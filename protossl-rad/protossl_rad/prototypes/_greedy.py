from functools import lru_cache
from typing import NamedTuple

import torch


class GreedyMatch(NamedTuple):
    # all indexed by subprototype id (not selection order), Q = number of queries (images or anchored patches)
    activations: torch.Tensor  # (Q, P) mean over subprototypes of the matched similarities, in [-1, 1]
    similarities: torch.Tensor  # (Q, P, K) cosine similarity of each subprototype to its matched patch
    patches: torch.Tensor  # (Q, P, K) flat patch index (row-major in the grid) matched to each subprototype


def check_feasible(grid_size: tuple[int, int], n_subprototypes: int, radius: int) -> None:
    # each step after the first searches the window around the last selected patch, which already holds at most
    # K - 2 other selected patches. a corner window has (r + 1)^2 - 1 other cells, so a free patch always exists
    # when K <= (r + 1)^2 (and the grid is wide enough to contain a full corner window)
    h, w = grid_size
    if n_subprototypes > (radius + 1) ** 2 or min(h, w) < radius + 1:
        raise ValueError(
            f"greedy matching with K={n_subprototypes} subprototypes and radius={radius} can run out of adjacent "
            f"patches on a {h}x{w} grid, need K <= (radius + 1)^2 and both grid sides >= radius + 1"
        )


@lru_cache(maxsize=16)
def neighbor_table(h: int, w: int, radius: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (N, (2r+1)^2) flat indices of each patch's window, row-major so indices are ascending, and a mask of which
    entries are inside the grid (out-of-grid entries point at the center patch and must be masked out)
    """
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    rows, cols = rows.flatten(), cols.flatten()
    offsets = torch.arange(-radius, radius + 1)
    dr, dc = torch.meshgrid(offsets, offsets, indexing="ij")
    nr = rows[:, None] + dr.flatten()[None]
    nc = cols[:, None] + dc.flatten()[None]
    valid = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
    idx = torch.where(valid, nr * w + nc, torch.arange(h * w)[:, None])
    return idx.to(device), valid.to(device)


@torch.no_grad()
def _select(
    sims: torch.Tensor,
    rows: torch.Tensor,
    anchors: torch.Tensor | None,
    neighbors: torch.Tensor,
    neighbors_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ProtoViT greedy matching. sims (B, N, P, K); each query q matches every prototype on image rows[q].
    step 1 takes the best (patch, subprototype) pair over the whole image, or with anchors the best subprototype
    at patch anchors[q]. every later step takes the best unused subprototype on an unused patch in the window
    around the last selected patch. returns (Q, P, K) patch and subprototype indices in selection order
    """
    n_queries = rows.shape[0]
    _, _, n_protos, n_sub = sims.shape
    proto_idx = torch.arange(n_protos, device=sims.device)

    if anchors is None:
        # one query per image (rows is arange). best patch per subprototype, then best subprototype (same order as
        # ProtoViT, so ties break the same way)
        best_per_sub, best_patch_per_sub = sims.max(dim=1)  # (Q, P, K)
        sub = best_per_sub.argmax(dim=-1)  # (Q, P)
        patch = best_patch_per_sub.gather(-1, sub[..., None]).squeeze(-1)
    else:
        sub = sims[rows, anchors].argmax(dim=-1)  # (Q, P)
        patch = anchors[:, None].expand(n_queries, n_protos)

    patches, subs = [patch], [sub]
    sub_used = torch.zeros(n_queries, n_protos, n_sub, dtype=torch.bool, device=sims.device)
    sub_used.scatter_(-1, sub[..., None], True)
    for _ in range(n_sub - 1):
        window = neighbors[patches[-1]]  # (Q, P, W)
        free = neighbors_valid[patches[-1]].clone()
        for prev in patches:
            free &= window != prev[..., None]
        cand = sims[rows[:, None, None], window, proto_idx[None, :, None]]  # (Q, P, W, K)
        cand = cand.masked_fill(~free[..., None] | sub_used[:, :, None, :], float("-inf"))
        flat = cand.flatten(2).argmax(dim=-1)
        sub = flat % n_sub
        patch = window.gather(-1, (flat // n_sub)[..., None]).squeeze(-1)
        patches.append(patch)
        subs.append(sub)
        sub_used.scatter_(-1, sub[..., None], True)
    return torch.stack(patches, dim=-1), torch.stack(subs, dim=-1)


def greedy_match(
    tokens: torch.Tensor,
    prototypes: torch.Tensor,
    grid_size: tuple[int, int],
    *,  # enforce kwargs
    radius: int,
    rows: torch.Tensor | None = None,
    anchors: torch.Tensor | None = None,
    grad_mask: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> GreedyMatch:
    """
    tokens (B, N, D) and prototypes (P, K, D), both already L2-normalized; N = h * w patches in row-major order.

    rows/anchors: one query per anchored patch, matching every prototype starting from patch anchors[q] of image
    rows[q] (iBOT). if both None, one query per image with a free first step (DINO).
    grad_mask: (B, N) bool, similarity terms whose matched patch is False are detached (the whole term, so neither
    the token nor the prototype gets gradient from it). they still take part in the search.
    chunk_size: number of prototypes matched at once. the (B, N, chunk, K) similarity tensor is the peak memory
    in forward and (per chunk) in backward
    """
    n_images, n_patches, dim = tokens.shape
    n_protos, n_sub, _ = prototypes.shape
    h, w = grid_size
    if h * w != n_patches:
        raise ValueError(f"grid {h}x{w} doesn't match {n_patches} tokens")
    check_feasible(grid_size, n_sub, radius)
    if (rows is None) != (anchors is None):
        raise ValueError("rows and anchors must be given together")
    if rows is None:
        rows = torch.arange(n_images, device=tokens.device)
    neighbors, neighbors_valid = neighbor_table(h, w, radius, tokens.device)

    chunk_size = chunk_size or n_protos
    flat_tokens = tokens.reshape(n_images * n_patches, dim)
    sims_out, patches_out = [], []
    for start in range(0, n_protos, chunk_size):
        chunk = prototypes[start : start + chunk_size]
        n_chunk = chunk.shape[0]
        sims = (flat_tokens @ chunk.reshape(n_chunk * n_sub, dim).T).view(n_images, n_patches, n_chunk, n_sub)
        patch_idx, sub_idx = _select(sims.detach(), rows, anchors, neighbors, neighbors_valid)

        # reorder by subprototype id, then gather only the matched similarities with grad so backward never needs
        # the full similarity tensor
        order = sub_idx.argsort(dim=-1)
        patch_idx = patch_idx.gather(-1, order)
        sub_idx = sub_idx.gather(-1, order)
        proto_idx = torch.arange(n_chunk, device=tokens.device)[None, :, None]
        matched = sims[rows[:, None, None], patch_idx, proto_idx, sub_idx]  # (Q, chunk, K)
        if grad_mask is not None:
            keep = grad_mask[rows[:, None, None], patch_idx]
            matched = torch.where(keep, matched, matched.detach())
        sims_out.append(matched)
        patches_out.append(patch_idx)

    similarities = torch.cat(sims_out, dim=1)
    return GreedyMatch(similarities.mean(dim=-1), similarities, torch.cat(patches_out, dim=1))
