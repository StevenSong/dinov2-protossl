import itertools

import pytest
import torch
import torch.nn.functional as F

from protossl_rad.prototypes import PrototypeBank, check_feasible, greedy_match, neighbor_table, subtract_cls


# ---------------------------------------------------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------------------------------------------------


def protovit_neighboring_mask(center_indices, h, w, radius, num_prototypes):
    # port of ProtoViT PPNet.neigboring_mask (model.py), generalized from the hardcoded 14x14 grid to h x w
    large_h, large_w = h + radius * 2, w + radius * 2
    large_matrix = torch.zeros(center_indices.shape[0], num_prototypes, large_h * large_w)
    small_total = (2 * radius + 1) ** 2
    small_matrix = torch.ones(center_indices.shape[0], num_prototypes, small_total)
    batch_size, num_points, _ = center_indices.shape
    small_size = int(small_matrix.shape[-1] ** 0.5)
    center_row, center_col = center_indices.squeeze(-1) // w, center_indices.squeeze(-1) % w
    start_row = torch.clamp(center_row + radius - small_size // 2, 0, large_h - small_size)
    start_col = torch.clamp(center_col + radius - small_size // 2, 0, large_w - small_size)
    for i in range(small_size):
        for j in range(small_size):
            large_idx = (start_row + i) * large_w + (start_col + j)
            small_idx = i * small_size + j
            large_matrix.view(batch_size, num_points, -1)[
                torch.arange(batch_size)[:, None], torch.arange(num_points), large_idx
            ] += small_matrix[..., small_idx]
    large_matrix = large_matrix.view(batch_size, num_points, large_h, large_w)
    return large_matrix[:, :, radius:-radius, radius:-radius].reshape(batch_size, num_points, -1)


def protovit_greedy_distance(dist_all, h, w, radius):
    # port of ProtoViT PPNet.greedy_distance with dist_all (B, P, N, K) given, and all slots equal (no pruning), in
    # which case values_slot == values_reordered. returns per-subprototype values and patch indices
    bsz, num_prototypes, n_patches, n_p = dist_all.shape
    mask_act = torch.ones((bsz, num_prototypes, n_patches))
    mask_subpatch = torch.ones((bsz, num_prototypes, n_p))
    mask_all = torch.ones((bsz, num_prototypes, n_patches, n_p))
    adjacent_mask = torch.ones((bsz, num_prototypes, n_patches))
    indices = torch.FloatTensor()
    values = torch.FloatTensor()
    subpatch_ids = torch.LongTensor()
    for _ in range(n_p):
        dist_all_masked = dist_all + (1 - mask_all * adjacent_mask.unsqueeze(-1)) * (-1e5)
        max_subs, max_subs_id = dist_all_masked.max(2)
        max_sub_act, max_sub_act_id = max_subs.max(-1)
        max_patch_id = max_subs_id.gather(-1, max_sub_act_id.unsqueeze(-1))
        adjacent_mask = protovit_neighboring_mask(max_patch_id, h, w, radius, num_prototypes)
        mask_act = mask_act.scatter(index=max_patch_id, dim=2, value=0)
        mask_subpatch = mask_subpatch.scatter(index=max_sub_act_id.unsqueeze(-1), dim=2, value=0)
        mask_all = mask_all * mask_act.unsqueeze(-1)
        mask_all = mask_all.permute(0, 1, 3, 2)
        mask_all = mask_all * mask_subpatch.unsqueeze(-1)
        mask_all = mask_all.permute(0, 1, 3, 2)
        subpatch_ids = torch.cat([subpatch_ids, max_sub_act_id.unsqueeze(-1)], dim=-1)
        indices = torch.cat([indices, max_patch_id], dim=-1)
        values = torch.cat([values, max_sub_act.unsqueeze(-1)], dim=-1)
    _, sub_indexes = subpatch_ids.to(torch.int64).sort(-1)
    return torch.gather(values, -1, sub_indexes), torch.gather(indices, -1, sub_indexes).long()


def brute_force_anchored(sims, h, w, radius, rows, anchors):
    # loop reference for anchored matching, sims (B, N, P, K)
    n_protos, n_sub = sims.shape[2:]
    values = torch.zeros(len(rows), n_protos, n_sub)
    patches = torch.zeros(len(rows), n_protos, n_sub, dtype=torch.long)
    for q, (b, a) in enumerate(zip(rows.tolist(), anchors.tolist())):
        for p in range(n_protos):
            s = sims[b, :, p]  # (N, K)
            k = int(s[a].argmax())
            chosen = {k: a}
            last = a
            for _ in range(n_sub - 1):
                lr, lc = divmod(last, w)
                best = None
                for n in range(h * w):
                    r, c = divmod(n, w)
                    if max(abs(r - lr), abs(c - lc)) > radius or n in chosen.values():
                        continue
                    for kk in range(n_sub):
                        if kk not in chosen and (best is None or s[n, kk] > best[0]):
                            best = (s[n, kk], n, kk)
                _, last, kk = best
                chosen[kk] = last
            for kk, n in chosen.items():
                values[q, p, kk] = s[n, kk]
                patches[q, p, kk] = n
    return values, patches


def random_inputs(b, h, w, p, k, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = F.normalize(torch.randn(b, h * w, d, generator=g), dim=-1)
    protos = F.normalize(torch.randn(p, k, d, generator=g), dim=-1)
    return tokens, protos


def all_sims(tokens, protos):
    return torch.einsum("bnd,pkd->bnpk", tokens, protos)


def chebyshev(a, b, w):
    return torch.maximum((a // w - b // w).abs(), (a % w - b % w).abs())


# ---------------------------------------------------------------------------------------------------------------------
# greedy matching
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "h,w,k,radius",
    [(14, 14, 4, 1), (5, 7, 4, 1), (2, 2, 4, 1), (6, 6, 9, 2), (4, 4, 1, 1), (37, 37, 4, 1)],
)
def test_image_matching_matches_protovit(h, w, k, radius):
    tokens, protos = random_inputs(3, h, w, 6, k, 16)
    match = greedy_match(tokens, protos, (h, w), radius=radius)
    ref_values, ref_patches = protovit_greedy_distance(all_sims(tokens, protos).permute(0, 2, 1, 3), h, w, radius)
    torch.testing.assert_close(match.similarities, ref_values)
    assert torch.equal(match.patches, ref_patches)
    # ProtoViT's score is the sum, ours the mean
    torch.testing.assert_close(match.activations, ref_values.sum(-1) / k)


@pytest.mark.parametrize("h,w,k,radius", [(6, 6, 4, 1), (3, 5, 4, 1), (5, 5, 9, 2)])
def test_anchored_matching_matches_brute_force(h, w, k, radius):
    tokens, protos = random_inputs(2, h, w, 5, k, 16)
    rows = torch.tensor([0, 0, 1, 1, 1])
    anchors = torch.tensor([0, h * w - 1, w + 1, w - 1, (h // 2) * w + w // 2])  # corners, edge, interior
    match = greedy_match(tokens, protos, (h, w), radius=radius, rows=rows, anchors=anchors)
    ref_values, ref_patches = brute_force_anchored(all_sims(tokens, protos), h, w, radius, rows, anchors)
    torch.testing.assert_close(match.similarities, ref_values)
    assert torch.equal(match.patches, ref_patches)
    # the anchor subprototype sits on the anchor patch and is the best subprototype there
    on_anchor = match.patches == anchors[:, None, None]
    assert torch.equal(on_anchor.sum(-1), torch.ones(len(rows), 5, dtype=torch.long))
    anchor_sims = all_sims(tokens, protos)[rows, anchors]  # (Q, P, K)
    torch.testing.assert_close(match.similarities[on_anchor], anchor_sims.max(-1).values.flatten())


@pytest.mark.parametrize("anchored", [False, True])
def test_matches_are_distinct_and_chained(anchored):
    h, w, k, radius = 8, 8, 4, 1
    tokens, protos = random_inputs(4, h, w, 32, k, 8, seed=1)
    kwargs = dict(rows=torch.arange(4).repeat(3), anchors=torch.randint(0, h * w, (12,))) if anchored else {}
    match = greedy_match(tokens, protos, (h, w), radius=radius, **kwargs)
    patches = match.patches
    # every subprototype gets its own patch
    assert (patches[..., :, None] != patches[..., None, :]).sum(-1).eq(k - 1).all()
    # every patch is within the radius of some other chosen patch (the one selected before it, chained)
    dists = chebyshev(patches[..., :, None], patches[..., None, :], w)
    dists = dists + torch.eye(k, dtype=torch.long) * 100
    assert (dists.min(-1).values <= radius).all()
    assert match.activations.abs().le(1 + 1e-6).all()


def test_chunking_is_exact():
    h, w = 6, 6
    tokens, protos = random_inputs(3, h, w, 11, 4, 16, seed=2)
    rows, anchors = torch.tensor([0, 1, 2, 2]), torch.tensor([0, 7, 35, 20])
    grad_mask = torch.rand(3, h * w, generator=torch.Generator().manual_seed(0)) < 0.5
    results = []
    for chunk_size in (None, 1, 4, 11, 100):
        t = tokens.clone().requires_grad_()
        p = protos.clone().requires_grad_()
        image = greedy_match(t, p, (h, w), radius=1, chunk_size=chunk_size)
        patch = greedy_match(
            t, p, (h, w), radius=1, rows=rows, anchors=anchors, grad_mask=grad_mask, chunk_size=chunk_size
        )
        (image.activations.square().sum() + patch.activations.sum()).backward()
        results.append((image.similarities, image.patches, patch.similarities, patch.patches, t.grad, p.grad))
    for other in results[1:]:
        for a, b in zip(results[0], other):
            torch.testing.assert_close(a, b)


def test_image_gradients_only_reach_matched_terms():
    h, w, k = 5, 5, 4
    tokens, protos = random_inputs(2, h, w, 3, k, 8, seed=3)
    tokens.requires_grad_()
    protos.requires_grad_()
    match = greedy_match(tokens, protos, (h, w), radius=1)
    match.activations.sum().backward()
    # d(mean_k <t_n(k), p_k>) / d t_n = sum of the subprototypes matched to n, / K
    expected_t = torch.zeros_like(tokens)
    expected_p = torch.zeros_like(protos)
    for b, p, kk in itertools.product(range(2), range(3), range(k)):
        n = match.patches[b, p, kk]
        expected_t[b, n] += protos[p, kk].detach() / k
        expected_p[p, kk] += tokens[b, n].detach() / k
    torch.testing.assert_close(tokens.grad, expected_t)
    torch.testing.assert_close(protos.grad, expected_p)


def test_grad_mask_detaches_whole_terms():
    h, w, k = 5, 5, 4
    tokens, protos = random_inputs(2, h, w, 7, k, 8, seed=4)
    tokens.requires_grad_()
    protos.requires_grad_()
    grad_mask = torch.zeros(2, h * w, dtype=torch.bool)
    rows, anchors = torch.tensor([0, 1, 1]), torch.tensor([12, 0, 24])
    grad_mask[rows, anchors] = True  # only the anchors are masked (student sees mask tokens there)
    grad_mask[0, 13] = True
    masked = greedy_match(tokens, protos, (h, w), radius=1, rows=rows, anchors=anchors, grad_mask=grad_mask)
    unmasked = greedy_match(tokens.detach(), protos.detach(), (h, w), radius=1, rows=rows, anchors=anchors)
    # detaching doesn't change the search or the values
    torch.testing.assert_close(masked.similarities, unmasked.similarities)
    assert torch.equal(masked.patches, unmasked.patches)
    masked.activations.sum().backward()
    expected_t = torch.zeros_like(tokens)
    expected_p = torch.zeros_like(protos)
    for q, p, kk in itertools.product(range(3), range(7), range(k)):
        b, n = rows[q], masked.patches[q, p, kk]
        if grad_mask[b, n]:
            expected_t[b, n] += protos[p, kk].detach() / k
            expected_p[p, kk] += tokens[b, n].detach() / k
    torch.testing.assert_close(tokens.grad, expected_t)
    torch.testing.assert_close(protos.grad, expected_p)
    # visible tokens get no gradient at all
    assert tokens.grad[~grad_mask].abs().sum() == 0


def test_neighbor_table():
    idx, valid = neighbor_table(3, 4, 1, torch.device("cpu"))
    assert idx.shape == (12, 9)
    # corner (0, 0): itself, right, below, below-right
    assert sorted(idx[0][valid[0]].tolist()) == [0, 1, 4, 5]
    # interior (1, 1) has the full window, in ascending order
    assert idx[5].tolist() == [0, 1, 2, 4, 5, 6, 8, 9, 10] and valid[5].all()


def test_feasibility():
    check_feasible((37, 37), 4, 1)
    check_feasible((2, 2), 4, 1)
    check_feasible((3, 3), 9, 2)
    with pytest.raises(ValueError):
        check_feasible((14, 14), 5, 1)
    with pytest.raises(ValueError):
        check_feasible((1, 14), 2, 1)
    with pytest.raises(ValueError):
        PrototypeBank(n_prototypes=4, n_subprototypes=5, dim=8, radius=1)
    tokens, protos = random_inputs(1, 4, 4, 2, 4, 8)
    with pytest.raises(ValueError):
        greedy_match(tokens, protos, (4, 5), radius=1)


# ---------------------------------------------------------------------------------------------------------------------
# bank
# ---------------------------------------------------------------------------------------------------------------------


def make_bank(p=6, k=4, d=16, **kwargs):
    torch.manual_seed(0)
    return PrototypeBank(n_prototypes=p, n_subprototypes=k, dim=d, **kwargs)


def test_bank_init_and_forward():
    bank = make_bank()
    assert bank.prototypes.shape == (6, 4, 16)
    torch.testing.assert_close(bank.prototypes.norm(dim=-1), torch.ones(6, 4))
    # gaussian init, not confined to the positive orthant
    assert (bank.prototypes < 0).any()

    cls = torch.randn(2, 16)
    patch = torch.randn(2, 20, 16)
    feats = subtract_cls(cls, patch)
    torch.testing.assert_close(feats, patch - cls[:, None])
    match = bank(feats * 5.0, (4, 5))  # tokens are normalized inside, scale doesn't matter
    ref = greedy_match(F.normalize(feats, dim=-1), bank.normalized_prototypes(), (4, 5), radius=1)
    torch.testing.assert_close(match.activations, ref.activations)
    assert match.activations.shape == (2, 6)

    rows, anchors = torch.tensor([1, 0]), torch.tensor([3, 19])
    anchored = bank.anchored(feats, (4, 5), rows=rows, anchors=anchors)
    assert anchored.activations.shape == (2, 6)


def test_coherence_loss():
    bank = make_bank(p=5, k=4, d=8)
    protos = F.normalize(bank.prototypes.detach(), dim=-1)
    all_pairs = sum(
        max(1 - float(protos[p, i] @ protos[p, j]) for i in range(4) for j in range(4) if i != j) for p in range(5)
    )
    torch.testing.assert_close(bank.coherence_loss(), torch.tensor(all_pairs))
    assert bank.coherence_loss(margin=10.0) == 0
    # identical subprototypes are fully coherent
    with torch.no_grad():
        bank.prototypes.copy_(bank.prototypes[:, :1].expand_as(bank.prototypes))
    assert bank.coherence_loss() < 1e-5
    bank.coherence_loss().backward()  # gradient defined for the parameter


def test_koleo_loss_excludes_peers():
    bank = make_bank(p=8, k=4, d=16)
    flat = F.normalize(bank.prototypes.detach(), dim=-1).flatten(0, 1)
    owner = torch.arange(8).repeat_interleave(4)
    expected = 0.0
    for i in range(32):
        dists = [float((flat[i] - flat[j]).norm()) for j in range(32) if owner[j] != owner[i]]
        expected += -torch.log(torch.tensor(min(dists)) + 1e-8)
    torch.testing.assert_close(bank.koleo_loss(), expected / 32)

    # collapsing a prototype's own subprototypes onto each other must not blow up the loss (peers are excluded)
    with torch.no_grad():
        bank.prototypes.copy_(bank.prototypes[:, :1].expand_as(bank.prototypes))
    assert torch.isfinite(bank.koleo_loss())
    bank.koleo_loss().backward()
    assert torch.isfinite(bank.prototypes.grad).all()


def test_half_precision_runs():
    bank = make_bank(p=4, k=4, d=16).half()
    feats = torch.randn(2, 9, 16).half()
    match = bank(feats, (3, 3))
    assert match.activations.dtype == torch.float16
    assert torch.isfinite(match.activations).all()
