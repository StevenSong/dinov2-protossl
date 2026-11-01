import torch
import torch.nn.functional as F
from torch import nn

from ._greedy import GreedyMatch, check_feasible, greedy_match


def subtract_cls(cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
    # ProtoViT features: final-norm patch tokens minus the final-norm CLS token, (B, D), (B, N, D) -> (B, N, D)
    return patch_tokens - cls_token[:, None]


def _normalize(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6 if x.dtype == torch.float16 else 1e-12
    return F.normalize(x, dim=-1, p=2, eps=eps)


class PrototypeBank(nn.Module):
    """
    P prototypes of K subprototypes each, compared to patch tokens by cosine similarity and greedily matched to
    adjacent patches (ProtoViT). activations are the MEAN over subprototypes, so they lie in [-1, 1]
    (ProtoViT sums, [-K, K]). no learnable slot pruning
    """

    def __init__(
        self,
        *,  # enforce kwargs
        n_prototypes: int,
        n_subprototypes: int,
        dim: int,
        radius: int = 1,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.n_subprototypes = n_subprototypes
        self.dim = dim
        self.radius = radius
        self.chunk_size = chunk_size
        # check_feasible needs a grid, the smallest one that could work tells us if K and radius are compatible at all
        check_feasible((radius + 1, radius + 1), n_subprototypes, radius)
        self.prototypes = nn.Parameter(torch.empty(n_prototypes, n_subprototypes, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # gaussian directions rather than ProtoViT's torch.rand, which puts every prototype in the positive orthant
        # while CLS-subtracted tokens are centered
        with torch.no_grad():
            nn.init.normal_(self.prototypes)
            self.prototypes.copy_(F.normalize(self.prototypes, dim=-1))

    def normalized_prototypes(self) -> torch.Tensor:
        return _normalize(self.prototypes)

    def forward(self, tokens: torch.Tensor, grid_size: tuple[int, int]) -> GreedyMatch:
        """
        image-level matching, tokens (B, N, D) (e.g. from subtract_cls, normalized here) -> one query per image
        """
        return greedy_match(
            _normalize(tokens),
            self.normalized_prototypes(),
            grid_size,
            radius=self.radius,
            chunk_size=self.chunk_size,
        )

    def anchored(
        self,
        tokens: torch.Tensor,
        grid_size: tuple[int, int],
        *,  # enforce kwargs
        rows: torch.Tensor,
        anchors: torch.Tensor,
        grad_mask: torch.Tensor | None = None,
    ) -> GreedyMatch:
        """
        patch-level matching, one query per (rows[q], anchors[q]) patch: each prototype puts its best subprototype
        at the anchor patch and greedily matches the rest around it. terms landing where grad_mask (B, N) is False
        are detached but still searched
        """
        return greedy_match(
            _normalize(tokens),
            self.normalized_prototypes(),
            grid_size,
            radius=self.radius,
            rows=rows,
            anchors=anchors,
            grad_mask=grad_mask,
            chunk_size=self.chunk_size,
        )

    def coherence_loss(self, margin: float = 0.0) -> torch.Tensor:
        """
        ProtoViT coherence: per prototype, the largest cosine distance over all pairs of its subprototypes, hinged
        at margin and SUMMED over prototypes (as ProtoViT, so its 3e-3 coefficient keeps its meaning)
        """
        protos = F.normalize(self.prototypes.float(), dim=-1)
        dist = 1 - protos @ protos.transpose(1, 2)  # (P, K, K), diagonal is 0 so it never wins the max
        worst = dist.flatten(1).max(dim=-1).values
        return F.relu(worst - margin).sum()

    @torch.no_grad()
    def _other_prototype_cosines(self, flat: torch.Tensor) -> torch.Tensor:
        # (P * K, P * K) cosines between subprototypes, -2 (below any cosine) for pairs within the same prototype
        dots = flat @ flat.T
        owner = torch.arange(self.n_prototypes, device=flat.device).repeat_interleave(self.n_subprototypes)
        return dots.masked_fill_(owner[:, None] == owner[None], -2.0)

    def koleo_loss(self, eps: float = 1e-8) -> torch.Tensor:
        """
        KoLeo (Sablayrolles et al. 2018) over all P * K subprototypes, where each one's nearest neighbor is searched
        only among other prototypes' subprototypes, so it doesn't fight the coherence loss
        """
        flat = F.normalize(self.prototypes.float(), dim=-1).flatten(0, 1)  # (P * K, D)
        nearest = self._other_prototype_cosines(flat).argmax(dim=-1)
        distances = F.pairwise_distance(flat, flat[nearest], p=2, eps=eps)
        return -torch.log(distances + eps).mean()

    @torch.no_grad()
    def statistics(self) -> dict[str, torch.Tensor]:
        """
        collapse diagnostics, all means: raw subprototype norm (weight decay acts on the raw parameter, so a shrinking
        norm raises the effective lr), worst cosine distance between a prototype's own subprototypes (the unhinged
        coherence term), cosine between a prototype's own subprototypes (-> 1 as they collapse into one) and cosine to
        the nearest subprototype of another prototype (-> 1 as prototypes collapse onto each other)
        """
        raw = self.prototypes.float()
        protos = F.normalize(raw, dim=-1)
        within = protos @ protos.transpose(1, 2)  # (P, K, K)
        stats = {
            "norm": raw.norm(dim=-1).mean(),
            "worst_within_dist": (1 - within).flatten(1).max(dim=-1).values.mean(),
            "nearest_other_cos": self._other_prototype_cosines(protos.flatten(0, 1)).max(dim=-1).values.mean(),
        }
        if self.n_subprototypes > 1:  # no pairs otherwise, and a NaN would trip the trainer's NaN check
            off_diagonal = ~torch.eye(self.n_subprototypes, dtype=torch.bool, device=raw.device)
            stats["within_cos"] = within[:, off_diagonal].mean()
        return stats
