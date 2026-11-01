# ProtoSSL: a ProtoViT prototype bank as the shared DINO / iBOT head. not exported from dinov2.layers so dinov2 only
# needs protossl_rad when the prototype head is used

import torch
from torch import nn

from protossl_rad.prototypes import PrototypeBank


class PrototypeHead(nn.Module):
    """
    PrototypeBank as one FSDP unit. FSDP only gathers the sharded bank inside forward, so everything that uses it
    (matching, the bank regularizers, statistics) goes through forward. the bank is named last_layer so
    optim.freeze_last_layer_epochs freezes it like the last layer of the DINO head
    """

    def __init__(
        self,
        *,  # enforce kwargs
        n_prototypes: int,
        n_subprototypes: int,
        dim: int,
        radius: int,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.last_layer = PrototypeBank(
            n_prototypes=n_prototypes,
            n_subprototypes=n_subprototypes,
            dim=dim,
            radius=radius,
            chunk_size=chunk_size,
        )

    def forward(
        self,
        image_queries: dict[str, tuple[torch.Tensor, tuple[int, int]]],
        patch_queries: dict[str, tuple] | None = None,
        *,  # enforce kwargs
        regularize: bool = False,
        coherence_margin: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """
        image_queries: name -> (tokens (B, N, D), grid), image-level matching -> activations (B, P)
        patch_queries: name -> (tokens, grid, rows, anchors, grad_mask), matching anchored at patch anchors[q] of image
            rows[q] -> activations (Q, P), see PrototypeBank.anchored
        regularize: also return the "koleo" and "coherence" losses and the bank statistics (as "bank_<name>")
        """
        bank = self.last_layer
        out = {name: bank(tokens, grid).activations for name, (tokens, grid) in image_queries.items()}
        for name, (tokens, grid, rows, anchors, grad_mask) in (patch_queries or {}).items():
            out[name] = bank.anchored(tokens, grid, rows=rows, anchors=anchors, grad_mask=grad_mask).activations
        if regularize:
            out["koleo"] = bank.koleo_loss()
            out["coherence"] = bank.coherence_loss(coherence_margin)
            out.update({f"bank_{k}": v for k, v in bank.statistics().items()})
        return out
