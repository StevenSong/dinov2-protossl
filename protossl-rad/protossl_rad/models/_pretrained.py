# ProtoSSL pretrained weights (teacher ViT + prototype bank) exported from the DINOv2 fork for downstream (SupProto) use.
# the export is a torch.save'd dict:
#   format          EXPORT_FORMAT
#   backbone        ViT state dict with DINOv2's native flat keys (as the released dinov2_vitb14_pretrain.pth)
#   prototypes      (P, K, D) raw (unnormalized) PrototypeBank.prototypes
#   vit             VisionTransformer kwargs
#   prototype_bank  {"n_prototypes": P, "n_subprototypes": K, "radius": r}
#   preprocessing   {"image_size", "mean", "std"} the training normalization, image_size the global crop size
#   meta            provenance, e.g. source checkpoint and training config

import torch

from ..prototypes import PrototypeBank
from ._vit import VisionTransformer

EXPORT_FORMAT = "protossl-vit-v1"


def _build(export: dict, chunk_size: int | None = None) -> tuple[VisionTransformer, PrototypeBank]:
    # strict: any missing / unexpected key or shape mismatch raises
    backbone = VisionTransformer(**export["vit"])
    backbone.load_state_dict(export["backbone"], strict=True)
    bank = PrototypeBank(**export["prototype_bank"], dim=backbone.embed_dim, chunk_size=chunk_size)
    bank.load_state_dict({"prototypes": export["prototypes"]}, strict=True)
    return backbone, bank


def make_export(
    *,  # enforce kwargs
    backbone: dict[str, torch.Tensor],
    prototypes: torch.Tensor,
    vit: dict,
    radius: int,
    preprocessing: dict,
    meta: dict | None = None,
) -> dict:
    """
    the export dict (see module comment) for torch.save, checked by loading it into the standalone models
    """
    export = {
        "format": EXPORT_FORMAT,
        "backbone": {k: v.detach().float().cpu().contiguous() for k, v in backbone.items()},
        "prototypes": prototypes.detach().float().cpu().contiguous(),
        "vit": dict(vit),
        "prototype_bank": {
            "n_prototypes": prototypes.shape[0],
            "n_subprototypes": prototypes.shape[1],
            "radius": radius,
        },
        "preprocessing": dict(preprocessing),
        "meta": dict(meta or {}),
    }
    _build(export)
    return export


def load_pretrained(
    path: str,
    *,  # enforce kwargs
    chunk_size: int | None = None,
) -> tuple[VisionTransformer, PrototypeBank, dict]:
    """
    an exported ProtoSSL checkpoint -> (ViT, prototype bank, the export's non-tensor entries). the bank matches
    the CLS-subtracted patch tokens it was trained on:
        out = vit.forward_features(images)
        match = bank(subtract_cls(out["x_norm_clstoken"], out["x_norm_patchtokens"]), (H // patch, W // patch))
    chunk_size: prototypes matched at once (a runtime memory bound, not stored)
    """
    export = torch.load(path, map_location="cpu", weights_only=True)
    if export.get("format") != EXPORT_FORMAT:
        raise ValueError(f"{path} is not a {EXPORT_FORMAT} export (format: {export.get('format')})")
    backbone, bank = _build(export, chunk_size=chunk_size)
    info = {k: v for k, v in export.items() if k not in ("backbone", "prototypes")}
    return backbone, bank, info
