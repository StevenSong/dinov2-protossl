# export a ProtoSSL teacher (ViT + prototype bank) for downstream use without dinov2, see protossl_rad.models:
#   python -m dinov2.train.export_protossl --config-file <output_dir>/config.yaml \
#       --checkpoint <output_dir>/eval/<iteration>/teacher_checkpoint.pth --output <export>.pth
# the student is not exported, it is only kept in the training checkpoints

import argparse
import os

import torch
from omegaconf import OmegaConf

from dinov2.configs import dinov2_default_config
from dinov2.data.cxr.augmentations import RAD_DINO_MEAN, RAD_DINO_STD
from dinov2.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dinov2.models import build_model_from_cfg
from dinov2.utils.utils import _flat_block_key

BANK_KEY = "dino_head.last_layer.prototypes"


def _normalization(cfg):
    # the mean / std the training augmentation normalized with
    if cfg.train.cxr_augmentation:
        return RAD_DINO_MEAN, RAD_DINO_STD
    if cfg.train.cell_augmentation:
        raise NotImplementedError("export of cell_augmentation runs")
    return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def export_protossl(cfg, teacher_state_dict: dict[str, torch.Tensor], meta: dict | None = None) -> dict:
    """
    training config + a teacher state dict as saved by train.do_test -> protossl_rad.models export dict
    """
    # imported here so dinov2 doesn't require protossl_rad unless the prototype head is used
    from protossl_rad.models import make_export

    if not cfg.proto.enabled:
        raise ValueError("not a ProtoSSL run (proto.enabled is false)")
    student = cfg.student
    if student.num_register_tokens or student.ffn_layer != "mlp" or student.channel_adaptive:
        raise NotImplementedError("protossl_rad's ViT has no register tokens, SwiGLU FFN or channel-adaptive input")

    backbone = {k.removeprefix("backbone."): v for k, v in teacher_state_dict.items() if k.startswith("backbone.")}
    others = sorted(set(teacher_state_dict) - {f"backbone.{k}" for k in backbone})
    if others != [BANK_KEY]:
        raise ValueError(f"expected the backbone and {BANK_KEY} in the teacher state dict, besides the backbone: {others}")
    # the training layout (e.g. block_chunks), checked strictly against the run's config
    model, embed_dim = build_model_from_cfg(cfg, only_teacher=True)
    model.load_state_dict(backbone, strict=True)
    backbone = {_flat_block_key(k): v for k, v in backbone.items()}

    mean, std = _normalization(cfg)
    vit = {
        "img_size": cfg.crops.global_crops_size,
        "patch_size": student.patch_size,
        "embed_dim": embed_dim,
        "depth": model.n_blocks,
        "num_heads": model.num_heads,
        "mlp_ratio": backbone["blocks.0.mlp.fc1.weight"].shape[0] / embed_dim,
        "in_chans": student.in_chans,
        "qkv_bias": student.qkv_bias,
        "proj_bias": student.proj_bias,
        "ffn_bias": student.ffn_bias,
        "layerscale": student.layerscale,
        "interpolate_antialias": student.interpolate_antialias,
        "interpolate_offset": student.interpolate_offset,
    }
    return make_export(
        backbone=backbone,
        prototypes=teacher_state_dict[BANK_KEY],
        vit=vit,
        radius=cfg.proto.radius,
        preprocessing={"image_size": cfg.crops.global_crops_size, "mean": list(mean), "std": list(std)},
        meta={**(meta or {}), "train_config": OmegaConf.to_yaml(cfg)},
    )


def get_args_parser():
    parser = argparse.ArgumentParser("Export a ProtoSSL teacher (ViT + prototype bank)")
    parser.add_argument("--config-file", required=True, help="config.yaml written to the training output_dir")
    parser.add_argument("--checkpoint", required=True, help="teacher_checkpoint.pth written by train.do_test")
    parser.add_argument("--output", required=True, help="export path, must not exist")
    return parser


def main(args):
    if os.path.exists(args.output):
        raise FileExistsError(args.output)
    # defaults first, for configs saved before newer keys existed
    cfg = OmegaConf.merge(OmegaConf.create(dinov2_default_config), OmegaConf.load(args.config_file))
    teacher = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["teacher"]
    meta = {"checkpoint": os.path.abspath(args.checkpoint), "config_file": os.path.abspath(args.config_file)}
    export = export_protossl(cfg, teacher, meta=meta)
    torch.save(export, args.output)
    print(f"Exported {args.checkpoint} to {args.output}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
