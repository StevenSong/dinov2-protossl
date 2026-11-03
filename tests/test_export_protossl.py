# CPU tests of the ProtoSSL export (dinov2.train.export_protossl) and protossl_rad's standalone ViT against dinov2's.
# run from the repo root:
#   python -m pytest tests
# the teacher is a plain module here, not FSDP-wrapped as in train.do_test
import pytest
import torch
from omegaconf import OmegaConf

from dinov2.configs import dinov2_default_config
from dinov2.data.cxr.augmentations import RAD_DINO_MEAN, RAD_DINO_STD
from dinov2.models.vision_transformer import vit_small
from dinov2.train.export_protossl import BANK_KEY, export_protossl, get_args_parser, main
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.utils.utils import _flat_block_key
from protossl_rad.models import load_pretrained, VisionTransformer
from protossl_rad.prototypes import subtract_cls

PATCH, GLOBAL = 14, 70  # 5x5 grid
SIZES = [(GLOBAL, GLOBAL), (42, 42), (56, 84)]  # native, interpolated, non-square


def perturb(module, seed=0):
    # the default init has zeros and constants (biases, pos_embed, mask_token, LayerScale) that would hide mixups
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return module


def assert_same_features(ref, out):
    for key in ("x_norm_clstoken", "x_norm_patchtokens", "x_prenorm"):
        assert torch.equal(ref[key], out[key]), key


def make_cfg(**overrides):
    cfg = {
        "proto": {"enabled": True, "n_prototypes": 8, "n_subprototypes": 4, "radius": 1, "chunk_size": 3},
        "dino": {"koleo_loss_weight": 0},
        "student": {"arch": "vit_small", "patch_size": PATCH, "block_chunks": 4},
        "crops": {"global_crops_size": GLOBAL},
        "train": {"cxr_augmentation": True},
        "compute_precision": {"grad_scaler": False},
    }
    return OmegaConf.merge(OmegaConf.create(dinov2_default_config), OmegaConf.create(cfg), OmegaConf.create(overrides))


def make_teacher(cfg):
    torch.manual_seed(0)
    teacher = SSLMetaArch(cfg).teacher
    return perturb(teacher).eval()


@pytest.mark.parametrize("block_chunks", [0, 4])
@pytest.mark.parametrize("layerscale", [1e-5, None])
@torch.no_grad()
def test_vit_matches_dinov2(block_chunks, layerscale):
    ref = vit_small(patch_size=PATCH, img_size=GLOBAL, init_values=layerscale, block_chunks=block_chunks)
    ref = perturb(ref).eval()
    vit = VisionTransformer(img_size=GLOBAL, patch_size=PATCH, embed_dim=384, depth=12, num_heads=6, layerscale=layerscale)
    vit.load_state_dict({_flat_block_key(k): v for k, v in ref.state_dict().items()}, strict=True)
    vit.eval()
    for size in SIZES:
        x = torch.randn(2, 3, *size)
        assert_same_features(ref.forward_features(x), vit.forward_features(x))


def test_vit_rejects_partial_patches():
    vit = VisionTransformer(img_size=GLOBAL, patch_size=PATCH, embed_dim=384, depth=1, num_heads=6)
    with pytest.raises(ValueError, match="multiple of the patch size"):
        vit.forward_features(torch.randn(1, 3, GLOBAL, GLOBAL + 1))


@torch.no_grad()
def test_export_roundtrip(tmp_path):
    cfg = make_cfg()
    teacher = make_teacher(cfg)
    # as train.do_test and train.setup save them
    checkpoint, config_file, output = tmp_path / "teacher_checkpoint.pth", tmp_path / "config.yaml", tmp_path / "out.pth"
    torch.save({"teacher": teacher.state_dict()}, checkpoint)
    OmegaConf.save(cfg, config_file)
    args = ["--config-file", str(config_file), "--checkpoint", str(checkpoint), "--output", str(output)]
    main(get_args_parser().parse_args(args))

    vit, bank, info = load_pretrained(str(output), chunk_size=3)
    vit.eval()
    exported = torch.load(output, map_location="cpu", weights_only=True)
    # native flat keys, though the run used block_chunks=4
    flat = vit_small(patch_size=PATCH, img_size=GLOBAL, init_values=1e-5, block_chunks=0).state_dict()
    assert set(exported["backbone"]) == set(flat)
    for size in SIZES:
        x = torch.randn(2, 3, *size)
        ref, out = teacher.backbone.forward_features(x), vit.forward_features(x)
        assert_same_features(ref, out)
        grid = (size[0] // PATCH, size[1] // PATCH)
        tokens = subtract_cls(out["x_norm_clstoken"], out["x_norm_patchtokens"])
        ref_act = teacher.dino_head({"x": (tokens, grid)})["x"]
        assert torch.equal(ref_act, bank(tokens, grid).activations)
    assert torch.equal(bank.prototypes, teacher.dino_head.last_layer.prototypes)

    assert info["prototype_bank"] == {"n_prototypes": 8, "n_subprototypes": 4, "radius": 1}
    assert info["preprocessing"] == {"image_size": GLOBAL, "mean": list(RAD_DINO_MEAN), "std": list(RAD_DINO_STD)}
    assert info["meta"]["checkpoint"] == str(checkpoint)
    assert OmegaConf.create(info["meta"]["train_config"]) == cfg

    with pytest.raises(FileExistsError):
        main(get_args_parser().parse_args(args))


def test_export_rejects_non_proto_runs():
    with pytest.raises(ValueError, match="proto.enabled"):
        export_protossl(make_cfg(proto={"enabled": False}), {})
    with pytest.raises(NotImplementedError):
        export_protossl(make_cfg(student={"num_register_tokens": 4}), {})


def test_export_rejects_mismatched_checkpoints():
    cfg = make_cfg()
    state_dict = make_teacher(cfg).state_dict()
    with pytest.raises(ValueError, match="besides the backbone"):
        export_protossl(cfg, {**state_dict, "ibot_head.last_layer.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="besides the backbone"):
        export_protossl(cfg, {k: v for k, v in state_dict.items() if k != BANK_KEY})
    # the checkpoint has LayerScale, the config doesn't
    with pytest.raises(RuntimeError, match="Unexpected key"):
        export_protossl(make_cfg(student={"layerscale": None}), state_dict)
