# CPU tests of protossl_rad.models (export format and loading). equivalence to dinov2's ViT and the dinov2-side
# export are tested in the dinov2 repo's tests/test_export_protossl.py

import pytest
import torch

from protossl_rad.models import EXPORT_FORMAT, load_pretrained, make_export, VisionTransformer
from protossl_rad.prototypes import PrototypeBank, subtract_cls

VIT = {"img_size": 42, "patch_size": 14, "embed_dim": 32, "depth": 2, "num_heads": 4, "layerscale": 1e-5}
PREPROCESSING = {"image_size": 42, "mean": [0.5] * 3, "std": [0.25] * 3}


def make_models(seed=0):
    torch.manual_seed(seed)
    vit = VisionTransformer(**VIT)
    with torch.no_grad():
        for t in [*vit.parameters(), *vit.buffers()]:
            t.add_(0.1 * torch.randn_like(t))
    bank = PrototypeBank(n_prototypes=5, n_subprototypes=4, dim=32, radius=1)
    return vit.eval(), bank


@torch.no_grad()
def test_roundtrip(tmp_path):
    vit, bank = make_models()
    export = make_export(
        backbone=vit.state_dict(),
        prototypes=bank.prototypes,
        vit=VIT,
        radius=1,
        preprocessing=PREPROCESSING,
        meta={"note": "test"},
    )
    torch.save(export, tmp_path / "export.pth")
    vit2, bank2, info = load_pretrained(str(tmp_path / "export.pth"), chunk_size=2)
    vit2.eval()
    assert bank2.chunk_size == 2
    assert torch.equal(bank2.prototypes, bank.prototypes)
    x = torch.randn(2, 3, 42, 56)
    out, out2 = vit.forward_features(x), vit2.forward_features(x)
    for key in out:
        assert torch.equal(out[key], out2[key]), key
    tokens = subtract_cls(out["x_norm_clstoken"], out["x_norm_patchtokens"])
    assert torch.equal(bank(tokens, (3, 4)).activations, bank2(tokens, (3, 4)).activations)
    assert info == {
        "format": EXPORT_FORMAT,
        "vit": VIT,
        "prototype_bank": {"n_prototypes": 5, "n_subprototypes": 4, "radius": 1},
        "preprocessing": PREPROCESSING,
        "meta": {"note": "test"},
    }


def test_mask_token_is_not_a_parameter():
    # loaded from DINOv2 checkpoints but unused, a parameter would trip DDP's unused parameter check
    vit, _ = make_models()
    assert "mask_token" in vit.state_dict()
    assert "mask_token" not in dict(vit.named_parameters())


def test_make_export_checks_config():
    vit, bank = make_models()
    kwargs = {"backbone": vit.state_dict(), "prototypes": bank.prototypes, "radius": 1, "preprocessing": PREPROCESSING}
    with pytest.raises(RuntimeError):
        make_export(vit={**VIT, "depth": 3}, **kwargs)
    with pytest.raises(RuntimeError):
        make_export(vit={**VIT, "layerscale": None}, **kwargs)
    with pytest.raises(RuntimeError):
        make_export(vit=VIT, **{**kwargs, "prototypes": bank.prototypes[..., :16]})


def test_load_rejects_other_files(tmp_path):
    vit, _ = make_models()
    torch.save(vit.state_dict(), tmp_path / "state_dict.pth")
    with pytest.raises(ValueError, match=EXPORT_FORMAT):
        load_pretrained(str(tmp_path / "state_dict.pth"))
