# CPU tests of the ProtoSSL training step (SSLMetaArch with proto.enabled). run from the repo root:
#   python -m pytest tests
# backbones are replaced by small stubs (the real ViT needs xformers for multi-crop forwards), no FSDP
import copy
import random

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

import dinov2.loss.ibot_patch_loss
from dinov2.configs import dinov2_default_config
from dinov2.data import MaskingGenerator, collate_data_and_cast
from dinov2.layers.prototype_head import PrototypeHead
from dinov2.train.ssl_meta_arch import SSLMetaArch, _random_window
from protossl_rad.prototypes import PrototypeBank, subtract_cls

PATCH = 14
GLOBAL, LOCAL, N_LOCAL, BATCH = 70, 42, 2, 4  # 5x5 global grid, 3x3 local grid
TEACHER_TEMP, STUDENT_TEMP = 0.05, 0.1


class StubBackbone(nn.Module):
    # patch embedding + mask tokens + a CLS token, returning the backbone output keys the training step reads
    def __init__(self, dim):
        super().__init__()
        self.embed = nn.Linear(3 * PATCH * PATCH, dim)
        self.mask_token = nn.Parameter(torch.randn(dim))
        self.cls = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def _one(self, x, masks):
        tokens = self.embed(F.unfold(x.float(), PATCH, stride=PATCH).transpose(1, 2))
        if masks is not None:
            tokens = torch.where(masks[..., None], self.mask_token, tokens)
        tokens = torch.tanh(tokens)
        return {"x_norm_clstoken": self.norm(self.cls(tokens.mean(1))), "x_norm_patchtokens": self.norm(tokens)}

    def forward(self, x, masks=None, is_training=False):
        if isinstance(x, list):
            return [self._one(xi, mi) for xi, mi in zip(x, masks)]
        return self._one(x, masks)


def make_cfg(**overrides):
    cfg = {
        "proto": {"enabled": True, "n_prototypes": 8, "n_subprototypes": 4, "radius": 1, "chunk_size": 3},
        "dino": {"koleo_loss_weight": 0},
        "student": {"arch": "vit_small", "patch_size": PATCH},
        "crops": {"global_crops_size": GLOBAL, "local_crops_size": LOCAL, "local_crops_number": N_LOCAL},
        "compute_precision": {"grad_scaler": False},
    }
    return OmegaConf.merge(OmegaConf.create(dinov2_default_config), OmegaConf.create(cfg), OmegaConf.create(overrides))


@pytest.fixture(autouse=True)
def torch_ibot_lossfunc(monkeypatch):
    # the xformers cross entropy is GPU-only, use the reference formula
    def lossfunc(t, s, temp):
        return torch.sum(t.float() * F.log_softmax(s.float() / temp, dim=-1), dim=-1)

    monkeypatch.setattr(dinov2.loss.ibot_patch_loss, "lossfunc", lossfunc)


def make_model(cfg):
    torch.manual_seed(0)
    model = SSLMetaArch(cfg)
    model.student.backbone = StubBackbone(model.embed_dim)
    model.teacher.backbone = StubBackbone(model.embed_dim)
    # a teacher that differs from the student, so student / teacher mixups change the loss
    model.teacher.dino_head.load_state_dict(model.student.dino_head.state_dict())
    with torch.no_grad():
        prototypes = model.teacher.dino_head.last_layer.prototypes
        prototypes.add_(0.3 * torch.randn_like(prototypes))
    for p in model.teacher.parameters():
        p.requires_grad = False
    model.need_to_synchronize_fsdp_streams = False  # no FSDP / CUDA
    return model


def make_batch(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    samples = [
        (
            {
                "global_crops": [torch.randn(3, GLOBAL, GLOBAL) for _ in range(2)],
                "local_crops": [torch.randn(3, LOCAL, LOCAL) for _ in range(N_LOCAL)],
            },
            (),
        )
        for _ in range(BATCH)
    ]
    grid = GLOBAL // PATCH
    return collate_data_and_cast(
        samples,
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        dtype=torch.half,
        n_tokens=grid * grid,
        mask_generator=MaskingGenerator((grid, grid), max_num_patches=0.5 * grid * grid),
    )


def cross_entropy(targets, logits):
    # (Q, P), (Q, P) -> (Q,) -sum_p t_p log softmax(s / T)_p
    return -(targets * F.log_softmax(logits / STUDENT_TEMP, dim=-1)).sum(-1)


def manual_loss(model, data):
    """
    the training objective written out with the bank called directly, first step (centers are still 0)
    """
    student, teacher, cfg = model.student, model.teacher, model.cfg
    s_bank, t_bank = student.dino_head.last_layer, teacher.dino_head.last_layer
    g_grid, l_grid = (GLOBAL // PATCH,) * 2, (LOCAL // PATCH,) * 2
    n_patches = g_grid[0] * g_grid[1]
    masks, idx = data["collated_masks"], data["mask_indices_list"]
    rows, anchors = idx // n_patches, idx % n_patches
    n_terms = 2 + 2 * N_LOCAL

    with torch.no_grad():
        t_out = teacher.backbone(data["collated_global_crops"])
        t_tokens = subtract_cls(t_out["x_norm_clstoken"], t_out["x_norm_patchtokens"])
        t_act = t_bank(t_tokens, g_grid).activations
        # crop A's teacher is the target of crop B's student and vice versa
        t_targets = F.softmax(torch.cat(t_act.chunk(2)[::-1]) / TEACHER_TEMP, dim=-1)
        t_ibot = F.softmax(t_bank.anchored(t_tokens, g_grid, rows=rows, anchors=anchors).activations / TEACHER_TEMP, -1)

    s_global, s_local = student.backbone(
        [data["collated_global_crops"], data["collated_local_crops"]], masks=[masks, None]
    )
    s_tokens = subtract_cls(s_global["x_norm_clstoken"], s_global["x_norm_patchtokens"])
    l_tokens = subtract_cls(s_local["x_norm_clstoken"], s_local["x_norm_patchtokens"])
    s_act = s_bank(s_tokens, g_grid).activations
    l_act = s_bank(l_tokens, l_grid).activations
    s_ibot = s_bank.anchored(s_tokens, g_grid, rows=rows, anchors=anchors, grad_mask=masks).activations

    local = sum(cross_entropy(t, s).mean() for s in l_act.chunk(N_LOCAL) for t in t_targets.chunk(2)) / n_terms
    global_ = cross_entropy(t_targets, s_act).mean() * 2 / n_terms
    ibot = (cross_entropy(t_ibot, s_ibot) * data["masks_weight"]).sum() / masks.shape[0]
    koleo = cfg.dino.loss_weight * s_bank.koleo_loss()
    coherence = cfg.proto.coherence_loss_weight * s_bank.coherence_loss(cfg.proto.coherence_margin)
    losses = {
        "dino_local_crops_loss": local,
        "dino_global_crops_loss": global_,
        "ibot_loss": ibot / 2,
        "koleo_loss": koleo,
        "coherence_loss": coherence,
    }
    return local + global_ + ibot + koleo + coherence, losses


def test_forward_backward_matches_manual():
    model = make_model(make_cfg())
    reference = copy.deepcopy(model)
    data = make_batch()

    loss_dict = model.forward_backward(data, teacher_temp=TEACHER_TEMP)
    total, expected = manual_loss(reference, data)
    total.backward()

    for k, v in expected.items():
        torch.testing.assert_close(loss_dict[k], v, msg=k)
    named_ref = dict(reference.student.named_parameters())
    n_with_grad = 0
    for name, p in model.student.named_parameters():
        ref = named_ref[name].grad
        if ref is None:
            assert p.grad is None or not p.grad.any(), name
            continue
        torch.testing.assert_close(p.grad, ref, msg=name)
        n_with_grad += 1
    assert model.student.dino_head.last_layer.prototypes.grad.abs().sum() > 0
    assert n_with_grad >= 6  # bank + every stub backbone parameter
    assert all(p.grad is None for p in model.teacher.parameters())


def test_diagnostics_logged():
    model = make_model(make_cfg())
    loss_dict = model.forward_backward(make_batch(), teacher_temp=TEACHER_TEMP)
    diag = {k for k in loss_dict if k.startswith("diag_")}
    assert diag == {
        "diag_window_gap_mean",
        "diag_window_gap_std",
        "diag_teacher_perplexity",
        "diag_bank_norm",
        "diag_bank_worst_within_dist",
        "diag_bank_within_cos",
        "diag_bank_nearest_other_cos",
    }
    assert all(torch.isfinite(loss_dict[k]) and not loss_dict[k].requires_grad for k in diag)
    assert 1 <= loss_dict["diag_teacher_perplexity"] <= model.cfg.proto.n_prototypes
    # bank initialized normalized, KoLeo/coherence see the same bank
    torch.testing.assert_close(loss_dict["diag_bank_norm"], torch.tensor(1.0))


def test_second_step_uses_center():
    # the DINO / iBOT centers are updated by the prototype activations (and read on the next step)
    model = make_model(make_cfg())
    model.forward_backward(make_batch(0), teacher_temp=TEACHER_TEMP)
    model.dino_loss.apply_center_update()
    model.ibot_patch_loss.apply_center_update()
    assert model.dino_loss.center.shape == (1, 8) and model.dino_loss.center.abs().sum() > 0
    assert model.ibot_patch_loss.center.shape == (1, 1, 8) and model.ibot_patch_loss.center.abs().sum() > 0
    loss_dict = model.forward_backward(make_batch(1), teacher_temp=TEACHER_TEMP)
    assert all(torch.isfinite(v) for v in loss_dict.values())


def test_sinkhorn_knopp_centering(monkeypatch):
    # sinkhorn_knopp all_reduces the masked patch count, which needs a process group; single process it's a no-op
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    model = make_model(make_cfg(train={"centering": "sinkhorn_knopp"}))
    loss_dict = model.forward_backward(make_batch(), teacher_temp=TEACHER_TEMP)
    assert all(torch.isfinite(v) for v in loss_dict.values())


def test_bank_is_last_layer_with_weight_decay():
    model = make_model(make_cfg())
    head = model.student.dino_head
    assert isinstance(head, PrototypeHead) and isinstance(head.last_layer, PrototypeBank)
    groups = [g for g in model.get_params_groups() if any(p is head.last_layer.prototypes for p in g["params"])]
    assert len(groups) == 1
    # frozen for optim.freeze_last_layer_epochs, decayed, no layerwise lr decay
    assert groups[0]["is_last_layer"] and groups[0]["wd_multiplier"] == 1.0 and groups[0]["lr_multiplier"] == 1.0
    assert model.dino_out_dim == model.ibot_out_dim == 8


@pytest.mark.parametrize(
    "overrides",
    [{"ibot": {"separate_head": True}}, {"dino": {"koleo_loss_weight": 0.1}}, {"dino": {"loss_weight": 0}}],
)
def test_invalid_configs(overrides):
    with pytest.raises(ValueError):
        SSLMetaArch(make_cfg(**overrides))


def test_koleo_weight_override():
    model = make_model(make_cfg(proto={"koleo_loss_weight": 0.25}))
    loss_dict = model.forward_backward(make_batch(), teacher_temp=TEACHER_TEMP)
    bank = model.student.dino_head.last_layer
    torch.testing.assert_close(loss_dict["koleo_loss"], 0.25 * bank.koleo_loss())


def test_random_window():
    tokens = torch.arange(2 * 5 * 6).view(2, 30, 1)
    for _ in range(20):
        window = _random_window(tokens, (5, 6), (3, 2)).view(2, 3, 2)
        top, left = divmod(int(window[0, 0, 0]), 6)
        expected = tokens.view(2, 5, 6)[:, top : top + 3, left : left + 2]
        assert torch.equal(window, expected)
    # windows larger than the grid are clipped to it
    assert _random_window(tokens, (5, 6), (7, 7)).shape == (2, 30, 1)


def test_default_config_disabled():
    assert OmegaConf.create(dinov2_default_config).proto.enabled is False
