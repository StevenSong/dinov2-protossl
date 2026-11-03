# a DINOv2 ViT without the dinov2 package, for downstream (SupProto) use of ProtoSSL weights. parameter names match
# dinov2.models.vision_transformer.DinoVisionTransformer with flat blocks (blocks.{i}.*), so released DINOv2 checkpoints
# and ProtoSSL exports load strictly. no masking, register tokens, stochastic depth or xformers

import math

import torch
import torch.nn.functional as F
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, proj_bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).unbind(2)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        *,  # enforce kwargs
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        layerscale: float | None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias)
        self.ls1 = LayerScale(dim, layerscale) if layerscale else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), bias=ffn_bias)
        self.ls2 = LayerScale(dim, layerscale) if layerscale else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class VisionTransformer(nn.Module):
    """
    DINOv2 ViT, same outputs as DinoVisionTransformer.forward_features (eval mode, no xformers) for the same weights,
    including the positional embedding interpolation for other image sizes. parameters are not initialized for
    training from scratch, the model is meant to load weights
    """

    def __init__(
        self,
        *,  # enforce kwargs
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        in_chans: int = 3,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        layerscale: float | None = None,  # LayerScale init value, None or 0 = no LayerScale
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        grid = img_size // patch_size
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid + 1, embed_dim))
        # DINOv2's iBOT mask token, unused here. a buffer so checkpoints load strictly but DDP sees no unused parameter
        self.register_buffer("mask_token", torch.zeros(1, embed_dim))
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                layerscale=layerscale,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, n_patches: int, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        # as DinoVisionTransformer.interpolate_pos_encoding, including its shortcut and interpolate_offset workaround
        N = self.pos_embed.shape[1] - 1
        if n_patches == N and height == width:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        dim = pos_embed.shape[-1]
        h0, w0 = height // self.patch_size, width // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        if self.interpolate_offset:
            kwargs = {"scale_factor": ((h0 + self.interpolate_offset) / M, (w0 + self.interpolate_offset) / M)}
        else:
            kwargs = {"size": (h0, w0)}
        patch_pos_embed = F.interpolate(
            pos_embed[:, 1:].reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert patch_pos_embed.shape[-2:] == (h0, w0)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((pos_embed[:, :1], patch_pos_embed), dim=1).to(dtype)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        (B, C, H, W) images, H and W multiples of the patch size -> "x_norm_clstoken" (B, D), "x_norm_patchtokens"
        (B, N, D) in row-major (H // patch, W // patch) order, "x_prenorm" (B, 1 + N, D)
        """
        _, _, height, width = x.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"image size {height}x{width} is not a multiple of the patch size {self.patch_size}")
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x.shape[1] - 1, height, width, x.dtype)
        for block in self.blocks:
            x = block(x)
        x_norm = self.norm(x)
        return {"x_norm_clstoken": x_norm[:, 0], "x_norm_patchtokens": x_norm[:, 1:], "x_prenorm": x}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_features(x)
