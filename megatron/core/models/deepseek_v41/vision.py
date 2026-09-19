# Copyright (c) 2026 DeepSeek-AI.
# Adapted from deepseek-ai/DeepSeek-V4.1-Flash (MIT; see LICENSE.deepseek).
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Differentiable DeepSeek-ViT with device-aware 2D RoPE and 3x3 pixel unshuffle."""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.transformer.module import mark_keep_in_fp32


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float, device: torch.device):
    """Build device-local row/column rotary tables for a rectangular patch grid."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the vision tower's half-split rotary transform in FP32."""
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class RMSNorm(nn.Module):
    """Vision RMSNorm with FP32 statistics and scale parameters."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mark_keep_in_fp32(nn.Parameter(torch.ones(dim, dtype=torch.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over hidden channels and restore the input dtype."""
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    """Biasful linear projection of flattened RGB image patches."""

    def __init__(self, args) -> None:
        super().__init__()
        self.proj = nn.Linear(3 * args.vision_patch_size**2, args.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project each RGB patch into a vision token."""
        return self.proj(x.flatten(1))


class Attention(nn.Module):
    """Bidirectional per-image self-attention with two-dimensional RoPE."""

    def __init__(self, args) -> None:
        super().__init__()
        self.n_heads = args.vision_n_heads
        self.head_dim = args.vision_dim // args.vision_n_heads
        self.wqkv = nn.Linear(args.vision_dim, 3 * args.vision_dim)
        self.wo = nn.Linear(args.vision_dim, args.vision_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Attend over all patches in one image."""
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class MLP(nn.Module):
    """Bias-free SwiGLU feed-forward layer for the vision tower."""

    def __init__(self, args) -> None:
        super().__init__()
        self.w1 = nn.Linear(args.vision_dim, 2 * args.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(args.vision_inter_dim, args.vision_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the gated feed-forward branch."""
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class Block(nn.Module):
    """Pre-normalized attention and feed-forward vision block."""

    def __init__(self, args) -> None:
        super().__init__()
        self.norm1 = RMSNorm(args.vision_dim)
        self.attn = Attention(args)
        self.norm2 = RMSNorm(args.vision_dim)
        self.mlp = MLP(args)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply the two residual branches of the vision block."""
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, args) -> None:
        super().__init__()
        self.rope_dim = args.vision_dim // args.vision_n_heads // 2
        self.rope_theta = args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.vision_n_layers)])
        self.norm = RMSNorm(args.vision_dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """Encode a complete rectangular image patch grid."""
        if patches.shape[0] != n_h * n_w:
            raise ValueError("Patch count must equal the image grid area")
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    """Spatial pixel unshuffle followed by a two-layer GELU projector."""

    def __init__(self, args) -> None:
        super().__init__()
        self.downsample_ratio = args.vision_downsample_ratio
        in_dim = args.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, args.dim)
        self.w2 = nn.Linear(args.dim, args.dim)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """Pad incomplete spatial groups and return row-major language embeddings."""
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
