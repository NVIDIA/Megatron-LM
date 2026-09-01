# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-native DeepSeek-V4-Flash-Vision encoder and aligner."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from examples.multimodal_dev.models.deepseek_v4.configuration import build_aligner_permutation
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig


class DeepSeekV4VisionRMSNorm(nn.Module):
    """RMSNorm matching the official vision reference implementation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mark_keep_in_fp32(nn.Parameter(torch.ones(hidden_size, dtype=torch.float32)))

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Normalize the final dimension in float32 and restore input dtype."""
        dtype = hidden_states.dtype
        states = hidden_states.float()
        states = states * torch.rsqrt(states.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * states).to(dtype)


def get_vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Build DeepSeek's row/column 2D rotary tables for one image."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = (torch.stack((hpos, wpos), dim=-1).reshape(-1, 2, 1).float() * inv_freq).flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_vision_rotary(hidden_states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply DeepSeek's half-split rotary transform in float32."""
    dtype = hidden_states.dtype
    first, second = hidden_states.float().chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(dtype)


class DeepSeekV4VisionPatchEmbed(nn.Module):
    """Biasful linear projection over flattened RGB patches."""

    def __init__(self, patch_size: int, hidden_size: int, params_dtype: torch.dtype) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(3 * patch_size**2, hidden_size, bias=True, dtype=params_dtype)

    def forward(self, patches: Tensor) -> Tensor:
        """Project ``[num_patches, 3, patch, patch]`` or flattened patches."""
        patches = patches.flatten(1)
        return self.proj(patches.to(dtype=self.proj.weight.dtype))


class DeepSeekV4VisionAttention(nn.Module):
    """Full bidirectional vision attention with 2D RoPE."""

    def __init__(self, hidden_size: int, num_heads: int, params_dtype: torch.dtype) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}."
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True, dtype=params_dtype)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=True, dtype=params_dtype)

    def forward(self, hidden_states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply full self-attention to one image's patch sequence."""
        num_tokens = hidden_states.shape[0]
        query, key, value = (
            projection.view(num_tokens, self.num_heads, self.head_dim)
            for projection in self.wqkv(hidden_states).chunk(3, dim=-1)
        )
        query = apply_vision_rotary(query, cos, sin)
        key = apply_vision_rotary(key, cos, sin)
        output = F.scaled_dot_product_attention(
            query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        )
        return self.wo(output.transpose(0, 1).reshape(num_tokens, -1))


class DeepSeekV4VisionMLP(nn.Module):
    """Bias-free SwiGLU MLP used by the vision tower."""

    def __init__(self, hidden_size: int, intermediate_size: int, params_dtype: torch.dtype) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, 2 * intermediate_size, bias=False, dtype=params_dtype)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=params_dtype)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Apply the SwiGLU feed-forward network."""
        gate, up = self.w1(hidden_states).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class DeepSeekV4VisionBlock(nn.Module):
    """Pre-norm attention and MLP residual block."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        eps: float,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.norm1 = DeepSeekV4VisionRMSNorm(hidden_size, eps)
        self.attn = DeepSeekV4VisionAttention(hidden_size, num_heads, params_dtype)
        self.norm2 = DeepSeekV4VisionRMSNorm(hidden_size, eps)
        self.mlp = DeepSeekV4VisionMLP(hidden_size, intermediate_size, params_dtype)

    def forward(self, hidden_states: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Run one vision block."""
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cos, sin)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class DeepSeekV4VisionAligner(nn.Module):
    """Pad-and-unfold 3x3 aligner that projects ViT rows to decoder width."""

    def __init__(
        self,
        vision_hidden_size: int,
        language_hidden_size: int,
        downsample_ratio: int,
        params_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.downsample_ratio = downsample_ratio
        in_features = vision_hidden_size * downsample_ratio**2
        self.w1 = nn.Linear(in_features, language_hidden_size, bias=True, dtype=params_dtype)
        self.w2 = nn.Linear(
            language_hidden_size, language_hidden_size, bias=True, dtype=params_dtype
        )

    def forward(self, hidden_states: Tensor, n_h: int, n_w: int) -> Tensor:
        """Downsample one ``n_h x n_w`` patch grid and project it."""
        ratio = self.downsample_ratio
        grid = hidden_states.view(n_h, n_w, -1).permute(2, 0, 1)
        grid = F.pad(grid, (0, -n_w % ratio, 0, -n_h % ratio))
        rows = F.unfold(grid.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(rows)))


class DeepSeekV4VisionEncoder(VisionModule):
    """DeepSeek-V4 ViT plus aligner, with one full-attention segment per image."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config=config)
        self.patch_size = config.vision_patch_size
        self.downsample_ratio = config.vision_downsample_ratio
        self.rope_theta = config.vision_rope_theta
        self.rope_dim = config.kv_channels // 2

        self.patch_embed = DeepSeekV4VisionPatchEmbed(
            self.patch_size, config.hidden_size, config.params_dtype
        )
        self.blocks = nn.ModuleList(
            [
                DeepSeekV4VisionBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    intermediate_size=config.ffn_hidden_size,
                    eps=config.layernorm_epsilon,
                    params_dtype=config.params_dtype,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norm = DeepSeekV4VisionRMSNorm(config.hidden_size, config.layernorm_epsilon)
        self.aligner = DeepSeekV4VisionAligner(
            vision_hidden_size=config.hidden_size,
            language_hidden_size=config.vision_out_hidden_size,
            downsample_ratio=self.downsample_ratio,
            params_dtype=config.params_dtype,
        )

    def _encode_one_image(self, patches: Tensor, n_h: int, n_w: int) -> Tensor:
        hidden_states = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(
            n_h, n_w, self.rope_dim, self.rope_theta, hidden_states.device
        )
        for block in self.blocks:
            if self.training and self.config.recompute_granularity == "full":
                hidden_states = checkpoint(block, hidden_states, cos, sin, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, cos, sin)
        hidden_states = self.norm(hidden_states)
        aligned = self.aligner(hidden_states, n_h, n_w)
        permutation = build_aligner_permutation(
            n_h, n_w, self.downsample_ratio, device=aligned.device
        )
        return aligned.index_select(0, permutation)

    def forward(self, pixel_values: Tensor, image_grid_thw: Tensor) -> Tensor:
        """Encode concatenated image patches in batch/image encounter order.

        Args:
            pixel_values: Concatenated flattened patches ``[total_patches, 3*p*p]``.
            image_grid_thw: ``[num_images, 3]`` rows containing ``(1, H, W)``.

        Returns:
            Decoder-width image embeddings in official N-layout IMAGE-token order.
        """
        if image_grid_thw is None or image_grid_thw.ndim != 2:
            raise ValueError("image_grid_thw must be a [num_images, 3] tensor.")
        if image_grid_thw.shape[1] == 2:
            image_grid_hw = image_grid_thw
        elif image_grid_thw.shape[1] == 3:
            if not torch.all(image_grid_thw[:, 0] == 1):
                raise ValueError("DeepSeek-V4-Vision currently supports images, not video grids.")
            image_grid_hw = image_grid_thw[:, 1:]
        else:
            raise ValueError(
                f"image_grid_thw must have 2 or 3 columns, got {image_grid_thw.shape[1]}."
            )

        outputs = []
        offset = 0
        for grid in image_grid_hw:
            n_h, n_w = int(grid[0].item()), int(grid[1].item())
            num_patches = n_h * n_w
            image_patches = pixel_values[offset : offset + num_patches]
            if image_patches.shape[0] != num_patches:
                raise ValueError("pixel_values has fewer rows than image_grid_thw describes.")
            outputs.append(self._encode_one_image(image_patches, n_h, n_w))
            offset += num_patches
        if offset != pixel_values.shape[0]:
            raise ValueError(
                f"pixel_values has {pixel_values.shape[0]} rows, but grids consume {offset}."
            )
        if not outputs:
            return pixel_values.new_empty((0, self.config.vision_out_hidden_size))
        return torch.cat(outputs, dim=0)
