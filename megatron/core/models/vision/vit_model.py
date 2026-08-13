# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""
Native Megatron-Core Vision Transformer.

Provides ViTModel (Pixtral/CLIP), QwenVLViTModel (Qwen3.5-MoE VL), and
KimiViTModel (Kimi-K2). All use mcore TransformerBlock — no HuggingFace at runtime.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from megatron.core.extensions.transformer_engine import TENorm

    HAVE_TE = True
except ImportError:
    TENorm = None
    HAVE_TE = False

_NORM_IMPL = TENorm


class PatchEmbedding(nn.Module):
    """Conv2d patch extractor — no positional embedding."""

    def __init__(self, in_channels: int, hidden_size: int, patch_dim: int, bias: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_dim, stride=patch_dim, bias=bias
        )  # pylint: disable=line-too-long
        self.patch_dim = patch_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """(B, C, H, W) → (B, N, hidden), (h_patches, w_patches)"""
        x = self.proj(x.to(self.proj.weight.dtype))
        h_patches, w_patches = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, hidden)
        return x, h_patches, w_patches

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-patchified input -> (B, N, hidden)."""
        weight = self.proj.weight.flatten(1)
        return F.linear(x.to(weight.dtype), weight, self.proj.bias)


def _dynamic_patch_grid(
    imgs_sizes: Union[List[Tuple[int, int]], torch.Tensor], patch_dim: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(imgs_sizes):
        sizes = imgs_sizes.to(device=device, dtype=torch.int64)
    else:
        sizes = torch.tensor(imgs_sizes, device=device, dtype=torch.int64)
    patch_hw = sizes // patch_dim
    seq_lens = patch_hw[:, 0] * patch_hw[:, 1]
    return patch_hw, seq_lens


def _cat_rope(chunks: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(chunks, dim=0)


class Pixtral2DRotaryEmbedding(nn.Module):
    """
    2D RoPE for Mistral-native Pixtral-family vision encoders.

    Mistral/vLLM applies RoPE as complex multiplication over adjacent hidden
    dimension pairs.  This returns repeat-interleaved angles and must be used
    with ``rotary_interleaved=True`` in TransformerConfig.
    """

    def __init__(self, head_dim: int, max_patches_per_side: int, rope_theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_patches_per_side = max_patches_per_side

        # freq_dim = head_dim // 4 (half for h, half for w in the head_dim//2 space).
        freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, h_patches: int, w_patches: int, device: torch.device) -> torch.Tensor:
        """
        Returns freqs of shape (h_patches * w_patches, 1, 1, head_dim).
        This is the format mcore's apply_rotary_pos_emb expects for rotary_pos_emb.
        mcore handles cos/sin internally.
        """
        base_freqs = self.freqs.to(device=device)
        h = torch.arange(h_patches, device=device, dtype=base_freqs.dtype)
        w = torch.arange(w_patches, device=device, dtype=base_freqs.dtype)

        freqs_h = torch.outer(h, base_freqs[::2])  # (h_patches, head_dim//4)
        freqs_w = torch.outer(w, base_freqs[1::2])  # (w_patches, head_dim//4)

        angles = torch.cat(
            [
                freqs_h[:, None, :].expand(-1, w_patches, -1),
                freqs_w[None, :, :].expand(h_patches, -1, -1),
            ],
            dim=-1,
        ).reshape(-1, self.head_dim // 2)

        # MCore interleaved RoPE pairs adjacent dims, so each complex-pair angle
        # is repeated into the real and imaginary slots.
        freqs = angles.repeat_interleave(2, dim=-1)
        return freqs[:, None, None, :]  # (N, 1, 1, head_dim) — mcore rotary_pos_emb format


class PixtralLargePatchMerger(nn.Module):
    """Spatial 2×2 patch merger for Pixtral-Large.

    Applies RMSNorm on pre-merge tokens, folds 2×2 spatial blocks into a single
    4*hidden vector, then projects back to hidden_size with a bias-less Linear.
    Matches Mistral-Large-3's `pre_mm_projector_norm` + `patch_merger.merging_layer`.
    """

    def __init__(self, transformer_config: TransformerConfig, spatial_merge_size: int = 2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        hidden = transformer_config.hidden_size
        self.pre_norm = _NORM_IMPL(
            config=transformer_config, hidden_size=hidden, eps=transformer_config.layernorm_epsilon
        )
        self.linear_fc1 = nn.Linear(hidden * spatial_merge_size**2, hidden, bias=False)

    def forward(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """(B, h_p*w_p, H) → (B, h_p/m*w_p/m, H)."""
        m = self.spatial_merge_size
        B, _, H = x.shape
        h_out, w_out = h_patches // m, w_patches // m
        x = self.pre_norm(x)
        # Match vLLM PatchMerger/unfold order: each 2x2 block is flattened as
        # (hidden, merge_h, merge_w), not patch-major (merge_h, merge_w, hidden).
        x = x.reshape(B, h_out, m, w_out, m, H)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()  # (B, h_out, w_out, H, m, m)
        x = x.reshape(B, h_out * w_out, H * m * m)  # (B, N_out, 4H)
        return self.linear_fc1(x)  # (B, N_out, H)


class ViTModel(MegatronModule):
    """
    Generic Vision Transformer — native mcore, no transformers dependency.

    Supports:
      - Pixtral (2D RoPE, SwiGLU, RMSNorm, no CLS, no bias)
      - Pixtral-Large (same + 2×2 patch merger after the transformer stack)
      - CLIP/SigLIP (learned absolute pos, GELU, LayerNorm, CLS token)
        when add_class_token=True and pos_emb_type='learned_absolute'

    Args:
        transformer_config: Standard mcore TransformerConfig. Configure:
            - normalization, norm_epsilon
            - gated_linear_unit, activation_func
            - add_bias_linear
            - num_layers, hidden_size, num_attention_heads, ffn_hidden_size
        transformer_layer_spec: Layer spec for TransformerBlock (bidirectional).
        patch_dim: Patch size in pixels (16 for Pixtral, 14 for CLIP/SigLIP).
        img_h / img_w: Maximum image dimensions (controls RoPE table size).
        add_class_token: Prepend a learnable CLS token (CLIP style).
        class_token_len: Width of CLS token.
        ln_pre: Apply RMSNorm/LayerNorm before the transformer stack.
        ln_pre_eps: Epsilon for pre-transformer norm (defaults to norm_epsilon).
        pos_emb_type: 'rope2d' | 'learned_absolute' | 'none'.
        rope_theta: RoPE base frequency.
        use_merger: If True, append a Pixtral-Large-style 2×2 patch merger that
            reduces sequence length by 4 and keeps the output width at hidden_size.
        spatial_merge_size: Merger block size (default 2 → 2×2 → 4× reduction).
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        patch_dim: int = 16,
        img_h: int = 1024,
        img_w: int = 1024,
        in_channels: int = 3,
        patch_embed_bias: bool = False,
        add_class_token: bool = False,
        class_token_len: int = 0,
        ln_pre: bool = True,
        ln_pre_eps: Optional[float] = None,
        pos_emb_type: str = 'rope2d',
        rope_theta: float = 10000.0,
        use_merger: bool = False,
        spatial_merge_size: int = 2,
        pg_collection=None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=transformer_config)
        assert HAVE_TE, (
            "TransformerEngine is required to construct this model "
            "(TENorm is used throughout). Install megatron-core with "
            "transformer-engine."
        )

        self.patch_dim = patch_dim
        self.img_h = img_h
        self.img_w = img_w
        self.add_class_token = add_class_token
        self.class_token_len = class_token_len if add_class_token else 0
        self.pos_emb_type = pos_emb_type
        hidden_size = transformer_config.hidden_size

        # Patch embedding (state dict: patch_embed.proj.*)
        self.patch_embed = PatchEmbedding(
            in_channels, hidden_size, patch_dim, bias=patch_embed_bias
        )  # pylint: disable=line-too-long

        # Pre-transformer norm (matches Pixtral's ln_pre and CLIP's ln_pre)
        if ln_pre:
            # TENorm auto-dispatches to RMSNorm or LayerNorm based on config.normalization
            self.ln_pre = _NORM_IMPL(
                config=transformer_config,
                hidden_size=hidden_size,
                eps=ln_pre_eps if ln_pre_eps is not None else transformer_config.layernorm_epsilon,
            )
        else:
            self.ln_pre = None

        # CLS token (CLIP style)
        if add_class_token:
            self.class_token = nn.Parameter(torch.zeros(1, class_token_len, hidden_size))

        # Positional embedding
        max_patches = (img_h // patch_dim) * (img_w // patch_dim)
        if pos_emb_type == 'learned_absolute':
            self.position_embeddings = nn.Embedding(max_patches + self.class_token_len, hidden_size)
        elif pos_emb_type == 'rope2d':
            head_dim = hidden_size // transformer_config.num_attention_heads
            max_patches_per_side = max(img_h, img_w) // patch_dim
            self.rope = Pixtral2DRotaryEmbedding(head_dim, max_patches_per_side, rope_theta)
        # 'none': no positional encoding

        # Transformer
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        # Optional Pixtral-Large-style 2×2 patch merger after the transformer stack.
        # Keeps output width at hidden_size; sequence length shrinks by merge_size**2.
        self.spatial_merge_size = spatial_merge_size if use_merger else 1
        if use_merger:
            self.merger = PixtralLargePatchMerger(transformer_config, spatial_merge_size)
        else:
            self.merger = None
        # Read by llava_model to size the vision projection input.
        self.out_hidden_size = hidden_size

    @property
    def num_patches_per_image(self) -> int:
        """Return the number of vision patches for a single fixed-size image."""
        return (self.img_h // self.patch_dim) * (self.img_w // self.patch_dim)

    def set_input_tensor(self, input_tensor):
        """Set the input tensor for the decoder (pipeline-parallel entrypoint)."""
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        imgs_sizes=None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W)
            attention_mask: optional mask for TransformerBlock

        Returns:
            (B, N, hidden_size)  where N includes CLS tokens if add_class_token=True
        """
        dynamic_resolution = pixel_values.dim() == 3
        if dynamic_resolution:
            assert imgs_sizes is not None, "imgs_sizes is required for dynamic-resolution ViTModel"
            assert not self.add_class_token, "dynamic-resolution CLS handling is not implemented"
            patch_hw, seq_lens = _dynamic_patch_grid(
                imgs_sizes, self.patch_dim, pixel_values.device
            )  # pylint: disable=line-too-long
            x = self.patch_embed.forward_patches(pixel_values)
        else:
            B = pixel_values.shape[0]
            x, h_patches, w_patches = self.patch_embed(pixel_values)

        # 2. Pre-transformer norm
        if self.ln_pre is not None:
            x = self.ln_pre(x)

        # 3. Positional embedding
        rotary_pos_emb = None
        if self.pos_emb_type == 'learned_absolute':
            assert not dynamic_resolution, "learned absolute ViT positions are fixed-size only"
            pos = torch.arange(x.shape[1], device=pixel_values.device)
            if self.add_class_token:
                pos = pos + self.class_token_len
            x = x + self.position_embeddings(pos)
        elif self.pos_emb_type == 'rope2d':
            # Shape: (N, 1, 1, head_dim//2) — mcore rotary_pos_emb format
            if dynamic_resolution:
                rotary_pos_emb = _cat_rope(
                    [self.rope(int(h), int(w), pixel_values.device) for h, w in patch_hw.tolist()]
                )
            else:
                rotary_pos_emb = self.rope(h_patches, w_patches, pixel_values.device)

        # 4. CLS token
        if self.add_class_token:
            cls = self.class_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        # 5. TransformerBlock: expects (S, B, hidden)
        x = x.transpose(0, 1).contiguous()
        x = self.decoder(
            hidden_states=x,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        x = x.transpose(0, 1).contiguous()  # (B, N, hidden)

        # 6. Optional 2×2 patch merger (Pixtral-Large). Merger only runs on the
        #    patch tokens — CLS tokens (when present) would need separate handling,
        #    but the Pixtral-Large config has no CLS tokens so we skip that branch.
        if self.merger is not None:
            if dynamic_resolution:
                chunks = torch.split(x, seq_lens.tolist(), dim=1)
                x = torch.cat(
                    [
                        self.merger(chunk, int(h), int(w))
                        for chunk, (h, w) in zip(chunks, patch_hw.tolist())
                    ],
                    dim=1,
                )
            else:
                x = self.merger(x, h_patches, w_patches)

        return x


# ---------------------------------------------------------------------------
# Qwen3.5-MoE VL vision encoder
# ---------------------------------------------------------------------------


class QwenVL2DRotaryEmbedding(nn.Module):
    """2D RoPE for Qwen VL.

    Uses the same inv_freq for both H and W; concatenates row_freqs + col_freqs
    then duplicates to fill head_dim, matching Qwen3_5MoeVisionRotaryEmbedding
    + rot_pos_emb exactly.
    """

    def __init__(self, head_dim: int, max_patches_per_side: int, rope_theta: float = 10000.0):
        super().__init__()
        dim = head_dim // 2  # e.g. 36 for head_dim=72
        # inv_freq: (dim//2,) — same formula as Qwen3_5MoeVisionRotaryEmbedding
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.head_dim = head_dim
        self.max_patches_per_side = max_patches_per_side
        # Precompute the full frequency table once. This avoids a per-forward
        # `.item()` host-device sync (previously used to size the table
        # dynamically to the current input) and the accompanying rebuild.
        positions = torch.arange(max_patches_per_side, dtype=inv_freq.dtype)
        self.register_buffer("freq_table", torch.outer(positions, inv_freq), persistent=False)

    def forward(self, row_ids: torch.Tensor, col_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            row_ids: (N,) integer row indices of each patch, in [0, max_patches_per_side).
            col_ids: (N,) integer col indices of each patch, in [0, max_patches_per_side).
        Returns:
            freqs: (N, 1, 1, head_dim) — mcore rotary_pos_emb format (raw freqs, not cos/sin)
        """
        freq_table = self.freq_table.to(device=row_ids.device)
        row_freqs = freq_table[row_ids]  # (N, dim//2)
        col_freqs = freq_table[col_ids]  # (N, dim//2)
        freqs = torch.cat([row_freqs, col_freqs], dim=-1)  # (N, dim) = (N, head_dim//2)
        freqs = torch.cat((freqs, freqs), dim=-1)  # (N, head_dim)
        return freqs[:, None, None, :]  # (N, 1, 1, head_dim)


class QwenLearnedPosEmbed(nn.Module):
    """Bilinear-interpolatable learned 2D position embedding.

    Stores a (num_grid_per_side x num_grid_per_side) grid of vectors and
    interpolates bilinearly to the actual (h_patches, w_patches) resolution.
    Matches Qwen3_5MoeVisionModel.fast_pos_embed_interpolate.
    """

    def __init__(self, num_grid_per_side: int, hidden_size: int):
        super().__init__()
        self.num_grid_per_side = num_grid_per_side
        # Named 'weight' so state_dict key is 'pos_embed.weight', matching checkpoint.
        self.weight = nn.Parameter(torch.empty(num_grid_per_side * num_grid_per_side, hidden_size))
        nn.init.normal_(self.weight)

    def forward(self, h_patches: int, w_patches: int, device: torch.device) -> torch.Tensor:
        """Returns (h_patches * w_patches, hidden_size) bilinear-interpolated embeddings."""
        g = self.num_grid_per_side
        # Bilinear interpolation indices (matches fast_pos_embed_interpolate)
        h_idx = torch.linspace(0, g - 1, h_patches, device=device)
        w_idx = torch.linspace(0, g - 1, w_patches, device=device)

        h_floor = h_idx.long().clamp(0, g - 1)
        w_floor = w_idx.long().clamp(0, g - 1)
        h_ceil = (h_floor + 1).clamp(0, g - 1)
        w_ceil = (w_floor + 1).clamp(0, g - 1)

        W = self.weight.to(device=device)
        dh = (h_idx - h_floor.float()).to(W.dtype)  # (h_p,)
        dw = (w_idx - w_floor.float()).to(W.dtype)  # (w_p,)

        # 2D index grid
        idx_ff = (h_floor[:, None] * g + w_floor[None, :]).reshape(-1)  # (h*w,)
        idx_fc = (h_floor[:, None] * g + w_ceil[None, :]).reshape(-1)
        idx_cf = (h_ceil[:, None] * g + w_floor[None, :]).reshape(-1)
        idx_cc = (h_ceil[:, None] * g + w_ceil[None, :]).reshape(-1)

        # Bilinear weights
        w_ff = ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1, 1)
        w_fc = ((1 - dh)[:, None] * dw[None, :]).reshape(-1, 1)
        w_cf = (dh[:, None] * (1 - dw)[None, :]).reshape(-1, 1)
        w_cc = (dh[:, None] * dw[None, :]).reshape(-1, 1)

        emb = W[idx_ff] * w_ff + W[idx_fc] * w_fc + W[idx_cf] * w_cf + W[idx_cc] * w_cc
        return emb  # (h_p * w_p, hidden)


class QwenPatchMerger(nn.Module):
    """Spatial 2×2 patch grouping for Qwen VL.

    The mcore path stops at ``encoder_tokens()``, exposing pre-projector
    tokens. ``linear_fc1`` / ``linear_fc2`` are present only to accept
    source-checkpoint keys (``visual.merger.mlp.*``); they aren't traversed
    at runtime. Their forward pass is available via ``forward()`` for callers
    that want Qwen's full merger, but the mcore inference path doesn't call
    it, and DDP/FSDP consumers should treat these params as static (they
    receive no gradient in the mcore path).

    Reshapes spatially adjacent 2×2 patches into single vectors, applies
    LayerNorm, then projects to out_hidden_size via 2-layer MLP.
    Matches Qwen3_5MoeVisionPatchMerger (use_postshuffle_norm=False default).
    """

    def __init__(self, hidden_size: int, spatial_merge_size: int, out_hidden_size: int):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.merged_hidden = hidden_size * spatial_merge_size**2
        self.patch_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_hidden, self.merged_hidden)
        self.linear_fc2 = nn.Linear(self.merged_hidden, out_hidden_size)
        # The mcore encoder path stops at encoder_tokens(); these two linears
        # exist only to accept source-checkpoint weights (visual.merger.mlp.*)
        # for key-compatible loading. Freeze them so the optimizer doesn't
        # allocate state for weights that never see a gradient.
        for p in (*self.linear_fc1.parameters(), *self.linear_fc2.parameters()):
            p.requires_grad = False

    def encoder_tokens(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """
        Args:
            x: (B, h_patches * w_patches, hidden)
        Returns:
            (B, h_out * w_out, hidden * merge_size^2) before Qwen's merger MLP.
        """
        B, N, H = x.shape
        m = self.spatial_merge_size
        h_out, w_out = h_patches // m, w_patches // m

        # Apply norm before merging (matches HF default use_postshuffle_norm=False)
        x = self.patch_norm(x)  # (B, N, H)

        # Input is already in block-first order (h_out, w_out, m, m) — each group of
        # m*m consecutive tokens belongs to the same 2×2 spatial block.
        return x.reshape(B, h_out * w_out, m * m * H)  # (B, N_out, merged_hidden)

    def forward(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """Return Qwen's source-model visual merger output."""
        x = self.encoder_tokens(x, h_patches, w_patches)

        x = self.linear_fc2(F.gelu(self.linear_fc1(x), approximate='tanh'))
        return x


class QwenPatchEmbedding(nn.Module):
    """Patch embedding with .proj Conv3d, matching checkpoint key naming (patch_embed.proj.*)."""

    def __init__(
        self, in_channels: int, hidden_size: int, patch_dim: int, temporal_patch_size: int
    ):  # pylint: disable=line-too-long
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=[temporal_patch_size, patch_dim, patch_dim],
            stride=[temporal_patch_size, patch_dim, patch_dim],
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project pixel patches into the transformer hidden dimension."""
        return self.proj(x)

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-patchified (B, N, C*P*P) input -> (B, N, hidden)."""
        full_weight = self.proj.weight.flatten(1)
        if x.shape[-1] == full_weight.shape[-1]:
            return F.linear(x.to(full_weight.dtype), full_weight, self.proj.bias)

        collapsed_weight = self.proj.weight.sum(dim=2).flatten(1)
        if x.shape[-1] == collapsed_weight.shape[-1]:
            return F.linear(x.to(collapsed_weight.dtype), collapsed_weight, self.proj.bias)

        raise ValueError(
            "Qwen patch input width must match either native temporal patches "
            f"({full_weight.shape[-1]}) or collapsed image patches "
            f"({collapsed_weight.shape[-1]}), got {x.shape[-1]}"
        )


class QwenVLViTModel(MegatronModule):
    """Native mcore vision encoder for Qwen3.5-MoE VL.

    Accepts standard (B, C, H, W) pixel values. Internally performs block-first
    patch reordering so that spatial 2×2 merge groups are consecutive in the
    sequence — this matches Qwen's fast_pos_embed_interpolate ordering.

    Args:
        transformer_config: TransformerConfig with:
            normalization='LayerNorm', layernorm_epsilon=1e-6,
            add_bias_linear=True, gated_linear_unit=False,
            activation_func=gelu_tanh, apply_rope_fusion=False
        transformer_layer_spec: Layer spec with no_mask attention.
        patch_dim: Spatial patch size in pixels (16 for Qwen).
        temporal_patch_size: Temporal merge (2 for Qwen).
        img_h / img_w: Max image dims for RoPE table (default 768 → 48 patches).
        spatial_merge_size: Spatial downsampling in merger (2 for Qwen).
        out_hidden_size: Source Qwen merger MLP output dimension.
        num_pos_per_side: Learned pos embed grid size (48 for Qwen).
        rope_theta: RoPE base frequency.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        patch_dim: int = 16,
        temporal_patch_size: int = 2,
        img_h: int = 768,
        img_w: int = 768,
        in_channels: int = 3,
        spatial_merge_size: int = 2,
        out_hidden_size: int = 2048,
        num_pos_per_side: int = 48,
        rope_theta: float = 10000.0,
        pg_collection=None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=transformer_config)
        assert HAVE_TE, (
            "TransformerEngine is required to construct this model "
            "(TENorm is used throughout). Install megatron-core with "
            "transformer-engine."
        )
        self.patch_dim = patch_dim
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.class_token_len = 0  # no CLS token
        hidden_size = transformer_config.hidden_size
        self.source_out_hidden_size = out_hidden_size
        self.out_hidden_size = hidden_size * spatial_merge_size**2

        # Conv3d patch embedding (state dict: patch_embed.proj.*)
        self.patch_embed = QwenPatchEmbedding(
            in_channels, hidden_size, patch_dim, temporal_patch_size
        )  # pylint: disable=line-too-long

        # Learned absolute position embeddings (bilinear interpolatable)
        self.pos_embed = QwenLearnedPosEmbed(num_pos_per_side, hidden_size)

        # 2D RoPE
        head_dim = hidden_size // transformer_config.num_attention_heads
        max_patches_per_side = max(img_h, img_w) // patch_dim
        self.rope = QwenVL2DRotaryEmbedding(head_dim, max_patches_per_side, rope_theta)

        # Transformer
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        # Patch merger
        self.merger = QwenPatchMerger(hidden_size, spatial_merge_size, out_hidden_size)

    def set_input_tensor(self, input_tensor):
        """Set the input tensor for the decoder (pipeline-parallel entrypoint)."""
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        imgs_sizes=None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W)
        Returns:
            (B, h_out * w_out, hidden * spatial_merge_size^2)
        """
        m = self.spatial_merge_size
        device = pixel_values.device

        dynamic_resolution = pixel_values.dim() == 3
        if dynamic_resolution:
            assert (
                imgs_sizes is not None
            ), "imgs_sizes is required for dynamic-resolution QwenVLViTModel"  # pylint: disable=line-too-long
            native_patch_order = (
                pixel_values.shape[-1] == self.patch_embed.proj.weight.flatten(1).shape[-1]
            )
            patch_hw, seq_lens = _dynamic_patch_grid(imgs_sizes, self.patch_dim, device)
            x = self.patch_embed.forward_patches(pixel_values)
            chunks = torch.split(x, seq_lens.tolist(), dim=1)
        else:
            native_patch_order = False
            B, C, H, W = pixel_values.shape
            h_p, w_p = H // self.patch_dim, W // self.patch_dim

            # 1. Extract patches and apply Conv3d
            # (B, C, H, W) → (B*h_p*w_p, C, temporal, patch_dim, patch_dim)
            x = pixel_values.unfold(2, self.patch_dim, self.patch_dim).unfold(
                3, self.patch_dim, self.patch_dim
            )  # pylint: disable=line-too-long
            # x: (B, C, h_p, w_p, patch_dim, patch_dim)
            x = x.permute(
                0, 2, 3, 1, 4, 5
            ).contiguous()  # (B, h_p, w_p, C, p, p)  # pylint: disable=line-too-long
            x = x.reshape(B * h_p * w_p, C, self.patch_dim, self.patch_dim)  # (N, C, p, p)
            # Duplicate temporal: (N, C, temporal, p, p)
            x = x.unsqueeze(2).expand(-1, -1, self.temporal_patch_size, -1, -1).contiguous()
            x = self.patch_embed.proj(
                x.to(self.patch_embed.proj.weight.dtype)
            )  # (N, hidden, 1, 1, 1)  # pylint: disable=line-too-long
            x = x.reshape(B, h_p * w_p, -1)  # (B, N, hidden)
            patch_hw = torch.tensor([[h_p, w_p]], device=device, dtype=torch.int64)
            seq_lens = torch.tensor([h_p * w_p], device=device, dtype=torch.int64)
            chunks = [x]

        reordered = []
        row_id_chunks = []
        col_id_chunks = []
        for chunk, (h_p, w_p) in zip(chunks, patch_hw.tolist()):
            h_out, w_out = h_p // m, w_p // m
            pos_emb = self.pos_embed(h_p, w_p, device)
            if native_patch_order:
                pos_emb = pos_emb.reshape(h_out, m, w_out, m, -1)
                pos_emb = pos_emb.permute(0, 2, 1, 3, 4).contiguous()
                pos_emb = pos_emb.reshape(h_p * w_p, -1)
                reordered.append(chunk + pos_emb.unsqueeze(0))
            else:
                chunk = chunk + pos_emb.unsqueeze(0)
                chunk = chunk.reshape(chunk.shape[0], h_out, m, w_out, m, -1)
                chunk = chunk.permute(0, 1, 3, 2, 4, 5).contiguous()
                reordered.append(chunk.reshape(chunk.shape[0], h_p * w_p, -1))

            block_rows = torch.arange(h_out, device=device)
            block_cols = torch.arange(w_out, device=device)
            intra = torch.arange(m, device=device)
            row_id_chunks.append(
                (block_rows[:, None, None, None] * m + intra[None, None, :, None])
                .expand(h_out, w_out, m, m)
                .reshape(-1)
            )
            col_id_chunks.append(
                (block_cols[None, :, None, None] * m + intra[None, None, None, :])
                .expand(h_out, w_out, m, m)
                .reshape(-1)
            )

        x = torch.cat(reordered, dim=1)
        row_ids = torch.cat(row_id_chunks, dim=0)
        col_ids = torch.cat(col_id_chunks, dim=0)

        # 4. 2D RoPE
        rotary_pos_emb = self.rope(row_ids, col_ids)  # (N, 1, 1, head_dim)

        # 5. TransformerBlock: (S, B, hidden)
        x = x.transpose(0, 1).contiguous()
        x = self.decoder(
            hidden_states=x,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        x = x.transpose(0, 1).contiguous()  # (B, N, hidden)

        # 6. Patch merger boundary. For ablations, expose the projector-independent
        #    tensor before Qwen's source-model merger MLP.
        if dynamic_resolution:
            chunks = torch.split(x, seq_lens.tolist(), dim=1)
            x = torch.cat(
                [
                    self.merger.encoder_tokens(chunk, int(h), int(w))
                    for chunk, (h, w) in zip(chunks, patch_hw.tolist())
                ],
                dim=1,
            )
        else:
            h_p, w_p = patch_hw[0].tolist()
            x = self.merger.encoder_tokens(x, int(h_p), int(w_p))

        return x


# ---------------------------------------------------------------------------
# Kimi-K2 vision encoder
# ---------------------------------------------------------------------------


class Kimi2DRotaryEmbedding(nn.Module):
    """2D RoPE for Kimi-K2 (interleaved complex style).

    Alternates x-position and y-position frequencies for complex pairs:
    pairs (0,1), (4,5), ... get col (x) rotation;
    pairs (2,3), (6,7), ... get row (y) rotation.

    Used with rotary_interleaved=True in TransformerConfig so that mcore's
    _rotate_half correctly implements complex multiplication by e^(i*theta).
    """

    def __init__(self, head_dim: int, max_patches_per_side: int, rope_theta: float = 10000.0):
        super().__init__()
        # Kimi: arange(0, head_dim, 4)[:head_dim//4] with exponent / head_dim
        # For head_dim=72: 18 base freqs using [0,4,8,...,68]/72  (matches _precompute_freqs_cis)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 4).float() / head_dim)
        )  # (head_dim//4,)  # pylint: disable=line-too-long
        self.head_dim = head_dim
        self.max_patches_per_side = max_patches_per_side
        # Precompute full frequency table to avoid a per-forward .item() sync
        # and rebuild (previously used to size the table to current input).
        positions = torch.arange(max_patches_per_side, dtype=inv_freq.dtype)
        self.register_buffer("freq_table", torch.outer(positions, inv_freq), persistent=False)

    def forward(self, row_ids: torch.Tensor, col_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns freqs of shape (N, 1, 1, head_dim) for mcore interleaved RoPE.

        row_ids / col_ids must be in [0, max_patches_per_side).
        """
        freq_table = self.freq_table.to(device=row_ids.device)
        col_freqs = freq_table[col_ids]  # (N, head_dim//4) — x/col-position freqs
        row_freqs = freq_table[row_ids]  # (N, head_dim//4) — y/row-position freqs

        # Interleave: [col_f0, row_f0, col_f1, row_f1, ..., col_f17, row_f17]
        # matches HF freqs_cis layout [x_cis_0, y_cis_0, x_cis_1, y_cis_1, ...]
        angles = torch.stack([col_freqs, row_freqs], dim=-1).flatten(-2)  # (N, head_dim//2)

        # Duplicate pairs for mcore interleaved RoPE: freqs[2i] = freqs[2i+1] = angles[i]
        freqs = angles.repeat_interleave(2, dim=-1)  # (N, head_dim)
        return freqs[:, None, None, :]  # (N, 1, 1, head_dim)


class KimiLearned2DPosEmbed(nn.Module):
    """Bicubic-interpolatable learned 2D spatial position embedding for Kimi.

    Matches Kimi's Learnable2DInterpPosEmbDivided_fixed for static images (T=1).
    For video (T>1), sinusoidal temporal embeddings would be added; we skip that
    for the static-image parity test.
    """

    def __init__(self, height: int, width: int, hidden_size: int):
        super().__init__()
        self.height = height
        self.width = width
        self.weight = nn.Parameter(torch.empty(height, width, hidden_size))
        nn.init.normal_(self.weight)

    def forward(self, h_patches: int, w_patches: int, device: torch.device) -> torch.Tensor:
        """Returns (h_patches * w_patches, hidden_size)."""
        if h_patches == self.height and w_patches == self.width:
            return self.weight.reshape(-1, self.weight.shape[-1])
        # Bicubic interpolation: (H, W, C) → (1, C, H, W) → interpolate → flatten
        w = self.weight.to(device=device)
        x = w.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        x = F.interpolate(x, size=(h_patches, w_patches), mode='bicubic', align_corners=False)
        return x.squeeze(0).permute(1, 2, 0).reshape(-1, w.shape[-1])  # (h*w, C)


class KimiPatchMerger(nn.Module):
    """Spatial 2×2 patch grouping for Kimi.

    Groups spatially adjacent 2×2 patches onto a new axis without reducing
    them, giving (B, h_out * w_out, 4 * hidden). Matches tpool_patch_merger at
    T=1, where the temporal mean degenerates to a reshape; a T>1 variant would
    mean over the temporal axis here.
    """

    def forward(self, x: torch.Tensor, h_patches: int, w_patches: int) -> torch.Tensor:
        """
        Args:
            x: (B, h_patches * w_patches, hidden)
        Returns:
            (B, h_out * w_out, 4 * hidden)
        """
        B, N, H = x.shape
        h_out, w_out = h_patches // 2, w_patches // 2

        # Reshape to (B, h_out, 2, w_out, 2, H) → (B, h_out, w_out, 2, 2, H)
        x = x.reshape(B, h_out, 2, w_out, 2, H)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, h_out, w_out, 2, 2, H)
        # Temporal pooling (T=1, so this is just a reshape):
        # For T>1 we'd mean over T; here T=1 so just collapse T dim
        x = x.reshape(B, h_out * w_out, 4, H)  # (B, N_out, 4, hidden)
        return x


class KimiViTModel(MegatronModule):
    """Native mcore vision encoder for Kimi-K2.

    Accepts (B, C, H, W) pixel values. Internally:
    1. Conv2d patch embed (14×14)
    2. Learnable 2D spatial position embedding (bicubic interpolated from 64×64 grid)
    3. 2D RoPE with interleaved complex-style rotation (rotary_interleaved=True)
    4. 27-layer bidirectional TransformerBlock
    5. Final LayerNorm
    6. 2×2 spatial group merger (patch grouping, no reduction)

    Args:
        transformer_config: TransformerConfig with:
            normalization='LayerNorm', add_bias_linear=True,
            gated_linear_unit=False, activation_func=gelu_tanh,
            rotary_interleaved=True, apply_rope_fusion=False
        transformer_layer_spec: Layer spec with no_mask attention.
        patch_dim: Spatial patch size (14 for Kimi).
        img_h / img_w: Max image dims for RoPE table (default 896 → 64 patches).
        pos_embed_height / pos_embed_width: Learned pos embed grid size (64 for Kimi).
        rope_theta: RoPE base frequency (10000 for Kimi).
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        patch_dim: int = 14,
        img_h: int = 896,
        img_w: int = 896,
        in_channels: int = 3,
        pos_embed_height: int = 64,
        pos_embed_width: int = 64,
        rope_theta: float = 10000.0,
        pg_collection=None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config=transformer_config)
        assert HAVE_TE, (
            "TransformerEngine is required to construct this model "
            "(TENorm is used throughout). Install megatron-core with "
            "transformer-engine."
        )
        self.patch_dim = patch_dim
        hidden_size = transformer_config.hidden_size
        self.class_token_len = 0  # no CLS token
        self.out_hidden_size = 4 * hidden_size  # KimiPatchMerger concatenates 2×2 patch groups

        # Conv2d patch embedding (state dict: patch_embed.proj.*)
        self.patch_embed = PatchEmbedding(in_channels, hidden_size, patch_dim, bias=True)

        # Learnable 2D spatial position embedding
        self.pos_embed = KimiLearned2DPosEmbed(pos_embed_height, pos_embed_width, hidden_size)

        # 2D RoPE (interleaved complex style)
        head_dim = hidden_size // transformer_config.num_attention_heads
        max_patches_per_side = max(img_h, img_w) // patch_dim
        self.rope = Kimi2DRotaryEmbedding(head_dim, max_patches_per_side, rope_theta)

        # Transformer
        self.decoder = TransformerBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=True,
            post_process=False,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        # Final layer norm (Kimi has a post-transformer norm)
        self.final_ln = _NORM_IMPL(
            config=transformer_config,
            hidden_size=hidden_size,
            eps=transformer_config.layernorm_epsilon,
        )

        # Spatial patch merger
        self.merger = KimiPatchMerger()

    def set_input_tensor(self, input_tensor):
        """Set the input tensor for the decoder (pipeline-parallel entrypoint)."""
        self.decoder.set_input_tensor(input_tensor)

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        imgs_sizes=None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W)
        Returns:
            (B, h_out * w_out, 4 * hidden_size)
        """
        device = pixel_values.device

        dynamic_resolution = pixel_values.dim() == 3
        if dynamic_resolution:
            assert (
                imgs_sizes is not None
            ), "imgs_sizes is required for dynamic-resolution KimiViTModel"  # pylint: disable=line-too-long
            patch_hw, seq_lens = _dynamic_patch_grid(imgs_sizes, self.patch_dim, device)
            x = self.patch_embed.forward_patches(pixel_values)
            chunks = torch.split(x, seq_lens.tolist(), dim=1)
        else:
            x, h_p, w_p = self.patch_embed(pixel_values)
            patch_hw = torch.tensor([[h_p, w_p]], device=device, dtype=torch.int64)
            seq_lens = torch.tensor([h_p * w_p], device=device, dtype=torch.int64)
            chunks = [x]

        encoded = []
        row_id_chunks = []
        col_id_chunks = []
        for chunk, (h_p, w_p) in zip(chunks, patch_hw.tolist()):
            pos_emb = self.pos_embed(h_p, w_p, device)
            encoded.append(chunk + pos_emb.unsqueeze(0))
            row_id_chunks.append(
                torch.arange(h_p, device=device).unsqueeze(1).expand(h_p, w_p).reshape(-1)
            )
            col_id_chunks.append(
                torch.arange(w_p, device=device).unsqueeze(0).expand(h_p, w_p).reshape(-1)
            )
        x = torch.cat(encoded, dim=1)
        row_ids = torch.cat(row_id_chunks, dim=0)
        col_ids = torch.cat(col_id_chunks, dim=0)
        rotary_pos_emb = self.rope(row_ids, col_ids)  # (N, 1, 1, head_dim)

        # 4. TransformerBlock: (S, B, hidden)
        x = x.transpose(0, 1).contiguous()
        x = self.decoder(
            hidden_states=x,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        x = x.transpose(0, 1).contiguous()  # (B, N, hidden)

        # 5. Final layer norm
        x = self.final_ln(x)

        # 6. Spatial 2×2 group merger → (B, h_out*w_out, 4, hidden)
        if dynamic_resolution:
            chunks = torch.split(x, seq_lens.tolist(), dim=1)
            x = torch.cat(
                [
                    self.merger(chunk, int(h), int(w)).reshape(
                        chunk.shape[0], -1, 4 * chunk.shape[-1]
                    )  # pylint: disable=line-too-long
                    for chunk, (h, w) in zip(chunks, patch_hw.tolist())
                ],
                dim=1,
            )
            return x

        h_p, w_p = patch_hw[0].tolist()
        x = self.merger(x, int(h_p), int(w_p))
        B, N_out, _, hidden = x.shape
        return x.reshape(B, N_out, 4 * hidden)  # (B, h_out*w_out, 4*hidden)
