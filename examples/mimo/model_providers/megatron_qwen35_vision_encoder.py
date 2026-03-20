# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron-native Qwen3.5 vision encoder for MIMO.

Implements the Qwen3.5 vision encoder (ViT + PatchMerger) with Megatron specs
and MegatronModule, replacing the HuggingFace wrapper.

Architecture exactly matches HuggingFace Qwen3_5MoeVisionModel:
  - 3D PatchEmbed (temporal_patch_size=2, patch_size=16)          → Qwen35VisionPatchEmbed
  - Learned 2D pos_embed with bilinear interpolation               → Qwen35VisionModel
  - 2D grid-based RoPE applied inside attention                    → Qwen35VisionRotaryEmbedding
  - 27 ViT blocks (pre-norm, full bidirectional attn, plain MLP)  → TransformerBlock via spec
  - PatchMerger (spatial_merge_size=2) → out_hidden_size           → Qwen35VisionPatchMerger

Key Megatron components used:
  - VisionModule / MegatronModule as base classes
  - TransformerBlock + TransformerLayer via ModuleSpec for the ViT blocks
  - ColumnParallelLinear / RowParallelLinear for all projection layers
  - TEDotProductAttention for FlashAttention with packed variable-length sequences
  - PackedSeqParams for packed-sequence (THD) attention format

Requirements:
  Transformer Engine (TE) must be installed for TEDotProductAttention, which
  supports the packed-sequence (THD) format needed for variable-length images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# Vision TransformerConfig
# ---------------------------------------------------------------------------


def get_qwen35_vision_transformer_config(
    params_dtype: torch.dtype = torch.bfloat16,
) -> TransformerConfig:
    """Return a TransformerConfig for the Qwen3.5 vision ViT (27 layers).

    All values match the HuggingFace Qwen3_5MoeVisionConfig defaults:
      depth=27, hidden_size=1152, num_heads=16, intermediate_size=4304,
      hidden_act='gelu_pytorch_tanh', bias=True everywhere.

    Note: ``apply_rope_fusion`` is disabled because we supply custom 2D
    grid-based rotary embeddings pre-computed outside the standard RoPE path.
    """
    cfg = TransformerConfig(
        num_layers=27,
        hidden_size=1152,
        num_attention_heads=16,
        kv_channels=72,             # head_dim = 1152 / 16
        num_query_groups=16,        # full MHA (not GQA)
        ffn_hidden_size=4304,
        add_bias_linear=True,       # all vision linear layers have bias
        normalization="LayerNorm",  # plain LN, not RMSNorm
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation_func=lambda x: F.gelu(x, approximate='tanh'),  # gelu_pytorch_tanh
        gated_linear_unit=False,
        apply_rope_fusion=False,    # custom 2D RoPE — disable TE/fused path
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        params_dtype=params_dtype,
    )
    return cfg


# ---------------------------------------------------------------------------
# Vision layer spec
# ---------------------------------------------------------------------------


def get_qwen35_vision_layer_spec() -> ModuleSpec:
    """Return a ModuleSpec for one Qwen3.5 vision ViT block.

    Each block is a pre-norm Transformer layer:
      input_layernorm → SelfAttention (no_mask, bias=True) → dropout+add
      pre_mlp_layernorm → MLP (GELU, bias=True) → dropout+add

    Uses non-TE ColumnParallelLinear/RowParallelLinear for the linear layers
    (preserving bias=True semantics) and TEDotProductAttention for
    FlashAttention with packed variable-length sequence (THD) support.
    """
    try:
        import apex  # noqa: F401
        from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
        LNImpl = FusedLayerNorm
    except ImportError:
        from megatron.core.transformer.torch_norm import WrappedTorchNorm
        LNImpl = WrappedTorchNorm

    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.transformer_layer import (
        TransformerLayer,
        TransformerLayerSubmodules,
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LNImpl,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.no_mask},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=RowParallelLinear,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
        ),
    )


# ---------------------------------------------------------------------------
# Submodule: PatchEmbed
# ---------------------------------------------------------------------------


class Qwen35VisionPatchEmbed(MegatronModule):
    """3D convolutional patch embedding for Qwen3.5 vision input.

    Converts flattened pixel patches (including temporal dimension for video)
    into token embeddings via a single 3D convolution, exactly as in
    ``Qwen3_5MoeVisionPatchEmbed``.

    Weight layout: ``proj`` is ``nn.Conv3d`` (no Megatron TP equivalent for
    3D conv, so it is replicated across TP ranks).

    Args:
        config: TransformerConfig; ``hidden_size`` is used as the embedding dim.
        patch_size: Spatial patch size (default 16).
        temporal_patch_size: Temporal patch size, i.e. frames merged per token
            (default 2).
        in_channels: Input channels (default 3 for RGB).
    """

    def __init__(
        self,
        config: TransformerConfig,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
    ) -> None:
        super().__init__(config=config)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


# ---------------------------------------------------------------------------
# Submodule: RoPE
# ---------------------------------------------------------------------------


class Qwen35VisionRotaryEmbedding(MegatronModule):
    """2D grid-based rotary positional embedding for Qwen3.5 vision ViT.

    Computes 2D sinusoidal position frequencies for (row, col) grid positions
    and returns raw angle values in the shape expected by Megatron's
    ``TransformerBlock`` rotary_pos_emb argument:
        ``[total_tokens, 1, 1, head_dim]``

    This exactly replicates ``Qwen3_5MoeVisionRotaryEmbedding`` +
    ``Qwen3_5MoeVisionModel.rot_pos_emb``.

    Args:
        config: TransformerConfig.
        theta: RoPE base frequency (default 10000.0).
    """

    def __init__(self, config: TransformerConfig, theta: float = 10000.0) -> None:
        super().__init__(config=config)
        self.head_dim = config.hidden_size // config.num_attention_heads  # 72
        self._rope_dim = self.head_dim // 2
        self._theta = theta

    def _get_inv_freq(self, device: torch.device) -> torch.Tensor:
        """Compute inv_freq on the given device (avoids meta-tensor issues)."""
        return 1.0 / (
            self._theta
            ** (torch.arange(0, self._rope_dim, 2, dtype=torch.float32, device=device) / self._rope_dim)
        )

    def _get_freq_table(self, seqlen: int, device: torch.device) -> torch.Tensor:
        """Frequency table for positions 0..seqlen-1.  Shape: [seqlen, head_dim//4]."""
        inv_freq = self._get_inv_freq(device)
        seq = torch.arange(seqlen, device=device, dtype=inv_freq.dtype)
        return torch.outer(seq, inv_freq)

    def get_pos_emb(
        self, grid_thw: torch.Tensor, spatial_merge_size: int = 2
    ) -> torch.Tensor:
        """Compute 2D vision RoPE for packed grid tokens.

        Replicates ``Qwen3_5MoeVisionModel.rot_pos_emb`` exactly.

        The tokens are enumerated in the same order as the patch merger:
        for each coarse block (merged_h × merged_w), all
        ``spatial_merge_size^2`` sub-pixels are contiguous.

        Args:
            grid_thw: Shape ``(N, 3)`` — temporal, height, width per image.
            spatial_merge_size: Spatial merge factor (default 2).

        Returns:
            emb: Shape ``[total_tokens, 1, 1, head_dim]`` — raw angle values
            compatible with Megatron's ``apply_rotary_pos_emb`` (THD path).
        """
        merge_size = spatial_merge_size
        device = grid_thw.device
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self._get_freq_table(max_hw, device=device)  # [max_hw, head_dim//4]

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h = height // merge_size
            merged_w = width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            # row_idx[br, bc, ir, ic] = br * merge_size + ir
            row_idx = (
                block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            )
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)  # [H*W, 2]
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        # freq_table[pos_ids]: [total_tokens, 2, head_dim//4]
        # flatten → [total_tokens, head_dim//2]
        embeddings = freq_table[pos_ids].flatten(1)
        # cat to full head_dim: [total_tokens, head_dim]
        emb = torch.cat((embeddings, embeddings), dim=-1)
        # reshape for Megatron THD RoPE: [T, 1, 1, head_dim]
        return emb[:, None, None, :]


# ---------------------------------------------------------------------------
# Submodule: PatchMerger
# ---------------------------------------------------------------------------


class Qwen35VisionPatchMerger(MegatronModule):
    """Spatial patch merger with projection to decoder hidden size.

    Replicates ``Qwen3_5MoeVisionPatchMerger`` (``use_postshuffle_norm=False``):
      1. LayerNorm per token (``config.hidden_size``, not merged size).
      2. Reshape: group ``spatial_merge_size^2`` adjacent tokens → merged vector.
      3. ColumnParallelLinear: ``merged_size → merged_size``.
      4. GELU activation.
      5. RowParallelLinear: ``merged_size → out_hidden_size``.

    Requires ``merged_size = hidden_size * spatial_merge_size^2`` to be
    divisible by the tensor-parallel size.

    Args:
        config: TransformerConfig; ``hidden_size`` is the per-token dim.
        spatial_merge_size: Spatial merge factor (default 2).
        out_hidden_size: Output projection dimension (default 4096).
    """

    def __init__(
        self,
        config: TransformerConfig,
        spatial_merge_size: int = 2,
        out_hidden_size: int = 4096,
    ) -> None:
        super().__init__(config=config)
        self.spatial_merge_size = spatial_merge_size
        merged_size = config.hidden_size * (spatial_merge_size ** 2)  # 4608
        self.merged_size = merged_size

        # LayerNorm on the per-token hidden dim (1152), not the merged dim.
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # FC1: merged_size → merged_size (column-parallel, no gather)
        self.linear_fc1 = ColumnParallelLinear(
            merged_size,
            merged_size,
            config=config,
            init_method=config.init_method,
            bias=True,
            gather_output=False,
            skip_bias_add=True,
        )
        self.act = nn.GELU()

        # FC2: merged_size → out_hidden_size (row-parallel, reduce at end)
        self.linear_fc2 = RowParallelLinear(
            merged_size,
            out_hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge and project tokens.

        Args:
            x: ``[total_tokens, hidden_size]`` pre-merge token embeddings.

        Returns:
            ``[total_tokens // spatial_merge_size^2, out_hidden_size]``
        """
        # Norm per token, then reshape to merge spatial_merge_size^2 tokens.
        x = self.norm(x)                      # [T, hidden_size]
        x = x.view(-1, self.merged_size)      # [T/4, merged_size]

        x, bias1 = self.linear_fc1(x)
        if bias1 is not None:
            x = x + bias1
        x = self.act(x)

        x, bias2 = self.linear_fc2(x)
        if bias2 is not None:
            x = x + bias2
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class Qwen35VisionModel(VisionModule):
    """Qwen3.5 vision encoder built entirely with Megatron specs and modules.

    Replicates ``Qwen3_5MoeVisionModel`` using Megatron primitives:
      - ``Qwen35VisionPatchEmbed``        (MegatronModule) — 3D patch extraction
      - ``nn.Embedding``                  — learned 2D position table
      - ``Qwen35VisionRotaryEmbedding``   (MegatronModule) — 2D grid RoPE
      - ``TransformerBlock`` via spec     — 27 bidirectional ViT layers
      - ``Qwen35VisionPatchMerger``       (MegatronModule) — spatial merging

    Forward inputs:
        hidden_states: flattened pixel patches fed to ``Qwen35VisionPatchEmbed``.
        grid_thw: ``(N, 3)`` tensor of ``(temporal, height, width)`` per image/video.

    Forward output:
        Merged token embeddings, shape ``[merged_tokens, out_hidden_size]``.

    Args:
        config: TransformerConfig for the ViT blocks.
        layer_spec: ``ModuleSpec`` for ``TransformerLayer``
            (e.g. from ``get_qwen35_vision_layer_spec()``).
        spatial_merge_size: Spatial merge factor (default 2).
        patch_size: ViT patch size (default 16).
        temporal_patch_size: Temporal patch size (default 2).
        in_channels: Input channels (default 3).
        num_position_embeddings: Learned position table size (default 2304,
            giving a 48×48 base grid).
        out_hidden_size: Merger output dimension (default 4096).
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_spec: ModuleSpec,
        spatial_merge_size: int = 2,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        num_position_embeddings: int = 2304,
        out_hidden_size: int = 4096,
    ) -> None:
        super().__init__(config=config)
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)  # 48

        # --- Patch embedding ---
        self.patch_embed = Qwen35VisionPatchEmbed(
            config=config,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
        )

        # --- Learned 2D position embeddings (bilinear-interpolated per forward) ---
        self.pos_embed = nn.Embedding(num_position_embeddings, config.hidden_size)

        # --- 2D RoPE for attention ---
        self.rotary_pos_emb_module = Qwen35VisionRotaryEmbedding(config=config)

        # --- ViT transformer blocks (27 layers) ---
        self.model_type = ModelType.encoder_or_decoder
        self.decoder = TransformerBlock(
            config=config,
            spec=layer_spec,
            pre_process=True,
            post_process=False,  # PatchMerger does its own LayerNorm
        )

        # --- Spatial patch merger ---
        self.merger = Qwen35VisionPatchMerger(
            config=config,
            spatial_merge_size=spatial_merge_size,
            out_hidden_size=out_hidden_size,
        )

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Support pipeline-parallel input injection."""
        self.decoder.set_input_tensor(input_tensor)

    # ------------------------------------------------------------------
    # Position-embedding helpers (ported 1-to-1 from HF)
    # ------------------------------------------------------------------

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of learned 2D position embeddings.

        Replicates ``Qwen3_5MoeVisionModel.fast_pos_embed_interpolate``
        exactly, including the spatial-merge reordering.

        Args:
            grid_thw: ``(N, 3)`` — temporal, height, width per image.

        Returns:
            ``[total_tokens, hidden_size]`` position embeddings.
        """
        grid_thw_list = grid_thw.tolist()
        grid_ts = [r[0] for r in grid_thw_list]
        grid_hs = [r[1] for r in grid_thw_list]
        grid_ws = [r[2] for r in grid_thw_list]
        device = grid_thw.device
        n = self.num_grid_per_side  # 48

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for _, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, n - 1, h)
            w_idxs = torch.linspace(0, n - 1, w)
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clip(max=n - 1)
            w_ceil = (w_floor + 1).clip(max=n - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            base_h = h_floor * n
            base_h_ceil = h_ceil * n
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        # Split by image and reorder tokens to match the spatial-merge layout
        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )
        merge_size = self.spatial_merge_size
        result = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)  # [t*h*w, d]
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            result.append(pos_embed)
        return torch.cat(result)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Encode image/video patches.

        Args:
            pixel_values: Raw pixel patches for the PatchEmbed.  Shape
                ``[total_patches, C * temporal_patch_size * patch_size^2]``.
            grid_thw: ``(N, 3)`` — temporal, height, width grid per
                image/video.

        Returns:
            Merged token embeddings,
            shape ``[merged_tokens, out_hidden_size]``.
        """
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        if grid_thw.ndim == 3:
            grid_thw = grid_thw.reshape(-1, grid_thw.shape[-1])

        # 1. Patch embedding → [total_tokens, hidden_size]
        hidden_states = self.patch_embed(pixel_values)

        # 2. Add bilinear-interpolated 2D position embeddings
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # 3. Precompute 2D RoPE: [total_tokens, 1, 1, head_dim] (raw angles)
        #    Passed as a single tensor; SelfAttention auto-duplicates it to
        #    (emb, emb) for Q and K.
        rotary_pos_emb = self.rotary_pos_emb_module.get_pos_emb(
            grid_thw, self.spatial_merge_size
        )

        # 4. Build PackedSeqParams for variable-length image attention (THD).
        #    cu_seqlens marks the start/end of each image's token sequence.
        seqlens = grid_thw[:, 1] * grid_thw[:, 2]          # H*W per image
        seqlens_rep = torch.repeat_interleave(seqlens, grid_thw[:, 0])  # × T frames
        cu_seqlens = F.pad(
            seqlens_rep.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0
        )
        max_seqlen = int(seqlens_rep.max().item())
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        )

        # 5. TransformerBlock expects [s, b, h].
        #    We use b=1 (single packed sequence); SelfAttention squeezes it
        #    back to [T, H, D] internally when qkv_format == 'thd'.
        hidden_states = hidden_states.unsqueeze(1)  # [T, 1, H]
        hidden_states = self.decoder(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        hidden_states = hidden_states.squeeze(1)    # [T, H]

        # 6. Spatial patch merger → [T/merge^2, out_hidden_size]
        return self.merger(hidden_states)
