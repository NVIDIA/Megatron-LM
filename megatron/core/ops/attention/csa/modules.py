# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn

from megatron.core.fp8_utils import get_fp8_disabled_context
from megatron.core.models.common.embeddings import RotaryEmbedding, apply_rotary_pos_emb
from megatron.core.ops.attention.csa import kernel_metadata as csa_metadata
from megatron.core.ops.attention.csa.reference import (
    _compute_unfused_csa_non_compressed_lse as _compute_unfused_csa_non_compressed_lse,
)
from megatron.core.ops.attention.csa.reference import (
    _get_compress_causal_mask_cached as _get_compress_causal_mask_cached,
)
from megatron.core.ops.attention.csa.reference import (
    _get_compress_topk_idxs_cached as _get_compress_topk_idxs_cached,
)
from megatron.core.ops.attention.csa.reference import (
    _get_compress_valid_counts_cached as _get_compress_valid_counts_cached,
)
from megatron.core.ops.attention.csa.reference import (
    _get_window_topk_idxs_cached as _get_window_topk_idxs_cached,
)
from megatron.core.ops.attention.csa.reference import _pool_compressor_values
from megatron.core.ops.attention.csa.reference import (
    get_compress_topk_idxs as get_compress_topk_idxs,
)
from megatron.core.ops.attention.csa.reference import get_window_topk_idxs as get_window_topk_idxs
from megatron.core.ops.attention.csa.reference import (
    unfused_compressed_sparse_attn as unfused_compressed_sparse_attn,
)
from megatron.core.ops.attention.dsa.kernel_metadata import HADAMARD_ROTATION
from megatron.core.ops.attention.dsa.reference import FusedDSAIndexerLoss, fused_qk_topk_naive
from megatron.core.ops.attention.dsa.rotation import rotate_activation
from megatron.core.ops.attention.kernel_metadata import DSV4_ROPE
from megatron.core.ops.kernel_metadata import DeterminismPolicy, validate_kernel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.dsa_loss import DSAIndexerLossAutoScaler, DSAIndexerLossLoggingHelper
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import nvtx_range_pop, nvtx_range_push

CSA_OPERATION_DETERMINISM: dict[str, str] = csa_metadata.CSA_OPERATION_DETERMINISM

# ---------------------------------------------------------------------------
# Helper functions for RoPE
# ---------------------------------------------------------------------------


def _apply_rope(
    x: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    rotary_pos_emb_module: RotaryEmbedding,
    config: TransformerConfig,
    rotary_seq_len: int,
    ratio: int = 1,
    cp_group: torch.distributed.ProcessGroup = None,
) -> torch.Tensor:
    """Apply RoPE to the last ``qk_pos_emb_head_dim`` dims, leaving the rest unchanged.

    Accepts both 3-D ``[seq, batch, head_dim]`` and 4-D ``[seq, batch, heads, head_dim]``
    inputs.  When the input is 3-D a temporary head dimension is inserted for
    ``apply_rotary_pos_emb`` and removed before returning.
    """
    if ratio == 1:
        total_seq_len = rotary_seq_len
    else:
        total_seq_len = rotary_seq_len * ratio
    # DSv4 reference (DS-Inf) RoPE is pure rotation (norm-preserving). Yarn's
    # concentration factor (mscale) is NOT part of the DSv4 model contract --
    # the model relies on Q/KV RMS-norm + unit-magnitude rotation. Force 1.0
    # regardless of which rotary class is in use.
    mscale = 1.0
    rotary_pos_cos = None
    rotary_pos_sin = None
    if config.apply_rope_fusion:
        # ``mscale=1.0`` keeps the cached cos/sin free of yarn's
        # concentration factor so the fused kernel sees the same
        # rotation as the unfused split-rotate path (DSv4 "pure
        # rotation" contract).
        rotary_pos_cos, rotary_pos_sin = rotary_pos_emb_module.get_cached_cos_sin(
            total_seq_len, dtype=x.dtype, packed_seq=False, mscale=mscale
        )
        rotary_pos_emb = None
    else:
        # Compressed-attention callers instantiate ``YarnRotaryEmbedding``
        # whenever ``compress_ratio > 1`` (regardless of ``config.rope_type``);
        # its ``forward`` returns ``(emb, mscale)``. Base ``RotaryEmbedding``
        # returns a single tensor. Unpack either form uniformly; the
        # caller-side ``mscale=1.0`` keeps the yarn concentration factor
        # out of the rotation.
        result = rotary_pos_emb_module(total_seq_len, packed_seq=False)
        if isinstance(result, tuple):
            rotary_pos_emb = result[0]
        else:
            rotary_pos_emb = result
    if rotary_pos_emb is not None and ratio > 1:
        rotary_pos_emb = rotary_pos_emb[:total_seq_len:ratio][:rotary_seq_len]
    if rotary_pos_cos is not None and ratio > 1:
        rotary_pos_cos = rotary_pos_cos[:total_seq_len:ratio][:rotary_seq_len]
    if rotary_pos_sin is not None and ratio > 1:
        rotary_pos_sin = rotary_pos_sin[:total_seq_len:ratio][:rotary_seq_len]

    squeeze_head = x.dim() == 3
    if squeeze_head:
        x = x.unsqueeze(-2)
    if config.apply_rope_fusion:
        from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace

        out = fused_mla_rope_inplace(
            x,
            rotary_pos_cos,
            rotary_pos_sin,
            nope_dim,
            pos_dim,
            None,
            cp_group.rank(),
            cp_group.size(),
            remove_interleaving=True,
        )
    else:
        x_nope, x_pe = torch.split(x, [nope_dim, pos_dim], dim=-1)
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=cp_group,
            mla_rotary_interleaved=True,
            mla_output_remove_interleaving=True,
        )
        out = torch.cat([x_nope, x_pe], dim=-1)
    if squeeze_head:
        out = out.squeeze(-2)
    return out


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------


class CompressorInterface(Protocol):
    """Runtime interface exposed by a CSA compressor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        """Compress an input sequence."""
        ...

    def backward_dw(self) -> None:
        """Compute deferred weight gradients."""
        ...


class CompressorBuilder(Protocol):
    """Builder protocol for CSA compressors."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
        rotary_pos_emb: nn.Module | None = None,
        pg_collection: ProcessGroupCollection,
        name: str | None = None,
    ) -> CompressorInterface:
        """Build a CSA compressor."""
        ...


@dataclass
class CompressorSubmodules:
    """Submodule specs for CSA and HCA Compressor."""

    linear_wkv: Union[ModuleSpec, type] = None
    linear_wgate: Union[ModuleSpec, type] = None
    norm: Union[ModuleSpec, type] = None


class Compressor(MegatronModule):
    """Gated pooling compressor for CSA and HCA sparse attention.

    Compresses a sequence of tokens into a shorter sequence by pooling groups of
    ``compress_ratio`` tokens using learned gated weights.

    For ``compress_ratio == 4``, overlapping compression is used (``coff = 2``).
    For ``compress_ratio == 128``, non-overlapping compression is used (``coff = 1``).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CompressorSubmodules,
        compress_ratio: int,
        head_dim: int,
        rotate: bool = False,
        rotary_pos_emb: nn.Module | None = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            name (str | None): module instance name passed top-down from its parent module
        """
        super().__init__(config=config)

        if pg_collection is None:
            raise ValueError("Compressor requires an explicit ProcessGroupCollection")
        policy = DeterminismPolicy.WARN if config.deterministic_mode else DeterminismPolicy.IGNORE
        validate_kernel(csa_metadata.CSA_POOLING, determinism=policy)
        if config.apply_rope_fusion:
            validate_kernel(DSV4_ROPE, determinism=policy)
        if rotate:
            validate_kernel(HADAMARD_ROTATION, determinism=policy)
        self.pg_collection = pg_collection

        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.coff = 1 + int(self.overlap)
        self.rotate = rotate
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim

        self.rotary_pos_emb = rotary_pos_emb

        proj_out_dim = self.coff * head_dim

        with get_fp8_disabled_context(config, is_init=True):
            self.linear_wkv = build_module(
                submodules.linear_wkv,
                config.hidden_size,
                proj_out_dim,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
                name=(name + ".linear_wkv") if name is not None else None,
            )

            self.linear_wgate = build_module(
                submodules.linear_wgate,
                config.hidden_size,
                proj_out_dim,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
                name=(name + ".linear_wgate") if name is not None else None,
            )

        # keep to high precision (FP32 in the reference DeepSeek V4 checkpoint)
        _ape = torch.empty(
            compress_ratio, proj_out_dim, device=torch.cuda.current_device(), dtype=torch.float32
        )
        config.init_method(_ape)
        self.ape = mark_keep_in_fp32(nn.Parameter(_ape))

        norm_config = copy.copy(config)
        norm_config.normalization = "RMSNorm"
        self.norm = build_module(
            submodules.norm, config=norm_config, hidden_size=head_dim, eps=config.layernorm_epsilon
        )

    def backward_dw(self):
        """Compute deferred weight gradients for the compressor projections."""
        self.linear_wkv.backward_dw()
        self.linear_wgate.backward_dw()

    def _overlap_transform(self, tensor: torch.Tensor, fill_value: float = 0) -> torch.Tensor:
        """Apply overlapping window transform for 4x compression.

        Input shape:  [n_groups, ratio, b, coff * head_dim]
        Output shape: [n_groups, 2 * ratio, b, head_dim]
        """
        n_groups, ratio, b_dim, _ = tensor.size()
        d = self.head_dim
        new_tensor = tensor.new_full((n_groups, 2 * ratio, b_dim, d), fill_value)
        new_tensor[:, ratio:] = tensor[:, :, :, d:]
        new_tensor[1:, :ratio] = tensor[:-1, :, :, :d]
        return new_tensor

    def _project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project compressor values and gates outside any enclosing FP8 context."""
        with get_fp8_disabled_context(self.config):
            kv, _ = self.linear_wkv(x)
            score, _ = self.linear_wgate(x)
        return kv, score

    def forward(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Compress hidden states into shorter KV sequence.

        Determinism:
            Unknown. Bit-exact CUDA pooling and gradient reductions have not been certified.

        Args:
            x: [sq, b, hidden_size]

        Returns:
            compressed_kv [sq // ratio, b, head_dim] or None if too short.
        """
        nvtx_range_push("compressor")

        sq, b, _ = x.size()
        ratio = self.compress_ratio

        if sq < ratio:
            nvtx_range_pop("compressor")
            return None

        kv, score = self._project(x)  # [sq, b, coff * head_dim]

        cutoff = (sq // ratio) * ratio
        if cutoff < sq:
            kv = kv[:cutoff]
            score = score[:cutoff]

        n_compressed = cutoff // ratio

        # Reshape: [n_compressed, ratio, b, coff * head_dim]
        kv = kv.view(n_compressed, ratio, b, -1)
        score = score.view(n_compressed, ratio, b, -1)

        # APE: [ratio, coff * head_dim] -> [1, ratio, 1, coff * head_dim]
        score = score + self.ape.view(1, ratio, 1, -1)

        if self.overlap:
            kv = self._overlap_transform(kv, fill_value=0)
            score = self._overlap_transform(score, fill_value=float("-inf"))

        kv = _pool_compressor_values(kv, score, x.dtype)  # [n_compressed, b, head_dim]

        kv = self.norm(kv)

        kv = _apply_rope(
            kv,
            self.head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            n_compressed,
            ratio=ratio,
            cp_group=self.pg_collection.cp,
        )

        if self.rotate:
            kv = rotate_activation(kv)

        nvtx_range_pop("compressor")
        return kv  # [n_compressed, b, head_dim]


# ---------------------------------------------------------------------------
# CSAIndexer
# ---------------------------------------------------------------------------


class CSAIndexerInterface(Protocol):
    """Runtime interface exposed by a CSA indexer."""

    index_topk: int
    softmax_scale: float
    pg_collection: ProcessGroupCollection

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute indexer projections before top-k selection."""
        ...

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute index scores and selected indices."""
        ...

    def backward_dw(self) -> None:
        """Compute deferred weight gradients."""
        ...


class CSAIndexerBuilder(Protocol):
    """Builder protocol for CSA indexers."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        compress_ratio: int,
        rotary_pos_emb: nn.Module | None = None,
        pg_collection: ProcessGroupCollection,
        name: str | None = None,
    ) -> CSAIndexerInterface:
        """Build a CSA indexer."""
        ...


@dataclass
class CSAIndexerSubmodules:
    """Submodule specs for CSAIndexer."""

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None
    compressor: CompressorBuilder | None = None


class CSAIndexer(MegatronModule):
    """Learned top-k retrieval over compressed positions for CSA sparse attention.

    Computes index scores to select the most relevant compressed KV positions for each
    query.  Reuses the scoring logic from ``DSAIndexer`` (einsum -> relu -> weight -> sum
    -> topk) and ``rotate_activation`` (Hadamard transform) from ``dsa.py``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CSAIndexerSubmodules,
        compress_ratio: int,
        rotary_pos_emb: nn.Module | None = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            name (str | None): module instance name passed top-down from its parent module
        """
        super().__init__(config=config)

        if pg_collection is None:
            raise ValueError("CSAIndexer requires an explicit ProcessGroupCollection")
        if config.apply_rope_fusion:
            validate_kernel(
                DSV4_ROPE,
                determinism=(
                    DeterminismPolicy.WARN
                    if config.deterministic_mode
                    else DeterminismPolicy.IGNORE
                ),
            )
        validate_kernel(
            HADAMARD_ROTATION,
            determinism=(
                DeterminismPolicy.WARN if config.deterministic_mode else DeterminismPolicy.IGNORE
            ),
        )
        self.pg_collection = pg_collection

        self.compress_ratio = compress_ratio
        self.hidden_size = config.hidden_size
        self.qk_pos_emb_head_dim = config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size
        )

        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        self.rotary_pos_emb = rotary_pos_emb

        # Q projection
        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
            name=(name + ".linear_wq_b") if name is not None else None,
        )

        # The reference DeepSeek V4 checkpoint keeps this projection in BF16.
        with get_fp8_disabled_context(config, is_init=True):
            self.linear_weights_proj = build_module(
                submodules.linear_weights_proj,
                self.hidden_size,
                self.index_n_heads,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
                name=(name + ".linear_weights_proj") if name is not None else None,
            )

        # Own compressor (smaller head_dim, with Hadamard rotation)
        self.compressor = not_none(submodules.compressor)(
            config=config,
            compress_ratio=compress_ratio,
            head_dim=self.index_head_dim,
            rotate=True,
            rotary_pos_emb=rotary_pos_emb,
            pg_collection=pg_collection,
            name=(name + ".compressor") if name is not None else None,
        )

    def backward_dw(self):
        """Compute deferred weight gradients for the indexer projections."""
        self.linear_wq_b.backward_dw()
        self.linear_weights_proj.backward_dw()
        self.compressor.backward_dw()

    def _project_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Project indexer weights outside any enclosing FP8 context."""
        with get_fp8_disabled_context(self.config):
            weights, _ = self.linear_weights_proj(x)
        return weights

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q, compressed K, and weights before top-k selection."""
        nvtx_range_push("indexer_before_topk")

        sq, bsz, _ = x.size()

        # Q path
        q, _ = self.linear_wq_b(qr)  # [sq, b, n_heads * head_dim]
        q = q.reshape(sq, bsz, self.index_n_heads, self.index_head_dim)
        q = _apply_rope(
            q,
            self.index_head_dim - self.qk_pos_emb_head_dim,
            self.qk_pos_emb_head_dim,
            self.rotary_pos_emb,
            self.config,
            sq,
            ratio=1,
            cp_group=self.pg_collection.cp,
        )
        q = rotate_activation(q)

        # K path: own compressor
        k = not_none(apply_module(self.compressor)(x))  # [sq//ratio, b, index_head_dim]

        weights = self._project_weights(x)  # [sq, b, n_heads]
        weights = weights * (self.index_n_heads**-0.5)

        nvtx_range_pop("indexer_before_topk")
        return q, k, weights

    def forward(
        self, x: torch.Tensor, qr: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (index_scores, topk_indices)."""
        nvtx_range_push("indexer")
        q, k, weights = self.forward_before_topk(x, qr)
        nvtx_range_push("indexer_qk_topk")
        effective_topk = min(self.index_topk, k.size(0))
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, effective_topk, mask)
        nvtx_range_pop("indexer_qk_topk")
        nvtx_range_pop("indexer")
        return index_scores, topk_indices


# ---------------------------------------------------------------------------
# CompressedSparseAttention (core attention)
# ---------------------------------------------------------------------------


class CompressedSparseAttentionInterface(Protocol):
    """Runtime interface exposed by compressed sparse attention."""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        x: torch.Tensor | None = None,
        qr: torch.Tensor | None = None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: object | None = None,
    ) -> torch.Tensor:
        """Apply compressed sparse attention."""
        ...

    def backward_dw(self) -> None:
        """Compute deferred weight gradients."""
        ...


class CompressedSparseAttentionBuilder(Protocol):
    """Builder protocol for compressed sparse attention."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        softmax_scale: float | None,
        k_channels: int | None,
        v_channels: int | None,
        cp_comm_type: str | None,
        pg_collection: ProcessGroupCollection,
        rotary_pos_emb: nn.Module | None,
        compress_ratio: int,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> CompressedSparseAttentionInterface:
        """Build compressed sparse attention."""
        ...


@dataclass
class CompressedSparseAttentionSubmodules:
    """Submodule specs for CompressedSparseAttention."""

    compressor: CompressorBuilder | None = None
    indexer: CSAIndexerBuilder | None = None


class CompressedSparseAttention(MegatronModule):
    """Sparse core attention for CompressedSparseAttention.

    Combines sliding-window attention with compressed KV attention. The spec always
    provides compressor and indexer submodule specs; which are built depends on the
    ``compress_ratio`` passed by the caller:

    * ``ratio <= 1``: window-only (neither compressor nor indexer is built).
    * ``ratio > 1``: window + compressed KV via ``Compressor``.
    * ``ratio == 4`` and not ``config.csa_dense_mode``: additionally builds
      ``CSAIndexer`` for learned top-k retrieval over compressed positions. Otherwise,
      all causally valid compressed positions are attended.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CompressedSparseAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str | None = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
        rotary_pos_emb: nn.Module | None = None,
        compress_ratio: int = 0,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ):
        """
        Args:
            name (str | None): module instance name passed top-down from its parent module
        """
        super().__init__(config=config)

        if pg_collection is None:
            raise ValueError(
                "CompressedSparseAttention requires an explicit ProcessGroupCollection"
            )
        policy = DeterminismPolicy.WARN if config.deterministic_mode else DeterminismPolicy.IGNORE
        for kernel in csa_metadata.KERNELS:
            validate_kernel(kernel, determinism=policy)
        self.pg_collection = pg_collection

        tp_size = self.pg_collection.tp.size()
        if tp_size != 1:
            raise ValueError(
                "CompressedSparseAttention supports only tensor-parallel size 1 in the "
                f"native SBHD slice, got tp_size={tp_size}."
            )

        self.layer_number = layer_number + self.config.num_layers if is_mtp_layer else layer_number
        self.compress_ratio = compress_ratio
        self.window_size = config.csa_window_size
        self.v_head_dim = config.v_head_dim

        self.num_attention_heads = config.num_attention_heads

        if softmax_scale is None:
            softmax_scale = config.v_head_dim**-0.5
        self.softmax_scale = softmax_scale

        # Learnable attention sink per head, kept in reference-checkpoint FP32.
        self.attn_sink = mark_keep_in_fp32(
            nn.Parameter(torch.zeros(self.num_attention_heads, dtype=torch.float32))
        )

        # Conditionally build Compressor (ratio > 1)
        self.compressor: CompressorInterface | None
        if self.compress_ratio > 1 and submodules.compressor is not None:
            self.compressor = submodules.compressor(
                config=config,
                compress_ratio=self.compress_ratio,
                head_dim=config.v_head_dim,
                rotate=False,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
                name=(name + ".compressor") if name is not None else None,
            )
        else:
            self.compressor = None

        # Conditionally build Indexer (ratio == 4)
        self.indexer: CSAIndexerInterface | None
        if (
            self.compress_ratio == 4
            and not config.csa_dense_mode
            and submodules.indexer is not None
        ):
            self.indexer = submodules.indexer(
                config=config,
                compress_ratio=self.compress_ratio,
                rotary_pos_emb=rotary_pos_emb,
                pg_collection=pg_collection,
                name=(name + ".indexer") if name is not None else None,
            )
        else:
            self.indexer = None

    def backward_dw(self):
        """Compute deferred gradients for the optional compressor and indexer projections."""
        if self.compressor is not None:
            self.compressor.backward_dw()
        if self.indexer is not None:
            self.indexer.backward_dw()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        x: torch.Tensor | None = None,
        qr: torch.Tensor | None = None,
        attn_mask_type: AttnMaskType | None = None,
        attention_bias: torch.Tensor | None = None,
        packed_seq_params: object | None = None,
    ) -> torch.Tensor:
        """Forward pass for CompressedSparseAttention.

        Args:
            query:  [sq, b, np, v_head_dim]
            key:    [sq, b, 1, v_head_dim]  (single-head MQA; head dim squeezed internally)
            value:  unused (key == value in MQA)
            attention_mask: Must be None; causal masking is applied internally.
            x:      [sq, b, hidden_size]  original hidden states.
            qr:     [sq, b, q_lora_rank]  compressed query representation.

        Returns:
            output: [sq, b, np * v_head_dim]
        """
        if attention_mask is not None:
            raise ValueError(
                "CompressedSparseAttention supports only an implicit causal mask in the "
                "native SBHD slice; padding and document-boundary masks are not supported."
            )
        if query.ndim != 4:
            raise ValueError(
                "CompressedSparseAttention query must have shape [seq, batch, heads, dim], "
                f"got {tuple(query.shape)}."
            )
        if query.size(2) != self.num_attention_heads:
            raise ValueError(
                "CompressedSparseAttention query head count must match the unsharded "
                f"attention sink ({self.num_attention_heads}), got {query.size(2)}."
            )
        nvtx_range_push("compressed_sparse_attn")
        assert (
            packed_seq_params is None
        ), "Packed sequence not supported for CompressedSparseAttention"

        sq, b, np, hn = query.size()

        # --- Step 1: Prepare single-head KV (squeeze singleton head dim) ---
        kv = key.squeeze(-2)  # [sq, b, 1, v_head_dim] -> [sq, b, v_head_dim]

        # --- Step 2: Compression ---
        if self.compressor is not None and self.compress_ratio > 1:
            compressed_kv = apply_module(self.compressor)(
                not_none(x)
            )  # [n_compressed, b, v_head_dim]
            if compressed_kv is not None:
                kv_full = torch.cat([kv, compressed_kv], dim=0)
                n_compressed = compressed_kv.size(0)
            else:
                kv_full = kv
                n_compressed = 0
        else:
            kv_full = kv
            n_compressed = 0

        offset = sq  # compressed indices start after original positions

        # --- Step 3: Window indices ---
        window_idxs = get_window_topk_idxs(self.window_size, b, sq, query.device)

        # --- Step 4: Compressed indices ---
        indexer_loss = None

        if self.compress_ratio > 1 and n_compressed > 0:
            nvtx_range_push("compressed_indices")
            if self.indexer is not None:
                x_det = not_none(x).detach()
                qr_det = not_none(qr).detach()

                causal_mask = (
                    _get_compress_causal_mask_cached(
                        self.compress_ratio, sq, n_compressed, str(x_det.device)
                    )
                    .unsqueeze(0)
                    .expand(b, -1, -1)
                )

                if self.training and torch.is_grad_enabled():
                    q_indexer, k_indexer, weights_indexer = self.indexer.forward_before_topk(
                        x_det, qr_det
                    )
                    indexer_loss_coeff = self.config.dsa_indexer_loss_coeff or 0.0
                    key_for_loss = compressed_kv.unsqueeze(2).expand(-1, -1, np, -1)
                    weights_for_unfused = weights_indexer.float() * self.indexer.softmax_scale
                    non_compressed_lse = _compute_unfused_csa_non_compressed_lse(
                        query, kv, self.attn_sink, window_idxs, self.softmax_scale
                    )
                    # The native reference intentionally recomputes sliding-window
                    # logits in the final attention call below. The teacher needs its
                    # detached denominator before indexer top-k is available; sharing
                    # it without materializing the full window gather requires a larger
                    # data-flow change. TODO(#6404): its fused training backend avoids
                    # this duplicate, but the native fallback should share the gathered
                    # window KV/logits instead of retaining this correctness-first helper.
                    topk_indices_compressed, indexer_loss = FusedDSAIndexerLoss.apply(
                        q_indexer,
                        weights_for_unfused,
                        k_indexer,
                        query.detach(),
                        key_for_loss.detach(),
                        self.softmax_scale,
                        min(self.indexer.index_topk, n_compressed),
                        indexer_loss_coeff,
                        causal_mask,
                        self.config.dsa_indexer_use_sparse_loss,
                        self.indexer.pg_collection,
                        None,
                        None,
                        None,
                        None,
                        self.config.calculate_per_token_loss,
                        True,
                        non_compressed_lse,
                    )
                    if indexer_loss_coeff > 0:
                        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                            loss=indexer_loss,
                            layer_number=self.layer_number,
                            num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
                        )
                else:
                    _, topk_indices_compressed = apply_module(self.indexer)(
                        x_det, qr_det, mask=causal_mask
                    )

                n_valid_per_pos = _get_compress_valid_counts_cached(
                    self.compress_ratio, sq, str(x_det.device)
                )
                valid = (topk_indices_compressed >= 0) & (topk_indices_compressed < n_valid_per_pos)
                compress_topk_idxs = torch.where(valid, topk_indices_compressed + offset, -1)
            else:
                compress_topk_idxs = get_compress_topk_idxs(
                    self.compress_ratio, b, sq, offset, query.device
                )

            topk_idxs = torch.cat([window_idxs, compress_topk_idxs], dim=-1)
            nvtx_range_pop("compressed_indices")
        else:
            topk_idxs = window_idxs

        topk_idxs = topk_idxs.int()

        # --- Step 5: Sparse attention ---
        nvtx_range_push("sparse_attn_kernel")
        output = unfused_compressed_sparse_attn(
            query, kv_full, self.attn_sink.float(), topk_idxs, self.softmax_scale
        )
        nvtx_range_pop("sparse_attn_kernel")

        # --- Step 6: Attach indexer loss ---
        if indexer_loss is not None and self.training and torch.is_grad_enabled():
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        nvtx_range_pop("compressed_sparse_attn")
        return output
