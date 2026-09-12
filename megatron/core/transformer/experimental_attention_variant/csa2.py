# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""V4.1 CSA2 training math with explicit, forward-local cross-layer sharing.

This uses the existing DSv4 attention wrapper, sparse attention kernels and native reference.
Ordinary indexer Top-K and chunked candidate scores also reuse DSv4 kernels.
Candidate-aware Reindex and indexer loss use indexed kernels. The r2 compressor uses
BF16 projections and cuDNN gated pooling, retaining the existing Linear and RMSNorm modules.
Shared graph tensors belong to one stack forward and are passed explicitly; modules retain
no activation state or inference caches.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
    _apply_rope,
    _compute_unfused_csa_non_compressed_lse,
    get_window_topk_idxs,
    get_window_topk_idxs_thd,
    unfused_compressed_sparse_attn,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import thd_layout_kernels
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
    CSA2CandidateBlocks,
    candidate_blocks_from_scores,
    fused_candidate_blocks,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_indexer import (
    CSA2IndexerInputs,
    fused_candidate_topk,
    fused_csa2_indexer_loss,
    fused_csa2_indexer_sparse_attn,
    prepare_csa2_indexer_inputs,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.fused_compressor import (
    maybe_compress_csa2_thd_fused,
    maybe_pool_csa2_r2_fused,
    maybe_prepare_csa2_r2_fused,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention import (
    _compact_flat_topk_idxs,
    _indexer_topk_core,
    build_flat_topk_idxs,
    csa_sparse_attn,
    get_flash_mla_topk_alignment,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
    CSA2THDCompressionLayout,
    CSA2THDLayout,
    build_csa2_thd_layout,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
    compute_dsa_indexer_loss,
)
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    use_fused_dsa_kernels,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import get_pg_size


@dataclass
class CSA2State:
    """Shared attention tensors for one full-sequence stack forward.

    Source layer IDs are zero-based, matching the configuration. ``global_indices`` uses
    compressed positions (per-batch for SBHD, physical packed rows for THD, and -1 for
    unavailable positions), never offsets into a consumer's concatenated local/global KV
    tensor. ``candidates`` keeps sequence-local block IDs and counts, without a dense mask.
    Float tensors retain their autograd graph,
    so every consumer contributes to its Full owner's gradient. Create a fresh state for each
    forward, including when another microbatch's backward is still outstanding.

    The fused path packs global/indexer K once per Full owner. ``fused_indices``
    contains compact physical addresses for batch-major local+global KV, with
    ``fused_topk_length`` selecting each row's valid prefix. Reuse consumers share
    both buffers; a new index source invalidates them. They are immutable and
    belong to this forward, just like the logical sharing tensors above.
    """

    global_kv: torch.Tensor | None = None
    indexer_k: torch.Tensor | None = None
    global_kv_flat: torch.Tensor | None = None
    indexer_k_flat: torch.Tensor | None = None
    fused_indices: torch.Tensor | None = None
    fused_topk_length: torch.Tensor | None = None
    fused_q_padding_mask: torch.Tensor | None = None
    fused_window_size: int | None = None
    # Packed windows depend only on the layout and window size, so Full/Reindex
    # sources can share this optional loss buffer independently of their Top-K.
    fused_window_indices: torch.Tensor | None = None
    global_indices: torch.Tensor | None = None
    candidates: CSA2CandidateBlocks | None = None
    kv_source_layer: int | None = None
    index_source_layer: int | None = None
    candidate_source_layer: int | None = None
    sequence_length: int | None = None
    batch_size: int | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    last_layer: int | None = None
    thd_layout: CSA2THDLayout | None = None
    compressed_layout: CSA2THDCompressionLayout | None = None

    def validate_forward(
        self, layer_idx: int, query: torch.Tensor, *, thd_layout: CSA2THDLayout | None = None
    ) -> None:
        """Reject reuse across forwards or incompatible sequence/batch layouts."""
        if self.last_layer is not None and layer_idx <= self.last_layer:
            raise ValueError(
                "CSA2State layers must run in increasing order; create a fresh CSA2State "
                "for each stack forward."
            )
        if thd_layout is not None:
            if query.ndim != 3 or query.shape[0] != thd_layout.total_tokens:
                raise ValueError(
                    "CSA2 THD queries must have shape [total_tokens, heads, head_dim]."
                )
            if query.device != thd_layout.valid_tokens.device:
                raise ValueError("CSA2 THD queries and layout must be on the same device.")
            sequence_length, batch_size = query.shape[0], 1
        else:
            sequence_length, batch_size = query.shape[:2]
        if self.sequence_length is None:
            self.sequence_length, self.batch_size = sequence_length, batch_size
            self.device, self.dtype = query.device, query.dtype
            self.thd_layout = thd_layout
        elif (sequence_length, batch_size, query.device, query.dtype) != (
            self.sequence_length,
            self.batch_size,
            self.device,
            self.dtype,
        ):
            raise ValueError(
                "CSA2State sequence length, batch size, device and dtype must match this "
                "forward; create a fresh CSA2State for each stack forward."
            )
        elif (self.thd_layout is None) != (thd_layout is None):
            raise ValueError("CSA2State cannot mix packed and unpacked layouts in one forward.")
        elif thd_layout is not None and self.thd_layout is not thd_layout:
            self.thd_layout.validate_layout(thd_layout)


def apply_csa2_thd_rope(
    x: torch.Tensor,
    rotary_pos_emb: nn.Module,
    config: MLATransformerConfig,
    layout: CSA2THDLayout | CSA2THDCompressionLayout,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Apply DSv4's segmented RoPE to token or compressed rows, excluding padding.

    Token positions restart at zero in every sequence; compressed row ``j`` uses
    position ``ratio * j``. Physical prefixes describe any reserved padding rows.
    Masking before rotation also gives fused RoPE private storage, so the main and
    indexer branches cannot overwrite their shared pre-RoPE latent.
    """
    if get_pg_size(cp_group) != 1:
        raise NotImplementedError("CSA2 THD RoPE currently requires CP=1.")
    compressed = isinstance(layout, CSA2THDCompressionLayout)
    valid = layout.valid_groups if compressed else layout.valid_tokens
    if x.ndim not in (3, 4) or x.shape[0] != valid.shape[0] or (x.ndim == 4 and x.shape[1] != 1):
        raise ValueError("CSA2 THD RoPE expects one row per token or compressed capacity slot.")
    if x.device != valid.device:
        raise ValueError("CSA2 THD RoPE input and layout must be on the same device.")
    mask = ~valid.reshape(-1, *([1] * (x.ndim - 1)))
    x = x.masked_fill(mask, 0)
    # All sequences can be shorter than the compression ratio. In that case
    # even a nonempty static capacity has no usable frequency-table entries.
    if x.shape[0] == 0 or layout.max_seqlen == 0 or layout.cu_seqlens_padded.numel() < 2:
        return x
    ratio = layout.ratio if compressed else 1
    pos_dim = config.qk_pos_emb_head_dim
    output = _apply_rope(
        x,
        nope_dim=x.shape[-1] - pos_dim,
        pos_dim=pos_dim,
        rotary_pos_emb_module=rotary_pos_emb,
        config=config,
        rotary_seq_len=0,
        ratio=ratio,
        cp_group=cp_group,
        cu_seqlens=layout.cu_seqlens_padded,
        max_seqlen_rope=layout.max_seqlen * ratio,
    )
    return output.masked_fill(mask, 0)


def select_candidate_blocks(
    logits: torch.Tensor, compress_lens: torch.Tensor | int, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Select block-max candidates, always including the newest visible position's block.

    ``logits`` has shape ``[..., global positions]`` and is already causally masked.
    ``compress_lens`` broadcasts against its leading dimensions with a trailing singleton
    dimension. The result is a position mask; the caller still applies the causal score mask
    because a selected block can include positions after the query.
    """
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("CSA2 candidate block count and size must be positive.")
    width = logits.shape[-1]
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.shape[-1]
    newest_block = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == newest_block, torch.inf
    )
    # ReLU index scores often tie at zero. Break ties by the earlier local
    # block so changing packed capacity cannot change which real blocks win.
    top_indices = scores.argsort(dim=-1, descending=True, stable=True)[
        ..., : min(topk_blocks, num_blocks)
    ]
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter(
        -1, top_indices, scores.gather(-1, top_indices) > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_candidate_blocks_thd(
    scores: torch.Tensor,
    layout: CSA2THDLayout,
    compressed_layout: CSA2THDCompressionLayout,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Apply CSA2 block selection in each packed sequence's local group coordinates."""
    if scores.shape[-1] == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    sequence_ids = layout.sequence_ids.clamp_min(0)
    starts = compressed_layout.cu_seqlens_padded[sequence_ids].long()
    ends = compressed_layout.cu_seqlens_padded[sequence_ids + 1].long()
    local_positions = torch.arange(compressed_layout.max_seqlen, device=scores.device)
    physical_positions = starts[:, None] + local_positions
    assigned = physical_positions < ends[:, None]
    safe_positions = physical_positions.clamp(max=scores.shape[-1] - 1)
    local_scores = scores.gather(1, safe_positions).masked_fill(~assigned, -torch.inf)
    # Force the latest visible group's block, excluding the physical padding
    # after each logical sequence. Rows without any visible group stay empty.
    visible = torch.where(local_scores.isfinite(), local_positions + 1, 0).amax(
        dim=-1, keepdim=True
    )
    keep = select_candidate_blocks(local_scores, visible, topk_blocks, block_size)
    keep = keep & local_scores.isfinite()
    # Clamping unused local capacity can repeat a physical column; accumulate
    # instead of letting a padded False overwrite an earlier selected True.
    return (
        torch.zeros_like(scores, dtype=torch.int32).scatter_add_(1, safe_positions, keep.int()) > 0
    )


class CSA2Compressor(MegatronModule):
    """Produce pre-RoPE global latents from complete, non-overlapping token groups."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: CompressorSubmodules,
        compress_ratio: int,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__(config=config)
        if compress_ratio not in (1, 2):
            raise ValueError("CSA2 compression requires ratio 1 or 2.")
        self.compress_ratio = compress_ratio
        self.tp_group = pg_collection.tp
        self.use_fused_compressor = use_fused_dsa_kernels(config)
        linear_kwargs = dict(
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_wkv = build_module(
            submodules.linear_wkv, config.hidden_size, config.v_head_dim, **linear_kwargs
        )
        self.linear_wgate = None
        if compress_ratio == 2:
            self.linear_wgate = build_module(
                submodules.linear_wgate, config.hidden_size, config.v_head_dim, **linear_kwargs
            )
        self.norm = build_module(
            submodules.norm,
            config=config,
            hidden_size=config.v_head_dim,
            eps=config.attention_latent_norm_epsilon,
        )

    def forward(
        self,
        x: torch.Tensor,
        packed_seq_params: PackedSeqParams | None = None,
        *,
        thd_layout: CSA2THDLayout | None = None,
        input_is_sanitized: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, CSA2THDCompressionLayout]:
        """Compress SBHD input or return THD latents with their physical layout.

        Packed inputs keep the usual ``[total_tokens, 1, hidden]`` shape. Their
        result uses DSv4's static compressed capacity, including zero-valued
        padding slots; ``layout.valid_groups`` identifies the complete real groups.
        A caller sharing an existing layout can pass ``thd_layout`` directly.
        ``input_is_sanitized`` lets attention reuse its already cleaned, finite
        hidden buffer for projection. Standalone callers leave it false so
        unused tokens, including incomplete groups, are cleared before Linear.
        """
        if packed_seq_params is not None or thd_layout is not None:
            if x.ndim != 3 or x.shape[1] != 1:
                raise ValueError("CSA2 THD compressor expects [total_tokens, 1, hidden].")
            if thd_layout is None:
                thd_layout = build_csa2_thd_layout(packed_seq_params, x.shape[0])
            elif packed_seq_params is not None:
                thd_layout.validate_compatible(packed_seq_params, x.shape[0])
            if x.shape[0] != thd_layout.total_tokens or x.device != thd_layout.valid_tokens.device:
                raise ValueError(
                    "CSA2 THD compressor input must match the layout's rows and device."
                )
            return self._forward_thd(
                x,
                thd_layout.for_compression(self.compress_ratio),
                thd_layout.cu_seqlens_padded,
                input_is_sanitized=input_is_sanitized,
            )
        return self._forward_sbhd(x)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Keep the original Megatron Linear graph and its precision contract."""
        latent, _ = self.linear_wkv(x)
        gate = self.linear_wgate(x)[0] if self.linear_wgate is not None else None
        return latent, gate

    def _pool_r2(
        self,
        latent: torch.Tensor,
        gate: torch.Tensor,
        dtype: torch.dtype,
        valid_groups: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool BF16 projections with cuDNN, using FP32 intermediates in both paths."""
        pooled = maybe_pool_csa2_r2_fused(
            latent, gate, dtype, valid_groups=valid_groups, enabled=self.use_fused_compressor
        )
        if pooled is not None:
            return pooled
        shape = (latent.shape[0] // 2, 2, *latent.shape[1:])
        return (
            (latent.float().reshape(shape) * gate.float().reshape(shape).softmax(dim=1))
            .sum(dim=1)
            .to(dtype)
        )

    def _forward_sbhd(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[sequence, batch, hidden]`` to ``[sequence // ratio, batch, head_dim]``."""
        if self.compress_ratio == 1:
            latent, _ = self._project(x)
        else:
            # An incomplete final group remains available through the sliding window only.
            complete_length = x.shape[0] // self.compress_ratio * self.compress_ratio
            xf = x[:complete_length]
            latent, gate = self._project(xf)
            latent = self._pool_r2(latent, gate, x.dtype)
        latent = latent.to(x.dtype)
        if latent.shape[0] == 0:
            # TE RMSNorm cannot normalize zero rows. Keep its zero weight gradient in
            # the graph when the sequence has no complete compression group.
            return latent * self.norm.weight
        return self.norm(latent)

    def _forward_thd(
        self,
        x: torch.Tensor,
        layout: CSA2THDCompressionLayout,
        cu_seqlens_padded: torch.Tensor,
        *,
        input_is_sanitized: bool = False,
    ) -> tuple[torch.Tensor, CSA2THDCompressionLayout]:
        """Project before gathering on the BF16 fused path; retain the native reference."""
        if (
            self.use_fused_compressor
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and layout.capacity > 0
            and not torch.are_deterministic_algorithms_enabled()
        ):
            latent = self._forward_thd_projected(
                x, layout, cu_seqlens_padded, input_is_sanitized=input_is_sanitized
            )
        else:
            latent = self._forward_thd_grouped(x, layout)
        latent = latent.to(x.dtype)
        if layout.capacity == 0:
            latent = latent * self.norm.weight
        else:
            latent = self.norm(latent)
        return latent.masked_fill(~layout.valid_groups[:, None, None], 0), layout

    def _forward_thd_projected(self, x, layout, cu_seqlens_padded, *, input_is_sanitized):
        """Reuse token-order hidden storage and let the V4 kernel gather KV/gate."""
        ratio = self.compress_ratio
        # At most ratio-1 physical tail tokens occur per sequence. This host-known
        # bound retains every physical segment while avoiding large orphan capacity.
        rows = min(
            x.shape[0], layout.capacity * ratio + (cu_seqlens_padded.numel() - 1) * (ratio - 1)
        )
        inputs = x[:rows]
        if not input_is_sanitized:
            # Only a token-sized mask is assembled here, never a grouped hidden
            # tensor. Integer addition handles repeated safe indices in invalid
            # groups without overwriting a real token's validity.
            counts = torch.zeros(rows, dtype=torch.int32, device=x.device)
            counts.scatter_add_(
                0,
                layout.source_indices.flatten(),
                layout.valid_groups[:, None].expand(-1, ratio).reshape(-1).int(),
            )
            inputs = inputs.masked_fill((counts == 0)[:, None, None], 0)
        latent, gate = self._project(inputs)
        if ratio == 1:
            # r1 physical compressed addresses equal the original token addresses.
            return latent.masked_fill(~layout.valid_groups[:, None, None], 0)
        pooled = maybe_compress_csa2_thd_fused(latent, gate, cu_seqlens_padded, layout)
        if pooled is not None:
            return pooled
        # Unsupported frontend/device or deterministic mode: gather only the
        # smaller projections and retain the FP32-intermediate pooling reference.
        invalid = ~layout.valid_groups[:, None, None, None]
        latent = latent[layout.source_indices].masked_fill(invalid, 0)
        gate = gate[layout.source_indices].masked_fill(invalid, 0)
        shape = (layout.capacity * ratio, 1, latent.shape[-1])
        return self._pool_r2(
            latent.reshape(shape), gate.reshape(shape), x.dtype, layout.valid_groups
        )

    def _forward_thd_grouped(self, x, layout):
        """Native/FP32 path: sanitize grouped input before the original Linear modules."""
        ratio = self.compress_ratio
        inputs = maybe_prepare_csa2_r2_fused(x, layout, enabled=self.use_fused_compressor)
        if inputs is None:
            grouped = x[layout.source_indices]
            # Neutralize invalid groups before GEMM, softmax and RMSNorm. Masking only
            # their final output would still allow NaN/Inf padding to poison backward.
            grouped = grouped.masked_fill(~layout.valid_groups[:, None, None, None], 0)
            inputs = grouped.reshape(layout.capacity * ratio, 1, x.shape[-1])
        latent, gate = self._project(inputs)
        if ratio == 2:
            latent = self._pool_r2(latent, gate, x.dtype, layout.valid_groups)
        return latent

    def backward_dw(self) -> None:
        """Follow the existing DSv4 linear gradient interface."""
        self.linear_wkv.backward_dw()
        if self.linear_wgate is not None:
            self.linear_wgate.backward_dw()


@dataclass
class CSA2IndexerSubmodules:
    """Projection and normalization specs for Full and Reindex CSA2 indexers."""

    linear_wq_b: ModuleSpec | type
    linear_wk: ModuleSpec | type
    k_norm: ModuleSpec | type
    linear_weights_proj: ModuleSpec | type


class CSA2Indexer(MegatronModule):
    """Select global positions using keys derived from the pre-RoPE compressor latent."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: CSA2IndexerSubmodules,
        compress_ratio: int,
        pg_collection: ProcessGroupCollection,
        owns_k: bool = True,
    ) -> None:
        super().__init__(config=config)
        self.compress_ratio = compress_ratio
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.n_heads = config.dsa_indexer_n_heads
        self.head_dim = config.dsa_indexer_head_dim
        self.topk = config.dsa_indexer_topk
        self.owns_k = owns_k
        self.use_fused_kernels = getattr(
            config, "dsa_kernel_backend", "none"
        ) == "cudnn" and use_fused_dsa_kernels(config)
        self.precision = getattr(config, "dsa_indexer_precision", "bf16")
        linear_kwargs = dict(
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            config.q_lora_rank,
            self.n_heads * self.head_dim,
            **linear_kwargs,
        )
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj, config.hidden_size, self.n_heads, **linear_kwargs
        )
        self.linear_wk = self.k_norm = None
        if owns_k:
            self.linear_wk = build_module(
                submodules.linear_wk, config.v_head_dim, self.head_dim, **linear_kwargs
            )
            self.k_norm = build_module(
                submodules.k_norm,
                config=config,
                hidden_size=self.head_dim,
                eps=config.attention_latent_norm_epsilon,
            )

    def project_keys(
        self,
        latent: torch.Tensor,
        rotary_pos_emb: nn.Module,
        *,
        thd_layout: CSA2THDCompressionLayout | None = None,
    ) -> torch.Tensor:
        """Produce graph-connected rotated index keys at a Full owner."""
        if not self.owns_k:
            raise ValueError("A CSA2 Reindex layer must consume its Full owner's indexer_k.")
        if thd_layout is not None:
            if (
                thd_layout.ratio != self.compress_ratio
                or latent.ndim != 3
                or latent.shape[:2] != (thd_layout.capacity, 1)
                or latent.device != thd_layout.valid_groups.device
            ):
                raise ValueError("CSA2 indexer latent must match its THD compression layout.")
            latent = latent.masked_fill(~thd_layout.valid_groups[:, None, None], 0)
        k, _ = self.linear_wk(latent)
        # An incomplete first compression group produces no key rows.
        k = self.k_norm(k) if k.shape[0] else k * self.k_norm.weight
        if thd_layout is not None:
            return apply_csa2_thd_rope(k, rotary_pos_emb, self.config, thd_layout, self.cp_group)
        pos_dim = self.config.qk_pos_emb_head_dim
        return _apply_rope(
            k,
            nope_dim=self.head_dim - pos_dim,
            pos_dim=pos_dim,
            rotary_pos_emb_module=rotary_pos_emb,
            config=self.config,
            rotary_seq_len=k.shape[0],
            ratio=self.compress_ratio,
            cp_group=self.cp_group,
            inplace=False,
        )

    def _project_inputs(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        rotary_pos_emb: nn.Module,
        *,
        indexer_k: torch.Tensor | None = None,
        thd_layout: CSA2THDLayout | None = None,
        compressed_layout: CSA2THDCompressionLayout | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project Q/W once, preserving the Full owner's shared K autograd graph."""
        if thd_layout is not None:
            if qr.ndim == 2:
                qr = qr.unsqueeze(1)
            if compressed_layout is None:
                compressed_layout = thd_layout.for_compression(self.compress_ratio)
            if compressed_layout.ratio != self.compress_ratio:
                raise ValueError("CSA2 indexer compression ratio must match its THD layout.")
            if x.shape[:2] != (thd_layout.total_tokens, 1) or qr.shape[:2] != x.shape[:2]:
                raise ValueError("CSA2 THD indexer inputs must have shape [total_tokens, 1, dim].")
            invalid = ~thd_layout.valid_tokens[:, None, None]
            x = x.masked_fill(invalid, 0)
            qr = qr.masked_fill(invalid, 0)
        elif compressed_layout is not None:
            raise ValueError("CSA2 compressed THD layout requires a token layout.")
        if indexer_k is None:
            if latent is None:
                raise ValueError("CSA2 indexer scores require latent or shared indexer_k.")
            indexer_k = self.project_keys(latent, rotary_pos_emb, thd_layout=compressed_layout)
        elif latent is not None:
            raise ValueError("Provide either latent or indexer_k, not both.")
        q, _ = self.linear_wq_b(qr)
        q = q.reshape(*q.shape[:-1], self.n_heads, self.head_dim)
        if thd_layout is not None:
            if indexer_k.shape[:2] != (compressed_layout.capacity, 1):
                raise ValueError("CSA2 shared indexer keys must match the THD compression layout.")
            indexer_k = indexer_k.masked_fill(~compressed_layout.valid_groups[:, None, None], 0)
            q = apply_csa2_thd_rope(q, rotary_pos_emb, self.config, thd_layout, self.cp_group)
            weights, _ = self.linear_weights_proj(x)
            return q.squeeze(1), indexer_k.squeeze(1), weights.squeeze(1)
        pos_dim = self.config.qk_pos_emb_head_dim
        q = _apply_rope(
            q,
            nope_dim=self.head_dim - pos_dim,
            pos_dim=pos_dim,
            rotary_pos_emb_module=rotary_pos_emb,
            config=self.config,
            rotary_seq_len=q.shape[0],
            cp_group=self.cp_group,
        )
        weights, _ = self.linear_weights_proj(x)
        return q, indexer_k, weights

    def _score_projected(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        candidates: CSA2CandidateBlocks | torch.Tensor | None = None,
        thd_layout: CSA2THDLayout | None = None,
        compressed_layout: CSA2THDCompressionLayout | None = None,
    ) -> torch.Tensor:
        """Native masked scores for supervision and candidate selection."""
        if isinstance(candidates, CSA2CandidateBlocks):
            candidates = candidates.to_mask(
                k.shape[0], thd_layout=thd_layout, compressed_layout=compressed_layout
            )
        scaled_weights = weights.float() * (self.head_dim**-0.5 * self.n_heads**-0.5)
        if thd_layout is not None:
            scores = torch.einsum("thd,cd->thc", q.float(), k.float()).relu()
            scores = (scores * scaled_weights.unsqueeze(-1)).sum(dim=1)
            valid = (
                thd_layout.valid_tokens[:, None]
                & compressed_layout.valid_groups[None, :]
                & (thd_layout.sequence_ids[:, None] == compressed_layout.sequence_ids[None, :])
                & (
                    compressed_layout.position_ids[None, :] + self.compress_ratio - 1
                    <= thd_layout.position_ids[:, None]
                )
            )
            if candidates is not None:
                valid = valid & candidates
            return scores.masked_fill(~valid, -torch.inf)
        scores = torch.einsum("sbhd,tbd->bsht", q.float(), k.float()).relu()
        scores = (scores * scaled_weights.permute(1, 0, 2).unsqueeze(-1)).sum(dim=2)
        visible = torch.arange(1, q.shape[0] + 1, device=q.device) // self.compress_ratio
        causal = torch.arange(k.shape[0], device=q.device)[None, :] < visible[:, None]
        scores = scores.masked_fill(~causal, -torch.inf)
        if candidates is not None:
            scores = scores.masked_fill(~candidates, -torch.inf)
        return scores

    def forward_before_topk(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        rotary_pos_emb: nn.Module,
        *,
        indexer_k: torch.Tensor | None = None,
        candidates: CSA2CandidateBlocks | torch.Tensor | None = None,
        thd_layout: CSA2THDLayout | None = None,
        compressed_layout: CSA2THDCompressionLayout | None = None,
    ) -> torch.Tensor:
        """Return graph-connected native scores for the indexer auxiliary objective.

        Scores have shape ``[B, S, C]`` in SBHD and ``[T, C]`` in THD. As in
        DSv4, MXFP8 only changes discrete selection; supervision uses the original
        unquantized projections, including shared keys from the Full owner.
        """
        if thd_layout is not None and compressed_layout is None:
            compressed_layout = thd_layout.for_compression(self.compress_ratio)
        q, k, weights = self._project_inputs(
            x,
            qr,
            latent,
            rotary_pos_emb,
            indexer_k=indexer_k,
            thd_layout=thd_layout,
            compressed_layout=compressed_layout,
        )
        return self._score_projected(
            q,
            k,
            weights,
            candidates=candidates,
            thd_layout=thd_layout,
            compressed_layout=compressed_layout,
        )

    def select_indices(self, scores: torch.Tensor) -> torch.Tensor:
        """Return sorted logical positions, with -1 for masked or unavailable entries."""
        # A stable tie order preserves selection across SBHD/THD and different
        # padding capacities. Physical order within each sequence is local order.
        indices = scores.argsort(dim=-1, descending=True, stable=True)[
            ..., : min(self.topk, scores.shape[-1])
        ]
        indices = indices.sort(dim=-1).values
        return indices.masked_fill(~scores.gather(-1, indices).isfinite(), -1).int()

    @torch.no_grad()
    def _fused_topk(self, inputs: CSA2IndexerInputs) -> torch.Tensor:
        """Adapt DSv4 score/Top-K kernels to r1/r2 and physical CSA2 THD rows."""
        if inputs.candidates is not None:
            return fused_candidate_topk(inputs, self.topk, self.precision)
        q, k, weights = inputs.q, inputs.k, inputs.weights
        thd_layout, compressed_layout = inputs.thd_layout, inputs.compressed_layout
        width = min(self.topk, inputs.key_capacity)
        shape = (*inputs.output_shape, width)
        if q.shape[0] == 0 or width == 0 or inputs.max_keys == 0:
            return torch.full(shape, -1, dtype=torch.int32, device=q.device)

        # MCore's MXFP8 entry point currently requires H64/D128. Extra heads
        # contribute exactly zero; scale by the real H32, never the padded H64.
        if self.precision == "mxfp8" and self.n_heads == 32:
            q = F.pad(q, (0, 0, 0, 32))
            weights = F.pad(weights, (0, 32))
        # The loss consumes the original BF16 weights; cuDNN selection expects
        # scaled/rounded weights. This temporary must never modify shared W.
        weights = (weights.float() * (self.head_dim**-0.5 * self.n_heads**-0.5)).to(weights.dtype)
        kwargs = {}
        if thd_layout is not None:
            cu_q, cu_k, max_q, max_k = inputs.packed_metadata
            kwargs = dict(
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_k,
                max_seqlen_q=max_q,
                # Round UP for an odd r2 Q span; its actual K segment still
                # contains only complete physical compression groups.
                max_seqlen_kv=max((max_q + self.compress_ratio - 1) // self.compress_ratio, max_k),
            )
        else:
            batch, seq_len = inputs.output_shape
            # Like V4 Path B, view the prepared rows as BSHD and call the core
            # directly. The public SBHD wrapper would copy Q/K/W again.
            q = q.view(batch, seq_len, q.shape[-2], q.shape[-1])
            k = k.view(batch, inputs.key_capacity, k.shape[-1])
            weights = weights.view(batch, seq_len, weights.shape[-1])
            if seq_len > inputs.key_capacity * self.compress_ratio:
                # The compact BSHD wrapper requires Sq <= Sk * r. Keep this
                # never-visible dummy key private to selection, not shared K.
                k = F.pad(k, (0, 0, 0, 1))

        indices, _, _, _ = _indexer_topk_core(
            q,
            k,
            weights,
            topk=width,
            ratio=self.compress_ratio,
            use_compact=True,
            precision=self.precision,
            # Native CSA2 also resolves exact ties toward earlier local keys.
            deterministic=True,
            **kwargs,
        )
        if thd_layout is not None:
            sequence_ids = thd_layout.sequence_ids.clamp_min(0)
            starts = inputs.starts.unsqueeze(-1)
            lengths = compressed_layout.cu_seqlens.diff()[sequence_ids].unsqueeze(-1)
            visible = inputs.visible.unsqueeze(-1)
            valid = (
                (indices >= 0)
                & (indices < torch.minimum(lengths, visible))
                & thd_layout.valid_tokens.unsqueeze(-1)
            )
            indices = torch.where(valid, indices + starts, -1)
        # Valid selected keys are ascending; invalid slots follow them. Shared
        # state contains physical global-K rows, never a local/global-KV offset.
        sentinel = torch.iinfo(torch.int32).max
        indices = indices.masked_fill(indices < 0, sentinel).sort(dim=-1).values
        return indices.masked_fill(indices == sentinel, -1).int().contiguous()

    @torch.no_grad()
    def _candidate_blocks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        selection_scores: torch.Tensor | None,
        thd_layout: CSA2THDLayout | None = None,
        compressed_layout: CSA2THDCompressionLayout | None = None,
    ) -> CSA2CandidateBlocks:
        """Generate compact candidate state from fused chunks or native reference scores."""
        if self.use_fused_kernels:
            return fused_candidate_blocks(
                prepare_csa2_indexer_inputs(
                    q,
                    k,
                    weights,
                    self.compress_ratio,
                    thd_layout=thd_layout,
                    compressed_layout=compressed_layout,
                ),
                topk_blocks=self.config.csa2_candidate_topk_blocks,
                block_size=self.config.csa2_candidate_block_size,
                precision=self.precision,
            )
        if thd_layout is None:
            visible = (torch.arange(q.shape[0], device=q.device) + 1) // self.compress_ratio
        else:
            sequence_ids = thd_layout.sequence_ids.clamp_min(0)
            starts = compressed_layout.cu_seqlens_padded[sequence_ids].long()
            ends = compressed_layout.cu_seqlens_padded[sequence_ids + 1].long()
            local = torch.arange(compressed_layout.max_seqlen, device=q.device)
            physical = starts[:, None] + local
            if k.shape[0] > 0:
                selection_scores = selection_scores.gather(1, physical.clamp(max=k.shape[0] - 1))
                selection_scores = selection_scores.masked_fill(
                    physical >= ends[:, None], -torch.inf
                )
            else:
                selection_scores = selection_scores.new_full(physical.shape, -torch.inf)
            visible = ((thd_layout.position_ids + 1) // self.compress_ratio).masked_fill(
                ~thd_layout.valid_tokens, 0
            )
        return candidate_blocks_from_scores(
            selection_scores,
            visible,
            self.config.csa2_candidate_topk_blocks,
            self.config.csa2_candidate_block_size,
        )

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        rotary_pos_emb: nn.Module,
        *,
        indexer_k: torch.Tensor | None = None,
        indexer_k_flat: torch.Tensor | None = None,
        candidates: CSA2CandidateBlocks | torch.Tensor | None = None,
        thd_layout: CSA2THDLayout | None = None,
        compressed_layout: CSA2THDCompressionLayout | None = None,
        return_scores: bool = False,
        return_candidates: bool = False,
        return_loss_inputs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | CSA2IndexerInputs | None, CSA2CandidateBlocks | None]:
        """Select keys and optionally return supervision scores and compact candidates.

        Ordinary fused Top-K avoids dense scores when supervision is disabled.
        Candidate generation and Reindex use bounded score chunks. Fused loss
        consumes graph-connected projections through ``return_loss_inputs``;
        ``return_scores`` explicitly requests the native reference scores.
        """
        if self.precision == "mxfp8" and not self.use_fused_kernels:
            raise ValueError("CSA2 MXFP8 selection requires the fused indexer backend")
        if return_scores and return_loss_inputs:
            raise ValueError("Request either native scores or fused loss inputs")
        if (
            self.use_fused_kernels
            and candidates is not None
            and not isinstance(candidates, CSA2CandidateBlocks)
        ):
            raise ValueError("Fused CSA2 Reindex requires compact candidate blocks")
        if thd_layout is not None and compressed_layout is None:
            compressed_layout = thd_layout.for_compression(self.compress_ratio)
        q, k, weights = self._project_inputs(
            x,
            qr,
            latent,
            rotary_pos_emb,
            indexer_k=indexer_k,
            thd_layout=thd_layout,
            compressed_layout=compressed_layout,
        )
        score_kwargs = dict(
            candidates=candidates, thd_layout=thd_layout, compressed_layout=compressed_layout
        )
        fused_topk = self.use_fused_kernels
        native_selection = not fused_topk
        scores = None
        if return_scores or (native_selection and self.precision == "bf16"):
            scores = self._score_projected(q, k, weights, **score_kwargs)
        # Prepare while the projection graph is enabled. Selection reads these
        # same tensors under no_grad; supervised consumers retain the K owner edge.
        inputs = (
            prepare_csa2_indexer_inputs(
                q, k, weights, self.compress_ratio, **score_kwargs, k_flat=indexer_k_flat
            )
            if fused_topk or return_loss_inputs
            else None
        )
        if fused_topk:
            # Selection/loss only need the prepared storage. Release local
            # references to the original Q/W before allocating kernel workspace.
            del q, k, weights
        with torch.no_grad():
            selection_scores = scores
            candidate_blocks = None
            if fused_topk and return_candidates:
                indices, candidate_blocks = self._fused_topk_and_candidates(inputs)
            elif fused_topk:
                indices = self._fused_topk(inputs)
            else:
                indices = self.select_indices(selection_scores)
            if return_candidates and not fused_topk:
                candidate_blocks = self._candidate_blocks(
                    q,
                    k,
                    weights,
                    selection_scores=selection_scores,
                    thd_layout=thd_layout,
                    compressed_layout=compressed_layout,
                )
        loss_inputs = inputs if return_loss_inputs else scores if return_scores else None
        return indices, loss_inputs, candidate_blocks

    def _fused_topk_and_candidates(
        self, inputs: CSA2IndexerInputs
    ) -> tuple[torch.Tensor, CSA2CandidateBlocks]:
        """Use one bounded scoring pass for source Top-K and exported block candidates."""
        return fused_candidate_blocks(
            inputs,
            topk_blocks=self.config.csa2_candidate_topk_blocks,
            block_size=self.config.csa2_candidate_block_size,
            precision=self.precision,
            token_topk=self.topk,
        )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        rotary_pos_emb: nn.Module,
        *,
        indexer_k: torch.Tensor | None = None,
        candidates: CSA2CandidateBlocks | torch.Tensor | None = None,
        thd_layout: CSA2THDLayout | None = None,
        compressed_layout: CSA2THDCompressionLayout | None = None,
    ) -> torch.Tensor:
        """Return sorted global-position indices, with ``-1`` for unavailable positions."""
        indices, _, _ = self.forward_with_scores(
            x,
            qr,
            latent,
            rotary_pos_emb,
            indexer_k=indexer_k,
            candidates=candidates,
            thd_layout=thd_layout,
            compressed_layout=compressed_layout,
        )
        return indices

    def backward_dw(self) -> None:
        """Follow the existing DSv4 linear gradient interface."""
        for linear in (self.linear_wq_b, self.linear_wk, self.linear_weights_proj):
            if linear is not None:
                linear.backward_dw()


class CompressedSparseAttention2(MegatronModule):
    """SWA, Full, Reindex and Reuse behind the existing DSv4 attention interface."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: CompressedSparseAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        pg_collection: ProcessGroupCollection,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        k_channels: int | None = None,
        v_channels: int | None = None,
        cp_comm_type: str | None = None,
        rotary_pos_emb: nn.Module | None = None,
        compress_ratio: int | None = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config)
        if config.dsv4_version != "v4.1" or attn_mask_type != AttnMaskType.causal:
            raise ValueError("CSA2 requires V4.1 causal attention.")
        if attention_type != "self" or is_mtp_layer:
            raise NotImplementedError("CSA2 currently supports backbone self-attention only.")
        if get_pg_size(pg_collection.tp) != 1 or get_pg_size(pg_collection.cp) != 1:
            raise NotImplementedError("Native CSA2 currently requires TP=CP=1.")
        if (attention_dropout or config.attention_dropout) != 0:
            raise NotImplementedError("Native CSA2 does not implement attention dropout.")
        layer_idx = layer_number - 1
        ratio = config.csa_compress_ratios[layer_idx]
        if compress_ratio is not None and compress_ratio != ratio:
            raise ValueError("CSA2 compress_ratio must match the configured layer ratio.")
        self.layer_idx = layer_idx
        self.is_kv_source = layer_idx in config.csa2_kv_source_layers
        self.is_index_source = layer_idx in config.csa2_index_source_layers
        self.kv_source_layer = (
            max(source for source in config.csa2_kv_source_layers if source <= layer_idx)
            if ratio
            else None
        )
        self.index_source_layer = (
            max(source for source in config.csa2_index_source_layers if source <= layer_idx)
            if ratio
            else None
        )
        self.is_candidate_source = layer_idx == config.csa2_candidate_source_layer
        self.uses_candidates = (
            config.csa2_candidate_source_layer is not None
            and layer_idx > config.csa2_candidate_source_layer
        )
        self.compress_ratio = ratio
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.rotary_pos_emb = rotary_pos_emb
        # Main attention and ordinary indexer Top-K reuse the DSv4 kernels.
        # Candidate operations and indexer supervision retain native math.
        self.use_fused_kernels = use_fused_dsa_kernels(config)
        self.softmax_scale = config.v_head_dim**-0.5 if softmax_scale is None else softmax_scale
        device = "cpu" if config.use_cpu_initialization else torch.cuda.current_device()
        self.attn_sink = mark_keep_in_fp32(
            nn.Parameter(
                torch.zeros(config.num_attention_heads, dtype=torch.float32, device=device)
            )
        )
        self.compressor = self.indexer = None
        if self.is_kv_source:
            kwargs = dict(config=config, compress_ratio=ratio, pg_collection=pg_collection)
            self.compressor = build_module(submodules.compressor, **kwargs)
        if self.is_index_source:
            self.indexer = build_module(
                submodules.indexer,
                config=config,
                compress_ratio=ratio,
                pg_collection=pg_collection,
                owns_k=self.is_kv_source,
            )

    def _shared_global_attention(
        self, x: torch.Tensor, qr: torch.Tensor, state: CSA2State, *, use_indexer_loss: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | CSA2IndexerInputs | None]:
        """Publish or consume graph tensors and indices in compressed KV coordinates."""
        thd_layout = state.thd_layout
        if self.is_kv_source:
            if thd_layout is None:
                latent = self.compressor(x)
                state.compressed_layout = None
            else:
                # forward() has already cleared padding in this shared hidden
                # buffer. Reuse it for both original Linear modules, as in V4.
                latent, state.compressed_layout = self.compressor(
                    x, thd_layout=thd_layout, input_is_sanitized=True
                )
            # The auxiliary objective trains the indexer, not the main compressor.
            # Detach BEFORE projection so Full and Reindex losses still reach the
            # owning indexer's K projection and normalization through the shared keys.
            indexer_latent = latent.detach() if use_indexer_loss else latent
            state.indexer_k = self.indexer.project_keys(
                indexer_latent, self.rotary_pos_emb, thd_layout=state.compressed_layout
            )
            if thd_layout is None:
                pos_dim = self.config.qk_pos_emb_head_dim
                state.global_kv = _apply_rope(
                    latent,
                    self.config.v_head_dim - pos_dim,
                    pos_dim,
                    self.rotary_pos_emb,
                    self.config,
                    latent.shape[0],
                    ratio=self.compress_ratio,
                    cp_group=self.cp_group,
                    # The indexer's linear backward retains this same latent,
                    # including when its input was detached from the compressor.
                    inplace=False,
                )
            else:
                state.global_kv = apply_csa2_thd_rope(
                    latent, self.rotary_pos_emb, self.config, state.compressed_layout, self.cp_group
                )
            state.kv_source_layer = self.layer_idx
            # Pack shared K once per owner, with graph-connected views/copies.
            # All supervised consumers use the same storage and autograd edge.
            state.global_kv_flat = state.indexer_k_flat = None
            if self.use_fused_kernels:
                state.global_kv_flat = (
                    state.global_kv.transpose(0, 1)
                    .reshape(-1, state.global_kv.shape[-1])
                    .contiguous()
                )
                state.indexer_k_flat = (
                    state.indexer_k.transpose(0, 1)
                    .reshape(-1, state.indexer_k.shape[-1])
                    .contiguous()
                )
            state.global_indices = state.candidates = None
            state.index_source_layer = state.candidate_source_layer = None
        elif (
            state.kv_source_layer != self.kv_source_layer
            or state.global_kv is None
            or state.indexer_k is None
        ):
            raise ValueError(
                f"CSA2 layer {self.layer_idx} requires KV and indexer K from Full layer "
                f"{self.kv_source_layer} in the same forward's CSA2State."
            )

        scores = None
        if thd_layout is not None and (
            state.compressed_layout is None or state.compressed_layout.ratio != self.compress_ratio
        ):
            raise ValueError("CSA2 shared compressed layout must match this layer's ratio.")
        if self.is_index_source:
            candidates = None
            if self.uses_candidates:
                if (
                    state.candidate_source_layer != self.config.csa2_candidate_source_layer
                    or state.candidates is None
                ):
                    raise ValueError(
                        f"CSA2 layer {self.layer_idx} requires candidates from layer "
                        f"{self.config.csa2_candidate_source_layer} in the same CSA2State."
                    )
                candidates = state.candidates
            with torch.set_grad_enabled(use_indexer_loss):
                state.global_indices, scores, candidate_blocks = self.indexer.forward_with_scores(
                    x.detach(),
                    qr.detach(),
                    None,
                    self.rotary_pos_emb,
                    indexer_k=state.indexer_k,
                    indexer_k_flat=state.indexer_k_flat,
                    candidates=candidates,
                    thd_layout=thd_layout,
                    compressed_layout=state.compressed_layout,
                    return_scores=use_indexer_loss
                    and not (self.use_fused_kernels and self.indexer.use_fused_kernels),
                    return_candidates=self.is_candidate_source,
                    return_loss_inputs=use_indexer_loss
                    and self.use_fused_kernels
                    and self.indexer.use_fused_kernels,
                )
            if self.is_candidate_source:
                state.candidates = candidate_blocks
                state.candidate_source_layer = self.layer_idx
            # The source's Top-K is selected over all causal positions. Only
            # later Reindex layers consume the newly produced candidate blocks.
            state.index_source_layer = self.layer_idx
            state.fused_indices = state.fused_topk_length = None
            state.fused_q_padding_mask = None
            state.fused_window_size = None
        elif state.index_source_layer != self.index_source_layer or state.global_indices is None:
            raise ValueError(
                f"CSA2 Reuse layer {self.layer_idx} requires indices from layer "
                f"{self.index_source_layer} in the same forward's CSA2State."
            )
        return state.global_kv, state.global_indices, scores if use_indexer_loss else None

    def _compute_indexer_loss(
        self,
        query: torch.Tensor,
        local_kv: torch.Tensor,
        global_kv: torch.Tensor,
        window_indices: torch.Tensor,
        global_indices: torch.Tensor,
        scores: torch.Tensor | CSA2IndexerInputs,
        *,
        thd_layout: CSA2THDLayout | None = None,
    ) -> torch.Tensor:
        """Apply the existing DSv4 KL objective to this Full/Reindex layer's teacher.

        Dense loss covers all causal global keys, restricted by the candidate mask
        on later Reindex layers. Sparse loss further restricts both distributions to
        this layer's top-k. The teacher includes SWA and sink mass before summing
        heads and renormalizing over global keys. Reuse layers add no objective.

        As in DSv4, this objective covers input tokens independently of the LM label
        mask. Token reduction and backward scaling use the existing DSA interfaces;
        the number of indexer layers only normalizes the reported metric.
        """
        if isinstance(scores, CSA2IndexerInputs):
            return fused_csa2_indexer_loss(
                scores,
                query,
                local_kv,
                global_kv,
                self.attn_sink,
                window_indices,
                global_indices,
                self.softmax_scale,
                self.config.dsa_indexer_loss_coeff,
                self.config.dsa_indexer_use_sparse_loss,
                self.config.calculate_per_token_loss,
            )
        if scores.shape[-1] == 0 or (thd_layout is not None and query.shape[0] == 0):
            # No complete compression group: preserve zero gradients for the indexer.
            return scores.sum() * 0
        non_compressed_lse = _compute_unfused_csa_non_compressed_lse(
            query, local_kv, self.attn_sink, window_indices, self.softmax_scale
        )
        mask = torch.zeros_like(scores).masked_fill(torch.isneginf(scores), -torch.inf)
        num_heads = query.shape[1] if thd_layout is not None else query.shape[2]
        query_valid_rows = None
        if thd_layout is not None:
            scores = scores.unsqueeze(0)
            global_indices = global_indices.unsqueeze(0)
            query_valid_rows = thd_layout.valid_tokens
        return compute_dsa_indexer_loss(
            # The shared helper adds masks in place; keep the scoring/selection tensor intact.
            scores.clone(),
            global_indices,
            query.detach(),
            global_kv.detach().unsqueeze(2).expand(-1, -1, num_heads, -1),
            self.softmax_scale,
            self.config.dsa_indexer_loss_coeff,
            self.config.dsa_indexer_use_sparse_loss,
            self.pg_collection,
            mask=mask,
            query_valid_rows=query_valid_rows,
            calculate_per_token_loss=self.config.calculate_per_token_loss,
            non_compressed_lse=non_compressed_lse,
        )

    def _attention_inputs(self, local, window, global_kv=None, topk=None):
        """Lower shared logical indices out of place for this consumer's KV buffer."""
        if global_kv is None:
            return local, window
        if local.ndim == 2:
            global_kv = global_kv.squeeze(1)
        indices = torch.cat((window, torch.where(topk >= 0, topk + local.shape[0], -1)), -1)
        return torch.cat((local, global_kv), dim=0), indices

    def _build_fused_thd_indices(self, query, topk, state, *, need_window=False):
        """Fuse packed window/global addressing, padding and compaction using V4."""
        layout = state.thd_layout
        window_size = self.config.csa_window_size
        window_out = None
        if need_window and (
            state.fused_window_indices is None
            or state.fused_window_indices.shape[-1] != window_size
        ):
            window_out = torch.empty(
                (query.shape[0], window_size), dtype=torch.int32, device=query.device
            )
        compressed_width = 0 if topk is None else topk.shape[-1]
        indices, lengths, _, padding_mask = thd_layout_kernels.build_attention_indices(
            layout.cu_seqlens_padded,
            global_start=0,
            l_local=query.shape[0],
            d_window=0,
            window_size=window_size,
            ratio=self.compress_ratio,
            compressed_width=compressed_width,
            compressed_topk=topk if compressed_width else None,
            cu_seqlens_compressed=(
                state.compressed_layout.cu_seqlens_padded if compressed_width else None
            ),
            compressed_base=query.shape[0],
            compressed_rows=state.global_kv_flat.shape[0] if compressed_width else 0,
            compressed_is_sequence_major=True,
            cu_seqlens_unpadded=layout.cu_seqlens,
            output_alignment=get_flash_mla_topk_alignment(),
            compressed_topk_is_physical=True,
            mask_padding_rows=True,
            window_indices_out=window_out,
        )
        if window_out is not None:
            state.fused_window_indices = window_out
        return indices, lengths, padding_mask

    def _fused_indices(self, query, window, topk, state, *, need_window=False):
        """Lower and compact once per index source; Reuse shares indices and lengths."""
        window_size = self.config.csa_window_size if window is None else window.shape[-1]
        have_window = (
            not need_window
            or window is not None
            or (
                state.fused_window_indices is not None
                and state.fused_window_indices.shape[-1] == window_size
            )
        )
        if (
            state.fused_indices is not None
            and state.fused_window_size == window_size
            and have_window
        ):
            return state.fused_indices
        if window is None:
            state.fused_indices, state.fused_topk_length, state.fused_q_padding_mask = (
                self._build_fused_thd_indices(query, topk, state, need_window=need_window)
            )
            state.fused_window_size = window_size
            return state.fused_indices
        if state.thd_layout is None:
            seq_len, batch = query.shape[:2]
            window = window.reshape(batch * seq_len, -1)
            topk = topk.reshape(batch * seq_len, -1)
            batch_ids = torch.arange(batch, device=query.device).repeat_interleave(seq_len)[:, None]
            window = torch.where(window >= 0, window + batch_ids * seq_len, -1)
            topk = torch.where(topk >= 0, topk + batch_ids * state.global_kv.shape[0], -1)
            local_rows = batch * seq_len
        else:
            local_rows = query.shape[0]
        state.fused_indices = (
            torch.cat((window, torch.where(topk >= 0, topk + local_rows, -1)), -1)
            .int()
            .contiguous()
        )
        if state.thd_layout is None:
            # Retain V4's sequence-major Q/output. Only key addresses are
            # batch-major; reorder query rows once per index source.
            state.fused_indices = (
                state.fused_indices.reshape(batch, seq_len, -1)
                .transpose(0, 1)
                .reshape(batch * seq_len, -1)
                .contiguous()
            )
        state.fused_indices, state.fused_topk_length = _compact_flat_topk_idxs(state.fused_indices)
        state.fused_q_padding_mask = (
            (state.fused_topk_length == 0) if state.thd_layout is not None else None
        )
        state.fused_window_size = window_size
        return state.fused_indices

    def _fused_attention(self, query, local, window, global_kv=None, topk=None, state=None):
        """Reuse V4 attention with shared indices and producer KV storage."""
        is_thd = query.ndim == 3
        if global_kv is None:
            padding_mask = None
            if is_thd:
                if window is None:
                    indices, topk_length, padding_mask = self._build_fused_thd_indices(
                        query, None, state
                    )
                else:
                    indices, topk_length = _compact_flat_topk_idxs(window.int().contiguous())
                    padding_mask = topk_length == 0
            else:
                indices, topk_length = build_flat_topk_idxs(
                    window, batch_size=query.shape[1], compact=True
                )
            return csa_sparse_attn(
                query,
                local,
                self.attn_sink,
                indices,
                self.softmax_scale,
                topk_length=topk_length,
                is_thd=is_thd,
                q_padding_mask=padding_mask,
            )
        indices = self._fused_indices(query, window, topk, state)
        if is_thd:
            q, local_flat = query.contiguous(), local.contiguous()
        else:
            q = query.reshape(-1, query.shape[-2], query.shape[-1])
            local_flat = local.transpose(0, 1).reshape(-1, local.shape[-1]).contiguous()
        global_flat = state.global_kv_flat
        kv = torch.cat((local_flat, global_flat))
        output = csa_sparse_attn(
            q,
            kv,
            self.attn_sink,
            indices,
            self.softmax_scale,
            topk_length=state.fused_topk_length,
            is_thd=True,
            kv_reconstruction_parts=(local_flat[:0], local_flat, global_flat),
            q_padding_mask=state.fused_q_padding_mask,
        )
        if not is_thd:
            output = output.reshape(query.shape[0], query.shape[1], -1)
        return output

    def _forward_unfused(self, query, local, window, x, qr, state, use_indexer_loss):
        """Native semantic reference, also handling empty packed batches."""
        loss = None
        kv, indices = local, window
        if self.compress_ratio:
            global_kv, topk, scores = self._shared_global_attention(
                x, qr, state, use_indexer_loss=use_indexer_loss
            )
            if scores is not None:
                loss = self._compute_indexer_loss(
                    query, local, global_kv, window, topk, scores, thd_layout=state.thd_layout
                )
            kv, indices = self._attention_inputs(local, window, global_kv, topk)
        # Repeated KV positions accumulate their gradients in FP32.
        output = unfused_compressed_sparse_attn(
            query, kv.float(), self.attn_sink, indices, self.softmax_scale
        )
        return output, loss

    def _forward_fused_no_indexer(self, query, local, window, state):
        """Window-only layer: one sparse-attention forward/backward."""
        return self._fused_attention(query, local, window, state=state), None

    def _forward_fused_indexer_training(self, query, local, window, x, qr, state):
        """Full/Reindex with supervision: shared selection, joint attention/KL autograd."""
        global_kv, topk, scores = self._shared_global_attention(x, qr, state, use_indexer_loss=True)
        need_window = (
            not isinstance(scores, CSA2IndexerInputs) or not self.config.dsa_indexer_use_sparse_loss
        )
        indices = self._fused_indices(query, window, topk, state, need_window=need_window)
        if window is None and need_window:
            window = state.fused_window_indices
        if isinstance(scores, CSA2IndexerInputs):
            return fused_csa2_indexer_sparse_attn(
                scores,
                query,
                local,
                global_kv,
                self.attn_sink,
                window,
                topk,
                self.softmax_scale,
                self.config.dsa_indexer_loss_coeff,
                self.config.dsa_indexer_use_sparse_loss,
                self.config.calculate_per_token_loss,
                global_kv_flat=state.global_kv_flat,
                attention_indices=indices,
                attention_topk_length=state.fused_topk_length,
                q_padding_mask=state.fused_q_padding_mask,
            )
        # Keep independently selectable native indexer math for diagnosis.
        output = self._fused_attention(query, local, window, global_kv, topk, state)
        loss = self._compute_indexer_loss(
            query, local, global_kv, window, topk, scores, thd_layout=state.thd_layout
        )
        return output, loss

    def _forward_fused_indexer_no_loss(self, query, local, window, x, qr, state):
        """Full/Reindex without auxiliary supervision; selection stays discrete."""
        global_kv, topk, _ = self._shared_global_attention(x, qr, state)
        return self._fused_attention(query, local, window, global_kv, topk, state), None

    def _forward_fused_reuse(self, query, local, window, x, qr, state):
        """Consume the owner's shared KV/Top-K without indexer projection or loss."""
        global_kv, topk, _ = self._shared_global_attention(x, qr, state)
        return self._fused_attention(query, local, window, global_kv, topk, state), None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        packed_seq_params: PackedSeqParams | None = None,
        x: torch.Tensor,
        qr: torch.Tensor,
        boundary_hidden: torch.Tensor | None = None,
        boundary_kv: torch.Tensor | None = None,
        csa2_state: CSA2State | None = None,
        thd_layout: CSA2THDLayout | None = None,
    ) -> torch.Tensor:
        """Attend jointly to causal window and selected global latents using one softmax.

        As in DSv4 CSA, attention_mask is unused: sparse indices and packed
        sequence metadata define causal visibility and sequence boundaries.
        """
        if boundary_hidden is not None or boundary_kv is not None:
            raise NotImplementedError("Native CSA2 currently requires CP=1.")
        is_thd = packed_seq_params is not None
        if is_thd:
            if packed_seq_params.qkv_format != "thd":
                raise NotImplementedError("CSA2 packed sequences require qkv_format='thd'.")
            if query.ndim != 3 or key.shape != (query.shape[0], 1, query.shape[-1]):
                raise ValueError("CSA2 THD expects query [T, heads, dim] and key [T, 1, dim].")
            if thd_layout is None:
                thd_layout = build_csa2_thd_layout(packed_seq_params, query.shape[0])
            else:
                thd_layout.validate_compatible(packed_seq_params, query.shape[0])
            # The DSv4 QKV wrapper returns packed qr without its dummy batch axis.
            if qr.ndim == 2:
                qr = qr.unsqueeze(1)
            if x.ndim != 3 or x.shape[:2] != (query.shape[0], 1) or qr.shape[:2] != x.shape[:2]:
                raise ValueError(
                    "CSA2 THD hidden states and query latents must have shape [T, 1, dim]."
                )
            invalid = ~thd_layout.valid_tokens[:, None, None]
            query = query.masked_fill(invalid, 0)
            key = key.masked_fill(invalid, 0)
            x = x.masked_fill(invalid, 0)
            qr = qr.masked_fill(invalid, 0)
        elif thd_layout is not None:
            raise ValueError("CSA2 THD layout requires packed sequence metadata.")
        elif query.ndim != 4 or query.shape[0] == 0:
            raise ValueError("CSA2 expects a nonempty [sequence, batch, heads, head_dim] query.")
        if csa2_state is None:
            if self.compress_ratio and not self.is_kv_source:
                raise ValueError(
                    "CSA2 Reindex/Reuse requires an explicit csa2_state shared with its Full "
                    "owner for this forward."
                )
            csa2_state = CSA2State()
        csa2_state.validate_forward(self.layer_idx, query, thd_layout=thd_layout)
        seq_len, batch = query.shape[0], 1 if is_thd else query.shape[1]
        if is_thd and self.use_fused_kernels and query.is_cuda and seq_len > 0:
            # Build the final physical indices after selection. Reuse layers
            # consume the source's cache without constructing a window matrix.
            indices = None
        elif is_thd:
            indices = get_window_topk_idxs_thd(
                self.config.csa_window_size, thd_layout.cu_seqlens_padded, total_q=seq_len
            )
            # DSv4's window helper returns segment-local positions; native THD
            # sparse attention consumes physical flat rows in the packed KV.
            starts = thd_layout.cu_seqlens_padded[thd_layout.sequence_ids.clamp_min(0)]
            indices = torch.where(indices >= 0, indices + starts[:, None], -1)
            valid_window = (
                (indices >= 0)
                & thd_layout.valid_tokens[:, None]
                & thd_layout.valid_tokens[indices.clamp_min(0).long()]
            )
            indices = indices.masked_fill(~valid_window, -1)
        else:
            indices = get_window_topk_idxs(
                self.config.csa_window_size, batch, seq_len, query.device
            )
        local = key.squeeze(-2)
        use_indexer_loss = (
            self.training
            and torch.is_grad_enabled()
            and (self.config.dsa_indexer_loss_coeff or 0.0) > 0
        )
        if not self.use_fused_kernels or seq_len == 0:
            output, indexer_loss = self._forward_unfused(
                query, local, indices, x, qr, csa2_state, use_indexer_loss
            )
        elif not self.compress_ratio:
            output, indexer_loss = self._forward_fused_no_indexer(query, local, indices, csa2_state)
        elif not self.is_index_source:
            output, indexer_loss = self._forward_fused_reuse(
                query, local, indices, x, qr, csa2_state
            )
        elif use_indexer_loss:
            output, indexer_loss = self._forward_fused_indexer_training(
                query, local, indices, x, qr, csa2_state
            )
        else:
            output, indexer_loss = self._forward_fused_indexer_no_loss(
                query, local, indices, x, qr, csa2_state
            )
        if is_thd:
            output = output.masked_fill(~thd_layout.valid_tokens[:, None], 0)
        if indexer_loss is not None:
            # The tracker averages microbatches, so report a token mean in either
            # reduction mode. Backward still receives the raw sum in per-token mode.
            logged_loss = indexer_loss.detach()
            if self.config.calculate_per_token_loss:
                token_count = (
                    thd_layout.valid_tokens.sum().clamp_min(1) if is_thd else seq_len * batch
                )
                logged_loss = logged_loss / token_count
            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss=logged_loss, layer_number=self.layer_idx + 1, num_layers=self.config.num_layers
            )
            output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)
        csa2_state.last_layer = self.layer_idx
        return output

    def backward_dw(self) -> None:
        """Flush the compressor and indexer linears using the existing DSv4 interface."""
        if self.compressor is not None:
            self.compressor.backward_dw()
        if self.indexer is not None:
            self.indexer.backward_dw()
