"""vLLM-visible sparse attention assembled on the Lite DS4 container."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from vllm import envs
from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    deepgemm_post_process_fp8_weight_block,
    per_token_group_quant_fp8,
)
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.utils.deep_gemm import fp8_gemm_nt
from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm.primitive.block_fp8 import (
    DeploymentBlockFP8Adapter,
    DeploymentFusedBlockFP8Adapter,
    quantize_block_fp8_weight,
    bind_source_scale_to_visible_weight,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.attention.backward import (
    attach_indexer_aux_loss,
    compressed_compact_graph,
    visible_sparse_attention,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.attention.runtime import (
    AttentionKernelMetadata,
    c128_all_visible_topk,
    compressed_width,
    compressor_operation,
    gather_cp_compressed_rows,
    official_compact_compressed_visible,
    official_indexer_topk,
    official_local_qk_visible,
    prepare_cp_compression_geometry,
    quantized_main_k_visible,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.dense import (
    block_fp8_linear,
    fused_block_fp8_linear,
    visible_linear,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.dense import (
    check_parameter_versions,
    fused_qkv_rms_norm,
    native_linear_vjp,
    parameter_versions,
    visible_functional_vjp,
)
from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention
from megatron.lite.primitive.parallel import ParallelState


def _fp32_linear(value: torch.Tensor, *weights: torch.Tensor) -> torch.Tensor:
    return torch.mm(
        value.contiguous(), torch.cat(weights, dim=0).T, out_dtype=torch.float32
    )


def insert_qkv(
    q: Tensor,
    kv: Tensor,
    cache: Tensor,
    slot_mapping: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    *,
    eps: float,
    block_size: int,
    padded_heads: int,
) -> Tensor:
    return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q,
        kv,
        cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        padded_heads,
        eps,
        block_size,
    )


def o_projection_visible(
    o: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    wo_a: Tensor,
    wo_b: Tensor,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    return_intermediate: bool = False,
    packed_weights: tuple[Any, torch.Tensor, torch.Tensor] | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    if packed_weights is None:
        with torch.no_grad():
            packed_weights = _pack_o_projection_weights(wo_a, wo_b, n_groups)
    packed_wa, wb_q, wb_s = packed_weights

    intermediate = None

    def packed_wb(value: Tensor) -> Tensor:
        nonlocal intermediate
        if return_intermediate:
            intermediate = value
        aligned = bool(envs.VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES)
        aq, a_s = per_token_group_quant_fp8(
            value,
            128,
            use_ue8m0=True,
            column_major_scales=True,
            tma_aligned_scales=aligned,
        )
        output = torch.empty(
            value.shape[0], wb_q.shape[0], dtype=torch.bfloat16, device=value.device
        )
        fp8_gemm_nt(
            (aq, a_s),
            (wb_q, wb_s),
            output,
            is_deep_gemm_e8m0_used=True,
        )
        return output

    with torch.no_grad():
        recipe, aligned = compute_fp8_einsum_recipe()
        output = deep_gemm_fp8_o_proj(
            o,
            positions,
            cos_sin_cache,
            packed_wa,
            packed_wb,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
            einsum_recipe=recipe,
            tma_aligned_scales=aligned,
        )
    if not return_intermediate:
        return output
    if intermediate is None:
        raise RuntimeError("vLLM o-projection did not expose its BF16 intermediate")
    return output, intermediate


def _pack_o_projection_weights(
    wo_a: Tensor,
    wo_b: Tensor,
    n_groups: int,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        canonical_wa = quantize_block_fp8_weight(wo_a)
        wa_q, wa_s = deepgemm_post_process_fp8_weight_block(
            wq=canonical_wa.qweight,
            ws=canonical_wa.scales,
            quant_block_shape=(128, 128),
            use_e8m0=True,
            is_bmm=True,
            bmm_batch_size=n_groups,
        )
        packed_wa = type("_PackedGroupedWeight", (), {})()
        packed_wa.weight = wa_q
        packed_wa.weight_scale = wa_s

        canonical_wb = quantize_block_fp8_weight(wo_b)
        wb_q, wb_s = deepgemm_post_process_fp8_weight_block(
            wq=canonical_wb.qweight,
            ws=canonical_wb.scales,
            quant_block_shape=(128, 128),
            use_e8m0=True,
        )
    return packed_wa, wb_q, wb_s


def _inverse_rope(o, positions, cache, nope_dim, rope_dim):
    prefix, rope = o[..., :nope_dim], o[..., nope_dim : nope_dim + rope_dim]
    selected = cache.index_select(0, positions.long()).float()
    cos = selected[..., : rope_dim // 2].unsqueeze(-2)
    sin = selected[..., rope_dim // 2 : rope_dim].unsqueeze(-2)
    even, odd = rope[..., 0::2].float(), rope[..., 1::2].float()
    rotated = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1)
    return torch.cat((prefix.float(), rotated.flatten(-2)), dim=-1).to(o.dtype)


def _inverse_rope_vjp(
    grad_inverse: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    nope_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    grad_prefix = grad_inverse[..., :nope_dim]
    grad_rope = grad_inverse[..., nope_dim : nope_dim + rope_dim]
    selected = cache.index_select(0, positions.long()).float()
    cos = selected[..., : rope_dim // 2].unsqueeze(-2)
    sin = selected[..., rope_dim // 2 : rope_dim].unsqueeze(-2)
    grad_even = grad_rope[..., 0::2].float()
    grad_odd = grad_rope[..., 1::2].float()
    original_even = grad_even * cos - grad_odd * sin
    original_odd = grad_even * sin + grad_odd * cos
    original_rope = torch.stack((original_even, original_odd), dim=-1).flatten(-2)
    return torch.cat((grad_prefix.float(), original_rope), dim=-1).to(
        grad_inverse.dtype
    )


def _te_grouped_gemm(
    lhs: tuple[torch.Tensor, ...],
    rhs: tuple[torch.Tensor, ...],
    output: torch.Tensor | tuple[torch.Tensor, ...],
    *,
    activation_dtype: torch.dtype,
    layout: str = "TN",
    m_splits: tuple[int, ...] | None = None,
    single_output: bool = False,
    grad: bool = False,
    use_split_accumulator: bool = False,
) -> None:
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

    outputs = [output] if isinstance(output, torch.Tensor) else list(output)
    general_grouped_gemm(
        list(lhs),
        list(rhs),
        outputs,
        [None] * len(lhs),
        activation_dtype,
        single_output=single_output,
        layout=layout,
        m_splits=list(m_splits) if m_splits is not None else None,
        grad=grad,
        use_split_accumulator=use_split_accumulator,
    )


@contextmanager
def _nvtx_range(name: str):
    if not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class _OProjectionVJP(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        visible_op: Callable,
        o: torch.Tensor,
        wo_a: torch.Tensor,
        wo_b: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        n_groups: int,
        nope_dim: int,
        rope_dim: int,
        o_lora_rank: int,
    ) -> torch.Tensor:
        visible_result = visible_op(o, wo_a, wo_b)
        if not isinstance(visible_result, tuple) or len(visible_result) != 2:
            raise RuntimeError(
                "training o-projection must return (visible_output, bf16_intermediate)"
            )
        output, z = visible_result
        ctx.save_for_backward(o, z, wo_a, wo_b, positions, cos_sin_cache)
        ctx.versions = parameter_versions((wo_a, wo_b))
        ctx.n_groups = n_groups
        ctx.nope_dim = nope_dim
        ctx.rope_dim = rope_dim
        ctx.o_lora_rank = o_lora_rank
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        with _nvtx_range("o_projection_backward"):
            return _OProjectionVJP._backward_impl(ctx, grad_output)

    @staticmethod
    def _backward_impl(ctx: Any, grad_output: torch.Tensor):
        o, z, wo_a, wo_b, positions, cos_sin_cache = ctx.saved_tensors
        check_parameter_versions((wo_a, wo_b), ctx.versions)
        n_groups = ctx.n_groups
        rank = ctx.o_lora_rank
        tokens = o.shape[0]
        activation_dtype = o.dtype

        inverse = _inverse_rope(
            o, positions, cos_sin_cache, ctx.nope_dim, ctx.rope_dim
        )
        grouped = inverse.reshape(tokens, n_groups, -1)
        grouped_packed = grouped.transpose(0, 1).contiguous()
        grouped_mats = tuple(grouped_packed.unbind(0))
        wa_packed = wo_a.reshape(n_groups, rank, -1).contiguous()
        wa_mats = tuple(wa_packed.unbind(0))
        splits = (tokens,) * n_groups

        with _nvtx_range("o_projection_backward/wo_b_vjp"):
            grad_z, grad_wo_b = native_linear_vjp(
                grad_output.reshape(tokens, -1),
                z,
                wo_b,
            )
        grad_z_packed = (
            grad_z.reshape(tokens, n_groups, rank).transpose(0, 1).contiguous()
        )
        grad_z_mats = tuple(grad_z_packed.unbind(0))

        grad_grouped_packed = torch.empty_like(grouped_packed)
        with _nvtx_range("o_projection_backward/wo_a_dgrad"):
            _te_grouped_gemm(
                wa_mats,
                grad_z_mats,
                grad_grouped_packed.view(n_groups * tokens, -1),
                activation_dtype=activation_dtype,
                layout="NN",
                m_splits=splits,
                single_output=True,
                grad=True,
                use_split_accumulator=True,
            )
        grad_inverse = (
            grad_grouped_packed.transpose(0, 1).contiguous().reshape(inverse.shape)
        )
        grad_o = _inverse_rope_vjp(
            grad_inverse,
            positions,
            cos_sin_cache,
            ctx.nope_dim,
            ctx.rope_dim,
        )

        grad_wo_a_packed = torch.empty_like(wa_packed)
        with _nvtx_range("o_projection_backward/wo_a_wgrad"):
            _te_grouped_gemm(
                grouped_mats,
                grad_z_mats,
                tuple(grad_wo_a_packed.unbind(0)),
                activation_dtype=activation_dtype,
                layout="NT",
                m_splits=splits,
                grad=True,
                use_split_accumulator=True,
            )
        return (
            None,
            grad_o,
            grad_wo_a_packed.reshape_as(wo_a),
            grad_wo_b,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _o_projection(
    visible_op: Callable,
    o: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    *,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
):
    if not o.is_cuda:
        def functional(o_, wa_, wb_):
            inverse = _inverse_rope(
                o_, positions, cos_sin_cache, nope_dim, rope_dim
            )
            grouped = inverse.reshape(inverse.shape[0], n_groups, -1)
            wa = wa_.to(grouped.dtype).reshape(n_groups, o_lora_rank, -1)
            z = torch.einsum("tgd,grd->tgr", grouped, wa)
            return F.linear(z.flatten(1), wb_.to(z.dtype)).to(o_.dtype)

        return visible_functional_vjp(
            visible_op, functional, (o, wo_a, wo_b), version_indices=(1, 2)
        )
    if not torch.is_grad_enabled():
        return visible_op(o, wo_a, wo_b)
    return _OProjectionVJP.apply(
        visible_op,
        o,
        wo_a,
        wo_b,
        positions,
        cos_sin_cache,
        n_groups,
        nope_dim,
        rope_dim,
        o_lora_rank,
    )


class VLLMAttention(CompressedSparseAttention):
    def __init__(
        self,
        config: DeepseekV4Config,
        *,
        ps=None,
        layer_idx: int,
        indexer_loss_coeff: float = 0.0,
        cache_deployment_weights: bool = False,
    ):
        ps = ps or ParallelState()
        super().__init__(config, layer_idx=layer_idx, ps=ps)
        self.config = config
        configured_ratio = (
            config.compress_ratios[layer_idx]
            if layer_idx < len(config.compress_ratios)
            else 0
        )
        self.compress_ratio = max(1, configured_ratio)
        self.indexer_loss_coeff = indexer_loss_coeff
        self.fused_linear = DeploymentFusedBlockFP8Adapter(
            cache_weight=cache_deployment_weights
        )
        self.q_linear = DeploymentBlockFP8Adapter(cache_weight=cache_deployment_weights)
        self.indexer_q_linear = DeploymentBlockFP8Adapter(
            cache_weight=cache_deployment_weights
        )
        self._cache_deployment_weights = cache_deployment_weights
        self._o_projection_weight_cache: (
            tuple[tuple[tuple[int, int], tuple[int, int]], tuple[Any, Tensor, Tensor]]
            | None
        ) = None
        self._projection_streams: list[torch.cuda.Stream] | None = None
        self._projection_events: list[torch.cuda.Event] | None = None

    def clear_deployment_weight_cache(self) -> None:
        self.fused_linear.clear_cache()
        self.q_linear.clear_cache()
        self.indexer_q_linear.clear_cache()
        self._o_projection_weight_cache = None

    def _packed_o_projection_weights(
        self,
        wo_a: Tensor,
        wo_b: Tensor,
    ) -> tuple[Any, Tensor, Tensor] | None:
        if not self._cache_deployment_weights:
            return None
        key = (
            (id(wo_a), wo_a._version),
            (id(wo_b), wo_b._version),
        )
        if (
            self._o_projection_weight_cache is None
            or self._o_projection_weight_cache[0] != key
        ):
            self._o_projection_weight_cache = (
                key,
                _pack_o_projection_weights(wo_a, wo_b, self.config.o_groups),
            )
        return self._o_projection_weight_cache[1]

    def _output_projection(
        self,
        result: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        heads_per_group = self.config.num_attention_heads // self.config.o_groups
        nope_dim = self.config.head_dim - self.config.qk_rope_head_dim
        bind_source_scale_to_visible_weight(self.wo_a, "weight", self.wo_a.weight)
        bind_source_scale_to_visible_weight(self.wo_b, "weight", self.wo_b.weight)
        needs_intermediate = torch.is_grad_enabled()
        packed_weights = self._packed_o_projection_weights(
            self.wo_a.weight,
            self.wo_b.weight,
        )
        return _o_projection(
            lambda value, wa, wb: o_projection_visible(
                value,
                positions,
                cos_sin_cache,
                wa,
                wb,
                n_groups=self.config.o_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=self.config.qk_rope_head_dim,
                o_lora_rank=self.config.o_lora_rank,
                return_intermediate=needs_intermediate,
                packed_weights=packed_weights,
            ),
            result,
            self.wo_a.weight,
            self.wo_b.weight,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            n_groups=self.config.o_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=self.config.qk_rope_head_dim,
            o_lora_rank=self.config.o_lora_rank,
        )

    def _input_projections(self, hidden_states: torch.Tensor):
        self._bind_input_projection_source_scales()

        def fused_projection():
            return fused_block_fp8_linear(
                self.fused_linear,
                hidden_states,
                self.wq_a.weight,
                self.wkv.weight,
            )

        aux_fns: list[Callable[[], torch.Tensor] | None] = [None, None, None]
        if self.compressor is not None:
            aux_fns[0] = lambda: fused_block_fp8_linear(
                _fp32_linear,
                hidden_states,
                self.compressor.wkv.weight,
                self.compressor.wgate.weight,
            )
        if self.indexer is not None:
            aux_fns[1] = lambda: visible_linear(
                linear_batch_invariant,
                hidden_states,
                self.indexer.weights_proj.weight,
            )
            aux_fns[2] = lambda: fused_block_fp8_linear(
                _fp32_linear,
                hidden_states,
                self.indexer.compressor.wkv.weight,
                self.indexer.compressor.wgate.weight,
            )
        if self._projection_streams is None:
            self._projection_streams = [torch.cuda.Stream() for _ in range(3)]
        if self._projection_events is None:
            self._projection_events = [torch.cuda.Event() for _ in range(4)]
        assert self._projection_events is not None
        from vllm.utils.multi_stream_utils import execute_in_parallel

        default_output, aux_outputs = execute_in_parallel(
            fused_projection,
            aux_fns,
            self._projection_events[0],
            self._projection_events[1:],
            self._projection_streams,
            enable=True,
        )
        # Transfer allocator ownership from the auxiliary streams.
        current_stream = torch.cuda.current_stream(hidden_states.device)
        for output in aux_outputs:
            if isinstance(output, torch.Tensor):
                output.record_stream(current_stream)
        return default_output, aux_outputs

    def _bind_input_projection_source_scales(self) -> None:
        projections = [self.wq_a, self.wkv]
        if self.compressor is not None:
            projections.extend((self.compressor.wkv, self.compressor.wgate))
        if self.indexer is not None:
            projections.extend(
                (self.indexer.compressor.wkv, self.indexer.compressor.wgate)
            )
        for projection in projections:
            bind_source_scale_to_visible_weight(projection, "weight", projection.weight)

    def _project_boundary_k(
        self,
        boundary_hidden: torch.Tensor,
        kv_visible: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        global_start: int,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        if boundary_hidden.shape[0] == 0:
            return kv_visible[:0]

        from megatron.core.transformer.experimental_attention_variant.csa_utils import (
            cp_utils,
        )

        rows = boundary_hidden.shape[0]
        boundary_qr_kv = fused_block_fp8_linear(
            self.fused_linear,
            boundary_hidden,
            self.wq_a.weight,
            self.wkv.weight,
        )
        boundary_qr, boundary_kv = boundary_qr_kv.split(
            [self.config.q_lora_rank, self.config.head_dim], dim=-1
        )
        _, boundary_kv = fused_qkv_rms_norm(
            fused_q_kv_rmsnorm,
            boundary_qr,
            boundary_kv,
            self.q_norm.weight,
            self.kv_norm.weight,
            self.config.rms_norm_eps,
        )
        positions = cp_utils._thd_cp_position_ids(
            cu_seqlens, global_start - rows, rows
        ).to(torch.int64)
        dummy_q = boundary_kv.new_zeros(
            (rows, self.config.num_attention_heads, self.config.head_dim)
        )
        _, boundary_k = official_local_qk_visible(
            dummy_q,
            boundary_kv,
            positions,
            cos_sin_cache,
            insert_qkv,
            eps=self.config.rms_norm_eps,
            rope_dim=self.config.qk_rope_head_dim,
            padded_heads=self.config.num_attention_heads,
        )
        return boundary_k

    def _forward_training_attention(
        self,
        hidden_states: torch.Tensor,
        metadata: AttentionKernelMetadata,
        q: torch.Tensor,
        kv: torch.Tensor,
        compressor_kv_score: torch.Tensor | None,
        index_q: torch.Tensor | None,
        indexer_weights: torch.Tensor | None,
        indexer_kv_score: torch.Tensor | None,
    ) -> torch.Tensor:
        cp_size = 1 if self.ps is None else self.ps.cp_size
        cp_rank = 0 if self.ps is None else self.ps.cp_rank
        cp_group = None if self.ps is None else self.ps.cp_group
        if cp_size > 1 and cp_group is None:
            raise RuntimeError("DS4 CP requires a model-owned CP group")
        if cp_size > 1 and self.indexer_loss_coeff:
            raise NotImplementedError(
                "DS4 CP indexer auxiliary loss is not implemented"
            )

        from megatron.core.transformer.experimental_attention_variant.csa_utils import (
            cp_layout_kernels,
            cp_utils,
        )

        psp = metadata.packed_seq_params
        cu_seqlens = (
            psp.cu_seqlens_q_padded
            if psp.cu_seqlens_q_padded is not None
            else psp.cu_seqlens_q
        )
        if cu_seqlens is None:
            raise RuntimeError("DS4 packed attention requires cu_seqlens_q")
        sequence_boundaries = metadata.sequence_boundaries
        compressed_boundaries = metadata.compressed_boundaries
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() != len(
            sequence_boundaries
        ):
            raise RuntimeError(
                "DS4 host sequence boundaries do not match GPU cu_seqlens shape"
            )
        if metadata.cp_size != cp_size:
            raise RuntimeError("DS4 host geometry CP size does not match model CP")
        l_local = hidden_states.shape[0]
        if sequence_boundaries[-1] != l_local * cp_size:
            raise RuntimeError(
                "DS4 host sequence terminal does not match CP-local hidden rows"
            )
        global_start = cp_rank * l_local
        positions = metadata.positions.reshape(-1).to(torch.int64)
        if positions.numel() != l_local:
            raise RuntimeError("DS4 CP position rows do not match local tokens")
        q_visible, kv_visible = official_local_qk_visible(
            q,
            kv,
            positions,
            metadata.cos_sin_cache,
            insert_qkv,
            eps=self.config.rms_norm_eps,
            rope_dim=self.config.qk_rope_head_dim,
            padded_heads=self.config.num_attention_heads,
        )

        boundary_hidden = (
            cp_utils.exchange_cp_boundary_hidden(
                hidden_states,
                self.compress_ratio,
                self.config.sliding_window,
                cp_group,
            )
            if cp_size > 1
            else hidden_states[:0]
        )
        d_window = boundary_hidden.shape[0]
        boundary_k_visible = self._project_boundary_k(
            boundary_hidden,
            kv_visible,
            cu_seqlens,
            global_start=global_start,
            cos_sin_cache=metadata.cos_sin_cache,
        )

        compressed_rank_major = hidden_states.new_empty((0, self.config.head_dim))
        cu_seqlens_compressed = None
        seq_to_rank_row = None
        compressed_topk = None
        ratio = max(1, self.compress_ratio)
        if self.compressor is not None and ratio > 1:
            compression_geometry = prepare_cp_compression_geometry(
                hidden_states,
                boundary_hidden,
                cu_seqlens,
                global_start=global_start,
                cp_size=cp_size,
                ratio=ratio,
            )
            cu_seqlens_compressed = compression_geometry.cu_seqlens_compressed
            hidden_compact = compression_geometry.hidden_compact
            group_ids = compression_geometry.compressed_group_ids
            seq_to_rank_row = compression_geometry.seq_to_rank_row
            if cp_size == 1:
                # Preserve the authoritative non-CP request geometry exactly.
                # CP-local capacities include padding and must not replace the
                # request-local compressed boundaries in this path.
                local_compressed_boundaries = compressed_boundaries
            else:
                local_boundaries = [0]
                local_end = global_start + l_local
                d_comp = 8 if ratio == 4 else ratio
                for seq_start, seq_end in zip(
                    sequence_boundaries, sequence_boundaries[1:]
                ):
                    owned_start = max(global_start, seq_start)
                    owned_end = min(local_end, seq_end)
                    first_group = max(
                        0,
                        (owned_start - seq_start - d_comp + ratio - 1) // ratio,
                    )
                    last_group = max(0, (owned_end - seq_start) // ratio)
                    local_boundaries.append(
                        local_boundaries[-1] + max(0, last_group - first_group)
                    )
                local_compressed_boundaries = tuple(local_boundaries)
            if cu_seqlens_compressed.numel() != len(compressed_boundaries):
                raise RuntimeError(
                    "DS4 host compressed boundaries do not match GPU metadata shape"
                )
            # Host boundaries describe the global sequence, while CP geometry
            # contains only this rank's compact rows.
            if cp_size == 1 and compressed_boundaries[-1] > group_ids.numel():
                raise RuntimeError(
                    "DS4 host compressed terminal exceeds compact GPU rows"
                )
            compact_score = fused_block_fp8_linear(
                _fp32_linear,
                hidden_compact,
                self.compressor.wkv.weight,
                self.compressor.wgate.weight,
            )
            compressed_graph = compressed_compact_graph(
                compact_score,
                self.compressor.ape,
                self.compressor.norm.weight,
                group_ids,
                metadata.cos_sin_cache,
                ratio=ratio,
                head_dim=self.config.head_dim,
                rope_dim=self.config.qk_rope_head_dim,
                eps=self.config.rms_norm_eps,
            )
            if ratio in (4, 128):
                if metadata.compressor_metadata is None:
                    raise RuntimeError(
                        "C4/C128 CP requires caller-owned official compressor metadata"
                    )
                compressed_graph = official_compact_compressed_visible(
                    compressed_graph,
                    compact_score,
                    self.compressor.ape,
                    self.compressor.norm.weight,
                    group_ids,
                    metadata.cos_sin_cache,
                    operation=compressor_operation,
                    runtime_metadata=metadata.compressor_metadata,
                    ratio=ratio,
                    head_dim=self.config.head_dim,
                    compressed_boundaries=local_compressed_boundaries,
                )
                if compressed_graph.shape[0] > group_ids.shape[0]:
                    raise RuntimeError(
                        "DS4 official compressor exceeded CP rank capacity"
                    )
                if cp_size > 1 and compressed_graph.shape[0] < group_ids.shape[0]:
                    compressed_graph = torch.cat(
                        (
                            compressed_graph,
                            compressed_graph.new_zeros(
                                (group_ids.shape[0] - compressed_graph.shape[0],)
                                + compressed_graph.shape[1:]
                            ),
                        ),
                        dim=0,
                    )
                compressed_local = compressed_graph
            else:
                compressed_local = quantized_main_k_visible(compressed_graph)
            if cp_size > 1:
                compressed_rank_major, _ = gather_cp_compressed_rows(
                    compressed_local,
                    seq_to_rank_row,
                    cp_group=cp_group,
                )
            else:
                compressed_rank_major = compressed_local

            width = compressed_width(
                int(psp.max_seqlen_q), ratio, self.config.index_topk
            )
            if self.indexer is not None:
                if indexer_kv_score is None or indexer_weights is None:
                    raise RuntimeError("C4 CP requires indexer projections")
                assert index_q is not None
                compact_index_score = fused_block_fp8_linear(
                    _fp32_linear,
                    hidden_compact.detach(),
                    self.indexer.compressor.wkv.weight,
                    self.indexer.compressor.wgate.weight,
                )
                index_k_local = compressed_compact_graph(
                    compact_index_score,
                    self.indexer.compressor.ape.detach(),
                    self.indexer.compressor.norm.weight.detach(),
                    group_ids,
                    metadata.cos_sin_cache,
                    ratio=ratio,
                    head_dim=self.config.index_head_dim,
                    rope_dim=self.config.qk_rope_head_dim,
                    eps=self.config.rms_norm_eps,
                )
                if metadata.indexer_compressor_metadata is None:
                    raise RuntimeError(
                        "C4 indexer requires official compressor metadata"
                    )
                index_k_local = official_compact_compressed_visible(
                    index_k_local,
                    compact_index_score,
                    self.indexer.compressor.ape.detach(),
                    self.indexer.compressor.norm.weight.detach(),
                    group_ids,
                    metadata.cos_sin_cache,
                    operation=compressor_operation,
                    runtime_metadata=metadata.indexer_compressor_metadata,
                    ratio=ratio,
                    head_dim=self.config.index_head_dim,
                    compressed_boundaries=local_compressed_boundaries,
                )
                if index_k_local.shape[0] > group_ids.shape[0]:
                    raise RuntimeError(
                        "DS4 official indexer compressor exceeded CP rank capacity"
                    )
                if cp_size > 1 and index_k_local.shape[0] < group_ids.shape[0]:
                    index_k_local = torch.cat(
                        (
                            index_k_local,
                            index_k_local.new_zeros(
                                (group_ids.shape[0] - index_k_local.shape[0],)
                                + index_k_local.shape[1:]
                            ),
                        ),
                        dim=0,
                    )
                if cp_size > 1:
                    index_k_rank_major, index_k_seq_major = gather_cp_compressed_rows(
                        index_k_local,
                        seq_to_rank_row,
                        cp_group=cp_group,
                    )
                else:
                    index_k_seq_major = index_k_local
                compressed_topk = official_indexer_topk(
                    index_q,
                    indexer_weights,
                    index_k_seq_major,
                    positions,
                    metadata.cos_sin_cache,
                    sequence_boundaries=sequence_boundaries,
                    compressed_boundaries=compressed_boundaries,
                    global_start=global_start,
                    ratio=ratio,
                    topk=width,
                )
            else:
                compressed_topk = c128_all_visible_topk(
                    positions, width=width, ratio=ratio
                )
        else:
            width = 0

        workspace = torch.cat(
            (boundary_k_visible, kv_visible, compressed_rank_major), dim=0
        ).view(-1, 1, self.config.head_dim)
        indexer_topk_for_loss = compressed_topk
        if compressed_topk is not None:
            # Canonical compressed order followed by chronological SWA.
            compressed_topk = torch.sort(
                compressed_topk, dim=-1, descending=True
            ).values
        indices, topk_length, _ = cp_layout_kernels.build_attention_indices(
            cu_seqlens,
            global_start,
            l_local,
            d_window,
            self.config.sliding_window,
            ratio,
            width,
            compressed_topk,
            cu_seqlens_compressed=cu_seqlens_compressed,
            seq_to_rank_row=seq_to_rank_row,
            for_indexer_loss=False,
        )
        if topk_length is None:
            raise RuntimeError("DS4 CP index lowering must return lengths")
        if compressed_topk is not None:
            # Rotate MCore [window|compressed] into vLLM [compressed|window].
            window_count = torch.minimum(
                positions + 1,
                torch.tensor(
                    self.config.sliding_window,
                    dtype=positions.dtype,
                    device=positions.device,
                ),
            ).to(torch.int64)
            compressed_count = topk_length.to(torch.int64) - window_count
            columns = torch.arange(indices.shape[-1], device=indices.device).unsqueeze(
                0
            )
            source = torch.where(
                columns < compressed_count.unsqueeze(1),
                window_count.unsqueeze(1) + columns,
                columns - compressed_count.unsqueeze(1),
            ).clamp_min(0)
            indices = torch.gather(indices, 1, source)
            indices = torch.where(
                columns < topk_length.to(torch.int64).unsqueeze(1),
                indices,
                torch.full_like(indices, -1),
            )
        indices = indices.unsqueeze(1)
        scale = self.config.head_dim**-0.5
        sink = self.sinks.float().contiguous()
        if cu_seqlens.numel() > 2:
            # vLLM's BI prefill path launches FlashMLA with request-local M/N.
            # Keep that geometry for every CP size. Build one canonical packed
            # request workspace and remap all physical CP indices in two mLite
            # CuTe kernels, then retain request-owned slices for FlashMLA.
            if cu_seqlens_compressed is None and seq_to_rank_row is None:
                cu_seqlens_compressed = torch.zeros_like(cu_seqlens)
                seq_to_rank_row = torch.empty(
                    0, dtype=torch.int32, device=cu_seqlens.device
                )
            elif cu_seqlens_compressed is None or seq_to_rank_row is None:
                raise RuntimeError(
                    "DS4 request-local compressed metadata must be jointly present"
                )
            if cu_seqlens_compressed.numel() != len(compressed_boundaries):
                raise RuntimeError(
                    "DS4 request-local host and GPU boundary counts differ"
                )
            from .request_local_layout import build_request_local_layout

            request_indices, workspace_row_map = build_request_local_layout(
                indices,
                cu_seqlens,
                cu_seqlens_compressed,
                seq_to_rank_row,
                global_start=global_start,
                l_local=l_local,
                d_window=d_window,
                physical_workspace_rows=workspace.shape[0],
                total_capacity=(
                    sequence_boundaries[-1] + compressed_boundaries[-1]
                ),
            )
            result = torch.empty_like(q_visible)
            for seq_idx in range(len(sequence_boundaries) - 1):
                seq_start = sequence_boundaries[seq_idx]
                seq_end = sequence_boundaries[seq_idx + 1]
                token_start = max(0, seq_start - global_start)
                token_end = min(l_local, seq_end - global_start)
                if token_end <= token_start:
                    continue

                local_indices = request_indices[token_start:token_end].clone()
                request_start = seq_start + compressed_boundaries[seq_idx]
                request_end = seq_end + compressed_boundaries[seq_idx + 1]
                local_row_map = workspace_row_map[request_start:request_end]
                valid_workspace_rows = local_row_map < workspace.shape[0]
                local_workspace = workspace.index_select(
                    0,
                    local_row_map.clamp_max(workspace.shape[0] - 1).to(
                        torch.int64
                    ),
                )
                local_workspace.masked_fill_(
                    ~valid_workspace_rows.view(
                        (-1,) + (1,) * (workspace.ndim - 1)
                    ),
                    0,
                )
                # The CuTe backward ABI requires every tensor data pointer to
                # be 16-byte aligned.  An int32 slice starts at a potentially
                # unaligned request offset, so give it request-owned storage.
                local_lengths = topk_length[token_start:token_end].clone()
                local_output = torch.empty_like(q_visible[token_start:token_end])

                def visible_attention(q_value, kv_value):
                    return flash_mla_sparse_fwd(
                        q=q_value,
                        kv=kv_value,
                        indices=local_indices,
                        sm_scale=scale,
                        attn_sink=sink,
                        topk_length=local_lengths,
                        out=local_output,
                    )

                result[token_start:token_end] = visible_sparse_attention(
                    visible_attention,
                    q_visible[token_start:token_end],
                    local_workspace,
                    local_indices,
                    local_lengths,
                    sink,
                    softmax_scale=scale,
                )
        else:
            output_buffer = torch.empty_like(q_visible)

            def visible_attention(q_value, kv_value):
                return flash_mla_sparse_fwd(
                    q=q_value,
                    kv=kv_value,
                    indices=indices,
                    sm_scale=scale,
                    attn_sink=sink,
                    topk_length=topk_length,
                    out=output_buffer,
                )

            result = visible_sparse_attention(
                visible_attention,
                q_visible,
                workspace,
                indices,
                topk_length,
                sink,
                softmax_scale=scale,
            )
        result = result[:, : self.config.num_attention_heads, :]
        if self.indexer_loss_coeff:
            assert indexer_topk_for_loss is not None
            assert index_q is not None
            assert indexer_kv_score is not None and indexer_weights is not None
            assert compressor_kv_score is not None and self.compressor is not None
            result = attach_indexer_aux_loss(
                result,
                q,
                index_q,
                indexer_kv_score,
                indexer_weights,
                compressor_kv_score,
                self.indexer.compressor.ape,
                self.indexer.compressor.norm.weight,
                self.compressor.ape,
                self.compressor.norm.weight,
                metadata.positions,
                metadata.cos_sin_cache,
                indexer_topk_for_loss,
                ratio=self.compress_ratio,
                rope_dim=self.config.qk_rope_head_dim,
                eps=self.config.rms_norm_eps,
                softmax_scale=scale,
                loss_coeff=self.indexer_loss_coeff,
            )
        return self._output_projection(
            result,
            positions,
            metadata.cos_sin_cache,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        metadata: AttentionKernelMetadata | None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError("layer-0 attention requires flat [tokens, hidden]")
        if metadata is None:
            raise NotImplementedError("layer-0 attention requires explicit metadata")
        # FSDP2 may cast nested metadata; FlashMLA requires FP32 RoPE cache.
        if metadata.cos_sin_cache.dtype != torch.float32:
            metadata.cos_sin_cache = metadata.cos_sin_cache.float()
        qr_kv, projection_outputs = self._input_projections(hidden_states)
        compressor_kv_score, indexer_weights, indexer_kv_score = projection_outputs
        qr, kv = qr_kv.split([self.config.q_lora_rank, self.config.head_dim], dim=-1)
        qr, kv = fused_qkv_rms_norm(
            fused_q_kv_rmsnorm,
            qr,
            kv,
            self.q_norm.weight,
            self.kv_norm.weight,
            self.config.rms_norm_eps,
        )
        bind_source_scale_to_visible_weight(self.wq_b, "weight", self.wq_b.weight)
        q = block_fp8_linear(self.q_linear, qr, self.wq_b.weight).view(
            -1, self.config.num_attention_heads, self.config.head_dim
        )
        index_q = (
            block_fp8_linear(
                self.indexer_q_linear,
                qr,
                bind_source_scale_to_visible_weight(
                    self.indexer.wq_b, "weight", self.indexer.wq_b.weight
                ),
            )
            .view(-1, self.config.index_n_heads, self.config.index_head_dim)
            .contiguous()
            if self.indexer is not None
            else None
        )
        return self._forward_training_attention(
            hidden_states,
            metadata,
            q.contiguous(),
            kv,
            compressor_kv_score,
            index_q,
            indexer_weights,
            indexer_kv_score,
        )
