# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Attention Residuals (AttnRes): softmax attention over residual-stream depth.

Reference: "Attention Residuals" (Kimi Team, arXiv:2603.15031).

AttnRes replaces the standard PreNorm residual accumulation with a per-token
softmax attention over depth sources. Treating each self-attention or MLP as
one sublayer, the input to sublayer ``l`` is

    h_l = sum_j alpha_j * V_j,   alpha = softmax_j( w_l . RMSNorm(V_j) )

where ``V = [b_0, b_1, ..., b_{n-1}, partial]``:

- ``b_0`` is the token embedding,
- ``b_i`` are the summed sublayer outputs of completed depth blocks
  (Block AttnRes groups ``attn_res_block_layers`` transformer layers, i.e.
  ``2 * attn_res_block_layers`` sublayers, per block),
- ``partial`` is the running intra-block partial sum.

``w_l`` is a per-sublayer learnable pseudo-query, initialized to zero so that
the initial attention weights are uniform. The state carried from sublayer to
sublayer is the partial sum (``partial += sublayer_out``); at each block
boundary the completed partial sum is appended to the source list and a new
partial sum starts. The final output aggregates all sources with one more
pseudo-query before the final layernorm.

This module contains:

- :class:`AttentionResidual`: the per-sublayer aggregation module
  (pseudo-query + key RMSNorm weight, fp32 softmax over depth).
- Block/boundary schedule helpers shared by ``TransformerBlock``,
  ``AttnResTransformerLayer``, and pipeline-parallel shape computation.
- Payload pack/unpack helpers for pipeline parallelism. Depth sources and the
  partial sum cross pipeline-stage boundaries as a single tensor concatenated
  along the sequence dimension: ``[num_slices * s, b, h]``.
"""

import functools
import logging
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


def is_attn_res_block_start(global_layer_number: int, block_layers: int) -> bool:
    """Whether this layer opens a new depth block.

    Layer numbers are 1-based global indices. The very first layer is always a
    block start: it appends the token embedding (the initial partial sum) as
    depth source ``b_0``.

    Unit semantics of ``block_layers`` (= ``config.attn_res_block_layers``):
    in GPT stacks a layer is one TransformerLayer (two sublayers, so the
    paper's block size S = 2 * block_layers and boundaries fall on layer
    starts only); in hybrid stacks a layer is one pattern entry (one sublayer,
    S = block_layers, which may be odd).
    """
    return (global_layer_number - 1) % block_layers == 0


def attn_res_num_sources(global_layer_number: int, block_layers: int) -> int:
    """Number of depth sources visible while running this layer.

    Counts the appends performed by all block-start layers up to and including
    this one: ``floor((l - 1) / k) + 1``. The attention sublayer of a
    block-start layer sees exactly this many sources and no partial sum;
    all other sublayers additionally see the running partial sum.
    """
    return (global_layer_number - 1) // block_layers + 1


def attn_res_final_num_sources(num_layers: int, block_layers: int) -> int:
    """Number of sources aggregated by the final output head.

    Equals ``attn_res_num_sources(num_layers) + 1``: all appended sources plus
    the trailing partial sum, which always holds the last (possibly partial)
    block and is never appended by a subsequent layer. MTP layers attend over
    this same set.
    """
    return attn_res_num_sources(num_layers, block_layers) + 1


def attn_res_num_payload_slices(num_layers_before: int, block_layers: int) -> int:
    """Number of ``[s, b, h]`` slices crossing a pipeline boundary.

    ``num_layers_before`` is the number of transformer layers completed before
    the boundary (i.e. the global layer count of the sending stage's last
    layer). The payload carries every source appended so far plus the running
    partial sum: ``floor((L - 1) / k) + 2``. A zero-layer boundary (standalone
    embedding stage under account_for_embedding_in_pipeline_split) carries a
    single slice: the embedding as the initial partial sum.
    """
    assert (
        num_layers_before >= 0
    ), f"invalid pipeline boundary: num_layers_before={num_layers_before}"
    return attn_res_num_sources(num_layers_before, block_layers) + 1


@functools.lru_cache(maxsize=128)
def _hybrid_entries_before_segment(
    main_pattern: str, num_layers: int, num_segments_expected: int, segment_index: int
):
    """Layer entries owned by pipeline segments before ``segment_index`` for a hybrid pattern.

    Pipe-based patterns use cumulative '|' segment lengths (possibly uneven).
    Segments are laid out chunk-major (``segment_index = vp_stage * pp_size +
    pp_rank``), matching ``select_pipeline_segment`` in
    hybrid_layer_allocation.py. Pipe-free patterns fall back to the legacy
    even split across ``num_segments_expected`` stages.
    """
    if '|' in main_pattern:
        segments = main_pattern.split('|')
        assert len(segments) == num_segments_expected, (
            f"hybrid pattern has {len(segments)} pipe segments but the pipeline layout "
            f"expects {num_segments_expected} (= pipeline_model_parallel_size x "
            "virtual_pipeline_model_parallel_size)"
        )
        return sum(len(segment) for segment in segments[:segment_index])
    return (num_layers // num_segments_expected) * segment_index


def _hybrid_layers_before_pp_rank(main_pattern: str, num_layers: int, pp_size: int, pp_rank: int):
    """Layer entries owned by pipeline stages before ``pp_rank`` (non-interleaved)."""
    return _hybrid_entries_before_segment(main_pattern, num_layers, pp_size, pp_rank)


def attn_res_payload_slices_for_pp_rank(
    config: TransformerConfig, boundary_recv_pp_rank: int
) -> int:
    """Payload slice count for the boundary received by ``boundary_recv_pp_rank``.

    The number of layers before that boundary equals the receiving stage's
    global layer offset. For GPT stacks that comes from
    :func:`get_transformer_layer_offset` (which also handles
    ``account_for_embedding/loss_in_pipeline_split``); for hybrid stacks it
    comes from the pattern's pipe segmentation.
    """
    if getattr(config, 'is_hybrid_model', False):
        main_pattern = (config.hybrid_layer_pattern or '').split('/')[0]
        layers_before = _hybrid_layers_before_pp_rank(
            main_pattern,
            config.num_layers,
            config.pipeline_model_parallel_size,
            boundary_recv_pp_rank,
        )
        return attn_res_num_payload_slices(layers_before, config.attn_res_block_layers)

    # Imported lazily to avoid a circular import with transformer_layer.py.
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    layers_before = get_transformer_layer_offset(
        config, vp_stage=None, pp_rank=boundary_recv_pp_rank
    )
    return attn_res_num_payload_slices(layers_before, config.attn_res_block_layers)


def _sources_formed_through(num_layers: int, block_layers: int) -> int:
    """Number of depth sources appended by layers ``1..num_layers`` (0 for 0 layers)."""
    if num_layers <= 0:
        return 0
    return attn_res_num_sources(num_layers, block_layers)


def _stage_layers_before(config: TransformerConfig, global_stage: int) -> int:
    """Global layers completed before interleaved stage ``global_stage``.

    Stage ``s`` runs on pipeline rank ``s % P`` as virtual chunk ``s // P``.
    Chunks cover contiguous global layer ranges in stage order, so this also
    equals the cumulative layer count through stage ``s - 1``.
    """
    pp_size = config.pipeline_model_parallel_size
    pp_rank = global_stage % pp_size
    vp_stage = global_stage // pp_size
    if getattr(config, 'is_hybrid_model', False):
        main_pattern = (config.hybrid_layer_pattern or '').split('/')[0]
        vp_size = config.virtual_pipeline_model_parallel_size or 1
        return _hybrid_entries_before_segment(
            main_pattern, config.num_layers, pp_size * vp_size, vp_stage * pp_size + pp_rank
        )
    # Imported lazily to avoid a circular import with transformer_layer.py.
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    return get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)


def attn_res_boundary_delta_slices(
    config: TransformerConfig, boundary_recv_pp_rank: int, boundary_recv_vp_stage: int
) -> int:
    """Delta payload slice count for the boundary INTO chunk (pp_rank, vp_stage).

    Under interleaved VPP each physical rank caches every source it has seen,
    so the boundary carries only the sources formed since the receiving rank's
    previous visit (its previous virtual chunk), plus the running partial sum.
    A first-round receiver (vp_stage == 0) has seen nothing and receives the
    full prefix — which is also the non-interleaved (V=1) payload, so this
    function generalizes :func:`attn_res_payload_slices_for_pp_rank`.
    """
    pp_size = config.pipeline_model_parallel_size
    block_layers = config.attn_res_block_layers
    stage = boundary_recv_vp_stage * pp_size + boundary_recv_pp_rank
    assert stage >= 1, "the first pipeline stage does not receive a payload"
    full = _sources_formed_through(_stage_layers_before(config, stage), block_layers)
    if stage < pp_size:
        known = 0
    else:
        # Layers completed through the end of the receiver's previous chunk
        # (= layers before the stage that follows it; chunk ranges are
        # contiguous in stage order).
        known = _sources_formed_through(
            _stage_layers_before(config, stage - pp_size + 1), block_layers
        )
    return full - known + 1


_UNIFORM_SLICES_MEMO: dict = {}


def attn_res_uniform_payload_slices(config: TransformerConfig) -> int:
    """Uniform (padded) payload width for the interleaved schedule.

    The interleaved pipeline schedule uses a single tensor shape for every
    send/recv, so all boundaries pad up to the maximum delta width. The
    maximum is enumerated over all P*V - 1 boundaries (no closed form: when
    ``attn_res_block_layers`` does not divide the chunk length, or hybrid
    segments are uneven, first-round or dense-segment boundaries can dominate).
    """
    pp_size = config.pipeline_model_parallel_size
    vp_size = config.virtual_pipeline_model_parallel_size or 1
    key = (
        config.num_layers,
        pp_size,
        vp_size,
        config.attn_res_block_layers,
        getattr(config, 'is_hybrid_model', False),
        getattr(config, 'hybrid_layer_pattern', None),
        config.account_for_embedding_in_pipeline_split,
        config.account_for_loss_in_pipeline_split,
    )
    if key not in _UNIFORM_SLICES_MEMO:
        _UNIFORM_SLICES_MEMO[key] = max(
            attn_res_boundary_delta_slices(config, s % pp_size, s // pp_size)
            for s in range(1, pp_size * vp_size)
        )
    return _UNIFORM_SLICES_MEMO[key]


def pack_attn_res_payload(values: Sequence[Tensor], pad_to_slices: Optional[int] = None) -> Tensor:
    """Concatenate depth sources + partial sum along the sequence dimension.

    Produces a fresh, viewless tensor, which is required by
    ``deallocate_output_tensor`` on the pipeline-parallel send path.
    ``pad_to_slices`` zero-pads up to the interleaved schedule's uniform
    payload width; the pad carries no gradient (dropped by cat-backward) and
    is zero-filled so NaN scanners and deterministic checks stay clean.
    """
    values = list(values)
    if pad_to_slices is not None and pad_to_slices > len(values):
        ref = values[0]
        pad = torch.zeros(
            ((pad_to_slices - len(values)) * ref.shape[0], *ref.shape[1:]),
            dtype=ref.dtype,
            device=ref.device,
        )
        values.append(pad)
    return torch.cat(values, dim=0)


def unpack_attn_res_payload(
    payload: Tensor, num_slices: int, padded_slices: Optional[int] = None
) -> Tuple[List[Tensor], Tensor]:
    """Split a received payload into (depth sources, partial sum).

    The slices are contiguous views of the received leaf tensor, so gradients
    accumulate into the single tensor handed back to the pipeline schedule.
    With ``padded_slices`` (interleaved schedule) the payload is padded to the
    uniform width: the true slices are sliced off FIRST and only then chunked —
    chunking the padded buffer would silently mix pad rows into every source
    whenever the padded width happens to divide evenly.
    """
    total_slices = padded_slices if padded_slices is not None else num_slices
    assert (
        total_slices >= num_slices
    ), f"padded slice count {total_slices} smaller than real slice count {num_slices}"
    assert payload.shape[0] % total_slices == 0, (
        f"attention residual payload sequence dim {payload.shape[0]} is not divisible by "
        f"the expected slice count {total_slices}; pipeline boundary bookkeeping is broken"
    )
    seq_length = payload.shape[0] // total_slices
    real = payload[: num_slices * seq_length]
    chunks = torch.chunk(real, num_slices, dim=0)
    assert len(chunks) == num_slices
    return list(chunks[:-1]), chunks[-1]


class _AttnResSourceCache:
    """Per-process (per-rank) depth-source cache for interleaved VPP.

    Keyed by the within-chunk microbatch index (identical across virtual
    chunks for the same data microbatch). ``sources[mb]`` holds the rank's
    full source list as of its latest visit, as detached ``requires_grad``
    LEAF tensors (see :func:`attn_res_tap_source`): later virtual chunks on
    this rank consume the leaves directly, so their backwards accumulate into
    ``leaf.grad`` without ever touching the producing chunk's graph — chunk
    graphs must stay disjoint because the schedule backward runs with
    keep_graph=False. The leaves share storage with activations the per-chunk
    autograd graphs keep alive anyway, so caching extends no lifetimes.
    """

    def __init__(self):
        self.sources: dict = {}

    def reset(self):
        """Clear all state (called at every schedule entry as a safety net)."""
        self.sources.clear()


_SOURCE_CACHE = _AttnResSourceCache()


def get_attn_res_source_cache() -> _AttnResSourceCache:
    """Return the process-wide source cache singleton."""
    return _SOURCE_CACHE


def attn_res_source_cache_reset():
    """Reset the source cache (schedule-entry safety net, mirrors paged_stash_reset)."""
    _SOURCE_CACHE.reset()


class _AttnResGradTap(torch.autograd.Function):
    """Identity wrapper applied where a depth source first materializes in a chunk.

    Pairs the in-graph tensor with its detached cache leaf. Later virtual
    chunks consume the leaf; their backwards (which the interleaved 1F1B
    schedule runs BEFORE this chunk's backward for the same microbatch)
    accumulate into ``leaf.grad``. This backward then drains ``leaf.grad``
    into the in-graph gradient, so every downstream contribution flows to the
    source's origin (an earlier recv leaf slice or a locally formed block sum)
    exactly once. The leaf reference is captured at forward time; nothing is
    keyed off mutable schedule state at backward time.
    """

    @staticmethod
    def forward(ctx, tensor, cache_leaf):
        """Return an identity view while retaining the cache leaf for backward."""
        ctx.cache_leaf = cache_leaf
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad_output):
        """Merge accumulated cache-leaf gradients into the source gradient."""
        leaf = ctx.cache_leaf
        if leaf.grad is not None:
            grad_output = grad_output + leaf.grad
            leaf.grad = None
        return grad_output, None


def attn_res_tap_source(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    """Materialize a depth source for cross-chunk reuse.

    Returns ``(in_graph, cache_leaf)``: ``in_graph`` replaces the source in
    the current chunk's dataflow (aggregations and the outgoing delta payload);
    ``cache_leaf`` goes into the rank-local source cache for later chunks.
    """
    cache_leaf = tensor.detach()
    cache_leaf.requires_grad_(True)
    return _AttnResGradTap.apply(tensor, cache_leaf), cache_leaf


class AttnResStageSources:
    """Depth-source bookkeeping for one stage (or virtual chunk) forward pass.

    Shared by ``TransformerBlock`` and ``HybridStack`` so the two entry/exit
    code paths cannot drift.

    Non-interleaved (V=1): a thin wrapper over the plain source list. The
    payload is the full prefix, no cache or grad tap is involved, and the
    dataflow is identical to the original single-stage-per-rank path.

    Interleaved VPP: reconstructs the full source list from the rank-local
    cache plus the incoming delta payload, taps every newly materialized
    source (:func:`attn_res_tap_source`) for cross-chunk gradient routing,
    updates the cache at exit, and emits delta-packed payloads padded to the
    schedule's uniform width.
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        pp_rank: int,
        vp_stage: Optional[int],
        microbatch_id: Optional[int],
        pre_process: bool,
    ):
        self.config = config
        self.interleaved = config.virtual_pipeline_model_parallel_size is not None
        self.pp_rank = pp_rank
        self.vp_stage = vp_stage or 0
        self.microbatch_id = microbatch_id
        self.pre_process = pre_process
        self.graph_sources: List[Tensor] = []
        self._cache_leaves: List[Tensor] = []
        if self.interleaved:
            assert microbatch_id is not None, (
                "attention residuals under interleaved VPP require the schedule's "
                "current_microbatch to be set on the layers (schedules.forward_step "
                "does this); it was not found"
            )

    @classmethod
    def enter(
        cls,
        config: TransformerConfig,
        hidden_states: Tensor,
        *,
        layers_before: int,
        pp_rank: int,
        vp_stage: Optional[int],
        microbatch_id: Optional[int],
        pre_process: bool,
    ) -> Tuple["AttnResStageSources", Tensor]:
        """Recover (sources, partial) at stage entry; returns (state, hidden_states)."""
        state = cls(
            config,
            pp_rank=pp_rank,
            vp_stage=vp_stage,
            microbatch_id=microbatch_id,
            pre_process=pre_process,
        )
        if pre_process:
            if state.interleaved:
                cache = get_attn_res_source_cache().sources
                assert microbatch_id not in cache, (
                    f"stale attention-residual source cache entry for microbatch "
                    f"{microbatch_id}; a previous iteration did not evict it"
                )
            # The embedding output is the initial partial sum; it becomes depth
            # source b_0 when layer 1 opens the first block.
            return state, hidden_states

        block_layers = config.attn_res_block_layers
        nvtx_range_push(msg="attn_res.unpack_payload")
        if not state.interleaved:
            sources, hidden_states = unpack_attn_res_payload(
                hidden_states, attn_res_num_payload_slices(layers_before, block_layers)
            )
            state.graph_sources = sources
        else:
            num_slices = attn_res_boundary_delta_slices(config, pp_rank, state.vp_stage)
            delta_sources, hidden_states = unpack_attn_res_payload(
                hidden_states, num_slices, padded_slices=attn_res_uniform_payload_slices(config)
            )
            cache = get_attn_res_source_cache().sources
            if state.vp_stage == 0:
                assert microbatch_id not in cache, (
                    f"stale attention-residual source cache entry for microbatch "
                    f"{microbatch_id}; a previous iteration did not evict it"
                )
                cached: List[Tensor] = []
            else:
                assert microbatch_id in cache, (
                    f"missing attention-residual source cache entry for microbatch "
                    f"{microbatch_id} on virtual stage {state.vp_stage}; the previous "
                    "chunk's forward did not store it"
                )
                cached = cache[microbatch_id]
            state.graph_sources = list(cached)
            state._cache_leaves = list(cached)
            for tensor in delta_sources:
                in_graph, leaf = attn_res_tap_source(tensor)
                state.graph_sources.append(in_graph)
                state._cache_leaves.append(leaf)
            expected = _sources_formed_through(layers_before, block_layers)
            assert len(state.graph_sources) == expected, (
                f"reconstructed {len(state.graph_sources)} depth sources but the layer "
                f"layout expects {expected} before this chunk; cache/delta bookkeeping "
                "is broken"
            )
        nvtx_range_pop(msg="attn_res.unpack_payload")
        return state, hidden_states

    def append_block_start(self, hidden_states: Tensor):
        """Record a completed depth block (the partial sum becomes a source)."""
        if self.interleaved:
            in_graph, leaf = attn_res_tap_source(hidden_states)
            self.graph_sources.append(in_graph)
            self._cache_leaves.append(leaf)
        else:
            self.graph_sources.append(hidden_states)

    def _update_cache(self):
        """Store or evict this rank's cache entry after the chunk's exit processing."""
        if not self.interleaved:
            return
        cache = get_attn_res_source_cache().sources
        vp_size = self.config.virtual_pipeline_model_parallel_size
        if self.vp_stage == vp_size - 1:
            # Last visit on this rank: nothing reads the entry afterwards
            # (backward uses autograd-saved references, never the cache).
            cache.pop(self.microbatch_id, None)
        else:
            cache[self.microbatch_id] = self._cache_leaves

    def exit_aggregate_values(self, partial: Tensor) -> List[Tensor]:
        """Values for the final output head: all sources plus the trailing partial."""
        values = [*self.graph_sources, partial]
        self._update_cache()
        return values

    def exit_pack(self, partial: Tensor) -> Tensor:
        """Pack the outgoing pipeline payload (full prefix, or delta+pad under VPP)."""
        if not self.interleaved:
            return pack_attn_res_payload([*self.graph_sources, partial])
        pp_size = self.config.pipeline_model_parallel_size
        recv_pp_rank = (self.pp_rank + 1) % pp_size
        recv_vp_stage = self.vp_stage + (1 if self.pp_rank == pp_size - 1 else 0)
        num_slices = attn_res_boundary_delta_slices(self.config, recv_pp_rank, recv_vp_stage)
        delta_count = num_slices - 1
        assert 0 <= delta_count <= len(self.graph_sources), (
            f"outgoing delta of {delta_count} sources but only "
            f"{len(self.graph_sources)} are available"
        )
        outgoing = self.graph_sources[len(self.graph_sources) - delta_count :]
        payload = pack_attn_res_payload(
            [*outgoing, partial], pad_to_slices=attn_res_uniform_payload_slices(self.config)
        )
        self._update_cache()
        return payload


def _attn_res_fwd_math(pseudo_query, key_norm_weight, eps, values):
    """Pure forward math: stats, fp32 depth softmax, weighted accumulation.

    Kept as a standalone function so it can be wrapped by ``torch.compile``
    (one specialization per source arity) without changing the custom
    autograd Function's saved-tensor policy.
    """
    q = (pseudo_query * key_norm_weight).float()  # [h]

    dots = []
    rstds = []
    for value in values:
        v32 = value.float()
        mean_sq = v32.pow(2).mean(dim=-1)  # [...]
        rstds.append(torch.rsqrt(mean_sq + eps))
        dots.append(torch.matmul(v32, q))
    dots = torch.stack(dots)  # [n, ...]
    rstds = torch.stack(rstds)  # [n, ...]
    alpha = torch.softmax(dots * rstds, dim=0)  # [n, ...] fp32

    out32 = None
    for j, value in enumerate(values):
        term = alpha[j].unsqueeze(-1) * value.float()
        out32 = term if out32 is None else out32 + term
    return out32.to(values[0].dtype), alpha, dots, rstds


def _attn_res_bwd_math(pseudo_query, key_norm_weight, alpha, dots, rstds, grad_output, values):
    """Pure backward math: value path, depth-softmax jacobian, key-RMSNorm path.

    Recomputes the fp32 upcasts and normalized keys from the saved value
    references plus per-token statistics. The dq reduction stays a gemv so the
    parameter gradients are deterministic.
    """
    q = (pseudo_query * key_norm_weight).float()  # [h]
    hidden_size = values[0].shape[-1]
    g32 = grad_output.float()

    # Value path + softmax backward. u_j = <g, V_j> per token.
    u = torch.stack([(g32 * value.float()).sum(dim=-1) for value in values])  # [n, ...]
    dlogits = alpha * (u - (alpha * u).sum(dim=0, keepdim=True))  # [n, ...]
    ddots = dlogits * rstds
    dmean_sq = dlogits * dots * (-0.5) * rstds.pow(3)

    dq = torch.zeros_like(q)
    grad_values = []
    for j, value in enumerate(values):
        v32 = value.float()
        gv = (
            alpha[j].unsqueeze(-1) * g32
            + ddots[j].unsqueeze(-1) * q
            + (dmean_sq[j] * (2.0 / hidden_size)).unsqueeze(-1) * v32
        )
        grad_values.append(gv.to(value.dtype))
        # dq += sum over tokens of ddots_j * V_j (deterministic gemv reduction).
        dq = dq + torch.matmul(v32.reshape(-1, hidden_size).transpose(0, 1), ddots[j].reshape(-1))

    grad_query = (dq * key_norm_weight.float()).to(pseudo_query.dtype)
    grad_norm_weight = (dq * pseudo_query.float()).to(key_norm_weight.dtype)
    return grad_query, grad_norm_weight, grad_values


_COMPILED_MATH: dict = {}
_COMPILE_FAILED = False


def _get_attn_res_math(use_compile: bool):
    """Return (fwd_math, bwd_math), compiled when requested and available.

    torch.compile specializes per source arity (the list length is a dynamo
    guard), so the one-time compile cost is bounded by the number of distinct
    depth arities (~N). Falls back to eager with a one-time warning if
    compilation is unavailable or fails at wrap time; a runtime compile
    failure inside the wrapped function is not caught.
    """
    global _COMPILE_FAILED
    if not use_compile or _COMPILE_FAILED:
        return _attn_res_fwd_math, _attn_res_bwd_math
    if not _COMPILED_MATH:
        try:
            _COMPILED_MATH['fwd'] = torch.compile(_attn_res_fwd_math)
            _COMPILED_MATH['bwd'] = torch.compile(_attn_res_bwd_math)
        except Exception:  # pylint: disable=broad-except
            _COMPILE_FAILED = True
            logging.getLogger(__name__).warning(
                "attn_res_impl='compile' unavailable (torch.compile failed to wrap); "
                "falling back to the eager attention-residual aggregation."
            )
            return _attn_res_fwd_math, _attn_res_bwd_math
    return _COMPILED_MATH['fwd'], _COMPILED_MATH['bwd']


class _AttnResAggregation(torch.autograd.Function):
    """Depth-softmax aggregation with a memory-lean, recomputing backward.

    A naive autograd implementation retains O(n) fp32 [.., h] intermediates
    (upcast copies and normalized keys) per sublayer. This Function saves only
    references to the incoming values plus per-token fp32 statistics
    (dots, inverse RMS, softmax weights; no hidden-size factor) and recomputes
    everything else in backward.

    All statistics, the softmax, and the weighted accumulation run in fp32;
    the output is cast back to the values' dtype so no fp32 ever leaks into
    the residual stream. With ``use_compile`` the forward/backward math bodies
    run as torch.compile-fused kernels (the eager loop is CPU-dispatch-bound:
    ~30 python ops and a dozen small kernels per aggregation), while the
    saved-tensor policy stays exactly the same.
    """

    @staticmethod
    def forward(ctx, pseudo_query, key_norm_weight, eps, use_compile, *values):
        """Aggregate depth sources and save compact state for backward."""
        fwd_math, _ = _get_attn_res_math(use_compile)
        out, alpha, dots, rstds = fwd_math(pseudo_query, key_norm_weight, eps, list(values))
        ctx.use_compile = use_compile
        ctx.save_for_backward(pseudo_query, key_norm_weight, alpha, dots, rstds, *values)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """Recompute aggregation intermediates and return input gradients."""
        nvtx_range_push(msg=f"attn_res.aggregate_bwd_n{len(ctx.saved_tensors) - 5}")
        pseudo_query, key_norm_weight, alpha, dots, rstds, *values = ctx.saved_tensors
        _, bwd_math = _get_attn_res_math(ctx.use_compile)
        grad_query, grad_norm_weight, grad_values = bwd_math(
            pseudo_query, key_norm_weight, alpha, dots, rstds, grad_output, list(values)
        )
        nvtx_range_pop(msg=f"attn_res.aggregate_bwd_n{len(values)}")
        return (grad_query, grad_norm_weight, None, None, *grad_values)


class AttentionResidual(MegatronModule):
    """Per-sublayer AttnRes aggregation: ``h = softmax-attention over depth sources``.

    One learnable zero-initialized pseudo-query and one RMSNorm weight per
    module (per sublayer). Zero init makes the initial attention weights
    uniform, which the paper identifies as required for training stability;
    together with RMSNorm scale invariance this makes the network functionally
    identical to the PreNorm baseline at initialization.

    The math is per-token and per-channel-local, so the module is transparent
    to TP (hidden dim intact under sequence parallelism), CP, and EP. The
    parameters are replicated; under sequence parallelism their gradients are
    all-reduced across the TP group via the ``sequence_parallel`` attribute.
    """

    def __init__(self, config: TransformerConfig, layer_number: Optional[int] = None):
        super().__init__(config)
        self.eps = config.layernorm_epsilon
        self.use_compile = getattr(config, 'attn_res_impl', 'eager') == 'compile'
        # Zero init is mandatory: uniform initial attention weights.
        self.pseudo_query = mark_keep_in_fp32(nn.Parameter(torch.zeros(config.hidden_size)))
        self.key_norm_weight = mark_keep_in_fp32(nn.Parameter(torch.ones(config.hidden_size)))
        if config.sequence_parallel:
            setattr(self.pseudo_query, 'sequence_parallel', True)
            setattr(self.key_norm_weight, 'sequence_parallel', True)

    def forward(self, values: Sequence[Tensor]) -> Tensor:
        """Aggregate depth sources (+ optional partial sum) into the sublayer input."""
        assert len(values) >= 1, "AttentionResidual requires at least one depth source"
        nvtx_range_push(msg=f"attn_res.aggregate_n{len(values)}")
        out = _AttnResAggregation.apply(
            self.pseudo_query, self.key_norm_weight, self.eps, self.use_compile, *values
        )
        nvtx_range_pop(msg=f"attn_res.aggregate_n{len(values)}")
        return out
