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

import logging
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


def is_attn_res_block_start(global_layer_number: int, block_layers: int) -> bool:
    """Whether the attention sublayer of this layer opens a new depth block.

    Block boundaries fall on transformer-layer starts only (``S`` is always an
    even number of sublayers). Layer numbers are 1-based global indices.
    The very first layer is always a block start: it appends the token
    embedding (the initial partial sum) as depth source ``b_0``.
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


def attn_res_payload_slices_for_pp_rank(
    config: TransformerConfig, boundary_recv_pp_rank: int
) -> int:
    """Payload slice count for the boundary received by ``boundary_recv_pp_rank``.

    The number of layers before that boundary equals the receiving stage's
    global layer offset. Uneven splits expressed through
    ``account_for_embedding_in_pipeline_split`` / ``account_for_loss_in_pipeline_split``
    are handled by :func:`get_transformer_layer_offset`.
    """
    # Imported lazily to avoid a circular import with transformer_layer.py.
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    layers_before = get_transformer_layer_offset(
        config, vp_stage=None, pp_rank=boundary_recv_pp_rank
    )
    return attn_res_num_payload_slices(layers_before, config.attn_res_block_layers)


def pack_attn_res_payload(values: Sequence[Tensor]) -> Tensor:
    """Concatenate depth sources + partial sum along the sequence dimension.

    Produces a fresh, viewless tensor, which is required by
    ``deallocate_output_tensor`` on the pipeline-parallel send path.
    """
    return torch.cat(list(values), dim=0)


def unpack_attn_res_payload(payload: Tensor, num_slices: int) -> Tuple[List[Tensor], Tensor]:
    """Split a received payload into (depth sources, partial sum).

    The slices are contiguous views of the received leaf tensor, so gradients
    accumulate into the single tensor handed back to the pipeline schedule.
    """
    assert payload.shape[0] % num_slices == 0, (
        f"attention residual payload sequence dim {payload.shape[0]} is not divisible by "
        f"the expected slice count {num_slices}; pipeline boundary bookkeeping is broken"
    )
    chunks = torch.chunk(payload, num_slices, dim=0)
    assert len(chunks) == num_slices
    return list(chunks[:-1]), chunks[-1]


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
