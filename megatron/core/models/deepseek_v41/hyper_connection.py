# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Single-pass hyper-connection wrapper for DeepSeek-V4.1 hybrid layers.

DeepSeek-V4 computes, for every sub-layer, the three mHC mappings (aggregation ``H_pre``,
expansion ``H_post``, residual mixing ``H_res``) from that sub-layer's own input. V4.1 keeps
the parameterisation but shifts ``H_pre`` by one sub-layer: the projection at sub-layer
``k`` produces ``H_post`` and ``H_res`` for ``k`` and ``H_pre`` for ``k + 1``. Sub-layer
``k`` therefore aggregates with the ``H_pre`` its predecessor produced (the first sub-layer
uses the identity mix: stream 0 only), and the output head aggregates with the ``H_pre``
of the last sub-layer. One projection per sub-layer instead of two: "single pass".

Numerics follow the reference: the flattened streams are normalised with
``rsqrt(mean(x^2) + eps)`` using the model RMS epsilon, and the mixing coefficients as well
as the mixing arithmetic stay in fp32 until the result is cast back to the activation dtype.
The wrapper reuses Megatron-Core's ``HyperConnectionModule`` for the parameters, the
projection and the Sinkhorn step, and implements the V4.1 data flow itself. It also hosts
the optional Engram memory applied to the residual streams before some attention layers,
and exposes the per-microbatch shared state to the CSA2 attention while the inner layer runs.

Design reference: ``Block.forward`` / ``hc_mixes`` in the official ``inference/model.py``;
independent implementation.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.deepseek_v41.engram import EngramMemory
from megatron.core.models.deepseek_v41.fused_proj_rms import (
    fused_v41_proj_rms_available,
    fused_v41_projection_and_rms,
)
from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.csa2.attention import CSA2Attention
from megatron.core.transformer.experimental_attention_variant.csa2.state import (
    DSv41SharedState,
    SharedStateSlot,
)
from megatron.core.transformer.hyper_connection import BroadcastTensorFused
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


def v41_projection_and_rms(x: Tensor, weight: Tensor, eps: float) -> Tuple[Tensor, Tensor]:
    """Mixing projection plus the reference RMS factor ``rsqrt(mean(x^2) + eps)``.

    ``x`` is ``[tokens, n*C]`` fp32, ``weight`` is ``[n^2 + 2n, n*C]`` fp32. Returns the raw
    projection and the per-token factor ``[tokens, 1]``.
    """
    proj = torch.matmul(x, weight.t())
    r = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return proj, r


def aggregate_streams_fp32(streams: Tensor, h_pre: Tensor, n: int) -> Tensor:
    """``H_pre``-weighted sum of the ``n`` streams, computed in fp32.

    ``streams``: ``[s, b, n*C]``; ``h_pre``: ``[s, b, n]``. Returns ``[s, b, C]`` in the
    stream dtype.
    """
    s, b, nc = streams.shape
    x = streams.float().view(s, b, n, nc // n)
    out = (x * h_pre.float().unsqueeze(-1)).sum(dim=2)
    return out.to(streams.dtype)


def residual_update_fp32(
    residual: Tensor, branch_output: Tensor, h_post: Tensor, h_res: Tensor, n: int
) -> Tensor:
    """``H_res^T @ residual + H_post (x) branch_output`` in fp32.

    ``residual``: ``[s, b, n*C]``; ``branch_output``: ``[s, b, C]``; ``h_post``: ``[s, b, n]``;
    ``h_res``: ``[s, b, n, n]`` with ``h_res[..., i, j]`` weighting stream ``i`` into stream
    ``j``. Returns ``[s, b, n*C]`` in the residual dtype.
    """
    s, b, nc = residual.shape
    res = residual.float().view(s, b, n, nc // n)
    mixed = torch.einsum("sbij,sbic->sbjc", h_res.float(), res)
    expanded = h_post.float().unsqueeze(-1) * branch_output.float().unsqueeze(2)
    return (mixed + expanded).to(residual.dtype).view(s, b, nc)


def aggregate_streams_fused(streams: Tensor, h_pre: Tensor, n: int) -> Tensor:
    """Same contract as :func:`aggregate_streams_fp32` through the merged fused mHC kernel
    (``fused_h_aggregate``: bf16 streams, fp32 ``H_pre``, fp32 accumulation, one bf16
    rounding; max deviation from the fp32 path is one bf16 ulp)."""
    from megatron.core.fusions.fused_mhc_kernels import fused_h_aggregate

    s, b, nc = streams.shape
    return fused_h_aggregate(streams.view(s, b, n, nc // n), h_pre.float())


def residual_update_fused(
    residual: Tensor, branch_output: Tensor, h_post: Tensor, h_res: Tensor, n: int
) -> Tensor:
    """Same contract as :func:`residual_update_fp32` through the merged fused mHC kernel
    (``fused_h_post_bda`` with fp32 coefficients returns fp32; cast once to the residual
    dtype, as the fp32 path does)."""
    from megatron.core.fusions.fused_mhc_kernels import fused_h_post_bda

    s, b, nc = residual.shape
    out = fused_h_post_bda(
        h_res.float(), residual.view(s, b, n, nc // n), h_post.float(), branch_output, None
    )
    return out.to(residual.dtype).view(s, b, nc)


class SinglePassHyperConnectionHybridLayer(HyperConnectionHybridLayer):
    """mHC wrapper with the V4.1 ``H_pre`` handoff, fp32 mixing, optional Engram, and state
    exposure."""

    # HybridStack / checkpointed_forward hand ``_extra_layer_kwargs()`` to layers that opt in.
    accepts_extra_layer_kwargs = True

    def __init__(
        self,
        config: TransformerConfig,
        layer: MegatronModule,
        engram: Optional[EngramMemory] = None,
        engram_index: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, layer=layer)
        hc = self.hyper_connection
        # Reference normalisation and epsilon (the V4.1 factor is rsqrt(mean(x^2) + eps)).
        hc.norm_eps = config.layernorm_epsilon
        hc._proj_rms_op = v41_projection_and_rms
        hc._proj_rms_compute_h_op = None
        self.n_streams = config.num_residual_streams
        # use_fused_mhc: the merged Triton/cuTile kernels for the stream aggregation and the
        # residual update (10-14x faster than the eager fp32 path at 16K x 4 x 5120, same
        # numerics to one bf16 ulp). The projection + RMS + compute_h kernel implements the
        # V4 normalisation and stays eager here; Sinkhorn already goes through hc._sinkhorn_op.
        # The projection + RMS factor uses the own V4.1 Triton kernels (fused_proj_rms.py:
        # bf16 streams read once, fp32-accurate 3xTF32 projection, rsqrt(mean(x^2) + eps)); the
        # merged fused projection kernel implements the V4 factor and is not reused.
        self._fused_proj_rms = False
        if getattr(config, "use_fused_mhc", False):
            self._aggregate_op = aggregate_streams_fused
            self._residual_update_op = residual_update_fused
            self._fused_proj_rms = fused_v41_proj_rms_available()
        else:
            self._aggregate_op = aggregate_streams_fp32
            self._residual_update_op = residual_update_fp32
        self.engram = engram
        self.engram_index = engram_index
        self.state_slot = SharedStateSlot()
        for module in layer.modules():
            if isinstance(module, CSA2Attention):
                module.shared_state_slot = self.state_slot

    def _identity_h_pre(self, hidden_states: Tensor) -> Tensor:
        s, b, _ = hidden_states.shape
        h_pre = torch.zeros(s, b, self.n_streams, dtype=torch.float32, device=hidden_states.device)
        h_pre[..., 0] = 1.0
        return h_pre

    def compute_mappings_fp32(self, streams: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """``(H_pre_next, H_post, H_res)`` in fp32: sigmoid + eps, 2*sigmoid, Sinkhorn."""
        hc = self.hyper_connection
        s, b, _ = streams.shape
        if self._fused_proj_rms and streams.is_cuda:
            proj, r = fused_v41_projection_and_rms(
                streams.reshape(s * b, -1), hc.mapping_proj.weight, hc.norm_eps
            )
            proj, r = proj.view(s, b, -1), r.view(s, b, 1)
        else:
            proj, r = hc._projection_and_get_norm(streams)
        h_pre, h_post, h_res = hc._compute_h(proj, r)
        h_res = hc._sinkhorn_op(
            h_res.view(s, b, self.n_streams, self.n_streams),
            hc.sinkhorn_iterations,
            hc.sinkhorn_eps,
        )
        return h_pre.float(), h_post.float(), h_res.float()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        sequence_len_offset: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        mhc_recompute_manager=None,
        dsv41_state: Optional[DSv41SharedState] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """One sub-layer update of the n-stream state with the shifted ``H_pre``."""
        if dsv41_state is None:
            raise RuntimeError(
                "SinglePassHyperConnectionHybridLayer requires the dsv41_state keyword; build the "
                "decoder with DSv41HybridStack (hybrid_dsv41_stack_spec)"
            )
        if mhc_recompute_manager is not None:
            raise NotImplementedError(
                "selective mHC recomputation is not supported by the DeepSeek-V4.1 wrapper yet"
            )

        if self.engram is not None:
            if dsv41_state.engram_hash_ids is None:
                raise RuntimeError("Engram layer reached without hash ids in the shared state")
            row_ids = dsv41_state.engram_hash_ids[:, :, self.engram_index]
            hidden_states = self.engram(hidden_states, row_ids)

        hs_for_mappings, hs_for_aggregate, hs_for_residual = BroadcastTensorFused.apply(
            hidden_states, self.hyper_connection._fused_add_3_op
        )
        h_pre_next, h_post, h_res = self.compute_mappings_fp32(hs_for_mappings)

        h_pre_in = dsv41_state.get_h_pre_for(self.layer_number)
        if h_pre_in is None:
            h_pre_in = self._identity_h_pre(hs_for_aggregate)
        aggregated = self._aggregate_op(hs_for_aggregate, h_pre_in, self.n_streams)
        dsv41_state.publish_h_pre(self.layer_number, h_pre_next)

        self.state_slot.state = dsv41_state
        try:
            fast_path = self._call_inner_transformer_layer_without_local_bda(
                aggregated,
                attention_mask,
                inference_context,
                rotary_pos_emb,
                sequence_len_offset,
                packed_seq_params,
                padding_mask,
                input_ids,
                mhc_recompute_manager=None,
            )
            if fast_path is None:
                layer_output, context = self._call_inner_layer(
                    aggregated,
                    attention_mask,
                    inference_context,
                    rotary_pos_emb,
                    sequence_len_offset,
                    packed_seq_params,
                    padding_mask,
                    input_ids,
                )
                if aggregated.dtype != layer_output.dtype:
                    aggregated = aggregated.to(layer_output.dtype)
                layer_output_with_bias = (layer_output - aggregated, None)
                dropout_prob = 0.0
            else:
                layer_output_with_bias, context, dropout_prob, _ = fast_path
        finally:
            self.state_slot.state = None

        branch_output, bias = layer_output_with_bias
        if bias is not None:
            branch_output = branch_output + bias
        if branch_output.shape != aggregated.shape:
            raise RuntimeError(
                "wrapped branch must preserve the hidden-state shape, got "
                f"{tuple(branch_output.shape)} vs {tuple(aggregated.shape)}"
            )
        if dropout_prob and self.training:
            raise NotImplementedError(
                "DeepSeek-V4.1 hyper-connections require hidden_dropout=0 (validated in config)"
            )

        hidden_states = self._residual_update_op(
            hs_for_residual, branch_output, h_post, h_res, self.n_streams
        )
        return hidden_states, context
