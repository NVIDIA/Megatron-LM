# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuDNN Frontend backend for GDP chunkwise context parallelism.

FE stores X = S^T in [H, V, K] order. Its forward summaries compose as
X_next = X @ M + E, and its backward summaries as dX_prev = dX @ B + G.
Only the boundary sequence contributes to inter-rank communication.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from cudnn.linear_attention import (
        gated_delta_product,
        gated_delta_product_summary,
        gated_delta_product_summary_bwd,
    )

except ImportError as exc:
    # The caller guards this optional backend; this marker handles direct imports in CI.
    raise ImportError("UnavailableError: cuDNN Frontend GDP CP backend is unavailable") from exc

from megatron.core.ssm.context_parallel.chunkwise import (
    CPBackwardSummary,
    CPBackwardUnpackedSummary,
    CPForwardSummary,
    CPForwardUnpackedSummary,
    CPSavedContext,
)
from megatron.core.ssm.context_parallel.gdp_common import (
    GDPChunkwiseContextParallel,
    GDPInputGradients,
    GDPInputs,
)


@dataclass(frozen=True)
class CuDNNGDPSavedMetadata:
    """Non-tensor fields for the FE forward-to-backward carryover."""

    scale: float
    num_householder: int
    checkpoint_every_n_tokens: int


@dataclass(frozen=True)
class CuDNNGDPBackwardContext:
    """The local output gradient and tensors retained by the shared autograd adapter."""

    saved_context: CPSavedContext
    output_grad: torch.Tensor


class CuDNNGatedDeltaProductCPBackend:
    """Adapt FE summary, forward, and backward operations to Megatron collectives."""

    autograd_function = GDPChunkwiseContextParallel

    def __init__(self, checkpoint_every_n_tokens: int = 192) -> None:
        self.checkpoint_every_n_tokens = checkpoint_every_n_tokens

    def cp_forward_prepare(self, inputs: GDPInputs) -> tuple[CPForwardSummary, GDPInputs]:
        """Summarize FE-native inputs for the trailing sequence."""
        if inputs.cu_seqlens is None:
            raise ValueError("The cuDNN GDP backend requires cu_seqlens")
        state_update, transition = gated_delta_product_summary(
            k=inputs.k,
            v=inputs.v,
            g=inputs.g,
            beta=inputs.beta,
            cu_seqlens=inputs.cu_seqlens,
            num_householder=inputs.num_householder,
            use_qk_l2norm_in_kernel=True,
            plan_name="gdp_summary_frost",
        )
        return (
            CPForwardUnpackedSummary(
                transition=transition[-1].contiguous(), state_update=state_update[-1].contiguous()
            ),
            inputs,
        )

    def cp_forward_apply(
        self, local_context: GDPInputs, preceding_summaries: CPForwardSummary
    ) -> tuple[torch.Tensor, CPSavedContext]:
        """Compose the prefix in FE orientation and seed only the first local sequence."""
        if not isinstance(preceding_summaries, CPForwardUnpackedSummary):
            raise TypeError("The cuDNN GDP backend requires unpacked forward summaries")
        local = local_context
        num_sequences = local.cu_seqlens.numel() - 1
        heads, dim_k = local.q.shape[1:]
        dim_v = local.v.shape[-1]
        initial_state = torch.zeros(
            num_sequences, heads, dim_v, dim_k, dtype=torch.float32, device=local.q.device
        )
        state = initial_state[0]
        for rank in range(preceding_summaries.transition.shape[0]):
            state = (
                state @ preceding_summaries.transition[rank]
                + preceding_summaries.state_update[rank]
            )
        initial_state[0].copy_(state)
        output, _, checkpoints = gated_delta_product(
            q=local.q,
            k=local.k,
            v=local.v,
            g=local.g,
            beta=local.beta,
            cu_seqlens=local.cu_seqlens,
            num_householder=local.num_householder,
            scale=local.scale,
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=True,
            checkpoint_every_n_tokens=self.checkpoint_every_n_tokens,
            plan_name="gdp_frost",
        )
        return output, CPSavedContext(
            tensors=(
                local.q,
                local.k,
                local.v,
                local.g,
                local.beta,
                local.cu_seqlens,
                checkpoints,
                initial_state,
            ),
            metadata=CuDNNGDPSavedMetadata(
                scale=local.scale,
                num_householder=local.num_householder,
                checkpoint_every_n_tokens=self.checkpoint_every_n_tokens,
            ),
        )

    def cp_backward_prepare(
        self, output_grad: torch.Tensor, saved_context: CPSavedContext
    ) -> tuple[CPBackwardSummary, CuDNNGDPBackwardContext]:
        """Communicate only the leading sequence's adjoint summary."""
        q, k, _, g, beta, cu_seqlens, _, _ = saved_context.tensors
        metadata = saved_context.metadata
        output_grad = output_grad.contiguous()
        state_grad, transition = gated_delta_product_summary_bwd(
            q=q,
            k=k,
            g=g,
            beta=beta,
            dO=output_grad,
            cu_seqlens=cu_seqlens,
            num_householder=metadata.num_householder,
            scale=metadata.scale,
            output_transition=True,
            use_qk_l2norm_in_kernel=True,
            plan_name="gdp_summary_frost",
        )
        return CPBackwardUnpackedSummary(
            transition=transition[0].contiguous(), local_state_grad=state_grad[0].contiguous()
        ), CuDNNGDPBackwardContext(saved_context, output_grad)

    def cp_backward_apply(
        self, backward_context: CuDNNGDPBackwardContext, following_summaries: CPBackwardSummary
    ) -> GDPInputGradients:
        """Fold the suffix in reverse rank order, then compute all local input gradients."""
        if not isinstance(following_summaries, CPBackwardUnpackedSummary):
            raise TypeError("The cuDNN GDP backend requires unpacked backward summaries")
        saved = backward_context.saved_context
        q, k, v, g, beta, cu_seqlens, checkpoints, initial_state = saved.tensors
        metadata = saved.metadata
        d_final_state = torch.zeros_like(initial_state)
        state_grad = d_final_state[-1]
        for rank in reversed(range(following_summaries.transition.shape[0])):
            state_grad = (
                state_grad @ following_summaries.transition[rank]
                + following_summaries.local_state_grad[rank]
            )
        d_final_state[-1].copy_(state_grad)
        # The public wrappers above register this explicit-backward custom op.
        dq, dk, dv, dg, dbeta, _, _, _ = torch.ops.cudnn.gated_delta_product_bwd(
            backward_context.output_grad,
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            metadata.num_householder,
            metadata.scale,
            initial_state=initial_state,
            d_final_state=d_final_state,
            state_checkpoints=checkpoints,
            checkpoint_every_n_tokens=metadata.checkpoint_every_n_tokens,
            use_qk_l2norm_in_kernel=True,
            plan_name="gdp_frost",
        )
        return dq, dk, dv, dg, dbeta
