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
    # Register the GDP custom ops before accessing torch.ops.cudnn.
    import cudnn.linear_attention.ops.gdp  # noqa: F401 # pylint: disable=unused-import

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
class CuDNNGDPForwardContext:
    """Inputs and summary intermediates retained until the forward operation."""

    inputs: GDPInputs
    intermediates: torch.Tensor


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
    intermediates: torch.Tensor


class CuDNNGatedDeltaProductCPBackend:
    """Adapt FE summary, forward, and backward operations to Megatron collectives."""

    autograd_function = GDPChunkwiseContextParallel

    def __init__(self, checkpoint_every_n_tokens: int = 192) -> None:
        self.checkpoint_every_n_tokens = checkpoint_every_n_tokens

    def cp_forward_prepare(
        self, inputs: GDPInputs
    ) -> tuple[CPForwardSummary, CuDNNGDPForwardContext]:
        """Summarize the trailing sequence and retain intermediates for forward."""
        if inputs.cu_seqlens is None:
            raise ValueError("The cuDNN GDP backend requires cu_seqlens")
        state_update, transition, intermediates = torch.ops.cudnn.gated_delta_product_summary(
            k=inputs.k,
            v=inputs.v,
            g=inputs.g,
            beta=inputs.beta,
            cu_seqlens=inputs.cu_seqlens[-2:],
            num_householder=inputs.num_householder,
            output_transition=True,
            output_context=True,
            use_qk_l2norm_in_kernel=True,
            plan_name="gdp_summary_frost",
        )
        return (
            CPForwardUnpackedSummary(transition=transition[0], state_update=state_update[0]),
            CuDNNGDPForwardContext(inputs, intermediates),
        )

    def cp_forward_apply(
        self, local_context: CuDNNGDPForwardContext, preceding_summaries: CPForwardSummary
    ) -> tuple[torch.Tensor, CPSavedContext]:
        """Pass the gathered prefix to FE to seed the first local sequence."""
        if not isinstance(preceding_summaries, CPForwardUnpackedSummary):
            raise TypeError("The cuDNN GDP backend requires unpacked forward summaries")
        local = local_context.inputs
        output, _, checkpoints, composed_initial_state = torch.ops.cudnn.gated_delta_product_fwd(
            q=local.q,
            k=local.k,
            v=local.v,
            g=local.g,
            beta=local.beta,
            cu_seqlens=local.cu_seqlens,
            num_householder=local.num_householder,
            scale=local.scale,
            preceding_state_update=preceding_summaries.state_update,
            preceding_transition=preceding_summaries.transition,
            context=local_context.intermediates,
            use_qk_l2norm_in_kernel=True,
            checkpoint_every_n_tokens=self.checkpoint_every_n_tokens,
            checkpoint_dtype=torch.float32,
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
                composed_initial_state,
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
        state_grad, transition, intermediates = torch.ops.cudnn.gated_delta_product_summary_bwd(
            q=q,
            k=k,
            g=g,
            beta=beta,
            dO=output_grad,
            cu_seqlens=cu_seqlens[:2],
            num_householder=metadata.num_householder,
            scale=metadata.scale,
            output_transition=True,
            output_context=True,
            checkpoint_every_n_tokens=metadata.checkpoint_every_n_tokens,
            use_qk_l2norm_in_kernel=True,
            plan_name="gdp_summary_frost",
        )
        return CPBackwardUnpackedSummary(
            transition=transition[0], local_state_grad=state_grad[0]
        ), CuDNNGDPBackwardContext(saved_context, output_grad, intermediates)

    def cp_backward_apply(
        self, backward_context: CuDNNGDPBackwardContext, following_summaries: CPBackwardSummary
    ) -> GDPInputGradients:
        """Pass the gathered suffix and saved prefix state to FE's backward operation."""
        if not isinstance(following_summaries, CPBackwardUnpackedSummary):
            raise TypeError("The cuDNN GDP backend requires unpacked backward summaries")
        saved = backward_context.saved_context
        q, k, v, g, beta, cu_seqlens, checkpoints, composed_initial_state = saved.tensors
        metadata = saved.metadata
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
            composed_initial_state=composed_initial_state,
            following_state_grad=following_summaries.local_state_grad,
            following_transition=following_summaries.transition,
            context=backward_context.intermediates,
            state_checkpoints=checkpoints,
            checkpoint_every_n_tokens=metadata.checkpoint_every_n_tokens,
            use_qk_l2norm_in_kernel=True,
            plan_name="gdp_frost",
        )
        return dq, dk, dv, dg, dbeta
