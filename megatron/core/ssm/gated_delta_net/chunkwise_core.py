# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GDN recurrence orchestration using TE local kernels and MCore CP summaries."""

from dataclasses import dataclass

import torch

from megatron.core.ssm.context_parallel.chunkwise import (
    CPBackwardUnpackedSummary,
    CPForwardUnpackedSummary,
    CPSavedContext,
    chunkwise_cp_backward,
    chunkwise_cp_forward,
)


@dataclass
class _Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    cu: torch.Tensor
    scale: float


class _Backend:
    """Adapt TE's value-by-key states to MCore's summary communication protocol."""

    def __init__(self, kernels, per_document=False):
        self.kernels = kernels
        self.per_document = per_document

    def cp_forward_prepare(self, inputs):
        update, transition = self.kernels.forward_summary(
            inputs.k, inputs.v, inputs.g, inputs.beta, inputs.cu
        )
        summary = CPForwardUnpackedSummary(
            (transition if self.per_document else transition[-1]).contiguous(),
            (update if self.per_document else update[-1]).contiguous(),
        )
        return summary, inputs

    def cp_forward_apply(self, local_context, preceding_summaries):
        x = local_context
        states = torch.zeros(
            x.cu.numel() - 1,
            x.v.shape[1],
            x.v.shape[2],
            x.k.shape[2],
            device=x.q.device,
            dtype=torch.float32,
        )
        carry = torch.zeros_like(states if self.per_document else states[0])
        # cuDNN stores value-by-key matrices, hence right multiplication.
        for transition, update in zip(
            preceding_summaries.transition, preceding_summaries.state_update
        ):
            carry = carry @ transition + update
        (states if self.per_document else states[0]).copy_(carry)
        output = self.kernels.forward(
            x.q, x.k, x.v, x.g, x.beta, x.cu, scale=x.scale, initial_state=states
        )
        return output, CPSavedContext(
            tensors=(x.q, x.k, x.v, x.g, x.beta, x.cu, states), metadata=x.scale
        )

    def cp_backward_prepare(self, output_grad, saved_context):
        q, k, _, g, beta, cu, _ = saved_context.tensors
        gamma, transition = self.kernels.backward_summary(
            q, k, g, beta, output_grad, cu, scale=saved_context.metadata
        )
        summary = CPBackwardUnpackedSummary(
            (transition if self.per_document else transition[0]).contiguous(),
            (gamma if self.per_document else gamma[0]).contiguous(),
        )
        return summary, (output_grad, saved_context)

    def cp_backward_apply(self, backward_context, following_summaries):
        output_grad, saved = backward_context
        q, k, v, g, beta, cu, states = saved.tensors
        final_grad = torch.zeros_like(states)
        carry = torch.zeros_like(states if self.per_document else states[0])
        for index in range(following_summaries.transition.shape[0] - 1, -1, -1):
            carry = (
                carry @ following_summaries.transition[index]
                + following_summaries.local_state_grad[index]
            )
        (final_grad if self.per_document else final_grad[-1]).copy_(carry)
        return self.kernels.backward(
            output_grad,
            q,
            k,
            v,
            g,
            beta,
            cu,
            scale=saved.metadata,
            initial_state=states,
            final_state_grad=final_grad,
        )


class _ChunkwiseGDN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu, kernels, group, preceding, following, per_document):
        rank = group.rank()
        result = chunkwise_cp_forward(
            _Backend(kernels, per_document),
            _Inputs(q, k, v, g, beta, cu, q.shape[-1] ** -0.5),
            group,
            slice(preceding, rank),
        )
        ctx.save_for_backward(*result.saved_context.tensors)
        ctx.scale = result.saved_context.metadata
        ctx.kernels = kernels
        ctx.group = group
        ctx.per_document = per_document
        ctx.following = slice(rank + 1, following)
        return result.output

    @staticmethod
    def backward(ctx, output_grad):
        grads = chunkwise_cp_backward(
            _Backend(ctx.kernels, ctx.per_document),
            output_grad,
            CPSavedContext(tensors=ctx.saved_tensors, metadata=ctx.scale),
            ctx.group,
            ctx.following,
        )
        return (*grads, None, None, None, None, None, None)


def chunkwise_gdn(q, k, v, g, beta, metadata, group, kernels, per_document=False):
    """Run expanded-head THD GDN with zero state at each global document start."""
    return _ChunkwiseGDN.apply(
        q,
        k,
        v,
        g,
        beta,
        metadata.local_cu_seqlens,
        kernels,
        group,
        metadata.preceding_rank_start,
        metadata.following_rank_stop,
        per_document,
    )
