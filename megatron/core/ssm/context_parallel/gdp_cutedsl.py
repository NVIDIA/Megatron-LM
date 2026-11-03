# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CuTeDSL backend for Gated Delta Product chunkwise context parallelism.

This backend is deterministic, but not batch invariant.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import torch

try:
    from gdp_attn import cp_backward_apply as cutedsl_cp_backward_apply
    from gdp_attn import cp_backward_prepare as cutedsl_cp_backward_prepare
    from gdp_attn import cp_forward_apply as cutedsl_cp_forward_apply
    from gdp_attn import cp_forward_prepare as cutedsl_cp_forward_prepare
    from gdp_attn.chunk_gated_delta_product import (
        GdpCpBackwardContext,
        GdpCpForwardLocalContext,
        GdpCpSavedContext,
    )
except ImportError as exc:
    # The caller guards this optional backend; this marker handles direct imports in CI.
    raise ImportError("UnavailableError: CuTeDSL GDP chunkwise CP backend is unavailable") from exc

from megatron.core.ssm.context_parallel.chunkwise import (
    CPBackwardPackedSummary,
    CPBackwardSummary,
    CPForwardPackedSummary,
    CPForwardSummary,
    CPSavedContext,
)
from megatron.core.ssm.context_parallel.gdp_common import (
    GDPChunkwiseContextParallel,
    GDPInputGradients,
    GDPInputs,
)


@dataclass(frozen=True)
class CuTeDSLGDPSavedMetadata:
    """Non-tensor fields needed to reconstruct the CuTeDSL backend's saved CP context."""

    scale: float
    num_householder: int
    use_qk_l2norm_in_kernel: bool
    recompute_chunk_num: int
    has_initial_states: bool


class CuTeDSLGatedDeltaProductCPBackend:
    """Adapt ``gdp_attn``'s four-function CP protocol to Megatron collectives."""

    autograd_function = GDPChunkwiseContextParallel

    def __init__(self, recompute_chunk_num: int = 0) -> None:
        self.recompute_chunk_num = recompute_chunk_num
        self._compile_pending = True

    def cp_forward_prepare(
        self, inputs: GDPInputs
    ) -> tuple[CPForwardSummary, GdpCpForwardLocalContext]:
        """Compute the trailing-boundary ``[E | M]`` summary for this rank."""
        if inputs.cu_seqlens is None:
            raise ValueError("The CuTeDSL GDP backend requires cu_seqlens")
        if self._compile_pending:
            # Cleared first, because _compile_boundary_variants calls this method.
            self._compile_pending = False
            self._compile_boundary_variants(inputs)
        packed_summary, local_context = cutedsl_cp_forward_prepare(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g,
            inputs.beta,
            inputs.cu_seqlens,
            inputs.scale,
            inputs.num_householder,
            use_qk_l2norm_in_kernel=True,
            recompute_chunk_num=self.recompute_chunk_num,
        )
        return CPForwardPackedSummary(packed=packed_summary), local_context

    def cp_forward_apply(
        self, local_context: GdpCpForwardLocalContext, preceding_summaries: CPForwardSummary
    ) -> tuple[torch.Tensor, CPSavedContext]:
        """Apply the causal prefix and retain the CuTeDSL backward checkpoint state."""
        packed_prefix = _forward_packed_tensor(preceding_summaries)
        output, saved_context = cutedsl_cp_forward_apply(
            local_context, packed_prefix, output_final_state=False
        )
        saved_tensors = [
            saved_context.q,
            saved_context.k,
            saved_context.v,
            saved_context.g,
            saved_context.beta,
            saved_context.cu_seqlens,
            saved_context.state,
        ]
        has_initial_states = saved_context.initial_states is not None
        if has_initial_states:
            saved_tensors.append(saved_context.initial_states)
        return output, CPSavedContext(
            tensors=tuple(saved_tensors),
            metadata=CuTeDSLGDPSavedMetadata(
                scale=saved_context.scale,
                num_householder=saved_context.num_householder,
                use_qk_l2norm_in_kernel=saved_context.use_qk_l2norm_in_kernel,
                recompute_chunk_num=saved_context.recompute_chunk_num,
                has_initial_states=has_initial_states,
            ),
        )

    def cp_backward_prepare(
        self, output_grad: torch.Tensor, saved_context: CPSavedContext
    ) -> tuple[CPBackwardSummary, GdpCpBackwardContext]:
        """Compute the leading-boundary ``[gamma | M^T]`` summary for this rank."""
        backend_saved_context = _restore_saved_context(saved_context)
        # boundary_len only tightens the launch grid; the backend's conservative None mode
        # preserves correctness without extracting a host scalar here.
        packed_summary, backward_context = cutedsl_cp_backward_prepare(
            output_grad, backend_saved_context, boundary_len=None
        )
        return CPBackwardPackedSummary(packed=packed_summary), backward_context

    def cp_backward_apply(
        self, backward_context: GdpCpBackwardContext, following_summaries: CPBackwardSummary
    ) -> GDPInputGradients:
        """Apply the reverse-causal suffix and compute all local input gradients."""
        return cutedsl_cp_backward_apply(
            backward_context, _backward_packed_tensor(following_summaries), dht=None
        )

    def _compile_boundary_variants(self, inputs: GDPInputs) -> None:
        """Compile the kernels of every CP boundary case on a dummy shard.

        gdp_attn compiles separate kernels for a shard with and without a preceding or following
        summary. Without this, a rank compiles a case in a later step, when a packed batch first
        puts a sequence boundary on its shard edge, while its CP group waits for it.
        """
        num_tokens = 256
        # Same dtypes, widths and row pitches as the real faces: gdp_attn keys kernels on them.
        q, k, v, g, beta = (
            torch.empty_strided(
                (num_tokens, face.shape[1]), face.stride(), dtype=face.dtype, device=face.device
            ).zero_()
            for face in (inputs.q, inputs.k, inputs.v, inputs.g, inputs.beta)
        )
        cu_seqlens = inputs.cu_seqlens.new_tensor([0, num_tokens])
        dummy_inputs = replace(inputs, q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens)
        # Zero or one preceding (following) summaries, sliced like the gathered ones in training.
        for num_preceding, num_following in itertools.product((0, 1), repeat=2):
            forward_summary, local_context = self.cp_forward_prepare(dummy_inputs)
            preceding = _forward_packed_tensor(forward_summary).unsqueeze(0)[:num_preceding]
            output, saved_context = self.cp_forward_apply(
                local_context, CPForwardPackedSummary(packed=preceding)
            )
            backward_summary, backward_context = self.cp_backward_prepare(
                torch.zeros_like(output), saved_context
            )
            following = _backward_packed_tensor(backward_summary).unsqueeze(0)[:num_following]
            self.cp_backward_apply(backward_context, CPBackwardPackedSummary(packed=following))


def _forward_packed_tensor(summary: CPForwardSummary) -> torch.Tensor:
    if not isinstance(summary, CPForwardPackedSummary):
        raise TypeError("The CuTeDSL GDP backend requires packed forward summaries")
    return summary.packed


def _backward_packed_tensor(summary: CPBackwardSummary) -> torch.Tensor:
    if not isinstance(summary, CPBackwardPackedSummary):
        raise TypeError("The CuTeDSL GDP backend requires packed backward summaries")
    return summary.packed


def _restore_saved_context(saved_context: CPSavedContext) -> GdpCpSavedContext:
    metadata = saved_context.metadata
    if not isinstance(metadata, CuTeDSLGDPSavedMetadata):
        raise TypeError("CuTeDSLGatedDeltaProductCPBackend received incompatible metadata")
    expected_count = 8 if metadata.has_initial_states else 7
    if len(saved_context.tensors) != expected_count:
        raise ValueError("CuTeDSLGatedDeltaProductCPBackend received an incompatible tensor bundle")

    q, k, v, g, beta, cu_seqlens, state = saved_context.tensors[:7]
    initial_states = saved_context.tensors[7] if metadata.has_initial_states else None
    return GdpCpSavedContext(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        state=state,
        initial_states=initial_states,
        scale=metadata.scale,
        num_householder=metadata.num_householder,
        use_qk_l2norm_in_kernel=metadata.use_qk_l2norm_in_kernel,
        recompute_chunk_num=metadata.recompute_chunk_num,
    )
