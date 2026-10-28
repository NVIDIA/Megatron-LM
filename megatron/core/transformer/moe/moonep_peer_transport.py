# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Peer transport policy for the accepted upstream MoonEP runtime."""


class MoonEPPeerTransport:
    """Submit upstream kernels on the stream selected by the execution runtime."""

    @staticmethod
    def push(storage, *, sources, scale_sources, experts_to_copy):
        """Copy the selected weights and scales to runtime slots."""
        from megatron.core.transformer.moe.ops.virtual_expert_triton import (
            launch_virtual_expert_weight_prefetch,
        )

        launch_virtual_expert_weight_prefetch(
            storage, sources=sources, scale_sources=scale_sources, experts_to_copy=experts_to_copy
        )

    @staticmethod
    def reduce(storage, *, native_grads, experts_to_copy, fc_layers):
        """Accumulate replica gradients into their owners using upstream ordering."""
        from megatron.core.transformer.moe.ops.virtual_expert_triton import (
            launch_virtual_expert_grad_reduce,
        )

        launch_virtual_expert_grad_reduce(
            storage, native_grads=native_grads, experts_to_copy=experts_to_copy, fc_layers=fc_layers
        )
