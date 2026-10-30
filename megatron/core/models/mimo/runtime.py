# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explicit encoder-graph lifetime for colocated packed multimodal training."""

from typing import Sequence

import torch
from torch.distributed import ProcessGroup

from megatron.core.models.mimo.comm.pack_bridge import (
    PackFeatureBridge,
    PackFeaturePlan,
    PackFeatureTransfer,
)


class MimoStepRuntime:
    """Own one modality's source graph and feature transfers for a source step.

    All transport ranks call the same rounds, including empty producers and
    text-only consumers. The host runs the encoder and decoder, explicitly
    completes each round after decoder backward, then calls ``finish_backward``
    before its normal gradient finalization. This class does not install hooks,
    normalize gradients, run an optimizer, or infer completion from leaf grads.

    Only one transfer may be pending. Checkpointing is supported at an idle
    source-step boundary; encoder graphs and pending transfers are not serialized.
    """

    def __init__(self, transport_group: ProcessGroup) -> None:
        self.bridge = PackFeatureBridge(transport_group)
        self._clear()

    def _clear(self) -> None:
        self.local_features = None
        self.feature_gradient = None
        self.pending = None
        self.num_rounds = self.completed_rounds = 0
        self.forward_only = False

    @property
    def active(self) -> bool:
        """Whether an encoder graph or evaluation step is still owned."""
        return self.local_features is not None

    def begin_step(
        self, local_features: torch.Tensor, num_rounds: int, *, forward_only: bool
    ) -> None:
        """Save projected encoder rows without detaching their original graph.

        Empty producers pass a floating ``[0, hidden]`` tensor on the same device
        and in the same dtype as other producers. Execution mode is explicit and
        must agree across the transport domain.
        """
        if self.active:
            raise RuntimeError("Finish the active MIMO source step before beginning another")
        if local_features.ndim != 2 or not local_features.is_floating_point():
            raise ValueError("local_features must be a floating-point [rows, hidden] tensor")
        if num_rounds < 1:
            raise ValueError("A source step must contain at least one decoder round")
        self.local_features = local_features
        self.num_rounds = num_rounds
        self.forward_only = forward_only
        if not forward_only:
            self.feature_gradient = torch.zeros_like(local_features, dtype=torch.float32)

    def route(
        self, plans: Sequence[PackFeaturePlan], cp_group: ProcessGroup
    ) -> PackFeatureTransfer:
        """Route this round's features; never implicitly complete an earlier round."""
        if not self.active:
            raise RuntimeError("begin_step must precede feature routing")
        if self.pending is not None:
            raise RuntimeError("Complete the pending feature transfer before routing another")
        if self.completed_rounds == self.num_rounds:
            raise RuntimeError("All planned decoder rounds have already completed")
        self.pending = self.bridge.forward(
            self.local_features, plans, cp_group, forward_only=self.forward_only
        )
        return self.pending

    def complete_round(self, transfer: PackFeatureTransfer) -> None:
        """After decoder backward, return and accumulate this transfer's gradients.

        A missing leaf gradient contributes zeros, including an unused modality.
        It does not indicate whether the host has executed backward. Evaluation
        releases the transfer without running backward communication.
        """
        if not self.active or transfer is not self.pending or transfer is None:
            raise RuntimeError(
                "Expected the current pending transfer, not a stale or completed one"
            )
        if not self.forward_only:
            self.feature_gradient.add_(self.bridge.backward(transfer))
        self.pending = None
        self.completed_rounds += 1

    def _check_finished(self, *, forward_only: bool) -> None:
        if not self.active or self.forward_only != forward_only:
            raise RuntimeError("No active MIMO step in the requested execution mode")
        if self.pending is not None or self.completed_rounds != self.num_rounds:
            raise RuntimeError("Explicitly complete every decoder round before finishing the step")

    def finish_backward(self) -> None:
        """Backpropagate through the encoder once, after every decoder contribution."""
        self._check_finished(forward_only=False)
        if self.local_features.requires_grad:
            self.local_features.backward(self.feature_gradient.to(self.local_features.dtype))
        self._clear()

    def finish_forward(self) -> None:
        """Release an evaluation step after all rounds, without a loss-closure hook."""
        self._check_finished(forward_only=True)
        self._clear()
