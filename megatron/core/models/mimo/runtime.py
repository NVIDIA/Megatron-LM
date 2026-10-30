# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explicit encoder-graph lifetime for colocated packed multimodal training."""

from typing import Callable, Hashable, Mapping, Sequence

import torch
from torch.distributed import ProcessGroup

from megatron.core.models.mimo.comm.pack_bridge import (
    FeatureSlice,
    PackFeatureBridge,
    PackFeaturePlan,
    PackFeatureTransfer,
)
from megatron.core.models.mimo.planning import plan_encoder_chunks


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


class LazyMimoStepRuntime(MimoStepRuntime):
    """Execute independent encoder chunks only while their consumers need them.

    The global packing plan remains unchanged. A caller supplies a local encoder
    callback which prepares only the requested media and returns their projected
    rows with an independent graph. Parameter gradients accumulate normally;
    gradient synchronization, normalization and optimizer updates stay with the
    host. A shared media graph stays live through its last consuming round.
    """

    def _clear(self) -> None:
        super()._clear()
        self.chunks = self.round_plans = ()
        self.live_chunks = {}
        self.encode = self.empty_template = None
        self._round_chunks = ()

    def begin_lazy_step(
        self,
        feature_plans: Sequence[Sequence[PackFeaturePlan]],
        feature_slices: Mapping[Hashable, FeatureSlice],
        encode: Callable[[Sequence[Hashable]], torch.Tensor],
        empty_template: torch.Tensor,
        *,
        forward_only: bool,
    ) -> None:
        """Plan chunk lifetimes without loading media or invoking the encoder.

        ``empty_template`` supplies the common device, dtype and hidden width;
        it must have shape ``[0, hidden]``. Empty producers still route every round.
        """
        if empty_template.ndim != 2 or empty_template.shape[0] != 0:
            raise ValueError("empty_template must have shape [0, hidden]")
        chunks = plan_encoder_chunks(feature_plans, feature_slices)
        if any(chunk.producer_rank not in self.bridge.ranks for chunk in chunks):
            raise ValueError("Encoder chunk producer is outside the transport group")
        self.begin_step(empty_template, len(feature_plans), forward_only=forward_only)
        self.round_plans = tuple(tuple(plans) for plans in feature_plans)
        self.chunks, self.encode, self.empty_template = chunks, encode, empty_template

    def route(
        self, plans: Sequence[PackFeaturePlan], cp_group: ProcessGroup
    ) -> PackFeatureTransfer:
        """Encode first-use chunks and rebase this round's source row offsets."""
        if not self.active or self.pending is not None or self.completed_rounds == self.num_rounds:
            raise RuntimeError("Routing requires an active step with no pending or exhausted round")
        index = self.completed_rounds
        if tuple(plans) != self.round_plans[index]:
            raise ValueError("Feature plans must follow the source step's saved round order")
        active = [chunk for chunk in self.chunks if chunk.first_round <= index <= chunk.last_round]
        self._round_chunks = tuple(
            chunk for chunk in active if chunk.producer_rank == self.bridge.rank
        )
        for chunk in self._round_chunks:
            if chunk.first_round == index:
                with torch.set_grad_enabled(not self.forward_only):
                    features = self.encode(chunk.media_ids)
                if (
                    features.shape != (chunk.num_rows, self.empty_template.shape[1])
                    or features.device != self.empty_template.device
                    or features.dtype != self.empty_template.dtype
                ):
                    raise ValueError(
                        "Encoder chunk output must match its planned rows and template"
                    )
                gradient = (
                    None if self.forward_only else torch.zeros_like(features, dtype=torch.float32)
                )
                self.live_chunks[chunk] = (features, gradient)

        offsets, rebased = {}, {}
        for chunk in active:
            offset = offsets.get(chunk.producer_rank, 0)
            for feature in chunk.features:
                rebased[feature] = FeatureSlice(chunk.producer_rank, offset, feature.length)
                offset += feature.length
            offsets[chunk.producer_rank] = offset
        buffers = [self.live_chunks[chunk][0].detach() for chunk in self._round_chunks]
        self.local_features = (
            torch.cat(buffers)
            if len(buffers) > 1
            else buffers[0] if buffers else self.empty_template
        )
        return super().route(
            tuple(
                PackFeaturePlan(plan.cp_ranks, tuple(rebased[f] for f in plan.features))
                for plan in plans
            ),
            cp_group,
        )

    def complete_round(self, transfer: PackFeatureTransfer) -> None:
        """Return visual gradients and immediately backward completed chunks."""
        if not self.active or transfer is not self.pending or transfer is None:
            raise RuntimeError(
                "Expected the current pending transfer, not a stale or completed one"
            )
        if not self.forward_only:
            gradient = self.bridge.backward(transfer)
            offset = 0
            for chunk in self._round_chunks:
                self.live_chunks[chunk][1].add_(gradient.narrow(0, offset, chunk.num_rows))
                offset += chunk.num_rows
        for chunk in self._round_chunks:
            if chunk.last_round == self.completed_rounds:
                features, gradient = self.live_chunks.pop(chunk)
                if not self.forward_only and features.requires_grad:
                    features.backward(gradient.to(features.dtype))
        self.pending = None
        self.local_features = self.empty_template
        self._round_chunks = ()
        self.completed_rounds += 1

    def finish_backward(self) -> None:
        """Finish after all chunk graphs have already been backpropagated."""
        self._check_finished(forward_only=False)
        if self.live_chunks:
            raise RuntimeError("Encoder chunks remain live after all decoder rounds")
        self._clear()
