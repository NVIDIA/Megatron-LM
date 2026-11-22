# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Full activation recompute for the EP A2A overlap schedule (``model_chunk_schedule_plan``).

Moved from ``megatron.core.models.common.model_chunk_schedule_plan``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from torch import Tensor

from megatron.core.utils import nvtx_range_pop, nvtx_range_push

if TYPE_CHECKING:
    from megatron.core.models.common.model_chunk_schedule_plan import (
        ModelChunkState,
        TransformerLayerSchedulePlan,
    )


# ModelChunkState fields the layer forwards mutate in place; each segment snapshots them
# so its backward-time replay sees the pre-segment values instead of end-of-chunk ones.
#
# ``mtp_input_mask`` belongs here for the same reason as ``input_ids``: MTP's
# ``_get_embeddings`` rolls the two together (one concatenated roll, so CP does a single
# boundary exchange) and writes both back to the chunk state. Leaving it out would let a
# replay roll an already-rolled mask, shifting it one position against ``input_ids``, and
# the mask drives ``torch.where(valid, decoder_input, decoder_input.detach())`` — so the
# embedding rows detached would be the wrong ones and the MTP/embedding gradients would be
# silently wrong rather than failing loudly.
_MUTABLE_CHUNK_STATE_FIELDS = (
    "input_ids",
    "position_ids",
    "padding_mask",
    "mtp_input_mask",
    "mtp_hidden_states",
)


def _copy_chunk_state_value(value: Any) -> Any:
    """Copy list-valued chunk state so a later in-place append cannot corrupt a snapshot."""
    return list(value) if isinstance(value, list) else value


class RecomputeSegment:
    """A contiguous group of layers recomputed as one unit.

    The initial forward runs under ``no_grad`` and is replayed with grad enabled at the
    start of the segment's own backward. A segment drops its layers' activations as soon
    as its own forward is done (``release_activations``) and the replayed ones as each
    node backwards (``ScheduleNode._release_state``), so only one segment's activations
    are live at a time and what crosses the forward->backward gap is the segment's input
    tensor plus the ``_MUTABLE_CHUNK_STATE_FIELDS`` snapshot. Layers left eager under
    ``block`` keep their activations throughout. The A2A issued by a replay is exposed by
    design; normal fwd/bwd A2A overlap is preserved.

    ``start_index`` is the segment's first layer index within the chunk, for NVTX labels.
    """

    def __init__(
        self,
        layers: List["TransformerLayerSchedulePlan"],
        chunk_state: ModelChunkState,
        start_index: int,
    ) -> None:
        assert len(layers) > 0, "a recompute segment must contain at least one layer"
        self.layers = layers
        self.chunk_state = chunk_state
        self.start_index = start_index
        # Captured at the start of the segment's initial forward.
        self.input_tensor = None
        self.state_snapshot = None
        self.rng_states = None
        for layer in layers:
            layer.recompute_segment = self
            # The initial forward retains no autograd graph for these layers.
            layer.set_forward_no_grad(True)

    def capture(self, layer: "TransformerLayerSchedulePlan", f_input: Tensor) -> None:
        """When the head layer starts its forward, snapshot what the replay needs:
        input tensor, mutable chunk state, RNG."""
        if layer is not self.layers[0]:
            return
        from megatron.core.tensor_parallel.random import _get_all_rng_states

        self.input_tensor = f_input
        cs = self.chunk_state
        self.state_snapshot = {
            name: _copy_chunk_state_value(getattr(cs, name)) for name in _MUTABLE_CHUNK_STATE_FIELDS
        }
        self.rng_states = _get_all_rng_states()

    def release_activations(self, layer: "TransformerLayerSchedulePlan") -> None:
        """Drop this segment's forward activations once its tail layer has forwarded.

        Runs while the chunk forward is still going, not at the end of it: the nodes hold
        their boundary tensors in ``inputs`` / ``output`` / ``detached`` until something
        unsets them, so deferring this to the end of the chunk would keep one such set
        alive per recomputed layer for the rest of the forward. The tail layer's output is
        the caller's ``f_input`` and is what the next segment captures, so it survives.

        Only the Python references are dropped. Resizing the storages (what
        ``ScheduleNode``'s ``free_input`` path and ``tensor_parallel.random.checkpoint``
        do) is not usable here: the replay rebuilds the graph on top of these very
        tensors, and freeing their storage corrupts it.
        """
        if layer is not self.layers[-1]:
            return
        for seg_layer in self.layers:
            seg_layer.reset_for_recompute()

    def release_input(self, layer: "TransformerLayerSchedulePlan") -> None:
        """Drop the retained input once the head layer has finished its backward."""
        if layer is not self.layers[0]:
            return
        self.input_tensor = None
        self.state_snapshot = None

    def recompute(self, layer: "TransformerLayerSchedulePlan") -> None:
        """Replay the segment with grad enabled when its tail layer enters the backward.

        Must run before any of that layer's nodes backwards: ``mtp_post_process`` is in the
        recompute scope, so its graph does not exist until the replay has run. Re-running
        the same nodes in the same order repopulates each node's ``inputs`` / ``output`` /
        ``detached`` / ``layer_state`` as in the initial forward.
        """
        if layer is not self.layers[-1]:
            return
        from megatron.core.tensor_parallel.random import _fork_rng, _set_all_rng_states

        assert self.input_tensor is not None, (
            "layer-level full recompute requires the retained segment input tensor, "
            "but it is missing."
        )
        cs = self.chunk_state

        for name, value in self.state_snapshot.items():
            setattr(cs, name, _copy_chunk_state_value(value))

        # The replayed nodes read inputs[i].grad to reach the previous segment, so the
        # retained input has to be a grad-tracking leaf.
        segment_input = self.input_tensor
        if not segment_input.requires_grad:
            segment_input.requires_grad_(True)

        for seg_layer in self.layers:
            seg_layer.set_forward_no_grad(False)

        # Replay the forward RNG stream so rng-forked ops reproduce the initial forward.
        with _fork_rng():
            _set_all_rng_states(*self.rng_states)
            f_input = segment_input
            for i, seg_layer in enumerate(self.layers):
                nvtx_msg = f"recompute_layer_{self.start_index + i}"
                nvtx_range_push(nvtx_msg)
                f_input = seg_layer.recompute_forward(f_input)
                nvtx_range_pop(nvtx_msg)
        self.rng_states = None
