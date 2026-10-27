# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Iterator, List, Optional, Tuple

import torch


class RouterReplayAction(Enum):
    """
    A Enum to define the actions for router replay.
    """

    RECORD = "record"  # Record the topk indices for replay
    REPLAY_FORWARD = "replay_forward"  # Replay the recorded topk indices for forward pass
    REPLAY_BACKWARD = "replay_backward"  # Replay topk indices for re-compute during backward pass


class RouterReplay:
    """
    A class to manage the recording and replaying of MoE routing decisions.
    It holds all router instances and provides static methods to globally
    control recording and replaying.
    """

    # Static variable to hold all router instances, one per MoE layer.
    global_router_replay_instances: List['RouterReplay'] = []

    @staticmethod
    def get_instances(is_mtp_layer: bool | None = None) -> List['RouterReplay']:
        """Get registered router replay instances, optionally filtered by layer type.

        Args:
            is_mtp_layer (bool | None): Select MTP routers when true, base-model routers
                when false, or all routers when unset.

        Returns:
            List[RouterReplay]: Matching router replay instances in registration order.
        """
        instances = RouterReplay.global_router_replay_instances
        if is_mtp_layer is None:
            return list(instances)
        return [instance for instance in instances if instance.is_mtp_layer == is_mtp_layer]

    @staticmethod
    def set_replay_data(all_layers_topk_indices: List[torch.Tensor]):
        """
        Distributes the topk indices for all layers to their respective RouterReplay instances.
        :param all_layers_topk_indices: A list of tensors, where each tensor contains the
                                        topk indices for a specific layer. The order
                                        must match the instantiation order of the routers.
        """
        if len(all_layers_topk_indices) != len(RouterReplay.global_router_replay_instances):
            raise ValueError(
                f"The number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match instances ({len(RouterReplay.global_router_replay_instances)})."
            )
        for i, router_instance in enumerate(RouterReplay.global_router_replay_instances):
            router_instance.set_target_indices(all_layers_topk_indices[i])

    @staticmethod
    def get_recorded_data(is_mtp_layer: bool | None = None) -> List[torch.Tensor | None]:
        """Collect recorded top-k indices, optionally filtered by layer type.

        Args:
            is_mtp_layer (bool | None): Select MTP routers when true, base-model routers
                when false, or all routers when unset.

        Returns:
            List[torch.Tensor | None]: Recorded indices in router registration order.
        """
        return [
            router.get_recorded_indices() for router in RouterReplay.get_instances(is_mtp_layer)
        ]

    @staticmethod
    def clear_global_indices():
        """Clears the recorded and target topk indices in all instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.clear_indices()

    @staticmethod
    def set_global_router_replay_action(
        router_replay_action: RouterReplayAction, *, is_mtp_layer: bool | None = None
    ) -> None:
        """Set the replay action, optionally filtered by layer type.

        Args:
            router_replay_action (RouterReplayAction): Replay action to set.
            is_mtp_layer (bool | None): Select MTP routers when true, base-model routers
                when false, or all routers when unset.
        """
        for router in RouterReplay.get_instances(is_mtp_layer):
            router.set_router_replay_action(router_replay_action)

    @staticmethod
    def clear_global_router_replay_action():
        """Clears the router replay action for all router instances."""
        for router in RouterReplay.global_router_replay_instances:
            router.clear_router_replay_action()

    @staticmethod
    def clear_global_router_replay_instances():
        """Clear the global list of router replay instances to prevent memory leaks."""
        RouterReplay.global_router_replay_instances.clear()

    @staticmethod
    def set_global_static_buffers(
        static_buffer: torch.Tensor,
        *,
        is_mtp_layer: bool | None = None,
        buffer_index: torch.Tensor | None = None,
        layer_indices: list[torch.Tensor] | None = None,
    ) -> None:
        """Set static buffers, optionally filtered by layer type.

        Args:
            static_buffer (torch.Tensor): Tensor shaped ``[max_tokens, num_layers, topk]``,
                with a leading buffer dimension when ``buffer_index`` is provided.
            is_mtp_layer (bool | None): Select MTP routers when true, base-model routers
                when false, or all routers when unset.
            buffer_index (torch.Tensor | None): GPU selector for double-buffered recording.
            layer_indices (list[torch.Tensor] | None): Stable GPU layer indices for
                double-buffered recording, one scalar per router.
        """
        instances = RouterReplay.get_instances(is_mtp_layer)
        num_layers = len(instances)
        assert static_buffer.shape[-2] == num_layers, (
            f"Buffer has {static_buffer.shape[-2]} layers but there are "
            f"{num_layers} RouterReplay instances."
        )
        for layer_idx, router_instance in enumerate(instances):
            if buffer_index is None:
                router_instance.set_static_buffer(static_buffer[:, layer_idx, :])
            else:
                assert layer_indices is not None
                router_instance.set_static_buffer(
                    static_buffer, buffer_index=buffer_index, layer_index=layer_indices[layer_idx]
                )

    @staticmethod
    def clear_global_static_buffers(*, is_mtp_layer: bool | None = None) -> None:
        """Clear static buffers, optionally filtered by layer type.

        Args:
            is_mtp_layer (bool | None): Select MTP routers when true, base-model routers
                when false, or all routers when unset.
        """
        for router in RouterReplay.get_instances(is_mtp_layer):
            router.clear_static_buffer()

    def __init__(self, is_mtp_layer: bool = False) -> None:
        """Initialize a RouterReplay instance for a specific layer.

        Args:
            is_mtp_layer (bool): Whether this recorder belongs to an MTP layer.
        """
        self.is_mtp_layer = is_mtp_layer
        self.target_topk_idx: Optional[torch.Tensor] = None  # Target topk indices for replay
        self.recorded_topk_idx: Optional[torch.Tensor] = None  # Recorded topk indices for replay
        self.router_replay_action: Optional[RouterReplayAction] = (
            None  # Router replay action for this layer
        )
        self.replay_backward_list: List[torch.Tensor] = (
            []
        )  # List of tensors for backward pass replay
        # Indices of the current router call, replayed in place of the micro-batch's
        # (set by ``replay_indices_for_call``, see ``ChunkedRouterReplay``)
        self.call_topk_idx: Optional[torch.Tensor] = None
        self.static_buffer: Optional[torch.Tensor] = None  # Static buffer for CUDA graph
        self.static_buffer_index: Optional[torch.Tensor] = None
        self.static_layer_index: Optional[torch.Tensor] = None
        self.layer_number: Optional[int] = None
        RouterReplay.global_router_replay_instances.append(self)

    def set_target_indices(self, topk_indices: torch.Tensor):
        """Sets the target topk indices for replay."""
        self.target_topk_idx = topk_indices
        self.replay_backward_list.append(topk_indices)

    def get_recorded_indices(self) -> Optional[torch.Tensor]:
        """Returns the recorded topk indices."""
        return self.recorded_topk_idx

    def clear_indices(self):
        """Clears the recorded and target topk indices."""
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.replay_backward_list = []

    @staticmethod
    @contextmanager
    def replay_indices_for_call(
        router_replay: Optional['RouterReplay'], topk_indices: Optional[torch.Tensor]
    ) -> Iterator[None]:
        """Make the replaying router calls inside the context use ``topk_indices``.

        These take the place of the micro-batch's target (``REPLAY_FORWARD``) and of a
        ``replay_backward_list`` entry (``REPLAY_BACKWARD``, nothing is popped). A no-op when
        ``router_replay`` or ``topk_indices`` is None.

        Args:
            router_replay (Optional[RouterReplay]): The router's replay state.
            topk_indices (Optional[torch.Tensor]): Indices of the call's tokens,
                ``[num_tokens, topk]``.
        """
        if router_replay is None or topk_indices is None:
            yield
            return
        router_replay.call_topk_idx = topk_indices
        try:
            yield
        finally:
            router_replay.call_topk_idx = None

    def set_router_replay_action(self, router_replay_action: RouterReplayAction):
        """Sets the router replay action for this layer."""
        self.router_replay_action = router_replay_action

    def clear_router_replay_action(self):
        """Clears the router replay action for this layer."""
        self.router_replay_action = None

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Callable[
            [torch.Tensor, int, Optional[int], Optional[int]], Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A wrapper for top-k computation that handles different replay actions.

        Args:
            scores (torch.Tensor): The scores to compute top-k on.
            topk (int): The number of top elements to select.
            num_groups (Optional[int]): Number of expert groups for group-limited routing.
            group_topk (Optional[int]): Number of groups to select for each token.
            default_compute_topk (Callable): The default top-k computation function, which
                                             should return a tuple of (values, indices).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the top-k values and indices.
        """
        if self.router_replay_action == RouterReplayAction.RECORD:
            probs, top_indices = default_compute_topk(
                scores, topk, num_groups=num_groups, group_topk=group_topk
            )
            self.record_indices(top_indices)
            return probs, top_indices
        elif self.call_topk_idx is not None and self.router_replay_action in (
            RouterReplayAction.REPLAY_FORWARD,
            RouterReplayAction.REPLAY_BACKWARD,
        ):
            top_indices = self.call_topk_idx.to(scores.device)
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_FORWARD:
            top_indices = self.target_topk_idx
            # Ensure indices are on the correct device
            top_indices = top_indices.to(scores.device)
            # Gather the scores for the replayed indices to get the probabilities
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        elif self.router_replay_action == RouterReplayAction.REPLAY_BACKWARD:
            top_indices = self.replay_backward_list.pop(0)
            # Ensure indices are on the correct device
            top_indices = top_indices.to(scores.device)
            # Gather the scores for the replayed indices to get the probabilities
            probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return default_compute_topk(scores, topk, num_groups, group_topk)

    def set_static_buffer(
        self,
        buffer: torch.Tensor,
        *,
        buffer_index: torch.Tensor | None = None,
        layer_index: torch.Tensor | None = None,
    ) -> None:
        """Sets a static buffer for CUDA graph compatible recording.

        Args:
            buffer (torch.Tensor): Per-layer destination shaped ``[max_tokens, topk]``
                for legacy recording, or the full contiguous
                ``[2, max_tokens, num_layers, topk]`` buffer for async recording.
            buffer_index (torch.Tensor | None): GPU selector for double-buffered recording.
            layer_index (torch.Tensor | None): GPU layer index within the shared buffer.
        """
        self.static_buffer = buffer
        self.static_buffer_index = buffer_index
        self.static_layer_index = layer_index

    def clear_static_buffer(self):
        """Clears the static buffer."""
        self.static_buffer = None
        self.static_buffer_index = None
        self.static_layer_index = None

    def record_indices(self, topk_indices: torch.Tensor) -> None:
        """Records the topk indices.

        If a static buffer is set (for CUDA graph compatibility), copies into it.
        Otherwise, just stores the tensor reference.

        Args:
            topk_indices (torch.Tensor): Selected expert indices for this layer's tokens.
        """
        if self.static_buffer_index is not None:
            rows = torch.arange(topk_indices.shape[0], device=topk_indices.device)
            # Keep the compiled input contiguous: per-layer views can trigger
            # copies spanning both banks, racing the other bank's D2H.
            self.static_buffer.index_put_(
                (
                    self.static_buffer_index[:, None],
                    rows[None, :],
                    self.static_layer_index[:, None],
                ),
                topk_indices.to(self.static_buffer.dtype).unsqueeze(0),
            )
            self.recorded_topk_idx = topk_indices
        elif self.static_buffer is not None:
            # Copy into static buffer for CUDA graph compatibility.
            num_tokens = topk_indices.shape[0]
            self.static_buffer[:num_tokens].copy_(topk_indices)
            self.recorded_topk_idx = self.static_buffer[:num_tokens]
        else:
            self.recorded_topk_idx = topk_indices


class ChunkedRouterReplay:
    """Routing replay for an MoE layer that is called once per sequence chunk.

    ``RouterReplay`` assumes one router call per micro-batch. ``target_topk_idx`` and each
    ``replay_backward_list`` entry hold the top-k of the whole micro-batch (``[s * b, topk]``),
    and ``get_replay_topk`` gathers the call's scores with them, so a call over a chunk
    (``[chunk_tokens, num_experts]`` scores) fails with a size mismatch. Recompute pops one entry
    per call, and recording keeps only the last call's indices.

    This gives each chunk its slice of the micro-batch's indices, which the caller passes to the
    chunk's MoE call (``MoELayer.forward(replay_topk_idx=...)``), and records the concatenated
    indices of all chunks once the last chunk is done. Passing the slice as an argument binds it
    to the call: with ``recompute_modules=["moe"]``, backward reruns each chunk's checkpointed
    MoE call on its own (not this loop), in whatever order autograd picks, and the rerun replays
    the same slice.

    Usage, around the per-chunk calls::

        replay = ChunkedRouterReplay(router_replay, chunk_tokens, pad_tokens, chunks_recompute)
        for i, chunk in enumerate(chunks):
            outputs.append(mlp(chunk, replay_topk_idx=replay.chunk_indices(i)))
            replay.after_chunk()
        replay.finish()
    """

    def __init__(
        self,
        router_replay: Optional[RouterReplay],
        chunk_tokens: List[int],
        pad_tokens: int = 0,
        chunks_recompute: bool = False,
    ) -> None:
        """
        Args:
            router_replay (Optional[RouterReplay]): The layer's replay state, or None when
                routing replay is disabled (every method is then a no-op).
            chunk_tokens (List[int]): Number of tokens (``s_chunk * b``) of each chunk, in order.
            pad_tokens (int): Number of dummy tokens appended after the real tokens to even out
                the chunk count. They replay the first real token's experts, and are excluded
                from the recorded indices.
            chunks_recompute (bool): Whether each chunk's MoE call is checkpointed on its own
                (selective ``"moe"`` recompute). Its rerun replays the chunk's slice of the
                forward target, so the micro-batch's ``replay_backward_list`` entry, which no
                rerun would pop, is dropped.
        """
        self.replay = router_replay
        self.action = router_replay.router_replay_action if router_replay is not None else None
        self.bounds = [0]
        for n in chunk_tokens:
            self.bounds.append(self.bounds[-1] + n)
        self.real_tokens = self.bounds[-1] - pad_tokens
        self.recorded: List[torch.Tensor] = []
        self.indices: Optional[torch.Tensor] = None
        if self.action == RouterReplayAction.REPLAY_FORWARD:
            target = self.replay.target_topk_idx
            backward_list = self.replay.replay_backward_list
            # set_target_indices appended the target as this micro-batch's entry.
            if chunks_recompute and backward_list and backward_list[-1] is target:
                backward_list.pop()
            self.indices = self._pad(target, pad_tokens)
        elif self.action == RouterReplayAction.REPLAY_BACKWARD and self.replay.replay_backward_list:
            self.indices = self._pad(self.replay.replay_backward_list.pop(0), pad_tokens)

    @staticmethod
    def _pad(indices: Optional[torch.Tensor], pad_tokens: int) -> Optional[torch.Tensor]:
        """Append ``pad_tokens`` rows of valid, distinct expert indices for the dummy tokens."""
        if not pad_tokens or indices is None:
            return indices
        if indices.shape[0] == 0:
            dummy = torch.arange(indices.shape[1], dtype=indices.dtype, device=indices.device)
            return dummy.expand(pad_tokens, -1)
        return torch.cat([indices, indices[:1].expand(pad_tokens, -1)], dim=0)

    def chunk_indices(self, i: int) -> Optional[torch.Tensor]:
        """Chunk ``i``'s replay indices, or None when not replaying."""
        if self.indices is None:
            return None
        return self.indices[self.bounds[i] : self.bounds[i + 1]]

    def after_chunk(self) -> None:
        """Keep the indices the last chunk recorded (a static buffer is reused by the next)."""
        if self.action == RouterReplayAction.RECORD and self.replay.recorded_topk_idx is not None:
            self.recorded.append(self.replay.recorded_topk_idx.clone())

    def finish(self) -> None:
        """Record the indices of all real tokens."""
        if self.action == RouterReplayAction.RECORD and self.recorded:
            self.replay.record_indices(torch.cat(self.recorded, dim=0)[: self.real_tokens])
