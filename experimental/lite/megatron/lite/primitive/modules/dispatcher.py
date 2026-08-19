# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Token dispatcher: AllToAll and DeepEP dispatch/combine."""

from __future__ import annotations

import os

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.modules.moe import _AllToAll
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.utils import ensure_divisible
from megatron.lite.primitive.utils.moe import permute, unpermute
from megatron.lite.primitive.alignment.deepep_route import (
    _VLLMEPGatherWithBF16Backward,
    _compact_route_preserving_metadata_inputs,
    _deepep_route_handle_received_rows,
    _scatter_deepep_routes_with_padding,
    _validate_and_order_route_preserving_outputs,
)

try:
    import deep_ep  # pyright: ignore[reportMissingImports]
    from deep_ep.utils import EventHandle, EventOverlap  # pyright: ignore[reportMissingImports]
except ImportError:
    deep_ep = None  # type: ignore
    EventHandle = None  # type: ignore
    EventOverlap = None  # type: ignore


_deepep_buffer = None
def _configure_deepep_deterministic_allocator() -> None:
    """Avoid DeepEP's known deterministic ``torch.empty`` stream race.

    Legacy normal DeepEP waits its communication stream before allocating
    output tensors.  PyTorch's deterministic debug fill is consequently
    enqueued after that wait on the compute stream and can race DeepEP's
    communication-stream writers.  ``fill_uninitialized_memory`` is only an
    uninitialized-memory debugging aid; disabling it does not relax PyTorch's
    deterministic algorithm selection.

    See https://github.com/deepseek-ai/DeepEP/issues/589.
    """

    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory
    ):
        torch.utils.deterministic.fill_uninitialized_memory = False


def _hidden_bytes(hidden_size: int) -> int:
    return hidden_size * 2


def _get_deepep_buffer(group: dist.ProcessGroup, hidden_bytes: int):
    """Return MCore's process-wide normal-DeepEP communication buffer.

    This deliberately mirrors ``megatron.core.transformer.moe.fused_a2a``:
    dispatch and combine look the buffer up from their process group and the
    payload width on every forward/backward entry.  In particular, the tiny
    route-fingerprint dispatch reuses the full-width primary buffer instead of
    creating a second per-layer DeepEP runtime.
    """

    if deep_ep is None:
        raise RuntimeError("DeepEP buffer requested but deep_ep is not installed.")

    _configure_deepep_deterministic_allocator()

    global _deepep_buffer
    group_size = dist.get_world_size(group=group)
    num_nvl_bytes = 0
    num_rdma_bytes = 0

    for config in (
        deep_ep.Buffer.get_dispatch_config(group_size),
        deep_ep.Buffer.get_combine_config(group_size),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group_size), num_nvl_bytes
        )
        # Match MCore: a node-local EP group neither needs an RDMA heap nor can
        # assume that a DeepEP build exposes internode size hints.
        if group_size > torch.cuda.device_count():
            num_rdma_bytes = max(
                config.get_rdma_buffer_size_hint(hidden_bytes, group_size),
                num_rdma_bytes,
            )

    if (
        _deepep_buffer is None
        or getattr(_deepep_buffer, "runtime", None) is None
        or _deepep_buffer.group != group
        or _deepep_buffer.num_nvl_bytes < num_nvl_bytes
        or _deepep_buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _deepep_buffer = deep_ep.Buffer(
            group=group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            explicitly_destroy=True,
        )
    return _deepep_buffer


def _build_deepep_buffer(group: dist.ProcessGroup, hidden_size: int):
    """Compatibility wrapper for callers that specify a BF16 hidden width."""

    return _get_deepep_buffer(group, _hidden_bytes(hidden_size))


def _use_moe_permute_fusion() -> bool:
    return os.environ.get("MEGATRON_LITE_MOE_PERMUTE_FUSION", "0") == "1"


def _tensor_hidden_bytes(x: torch.Tensor) -> int:
    return x.size(1) * max(x.element_size(), 2)


def _dispatch_deepep_raw(
    group,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    num_experts: int,
    *,
    async_finish: bool,
    allocate_on_comm_stream: bool,
):
    """Run the shared normal-DeepEP dispatch without owning autograd state."""
    buffer = _get_deepep_buffer(group, _tensor_hidden_bytes(hidden_states))
    hidden_states = hidden_states.contiguous()
    topk_indices = topk_indices.contiguous()
    topk_scores = topk_scores.float().contiguous()
    previous_event = (
        EventOverlap(EventHandle())
        if async_finish and EventHandle is not None and EventOverlap is not None
        else None
    )
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        event,
    ) = buffer.get_dispatch_layout(
        topk_indices,
        num_experts=num_experts,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    result = buffer.dispatch(
        hidden_states,
        topk_idx=topk_indices,
        topk_weights=topk_scores,
        num_tokens_per_rank=num_tokens_per_rank,
        num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
        is_token_in_rank=is_token_in_rank,
        num_tokens_per_expert=num_tokens_per_expert,
        previous_event=event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )
    if async_finish:
        result[-1].current_stream_wait()
    return result


class _DeepEPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        group,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        num_experts: int,
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ):
        (
            recv_hidden,
            recv_indices,
            recv_probs,
            recv_per_expert,
            handle,
            _,
        ) = _dispatch_deepep_raw(
            group,
            hidden_states,
            topk_indices,
            topk_scores,
            num_experts,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        # Match MCore/Slime's normal-DeepEP contract: the public API returns
        # receive counts as a Python list and the dispatcher materializes that
        # metadata on CPU.  Do not enqueue an unrelated CUDA allocation while
        # DeepEP's communication result is becoming visible.  The aligned
        # scatter consumes CPU counts directly and derives its CUDA counts from
        # the received route metadata when needed.
        recv_per_expert_tensor = torch.tensor(recv_per_expert, dtype=torch.int64)
        return recv_hidden, recv_indices, recv_probs, recv_per_expert_tensor, handle

    @staticmethod
    def backward(
        ctx, grad_recv_hidden, grad_recv_indices, grad_recv_probs, grad_recv_per_expert, grad_handle
    ):
        del grad_recv_indices, grad_recv_per_expert, grad_handle
        previous_event = (
            EventOverlap(EventHandle())
            if ctx.async_finish and EventHandle is not None and EventOverlap is not None
            else None
        )
        buffer = _get_deepep_buffer(ctx.group, _tensor_hidden_bytes(grad_recv_hidden))
        grad_scores = None if grad_recv_probs is None else grad_recv_probs.float()
        grad_hidden, grad_topk_scores, after_event = buffer.combine(
            grad_recv_hidden.contiguous(),
            ctx.handle,
            topk_weights=grad_scores,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return None, grad_hidden, None, grad_topk_scores, None, None, None


class _DeepEPCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        group,
        rank_grouped: torch.Tensor,
        handle,
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ):
        buffer = _get_deepep_buffer(group, _tensor_hidden_bytes(rank_grouped))
        previous_event = (
            EventOverlap(EventHandle())
            if async_finish and EventHandle is not None and EventOverlap is not None
            else None
        )
        combined, _, after_event = buffer.combine(
            rank_grouped,
            handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event.current_stream_wait()
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined

    @staticmethod
    def backward(ctx, grad_output):
        previous_event = (
            EventOverlap(EventHandle())
            if ctx.async_finish and EventHandle is not None and EventOverlap is not None
            else None
        )
        buffer = _get_deepep_buffer(ctx.group, _tensor_hidden_bytes(grad_output))
        grad_rank_grouped, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return None, grad_rank_grouped, None, None, None


class TokenDispatcher:

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        ps: ParallelState,
        *,
        use_deepep: bool = True,
        deepep_align_to_low_latency: bool = False,
        moe_permute_fusion: bool | None = None,
        capacity_factor: float | None = None,
    ):
        self.ps = ps
        self.num_experts = num_experts
        self.ep_size = ps.ep_size
        self.num_local_experts = ensure_divisible(num_experts, ps.ep_size)
        self.moe_permute_fusion = (
            _use_moe_permute_fusion() if moe_permute_fusion is None else bool(moe_permute_fusion)
        )

        self.use_deepep = use_deepep and deep_ep is not None and ps.ep_size > 1
        self.deepep_align_to_low_latency = bool(deepep_align_to_low_latency)
        self.capacity_factor = capacity_factor
        if self.deepep_align_to_low_latency and ps.ep_size > 1 and not self.use_deepep:
            raise RuntimeError(
                "low-latency semantic alignment at EP>1 requires normal DeepEP"
            )
        if self.use_deepep:
            assert ps.tp_ep_group is not None
            self._deepep_group = ps.tp_ep_group
            # Match MCore's _DeepepManager initialization exactly.  DeepEP's
            # SM allocation is process-global; leaving it at the package
            # default makes this shared dispatcher differ from both mLite.lite
            # through MCore and Slime, and can starve later collectives while
            # grouped expert kernels are active.
            deep_ep.Buffer.set_num_sms(20)
            # Match MCore/Slime: get_buffer() is entered by the first dispatch,
            # after a colocated trainer has woken and restored its CUDA state.
            # Eager construction here makes the DeepEP runtime span VERL's
            # initial model offload/empty-cache boundary, a lifecycle that the
            # mature fused_a2a manager deliberately avoids.
            self.buffer = None

        self._row_id_map: torch.Tensor | None = None
        self._restore_shape: tuple | None = None
        self._input_splits: list[int] | None = None
        self._output_splits: list[int] | None = None
        self._handle = None
        self._deepep_event = None
        if self.ep_size > 1 and self.num_local_experts > 1:
            chunk_idxs = torch.arange(self.ep_size * self.num_local_experts, device="cpu")
            self._sort_by_experts = (
                chunk_idxs.reshape(self.ep_size, self.num_local_experts).T.ravel().tolist()
            )
            self._restore_by_ranks = (
                chunk_idxs.reshape(self.num_local_experts, self.ep_size).T.ravel().tolist()
            )

    def _ensure_deepep_buffer(self, hidden_states: torch.Tensor):
        """Lazily enter MCore/Slime's process-wide normal DeepEP runtime."""

        if not self.use_deepep:
            raise RuntimeError("DeepEP buffer requested while DeepEP is disabled")
        self.buffer = _get_deepep_buffer(
            self._deepep_group, _tensor_hidden_bytes(hidden_states)
        )
        return self.buffer

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
        *,
        router_token_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Match slime's _DeepepManager.setup_metadata contract.  The fixed-topk
        # route path is valid only when neither expert-capacity dropping nor a
        # router token mask can introduce -1 sentinels.
        source_fixed_topk_valid = (
            self.capacity_factor is None and router_token_masks is None
        )
        if self.capacity_factor is not None:
            topk_indices = topk_indices.masked_fill(topk_scores == 0, -1)
        if router_token_masks is not None:
            topk_indices = topk_indices.masked_fill(
                router_token_masks.view(-1, 1), -1
            )
        if self.deepep_align_to_low_latency:
            return self._dispatch_low_latency_aligned(
                hidden_states,
                topk_scores,
                topk_indices,
                source_fixed_topk_valid=source_fixed_topk_valid,
            )
        if self.ep_size <= 1:
            return self._dispatch_local(hidden_states, topk_scores, topk_indices)
        if self.use_deepep:
            return self._dispatch_deepep(hidden_states, topk_scores, topk_indices)
        dispatched, tpe, sorted_scores = self._dispatch_alltoall(
            hidden_states, topk_scores, topk_indices
        )
        return dispatched, tpe, sorted_scores

    def combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        if self.deepep_align_to_low_latency:
            return self._combine_low_latency_aligned(expert_output)
        if self.ep_size <= 1:
            return self._combine_local(expert_output)
        if self.use_deepep:
            return self._combine_deepep(expert_output)
        return self._combine_alltoall(expert_output)

    def _dispatch_low_latency_aligned(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
        *,
        source_fixed_topk_valid: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normal DeepEP transport with vLLM LL route/layout semantics."""

        if self.use_deepep:
            self._ensure_deepep_buffer(hidden_states)

        if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
            raise TypeError("aligned DeepEP requires BF16 [tokens, hidden]")
        if hidden_states.shape[1] < 16:
            raise ValueError("aligned DeepEP requires hidden size >= 16")
        if topk_indices.shape != topk_scores.shape:
            raise ValueError("top-k IDs and scores must have identical shapes")
        topk_indices = topk_indices.long().contiguous()
        topk_scores = topk_scores.float().contiguous()

        if self.ep_size > 1:
            (
                received_hidden,
                received_indices,
                received_weights,
                received_per_expert,
                _,
            ) = _DeepEPDispatch.apply(
                self._deepep_group,
                hidden_states,
                topk_indices,
                topk_scores,
                self.num_experts,
                False,
                False,
            )
            (
                route_indices,
                route_weights,
                route_fingerprints,
                source_output_index,
                source_all_routes_valid,
            ) = _compact_route_preserving_metadata_inputs(
                hidden_states,
                topk_indices,
                topk_scores,
                assume_all_routes_valid=source_fixed_topk_valid,
            )
            (
                received_fingerprints,
                received_route_indices,
                received_route_weights,
                _,
                route_handle,
                _,
            ) = _dispatch_deepep_raw(
                self._deepep_group,
                route_fingerprints,
                route_indices,
                route_weights,
                self.num_experts,
                async_finish=False,
                allocate_on_comm_stream=False,
            )
        else:
            received_hidden = hidden_states
            received_indices = topk_indices
            received_weights = topk_scores
            positions = torch.nonzero(topk_indices >= 0, as_tuple=False)
            token_rows = positions[:, 0]
            topk_slots = positions[:, 1]
            received_fingerprints = hidden_states.detach().narrow(
                1, 0, 16
            ).index_select(0, token_rows)
            received_route_indices = topk_indices[token_rows, topk_slots]
            received_route_weights = topk_scores[token_rows, topk_slots]
            route_handle = None
            source_output_index = torch.arange(
                topk_indices.numel(), device=topk_indices.device, dtype=torch.long
            ).reshape_as(topk_indices)
            source_all_routes_valid = True
            received_per_expert = torch.bincount(
                received_route_indices.reshape(-1).long(),
                minlength=self.num_local_experts,
            )

        expected_route_count = (
            _deepep_route_handle_received_rows(route_handle)
            if route_handle is not None
            else int((received_indices >= 0).sum().item())
        )
        (
            expert_hidden,
            expert_probs,
            output_index,
            sanitized_indices,
            _,
            all_routes_valid,
            device_tokens_per_expert,
            positions,
        ) = _scatter_deepep_routes_with_padding(
            received_hidden,
            received_indices,
            received_weights,
            received_per_expert,
            return_route_positions=True,
            expected_route_count=expected_route_count,
        )
        metadata_route_rows = _validate_and_order_route_preserving_outputs(
            expert_hidden,
            received_hidden,
            sanitized_indices,
            received_weights,
            output_index,
            received_fingerprints,
            received_route_indices.reshape(-1),
            received_route_weights.reshape(-1),
            order_outputs=False,
            route_positions=positions,
            return_route_rows=True,
        )
        self._aligned_received_output_index = output_index
        self._aligned_received_positions = positions
        self._aligned_metadata_route_rows = metadata_route_rows
        self._aligned_route_handle = route_handle
        self._aligned_source_indices = topk_indices
        self._aligned_source_weights = topk_scores
        self._aligned_source_shape = hidden_states.shape
        self._aligned_source_output_index = source_output_index
        self._aligned_source_all_routes_valid = source_all_routes_valid
        self._aligned_device_tokens_per_expert = device_tokens_per_expert
        self._local_tpe_list = received_per_expert.tolist()
        return expert_hidden, received_per_expert, expert_probs

    def _combine_low_latency_aligned(
        self, expert_output: torch.Tensor
    ) -> torch.Tensor:
        route_outputs = expert_output.index_select(
            0, self._aligned_metadata_route_rows
        )
        if self.ep_size > 1:
            # Match Slime's route-preserving DeepEP lifecycle in both training
            # and forward-only execution: submit combine asynchronously through
            # the autograd wrapper, then make the current stream wait for the
            # returned event.  Calling Buffer.combine synchronously only in
            # no-grad mode leaves a different channel/handle completion order
            # across pipeline microbatches.
            source_routes = _DeepEPCombine.apply(
                self._deepep_group,
                route_outputs,
                self._aligned_route_handle,
                True,
                False,
            )
        else:
            source_routes = route_outputs
        output = _VLLMEPGatherWithBF16Backward.apply(
            source_routes,
            self._aligned_source_indices,
            self._aligned_source_weights,
            self._aligned_source_output_index,
            True,
            self._aligned_source_all_routes_valid,
        )
        for name in (
            "_aligned_received_output_index",
            "_aligned_received_positions",
            "_aligned_metadata_route_rows",
            "_aligned_route_handle",
            "_aligned_source_indices",
            "_aligned_source_weights",
            "_aligned_source_shape",
            "_aligned_source_output_index",
            "_aligned_source_all_routes_valid",
            "_aligned_device_tokens_per_expert",
        ):
            delattr(self, name)
        self._local_tpe_list = None
        return output

    def submit_deepep_combine(
        self, expert_output: torch.Tensor, *, allocate_on_comm_stream: bool = False
    ):
        if not self.use_deepep:
            raise RuntimeError("submit_deepep_combine requires DeepEP combine.")
        rank_grouped = unpermute(
            expert_output,
            self._row_id_map,
            restore_shape=self._restore_shape,
            fused=self.moe_permute_fusion,
        )
        previous_event = (
            EventOverlap(EventHandle())
            if EventHandle is not None and EventOverlap is not None
            else None
        )
        buffer = _get_deepep_buffer(
            self._deepep_group, _tensor_hidden_bytes(rank_grouped)
        )
        combined = buffer.combine(
            rank_grouped,
            self._handle,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        event = None
        if isinstance(combined, tuple):
            if len(combined) >= 3:
                event = combined[2]
            combined = combined[0]
        return {"combined": combined, "event": event}

    def finish_deepep_combine(self, state):
        if not self.use_deepep:
            raise RuntimeError("finish_deepep_combine requires DeepEP combine.")
        event = state.get("event")
        if event is not None:
            event.current_stream_wait()
        self._row_id_map = None
        self._restore_shape = None
        self._handle = None
        self._local_tpe_list = None
        return state["combined"]

    def _dispatch_local(self, hidden_states, topk_scores, topk_indices):
        t, h = hidden_states.shape
        e = self.num_experts

        routing_map = torch.zeros(t, e, dtype=torch.bool, device=hidden_states.device)
        routing_map.scatter_(1, topk_indices, True)
        num_out = int(routing_map.sum().item())

        probs_2d = torch.zeros(t, e, dtype=topk_scores.dtype, device=hidden_states.device)
        probs_2d.scatter_add_(1, topk_indices, topk_scores)

        permuted, permuted_probs, sorted_indices = permute(
            hidden_states,
            routing_map,
            probs=probs_2d,
            num_out_tokens=num_out,
            fused=self.moe_permute_fusion,
        )[:3]

        self._row_id_map = sorted_indices
        self._restore_shape = hidden_states.shape

        tokens_per_expert = routing_map.sum(dim=0).to(torch.int64)
        return permuted, tokens_per_expert, permuted_probs

    def _combine_local(self, expert_output):
        result = unpermute(
            expert_output,
            self._row_id_map,
            restore_shape=self._restore_shape,
            fused=self.moe_permute_fusion,
        )
        self._row_id_map = None
        self._restore_shape = None
        return result

    def _dispatch_alltoall(self, hidden_states, topk_scores, topk_indices):
        t, h = hidden_states.shape
        e = self.num_experts

        routing_map = torch.zeros(t, e, dtype=torch.bool, device=hidden_states.device)
        routing_map.scatter_(1, topk_indices, True)
        # Use the actual number of routed (token, expert) pairs from routing_map
        # rather than t * topk: hash routing (ds4) can map a token's topk slots to
        # DUPLICATE experts, which scatter_ dedups, so t*topk would overcount and
        # leave permuted.size(0) != sum(input_splits) (all-to-all split mismatch).
        # Unique-topk routers (every other model) have routing_map.sum() == t*topk,
        # so this is a no-op for them.
        num_out = int(routing_map.sum().item())

        probs_2d = torch.zeros(t, e, dtype=topk_scores.dtype, device=hidden_states.device)
        probs_2d.scatter_add_(1, topk_indices, topk_scores)

        permuted, permuted_probs, sorted_indices = permute(
            hidden_states,
            routing_map,
            probs=probs_2d,
            num_out_tokens=num_out,
            fused=self.moe_permute_fusion,
        )[:3]
        self._row_id_map = sorted_indices
        self._restore_shape = hidden_states.shape

        tokens_per_expert = routing_map.sum(dim=0).to(torch.int64)
        tpe_by_rank = tokens_per_expert.view(self.ep_size, self.num_local_experts).sum(dim=1)
        self._input_splits = tpe_by_rank.tolist()

        global_tpe_flat = tokens_per_expert.new_empty(self.ep_size * e)
        dist.all_gather_into_tensor(global_tpe_flat, tokens_per_expert, group=self.ps.ep_group)
        global_tpe_2d = global_tpe_flat.view(self.ep_size, e)
        ep_rank = dist.get_rank(group=self.ps.ep_group)
        my_start = ep_rank * self.num_local_experts
        recv_tpe_2d = global_tpe_2d[:, my_start : my_start + self.num_local_experts].contiguous()
        self._output_splits = recv_tpe_2d.sum(dim=1).tolist()

        recv_flat = _AllToAll.apply(
            permuted, self._input_splits, self._output_splits, self.ps.ep_group
        )
        recv_scores = _AllToAll.apply(
            permuted_probs.unsqueeze(-1), self._input_splits, self._output_splits, self.ps.ep_group
        )

        if self.num_local_experts > 1:
            chunk_sizes = recv_tpe_2d.ravel().tolist()
            chunks = torch.split(recv_flat, chunk_sizes, dim=0)
            score_chunks = torch.split(recv_scores, chunk_sizes, dim=0)
            sort_idxs = self._sort_by_experts
            restore_idxs = self._restore_by_ranks
            dispatched = torch.cat([chunks[i] for i in sort_idxs], dim=0)
            permuted_probs_out = torch.cat([score_chunks[i] for i in sort_idxs], dim=0)
            self._combine_chunk_sizes = [chunk_sizes[i] for i in sort_idxs]
            self._combine_restore_idxs = restore_idxs
        else:
            dispatched = recv_flat
            permuted_probs_out = recv_scores
            self._combine_chunk_sizes = None
            self._combine_restore_idxs = None

        recv_tpe = recv_tpe_2d.sum(dim=0)
        return dispatched, recv_tpe, permuted_probs_out.squeeze(-1)

    def _combine_alltoall(self, expert_output):
        if self._combine_chunk_sizes is not None:
            chunks = torch.split(expert_output, self._combine_chunk_sizes, dim=0)
            restore_idxs = (
                self._combine_restore_idxs
                if self._combine_restore_idxs is not None
                else self._restore_by_ranks
            )
            rank_grouped = torch.cat([chunks[i] for i in restore_idxs], dim=0)
        else:
            rank_grouped = expert_output

        combined = _AllToAll.apply(
            rank_grouped, self._output_splits, self._input_splits, self.ps.ep_group
        )
        result = unpermute(
            combined,
            self._row_id_map,
            restore_shape=self._restore_shape,
            fused=self.moe_permute_fusion,
        )
        self._row_id_map = None
        self._restore_shape = None
        self._input_splits = None
        self._output_splits = None
        self._combine_chunk_sizes = None
        self._combine_restore_idxs = None
        self._local_tpe_list = None
        return result

    def submit_deepep_dispatch(
        self, hidden_states, topk_scores, topk_indices, *, allocate_on_comm_stream: bool = False
    ):
        if not self.use_deepep:
            raise RuntimeError("submit_deepep_dispatch requires DeepEP dispatch.")
        hidden_states = hidden_states.contiguous()
        topk_indices = topk_indices.contiguous()
        topk_scores = topk_scores.float().contiguous()
        previous_event = (
            EventOverlap(EventHandle())
            if EventHandle is not None and EventOverlap is not None
            else None
        )
        buffer = self._ensure_deepep_buffer(hidden_states)
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            event,
        ) = buffer.get_dispatch_layout(
            topk_indices,
            num_experts=self.num_experts,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        recv_hidden, recv_indices, recv_probs, recv_per_expert, handle, event = (
            buffer.dispatch(
                hidden_states,
                topk_idx=topk_indices,
                topk_weights=topk_scores,
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                previous_event=event,
                async_finish=True,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        )
        return {
            "recv_hidden": recv_hidden,
            "recv_indices": recv_indices,
            "recv_probs": recv_probs,
            "recv_per_expert": recv_per_expert,
            "handle": handle,
            "event": event,
        }

    def finish_deepep_dispatch(self, state):
        if not self.use_deepep:
            raise RuntimeError("finish_deepep_dispatch requires DeepEP dispatch.")
        self._handle = state["handle"]
        self._deepep_event = state["event"]
        self.wait_dispatch_event()
        return self._finish_deepep_dispatch(
            state["recv_hidden"],
            state["recv_indices"],
            state["recv_probs"],
            state["recv_per_expert"],
        )

    def _finish_deepep_dispatch(
        self,
        recv_hidden: torch.Tensor,
        recv_indices: torch.Tensor,
        recv_probs: torch.Tensor,
        recv_per_expert,
    ):
        if isinstance(recv_per_expert, torch.Tensor):
            recv_per_expert = [int(x) for x in recv_per_expert.detach().cpu().tolist()]
        rows = recv_hidden.size(0)
        recv_indices = recv_indices.to(torch.long)
        routing_map = torch.zeros(
            rows, self.num_local_experts, dtype=torch.bool, device=recv_hidden.device
        )
        probs_2d = torch.zeros(
            rows, self.num_local_experts, dtype=recv_probs.dtype, device=recv_hidden.device
        )
        valid = recv_indices >= 0
        row_ids = torch.arange(rows, device=recv_hidden.device).unsqueeze(1)
        row_ids = row_ids.expand_as(recv_indices)[valid]
        expert_ids = recv_indices[valid]
        routing_map[row_ids, expert_ids] = True
        # DS4 hash routing can map multiple top-k slots of one token to the
        # same expert. The boolean routing map intentionally deduplicates those
        # routes, so their weights must be accumulated and expert counts must
        # be derived from that same deduplicated map. Using DeepEP's raw slot
        # counts here over-allocates permute rows and feeds uninitialized
        # expert outputs into combine.
        probs_2d.index_put_(
            (row_ids, expert_ids),
            recv_probs[valid],
            accumulate=True,
        )
        local_tpe = routing_map.sum(dim=0).to(torch.int64)
        self._local_tpe_list = [int(x) for x in local_tpe.detach().cpu().tolist()]
        num_out = int(local_tpe.sum().item())
        dispatched, permuted_probs, sorted_indices = permute(
            recv_hidden,
            routing_map,
            probs=probs_2d,
            num_out_tokens=num_out,
            fused=self.moe_permute_fusion,
        )[:3]
        self._row_id_map = sorted_indices
        self._restore_shape = recv_hidden.shape
        if os.environ.get("MEGATRON_LITE_DEEPEP_DEBUG_METADATA") == "1":
            ep_rank = dist.get_rank(group=self.ps.ep_group)
            print(
                "[DEEPEP_METADATA] "
                f"ep_rank={ep_rank} recv_rows={int(recv_hidden.shape[0])} "
                f"expert_rows={int(dispatched.shape[0])} "
                f"recv_indices_shape={tuple(recv_indices.shape)} "
                f"recv_per_expert_len={len(recv_per_expert)} "
                f"recv_per_expert_sum={sum(int(x) for x in recv_per_expert)} "
                f"recv_per_expert_head={recv_per_expert[: self.num_local_experts]} "
                f"local_tpe_sum={int(local_tpe.sum().item())}",
                flush=True,
            )
        if os.environ.get("MEGATRON_LITE_DEEPEP_SKIP_DISPATCH_METADATA_CHECK") != "1" and int(
            local_tpe.sum().item()
        ) != int(dispatched.shape[0]):
            ep_rank = dist.get_rank(group=self.ps.ep_group)
            raise RuntimeError(
                "DeepEP dispatch metadata mismatch: "
                f"ep_rank={ep_rank} dispatched_tokens={int(dispatched.shape[0])} "
                f"local_tpe={local_tpe.tolist()} recv_per_expert_len={len(recv_per_expert)}"
            )
        return dispatched, local_tpe, permuted_probs

    def _dispatch_deepep(self, hidden_states, topk_scores, topk_indices):
        self._ensure_deepep_buffer(hidden_states)
        if torch.is_grad_enabled():
            recv_hidden, recv_indices, recv_probs, recv_per_expert, handle = _DeepEPDispatch.apply(
                self._deepep_group,
                hidden_states,
                topk_indices,
                topk_scores.float(),
                self.num_experts,
                False,
                False,
            )
            self._handle = handle
            self._deepep_event = None
            return self._finish_deepep_dispatch(
                recv_hidden, recv_indices, recv_probs, recv_per_expert
            )
        state = self.submit_deepep_dispatch(
            hidden_states, topk_scores, topk_indices, allocate_on_comm_stream=False
        )
        return self.finish_deepep_dispatch(state)

    def wait_dispatch_event(self):
        if self._deepep_event is not None:
            self._deepep_event.current_stream_wait()
            self._deepep_event = None

    def _combine_deepep(self, expert_output):
        rank_grouped = unpermute(
            expert_output,
            self._row_id_map,
            restore_shape=self._restore_shape,
            fused=self.moe_permute_fusion,
        )
        if torch.is_grad_enabled():
            combined = _DeepEPCombine.apply(
                self._deepep_group, rank_grouped, self._handle, False, False
            )
        else:
            buffer = _get_deepep_buffer(
                self._deepep_group, _tensor_hidden_bytes(rank_grouped)
            )
            combined = buffer.combine(rank_grouped, self._handle)
        if isinstance(combined, tuple):
            combined = combined[0]
        self._row_id_map = None
        self._restore_shape = None
        self._handle = None
        self._local_tpe_list = None
        return combined


__all__ = ["TokenDispatcher"]
