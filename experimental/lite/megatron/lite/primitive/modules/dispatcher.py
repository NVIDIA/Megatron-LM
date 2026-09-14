# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Token dispatcher: AllToAll and DeepEP dispatch/combine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.modules.moe import _AllToAll
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.utils import ensure_divisible
from megatron.lite.primitive.utils.moe import permute, unpermute

try:
    import deep_ep  # pyright: ignore[reportMissingImports]
    from deep_ep.utils import EventHandle, EventOverlap  # pyright: ignore[reportMissingImports]
except ImportError:
    deep_ep = None  # type: ignore
    EventHandle = None  # type: ignore
    EventOverlap = None  # type: ignore


def _hidden_bytes(hidden_size: int) -> int:
    return hidden_size * 2


@dataclass(frozen=True)
class _DeepEPBufferAllocation:
    buffer: Any
    num_nvl_bytes: int
    num_rdma_bytes: int

    @property
    def resident_bytes(self) -> int:
        return self.num_nvl_bytes + self.num_rdma_bytes


def _build_deepep_buffer(group: dist.ProcessGroup, hidden_size: int) -> _DeepEPBufferAllocation:
    if deep_ep is None:
        raise RuntimeError("DeepEP buffer requested but deep_ep is not installed.")

    group_size = dist.get_world_size(group=group)
    hidden_bytes = _hidden_bytes(hidden_size)
    num_nvl_bytes = 0
    num_rdma_bytes = 0

    for config in (
        deep_ep.Buffer.get_dispatch_config(group_size),
        deep_ep.Buffer.get_combine_config(group_size),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group_size), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group_size), num_rdma_bytes
        )

    return _DeepEPBufferAllocation(
        buffer=deep_ep.Buffer(
            group=group, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=num_rdma_bytes
        ),
        num_nvl_bytes=num_nvl_bytes,
        num_rdma_bytes=num_rdma_bytes,
    )


def _event_is_waitable(event) -> bool:
    if event is None:
        return False
    if hasattr(event, "current_stream_wait") and getattr(event, "event", None) is None:
        return False
    return True


def _event_current_stream_wait(event) -> None:
    if event is None:
        return
    if hasattr(event, "current_stream_wait"):
        if getattr(event, "event", None) is None:
            return
        event.current_stream_wait()
    else:
        torch.cuda.current_stream().wait_event(event)


def _record_current_stream_event_if_unwaitable(event, tensor: torch.Tensor):
    if not _event_is_waitable(event) and torch.cuda.is_available() and tensor.is_cuda:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(tensor.device))
    return event


def _use_moe_permute_fusion() -> bool:
    return os.environ.get("MEGATRON_LITE_MOE_PERMUTE_FUSION", "0") == "1"


def _tensor_hidden_bytes(x: torch.Tensor) -> int:
    return x.size(1) * max(x.element_size(), 2)


class _DeepEPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        buffer,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_scores: torch.Tensor,
        num_experts: int,
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ):
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
        (recv_hidden, recv_indices, recv_probs, recv_per_expert, handle, after_event) = (
            buffer.dispatch(
                hidden_states.contiguous(),
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
        )
        if async_finish:
            after_event.current_stream_wait()

        ctx.buffer = buffer
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        recv_per_expert_tensor = torch.tensor(
            recv_per_expert, dtype=torch.int64, device=recv_hidden.device
        )
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
        grad_scores = None if grad_recv_probs is None else grad_recv_probs.float()
        grad_hidden, grad_topk_scores, after_event = ctx.buffer.combine(
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
        buffer,
        rank_grouped: torch.Tensor,
        handle,
        async_finish: bool,
        allocate_on_comm_stream: bool,
    ):
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
        ctx.buffer = buffer
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
        grad_rank_grouped, _, _, _, _, after_event = ctx.buffer.dispatch(
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
        moe_permute_fusion: bool | None = None,
    ):
        self.ps = ps
        self.num_experts = num_experts
        self.ep_size = ps.ep_size
        self.num_local_experts = ensure_divisible(num_experts, ps.ep_size)
        self.moe_permute_fusion = (
            _use_moe_permute_fusion() if moe_permute_fusion is None else bool(moe_permute_fusion)
        )

        self.use_deepep = use_deepep and deep_ep is not None and ps.ep_size > 1
        if self.use_deepep:
            assert ps.tp_ep_group is not None
            allocation = _build_deepep_buffer(ps.tp_ep_group, hidden_size)
            self.buffer = allocation.buffer
            self.deepep_buffer_resident_bytes = allocation.resident_bytes

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

    def dispatch(
        self, hidden_states: torch.Tensor, topk_scores: torch.Tensor, topk_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.ep_size <= 1:
            return self._dispatch_local(hidden_states, topk_scores, topk_indices)
        if self.use_deepep:
            return self._dispatch_deepep(hidden_states, topk_scores, topk_indices)
        dispatched, tpe, sorted_scores = self._dispatch_alltoall(
            hidden_states, topk_scores, topk_indices
        )
        return dispatched, tpe, sorted_scores

    def combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        if self.ep_size <= 1:
            return self._combine_local(expert_output)
        if self.use_deepep:
            return self._combine_deepep(expert_output)
        return self._combine_alltoall(expert_output)

    def submit_deepep_combine(
        self,
        expert_output: torch.Tensor,
        *,
        allocate_on_comm_stream: bool = False,
        async_finish: bool = True,
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
            if async_finish and EventHandle is not None and EventOverlap is not None
            else None
        )
        combined = self.buffer.combine(
            rank_grouped,
            self._handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        event = None
        if isinstance(combined, tuple):
            if len(combined) >= 3:
                event = combined[2]
            combined = combined[0]
        event = _record_current_stream_event_if_unwaitable(event, rank_grouped)
        return {
            "combined": combined,
            "event": event,
            "rank_grouped": rank_grouped,
            "handle": self._handle,
        }

    def prepare_deepep_combine(self, expert_output: torch.Tensor):
        if not self.use_deepep:
            raise RuntimeError("prepare_deepep_combine requires DeepEP combine.")
        rank_grouped = unpermute(
            expert_output,
            self._row_id_map,
            restore_shape=self._restore_shape,
            fused=self.moe_permute_fusion,
        )
        return rank_grouped, self._handle

    def submit_deepep_combine_prepared(
        self,
        rank_grouped: torch.Tensor,
        handle,
        *,
        allocate_on_comm_stream: bool = False,
        async_finish: bool = True,
    ):
        if not self.use_deepep:
            raise RuntimeError("submit_deepep_combine_prepared requires DeepEP combine.")
        previous_event = (
            EventOverlap(EventHandle())
            if async_finish and EventHandle is not None and EventOverlap is not None
            else None
        )
        combined = self.buffer.combine(
            rank_grouped,
            handle,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        event = None
        if isinstance(combined, tuple):
            if len(combined) >= 3:
                event = combined[2]
            combined = combined[0]
        event = _record_current_stream_event_if_unwaitable(event, rank_grouped)
        return {
            "combined": combined,
            "event": event,
            "rank_grouped": rank_grouped,
            "handle": handle,
        }

    def clear_deepep_combine_state(self):
        self._row_id_map = None
        self._restore_shape = None
        self._handle = None
        self._local_tpe_list = None

    def submit_deepep_combine_backward(
        self, grad_output: torch.Tensor, handle, *, allocate_on_comm_stream: bool = False
    ):
        if not self.use_deepep:
            raise RuntimeError("submit_deepep_combine_backward requires DeepEP.")
        previous_event = (
            EventOverlap(EventHandle())
            if EventHandle is not None and EventOverlap is not None
            else None
        )
        grad_rank_grouped, _, _, _, _, event = self.buffer.dispatch(
            grad_output.contiguous(),
            handle=handle,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        return {"grad_rank_grouped": grad_rank_grouped, "event": event}

    def finish_deepep_combine_backward(self, state):
        if not self.use_deepep:
            raise RuntimeError("finish_deepep_combine_backward requires DeepEP.")
        _event_current_stream_wait(state.get("event"))
        return state["grad_rank_grouped"]

    def submit_deepep_dispatch_backward(
        self,
        grad_recv_hidden: torch.Tensor,
        grad_recv_probs: torch.Tensor | None,
        handle,
        *,
        allocate_on_comm_stream: bool = False,
    ):
        if not self.use_deepep:
            raise RuntimeError("submit_deepep_dispatch_backward requires DeepEP.")
        previous_event = (
            EventOverlap(EventHandle())
            if EventHandle is not None and EventOverlap is not None
            else None
        )
        grad_scores = None if grad_recv_probs is None else grad_recv_probs.float()
        grad_hidden, grad_topk_scores, event = self.buffer.combine(
            grad_recv_hidden.contiguous(),
            handle,
            topk_weights=grad_scores,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        return {"grad_hidden": grad_hidden, "grad_topk_scores": grad_topk_scores, "event": event}

    def finish_deepep_dispatch_backward(self, state):
        if not self.use_deepep:
            raise RuntimeError("finish_deepep_dispatch_backward requires DeepEP.")
        _event_current_stream_wait(state.get("event"))
        return state["grad_hidden"], state["grad_topk_scores"]

    def finish_deepep_combine(self, state):
        if not self.use_deepep:
            raise RuntimeError("finish_deepep_combine requires DeepEP combine.")
        _event_current_stream_wait(state.get("event"))
        combined = state["combined"]
        state.clear()
        self.clear_deepep_combine_state()
        return combined

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
        self,
        hidden_states,
        topk_scores,
        topk_indices,
        *,
        allocate_on_comm_stream: bool = False,
        async_finish: bool = True,
    ):
        if not self.use_deepep:
            raise RuntimeError("submit_deepep_dispatch requires DeepEP dispatch.")
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
        ) = self.buffer.get_dispatch_layout(
            topk_indices,
            num_experts=self.num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        hidden_states_contig = hidden_states.contiguous()
        recv_hidden, recv_indices, recv_probs, recv_per_expert, handle, event = (
            self.buffer.dispatch(
                hidden_states_contig,
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
        )
        event = _record_current_stream_event_if_unwaitable(event, hidden_states_contig)
        return {
            "_dispatch_inputs": (
                hidden_states_contig,
                topk_indices,
                topk_scores,
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
            ),
            "recv_hidden": recv_hidden,
            "recv_indices": recv_indices,
            "recv_probs": recv_probs,
            "recv_per_expert": recv_per_expert,
            "handle": handle,
            "event": event,
        }

    def _resolve_deepep_recv_per_expert(self, state):
        return state["recv_per_expert"]

    def finish_deepep_dispatch(self, state, *, materialize_local_tpe: bool = True):
        if not self.use_deepep:
            raise RuntimeError("finish_deepep_dispatch requires DeepEP dispatch.")
        self._handle = state["handle"]
        self._deepep_event = state["event"]
        self.wait_dispatch_event()
        recv_per_expert = self._resolve_deepep_recv_per_expert(state)
        return self._finish_deepep_dispatch(
            state["recv_hidden"],
            state["recv_indices"],
            state["recv_probs"],
            recv_per_expert,
            materialize_local_tpe=materialize_local_tpe,
        )

    def _finish_deepep_dispatch(
        self,
        recv_hidden: torch.Tensor,
        recv_indices: torch.Tensor,
        recv_probs: torch.Tensor,
        recv_per_expert,
        *,
        materialize_local_tpe: bool = True,
    ):
        dispatched, local_tpe, permuted_probs, metadata = self._finish_deepep_dispatch_external(
            recv_hidden,
            recv_indices,
            recv_probs,
            recv_per_expert,
            materialize_local_tpe=materialize_local_tpe,
        )
        self._local_tpe_list = metadata["local_tpe_list"]
        self._row_id_map = metadata["row_id_map"]
        self._restore_shape = metadata["restore_shape"]
        return dispatched, local_tpe, permuted_probs

    def _finish_deepep_dispatch_external(
        self,
        recv_hidden: torch.Tensor,
        recv_indices: torch.Tensor,
        recv_probs: torch.Tensor,
        recv_per_expert,
        *,
        force_manual_map: bool = False,
        force_direct_permute: bool = False,
        materialize_local_tpe: bool = True,
    ):
        if isinstance(recv_per_expert, torch.Tensor):
            recv_per_expert = [int(x) for x in recv_per_expert.detach().cpu().tolist()]
        local_tpe_list = [int(x) for x in recv_per_expert[: self.num_local_experts]]
        local_tpe = (
            torch.tensor(local_tpe_list, dtype=torch.int64, device=recv_hidden.device)
            if materialize_local_tpe
            else None
        )
        rows = recv_hidden.size(0)
        recv_indices = recv_indices.to(torch.long)
        if recv_indices.dim() != 2:
            raise RuntimeError("DeepEP dispatch indices must have shape [recv_rows, topk]")
        valid = recv_indices >= 0
        num_out = sum(int(x) for x in recv_per_expert)
        use_direct_permute = force_direct_permute
        need_manual_map = force_manual_map or use_direct_permute
        valid_coords = valid.nonzero(as_tuple=False)
        valid_row_ids = valid_coords[:, 0]
        valid_topk_slots = valid_coords[:, 1]
        valid_expert_ids = recv_indices[valid_row_ids, valid_topk_slots]
        valid_prob_flat_indices = valid_row_ids * recv_indices.size(1) + valid_topk_slots
        manual_order = None
        manual_prob_flat_indices = None
        if need_manual_map:
            manual_order = torch.argsort(valid_expert_ids * rows + valid_row_ids, stable=True)
            manual_row_id_map = valid_row_ids.index_select(0, manual_order)
            manual_prob_flat_indices = valid_prob_flat_indices.index_select(0, manual_order)
        else:
            manual_row_id_map = None
        if use_direct_permute:
            assert manual_order is not None
            sorted_indices = manual_row_id_map
            assert sorted_indices is not None
            dispatched = recv_hidden.index_select(0, sorted_indices)
            permuted_probs = recv_probs.reshape(-1).index_select(0, manual_prob_flat_indices)
        else:
            routing_map = torch.zeros(
                rows, self.num_local_experts, dtype=torch.bool, device=recv_hidden.device
            )
            probs_2d = torch.zeros(
                rows, self.num_local_experts, dtype=recv_probs.dtype, device=recv_hidden.device
            )
            routing_map[valid_row_ids, valid_expert_ids] = True
            probs_2d.index_put_(
                (valid_row_ids, valid_expert_ids),
                recv_probs.reshape(-1).index_select(0, valid_prob_flat_indices),
                accumulate=True,
            )
            dispatched, permuted_probs, sorted_indices = permute(
                recv_hidden,
                routing_map,
                probs=probs_2d,
                num_out_tokens=num_out,
                fused=self.moe_permute_fusion,
            )[:3]
        restore_shape = recv_hidden.shape
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
                f"local_tpe_sum={sum(local_tpe_list)}",
                flush=True,
            )
        if sum(local_tpe_list) != int(dispatched.shape[0]):
            ep_rank = dist.get_rank(group=self.ps.ep_group)
            raise RuntimeError(
                "DeepEP dispatch metadata mismatch: "
                f"ep_rank={ep_rank} dispatched_tokens={int(dispatched.shape[0])} "
                f"local_tpe={local_tpe_list} recv_per_expert_len={len(recv_per_expert)}"
            )
        metadata = {
            "row_id_map": sorted_indices,
            "restore_shape": restore_shape,
            "manual_row_id_map": manual_row_id_map,
            "manual_prob_flat_indices": manual_prob_flat_indices,
            "local_tpe_list": local_tpe_list,
        }
        return dispatched, local_tpe, permuted_probs, metadata

    def finish_deepep_dispatch_external_with_options(
        self,
        state,
        *,
        force_manual_map: bool = False,
        force_direct_permute: bool = False,
        materialize_local_tpe: bool = True,
    ):
        if not self.use_deepep:
            raise RuntimeError("finish_deepep_dispatch_external requires DeepEP dispatch.")
        _event_current_stream_wait(state.get("event"))
        recv_per_expert = self._resolve_deepep_recv_per_expert(state)
        dispatched, local_tpe, permuted_probs, metadata = self._finish_deepep_dispatch_external(
            state["recv_hidden"],
            state["recv_indices"],
            state["recv_probs"],
            recv_per_expert,
            force_manual_map=force_manual_map,
            force_direct_permute=force_direct_permute,
            materialize_local_tpe=materialize_local_tpe,
        )
        metadata["handle"] = state["handle"]
        return dispatched, local_tpe, permuted_probs, metadata

    def _dispatch_deepep(self, hidden_states, topk_scores, topk_indices):
        if torch.is_grad_enabled():
            recv_hidden, recv_indices, recv_probs, recv_per_expert, handle = _DeepEPDispatch.apply(
                self.buffer,
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
            _event_current_stream_wait(self._deepep_event)
            self._deepep_event = None

    def _combine_deepep(self, expert_output):
        rank_grouped = unpermute(
            expert_output,
            self._row_id_map,
            restore_shape=self._restore_shape,
            fused=self.moe_permute_fusion,
        )
        if torch.is_grad_enabled():
            combined = _DeepEPCombine.apply(self.buffer, rank_grouped, self._handle, False, False)
        else:
            combined = self.buffer.combine(rank_grouped, self._handle)
        if isinstance(combined, tuple):
            combined = combined[0]
        self.clear_deepep_combine_state()
        return combined


__all__ = ["TokenDispatcher"]
