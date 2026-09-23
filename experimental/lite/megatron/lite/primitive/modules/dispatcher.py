# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Token dispatcher: AllToAll, DeepEP and HybridEP dispatch/combine."""

from __future__ import annotations

import os

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

try:
    from deep_ep import HybridEPBuffer  # pyright: ignore[reportMissingImports]
except ImportError:
    HybridEPBuffer = None  # type: ignore

DISPATCH_BACKENDS = ("alltoall", "deepep", "hybridep")
_HYBRIDEP_INT16_EXPERT_LIMIT = 1 << 15
_hybridep_buffers: dict[tuple, object] = {}


def _hidden_bytes(hidden_size: int) -> int:
    return hidden_size * 2


def _build_deepep_buffer(group: dist.ProcessGroup, hidden_size: int):
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

    return deep_ep.Buffer(group=group, num_nvl_bytes=num_nvl_bytes, num_rdma_bytes=num_rdma_bytes)


def _get_hybridep_buffer(group: dist.ProcessGroup, hidden_size: int, num_local_experts: int, max_tokens: int):
    if HybridEPBuffer is None:
        raise RuntimeError("HybridEP buffer requested but deep_ep.HybridEPBuffer is not installed.")
    key = (id(group), hidden_size, num_local_experts)
    buf = _hybridep_buffers.get(key)
    if buf is None:
        buf = HybridEPBuffer(
            group=group,
            hidden_dim=hidden_size,
            max_num_of_tokens_per_rank=max_tokens,
            num_local_experts=num_local_experts,
        )
        _hybridep_buffers[key] = buf
    return buf


def reset_hybridep_buffers() -> None:
    """Drop cached HybridEP buffers (call after destroying the process groups they were built on)."""
    _hybridep_buffers.clear()


def _routing_tensors(
    topk_scores: torch.Tensor, topk_indices: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    t = topk_indices.size(0)
    routing_map = torch.zeros(t, num_experts, dtype=torch.bool, device=topk_indices.device)
    routing_map.scatter_(1, topk_indices, True)
    probs_2d = torch.zeros(t, num_experts, dtype=topk_scores.dtype, device=topk_indices.device)
    probs_2d.scatter_add_(1, topk_indices, topk_scores)
    return routing_map, probs_2d


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
                topk_weights=topk_scores.float(),
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


class _HybridEPDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        buffer,
        hidden_states: torch.Tensor,
        routing_map: torch.Tensor | None,
        topk_idx: torch.Tensor | None,
        probs: torch.Tensor,
        num_local_experts: int,
        num_experts: int,
        num_tokens_per_rank: int,
    ):
        if topk_idx is not None:
            routing_kwargs = {"topk_idx": topk_idx, "num_of_experts": num_experts}
        else:
            routing_kwargs = {"routing_map": routing_map}
        dispatched, dispatched_probs, _, tokens_per_expert, handle = buffer.dispatch_with_permute(
            hidden=hidden_states,
            probs=probs,
            scaling_factor=None,
            num_of_experts_per_rank=num_local_experts,
            num_of_tokens_per_rank=num_tokens_per_rank,
            **routing_kwargs,
        )
        ctx.buffer = buffer
        ctx.handle = handle
        return dispatched, dispatched_probs, tokens_per_expert, handle

    @staticmethod
    def backward(ctx, grad_hidden, grad_probs, grad_tokens_per_expert, grad_handle):
        del grad_tokens_per_expert, grad_handle
        combined_hidden, combined_probs = ctx.buffer.combine_with_unpermute(
            hidden=grad_hidden, probs=grad_probs, handle=ctx.handle
        )
        return None, combined_hidden, None, None, combined_probs, None, None, None


class _HybridEPCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, buffer, expert_output: torch.Tensor, handle, num_permuted_tokens):
        combined, _ = buffer.combine_with_unpermute(hidden=expert_output, handle=handle)
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.num_permuted_tokens = num_permuted_tokens
        return combined

    @staticmethod
    def backward(ctx, grad_output):
        dispatched, _, _, _, _ = ctx.buffer.dispatch_with_permute(
            hidden=grad_output,
            scaling_factor=None,
            handle=ctx.handle,
            num_permuted_tokens=ctx.num_permuted_tokens,
        )
        return None, dispatched, None, None


class TokenDispatcher:

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        ps: ParallelState,
        *,
        use_deepep: bool = True,
        dispatch_backend: str | None = None,
        moe_permute_fusion: bool | None = None,
        hybridep_dense_routing: bool = True,
        hybridep_max_tokens: int | None = None,
    ):
        self.ps = ps
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.ep_size = ps.ep_size
        self.num_local_experts = ensure_divisible(num_experts, ps.ep_size)
        self.moe_permute_fusion = (
            _use_moe_permute_fusion() if moe_permute_fusion is None else bool(moe_permute_fusion)
        )

        if dispatch_backend is None:
            dispatch_backend = "deepep" if use_deepep else "alltoall"
        if dispatch_backend not in DISPATCH_BACKENDS:
            raise ValueError(f"dispatch_backend must be one of {DISPATCH_BACKENDS}, got {dispatch_backend!r}")
        if dispatch_backend == "hybridep" and HybridEPBuffer is None:
            raise RuntimeError("dispatch_backend='hybridep' requires deep_ep.HybridEPBuffer (DeepEP hybrid-ep branch).")
        self.dispatch_backend = dispatch_backend
        self.use_deepep = dispatch_backend == "deepep" and deep_ep is not None and ps.ep_size > 1
        self.use_hybridep = dispatch_backend == "hybridep" and ps.ep_size > 1
        if self.use_deepep:
            assert ps.tp_ep_group is not None
            self.buffer = _build_deepep_buffer(ps.tp_ep_group, hidden_size)
        # HybridEP dense routing needs int16 expert ids; duplicate experts per token are malformed there.
        self.hybridep_dense_routing = hybridep_dense_routing and num_experts <= _HYBRIDEP_INT16_EXPERT_LIMIT
        self.hybridep_max_tokens = hybridep_max_tokens
        self._hybridep_buffer = None
        self._num_permuted_tokens = None

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
        if self.use_hybridep:
            return self._dispatch_hybridep(hidden_states, topk_scores, topk_indices)
        dispatched, tpe, sorted_scores = self._dispatch_alltoall(
            hidden_states, topk_scores, topk_indices
        )
        return dispatched, tpe, sorted_scores

    def combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        if self.ep_size <= 1:
            return self._combine_local(expert_output)
        if self.use_deepep:
            return self._combine_deepep(expert_output)
        if self.use_hybridep:
            return self._combine_hybridep(expert_output)
        return self._combine_alltoall(expert_output)

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
        combined = self.buffer.combine(
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
        routing_map, probs_2d = _routing_tensors(topk_scores, topk_indices, self.num_experts)
        num_out = int(routing_map.sum().item())

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
        e = self.num_experts
        routing_map, probs_2d = _routing_tensors(topk_scores, topk_indices, e)
        # routing_map.sum() (not t*topk): hash routing may repeat an expert within a token's top-k.
        num_out = int(routing_map.sum().item())

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

    def _hybridep_group(self) -> dist.ProcessGroup:
        group = self.ps.tp_ep_group if self.ps.tp_ep_group is not None else self.ps.ep_group
        assert group is not None
        return group

    def _dispatch_hybridep(self, hidden_states, topk_scores, topk_indices):
        group = self._hybridep_group()
        t = hidden_states.size(0)
        if self._hybridep_buffer is None:
            self._hybridep_buffer = _get_hybridep_buffer(
                group, self.hidden_size, self.num_local_experts, self.hybridep_max_tokens or t
            )
        routing_map, probs_2d = _routing_tensors(topk_scores, topk_indices, self.num_experts)
        # HybridEP only supports fp32 probs (same cast as mcore _HybridEPManager.dispatch).
        probs_2d = probs_2d.float()
        topk_idx = topk_indices.to(torch.int16).contiguous() if self.hybridep_dense_routing else None
        # Ranks in the group may hold different token counts; pass the group-wide max as slot count.
        num_tokens_per_rank = torch.tensor([t], device=hidden_states.device, dtype=torch.long)
        dist.all_reduce(num_tokens_per_rank, op=dist.ReduceOp.MAX, group=group)
        dispatched, dispatched_probs, tokens_per_expert, handle = _HybridEPDispatch.apply(
            self._hybridep_buffer,
            hidden_states,
            None if topk_idx is not None else routing_map,
            topk_idx,
            probs_2d,
            self.num_local_experts,
            self.num_experts,
            int(num_tokens_per_rank.item()),
        )
        tokens_per_expert = tokens_per_expert.to(torch.int64)
        self._handle = handle
        self._num_permuted_tokens = tokens_per_expert.sum()
        self._local_tpe_list = None
        return dispatched, tokens_per_expert, dispatched_probs

    def _combine_hybridep(self, expert_output):
        combined = _HybridEPCombine.apply(
            self._hybridep_buffer, expert_output, self._handle, self._num_permuted_tokens
        )
        self._handle = None
        self._num_permuted_tokens = None
        self._local_tpe_list = None
        return combined

    def submit_deepep_dispatch(
        self, hidden_states, topk_scores, topk_indices, *, allocate_on_comm_stream: bool = False
    ):
        if not self.use_deepep:
            raise RuntimeError("submit_deepep_dispatch requires DeepEP dispatch.")
        previous_event = (
            EventOverlap(EventHandle())
            if EventHandle is not None and EventOverlap is not None
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
            async_finish=True,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        topk_scores = topk_scores.float()
        recv_hidden, recv_indices, recv_probs, recv_per_expert, handle, event = (
            self.buffer.dispatch(
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
        local_tpe = torch.tensor(
            recv_per_expert[: self.num_local_experts], dtype=torch.int64, device=recv_hidden.device
        )
        self._local_tpe_list = [int(x) for x in recv_per_expert[: self.num_local_experts]]
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
        probs_2d.index_put_((row_ids, expert_ids), recv_probs[valid], accumulate=True)
        num_out = sum(int(x) for x in recv_per_expert)
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
            combined = _DeepEPCombine.apply(self.buffer, rank_grouped, self._handle, False, False)
        else:
            combined = self.buffer.combine(rank_grouped, self._handle)
        if isinstance(combined, tuple):
            combined = combined[0]
        self._row_id_map = None
        self._restore_shape = None
        self._handle = None
        self._local_tpe_list = None
        return combined


__all__ = ["DISPATCH_BACKENDS", "TokenDispatcher", "reset_hybridep_buffers"]
