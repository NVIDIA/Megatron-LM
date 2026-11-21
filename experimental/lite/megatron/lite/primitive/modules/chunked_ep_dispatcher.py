# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""ChunkedEP transport: asynchronous extensions over the shared DeepEP setup."""

from __future__ import annotations

import torch
import torch.distributed as dist

from megatron.lite.primitive.modules.dispatcher import (
    EventHandle,
    EventOverlap,
    TokenDispatcher as _BaseDispatcher,
)
from megatron.lite.primitive.utils.moe import permute, unpermute


def _previous_event():
    if EventHandle is not None and EventOverlap is not None:
        return EventOverlap(EventHandle())
    return None


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


def _require_completion_event(event):
    if not _event_is_waitable(event):
        raise RuntimeError("DeepEP async transport requires a completion event")


class ChunkedDispatcher(_BaseDispatcher):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.use_deepep:
            raise RuntimeError("ChunkedDispatcher requires DeepEP and EP > 1")

    def prepare_deepep_combine(self, expert_output: torch.Tensor):
        rank_grouped = unpermute(
            expert_output,
            self._row_id_map,
            restore_shape=self._restore_shape,
            fused=self.moe_permute_fusion,
        )
        return rank_grouped, self._handle

    def submit_deepep_combine_prepared(
        self, rank_grouped: torch.Tensor, handle, *, allocate_on_comm_stream: bool = False
    ):
        previous_event = _previous_event()
        combined = self.buffer.combine(
            rank_grouped,
            handle,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        event = None
        if isinstance(combined, tuple):
            if len(combined) >= 3:
                event = combined[2]
            combined = combined[0]
        _require_completion_event(event)
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
        grad_rank_grouped, _, _, _, _, event = self.buffer.dispatch(
            grad_output.contiguous(),
            handle=handle,
            previous_event=_previous_event(),
            async_finish=True,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        _require_completion_event(event)
        return {"grad_rank_grouped": grad_rank_grouped, "event": event}

    def finish_deepep_combine_backward(self, state):
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
        grad_scores = None if grad_recv_probs is None else grad_recv_probs.float()
        grad_hidden, grad_topk_scores, event = self.buffer.combine(
            grad_recv_hidden.contiguous(),
            handle,
            topk_weights=grad_scores,
            previous_event=_previous_event(),
            async_finish=True,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        _require_completion_event(event)
        return {"grad_hidden": grad_hidden, "grad_topk_scores": grad_topk_scores, "event": event}

    def finish_deepep_dispatch_backward(self, state):
        _event_current_stream_wait(state.get("event"))
        return state["grad_hidden"], state["grad_topk_scores"]

    def finish_deepep_combine(self, state):
        _event_current_stream_wait(state.get("event"))
        combined = state["combined"]
        state.clear()
        self.clear_deepep_combine_state()
        return combined

    def submit_deepep_dispatch(
        self, hidden_states, topk_scores, topk_indices, *, allocate_on_comm_stream: bool = False
    ):
        topk_indices = topk_indices.contiguous()
        topk_scores = topk_scores.float().contiguous()
        previous_event = _previous_event()
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
                async_finish=True,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        )
        _require_completion_event(event)
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

    def finish_deepep_dispatch(self, state):
        self._handle = state["handle"]
        self._deepep_event = state["event"]
        self.wait_dispatch_event()
        dispatched, local_tpe, permuted_probs, metadata = self._finish_deepep_dispatch_external(
            state
        )
        self._local_tpe_list = metadata["local_tpe_list"]
        self._row_id_map = metadata["row_id_map"]
        self._restore_shape = metadata["restore_shape"]
        return dispatched, local_tpe, permuted_probs

    def _finish_deepep_dispatch_external(
        self, state, *, manual_backward: bool = False, output_allocation=None
    ):
        recv_hidden, recv_indices, recv_probs, recv_per_expert = (
            state[name] for name in ("recv_hidden", "recv_indices", "recv_probs", "recv_per_expert")
        )
        if isinstance(recv_per_expert, torch.Tensor):
            recv_per_expert = [int(x) for x in recv_per_expert.detach().cpu().tolist()]
        local_tpe_list = [int(x) for x in recv_per_expert[: self.num_local_experts]]
        rows = recv_hidden.size(0)
        recv_indices = recv_indices.to(torch.long)
        if recv_indices.dim() != 2:
            raise RuntimeError("DeepEP dispatch indices must have shape [recv_rows, topk]")
        valid = recv_indices >= 0
        num_out = sum(int(x) for x in recv_per_expert)
        valid_coords = valid.nonzero(as_tuple=False)
        valid_row_ids = valid_coords[:, 0]
        valid_topk_slots = valid_coords[:, 1]
        valid_expert_ids = recv_indices[valid_row_ids, valid_topk_slots]
        valid_prob_flat_indices = valid_row_ids * recv_indices.size(1) + valid_topk_slots
        manual_row_id_map = None
        manual_prob_flat_indices = None
        if manual_backward:
            manual_order = torch.argsort(valid_expert_ids * rows + valid_row_ids, stable=True)
            manual_row_id_map = valid_row_ids.index_select(0, manual_order)
            manual_prob_flat_indices = valid_prob_flat_indices.index_select(0, manual_order)
            sorted_indices = manual_row_id_map
            if output_allocation is None:
                dispatched = recv_hidden.index_select(0, sorted_indices)
            else:
                shape = (sorted_indices.numel(), recv_hidden.size(1))
                dispatched = output_allocation("fc1_input", shape)
                if (
                    dispatched.shape != shape
                    or dispatched.dtype != recv_hidden.dtype
                    or dispatched.device != recv_hidden.device
                    or not dispatched.is_contiguous()
                ):
                    raise RuntimeError("Invalid caller-owned dispatch output")
                with torch.no_grad():
                    torch.index_select(recv_hidden, 0, sorted_indices, out=dispatched)
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
        return dispatched, None, permuted_probs, metadata

    def finish_deepep_dispatch_for_backward(self, state, *, output_allocation=None):
        _event_current_stream_wait(state.get("event"))
        dispatched, local_tpe, permuted_probs, metadata = self._finish_deepep_dispatch_external(
            state, manual_backward=True, output_allocation=output_allocation
        )
        metadata["handle"] = state["handle"]
        return dispatched, local_tpe, permuted_probs, metadata

    def wait_dispatch_event(self):
        if self._deepep_event is not None:
            _event_current_stream_wait(self._deepep_event)
            self._deepep_event = None


__all__ = ["ChunkedDispatcher"]
