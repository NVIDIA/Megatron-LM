"""HybridEP route-slot transport for the DeepSeek-V4 vLLM implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

try:
    import deep_ep
except ImportError:  # pragma: no cover - depends on the runtime image
    deep_ep = None


_buffer = None
_buffer_capacity = 0
_buffer_signature = None
_PAD_MULTIPLE = 128


@dataclass
class HybridEPRouteState:
    buffer: object
    handle: object
    source_route_count: int
    source_indices: torch.Tensor
    source_weights: torch.Tensor
    source_output_index: torch.Tensor
    source_all_routes_valid: bool


@dataclass
class HybridEPDispatchResult:
    hidden: torch.Tensor
    tokens_per_expert: torch.Tensor
    tokens_per_expert_list: list[int]
    probs: torch.Tensor
    state: HybridEPRouteState


def require_available() -> None:
    if deep_ep is None or not hasattr(deep_ep, "HybridEPBuffer"):
        raise RuntimeError(
            "hybridep requires a merged DeepEP runtime exporting HybridEPBuffer"
        )


def _get_buffer(
    group: dist.ProcessGroup,
    hidden_size: int,
    num_local_experts: int,
    required_route_capacity: int,
    hybridep_max_tokens_per_rank: int,
):
    require_available()
    if (
        not isinstance(hybridep_max_tokens_per_rank, int)
        or isinstance(hybridep_max_tokens_per_rank, bool)
        or hybridep_max_tokens_per_rank <= 0
    ):
        raise ValueError("HybridEP max_tokens_per_rank must be a positive integer")
    if hybridep_max_tokens_per_rank < required_route_capacity:
        raise RuntimeError(
            f"HybridEP route capacity {hybridep_max_tokens_per_rank} is below required "
            f"{required_route_capacity}"
        )
    # Match Megatron Core: initialize from the current input shape and only
    # grow when a later forward needs more capacity. The configured value is
    # an upper bound, not an eager allocation size.
    capacity = required_route_capacity

    signature = (group, hidden_size, num_local_experts)
    global _buffer, _buffer_capacity, _buffer_signature
    rebuild = (
        _buffer is None
        or getattr(_buffer, "runtime", None) is None
        or _buffer_signature != signature
        or _buffer_capacity < capacity
    )
    if rebuild:
        _buffer = deep_ep.HybridEPBuffer(
            group=group,
            hidden_dim=hidden_size,
            max_num_of_tokens_per_rank=capacity,
            num_local_experts=num_local_experts,
            use_fp8=False,
        )
        _buffer_capacity = capacity
        _buffer_signature = signature
    if getattr(_buffer, "runtime", None) is None:
        raise RuntimeError("HybridEPBuffer was created without an active runtime")
    return _buffer


class _DispatchRoutes(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        buffer,
        route_hidden: torch.Tensor,
        route_indices: torch.Tensor,
        num_experts: int,
        num_local_experts: int,
    ):
        dispatched, _, _, tokens_per_expert, handle = (
            buffer.dispatch_with_permute(
                hidden=route_hidden.contiguous(),
                topk_idx=route_indices.reshape(-1, 1).contiguous(),
                topk_weights=None,
                num_of_experts=num_experts,
                scaling_factor=None,
                num_of_experts_per_rank=num_local_experts,
                pad_multiple=_PAD_MULTIPLE,
                num_permuted_tokens=None,
                non_blocking=False,
            )
        )
        ctx.buffer = buffer
        ctx.handle = handle
        return dispatched, tokens_per_expert, handle

    @staticmethod
    def backward(ctx, grad_dispatched, _grad_counts, _grad_handle):
        grad_routes, _ = ctx.buffer.combine_with_unpermute(
            hidden=grad_dispatched.contiguous(),
            handle=ctx.handle,
            pad_multiple=_PAD_MULTIPLE,
        )
        return None, grad_routes, None, None, None


class _CombineRoutes(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        buffer,
        expert_output: torch.Tensor,
        handle,
        source_route_count: int,
    ):
        source_routes, _ = buffer.combine_with_unpermute(
            hidden=expert_output.contiguous(),
            handle=handle,
            pad_multiple=_PAD_MULTIPLE,
        )
        if source_routes.shape[0] < source_route_count:
            raise RuntimeError(
                "HybridEP combine returned fewer rows than the source route count"
            )
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.num_permuted_tokens = expert_output.shape[0]
        ctx.native_source_rows = source_routes.shape[0]
        ctx.source_route_count = source_route_count
        return source_routes.narrow(0, 0, source_route_count)

    @staticmethod
    def backward(ctx, grad_source_routes):
        native_grad = torch.zeros(
            (ctx.native_source_rows, grad_source_routes.shape[1]),
            device=grad_source_routes.device,
            dtype=grad_source_routes.dtype,
        )
        native_grad.narrow(0, 0, ctx.source_route_count).copy_(grad_source_routes)
        grad_expert, _, _, _, _ = ctx.buffer.dispatch_with_permute(
            hidden=native_grad.contiguous(),
            scaling_factor=None,
            num_permuted_tokens=ctx.num_permuted_tokens,
            pad_multiple=_PAD_MULTIPLE,
            handle=ctx.handle,
        )
        return None, grad_expert, None, None


def dispatch_routes(
    hidden_states: torch.Tensor,
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    num_experts: int,
    num_local_experts: int,
    group: dist.ProcessGroup,
    hybridep_max_tokens_per_rank: int,
) -> HybridEPDispatchResult:
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise TypeError("hybridep requires BF16 [tokens, hidden]")
    if topk_indices.shape != topk_scores.shape:
        raise ValueError("hybridep top-k IDs and scores must have identical shapes")
    if topk_indices.ndim != 2 or topk_indices.shape[0] != hidden_states.shape[0]:
        raise ValueError("hybridep routes must describe every input token")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("hybridep route IDs must be integer")
    if topk_scores.dtype != torch.float32:
        raise TypeError("hybridep route weights must be FP32")

    valid = (topk_indices >= 0) & (topk_indices < num_experts)
    if hidden_states.is_cuda:
        torch._assert_async(
            torch.all(valid),
            "CUDA HybridEP requires every DS4 top-k route to be valid",
        )
        route_hidden = hidden_states.repeat_interleave(topk_indices.shape[1], dim=0)
        route_indices = topk_indices.reshape(-1)
        source_route_count = topk_indices.numel()
        source_output_index = torch.arange(
            source_route_count, device=topk_indices.device, dtype=torch.long
        ).reshape_as(topk_indices)
        source_all_routes_valid = True
    else:
        # CPU compaction is retained only as a unit-test/reference path.
        positions = torch.nonzero(valid, as_tuple=False)
        token_rows = positions[:, 0]
        topk_slots = positions[:, 1]
        route_hidden = hidden_states.index_select(0, token_rows).contiguous()
        route_indices = topk_indices[token_rows, topk_slots].reshape(-1)
        source_route_count = positions.shape[0]
        source_output_index = torch.full_like(topk_indices, -1, dtype=torch.long)
        source_output_index[token_rows, topk_slots] = torch.arange(
            source_route_count, device=topk_indices.device, dtype=torch.long
        )
        source_all_routes_valid = source_route_count == topk_indices.numel()
    if dist.is_initialized():
        # HybridEP requires every rank to enter with the same route capacity.
        # Its ABI accepts that capacity as a host integer, so exchange only
        # Python shape metadata here; never read a CUDA count tensor.
        route_counts = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(route_counts, source_route_count, group=group)
        padded_route_count = max(int(count) for count in route_counts)
    else:
        padded_route_count = source_route_count
    if source_route_count < padded_route_count:
        padding = padded_route_count - source_route_count
        route_hidden = torch.cat(
            (
                route_hidden,
                torch.zeros(
                    padding,
                    hidden_states.shape[1],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ),
            ),
            dim=0,
        )
        route_indices = torch.cat(
            (
                route_indices,
                torch.full(
                    (padding,),
                    -1,
                    device=route_indices.device,
                    dtype=route_indices.dtype,
                ),
            ),
            dim=0,
        )
    buffer = _get_buffer(
        group,
        hidden_states.shape[1],
        num_local_experts,
        padded_route_count,
        hybridep_max_tokens_per_rank,
    )
    (
        expert_hidden,
        padded_counts,
        handle,
    ) = _DispatchRoutes.apply(
        buffer,
        route_hidden,
        route_indices,
        num_experts,
        num_local_experts,
    )
    if not isinstance(padded_counts, torch.Tensor):
        raise TypeError("HybridEP tokens_per_expert must be a tensor")
    if padded_counts.device.type != "cpu":
        raise RuntimeError("HybridEP expert counts must remain CPU metadata")
    counts_list = [int(value) for value in padded_counts.tolist()]
    if sum(counts_list) != expert_hidden.shape[0]:
        raise RuntimeError(
            "HybridEP padded expert counts do not cover dispatched rows"
        )
    return HybridEPDispatchResult(
        hidden=expert_hidden,
        tokens_per_expert=padded_counts,
        tokens_per_expert_list=counts_list,
        probs=torch.zeros(
            expert_hidden.shape[0],
            device=expert_hidden.device,
            dtype=torch.float32,
        ),
        state=HybridEPRouteState(
            buffer=buffer,
            handle=handle,
            source_route_count=source_route_count,
            source_indices=topk_indices,
            source_weights=topk_scores,
            source_output_index=source_output_index,
            source_all_routes_valid=source_all_routes_valid,
        ),
    )


def combine_routes(
    expert_output: torch.Tensor, state: HybridEPRouteState
) -> torch.Tensor:
    return _CombineRoutes.apply(
        state.buffer,
        expert_output,
        state.handle,
        state.source_route_count,
    )
