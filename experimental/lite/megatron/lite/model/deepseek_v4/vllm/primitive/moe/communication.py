"""Slime-style route alignment over normal DeepEP."""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from megatron.lite.primitive.modules.dispatcher import (
    TokenDispatcher,
    _DeepEPCombine,
    _DeepEPDispatch,
    deep_ep,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.moe import hybridep

_HOT_PATH_ASSERTS = os.getenv("MLITE_VLLM_HOT_PATH_ASSERTS") == "1"


@contextmanager
def _moe_nvtx_range(name: str):
    if os.environ.get("MLITE_STEP_NVTX") != "1" or not torch.cuda.is_available():
        yield
        return
    with torch.cuda.nvtx.range(name):
        yield


@triton.jit
def _route_hash_kernel(
    fingerprint_words,
    indices,
    weight_bits,
    hashes,
    FINGERPRINT_WORDS: tl.constexpr,
):
    row = tl.program_id(0) + tl.arange(0, 1)
    value = tl.full((1,), 1469598103934665603, tl.int64)
    for column in range(FINGERPRINT_WORDS):
        word = tl.load(
            fingerprint_words + row * FINGERPRINT_WORDS + column
        ).to(tl.int64)
        value = (value ^ (word & 0xFFFF)) * 1099511628211
    value = (value ^ tl.load(indices + row).to(tl.int64)) * 1099511628211
    bits = tl.load(weight_bits + row).to(tl.int64) & 0xFFFFFFFF
    tl.store(hashes + row, (value ^ bits) * 1099511628211)


@triton.jit
def _scatter_routes_forward_kernel(
    hidden_states,
    output_index,
    output,
    num_slots,
    hidden_size: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hidden_block = tl.program_id(0)
    first_slot = tl.program_id(1)
    num_slot_programs = tl.num_programs(1)
    hidden_offsets = hidden_block * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < hidden_size

    for slot_i32 in range(first_slot, num_slots, num_slot_programs):
        slot = slot_i32.to(tl.int64)
        destination_i32 = tl.load(output_index + slot)
        destination = destination_i32.to(tl.int64)
        token = slot // topk
        valid = destination_i32 >= 0
        values = tl.load(
            hidden_states + token * hidden_size + hidden_offsets,
            mask=hidden_mask & valid,
            other=0.0,
        )
        tl.store(
            output + destination * hidden_size + hidden_offsets,
            values,
            mask=hidden_mask & valid,
        )


@triton.jit
def _scatter_routes_backward_kernel(
    grad_output,
    output_index,
    grad_input,
    num_tokens,
    grad_output_rows,
    hidden_size: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hidden_block = tl.program_id(0)
    first_token = tl.program_id(1)
    num_token_programs = tl.num_programs(1)
    hidden_offsets = hidden_block * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < hidden_size

    for token_i32 in range(first_token, num_tokens, num_token_programs):
        token = token_i32.to(tl.int64)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for route_slot in range(topk):
            route_i32 = tl.load(output_index + token * topk + route_slot)
            valid = (route_i32 >= 0) & (route_i32 < grad_output_rows)
            safe_route = tl.maximum(route_i32, 0).to(tl.int64)
            route_grad = tl.load(
                grad_output + safe_route * hidden_size + hidden_offsets,
                mask=hidden_mask & valid,
                other=0.0,
            ).to(tl.float32)
            accumulator = (accumulator + route_grad).to(tl.bfloat16).to(tl.float32)
        tl.store(
            grad_input + token * hidden_size + hidden_offsets,
            accumulator,
            mask=hidden_mask,
        )


@triton.jit
def _ordered_route_grad_kernel(
    grad_output,
    topk_weights,
    output_index,
    grad_routes,
    num_slots,
    hidden_size: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    hidden_block = tl.program_id(0)
    first_slot = tl.program_id(1)
    num_slot_programs = tl.num_programs(1)
    hidden_offsets = hidden_block * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < hidden_size

    for slot_i32 in range(first_slot, num_slots, num_slot_programs):
        slot = slot_i32.to(tl.int64)
        route_i32 = tl.load(output_index + slot)
        route = route_i32.to(tl.int64)
        token = slot // topk
        valid = route_i32 >= 0
        weight = tl.load(topk_weights + slot).to(tl.float32)
        token_grad = tl.load(
            grad_output + token * hidden_size + hidden_offsets,
            mask=hidden_mask & valid,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            grad_routes + route * hidden_size + hidden_offsets,
            token_grad * weight,
            mask=hidden_mask & valid,
        )


@triton.jit
def _compact_route_positions_kernel(
    valid,
    prefix,
    positions,
    num_slots,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    flat_slot = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = flat_slot < num_slots
    is_valid = tl.load(valid + flat_slot, mask=in_bounds, other=0).to(tl.int1)
    compact_row = tl.load(prefix + flat_slot, mask=in_bounds, other=0).to(tl.int64) - 1
    write_mask = in_bounds & is_valid
    tl.store(
        positions + compact_row * 2,
        flat_slot // topk,
        mask=write_mask,
    )
    tl.store(
        positions + compact_row * 2 + 1,
        flat_slot % topk,
        mask=write_mask,
    )


def scatter_routes_forward(
    hidden_states: torch.Tensor,
    output_index: torch.Tensor,
    output: torch.Tensor,
) -> None:
    hidden_size = hidden_states.shape[1]
    block_d = 1024 if hidden_size >= 1024 else triton.next_power_of_2(hidden_size)
    num_slot_programs = min(output_index.numel(), 8192)
    grid = (triton.cdiv(hidden_size, block_d), num_slot_programs)
    _scatter_routes_forward_kernel[grid](
        hidden_states,
        output_index,
        output,
        output_index.numel(),
        hidden_size=hidden_size,
        topk=output_index.shape[1],
        BLOCK_D=block_d,
        num_warps=4,
    )


def scatter_routes_backward(
    grad_output: torch.Tensor,
    output_index: torch.Tensor,
    grad_input: torch.Tensor,
) -> None:
    hidden_size = grad_output.shape[1]
    block_d = 1024 if hidden_size >= 1024 else triton.next_power_of_2(hidden_size)
    num_token_programs = min(output_index.shape[0], 2048)
    grid = (triton.cdiv(hidden_size, block_d), num_token_programs)
    _scatter_routes_backward_kernel[grid](
        grad_output,
        output_index,
        grad_input,
        output_index.shape[0],
        grad_output.shape[0],
        hidden_size=hidden_size,
        topk=output_index.shape[1],
        BLOCK_D=block_d,
        num_warps=4,
    )


def ordered_route_grad(
    grad_output: torch.Tensor,
    topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    grad_routes: torch.Tensor,
) -> None:
    hidden_size = grad_output.shape[1]
    block_d = 1024 if hidden_size >= 1024 else triton.next_power_of_2(hidden_size)
    num_slot_programs = min(output_index.numel(), 8192)
    grid = (triton.cdiv(hidden_size, block_d), num_slot_programs)
    _ordered_route_grad_kernel[grid](
        grad_output,
        topk_weights,
        output_index,
        grad_routes,
        output_index.numel(),
        hidden_size=hidden_size,
        topk=output_index.shape[1],
        BLOCK_D=block_d,
        num_warps=4,
    )


def compact_route_positions(
    valid: torch.Tensor,
    num_routes: int,
) -> torch.Tensor:
    flat_valid = valid.reshape(-1).contiguous()
    prefix = torch.cumsum(flat_valid, dim=0, dtype=torch.int32)
    positions = torch.empty(
        (num_routes, 2),
        device=valid.device,
        dtype=torch.long,
    )
    if flat_valid.numel() and _HOT_PATH_ASSERTS:
        torch._assert_async(
            prefix[-1] == num_routes,
            "DeepEP compact route count differs from its metadata handle",
        )
    if num_routes:
        block_size = 256
        grid = (triton.cdiv(flat_valid.numel(), block_size),)
        _compact_route_positions_kernel[grid](
            flat_valid,
            prefix,
            positions,
            flat_valid.numel(),
            topk=valid.shape[1],
            BLOCK_SIZE=block_size,
        )
    return positions


_BACKWARD_CHUNK_ROWS = 1024


def _ordered_route_backward(
    *,
    route_values: torch.Tensor,
    topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    grad_output: torch.Tensor,
    grad_routes: torch.Tensor | None,
    grad_weights: torch.Tensor | None,
    static_mapping_valid: bool,
) -> None:
    aliases_values = (
        grad_routes is not None
        and grad_routes.untyped_storage().data_ptr()
        == route_values.untyped_storage().data_ptr()
    )
    num_routes = output_index.numel()
    topk = output_index.shape[1]
    if static_mapping_valid:
        flat_positions = torch.arange(
            num_routes, device=output_index.device, dtype=torch.long
        )
        token_rows = torch.div(flat_positions, topk, rounding_mode="floor")
        columns = flat_positions.remainder(topk)
    else:
        token_rows, columns = torch.nonzero(output_index >= 0, as_tuple=True)
    route_rows = output_index[token_rows, columns].long()

    def chunks():
        for start in range(0, route_rows.numel(), _BACKWARD_CHUNK_ROWS):
            end = min(start + _BACKWARD_CHUNK_ROWS, route_rows.numel())
            yield token_rows[start:end], columns[start:end], route_rows[start:end]

    if aliases_values and grad_weights is not None:
        for token_chunk, column_chunk, route_chunk in chunks():
            token_grads = grad_output.index_select(0, token_chunk)
            selected = route_values.index_select(0, route_chunk)
            grad_weights[token_chunk, column_chunk] = (
                token_grads.float() * selected.float()
            ).sum(dim=-1)
    if aliases_values and not static_mapping_valid:
        grad_routes.zero_()

    fused_route_grad = False
    if grad_routes is not None and grad_output.is_cuda:
        ordered_route_grad(
            grad_output.contiguous(),
            topk_weights.contiguous(),
            output_index.contiguous(),
            grad_routes,
        )
        fused_route_grad = True

    if (grad_routes is not None and not fused_route_grad) or (
        grad_weights is not None and not aliases_values
    ):
        for token_chunk, column_chunk, route_chunk in chunks():
            token_grads = grad_output.index_select(0, token_chunk)
            if grad_routes is not None and not fused_route_grad:
                weights = topk_weights[token_chunk, column_chunk]
                grad_routes.index_copy_(
                    0,
                    route_chunk,
                    (token_grads.float() * weights.float().unsqueeze(1)).to(
                        grad_routes.dtype
                    ),
                )
            if grad_weights is not None and not aliases_values:
                selected = route_values.index_select(0, route_chunk)
                grad_weights[token_chunk, column_chunk] = (
                    token_grads.float() * selected.float()
                ).sum(dim=-1)


class _DeepEPScatterWithDeterministicBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        output_index: torch.Tensor,
        total_rows: int,
    ) -> torch.Tensor:
        output = hidden_states.new_zeros((int(total_rows), hidden_states.shape[1]))
        if hidden_states.is_cuda:
            scatter_routes_forward(
                hidden_states.contiguous(),
                output_index.contiguous(),
                output,
            )
        else:
            valid_positions = torch.nonzero(output_index >= 0, as_tuple=False)
            for start in range(0, valid_positions.shape[0], _BACKWARD_CHUNK_ROWS):
                positions = valid_positions[start : start + _BACKWARD_CHUNK_ROWS]
                token_rows = positions[:, 0]
                topk_columns = positions[:, 1]
                destination_rows = output_index[token_rows, topk_columns].to(
                    dtype=torch.long
                )
                output.index_copy_(
                    0,
                    destination_rows,
                    hidden_states.index_select(0, token_rows),
                )

        ctx.input_shape = tuple(hidden_states.shape)
        ctx.input_dtype = hidden_states.dtype
        ctx.input_device = hidden_states.device
        ctx.save_for_backward(output_index)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.needs_input_grad[0]:
            return None, None, None

        (output_index,) = ctx.saved_tensors
        grad_input = torch.zeros(
            ctx.input_shape,
            dtype=ctx.input_dtype,
            device=ctx.input_device,
        )
        if grad_output.is_cuda:
            with _moe_nvtx_range("moe_bwd/route_scatter"):
                scatter_routes_backward(
                    grad_output.contiguous(),
                    output_index.contiguous(),
                    grad_input,
                )
        else:
            for start in range(0, output_index.shape[0], _BACKWARD_CHUNK_ROWS):
                end = min(start + _BACKWARD_CHUNK_ROWS, output_index.shape[0])
                grad_chunk = grad_input[start:end]
                for column in range(output_index.shape[1]):
                    route_rows = output_index[start:end, column].to(dtype=torch.long)
                    valid_rows = route_rows >= 0
                    safe_route_rows = route_rows.clamp(
                        min=0, max=max(grad_output.shape[0] - 1, 0)
                    )
                    if grad_output.shape[0] == 0:
                        continue
                    selected = grad_output.index_select(0, safe_route_rows)
                    selected.masked_fill_(~valid_rows.unsqueeze(1), 0)
                    grad_chunk.add_(selected)
        return grad_input, None, None


def _scatter_deepep_routes_with_padding(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    *,
    expected_route_count: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if hidden_states.ndim != 2 or topk_indices.ndim != 2:
        raise ValueError("DeepEP scatter expects [tokens, hidden] and [tokens, topk]")
    if topk_indices.shape != topk_weights.shape:
        raise ValueError("DeepEP top-k indices and weights must have identical shapes")
    if hidden_states.shape[0] != topk_indices.shape[0]:
        raise ValueError("DeepEP hidden-state and top-k token dimensions differ")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"DeepEP top-k indices must be integer, got {topk_indices.dtype}"
        )
    if topk_weights.dtype != torch.float32:
        raise TypeError(f"DeepEP top-k weights must be FP32, got {topk_weights.dtype}")

    if tokens_per_expert.device.type == "cpu":
        count_values = tuple(
            int(value) for value in tokens_per_expert.reshape(-1).tolist()
        )
        total_rows = sum(count_values)
    else:
        count_values = None

    num_experts = tokens_per_expert.numel()
    valid = (topk_indices >= 0) & (topk_indices < num_experts)
    sanitized_indices = topk_indices.masked_fill(~valid, -1)
    real_counts = torch.bincount(
        sanitized_indices.masked_select(valid).to(dtype=torch.long),
        minlength=num_experts,
    )

    counts_are_exact_unpadded = (
        count_values is not None
        and expected_route_count is not None
        and total_rows == expected_route_count
    )
    if counts_are_exact_unpadded:
        counts = real_counts
        if _HOT_PATH_ASSERTS:
            torch._assert_async(
                real_counts.sum() == expected_route_count,
                "DeepEP received route count differs from its route-preserving metadata",
            )
    else:
        counts = tokens_per_expert.to(
            device=topk_indices.device,
            dtype=torch.long,
        ).reshape(-1)
    if _HOT_PATH_ASSERTS:
        torch._assert_async(
            torch.all(real_counts <= counts),
            "DeepEP real route count exceeds its aligned expert count",
        )

    if count_values is None:
        total_rows = int(counts.sum().item())
    permuted_probs = topk_weights.new_zeros((total_rows,))
    output_index = torch.full_like(topk_indices, -1)
    expert_offsets = torch.cumsum(counts, dim=0) - counts

    if expected_route_count is not None and valid.is_cuda:
        occurrences = compact_route_positions(valid.contiguous(), expected_route_count)
    else:
        occurrences = torch.nonzero(valid, as_tuple=False)
    if occurrences.numel():
        token_rows = occurrences[:, 0]
        topk_columns = occurrences[:, 1]
        route_experts = sanitized_indices[token_rows, topk_columns].to(dtype=torch.long)
        expert_order = torch.argsort(route_experts, stable=True)
        token_rows = token_rows.index_select(0, expert_order)
        topk_columns = topk_columns.index_select(0, expert_order)
        route_experts = route_experts.index_select(0, expert_order)

        real_offsets = torch.cumsum(real_counts, dim=0) - real_counts
        within_expert = torch.arange(
            route_experts.numel(),
            device=hidden_states.device,
            dtype=torch.long,
        ) - real_offsets.index_select(0, route_experts)
        destination_rows = expert_offsets.index_select(0, route_experts) + within_expert
        permuted_probs.index_copy_(
            0,
            destination_rows,
            topk_weights[token_rows, topk_columns],
        )
        output_index[token_rows, topk_columns] = destination_rows.to(
            dtype=output_index.dtype
        )

    if _HOT_PATH_ASSERTS:
        torch._assert_async(
            torch.all(~valid | (output_index >= 0)),
            "A valid DeepEP route has no expert-major output row",
        )
    permuted_hidden = _DeepEPScatterWithDeterministicBackward.apply(
        hidden_states,
        output_index,
        total_rows,
    )
    return (
        permuted_hidden,
        permuted_probs,
        output_index,
        sanitized_indices,
        occurrences,
    )


class _VLLMEPGatherWithBF16Backward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        output_index: torch.Tensor,
        reuse_input_for_grad: bool,
        static_mapping_valid: bool,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                "vLLM ep_gather requires BF16 [expert_rows, hidden], got "
                f"{hidden_states.dtype} {tuple(hidden_states.shape)}"
            )
        if (
            topk_indices.shape != topk_weights.shape
            or topk_indices.shape != output_index.shape
        ):
            raise ValueError(
                "DeepEP gather IDs, weights, and output indices must align"
            )
        from vllm.model_executor.layers.fused_moe.deep_gemm_utils import ep_gather

        output_shape = (topk_indices.shape[0], hidden_states.shape[1])
        output = torch.empty(
            output_shape,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        ep_gather(
            hidden_states,
            topk_indices,
            topk_weights,
            output_index,
            None,
            output,
        )
        ctx.reuse_input_for_grad = bool(reuse_input_for_grad)
        ctx.static_mapping_valid = bool(static_mapping_valid)
        ctx.save_for_backward(hidden_states, topk_weights, output_index)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, topk_weights, output_index = ctx.saved_tensors
        needs_hidden = ctx.needs_input_grad[0]
        needs_weights = ctx.needs_input_grad[2]
        if needs_hidden and ctx.reuse_input_for_grad:
            grad_hidden = hidden_states.detach()
        else:
            grad_hidden = torch.zeros_like(hidden_states) if needs_hidden else None
        grad_weights = torch.zeros_like(topk_weights) if needs_weights else None

        with _moe_nvtx_range("moe_bwd/route_gather"):
            _ordered_route_backward(
                route_values=hidden_states,
                topk_weights=topk_weights,
                output_index=output_index,
                grad_output=grad_output,
                grad_routes=grad_hidden,
                grad_weights=grad_weights,
                static_mapping_valid=ctx.static_mapping_valid,
            )

        return grad_hidden, None, grad_weights, None, None, None


def _compact_route_preserving_metadata_inputs(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise TypeError("Route metadata requires BF16 [tokens, hidden]")
    if hidden_states.shape[1] < 16:
        raise ValueError("Route fingerprints require hidden size >= 16")
    if topk_indices.ndim != 2 or topk_weights.shape != topk_indices.shape:
        raise ValueError("Route top-k IDs and weights must align")
    if topk_indices.shape[0] != hidden_states.shape[0]:
        raise ValueError("Route token counts must align")
    if topk_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("Route IDs must be integer")
    if topk_weights.dtype != torch.float32:
        raise TypeError("Route weights must be FP32")

    flat_indices = topk_indices.reshape(-1)
    valid_positions = torch.nonzero(flat_indices >= 0, as_tuple=False).reshape(-1)
    if valid_positions.numel() == 0:
        raise RuntimeError("Route-preserving DeepEP received no valid expert routes")
    compact_indices = flat_indices.index_select(0, valid_positions)
    compact_weights = topk_weights.detach().reshape(-1).index_select(0, valid_positions)
    token_rows = torch.div(
        valid_positions, topk_indices.shape[1], rounding_mode="floor"
    )
    fingerprints = (
        hidden_states.detach().narrow(1, 0, 16).index_select(0, token_rows).contiguous()
    )
    output_index = torch.full_like(topk_indices, -1, dtype=torch.long)
    output_index.reshape(-1).index_copy_(
        0,
        valid_positions,
        torch.arange(
            valid_positions.numel(), device=valid_positions.device, dtype=torch.long
        ),
    )
    return (
        compact_indices.reshape(-1, 1).contiguous(),
        compact_weights.reshape(-1, 1).contiguous(),
        fingerprints,
        output_index,
        valid_positions.numel() == flat_indices.numel(),
    )


def _deepep_route_handle_received_rows(handle: tuple) -> int:
    if not isinstance(handle, tuple):
        raise TypeError(
            f"DeepEP route handle must be a tuple, got {type(handle).__name__}"
        )
    if len(handle) == 6:
        received_metadata = handle[3]
    elif len(handle) == 10:
        received_metadata = handle[7]
    else:
        raise ValueError(
            f"Unsupported normal DeepEP route handle length: {len(handle)}"
        )
    if not isinstance(received_metadata, torch.Tensor) or received_metadata.ndim < 1:
        raise TypeError("DeepEP route handle has invalid received-source metadata")
    return received_metadata.shape[0]


def _route_hashes(
    fingerprints: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    words = fingerprints.contiguous().view(torch.int16).flatten(1)
    indices = indices.contiguous()
    weight_bits = weights.contiguous().view(torch.int32)
    if words.shape[0] == 0:
        return torch.empty((0,), dtype=torch.int64, device=words.device)
    if words.is_cuda:
        hashes = torch.empty(
            (words.shape[0],), dtype=torch.int64, device=words.device
        )
        _route_hash_kernel[(words.shape[0],)](
            words,
            indices,
            weight_bits,
            hashes,
            FINGERPRINT_WORDS=words.shape[1],
            num_warps=1,
        )
        return hashes
    hashes = torch.full(
        (words.shape[0],),
        1469598103934665603,
        dtype=torch.int64,
        device=words.device,
    )
    for column in range(words.shape[1]):
        hashes = (
            hashes ^ (words[:, column].to(torch.int64) & 0xFFFF)
        ) * 1099511628211
    hashes = (hashes ^ indices.to(torch.int64)) * 1099511628211
    return (hashes ^ (weight_bits.to(torch.int64) & 0xFFFFFFFF)) * 1099511628211


def _validate_and_order_route_preserving_outputs(
    expert_outputs: torch.Tensor,
    received_tokens: torch.Tensor,
    received_topk_indices: torch.Tensor,
    received_topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    route_fingerprints: torch.Tensor,
    route_indices: torch.Tensor,
    route_weights: torch.Tensor,
    *,
    order_outputs: bool = True,
    route_positions: torch.Tensor | None = None,
    return_route_rows: bool = False,
) -> torch.Tensor:
    """Match primary rows to Slime fingerprints and expert IDs."""
    if expert_outputs.ndim != 2 or received_tokens.ndim != 2:
        raise ValueError("Route-preserving DeepEP expects 2D hidden tensors")
    if received_topk_indices.shape != received_topk_weights.shape:
        raise ValueError("Received DeepEP IDs and weights do not align")
    if output_index.shape != received_topk_indices.shape:
        raise ValueError("Received DeepEP route mapping does not align")

    positions = (
        torch.nonzero(output_index >= 0, as_tuple=False)
        if route_positions is None
        else route_positions
    )
    if positions.shape[0] != route_indices.numel():
        raise RuntimeError(
            "Route-preserving DeepEP route count mismatch: "
            f"primary={positions.shape[0]} metadata={route_indices.numel()}"
        )
    token_rows = positions[:, 0]
    topk_slots = positions[:, 1]
    expected_indices = received_topk_indices[token_rows, topk_slots].reshape(-1)
    expected_fingerprints = received_tokens.narrow(1, 0, 16).index_select(0, token_rows)
    if route_fingerprints.shape != expected_fingerprints.shape:
        raise RuntimeError(
            "Route-preserving DeepEP fingerprint shape mismatch: "
            f"{tuple(route_fingerprints.shape)} != {tuple(expected_fingerprints.shape)}"
        )

    expected_weights = received_topk_weights[token_rows, topk_slots].reshape(-1)
    # The hidden-state and metadata dispatches are separate DeepEP operations.
    # At full-model scale they may preserve the same routes while choosing a
    # different arrival order within an expert. Match the two streams by a
    # bitwise route fingerprint before validating and building combine rows.
    expected_hashes = _route_hashes(
        expected_fingerprints, expected_indices, expected_weights
    )
    route_hashes = _route_hashes(
        route_fingerprints,
        route_indices.to(dtype=expected_indices.dtype),
        route_weights.to(dtype=expected_weights.dtype),
    )
    expected_order = torch.argsort(expected_hashes, stable=True)
    route_order = torch.argsort(route_hashes, stable=True)
    expected_for_route = torch.empty_like(expected_order)
    expected_for_route.scatter_(0, route_order, expected_order)
    expected_indices = expected_indices.index_select(0, expected_for_route)
    expected_weights = expected_weights.index_select(0, expected_for_route)
    expected_fingerprints = expected_fingerprints.index_select(0, expected_for_route)
    if _HOT_PATH_ASSERTS:
        torch._assert_async(
            torch.all(expected_indices == route_indices.to(dtype=expected_indices.dtype)),
            "Route-preserving DeepEP metadata changed local expert order",
        )
        torch._assert_async(
            torch.all(
                expected_weights.contiguous().view(torch.int32)
                == route_weights.to(dtype=expected_weights.dtype)
                .contiguous()
                .view(torch.int32)
            ),
            "Route-preserving DeepEP metadata changed route probability order",
        )
        torch._assert_async(
            torch.all(expected_fingerprints == route_fingerprints),
            "Route-preserving DeepEP metadata changed source-token order",
        )
    primary_route_rows = output_index[token_rows, topk_slots].to(
        dtype=torch.long
    ).index_select(0, expected_for_route)
    route_rows = primary_route_rows
    if return_route_rows:
        return route_rows
    if not order_outputs:
        return expert_outputs
    return expert_outputs.index_select(0, route_rows)


_deepep_buffer = None


def _configure_deepep_deterministic_allocator() -> None:
    """Prevent deterministic debug-fill from racing normal DeepEP writes."""
    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.utils.deterministic.fill_uninitialized_memory
    ):
        torch.utils.deterministic.fill_uninitialized_memory = False


def _get_deepep_buffer(group: dist.ProcessGroup, hidden_bytes: int):
    """Reuse the process-wide normal-DeepEP buffer used by MCore and Slime."""
    if deep_ep is None:
        raise RuntimeError("DeepEP is required for vLLM-aligned EP>1")
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
            num_nvl_bytes,
            config.get_nvl_buffer_size_hint(hidden_bytes, group_size),
        )
        if group_size > torch.cuda.device_count():
            num_rdma_bytes = max(
                num_rdma_bytes,
                config.get_rdma_buffer_size_hint(hidden_bytes, group_size),
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


def _dispatch_route_metadata(
    buffer,
    fingerprints: torch.Tensor,
    route_indices: torch.Tensor,
    route_weights: torch.Tensor,
    num_experts: int,
):
    layout = buffer.get_dispatch_layout(
        route_indices,
        num_experts=num_experts,
        async_finish=False,
        allocate_on_comm_stream=False,
    )
    return buffer.dispatch(
        fingerprints.contiguous(),
        topk_idx=route_indices.contiguous(),
        topk_weights=route_weights.float().contiguous(),
        num_tokens_per_rank=layout[0],
        num_tokens_per_rdma_rank=layout[1],
        num_tokens_per_expert=layout[2],
        is_token_in_rank=layout[3],
        previous_event=layout[4],
        async_finish=False,
        allocate_on_comm_stream=False,
    )


class VLLMAlignedNormalDeepEPDispatcher(TokenDispatcher):
    """Match vLLM route identity while using only normal DeepEP transport."""

    def __init__(self, *args, **kwargs):
        kwargs["use_deepep"] = False
        super().__init__(*args, **kwargs)
        self.use_deepep = self.ep_size > 1
        self.buffer = None
        if self.use_deepep:
            if deep_ep is None:
                raise RuntimeError("DeepEP is required for vLLM-aligned EP>1")
            if self.ps.tp_ep_group is None:
                raise RuntimeError("vLLM alignment at EP>1 requires an EP group")
            self._deepep_group = self.ps.tp_ep_group
            deep_ep.Buffer.set_num_sms(20)

    def _ensure_deepep_buffer(self, hidden_states: torch.Tensor):
        if not self.use_deepep:
            raise RuntimeError("DeepEP buffer requested at EP=1")
        self.buffer = _get_deepep_buffer(
            self._deepep_group,
            hidden_states.shape[1] * max(hidden_states.element_size(), 2),
        )
        return self.buffer

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._dispatch_aligned(hidden_states, topk_scores, topk_indices)

    def combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        route_outputs = expert_output.index_select(0, self._metadata_route_rows)
        if self.ep_size > 1:
            with _moe_nvtx_range("moe/deepep/combine"):
                source_routes = _DeepEPCombine.apply(
                    self.buffer,
                    route_outputs,
                    self._route_handle,
                    True,
                    False,
                )
        else:
            source_routes = route_outputs
        output = _VLLMEPGatherWithBF16Backward.apply(
            source_routes,
            self._source_indices,
            self._source_weights,
            self._source_output_index,
            True,
            self._source_all_routes_valid,
        )
        for name in (
            "_metadata_route_rows",
            "_route_handle",
            "_source_indices",
            "_source_weights",
            "_source_output_index",
            "_source_all_routes_valid",
        ):
            delattr(self, name)
        self._local_tpe_list = None
        return output

    def _dispatch_aligned(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
            raise TypeError("aligned DeepEP requires BF16 [tokens, hidden]")
        if hidden_states.shape[1] < 16:
            raise ValueError("aligned DeepEP requires hidden size >= 16")
        if topk_indices.shape != topk_scores.shape:
            raise ValueError("top-k IDs and scores must have identical shapes")
        topk_indices = topk_indices.long().contiguous()
        topk_scores = topk_scores.float().contiguous()
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
        )
        if self.ep_size > 1:
            buffer = self._ensure_deepep_buffer(hidden_states)
            with _moe_nvtx_range("moe/deepep/primary_dispatch"):
                (
                    received_hidden,
                    received_indices,
                    received_weights,
                    received_per_expert_cpu,
                    _,
                ) = _DeepEPDispatch.apply(
                    buffer,
                    hidden_states,
                    topk_indices,
                    topk_scores,
                    self.num_experts,
                    False,
                    False,
                )
            if received_per_expert_cpu.device.type != "cpu":
                raise RuntimeError("DeepEP expert counts must remain CPU metadata")
            self._local_tpe_list = received_per_expert_cpu.tolist()
            received_per_expert = received_per_expert_cpu
            with _moe_nvtx_range("moe/deepep/metadata_dispatch"):
                (
                    received_fingerprints,
                    received_route_indices,
                    received_route_weights,
                    _,
                    route_handle,
                    _,
                ) = _dispatch_route_metadata(
                    buffer,
                    route_fingerprints,
                    route_indices,
                    route_weights,
                    self.num_experts,
                )
        else:
            received_hidden = hidden_states
            received_indices = topk_indices
            received_weights = topk_scores
            received_fingerprints = route_fingerprints
            received_route_indices = route_indices
            received_route_weights = route_weights
            route_handle = None
            received_per_expert = torch.bincount(
                received_route_indices.reshape(-1).long(),
                minlength=self.num_local_experts,
            )
            self._local_tpe_list = received_per_expert.tolist()
        expected_route_count = (
            _deepep_route_handle_received_rows(route_handle)
            if route_handle is not None
            else int((received_indices >= 0).sum().item())
        )
        with _moe_nvtx_range("moe/deepep/scatter_routes"):
            (
                expert_hidden,
                expert_probs,
                output_index,
                sanitized_indices,
                positions,
            ) = _scatter_deepep_routes_with_padding(
                received_hidden,
                received_indices,
                received_weights,
                received_per_expert,
                expected_route_count=expected_route_count,
            )
        with _moe_nvtx_range("moe/deepep/order_routes"):
            self._metadata_route_rows = _validate_and_order_route_preserving_outputs(
                expert_hidden,
                received_hidden,
                sanitized_indices,
                received_weights,
                output_index,
                received_fingerprints,
                received_route_indices.reshape(-1),
                received_route_weights.reshape(-1),
                route_positions=positions,
                return_route_rows=True,
            )
        self._route_handle = route_handle
        self._source_indices = topk_indices
        self._source_weights = topk_scores
        self._source_output_index = source_output_index
        self._source_all_routes_valid = source_all_routes_valid
        return expert_hidden, received_per_expert, expert_probs


class VLLMAlignedHybridEPDispatcher(TokenDispatcher):
    """Preserve vLLM route slots over the dedicated HybridEP transport."""

    def __init__(
        self,
        *args,
        hybridep_max_tokens_per_rank: int | None = None,
        **kwargs,
    ):
        if (
            not isinstance(hybridep_max_tokens_per_rank, int)
            or isinstance(hybridep_max_tokens_per_rank, bool)
            or hybridep_max_tokens_per_rank <= 0
        ):
            raise ValueError(
                "hybridep_max_tokens_per_rank must be a positive integer"
            )
        kwargs["use_deepep"] = False
        super().__init__(*args, **kwargs)
        if self.ep_size <= 1:
            raise RuntimeError("hybridep requires expert parallel size greater than one")
        if self.ps.ep_group is None:
            raise RuntimeError("hybridep requires an expert-parallel process group")
        hybridep.require_available()
        self._hybridep_group = self.ps.ep_group
        self._hybridep_max_tokens_per_rank = hybridep_max_tokens_per_rank
        self._hybridep_state = None

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._hybridep_state is not None:
            raise RuntimeError("hybridep dispatch state is still awaiting combine")
        result = hybridep.dispatch_routes(
            hidden_states,
            topk_scores.float().contiguous(),
            topk_indices.long().contiguous(),
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            group=self._hybridep_group,
            hybridep_max_tokens_per_rank=self._hybridep_max_tokens_per_rank,
        )
        self._hybridep_state = result.state
        self._local_tpe_list = result.tokens_per_expert_list
        return result.hidden, result.tokens_per_expert, result.probs

    def combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        if self._hybridep_state is None:
            raise RuntimeError("hybridep combine has no matching dispatch state")
        state = self._hybridep_state
        source_routes = hybridep.combine_routes(expert_output, state)
        output = _VLLMEPGatherWithBF16Backward.apply(
            source_routes,
            state.source_indices,
            state.source_weights,
            state.source_output_index,
            False,
            state.source_all_routes_valid,
        )
        self._hybridep_state = None
        self._local_tpe_list = None
        return output
