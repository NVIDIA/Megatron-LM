# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Differentiable row-sharded embedding lookup for Engram.

Adapted from NVIDIA/Megatron-LM#7231 by BestJuly, including uneven logical
checkpoint shapes and rank-local initialization that preserves replicated RNG.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import MethodType
from typing import Callable, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import (
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.fp8_utils import (
    copy_tensor_to_quantized_param,
    dequantize_fp8_tensor,
    is_mxfp8tensor,
)
from megatron.core.ops._backends import require
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import (
    ensure_params_ready,
    get_pg_rank,
    get_pg_size,
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _engram_accumulate_rows_kernel(
        target,
        rows,
        order,
        gradients,
        N,
        D: tl.constexpr,
        TARGET_STRIDE: tl.constexpr,
        GRAD_STRIDE: tl.constexpr,
        ROW_STRIDE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """One writer per sorted segment; sum duplicates in FP32 without atomics."""
        first = tl.program_id(0)
        source = tl.load(order + first)
        row = tl.load(rows + source * ROW_STRIDE)
        previous = tl.load(order + first - 1, first > 0, other=0)
        previous_row = tl.load(rows + previous * ROW_STRIDE)
        if (first == 0) | (row != previous_row):
            cols = tl.arange(0, BLOCK_D)
            values = tl.full((BLOCK_D,), 0, tl.float32)
            more = True
            start = first
            while more:
                positions = start + tl.arange(0, BLOCK_ROWS)
                indices = tl.load(order + positions, positions < N, other=0)
                ids = tl.load(rows + indices * ROW_STRIDE, positions < N, other=-1)
                valid = (positions < N) & (ids == row)
                chunk = tl.load(
                    gradients + indices[:, None] * GRAD_STRIDE + cols[None, :],
                    valid[:, None] & (cols[None, :] < D),
                    other=0,
                ).to(tl.float32)
                values += tl.sum(chunk, axis=0)
                start += BLOCK_ROWS
                more = (start < N) & (tl.sum(valid.to(tl.int32), axis=0) == BLOCK_ROWS)
            old = tl.load(target + row * TARGET_STRIDE + cols, cols < D, other=0).to(tl.float32)
            tl.store(target + row * TARGET_STRIDE + cols, old + values, cols < D)

    @triton.jit
    def _mxfp8_gather_pack_kernel(
        data,
        scales,
        row_ids,
        table_order,
        packed,
        start,
        count,
        D: tl.constexpr,
        DATA_STRIDE: tl.constexpr,
        SCALE_STRIDE: tl.constexpr,
        ID_STRIDE: tl.constexpr,
        HAS_TABLE_ORDER: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Gather one table directly into its slots in the EP return payload."""
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, col = offset // (D + D // 32), offset % (D + D // 32)
        valid = row < count
        table_row = tl.load(row_ids + (start + row) * ID_STRIDE, valid, other=0)
        value = tl.load(data + table_row * DATA_STRIDE + col, valid & (col < D), other=0)
        scale = tl.load(scales + table_row * SCALE_STRIDE + col - D, valid & (col >= D), other=0)
        destination = start + row
        if HAS_TABLE_ORDER:
            destination = tl.load(table_order + destination, valid, other=0)
        tl.store(packed + destination * (D + D // 32) + col, tl.where(col < D, value, scale), valid)

    @triton.jit
    def _engram_inverse_permutation_kernel(order, inverse, count, BLOCK: tl.constexpr):
        """Map original token/head positions to received EP payload rows."""
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        original = tl.load(order + index, index < count, other=0)
        tl.store(inverse + original, index, index < count)

    @triton.jit
    def _mxfp8_repack_activation_kernel(
        packed,
        inverse,
        row_data,
        row_scales,
        col_data,
        col_scales,
        tokens,
        padded_tokens,
        D: tl.constexpr,
        WIDTH: tl.constexpr,
        ROW_SCALE_STRIDE: tl.constexpr,
        COL_SCALE_STRIDE: tl.constexpr,
        DTYPE: tl.constexpr,
        HAS_ORDER: tl.constexpr,
        NEEDS_WGRAD: tl.constexpr,
    ):
        """Restore heads and produce both GEMM orientations without a BF16 buffer."""
        rows = tl.program_id(0) * 32 + tl.arange(0, 32)
        cols = tl.program_id(1) * 32 + tl.arange(0, 32)
        valid_rows = (rows < tokens) & (tl.program_id(1) * 32 < WIDTH)
        # D is a multiple of 32: this block shares one retrieved row and scale per token.
        source = rows * (WIDTH // D) + tl.program_id(1) * 32 // D
        if HAS_ORDER:
            source = tl.load(inverse + source, valid_rows, other=0)
        raw = tl.load(
            packed + source[:, None] * (D + D // 32) + cols[None, :] % D,
            valid_rows[:, None],
            other=0,
        )
        scales = tl.load(
            packed + source * (D + D // 32) + D + tl.program_id(1) % (D // 32),
            valid_rows,
            other=127,
        )
        data_mask = (rows[:, None] < padded_tokens) & (cols[None, :] < WIDTH)
        tl.store(row_data + rows[:, None] * WIDTH + cols[None, :], raw, data_mask)
        # The grid covers TE's complete scale allocation, including 128-by-4 padding.
        tl.store(row_scales + rows * ROW_SCALE_STRIDE + tl.program_id(1), scales)
        if NEEDS_WGRAD:
            # Match TE dequantization and the logical dtype rounding before wgrad
            # quantization. In particular TE decodes E8M0 zero as FP32 zero.
            dequant_scale = (scales.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
            values = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * dequant_scale[:, None]
            values = values.to(DTYPE).to(tl.float32)
            # TE fmax ignores NaNs when reducing amax, but leaves them in the data.
            amax = tl.max(tl.where(values == values, tl.abs(values), 0.0), axis=0)
            scaled_amax = amax * (1.0 / 448.0)
            bits = scaled_amax.to(tl.uint32, bitcast=True)
            exponent, mantissa = (bits >> 23) & 255, bits & 0x7FFFFF
            # TE ptx::float_to_e8m0: round toward +inf, saturate, including subnormals.
            round_up = (mantissa > 0) & (exponent < 254)
            round_up = round_up & ((exponent != 0) | (mantissa > 0x400000))
            exponent = tl.minimum(exponent + round_up.to(tl.uint32), 254)
            reciprocal_bits = tl.where(exponent == 254, 0x00400000, (254 - exponent) << 23)
            reciprocal = reciprocal_bits.to(tl.float32, bitcast=True)
            quantized = (values * reciprocal[None, :]).to(
                tl.float8e4nv, fp_downcast_rounding='rtne'
            )
            tl.store(
                col_data + rows[:, None] * WIDTH + cols[None, :],
                quantized.to(tl.uint8, bitcast=True),
                data_mask,
            )
            tl.store(
                col_scales + tl.program_id(0) * COL_SCALE_STRIDE + cols,
                tl.where((tl.program_id(0) * 32 < padded_tokens) & (cols < WIDTH), exponent, 127),
            )


def _mxfp8_pack_lookup(
    weights, row_ids, table_order, table_counts, embedding_dim, *, use_fused=True
):
    """Pack raw table values/scales in request order, with a native reference fallback."""
    packed = torch.empty(
        (row_ids.numel(), embedding_dim + embedding_dim // 32),
        device=row_ids.device,
        dtype=torch.uint8,
    )
    start = 0
    for weight, count in zip(weights, table_counts):
        if not is_mxfp8tensor(weight):
            raise RuntimeError('Move Engram FP8 tables to CUDA before performing lookups')
        storage = weight if hasattr(weight, '_rowwise_data') else weight.data
        if getattr(storage, '_with_gemm_swizzled_scales', False):
            raise ValueError('Engram FP8 table lookup requires unswizzled rowwise scales')
        if count:
            if use_fused and HAVE_TRITON and row_ids.is_cuda:
                _mxfp8_gather_pack_kernel[(triton.cdiv(count * packed.shape[1], 1024),)](
                    storage._rowwise_data,
                    storage._rowwise_scale_inv,
                    row_ids,
                    table_order,
                    packed,
                    start,
                    count,
                    embedding_dim,
                    storage._rowwise_data.stride(0),
                    storage._rowwise_scale_inv.stride(0),
                    row_ids.stride(0),
                    table_order is not None,
                    1024,
                )
            else:
                ids = row_ids.narrow(0, start, count)
                values = torch.cat(
                    (
                        storage._rowwise_data.index_select(0, ids),
                        storage._rowwise_scale_inv[:, : embedding_dim // 32].index_select(0, ids),
                    ),
                    dim=-1,
                )
                if table_order is None:
                    packed[start : start + count].copy_(values)
                else:
                    packed.index_copy_(0, table_order.narrow(0, start, count), values)
        start += count
    return packed


def _mxfp8_quantizer():
    """Use the same E4M3/E8M0 representation as TE's MXFP8 Linear."""
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine_torch import DType

    quantizer = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=True, columnwise=False)
    quantizer.optimize_for_gemm = False
    return quantizer


def _mxfp8_lookup_activation(
    packed,
    embedding_dim,
    output_width,
    dtype,
    needs_wgrad,
    order=None,
    *,
    use_fused=True,
    inverse=None,
):
    """Repack retrieved bytes for GEMM; only the batch needs a columnwise representation."""
    if order is not None and inverse is not None:
        raise ValueError("Specify either request order or an explicit inverse map")
    rows = packed.shape[0] if inverse is None else inverse.numel()
    tokens = rows * embedding_dim // output_width
    # An empty requester still participates in EP and backward; give GEMM a valid tile.
    padded_tokens = max(32, (tokens + 31) // 32 * 32)
    quantizer = _mxfp8_quantizer()
    if use_fused and HAVE_TRITON and packed.is_cuda:
        quantizer.set_usage(rowwise=True, columnwise=needs_wgrad)
        output = quantizer.make_empty(
            (padded_tokens, output_width), dtype=dtype, device=packed.device
        )
        if order is not None and order.numel():
            inverse = torch.empty_like(order)
            _engram_inverse_permutation_kernel[(triton.cdiv(order.numel(), 1024),)](
                order, inverse, order.numel(), 1024
            )
        _mxfp8_repack_activation_kernel[
            (output._rowwise_scale_inv.shape[0] // 32, output._rowwise_scale_inv.shape[1])
        ](
            packed,
            inverse,
            output._rowwise_data,
            output._rowwise_scale_inv,
            output._columnwise_data,
            output._columnwise_scale_inv,
            tokens,
            padded_tokens,
            embedding_dim,
            output_width,
            output._rowwise_scale_inv.stride(0),
            output._columnwise_scale_inv.stride(0) if needs_wgrad else 0,
            {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[dtype],
            inverse is not None,
            needs_wgrad,
        )
        return output

    if inverse is not None:
        packed = packed.index_select(0, inverse)
    if order is not None:
        ordered = torch.empty_like(packed)
        ordered.index_copy_(0, order, packed)
        packed = ordered
    output = quantizer.make_empty((padded_tokens, output_width), dtype=dtype, device=packed.device)
    output._rowwise_data.zero_()
    # E8M0 byte 127 encodes scale 1, including the unused alignment padding.
    output._rowwise_scale_inv.fill_(127)
    output._rowwise_data[:tokens].copy_(packed[:, :embedding_dim].reshape(tokens, output_width))
    output._rowwise_scale_inv[:tokens, : output_width // 32].copy_(
        packed[:, embedding_dim:].reshape(tokens, output_width // 32)
    )
    if needs_wgrad:
        # The forward uses the original retrieved rowwise bytes. wgrad needs columnwise
        # quantization of this batch, whose row ordering differs from the embedding table.
        dequantized = output.dequantize()
        quantizer.set_usage(rowwise=False, columnwise=True)
        columnwise = quantizer(dequantized)
        output._columnwise_data = columnwise._columnwise_data
        output._columnwise_scale_inv = columnwise._columnwise_scale_inv
        output._get_quantizer().set_usage(rowwise=True, columnwise=True)
    return output


def _embedding_row_gradients(row_ids, grad_output, shape, dtype):
    """Accumulate repeated row IDs in FP32, then cast once per touched row."""
    grad_weight = torch.zeros(shape, dtype=dtype, device=grad_output.device)
    if row_ids.numel():
        order = torch.argsort(row_ids, stable=True)
        sorted_rows = row_ids.index_select(0, order)
        accumulation_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        sorted_grads = grad_output.index_select(0, order).to(accumulation_dtype)
        unique_rows, counts = torch.unique_consecutive(sorted_rows, return_counts=True)
        row_grads = torch.segment_reduce(sorted_grads, 'sum', lengths=counts)
        grad_weight.index_copy_(0, unique_rows, row_grads.to(dtype))
    return grad_weight


class _EngramExchangePlan(NamedTuple):
    """Lookup-local sizes for interleaved dispatches, retained through backward."""

    max_rows: int
    recv_rows: tuple[int, ...]


class _EngramRowExchangePlan(NamedTuple):
    """Global peer counts for coalesced one-sided transfers, in source order."""

    max_rows: int
    peer_counts: Tensor


class _EngramAllToAll:
    """Exact variable-split exchanges, independent of the MoE dispatcher and its buffers.

    HybridEP dispatch treats each row as a top-1 token and each shard rank as an expert.
    Packets contain an integer ordering key followed by opaque bytes. Dispatch copies
    UINT16 words without arithmetic; combine would sum BF16 values and corrupt integer
    IDs and MXFP8 data/scales, so the return and gradient paths also use dispatch.
    """

    def __init__(self, group, backend="nccl", chunk_size=4096, max_row_bytes=16):
        if backend not in ("nccl", "hybridep", "symmetric"):
            raise ValueError(f"Unknown Engram communication backend: {backend}")
        if chunk_size <= 0 or chunk_size % 64:
            raise ValueError("Engram hybridep_chunk_size must be a positive multiple of 64")
        self.group = group
        self.size, self.rank = get_pg_size(group), get_pg_rank(group)
        self.backend, self.chunk_size = backend, chunk_size
        # HybridEP's UINT16 TMA path requires a multiple of 16 channels (32 bytes).
        self.max_packet_bytes = (max(16, max_row_bytes) + 8 + 31) // 32 * 32
        self.buffer = None
        self.workspace_bytes = 64 * 1024**2
        self.device_transport = (
            require(
                "megatron.core.ops.engram", "EngramDeviceTransport", needed_by="Engram symmetric"
            ).EngramDeviceTransport(group)
            if backend == "symmetric"
            else None
        )

    def exchange_counts(self, send_counts, *, global_counts=False):
        """Return peer splits, owner table counts, and forward/reverse exchange plans."""
        if self.backend == "nccl" and not global_counts:
            recv_counts = torch.empty_like(send_counts)
            torch.distributed.all_to_all_single(recv_counts, send_counts, group=self.group)
            sent, received = torch.stack((send_counts, recv_counts)).view(2, self.size, -1).cpu()
            plans = (None, None)
        else:
            gathered = send_counts.new_empty(self.size * send_counts.numel())
            torch.distributed.all_gather_into_tensor(gathered, send_counts, group=self.group)
            # One host transfer per lookup provides exact counts for every round in
            # all three directions. Never cache the current plan on the transport:
            # another microbatch may run before this lookup's backward.
            counts = gathered.view(self.size, self.size, -1).cpu()
            sent, received = counts[self.rank], counts[:, self.rank]
            peer_counts = counts.sum(dim=-1)
            plans = (
                (self._hybridep_plan(peer_counts), self._hybridep_plan(peer_counts.T))
                if self.backend == 'hybridep' and not global_counts
                else (
                    _EngramRowExchangePlan(int(peer_counts.sum(1).max()), peer_counts),
                    _EngramRowExchangePlan(
                        int(peer_counts.sum(0).max()), peer_counts.T.contiguous()
                    ),
                )
            )
        return (
            sent.sum(dim=1).tolist(),
            received.sum(dim=1).tolist(),
            received.sum(dim=0).tolist(),
            plans,
        )

    def _hybridep_plan(self, peer_counts):
        """Count rows destined for this rank when round r sends indices r::rounds."""
        max_rows = int(peer_counts.sum(dim=1).max())
        rounds = (max_rows + self.chunk_size - 1) // self.chunk_size
        if rounds == 0:
            return _EngramExchangePlan(0, ())
        ends = peer_counts.cumsum(dim=1)[:, self.rank]
        starts = ends - peer_counts[:, self.rank]
        offsets = torch.arange(rounds, dtype=torch.int64, device="cpu")[:, None]
        # Number of indices congruent to r modulo rounds in each [start, end).
        received = ((ends + rounds - 1 - offsets) // rounds) - (
            (starts + rounds - 1 - offsets) // rounds
        )
        return _EngramExchangePlan(max_rows, tuple(received.sum(dim=1).tolist()))

    def exchange(self, input_, output_splits, input_splits, plan=None):
        """Return rows ordered by source rank, then by their original position."""
        if self.backend == "nccl":
            output = input_.new_empty((sum(output_splits), *input_.shape[1:]))
            torch.distributed.all_to_all_single(
                output,
                input_.contiguous(),
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=self.group,
            )
            return output
        if self.backend == 'symmetric':
            return self._symmetric_exchange(input_, output_splits, input_splits, plan)
        return self._hybridep_exchange(input_, output_splits, input_splits, plan)

    def _symmetric_exchange(self, input_, output_splits, input_splits, plan):
        if not isinstance(plan, _EngramRowExchangePlan):
            raise ValueError("Engram symmetric requires a lookup-local row exchange plan")
        return self.device_transport.exchange(
            input_, output_splits, input_splits, plan, self.workspace_bytes
        )

    def lookup(self, module, ticket):
        """Read local-domain tables directly; GIN services only remote-domain rows."""
        data, scales = [], []
        for table in module.tables:
            weight = table.weight
            if table.fp8_storage:
                if not is_mxfp8tensor(weight):
                    raise RuntimeError("Move Engram FP8 tables to CUDA before performing lookups")
                storage = weight if hasattr(weight, "_rowwise_data") else weight.data
                if getattr(storage, "_with_gemm_swizzled_scales", False):
                    raise ValueError("Engram FP8 lookup requires unswizzled rowwise scales")
                data.append(storage._rowwise_data)
                scales.append(storage._rowwise_scale_inv)
            else:
                data.append(weight)
        payload = self.device_transport.lookup(
            module,
            data,
            scales,
            ticket.requests,
            ticket.send_splits,
            ticket.recv_splits,
            ticket.plans,
            self.workspace_bytes,
        )
        return payload if scales else payload.view(module.tables[0].weight.dtype)

    def _hybridep_exchange(self, input_, output_splits, input_splits, plan):
        if isinstance(plan, _EngramRowExchangePlan):
            plan = self._hybridep_plan(plan.peer_counts)
        if not input_.is_cuda:
            raise ValueError("Engram HybridEP communication requires CUDA tensors")
        if self.size > 8192:
            raise ValueError("Engram HybridEP top-1 routing supports at most 8192 EP ranks")
        if sum(input_splits) != input_.shape[0]:
            raise ValueError("Engram input splits must cover every input row")
        if plan is None:
            raise ValueError("Engram HybridEP exchange requires a lookup-local plan")

        # All ranks participate in the same bounded number of dispatches, including
        # empty requesters/owners. Capacity never scales with the full sequence length.
        max_rows, recv_rows = plan
        rounds = len(recv_rows)
        output_rows = sum(output_splits)
        if sum(recv_rows) != output_rows or input_.shape[0] > max_rows:
            raise ValueError("Engram HybridEP plan does not match the exchange splits")
        if max_rows == 0:
            return input_.new_empty((output_rows, *input_.shape[1:]))

        if self.buffer is None:
            try:
                from deep_ep import HybridEPBuffer
            except ImportError as exc:
                raise ImportError("Engram hybridep requires DeepEP with HybridEPBuffer") from exc
            self.buffer = HybridEPBuffer(
                self.group,
                hidden_dim=self.max_packet_bytes // 2,
                max_num_of_tokens_per_rank=self.chunk_size,
                # Four slots satisfy HybridEP's TMA alignment even for NVLink domains
                # of size one or two. Only slot zero on each rank receives requests.
                num_local_experts=4,
                use_fp8=False,
                # Match the supported MoE setting; custom all-gather can hang on MNNVL.
                enable_custom_allgather=False,
            )

        payload = input_.contiguous().view(torch.uint8).flatten(1)
        row_bytes = payload.shape[1]
        packet_bytes = (8 + row_bytes + 31) // 32 * 32
        destinations = torch.repeat_interleave(
            torch.arange(self.size, dtype=torch.int16, device=input_.device) * 4,
            torch.tensor(input_splits, dtype=torch.int64, device=input_.device),
            output_size=input_.shape[0],
        )
        output = payload.new_empty((output_rows, row_bytes))
        keys = torch.empty(output_rows, dtype=torch.int64, device=input_.device)
        packet = payload.new_zeros((self.chunk_size, packet_bytes))
        routes = torch.full((self.chunk_size, 1), -1, dtype=torch.int16, device=input_.device)
        received_rows = 0
        for round_id, expected_rows in enumerate(recv_rows):
            # Spread every owner's sorted segment across rounds instead of sending
            # to just the first few owners in round zero. Keep the original indices
            # in the packet keys to restore the same order as NCCL after reception.
            count = max(0, (input_.shape[0] + rounds - 1 - round_id) // rounds)
            routes.fill_(-1)
            if count:
                source_keys = torch.arange(round_id, input_.shape[0], rounds, device=input_.device)
                source_keys += self.rank * max_rows
                packet[:count, :8].copy_(source_keys.view(torch.uint8).view(count, 8))
                packet[:count, 8 : 8 + row_bytes].copy_(payload[round_id::rounds])
                routes[:count, 0].copy_(destinations[round_id::rounds])
            received, _, _, _, _ = self.buffer.dispatch_with_permute(
                hidden=packet.view(torch.bfloat16),
                topk_idx=routes,
                num_of_experts=self.size * 4,
                num_of_experts_per_rank=4,
                use_fp8=False,
                pad_multiple=1,
                num_permuted_tokens=expected_rows,
                non_blocking=True,
            )
            # No capacity/drop budget: keep every routed row, even for an all-to-one
            # hotspot. Copy before reusing HybridEP's workspace in the next dispatch.
            received = received.view(torch.uint8)
            count = received.shape[0]
            end = received_rows + count
            keys[received_rows:end].copy_(received[:, :8].contiguous().view(torch.int64).flatten())
            output[received_rows:end].copy_(received[:, 8 : 8 + row_bytes])
            received_rows = end
        if received_rows != output_rows:
            raise RuntimeError(
                f"Engram HybridEP received {received_rows} rows, expected {output_rows}"
            )
        order = torch.argsort(keys)
        return output.index_select(0, order).view(input_.dtype).view(output_rows, *input_.shape[1:])


class _MXFP8EngramLookup(torch.autograd.Function):
    """Exchange FP8 bytes/scales forward and high-precision row gradients backward."""

    @staticmethod
    def forward(ctx, module, hash_ids, output_width, needs_wgrad, *weights):
        requests, order, send_splits, recv_splits, table_counts, plans = module._route_requests(
            hash_ids
        )
        # V4.1 combines its hash heads into one table, so no table permutation is needed.
        table_order = torch.argsort(requests[:, 0], stable=True) if len(weights) > 1 else None
        row_ids = requests[:, 1]
        if table_order is not None:
            row_ids = row_ids.index_select(0, table_order)
        width = module.embedding_dim
        packed = _mxfp8_pack_lookup(weights, row_ids, table_order, table_counts, width)
        if module.ep_size > 1:
            packed = module.communication.exchange(packed, send_splits, recv_splits, plans[1])
        else:
            order = None
        ctx.save_for_backward(row_ids, table_order, order)
        ctx.communication, ctx.ep_size = module.communication, module.ep_size
        ctx.send_splits, ctx.recv_splits = send_splits, recv_splits
        ctx.request_plan = plans[0]
        ctx.table_counts = table_counts
        ctx.shapes = [tuple(weight.shape) for weight in weights]
        ctx.dtypes = [weight.dtype for weight in weights]
        ctx.width, ctx.rows = width, hash_ids.numel()
        return _mxfp8_lookup_activation(
            packed, width, output_width, weights[0].dtype, needs_wgrad, order
        )

    @staticmethod
    def backward(ctx, grad_output):
        row_ids, table_order, order = ctx.saved_tensors
        # Discard token alignment padding before reversing the request permutation.
        grad = grad_output.reshape(-1)[: ctx.rows * ctx.width].view(ctx.rows, ctx.width)
        if order is not None:
            grad = grad.index_select(0, order)
        if ctx.ep_size > 1:
            grad = ctx.communication.exchange(
                grad, ctx.recv_splits, ctx.send_splits, ctx.request_plan
            )
        if table_order is not None:
            grad = grad.index_select(0, table_order)
        gradients, start = [], 0
        for count, shape, dtype in zip(ctx.table_counts, ctx.shapes, ctx.dtypes):
            gradients.append(
                _embedding_row_gradients(
                    row_ids.narrow(0, start, count), grad.narrow(0, start, count), shape, dtype
                )
            )
            start += count
        return None, None, None, None, *gradients


class _DifferentiableEngramAllToAll(torch.autograd.Function):
    """Variable-split all-to-all with Engram-specific forward/backward NVTX ranges."""

    @staticmethod
    def forward(
        ctx, communication, input_: Tensor, output_splits: list[int], input_splits: list[int], plans
    ):
        """Exchange owner embeddings while retaining peer splits for the reverse exchange."""
        ctx.communication = communication
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        ctx.request_plan = plans[0]
        message = "engram.lookup.return-a2a.forward"
        nvtx_range_push(message)
        try:
            return communication.exchange(input_, output_splits, input_splits, plans[1])
        finally:
            nvtx_range_pop(message)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Return embedding gradients to the owners using the reversed peer splits."""
        message = "engram.lookup.return-a2a.backward"
        nvtx_range_push(message)
        try:
            grad_input = ctx.communication.exchange(
                grad_output, ctx.input_splits, ctx.output_splits, ctx.request_plan
            )
        finally:
            nvtx_range_pop(message)
        return None, grad_input, None, None, None


class _DeterministicEmbedding(torch.autograd.Function):
    """Embedding lookup with sorted, segmented accumulation for repeated-row gradients."""

    @staticmethod
    def forward(ctx, weight: Tensor, row_ids: Tensor) -> Tensor:
        """Gather requested rows and save their IDs for deterministic accumulation."""
        ctx.save_for_backward(row_ids)
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        return weight.index_select(0, row_ids)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        """Sum duplicate-row gradients in sorted order and write each unique row once."""
        message = "engram.embedding.backward"
        nvtx_range_push(message)
        try:
            (row_ids,) = ctx.saved_tensors
            return (
                _embedding_row_gradients(row_ids, grad_output, ctx.weight_shape, ctx.weight_dtype),
                None,
            )
        finally:
            nvtx_range_pop(message)


def get_contiguous_row_range(global_rows: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return the balanced, unpadded contiguous row interval owned by ``rank``."""
    if global_rows < 0:
        raise ValueError(f"global_rows must be nonnegative, got {global_rows}.")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")
    base, remainder = divmod(global_rows, world_size)
    start = rank * base + min(rank, remainder)
    local_rows = base + int(rank < remainder)
    return start, start + local_rows


class EPShardedEmbeddingTable(MegatronModule):
    """One prime-sized embedding table sharded over an explicit EP group."""

    def __init__(
        self,
        config,
        global_num_embeddings: int,
        embedding_dim: int,
        init_method: Callable[[Tensor], None],
        ep_group=None,
        tp_group=None,
        expt_dp_group=None,
        fp8_storage: bool = False,
    ) -> None:
        super().__init__(config=config)
        self.global_num_embeddings = global_num_embeddings
        self.embedding_dim = embedding_dim
        self.fp8_storage = fp8_storage
        if fp8_storage and embedding_dim % 32:
            raise ValueError('Engram MXFP8 requires embedding_dim divisible by 32')
        if fp8_storage and config.params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError('Engram MXFP8 primary weights require FP16/BF16 model parameters')
        self.tp_group = tp_group
        self.expt_dp_group = expt_dp_group
        self.deterministic_mode = getattr(config, "deterministic_mode", False)
        self.ep_rank = get_pg_rank(ep_group)
        self.ep_size = get_pg_size(ep_group)
        self.row_start, self.row_end = get_contiguous_row_range(
            global_num_embeddings, self.ep_rank, self.ep_size
        )
        physical_rows = self.row_end - self.row_start
        if fp8_storage:
            physical_rows = (physical_rows + 31) // 32 * 32

        device = (
            torch.device("cpu") if config.use_cpu_initialization else torch.cuda.current_device()
        )
        self.weight = nn.Parameter(
            torch.empty((physical_rows, embedding_dim), device=device, dtype=config.params_dtype)
        )
        if config.perform_initialization:
            self._initialize_weight(init_method)
        with torch.no_grad():
            self.weight[self.local_num_embeddings :].zero_()

        # The row shard is distinct across EP and synchronized only over expert-DP.
        self.weight.allreduce = False
        self.weight.is_engram_embedding = True
        self.weight.is_embedding_or_output_parameter = True
        # Matrix optimizers need the logical row partition, independently of DP replication.
        self.weight.row_shard_group = ep_group
        self.weight.row_shard_global_rows = global_num_embeddings
        self.weight.row_shard_start = self.row_start
        self.weight.row_shard_local_rows = self.local_num_embeddings
        self.weight.mxfp8_requires_columnwise = False
        if config.sequence_parallel:
            self.weight.sequence_parallel = True
        self._quantize_weight()

    def _quantize_weight(self):
        """Materialize primary FP8 storage, retaining the original optimizer initializer."""
        if not self.fp8_storage or self.weight.device.type != 'cuda' or is_mxfp8tensor(self.weight):
            return
        source = self.weight
        quantizer = _mxfp8_quantizer()
        weight = quantizer.make_empty(source.shape, dtype=source.dtype, device=source.device)
        if source.numel():
            copy_tensor_to_quantized_param(weight, source.detach())
        self.weight = nn.Parameter(weight, requires_grad=source.requires_grad)
        self.weight.__dict__.update(source.__dict__)
        if torch.is_grad_enabled():
            # Match TE Linear's native primary-weight contract: only optimizer owners keep
            # FP32 masters, and the factory releases all remaining CPU initializers.
            self.weight._high_precision_init_val = source.detach().cpu()
            self.weight.get_high_precision_init_val = MethodType(
                lambda parameter: parameter._high_precision_init_val, self.weight
            )
            self.weight.clear_high_precision_init_val = MethodType(
                lambda parameter: setattr(parameter, '_high_precision_init_val', None), self.weight
            )

    def _apply(self, fn, recurse=True):
        """Preserve parameter metadata through device/dtype moves and FP8 materialization."""
        source = self.weight
        attributes = dict(source.__dict__)
        result = super()._apply(fn, recurse=recurse)
        # PyTorch swaps traceable tensor subclasses even for no-op conversions. As in
        # TE's BaseModule, restore external metadata without replacing the new storage.
        for name, value in attributes.items():
            if name in self.weight.__dict__:
                continue
            if isinstance(value, MethodType) and value.__self__ is source:
                value = MethodType(value.__func__, self.weight)
            setattr(self.weight, name, value)
        # CPU initialization may precede the model's move to its training device.
        self._quantize_weight()
        return result

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Use the bounded quantization path for high-precision checkpoint weights."""
        key = prefix + 'weight'
        source = state_dict.get(key)
        if (
            is_mxfp8tensor(self.weight)
            and isinstance(source, Tensor)
            and source.shape == self.weight.shape
        ):
            with torch.no_grad():
                copy_tensor_to_quantized_param(self.weight, source)
            # The normal loader still checks missing/unexpected keys. Give it the
            # already loaded parameter so it does not requantize the entire table.
            state_dict = state_dict.copy()
            state_dict[key] = self.weight
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _initialize_weight(self, init_method: Callable[[Tensor], None]) -> None:
        """Initialize the row shard on a forked RNG stream keyed by (table, EP rank).

        EP ranks own different row counts (prime table sizes never divide evenly), so drawing
        from the default generator would desynchronize it across EP ranks and silently diverge
        every subsequently initialized replicated parameter across data-parallel peers. The
        fork keeps the default stream untouched; the shard seed varies only with the globally
        unique prime table size and the EP rank, so TP and expert-DP replicas stay identical
        while EP shards are decorrelated.
        """
        base_seed = torch.initial_seed() % (2**31)
        shard_seed = base_seed + 100003 * self.global_num_embeddings + self.ep_rank
        devices = [self.weight.device] if self.weight.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if devices:
                torch.cuda.manual_seed(shard_seed)
            else:
                # Seed only the CPU generator: torch.manual_seed would also reseed every CUDA
                # default generator with the EP-rank-dependent shard seed, and fork_rng([])
                # restores only the CPU state.
                torch.default_generator.manual_seed(shard_seed)
            init_method(self.weight[: self.local_num_embeddings])

    @property
    def local_num_embeddings(self) -> int:
        """Number of rows owned by this rank."""
        return self.row_end - self.row_start

    def forward(self, local_row_ids: Tensor) -> Tensor:
        """Look up owner-local row IDs."""
        if self.fp8_storage:
            raise RuntimeError('Use lookup_quantized for primary MXFP8 Engram tables')
        if self.deterministic_mode:
            return _DeterministicEmbedding.apply(self.weight, local_row_ids)
        return F.embedding(local_row_ids, self.weight)

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Represent this uneven EP row shard with its exact global logical shape."""
        del metadata
        prepend_axis_num = len(sharded_offsets)
        global_shape = [1] * prepend_axis_num + [self.global_num_embeddings, self.embedding_dim]
        global_offset = [0] * prepend_axis_num + [self.row_start, 0]
        for axis, rank_offset, axis_fragmentation in sharded_offsets:
            global_shape[axis] = axis_fragmentation
            global_offset[axis] = rank_offset

        replica_id = (0, get_pg_rank(self.tp_group), get_pg_rank(self.expt_dp_group))
        key = f"{prefix}weight"
        if self.fp8_storage:
            logical_shape = (self.local_num_embeddings, self.embedding_dim)
            physical_shape = tuple(self.weight.shape)

            @torch.no_grad()
            def build(key, data, replica_id, flattened_range):
                if flattened_range is not None:
                    raise ValueError('Engram MXFP8 checkpoints require whole row-shard ownership')
                if is_mxfp8tensor(data):
                    data = dequantize_fp8_tensor(data)
                return [
                    ShardedTensor(
                        key=key,
                        data=data[: logical_shape[0]],
                        dtype=data.dtype,
                        local_shape=logical_shape,
                        global_shape=tuple(global_shape),
                        global_offset=tuple(global_offset),
                        axis_fragmentations=None,
                        replica_id=replica_id,
                        prepend_axis_num=prepend_axis_num,
                    )
                ]

            def merge(tensors):
                data = tensors[0]
                padded = data.new_zeros(physical_shape)
                padded[: logical_shape[0]].copy_(data)
                return padded

            return {key: ShardedTensorFactory(key, self.weight, build, merge, replica_id)}
        return {
            key: ShardedTensor(
                key=key,
                data=self.weight,
                dtype=self.weight.dtype,
                local_shape=tuple(self.weight.shape),
                global_shape=tuple(global_shape),
                global_offset=tuple(global_offset),
                axis_fragmentations=None,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            )
        }


class EPShardedMultiTableEmbedding(MegatronModule):
    """Batch variable-size requests for multiple independently EP-sharded tables."""

    def __init__(
        self,
        config,
        table_sizes: tuple[int, ...],
        embedding_dim: int,
        init_method: Callable[[Tensor], None],
        ep_group=None,
        tp_group=None,
        expt_dp_group=None,
        fp8_storage: bool = False,
        communication_backend: str = "nccl",
        hybridep_chunk_size: int = 4096,
    ) -> None:
        super().__init__(config=config)
        self.table_sizes = tuple(table_sizes)
        self.embedding_dim = embedding_dim
        self.ep_group = ep_group
        self.ep_size = get_pg_size(ep_group)
        self.communication = _EngramAllToAll(
            ep_group,
            communication_backend,
            hybridep_chunk_size,
            max_row_bytes=embedding_dim * config.params_dtype.itemsize,
        )
        self.tables = nn.ModuleList(
            [
                EPShardedEmbeddingTable(
                    config=config,
                    global_num_embeddings=table_size,
                    embedding_dim=embedding_dim,
                    init_method=init_method,
                    ep_group=ep_group,
                    tp_group=tp_group,
                    expt_dp_group=expt_dp_group,
                    fp8_storage=fp8_storage,
                )
                for table_size in self.table_sizes
            ]
        )

        # Per-table balanced-partition constants let forward resolve the owning EP rank of any
        # global row in closed form, without materializing per-request [N, ep_size] boundaries.
        device = self.tables[0].weight.device
        bases_and_remainders = [divmod(table_size, self.ep_size) for table_size in self.table_sizes]
        self.register_buffer(
            "table_row_base",
            torch.tensor(
                [base for base, _ in bases_and_remainders], dtype=torch.int64, device=device
            ),
            persistent=False,
        )
        self.register_buffer(
            "table_row_remainder",
            torch.tensor(
                [remainder for _, remainder in bases_and_remainders],
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )

    @property
    def local_parameter_count(self) -> int:
        """Local sparse parameter count."""
        return sum(table.local_num_embeddings * self.embedding_dim for table in self.tables)

    @property
    def global_parameter_count(self) -> int:
        """Global logical sparse parameter count."""
        return sum(self.table_sizes) * self.embedding_dim

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Preserve each table's irregular EP row metadata through the ModuleList container."""
        sharded_state_dict = {}
        for table_id, table in enumerate(self.tables):
            sharded_state_dict.update(
                table.sharded_state_dict(
                    prefix=f"{prefix}tables.{table_id}.",
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
            )
        return sharded_state_dict

    def _lookup_received_requests(self, requests: Tensor, table_counts: list[int]) -> Tensor:
        """Look up owner-local rows, grouping the received requests by table.

        ``table_counts`` comes from the request-count exchange, so no extra host
        synchronization is needed on the owner-side critical path.
        """
        order = torch.argsort(requests[:, 0], stable=True)
        sorted_rows = requests[:, 1].index_select(0, order)
        values = []
        start = 0
        for table, count in zip(self.tables, table_counts):
            values.append(table(sorted_rows.narrow(0, start, count)))
            start += count
        output = self.tables[0].weight.new_empty((requests.shape[0], self.embedding_dim))
        output.index_copy_(0, order, torch.cat(values))
        return output

    def forward(self, hash_ids: Tensor) -> Tensor:
        """Route global table rows to owners and restore token/head ordering.

        Args:
            hash_ids: Per-table row IDs with shape ``[..., num_tables]``.

        Returns:
            Retrieved embeddings with shape ``[..., num_tables, embedding_dim]``.
        """
        if hash_ids.shape[-1] != len(self.tables):
            raise ValueError(
                f"Expected {len(self.tables)} Engram hash heads, got {hash_ids.shape[-1]}."
            )
        original_shape = hash_ids.shape

        if self.ep_size == 1:
            # Per-table layout is statically known: no routing, sorting, or host sync needed.
            values = [
                table(hash_ids[..., table_id].reshape(-1).to(torch.int64))
                for table_id, table in enumerate(self.tables)
            ]
            return torch.stack(values, dim=-2).view(*original_shape, self.embedding_dim)

        received_requests, owner_order, send_splits, recv_splits, table_counts, plans = (
            self._route_requests(hash_ids)
        )
        message = "engram.lookup.local-embedding"
        nvtx_range_push(message)
        try:
            owner_embeddings = self._lookup_received_requests(received_requests, table_counts)
        finally:
            nvtx_range_pop(message)
        sorted_embeddings = _DifferentiableEngramAllToAll.apply(
            self.communication, owner_embeddings, send_splits, recv_splits, plans
        )
        output = torch.empty_like(sorted_embeddings)
        output.index_copy_(0, owner_order, sorted_embeddings)
        return output.view(*original_shape, self.embedding_dim)

    def lookup_quantized(self, hash_ids, output_width, *, needs_wgrad=True):
        """Return a token-padded MXFP8 matrix with adjacent hash heads concatenated.

        The caller trims the Linear output to ``hash_ids.numel() * embedding_dim /
        output_width`` tokens. Byte payloads and scales share one EP exchange; their
        autograd boundary returns BF16/FP32 gradients to the original table parameters.
        """
        if output_width <= 0 or output_width % self.embedding_dim:
            raise ValueError('output_width must be a positive multiple of embedding_dim')
        if hash_ids.numel() * self.embedding_dim % output_width:
            raise ValueError('Quantized Engram lookup must contain complete output rows')
        weights = tuple(table.weight for table in self.tables)
        # This path reads raw FP8 storage without calling the tables' forward hooks.
        ensure_params_ready(weights)
        return _MXFP8EngramLookup.apply(
            self, hash_ids, output_width, torch.is_grad_enabled() and needs_wgrad, *weights
        )

    def _route_requests(self, hash_ids):
        """Share the same exact row ownership and request order between BF16 and FP8."""
        if hash_ids.shape[-1] != len(self.tables):
            raise ValueError(f'Expected {len(self.tables)} Engram hash heads')
        original_shape = hash_ids.shape
        rows = hash_ids.reshape(-1).to(torch.int64)
        table_ids = (
            torch.arange(len(self.tables), device=hash_ids.device, dtype=torch.int64)
            .view(*([1] * (hash_ids.ndim - 1)), -1)
            .expand(original_shape)
            .reshape(-1)
        )
        return self._route_rows(rows, table_ids)

    def _route_rows(self, rows, table_ids, *, global_counts=False, exchange_requests=True):
        """Route arbitrary (table,row) requests, including deduplicated requests."""
        if self.ep_size == 1:
            return (
                torch.stack((table_ids, rows), dim=-1),
                torch.arange(rows.numel(), device=rows.device),
                [rows.numel()],
                [rows.numel()],
                (
                    [rows.numel()]
                    if len(self.tables) == 1
                    else torch.bincount(table_ids, minlength=len(self.tables)).cpu().tolist()
                ),
                (None, None),
            )

        message = "engram.lookup.owner-map"
        nvtx_range_push(message)
        try:
            # Balanced contiguous partition: ranks below `remainder` own `base + 1` rows and the
            # rest own `base`, so the owner of a global row follows in O(1) per request.
            base = self.table_row_base.index_select(0, table_ids)
            remainder = self.table_row_remainder.index_select(0, table_ids)
            large_region_end = (base + 1) * remainder
            owners = torch.where(
                rows < large_region_end,
                rows // (base + 1),
                remainder + (rows - large_region_end) // base.clamp_min(1),
            )
            owner_starts = owners * base + torch.minimum(owners, remainder)
            local_rows = rows - owner_starts

            owner_order = torch.argsort(owners, stable=True)
            sorted_owners = owners.index_select(0, owner_order)
            requests = torch.stack(
                (table_ids.index_select(0, owner_order), local_rows.index_select(0, owner_order)),
                dim=-1,
            )
            # Count per (peer, table) rather than per peer: the same exchange then yields the
            # all-to-all splits and the owner-side per-table counts from one host transfer.
            num_tables = len(self.tables)
            send_counts = torch.bincount(
                sorted_owners * num_tables + requests[:, 0], minlength=self.ep_size * num_tables
            ).to(torch.int64)
        finally:
            nvtx_range_pop(message)

        message = "engram.lookup.count-a2a"
        nvtx_range_push(message)
        try:
            send_splits, recv_splits, table_counts, plans = self.communication.exchange_counts(
                send_counts, global_counts=global_counts
            )
        finally:
            nvtx_range_pop(message)

        message = "engram.lookup.request-a2a"
        nvtx_range_push(message)
        try:
            received_requests = (
                self.communication.exchange(requests, recv_splits, send_splits, plans[0])
                if exchange_requests
                else requests
            )
        finally:
            nvtx_range_pop(message)

        return received_requests, owner_order, send_splits, recv_splits, table_counts, plans


@dataclass
class EngramLookupTicket:
    """One request route; payload ownership ends at lookup consumption.

    ``requests`` holds requester-local IDs for direct table reads with the
    symmetric backend, and owner-received IDs for the other transports.
    """

    module: object
    shape: tuple
    inverse: Tensor
    requests: Tensor
    order: Tensor
    send_splits: list
    recv_splits: list
    table_counts: list
    plans: tuple
    capacity: int
    generation: int
    payload: Tensor | None = None
    ready: object = None
    gradient_ready: object = None
    consumed: bool = False


class EngramRuntime:
    """Bounded stage-local row prefetch and gradient delivery.

    Parameters remain owned by the modules and DDP. This object owns only request
    routes, prefetched rows and deferred row gradients. Budget accounting uses
    group-wide upper bounds, so every shard makes the same eviction/flush choice.
    Model activations, DDP buffers and opaque backend workspaces are not charged
    to this budget. A single stream serializes reusable transport workspaces.
    """

    def __init__(self, buffer_budget_mb=1024, *, defer_grad_transfer=False):
        self.budget = buffer_budget_mb * 1024**2
        self.defer_grad_transfer = defer_grad_transfer
        self.stream = None
        self.cached = []
        self.pending = []
        self.used = 0
        self.resident = 0
        self.generation = 0
        self.peak = 0
        self.communications = {}

    def attach(self, module):
        """Share transport workspace between tables with the same row format/group."""
        transport = module.communication
        # Duplicate gradients are summed before communication, in at least FP32.
        # The legacy lookup sends model-dtype gradients and sized its workspace
        # accordingly; enlarge it before HybridEP allocates its first buffer.
        grad_bytes = module.embedding_dim * max(4, module.tables[0].weight.dtype.itemsize)
        packet_bytes = (max(16, grad_bytes) + 8 + 31) // 32 * 32
        if packet_bytes > transport.max_packet_bytes:
            if transport.buffer is not None:
                raise RuntimeError('Attach the Engram runtime before the first lookup')
            transport.max_packet_bytes = packet_bytes
        key = (transport.group, transport.backend, transport.chunk_size, transport.max_packet_bytes)
        if key not in self.communications and transport.backend == 'symmetric':
            transport.workspace_bytes = self.budget // 4
            self.resident += transport.workspace_bytes
            if self.resident >= self.budget:
                raise ValueError('Engram runtime budget cannot hold all transport workspaces')
            self.used += transport.workspace_bytes
        module.communication = self.communications.setdefault(key, transport)
        module.runtime = self
        if self.defer_grad_transfer:
            for table in module.tables:
                table.weight.deferred_grad_producer = self

    def _stream_context(self, device):
        if device.type != 'cuda':
            return nullcontext()
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        return torch.cuda.stream(self.stream)

    def _reserve(self, size):
        if size + self.resident > self.budget:
            raise RuntimeError(
                f'Engram lookup needs {size / 1024**2:.1f} MiB of runtime buffers; '
                f'increase buffer_budget_mb (currently {self.budget // 1024**2})'
            )
        if self.used + size > self.budget:
            self.finish_deferred_gradients()
        while self.used + size > self.budget and self.cached:
            # Only optional prefetch payloads are evicted. Routes remain valid and
            # can fetch the same parameter version when their consumer arrives.
            old = self.cached.pop()
            if old.payload is not None:
                self._release_payload(old)
        if self.used + size > self.budget:
            raise RuntimeError('Engram runtime has unreleased buffers')
        self.used += size
        self.peak = max(self.peak, self.used)

    @torch.no_grad()
    def prefetch(self, module, hash_ids):
        """Deduplicate requests and fetch rows before the hidden-state consumer."""
        weights = tuple(table.weight for table in module.tables)
        ensure_params_ready(weights)
        if hash_ids.shape[-1] != len(weights):
            raise ValueError('Engram prefetch expects one final dimension per table')
        # Encode (table,row) without materializing an [N, num_shards] comparison.
        stride = max(module.table_sizes)
        tables = (
            torch.arange(len(weights), device=hash_ids.device).expand(hash_ids.shape).reshape(-1)
        )
        unique, inverse = torch.unique(
            hash_ids.reshape(-1).long() + tables * stride, sorted=True, return_inverse=True
        )
        with self._stream_context(hash_ids.device):
            if unique.is_cuda:
                unique.record_stream(self.stream)
            requests, order, send, recv, counts, plans = module._route_rows(
                unique % stride,
                unique // stride,
                global_counts=True,
                exchange_requests=module.communication.backend != "symmetric",
            )
        max_send = plans[0].max_rows if module.ep_size > 1 else unique.numel()
        max_recv = plans[1].max_rows if module.ep_size > 1 else unique.numel()
        # Covers both payload directions and FP32 unique-row gradients. The route
        # itself survives in autograd, like the old lookup's saved row IDs.
        grad_bytes = module.embedding_dim * max(4, weights[0].dtype.itemsize)
        capacity = (max_send + max_recv) * (grad_bytes + 16)
        if module.communication.backend == "symmetric":
            # The gradient and its opaque (row ID, gradient) packet coexist.
            capacity += (max_send + max_recv) * grad_bytes
        ticket = EngramLookupTicket(
            module,
            tuple(hash_ids.shape),
            inverse,
            requests,
            order,
            send,
            recv,
            counts,
            plans,
            capacity,
            self.generation,
        )
        self._materialize(ticket, required=False)
        return ticket

    @torch.no_grad()
    def _materialize(self, ticket, *, required=True):
        # Lookahead must not evict a nearer consumer just to prefetch a later
        # microbatch. Keep the route and fetch its payload at consumption instead.
        if not required and self.used + ticket.capacity > self.budget:
            return
        self._reserve(ticket.capacity)
        module = ticket.module
        weights = tuple(table.weight for table in module.tables)
        ensure_params_ready(weights)
        with self._stream_context(ticket.requests.device):
            if module.communication.backend == "symmetric":
                payload = module.communication.lookup(module, ticket)
            elif module.tables[0].fp8_storage:
                order = (
                    torch.argsort(ticket.requests[:, 0], stable=True) if len(weights) > 1 else None
                )
                rows = ticket.requests[:, 1]
                if order is not None:
                    rows = rows.index_select(0, order)
                payload = _mxfp8_pack_lookup(
                    weights, rows, order, ticket.table_counts, module.embedding_dim
                )
            else:
                payload = module._lookup_received_requests(ticket.requests, ticket.table_counts)
            if module.ep_size > 1 and module.communication.backend != "symmetric":
                payload = module.communication.exchange(
                    payload, ticket.send_splits, ticket.recv_splits, ticket.plans[1]
                )
            ticket.payload = payload
            if payload.is_cuda:
                ticket.ready = torch.cuda.Event()
                ticket.ready.record(self.stream)
        self.cached.append(ticket)

    def _release_payload(self, ticket):
        if ticket.payload is not None:
            if ticket.payload.is_cuda:
                ticket.payload.record_stream(torch.cuda.current_stream(ticket.payload.device))
            ticket.payload = None
            self.used -= ticket.capacity

    def consume(self, ticket, output_width=None, *, needs_wgrad=True):
        """Attach the table-gradient edge exactly once, outside recompute/capture."""
        if ticket.generation != self.generation:
            raise RuntimeError('Engram lookup ticket belongs to an earlier training step')
        if ticket.consumed:
            raise RuntimeError('Engram lookup ticket has already been consumed')
        if ticket.payload is None:
            self._materialize(ticket)
        if ticket.ready is not None:
            consumer = torch.cuda.current_stream(ticket.payload.device)
            consumer.wait_event(ticket.ready)
            # Forward-only execution may release the ticket immediately after
            # launching the inverse permutation on the consumer stream.
            ticket.order.record_stream(consumer)
            ticket.inverse.record_stream(consumer)
        weights = tuple(table.weight for table in ticket.module.tables)
        output = _PrefetchedEngramLookup.apply(
            self, ticket, output_width, torch.is_grad_enabled() and needs_wgrad, *weights
        )
        ticket.consumed = True
        self.cached = [entry for entry in self.cached if entry is not ticket]
        self._release_payload(ticket)
        return output

    def enqueue(self, ticket, gradient):
        """Retain unique-row gradients until pressure or gradient finalization."""
        self._reserve(ticket.capacity)
        if gradient.is_cuda:
            ticket.gradient_ready = torch.cuda.Event()
            ticket.gradient_ready.record(torch.cuda.current_stream(gradient.device))
        self.pending.append((ticket, gradient))

    @torch.no_grad()
    def _deliver(self, ticket, gradient, *, accumulate):
        module = ticket.module
        requests = ticket.requests
        if module.ep_size > 1:
            if module.communication.backend == "symmetric":
                # Direct LSA lookup never sent IDs to the owner. Send each ID
                # together with its already locally reduced FP32 gradient.
                raw = gradient.contiguous().view(torch.uint8)
                packet = torch.cat((requests.contiguous().view(torch.uint8), raw), dim=1)
                received = module.communication.exchange(
                    packet, ticket.recv_splits, ticket.send_splits, ticket.plans[0]
                )
                requests = received[:, :16].contiguous().view(torch.int64)
                gradient = received[:, 16:].contiguous().view(gradient.dtype)
            else:
                gradient = module.communication.exchange(
                    gradient, ticket.recv_splits, ticket.send_splits, ticket.plans[0]
                )
        rows = requests[:, 1]
        if len(module.tables) > 1:
            table_order = torch.argsort(requests[:, 0], stable=True)
            rows = rows.index_select(0, table_order)
            gradient = gradient.index_select(0, table_order)
        result, start = [], 0
        for table, count in zip(module.tables, ticket.table_counts):
            weight = table.weight
            local_rows, local_grad = rows[start : start + count], gradient[start : start + count]
            if accumulate:
                target = getattr(weight, 'main_grad', None)
                if target is None:
                    if weight.grad is None:
                        weight.grad = torch.zeros(
                            weight.shape, dtype=weight.dtype, device=weight.device
                        )
                    target = weight.grad
                _accumulate_embedding_rows(target, local_rows, local_grad)
                weight.grad_added_to_main_grad = True
            else:
                result.append(
                    _embedding_row_gradients(local_rows, local_grad, weight.shape, weight.dtype)
                )
            start += count
        return result

    @torch.no_grad()
    def _deliver_batch(self, entries):
        """Coalesce queued microbatches into one gradient exchange per table module."""
        if len(entries) == 1:
            return self._deliver(*entries[0], accumulate=True)
        module = entries[0][0].module
        size = module.ep_size
        send = [sum(ticket.send_splits[peer] for ticket, _ in entries) for peer in range(size)]
        recv = [sum(ticket.recv_splits[peer] for ticket, _ in entries) for peer in range(size)]
        send_parts = [gradient.split(ticket.send_splits) for ticket, gradient in entries]
        recv_parts = [
            ticket.requests.split(
                ticket.send_splits
                if module.communication.backend == "symmetric"
                else ticket.recv_splits
            )
            for ticket, _ in entries
        ]
        gradient = torch.cat([parts[peer] for peer in range(size) for parts in send_parts])
        requests = torch.cat([parts[peer] for peer in range(size) for parts in recv_parts])
        plan = None
        if size > 1:
            counts = sum(ticket.plans[0].peer_counts for ticket, _ in entries)
            plan = _EngramRowExchangePlan(int(counts.sum(1).max()), counts)
        ticket = replace(
            entries[0][0],
            requests=requests,
            send_splits=send,
            recv_splits=recv,
            plans=(plan, None),
            table_counts=[
                sum(ticket.table_counts[i] for ticket, _ in entries)
                for i in range(len(module.tables))
            ],
        )
        return self._deliver(ticket, gradient, accumulate=True)

    def finish_deferred_gradients(self):
        """Deliver all row gradients before DDP reduction, clipping or optimizer step."""
        pending, self.pending = self.pending, []
        by_module = {}
        for ticket, gradient in pending:
            by_module.setdefault(ticket.module, []).append((ticket, gradient))
        for entries in by_module.values():
            with self._stream_context(entries[0][1].device):
                for ticket, _ in entries:
                    if ticket.gradient_ready is not None:
                        self.stream.wait_event(ticket.gradient_ready)
                self._deliver_batch(entries)
                for ticket, gradient in entries:
                    if gradient.is_cuda:
                        gradient.record_stream(self.stream)
                    self.used -= ticket.capacity
        # Pressure may already have flushed the last pending entry on a backward
        # stream. DDP must still join that work before publishing main_grad.
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

    def reset(self):
        """Discard the prior schedule's tickets on the training loop's zero-grad boundary."""
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        for ticket in self.cached:
            ticket.payload = None
        self.cached.clear()
        self.pending.clear()
        self.used = self.resident
        self.generation += 1

    def close(self):
        """Finish queued gradients and release device transports before group teardown."""
        self.finish_deferred_gradients()
        if self.stream is not None:
            self.stream.synchronize()
        for transport in self.communications.values():
            if transport.device_transport is not None:
                transport.device_transport.close()


def _accumulate_embedding_rows(target, rows, gradients):
    """Deterministically sum duplicates in FP32 and add only touched rows to main_grad."""
    if not rows.numel():
        return
    order = torch.argsort(rows, stable=True)
    if (
        HAVE_TRITON
        and target.is_cuda
        and target.dtype != torch.float64
        and gradients.dtype != torch.float64
        and target.stride(1) == gradients.stride(1) == 1
    ):
        _engram_accumulate_rows_kernel[(rows.numel(),)](
            target,
            rows,
            order,
            gradients,
            rows.numel(),
            target.shape[1],
            target.stride(0),
            gradients.stride(0),
            rows.stride(0),
            triton.next_power_of_2(target.shape[1]),
            16,
        )
        return
    ids, counts = torch.unique_consecutive(rows[order], return_counts=True)
    values = gradients[order].to(torch.float64 if target.dtype == torch.float64 else torch.float32)
    values = torch.segment_reduce(values, 'sum', lengths=counts)
    target.index_copy_(0, ids, (target.index_select(0, ids) + values).to(target.dtype))


class _PrefetchedEngramLookup(torch.autograd.Function):
    """Bridge a no-grad row fetch to the owning table parameters."""

    @staticmethod
    def forward(ctx, runtime, ticket, output_width, needs_wgrad, *weights):
        inverse_order = torch.empty_like(ticket.order)
        inverse_order[ticket.order] = torch.arange(ticket.order.numel(), device=ticket.order.device)
        mapping = inverse_order[ticket.inverse]
        ctx.runtime, ctx.ticket = runtime, ticket
        ctx.save_for_backward(mapping)
        module = ticket.module
        if output_width is not None:
            return _mxfp8_lookup_activation(
                ticket.payload,
                module.embedding_dim,
                output_width,
                weights[0].dtype,
                needs_wgrad,
                inverse=mapping,
            )
        return ticket.payload.index_select(0, mapping).view(*ticket.shape, module.embedding_dim)

    @staticmethod
    def backward(ctx, grad_output):
        ticket, module = ctx.ticket, ctx.ticket.module
        (mapping,) = ctx.saved_tensors
        count, width = mapping.numel(), module.embedding_dim
        grad = grad_output.reshape(-1)[: count * width].view(count, width)
        dtype = torch.float64 if grad.dtype == torch.float64 else torch.float32
        unique_grad = torch.zeros((ticket.order.numel(), width), device=grad.device, dtype=dtype)
        _accumulate_embedding_rows(unique_grad, mapping, grad)
        if ctx.runtime.defer_grad_transfer:
            ctx.runtime.enqueue(ticket, unique_grad)
            gradients = [None] * len(module.tables)
        else:
            with ctx.runtime._stream_context(grad.device):
                gradients = ctx.runtime._deliver(ticket, unique_grad, accumulate=False)
                if unique_grad.is_cuda:
                    unique_grad.record_stream(ctx.runtime.stream)
            if unique_grad.is_cuda:
                consumer = torch.cuda.current_stream(grad.device)
                consumer.wait_stream(ctx.runtime.stream)
                for gradient in gradients:
                    gradient.record_stream(consumer)
        return None, None, None, None, *gradients
