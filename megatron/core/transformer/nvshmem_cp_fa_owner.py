# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in NVSHMEM saved-state attention backend for the validated CP=4 contract."""

from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.nn.functional as F

_FA4_FWD = _FA4_BWD = _FA4_TWO_SECTION_BWD = None
if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_GLOBAL") == "1":
    import cutlass as _fa4_cutlass
    import cutlass.cute as _fa4_cute
    from flash_attn.cute.interface import (
        _flash_attn_bwd as _FA4_BWD,
        _flash_attn_fwd as _FA4_FWD,
    )
    try:
        from flash_attn.cute.interface import (
            _flash_attn_bwd_two_section_causal as _FA4_TWO_SECTION_BWD,
        )
    except ImportError:
        _FA4_TWO_SECTION_BWD = None

from megatron.core.cp_timing import cp_timing_enabled, record_cp_timing, tensor_meta
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.nvshmem_cp_attention import (
    _ensure_nvshmem_initialized,
    _next_grad_return_epoch,
    _publish_block_ready_epoch,
    _sync_timeout_seconds,
    _wait_for_block_ready_epoch,
    _wait_for_grad_committed_epoch,
    _workspace,
)
from megatron.core.typed_torch import apply_module

_NATIVE_FUSED_BACKWARD_CALL_COUNTERS: dict[tuple[int, int], int] = {}
_BRANCH_B_PIPELINE_COPY_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}
_BRANCH_B_PIPELINE_COMPUTE_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}
_BRANCH_B_DUAL_SECTION_BWD_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}
_BRANCH_B_DEFERRED_PHASE_EVENTS: list[dict[str, object]] = []
_BRANCH_B_DEFERRED_FORWARD_DETAIL_EVENTS: list[dict[str, object]] = []
_BRANCH_B_FA4_MASKS: dict[tuple[int, int, int], object] = {}
_BRANCH_B_FA4_PREWARMED: set[tuple] = set()
_BRANCH_B_FA4_BLOCK_SPARSE: dict[tuple, tuple] = {}


class NvshmemCpFaOwnerError(RuntimeError):
    """Raised when the experimental FA-owner path cannot run."""


def _import_te_for_native_fused_owner():
    import transformer_engine_torch as tex

    return tex


def _fa_owner_io_v1_block_ready_native_fused(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask_type: AttnMaskType,
    layer_number: int | None,
) -> torch.Tensor:
    """Fail-closed route for the production Branch-B native fused owner.

    The native owner consumes block-ready symmetric peer K/V and owns forward
    attention, backward dQ/dK/dV, and remote K/V gradient return.
    """

    if os.getenv("MEGATRON_NVSHMEM_CP_BLOCK_READY_PROTOCOL") != "1":
        raise NvshmemCpFaOwnerError(
            "block_ready_native_fused requires MEGATRON_NVSHMEM_CP_BLOCK_READY_PROTOCOL=1."
        )
    if attn_mask_type != AttnMaskType.causal:
        raise NvshmemCpFaOwnerError(
            f"block_ready_native_fused currently targets causal CP attention, got {attn_mask_type}."
        )
    return _BlockReadyNativeFusedOwnerAutograd.apply(query, key, value, True, layer_number)


def _branch_b_section_for_peer(peer: int, rank: int) -> str:
    if peer == rank:
        return "diagonal"
    if peer < rank:
        return "lower-triangle"
    return "upper-triangle"


def _branch_b_pipeline_copy_stream(
    device: torch.device, lane: int = 0
) -> torch.cuda.Stream:
    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    key = (device_index, int(lane))
    stream = _BRANCH_B_PIPELINE_COPY_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _BRANCH_B_PIPELINE_COPY_STREAMS[key] = stream
    return stream


def _branch_b_pipeline_compute_stream(
    device: torch.device, lane: int
) -> torch.cuda.Stream:
    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    key = (device_index, int(lane))
    stream = _BRANCH_B_PIPELINE_COMPUTE_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _BRANCH_B_PIPELINE_COMPUTE_STREAMS[key] = stream
    return stream


def _branch_b_dual_section_bwd_stream(
    device: torch.device, lane: int
) -> torch.cuda.Stream:
    device_index = int(device.index if device.index is not None else torch.cuda.current_device())
    key = (device_index, int(lane))
    stream = _BRANCH_B_DUAL_SECTION_BWD_STREAMS.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _BRANCH_B_DUAL_SECTION_BWD_STREAMS[key] = stream
    return stream


def _branch_b_two_prefix_inputs(
    query: torch.Tensor,
    peer_keys: list[torch.Tensor],
    peer_values: list[torch.Tensor],
    cp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack the two global causal prefixes consumed by one native CP rank."""

    if len(peer_keys) != 4 or len(peer_values) != 4 or not 0 <= cp_rank < 4:
        raise NvshmemCpFaOwnerError("Two-prefix Branch-B currently requires CP=4.")
    if int(query.shape[0]) % 2 != 0:
        raise NvshmemCpFaOwnerError("Two-prefix Branch-B requires even local sequence length.")
    half = int(query.shape[0]) // 2
    q_first = query[:half].contiguous()
    q_second = query[half:].contiguous()
    first_k = torch.cat([peer_keys[owner][:half] for owner in range(cp_rank + 1)])
    first_v = torch.cat([peer_values[owner][:half] for owner in range(cp_rank + 1)])
    second_owners = list(range(3, cp_rank - 1, -1))
    second_k = torch.cat(
        [peer_keys[owner][:half] for owner in range(4)]
        + [peer_keys[owner][half:] for owner in second_owners]
    )
    second_v = torch.cat(
        [peer_values[owner][:half] for owner in range(4)]
        + [peer_values[owner][half:] for owner in second_owners]
    )
    return q_first, q_second, first_k, first_v, second_k, second_v


def _branch_b_grouped_sections_forward(
    *,
    tex,
    query: torch.Tensor,
    peer_keys: list[torch.Tensor],
    peer_values: list[torch.Tensor],
    cp_rank: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Batch independent same-shape peer sections without changing merge semantics."""

    if len(peer_keys) != 4 or len(peer_values) != 4 or not 0 <= cp_rank < 4:
        raise NvshmemCpFaOwnerError("Grouped-section Branch-B currently requires CP=4.")
    half = int(query.shape[0]) // 2
    scale = 1.0 / math.sqrt(float(query.shape[-1]))
    layout = tex.NVTE_QKV_Layout.NVTE_SBHD_SBHD_SBHD
    mask = tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK
    outputs: list[torch.Tensor | None] = [None] * 4
    lses: list[torch.Tensor | None] = [None] * 4
    rngs: list[torch.Tensor | None] = [None] * 4
    sections = [_branch_b_section_for_peer(peer, cp_rank) for peer in range(4)]

    diagonal = tex.nvshmem_cp_forward_section_fused_attn_execute(
        query,
        peer_keys[cp_rank],
        peer_values[cp_rank],
        int(query.shape[0]),
        int(query.shape[0]),
        cp_rank,
        4,
        cp_rank,
        "sbhd",
        "diagonal",
        float(scale),
        0.0,
        layout,
        mask,
        query.dtype,
        True,
    )
    outputs[cp_rank], lses[cp_rank], rngs[cp_rank] = diagonal[:3]

    lower = list(range(cp_rank))
    if lower:
        grouped = tex.nvshmem_cp_grouped_noncausal_fused_attn_forward_execute(
            torch.cat([query] * len(lower), dim=1),
            torch.cat([peer_keys[peer][:half] for peer in lower], dim=1),
            torch.cat([peer_values[peer][:half] for peer in lower], dim=1),
            "sbhd",
            float(scale),
            query.dtype,
            True,
        )
        for batch_index, peer in enumerate(lower):
            outputs[peer] = grouped[0][:, batch_index : batch_index + 1]
            lses[peer] = grouped[1][batch_index : batch_index + 1]
            rngs[peer] = grouped[2]

    upper = list(range(cp_rank + 1, 4))
    if upper:
        grouped = tex.nvshmem_cp_grouped_noncausal_fused_attn_forward_execute(
            torch.cat([query[half:]] * len(upper), dim=1),
            torch.cat([peer_keys[peer] for peer in upper], dim=1),
            torch.cat([peer_values[peer] for peer in upper], dim=1),
            "sbhd",
            float(scale),
            query.dtype,
            True,
        )
        for batch_index, peer in enumerate(upper):
            outputs[peer] = grouped[0][:, batch_index : batch_index + 1]
            lses[peer] = grouped[1][batch_index : batch_index + 1]
            rngs[peer] = grouped[2]

    if any(item is None for item in (*outputs, *lses, *rngs)):
        raise NvshmemCpFaOwnerError("Grouped-section Branch-B missed a peer result.")
    merged = tex.nvshmem_cp_forward_lse_merge(
        outputs, lses, sections, int(query.shape[0]), "sbhd"
    )
    forward_aux: list[torch.Tensor] = [merged[1]]
    for peer in range(4):
        forward_aux.extend((lses[peer], rngs[peer]))
    return merged[0], forward_aux


def _branch_b_two_prefix_backward(
    *,
    tex,
    query: torch.Tensor,
    peer_keys: list[torch.Tensor],
    peer_values: list[torch.Tensor],
    output: torch.Tensor,
    grad_output: torch.Tensor,
    forward_aux: list[torch.Tensor],
    ws,
    cp_rank: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run two prefix backwards, scatter dK/dV by owner, and publish symmetric slots."""

    if len(forward_aux) != 4:
        raise NvshmemCpFaOwnerError(
            f"Two-prefix Branch-B backward expects four aux tensors, got {len(forward_aux)}."
        )
    q_first, q_second, first_k, first_v, second_k, second_v = (
        _branch_b_two_prefix_inputs(query, peer_keys, peer_values, cp_rank)
    )
    half = int(query.shape[0]) // 2
    first = tex.nvshmem_cp_prefix_fused_attn_backward_execute(
        q_first,
        first_k,
        first_v,
        output[:half].contiguous(),
        grad_output[:half].contiguous(),
        forward_aux[0],
        forward_aux[1],
        "sbhd",
        float(softmax_scale),
        False,
    )
    second = tex.nvshmem_cp_prefix_fused_attn_backward_execute(
        q_second,
        second_k,
        second_v,
        output[half:].contiguous(),
        grad_output[half:].contiguous(),
        forward_aux[2],
        forward_aux[3],
        "sbhd",
        float(softmax_scale),
        False,
    )
    if len(first) < 3 or len(second) < 3:
        raise NvshmemCpFaOwnerError("Two-prefix TE backward did not return dQ/dK/dV.")
    dq = torch.cat((first[0], second[0]), dim=0)
    dk_by_owner = [torch.zeros_like(query) for _ in range(4)]
    dv_by_owner = [torch.zeros_like(query) for _ in range(4)]
    for owner in range(cp_rank + 1):
        start = owner * half
        dk_by_owner[owner][:half].add_(first[1][start : start + half])
        dv_by_owner[owner][:half].add_(first[2][start : start + half])
    for owner in range(4):
        start = owner * half
        dk_by_owner[owner][:half].add_(second[1][start : start + half])
        dv_by_owner[owner][:half].add_(second[2][start : start + half])
    for index, owner in enumerate(range(3, cp_rank - 1, -1), start=4):
        start = index * half
        dk_by_owner[owner][half:].add_(second[1][start : start + half])
        dv_by_owner[owner][half:].add_(second[2][start : start + half])
    dk, dv = tex.nvshmem_cp_prefix_grad_return_execute(
        dk_by_owner,
        dv_by_owner,
        peer_keys[cp_rank],
        peer_values[cp_rank],
        ws.grad_key_return,
        ws.grad_value_return,
        ws.grad_committed_epoch,
        ws.peer_grad_key_returns,
        ws.peer_grad_value_returns,
        ws.peer_grad_committed_epochs,
        4,
        cp_rank,
    )
    return dq, dk, dv


def _branch_b_concurrent_sections_backward(
    *,
    tex,
    query: torch.Tensor,
    peer_keys: list[torch.Tensor],
    peer_values: list[torch.Tensor],
    output: torch.Tensor,
    grad_output: torch.Tensor,
    forward_aux: list[torch.Tensor],
    ws,
    cp_rank: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute independent peer sections on separate streams, then assemble dQ/dKV."""

    if len(forward_aux) != 9:
        raise NvshmemCpFaOwnerError(
            f"Concurrent Branch-B backward expects nine aux tensors, got {len(forward_aux)}."
        )
    if query.dtype != torch.bfloat16:
        raise NvshmemCpFaOwnerError("Concurrent Branch-B backward currently requires bf16.")

    current_stream = torch.cuda.current_stream(query.device)
    inputs_ready = torch.cuda.Event(enable_timing=False)
    inputs_ready.record(current_stream)
    layout = tex.NVTE_QKV_Layout.NVTE_SBHD_SBHD_SBHD
    mask = tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK
    results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [None] * 4
    done_events: list[torch.cuda.Event] = []

    for peer in range(4):
        section_stream = _branch_b_pipeline_compute_stream(query.device, peer)
        section = _branch_b_section_for_peer(peer, cp_rank)
        with torch.cuda.stream(section_stream):
            section_stream.wait_event(inputs_ready)
            native = tex.nvshmem_cp_block_ready_fused_backward_section_execute(
                query,
                peer_keys[peer],
                peer_values[peer],
                output,
                grad_output,
                forward_aux[0],
                forward_aux[1 + 2 * peer + 1],
                int(query.shape[0]),
                int(query.shape[0]),
                peer,
                4,
                cp_rank,
                "sbhd",
                section,
                float(softmax_scale),
                0.0,
                layout,
                mask,
                query.dtype,
                tex.DType.kBFloat16,
                False,
            )
            result = (native[0], native[1], native[2])
            for tensor in result:
                tensor.record_stream(section_stream)
            done = torch.cuda.Event(enable_timing=False)
            done.record(section_stream)
            done_events.append(done)
            results[peer] = result

    for done in done_events:
        current_stream.wait_event(done)

    dq = torch.zeros_like(query)
    dk_by_owner = [torch.zeros_like(query) for _ in range(4)]
    dv_by_owner = [torch.zeros_like(query) for _ in range(4)]
    half = int(query.shape[0]) // 2
    for peer, result in enumerate(results):
        if result is None:
            raise NvshmemCpFaOwnerError(f"Missing concurrent backward result for peer {peer}.")
        section = _branch_b_section_for_peer(peer, cp_rank)
        if section == "upper-triangle":
            dq[half:].add_(result[0])
        else:
            dq.add_(result[0])
        if section == "lower-triangle":
            dk_by_owner[peer][:half].copy_(result[1])
            dv_by_owner[peer][:half].copy_(result[2])
        else:
            dk_by_owner[peer].copy_(result[1])
            dv_by_owner[peer].copy_(result[2])

    dk, dv = tex.nvshmem_cp_prefix_grad_return_execute(
        dk_by_owner,
        dv_by_owner,
        peer_keys[cp_rank],
        peer_values[cp_rank],
        ws.grad_key_return,
        ws.grad_value_return,
        ws.grad_committed_epoch,
        ws.peer_grad_key_returns,
        ws.peer_grad_value_returns,
        ws.peer_grad_committed_epochs,
        4,
        cp_rank,
    )
    return dq, dk, dv


def _branch_b_grouped_sections_backward(
    *,
    tex,
    query: torch.Tensor,
    peer_keys: list[torch.Tensor],
    peer_values: list[torch.Tensor],
    output: torch.Tensor,
    grad_output: torch.Tensor,
    forward_aux: list[torch.Tensor],
    ws,
    cp_rank: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch same-shape lower and upper peer sections during backward."""

    if len(forward_aux) != 9 or query.dtype != torch.bfloat16:
        raise NvshmemCpFaOwnerError(
            "Grouped Branch-B backward requires bf16 and nine forward aux tensors."
        )
    half = int(query.shape[0]) // 2
    merged_lse = forward_aux[0]
    layout = tex.NVTE_QKV_Layout.NVTE_SBHD_SBHD_SBHD
    mask = tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK
    dq = torch.zeros_like(query)
    dk_by_owner = [torch.zeros_like(query) for _ in range(4)]
    dv_by_owner = [torch.zeros_like(query) for _ in range(4)]
    timing_meta = {
        "cp_rank": int(cp_rank),
        "cp_world_size": 4,
        "query": tensor_meta(query),
    }

    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_TRUE_FUSED_BACKWARD_NO_TE_BWD") == "1":
        helper = getattr(
            tex,
            "nvshmem_cp_branch_b_true_fused_backward_no_te_bwd_execute",
            None,
        )
        if helper is None:
            raise NvshmemCpFaOwnerError(
                "Branch-B true fused backward selected, but TE does not expose "
                "nvshmem_cp_branch_b_true_fused_backward_no_te_bwd_execute. "
                "This path is intentionally fail-closed: the next production candidate "
                "must avoid TE grouped/section backward calls rather than falling back to "
                "grouped-call helpers."
            )
        fused = helper(
            query,
            peer_keys,
            peer_values,
            output,
            grad_output,
            forward_aux,
            int(cp_rank),
            "sbhd",
            float(softmax_scale),
            False,
        )
        if len(fused) != 9:
            raise NvshmemCpFaOwnerError(
                "Branch-B true fused backward TE helper must return "
                "[dq, dk0, dk1, dk2, dk3, dv0, dv1, dv2, dv3]."
            )
        dq = fused[0]
        dk_by_owner = list(fused[1:5])
        dv_by_owner = list(fused[5:9])
        dk, dv = tex.nvshmem_cp_prefix_grad_return_execute(
            dk_by_owner,
            dv_by_owner,
            peer_keys[cp_rank],
            peer_values[cp_rank],
            ws.grad_key_return,
            ws.grad_value_return,
            ws.grad_committed_epoch,
            ws.peer_grad_key_returns,
            ws.peer_grad_value_returns,
            ws.peer_grad_committed_epochs,
            4,
            cp_rank,
        )
        return dq, dk, dv

    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_GROUPED_BACKWARD_NO_MATERIALIZE") == "1":
        helper = getattr(
            tex,
            "nvshmem_cp_grouped_noncausal_fused_attn_backward_no_materialize_execute",
            None,
        )
        if helper is None:
            raise NvshmemCpFaOwnerError(
                "Branch-B no-materialize grouped backward selected, but TE does not expose "
                "nvshmem_cp_grouped_noncausal_fused_attn_backward_no_materialize_execute. "
                "This path is intentionally fail-closed so it cannot fall back to the "
                "falsified Python torch.cat materialization path."
            )
        grouped = helper(
            query,
            peer_keys,
            peer_values,
            output,
            grad_output,
            forward_aux,
            int(cp_rank),
            "sbhd",
            float(softmax_scale),
            False,
        )
        if len(grouped) != 9:
            raise NvshmemCpFaOwnerError(
                "Branch-B no-materialize grouped backward TE helper must return "
                "[dq, dk0, dk1, dk2, dk3, dv0, dv1, dv2, dv3]."
            )
        dq = grouped[0]
        dk_by_owner = list(grouped[1:5])
        dv_by_owner = list(grouped[5:9])
        dk, dv = tex.nvshmem_cp_prefix_grad_return_execute(
            dk_by_owner,
            dv_by_owner,
            peer_keys[cp_rank],
            peer_values[cp_rank],
            ws.grad_key_return,
            ws.grad_value_return,
            ws.grad_committed_epoch,
            ws.peer_grad_key_returns,
            ws.peer_grad_value_returns,
            ws.peer_grad_committed_epochs,
            4,
            cp_rank,
        )
        return dq, dk, dv

    phase = _phase_start()
    diagonal = tex.nvshmem_cp_block_ready_fused_backward_section_execute(
        query,
        peer_keys[cp_rank],
        peer_values[cp_rank],
        output,
        grad_output,
        merged_lse,
        forward_aux[1 + 2 * cp_rank + 1],
        int(query.shape[0]),
        int(query.shape[0]),
        cp_rank,
        4,
        cp_rank,
        "sbhd",
        "diagonal",
        float(softmax_scale),
        0.0,
        layout,
        mask,
        query.dtype,
        tex.DType.kBFloat16,
        False,
    )
    _phase_end(
        phase,
        "branch_b_grouped_backward_diagonal_section",
        section="diagonal",
        **timing_meta,
    )
    phase = _phase_start()
    dq.add_(diagonal[0])
    dk_by_owner[cp_rank].copy_(diagonal[1])
    dv_by_owner[cp_rank].copy_(diagonal[2])
    _phase_end(
        phase,
        "branch_b_grouped_backward_diagonal_scatter",
        section="diagonal",
        **timing_meta,
    )

    lower = list(range(cp_rank))
    if lower:
        count = len(lower)
        phase = _phase_start()
        lower_q = torch.cat([query] * count, dim=1)
        lower_k = torch.cat([peer_keys[peer][:half] for peer in lower], dim=1)
        lower_v = torch.cat([peer_values[peer][:half] for peer in lower], dim=1)
        lower_out = torch.cat([output] * count, dim=1)
        lower_grad_output = torch.cat([grad_output] * count, dim=1)
        lower_lse = torch.cat([merged_lse] * count, dim=0)
        _phase_end(
            phase,
            "branch_b_grouped_backward_lower_materialize",
            section="lower-triangle",
            peer_count=int(count),
            materialized_q=tensor_meta(lower_q),
            materialized_k=tensor_meta(lower_k),
            **timing_meta,
        )
        phase = _phase_start()
        grouped = tex.nvshmem_cp_grouped_noncausal_fused_attn_backward_execute(
            lower_q,
            lower_k,
            lower_v,
            lower_out,
            lower_grad_output,
            lower_lse,
            forward_aux[1 + 2 * lower[0] + 1],
            "sbhd",
            float(softmax_scale),
            False,
        )
        _phase_end(
            phase,
            "branch_b_grouped_backward_lower_te_bwd",
            section="lower-triangle",
            peer_count=int(count),
            **timing_meta,
        )
        phase = _phase_start()
        dq.add_(grouped[0].sum(dim=1, keepdim=True))
        for batch_index, peer in enumerate(lower):
            dk_by_owner[peer][:half].copy_(grouped[1][:, batch_index : batch_index + 1])
            dv_by_owner[peer][:half].copy_(grouped[2][:, batch_index : batch_index + 1])
        _phase_end(
            phase,
            "branch_b_grouped_backward_lower_scatter",
            section="lower-triangle",
            peer_count=int(count),
            **timing_meta,
        )

    upper = list(range(cp_rank + 1, 4))
    if upper:
        count = len(upper)
        phase = _phase_start()
        upper_q = torch.cat([query[half:]] * count, dim=1)
        upper_k = torch.cat([peer_keys[peer] for peer in upper], dim=1)
        upper_v = torch.cat([peer_values[peer] for peer in upper], dim=1)
        upper_out = torch.cat([output[half:]] * count, dim=1)
        upper_grad_output = torch.cat([grad_output[half:]] * count, dim=1)
        upper_lse = torch.cat([merged_lse[:, :, half:]] * count, dim=0)
        _phase_end(
            phase,
            "branch_b_grouped_backward_upper_materialize",
            section="upper-triangle",
            peer_count=int(count),
            materialized_q=tensor_meta(upper_q),
            materialized_k=tensor_meta(upper_k),
            **timing_meta,
        )
        phase = _phase_start()
        grouped = tex.nvshmem_cp_grouped_noncausal_fused_attn_backward_execute(
            upper_q,
            upper_k,
            upper_v,
            upper_out,
            upper_grad_output,
            upper_lse,
            forward_aux[1 + 2 * upper[0] + 1],
            "sbhd",
            float(softmax_scale),
            False,
        )
        _phase_end(
            phase,
            "branch_b_grouped_backward_upper_te_bwd",
            section="upper-triangle",
            peer_count=int(count),
            **timing_meta,
        )
        phase = _phase_start()
        dq[half:].add_(grouped[0].sum(dim=1, keepdim=True))
        for batch_index, peer in enumerate(upper):
            dk_by_owner[peer].copy_(grouped[1][:, batch_index : batch_index + 1])
            dv_by_owner[peer].copy_(grouped[2][:, batch_index : batch_index + 1])
        _phase_end(
            phase,
            "branch_b_grouped_backward_upper_scatter",
            section="upper-triangle",
            peer_count=int(count),
            **timing_meta,
        )

    phase = _phase_start()
    dk, dv = tex.nvshmem_cp_prefix_grad_return_execute(
        dk_by_owner,
        dv_by_owner,
        peer_keys[cp_rank],
        peer_values[cp_rank],
        ws.grad_key_return,
        ws.grad_value_return,
        ws.grad_committed_epoch,
        ws.peer_grad_key_returns,
        ws.peer_grad_value_returns,
        ws.peer_grad_committed_epochs,
        4,
        cp_rank,
    )
    _phase_end(phase, "branch_b_grouped_backward_grad_return", **timing_meta)
    return dq, dk, dv


def _branch_b_pipelined_stage_forward(
    *,
    tex,
    query: torch.Tensor,
    peer_keys: list[torch.Tensor],
    peer_values: list[torch.Tensor],
    peer_pes: list[int],
    cp_rank: int,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    torch.Tensor,
]:
    """Pipeline remote K/V staging with TE fused per-peer forward sections."""

    cp_size = len(peer_keys)
    if cp_size != 4 or len(peer_values) != cp_size:
        raise NvshmemCpFaOwnerError(
            f"Pipelined Branch-B staging requires CP=4 peer K/V lists, got {cp_size}."
        )
    compute_stream = torch.cuda.current_stream(query.device)
    copy_lanes = max(
        1,
        min(
            cp_size - 1,
            int(os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_PY_STAGE_COPY_STREAMS", "1")),
        ),
    )
    # The fallback Megatron QKV producer returns Q as a strided view into the
    # interleaved QKV tensor. TE's low-level fused-attention entrypoint treats
    # separate SBHD inputs as dense. Materialize Q once for all four sections;
    # upper-triangle happened to do this in section preparation already.
    query_work = query.contiguous()
    staged_keys = list(peer_keys)
    staged_values = list(peer_values)
    ready_events: list[torch.cuda.Event | None] = [None] * cp_size
    peer_order = [(cp_rank - step) % cp_size for step in range(cp_size)]
    section_aware_half_stage = (
        os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_SECTION_AWARE_HALF_STAGE") == "1"
    )
    nvshmem_getmem_stage = (
        os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_STAGE_GETMEM") == "1"
    )
    if len(peer_pes) != cp_size:
        raise NvshmemCpFaOwnerError(
            f"Pipelined Branch-B staging expects {cp_size} NVSHMEM PEs, got {len(peer_pes)}."
        )
    if nvshmem_getmem_stage and section_aware_half_stage:
        raise NvshmemCpFaOwnerError(
            "NVSHMEM getmem staging and section-aware half staging are separate experiments."
        )
    nvshmem_bindings = _ensure_nvshmem_initialized() if nvshmem_getmem_stage else None
    half = int(query.shape[0]) // 2
    if section_aware_half_stage and int(query.shape[0]) % 2 != 0:
        raise NvshmemCpFaOwnerError(
            "Section-aware Branch-B staging requires an even local sequence length."
        )

    # All remote copies are queued in peer order. The compute stream waits only
    # when it reaches that peer, allowing later copies to overlap earlier TE
    # fused-attention sections.
    for copy_index, peer in enumerate(peer_order[1:]):
        copy_stream = _branch_b_pipeline_copy_stream(
            query.device, copy_index % copy_lanes
        )
        with torch.cuda.stream(copy_stream):
            staged_key = torch.empty_like(peer_keys[peer], memory_format=torch.contiguous_format)
            staged_value = torch.empty_like(
                peer_values[peer], memory_format=torch.contiguous_format
            )
            if nvshmem_getmem_stage:
                nvshmem_bindings.getmem_on_stream(
                    int(staged_key.data_ptr()),
                    int(peer_keys[cp_rank].data_ptr()),
                    int(peer_keys[cp_rank].numel() * peer_keys[cp_rank].element_size()),
                    int(peer_pes[peer]),
                    int(copy_stream.cuda_stream),
                )
                nvshmem_bindings.getmem_on_stream(
                    int(staged_value.data_ptr()),
                    int(peer_values[cp_rank].data_ptr()),
                    int(peer_values[cp_rank].numel() * peer_values[cp_rank].element_size()),
                    int(peer_pes[peer]),
                    int(copy_stream.cuda_stream),
                )
                nvshmem_bindings.quiet_on_stream(int(copy_stream.cuda_stream))
            elif section_aware_half_stage and peer < cp_rank:
                # A lower-triangle CP section can consume only the peer's first
                # native chunk. Its second chunk is causally later than both
                # local query chunks, so do not move bytes the TE section never
                # reads. The full allocation preserves the TE SBHD ABI.
                staged_key[:half].copy_(peer_keys[peer][:half], non_blocking=True)
                staged_value[:half].copy_(peer_values[peer][:half], non_blocking=True)
                # TE's section preparation is logically first-half-only, but
                # call-reused allocator contents were observed to perturb its
                # backward path. Keep the inactive tail deterministic without
                # fetching it across the peer mapping.
                staged_key[half:].zero_()
                staged_value[half:].zero_()
            else:
                staged_key.copy_(peer_keys[peer], non_blocking=True)
                staged_value.copy_(peer_values[peer], non_blocking=True)
            staged_key.record_stream(copy_stream)
            staged_value.record_stream(copy_stream)
            ready = torch.cuda.Event(enable_timing=False)
            ready.record(copy_stream)
            staged_keys[peer] = staged_key
            staged_values[peer] = staged_value
            ready_events[peer] = ready

    softmax_scale = 1.0 / math.sqrt(float(query.shape[-1]))
    grouped_sections = (
        os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_GROUPED_SECTIONS") == "1"
    )
    if grouped_sections:
        for ready in ready_events:
            if ready is not None:
                compute_stream.wait_event(ready)
        output, forward_aux = _branch_b_grouped_sections_forward(
            tex=tex,
            query=query_work,
            peer_keys=staged_keys,
            peer_values=staged_values,
            cp_rank=cp_rank,
        )
        return output, forward_aux, staged_keys, staged_values, query_work

    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_TWO_PREFIX_ATTENTION") == "1":
        for ready in ready_events:
            if ready is not None:
                compute_stream.wait_event(ready)
        q_first, q_second, first_k, first_v, second_k, second_v = (
            _branch_b_two_prefix_inputs(
                query_work, staged_keys, staged_values, cp_rank
            )
        )
        first = tex.nvshmem_cp_prefix_fused_attn_forward_execute(
            q_first, first_k, first_v, "sbhd", float(softmax_scale), query.dtype, True
        )
        second = tex.nvshmem_cp_prefix_fused_attn_forward_execute(
            q_second, second_k, second_v, "sbhd", float(softmax_scale), query.dtype, True
        )
        if len(first) < 3 or len(second) < 3:
            raise NvshmemCpFaOwnerError(
                "Two-prefix Branch-B forward must return output, LSE, and RNG state."
            )
        output = torch.cat((first[0], second[0]), dim=0)
        native_section_backward = (
            os.getenv(
                "MEGATRON_NVSHMEM_CP_BRANCH_B_TWO_PREFIX_NATIVE_SECTION_BACKWARD"
            )
            == "1"
        )
        if native_section_backward:
            first_lse = first[1].squeeze(-1) if first[1].dim() == 4 else first[1]
            second_lse = second[1].squeeze(-1) if second[1].dim() == 4 else second[1]
            merged_lse = torch.cat((first_lse, second_lse), dim=2)
            merged_lse_4d = merged_lse.unsqueeze(-1)
            forward_aux = [merged_lse]
            for peer in range(cp_size):
                if peer > cp_rank:
                    forward_aux.extend((second[1], second[2]))
                else:
                    forward_aux.extend((merged_lse_4d, first[2]))
        else:
            forward_aux = [first[1], first[2], second[1], second[2]]
        return output, forward_aux, staged_keys, staged_values, query_work

    out_by_peer: list[torch.Tensor | None] = [None] * cp_size
    lse_by_peer: list[torch.Tensor | None] = [None] * cp_size
    rng_by_peer: list[torch.Tensor | None] = [None] * cp_size
    section_by_peer: list[str | None] = [None] * cp_size
    qkv_layout = tex.NVTE_QKV_Layout.NVTE_SBHD_SBHD_SBHD
    attn_mask_type = tex.NVTE_Mask_Type.NVTE_CAUSAL_MASK
    concurrent_sections = (
        os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_CONCURRENT_FORWARD_SECTIONS") == "1"
    )
    query_ready: torch.cuda.Event | None = None
    section_done: list[torch.cuda.Event | None] = [None] * cp_size
    if concurrent_sections:
        query_ready = torch.cuda.Event(enable_timing=False)
        query_ready.record(compute_stream)
    for ring_step, peer in enumerate(peer_order):
        ready = ready_events[peer]
        section = _branch_b_section_for_peer(peer, cp_rank)
        section_stream = (
            _branch_b_pipeline_compute_stream(query.device, peer)
            if concurrent_sections
            else compute_stream
        )
        if ready is not None:
            section_stream.wait_event(ready)
        if query_ready is not None:
            section_stream.wait_event(query_ready)
        with torch.cuda.stream(section_stream):
            native_outputs = tex.nvshmem_cp_forward_section_fused_attn_execute(
                query_work,
                staged_keys[peer],
                staged_values[peer],
                int(query_work.shape[0]),
                int(query_work.shape[0]),
                int(ring_step),
                int(cp_size),
                int(cp_rank),
                "sbhd",
                section,
                float(softmax_scale),
                0.0,
                qkv_layout,
                attn_mask_type,
                query.dtype,
                True,
            )
        if not isinstance(native_outputs, (list, tuple)) or len(native_outputs) < 3:
            raise NvshmemCpFaOwnerError(
                "Pipelined Branch-B section must return output, LSE, and RNG state."
            )
        out_step, softmax_lse_step, rng_state = native_outputs[:3]
        out_by_peer[peer] = out_step
        lse_by_peer[peer] = softmax_lse_step
        rng_by_peer[peer] = rng_state
        section_by_peer[peer] = section
        if concurrent_sections:
            out_step.record_stream(section_stream)
            softmax_lse_step.record_stream(section_stream)
            rng_state.record_stream(section_stream)
            done = torch.cuda.Event(enable_timing=False)
            done.record(section_stream)
            section_done[peer] = done

    if concurrent_sections:
        for done in section_done:
            if done is None:
                raise NvshmemCpFaOwnerError("Missing concurrent Branch-B section event.")
            compute_stream.wait_event(done)

    if any(item is None for item in (*out_by_peer, *lse_by_peer, *section_by_peer)):
        raise NvshmemCpFaOwnerError("Missing Branch-B per-peer forward merge state.")
    merge_outputs = [item for item in out_by_peer if item is not None]
    merge_lse = [item for item in lse_by_peer if item is not None]
    merge_sections = [item for item in section_by_peer if item is not None]

    merged = tex.nvshmem_cp_forward_lse_merge(
        merge_outputs,
        merge_lse,
        merge_sections,
        int(query.shape[0]),
        "sbhd",
    )
    if not isinstance(merged, (list, tuple)) or len(merged) != 2:
        raise NvshmemCpFaOwnerError(
            "Pipelined Branch-B LSE merge must return output and merged LSE."
        )
    output, merged_softmax_lse = merged
    forward_aux: list[torch.Tensor] = [merged_softmax_lse]
    for peer in range(cp_size):
        lse = lse_by_peer[peer]
        rng = rng_by_peer[peer]
        if lse is None or rng is None:
            raise NvshmemCpFaOwnerError(
                f"Missing Branch-B forward aux tensors for CP peer {peer}."
            )
        forward_aux.extend((lse, rng))
    return output, forward_aux, staged_keys, staged_values, query_work


def _branch_b_fa4_trace(cp_rank: int, step: str) -> None:
    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_TRACE") == "1":
        print(f"[branch-b-fa4 rank={cp_rank}] {step}", flush=True)


def _branch_b_fa4_trace_sync() -> None:
    """Synchronize only when a trace must describe completed GPU work."""
    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_TRACE") == "1":
        torch.cuda.current_stream().synchronize()


def _branch_b_fa4_mask(cp_rank: int, half: int, cp_size: int):
    """Compile/cache the native two-chunk causal mask for one CP rank."""
    cache_key = (cp_rank, half, cp_size)
    cached = _BRANCH_B_FA4_MASKS.get(cache_key)
    if cached is not None:
        return cached
    cutlass = _fa4_cutlass
    cute = _fa4_cute

    @cute.jit
    def native_cp_causal_mask(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        chunk = cutlass.Int32(half)
        first_global_q = cp_rank * chunk + q_idx
        second_global_q = (2 * cp_size - 1 - cp_rank) * chunk + (q_idx - chunk)
        global_q = cute.where(q_idx < chunk, first_global_q, second_global_q)
        return kv_idx <= global_q

    _BRANCH_B_FA4_MASKS[cache_key] = native_cp_causal_mask
    return native_cp_causal_mask


def _branch_b_fa4_block_sparse(
    cp_rank: int, local_seq: int, cp_size: int, device: torch.device
):
    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_BLOCK_SPARSE") != "1":
        return None, None
    cache_key = (cp_rank, local_seq, cp_size, device)
    cached = _BRANCH_B_FA4_BLOCK_SPARSE.get(cache_key)
    if cached is not None:
        return cached
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from torch.nn.attention.flex_attention import create_block_mask

    half = local_seq // 2
    global_seq = local_seq * cp_size

    def eager_mask(batch, head, q_idx, kv_idx):
        del batch, head
        global_q = torch.where(
            q_idx < half,
            cp_rank * half + q_idx,
            (2 * cp_size - 1 - cp_rank) * half + (q_idx - half),
        )
        return kv_idx <= global_q

    block_mask = create_block_mask(
        eager_mask,
        1,
        1,
        local_seq,
        global_seq,
        device=str(device),
        BLOCK_SIZE=(256, 128),
    )
    (
        _seq_q,
        _seq_k,
        kv_mask_cnt,
        kv_mask_idx,
        full_kv_cnt,
        full_kv_idx,
        q_mask_cnt,
        q_mask_idx,
        full_q_cnt,
        full_q_idx,
        *_,
    ) = block_mask.as_tuple()
    forward = BlockSparseTensorsTorch(
        mask_block_cnt=kv_mask_cnt,
        mask_block_idx=kv_mask_idx,
        full_block_cnt=full_kv_cnt,
        full_block_idx=full_kv_idx,
        block_size=(256, 128),
    )
    backward = BlockSparseTensorsTorch(
        mask_block_cnt=q_mask_cnt,
        mask_block_idx=q_mask_idx,
        full_block_cnt=full_q_cnt,
        full_block_idx=full_q_idx,
        block_size=(256, 128),
    )
    _BRANCH_B_FA4_BLOCK_SPARSE[cache_key] = (forward, backward)
    return forward, backward


def _branch_b_fa4_prewarm(query: torch.Tensor, cp_rank: int, cp_size: int) -> None:
    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_PREWARM") != "1":
        return
    key = (query.device, tuple(query.shape), query.dtype, cp_rank, cp_size)
    if key in _BRANCH_B_FA4_PREWARMED:
        return
    half = int(query.shape[0]) // 2
    q = torch.zeros_like(query).transpose(0, 1).contiguous()
    global_shape = (1, int(query.shape[0]) * cp_size, *query.shape[2:])
    k = torch.zeros(global_shape, device=query.device, dtype=query.dtype)
    v = torch.zeros_like(k)
    softmax_scale = 1.0 / math.sqrt(float(query.shape[-1]))
    _branch_b_fa4_trace(cp_rank, "P00 before_prewarm_fwd")
    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_MULTIBASE") == "1":
        peers_k = tuple(torch.zeros_like(q) for _ in range(4))
        peers_v = tuple(torch.zeros_like(q) for _ in range(4))
        _FA4_FWD(
            q,
            peers_k[0],
            peers_v[0],
            softmax_scale=softmax_scale,
            mask_mod=_branch_b_fa4_mask(cp_rank, half, cp_size),
            peer_k=peers_k,
            peer_v=peers_v,
        )
    elif os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_PACKED_VARLEN") == "1":
        prefix1 = (cp_rank + 1) * half
        prefix2 = (2 * cp_size - cp_rank) * half
        q_packed = q[0]
        k_packed = torch.cat((k[0, :prefix1], k[0, :prefix2]), dim=0)
        v_packed = torch.cat((v[0, :prefix1], v[0, :prefix2]), dim=0)
        cu_q = torch.tensor([0, half, 2 * half], dtype=torch.int32, device=query.device)
        cu_k = torch.tensor(
            [0, prefix1, prefix1 + prefix2], dtype=torch.int32, device=query.device
        )
        _FA4_FWD(
            q_packed, k_packed, v_packed,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=half, max_seqlen_k=max(prefix1, prefix2),
            softmax_scale=softmax_scale, causal=True,
        )
    elif os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_SHARED_KV_BATCH") == "1":
        prefix1 = (cp_rank + 1) * half
        prefix2 = (2 * cp_size - cp_rank) * half
        q_batch = q.view(2, half, q.shape[2], q.shape[3])
        k_batch = k[:, :prefix2].expand(2, -1, -1, -1)
        v_batch = v[:, :prefix2].expand(2, -1, -1, -1)
        seqused_q = torch.full((2,), half, dtype=torch.int32, device=query.device)
        seqused_k = torch.tensor([prefix1, prefix2], dtype=torch.int32, device=query.device)
        _FA4_FWD(
            q_batch, k_batch, v_batch,
            seqused_q=seqused_q, seqused_k=seqused_k,
            softmax_scale=softmax_scale, causal=True,
        )
    elif os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_TWO_PREFIX") == "1":
        for q_part, prefix_chunks in (
            (q[:, :half], cp_rank + 1),
            (q[:, half:], 2 * cp_size - cp_rank),
        ):
            prefix = prefix_chunks * half
            _FA4_FWD(
                q_part,
                k[:, :prefix],
                v[:, :prefix],
                softmax_scale=softmax_scale,
                causal=True,
            )
    else:
        block_sparse_fwd, _ = _branch_b_fa4_block_sparse(
            cp_rank, int(query.shape[0]), cp_size, query.device
        )
        _FA4_FWD(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            mask_mod=_branch_b_fa4_mask(cp_rank, half, cp_size),
            block_sparse_tensors=block_sparse_fwd,
        )
    torch.cuda.current_stream().synchronize()
    _BRANCH_B_FA4_PREWARMED.add(key)
    _branch_b_fa4_trace(cp_rank, "P10 after_prewarm_fwd")


def _branch_b_fa4_global_forward(
    query: torch.Tensor,
    peer_keys: list[torch.Tensor],
    peer_values: list[torch.Tensor],
    cp_rank: int,
    peer_pes: list[int] | None = None,
    layer_number: int = 0,
) -> tuple[torch.Tensor, tuple]:
    """Run one FA4 graph over globally ordered K/V and native-split local Q."""
    if len(peer_keys) != 4 or len(peer_values) != 4:
        raise NvshmemCpFaOwnerError("FA4 Branch-B prototype currently requires CP=4.")
    half = int(query.shape[0]) // 2
    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_MULTIBASE") == "1":
        q_fa4 = query.view(1, query.shape[0], query.shape[2], query.shape[3])
        peer_k_fa4 = tuple(
            item.view(1, item.shape[0], item.shape[2], item.shape[3])
            for item in peer_keys
        )
        peer_v_fa4 = tuple(
            item.view(1, item.shape[0], item.shape[2], item.shape[3])
            for item in peer_values
        )
        softmax_scale = 1.0 / math.sqrt(float(query.shape[-1]))
        mask_mod = _branch_b_fa4_mask(cp_rank, half, 4)
        out_fa4, lse, _, _ = _FA4_FWD(
            q_fa4,
            peer_k_fa4[0],
            peer_v_fa4[0],
            softmax_scale=softmax_scale,
            mask_mod=mask_mod,
            peer_k=peer_k_fa4,
            peer_v=peer_v_fa4,
            return_lse=True,
        )
        state = (
            "multibase", q_fa4, peer_k_fa4, peer_v_fa4, out_fa4, lse,
            mask_mod, softmax_scale,
        )
        return out_fa4.view_as(query), state
    deferred_detail = (
        os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_DEFERRED_PHASE_EVENTS") == "1"
        and query.is_cuda
    )
    materialization_start = materialization_end = None
    if deferred_detail:
        materialization_start = torch.cuda.Event(enable_timing=True)
        materialization_end = torch.cuda.Event(enable_timing=True)
        materialization_start.record()
    _branch_b_fa4_trace(cp_rank, "F10 before_global_kv_cat")
    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_GLOBAL_GETMEM") == "1":
        if peer_pes is None or len(peer_pes) != 4:
            raise NvshmemCpFaOwnerError("FA4 global getmem staging requires four NVSHMEM PEs.")
        nvshmem = _ensure_nvshmem_initialized()
        global_shape = (query.shape[0] * 4, *peer_keys[cp_rank].shape[1:])
        global_key = torch.empty(global_shape, device=query.device, dtype=peer_keys[cp_rank].dtype)
        global_value = torch.empty(
            global_shape, device=query.device, dtype=peer_values[cp_rank].dtype
        )
        stream = torch.cuda.current_stream(query.device)
        half_bytes = half * peer_keys[cp_rank][0].numel() * peer_keys[cp_rank].element_size()
        key_base = int(peer_keys[cp_rank].data_ptr())
        value_base = int(peer_values[cp_rank].data_ptr())
        for owner in range(4):
            pe = int(peer_pes[owner])
            for source_half, chunk in ((0, owner), (1, 7 - owner)):
                byte_offset = source_half * half_bytes
                nvshmem.getmem_on_stream(
                    int(global_key[chunk * half :].data_ptr()),
                    key_base + byte_offset,
                    half_bytes,
                    pe,
                    int(stream.cuda_stream),
                )
                nvshmem.getmem_on_stream(
                    int(global_value[chunk * half :].data_ptr()),
                    value_base + byte_offset,
                    half_bytes,
                    pe,
                    int(stream.cuda_stream),
                )
        nvshmem.quiet_on_stream(int(stream.cuda_stream))
    else:
        global_key = torch.cat(
            [item[:half] for item in peer_keys]
            + [item[half:] for item in reversed(peer_keys)],
            dim=0,
        )
        global_value = torch.cat(
            [item[:half] for item in peer_values]
            + [item[half:] for item in reversed(peer_values)],
            dim=0,
        )
    _branch_b_fa4_trace_sync()
    _branch_b_fa4_trace(cp_rank, "F20 after_global_kv_cat")
    if query.shape[1] == 1:
        q_fa4 = query.view(1, query.shape[0], query.shape[2], query.shape[3])
        k_fa4 = global_key.view(1, global_key.shape[0], global_key.shape[2], global_key.shape[3])
        v_fa4 = global_value.view(
            1, global_value.shape[0], global_value.shape[2], global_value.shape[3]
        )
    else:
        q_fa4 = query.transpose(0, 1).contiguous()
        k_fa4 = global_key.transpose(0, 1).contiguous()
        v_fa4 = global_value.transpose(0, 1).contiguous()
    if materialization_end is not None:
        materialization_end.record()
    softmax_scale = 1.0 / math.sqrt(float(query.shape[-1]))
    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_PACKED_VARLEN") == "1":
        prefix1 = (cp_rank + 1) * half
        prefix2 = (8 - cp_rank) * half
        q_packed = q_fa4[0]
        k_packed = torch.cat((k_fa4[0, :prefix1], k_fa4[0, :prefix2]), dim=0)
        v_packed = torch.cat((v_fa4[0, :prefix1], v_fa4[0, :prefix2]), dim=0)
        cu_q = torch.tensor([0, half, 2 * half], dtype=torch.int32, device=query.device)
        cu_k = torch.tensor(
            [0, prefix1, prefix1 + prefix2], dtype=torch.int32, device=query.device
        )
        packed_result = _FA4_FWD(
            q_packed, k_packed, v_packed,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=half, max_seqlen_k=max(prefix1, prefix2),
            softmax_scale=softmax_scale, causal=True, return_lse=True,
        )
        out_fa4 = packed_result[0].unsqueeze(0)
        state = (
            "packed_varlen", q_packed, k_packed, v_packed, packed_result,
            cu_q, cu_k, prefix1, prefix2, softmax_scale,
        )
        _branch_b_fa4_trace_sync()
        _branch_b_fa4_trace(cp_rank, "F40 after_fa4_fwd")
        return (
            out_fa4.view_as(query)
            if query.shape[1] == 1
            else out_fa4.transpose(0, 1).contiguous()
        ), state
    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_SHARED_KV_BATCH") == "1":
        prefix1 = (cp_rank + 1) * half
        prefix2 = (8 - cp_rank) * half
        q_batch = q_fa4.view(2, half, q_fa4.shape[2], q_fa4.shape[3])
        k_batch = k_fa4[:, :prefix2].expand(2, -1, -1, -1)
        v_batch = v_fa4[:, :prefix2].expand(2, -1, -1, -1)
        seqused_q = torch.full((2,), half, dtype=torch.int32, device=query.device)
        seqused_k = torch.tensor([prefix1, prefix2], dtype=torch.int32, device=query.device)
        fa4_start = fa4_end = None
        if deferred_detail:
            fa4_start = torch.cuda.Event(enable_timing=True)
            fa4_end = torch.cuda.Event(enable_timing=True)
            fa4_start.record()
        batch_result = _FA4_FWD(
            q_batch, k_batch, v_batch,
            seqused_q=seqused_q, seqused_k=seqused_k,
            softmax_scale=softmax_scale, causal=True, return_lse=True,
        )
        if fa4_end is not None:
            fa4_end.record()
            _BRANCH_B_DEFERRED_FORWARD_DETAIL_EVENTS.append(
                {
                    "layer_number": int(layer_number),
                    "cp_rank": int(cp_rank),
                    "materialization_start": materialization_start,
                    "materialization_end": materialization_end,
                    "fa4_start": fa4_start,
                    "fa4_end": fa4_end,
                }
            )
        out_fa4 = torch.cat((batch_result[0][0:1], batch_result[0][1:2]), dim=1)
        state = (
            "shared_kv_batch", q_batch, k_batch, v_batch, batch_result,
            seqused_q, seqused_k, prefix1, prefix2, softmax_scale,
        )
        _branch_b_fa4_trace_sync()
        _branch_b_fa4_trace(cp_rank, "F40 after_fa4_fwd")
        return (
            out_fa4.view_as(query)
            if query.shape[1] == 1
            else out_fa4.transpose(0, 1).contiguous()
        ), state
    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_TWO_PREFIX") == "1":
        prefix1 = (cp_rank + 1) * half
        prefix2 = (8 - cp_rank) * half
        result1 = _FA4_FWD(
            q_fa4[:, :half],
            k_fa4[:, :prefix1],
            v_fa4[:, :prefix1],
            softmax_scale=softmax_scale,
            causal=True,
            return_lse=True,
        )
        result2 = _FA4_FWD(
            q_fa4[:, half:],
            k_fa4[:, :prefix2],
            v_fa4[:, :prefix2],
            softmax_scale=softmax_scale,
            causal=True,
            return_lse=True,
        )
        out_fa4 = torch.cat((result1[0], result2[0]), dim=1)
        state = (
            "two_prefix", q_fa4, k_fa4, v_fa4, result1, result2,
            prefix1, prefix2, softmax_scale,
        )
        _branch_b_fa4_trace_sync()
        _branch_b_fa4_trace(cp_rank, "F40 after_fa4_fwd")
        return (
            out_fa4.view_as(query)
            if query.shape[1] == 1
            else out_fa4.transpose(0, 1).contiguous()
        ), state
    mask_mod = _branch_b_fa4_mask(cp_rank, half, 4)
    block_sparse_fwd, block_sparse_bwd = _branch_b_fa4_block_sparse(
        cp_rank, int(query.shape[0]), 4, query.device
    )
    _branch_b_fa4_trace(
        cp_rank,
        "F30 before_fa4_fwd "
        f"q={tuple(q_fa4.shape)}/{q_fa4.dtype}/{q_fa4.stride()} "
        f"k={tuple(k_fa4.shape)}/{k_fa4.dtype}/{k_fa4.stride()} "
        f"v={tuple(v_fa4.shape)}/{v_fa4.dtype}/{v_fa4.stride()} "
        f"stream={torch.cuda.current_stream().cuda_stream}",
    )
    out_fa4, lse, p, row_max = _FA4_FWD(
        q_fa4,
        k_fa4,
        v_fa4,
        softmax_scale=softmax_scale,
        mask_mod=mask_mod,
        block_sparse_tensors=block_sparse_fwd,
        return_lse=True,
    )
    _branch_b_fa4_trace_sync()
    _branch_b_fa4_trace(cp_rank, "F40 after_fa4_fwd")
    state = (
        q_fa4, k_fa4, v_fa4, out_fa4, lse, p, row_max, mask_mod,
        softmax_scale, block_sparse_bwd,
    )
    return out_fa4.transpose(0, 1).contiguous(), state


def _branch_b_fa4_global_backward(
    tex,
    fa4_state,
    grad_output: torch.Tensor,
    peer_keys: list[torch.Tensor],
    peer_values: list[torch.Tensor],
    ws,
    cp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase_timing = os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_BWD_PHASE_TIMING") == "1"
    phase_events: list[tuple[str, torch.cuda.Event]] = []

    def record_phase(name: str) -> None:
        if phase_timing:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            phase_events.append((name, event))

    def emit_phase_report(**extra: object) -> None:
        if not phase_timing or not phase_events:
            return
        phase_events[-1][1].synchronize()
        phases_ms = {
            f"{left_name}_to_{right_name}_ms": left_event.elapsed_time(right_event)
            for (left_name, left_event), (right_name, right_event) in zip(
                phase_events, phase_events[1:]
            )
        }
        payload = {"cp_rank": cp_rank, "phases_ms": phases_ms}
        payload.update(extra)
        print("[branch-b-fa4-bwd-phase] " + json.dumps(payload, sort_keys=True), flush=True)

    def torch_global_grad_return(
        dk_global_tensor: torch.Tensor,
        dv_global_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slow correctness oracle for global dK/dV owner return.

        Each CP rank computes dK/dV contributions for its local Q against the
        globally ordered K/V layout:
        [chunk0, chunk1, chunk2, chunk3, chunk4, chunk5, chunk6, chunk7].
        The local native CP owner layout is [chunk rank, chunk 7-rank].
        """

        from megatron.core import parallel_state

        cp_group = parallel_state.get_context_parallel_group(check_initialized=False)
        gathered_dk = [torch.empty_like(dk_global_tensor) for _ in range(4)]
        gathered_dv = [torch.empty_like(dv_global_tensor) for _ in range(4)]
        torch.distributed.all_gather(gathered_dk, dk_global_tensor.contiguous(), group=cp_group)
        torch.distributed.all_gather(gathered_dv, dv_global_tensor.contiguous(), group=cp_group)
        half_tokens = int(peer_keys[cp_rank].shape[0]) // 2

        subset_return = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_TORCH_TE_CP4_SUBSET_RETURN") == "1"
        )
        if subset_return and len(gathered_dk) != 4:
            raise NvshmemCpFaOwnerError(
                "MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_TORCH_TE_CP4_SUBSET_RETURN=1 "
                f"requires CP=4, got {len(gathered_dk)}."
            )

        def te_cp4_sources(owner_rank: int, local_half: int) -> list[int]:
            table = {
                (0, 0): [0, 1, 2, 3],
                (0, 1): [0],
                (1, 0): [0, 1, 2, 3],
                (1, 1): [0, 1],
                (2, 0): [0, 1, 2],
                (2, 1): [0, 1, 2],
                (3, 0): [0, 1, 2, 3],
                (3, 1): [0, 1, 2, 3],
            }
            return table[(owner_rank, local_half)]

        def owner_sum(
            chunks: list[torch.Tensor],
            chunk_index: int,
            *,
            local_half: int,
        ) -> torch.Tensor:
            start = chunk_index * half_tokens
            source_ranks = (
                te_cp4_sources(cp_rank, local_half)
                if subset_return
                else range(len(chunks))
            )
            total = None
            for source_rank in source_ranks:
                item = chunks[int(source_rank)].narrow(0, start, half_tokens).float()
                total = item if total is None else total + item
            if total is None:
                raise NvshmemCpFaOwnerError("empty K/V gradient source subset.")
            return total.to(peer_keys[cp_rank].dtype)

        dk_first = owner_sum(gathered_dk, cp_rank, local_half=0)
        dk_second = owner_sum(gathered_dk, 7 - cp_rank, local_half=1)
        dv_first = owner_sum(gathered_dv, cp_rank, local_half=0)
        dv_second = owner_sum(gathered_dv, 7 - cp_rank, local_half=1)
        return torch.cat((dk_first, dk_second), dim=0), torch.cat((dv_first, dv_second), dim=0)

    record_phase("start")
    if isinstance(fa4_state[0], str) and fa4_state[0] == "multibase":
        (
            _marker, q_fa4, peer_k_fa4, peer_v_fa4, out_fa4, lse,
            mask_mod, softmax_scale,
        ) = fa4_state
        dq_fa4, dk_fa4, dv_fa4 = _FA4_BWD(
            q_fa4,
            peer_k_fa4[0],
            peer_v_fa4[0],
            out_fa4,
            grad_output.view_as(out_fa4),
            lse,
            softmax_scale=softmax_scale,
            mask_mod=mask_mod,
            peer_k=peer_k_fa4,
            peer_v=peer_v_fa4,
        )
    elif isinstance(fa4_state[0], str) and fa4_state[0] == "packed_varlen":
        (
            _marker, q_packed, k_packed, v_packed, packed_result,
            cu_q, cu_k, prefix1, prefix2, softmax_scale,
        ) = fa4_state
        half = int(q_packed.shape[0]) // 2
        dq_packed, dk_packed, dv_packed = _FA4_BWD(
            q_packed, k_packed, v_packed, packed_result[0],
            grad_output.transpose(0, 1).contiguous()[0], packed_result[1],
            softmax_scale=softmax_scale, causal=True,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=half, max_seqlen_k=max(prefix1, prefix2),
        )
        dq_fa4 = dq_packed.unsqueeze(0)
        direct_owner_return = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_DIRECT_OWNER_RETURN") == "1"
            and os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_GLOBAL_GRAD_RETURN") == "1"
        )
        if direct_owner_return:
            native_direct_return = (
                os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_NATIVE_DIRECT_OWNER_RETURN") == "1"
            )
            if not native_direct_return:
                raise NvshmemCpFaOwnerError(
                    "FA4 packed-varlen direct owner return currently requires "
                    "MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_NATIVE_DIRECT_OWNER_RETURN=1."
                )
            if int(prefix1) % int(half) != 0 or int(prefix2) % int(half) != 0:
                raise NvshmemCpFaOwnerError(
                    "FA4 packed-varlen direct owner return expects prefix lengths aligned "
                    "to CP half chunks."
                )
            tex = _import_te_for_native_fused_owner()
            native_return = getattr(
                tex, "nvshmem_cp_two_prefix_direct_owner_grad_return_execute", None
            )
            if native_return is None:
                raise NvshmemCpFaOwnerError(
                    "FA4 packed-varlen direct owner return requires transformer_engine_torch."
                    "nvshmem_cp_two_prefix_direct_owner_grad_return_execute."
                )
            dk1 = dk_packed[:prefix1].unsqueeze(0).contiguous()
            dv1 = dv_packed[:prefix1].unsqueeze(0).contiguous()
            dk2 = dk_packed[prefix1 : prefix1 + prefix2].unsqueeze(0).contiguous()
            dv2 = dv_packed[prefix1 : prefix1 + prefix2].unsqueeze(0).contiguous()
            dq = dq_fa4.view(dq_fa4.shape[1], 1, dq_fa4.shape[2], dq_fa4.shape[3])
            dk, dv = native_return(
                dk1,
                dv1,
                dk2,
                dv2,
                int(prefix1),
                int(prefix2),
                peer_keys[cp_rank],
                peer_values[cp_rank],
                ws.grad_key_return,
                ws.grad_value_return,
                ws.grad_committed_epoch,
                ws.peer_grad_key_returns,
                ws.peer_grad_value_returns,
                ws.peer_grad_committed_epochs,
                4,
                cp_rank,
            )
            record_phase("after_packed_varlen_native_direct_owner_return")
            if phase_timing:
                phase_events[-1][1].synchronize()
                phases_ms = {
                    f"{left_name}_to_{right_name}_ms": left_event.elapsed_time(right_event)
                    for (left_name, left_event), (right_name, right_event) in zip(
                        phase_events, phase_events[1:]
                    )
                }
                print(
                    "[branch-b-fa4-bwd-phase] "
                    + json.dumps(
                        {
                            "cp_rank": cp_rank,
                            "direct_owner_return": True,
                            "native_direct_owner_return": True,
                            "packed_varlen": True,
                            "phases_ms": phases_ms,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return dq, dk, dv
        global_seq = int(peer_keys[cp_rank].shape[0]) * 4
        dk_fa4 = torch.zeros(
            (1, global_seq, *dk_packed.shape[1:]), device=dk_packed.device, dtype=dk_packed.dtype
        )
        dv_fa4 = torch.zeros(
            (1, global_seq, *dv_packed.shape[1:]), device=dv_packed.device, dtype=dv_packed.dtype
        )
        dk_fa4[0, :prefix1].add_(dk_packed[:prefix1])
        dk_fa4[0, :prefix2].add_(dk_packed[prefix1:])
        dv_fa4[0, :prefix1].add_(dv_packed[:prefix1])
        dv_fa4[0, :prefix2].add_(dv_packed[prefix1:])
    elif isinstance(fa4_state[0], str) and fa4_state[0] == "shared_kv_batch":
        (
            _marker, q_batch, k_batch, v_batch, batch_result,
            seqused_q, seqused_k, prefix1, prefix2, softmax_scale,
        ) = fa4_state
        half = int(q_batch.shape[1])
        grad_bshd = (
            grad_output.view(1, grad_output.shape[0], grad_output.shape[2], grad_output.shape[3])
            if grad_output.shape[1] == 1
            else grad_output.transpose(0, 1).contiguous()
        )
        dout_batch = grad_bshd.view(2, half, grad_bshd.shape[2], grad_bshd.shape[3])
        two_section_tail_dq = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_TWO_SECTION_TAIL_DQ_RECOMPUTE") == "1"
        )
        two_section_causal_bwd = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_TWO_SECTION_CAUSAL_BWD") == "1"
        )
        dual_section_stream_bwd = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_DUAL_SECTION_STREAM_BWD") == "1"
        )
        two_section_bwd = (
            two_section_tail_dq or two_section_causal_bwd or dual_section_stream_bwd
        )
        direct_owner_return = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_DIRECT_OWNER_RETURN") == "1"
            and os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_GLOBAL_GRAD_RETURN") == "1"
        )
        if two_section_bwd:
            if direct_owner_return:
                raise NvshmemCpFaOwnerError(
                    "FA4 two-section backward returns compact combined dK/dV; "
                    "use global grad return, not direct owner return."
                )
            if dual_section_stream_bwd:
                current_stream = torch.cuda.current_stream(q_batch.device)
                section0_stream = _branch_b_dual_section_bwd_stream(q_batch.device, 0)
                section1_stream = _branch_b_dual_section_bwd_stream(q_batch.device, 1)
                section0_stream.wait_stream(current_stream)
                section1_stream.wait_stream(current_stream)
                with torch.cuda.stream(section0_stream):
                    dq1, dk1, dv1 = _FA4_BWD(
                        q_batch[0:1],
                        k_batch[0:1, :prefix1],
                        v_batch[0:1, :prefix1],
                        batch_result[0][0:1],
                        dout_batch[0:1],
                        batch_result[1][0:1],
                        softmax_scale=softmax_scale,
                        causal=True,
                    )
                with torch.cuda.stream(section1_stream):
                    dq2, dk2, dv2 = _FA4_BWD(
                        q_batch[1:2],
                        k_batch[0:1, :prefix2],
                        v_batch[0:1, :prefix2],
                        batch_result[0][1:2],
                        dout_batch[1:2],
                        batch_result[1][1:2],
                        softmax_scale=softmax_scale,
                        causal=True,
                    )
                current_stream.wait_stream(section0_stream)
                current_stream.wait_stream(section1_stream)
                dq_fa4 = torch.cat((dq1, dq2), dim=1)
                global_seq = int(peer_keys[cp_rank].shape[0]) * 4
                dk_fa4 = torch.zeros(
                    (1, global_seq, *dk2.shape[2:]),
                    device=dk2.device,
                    dtype=dk2.dtype,
                )
                dv_fa4 = torch.zeros(
                    (1, global_seq, *dv2.shape[2:]),
                    device=dv2.device,
                    dtype=dv2.dtype,
                )
                dk_fa4[:, :prefix1].add_(dk1)
                dk_fa4[:, :prefix2].add_(dk2)
                dv_fa4[:, :prefix1].add_(dv1)
                dv_fa4[:, :prefix2].add_(dv2)
                record_phase("after_shared_kv_dual_section_stream_bwd")
            elif _FA4_TWO_SECTION_BWD is None:
                raise NvshmemCpFaOwnerError(
                    "FA4 two-section tail dQ recompute requires "
                    "flash_attn.cute.interface._flash_attn_bwd_two_section_causal."
                )
            else:
                dq1, dq2, dk_batch, dv_batch = _FA4_TWO_SECTION_BWD(
                    q_batch[0:1],
                    q_batch[1:2],
                    k_batch[0:1],
                    v_batch[0:1],
                    batch_result[0][0:1],
                    batch_result[0][1:2],
                    dout_batch[0:1],
                    dout_batch[1:2],
                    batch_result[1][0:1],
                    batch_result[1][1:2],
                    int(prefix1),
                    int(prefix2),
                    softmax_scale=softmax_scale,
                )
                dq_fa4 = torch.cat((dq1, dq2), dim=1)
                global_seq = int(peer_keys[cp_rank].shape[0]) * 4
                if int(prefix2) == global_seq:
                    dk_fa4 = dk_batch
                    dv_fa4 = dv_batch
                else:
                    dk_fa4 = torch.zeros(
                        (1, global_seq, *dk_batch.shape[2:]),
                        device=dk_batch.device,
                        dtype=dk_batch.dtype,
                    )
                    dv_fa4 = torch.zeros(
                        (1, global_seq, *dv_batch.shape[2:]),
                        device=dv_batch.device,
                        dtype=dv_batch.dtype,
                    )
                    dk_fa4[:, :prefix2].copy_(dk_batch[:, :prefix2])
                    dv_fa4[:, :prefix2].copy_(dv_batch[:, :prefix2])
                record_phase(
                    "after_shared_kv_two_section_tail_dq"
                    if two_section_tail_dq
                    else "after_shared_kv_two_section_causal_bwd"
                )
        else:
            dq_batch, dk_batch, dv_batch = _FA4_BWD(
                q_batch, k_batch, v_batch, batch_result[0], dout_batch, batch_result[1],
                softmax_scale=softmax_scale, causal=True,
                seqused_q=seqused_q, seqused_k=seqused_k,
            )
            dq_fa4 = torch.cat((dq_batch[0:1], dq_batch[1:2]), dim=1)
        if direct_owner_return:
            if int(prefix1) % int(half) != 0 or int(prefix2) % int(half) != 0:
                raise NvshmemCpFaOwnerError(
                    "FA4 shared-KV batch direct owner return expects prefix lengths "
                    "aligned to CP half chunks."
                )
            native_direct_return = (
                os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_NATIVE_DIRECT_OWNER_RETURN") == "1"
            )
            if not native_direct_return:
                raise NvshmemCpFaOwnerError(
                    "FA4 shared-KV batch direct owner return currently requires "
                    "MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_NATIVE_DIRECT_OWNER_RETURN=1."
                )
            tex = _import_te_for_native_fused_owner()
            native_return = getattr(
                tex, "nvshmem_cp_two_prefix_direct_owner_grad_return_execute", None
            )
            if native_return is None:
                available = [
                    name
                    for name in dir(tex)
                    if "grad_return" in name or "two_prefix" in name
                ]
                raise NvshmemCpFaOwnerError(
                    "FA4 shared-KV batch direct owner return requires transformer_engine_torch."
                    "nvshmem_cp_two_prefix_direct_owner_grad_return_execute. "
                    f"loaded_module={getattr(tex, '__file__', '<unknown>')} "
                    f"available_related={available}"
                )
            dq = dq_fa4.view(dq_fa4.shape[1], 1, dq_fa4.shape[2], dq_fa4.shape[3])
            dk, dv = native_return(
                dk_batch[0:1, :prefix1].contiguous(),
                dv_batch[0:1, :prefix1].contiguous(),
                dk_batch[1:2, :prefix2].contiguous(),
                dv_batch[1:2, :prefix2].contiguous(),
                int(prefix1),
                int(prefix2),
                peer_keys[cp_rank],
                peer_values[cp_rank],
                ws.grad_key_return,
                ws.grad_value_return,
                ws.grad_committed_epoch,
                ws.peer_grad_key_returns,
                ws.peer_grad_value_returns,
                ws.peer_grad_committed_epochs,
                4,
                cp_rank,
            )
            record_phase("after_shared_kv_batch_native_direct_owner_return")
            if phase_timing:
                phase_events[-1][1].synchronize()
                phases_ms = {
                    f"{left_name}_to_{right_name}_ms": left_event.elapsed_time(right_event)
                    for (left_name, left_event), (right_name, right_event) in zip(
                        phase_events, phase_events[1:]
                    )
                }
                print(
                    "[branch-b-fa4-bwd-phase] "
                    + json.dumps(
                        {
                            "cp_rank": cp_rank,
                            "direct_owner_return": True,
                            "native_direct_owner_return": True,
                            "shared_kv_batch": True,
                            "phases_ms": phases_ms,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
            )
            return dq, dk, dv
        if not two_section_bwd:
            global_seq = int(peer_keys[cp_rank].shape[0]) * 4
            dk_fa4 = torch.zeros(
                (1, global_seq, *dk_batch.shape[2:]), device=dk_batch.device, dtype=dk_batch.dtype
            )
            dv_fa4 = torch.zeros(
                (1, global_seq, *dv_batch.shape[2:]), device=dv_batch.device, dtype=dv_batch.dtype
            )
            dk_fa4[:, :prefix1].add_(dk_batch[0:1, :prefix1])
            dk_fa4[:, :prefix2].add_(dk_batch[1:2, :prefix2])
            dv_fa4[:, :prefix1].add_(dv_batch[0:1, :prefix1])
            dv_fa4[:, :prefix2].add_(dv_batch[1:2, :prefix2])
    elif isinstance(fa4_state[0], str) and fa4_state[0] == "two_prefix":
        (
            _marker, q_fa4, k_fa4, v_fa4, result1, result2,
            prefix1, prefix2, softmax_scale,
        ) = fa4_state
        grad_bshd = (
            grad_output.view(1, grad_output.shape[0], grad_output.shape[2], grad_output.shape[3])
            if grad_output.shape[1] == 1
            else grad_output.transpose(0, 1).contiguous()
        )
        half = int(q_fa4.shape[1]) // 2
        record_phase("before_prefix1")
        dq1, dk1, dv1 = _FA4_BWD(
            q_fa4[:, :half], k_fa4[:, :prefix1], v_fa4[:, :prefix1],
            result1[0], grad_bshd[:, :half], result1[1],
            softmax_scale=softmax_scale, causal=True,
        )
        record_phase("after_prefix1")
        dq2, dk2, dv2 = _FA4_BWD(
            q_fa4[:, half:], k_fa4[:, :prefix2], v_fa4[:, :prefix2],
            result2[0], grad_bshd[:, half:], result2[1],
            softmax_scale=softmax_scale, causal=True,
        )
        record_phase("after_prefix2")
        dq_fa4 = torch.cat((dq1, dq2), dim=1)
        direct_owner_return = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_DIRECT_OWNER_RETURN") == "1"
            and os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_GLOBAL_GRAD_RETURN") == "1"
        )
        if direct_owner_return:
            if q_fa4.shape[0] != 1 or int(prefix1) % int(half) != 0 or int(prefix2) % int(half) != 0:
                raise NvshmemCpFaOwnerError(
                    "FA4 direct owner return currently expects B=1 and prefix lengths "
                    "aligned to CP half chunks."
                )
            dq = dq_fa4.view(dq_fa4.shape[1], 1, dq_fa4.shape[2], dq_fa4.shape[3])
            native_direct_return = (
                os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_NATIVE_DIRECT_OWNER_RETURN") == "1"
            )
            if native_direct_return:
                tex = _import_te_for_native_fused_owner()
                native_return = getattr(
                    tex, "nvshmem_cp_two_prefix_direct_owner_grad_return_execute", None
                )
                if native_return is None:
                    available = [
                        name
                        for name in dir(tex)
                        if "grad_return" in name or "two_prefix" in name
                    ]
                    raise NvshmemCpFaOwnerError(
                        "MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_NATIVE_DIRECT_OWNER_RETURN=1 "
                        "requires transformer_engine_torch."
                        "nvshmem_cp_two_prefix_direct_owner_grad_return_execute. "
                        f"loaded_module={getattr(tex, '__file__', '<unknown>')} "
                        f"available_related={available}"
                    )
                dk, dv = native_return(
                    dk1,
                    dv1,
                    dk2,
                    dv2,
                    int(prefix1),
                    int(prefix2),
                    peer_keys[cp_rank],
                    peer_values[cp_rank],
                    ws.grad_key_return,
                    ws.grad_value_return,
                    ws.grad_committed_epoch,
                    ws.peer_grad_key_returns,
                    ws.peer_grad_value_returns,
                    ws.peer_grad_committed_epochs,
                    4,
                    cp_rank,
                )
                record_phase("after_native_direct_owner_return")
                if phase_timing:
                    phase_events[-1][1].synchronize()
                    phases_ms = {
                        f"{left_name}_to_{right_name}_ms": left_event.elapsed_time(right_event)
                        for (left_name, left_event), (right_name, right_event) in zip(
                            phase_events, phase_events[1:]
                        )
                    }
                    print(
                        "[branch-b-fa4-bwd-phase] "
                        + json.dumps(
                            {
                                "cp_rank": cp_rank,
                                "direct_owner_return": True,
                                "native_direct_owner_return": True,
                                "phases_ms": phases_ms,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                return dq, dk, dv
            prefix1_chunks = int(prefix1) // int(half)
            prefix2_chunks = int(prefix2) // int(half)

            def _return_slot(tensor: torch.Tensor, owner: int) -> torch.Tensor:
                if tensor.dim() == peer_keys[cp_rank].dim() + 1:
                    return tensor[cp_rank]
                return tensor

            def _copy_chunk(
                *,
                dst: torch.Tensor,
                src1: torch.Tensor,
                src2: torch.Tensor,
                chunk: int,
                half_index: int,
            ) -> None:
                dst_half = dst.narrow(0, half_index * half, half)
                dst_half.zero_()
                if chunk < prefix2_chunks:
                    src2_chunk = src2.narrow(1, chunk * half, half).view_as(dst_half)
                    dst_half.copy_(src2_chunk)
                if chunk < prefix1_chunks:
                    src1_chunk = src1.narrow(1, chunk * half, half).view_as(dst_half)
                    dst_half.add_(src1_chunk)

            for owner in range(4):
                key_slot = _return_slot(ws.peer_grad_key_returns[owner], owner)
                value_slot = _return_slot(ws.peer_grad_value_returns[owner], owner)
                _copy_chunk(dst=key_slot, src1=dk1, src2=dk2, chunk=owner, half_index=0)
                _copy_chunk(dst=key_slot, src1=dk1, src2=dk2, chunk=7 - owner, half_index=1)
                _copy_chunk(dst=value_slot, src1=dv1, src2=dv2, chunk=owner, half_index=0)
                _copy_chunk(dst=value_slot, src1=dv1, src2=dv2, chunk=7 - owner, half_index=1)

            if torch.cuda.is_available() and grad_output.is_cuda:
                torch.cuda.current_stream().synchronize()
            epoch = _next_grad_return_epoch(ws)
            for owner in range(4):
                peer_epoch = ws.peer_grad_committed_epochs[owner]
                if peer_epoch.dim() >= 1 and peer_epoch.size(0) > cp_rank:
                    peer_epoch[cp_rank] = epoch
                else:
                    peer_epoch.fill_(epoch)
            _wait_for_grad_committed_epoch(ws, epoch, _sync_timeout_seconds())
            dk = ws.grad_key_return.sum(dim=0).to(peer_keys[cp_rank].dtype)
            dv = ws.grad_value_return.sum(dim=0).to(peer_values[cp_rank].dtype)
            record_phase("after_direct_owner_return")
            if phase_timing:
                phase_events[-1][1].synchronize()
                phases_ms = {
                    f"{left_name}_to_{right_name}_ms": left_event.elapsed_time(right_event)
                    for (left_name, left_event), (right_name, right_event) in zip(
                        phase_events, phase_events[1:]
                    )
                }
                print(
                    "[branch-b-fa4-bwd-phase] "
                    + json.dumps(
                        {
                            "cp_rank": cp_rank,
                            "direct_owner_return": True,
                            "phases_ms": phases_ms,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return dq, dk, dv
        if prefix2 == k_fa4.shape[1]:
            dk_fa4 = dk2
            dv_fa4 = dv2
        else:
            dk_fa4 = torch.zeros_like(k_fa4)
            dv_fa4 = torch.zeros_like(v_fa4)
            dk_fa4[:, :prefix2].copy_(dk2)
            dv_fa4[:, :prefix2].copy_(dv2)
        dk_fa4[:, :prefix1].add_(dk1)
        dv_fa4[:, :prefix1].add_(dv1)
        record_phase("after_global_assembly")
    else:
        (
            q_fa4, k_fa4, v_fa4, out_fa4, lse, p, row_max, mask_mod,
            softmax_scale, block_sparse_bwd,
        ) = fa4_state
        dq_fa4, dk_fa4, dv_fa4 = _FA4_BWD(
            q_fa4,
            k_fa4,
            v_fa4,
            out_fa4,
            grad_output.transpose(0, 1).contiguous(),
            lse,
            softmax_scale=softmax_scale,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_bwd,
        )
    if dq_fa4.shape[0] == 1:
        dq = dq_fa4.view(dq_fa4.shape[1], 1, dq_fa4.shape[2], dq_fa4.shape[3])
        dk_global = dk_fa4.view(
            dk_fa4.shape[1], 1, dk_fa4.shape[2], dk_fa4.shape[3]
        )
        dv_global = dv_fa4.view(
            dv_fa4.shape[1], 1, dv_fa4.shape[2], dv_fa4.shape[3]
        )
    else:
        dq = dq_fa4.transpose(0, 1).contiguous()
        dk_global = dk_fa4.transpose(0, 1).contiguous()
        dv_global = dv_fa4.transpose(0, 1).contiguous()
    if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_GLOBAL_GRAD_RETURN") == "1":
        if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_TORCH_GLOBAL_GRAD_RETURN") == "1":
            dk, dv = torch_global_grad_return(dk_global, dv_global)
            record_phase("after_torch_global_grad_return")
            return dq, dk, dv
        if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_NATIVE_TE_CP4_SUBSET_RETURN") == "1":
            subset_return = getattr(tex, "nvshmem_cp_global_grad_return_subset_execute", None)
            if subset_return is None:
                raise NvshmemCpFaOwnerError(
                    "MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_NATIVE_TE_CP4_SUBSET_RETURN=1 "
                    "requires transformer_engine_torch."
                    "nvshmem_cp_global_grad_return_subset_execute."
                )
            source_table = {
                0: [[0, 1, 2, 3], [0]],
                1: [[0, 1, 2, 3], [0, 1]],
                2: [[0, 1, 2, 3], [0, 1, 2]],
                3: [[0, 1, 2, 3], [0, 1, 2, 3]],
            }
            dk, dv = subset_return(
                dk_global,
                dv_global,
                peer_keys[cp_rank],
                peer_values[cp_rank],
                ws.grad_key_return,
                ws.grad_value_return,
                ws.grad_committed_epoch,
                ws.peer_grad_key_returns,
                ws.peer_grad_value_returns,
                ws.peer_grad_committed_epochs,
                source_table[int(cp_rank)],
                4,
                cp_rank,
            )
            record_phase("after_native_te_cp4_subset_grad_return")
            emit_phase_report(native_te_cp4_subset_return=True)
            return dq, dk, dv
        global_return = getattr(tex, "nvshmem_cp_global_grad_return_execute", None)
        if global_return is None:
            raise NvshmemCpFaOwnerError(
                "MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_GLOBAL_GRAD_RETURN=1 requires "
                "transformer_engine_torch.nvshmem_cp_global_grad_return_execute."
            )
        dk, dv = global_return(
            dk_global,
            dv_global,
            peer_keys[cp_rank],
            peer_values[cp_rank],
            ws.grad_key_return,
            ws.grad_value_return,
            ws.grad_committed_epoch,
            ws.peer_grad_key_returns,
            ws.peer_grad_value_returns,
            ws.peer_grad_committed_epochs,
            4,
            cp_rank,
        )
        record_phase("after_grad_return")
        if phase_timing:
            phase_events[-1][1].synchronize()
            phases_ms = {
                f"{left_name}_to_{right_name}_ms": left_event.elapsed_time(right_event)
                for (left_name, left_event), (right_name, right_event) in zip(
                    phase_events, phase_events[1:]
                )
            }
            print(
                "[branch-b-fa4-bwd-phase] "
                + json.dumps({"cp_rank": cp_rank, "phases_ms": phases_ms}, sort_keys=True),
                flush=True,
            )
        return dq, dk, dv
    half = int(dq.shape[0]) // 2
    dk_chunks = list(dk_global.split(half, dim=0))
    dv_chunks = list(dv_global.split(half, dim=0))
    dk_by_owner = [
        torch.cat((dk_chunks[owner], dk_chunks[7 - owner]), dim=0)
        for owner in range(4)
    ]
    dv_by_owner = [
        torch.cat((dv_chunks[owner], dv_chunks[7 - owner]), dim=0)
        for owner in range(4)
    ]
    dk, dv = tex.nvshmem_cp_prefix_grad_return_execute(
        dk_by_owner,
        dv_by_owner,
        peer_keys[cp_rank],
        peer_values[cp_rank],
        ws.grad_key_return,
        ws.grad_value_return,
        ws.grad_committed_epoch,
        ws.peer_grad_key_returns,
        ws.peer_grad_value_returns,
        ws.peer_grad_committed_epochs,
        4,
        cp_rank,
    )
    return dq, dk, dv


class _BlockReadyNativeFusedOwnerAutograd(torch.autograd.Function):
    """Training boundary for the production Branch-B native fused owner.

    The forward native symbol must return attention output plus saved forward
    aux tensors. The backward native symbol consumes those aux tensors and
    grad_output, owns dQ and owner dK/dV computation, and returns remote K/V
    gradients through NVSHMEM symmetric gradient-return buffers. This class
    intentionally fails closed if either symbol or contract is missing.
    """

    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool, layer_number):
        deferred_timing = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_DEFERRED_PHASE_EVENTS") == "1"
            and query.is_cuda
        )
        deferred_forward_start = deferred_forward_end = None
        if deferred_timing:
            deferred_forward_start = torch.cuda.Event(enable_timing=True)
            deferred_forward_end = torch.cuda.Event(enable_timing=True)
            deferred_forward_start.record()
        fa4_global = os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_FA4_GLOBAL") == "1"
        try:
            tex = _import_te_for_native_fused_owner()
        except ImportError as exc:
            raise NvshmemCpFaOwnerError(
                "MEGATRON_NVSHMEM_CP_FA_OWNER_IO_V1_BACKWARD_IMPL="
                "block_ready_native_fused requires transformer_engine_torch with the "
                "selected backend primitives."
            ) from exc

        native_owner = getattr(tex, "nvshmem_cp_block_ready_fused_owner_execute", None)
        native_backward = getattr(tex, "nvshmem_cp_block_ready_fused_owner_backward_execute", None)
        global_grad_return = getattr(tex, "nvshmem_cp_global_grad_return_execute", None)
        if fa4_global and global_grad_return is None:
            raise NvshmemCpFaOwnerError(
                "FA4-global Branch-B requires transformer_engine_torch."
                "nvshmem_cp_global_grad_return_execute."
            )
        if not fa4_global and (native_owner is None or native_backward is None):
            raise NvshmemCpFaOwnerError(
                "MEGATRON_NVSHMEM_CP_FA_OWNER_IO_V1_BACKWARD_IMPL="
                "block_ready_native_fused requires native fused Branch-B owner symbols "
                "named nvshmem_cp_block_ready_fused_owner_execute and "
                "nvshmem_cp_block_ready_fused_owner_backward_execute. The symbols must "
                "consume symmetric peer K/V plus block-ready flags, save forward aux tensors, "
                "compute memory-safe forward attention, implement backward dQ and owner dK/dV, "
                "and return remote K/V gradients without Python-staged large buffers."
            )

        if os.getenv("MEGATRON_NVSHMEM_CP_BLOCK_READY_PROTOCOL") != "1":
            raise NvshmemCpFaOwnerError(
                "block_ready_native_fused requires MEGATRON_NVSHMEM_CP_BLOCK_READY_PROTOCOL=1."
            )
        if not bool(is_causal):
            raise NvshmemCpFaOwnerError("block_ready_native_fused currently targets causal CP attention.")

        if fa4_global:
            from megatron.core import parallel_state

            _branch_b_fa4_prewarm(
                query,
                int(parallel_state.get_context_parallel_rank()),
                int(parallel_state.get_context_parallel_world_size()),
            )
        ws = _workspace(key.shape, key.dtype, key.device, layer_number)
        cp_size = len(ws.cp_group_ranks)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        try:
            cp_rank = ws.cp_group_ranks.index(int(rank))
        except ValueError as exc:
            raise NvshmemCpFaOwnerError(
                f"Global rank {rank} is not in the NVSHMEM CP group {ws.cp_group_ranks}."
            ) from exc

        if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_CLEAR_GRAD_RETURN_BEFORE_FORWARD") == "1":
            with torch.no_grad():
                ws.grad_key_return.zero_()
                ws.grad_value_return.zero_()
                if (
                    os.getenv(
                        "MEGATRON_NVSHMEM_CP_BRANCH_B_RESET_GRAD_EPOCH_BEFORE_FORWARD"
                    )
                    == "1"
                ):
                    ws.grad_committed_epoch.zero_()
                if (
                    os.getenv(
                        "MEGATRON_NVSHMEM_CP_BRANCH_B_RESET_CARRIER_EPOCH_BEFORE_FORWARD"
                    )
                    == "1"
                    and getattr(ws, "carrier_epoch", None) is not None
                    and ws.carrier_epoch.numel() > 0
                ):
                    ws.carrier_epoch.zero_()
            if torch.cuda.is_available() and query.is_cuda:
                torch.cuda.current_stream().synchronize()
        with torch.no_grad():
            if ws.key.data_ptr() != key.data_ptr():
                raise NvshmemCpFaOwnerError(
                    "block_ready_native_fused requires K to already reside in the NVSHMEM symmetric workspace. "
                    "Use a zero-copy producer; do not copy K here."
                )
            if ws.value.data_ptr() != value.data_ptr():
                raise NvshmemCpFaOwnerError(
                    "block_ready_native_fused requires V to already reside in the NVSHMEM symmetric workspace. "
                    "Use a zero-copy producer; do not copy V here."
                )
        timing_on = cp_timing_enabled()
        forward_total_t0 = time.perf_counter() if timing_on else None
        publish_t0 = time.perf_counter() if timing_on else None
        with torch.no_grad():
            forward_epoch = int(ws.block_ready_epoch.detach().min().item()) + 1
            _publish_block_ready_epoch(ws, forward_epoch)
        publish_wall_ms = (
            (time.perf_counter() - publish_t0) * 1000.0
            if publish_t0 is not None
            else None
        )
        skip_global_wait = (
            os.getenv("MEGATRON_NVSHMEM_CP_BLOCK_READY_NATIVE_FUSED_SKIP_GLOBAL_WAIT") == "1"
        )
        native_peer_wait = (
            os.getenv("MEGATRON_NVSHMEM_CP_BLOCK_READY_NATIVE_FUSED_PEER_WAIT") == "1"
        )
        stream_wait = os.getenv("MEGATRON_NVSHMEM_CP_BLOCK_READY_STREAM_WAIT") == "1"
        if skip_global_wait and not native_peer_wait:
            raise NvshmemCpFaOwnerError(
                "MEGATRON_NVSHMEM_CP_BLOCK_READY_NATIVE_FUSED_SKIP_GLOBAL_WAIT=1 requires "
                "MEGATRON_NVSHMEM_CP_BLOCK_READY_NATIVE_FUSED_PEER_WAIT=1 so peer K/V is "
                "stream-waited in the native owner before consumption."
            )
        if stream_wait and native_peer_wait:
            raise NvshmemCpFaOwnerError(
                "MEGATRON_NVSHMEM_CP_BLOCK_READY_STREAM_WAIT=1 is a pre-owner block-ready "
                "wait and must not be combined with per-section native peer wait."
            )
        wait_t0 = time.perf_counter() if timing_on else None
        if not skip_global_wait:
            if stream_wait:
                stream_wait_fn = getattr(
                    tex, "nvshmem_block_ready_wait_int32_on_current_stream", None
                )
                if stream_wait_fn is None:
                    raise NvshmemCpFaOwnerError(
                        "MEGATRON_NVSHMEM_CP_BLOCK_READY_STREAM_WAIT=1 requires a rebuilt "
                        "transformer_engine_torch exposing "
                        "nvshmem_block_ready_wait_int32_on_current_stream."
                    )
                for peer_epoch in ws.peer_block_ready_epochs:
                    stream_wait_fn(peer_epoch, int(forward_epoch))
            else:
                if torch.cuda.is_available() and query.is_cuda:
                    torch.cuda.current_stream().synchronize()
                _wait_for_block_ready_epoch(ws, forward_epoch, _sync_timeout_seconds())
        wait_wall_ms = (
            (time.perf_counter() - wait_t0) * 1000.0 if wait_t0 is not None else None
        )
        native_start = native_end = None
        if timing_on and torch.cuda.is_available() and query.is_cuda:
            native_start = torch.cuda.Event(enable_timing=True)
            native_end = torch.cuda.Event(enable_timing=True)
            native_start.record()
        native_peer_keys = list(ws.peer_keys)
        native_peer_values = list(ws.peer_values)
        reuse_staged_peer_kv = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_PY_STAGE_REMOTE_KV_REUSE_BWD") == "1"
        )
        pipelined_stage = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_PY_PIPELINED_STAGE") == "1"
        )
        if reuse_staged_peer_kv and not pipelined_stage:
            # TE fused-attention kernels lose locality when they repeatedly load
            # mapped peer memory. Stage only remote owners once and retain the
            # local copies for backward; the local owner's symmetric tensors are
            # already device-local and need no copy.
            _branch_b_fa4_trace(cp_rank, "F01 before_peer_stage")
            native_peer_keys = [
                peer_key
                if peer == cp_rank
                else peer_key.clone(memory_format=torch.contiguous_format)
                for peer, peer_key in enumerate(native_peer_keys)
            ]
            native_peer_values = [
                peer_value
                if peer == cp_rank
                else peer_value.clone(memory_format=torch.contiguous_format)
                for peer, peer_value in enumerate(native_peer_values)
            ]
            torch.cuda.current_stream().synchronize()
            _branch_b_fa4_trace(cp_rank, "F02 after_peer_stage")

        fa4_state = None
        if fa4_global:
            _branch_b_fa4_trace(cp_rank, "F03 enter_fa4_branch")
            native_query = query.contiguous()
            output, fa4_state = _branch_b_fa4_global_forward(
                native_query,
                native_peer_keys,
                native_peer_values,
                cp_rank,
                list(ws.cp_group_ranks),
                int(layer_number or 0),
            )
            # The FA4 autograd graph owns its LSE and row-max state. Keep a
            # tensor placeholder so the surrounding Branch-B save contract
            # remains non-empty without exposing FA4 private auxiliaries.
            forward_aux_tensors = [torch.empty(0, device=query.device)]
            result = (output, forward_aux_tensors)
        elif pipelined_stage:
            (
                output,
                forward_aux_tensors,
                native_peer_keys,
                native_peer_values,
                native_query,
            ) = (
                _branch_b_pipelined_stage_forward(
                    tex=tex,
                    query=query,
                    peer_keys=native_peer_keys,
                    peer_values=native_peer_values,
                    peer_pes=list(ws.cp_group_ranks),
                    cp_rank=cp_rank,
                )
            )
            result = (output, forward_aux_tensors)
            reuse_staged_peer_kv = True
        else:
            native_query = query.contiguous()
            result = native_owner(
                native_query,
                key,
                value,
                native_peer_keys,
                native_peer_values,
                ws.block_ready_epoch,
                ws.peer_block_ready_epochs,
                ws.grad_key_return,
                ws.grad_value_return,
                ws.grad_committed_epoch,
                ws.peer_grad_key_returns,
                ws.peer_grad_value_returns,
                ws.peer_grad_committed_epochs,
                cp_size,
                cp_rank,
                "sbhd",
                True,
            )
        native_forward_ms = None
        if native_end is not None and native_start is not None:
            native_end.record()
            native_end.synchronize()
            native_forward_ms = float(native_start.elapsed_time(native_end))
        if not (isinstance(result, (list, tuple)) and len(result) == 2):
            raise NvshmemCpFaOwnerError(
                "nvshmem_cp_block_ready_fused_owner_execute must return "
                "(attention_output, forward_aux_tensors). The saved forward aux tensors are "
                "required for nvshmem_cp_block_ready_fused_owner_backward_execute."
            )
        output, forward_aux_tensors = result
        if not isinstance(forward_aux_tensors, (list, tuple)) or not forward_aux_tensors:
            raise NvshmemCpFaOwnerError(
                "nvshmem_cp_block_ready_fused_owner_execute must return non-empty saved forward_aux_tensors "
                "including softmax LSE/RNG state for the native fused backward."
            )
        if output.shape != query.shape:
            raise NvshmemCpFaOwnerError(
                f"nvshmem_cp_block_ready_fused_owner_execute returned output shape {list(output.shape)}, "
                f"expected {list(query.shape)}."
            )
        if timing_on:
            record_cp_timing(
                {
                    "event": "branch_b_native_fused_owner_forward",
                    "layer_number": layer_number,
                    "cp_rank": int(cp_rank),
                    "cp_world_size": int(cp_size),
                    "query": tensor_meta(query),
                    "key": tensor_meta(key),
                    "value": tensor_meta(value),
                    "output": tensor_meta(output),
                    "forward_aux_count": int(len(forward_aux_tensors)),
                    "publish_wall_ms": publish_wall_ms,
                    "block_ready_wait_wall_ms": wait_wall_ms,
                    "native_forward_elapsed_ms": native_forward_ms,
                    "elapsed_ms": (
                        (time.perf_counter() - forward_total_t0) * 1000.0
                        if forward_total_t0 is not None
                        else None
                    ),
                    "skip_global_wait": bool(skip_global_wait),
                    "native_peer_wait": bool(native_peer_wait),
                    "block_ready_stream_wait": bool(stream_wait),
                    "timing_scope": "Branch-B native fused owner forward; diagnostic timing may perturb runtime.",
                }
            )

        ctx.ws = ws
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.layer_number = layer_number
        ctx.native_backward = native_backward
        ctx.forward_aux_tensors = list(forward_aux_tensors)
        ctx.native_peer_keys = native_peer_keys
        ctx.native_peer_values = native_peer_values
        ctx.reuse_staged_peer_kv = reuse_staged_peer_kv
        ctx.pipelined_stage = pipelined_stage
        ctx.fa4_global = fa4_global
        ctx.fa4_state = fa4_state
        ctx.two_prefix_attention = (
            pipelined_stage
            and os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_TWO_PREFIX_ATTENTION") == "1"
        )
        ctx.two_prefix_native_section_backward = (
            ctx.two_prefix_attention
            and os.getenv(
                "MEGATRON_NVSHMEM_CP_BRANCH_B_TWO_PREFIX_NATIVE_SECTION_BACKWARD"
            )
            == "1"
        )
        ctx.is_causal = bool(is_causal)
        ctx.softmax_scale = 1.0 / math.sqrt(float(query.shape[-1]))
        ctx.query_dtype = query.dtype
        ctx.key_dtype = key.dtype
        ctx.value_dtype = value.dtype
        ctx.forward_aux_count = len(forward_aux_tensors)
        if deferred_forward_end is not None:
            deferred_forward_end.record()
        ctx.deferred_forward_start = deferred_forward_start
        ctx.deferred_forward_end = deferred_forward_end
        ctx.save_for_backward(native_query, key, value, output, *forward_aux_tensors)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        deferred_backward_start = deferred_backward_end = None
        if ctx.deferred_forward_start is not None:
            deferred_backward_start = torch.cuda.Event(enable_timing=True)
            deferred_backward_end = torch.cuda.Event(enable_timing=True)
            deferred_backward_start.record()
        saved = ctx.saved_tensors
        query, key, value, output = saved[:4]
        forward_aux_tensors = list(saved[4 : 4 + ctx.forward_aux_count])
        ws = ctx.ws
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        layer_number = int(getattr(ctx, "layer_number", 0) or 0)
        call_key = (int(rank), layer_number)
        call_index = _NATIVE_FUSED_BACKWARD_CALL_COUNTERS.get(call_key, 0)
        _NATIVE_FUSED_BACKWARD_CALL_COUNTERS[call_key] = call_index + 1
        if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_CLEAR_GRAD_RETURN_BEFORE_BACKWARD") == "1":
            ws.grad_key_return.zero_()
            ws.grad_value_return.zero_()
            if getattr(ws, "raw_key_return", None) is not None and ws.raw_key_return.numel() > 0:
                ws.raw_key_return.zero_()
                ws.raw_value_return.zero_()
            if torch.cuda.is_available() and grad_output.is_cuda:
                torch.cuda.current_stream().synchronize()
        timing_on = cp_timing_enabled()
        backward_total_t0 = time.perf_counter() if timing_on else None
        native_start = native_end = None
        if timing_on and torch.cuda.is_available() and grad_output.is_cuda:
            if (
                os.getenv("MEGATRON_NVSHMEM_CP_BACKWARD_DEVICE_SYNC_TIMING") == "1"
            ):
                torch.cuda.synchronize(grad_output.device)
            native_start = torch.cuda.Event(enable_timing=True)
            native_end = torch.cuda.Event(enable_timing=True)
            native_start.record()
        native_peer_keys = (
            ctx.native_peer_keys if ctx.reuse_staged_peer_kv else list(ws.peer_keys)
        )
        native_peer_values = (
            ctx.native_peer_values if ctx.reuse_staged_peer_kv else list(ws.peer_values)
        )
        concurrent_sections_backward = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_CONCURRENT_BACKWARD_SECTIONS") == "1"
        )
        grouped_sections_backward = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_GROUPED_BACKWARD_SECTIONS") == "1"
        )
        true_fused_no_te_bwd = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_TRUE_FUSED_BACKWARD_NO_TE_BWD") == "1"
        )
        if true_fused_no_te_bwd:
            tex = _import_te_for_native_fused_owner()
            helper = getattr(
                tex,
                "nvshmem_cp_branch_b_true_fused_backward_no_te_bwd_execute",
                None,
            )
            if helper is None:
                raise NvshmemCpFaOwnerError(
                    "Branch-B true fused backward selected, but TE does not expose "
                    "nvshmem_cp_branch_b_true_fused_backward_no_te_bwd_execute. "
                    "This path is intentionally fail-closed: the next production candidate "
                    "must avoid TE grouped/section backward calls rather than falling back to "
                    "grouped-call helpers or the default native section loop."
                )
            fused = helper(
                query,
                native_peer_keys,
                native_peer_values,
                output,
                grad_output,
                forward_aux_tensors,
                int(ctx.cp_rank),
                "sbhd",
                float(ctx.softmax_scale),
                False,
            )
            if len(fused) != 9:
                raise NvshmemCpFaOwnerError(
                    "Branch-B true fused backward TE helper must return "
                    "[dq, dk0, dk1, dk2, dk3, dv0, dv1, dv2, dv3]."
                )
            dq = fused[0]
            dk_by_owner = list(fused[1:5])
            dv_by_owner = list(fused[5:9])
            dk, dv = tex.nvshmem_cp_prefix_grad_return_execute(
                dk_by_owner,
                dv_by_owner,
                native_peer_keys[ctx.cp_rank],
                native_peer_values[ctx.cp_rank],
                ws.grad_key_return,
                ws.grad_value_return,
                ws.grad_committed_epoch,
                ws.peer_grad_key_returns,
                ws.peer_grad_value_returns,
                ws.peer_grad_committed_epochs,
                4,
                ctx.cp_rank,
            )
            result = (dq, dk, dv)
        elif ctx.fa4_global:
            tex = _import_te_for_native_fused_owner()
            result = _branch_b_fa4_global_backward(
                tex=tex,
                fa4_state=ctx.fa4_state,
                grad_output=grad_output,
                peer_keys=native_peer_keys,
                peer_values=native_peer_values,
                ws=ws,
                cp_rank=ctx.cp_rank,
            )
        elif grouped_sections_backward:
            tex = _import_te_for_native_fused_owner()
            result = _branch_b_grouped_sections_backward(
                tex=tex,
                query=query,
                peer_keys=native_peer_keys,
                peer_values=native_peer_values,
                output=output,
                grad_output=grad_output,
                forward_aux=forward_aux_tensors,
                ws=ws,
                cp_rank=ctx.cp_rank,
                softmax_scale=ctx.softmax_scale,
            )
        elif concurrent_sections_backward:
            tex = _import_te_for_native_fused_owner()
            result = _branch_b_concurrent_sections_backward(
                tex=tex,
                query=query,
                peer_keys=native_peer_keys,
                peer_values=native_peer_values,
                output=output,
                grad_output=grad_output,
                forward_aux=forward_aux_tensors,
                ws=ws,
                cp_rank=ctx.cp_rank,
                softmax_scale=ctx.softmax_scale,
            )
        elif ctx.two_prefix_attention and not ctx.two_prefix_native_section_backward:
            tex = _import_te_for_native_fused_owner()
            result = _branch_b_two_prefix_backward(
                tex=tex,
                query=query,
                peer_keys=native_peer_keys,
                peer_values=native_peer_values,
                output=output,
                grad_output=grad_output,
                forward_aux=forward_aux_tensors,
                ws=ws,
                cp_rank=ctx.cp_rank,
                softmax_scale=ctx.softmax_scale,
            )
        else:
            result = ctx.native_backward(
                query,
                key,
                value,
                native_peer_keys,
                native_peer_values,
                output,
                grad_output,
                forward_aux_tensors,
                ws.block_ready_epoch,
                ws.peer_block_ready_epochs,
                ws.grad_key_return,
                ws.grad_value_return,
                ws.grad_committed_epoch,
                ws.peer_grad_key_returns,
                ws.peer_grad_value_returns,
                ws.peer_grad_committed_epochs,
                ws.raw_key_return,
                ws.raw_value_return,
                list(ws.peer_raw_key_returns),
                list(ws.peer_raw_value_returns),
                ws.carrier_key_slots,
                ws.carrier_value_slots,
                ws.carrier_epoch,
                ws.peer_carrier_key_slots,
                ws.peer_carrier_value_slots,
                ws.peer_carrier_epochs,
                ctx.cp_size,
                ctx.cp_rank,
                "sbhd",
                ctx.is_causal,
                float(ctx.softmax_scale),
                0.0,
            )
        native_backward_ms = None
        if native_end is not None and native_start is not None:
            if (
                os.getenv("MEGATRON_NVSHMEM_CP_BACKWARD_DEVICE_SYNC_TIMING") == "1"
            ):
                torch.cuda.synchronize(grad_output.device)
            native_end.record()
            native_end.synchronize()
            native_backward_ms = float(native_start.elapsed_time(native_end))
        if not (isinstance(result, (list, tuple)) and len(result) == 3):
            raise NvshmemCpFaOwnerError(
                "nvshmem_cp_block_ready_fused_owner_backward_execute must return "
                "(grad_query, grad_key, grad_value) and own remote K/V gradient return internally."
            )
        dq, dk, dv = result
        post_native_sync_t0 = time.perf_counter() if timing_on else None
        skip_post_native_sync = (
            os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_SKIP_POST_NATIVE_BACKWARD_SYNC") == "1"
        )
        if torch.cuda.is_available() and grad_output.is_cuda and not skip_post_native_sync:
            torch.cuda.current_stream().synchronize()
        post_native_sync_wall_ms = (
            (time.perf_counter() - post_native_sync_t0) * 1000.0
            if post_native_sync_t0 is not None
            else None
        )
        if timing_on:
            record_cp_timing(
                {
                    "event": "branch_b_native_fused_owner_backward",
                    "layer_number": getattr(ctx, "layer_number", None),
                    "call_index": int(call_index),
                    "cp_rank": int(ctx.cp_rank),
                    "cp_world_size": int(ctx.cp_size),
                    "grad_output": tensor_meta(grad_output),
                    "dq": tensor_meta(dq),
                    "dk": tensor_meta(dk),
                    "dv": tensor_meta(dv),
                    "grad_key_return": tensor_meta(ws.grad_key_return),
                    "grad_value_return": tensor_meta(ws.grad_value_return),
                    "native_backward_elapsed_ms": native_backward_ms,
                    "post_native_sync_wall_ms": post_native_sync_wall_ms,
                    "elapsed_ms": (
                        (time.perf_counter() - backward_total_t0) * 1000.0
                        if backward_total_t0 is not None
                        else None
                    ),
                    "skip_post_native_sync": bool(skip_post_native_sync),
                    "timing_scope": "Branch-B native fused owner backward; diagnostic timing may perturb runtime.",
                }
            )
        if os.getenv("MEGATRON_NVSHMEM_CP_BRANCH_B_EVICT_WORKSPACE_AFTER_BACKWARD") == "1":
            import megatron.core.transformer.nvshmem_cp_attention as nvshmem_cp_attention

            ws_key = (
                tuple(key.shape),
                str(key.dtype),
                int(key.device.index or 0),
                int(getattr(ctx, "layer_number", None) or 0),
            )
            nvshmem_cp_attention._WORKSPACES.pop(ws_key, None)
        if deferred_backward_end is not None:
            deferred_backward_end.record()
            _BRANCH_B_DEFERRED_PHASE_EVENTS.append(
                {
                    "layer_number": layer_number,
                    "cp_rank": int(ctx.cp_rank),
                    "forward_start": ctx.deferred_forward_start,
                    "forward_end": ctx.deferred_forward_end,
                    "backward_start": deferred_backward_start,
                    "backward_end": deferred_backward_end,
                }
            )
        return dq.to(ctx.query_dtype), dk.to(ctx.key_dtype), dv.to(ctx.value_dtype), None, None


def _phase_timing_enabled() -> bool:
    return cp_timing_enabled() and os.getenv("MEGATRON_NVSHMEM_CP_OWNER_PHASE_TIMING") == "1"


def _phase_start():
    if not _phase_timing_enabled():
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _phase_end(start_event, event_name: str, **metadata) -> None:
    if start_event is None:
        return
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    end_event.synchronize()
    record_cp_timing(
        {
            "event": event_name,
            "elapsed_ms": float(start_event.elapsed_time(end_event)),
            **metadata,
        }
    )


class _PublishSymmetricKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, key: torch.Tensor, value: torch.Tensor, layer_number):
        ws = _workspace(tuple(key.shape), key.dtype, key.device, layer_number)
        ws.key.copy_(key)
        ws.value.copy_(value)
        return ws.key, ws.value

    @staticmethod
    def backward(ctx, grad_key, grad_value):
        return grad_key, grad_value, None


def _normal_qkv_publish(module, hidden_states):
    query, key_normal, value_normal = module.get_query_key_value_tensors(hidden_states)
    key, value = _PublishSymmetricKV.apply(key_normal, value_normal, module.layer_number)
    return query, key, value, key_normal, value_normal


def nvshmem_cp_symmetric_qkv_self_attention_forward(
    *,
    module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    key_value_states: torch.Tensor | None = None,
    inference_context=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    rotary_pos_cos_sin=None,
    attention_bias=None,
    packed_seq_params=None,
    sequence_len_offset: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the fail-closed symmetric-QKV backend at the Megatron attention boundary."""

    unsupported = {
        "key_value_states": key_value_states,
        "inference_context": inference_context,
        "rotary_pos_emb": rotary_pos_emb,
        "rotary_pos_cos": rotary_pos_cos,
        "rotary_pos_sin": rotary_pos_sin,
        "rotary_pos_cos_sin": rotary_pos_cos_sin,
        "attention_bias": attention_bias,
        "packed_seq_params": packed_seq_params,
        "sequence_len_offset": sequence_len_offset,
    }
    active = [name for name, value in unsupported.items() if value is not None]
    if active:
        raise NvshmemCpFaOwnerError(
            f"The experimental NVSHMEM backend does not support: {', '.join(active)}."
        )
    if module.attention_type != "self":
        raise NvshmemCpFaOwnerError("The experimental NVSHMEM backend supports self-attention only.")
    if module.training is False:
        raise NvshmemCpFaOwnerError("The experimental NVSHMEM backend currently targets training only.")
    if getattr(module.config, "attention_output_gate", False):
        raise NvshmemCpFaOwnerError("The experimental NVSHMEM backend does not support output gates.")
    if getattr(module, "offload_qkv_linear", False) or getattr(module, "offload_core_attention", False):
        raise NvshmemCpFaOwnerError("The experimental NVSHMEM backend does not support activation offload.")

    bindings = _ensure_nvshmem_initialized()
    phase = _phase_start()
    query, key, value, _, _ = _normal_qkv_publish(module, hidden_states)
    _phase_end(phase, "nvshmem_symmetric_qkv_producer", layer_number=module.layer_number)

    try:
        bindings.quiet_on_stream(torch.cuda.current_stream().cuda_stream)
    except Exception:
        torch.cuda.synchronize()
    ws = _workspace(tuple(key.shape), key.dtype, key.device, module.layer_number)
    epoch = int(ws.committed_epoch[0].item()) + 1
    ws.committed_epoch[0] = epoch
    _publish_block_ready_epoch(ws, epoch)
    torch.cuda.current_stream().synchronize()
    _wait_for_block_ready_epoch(ws, epoch, _sync_timeout_seconds())

    core_attn_out = _fa_owner_io_v1_block_ready_native_fused(
        query=query,
        key=key,
        value=value,
        attn_mask_type=module.attn_mask_type,
        layer_number=module.layer_number,
    )
    hidden_size = module.num_attention_heads_per_partition * module.hidden_size_per_attention_head
    core_attn_out = core_attn_out.reshape(hidden_states.size(0), hidden_states.size(1), hidden_size)
    return module.linear_proj(core_attn_out)
