# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import megatron.core.context_parallel.layout as context_parallel_layout_module
from megatron.core import parallel_state
from megatron.core.context_parallel import (
    ContextParallelLayoutManager,
    THDCPLayoutPlan,
    build_thd_cp_layout_plan,
    contiguous_to_zigzag,
    zigzag_to_contiguous,
)
from megatron.core.context_parallel.layout import (
    _build_group_rank_by_logical_rank,
    _build_layout_redistribution_plan,
    _build_thd_cp_layout_plan_from_rank_order_indices,
    _build_thd_rank_order_indices,
    _local_segment_ids,
)
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.test_utilities import Utils


def _reference_thd_rank_order_indices(cu_seqlens, cp_size):
    """Return per-sequence dual-chunk indices concatenated in CP-rank order."""
    rank_order_indices = []
    for cp_rank in range(cp_size):
        rank_indices = []
        for sequence_start, sequence_stop in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            sequence_length = sequence_stop - sequence_start
            chunk_length = sequence_length // (2 * cp_size)
            front_start = sequence_start + cp_rank * chunk_length
            back_start = sequence_start + (2 * cp_size - cp_rank - 1) * chunk_length
            rank_indices.extend(range(front_start, front_start + chunk_length))
            rank_indices.extend(range(back_start, back_start + chunk_length))
        rank_order_indices.extend(rank_indices)
    return torch.tensor(rank_order_indices, dtype=torch.int64)


def _simulate_thd_plan(plans_by_group_rank, inputs_by_group_rank, *, reverse=False):
    """Apply the rank-local plans without distributed communication."""
    group_size = len(plans_by_group_rank)
    send_buffers = []
    split_sizes = []
    for group_rank in range(group_size):
        _, plan = plans_by_group_rank[group_rank]
        if reverse:
            send_indices = plan.reverse_send_indices
            rank_split_sizes = plan.forward_output_split_sizes
        else:
            send_indices = plan.forward_send_indices
            rank_split_sizes = plan.forward_input_split_sizes
        send_buffers.append(inputs_by_group_rank[group_rank].index_select(0, send_indices))
        split_sizes.append(rank_split_sizes)

    outputs = []
    for target_group_rank in range(group_size):
        received = []
        for source_group_rank in range(group_size):
            start = sum(split_sizes[source_group_rank][:target_group_rank])
            count = split_sizes[source_group_rank][target_group_rank]
            received.append(send_buffers[source_group_rank][start : start + count])
        received = torch.cat(received)
        _, plan = plans_by_group_rank[target_group_rank]
        if reverse:
            outputs.append(received.index_select(0, plan.reverse_receive_indices))
        else:
            output = received.new_full((plan.zigzag_local_token_count,), -1)
            outputs.append(output.index_copy(0, plan.forward_receive_positions, received))
    return outputs


def test_layout_conversion_coalesces_layout_runs(monkeypatch):
    layer_layouts = ("zigzag", "zigzag", "contiguous", "contiguous", "zigzag")
    cp_group = SimpleNamespace(size=lambda: 2)
    tp_group = object()
    tp_cp_group = object()
    manager = ContextParallelLayoutManager(
        layer_layouts=layer_layouts,
        boundary_layout="contiguous",
        sequence_parallel=True,
        cp_group=cp_group,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
    )
    calls = []

    def fake_contiguous_to_zigzag(
        hidden_states, cp_group, sequence_parallel, tp_group, tp_cp_group, thd_plan
    ):
        calls.append(("to_zigzag", cp_group, sequence_parallel, tp_group, tp_cp_group, thd_plan))
        return hidden_states + 1

    def fake_zigzag_to_contiguous(
        hidden_states, cp_group, sequence_parallel, tp_group, tp_cp_group, thd_plan
    ):
        calls.append(
            ("to_contiguous", cp_group, sequence_parallel, tp_group, tp_cp_group, thd_plan)
        )
        return hidden_states + 2

    monkeypatch.setattr(
        context_parallel_layout_module, "contiguous_to_zigzag", fake_contiguous_to_zigzag
    )
    monkeypatch.setattr(
        context_parallel_layout_module, "zigzag_to_contiguous", fake_zigzag_to_contiguous
    )

    hidden_states = torch.zeros(2, 1, 1)
    thd_plan = object()
    layout_state = manager.build_forward_state(None)
    assert layout_state is not None
    layout_state.thd_plan = thd_plan
    for layer_index in range(len(layer_layouts)):
        hidden_states, layer_packed_seq_params = layout_state.prepare_layer(
            layer_index, hidden_states
        )
        assert layer_packed_seq_params is None
        hidden_states = layout_state.finalize_layer(layer_index, hidden_states)

    assert calls == [
        ("to_zigzag", cp_group, True, tp_group, tp_cp_group, thd_plan),
        ("to_contiguous", cp_group, True, tp_group, tp_cp_group, thd_plan),
        ("to_zigzag", cp_group, True, tp_group, tp_cp_group, thd_plan),
        ("to_contiguous", cp_group, True, tp_group, tp_cp_group, thd_plan),
    ]
    torch.testing.assert_close(hidden_states, torch.full_like(hidden_states, 6))


def test_matching_cp_layouts_do_not_convert(monkeypatch):
    manager = ContextParallelLayoutManager(
        layer_layouts=("zigzag",),
        boundary_layout="zigzag",
        sequence_parallel=False,
        cp_group=SimpleNamespace(size=lambda: 2),
        tp_group=object(),
        tp_cp_group=object(),
    )

    def fail_conversion(*_args):
        raise AssertionError("matching layouts must not convert")

    monkeypatch.setattr(context_parallel_layout_module, "contiguous_to_zigzag", fail_conversion)
    monkeypatch.setattr(context_parallel_layout_module, "zigzag_to_contiguous", fail_conversion)

    hidden_states = torch.zeros(2, 1, 1)
    assert manager.prepare_layer_input(0, hidden_states) is hidden_states
    assert manager.finalize_layer_output(0, hidden_states) is hidden_states
    assert manager.build_forward_state(None) is None


def test_packed_zigzag_layout_uses_padded_metadata(monkeypatch):
    cp_group = SimpleNamespace(size=lambda: 2)
    tp_group = object()
    tp_cp_group = object()
    manager = ContextParallelLayoutManager(
        layer_layouts=("contiguous", "zigzag"),
        boundary_layout="contiguous",
        sequence_parallel=False,
        cp_group=cp_group,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
    )
    cu_seqlens = torch.tensor((0, 7, 16), dtype=torch.int32)
    target_cu_seqlens_padded = torch.tensor((0, 8, 24), dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=9,
        max_seqlen_kv=9,
        total_tokens=16,
        cp_partition_mode="contiguous",
    )
    plan = SimpleNamespace(
        cu_seqlens_padded=target_cu_seqlens_padded, max_seqlen_padded=16, pad_between_seqs=True
    )
    monkeypatch.setattr(
        context_parallel_layout_module, "build_thd_cp_layout_plan", lambda *_args, **_kwargs: plan
    )

    layout_state = manager.build_forward_state(packed_seq_params)

    assert layout_state is not None
    assert layout_state.thd_plan is plan
    zigzag_params = layout_state.zigzag_packed_seq_params
    assert zigzag_params is not None
    assert zigzag_params.cu_seqlens_q is cu_seqlens
    assert zigzag_params.cu_seqlens_kv is cu_seqlens
    assert zigzag_params.cu_seqlens_q_padded is target_cu_seqlens_padded
    assert zigzag_params.cu_seqlens_kv_padded is target_cu_seqlens_padded
    assert zigzag_params.max_seqlen_q == 16
    assert zigzag_params.max_seqlen_kv == 16
    assert zigzag_params.pad_between_seqs
    assert zigzag_params.cp_partition_mode == "zigzag"
    assert zigzag_params.total_tokens is None
    assert zigzag_params.seq_idx is None

    monkeypatch.setattr(manager, "prepare_layer_input", lambda _index, hidden, _plan: hidden)
    hidden_states = torch.zeros(2, 1, 1)
    _, layer_packed_seq_params = layout_state.prepare_layer(0, hidden_states)
    assert layer_packed_seq_params is packed_seq_params
    _, layer_packed_seq_params = layout_state.prepare_layer(1, hidden_states)
    assert layer_packed_seq_params is zigzag_params


@pytest.mark.parametrize(
    ("cp_global_ranks", "tp_global_ranks", "current_global_rank", "expected"),
    [
        ((1, 3, 5, 7), (4, 5), 5, tuple(range(8))),
        ((4, 5, 6, 7), (2, 6), 6, (0, 4, 1, 5, 2, 6, 3, 7)),
    ],
    ids=["tp-fastest", "cp-fastest"],
)
def test_group_rank_mapping_handles_parallel_rank_order(
    cp_global_ranks, tp_global_ranks, current_global_rank, expected
):
    assert (
        _build_group_rank_by_logical_rank(
            cp_global_ranks=cp_global_ranks,
            tp_global_ranks=tp_global_ranks,
            tp_cp_global_ranks=tuple(range(8)),
            current_global_rank=current_global_rank,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("cp_size", "tp_size"), [(2, 1), (3, 2), (4, 4)], ids=["cp2", "cp3-tp2", "cp4-tp4"]
)
@pytest.mark.parametrize(
    ("source_layout", "target_layout"), [("contiguous", "zigzag"), ("zigzag", "contiguous")]
)
def test_layout_redistribution_plan_restores_target_segments(
    cp_size, tp_size, source_layout, target_layout
):
    group_size = cp_size * tp_size
    sent_by_rank = []
    for source_logical_rank in range(group_size):
        source_cp_rank, source_tp_rank = divmod(source_logical_rank, tp_size)
        plan = _build_layout_redistribution_plan(
            source_layout,
            target_layout,
            cp_size,
            source_cp_rank,
            tp_size=tp_size,
            tp_rank=source_tp_rank,
        )
        source_ids = _local_segment_ids(
            source_layout, cp_size, source_cp_rank, tp_size=tp_size, tp_rank=source_tp_rank
        )
        send_ids = [source_ids[slot] for slot in plan.send_slots]
        offset = 0
        destinations = []
        for target_rank, count in enumerate(plan.input_segment_counts):
            destinations.extend(
                (target_rank, segment_id) for segment_id in send_ids[offset : offset + count]
            )
            offset += count
        sent_by_rank.append(destinations)

    for target_logical_rank in range(group_size):
        target_cp_rank, target_tp_rank = divmod(target_logical_rank, tp_size)
        plan = _build_layout_redistribution_plan(
            source_layout,
            target_layout,
            cp_size,
            target_cp_rank,
            tp_size=tp_size,
            tp_rank=target_tp_rank,
        )
        received = [
            segment_id
            for source_entries in sent_by_rank
            for destination, segment_id in source_entries
            if destination == target_logical_rank
        ]
        actual = tuple(received[index] for index in plan.receive_permutation)
        assert actual == _local_segment_ids(
            target_layout, cp_size, target_cp_rank, tp_size=tp_size, tp_rank=target_tp_rank
        )


def test_thd_rank_order_indices_pad_uneven_sequences():
    rank_order_indices, padded_cu_seqlens = _build_thd_rank_order_indices(
        torch.tensor((0, 3, 8), dtype=torch.int32), None, cp_size=2, tp_size=1
    )

    torch.testing.assert_close(padded_cu_seqlens, torch.tensor((0, 4, 12), dtype=torch.int32))
    torch.testing.assert_close(
        rank_order_indices,
        torch.tensor((0, -1, 3, 4, -1, -1, 1, 2, 5, 6, 7, -1), dtype=torch.int64),
    )


@pytest.mark.parametrize(
    ("cp_size", "tp_size", "cu_seqlens", "group_rank_by_logical_rank"),
    [(2, 1, (0, 8, 24), tuple(range(2))), (4, 2, (0, 16, 48), (0, 4, 1, 5, 2, 6, 3, 7))],
    ids=["cp2", "cp4-tp2-cp-fastest"],
)
def test_thd_layout_plan_routes_per_sequence_zigzag_tokens(
    cp_size, tp_size, cu_seqlens, group_rank_by_logical_rank
):
    rank_order_indices = _reference_thd_rank_order_indices(cu_seqlens, cp_size)
    group_size = cp_size * tp_size
    local_sequence_length = rank_order_indices.numel() // group_size
    plans_by_group_rank = [None] * group_size
    contiguous_by_group_rank = [None] * group_size

    for logical_rank, group_rank in enumerate(group_rank_by_logical_rank):
        cp_rank, tp_rank = divmod(logical_rank, tp_size)
        plan = _build_thd_cp_layout_plan_from_rank_order_indices(
            rank_order_indices,
            source_token_count=rank_order_indices.numel(),
            cu_seqlens_padded=torch.tensor(cu_seqlens, dtype=torch.int32),
            cp_size=cp_size,
            cp_rank=cp_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            group_rank_by_logical_rank=group_rank_by_logical_rank,
        )
        assert not plan.pad_between_seqs
        plans_by_group_rank[group_rank] = (logical_rank, plan)
        contiguous_by_group_rank[group_rank] = torch.arange(
            logical_rank * local_sequence_length,
            (logical_rank + 1) * local_sequence_length,
            dtype=torch.int64,
        )

    zigzag_by_group_rank = _simulate_thd_plan(plans_by_group_rank, contiguous_by_group_rank)
    expected_by_logical_rank = rank_order_indices.view(group_size, local_sequence_length)
    for group_rank, (logical_rank, _) in enumerate(plans_by_group_rank):
        torch.testing.assert_close(
            zigzag_by_group_rank[group_rank], expected_by_logical_rank[logical_rank]
        )

    restored_by_group_rank = _simulate_thd_plan(
        plans_by_group_rank, zigzag_by_group_rank, reverse=True
    )
    for restored, expected in zip(restored_by_group_rank, contiguous_by_group_rank):
        torch.testing.assert_close(restored, expected)


@pytest.mark.parametrize(
    ("cp_size", "tp_size", "cu_seqlens", "cu_seqlens_padded"),
    [(4, 1, (0, 3, 10, 16), None), (4, 2, (0, 7, 16), None), (4, 1, (0, 5, 11), (0, 8, 16))],
    ids=["cp4", "cp4-tp2", "existing-physical-padding"],
)
def test_thd_layout_plan_handles_padding(cp_size, tp_size, cu_seqlens, cu_seqlens_padded):
    actual_cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32)
    source_cu_seqlens_padded = (
        torch.tensor(cu_seqlens_padded, dtype=torch.int32)
        if cu_seqlens_padded is not None
        else None
    )
    rank_order_indices, target_cu_seqlens_padded = _build_thd_rank_order_indices(
        actual_cu_seqlens, source_cu_seqlens_padded, cp_size, tp_size
    )
    assert target_cu_seqlens_padded.dtype == torch.int32
    if source_cu_seqlens_padded is not None:
        torch.testing.assert_close(target_cu_seqlens_padded, source_cu_seqlens_padded)
    group_size = cp_size * tp_size
    source_token_count = (
        cu_seqlens[-1] if source_cu_seqlens_padded is None else cu_seqlens_padded[-1]
    )
    local_source_token_count = source_token_count // group_size
    plans_by_group_rank = []
    inputs_by_group_rank = []

    for logical_rank in range(group_size):
        cp_rank, tp_rank = divmod(logical_rank, tp_size)
        plan = _build_thd_cp_layout_plan_from_rank_order_indices(
            rank_order_indices,
            source_token_count=source_token_count,
            cu_seqlens_padded=target_cu_seqlens_padded,
            cp_size=cp_size,
            cp_rank=cp_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            pad_between_seqs=True,
        )
        assert plan.pad_between_seqs
        plans_by_group_rank.append((logical_rank, plan))
        inputs_by_group_rank.append(
            torch.arange(
                logical_rank * local_source_token_count,
                (logical_rank + 1) * local_source_token_count,
            )
        )

    zigzag_by_group_rank = _simulate_thd_plan(plans_by_group_rank, inputs_by_group_rank)
    torch.testing.assert_close(torch.cat(zigzag_by_group_rank), rank_order_indices)

    restored_by_group_rank = _simulate_thd_plan(
        plans_by_group_rank, zigzag_by_group_rank, reverse=True
    )
    for restored, expected in zip(restored_by_group_rank, inputs_by_group_rank):
        torch.testing.assert_close(restored, expected)


_DISTRIBUTED_LAYOUT_CASES = [
    pytest.param(
        1,
        Utils.world_size,
        marks=pytest.mark.skipif(
            not torch.cuda.is_available() or Utils.world_size < 2,
            reason="CP layout conversion requires at least two GPUs",
        ),
        id="cp",
    ),
    pytest.param(
        2,
        4,
        marks=pytest.mark.skipif(
            not torch.cuda.is_available() or Utils.world_size < 8 or Utils.world_size % 8 != 0,
            reason="TP2 x CP4 layout conversion requires a multiple of eight GPUs",
        ),
        id="tp2-cp4",
    ),
]


@pytest.mark.internal
@pytest.mark.parametrize(("tp_size", "cp_size"), _DISTRIBUTED_LAYOUT_CASES)
def test_layout_all_to_all_round_trip_and_backward(tp_size, cp_size):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    try:
        sequence_parallel = tp_size > 1
        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = dist.get_rank(group=cp_group)
        tp_group = parallel_state.get_tensor_model_parallel_group() if sequence_parallel else None
        tp_cp_group = (
            parallel_state.get_tensor_and_context_parallel_group() if sequence_parallel else None
        )
        tp_rank = dist.get_rank(group=tp_group) if sequence_parallel else 0
        segment_len = 3
        contiguous_ids = _local_segment_ids(
            "contiguous", cp_size, cp_rank, tp_size=tp_size, tp_rank=tp_rank
        )
        global_segment_count = cp_size * tp_size * len(contiguous_ids)
        global_values = torch.arange(
            global_segment_count * segment_len, dtype=torch.float32, device="cuda"
        ).view(-1, 1, 1)
        contiguous = torch.cat(
            [
                global_values[segment_id * segment_len : (segment_id + 1) * segment_len]
                for segment_id in contiguous_ids
            ]
        ).requires_grad_(True)

        zigzag = contiguous_to_zigzag(
            contiguous, cp_group, sequence_parallel, tp_group, tp_cp_group
        )
        zigzag_ids = _local_segment_ids(
            "zigzag", cp_size, cp_rank, tp_size=tp_size, tp_rank=tp_rank
        )
        expected_zigzag = torch.cat(
            [
                global_values[segment_id * segment_len : (segment_id + 1) * segment_len]
                for segment_id in zigzag_ids
            ]
        )
        torch.testing.assert_close(zigzag, expected_zigzag)

        restored = zigzag_to_contiguous(zigzag, cp_group, sequence_parallel, tp_group, tp_cp_group)
        torch.testing.assert_close(restored, contiguous)
        restored.sum().backward()
        torch.testing.assert_close(contiguous.grad, torch.ones_like(contiguous))
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize(("tp_size", "cp_size"), _DISTRIBUTED_LAYOUT_CASES)
def test_thd_layout_all_to_all_pads_and_round_trips(tp_size, cp_size):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    try:
        sequence_parallel = tp_size > 1
        cp_group = parallel_state.get_context_parallel_group()
        cp_rank = dist.get_rank(group=cp_group)
        tp_group = parallel_state.get_tensor_model_parallel_group() if sequence_parallel else None
        tp_cp_group = (
            parallel_state.get_tensor_and_context_parallel_group() if sequence_parallel else None
        )
        tp_rank = dist.get_rank(group=tp_group) if sequence_parallel else 0
        group_size = tp_size * cp_size
        cu_seqlens_values = (0, 2 * cp_size - 1, 4 * cp_size)
        total_tokens = cu_seqlens_values[-1]
        local_sequence_length = total_tokens // group_size
        logical_rank = cp_rank * tp_size + tp_rank
        cu_seqlens = torch.tensor(cu_seqlens_values, dtype=torch.int32, device="cuda")
        global_values = torch.arange(total_tokens, dtype=torch.float32, device="cuda").view(
            -1, 1, 1
        )
        contiguous = (
            global_values[
                logical_rank * local_sequence_length : (logical_rank + 1) * local_sequence_length
            ]
            .clone()
            .requires_grad_(True)
        )
        plan = build_thd_cp_layout_plan(
            cu_seqlens,
            total_tokens,
            cp_group,
            sequence_parallel=sequence_parallel,
            tp_group=tp_group,
            tp_cp_group=tp_cp_group,
        )
        assert plan.pad_between_seqs
        zigzag = contiguous_to_zigzag(
            contiguous, cp_group, sequence_parallel, tp_group, tp_cp_group, plan
        )
        rank_order_indices, _ = _build_thd_rank_order_indices(cu_seqlens, None, cp_size, tp_size)
        expected = global_values.new_zeros(
            (plan.zigzag_local_token_count, *global_values.shape[1:])
        )
        expected_indices = rank_order_indices.view(group_size, -1)[logical_rank]
        valid_positions = torch.nonzero(expected_indices >= 0, as_tuple=False).flatten()
        expected.index_copy_(
            0,
            valid_positions,
            global_values.index_select(0, expected_indices.index_select(0, valid_positions)),
        )
        torch.testing.assert_close(zigzag, expected)

        restored = zigzag_to_contiguous(
            zigzag, cp_group, sequence_parallel, tp_group, tp_cp_group, plan
        )
        torch.testing.assert_close(restored, contiguous)
        restored.sum().backward()
        torch.testing.assert_close(contiguous.grad, torch.ones_like(contiguous))
    finally:
        Utils.destroy_model_parallel()
