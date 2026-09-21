# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from types import SimpleNamespace

import pytest
import torch

import megatron.core.context_parallel_layout.conversion as context_parallel_layout_conversion
import megatron.core.context_parallel_layout.routes as context_parallel_layout_routes
from megatron.core import parallel_state
from megatron.core.context_parallel_layout import (
    CpPartitionModeConverter,
    ThdCpRoute,
    convert_module_input_tensors_cp_partition_mode,
    prebuild_thd_cp_partition_routes,
)
from megatron.core.context_parallel_layout.routes import (
    build_thd_cp_partition_route,
    build_thd_tp_cp_partition_route,
    build_tp_cp_group_rank_by_logical_rank,
    get_thd_cp_partition_route,
    get_thd_tp_cp_partition_route,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from tests.unit_tests.test_utilities import Utils


class _FakeGroup:

    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _make_sequence_tensor(total_seq_len, seq_dim, device):
    if seq_dim == 0:
        shape = (total_seq_len, 3, 5)
    elif seq_dim == 1:
        shape = (3, total_seq_len, 5)
    else:
        raise ValueError(f"Unsupported test seq_dim {seq_dim}.")
    return torch.arange(torch.prod(torch.tensor(shape)), device=device, dtype=torch.float32).view(
        *shape
    )


def _get_sequence_parallel_shard(tensor, seq_dim, tp_group):
    tp_size = tp_group.size()
    tp_rank = tp_group.rank()
    assert tensor.size(seq_dim) % tp_size == 0
    return tensor.chunk(tp_size, dim=seq_dim)[tp_rank].contiguous()


def _get_sbhd_tensor_on_this_cp_rank(tensor, seq_dim, cp_group, cp_partition_mode):
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    if cp_partition_mode == "zigzag":
        cp_idx = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], device=tensor.device)
    elif cp_partition_mode == "contiguous":
        cp_idx = torch.tensor([2 * cp_rank, 2 * cp_rank + 1], device=tensor.device)
    else:
        raise ValueError(f"Unsupported test CP partition mode {cp_partition_mode!r}.")
    tensor = tensor.view(*tensor.shape[:seq_dim], 2 * cp_size, -1, *tensor.shape[(seq_dim + 1) :])
    tensor = tensor.index_select(seq_dim, cp_idx)
    return tensor.view(*tensor.shape[:seq_dim], -1, *tensor.shape[(seq_dim + 2) :])


def _get_test_thd_token_indices(cu_seqlens, cp_size, cp_rank, cp_partition_mode):
    cu = cu_seqlens.to(dtype=torch.long).tolist()
    compact_cu = [cu[0]]
    for value in cu[1:]:
        if value != compact_cu[-1]:
            compact_cu.append(value)

    total_tokens = compact_cu[-1]
    if cp_partition_mode == "contiguous":
        part_len = total_tokens // cp_size
        start = cp_rank * part_len
        return torch.arange(start, start + part_len, dtype=torch.long)
    if cp_partition_mode != "zigzag":
        raise ValueError(f"Unsupported test CP partition mode {cp_partition_mode!r}.")

    token_indices = []
    for seq_start, seq_end in zip(compact_cu[:-1], compact_cu[1:]):
        chunk_len = (seq_end - seq_start) // (2 * cp_size)
        first_start = seq_start + cp_rank * chunk_len
        second_chunk = 2 * cp_size - cp_rank - 1
        second_start = seq_start + second_chunk * chunk_len
        token_indices.extend(range(first_start, first_start + chunk_len))
        token_indices.extend(range(second_start, second_start + chunk_len))
    return torch.tensor(token_indices, dtype=torch.long)


def _get_test_thd_sp_shard_token_indices(
    cu_seqlens, cp_size, cp_rank, tp_size, tp_rank, cp_partition_mode
):
    """Global token indices held by one (cp_rank, tp_rank) sequence-parallel THD shard."""
    cp_local = _get_test_thd_token_indices(cu_seqlens, cp_size, cp_rank, cp_partition_mode)
    assert cp_local.numel() % tp_size == 0
    return cp_local.chunk(tp_size)[tp_rank].contiguous()


def _simulate_thd_route_all_to_all(routes, source_tokens_by_rank, source_layout, target_layout):
    """Run the pack -> all-to-all-v -> scatter exchange of THD routes on the host.

    ``routes`` and ``source_tokens_by_rank`` are indexed by communication-group rank.
    Returns the per-rank tensors in target-layout local order.
    """
    group_size = len(routes)
    if source_layout == "zigzag" and target_layout == "contiguous":
        selected = [
            (r.zigzag_index, r.contiguous_index, r.zigzag_split_sizes, r.contiguous_split_sizes)
            for r in routes
        ]
    else:
        selected = [
            (r.contiguous_index, r.zigzag_index, r.contiguous_split_sizes, r.zigzag_split_sizes)
            for r in routes
        ]

    send_buffers = []
    for rank, (send_index, _, send_split_sizes, _) in enumerate(selected):
        assert sum(send_split_sizes) == source_tokens_by_rank[rank].numel()
        send_buffers.append(
            source_tokens_by_rank[rank]
            if send_index is None
            else source_tokens_by_rank[rank].index_select(0, send_index)
        )

    outputs = []
    for dst_rank in range(group_size):
        recv_chunks = []
        for src_rank in range(group_size):
            send_split_sizes = selected[src_rank][2]
            send_offset = sum(send_split_sizes[:dst_rank])
            recv_chunks.append(
                send_buffers[src_rank].narrow(0, send_offset, send_split_sizes[dst_rank])
            )
        recv_buf = torch.cat(recv_chunks, dim=0)
        _, recv_index, _, recv_split_sizes = selected[dst_rank]
        assert recv_buf.numel() == sum(recv_split_sizes)
        if recv_index is None:
            outputs.append(recv_buf)
        else:
            out = torch.empty(sum(recv_split_sizes), dtype=recv_buf.dtype)
            out.index_copy_(0, recv_index, recv_buf)
            outputs.append(out)
    return outputs


def _convert_thd_sp_shard_via_tp_gather(
    x, *, source_layout, target_layout, cu_seqlens, cp_group, tp_group
):
    """Pre-fusion reference: TP gather -> CP-only THD conversion -> TP scatter.

    This is the composition the fused TP x CP all-to-all replaced; it is kept here
    so the two can be compared bit for bit in both directions.
    """
    gathered = gather_from_sequence_parallel_region(
        input_=x, tensor_parallel_output_grad=False, group=tp_group
    )
    converted = context_parallel_layout_conversion.convert_cp_partition_mode(
        x=gathered,
        source_partition_mode=source_layout,
        target_partition_mode=target_layout,
        seq_dim=0,
        cu_seqlens=cu_seqlens,
        cp_group=cp_group,
    )
    return scatter_to_sequence_parallel_region(input_=converted, group=tp_group)


def _assert_thd_routes_equal(actual, expected):
    for field in ("zigzag_index", "contiguous_index"):
        actual_index = getattr(actual, field)
        expected_index = getattr(expected, field)
        if expected_index is None:
            assert actual_index is None
        else:
            assert torch.equal(actual_index, expected_index)
    assert actual.zigzag_split_sizes == expected.zigzag_split_sizes
    assert actual.contiguous_split_sizes == expected.contiguous_split_sizes


@pytest.mark.parametrize(
    ("source_layout", "target_layout"), [("zigzag", "contiguous"), ("contiguous", "zigzag")]
)
@pytest.mark.parametrize(
    ("cp_size", "tp_size", "group_rank_by_logical_rank"),
    [(3, 1, (0, 1, 2)), (2, 2, (0, 2, 1, 3)), (2, 4, tuple(range(8)))],
)
def test_sbhd_layout_redistribution_plan_reassembles_target_segments(
    source_layout, target_layout, cp_size, tp_size, group_rank_by_logical_rank
):
    group_size = cp_size * tp_size
    plans = [None] * group_size
    sends = [[None] * group_size for _ in range(group_size)]

    for logical_rank, group_rank in enumerate(group_rank_by_logical_rank):
        cp_rank, tp_rank = divmod(logical_rank, tp_size)
        source_ids = context_parallel_layout_conversion._local_sbhd_segment_ids(
            layout=source_layout, cp_size=cp_size, cp_rank=cp_rank, tp_size=tp_size, tp_rank=tp_rank
        )
        plan = context_parallel_layout_conversion._build_sbhd_layout_redistribution_plan(
            source_layout=source_layout,
            target_layout=target_layout,
            cp_size=cp_size,
            cp_rank=cp_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
            group_rank_by_logical_rank=group_rank_by_logical_rank,
        )
        plans[group_rank] = plan
        packed_ids = tuple(source_ids[slot] for slot in plan.send_slots)
        offset = 0
        for destination, count in enumerate(plan.input_segment_counts):
            sends[group_rank][destination] = packed_ids[offset : offset + count]
            offset += count

    for logical_rank, group_rank in enumerate(group_rank_by_logical_rank):
        cp_rank, tp_rank = divmod(logical_rank, tp_size)
        plan = plans[group_rank]
        received_ids = tuple(
            segment_id
            for source_group_rank in range(group_size)
            for segment_id in sends[source_group_rank][group_rank]
        )
        output_ids = tuple(received_ids[index] for index in plan.receive_permutation)
        assert output_ids == context_parallel_layout_conversion._local_sbhd_segment_ids(
            layout=target_layout, cp_size=cp_size, cp_rank=cp_rank, tp_size=tp_size, tp_rank=tp_rank
        )


def test_sbhd_layout_redistribution_rejects_odd_tensor_parallel_size():
    with pytest.raises(ValueError, match="even tensor-parallel size"):
        context_parallel_layout_conversion._sbhd_segments_per_rank(tp_size=3)


@pytest.mark.parametrize(
    ("source_layout", "target_layout"), [("zigzag", "contiguous"), ("contiguous", "zigzag")]
)
@pytest.mark.parametrize(
    ("cu_seqlens", "cp_size"),
    [
        (torch.tensor([0, 16, 40]), 2),
        (torch.tensor([0, 32, 96, 128]), 4),
        (torch.tensor([0, 32, 96, 128, 128, 128]), 4),
    ],
)
def test_thd_cp_partition_route_reassembles_target_layout(
    source_layout, target_layout, cu_seqlens, cp_size
):
    source_indices = [
        _get_test_thd_token_indices(cu_seqlens, cp_size, rank, source_layout)
        for rank in range(cp_size)
    ]
    target_indices = [
        _get_test_thd_token_indices(cu_seqlens, cp_size, rank, target_layout)
        for rank in range(cp_size)
    ]
    routes = [build_thd_cp_partition_route(cu_seqlens, cp_size, rank) for rank in range(cp_size)]
    if source_layout == "zigzag" and target_layout == "contiguous":
        selected_routes = [
            (
                route.zigzag_index,
                route.contiguous_index,
                route.zigzag_split_sizes,
                route.contiguous_split_sizes,
            )
            for route in routes
        ]
    else:
        selected_routes = [
            (
                route.contiguous_index,
                route.zigzag_index,
                route.contiguous_split_sizes,
                route.zigzag_split_sizes,
            )
            for route in routes
        ]
    for rank, (_, _, send_split_sizes, recv_split_sizes) in enumerate(selected_routes):
        assert sum(send_split_sizes) == source_indices[rank].numel()
        assert sum(recv_split_sizes) == target_indices[rank].numel()
        assert source_indices[rank].numel() == target_indices[rank].numel()

    send_buffers = []
    for rank, (send_index, _, _, _) in enumerate(selected_routes):
        send_buffers.append(
            source_indices[rank]
            if send_index is None
            else source_indices[rank].index_select(0, send_index)
        )

    for dst_rank in range(cp_size):
        recv_chunks = []
        for src_rank in range(cp_size):
            _, _, send_split_sizes, _ = selected_routes[src_rank]
            send_offset = sum(send_split_sizes[:dst_rank])
            send_len = send_split_sizes[dst_rank]
            recv_chunks.append(send_buffers[src_rank].narrow(0, send_offset, send_len))
        recv_buf = torch.cat(recv_chunks, dim=0)
        _, recv_index, _, recv_split_sizes = selected_routes[dst_rank]
        local_target_length = sum(recv_split_sizes)
        if recv_index is None:
            out = recv_buf
        else:
            out = torch.empty(local_target_length, dtype=recv_buf.dtype)
            out.index_copy_(0, recv_index, recv_buf)
        assert torch.equal(out, target_indices[dst_rank])


def test_thd_cp_partition_route_stores_bidirectional_layout_views():
    route = build_thd_cp_partition_route(torch.tensor([0, 8, 12, 16]), cp_size=2, cp_rank=0)

    assert isinstance(route, ThdCpRoute)
    assert route.zigzag_index is None
    assert route.zigzag_split_sizes == [4, 4]
    assert route.contiguous_index.tolist() == [0, 1, 6, 7, 2, 3, 4, 5]
    assert route.contiguous_split_sizes == [4, 4]

    c2z_send_index = route.contiguous_index
    c2z_recv_index = route.zigzag_index
    c2z_send_splits = route.contiguous_split_sizes
    c2z_recv_splits = route.zigzag_split_sizes
    z2c_send_index = route.zigzag_index
    z2c_recv_index = route.contiguous_index
    z2c_send_splits = route.zigzag_split_sizes
    z2c_recv_splits = route.contiguous_split_sizes

    assert c2z_send_index is route.contiguous_index
    assert c2z_recv_index is route.zigzag_index
    assert c2z_send_splits is route.contiguous_split_sizes
    assert c2z_recv_splits is route.zigzag_split_sizes
    assert z2c_send_index is route.zigzag_index
    assert z2c_recv_index is route.contiguous_index
    assert z2c_send_splits is route.zigzag_split_sizes
    assert z2c_recv_splits is route.contiguous_split_sizes


def test_build_thd_cp_partition_route_rejects_decreasing_boundaries():
    with pytest.raises(ValueError, match="nondecreasing"):
        build_thd_cp_partition_route(
            torch.tensor([0, 16, 8], dtype=torch.int32), cp_size=2, cp_rank=0
        )


@pytest.mark.internal
@pytest.mark.parametrize(
    ("source_layout", "target_layout"), [("zigzag", "contiguous"), ("contiguous", "zigzag")]
)
@pytest.mark.parametrize("seq_dim", [0, 1])
def test_sbhd_convert_cp_partition_mode_matches_direct_target_shard(
    source_layout, target_layout, seq_dim
):
    if not torch.cuda.is_available() or Utils.world_size < 2:
        pytest.skip("SBHD CP partition-mode conversion needs at least two CUDA ranks.")

    cp_size = 2
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_size)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        full_tensor = _make_sequence_tensor(
            total_seq_len=32,
            seq_dim=seq_dim,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
        source_shard = _get_sbhd_tensor_on_this_cp_rank(
            full_tensor, seq_dim, cp_group, cp_partition_mode=source_layout
        )

        converted = context_parallel_layout_conversion.convert_cp_partition_mode(
            x=source_shard,
            source_partition_mode=source_layout,
            target_partition_mode=target_layout,
            seq_dim=seq_dim,
            cp_group=cp_group,
        )
        expected = _get_sbhd_tensor_on_this_cp_rank(
            full_tensor, seq_dim, cp_group, cp_partition_mode=target_layout
        )

        torch.testing.assert_close(converted, expected, atol=0.0, rtol=0.0)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize(
    ("source_layout", "target_layout", "seq_dim", "sequence_parallel"),
    [
        pytest.param("zigzag", "contiguous", 0, False, id="zigzag-contiguous-seq0"),
        pytest.param("zigzag", "contiguous", 1, False, id="zigzag-contiguous-seq1"),
        pytest.param("contiguous", "zigzag", 0, False, id="contiguous-zigzag-seq0"),
        pytest.param("contiguous", "zigzag", 1, False, id="contiguous-zigzag-seq1"),
        pytest.param("zigzag", "contiguous", 0, True, id="sp-zigzag-contiguous-seq0"),
        pytest.param("zigzag", "contiguous", 1, True, id="sp-zigzag-contiguous-seq1"),
        pytest.param("contiguous", "zigzag", 0, True, id="sp-contiguous-zigzag-seq0"),
        pytest.param("contiguous", "zigzag", 1, True, id="sp-contiguous-zigzag-seq1"),
    ],
)
def test_sbhd_convert_cp_partition_mode_backward_matches_direct_source_shard(
    source_layout, target_layout, seq_dim, sequence_parallel
):
    min_world_size = 4 if sequence_parallel else 2
    if not torch.cuda.is_available() or Utils.world_size < min_world_size:
        pytest.skip(
            f"SBHD CP partition-mode conversion backward needs at least {min_world_size} "
            "CUDA ranks."
        )

    cp_size = 2
    tp_size = 2 if sequence_parallel else 1
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    try:
        cp_group = parallel_state.get_context_parallel_group()
        tp_group = parallel_state.get_tensor_model_parallel_group() if sequence_parallel else None
        tp_cp_group = (
            parallel_state.get_tensor_and_context_parallel_group() if sequence_parallel else None
        )
        full_tensor = _make_sequence_tensor(
            total_seq_len=32,
            seq_dim=seq_dim,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
        full_upstream_grad = full_tensor.mul(0.125).add(1.0)
        source_shard = _get_sbhd_tensor_on_this_cp_rank(
            full_tensor, seq_dim, cp_group, cp_partition_mode=source_layout
        )
        if sequence_parallel:
            source_shard = _get_sequence_parallel_shard(source_shard, seq_dim, tp_group)
        source_shard = source_shard.detach().requires_grad_(True)

        convert_kwargs = (
            {"sequence_parallel": True, "tp_group": tp_group, "tp_cp_group": tp_cp_group}
            if sequence_parallel
            else {}
        )
        converted = context_parallel_layout_conversion.convert_cp_partition_mode(
            x=source_shard,
            source_partition_mode=source_layout,
            target_partition_mode=target_layout,
            seq_dim=seq_dim,
            cp_group=cp_group,
            **convert_kwargs,
        )
        expected_target = _get_sbhd_tensor_on_this_cp_rank(
            full_tensor, seq_dim, cp_group, cp_partition_mode=target_layout
        )
        if sequence_parallel:
            expected_target = _get_sequence_parallel_shard(expected_target, seq_dim, tp_group)
        torch.testing.assert_close(converted, expected_target, atol=0.0, rtol=0.0)

        target_upstream_grad = _get_sbhd_tensor_on_this_cp_rank(
            full_upstream_grad, seq_dim, cp_group, cp_partition_mode=target_layout
        )
        if sequence_parallel:
            target_upstream_grad = _get_sequence_parallel_shard(
                target_upstream_grad, seq_dim, tp_group
            )
        converted.mul(target_upstream_grad).sum().backward()
        expected_source_grad = _get_sbhd_tensor_on_this_cp_rank(
            full_upstream_grad, seq_dim, cp_group, cp_partition_mode=source_layout
        )
        if sequence_parallel:
            expected_source_grad = _get_sequence_parallel_shard(
                expected_source_grad, seq_dim, tp_group
            )

        torch.testing.assert_close(source_shard.grad, expected_source_grad, atol=0.0, rtol=0.0)
    finally:
        Utils.destroy_model_parallel()


def test_prebuild_thd_cp_partition_routes_populates_direct_fields():
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 16, 40]),
        cu_seqlens_q_padded=None,
        cp_partition_route=None,
    )
    cp_group = _FakeGroup(size=2, rank=0)
    prebuild_thd_cp_partition_routes(packed_seq_params, cp_group)

    route = get_thd_cp_partition_route(packed_seq_params, "zigzag", "contiguous")
    same_route = get_thd_cp_partition_route(packed_seq_params, "zigzag", "contiguous")
    reverse_route = get_thd_cp_partition_route(packed_seq_params, "contiguous", "zigzag")

    assert same_route is route
    assert reverse_route is route
    assert packed_seq_params.cp_partition_route is route
    assert packed_seq_params.thd_cp_host_cu_seqlens_q == [0, 16, 40]
    assert packed_seq_params.thd_cp_host_cu_seqlens_kv is (
        packed_seq_params.thd_cp_host_cu_seqlens_q
    )


def test_prebuild_thd_cp_partition_routes_materializes_aliased_qkv_once(monkeypatch):
    cu_q = torch.tensor([0, 16, 40, 40], dtype=torch.int32)
    expected_route = build_thd_cp_partition_route(cu_q, cp_size=2, cp_rank=1)
    materialized = []
    original_materialize = context_parallel_layout_routes._materialize_thd_cu_seqlens_to_list

    def track_materialize(cu_seqlens):
        materialized.append(cu_seqlens)
        return original_materialize(cu_seqlens)

    monkeypatch.setattr(
        context_parallel_layout_routes, "_materialize_thd_cu_seqlens_to_list", track_materialize
    )
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=cu_q,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv=cu_q,
        cu_seqlens_kv_padded=None,
        cp_partition_route=None,
    )

    prebuild_thd_cp_partition_routes(packed_seq_params, _FakeGroup(size=2, rank=1))

    assert len(materialized) == 1
    assert materialized[0] is cu_q
    assert packed_seq_params.thd_cp_host_cu_seqlens_q == [0, 16, 40]
    assert packed_seq_params.thd_cp_host_cu_seqlens_kv is (
        packed_seq_params.thd_cp_host_cu_seqlens_q
    )
    _assert_thd_routes_equal(packed_seq_params.cp_partition_route, expected_route)


def test_prebuild_thd_cp_partition_routes_materializes_distinct_qkv_jointly_once(monkeypatch):
    cu_q = torch.tensor([0, 16, 40, 40], dtype=torch.int32)
    cu_kv = torch.tensor([0, 8, 40, 40], dtype=torch.int64)
    expected_route = build_thd_cp_partition_route(cu_q, cp_size=2, cp_rank=0)
    materialized = []
    original_materialize = context_parallel_layout_routes._materialize_thd_cu_seqlens_to_list

    def track_materialize(cu_seqlens):
        materialized.append(cu_seqlens)
        return original_materialize(cu_seqlens)

    monkeypatch.setattr(
        context_parallel_layout_routes, "_materialize_thd_cu_seqlens_to_list", track_materialize
    )
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=cu_q,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv=cu_kv,
        cu_seqlens_kv_padded=None,
        cp_partition_route=None,
    )

    prebuild_thd_cp_partition_routes(packed_seq_params, _FakeGroup(size=2, rank=0))

    assert len(materialized) == 1
    assert torch.equal(materialized[0], torch.cat((cu_q, cu_kv)))
    assert packed_seq_params.thd_cp_host_cu_seqlens_q == [0, 16, 40]
    assert packed_seq_params.thd_cp_host_cu_seqlens_kv == [0, 8, 40]
    assert packed_seq_params.thd_cp_host_cu_seqlens_kv is not (
        packed_seq_params.thd_cp_host_cu_seqlens_q
    )
    _assert_thd_routes_equal(packed_seq_params.cp_partition_route, expected_route)


def test_prebuild_thd_cp_partition_routes_preserves_mixed_device_fallback(monkeypatch):
    cu_q = torch.tensor([0, 16, 40])
    cu_kv = torch.empty(3, device="meta", dtype=torch.int32)
    materialized = []

    def fake_compact(cu_seqlens):
        materialized.append(cu_seqlens)
        return [0, 16, 40] if cu_seqlens is cu_q else [0, 8, 40]

    monkeypatch.setattr(
        context_parallel_layout_routes, "_compact_thd_cu_seqlens_to_list", fake_compact
    )
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=cu_q,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv=cu_kv,
        cu_seqlens_kv_padded=None,
        cp_partition_route=None,
    )

    prebuild_thd_cp_partition_routes(packed_seq_params, _FakeGroup(size=2, rank=0))

    assert len(materialized) == 2
    assert materialized[0] is cu_q
    assert materialized[1] is cu_kv
    assert packed_seq_params.thd_cp_host_cu_seqlens_q == [0, 16, 40]
    assert packed_seq_params.thd_cp_host_cu_seqlens_kv == [0, 8, 40]


def test_prebuild_thd_cp_partition_routes_raises_route_errors():
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 10, 18]),
        cu_seqlens_q_padded=None,
        cp_partition_route=None,
    )
    cp_group = _FakeGroup(size=2, rank=0)

    with pytest.raises(ValueError, match="divisible"):
        prebuild_thd_cp_partition_routes(packed_seq_params, cp_group)


def test_cp_partition_mode_converter_recurses_over_tensor_containers(monkeypatch):
    calls = []

    def fake_convert(*, x, cp_group, **kwargs):
        calls.append((x, cp_group, kwargs))
        return x + 10

    monkeypatch.setattr(
        context_parallel_layout_conversion, "convert_cp_partition_mode", fake_convert
    )
    cp_group = SimpleNamespace(size=lambda: 2)
    tp_cp_group = object()
    config = SimpleNamespace(cuda_graph_impl=None)
    cu_seqlens = torch.tensor([0, 8])
    untouched = object()
    value = (torch.tensor([1]), [None, untouched, torch.tensor([2])])
    route = object()
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_q_padded=None,
        cp_partition_mode="zigzag",
        cp_partition_route=route,
    )

    converter = CpPartitionModeConverter(
        cp_group=cp_group,
        packed_seq_params=packed_seq_params,
        source_partition_mode="zigzag",
        target_partition_mode="contiguous",
        config=config,
        tp_cp_group=tp_cp_group,
    )
    converted = converter.convert(value=value, seq_dim=lambda tensor: tensor.dim() - 1)

    assert torch.equal(converted[0], torch.tensor([11]))
    assert converted[1][0] is None
    assert converted[1][1] is untouched
    assert torch.equal(converted[1][2], torch.tensor([12]))
    assert [call[1] for call in calls] == [cp_group, cp_group]
    assert [call[2]["seq_dim"] for call in calls] == [0, 0]
    assert all(call[2]["cu_seqlens"] is cu_seqlens for call in calls)
    assert all(call[2]["tp_cp_group"] is tp_cp_group for call in calls)
    assert packed_seq_params.cp_partition_mode == "contiguous"
    assert packed_seq_params.cp_partition_route is route


def test_cp_partition_mode_converter_rejects_thd_full_iteration_cuda_graph_conversion():
    cp_group = SimpleNamespace(size=lambda: 2)
    packed_seq_params = SimpleNamespace(qkv_format="thd")
    config = SimpleNamespace(cuda_graph_impl="full_iteration")

    CpPartitionModeConverter(
        cp_group=cp_group,
        packed_seq_params=packed_seq_params,
        source_partition_mode="zigzag",
        target_partition_mode="zigzag",
        config=config,
    )

    with pytest.raises(ValueError, match="Full-iteration CUDA graph"):
        CpPartitionModeConverter(
            cp_group=cp_group,
            packed_seq_params=packed_seq_params,
            source_partition_mode="zigzag",
            target_partition_mode="contiguous",
            config=config,
        )


def test_module_input_conversion_treats_missing_packed_seq_params_as_sbhd(monkeypatch):
    calls = []

    def fake_convert(*, x, cp_group, **kwargs):
        calls.append((x, cp_group, kwargs))
        return x + 1

    monkeypatch.setattr(
        context_parallel_layout_conversion, "convert_cp_partition_mode", fake_convert
    )
    cp_group = SimpleNamespace(size=lambda: 2)
    tp_cp_group = object()
    hidden_states = torch.ones(8, 1, 4)

    converted, converter = convert_module_input_tensors_cp_partition_mode(
        hidden_states=hidden_states,
        packed_seq_params=None,
        cp_group=cp_group,
        tp_group=None,
        tp_cp_group=tp_cp_group,
        target_partition_mode="contiguous",
        sequence_parallel=False,
        config=SimpleNamespace(cp_partition_mode="zigzag", cuda_graph_impl=None),
    )

    assert converter is not None
    assert torch.equal(converted, hidden_states + 1)
    assert calls[0][2]["source_partition_mode"] == "zigzag"
    assert calls[0][2]["target_partition_mode"] == "contiguous"
    assert calls[0][2]["cu_seqlens"] is None
    assert calls[0][2]["tp_cp_group"] is tp_cp_group
    assert converter.tp_cp_group is tp_cp_group


def test_public_conversion_apis_default_to_no_cp_group():
    hidden_states = torch.ones(8, 1, 4)
    config = SimpleNamespace(cp_partition_mode="zigzag", cuda_graph_impl=None)
    converter = CpPartitionModeConverter(
        packed_seq_params=None,
        source_partition_mode="zigzag",
        target_partition_mode="contiguous",
        config=config,
    )

    assert converter.convert(value=hidden_states) is hidden_states
    converted, back_to_input_converter = convert_module_input_tensors_cp_partition_mode(
        hidden_states=hidden_states,
        packed_seq_params=None,
        target_partition_mode="contiguous",
        sequence_parallel=False,
        config=config,
    )
    assert converted is hidden_states
    assert back_to_input_converter is None
    assert (
        context_parallel_layout_conversion.convert_cp_partition_mode(
            x=hidden_states, source_partition_mode="zigzag", target_partition_mode="contiguous"
        )
        is hidden_states
    )


@pytest.mark.parametrize(
    ("sequence_parallel", "tp_size"),
    [(False, None), (False, 2), (True, None), (True, 1), (True, 2)],
)
def test_sbhd_conversion_uses_one_redistribution_path(monkeypatch, sequence_parallel, tp_size):
    calls = []
    cp_group = _FakeGroup(size=2, rank=0)
    tp_group = _FakeGroup(size=tp_size, rank=0) if tp_size is not None else None
    tp_cp_group = _FakeGroup(size=2 * tp_size, rank=0) if tp_size is not None else None
    x = torch.arange(24).view(2, 6, 2)

    def fake_redistribute(**kwargs):
        calls.append(kwargs)
        return kwargs["input_"] + 1

    monkeypatch.setattr(
        context_parallel_layout_conversion, "_redistribute_sbhd_layout", fake_redistribute
    )

    converted = context_parallel_layout_conversion.convert_cp_partition_mode(
        x=x,
        source_partition_mode="zigzag",
        target_partition_mode="contiguous",
        seq_dim=1,
        sequence_parallel=sequence_parallel,
        cp_group=cp_group,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
    )

    torch.testing.assert_close(converted, x + 1)
    assert len(calls) == 1
    call = calls[0]
    assert torch.equal(call.pop("input_"), x.movedim(1, 0))
    assert call == {
        "cp_group": cp_group,
        "source_layout": "zigzag",
        "target_layout": "contiguous",
        "sequence_parallel": sequence_parallel,
        "tp_group": tp_group,
        "tp_cp_group": tp_cp_group,
    }


def test_sequence_parallel_thd_conversion_warns_about_naive_fallback(monkeypatch):
    from megatron.core.tensor_parallel import mappings

    calls = []
    cp_group = _FakeGroup(size=2, rank=0)
    tp_group = _FakeGroup(size=2, rank=0)
    cu_seqlens = torch.tensor([0, 12])
    x = torch.arange(24).view(2, 6, 2)

    def fake_gather(*, input_, tensor_parallel_output_grad, group):
        calls.append(("gather", tensor_parallel_output_grad, group))
        return input_

    def fake_redistribute(**kwargs):
        calls.append(("thd", kwargs))
        return kwargs["x"]

    def fake_scatter(*, input_, group):
        calls.append(("scatter", group))
        return input_

    monkeypatch.setattr(mappings, "gather_from_sequence_parallel_region", fake_gather)
    monkeypatch.setattr(
        context_parallel_layout_conversion, "_redistribute_thd_layout", fake_redistribute
    )
    monkeypatch.setattr(mappings, "scatter_to_sequence_parallel_region", fake_scatter)

    with pytest.warns(RuntimeWarning, match="naive TP gather"):
        converted = context_parallel_layout_conversion.convert_cp_partition_mode(
            x=x,
            source_partition_mode="zigzag",
            target_partition_mode="contiguous",
            seq_dim=1,
            cu_seqlens=cu_seqlens,
            sequence_parallel=True,
            cp_group=cp_group,
            tp_group=tp_group,
        )

    assert torch.equal(converted, x)
    assert calls[0] == ("gather", False, tp_group)
    call_name, thd_call = calls[1]
    assert call_name == "thd"
    assert torch.equal(thd_call.pop("x"), x.movedim(1, 0))
    assert thd_call.pop("cu_seqlens") is cu_seqlens
    assert thd_call == {
        "cp_group": cp_group,
        "seq_dim": 0,
        "source_partition_mode": "zigzag",
        "target_partition_mode": "contiguous",
        "thd_cp_partition_route": None,
    }
    assert calls[2] == ("scatter", tp_group)


# -----------------------------------------------------------------------------
# Fused TP x CP THD routes for sequence-parallel shards
# -----------------------------------------------------------------------------

_TP_CP_ROUTE_CASES = [
    pytest.param(torch.tensor([0, 16, 40]), 2, 2, (0, 2, 1, 3), id="straddle-cp2-tp2-permuted"),
    pytest.param(torch.tensor([0, 128]), 4, 2, tuple(range(8)), id="one-seq-cp4-tp2"),
    pytest.param(
        torch.tensor([0, 32, 96, 128]), 4, 2, (0, 4, 1, 5, 2, 6, 3, 7), id="many-cp4-tp2-permuted"
    ),
    pytest.param(
        torch.tensor([0, 32, 96, 128, 128, 128]),
        2,
        4,
        (7, 6, 5, 4, 3, 2, 1, 0),
        id="padded-tail-cp2-tp4-reversed",
    ),
    pytest.param(torch.tensor([0, 48, 128]), 2, 4, tuple(range(8)), id="uneven-cp2-tp4"),
]


@pytest.mark.parametrize(
    ("source_layout", "target_layout"), [("zigzag", "contiguous"), ("contiguous", "zigzag")]
)
@pytest.mark.parametrize(
    ("cu_seqlens", "cp_size", "tp_size", "group_rank_by_logical_rank"), _TP_CP_ROUTE_CASES
)
def test_thd_tp_cp_partition_route_reassembles_target_layout(
    source_layout, target_layout, cu_seqlens, cp_size, tp_size, group_rank_by_logical_rank
):
    group_size = cp_size * tp_size
    routes = [None] * group_size
    source_tokens = [None] * group_size
    target_tokens = [None] * group_size
    for logical_rank, group_rank in enumerate(group_rank_by_logical_rank):
        cp_rank, tp_rank = divmod(logical_rank, tp_size)
        source_tokens[group_rank] = _get_test_thd_sp_shard_token_indices(
            cu_seqlens, cp_size, cp_rank, tp_size, tp_rank, source_layout
        )
        target_tokens[group_rank] = _get_test_thd_sp_shard_token_indices(
            cu_seqlens, cp_size, cp_rank, tp_size, tp_rank, target_layout
        )
        route = build_thd_tp_cp_partition_route(
            cu_seqlens, cp_size, cp_rank, tp_size, tp_rank, group_rank_by_logical_rank
        )
        assert isinstance(route, ThdCpRoute)
        assert len(route.zigzag_split_sizes) == group_size
        assert len(route.contiguous_split_sizes) == group_size
        routes[group_rank] = route

    outputs = _simulate_thd_route_all_to_all(routes, source_tokens, source_layout, target_layout)
    for group_rank in range(group_size):
        assert torch.equal(outputs[group_rank], target_tokens[group_rank])


@pytest.mark.parametrize(
    ("source_layout", "target_layout"), [("zigzag", "contiguous"), ("contiguous", "zigzag")]
)
@pytest.mark.parametrize(
    ("cu_seqlens", "cp_size", "tp_size", "group_rank_by_logical_rank"), _TP_CP_ROUTE_CASES
)
def test_thd_tp_cp_partition_route_matches_composed_tp_gather_cp_route(
    source_layout, target_layout, cu_seqlens, cp_size, tp_size, group_rank_by_logical_rank
):
    """The fused route must reproduce TP gather -> CP route -> TP scatter exactly."""
    group_size = cp_size * tp_size
    fused_routes = [None] * group_size
    fused_sources = [None] * group_size
    for logical_rank, group_rank in enumerate(group_rank_by_logical_rank):
        cp_rank, tp_rank = divmod(logical_rank, tp_size)
        fused_sources[group_rank] = _get_test_thd_sp_shard_token_indices(
            cu_seqlens, cp_size, cp_rank, tp_size, tp_rank, source_layout
        )
        fused_routes[group_rank] = build_thd_tp_cp_partition_route(
            cu_seqlens, cp_size, cp_rank, tp_size, tp_rank, group_rank_by_logical_rank
        )
    fused_outputs = _simulate_thd_route_all_to_all(
        fused_routes, fused_sources, source_layout, target_layout
    )

    # Composed reference: gather the TP shards of every CP rank, run the CP-only
    # route over the CP group, then split the result back into TP shards.
    gathered_sources = []
    for cp_rank in range(cp_size):
        shards = [
            fused_sources[group_rank_by_logical_rank[cp_rank * tp_size + tp_rank]]
            for tp_rank in range(tp_size)
        ]
        gathered_sources.append(torch.cat(shards, dim=0))
    cp_routes = [build_thd_cp_partition_route(cu_seqlens, cp_size, rank) for rank in range(cp_size)]
    cp_outputs = _simulate_thd_route_all_to_all(
        cp_routes, gathered_sources, source_layout, target_layout
    )
    for cp_rank in range(cp_size):
        for tp_rank, shard in enumerate(cp_outputs[cp_rank].chunk(tp_size)):
            group_rank = group_rank_by_logical_rank[cp_rank * tp_size + tp_rank]
            assert torch.equal(fused_outputs[group_rank], shard)


def test_thd_tp_cp_partition_route_requires_tp_divisible_cp_shards():
    with pytest.raises(ValueError, match="divisible by tp_size"):
        build_thd_tp_cp_partition_route(
            torch.tensor([0, 20]),
            cp_size=2,
            cp_rank=0,
            tp_size=4,
            tp_rank=0,
            group_rank_by_logical_rank=tuple(range(8)),
        )


def test_thd_tp_cp_partition_route_rejects_invalid_group_rank_mapping():
    with pytest.raises(ValueError, match="permutation"):
        build_thd_tp_cp_partition_route(
            torch.tensor([0, 32]),
            cp_size=2,
            cp_rank=0,
            tp_size=2,
            tp_rank=0,
            group_rank_by_logical_rank=(0, 0, 1, 2),
        )


def test_tp_cp_group_rank_mapping_follows_cartesian_product():
    # TP is the fastest-varying dimension of the TP x CP group (Megatron's default order).
    assert build_tp_cp_group_rank_by_logical_rank((0, 2), (0, 1), (0, 1, 2, 3), 0) == (0, 1, 2, 3)
    assert build_tp_cp_group_rank_by_logical_rank((1, 3), (2, 3), (0, 1, 2, 3), 3) == (0, 1, 2, 3)
    # A TP x CP group built in a different order still maps every coordinate to its rank.
    assert build_tp_cp_group_rank_by_logical_rank((0, 2), (0, 1), (0, 2, 1, 3), 0) == (0, 2, 1, 3)
    with pytest.raises(RuntimeError, match="Cartesian product"):
        # A dynamic CP sub-group spanning another data-parallel replica.
        build_tp_cp_group_rank_by_logical_rank((0, 4), (0, 1), (0, 1, 2, 3), 0)
    with pytest.raises(RuntimeError, match="Cartesian product"):
        # Groups whose coordinate sums collide cannot be a product inside the group.
        build_tp_cp_group_rank_by_logical_rank((0, 1), (0, 1), (0, 1, 2, 3), 0)


def test_sequence_parallel_thd_conversion_dispatches_to_fused_tp_cp_route(monkeypatch):
    from megatron.core.tensor_parallel import mappings

    calls = []
    cp_group = _FakeGroup(size=2, rank=1)
    tp_group = _FakeGroup(size=2, rank=0)
    tp_cp_group = _FakeGroup(size=4, rank=2)
    cu_seqlens = torch.tensor([0, 16, 40])
    x = torch.arange(20).view(2, 10, 1)

    def fail(*args, **kwargs):
        raise AssertionError("the composed TP gather/scatter fallback must not run")

    def fake_fused(**kwargs):
        calls.append(kwargs)
        return kwargs["x"] + 1

    monkeypatch.setattr(mappings, "gather_from_sequence_parallel_region", fail)
    monkeypatch.setattr(mappings, "scatter_to_sequence_parallel_region", fail)
    monkeypatch.setattr(context_parallel_layout_conversion, "_redistribute_thd_layout", fail)
    monkeypatch.setattr(
        context_parallel_layout_conversion,
        "resolve_tp_cp_group_rank_by_logical_rank",
        lambda cp, tp, tp_cp: (0, 2, 1, 3),
    )
    monkeypatch.setattr(
        context_parallel_layout_conversion, "_redistribute_thd_tp_cp_layout", fake_fused
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        converted = context_parallel_layout_conversion.convert_cp_partition_mode(
            x=x,
            source_partition_mode="contiguous",
            target_partition_mode="zigzag",
            seq_dim=1,
            cu_seqlens=cu_seqlens,
            sequence_parallel=True,
            cp_group=cp_group,
            tp_group=tp_group,
            tp_cp_group=tp_cp_group,
        )

    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert torch.equal(converted, x + 1)
    assert len(calls) == 1
    call = calls[0]
    assert torch.equal(call.pop("x"), x)
    assert call.pop("cu_seqlens") is cu_seqlens
    assert call == {
        "cp_group": cp_group,
        "tp_group": tp_group,
        "tp_cp_group": tp_cp_group,
        "group_rank_by_logical_rank": (0, 2, 1, 3),
        "seq_dim": 1,
        "source_partition_mode": "contiguous",
        "target_partition_mode": "zigzag",
        "thd_tp_cp_partition_route": None,
    }


def test_sequence_parallel_thd_conversion_falls_back_when_groups_cannot_be_fused(monkeypatch):
    """A TP x CP group that does not contain the CP group (dynamic CP) keeps the old path."""
    from megatron.core.tensor_parallel import mappings

    calls = []
    cp_group = _FakeGroup(size=2, rank=0)
    tp_group = _FakeGroup(size=2, rank=0)
    # Static TP x CP group of a larger CP size than this microbatch's dynamic CP sub-group.
    tp_cp_group = _FakeGroup(size=8, rank=0)
    cu_seqlens = torch.tensor([0, 12])
    x = torch.arange(24).view(6, 2, 2)

    monkeypatch.setattr(
        mappings, "gather_from_sequence_parallel_region", lambda *, input_, **kw: input_
    )
    monkeypatch.setattr(
        mappings, "scatter_to_sequence_parallel_region", lambda *, input_, group: input_
    )
    monkeypatch.setattr(
        context_parallel_layout_conversion,
        "_redistribute_thd_layout",
        lambda **kwargs: calls.append(kwargs) or kwargs["x"],
    )
    monkeypatch.setattr(
        context_parallel_layout_conversion,
        "_redistribute_thd_tp_cp_layout",
        lambda **kwargs: pytest.fail("fused path must not run for unfusable groups"),
    )

    with pytest.warns(RuntimeWarning, match="naive TP gather"):
        converted = context_parallel_layout_conversion.convert_cp_partition_mode(
            x=x,
            source_partition_mode="zigzag",
            target_partition_mode="contiguous",
            seq_dim=0,
            cu_seqlens=cu_seqlens,
            sequence_parallel=True,
            cp_group=cp_group,
            tp_group=tp_group,
            tp_cp_group=tp_cp_group,
        )

    assert torch.equal(converted, x)
    assert len(calls) == 1 and calls[0]["cp_group"] is cp_group


def test_prebuild_thd_cp_partition_routes_also_builds_fused_tp_cp_route(monkeypatch):
    cp_group = _FakeGroup(size=2, rank=1)
    tp_group = _FakeGroup(size=2, rank=0)
    tp_cp_group = _FakeGroup(size=4, rank=2)
    mapping = (0, 2, 1, 3)
    monkeypatch.setattr(
        context_parallel_layout_routes,
        "resolve_tp_cp_group_rank_by_logical_rank",
        lambda cp, tp, tp_cp: mapping if tp_cp is tp_cp_group else None,
    )
    cu_q = torch.tensor([0, 16, 40], dtype=torch.int32)
    packed_seq_params = SimpleNamespace(
        qkv_format="thd", cu_seqlens_q=cu_q, cu_seqlens_q_padded=None, cp_partition_route=None
    )

    prebuild_thd_cp_partition_routes(
        packed_seq_params, cp_group, tp_group=tp_group, tp_cp_group=tp_cp_group
    )

    route = packed_seq_params.tp_cp_partition_route
    assert isinstance(route, ThdCpRoute)
    expected = build_thd_tp_cp_partition_route(cu_q, 2, 1, 2, 0, mapping)
    _assert_thd_routes_equal(route, expected)
    assert packed_seq_params.thd_cp_host_cu_seqlens_q == [0, 16, 40]

    # The lookup returns the prebuilt route for both directions without rebuilding.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for source, target in (("zigzag", "contiguous"), ("contiguous", "zigzag")):
            assert (
                get_thd_tp_cp_partition_route(
                    packed_seq_params,
                    source,
                    target,
                    cp_group=cp_group,
                    tp_group=tp_group,
                    tp_cp_group=tp_cp_group,
                )
                is route
            )
        assert (
            get_thd_tp_cp_partition_route(
                packed_seq_params,
                "zigzag",
                "zigzag",
                cp_group=cp_group,
                tp_group=tp_group,
                tp_cp_group=tp_cp_group,
            )
            is None
        )
        # Unfusable groups: no route, the caller falls back to the composed path.
        assert (
            get_thd_tp_cp_partition_route(
                packed_seq_params,
                "zigzag",
                "contiguous",
                cp_group=cp_group,
                tp_group=tp_group,
                tp_cp_group=_FakeGroup(size=4, rank=2),
            )
            is None
        )


def test_prebuild_thd_cp_partition_routes_skips_fused_route_without_tp(monkeypatch):
    monkeypatch.setattr(
        context_parallel_layout_routes,
        "resolve_tp_cp_group_rank_by_logical_rank",
        lambda *args: pytest.fail("mapping must not be resolved without TP"),
    )
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 16, 40]),
        cu_seqlens_q_padded=None,
        cp_partition_route=None,
    )
    prebuild_thd_cp_partition_routes(
        packed_seq_params,
        _FakeGroup(size=2, rank=0),
        tp_group=_FakeGroup(size=1, rank=0),
        tp_cp_group=_FakeGroup(size=2, rank=0),
    )
    assert packed_seq_params.cp_partition_route is not None
    assert not hasattr(packed_seq_params, "tp_cp_partition_route")


def test_get_thd_tp_cp_partition_route_builds_from_host_boundaries_or_warns(monkeypatch):
    cp_group = _FakeGroup(size=2, rank=0)
    tp_group = _FakeGroup(size=2, rank=1)
    tp_cp_group = _FakeGroup(size=4, rank=1)
    mapping = (0, 1, 2, 3)
    monkeypatch.setattr(
        context_parallel_layout_routes,
        "resolve_tp_cp_group_rank_by_logical_rank",
        lambda cp, tp, tp_cp: mapping,
    )
    cu_q = torch.tensor([0, 16, 40], dtype=torch.int32)
    expected = build_thd_tp_cp_partition_route(cu_q, 2, 0, 2, 1, mapping)

    # Host boundaries left by the CP route prebuild are reused without a warning.
    with_host = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=cu_q,
        cu_seqlens_q_padded=None,
        cp_partition_route=None,
        thd_cp_host_cu_seqlens_q=[0, 16, 40],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        route = get_thd_tp_cp_partition_route(
            with_host,
            "contiguous",
            "zigzag",
            cp_group=cp_group,
            tp_group=tp_group,
            tp_cp_group=tp_cp_group,
        )
    _assert_thd_routes_equal(route, expected)
    assert with_host.tp_cp_partition_route is route
    # The cached route is handed back as is for both directions.
    assert (
        get_thd_tp_cp_partition_route(
            with_host,
            "zigzag",
            "contiguous",
            cp_group=cp_group,
            tp_group=tp_group,
            tp_cp_group=tp_cp_group,
        )
        is route
    )

    # Without a prebuild the lookup still works but flags the device-to-host copy.
    without_host = SimpleNamespace(
        qkv_format="thd", cu_seqlens_q=cu_q, cu_seqlens_q_padded=None, cp_partition_route=None
    )
    with pytest.warns(FutureWarning, match="TP x CP layout route"):
        route = get_thd_tp_cp_partition_route(
            without_host,
            "contiguous",
            "zigzag",
            cp_group=cp_group,
            tp_group=tp_group,
            tp_cp_group=tp_cp_group,
        )
    _assert_thd_routes_equal(route, expected)


def test_cp_partition_mode_converter_prefers_fused_route_for_sequence_parallel_thd(monkeypatch):
    calls = []

    def fake_convert(*, x, **kwargs):
        calls.append(kwargs)
        return x

    cp_group = _FakeGroup(size=2, rank=0)
    tp_group = _FakeGroup(size=2, rank=0)
    tp_cp_group = _FakeGroup(size=4, rank=0)
    cp_route = object()
    tp_cp_route = object()
    monkeypatch.setattr(
        context_parallel_layout_conversion, "convert_cp_partition_mode", fake_convert
    )
    monkeypatch.setattr(
        context_parallel_layout_conversion,
        "get_thd_tp_cp_partition_route",
        lambda *args, **kwargs: tp_cp_route,
    )
    monkeypatch.setattr(
        context_parallel_layout_conversion,
        "get_thd_cp_partition_route",
        lambda *args, **kwargs: pytest.fail("CP-only route is not needed on the fused path"),
    )
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 8]),
        cu_seqlens_q_padded=None,
        cp_partition_mode="contiguous",
        cp_partition_route=cp_route,
        tp_cp_partition_route=tp_cp_route,
    )
    converter = CpPartitionModeConverter(
        cp_group=cp_group,
        packed_seq_params=packed_seq_params,
        source_partition_mode="contiguous",
        target_partition_mode="zigzag",
        config=SimpleNamespace(cuda_graph_impl=None),
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
    )
    converter.convert(torch.zeros(4, 1, 2), seq_dim=0, sequence_parallel=True)
    assert calls[0]["thd_tp_cp_partition_route"] is tp_cp_route
    assert calls[0]["thd_cp_partition_route"] is None

    # Without sequence parallelism only the CP route is looked up.
    monkeypatch.setattr(
        context_parallel_layout_conversion,
        "get_thd_tp_cp_partition_route",
        lambda *args, **kwargs: pytest.fail("fused route is not needed without SP"),
    )
    monkeypatch.setattr(
        context_parallel_layout_conversion,
        "get_thd_cp_partition_route",
        lambda *args, **kwargs: cp_route,
    )
    converter.convert(torch.zeros(4, 1, 2), seq_dim=0, sequence_parallel=False)
    assert calls[1]["thd_cp_partition_route"] is cp_route
    assert calls[1]["thd_tp_cp_partition_route"] is None


@pytest.mark.internal
@pytest.mark.parametrize(
    ("source_layout", "target_layout"), [("zigzag", "contiguous"), ("contiguous", "zigzag")]
)
@pytest.mark.parametrize(("tp_size", "cp_size"), [(2, 2), (2, 4), (4, 2)])
@pytest.mark.parametrize(
    "cu_seqlens_list",
    [[0, 128], [0, 16, 48, 64, 128], [0, 48, 128], [0, 32, 96, 128, 128]],
    ids=["one-seq", "many-seqs", "straddling", "padded-tail"],
)
def test_sequence_parallel_thd_conversion_matches_unfused_reference(
    source_layout, target_layout, tp_size, cp_size, cu_seqlens_list
):
    """Fused TP x CP conversion == TP gather -> CP all-to-all -> TP scatter, bit for bit."""
    required_world_size = tp_size * cp_size
    if (
        not torch.cuda.is_available()
        or Utils.world_size < required_world_size
        or Utils.world_size % required_world_size != 0
    ):
        pytest.skip(
            f"Sequence-parallel THD conversion needs a multiple of {required_world_size} "
            "CUDA ranks."
        )

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
    )
    try:
        cp_group = parallel_state.get_context_parallel_group()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_cp_group = parallel_state.get_tensor_and_context_parallel_group()
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        cu_seqlens = torch.tensor(cu_seqlens_list, device=device, dtype=torch.int32)
        full_tensor = _make_sequence_tensor(total_seq_len=128, seq_dim=0, device=device)
        full_upstream_grad = full_tensor.mul(0.125).add(1.0)
        source_indices = _get_test_thd_sp_shard_token_indices(
            cu_seqlens.cpu(), cp_size, cp_group.rank(), tp_size, tp_group.rank(), source_layout
        ).to(device)
        target_indices = _get_test_thd_sp_shard_token_indices(
            cu_seqlens.cpu(), cp_size, cp_group.rank(), tp_size, tp_group.rank(), target_layout
        ).to(device)
        expected_target = full_tensor.index_select(0, target_indices)
        target_upstream_grad = full_upstream_grad.index_select(0, target_indices)
        expected_source_grad = full_upstream_grad.index_select(0, source_indices)

        fused_input = full_tensor.index_select(0, source_indices).requires_grad_(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fused_output = context_parallel_layout_conversion.convert_cp_partition_mode(
                x=fused_input,
                source_partition_mode=source_layout,
                target_partition_mode=target_layout,
                seq_dim=0,
                cu_seqlens=cu_seqlens,
                sequence_parallel=True,
                cp_group=cp_group,
                tp_group=tp_group,
                tp_cp_group=tp_cp_group,
            )
        assert not [
            w
            for w in caught
            if issubclass(w.category, RuntimeWarning) and "naive" in str(w.message)
        ], "static TP x CP groups must use the fused route"
        fused_output.mul(target_upstream_grad).sum().backward()

        reference_input = full_tensor.index_select(0, source_indices).requires_grad_(True)
        reference_output = _convert_thd_sp_shard_via_tp_gather(
            reference_input,
            source_layout=source_layout,
            target_layout=target_layout,
            cu_seqlens=cu_seqlens,
            cp_group=cp_group,
            tp_group=tp_group,
        )
        reference_output.mul(target_upstream_grad).sum().backward()

        torch.testing.assert_close(fused_output, expected_target, atol=0.0, rtol=0.0)
        torch.testing.assert_close(reference_output, expected_target, atol=0.0, rtol=0.0)
        torch.testing.assert_close(fused_input.grad, expected_source_grad, atol=0.0, rtol=0.0)
        torch.testing.assert_close(reference_input.grad, expected_source_grad, atol=0.0, rtol=0.0)

        # The converter path uses the routes prebuilt on a real PackedSeqParams.
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cp_partition_mode=source_layout,
        )
        prebuild_thd_cp_partition_routes(
            packed_seq_params, cp_group, tp_group=tp_group, tp_cp_group=tp_cp_group
        )
        assert isinstance(packed_seq_params.tp_cp_partition_route, ThdCpRoute)
        converter = CpPartitionModeConverter(
            packed_seq_params=packed_seq_params,
            source_partition_mode=source_layout,
            target_partition_mode=target_layout,
            config=SimpleNamespace(cuda_graph_impl=None),
            cp_group=cp_group,
            tp_group=tp_group,
            tp_cp_group=tp_cp_group,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            converter_output = converter.convert(
                full_tensor.index_select(0, source_indices), seq_dim=0, sequence_parallel=True
            )
        torch.testing.assert_close(converter_output, expected_target, atol=0.0, rtol=0.0)
        assert packed_seq_params.cp_partition_mode == target_layout
    finally:
        Utils.destroy_model_parallel()
