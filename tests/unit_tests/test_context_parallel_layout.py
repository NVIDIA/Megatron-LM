# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.context_parallel_layout.conversion as context_parallel_layout_conversion
from megatron.core import parallel_state
from megatron.core.context_parallel_layout import (
    CpPartitionModeConverter,
    ThdCpRoute,
    convert_module_input_tensors_cp_partition_mode,
    prebuild_thd_cp_partition_routes,
)
from megatron.core.context_parallel_layout.routes import (
    build_thd_cp_partition_route,
    get_thd_cp_partition_route,
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
            source_shard,
            cp_group,
            source_partition_mode=source_layout,
            target_partition_mode=target_layout,
            seq_dim=seq_dim,
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
        pytest.param("zigzag", "contiguous", 0, True, id="sequence-parallel"),
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
            {"sequence_parallel": True, "tp_group": tp_group} if sequence_parallel else {}
        )
        converted = context_parallel_layout_conversion.convert_cp_partition_mode(
            source_shard,
            cp_group,
            source_partition_mode=source_layout,
            target_partition_mode=target_layout,
            seq_dim=seq_dim,
            **convert_kwargs,
        )
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

    def fake_convert(tensor, cp_group, **kwargs):
        calls.append((tensor, cp_group, kwargs))
        return tensor + 10

    monkeypatch.setattr(
        context_parallel_layout_conversion, "convert_cp_partition_mode", fake_convert
    )
    cp_group = SimpleNamespace(size=lambda: 2)
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
    )
    converted = converter.convert(value, seq_dim=lambda tensor: tensor.dim() - 1)

    assert torch.equal(converted[0], torch.tensor([11]))
    assert converted[1][0] is None
    assert converted[1][1] is untouched
    assert torch.equal(converted[1][2], torch.tensor([12]))
    assert [call[1] for call in calls] == [cp_group, cp_group]
    assert [call[2]["seq_dim"] for call in calls] == [0, 0]
    assert all(call[2]["cu_seqlens"] is cu_seqlens for call in calls)
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

    def fake_convert(tensor, cp_group, **kwargs):
        calls.append((tensor, cp_group, kwargs))
        return tensor + 1

    monkeypatch.setattr(
        context_parallel_layout_conversion, "convert_cp_partition_mode", fake_convert
    )
    cp_group = SimpleNamespace(size=lambda: 2)
    hidden_states = torch.ones(8, 1, 4)

    converted, converter = convert_module_input_tensors_cp_partition_mode(
        hidden_states=hidden_states,
        packed_seq_params=None,
        cp_group=cp_group,
        tp_group=None,
        target_partition_mode="contiguous",
        sequence_parallel=False,
        config=SimpleNamespace(cp_partition_mode="zigzag", cuda_graph_impl=None),
    )

    assert converter is not None
    assert torch.equal(converted, hidden_states + 1)
    assert calls[0][2]["source_partition_mode"] == "zigzag"
    assert calls[0][2]["target_partition_mode"] == "contiguous"
    assert calls[0][2]["cu_seqlens"] is None
