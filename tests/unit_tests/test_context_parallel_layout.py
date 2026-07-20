# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.context_parallel_layout as context_parallel_layout
from megatron.core.context_parallel_layout import (
    build_thd_cp_partition_route,
    build_cp_partition_mode_plan,
    convert_cp_partition_mode_nested,
    get_context_parallel_layout_chunk_indices,
    get_or_build_thd_cp_partition_route,
    get_required_cp_partition_mode_for_layer,
    get_thd_context_parallel_rank_indices,
    prebuild_thd_cp_partition_route_cache,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_stage_input_cp_partition_mode,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    get_hybrid_stage_input_cp_partition_mode,
)


class _PipelineLayout:

    def __init__(self, offset):
        self.offset = offset

    def get_layer_offset(self, **_kwargs):
        return self.offset


class IdentityOp:
    pass


class GatedDeltaNet:
    pass


class _FakeGroup:

    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _token_ranges(*spans):
    return [token for start, end in spans for token in range(start, end)]


def test_context_parallel_layout_chunk_indices():
    assert get_context_parallel_layout_chunk_indices(4, 2, "zigzag").tolist() == [2, 5]
    assert get_context_parallel_layout_chunk_indices(4, 2, "contiguous").tolist() == [4, 5]


def test_thd_context_parallel_rank_indices_match_per_sequence_chunk_order():
    cu_seqlens = torch.tensor([0, 16, 40])

    assert get_thd_context_parallel_rank_indices(
        cu_seqlens, 2, 0, "zigzag"
    ).tolist() == _token_ranges((0, 4), (12, 16), (16, 22), (34, 40))
    assert get_thd_context_parallel_rank_indices(
        cu_seqlens, 2, 1, "zigzag"
    ).tolist() == _token_ranges((4, 12), (22, 34))
    assert get_thd_context_parallel_rank_indices(cu_seqlens, 2, 0, "contiguous").tolist() == list(
        range(0, 20)
    )
    assert get_thd_context_parallel_rank_indices(cu_seqlens, 2, 1, "contiguous").tolist() == list(
        range(20, 40)
    )


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
def test_thd_context_parallel_rank_indices_cover_all_tokens_once(layout):
    cu_seqlens = torch.tensor([0, 32, 96, 128])
    cp_size = 4

    rank_indices = [
        get_thd_context_parallel_rank_indices(cu_seqlens, cp_size, rank, layout)
        for rank in range(cp_size)
    ]

    assert [indices.numel() for indices in rank_indices] == [32, 32, 32, 32]
    assert torch.cat(rank_indices).sort().values.tolist() == list(range(128))


@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
def test_thd_context_parallel_rank_indices_ignore_duplicate_boundaries(layout):
    compact_cu_seqlens = torch.tensor([0, 16, 40])
    padded_cu_seqlens = torch.tensor([0, 16, 40, 40, 40])

    for rank in range(2):
        assert torch.equal(
            get_thd_context_parallel_rank_indices(padded_cu_seqlens, 2, rank, layout),
            get_thd_context_parallel_rank_indices(compact_cu_seqlens, 2, rank, layout),
        )


def test_thd_context_parallel_rank_indices_reject_uneven_chunks():
    with pytest.raises(ValueError, match="divisible"):
        get_thd_context_parallel_rank_indices(torch.tensor([0, 10]), 2, 0, "zigzag")


def test_thd_contiguous_rank_indices_allow_uneven_sequence_lengths():
    cu_seqlens = torch.tensor([0, 10, 18])

    assert get_thd_context_parallel_rank_indices(cu_seqlens, 2, 0, "contiguous").tolist() == list(
        range(0, 9)
    )
    assert get_thd_context_parallel_rank_indices(cu_seqlens, 2, 1, "contiguous").tolist() == list(
        range(9, 18)
    )


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
        get_thd_context_parallel_rank_indices(cu_seqlens, cp_size, rank, source_layout)
        for rank in range(cp_size)
    ]
    target_indices = [
        get_thd_context_parallel_rank_indices(cu_seqlens, cp_size, rank, target_layout)
        for rank in range(cp_size)
    ]
    routes = [
        build_thd_cp_partition_route(cu_seqlens, cp_size, rank, source_layout, target_layout)
        for rank in range(cp_size)
    ]
    for route in routes:
        assert route.send_rows_are_identity == torch.equal(
            route.send_rows, torch.arange(route.send_rows.numel(), dtype=route.send_rows.dtype)
        )
        assert route.recv_rows_are_identity == torch.equal(
            route.recv_rows, torch.arange(route.recv_rows.numel(), dtype=route.recv_rows.dtype)
        )
    send_buffers = [
        source_indices[rank].index_select(0, routes[rank].send_rows) for rank in range(cp_size)
    ]

    for dst_rank in range(cp_size):
        recv_chunks = []
        for src_rank in range(cp_size):
            src_route = routes[src_rank]
            send_offset = sum(src_route.input_split_sizes[:dst_rank])
            send_len = src_route.input_split_sizes[dst_rank]
            recv_chunks.append(send_buffers[src_rank].narrow(0, send_offset, send_len))
        recv_buf = torch.cat(recv_chunks, dim=0)
        out = torch.empty(routes[dst_rank].local_target_length, dtype=recv_buf.dtype)
        out.index_copy_(0, routes[dst_rank].recv_rows, recv_buf)
        assert torch.equal(out, target_indices[dst_rank])


def test_thd_cp_partition_route_cache_reuses_same_microbatch_route():
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 16, 40]),
        cu_seqlens_q_padded=None,
        cp_partition_route_cache=None,
    )
    cp_group = _FakeGroup(size=2, rank=0)

    route = get_or_build_thd_cp_partition_route(
        packed_seq_params, cp_group, "zigzag", "contiguous"
    )
    same_route = get_or_build_thd_cp_partition_route(
        packed_seq_params, cp_group, "zigzag", "contiguous"
    )
    reverse_route = get_or_build_thd_cp_partition_route(
        packed_seq_params, cp_group, "contiguous", "zigzag"
    )

    assert same_route is route
    assert reverse_route is not route
    assert len(packed_seq_params.cp_partition_route_cache) == 2


def test_prebuild_thd_cp_partition_route_cache_populates_lazy_lookup():
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 16, 40]),
        cu_seqlens_q_padded=None,
        cp_partition_route_cache=None,
    )
    cp_group = _FakeGroup(size=2, rank=0)

    prebuild_thd_cp_partition_route_cache(packed_seq_params, cp_group)

    assert len(packed_seq_params.cp_partition_route_cache) == 2
    route = get_or_build_thd_cp_partition_route(
        packed_seq_params, cp_group, "zigzag", "contiguous"
    )
    assert route is next(
        route
        for route in packed_seq_params.cp_partition_route_cache.values()
        if route.source_partition_mode == "zigzag"
        and route.target_partition_mode == "contiguous"
    )


def test_prebuild_thd_cp_partition_route_cache_is_best_effort():
    packed_seq_params = SimpleNamespace(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 10, 18]),
        cu_seqlens_q_padded=None,
        cp_partition_route_cache=None,
    )
    cp_group = _FakeGroup(size=2, rank=0)

    prebuild_thd_cp_partition_route_cache(packed_seq_params, cp_group)

    assert packed_seq_params.cp_partition_route_cache == {}


def test_thd_context_parallel_rank_indices_reject_decreasing_boundaries():
    with pytest.raises(ValueError, match="nondecreasing"):
        get_thd_context_parallel_rank_indices(torch.tensor([0, 16, 8]), 2, 0, "zigzag")


def test_thd_context_parallel_rank_indices_reject_unknown_layout():
    with pytest.raises(ValueError, match="Unsupported"):
        get_thd_context_parallel_rank_indices(torch.tensor([0, 16]), 2, 0, "interleaved")


def test_convert_cp_partition_mode_nested_recurses_over_tensor_containers(monkeypatch):
    calls = []

    def fake_convert(tensor, cp_group, **kwargs):
        calls.append((tensor, cp_group, kwargs))
        return tensor + 10

    monkeypatch.setattr(context_parallel_layout, "convert_cp_partition_mode", fake_convert)
    cp_group = object()
    cu_seqlens = torch.tensor([0, 8])
    untouched = object()
    value = (torch.tensor([1]), [None, untouched, torch.tensor([2])])

    converted = convert_cp_partition_mode_nested(
        value,
        cp_group,
        source_partition_mode="zigzag",
        target_partition_mode="contiguous",
        seq_dim=lambda tensor: tensor.dim() - 1,
        cu_seqlens=cu_seqlens,
    )

    assert torch.equal(converted[0], torch.tensor([11]))
    assert converted[1][0] is None
    assert converted[1][1] is untouched
    assert torch.equal(converted[1][2], torch.tensor([12]))
    assert [call[1] for call in calls] == [cp_group, cp_group]
    assert [call[2]["seq_dim"] for call in calls] == [0, 0]
    assert all(call[2]["cu_seqlens"] is cu_seqlens for call in calls)


def test_required_partition_mode_rejects_unknown_layer_type():
    with pytest.raises(ValueError, match="Cannot determine CP partition mode"):
        get_required_cp_partition_mode_for_layer(object(), SimpleNamespace(cp_comm_type=None))


def test_build_cp_partition_mode_plan_requires_stage_entry_layout():
    config = SimpleNamespace(
        context_parallel_size=2, dynamic_context_parallel=False, cp_comm_type=None
    )

    with pytest.raises(ValueError, match="cp_stage_entry_partition_mode"):
        build_cp_partition_mode_plan([], config, None, owner_name="TestBlock")


def test_build_cp_partition_mode_plan_tracks_exit_layout():
    config = SimpleNamespace(
        context_parallel_size=2,
        dynamic_context_parallel=False,
        cp_comm_type=None,
        linear_cp_mode="chunkwise",
    )

    entry, plan, exit_layout = build_cp_partition_mode_plan(
        [IdentityOp(), GatedDeltaNet()], config, "zigzag", owner_name="TestBlock"
    )

    assert entry == "zigzag"
    assert plan == [None, "contiguous"]
    assert exit_layout == "contiguous"


def test_build_cp_partition_mode_plan_skips_layer_inspection_without_cp():
    config = SimpleNamespace(context_parallel_size=1, dynamic_context_parallel=False)

    entry, plan, exit_layout = build_cp_partition_mode_plan(
        [object(), object()], config, None, owner_name="TestBlock"
    )

    assert entry == "zigzag"
    assert plan == [None, None]
    assert exit_layout == "zigzag"


def test_gated_delta_net_chunkwise_layout_plan_follows_linear_attention_pattern():
    config = SimpleNamespace(
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=2,
        linear_cp_mode="chunkwise",
        num_layers=4,
        pipeline_model_parallel_layout=None,
        pipeline_model_parallel_size=1,
    )

    assert (
        get_experimental_attention_variant_stage_input_cp_partition_mode(config)
        == "contiguous"
    )

    config.pipeline_model_parallel_layout = _PipelineLayout(offset=2)
    assert (
        get_experimental_attention_variant_stage_input_cp_partition_mode(config)
        == "zigzag"
    )


def test_gated_delta_net_headwise_layout_plan_uses_contiguous():
    config = SimpleNamespace(
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1, 0],
        linear_cp_mode="headwise",
        num_layers=2,
        pipeline_model_parallel_layout=None,
        pipeline_model_parallel_size=1,
    )

    assert (
        get_experimental_attention_variant_stage_input_cp_partition_mode(config)
        == "contiguous"
    )


def test_hybrid_stage_input_layout_follows_previous_sensitive_layer():
    config = SimpleNamespace(experimental_attention_variant=None, linear_cp_mode="chunkwise")

    assert get_hybrid_stage_input_cp_partition_mode(config, "M-G", 0) == "zigzag"
    assert get_hybrid_stage_input_cp_partition_mode(config, "M-G", 2) == "zigzag"
    assert get_hybrid_stage_input_cp_partition_mode(config, "M-G", 3) == "contiguous"


def test_hybrid_stage_input_layout_uses_future_layer_before_first_sensitive_layer():
    config = SimpleNamespace(experimental_attention_variant=None, linear_cp_mode="chunkwise")

    assert get_hybrid_stage_input_cp_partition_mode(config, "-G", 0) == "contiguous"


def test_hybrid_stage_input_layout_handles_dsv4_symbols():
    config = SimpleNamespace(
        experimental_attention_variant="dsv4_hybrid", linear_cp_mode="chunkwise"
    )

    assert get_hybrid_stage_input_cp_partition_mode(config, "D-E", 0) == "contiguous"
    assert get_hybrid_stage_input_cp_partition_mode(config, "C-E", 0) == "contiguous"
