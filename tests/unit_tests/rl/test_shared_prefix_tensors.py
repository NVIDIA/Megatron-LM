# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from megatron.rl.shared_prefix_packing import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
    SharedPrefixRow,
    build_shared_prefix_layout,
)
from megatron.rl.shared_prefix_tensors import (
    build_shared_prefix_rows,
    get_shared_prefix_context_parallel_indices,
    get_shared_prefix_physical_alignment,
    materialize_shared_prefix_layout,
    materialize_shared_prefix_token_aligned_tensor,
    resolve_shared_prefix_parallel_topology,
    resolve_shared_prefix_physical_padding_multiple,
    shard_shared_prefix_tensor_bin_for_context_parallel,
)
from tests.unit_tests.rl.shared_prefix_oracles import (
    build_star_attention_allow_mask,
    plan_and_materialize,
)


def test_singleton_fallback_roots_preserve_causality_and_materialized_padding():
    inputs = torch.tensor([[10, 11, 20, 21], [10, 11, 30, 31], [40, 41, 42, 0]])
    rows = [
        SharedPrefixRow(0, "shared", (10, 11), 2),
        SharedPrefixRow(1, "shared", (10, 11), 2),
        SharedPrefixRow(2, "fallback", (40, 41), 1),
    ]
    shared = build_shared_prefix_layout(rows[:2], sequence_length_pad_multiple=4)
    fallback = build_shared_prefix_layout(
        rows[2:], sequence_length_pad_multiple=4, allow_singleton=True
    )
    forest = SharedPrefixForestLayout((shared, fallback), mtp_loss_group_root_counts=(1, 1))
    mask = build_star_attention_allow_mask(forest)
    expected = torch.block_diag(
        build_star_attention_allow_mask(shared), torch.ones(4, 4, dtype=torch.bool).tril()
    )
    torch.testing.assert_close(mask, expected, rtol=0, atol=0)
    materialized = materialize_shared_prefix_layout(
        inputs, input_lengths=torch.tensor([4, 4, 3]), layout=forest
    )
    assert materialized.layout.mtp_loss_group_root_counts == (1, 1)
    torch.testing.assert_close(
        materialized.packed_input_ids,
        torch.tensor([10, 11, 20, 21, 30, 31, 40, 41, 42, 0]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("tp_size", "cp_size", "expected"), [(1, 1, 1), (1, 2, 4), (2, 1, 4), (2, 2, 8), (4, 4, 32)]
)
def test_shared_prefix_physical_alignment_contract(
    tp_size: int, cp_size: int, expected: int
) -> None:
    assert get_shared_prefix_physical_alignment(tp_size=tp_size, cp_size=cp_size) == expected


@pytest.mark.parametrize(
    ("tp_size", "cp_size", "sequence_parallel"),
    [
        (True, 1, False),
        (4.0, 1, True),
        ("4", 1, True),
        (1, False, False),
        (1, 2.0, False),
        (1, "2", False),
        (1, 1, 0),
        (1, 1, "false"),
    ],
)
def test_shared_prefix_parallel_topology_rejects_coercible_values(
    tp_size: object, cp_size: object, sequence_parallel: object
) -> None:
    with pytest.raises(ValueError):
        resolve_shared_prefix_parallel_topology(
            tp_size=tp_size, cp_size=cp_size, sequence_parallel=sequence_parallel
        )


@pytest.mark.parametrize("raw", [True, False, 0, -8, 8.0, "8"])
def test_physical_padding_resolver_rejects_invalid_explicit_values(raw: object) -> None:
    with pytest.raises(ValueError, match="padding_multiple"):
        resolve_shared_prefix_physical_padding_multiple(
            tp_size=2, cp_size=2, padding_multiple=raw  # type: ignore[arg-type]
        )


def test_physical_padding_resolver_defaults_to_q_and_accepts_m_multiple() -> None:
    assert (
        resolve_shared_prefix_physical_padding_multiple(tp_size=2, cp_size=2, padding_multiple=None)
        == 8
    )
    assert (
        resolve_shared_prefix_physical_padding_multiple(tp_size=2, cp_size=2, padding_multiple=32)
        == 32
    )
    with pytest.raises(ValueError, match="topology alignment 8"):
        resolve_shared_prefix_physical_padding_multiple(tp_size=2, cp_size=2, padding_multiple=12)


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    return (
        torch.tensor(
            [[10, 11, 12, 20, 21, -1], [10, 11, 12, 30, 31, 32], [40, 41, 50, 51, -1, -1]],
            dtype=torch.long,
        ),
        torch.tensor([5, 6, 4]),
        torch.tensor([3, 3, 2]),
        ["shared", "shared", "singleton"],
    )


def test_tensor_plan_materializes_prompt_once_and_preserves_fallbacks() -> None:
    input_ids, input_lengths, prompt_lengths, group_ids = _batch()

    shared_bins, fallback_row_indices = plan_and_materialize(
        input_ids=input_ids,
        input_lengths=input_lengths,
        prompt_lengths=prompt_lengths,
        group_ids=group_ids,
        bin_capacity=16,
    )

    assert len(shared_bins) == 1
    shared_bin = shared_bins[0]
    assert shared_bin.layout.row_indices == (1, 0)
    torch.testing.assert_close(
        shared_bin.packed_input_ids, torch.tensor([10, 11, 12, 30, 31, 32, 20, 21])
    )
    torch.testing.assert_close(shared_bin.position_ids, torch.tensor([0, 1, 2, 3, 4, 5, 3, 4]))
    assert fallback_row_indices == (2,)


def test_tensor_indices_encode_gather_fanout_and_scatter() -> None:
    input_ids, input_lengths, prompt_lengths, group_ids = _batch()
    shared_bin = plan_and_materialize(
        input_ids=input_ids,
        input_lengths=input_lengths,
        prompt_lengths=prompt_lengths,
        group_ids=group_ids,
        bin_capacity=16,
    )[0][0]
    indices = shared_bin.indices

    torch.testing.assert_close(indices.token_gather_rows, torch.tensor([1, 1, 1, 1, 1, 1, 0, 0]))
    torch.testing.assert_close(indices.token_gather_columns, torch.tensor([0, 1, 2, 3, 4, 5, 3, 4]))
    torch.testing.assert_close(indices.completion_positions, torch.tensor([3, 4, 5, 6, 7]))
    torch.testing.assert_close(indices.predecessor_positions, torch.tensor([2, 3, 4, 2, 6]))
    assert shared_bin.layout.completion_scatter_rows == (1, 1, 1, 0, 0)
    torch.testing.assert_close(indices.completion_scatter_columns, torch.tensor([2, 3, 4, 2, 3]))


def test_context_parallel_indices_match_standard_two_chunk_zigzag() -> None:
    rank0 = get_shared_prefix_context_parallel_indices(16, cp_rank=0, cp_size=2)
    rank1 = get_shared_prefix_context_parallel_indices(16, cp_rank=1, cp_size=2)

    torch.testing.assert_close(rank0, torch.tensor([0, 1, 2, 3, 12, 13, 14, 15]))
    torch.testing.assert_close(rank1, torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]))
    torch.testing.assert_close(torch.cat((rank0, rank1)).sort().values, torch.arange(16))


def test_context_parallel_shard_pads_only_after_real_star_tokens() -> None:
    input_ids = torch.tensor([[10, 11, 12, 20, 0], [10, 11, 12, 30, 31]])
    tensor_bin = plan_and_materialize(
        input_ids=input_ids,
        input_lengths=torch.tensor([4, 5]),
        prompt_lengths=torch.tensor([3, 3]),
        group_ids=["g", "g"],
        bin_capacity=8,
    )[0][0]

    rank0 = shard_shared_prefix_tensor_bin_for_context_parallel(tensor_bin, cp_rank=0, cp_size=2)
    rank1 = shard_shared_prefix_tensor_bin_for_context_parallel(tensor_bin, cp_rank=1, cp_size=2)

    assert tensor_bin.layout.total_length == 6
    assert rank0.padded_total_length == rank1.padded_total_length == 8
    torch.testing.assert_close(rank0.global_token_indices, torch.tensor([0, 1, 6, 7]))
    torch.testing.assert_close(rank1.global_token_indices, torch.tensor([2, 3, 4, 5]))
    torch.testing.assert_close(rank0.packed_input_ids, torch.tensor([10, 11, 0, 0]))
    torch.testing.assert_close(rank1.packed_input_ids, torch.tensor([12, 30, 31, 20]))


def test_physical_branch_tails_match_dense_router_count_semantics() -> None:
    input_ids = torch.tensor([[10, 11, 12, 20, 21, 91, 92], [10, 11, 12, 30, 31, 32, 33]])
    tensor_bin = plan_and_materialize(
        input_ids=input_ids,
        input_lengths=torch.tensor([5, 7]),
        prompt_lengths=torch.tensor([3, 3]),
        group_ids=["g", "g"],
        bin_capacity=16,
        sequence_length_pad_multiple=4,
    )[0][0]
    layout = tensor_bin.layout

    assert layout.completion_lengths == (4, 2)
    assert layout.physical_completion_lengths == (5, 5)
    assert layout.physical_total_length == 13
    torch.testing.assert_close(
        tensor_bin.packed_input_ids, torch.tensor([10, 11, 12, 30, 31, 32, 33, 0, 20, 21, 0, 0, 0])
    )
    torch.testing.assert_close(
        tensor_bin.position_ids, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7])
    )
    torch.testing.assert_close(
        tensor_bin.indices.physical_padding_positions, torch.tensor([7, 10, 11, 12])
    )

    # This deterministic route surrogate depends on the same token and RoPE
    # position inputs as the real router. Prompt routes are counted twice;
    # every per-row physical tail route is counted once.
    star_routes = (tensor_bin.packed_input_ids + tensor_bin.position_ids) % 4
    multiplicities = torch.tensor([2, 2, 2] + [1] * 10)
    star_counts = torch.zeros(4, dtype=torch.long)
    star_counts.scatter_add_(0, star_routes, multiplicities)

    dense_tokens = torch.tensor([[10, 11, 12, 20, 21, 0, 0, 0], [10, 11, 12, 30, 31, 32, 33, 0]])
    dense_positions = torch.arange(8).expand_as(dense_tokens)
    dense_counts = torch.bincount(((dense_tokens + dense_positions) % 4).flatten(), minlength=4)
    torch.testing.assert_close(star_counts, dense_counts)

    mask = build_star_attention_allow_mask(tensor_bin.layout)
    assert mask.shape == (13, 13)
    assert not mask[7, 8]
    assert not mask[12, 3]
    assert mask[7, 3]
    assert mask[12, 8]

    shard = shard_shared_prefix_tensor_bin_for_context_parallel(
        tensor_bin, cp_rank=0, cp_size=2, padding_multiple=8
    )
    assert shard.padded_total_length == 16


def test_token_aligned_mtp_mask_uses_star_order_and_zeros_physical_padding() -> None:
    input_ids = torch.tensor([[10, 11, 12, 20, 21, 91, 92], [10, 11, 12, 30, 31, 32, 33]])
    tensor_bin = plan_and_materialize(
        input_ids=input_ids,
        input_lengths=torch.tensor([5, 7]),
        prompt_lengths=torch.tensor([3, 3]),
        group_ids=["g", "g"],
        bin_capacity=16,
        sequence_length_pad_multiple=4,
    )[0][0]
    source_mtp_mask = torch.tensor([[0, 0, 0, 1, 1, 7, 7], [0, 0, 0, 1, 1, 1, 1]])

    packed_mtp_mask = materialize_shared_prefix_token_aligned_tensor(
        source_mtp_mask, tensor_bin=tensor_bin, padding_value=0
    )

    torch.testing.assert_close(
        packed_mtp_mask, torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0])
    )


def test_tp_sp_shard_composes_interior_and_topology_padding() -> None:
    """M-aligned branches and the final star tail remain distinct metadata."""
    input_ids = torch.tensor(
        [
            [10, 11, 12, 13, 20, 21, 22, 0, 0],
            [10, 11, 12, 13, 30, 31, 32, 33, 0],
            [10, 11, 12, 13, 40, 41, 42, 43, 44],
        ]
    )
    tensor_bin = plan_and_materialize(
        input_ids=input_ids,
        input_lengths=torch.tensor([7, 8, 9]),
        prompt_lengths=torch.tensor([4, 4, 4]),
        group_ids=["g", "g", "g"],
        bin_capacity=64,
        sequence_length_pad_multiple=16,
    )[0][0]

    assert tensor_bin.layout.completion_lengths == (5, 4, 3)
    assert tensor_bin.layout.physical_completion_lengths == (12, 12, 12)
    assert tensor_bin.layout.physical_total_length == 40
    assert tensor_bin.indices.physical_padding_positions.numel() == 24

    shards = [
        shard_shared_prefix_tensor_bin_for_context_parallel(
            tensor_bin, cp_rank=rank, cp_size=2, tp_size=2, padding_multiple=16
        )
        for rank in range(2)
    ]
    assert all(shard.padding_multiple == 16 for shard in shards)
    assert all(shard.padded_total_length == 48 for shard in shards)
    assert all(shard.packed_input_ids.numel() == 24 for shard in shards)
    torch.testing.assert_close(
        torch.cat([shard.global_token_indices for shard in shards]).sort().values, torch.arange(48)
    )
    # The final eight positions are topology-only padding, separate from the
    # 24 per-branch native padding positions retained in the logical layout.
    restored = torch.empty(48, dtype=torch.long)
    for shard in shards:
        restored[shard.global_token_indices] = shard.packed_input_ids
    torch.testing.assert_close(restored[40:], torch.zeros(8, dtype=torch.long))


@pytest.mark.parametrize(
    "padded_length,cp_rank,cp_size,message",
    [
        (8, 0, 0, "cp_size must be positive"),
        (8, 2, 2, "cp_rank must be in"),
        (6, 0, 2, r"divisible by 2 \* cp_size"),
    ],
)
def test_context_parallel_indices_reject_invalid_topology(
    padded_length: int, cp_rank: int, cp_size: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        get_shared_prefix_context_parallel_indices(padded_length, cp_rank=cp_rank, cp_size=cp_size)


def test_cp1_star_attention_mask_has_exact_branch_isolation() -> None:
    input_ids, input_lengths, prompt_lengths, group_ids = _batch()
    (shared_bin,), _ = plan_and_materialize(
        input_ids=input_ids,
        input_lengths=input_lengths,
        prompt_lengths=prompt_lengths,
        group_ids=group_ids,
        bin_capacity=16,
    )
    mask = build_star_attention_allow_mask(shared_bin.layout)

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(mask, expected)


def test_group_id_and_exact_prompt_are_both_required() -> None:
    input_ids = torch.tensor([[1, 2, 3], [1, 2, 4], [1, 9, 5], [1, 2, 6]])
    shared_bins, fallback_row_indices = plan_and_materialize(
        input_ids=input_ids,
        input_lengths=torch.tensor([3, 3, 3, 3]),
        prompt_lengths=torch.tensor([2, 2, 2, 2]),
        group_ids=["a", "b", "a", "a"],
        bin_capacity=8,
    )

    assert [shared.layout.row_indices for shared in shared_bins] == [(0, 3)]
    assert fallback_row_indices == (1, 2)


def test_materializer_rejects_stale_prompt_tokens() -> None:
    layout = build_shared_prefix_layout(
        [SharedPrefixRow(0, "g", (1, 2), 1), SharedPrefixRow(1, "g", (1, 2), 1)]
    )
    changed_input_ids = torch.tensor([[1, 2, 3], [1, 9, 4]])

    with pytest.raises(ValueError, match="differs from the planned exact prompt"):
        materialize_shared_prefix_layout(
            changed_input_ids, input_lengths=torch.tensor([3, 3]), layout=layout
        )


def test_materializer_rejects_a_layout_past_the_unpadded_row() -> None:
    layout = build_shared_prefix_layout(
        [SharedPrefixRow(0, "g", (1, 2), 2), SharedPrefixRow(1, "g", (1, 2), 1)]
    )
    input_ids = torch.tensor([[1, 2, 3, -1], [1, 2, 4, -1]])

    with pytest.raises(ValueError, match="completion length differs from source row"):
        materialize_shared_prefix_layout(
            input_ids, input_lengths=torch.tensor([3, 3]), layout=layout
        )


def test_materializer_rejects_a_row_that_grew_past_its_layout() -> None:
    """A stale layout must not silently drop the source row's extra completion tokens."""
    layout = build_shared_prefix_layout(
        [SharedPrefixRow(0, "g", (1, 2), 2), SharedPrefixRow(1, "g", (1, 2), 1)]
    )
    input_ids = torch.tensor([[1, 2, 7, 8, 9], [1, 2, 4, 0, 0]])

    with pytest.raises(ValueError, match="completion length differs from source row"):
        materialize_shared_prefix_layout(
            input_ids, input_lengths=torch.tensor([5, 3]), layout=layout
        )


def test_layout_derives_every_index_map_from_its_rows() -> None:
    """A hand-built layout cannot carry maps that disagree with its rows."""
    layout = SharedPrefixLayout(
        group_id="g",
        prompt_token_ids=(1, 2),
        row_indices=(0, 1),
        completion_lengths=(1, 1),
        physical_completion_lengths=[2, 1],
    )
    assert layout.physical_completion_lengths == (2, 1)
    assert layout.total_length == 4
    assert layout.physical_total_length == 5
    assert layout.position_ids == (0, 1, 2, 3, 2)
    assert layout.completion_positions == (2, 4)
    assert layout.predecessor_positions == (1, 1)
    assert layout.token_gather_rows == (0, 0, 0, 0, 1)
    assert layout.token_gather_columns == (0, 1, 2, 3, 2)
    assert layout.physical_padding_positions == (3,)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            dict(completion_lengths=(2, 1), physical_completion_lengths=(1, 1)),
            "no longer than storage",
        ),
        (dict(physical_completion_lengths=(0, 1)), "positive and contiguous"),
        (dict(physical_completion_lengths=(1,)), "must have equal length"),
        (dict(row_indices=(0,)), "each completion node must own one source row"),
        (dict(row_indices=(1, 1)), "unique nonnegative integers"),
        (dict(row_indices=(0, True)), "unique nonnegative integers"),
        (dict(completion_lengths=(1, 0)), "positive"),
        (dict(prompt_token_ids=()), "non-empty prompt"),
        (dict(group_id=""), "non-empty group_id"),
    ],
)
def test_layout_rejects_inconsistent_rows(kwargs: dict, message: str) -> None:
    fields = dict(
        group_id="g", prompt_token_ids=(1, 2), row_indices=(0, 1), completion_lengths=(1, 1)
    )
    fields.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SharedPrefixLayout(**fields)


@pytest.mark.parametrize(
    "input_ids, input_lengths, prompt_lengths, group_ids, message",
    [
        (
            torch.tensor([1, 2, 3]),
            torch.tensor([3]),
            torch.tensor([2]),
            ["g"],
            "input_ids must have shape",
        ),
        (
            torch.zeros((2, 3), dtype=torch.float32),
            torch.tensor([3, 3]),
            torch.tensor([2, 2]),
            ["g", "g"],
            "integer dtype",
        ),
        (
            torch.zeros((2, 3), dtype=torch.long),
            torch.tensor([3, 4]),
            torch.tensor([2, 2]),
            ["g", "g"],
            "input_lengths must be within",
        ),
        (
            torch.zeros((2, 3), dtype=torch.long),
            torch.tensor([3, 2]),
            torch.tensor([2, 3]),
            ["g", "g"],
            "prompt_lengths must satisfy",
        ),
        (
            torch.zeros((2, 3), dtype=torch.long),
            torch.tensor([3, 3]),
            torch.tensor([2, 2]),
            ["g"],
            "group_ids must have 2 entries",
        ),
    ],
)
def test_planning_rejects_invalid_batch_metadata(
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    group_ids: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plan_and_materialize(
            input_ids=input_ids,
            input_lengths=input_lengths,
            prompt_lengths=prompt_lengths,
            group_ids=group_ids,
            bin_capacity=8,
        )


@pytest.mark.parametrize("field", ["input_ids", "input_lengths", "prompt_lengths"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.bool])
def test_row_builder_rejects_noninteger_tokens_and_lengths(field, dtype):
    batch = {
        "input_ids": torch.tensor([[2, 3, 4, 5], [2, 3, 6, 7]]),
        "input_lengths": torch.tensor([4, 4]),
        "prompt_lengths": torch.tensor([2, 2]),
        "group_ids": ["g", "g"],
    }
    batch[field] = batch[field].to(dtype)
    if dtype == torch.float32:
        batch[field] += 0.25
    with pytest.raises(ValueError, match="integer dtype"):
        build_shared_prefix_rows(**batch)
