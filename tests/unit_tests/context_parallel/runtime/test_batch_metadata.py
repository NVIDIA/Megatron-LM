# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime group metadata from the packed batch producer, including static compatibility."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from megatron.core.context_parallel import get_batches_on_this_cp_rank
from megatron.core.datasets.data_schedule import get_batch_on_this_rank_for_sequence_packing
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import _resolve_dynamic_cp_group_for_batch, get_batch_on_this_cp_rank

pytestmark = pytest.mark.launch_on_gb200


def _mock_group(size, rank=0):
    group = MagicMock()
    group.size.return_value = size
    group.rank.return_value = rank
    return group


def test_dynamic_cp_singleton_runtime_group_keeps_the_batch_unsharded():
    """A CP-off microbatch (local_cp_size == 1) resolves and records the singleton group."""
    static_cp_group, singleton = _mock_group(2), _mock_group(1)
    tokens = torch.arange(8).view(1, 8)
    batch = {
        "tokens": tokens.clone(),
        "labels": tokens.clone() + 1,
        "loss_mask": torch.ones(1, 8),
        "position_ids": tokens.clone(),
        "cu_seqlens": torch.tensor([[0, 8]], dtype=torch.int32),
        "local_cp_size": torch.tensor([1], dtype=torch.int32),
    }
    requested_sizes = []

    def group_func(group_size):
        requested_sizes.append(group_size)
        return singleton

    with (
        patch("torch.distributed.get_world_size", side_effect=lambda group: group.size()),
        patch("torch.distributed.get_rank", side_effect=lambda group: group.rank()),
    ):
        batch = get_batch_on_this_cp_rank(
            batch, dynamic_cp=True, cp_group=static_cp_group, dynamic_cp_group_func=group_func
        )

    assert requested_sizes == [1]
    assert batch["hybrid_cp_group"] is singleton
    torch.testing.assert_close(batch["tokens"], tokens)
    torch.testing.assert_close(batch["labels"], tokens + 1)


def test_dynamic_cp_group_resolution_prefers_the_scheduled_group():
    scheduled = _mock_group(2)
    batch = {"local_cp_size": 2, "hybrid_cp_group": scheduled}

    def group_func(group_size):
        raise AssertionError("an attached runtime group must not be looked up again")

    assert _resolve_dynamic_cp_group_for_batch(batch, group_func) is scheduled


@pytest.mark.parametrize(
    "local_cp_size, group_func, match",
    [
        (None, lambda group_size: _mock_group(2), "local_cp_size is required"),
        (torch.tensor([2]), None, "dynamic_cp_group_func is required"),
        (torch.tensor([4]), lambda group_size: _mock_group(2), "does not match local_cp_size"),
    ],
)
def test_dynamic_cp_group_resolution_rejects_inconsistent_metadata(
    local_cp_size, group_func, match
):
    with pytest.raises(ValueError, match=match):
        _resolve_dynamic_cp_group_for_batch({"local_cp_size": local_cp_size}, group_func)


@pytest.mark.parametrize("is_hybrid_cp", [True, False])
def test_layout_views_carry_the_runtime_tp_cp_group_under_dynamic_cp(is_hybrid_cp):
    singleton = _mock_group(1)
    tp_cp_group = object()
    tokens = torch.arange(8).view(1, 8)
    batch = {
        "tokens": tokens,
        "labels": tokens + 1,
        "loss_mask": torch.ones(1, 8),
        "position_ids": tokens.clone(),
        "cu_seqlens": torch.tensor([[0, 8]], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([[0, 8]], dtype=torch.int32),
        "max_seqlen": torch.tensor([8], dtype=torch.int32),
        "local_cp_size": torch.tensor([1], dtype=torch.int32) if is_hybrid_cp else None,
        "hybrid_cp_group": singleton if is_hybrid_cp else None,
    }

    with (
        patch("torch.distributed.get_world_size", side_effect=lambda group: group.size()),
        patch("torch.distributed.get_rank", side_effect=lambda group: group.rank()),
    ):
        cp_batch = get_batches_on_this_cp_rank(
            batch,
            boundary_layout="contiguous",
            dynamic_cp=is_hybrid_cp,
            cp_group=singleton,
            additional_layouts={"zigzag"},
            tp_cp_group=tp_cp_group,
        )

    for layout in ("contiguous", "zigzag"):
        expected = tp_cp_group if is_hybrid_cp else None
        assert cp_batch.get_packed_seq_params(layout).tp_cp_group is expected


@pytest.mark.parametrize("linear_cp_layout", ["contiguous", "zigzag"])
@pytest.mark.parametrize("cp_size", [1, 2])
def test_scheduler_builds_static_cp_layout_views(cp_size, linear_cp_layout):
    """Without dynamic CP the views are sharded over the static CP group (no local_cp_size)."""

    class _Group:
        def __init__(self, size, rank=0):
            self._size = size
            self._rank = rank

        def size(self):
            return self._size

        def rank(self):
            return self._rank

    def _zigzag(cu_seqlens, total_tokens, cp_size, cp_rank):
        chunks = torch.arange(total_tokens).reshape(2 * cp_size, -1)
        return torch.cat((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]))

    static_cp_group = _Group(cp_size)
    pg_collection = ProcessGroupCollection(tp=_Group(1), pp=_Group(1), cp=static_cp_group)
    config = SimpleNamespace(
        linear_cp_layout=linear_cp_layout, attention_cp_layout="zigzag", sequence_parallel=False
    )
    tokens = torch.arange(8, dtype=torch.int64)
    batch = {
        "tokens": tokens.clone(),
        "labels": tokens.clone() + 100,
        "loss_mask": torch.ones(8, dtype=torch.float32),
        "position_ids": tokens.clone(),
        "cu_seqlens": torch.tensor([0, 8], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 8], dtype=torch.int32),
        "max_seqlen": torch.tensor([8], dtype=torch.int32),
    }

    with (
        patch("torch.cuda.current_device", return_value=torch.device("cpu")),
        patch("torch.distributed.get_process_group_ranks", return_value=[0]),
        patch("torch.distributed.get_world_size", side_effect=lambda group: group.size()),
        patch("torch.distributed.get_rank", side_effect=lambda group: group.rank()),
        patch("megatron.core.datasets.data_schedule.broadcast_tensor"),
        # Static zigzag views are partitioned per document with TE, which needs CUDA.
        patch("megatron.core.utils.tex", SimpleNamespace(thd_get_partitioned_indices=_zigzag)),
    ):
        cp_batch = get_batch_on_this_rank_for_sequence_packing(
            iter([batch]),
            dynamic_cp=False,
            pg_collection=pg_collection,
            config=config,
            return_context_parallel_batch=True,
        )

    assert set(cp_batch.batches_by_layout) == {linear_cp_layout, "zigzag"}
    assert cp_batch.get_packed_seq_params("zigzag").local_cp_size is None
    if linear_cp_layout == "contiguous":
        expected_contiguous = tokens[: 8 // cp_size].view(1, -1)
        torch.testing.assert_close(cp_batch.get_batch("contiguous")["tokens"], expected_contiguous)
    expected_zigzag = tokens[[0, 1, 6, 7]] if cp_size == 2 else tokens
    torch.testing.assert_close(cp_batch.get_batch("zigzag")["tokens"], expected_zigzag.view(1, -1))


@pytest.mark.parametrize("sequence_parallel", [True, False])
def test_packed_params_carry_the_runtime_tp_cp_group(sequence_parallel):
    """Under dynamic CP with SP, layout conversion needs the TP x runtime-CP group per microbatch."""

    class _Group:
        def __init__(self, size, rank=0):
            self._size = size
            self._rank = rank

        def size(self):
            return self._size

        def rank(self):
            return self._rank

    runtime_cp_group, runtime_tp_cp_group = _Group(1), object()
    pg_collection = ProcessGroupCollection(tp=_Group(2), pp=_Group(1), cp=_Group(1))
    config = SimpleNamespace(sequence_parallel=sequence_parallel)
    tokens = torch.arange(8, dtype=torch.int64)
    batch = {
        "tokens": tokens.clone(),
        "labels": tokens.clone() + 100,
        "loss_mask": torch.ones(8, dtype=torch.float32),
        "position_ids": tokens.clone(),
        "cu_seqlens": torch.tensor([0, 8], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 8], dtype=torch.int32),
        "max_seqlen": torch.tensor([8], dtype=torch.int32),
        "local_cp_size": torch.tensor([1], dtype=torch.int32),
    }

    with (
        patch("torch.cuda.current_device", return_value=torch.device("cpu")),
        patch("torch.distributed.get_process_group_ranks", return_value=[0, 1]),
        patch("torch.distributed.get_world_size", side_effect=lambda group: group.size()),
        patch("torch.distributed.get_rank", side_effect=lambda group: group.rank()),
        patch("megatron.core.datasets.data_schedule.broadcast_tensor"),
        patch("megatron.core.datasets.data_schedule_utils.broadcast_tensor"),
    ):
        *_, packed_seq_params, _ = get_batch_on_this_rank_for_sequence_packing(
            iter([batch]),
            dynamic_cp=True,
            pg_collection=pg_collection,
            config=config,
            dynamic_cp_group_func=lambda group_size: runtime_cp_group,
            dynamic_tp_cp_group_func=lambda group_size: runtime_tp_cp_group,
        )

    assert packed_seq_params.cp_group is runtime_cp_group
    expected = runtime_tp_cp_group if sequence_parallel else None
    assert packed_seq_params.tp_cp_group is expected


def test_dynamic_sequence_packing_requires_config():
    with pytest.raises(ValueError, match="config is required for dynamic CP"):
        get_batch_on_this_rank_for_sequence_packing(
            None,
            dynamic_cp=True,
            pg_collection=ProcessGroupCollection(
                tp=_mock_group(1), pp=_mock_group(1), cp=_mock_group(1)
            ),
        )
