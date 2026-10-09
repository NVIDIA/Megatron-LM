# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets import data_schedule
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler, DpBalancedScheduler
from megatron.core.datasets.data_schedule_utils import (
    align_sample_id_groups,
    get_packed_sequence_length,
    pad_packed_batch_before_cp_slice,
)


def _group(size, rank=0, stride=1):
    return SimpleNamespace(
        size=lambda: size, rank=lambda: rank, ranks=list(range(0, size * stride, stride))
    )


def _batch(lengths):
    lengths = torch.tensor(lengths, dtype=torch.int32)
    boundaries = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    tokens = torch.arange(int(boundaries[-1]), dtype=torch.int64)
    return {
        "tokens": tokens.clone(),
        "labels": tokens.clone() + 1,
        "position_ids": tokens.clone(),
        "loss_mask": torch.ones_like(tokens, dtype=torch.float32),
        "cu_seqlens": boundaries.clone(),
        "cu_seqlens_padded": boundaries.clone(),
        "max_seqlen": lengths.max(),
    }


@pytest.fixture
def cpu_batch_communication(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: group.size())
    monkeypatch.setattr(
        torch.distributed, "get_rank", lambda group=None: group.rank() if group else 0
    )
    monkeypatch.setattr(data_schedule, "broadcast_tensor", lambda *args: None)
    monkeypatch.setattr(data_schedule, "broadcast_scalars", lambda values, *args, **kwargs: values)

    def partition(boundaries, total_tokens, cp_size, cp_rank):
        chunks = torch.arange(total_tokens).reshape(2 * cp_size, -1)
        return torch.cat((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]))

    monkeypatch.setattr(
        data_schedule, "tex", SimpleNamespace(thd_get_partitioned_indices=partition)
    )


@pytest.mark.parametrize("dynamic_cp", [False, True])
@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("pp_rank,mtp", [(0, False), (1, False), (2, False), (1, True)])
def test_cp_slice_respects_pipeline_field_ownership(
    cpu_batch_communication, dynamic_cp, cp_rank, pp_rank, mtp
):
    batch = _batch([8])
    batch["local_cp_size"] = torch.tensor(2, dtype=torch.int32)
    owned = set()
    if pp_rank == 0 or mtp:
        owned.update(("tokens", "position_ids"))
    if pp_rank == 2 or mtp:
        owned.update(("labels", "loss_mask"))
    for key in ("tokens", "position_ids", "labels", "loss_mask"):
        if key not in owned:
            del batch[key]
    cp = _group(2, cp_rank)
    result = data_schedule.get_batch_on_this_rank_for_sequence_packing(
        iter([batch]),
        mtp_on_this_rank=mtp,
        dynamic_cp=dynamic_cp,
        dynamic_cp_group_func=lambda group_size: cp,
        pg_collection=SimpleNamespace(tp=_group(1), pp=_group(3, pp_rank), cp=cp),
        config=SimpleNamespace(),
    )
    expected = torch.tensor([0, 1, 6, 7] if cp_rank == 0 else [2, 3, 4, 5]).view(1, -1)
    for key, tensor in zip(
        ("tokens", "labels", "loss_mask", "attention_mask", "position_ids"), result
    ):
        if key not in owned:
            assert tensor is None
        elif key == "labels":
            torch.testing.assert_close(tensor, expected + 1)
        elif key == "loss_mask":
            torch.testing.assert_close(tensor, torch.ones(1, 4))
        else:
            torch.testing.assert_close(tensor, expected)
    assert result[-1].shape == (1, 4)


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_static_packing_returns_hybrid_layouts(cpu_batch_communication, cp_size, sequence_parallel):
    tp = _group(2 if sequence_parallel else 1)
    cp = _group(cp_size, stride=tp.size())
    tp_cp = _group(tp.size() * cp_size)
    result = data_schedule.get_batch_on_this_rank_for_sequence_packing(
        iter([_batch([16])]),
        dynamic_cp=False,
        pg_collection=SimpleNamespace(tp=tp, pp=_group(1), cp=cp, tp_cp=tp_cp),
        config=SimpleNamespace(
            sequence_parallel=sequence_parallel,
            linear_cp_layout="contiguous",
            attention_cp_layout="zigzag",
        ),
        return_context_parallel_batch=True,
    )
    assert set(result.batches_by_layout) == {"contiguous", "zigzag"}
    for layout in result.batches_by_layout:
        assert result.get_batch(layout)["tokens"].shape == (1, 16 // cp_size)
        assert result.get_packed_seq_params(layout).local_cp_size is None


@pytest.mark.parametrize(
    "world_size,lengths,capacity", [(6, [192] * 3, 128), (2, [6144, 2048], 4096)]
)
def test_vpp_alignment_splits_non_power_of_two_and_full_domain_batches(
    world_size, lengths, capacity
):
    scheduler = DefaultDynamicCPScheduler(capacity, 1, world_size, 2)
    groups = scheduler.get_groups_and_subsamples(list(enumerate(lengths)))
    assert len(groups) == 2
    seen = []
    for microbatch in groups:
        assert len(microbatch) == world_size
        assert all(microbatch)
        ids = set(sample_id for rank_ids in microbatch for sample_id in rank_ids)
        seen.extend(ids)
        for sample_id in ids:
            ranks = [rank for rank, rank_ids in enumerate(microbatch) if sample_id in rank_ids]
            assert len(ranks) in scheduler.cp_group_sizes
            assert ranks == list(range(ranks[0], ranks[0] + len(ranks)))
            assert ranks[0] % len(ranks) == 0
        for rank_ids in microbatch:
            cp_size = sum(microbatch_ids == rank_ids for microbatch_ids in microbatch)
            assert sum(lengths[sample_id] for sample_id in rank_ids) / cp_size <= capacity
    assert sorted(seen) == list(range(len(lengths)))


def test_dynamic_cp_capacity_counts_per_sequence_padding():
    scheduler = DefaultDynamicCPScheduler(2048, 1, 6, None, min_cp_size=6, pad_sequences=True)
    groups = scheduler.get_groups_and_subsamples(list(enumerate([1024] * 12)))
    assert len(groups) == 2
    seen = []
    for microbatch in groups:
        ids = microbatch[0]
        assert all(rank_ids == ids for rank_ids in microbatch)
        seen.extend(ids)
        batch = _batch([1024] * len(ids))
        batch["padding_mask"] = torch.zeros_like(batch["tokens"], dtype=torch.bool)
        pad_packed_batch_before_cp_slice(
            batch,
            SimpleNamespace(pad_packed_seq_alignment=32, max_seqlen_per_dp_cp_rank=2048),
            6,
            1,
        )
        assert batch["tokens"].numel() // 6 <= 2048
        assert (batch["tokens"].numel() // 6) % 32 == 0
        assert batch["loss_mask"].sum() == 1024 * len(ids)
    assert sorted(seen) == list(range(12))


@pytest.mark.parametrize('pad_sequences', [False, True])
def test_static_cp_capacity_counts_per_sequence_padding(pad_sequences):
    scheduler = DpBalancedScheduler(2048, 6, 1, None, pad_sequences=pad_sequences)
    groups = scheduler.get_groups_and_subsamples(list(enumerate([1024] * 12)))
    assert len(groups) == (2 if pad_sequences else 1)
    assert [sid for group in groups for sid in group[0]] == list(range(12))


@pytest.mark.parametrize('scheduler_type', ['dp_balanced', 'default_dynamic_cp'])
@pytest.mark.parametrize('alignment', [None, 32, 'max'])
def test_wrap_iterator_enables_padding_aware_capacity(monkeypatch, scheduler_type, alignment):
    def run(scheduler, *args):
        assert scheduler.pad_sequences == (alignment is not None)
        expected_capacity = 2048 if alignment == 32 else 2050
        assert scheduler.max_seqlen_per_dp_cp_rank == expected_capacity
        return None, 0, 0, 0

    monkeypatch.setattr(DpBalancedScheduler, 'run', run)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    data_schedule.wrap_data_iterator(
        None,
        SimpleNamespace(
            sequence_packing_scheduler=scheduler_type,
            min_dynamic_context_parallel_size=1,
            max_seqlen_per_dp_cp_rank=2050,
            pad_packed_seq_alignment=alignment,
            virtual_pipeline_model_parallel_size=None,
        ),
        1,
        pg_collection=SimpleNamespace(dp_cp=_group(6), dp=_group(6), tp=_group(1), pp=_group(1)),
    )


def test_vpp_alignment_rejects_unsplittable_batch_without_looping():
    with pytest.raises(ValueError, match='Cannot align packed microbatches'):
        align_sample_id_groups([[[0], [0]]], 2)


@pytest.mark.parametrize('world_size', [1, 2, 4, 6, 8, 12, 14, 16])
@pytest.mark.parametrize('pad_sequences', [False, True])
def test_dynamic_cp_schedule_preserves_samples_groups_and_capacity(world_size, pad_sequences):
    rng = random.Random(7047)
    for _ in range(25):
        lengths = [rng.randrange(1, world_size * 32 + 1) for _ in range(4 * rng.randrange(1, 8))]
        scheduler = DefaultDynamicCPScheduler(32, 1, world_size, 4, pad_sequences=pad_sequences)
        batches = scheduler.get_groups_and_subsamples(list(enumerate(lengths)))
        assert len(batches) % 4 == 0
        seen = []
        for batch in batches:
            rank = 0
            while rank < world_size:
                ids = batch[rank]
                size = batch.count(ids)
                assert ids and size in scheduler.cp_group_sizes
                assert rank % size == 0
                assert batch[rank : rank + size] == [ids] * size
                assert (
                    sum(
                        get_packed_sequence_length(lengths[sid], size, pad_sequences) for sid in ids
                    )
                    <= 32 * size
                )
                seen.extend(ids)
                rank += size
        assert sorted(seen) == list(range(len(lengths)))
