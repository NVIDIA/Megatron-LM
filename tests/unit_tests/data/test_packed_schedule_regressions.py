# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import random
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core import utils as core_utils
from megatron.core.datasets import data_schedule
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler, DpBalancedScheduler
from megatron.core.datasets.data_schedule_utils import (
    align_sample_id_groups,
    get_packed_sequence_length,
    pad_packed_batch_before_cp_slice,
)
from megatron.core.process_groups_config import ProcessGroupCollection


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
def no_global_process_groups(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Sequence packing must use the supplied process group collection")

    for name, value in vars(parallel_state).items():
        if name.startswith("get_") and callable(value):
            monkeypatch.setattr(parallel_state, name, fail)


@pytest.fixture
def cpu_batch_communication(monkeypatch, no_global_process_groups):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: group.size())
    monkeypatch.setattr(
        torch.distributed, "get_rank", lambda group=None: group.rank() if group else 0
    )
    monkeypatch.setattr(data_schedule, "broadcast_tensor", lambda *args: None)
    monkeypatch.setattr(data_schedule, "broadcast_scalars", lambda values, *args, **kwargs: values)

    def partition(boundaries, total_tokens, cp_size, cp_rank):
        # Transformer Engine rejects cu_seqlens that are not 1-D.
        assert boundaries.dim() == 1, f"cu_seqlens must be 1-D, got {tuple(boundaries.shape)}"
        chunks = torch.arange(total_tokens).reshape(2 * cp_size, -1)
        return torch.cat((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]))

    tex = SimpleNamespace(thd_get_partitioned_indices=partition)
    monkeypatch.setattr(data_schedule, "tex", tex)
    # Static zigzag CP batches are partitioned per document in megatron.core.utils.
    monkeypatch.setattr(core_utils, "tex", tex)


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
        pg_collection=ProcessGroupCollection(
            tp=_group(1), pp=_group(3, pp_rank), cp=cp, dynamic_cp={2: cp}
        ),
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


@pytest.mark.parametrize("linear_cp_layout", ["contiguous", "zigzag"])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_static_packing_returns_hybrid_layouts(
    cpu_batch_communication, cp_size, sequence_parallel, linear_cp_layout
):
    tp = _group(2 if sequence_parallel else 1)
    cp = _group(cp_size, stride=tp.size())
    tp_cp = _group(tp.size() * cp_size)
    result = data_schedule.get_batch_on_this_rank_for_sequence_packing(
        iter([_batch([16])]),
        dynamic_cp=False,
        pg_collection=ProcessGroupCollection(tp=tp, pp=_group(1), cp=cp, tp_cp=tp_cp),
        config=SimpleNamespace(
            sequence_parallel=sequence_parallel,
            linear_cp_layout=linear_cp_layout,
            attention_cp_layout="zigzag",
        ),
        return_context_parallel_batch=True,
    )
    assert set(result.batches_by_layout) == {linear_cp_layout, "zigzag"}
    zigzag_tokens = torch.arange(16).view(2 * cp_size, -1)[[0, -1]].view(1, -1)
    torch.testing.assert_close(result.get_batch("zigzag")["tokens"], zigzag_tokens)
    for layout in result.batches_by_layout:
        batch = result.get_batch(layout)
        assert batch["tokens"].shape == (1, 16 // cp_size)
        assert batch["cu_seqlens"].shape == batch["cu_seqlens_padded"].shape == (1, 2)
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
def test_wrap_iterator_enables_padding_aware_capacity(
    monkeypatch, no_global_process_groups, scheduler_type, alignment
):
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
        pg_collection=ProcessGroupCollection(
            dp_cp=_group(6),
            dp=_group(6),
            tp=_group(1),
            pp=_group(1),
            dynamic_cp={size: _group(size) for size in (1, 2, 6)},
        ),
    )


@pytest.mark.parametrize("local_cp_size", [1, 2, 6])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("last_cp_rank", [False, True])
def test_dynamic_layouts_use_injected_runtime_groups(
    cpu_batch_communication, monkeypatch, local_cp_size, sequence_parallel, last_cp_rank
):
    tp = _group(2 if sequence_parallel else 1)
    cp_rank = local_cp_size - 1 if last_cp_rank else 0
    cp = _group(local_cp_size, cp_rank, stride=tp.size())
    tp_cp = _group(tp.size() * local_cp_size, cp_rank * tp.size())
    pg = ProcessGroupCollection(
        tp=tp,
        pp=_group(1),
        cp=_group(2, stride=tp.size()),
        tp_cp=_group(2 * tp.size()),
        dynamic_cp={local_cp_size: cp},
        dynamic_tp_cp={local_cp_size: tp_cp},
    )
    original = data_schedule.get_batches_on_this_cp_rank

    def get_layouts(batch, **kwargs):
        assert kwargs["cp_group"] is cp
        assert kwargs["tp_group"] is tp
        assert kwargs["tp_cp_group"] is (tp_cp if sequence_parallel else None)
        return original(batch, **kwargs)

    monkeypatch.setattr(data_schedule, "get_batches_on_this_cp_rank", get_layouts)
    batch = _batch([48])
    batch["local_cp_size"] = torch.tensor(local_cp_size, dtype=torch.int32)
    result = data_schedule.get_batch_on_this_rank_for_sequence_packing(
        iter([batch]),
        dynamic_cp=True,
        pg_collection=pg,
        config=SimpleNamespace(
            sequence_parallel=sequence_parallel,
            linear_cp_layout="contiguous",
            attention_cp_layout="zigzag",
        ),
        return_context_parallel_batch=True,
    )
    local_tokens = 48 // local_cp_size
    expected = torch.arange(cp_rank * local_tokens, (cp_rank + 1) * local_tokens).view(1, -1)
    torch.testing.assert_close(result.get_batch("contiguous")["tokens"], expected)
    for layout in result.batches_by_layout:
        params = result.get_packed_seq_params(layout)
        assert params.cp_group is cp
        assert params.local_cp_size == local_cp_size
        assert result.get_batch(layout)["padding_mask"].shape == (1, local_tokens)
    assert (result.thd_plan is not None) == (local_cp_size > 1)
    if result.thd_plan is not None:
        assert len(result.thd_plan.forward_input_split_sizes) == (
            local_cp_size * tp.size() if sequence_parallel else local_cp_size
        )


def test_dynamic_group_callbacks_override_collection_maps(cpu_batch_communication):
    cp = _group(2, stride=2)
    tp_cp = _group(4)
    calls = []

    def runtime_cp(group_size):
        calls.append(("cp", group_size))
        return cp

    def runtime_tp_cp(group_size):
        calls.append(("tp_cp", group_size))
        return tp_cp

    batch = _batch([16])
    batch["local_cp_size"] = torch.tensor(2, dtype=torch.int32)
    result = data_schedule.get_batch_on_this_rank_for_sequence_packing(
        iter([batch]),
        dynamic_cp=True,
        pg_collection=ProcessGroupCollection(
            tp=_group(2),
            pp=_group(1),
            cp=_group(4),
            dynamic_cp={2: _group(2, 1)},
            dynamic_tp_cp={2: _group(4, 2)},
        ),
        dynamic_cp_group_func=runtime_cp,
        dynamic_tp_cp_group_func=runtime_tp_cp,
        config=SimpleNamespace(
            sequence_parallel=True, linear_cp_layout="contiguous", attention_cp_layout="zigzag"
        ),
        return_context_parallel_batch=True,
    )
    assert calls and all(size == 2 for _, size in calls)
    assert ("cp", 2) in calls and ("tp_cp", 2) in calls
    assert result.get_packed_seq_params("contiguous").cp_group is cp
    torch.testing.assert_close(
        result.get_batch("contiguous")["tokens"], torch.arange(8).view(1, -1)
    )


@pytest.mark.parametrize("entrypoint", ["batch", "iterator"])
def test_sequence_packing_requires_collection(no_global_process_groups, entrypoint):
    with pytest.raises(ValueError, match="pg_collection is required"):
        if entrypoint == "batch":
            data_schedule.get_batch_on_this_rank_for_sequence_packing(None)
        else:
            data_schedule.wrap_data_iterator(None, SimpleNamespace(), 1)


@pytest.mark.parametrize("missing", ["tp", "pp", "cp"])
def test_packed_batch_requires_static_groups(no_global_process_groups, missing):
    groups = {name: _group(1) for name in ("tp", "pp", "cp") if name != missing}
    with pytest.raises(ValueError, match="must define tp, pp, and cp"):
        data_schedule.get_batch_on_this_rank_for_sequence_packing(
            None, pg_collection=ProcessGroupCollection(**groups)
        )


@pytest.mark.parametrize("groups", [None, {}, {1: _group(1)}])
def test_packed_batch_rejects_missing_runtime_cp_group(cpu_batch_communication, groups):
    batch = _batch([16])
    batch["local_cp_size"] = torch.tensor(2, dtype=torch.int32)
    with pytest.raises(ValueError, match=r"pg_collection.dynamic_cp"):
        data_schedule.get_batch_on_this_rank_for_sequence_packing(
            iter([batch]),
            dynamic_cp=True,
            pg_collection=ProcessGroupCollection(
                tp=_group(1), pp=_group(1), cp=_group(2), dynamic_cp=groups
            ),
            config=SimpleNamespace(),
        )


@pytest.mark.parametrize("dynamic_cp", [False, True])
def test_sequence_parallel_rejects_missing_tp_cp_group(cpu_batch_communication, dynamic_cp):
    batch = _batch([16])
    batch["local_cp_size"] = torch.tensor(2, dtype=torch.int32)
    pg = ProcessGroupCollection(tp=_group(2), pp=_group(1), cp=_group(2), dynamic_cp={2: _group(2)})
    error = r"pg_collection.dynamic_tp_cp.*runtime CP size 2" if dynamic_cp else "tp_cp_group"
    with pytest.raises(ValueError, match=error):
        data_schedule.get_batch_on_this_rank_for_sequence_packing(
            iter([batch]),
            dynamic_cp=dynamic_cp,
            pg_collection=pg,
            config=SimpleNamespace(
                sequence_parallel=True, linear_cp_layout="contiguous", attention_cp_layout="zigzag"
            ),
            return_context_parallel_batch=True,
        )


def _dynamic_config(min_cp_size):
    return SimpleNamespace(
        sequence_packing_scheduler="default_dynamic_cp",
        min_dynamic_context_parallel_size=min_cp_size,
        max_seqlen_per_dp_cp_rank=16,
        virtual_pipeline_model_parallel_size=None,
    )


@pytest.mark.parametrize("cp_sizes,min_cp_size", [((2, 6), 2), ((6,), 6)])
def test_dynamic_scheduler_uses_only_supplied_group_sizes(
    cpu_batch_communication, monkeypatch, cp_sizes, min_cp_size
):
    pg = ProcessGroupCollection(
        dp=_group(3),
        dp_cp=_group(6),
        tp=_group(1),
        pp=_group(1),
        dynamic_cp={size: _group(size) for size in cp_sizes},
    )

    def run(scheduler, iterator, count, dp_group, tp_group, pp_group, dp_cp_group, *args):
        assert scheduler.cp_group_sizes == cp_sizes
        assert (dp_group, tp_group, pp_group, dp_cp_group) == (pg.dp, pg.tp, pg.pp, pg.dp_cp)
        batches = scheduler.get_groups_and_subsamples(list(enumerate([16] * 12)))
        for batch in batches:
            for ids in batch:
                assert batch.count(ids) in cp_sizes
        return iterator, len(batches), 192, 3072

    monkeypatch.setattr(DefaultDynamicCPScheduler, "run", run)
    result = data_schedule.wrap_data_iterator(None, _dynamic_config(min_cp_size), 1, pg)
    assert result[1] > 0


@pytest.mark.parametrize(
    "cp_sizes,min_cp_size,error",
    [
        ((), 1, r"pg_collection.dynamic_cp is required"),
        ((2, 6), 1, "min_cp_size=1.*expected one of"),
        ((3, 6), 3, "Unsupported Dynamic CP group sizes"),
    ],
)
def test_dynamic_scheduler_rejects_unavailable_group_sizes(
    cpu_batch_communication, cp_sizes, min_cp_size, error
):
    pg = ProcessGroupCollection(
        dp=_group(3),
        dp_cp=_group(6),
        tp=_group(1),
        pp=_group(1),
        dynamic_cp={size: _group(size) for size in cp_sizes},
    )
    with pytest.raises(ValueError, match=error):
        data_schedule.wrap_data_iterator(None, _dynamic_config(min_cp_size), 1, pg)


def test_dynamic_scheduler_does_not_invent_full_domain_group():
    scheduler = DefaultDynamicCPScheduler(16, 2, 3, None, min_cp_size=2, cp_group_sizes=(2,))
    with pytest.raises(ValueError, match=r"full-domain group 6 is unavailable"):
        scheduler.get_groups_and_subsamples([(0, 16)])

    groups = scheduler.get_groups_and_subsamples([(0, 32), (1, 32), (2, 32)])
    assert groups == [[[0], [0], [1], [1], [2], [2]]]


def test_vpp_alignment_does_not_invent_full_domain_group():
    with pytest.raises(ValueError, match="Cannot align packed microbatches"):
        align_sample_id_groups(
            [[[0], [0], [1], [1], [2], [2]]],
            2,
            cp_group_sizes=(2,),
            sample_lengths={0: 32, 1: 32, 2: 32},
            max_seq_len_per_rank=16,
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
