# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU checks of the interleaved (VPP) plumbing that THD full-iteration CUDA graphs rely on.

The GLM-5.2 1M candidate runs PP4 x VPP2 with the layout
``Et*10|t*12|t*12|t*12|t*12|t*12|t*8|m*7L``, packed THD batches, static-certified dynamic
CP, a full-iteration CUDA graph and repeated MTP on an MTP-only last chunk. These tests pin
the pieces that are pure host logic: the interleaved schedule shape, MTP placement, the
dynamic-CP scheduler's per-chunk iterators, the full-iteration wrapper's per-(chunk,
microbatch) static slots, and the world-wide dynamic-CP certificate over those slots.
"""

import itertools
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.datasets import data_schedule
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.pipeline_parallel.schedules import (
    convert_schedule_table_to_order,
    get_pp_rank_microbatches,
    get_schedule_table,
)
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks, mtp_on_this_rank
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

PP = 4
VPP = 2
NUM_MICROBATCHES = 8
GROUP = PP  # microbatch_group_size_per_vp_stage defaults to the PP size.
VPP2_LAYOUT = "Et*10|t*12|t*12|t*12|t*12|t*12|t*8|m*7L"
TWIN_LAYOUT = "Et*22|t*24|t*24|t*8m*7L"


class _FakeGroup:
    def __init__(self, rank, size, ranks=None):
        self._rank = rank
        self._size = size
        self.ranks = list(range(size)) if ranks is None else list(ranks)

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _p2p(rank, vpp=VPP):
    return SimpleNamespace(pp_group=_FakeGroup(rank, PP), virtual_pipeline_model_parallel_size=vpp)


def _decoder_layers_per_chunk(layout_str):
    layout = PipelineParallelLayerLayout(layout_str, PP)
    return [
        [layout.layout[pp][vp].count(LayerType.decoder) for vp in range(len(layout.layout[pp]))]
        for pp in range(PP)
    ]


def _max_outstanding(order, layers_per_chunk):
    """Largest number of forwarded-but-not-backwarded decoder-layer microbatches on a rank."""
    outstanding = [0] * len(layers_per_chunk)
    peak_per_chunk = [0] * len(layers_per_chunk)
    peak_layers = 0
    for step in order:
        chunk = abs(step) - 1
        if step > 0:
            outstanding[chunk] += 1
            peak_per_chunk[chunk] = max(peak_per_chunk[chunk], outstanding[chunk])
            peak_layers = max(
                peak_layers, sum(n * l for n, l in zip(outstanding, layers_per_chunk))
            )
        else:
            outstanding[chunk] -= 1
            assert outstanding[chunk] >= 0
    assert outstanding == [0] * len(layers_per_chunk)
    return peak_per_chunk, peak_layers


def test_1m_vpp2_schedule_shape():
    table = get_schedule_table(NUM_MICROBATCHES, VPP, GROUP)
    # Breadth-first groups of PP microbatches per chunk: c0 mb0-3, c1 mb0-3, c0 mb4-7, c1 mb4-7.
    assert table == (
        [(mb, 0) for mb in range(4)]
        + [(mb, 1) for mb in range(4)]
        + [(mb, 0) for mb in range(4, 8)]
        + [(mb, 1) for mb in range(4, 8)]
    )
    # The interleaved schedule requires no partial last group smaller than PP.
    assert NUM_MICROBATCHES % GROUP == 0
    warmups = []
    for rank in range(PP):
        total, all_warmup, warmup, remaining = get_pp_rank_microbatches(
            NUM_MICROBATCHES, VPP, GROUP, p2p_communicator=_p2p(rank)
        )
        assert total == NUM_MICROBATCHES * VPP and not all_warmup
        assert warmup + remaining == total
        warmups.append(warmup)
    assert warmups == [10, 8, 6, 4]


def test_1m_vpp2_outstanding_activations_per_rank():
    """In-flight decoder-layer microbatches, the term that scales full-recompute memory."""
    table = get_schedule_table(NUM_MICROBATCHES, VPP, GROUP)
    vpp_layers = _decoder_layers_per_chunk(VPP2_LAYOUT)
    assert vpp_layers == [[10, 12], [12, 12], [12, 8], [12, 0]]
    peaks = {}
    for rank in range(PP):
        _, _, warmup, _ = get_pp_rank_microbatches(
            NUM_MICROBATCHES, VPP, GROUP, p2p_communicator=_p2p(rank)
        )
        order = convert_schedule_table_to_order(warmup, VPP, table)
        assert sorted(order) == sorted([1, 2, -1, -2] * NUM_MICROBATCHES)
        peaks[rank] = _max_outstanding(order, vpp_layers[rank])
    assert peaks == {
        0: ([8, 4], 118),
        1: ([8, 4], 108),
        2: ([7, 3], 84),
        # The MTP-only chunk has no decoder layer and at most one microbatch in flight.
        3: ([5, 1], 60),
    }

    # Non-interleaved twin: rank r keeps PP - r microbatches of all its layers in flight.
    twin_layers = _decoder_layers_per_chunk(TWIN_LAYOUT)
    twin_table = get_schedule_table(NUM_MICROBATCHES, 1, NUM_MICROBATCHES)
    twin = {}
    for rank in range(PP):
        _, _, warmup, _ = get_pp_rank_microbatches(
            NUM_MICROBATCHES, 1, NUM_MICROBATCHES, p2p_communicator=_p2p(rank, vpp=None)
        )
        assert warmup == PP - rank - 1
        order = convert_schedule_table_to_order(warmup, 1, twin_table)
        twin[rank] = _max_outstanding(order, twin_layers[rank])[1]
    assert twin == {0: 88, 1: 72, 2: 48, 3: 8}


def test_1m_vpp2_layout_places_mtp_on_the_last_chunk_of_the_last_rank(monkeypatch):
    # get_pg_rank only reads an explicit group once torch.distributed is initialized.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    layout = PipelineParallelLayerLayout(VPP2_LAYOUT, PP)
    assert layout.virtual_pipeline_model_parallel_size == VPP
    # MTP on the last PP rank is not "standalone", so the fixed-shape PP P2P stays in use.
    assert layout.validate_layer_layout(num_layers=78, mtp_num_layers=7) is False
    on = {
        (pp, vp): mtp_on_this_rank(
            layout, 7, ignore_virtual=False, vp_stage=vp, pp_group=_FakeGroup(pp, PP)
        )
        for pp, vp in itertools.product(range(PP), range(VPP))
    }
    assert on == {key: key == (3, 1) for key in on}
    assert [
        mtp_on_this_rank(layout, 7, ignore_virtual=True, pp_group=_FakeGroup(pp, PP))
        for pp in range(PP)
    ] == [False, False, False, True]
    config = SimpleNamespace(mtp_num_layers=7, pipeline_model_parallel_layout=layout)
    assert get_mtp_ranks([0, 64, 128, 192], config) == [192]
    # The MTP-only chunk holds the loss but no decoder layer.
    assert layout.layout[3][1] == [LayerType.mtp] * 7 + [LayerType.loss]


def test_1m_vpp2_layer_offsets(monkeypatch):
    layout = PipelineParallelLayerLayout(VPP2_LAYOUT, PP)
    monkeypatch.setattr(
        parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: VPP
    )
    offsets = {
        (pp, vp): layout.get_layer_offset(vp_stage=vp, pp_rank=pp)
        for vp in range(VPP)
        for pp in range(PP)
    }
    assert offsets == {
        (0, 0): 0,
        (1, 0): 10,
        (2, 0): 22,
        (3, 0): 34,
        (0, 1): 46,
        (1, 1): 58,
        (2, 1): 70,
        (3, 1): 78,
    }
    assert layout.get_num_layers_to_build(vp_stage=1, pp_rank=3) == 0
    assert layout.get_num_layers_to_build(LayerType.mtp, vp_stage=1, pp_rank=3) == 7


# ------------------------------------------------------------------------------------------
# Dynamic-CP scheduler under VPP
# ------------------------------------------------------------------------------------------

# Scaled-down 1M shape: every sample needs the whole DPxCP group, as 1,015,808 tokens at
# 16,384 per rank need all 64 CP ranks.
CP = 4
CAP_PER_RANK = 16
SAMPLE_LEN = 62  # ceil(log2(62 / 16)) = 2 -> CP4, padded to 2 * CP = 8 -> 64
PADDED_LEN = 64


def _sample(index):
    tokens = torch.arange(PADDED_LEN, dtype=torch.int64) + 100 * index
    return {
        "tokens": tokens,
        "labels": tokens + 1,
        "loss_mask": torch.ones(PADDED_LEN, dtype=torch.float32),
        "position_ids": torch.arange(PADDED_LEN, dtype=torch.int64),
        "original_seq_len": torch.tensor([SAMPLE_LEN], dtype=torch.int32),
        "padded_seq_len": torch.tensor([PADDED_LEN], dtype=torch.int32),
    }


class _CountingIterator:
    def __init__(self, samples):
        self.samples = iter(samples)
        self.calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        return next(self.samples)


def _run_vpp_scheduler(monkeypatch, pp_rank, num_samples):
    layout = PipelineParallelLayerLayout(VPP2_LAYOUT, PP)
    config = SimpleNamespace(
        virtual_pipeline_model_parallel_size=VPP,
        microbatch_group_size_per_vp_stage=GROUP,
        pipeline_model_parallel_layout=layout,
        mtp_num_layers=7,
    )
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda: pp_rank)
    samples = [_sample(i) for i in range(num_samples)]

    def fake_fetch(data_iterator, num_microbatches, dp_group):
        batch = [next(data_iterator) for _ in range(num_microbatches)]
        ids = list(range(len(batch)))
        return (
            batch,
            [(i, PADDED_LEN) for i in ids],
            torch.tensor(ids, dtype=torch.int32),
            torch.tensor([0, len(batch)], dtype=torch.int32),
            [PADDED_LEN] * len(batch),
        )

    def fake_reroute(batch, global_ids_this_rank, *_args):
        # One DP rank owns every sample; the DCP all-to-all is the identity here.
        return {int(i): sample for i, sample in zip(global_ids_this_rank.tolist(), batch)}

    monkeypatch.setattr(data_schedule, "get_batch_and_global_seqlens", fake_fetch)
    monkeypatch.setattr(data_schedule, "reroute_samples_to_dcp_ranks", fake_reroute)
    scheduler = DefaultDynamicCPScheduler(CAP_PER_RANK, CP, 1, GROUP, max_num_seqs=2, min_cp_size=1)
    chunk0 = _CountingIterator(samples)
    chunk1 = _CountingIterator(list(samples))
    result = scheduler.run(
        [chunk0, chunk1],
        num_samples,
        _FakeGroup(0, 1),
        _FakeGroup(0, 1),
        _FakeGroup(pp_rank, PP),
        _FakeGroup(0, CP),
        torch.device("cpu"),
        config,
    )
    return result, chunk0, chunk1


@pytest.mark.parametrize("pp_rank", range(PP))
def test_vpp_dynamic_cp_scheduler_builds_one_iterator_per_chunk(monkeypatch, pp_rank):
    (iterators, num_microbatches, seqlen_sum, _), chunk0, chunk1 = _run_vpp_scheduler(
        monkeypatch, pp_rank, NUM_MICROBATCHES
    )
    # One full-CP group per sample, already a multiple of the VPP group size.
    assert num_microbatches == NUM_MICROBATCHES
    assert seqlen_sum == NUM_MICROBATCHES * PADDED_LEN
    # The scheduler reads the first chunk's loader only; later chunks' loaders stay untouched.
    assert chunk0.calls == NUM_MICROBATCHES and chunk1.calls == 0
    assert isinstance(iterators, list) and len(iterators) == VPP
    per_chunk = [[next(iterator) for _ in range(NUM_MICROBATCHES)] for iterator in iterators]
    for iterator in iterators:
        with pytest.raises(StopIteration):
            next(iterator)

    metadata_keys = {
        "max_seqlen",
        "cu_seqlens",
        "cu_seqlens_padded",
        "zigzag_cp_min_chunk_size",
        "local_cp_size",
    }
    # Data-carrying chunks: the first chunk of PP rank 0 (tokens) and the MTP + loss chunk of
    # PP rank 3 (tokens, labels and loss mask). Every other chunk gets metadata only.
    data_keys = {
        0: {0: {"tokens", "position_ids"}},
        1: {},
        2: {},
        3: {1: {"tokens", "position_ids", "labels", "loss_mask"}},
    }[pp_rank]
    for vp_stage, samples in enumerate(per_chunk):
        for microbatch, sample in enumerate(samples):
            expected = metadata_keys | data_keys.get(vp_stage, set())
            if vp_stage in data_keys:
                assert set(sample) == expected, (pp_rank, vp_stage, microbatch)
            else:
                assert set(sample) == metadata_keys, (pp_rank, vp_stage, microbatch)
            assert int(sample["local_cp_size"]) == CP
            # The MTP certificate needs zigzag half-chunks of at least mtp_num_layers + 1 rows.
            assert int(sample["zigzag_cp_min_chunk_size"]) == PADDED_LEN // (2 * CP)
            assert sample["cu_seqlens"].tolist() == [0, SAMPLE_LEN]
            assert sample["cu_seqlens_padded"].tolist() == [0, PADDED_LEN]
    if pp_rank == 3:
        assert torch.equal(per_chunk[1][0]["tokens"], _sample(0)["tokens"])
        assert torch.equal(per_chunk[1][0]["labels"], _sample(0)["labels"])


def test_vpp_dynamic_cp_scheduler_refuses_unsplittable_partial_groups(monkeypatch):
    # Six full-CP samples cannot be split into a multiple of PP microbatches; the
    # scheduler fails instead of producing a partial last VPP group.
    with pytest.raises(AssertionError, match="no tail microbatch has enough ids to split"):
        _run_vpp_scheduler(monkeypatch, 0, 6)


# ------------------------------------------------------------------------------------------
# Full-iteration wrapper: one static slot per (model chunk, microbatch)
# ------------------------------------------------------------------------------------------


class _RecordingLoader:
    def __init__(self):
        self.calls = []

    def __call__(self, batch, stage, slot):
        self.calls.append((stage, slot, batch["vp_stage"], batch["microbatch"]))
        return {"slot": slot, "vp_stage": batch["vp_stage"], "microbatch": batch["microbatch"]}


def _wrapper_with_preparation(prepared):
    wrapper = FullCudaGraphWrapper.__new__(FullCudaGraphWrapper)
    wrapper.static_loader = _RecordingLoader()
    counters = {}

    def prepare(data_iterator, vp_stage):
        index = counters.get(vp_stage, 0)
        counters[vp_stage] = index + 1
        prepared.append((None if data_iterator is None else id(data_iterator), vp_stage))
        if data_iterator is not None:
            next(data_iterator)
        return {"vp_stage": vp_stage, "microbatch": index}

    wrapper.batch_preparation_fn = prepare
    return wrapper


@pytest.mark.parametrize("has_data", [True, False])
def test_full_iteration_wrapper_assigns_chunk_major_static_slots(has_data):
    prepared = []
    wrapper = _wrapper_with_preparation(prepared)
    model = [torch.nn.Identity(), torch.nn.Identity()]
    if has_data:
        data_iterator = [iter(range(NUM_MICROBATCHES)), iter(range(NUM_MICROBATCHES))]
    else:
        # TP ranks other than 0 receive no iterator, but still prepare (and broadcast) every
        # (chunk, microbatch) pair in the same order as TP rank 0.
        data_iterator = [None, None]
    per_chunk = wrapper._data_read_with_batch_preparation(
        data_iterator, model, "training", NUM_MICROBATCHES
    )

    assert [vp for _, vp in prepared] == [0] * NUM_MICROBATCHES + [1] * NUM_MICROBATCHES
    if has_data:
        assert [it for it, _ in prepared] == (
            [id(data_iterator[0])] * NUM_MICROBATCHES + [id(data_iterator[1])] * NUM_MICROBATCHES
        )
    assert [slot for _, slot, _, _ in wrapper.static_loader.calls] == list(
        range(VPP * NUM_MICROBATCHES)
    )
    assert len(per_chunk) == VPP
    for vp_stage, iterator in enumerate(per_chunk):
        slots = list(iterator)
        assert [slot["slot"] for slot in slots] == [
            vp_stage * NUM_MICROBATCHES + microbatch for microbatch in range(NUM_MICROBATCHES)
        ]
        assert all(slot["vp_stage"] == vp_stage for slot in slots)
        assert [slot["microbatch"] for slot in slots] == list(range(NUM_MICROBATCHES))


# ------------------------------------------------------------------------------------------
# Static dynamic-CP certificate over VPP slots
# ------------------------------------------------------------------------------------------


def _slot_metadata(rank, cp_size, num_slots, *, override=None):
    stage = rank // cp_size
    members = tuple(range(stage * cp_size, (stage + 1) * cp_size))
    metadata = [
        {
            "local_cp_size": cp_size,
            "cp_group_global_ranks": members,
            "cp_group_rank": rank - stage * cp_size,
        }
        for _ in range(num_slots)
    ]
    if override is not None:
        slot, value = override
        metadata[slot] = value
    return metadata


def _gather(world_size, metadata_by_rank, appended):
    rows = [
        FullCudaGraphWrapper._encode_new_dynamic_cp_group_state(
            metadata_by_rank[rank], False, appended[rank], world_size
        )
        for rank in range(world_size)
    ]
    assert len({len(row) for row in rows}) == 1
    return torch.tensor(rows, dtype=torch.int64)


def test_dynamic_cp_certificate_accepts_vpp_slots_on_every_pp_stage():
    # 8 ranks = PP2 x CP4 (each PP stage owns 4 consecutive ranks); 2 chunks x 4 microbatches.
    world, cp, slots = 8, 4, VPP * 4
    metadata = {rank: _slot_metadata(rank, cp, slots) for rank in range(world)}
    gathered = _gather(world, metadata, {rank: slots for rank in range(world)})
    assert (
        FullCudaGraphWrapper._validate_new_dynamic_cp_group_state(gathered, slots, slots, world)
        is None
    )


def test_dynamic_cp_certificate_rejects_one_incoherent_vpp_slot():
    world, cp, slots = 8, 4, VPP * 4
    metadata = {rank: _slot_metadata(rank, cp, slots) for rank in range(world)}
    # Rank 5 (PP stage 1) realizes a different CP group for chunk 1, microbatch 3 (slot 7).
    metadata[5] = _slot_metadata(
        5,
        cp,
        slots,
        override=(7, {"local_cp_size": 2, "cp_group_global_ranks": (4, 5), "cp_group_rank": 1}),
    )
    gathered = _gather(world, metadata, {rank: slots for rank in range(world)})
    error = FullCudaGraphWrapper._validate_new_dynamic_cp_group_state(gathered, slots, slots, world)
    assert error is not None and error.startswith("slot 7 ")


def test_dynamic_cp_certificate_rejects_a_rank_missing_a_chunk():
    world, cp, slots = 8, 4, VPP * 4
    metadata = {rank: _slot_metadata(rank, cp, slots) for rank in range(world)}
    # A rank that prepared only one chunk's slots must not pass the shared certificate.
    metadata[6] = _slot_metadata(6, cp, slots // VPP) + [None] * (slots - slots // VPP)
    appended = {rank: slots for rank in range(world)}
    appended[6] = slots // VPP
    gathered = _gather(world, metadata, appended)
    error = FullCudaGraphWrapper._validate_new_dynamic_cp_group_state(gathered, slots, slots, world)
    assert error == "ranks appended different numbers of static DCP microbatch slots"
