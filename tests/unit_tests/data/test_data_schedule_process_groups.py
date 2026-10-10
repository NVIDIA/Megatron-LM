# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Process groups of the hybrid-CP and sequence-packing data schedules.

GTP-remat peers read distinct samples, so an explicit collection must select the same
GTP-remat-inclusive data domain as the global default. The sample reroute must map global
ranks to group ranks without assuming the default rank layout.
"""

import contextlib
import itertools
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.datasets import data_schedule
from megatron.core.datasets.data_schedule import HybridCPDataLoaderWrapper, wrap_data_iterator
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils

MAX_SEQLEN_PER_RANK = 128
NUM_SAMPLES_PER_RANK = 8
NUM_SUBSAMPLES_PER_RANK = 6


class _FakeGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


def _require_world_size(min_size, model_size=1):
    if Utils.world_size < min_size or Utils.world_size % model_size != 0:
        pytest.skip(f"needs a world size of at least {min_size} divisible by {model_size}")


def _ranks(group):
    return dist.get_process_group_ranks(group)


@contextlib.contextmanager
def _forbid_global_groups(monkeypatch):
    """Fail if the explicit-collection path reads a global group."""

    def _global_read(*args, **kwargs):
        raise AssertionError("the explicit pg_collection path read a global process group")

    with monkeypatch.context() as patch:
        for name in (
            "get_data_parallel_group",
            "get_tensor_model_parallel_group",
            "get_pipeline_model_parallel_group",
        ):
            patch.setattr(parallel_state, name, _global_read)
        yield


def _hybrid_cp_config():
    return SimpleNamespace(max_seqlen_per_dp_cp_rank=MAX_SEQLEN_PER_RANK)


def _packing_config():
    return SimpleNamespace(
        max_seqlen_per_dp_cp_rank=MAX_SEQLEN_PER_RANK,
        microbatch_group_size_per_vp_stage=None,
        virtual_pipeline_model_parallel_size=None,
        sequence_packing_scheduler="dp_balanced",
    )


def _hybrid_cp_length(sample_id):
    # Every sixth sub-sample needs two ranks; the others vary so a misrouted one changes sizes.
    return 2 * MAX_SEQLEN_PER_RANK - 32 if sample_id % 6 == 0 else 16 * (1 + sample_id % 5)


def _packing_length(sample_id):
    return 16 * (1 + (7 * sample_id) % 5)


def _sample(sample_id, length):
    """A sample whose tokens all equal ``sample_id + 1`` so the receiver can identify it."""
    tokens = torch.full((length,), sample_id + 1, dtype=torch.int64, device="cuda")
    return {
        "tokens": tokens,
        "labels": tokens + 1,
        "loss_mask": torch.ones(length, dtype=torch.float32, device="cuda"),
        "position_ids": torch.arange(length, dtype=torch.int64, device="cuda"),
    }


def _hybrid_cp_batch(dp_rank):
    """One packed sample of sub-samples; DP rank r owns global ids r*K .. r*K + K - 1."""
    ids = range(dp_rank * NUM_SUBSAMPLES_PER_RANK, (dp_rank + 1) * NUM_SUBSAMPLES_PER_RANK)
    subsamples = [_sample(i, _hybrid_cp_length(i)) for i in ids]
    packed = {key: torch.cat([s[key] for s in subsamples]) for key in subsamples[0]}
    lengths = [_hybrid_cp_length(i) for i in ids]
    packed["cu_seqlens"] = torch.tensor(
        [0, *itertools.accumulate(lengths)], dtype=torch.int32, device="cuda"
    )
    return [packed]


def _run_hybrid_cp(pg_collection, dp_group):
    wrapper = HybridCPDataLoaderWrapper(
        iter([_hybrid_cp_batch(dp_group.rank())]), _hybrid_cp_config(), pg_collection=pg_collection
    )
    samples, sample_id_groups = next(wrapper)
    return {gid: sample["tokens"].cpu() for gid, sample in samples.items()}, sample_id_groups


def _check_hybrid_cp_result(tokens, sample_id_groups, dp_cp_rank):
    """This rank received exactly the sub-samples scheduled on it, with their own data."""
    expected = sorted({gid for group in sample_id_groups for gid in group[dp_cp_rank]})
    assert sorted(tokens) == expected
    for gid, sample_tokens in tokens.items():
        length = _hybrid_cp_length(gid)
        assert torch.equal(sample_tokens, torch.full((length,), gid + 1, dtype=torch.int64))


def _packing_samples(dp_rank):
    """Single-sequence samples; DP rank r owns global ids r*N .. r*N + N - 1."""
    samples = []
    for i in range(dp_rank * NUM_SAMPLES_PER_RANK, (dp_rank + 1) * NUM_SAMPLES_PER_RANK):
        sample = _sample(i, _packing_length(i))
        sample["cu_seqlens"] = torch.tensor(
            [0, _packing_length(i)], dtype=torch.int32, device="cuda"
        )
        samples.append(sample)
    return samples


def _run_packing(pg_collection, dp_group, dp_cp_group, cp_size, is_reader):
    """Wrap a deterministic data iterator; check the packed samples over the data domain."""
    data_iterator = iter(_packing_samples(dp_group.rank())) if is_reader else None
    new_iterator, num_micro_batches, _, _ = wrap_data_iterator(
        data_iterator, _packing_config(), NUM_SAMPLES_PER_RANK, pg_collection=pg_collection
    )
    if not is_reader:
        assert new_iterator is None
        return num_micro_batches, []
    packed = [next(new_iterator) for _ in range(num_micro_batches)]

    sample_ids = []
    for sample in packed:
        bounds = sample["cu_seqlens_padded"].tolist()
        tokens = sample["tokens"].cpu()
        for start, end in zip(bounds[:-1], bounds[1:]):
            sample_id = int(tokens[start]) - 1
            assert end - start == _packing_length(sample_id)
            assert torch.all(tokens[start:end] == sample_id + 1)
            sample_ids.append(sample_id)

    # Every sample of the data domain lands once on each CP rank of one DP replica.
    gathered = [None] * dp_cp_group.size()
    dist.all_gather_object(gathered, sample_ids, group=dp_cp_group)
    counts = Counter(sample_id for rank_ids in gathered for sample_id in rank_ids)
    assert counts == {i: cp_size for i in range(dp_group.size() * NUM_SAMPLES_PER_RANK)}
    return num_micro_batches, [sample["tokens"].tolist() for sample in packed]


def _capture_scheduler_groups(monkeypatch, pg_collection):
    """Return the (dp, tp, pp, dp_cp) groups and sizes wrap_data_iterator schedules over."""
    captured = []

    def _run(self, data_iterator, num_microbatches, dp_group, tp_group, pp_group, dp_cp_group, *_):
        captured.append((self.dp_size, self.cp_size, dp_group, tp_group, pp_group, dp_cp_group))
        return None, 0, 0.0, 0.0

    with monkeypatch.context() as patch:
        patch.setattr(data_schedule.DpBalancedScheduler, "run", _run)
        wrap_data_iterator(None, _packing_config(), 1, pg_collection=pg_collection)
    return captured[0]


def _fake_collection(**overrides):
    groups = dict(
        dp=_FakeGroup(2),
        dp_cp=_FakeGroup(2),
        dp_gtp_remat=_FakeGroup(4),
        dp_cp_gtp_remat=_FakeGroup(4),
        tp=_FakeGroup(1),
        pp=_FakeGroup(1),
    )
    groups.update(overrides)
    return ProcessGroupCollection(**{k: v for k, v in groups.items() if v != "absent"})


def test_explicit_collection_uses_gtp_remat_groups(monkeypatch):
    pg_collection = _fake_collection()
    wrapper = HybridCPDataLoaderWrapper(None, _hybrid_cp_config(), pg_collection=pg_collection)
    assert wrapper.dp_group is pg_collection.dp_gtp_remat
    assert wrapper.dp_cp_group is pg_collection.dp_cp_gtp_remat

    dp_size, cp_size, dp_group, tp_group, pp_group, dp_cp_group = _capture_scheduler_groups(
        monkeypatch, pg_collection
    )
    assert (dp_size, cp_size) == (4, 1)
    assert dp_group is pg_collection.dp_gtp_remat and dp_cp_group is pg_collection.dp_cp_gtp_remat
    assert tp_group is pg_collection.tp and pp_group is pg_collection.pp


def test_collection_without_gtp_remat_fields_uses_replicate_groups(monkeypatch):
    pg_collection = _fake_collection(dp_gtp_remat="absent", dp_cp_gtp_remat="absent")
    wrapper = HybridCPDataLoaderWrapper(None, _hybrid_cp_config(), pg_collection=pg_collection)
    assert wrapper.dp_group is pg_collection.dp
    assert wrapper.dp_cp_group is pg_collection.dp_cp

    _, _, dp_group, _, _, dp_cp_group = _capture_scheduler_groups(monkeypatch, pg_collection)
    assert dp_group is pg_collection.dp and dp_cp_group is pg_collection.dp_cp


@pytest.mark.parametrize("field", ["dp_gtp_remat", "dp_cp_gtp_remat"])
def test_hybrid_cp_wrapper_rejects_a_none_group(field):
    with pytest.raises(ValueError, match="pg_collection must set"):
        HybridCPDataLoaderWrapper(
            None, _hybrid_cp_config(), pg_collection=_fake_collection(**{field: None})
        )


@pytest.mark.parametrize("field", ["dp_gtp_remat", "dp_cp_gtp_remat", "tp", "pp"])
def test_wrap_data_iterator_rejects_a_none_group(field):
    with pytest.raises(ValueError, match="pg_collection must set"):
        wrap_data_iterator(
            None, _packing_config(), 1, pg_collection=_fake_collection(**{field: None})
        )


@pytest.mark.parametrize(("tp", "cp", "gtp"), [(1, 1, 1), (1, 1, 2), (2, 1, 2), (1, 2, 2)])
def test_explicit_collection_selects_the_global_data_domain(monkeypatch, tp, cp, gtp):
    """use_mpu_process_groups() selects the ranks of the global (GTP-remat-inclusive) groups."""
    _require_world_size(4, tp * cp * gtp)
    Utils.initialize_model_parallel(tp, 1, context_parallel_size=cp, gtp_remat_size=gtp)
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        global_dp_cp = _ranks(parallel_state.get_data_parallel_group(with_context_parallel=True))
        global_dp = _ranks(parallel_state.get_data_parallel_group())

        fallback = HybridCPDataLoaderWrapper(None, _hybrid_cp_config())
        with _forbid_global_groups(monkeypatch):
            explicit = HybridCPDataLoaderWrapper(
                None, _hybrid_cp_config(), pg_collection=pg_collection
            )
        assert _ranks(fallback.dp_cp_group) == _ranks(explicit.dp_cp_group) == global_dp_cp
        assert _ranks(fallback.dp_group) == _ranks(explicit.dp_group) == global_dp

        fallback = _capture_scheduler_groups(monkeypatch, None)
        with _forbid_global_groups(monkeypatch):
            explicit = _capture_scheduler_groups(monkeypatch, pg_collection)
        assert fallback[:2] == explicit[:2]
        assert [_ranks(g) for g in fallback[2:]] == [_ranks(g) for g in explicit[2:]]
        assert _ranks(explicit[2]) == global_dp and _ranks(explicit[5]) == global_dp_cp
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("gtp", [1, 2])
def test_hybrid_cp_explicit_collection_matches_fallback(monkeypatch, gtp):
    _require_world_size(4, gtp)
    Utils.initialize_model_parallel(1, 1, gtp_remat_size=gtp)
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        dp_group = parallel_state.get_data_parallel_group()

        fallback = _run_hybrid_cp(None, dp_group)
        with _forbid_global_groups(monkeypatch):
            explicit = _run_hybrid_cp(pg_collection, dp_group)
        for tokens, sample_id_groups in (fallback, explicit):
            _check_hybrid_cp_result(tokens, sample_id_groups, dp_cp_group.rank())
        assert explicit[1] == fallback[1]
        assert explicit[0].keys() == fallback[0].keys()
        assert all(torch.equal(explicit[0][gid], fallback[0][gid]) for gid in fallback[0])
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    ("tp", "cp", "gtp"), [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2), (1, 2, 2)]
)
def test_wrap_data_iterator_explicit_collection_matches_fallback(monkeypatch, tp, cp, gtp):
    _require_world_size(4, tp * cp * gtp)
    Utils.initialize_model_parallel(tp, 1, context_parallel_size=cp, gtp_remat_size=gtp)
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        dp_group = parallel_state.get_data_parallel_group()
        is_reader = parallel_state.get_tensor_model_parallel_rank() == 0

        fallback = _run_packing(None, dp_group, dp_cp_group, cp, is_reader)
        with _forbid_global_groups(monkeypatch):
            explicit = _run_packing(pg_collection, dp_group, dp_cp_group, cp, is_reader)
        assert explicit == fallback
    finally:
        Utils.destroy_model_parallel()


def test_wrap_data_iterator_with_pipeline_before_data_parallel_order(monkeypatch):
    """The 'tp-cp-ep-pp-dp' order interleaves the pipeline stages of a DP group."""
    _require_world_size(4, 2)
    Utils.initialize_model_parallel(1, 2, order="tp-cp-ep-pp-dp")
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        dp_cp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        dp_group = parallel_state.get_data_parallel_group()
        assert _ranks(dp_group) != list(range(_ranks(dp_group)[0], _ranks(dp_group)[-1] + 1))

        fallback = _run_packing(None, dp_group, dp_cp_group, 1, True)
        with _forbid_global_groups(monkeypatch):
            explicit = _run_packing(pg_collection, dp_group, dp_cp_group, 1, True)
        assert explicit == fallback
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("entry_point", ["hybrid_cp", "sequence_packing"])
def test_reroute_on_grids_at_rank_offsets(entry_point):
    """Two grids split the world; the upper one starts at a nonzero rank offset."""
    _require_world_size(4, 2)
    Utils.initialize_distributed()
    parallel_state.destroy_model_parallel()
    half = Utils.world_size // 2
    grids = [
        HyperCommGrid([1, 1, half, 1], ["tp", "cp", "dp", "pp"], rank_offset=offset)
        for offset in (0, half)
    ]
    # Group creation is collective: every rank creates every group in the same order.
    grid_groups = [
        {
            "tp": grid.create_pg("tp"),
            "pp": grid.create_pg("pp"),
            "dp": grid.create_pg("dp"),
            "dp_cp": grid.create_pg(["cp", "dp"]),
        }
        for grid in grids
    ]
    try:
        groups = grid_groups[dist.get_rank() // half]
        pg_collection = ProcessGroupCollection(**groups)
        if entry_point == "hybrid_cp":
            tokens, sample_id_groups = _run_hybrid_cp(pg_collection, groups["dp"])
            _check_hybrid_cp_result(tokens, sample_id_groups, groups["dp_cp"].rank())
        else:
            _run_packing(pg_collection, groups["dp"], groups["dp_cp"], 1, True)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        for grid in grids:
            grid.destroy()
