# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The training loop selects its data-parallel groups from a collection with one helper.

The helper returns the data-parallel group that spans the GTP weight-remat axis, the group that
parallel_state.get_data_parallel_group() returns by default. The CPU tests use stub groups; the
distributed tests need 4 ranks.
"""

import warnings
from contextlib import contextmanager
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.training import training
from tests.unit_tests.test_utilities import Utils


def _raise_global_read(*args, **kwargs):
    raise AssertionError("read the global parallel state")


class _Group:
    def __init__(self, size=2, rank=0):
        self._size, self._rank = size, rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


@pytest.fixture(autouse=True)
def fresh_fallback_record(monkeypatch):
    """Forget which fallbacks warned, so that each test sees its own warning."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


@contextmanager
def _record_fallbacks():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", ProcessGroupFallbackWarning)
        yield record


def _owners(record):
    """The functions named by the fallback warnings, in order."""
    return [
        str(w.message).split()[0]
        for w in record
        if issubclass(w.category, ProcessGroupFallbackWarning)
    ]


_FIELDS = {True: ("dp_cp_gtp_remat", "dp_cp"), False: ("dp_gtp_remat", "dp")}


@pytest.mark.parametrize("with_context_parallel", [True, False])
class TestDataParallelGroup:

    def test_uses_the_gtp_inclusive_group(self, with_context_parallel):
        name, replicate_name = _FIELDS[with_context_parallel]
        pg_collection = ProcessGroupCollection(**{name: _Group(), replicate_name: _Group()})

        with _record_fallbacks() as record:
            group = training._get_data_parallel_group(
                pg_collection, with_context_parallel=with_context_parallel, owner="owner"
            )

        assert group is vars(pg_collection)[name]
        assert _owners(record) == []

    @pytest.mark.parametrize("gtp_remat", [None, _Group(size=1)], ids=["none", "one_rank"])
    def test_without_gtp_remat_the_replicate_group_serves(
        self, monkeypatch, with_context_parallel, gtp_remat
    ):
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        _, replicate_name = _FIELDS[with_context_parallel]
        pg_collection = ProcessGroupCollection(**{replicate_name: _Group()}, gtp_remat=gtp_remat)

        with _record_fallbacks() as record:
            group = training._get_data_parallel_group(
                pg_collection, with_context_parallel=with_context_parallel, owner="owner"
            )

        assert group is vars(pg_collection)[replicate_name]
        assert _owners(record) == []

    def test_without_gtp_remat_fields_the_replicate_group_serves_with_a_warning(
        self, with_context_parallel
    ):
        # A collection that predates the GTP weight-remat fields keeps the replicate group.
        name, replicate_name = _FIELDS[with_context_parallel]
        pg_collection = ProcessGroupCollection(**{replicate_name: _Group()})

        with _record_fallbacks() as record:
            group = training._get_data_parallel_group(
                pg_collection, with_context_parallel=with_context_parallel, owner="owner"
            )

        assert group is vars(pg_collection)[replicate_name]
        assert _owners(record) == ["owner"]
        assert f"`pg_collection.{name}`" in str(record[0].message)

    def test_active_gtp_remat_requires_the_gtp_inclusive_group(
        self, monkeypatch, with_context_parallel
    ):
        # The replicate group holds only a part of the ranks that read distinct data.
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        name, replicate_name = _FIELDS[with_context_parallel]
        pg_collection = ProcessGroupCollection(
            **{replicate_name: _Group()}, gtp_remat=_Group(size=2)
        )

        with pytest.raises(ValueError, match=f"owner requires pg_collection to set {name}"):
            training._get_data_parallel_group(
                pg_collection, with_context_parallel=with_context_parallel, owner="owner"
            )

    @pytest.mark.parametrize("gtp_remat", [{}, {"gtp_remat": None}], ids=["unset", "none"])
    def test_a_collection_without_either_group_is_rejected(self, with_context_parallel, gtp_remat):
        name, replicate_name = _FIELDS[with_context_parallel]
        pg_collection = ProcessGroupCollection(**{replicate_name: None}, **gtp_remat)

        with pytest.raises(
            ValueError, match=f"owner requires pg_collection to set {name}, or {replicate_name}"
        ):
            training._get_data_parallel_group(
                pg_collection, with_context_parallel=with_context_parallel, owner="owner"
            )

    def test_omitted_collection_warns_and_uses_the_global_group(
        self, monkeypatch, with_context_parallel
    ):
        global_group = _Group()
        get_group = mock.Mock(return_value=global_group)
        monkeypatch.setattr(parallel_state, "get_data_parallel_group", get_group)

        with _record_fallbacks() as record:
            group = training._get_data_parallel_group(
                None, with_context_parallel=with_context_parallel, owner="owner"
            )

        assert group is global_group
        get_group.assert_called_once_with(with_context_parallel=with_context_parallel)
        assert _owners(record) == ["owner"]


def test_reduce_sum_reduces_over_the_collection_group():
    """The full-dataset probe sums over the data-parallel group that the helper selects."""
    pg_collection = ProcessGroupCollection(dp_cp=_Group(), gtp_remat=None)

    with (
        mock.patch.object(torch.distributed, "all_reduce") as all_reduce,
        mock.patch.object(torch.cuda, "current_device", return_value="cpu"),
    ):
        training._reduce_sum_across_data_parallel_group(
            3.0, with_context_parallel=True, pg_collection=pg_collection
        )

    assert all_reduce.call_args.kwargs["group"] is pg_collection.dp_cp
    with pytest.raises(ValueError, match="requires pg_collection to set dp_cp_gtp_remat"):
        training._reduce_sum_across_data_parallel_group(
            3.0, with_context_parallel=True, pg_collection=ProcessGroupCollection(gtp_remat=None)
        )


_needs_4_ranks = pytest.mark.skipif(
    Utils.world_size < 4 or Utils.world_size % 4 != 0,
    reason="Requires a world size that is a multiple of 4",
)


@_needs_4_ranks
class TestDataParallelGroupOnAGrid:

    @pytest.fixture(autouse=True)
    def grid(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("with_context_parallel", [True, False])
    def test_standard_collection_selects_the_global_groups(self, with_context_parallel):
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        global_group = parallel_state.get_data_parallel_group(
            with_context_parallel=with_context_parallel
        )

        with _record_fallbacks() as record:
            group = training._get_data_parallel_group(
                pg_collection, with_context_parallel=with_context_parallel, owner="owner"
            )
        assert _owners(record) == []
        assert group is global_group

        with _record_fallbacks() as record:
            fallback_group = training._get_data_parallel_group(
                None, with_context_parallel=with_context_parallel, owner="owner"
            )
        assert _owners(record) == ["owner"]
        assert fallback_group is global_group

    @pytest.mark.parametrize("gtp_inclusive", [True, False])
    def test_sums_over_the_collection_group_without_global_reads(self, monkeypatch, gtp_inclusive):
        # Adjacent ranks, unlike the global DP group of TP=2, whose ranks are two apart.
        pair_group, _ = torch.distributed.new_subgroups_by_enumeration(
            [[rank, rank + 1] for rank in range(0, Utils.world_size, 2)]
        )
        if gtp_inclusive:
            pg_collection = ProcessGroupCollection(dp_cp_gtp_remat=pair_group)
        else:
            pg_collection = ProcessGroupCollection(dp_cp=pair_group, gtp_remat=None)
        monkeypatch.setattr(parallel_state, "get_data_parallel_group", _raise_global_read)
        rank = torch.distributed.get_rank()

        total = training._reduce_sum_across_data_parallel_group(
            float(rank + 1), with_context_parallel=True, pg_collection=pg_collection
        )

        first_rank_of_pair = rank - rank % 2
        assert total == (first_rank_of_pair + 1) + (first_rank_of_pair + 2)
