# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tensor-parallel helpers read the global grid only when the caller omits its groups, and warn.

model_parallel_cuda_manual_seed, the tensor-parallel and GTP duplicate filters, and the GTP
checkpoint replica rank take their groups from the caller. When the caller omits them they fall
back to the global parallel grid: the fallback emits a ProcessGroupFallbackWarning and selects
the same ranks as the global accessors.
"""

import warnings

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.tensor_parallel import gtp_api
from megatron.core.tensor_parallel import random as random_module
from megatron.core.tensor_parallel.generalized_tensor_parallelism import gtp_replica_rank
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    param_is_not_gtp_duplicate,
    param_is_not_tensor_parallel_duplicate,
)
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_gtp_remat_rng_tracker_name,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import (
    Utils,
    forbid_global_process_groups,
    new_group_with_same_ranks,
)

VERSIONS = "deprecated since Megatron Core 0.21 and will be removed in 0.23"

# The groups model_parallel_cuda_manual_seed reads from a collection.
SEED_GROUPS = ("tp", "ep", "expt_tp", "gtp_remat", "expt_gtp_remat")

# (tp, ep, etp, gtp, egtp) grids with non-trivial ranks on every seeded axis of 4 ranks.
SEED_GRIDS = [(2, 1, 1, 2, 2), (1, 2, 2, 2, 1)]


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _fallback_messages(record):
    return [str(w.message) for w in record if issubclass(w.category, ProcessGroupFallbackWarning)]


def _initialize_seed_grid(tp, ep, etp, gtp, egtp):
    if (
        Utils.world_size < 4
        or Utils.world_size % (tp * gtp)
        or Utils.world_size % (etp * ep * egtp)
    ):
        pytest.skip("needs a world size of at least 4 that the grid divides")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
        gtp_remat_size=gtp,
        expert_gtp_remat_size=egtp,
    )


def _tracker_states():
    return {name: state.clone() for name, state in get_cuda_rng_tracker().get_states().items()}


def _assert_same_states(actual, expected):
    assert actual.keys() == expected.keys()
    for name, state in expected.items():
        assert torch.equal(actual[name], state), name


def _seed_from_global_accessors(seed):
    """Seed with every rank and world size the global accessors give."""
    model_parallel_cuda_manual_seed(
        seed,
        force_reset_rng=True,
        tp_rank=parallel_state.get_tensor_model_parallel_rank(),
        ep_rank=parallel_state.get_expert_model_parallel_rank(),
        etp_rank=parallel_state.get_expert_tensor_parallel_rank(),
        gtp_remat_rank=parallel_state.get_gtp_weight_remat_rank(),
        egtp_remat_rank=parallel_state.get_expert_gtp_weight_remat_rank(),
        gtp_remat_world_size=parallel_state.get_gtp_weight_remat_world_size(),
        egtp_remat_world_size=parallel_state.get_expert_gtp_weight_remat_world_size(),
    )


def _param(*, expert=False):
    param = torch.nn.Parameter(torch.empty(2))
    if expert:
        # allreduce=False marks a parameter that uses the expert topology.
        param.allreduce = False
    return param


class TestModelParallelCudaManualSeed:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("grid", SEED_GRIDS)
    def test_given_groups_read_no_global_group(self, grid):
        _initialize_seed_grid(*grid)
        _seed_from_global_accessors(1234)
        expected = _tracker_states()
        mpu_groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=list(SEED_GROUPS))
        own_groups = {
            name: new_group_with_same_ranks(vars(mpu_groups)[name]) for name in SEED_GROUPS
        }
        pg_collection = ProcessGroupCollection(**own_groups)
        try:
            with forbid_global_process_groups(), warnings.catch_warnings():
                warnings.simplefilter("error", ProcessGroupFallbackWarning)
                model_parallel_cuda_manual_seed(
                    1234, force_reset_rng=True, pg_collection=pg_collection
                )
            _assert_same_states(_tracker_states(), expected)
        finally:
            for group in own_groups.values():
                torch.distributed.destroy_process_group(group)

    @pytest.mark.parametrize("grid", SEED_GRIDS)
    def test_omitted_groups_warn_and_seed_from_the_global_grid(self, grid):
        _initialize_seed_grid(*grid)
        _seed_from_global_accessors(1234)
        expected = _tracker_states()
        assert get_gtp_remat_rng_tracker_name() in expected

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(2):
                model_parallel_cuda_manual_seed(1234, force_reset_rng=True)

        messages = _fallback_messages(record)
        assert len(messages) == 1
        assert "model_parallel_cuda_manual_seed was called without `pg_collection`" in messages[0]
        assert VERSIONS in messages[0]
        _assert_same_states(_tracker_states(), expected)

    def test_omitted_gtp_axes_are_off_without_a_global_grid(self, monkeypatch):
        """A caller that passes its TP and expert ranks but builds no global grid still seeds."""
        Utils.initialize_model_parallel()
        Utils.destroy_model_parallel()
        assert not parallel_state.model_parallel_is_initialized()

        def resolve_process_groups(*args, **kwargs):
            raise AssertionError("resolved the omitted GTP_remat groups as required groups")

        # The GTP_remat axes are off without a grid; they are not resolved as groups the caller
        # must have.
        monkeypatch.setattr(random_module, "resolve_process_groups", resolve_process_groups)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            model_parallel_cuda_manual_seed(
                1234, force_reset_rng=True, tp_rank=0, ep_rank=0, etp_rank=0
            )

        messages = _fallback_messages(record)
        assert len(messages) == 1
        assert "model_parallel_cuda_manual_seed was called without `pg_collection`" in messages[0]
        states = _tracker_states()
        assert get_gtp_remat_rng_tracker_name() not in states
        assert get_gtp_remat_rng_tracker_name(is_expert=True) not in states


@pytest.mark.skipif(Utils.world_size < 2, reason="needs at least 2 ranks for TP=2")
class TestTensorParallelDuplicateFilter:
    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_given_group_reads_no_global_group(self):
        group = new_group_with_same_ranks(parallel_state.get_tensor_model_parallel_group())
        try:
            with forbid_global_process_groups(), warnings.catch_warnings():
                warnings.simplefilter("error", ProcessGroupFallbackWarning)
                for param in (_param(), _param(expert=True)):
                    assert param_is_not_tensor_parallel_duplicate(param, tp_group=group) == (
                        group.rank() == 0
                    )
        finally:
            torch.distributed.destroy_process_group(group)

    def test_omitted_group_warns_and_uses_the_global_rank(self):
        expected = parallel_state.get_tensor_model_parallel_rank() == 0
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(2):
                assert param_is_not_tensor_parallel_duplicate(_param()) == expected
                # Without either group, an expert parameter also falls back to the TP group.
                assert param_is_not_tensor_parallel_duplicate(_param(expert=True)) == expected

        messages = _fallback_messages(record)
        assert len(messages) == 2
        owner = "param_is_not_tensor_parallel_duplicate was called without"
        assert f"{owner} `tp_group`" in messages[0]
        assert f"{owner} `expert_tp_group`" in messages[1]
        assert all(VERSIONS in message for message in messages)


@pytest.mark.skipif(
    Utils.world_size < 2 or Utils.world_size % 2, reason="needs an even number of ranks"
)
class TestGTPDuplicateFilter:
    def setup_method(self, method):
        Utils.initialize_model_parallel(gtp_remat_size=2, expert_gtp_remat_size=2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_given_groups_read_no_global_group(self):
        dense_group = new_group_with_same_ranks(parallel_state.get_gtp_weight_remat_group())
        expert_group = new_group_with_same_ranks(parallel_state.get_expert_gtp_weight_remat_group())
        groups = {"gtp_group": dense_group, "expert_gtp_group": expert_group}
        try:
            with forbid_global_process_groups(), warnings.catch_warnings():
                warnings.simplefilter("error", ProcessGroupFallbackWarning)
                assert param_is_not_gtp_duplicate(_param(), **groups) == (dense_group.rank() == 0)
                assert param_is_not_gtp_duplicate(_param(expert=True), **groups) == (
                    expert_group.rank() == 0
                )
                # Groups taken from a collection: None is a GTP_remat axis that is off.
                assert param_is_not_gtp_duplicate(_param(), use_global_fallback=False)
        finally:
            torch.distributed.destroy_process_group(dense_group)
            torch.distributed.destroy_process_group(expert_group)

    def test_omitted_groups_warn_and_use_the_global_ranks(self):
        expected_dense = parallel_state.get_gtp_weight_remat_rank() == 0
        expected_expert = parallel_state.get_expert_gtp_weight_remat_rank() == 0
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(2):
                assert param_is_not_gtp_duplicate(_param()) == expected_dense
                assert param_is_not_gtp_duplicate(_param(expert=True)) == expected_expert

        messages = _fallback_messages(record)
        assert len(messages) == 2
        assert "param_is_not_gtp_duplicate was called without `gtp_group`" in messages[0]
        assert "param_is_not_gtp_duplicate was called without `expert_gtp_group`" in messages[1]
        assert all(VERSIONS in message for message in messages)

    def test_omitted_groups_keep_every_param_without_a_global_grid(self):
        Utils.destroy_model_parallel()
        assert not parallel_state.model_parallel_is_initialized()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert param_is_not_gtp_duplicate(_param())
            assert param_is_not_gtp_duplicate(_param(expert=True))
        assert len(_fallback_messages(record)) == 2


@pytest.mark.skipif(
    Utils.world_size < 4 or Utils.world_size % 4, reason="needs a multiple of 4 ranks"
)
class TestGTPReplicaRank:
    def setup_method(self, method):
        Utils.initialize_model_parallel(gtp_remat_size=2, expert_gtp_remat_size=2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_given_or_stamped_group_reads_no_global_group(self):
        replica_group = new_group_with_same_ranks(
            parallel_state.get_data_parallel_group(with_context_parallel=True, with_gtp_remat=False)
        )
        stamped = torch.empty(1)
        stamped.gtp_replica_group = replica_group
        try:
            with forbid_global_process_groups(), warnings.catch_warnings():
                warnings.simplefilter("error", ProcessGroupFallbackWarning)
                assert gtp_replica_rank(stamped) == replica_group.rank()
                assert gtp_replica_rank(torch.empty(1), replica_group) == replica_group.rank()
        finally:
            torch.distributed.destroy_process_group(replica_group)

    def test_unstamped_param_warns_and_uses_the_global_replica_rank(self):
        expected_dense = parallel_state.get_data_parallel_rank(
            with_context_parallel=True, with_gtp_remat=False
        )
        expected_expert = parallel_state.get_expert_data_parallel_rank(with_gtp_remat=False)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(2):
                assert gtp_replica_rank(_param()) == expected_dense
                assert gtp_replica_rank(_param(expert=True)) == expected_expert

        messages = _fallback_messages(record)
        assert len(messages) == 1
        assert "gtp_replica_rank was called without `explicit_group`" in messages[0]
        assert VERSIONS in messages[0]

    def test_unstamped_param_without_a_global_grid_raises(self):
        Utils.destroy_model_parallel()
        with pytest.raises(RuntimeError, match="gtp_replica_group"):
            gtp_replica_rank(_param())


@pytest.mark.skipif(
    Utils.world_size < 2 or Utils.world_size % 2, reason="needs an even number of ranks"
)
class TestGTPReplicaGroupStamping:
    """GTP-remat layers stamp their collection's replica group on their sharded weights."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("layer", ["embedding", "column", "row", "expert_column"])
    @pytest.mark.parametrize("sets_replica_group", [True, False], ids=["set", "unset"])
    def test_layers_stamp_the_replica_group_of_their_collection(
        self, monkeypatch, layer, sets_replica_group
    ):
        Utils.initialize_model_parallel(gtp_remat_size=2, expert_gtp_remat_size=2)
        is_expert = layer == "expert_column"
        mpu_groups = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "expt_tp", "gtp_remat", "expt_gtp_remat", "dp_cp", "expt_dp"]
        )
        replica_field = "expt_dp" if is_expert else "dp_cp"
        replica_group = new_group_with_same_ranks(vars(mpu_groups)[replica_field])
        groups = {
            name: vars(mpu_groups)[name]
            for name in ("tp", "expt_tp", "gtp_remat", "expt_gtp_remat")
        }
        if sets_replica_group:
            groups[replica_field] = replica_group
        pg_collection = ProcessGroupCollection(**groups)
        tp_group = pg_collection.expt_tp if is_expert else pg_collection.tp

        wrapped = []

        def wrap_module_params_gtp(module, weight_names, gtp_remat_group, replica_group=None):
            wrapped.append((gtp_remat_group, replica_group))

        # The layers import the GTP wrapper when they shard a weight; record what they pass it.
        monkeypatch.setattr(
            gtp_api, "wrap_module_params_gtp", wrap_module_params_gtp, raising=False
        )
        config = TransformerConfig(
            num_layers=1, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        common = dict(
            config=config,
            init_method=torch.nn.init.zeros_,
            tp_group=tp_group,
            pg_collection=pg_collection,
        )
        try:
            with forbid_global_process_groups():
                if layer == "embedding":
                    VocabParallelEmbedding(16, 64, **common)
                elif layer == "row":
                    RowParallelLinear(
                        64, 64, bias=False, input_is_parallel=True, skip_bias_add=False, **common
                    )
                else:
                    ColumnParallelLinear(64, 64, bias=False, is_expert=is_expert, **common)
        finally:
            torch.distributed.destroy_process_group(replica_group)

        gtp_remat_group = pg_collection.expt_gtp_remat if is_expert else pg_collection.gtp_remat
        expected_replica_group = replica_group if sets_replica_group else None
        assert wrapped == [(gtp_remat_group, expected_replica_group)]
