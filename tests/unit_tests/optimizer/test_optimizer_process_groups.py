# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The optimizers take their process groups from the model's collection, and warn when a caller
omits it and they fall back to the global process groups."""

import contextlib
import warnings
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import (
    Float16OptimizerWithFloat16Params,
    _backfill_gtp_sharded_param_map,
)
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.tensor_parallel import gtp_api
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, forbid_global_process_groups


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


@pytest.fixture
def model_parallel():
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


@contextlib.contextmanager
def _without_global_groups():
    """Fail on any read of the global grid and on any process-group fallback warning."""
    with forbid_global_process_groups(), warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        yield


def _ddp_linear():
    """A bf16 linear layer under DDP, for the non-distributed optimizer."""
    model = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16, device="cuda")
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    return DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )


def _optimizer_config():
    return OptimizerConfig(optimizer="adam", lr=0.01, bf16=True, use_distributed_optimizer=False)


class TestGetMegatronOptimizer:

    def test_builds_on_the_collection_it_is_given(self, model_parallel):
        model = _ddp_linear()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        with _without_global_groups():
            optimizer = get_megatron_optimizer(
                _optimizer_config(),
                [model],
                use_gloo_process_groups=False,
                pg_collection=pg_collection,
            )

        (dense_optimizer,) = optimizer.chained_optimizers
        assert isinstance(dense_optimizer, Float16OptimizerWithFloat16Params)
        assert dense_optimizer.pg_collection is pg_collection

    def test_without_a_collection_warns_and_uses_the_global_groups(self, model_parallel):
        model = _ddp_linear()

        with pytest.warns(
            ProcessGroupFallbackWarning,
            match="get_megatron_optimizer was called without `pg_collection`",
        ):
            optimizer = get_megatron_optimizer(
                _optimizer_config(), [model], use_gloo_process_groups=False
            )

        (dense_optimizer,) = optimizer.chained_optimizers
        # The fallback sets every field to the group use_mpu_process_groups() returns.
        expected = ProcessGroupCollection.use_mpu_process_groups()
        assert vars(dense_optimizer.pg_collection) == vars(expected)
        assert dense_optimizer.tp_group is parallel_state.get_tensor_model_parallel_group()
        assert dense_optimizer.grad_stats_parallel_group is (
            parallel_state.get_model_parallel_group()
        )


class TestLayerWiseDistributedOptimizer:

    def test_wraps_its_optimizers_with_its_collection(self, model_parallel):
        model = _ddp_linear()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        optimizer = LayerWiseDistributedOptimizer(
            [torch.optim.Adam(model.parameters(), lr=0.01)],
            _optimizer_config(),
            pg_collection,
            model_chunks=[model],
        )

        (wrapped_optimizer,) = optimizer.chained_optimizers
        assert isinstance(wrapped_optimizer, Float16OptimizerWithFloat16Params)
        assert wrapped_optimizer.pg_collection is pg_collection


@pytest.fixture
def recorded_rebuilds(monkeypatch):
    """Record the groups each GTP shard is rebuilt with, instead of building its entry."""
    rebuilds = []

    def rebuild(state_dict, prefix, tensor_parallel_layers_axis_map=None, *, tp_group, dp_cp_group):
        rebuilds.append((tp_group, dp_cp_group))
        return {key: SimpleNamespace(key=key) for key in state_dict}

    # Set rather than patch: the names exist only when the GTP implementation imports.
    monkeypatch.setattr(
        gtp_api,
        "is_gtp_param",
        lambda param: getattr(param, "is_gtp_weight_remat", False),
        raising=False,
    )
    monkeypatch.setattr(
        gtp_api, "make_sharded_tensors_for_checkpoint_with_gtp_remat", rebuild, raising=False
    )
    return rebuilds


def _gtp_shard():
    """A parameter that reads as a GTP_remat shard."""
    param = torch.nn.Parameter(torch.zeros(4, 4))
    param.is_gtp_weight_remat = True
    param._debug_name = "decoder.layers.0.mixer.in_proj.weight"
    return param


class TestGtpShardCheckpointEntries:

    @pytest.mark.parametrize(
        "dp_cp_fields, expected_dp_cp_field",
        [
            ({"dp_cp": "group"}, "dp_cp"),
            ({"dp_cp": "group", "dp_cp_gtp_remat": None}, "dp_cp"),
            ({"dp_cp": "group", "dp_cp_gtp_remat": "group"}, "dp_cp_gtp_remat"),
            ({"dp_cp_gtp_remat": "group"}, "dp_cp_gtp_remat"),
        ],
    )
    def test_rebuilds_a_shard_with_the_collection_groups(
        self, recorded_rebuilds, dp_cp_fields, expected_dp_cp_field
    ):
        # Stand-ins: the rebuild is recorded, so only the identity of each group matters.
        groups = {"tp": object()}
        groups.update(
            {name: None if value is None else object() for name, value in dp_cp_fields.items()}
        )
        id_to_sharded_param_map = {}

        with _without_global_groups():
            _backfill_gtp_sharded_param_map(
                id_to_sharded_param_map,
                [[_gtp_shard()]],
                pg_collection=ProcessGroupCollection(**groups),
            )

        assert list(id_to_sharded_param_map) == [0]
        ((rebuilt_tp_group, rebuilt_dp_cp_group),) = recorded_rebuilds
        assert rebuilt_tp_group is groups["tp"]
        # The model's own entries use the DP x CP group with the GTP-remat peers when the
        # collection sets one, and dp_cp otherwise.
        assert rebuilt_dp_cp_group is groups[expected_dp_cp_field]

    def test_without_a_collection_warns_and_uses_the_global_groups(
        self, model_parallel, recorded_rebuilds
    ):
        id_to_sharded_param_map = {}

        with pytest.warns(
            ProcessGroupFallbackWarning,
            match="Float16OptimizerWithFloat16Params was called without `pg_collection`",
        ):
            _backfill_gtp_sharded_param_map(id_to_sharded_param_map, [[_gtp_shard()]])

        assert list(id_to_sharded_param_map) == [0]
        ((tp_group, dp_cp_group),) = recorded_rebuilds
        assert tp_group is parallel_state.get_tensor_model_parallel_group()
        assert dp_cp_group is parallel_state.get_data_parallel_group(with_context_parallel=True)

    def test_a_shard_with_a_model_entry_needs_no_groups(self, recorded_rebuilds):
        shard = _gtp_shard()
        # A native-FP8 shard's model entry holds a dequantized copy that links back to it.
        entry = SimpleNamespace(data=SimpleNamespace(_gtp_dequant_src=shard), key="weight")
        id_to_sharded_param_map = {}

        with _without_global_groups():
            _backfill_gtp_sharded_param_map(id_to_sharded_param_map, [[shard]], {"weight": entry})

        assert id_to_sharded_param_map == {0: entry}
        assert recorded_rebuilds == []

    @pytest.mark.parametrize(
        "dp_cp_fields, message",
        [
            ({}, "requires pg_collection to set dp_cp"),
            ({"dp_cp_gtp_remat": None}, "requires pg_collection to set dp_cp"),
            ({"dp_cp": None}, "sets dp_cp to None and has no dp_cp_gtp_remat"),
        ],
    )
    def test_rejects_a_collection_without_a_dp_cp_group(
        self, recorded_rebuilds, dp_cp_fields, message
    ):
        pg_collection = ProcessGroupCollection(tp=object(), **dp_cp_fields)

        with pytest.raises(ValueError, match=message):
            _backfill_gtp_sharded_param_map({}, [[_gtp_shard()]], pg_collection=pg_collection)
        assert recorded_rebuilds == []
