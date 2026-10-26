# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Optimizer fallbacks to WORLD or to the global grid warn instead of staying silent.

``torch.distributed`` reads a ``None`` grad-stats group as WORLD, so an optimizer without one
reduces its gradient norm, zero count and found-inf flag over every rank in the job. The
fallback keeps working during the deprecation period: a caller such as Megatron-Bridge, which
sets ``intra_dist_opt=None`` for a single optimizer instance, keeps training, and is told that
the fallback goes away.
"""

import os
import warnings
from unittest import mock

import pytest
import torch
from packaging.version import Version

from megatron.core import parallel_state, process_groups_config
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer import optimizer as optimizer_module
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import FP32Optimizer
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def fresh_warning_registries(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())
    monkeypatch.setattr(optimizer_module, "_warned_world_grad_stats_owners", set(), raising=False)


def _stub_optimizer(cls, **attributes):
    """An instance of ``cls`` that carries only ``attributes``, without running __init__."""
    optimizer = object.__new__(cls)
    for name, value in attributes.items():
        setattr(optimizer, name, value)
    return optimizer


def _fallback_messages(record):
    return [str(w.message) for w in record if issubclass(w.category, ProcessGroupFallbackWarning)]


class TestGradStatsGroupGetters:
    """The getters return the group the reductions use, and warn when it is WORLD or global."""

    @pytest.mark.parametrize(
        "cls,attributes",
        [
            (DistributedOptimizer, {}),
            (DistributedOptimizer, {"grad_stats_parallel_group": None}),
            (FP32Optimizer, {"grad_stats_parallel_group": None}),
        ],
        ids=["distributed-unset", "distributed-none", "non-distributed-none"],
    )
    def test_missing_group_reduces_over_world_with_a_warning(self, cls, attributes):
        optimizer = _stub_optimizer(cls, **attributes)
        with pytest.warns(ProcessGroupFallbackWarning) as record:
            assert optimizer.get_grad_stats_parallel_group() is None
        (message,) = _fallback_messages(record)
        assert message.startswith(f"{cls.__name__} has no grad_stats_parallel_group")
        assert "over the default (WORLD) process group" in message
        assert "deprecated since Megatron Core 0.21 and will be removed in 0.23" in message

    @pytest.mark.parametrize("cls", [DistributedOptimizer, FP32Optimizer])
    def test_set_group_is_returned_without_a_warning(self, cls):
        group = object()
        optimizer = _stub_optimizer(cls, grad_stats_parallel_group=group)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            assert optimizer.get_grad_stats_parallel_group() is group

    def test_unset_group_of_a_non_distributed_optimizer_reads_the_global_grid_with_a_warning(self):
        global_model_parallel_group = object()
        optimizer = _stub_optimizer(FP32Optimizer)
        with mock.patch.object(
            parallel_state, "get_model_parallel_group", return_value=global_model_parallel_group
        ):
            with pytest.warns(ProcessGroupFallbackWarning) as record:
                group = optimizer.get_grad_stats_parallel_group()
        assert group is global_model_parallel_group
        (message,) = _fallback_messages(record)
        assert message.startswith("FP32Optimizer was called without `grad_stats_parallel_group`")
        assert "deprecated since Megatron Core 0.21 and will be removed in 0.23" in message

    def test_world_fallback_warns_once_per_owner_and_raises_under_an_error_filter(self):
        optimizer = _stub_optimizer(DistributedOptimizer, grad_stats_parallel_group=None)
        # A warning that a filter turns into an error is not recorded, so strict mode raises
        # on every call, not only the first.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            for _ in range(2):
                with pytest.raises(ProcessGroupFallbackWarning):
                    optimizer.get_grad_stats_parallel_group()
        with pytest.warns(ProcessGroupFallbackWarning):
            optimizer.get_grad_stats_parallel_group()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            assert optimizer.get_grad_stats_parallel_group() is None
        # Another owner still hears about its own fallback.
        with pytest.warns(ProcessGroupFallbackWarning, match="^FP32Optimizer has no"):
            _stub_optimizer(
                FP32Optimizer, grad_stats_parallel_group=None
            ).get_grad_stats_parallel_group()


class TestDistributedOptimizerWithoutInstanceGroup:
    """A collection with ``intra_dist_opt=None`` keeps training, with a deprecation warning."""

    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_step_reduces_over_world_and_warns(self):
        model = torch.nn.Linear(16, 16, device='cuda')
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            DistributedDataParallelConfig(use_distributed_optimizer=True),
            model,
        )
        # Megatron-Bridge builds such a collection for a single optimizer instance.
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.intra_dist_opt = None
        optimizer = get_megatron_optimizer(
            OptimizerConfig(optimizer='adam', lr=1e-3, use_distributed_optimizer=True),
            [model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )
        model(torch.randn(4, 16, device='cuda')).sum().backward()
        model.finish_grad_sync()

        with pytest.warns(
            ProcessGroupFallbackWarning, match="^DistributedOptimizer has no grad_stats_parallel"
        ):
            assert not optimizer.prepare_grads()
            world_norm = float(optimizer.get_grad_norm())
        assert world_norm > 0.0

        # With one optimizer instance on the standard grid, WORLD holds the instance's ranks.
        (distributed_optimizer,) = optimizer.chained_optimizers
        distributed_optimizer.grad_stats_parallel_group = (
            parallel_state.get_intra_distributed_optimizer_instance_group()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            assert float(optimizer.get_grad_norm()) == pytest.approx(world_norm, rel=1e-6)
        assert optimizer.step_with_ready_grads()


@pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip layer-wise optimizer for LTS test",
)
class TestLayerWiseOptimizerWithoutInstanceGroup:
    """The layer-wise optimizer reads its grad-stats group directly, so it warns when built."""

    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_construction_warns_that_statistics_are_reduced_over_world(self):
        model = torch.nn.Linear(10, 5).bfloat16().cuda()
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1),
            DistributedDataParallelConfig(use_distributed_optimizer=False),
            model,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()
        pg_collection.intra_dist_opt = None
        config = OptimizerConfig(
            optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=False
        )

        with pytest.warns(
            ProcessGroupFallbackWarning,
            match="^LayerWiseDistributedOptimizer has no grad_stats_parallel_group",
        ):
            optimizer = LayerWiseDistributedOptimizer(
                [torch.optim.Adam(model.parameters(), lr=config.lr)], config, pg_collection
            )
        assert optimizer.grad_stats_parallel_group is None
