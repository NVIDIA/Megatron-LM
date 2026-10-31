# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The training loop hands its process-group collection to the Megatron Core entry points.

A group that the training loop reads from the global grid instead of the collection is invisible
when the collection holds the global groups themselves. These tests therefore give the
collection distinct communicators over the same ranks, or a layout that differs from the global
grid, so that a global read changes which group is used.
"""

import importlib.util
import pathlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_no_pipelining,
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.training import global_vars, training
from megatron.training.arguments import parse_args
from megatron.training.models.dist_utils import _ddp_wrap
from tests.unit_tests.dist_checkpointing.utils import init_basic_mock_args
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    Utils.world_size < 4 or Utils.world_size % 2 != 0, reason="Requires an even world of 4+ ranks"
)

_CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_process_group_usage",
    pathlib.Path(__file__).resolve().parents[3] / "tools" / "check_process_group_usage.py",
)
_checker = importlib.util.module_from_spec(_CHECKER_SPEC)
_CHECKER_SPEC.loader.exec_module(_checker)


class _TinyModel(MegatronModule):
    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(16, 16)


def _config():
    return TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=1, bf16=True)


def _same_ranks(group):
    """A new communicator over the ranks of ``group``."""
    return torch.distributed.new_group(ranks=torch.distributed.get_process_group_ranks(group))


def _ranks(group):
    return torch.distributed.get_process_group_ranks(group)


def _forbid_global_grid(monkeypatch):
    """Make the parallel_state accessors the usage checker counts, and the collection shim built
    on them, raise when called."""

    def read_global_grid(*args, **kwargs):
        raise AssertionError("read a process group of the global parallel grid")

    for name, value in list(vars(parallel_state).items()):
        if callable(value) and _checker._is_deprecated_accessor(name):
            monkeypatch.setattr(parallel_state, name, read_global_grid)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", read_global_grid)


class _DenseAndExpertLinear(torch.nn.Module):
    """One dense and one expert-parallel weight, so the optimizer builds both of its groups."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = torch.nn.Linear(16, 16, bias=False, device="cuda")
        self.experts = torch.nn.Linear(16, 16, bias=False, device="cuda")
        self.experts.weight.allreduce = False


class _OptimizerBuilt(Exception):
    """Stops setup_model_and_optimizer once it has built the optimizer."""


class TestTrainingPgCollection:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.usefixtures("run_config")
    @pytest.mark.parametrize("create_all_gather_group", [False, True])
    def test_collection_carries_all_gather_groups_when_requested(self, create_all_gather_group):
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        args = SimpleNamespace(
            create_all_gather_group=create_all_gather_group, expert_model_parallel_size=2
        )

        pg_collection = training._build_pg_collection_from_parallel_state(args)

        if not create_all_gather_group:
            assert pg_collection.dp_cp_ag is None and pg_collection.expt_dp_ag is None
            return
        # Separate communicators over the data-parallel ranks that Megatron-FSDP gathers over.
        assert pg_collection.dp_cp_ag is not pg_collection.dp_cp
        assert _ranks(pg_collection.dp_cp_ag) == _ranks(pg_collection.dp_cp)
        assert pg_collection.expt_dp_ag is not pg_collection.expt_dp
        assert _ranks(pg_collection.expt_dp_ag) == _ranks(pg_collection.expt_dp)

    @pytest.mark.usefixtures("run_config")
    @pytest.mark.parametrize("mismatched", ["dense", "expert"])
    def test_all_gather_groups_must_span_the_data_parallel_ranks(self, monkeypatch, mismatched):
        Utils.initialize_model_parallel(expert_model_parallel_size=2)
        grid = ProcessGroupCollection.use_mpu_process_groups()
        # The expert-parallel ranks stand in for an all-gather group laid out in another rank
        # order: they differ from both the dense and the expert data-parallel ranks.
        if mismatched == "dense":
            all_gather_groups = (grid.ep, grid.expt_dp)
        else:
            all_gather_groups = (grid.dp_cp, grid.ep)
        monkeypatch.setattr(
            training, "create_all_gather_groups", lambda **kwargs: all_gather_groups
        )
        args = SimpleNamespace(create_all_gather_group=True, expert_model_parallel_size=2)

        with pytest.raises(ValueError, match="all-gather groups whose ranks differ"):
            training._build_pg_collection_from_parallel_state(args)

    @pytest.mark.usefixtures("run_config")
    @pytest.mark.parametrize("use_gloo_process_groups", [True, False])
    def test_collection_carries_the_grid_gloo_groups(self, use_gloo_process_groups):
        Utils.initialize_model_parallel(create_gloo_process_groups=use_gloo_process_groups)
        args = SimpleNamespace(create_all_gather_group=False, expert_model_parallel_size=1)

        pg_collection = training._build_pg_collection_from_parallel_state(args)

        if not use_gloo_process_groups:
            assert pg_collection.intra_dp_cp_gloo is None
            assert pg_collection.intra_expt_dp_gloo is None
            return
        # The Gloo mirrors of the groups the distributed optimizer shards over.
        assert pg_collection.intra_dp_cp_gloo is not None
        assert pg_collection.intra_dp_cp_gloo is parallel_state.get_data_parallel_group_gloo(
            with_context_parallel=True, partial_data_parallel=True
        )
        assert pg_collection.intra_expt_dp_gloo is not None
        assert pg_collection.intra_expt_dp_gloo is (
            parallel_state.get_expert_data_parallel_group_gloo(partial_expert_data_parallel=True)
        )

    @pytest.mark.parametrize("use_gloo_process_groups", [True, False])
    def test_setup_builds_the_optimizer_on_the_collection(
        self, monkeypatch, run_config, use_gloo_process_groups
    ):
        Utils.initialize_model_parallel(create_gloo_process_groups=use_gloo_process_groups)
        run_config.dist.use_gloo_process_groups = use_gloo_process_groups
        # The collection pretrain() builds and hands to setup_model_and_optimizer().
        pg_collection = training._build_pg_collection_from_parallel_state(
            SimpleNamespace(create_all_gather_group=False, expert_model_parallel_size=1)
        )
        if use_gloo_process_groups:
            # Distinct Gloo communicators over the data-parallel ranks (every rank on this grid),
            # so the optimizer holds them only if it takes its Gloo groups from the collection.
            pg_collection.intra_dp_cp_gloo = torch.distributed.new_group(
                ranks=_ranks(pg_collection.dp_cp), backend="gloo"
            )
            pg_collection.intra_expt_dp_gloo = torch.distributed.new_group(
                ranks=_ranks(pg_collection.expt_dp), backend="gloo"
            )
        config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=1)
        model = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(use_distributed_optimizer=True),
            _DenseAndExpertLinear(config),
            pg_collection=pg_collection,
        )
        get_model = Mock(return_value=[model])
        build_optimizer = training.get_megatron_optimizer
        optimizer_calls = []
        optimizers = []

        def get_megatron_optimizer(*args, **kwargs):
            optimizer_calls.append(kwargs)
            with pytest.MonkeyPatch.context() as patch:
                _forbid_global_grid(patch)
                optimizers.append(build_optimizer(*args, **kwargs))
            return optimizers[-1]

        monkeypatch.setattr(
            training,
            "get_args",
            lambda: SimpleNamespace(
                logits_save_dir=None, logits_load_dir=None, dump_param_to_param_group_map=None
            ),
        )
        monkeypatch.setattr(training, "get_timers", Mock())
        monkeypatch.setattr(training, "get_one_logger", lambda: None)
        monkeypatch.setattr(training, "has_nvidia_modelopt", False)
        monkeypatch.setattr(training, "is_gtp_remat_active", lambda args: False)
        monkeypatch.setattr(training, "get_model", get_model)
        monkeypatch.setattr(
            training,
            "get_megatron_optimizer_config",
            lambda args: (
                OptimizerConfig(optimizer="adam", lr=1e-3, use_distributed_optimizer=True),
                None,
            ),
        )
        monkeypatch.setattr(training, "get_megatron_optimizer", get_megatron_optimizer)
        monkeypatch.setattr(
            training, "get_optimizer_param_scheduler", Mock(side_effect=_OptimizerBuilt)
        )

        with pytest.raises(_OptimizerBuilt):
            training.setup_model_and_optimizer(
                ModelType.encoder_or_decoder,
                Mock(),
                cfg_container=run_config,
                pg_collection=pg_collection,
            )

        # The model and its optimizer are built on the one collection, Gloo groups included.
        assert get_model.call_args.kwargs["pg_collection"] is pg_collection
        (optimizer_kwargs,) = optimizer_calls
        assert optimizer_kwargs["pg_collection"] is pg_collection
        assert optimizer_kwargs["use_gloo_process_groups"] is use_gloo_process_groups
        dense_optimizer, expert_optimizer = optimizers[0].chained_optimizers
        assert isinstance(dense_optimizer, DistributedOptimizer)
        assert isinstance(expert_optimizer, DistributedOptimizer)
        assert dense_optimizer.data_parallel_group_gloo is pg_collection.intra_dp_cp_gloo
        assert expert_optimizer.data_parallel_group_gloo is pg_collection.intra_expt_dp_gloo

    @pytest.mark.usefixtures("run_config")
    def test_get_model_wraps_ddp_with_the_given_collection(self, monkeypatch):
        Utils.initialize_model_parallel()
        args = init_basic_mock_args(parse_args(ignore_unknown_args=True), tp=1, pp=1, bf16=True)
        monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp = _same_ranks(pg_collection.dp)
        pg_collection.dp_cp = _same_ranks(pg_collection.dp_cp)

        def model_provider(pre_process, post_process, config, pg_collection):
            return _TinyModel(config)

        model = training.get_model(model_provider, config=_config(), pg_collection=pg_collection)

        assert model[0].dp_group is pg_collection.dp
        assert model[0].dp_cp_group is pg_collection.dp_cp
        assert model[0].intra_dp_cp_group is pg_collection.dp_cp

    @pytest.mark.skipif(not training.HAVE_FSDP2, reason="Requires Torch FSDP2")
    @pytest.mark.parametrize("has_gtp_remat_fields", [True, False])
    def test_torch_fsdp2_shards_over_the_collection_dp_cp_group(self, has_gtp_remat_fields):
        Utils.initialize_model_parallel()
        dp_cp_group = _same_ranks(
            parallel_state.get_data_parallel_group(with_context_parallel=True)
        )
        if has_gtp_remat_fields:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
            pg_collection.dp_cp_gtp_remat = dp_cp_group
        else:
            pg_collection = ProcessGroupCollection(dp_cp=dp_cp_group)
        ddp_config = DistributedDataParallelConfig()

        # get_model's wrapper and the model builders' wrapper.
        wrapped = training.wrap_model_chunks_with_ddp(
            [_TinyModel(_config()).cuda()],
            _config(),
            ddp_config,
            DP=training.torch_FSDP,
            pg_collection=pg_collection,
        )
        built = _ddp_wrap(
            [_TinyModel(_config()).cuda()],
            data_parallel_random_init=False,
            ddp_config=ddp_config,
            overlap_param_gather_with_optimizer_step=False,
            use_torch_fsdp2=True,
            pg_collection=pg_collection,
        )

        assert wrapped[0].process_group is dp_cp_group
        assert built[0].process_group is dp_cp_group

    def test_schedule_follows_the_collection_pipeline_group(self, monkeypatch):
        # The global grid has no pipeline parallelism; the collection has two stages.
        Utils.initialize_model_parallel()
        pp_group, _ = torch.distributed.new_subgroups_by_enumeration(
            [[rank, rank + 1] for rank in range(0, Utils.world_size, 2)]
        )
        pg_collection = ProcessGroupCollection(pp=pp_group)

        def schedule(vp_size):
            config = SimpleNamespace(virtual_pipeline_model_parallel_size=vp_size)
            return training._get_forward_backward_func(pg_collection, config)

        assert schedule(None) is forward_backward_pipelining_without_interleaving
        assert schedule(2) is forward_backward_pipelining_with_interleaving
        # Without a collection the global grid decides, through the deprecated fallback.
        monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())
        with pytest.warns(ProcessGroupFallbackWarning, match="get_forward_backward_func"):
            assert training._get_forward_backward_func(None, None) is forward_backward_no_pipelining
