# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for MIMO's nested DDP overlap lifecycle."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import (
    MimoOptimizer,
    _optimizer_config_for_module,
    get_mimo_optimizer,
)
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig


def _overlap_stub(modules):
    stub = SimpleNamespace()
    stub._active_ddp_modules = lambda: iter(modules)
    for name in (
        "no_sync",
        "enable_forward_pre_hook",
        "disable_forward_pre_hook",
        "start_param_sync",
        "start_grad_sync",
        "free_overlap_buffers",
    ):
        setattr(stub, name, getattr(MimoModel, name).__get__(stub))
    return stub


def _ddp(*, grad_overlap, param_overlap, events, name):
    module = MagicMock()
    module.ddp_config = SimpleNamespace(
        overlap_grad_reduce=grad_overlap, overlap_param_gather=param_overlap
    )

    @contextmanager
    def no_sync():
        events.append(f"{name}:enter")
        try:
            yield
        finally:
            events.append(f"{name}:exit")

    module.no_sync.side_effect = no_sync
    return module


def test_nested_overlap_lifecycle_routes_only_to_enabled_modules():
    events = []
    language = _ddp(grad_overlap=True, param_overlap=True, events=events, name="language")
    encoder = _ddp(grad_overlap=True, param_overlap=False, events=events, name="encoder")
    inactive = _ddp(grad_overlap=False, param_overlap=False, events=events, name="inactive")
    model = _overlap_stub([language, encoder, inactive])

    with model.no_sync():
        events.append("body")

    assert events == ["language:enter", "encoder:enter", "body", "encoder:exit", "language:exit"]
    inactive.no_sync.assert_not_called()

    model.enable_forward_pre_hook()
    model.disable_forward_pre_hook(param_sync=False)
    model.start_param_sync(force_sync=True, force_dispatch=True)
    model.start_grad_sync()
    model.free_overlap_buffers()

    language.enable_forward_pre_hook.assert_called_once_with()
    language.disable_forward_pre_hook.assert_called_once_with(param_sync=False)
    language.start_param_sync.assert_called_once_with(force_sync=True, force_dispatch=True)
    language.start_grad_sync.assert_called_once_with()
    language.free_overlap_buffers.assert_called_once_with()

    encoder.enable_forward_pre_hook.assert_not_called()
    encoder.start_param_sync.assert_not_called()
    encoder.start_grad_sync.assert_called_once_with()
    inactive.start_grad_sync.assert_not_called()


def test_nested_forward_pre_hook_handles_are_visible_to_stock_train_loop():
    language = MagicMock()
    encoder = MagicMock()
    language.remove_forward_pre_hook_handles = {object(): object()}
    encoder.remove_forward_pre_hook_handles = {}
    model = _overlap_stub([language, encoder])

    handles = MimoModel.remove_forward_pre_hook_handles.fget(model)

    assert handles == language.remove_forward_pre_hook_handles


def test_mimo_optimizer_exposes_inner_optimizers_to_stock_train_loop():
    dense = object()
    expert = object()
    module_optimizer = SimpleNamespace(chained_optimizers=[dense, expert])
    standalone_optimizer = object()
    optimizer = SimpleNamespace(_active_optimizers=[module_optimizer, standalone_optimizer])

    optimizers = MimoOptimizer.chained_optimizers.fget(optimizer)

    assert optimizers == [dense, expert, standalone_optimizer]


def test_encoder_optimizer_uses_nonoverlapped_mxfp8_param_copy():
    global_config = OptimizerConfig(
        fp8_recipe='mxfp8', reuse_grad_buf_for_mxfp8_param_ag=True, overlap_param_gather=True
    )
    module = SimpleNamespace(
        ddp_config=DistributedDataParallelConfig(
            overlap_param_gather=False,
            use_distributed_optimizer=True,
            fp8_param_gather=True,
            reuse_grad_buf_for_mxfp8_param_ag=True,
        )
    )
    module_config = _optimizer_config_for_module(global_config, module)
    assert module_config is not global_config
    assert global_config.reuse_grad_buf_for_mxfp8_param_ag
    assert global_config.overlap_param_gather
    assert module_config.reuse_grad_buf_for_mxfp8_param_ag
    assert not module_config.overlap_param_gather
    assert module_config.use_distributed_optimizer
    assert not global_config.use_distributed_optimizer
    optimizer = SimpleNamespace(
        config=module_config,
        timers=None,
        is_stub_optimizer=False,
        optimizer=MagicMock(),
        _copy_main_params_to_model_params=MagicMock(),
        _copy_main_params_to_param_buffer=MagicMock(),
    )

    assert MixedPrecisionOptimizer.step_with_ready_grads(optimizer)
    optimizer._copy_main_params_to_model_params.assert_not_called()
    optimizer._copy_main_params_to_param_buffer.assert_called_once_with()


def test_optimizer_config_rejects_module_without_ddp_config():
    with pytest.raises(ValueError, match="must have a ddp_config"):
        _optimizer_config_for_module(OptimizerConfig(), SimpleNamespace())


@pytest.mark.parametrize("optimizer_type", [MixedPrecisionOptimizer, FullyShardedOptimizer])
def test_mimo_optimizer_scopes_param_group_alignment_to_module(mocker, optimizer_type):
    intra_dist_opt = object()
    full_module_group = object()
    original_grad_group = object()
    pg_collection = SimpleNamespace(intra_dist_opt=intra_dist_opt)
    module = SimpleNamespace(
        pg_collection=pg_collection,
        ddp_config=DistributedDataParallelConfig(overlap_param_gather=True),
        parameters=lambda: iter((SimpleNamespace(requires_grad=True),)),
    )
    grid = SimpleNamespace(
        is_current_rank_in_grid=lambda: True,
        dim_names=["tp", "cp", "dp", "pp"],
        get_pg=MagicMock(return_value=full_module_group),
    )
    mimo_model = SimpleNamespace(
        mimo_config=SimpleNamespace(module_to_grid_map={MIMO_LANGUAGE_MODULE_KEY: grid}),
        language_model=object(),
        modality_submodules={},
    )
    inner_optimizer = MagicMock(spec=optimizer_type)
    inner_optimizer.grad_stats_parallel_group = original_grad_group
    optimizer_factory = mocker.patch(
        "megatron.core.optimizer.get_megatron_optimizer", return_value=inner_optimizer
    )

    get_mimo_optimizer(
        mimo_model,
        OptimizerConfig(use_distributed_optimizer=True, reuse_grad_buf_for_mxfp8_param_ag=True),
        module_overrides={MIMO_LANGUAGE_MODULE_KEY: module},
    )

    assert optimizer_factory.call_args.kwargs['param_group_process_group'] is intra_dist_opt
    assert optimizer_factory.call_args.kwargs['pg_collection'] is pg_collection
    assert optimizer_factory.call_args.kwargs["model_chunks"] == [module]
    module_config = optimizer_factory.call_args.kwargs["config"]
    assert not module_config.use_distributed_optimizer
    assert not module_config.reuse_grad_buf_for_mxfp8_param_ag
    expected_grad_group = (
        full_module_group if optimizer_type is FullyShardedOptimizer else original_grad_group
    )
    assert inner_optimizer.grad_stats_parallel_group is expected_grad_group
    if optimizer_type is FullyShardedOptimizer:
        grid.get_pg.assert_called_once_with(grid.dim_names)


def test_mimo_optimizer_stages_each_active_optimizer_before_param_sync():
    language_optimizer = MagicMock()
    encoder_optimizer = MagicMock()
    optimizer = SimpleNamespace(_active_optimizers=[language_optimizer, encoder_optimizer])

    MimoOptimizer.prepare_model_params_for_param_sync(optimizer)

    language_optimizer.prepare_model_params_for_param_sync.assert_called_once_with()
    encoder_optimizer.prepare_model_params_for_param_sync.assert_called_once_with()
