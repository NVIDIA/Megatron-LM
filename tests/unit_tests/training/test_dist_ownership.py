# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Behavioral coverage for config-owned distributed initialization."""

from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from dataclasses import asdict, fields
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.training import arguments, global_vars, initialize, training
from megatron.training.argument_utils import inference_cfg_container_from_args
from megatron.training.config import DistributedInitConfig


def _args():
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    args.world_size = 4
    # These topology values are normally resolved by validate_args, not argparse.
    args.gtp_weight_remat_size = 1
    args.expert_gtp_weight_remat_size = 1
    args.virtual_pipeline_model_parallel_size = None
    args.expert_tensor_parallel_size = 1
    return args


@pytest.mark.parametrize("enabled", [False, True])
def test_distributed_cli_preserves_flags_and_single_fsdp_owner(enabled):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    flags = ["--use-megatron-fsdp", "--fake-process-group"] if enabled else []
    args = parser.parse_args(flags)
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    assert args.use_megatron_fsdp is enabled
    assert cfg.dist.fake_process_group is enabled
    assert "use_megatron_fsdp" not in asdict(cfg.dist)


def test_distributed_adapter_does_not_alias_mutable_args():
    args = _args()
    args.high_priority_stream_groups = ["dp"]
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    args.high_priority_stream_groups.append("tp")
    assert cfg.dist.high_priority_stream_groups == ["dp"]


@pytest.mark.parametrize("deleted", [False, True])
@pytest.mark.parametrize("already_initialized", [False, True])
@pytest.mark.parametrize("fake", [False, True])
def test_distributed_initialization_uses_config(
    monkeypatch, tmp_path, run_config, deleted, already_initialized, fake
):
    args = _args()
    cfg = run_config
    cfg.dist = DistributedInitConfig(
        distributed_backend="gloo",
        fake_process_group=fake,
        distributed_timeout_minutes=37,
        local_rank=1,
        use_sharp=True,
        sharp_enabled_group="dp_replica",
        use_gloo_process_groups=False,
        high_priority_stream_groups=["tp"],
        use_tp_pp_dp_mapping=True,
        nccl_communicator_config_path="owned.yaml",
        flight_recorder_dump_path=str(tmp_path),
        flight_recorder_trace_buffer_size=789,
        flight_recorder_dump_on_timeout=False,
        flight_recorder_include_stack_trace=True,
        flight_recorder_include_only_active=False,
        flight_recorder_extra_dump_on_exec=False,
    )
    if deleted:
        for field in fields(cfg.dist):
            delattr(args, field.name)
    original = vars(args).copy()
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize, "print_rank_0", Mock())
    monkeypatch.setattr(initialize, "warn_rank_0", Mock())
    monkeypatch.setattr(initialize.torch.cuda, "device_count", lambda: 2)
    device = Mock()
    monkeypatch.setattr(initialize.torch.cuda, "set_device", device)
    monkeypatch.setattr(initialize.torch.distributed, "is_initialized", lambda: already_initialized)
    monkeypatch.setattr(initialize.torch.distributed, "get_rank", lambda: args.rank)
    monkeypatch.setattr(initialize.torch.distributed, "get_world_size", lambda: args.world_size)
    init_pg, init_mpu = Mock(), Mock()
    monkeypatch.setattr(initialize.torch.distributed, "init_process_group", init_pg)
    monkeypatch.setattr(initialize.inprocess_restart, "maybe_force_nccl_backend_init", Mock())
    monkeypatch.setattr(initialize.mpu, "model_parallel_is_initialized", lambda: False)
    monkeypatch.setattr(initialize.mpu, "initialize_model_parallel", init_mpu)
    for name in (
        "get_tensor_model_parallel_world_size",
        "get_pipeline_model_parallel_world_size",
        "get_gtp_weight_remat_world_size",
    ):
        monkeypatch.setattr(initialize.mpu, name, lambda: 1)
    for name in (
        "TORCH_FR_DUMP_TEMP_FILE",
        "TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
        "TORCH_NCCL_TRACE_BUFFER_SIZE",
        "TORCH_NCCL_DUMP_ON_TIMEOUT",
        "TORCH_INCLUDE_STACK_TRACE",
        "TORCH_INCLUDE_ONLY_ACTIVE",
        "TORCH_NCCL_EXTRA_DUMP_ON_EXEC",
    ):
        monkeypatch.delenv(name, raising=False)
    # Preserve the existing environment-over-config precedence.
    monkeypatch.setenv("TORCH_NCCL_TRACE_BUFFER_SIZE", "123")
    store = object()
    initialize._initialize_distributed(None, None, store)
    if already_initialized:
        init_pg.assert_not_called()
        device.assert_not_called()
    else:
        device.assert_called_once_with(1)
        kwargs = init_pg.call_args.kwargs
        if fake:
            from torch.testing._internal.distributed.fake_pg import FakeStore

            assert isinstance(kwargs["store"], FakeStore)
        else:
            assert kwargs["store"] is store
        assert kwargs == {
            "backend": "fake" if fake else "gloo",
            "store": kwargs["store"],
            "world_size": 4,
            "rank": 0,
            "timeout": timedelta(minutes=37),
        }
        assert initialize.os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "123"
        assert initialize.os.environ["TORCH_INCLUDE_STACK_TRACE"] == "1"
        assert initialize.os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "0"
    kwargs = init_mpu.call_args.kwargs
    assert kwargs["distributed_timeout_minutes"] == 37
    assert kwargs["nccl_communicator_config_path"] == "owned.yaml"
    assert kwargs["order"] == "tp-cp-ep-pp-dp"
    assert kwargs["high_priority_stream_groups"] == ["tp"]
    assert kwargs["use_sharp"] is True
    assert kwargs["sharp_enabled_group"] == "dp_replica"
    assert kwargs["create_gloo_process_groups"] is False
    assert vars(args) == original


@pytest.mark.parametrize("lazy", [False, True])
def test_lazy_initialization_reads_config(monkeypatch, run_config, lazy):
    args = _args()
    del args.lazy_mpu_init
    run_config.dist.lazy_mpu_init = lazy
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    for name in (
        "setup_logging",
        "initialize_rerun_state_machine",
        "_init_autoresume",
        "set_default_log_ranks",
    ):
        monkeypatch.setattr(initialize, name, Mock())
    monkeypatch.setattr(initialize.mpu, "set_tensor_model_parallel_world_size", Mock())
    monkeypatch.setattr(initialize.mpu, "set_tensor_model_parallel_rank", Mock())
    distributed = Mock()
    monkeypatch.setattr(initialize, "_initialize_distributed", distributed)
    finish = initialize.initialize_megatron(
        allow_no_cuda=True, skip_random_seed=True, skip_dependency_compilation=True
    )
    if lazy:
        distributed.assert_not_called()
        assert args.use_cpu_initialization is True
        finish()
    distributed.assert_called_once()


def test_telemetry_uses_owned_local_rank(monkeypatch, run_config):
    run_config.dist.local_rank = 2
    device = Mock(return_value={})
    monkeypatch.setattr(global_vars, "_detect_gpu_identity", device)
    attrs = global_vars.build_telemetry_resource_attrs(Namespace(local_rank=7))
    assert attrs["dl.local_rank"] == 2
    device.assert_called_once_with(2)


@pytest.mark.parametrize("fsdp2", [False, True])
@pytest.mark.parametrize("megatron_fsdp", [False, True])
def test_native_builder_uses_dist_and_ddp_owners(monkeypatch, run_config, fsdp2, megatron_fsdp):
    args = _args()
    del args.use_torch_fsdp2, args.use_megatron_fsdp
    run_config.dist.use_torch_fsdp2 = fsdp2
    run_config.ddp = (
        training.TorchFullyShardedDataParallelConfig(use_megatron_fsdp=megatron_fsdp)
        if fsdp2
        else training.DistributedDataParallelConfig(use_megatron_fsdp=megatron_fsdp)
    )
    builder = Mock()
    run_config.model = SimpleNamespace(get_builder_cls=lambda: Mock(return_value=builder))
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", Mock())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(training, "is_gtp_remat_active", lambda args: False)
    monkeypatch.setattr(training, "_add_model_freeze_pre_wrap_hook", Mock())
    monkeypatch.setattr("megatron.training.utils.start_memory_history_recording", Mock())
    # A passed training container must not be replaced by the global container.
    monkeypatch.setattr(training, "get_run_config", Mock(side_effect=AssertionError("global read")))

    class ReachedBuilder(Exception):
        pass

    builder.build_distributed_models.side_effect = ReachedBuilder
    with pytest.raises(ReachedBuilder):
        training.setup_model_and_optimizer(Mock(), cfg_container=run_config)
    kwargs = builder.build_distributed_models.call_args.kwargs
    assert kwargs["ddp_config"] is run_config.ddp
    assert kwargs["use_torch_fsdp2"] is fsdp2
    assert kwargs["use_megatron_fsdp"] is megatron_fsdp


@pytest.mark.parametrize("fsdp2", [False, True])
@pytest.mark.parametrize("deleted", [False, True])
def test_model_wrapper_uses_dist_config_with_inference_container(monkeypatch, fsdp2, deleted):
    args = _args()
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    cfg.dist.use_torch_fsdp2 = fsdp2
    args.use_torch_fsdp2 = not fsdp2
    if deleted:
        del args.use_torch_fsdp2
    args.use_cpu_initialization = True
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", cfg)
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(training, "HAVE_FSDP2", True)
    monkeypatch.setattr(training, "get_pg_size", lambda group: 1)
    monkeypatch.setattr(training, "get_pg_rank", lambda group: 0)
    monkeypatch.setattr(training, "is_pp_first_stage", lambda group: True)
    monkeypatch.setattr(training, "is_pp_last_stage", lambda group: True)
    monkeypatch.setattr(training, "correct_amax_history_if_needed", Mock())
    monkeypatch.setattr(training.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(training.torch.cuda, "current_stream", Mock())
    monkeypatch.setattr(training.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        training, "get_model_config", lambda model: SimpleNamespace(cuda_graph_impl="none")
    )
    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(model, "cuda", Mock(return_value=model))
    wrap = Mock(return_value=[model])
    monkeypatch.setattr(training, "wrap_model_chunks_with_ddp", wrap)
    groups = SimpleNamespace(dp=None, cp=None, tp=None, gtp_remat=None, pp=None, dp_cp=None)
    assert training.get_model(lambda **kwargs: model, pg_collection=groups) == [model]
    assert wrap.call_args.kwargs["DP"] is (training.torch_FSDP if fsdp2 else training.DDP)
    expected_config = (
        training.TorchFullyShardedDataParallelConfig
        if fsdp2
        else training.DistributedDataParallelConfig
    )
    assert isinstance(wrap.call_args.args[2], expected_config)
    assert not hasattr(cfg, "ddp")
    assert model.cuda.call_count == (0 if fsdp2 else 1)
