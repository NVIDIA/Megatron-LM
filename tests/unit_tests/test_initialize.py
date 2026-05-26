# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.training import initialize


def _make_initialize_args(**overrides):
    args = SimpleNamespace(
        ckpt_convert_format=None,
        ckpt_convert_save=None,
        load=None,
        pretrained_checkpoint=None,
        exit_on_missing_checkpoint=False,
        use_checkpoint_args=False,
        non_persistent_ckpt_type=None,
        yaml_cfg=None,
        async_save=False,
        use_persistent_ckpt_worker=False,
        rank=0,
        rerun_mode="disabled",
        error_injection_rate=0.0,
        error_injection_type="correct_result",
        result_rejected_tracker_filename=None,
        batch_invariant_mode=False,
        lazy_mpu_init=False,
        tensor_model_parallel_size=1,
        use_cpu_initialization=False,
        tp_comm_overlap=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_initialize_megatron_sets_globals_and_can_skip_mpu(monkeypatch):
    calls = []
    args = _make_initialize_args(batch_invariant_mode=True)
    monkeypatch.setattr(initialize, "validate_args", lambda args, defaults: calls.append(("validate", defaults)))
    monkeypatch.setattr(initialize, "set_global_variables", lambda args: calls.append(("globals", args.rank)))
    monkeypatch.setattr(initialize, "setup_logging", lambda: calls.append("logging"))
    monkeypatch.setattr(
        initialize,
        "initialize_rerun_state_machine",
        lambda **kwargs: calls.append(("rerun", kwargs["mode"].value)),
    )
    monkeypatch.setattr(initialize, "print_rank_0", lambda message: calls.append(("print", message)))
    monkeypatch.setattr(initialize, "enable_batch_invariant_mode", lambda: calls.append("batch-invariant"))
    monkeypatch.setattr(initialize, "get_args", lambda: args)

    result = initialize.initialize_megatron(
        allow_no_cuda=True,
        skip_mpu_initialization=True,
        parsed_args=args,
    )

    assert result is None
    assert ("validate", {}) in calls
    assert ("globals", 0) in calls
    assert ("rerun", "disabled") in calls
    assert "batch-invariant" in calls


def test_initialize_megatron_lazy_mpu_returns_finish_callback(monkeypatch):
    calls = []
    args = _make_initialize_args(lazy_mpu_init=True, tensor_model_parallel_size=4, rank=2)
    monkeypatch.setattr(initialize, "validate_args", lambda args, defaults: calls.append("validate"))
    monkeypatch.setattr(initialize, "set_global_variables", lambda args: calls.append("globals"))
    monkeypatch.setattr(initialize, "setup_logging", lambda: calls.append("logging"))
    monkeypatch.setattr(initialize, "initialize_rerun_state_machine", lambda **kwargs: calls.append("rerun"))
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize.mpu, "set_tensor_model_parallel_world_size", lambda size: calls.append(("tp-size", size)))
    monkeypatch.setattr(initialize.mpu, "set_tensor_model_parallel_rank", lambda rank: calls.append(("tp-rank", rank)))

    finish = initialize.initialize_megatron(allow_no_cuda=True, parsed_args=args)

    assert callable(finish)
    assert args.use_cpu_initialization is True
    assert ("tp-size", 4) in calls
    assert ("tp-rank", 2) in calls


def test_initialize_megatron_eager_path_runs_finish_autoresume_compile_and_tp_overlap(monkeypatch):
    calls = []
    args = _make_initialize_args(
        seed=123,
        data_parallel_random_init=False,
        te_rng_tracker=False,
        inference_rng_tracker=False,
        cuda_graph_impl="none",
        num_experts=None,
        tp_comm_overlap=True,
    )
    monkeypatch.setattr(initialize, "validate_args", lambda args, defaults: calls.append("validate"))
    monkeypatch.setattr(initialize, "set_global_variables", lambda args: calls.append("globals"))
    monkeypatch.setattr(initialize, "setup_logging", lambda: calls.append("logging"))
    monkeypatch.setattr(initialize, "initialize_rerun_state_machine", lambda **kwargs: calls.append("rerun"))
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize, "_initialize_distributed", lambda *items: calls.append("distributed"))
    monkeypatch.setattr(initialize, "_set_random_seed", lambda *items, **kwargs: calls.append(("seed", items[0], kwargs)))
    monkeypatch.setattr(initialize, "_init_autoresume", lambda: calls.append("autoresume"))
    monkeypatch.setattr(initialize, "_compile_dependencies", lambda: calls.append("compile"))
    monkeypatch.setattr(initialize, "_initialize_tp_communicators", lambda: calls.append("tp-overlap"))
    monkeypatch.setattr(initialize, "print_rank_0", lambda message: calls.append(("print", message)))

    result = initialize.initialize_megatron(allow_no_cuda=True, parsed_args=args)

    assert result is None
    assert "distributed" in calls
    assert any(call[0] == "seed" and call[1] == 123 for call in calls if isinstance(call, tuple))
    assert "autoresume" in calls
    assert "compile" in calls
    assert "tp-overlap" in calls


def test_initialize_megatron_validates_yaml_and_checkpoint_arg_requirements(monkeypatch):
    yaml_args = _make_initialize_args(yaml_cfg="config.yaml")
    calls = []
    monkeypatch.setattr(initialize, "validate_yaml", lambda args, defaults: calls.append(("yaml", defaults)) or args)
    monkeypatch.setattr(initialize, "set_global_variables", lambda args: calls.append("globals"))
    monkeypatch.setattr(initialize, "setup_logging", lambda: calls.append("logging"))
    monkeypatch.setattr(initialize, "initialize_rerun_state_machine", lambda **kwargs: calls.append("rerun"))
    monkeypatch.setattr(initialize, "get_args", lambda: yaml_args)

    initialize.initialize_megatron(
        allow_no_cuda=True,
        skip_mpu_initialization=True,
        parsed_args=yaml_args,
        args_defaults={"alpha": 1},
    )

    assert ("yaml", {"alpha": 1}) in calls

    bad_args = _make_initialize_args(use_checkpoint_args=True, load=None, pretrained_checkpoint=None)
    try:
        initialize.initialize_megatron(
            allow_no_cuda=True,
            skip_mpu_initialization=True,
            parsed_args=bad_args,
        )
    except AssertionError as exc:
        assert "--use-checkpoint-args requires" in str(exc)
    else:
        raise AssertionError("expected missing checkpoint args to raise")


def test_set_random_seed_offsets_pipeline_and_data_parallel_ranks(monkeypatch):
    manual_seed_calls = []
    monkeypatch.setattr(initialize.mpu, "get_pipeline_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(initialize.mpu, "get_data_parallel_rank", lambda: 3)
    monkeypatch.setattr(initialize.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(initialize.torch, "manual_seed", lambda seed: manual_seed_calls.append(seed))

    initialize._set_random_seed(11, data_parallel_random_init=True)

    assert manual_seed_calls == [241]
    assert random.randint(0, 10) >= 0
    assert np.random.randint(0, 10) >= 0

    with pytest.raises(ValueError, match="positive integer"):
        initialize._set_random_seed(0)


def test_write_args_to_tensorboard_writes_all_namespace_fields(monkeypatch):
    calls = []
    args = SimpleNamespace(iteration=7, alpha=1, beta="two")
    writer = SimpleNamespace(add_text=lambda name, value, global_step: calls.append((name, value, global_step)))
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize, "get_tensorboard_writer", lambda: writer)

    initialize.write_args_to_tensorboard()

    assert ("alpha", "1", 7) in calls
    assert ("beta", "two", 7) in calls


def test_init_autoresume_runs_barriers_when_available(monkeypatch):
    calls = []
    autoresume = SimpleNamespace(init=lambda: calls.append("init"))
    monkeypatch.setattr(initialize, "get_adlr_autoresume", lambda: autoresume)
    monkeypatch.setattr(initialize.torch.distributed, "barrier", lambda: calls.append("barrier"))

    initialize._init_autoresume()

    assert calls == ["barrier", "init", "barrier"]


def test_set_jit_fusion_options_handles_modern_nvfuser_and_legacy_paths(monkeypatch):
    calls = []
    monkeypatch.setattr(initialize, "_warmup_jit_function", lambda: calls.append("warmup"))

    monkeypatch.setattr(initialize, "is_torch_min_version", lambda version: version == "2.2.0a0")
    initialize.set_jit_fusion_options()
    assert calls == ["warmup"]

    calls.clear()
    monkeypatch.setattr(initialize, "is_torch_min_version", lambda version: version == "1.10.0a0")
    for name in [
        "_jit_set_profiling_executor",
        "_jit_set_profiling_mode",
        "_jit_override_can_fuse_on_cpu",
        "_jit_override_can_fuse_on_gpu",
        "_jit_set_texpr_fuser_enabled",
        "_jit_set_nvfuser_enabled",
        "_debug_set_autodiff_subgraph_inlining",
    ]:
        monkeypatch.setattr(
            initialize.torch._C,
            name,
            lambda value, name=name: calls.append((name, value)),
            raising=False,
        )
    initialize.set_jit_fusion_options()
    assert ("_jit_set_nvfuser_enabled", True) in calls
    assert calls[-1] == "warmup"

    calls.clear()
    monkeypatch.setattr(initialize, "is_torch_min_version", lambda version: False)
    initialize.set_jit_fusion_options()
    assert ("_jit_override_can_fuse_on_cpu", True) in calls
    assert ("_jit_override_can_fuse_on_gpu", True) in calls
    assert calls[-1] == "warmup"


def test_warmup_jit_function_uses_configured_fusion_shapes(monkeypatch):
    calls = []

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape
            self.requires_grad = False

        def expand_as(self, other):
            calls.append(("expand", other.shape))
            return self

    def fake_rand(*items, **kwargs):
        calls.append(("rand", items[0], kwargs["dtype"], kwargs["device"]))
        return FakeTensor(items[0])

    args = SimpleNamespace(
        bf16=True,
        fp16=False,
        ffn_hidden_size=8,
        tensor_model_parallel_size=2,
        seq_length=6,
        context_parallel_size=1,
        micro_batch_size=2,
        swiglu=True,
        sequence_parallel=True,
        hidden_size=4,
    )
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize.torch, "rand", fake_rand)
    monkeypatch.setattr(initialize, "bias_swiglu", lambda input_, bias: calls.append("swiglu") or input_)
    monkeypatch.setattr(initialize, "bias_gelu", lambda bias, input_: calls.append("gelu") or input_)
    monkeypatch.setattr(
        initialize,
        "bias_dropout_add_fused_train",
        lambda pair, residual, dropout: calls.append(("dropout", dropout)) or residual,
    )
    monkeypatch.setattr(initialize.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(initialize.torch.cuda, "empty_cache", lambda: calls.append("empty-cache"))

    initialize._warmup_jit_function()

    assert ("rand", 4, torch.bfloat16, "cuda") in calls
    assert ("rand", (6, 2, 4), torch.bfloat16, "cuda") in calls
    assert ("rand", (3, 2, 4), torch.bfloat16, "cuda") in calls
    assert calls.count("swiglu") == 10
    assert calls.count(("dropout", 0.1)) == 10
    assert "empty-cache" in calls


def test_compile_dependencies_runs_compile_only_on_rank_zero(monkeypatch):
    calls = []
    monkeypatch.setattr(initialize, "get_args", lambda: SimpleNamespace())
    monkeypatch.setattr(initialize.time, "time", lambda: 10.0)
    monkeypatch.setattr(initialize.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(initialize.torch.distributed, "barrier", lambda: calls.append("barrier"))
    monkeypatch.setattr(
        "megatron.core.datasets.utils.compile_helpers",
        lambda: calls.append("compile"),
    )

    initialize._compile_dependencies()

    assert calls == ["compile", "barrier"]

    calls.clear()
    monkeypatch.setattr(initialize.torch.distributed, "get_rank", lambda: 1)
    initialize._compile_dependencies()
    assert calls == ["barrier"]


def test_initialize_distributed_fast_path_when_torch_dist_already_initialized(monkeypatch):
    calls = []
    args = SimpleNamespace(rank=-1, world_size=-1)
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(initialize.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(initialize.torch.distributed, "get_rank", lambda: 3)
    monkeypatch.setattr(initialize.torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(initialize, "print_rank_0", lambda message: calls.append(message))

    initialize._initialize_distributed(None, None, store=None)

    assert args.rank == 3
    assert args.world_size == 8
    assert any("already initialized" in message for message in calls)


def test_initialize_distributed_sets_flight_recorder_env_and_init_group(monkeypatch, tmp_path):
    calls = []
    args = SimpleNamespace(
        local_rank=0,
        cuda_graph_impl=None,
        flight_recorder_dump_path=str(tmp_path),
        flight_recorder_trace_buffer_size=4096,
        flight_recorder_dump_on_timeout=True,
        flight_recorder_include_stack_trace=True,
        flight_recorder_include_only_active=False,
        flight_recorder_extra_dump_on_exec=True,
        distributed_backend="nccl",
        world_size=4,
        rank=2,
        distributed_timeout_minutes=7,
        fake_process_group=False,
    )
    for env_name in [
        "TORCH_FR_DUMP_TEMP_FILE",
        "TORCH_NCCL_DEBUG_INFO_TEMP_FILE",
        "TORCH_NCCL_TRACE_BUFFER_SIZE",
        "TORCH_NCCL_DUMP_ON_TIMEOUT",
        "TORCH_INCLUDE_STACK_TRACE",
        "TORCH_INCLUDE_ONLY_ACTIVE",
        "TORCH_NCCL_EXTRA_DUMP_ON_EXEC",
    ]:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(initialize.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        initialize.torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs)),
    )
    monkeypatch.setattr(initialize.inprocess_restart, "maybe_force_nccl_backend_init", lambda device_id: calls.append(("nccl", device_id)))
    monkeypatch.setattr(initialize, "print_rank_0", lambda message: calls.append(("print", message)))
    monkeypatch.setattr(initialize, "warn_rank_0", lambda message: calls.append(("warn", message)))

    initialize._initialize_distributed(None, None, store="store")

    init_kwargs = next(item[1] for item in calls if item[0] == "init")
    assert init_kwargs["backend"] == "nccl"
    assert init_kwargs["world_size"] == 4
    assert init_kwargs["rank"] == 2
    assert init_kwargs["store"] == "store"
    assert ("nccl", None) in calls
    assert initialize.os.environ["TORCH_FR_DUMP_TEMP_FILE"].endswith("_dump_")
    assert initialize.os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "4096"


def test_setup_logging_uses_env_and_argument_precedence(monkeypatch):
    root_logger = logging.getLogger()
    original_level = root_logger.level
    try:
        monkeypatch.setenv("MEGATRON_LOGGING_LEVEL", str(logging.WARNING))
        monkeypatch.setattr(initialize, "is_rank0", lambda: True)
        monkeypatch.setattr(initialize, "get_args", lambda: SimpleNamespace(logging_level=None))

        initialize.setup_logging()
        assert root_logger.level == logging.WARNING

        monkeypatch.setattr(initialize, "get_args", lambda: SimpleNamespace(logging_level=logging.ERROR))
        initialize.setup_logging()
        assert root_logger.level == logging.ERROR
    finally:
        root_logger.setLevel(original_level)
