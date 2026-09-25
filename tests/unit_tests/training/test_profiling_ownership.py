# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Behavioral ownership checks without starting an actual profiler."""

from argparse import ArgumentParser
from contextlib import nullcontext
from dataclasses import asdict, fields
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from megatron.training import argument_utils, arguments, checkpointing, global_vars, training
from megatron.training.argument_utils import profiling_config_from_args
from megatron.training.config import ProfilingConfig
from megatron.training.utils import start_memory_history_recording


def cli_config(*options):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args(options)
    return args, profiling_config_from_args(args)


def remove_profiling_args(args):
    for field in fields(ProfilingConfig):
        name = "profile" if field.name == "use_nsys_profiler" else field.name
        delattr(args, name)


@pytest.mark.parametrize("cleanup", ["unset_global_variables", "destroy_global_vars"])
def test_run_config_lifecycle(monkeypatch, cleanup):
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
    with pytest.raises(AssertionError, match="run config is not initialized"):
        global_vars.get_run_config()
    config = SimpleNamespace(profiling=ProfilingConfig())
    global_vars.set_run_config(config)
    assert global_vars.get_run_config() is config
    with pytest.raises(AssertionError, match="run config is already initialized"):
        global_vars.set_run_config(SimpleNamespace(profiling=ProfilingConfig()))
    # Both cleanup paths clear the owner before a subsequent run.
    getattr(global_vars, cleanup)()
    with pytest.raises(AssertionError, match="run config is not initialized"):
        global_vars.get_run_config()
    global_vars.set_run_config(config)
    assert global_vars.get_run_config() is config


@pytest.fixture
def inline_profiler(monkeypatch, run_config):
    """Exercise the real train preamble/loop, stopping before the first train step."""
    args, _ = cli_config("--micro-batch-size", "1")
    remove_profiling_args(args)
    args.iteration = args.consumed_train_samples = args.num_floating_point_operations_so_far = 0
    args.train_iters = 10
    args.train_sync_interval = 0
    args.gpu_sniff_test_interval = args.check_weight_hash_across_dp_replicas_interval = None
    args.log_straggler = args.manual_gc = False
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", Mock(return_value=Mock()))
    monkeypatch.setattr(training, "get_energy_monitor", lambda: None)
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "get_attr_wrapped_model", Mock(return_value=Mock()))
    monkeypatch.setattr(training, "get_rerun_state_machine", Mock(return_value=Mock()))
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(training, "get_forward_backward_func", Mock())
    monkeypatch.setattr(training, "one_logger_utils", Mock())
    monkeypatch.setattr(training, "should_disable_forward_pre_hook", lambda _: False)
    monkeypatch.setattr(training, "_otel_managed_span", lambda *a, **kw: nullcontext())
    for name in (
        "write_args_to_tensorboard",
        "print_datetime",
        "_end_otel_startup_span",
        "_start_otel_train_span",
        "_maybe_reroot_otel_interval",
    ):
        monkeypatch.setattr(training, name, Mock())

    class BeforeTrainStep(Exception):
        pass

    monkeypatch.setattr(
        training.ft_integration, "on_checkpointing_start", Mock(side_effect=BeforeTrainStep)
    )

    def start(config, *, iteration=0, rank=0, tensorboard_dir=None):
        run_config.profiling = config
        args.iteration = iteration
        args.tensorboard_dir = tensorboard_dir
        monkeypatch.setattr(training.torch.distributed, "get_rank", lambda: rank)
        with pytest.raises(BeforeTrainStep):
            training.train(
                forward_step_func=Mock(),
                model=[Mock()],
                optimizer=None,
                opt_param_scheduler=None,
                train_data_iterator=None,
                valid_data_iterator=None,
                process_non_loss_data_func=None,
                config=SimpleNamespace(finalize_model_grads_func=Mock()),
                checkpointing_context={},
                non_loss_data_func=None,
            )

    return start


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_and_native_settings_match_without_aliasing(enabled):
    args, config = cli_config(*(["--profile"] if enabled else []), "--profile-ranks", "1", "3")
    native = ProfilingConfig(use_nsys_profiler=enabled, profile_ranks=[1, 3])
    assert asdict(config) == asdict(native)
    args.profile_ranks.append(7)
    args.profile = not enabled
    assert config.profile_ranks == [1, 3]
    assert config.use_nsys_profiler is enabled


def test_legacy_inference_owner_does_not_construct_unused_training_configs():
    args, profiling = cli_config(
        "--profile", "--optimizer", "sgd", "--use-precision-aware-optimizer"
    )
    config = argument_utils.inference_cfg_container_from_args(args, build_model_config=False)
    assert config.model is None
    assert asdict(config.profiling) == asdict(profiling)


@pytest.mark.parametrize("explicit_none", [False, True])
def test_inference_factory_keeps_automatic_model_construction(monkeypatch, explicit_none):
    args, _ = cli_config()
    model = object()
    build = Mock(return_value=model)
    monkeypatch.setattr(argument_utils, "gpt_config_from_args", build)
    kwargs = {"model_cfg": None} if explicit_none else {}
    config = argument_utils.inference_cfg_container_from_args(args, **kwargs)
    assert config.model is model
    build.assert_called_once_with(args)


@pytest.mark.parametrize("native", [False, True])
def test_active_pytorch_window_validation(native, inline_profiler):
    if native:
        config = ProfilingConfig(
            use_nsys_profiler=True,
            use_pytorch_profiler=True,
            profile_step_start=5,
            profile_step_end=5,
        )
    else:
        _, config = cli_config(
            "--profile",
            "--use-pytorch-profiler",
            "--profile-step-start",
            "5",
            "--profile-step-end",
            "5",
        )
    # Construction is valid for non-training consumers; an active selected
    # profiler checks its requirements immediately before starting.
    with pytest.raises(ValueError, match="profile_step_end > profile_step_start"):
        inline_profiler(config)
    config.profile_ranks = [1]
    inline_profiler(config)  # Excluded ranks do not validate the window.
    config.profile_ranks = []
    config.use_nsys_profiler = False
    inline_profiler(config)
    config.use_nsys_profiler = True
    config.use_pytorch_profiler = False
    inline_profiler(config)


@pytest.mark.parametrize("profile", [None, False, True])
def test_legacy_namespace_profile_alias_is_optional(profile):
    args = SimpleNamespace(record_memory_history=True, memory_snapshot_path="legacy.pickle")
    if profile is not None:
        args.profile = profile
    before = vars(args).copy()
    config = profiling_config_from_args(args)
    assert config.use_nsys_profiler is (profile is True)
    assert config.record_memory_history is True
    assert config.memory_snapshot_path == "legacy.pickle"
    assert vars(args) == before


@pytest.mark.parametrize(
    "enabled,ranks,rank,expected",
    [(False, [], 0, False), (True, [1], 0, False), (True, [], 0, True), (True, [1], 1, True)],
)
def test_nsys_windows_and_nvtx_use_config(
    monkeypatch, inline_profiler, enabled, ranks, rank, expected
):
    args, config = cli_config(
        "--profile",
        "--profile-step-start",
        "2",
        "--profile-step-end",
        "4",
        "--record-shapes",
        "--nvtx-ranges",
    )
    config.use_nsys_profiler = enabled
    config.profile_ranks = ranks
    remove_profiling_args(args)
    cudart = Mock()
    context = MagicMock()
    nvtx = Mock()
    monkeypatch.setattr(training.torch.cuda, "cudart", Mock(return_value=cudart))
    monkeypatch.setattr(training.torch.cuda, "check_error", Mock())
    emit = Mock(return_value=context)
    monkeypatch.setattr(training.torch.autograd.profiler, "emit_nvtx", emit)
    monkeypatch.setattr(training, "configure_nvtx_profiling", nvtx)
    inline_profiler(config, iteration=2, rank=rank)
    for iteration in range(1, 7):
        training.post_training_step_callbacks([], None, None, iteration, None, 0, context)
    assert cudart.cudaProfilerStart.call_count == int(expected)
    assert cudart.cudaProfilerStop.call_count == int(expected)
    assert emit.call_count == int(expected)
    assert context.__enter__.call_count == int(expected)
    assert context.__exit__.call_count == int(expected)
    if expected:
        assert [call.args for call in nvtx.call_args_list] == [(True,), (False,)]
        emit.assert_called_once_with(record_shapes=True)
    else:
        nvtx.assert_not_called()


@pytest.mark.parametrize("start", [0, 2])
@pytest.mark.parametrize("chakra", [False, True])
def test_pytorch_consumer_inputs_and_resume_progress(
    monkeypatch, inline_profiler, tmp_path, start, chakra
):
    config = ProfilingConfig(
        use_nsys_profiler=True,
        use_pytorch_profiler=True,
        profile_step_start=start,
        profile_step_end=5,
        pytorch_profiler_collect_shapes=True,
        pytorch_profiler_collect_callstack=True,
        pytorch_profiler_collect_chakra=chakra,
    )
    observer = Mock()
    observer.register_callback.return_value = observer
    backend = Mock()
    backend.execution_trace_observer = observer if chakra else None
    factory = Mock(return_value=backend)
    schedule = Mock()
    monkeypatch.setattr(training.torch.profiler, "profile", factory)
    monkeypatch.setattr(training.torch.profiler, "schedule", schedule)
    monkeypatch.setattr(
        training.torch.profiler, "ExecutionTraceObserver", Mock(return_value=observer)
    )
    inline_profiler(config, iteration=4, tensorboard_dir=str(tmp_path / "tb"))
    schedule.assert_called_once_with(
        wait=max(start - 1, 0), warmup=int(start > 0), active=5 - start, repeat=1
    )
    options = factory.call_args.kwargs
    assert options["record_shapes"] is True and options["with_stack"] is True
    assert options["execution_trace_observer"] is (observer if chakra else None)
    options["on_trace_ready"](backend)
    backend.export_chrome_trace.assert_called_once_with(
        f"{tmp_path}/tb/../torch_profile/rank-0.json.gz"
    )
    # Existing semantics: PyTorch scheduling starts at loop entry on resume;
    # global progress is supplied explicitly for the stop boundary.
    training.post_training_step_callbacks([], None, None, 4, backend, 0)
    backend.stop.assert_not_called()
    training.post_training_step_callbacks([], None, None, 5, backend, 0)
    backend.start.assert_called_once()
    backend.step.assert_called_once()
    backend.stop.assert_called_once()
    assert observer.unregister_callback.call_count == int(chakra)


def test_checkpoint_profiling_does_not_override_current_run(monkeypatch):
    args, config = cli_config("--profile-step-start", "7", "--profile-ranks", "2")
    args.rank = 0
    args.load = "checkpoint"
    saved_args = SimpleNamespace(profile=True, profile_step_start=100, profile_ranks=[99])
    state = {"args": saved_args, "iteration": 17, "checkpoint_version": 3.0}
    monkeypatch.setattr(
        checkpointing,
        "_load_base_checkpoint",
        Mock(return_value=(state, "checkpoint", False, None)),
    )
    checkpointing.load_args_from_checkpoint(args)
    assert args.iteration == 17
    assert asdict(profiling_config_from_args(args)) == asdict(config)


@pytest.mark.parametrize(
    "enabled,ranks,rank,expected",
    [(False, [], 0, False), (True, [1], 0, False), (True, [], 0, True), (True, [1], 1, True)],
)
def test_memory_history_uses_config(monkeypatch, enabled, ranks, rank, expected):
    from megatron.training.utils import utils as memory_utils

    config = ProfilingConfig(
        record_memory_history=enabled, profile_ranks=ranks, memory_snapshot_path="memory.pickle"
    )
    record = Mock()
    attach = Mock()
    dump = Mock()
    monkeypatch.setattr(memory_utils, "safe_get_rank", lambda: rank)
    monkeypatch.setattr(memory_utils.torch.cuda.memory, "_record_memory_history", record)
    monkeypatch.setattr(memory_utils.torch._C, "_cuda_attach_out_of_memory_observer", attach)
    monkeypatch.setattr(memory_utils.torch.cuda.memory, "_dump_snapshot", dump)
    start_memory_history_recording(config)
    assert record.call_count == int(expected)
    assert attach.call_count == int(expected)
    if expected:
        attach.call_args.args[0](0, 1, 2, 3)
        dump.assert_called_once_with(f"memory_oom_rank_{rank}.pickle")


@pytest.mark.parametrize(
    "enabled,ranks,rank,backend,expected",
    [
        (False, [], 0, "nccl", False),
        (True, [1], 0, "nccl", False),
        (True, [], 0, "nccl", True),
        (True, [1], 1, "nccl", True),
        (True, [1], 0, "fake", True),
    ],
)
def test_training_log_memory_snapshot_without_profiling_args(
    monkeypatch, run_config, enabled, ranks, rank, backend, expected
):
    from megatron.training import training

    args, config = cli_config("--log-interval", "1", "--micro-batch-size", "1")
    run_config.profiling = config
    config.record_memory_history = enabled
    config.profile_ranks = ranks
    config.memory_snapshot_path = "owned.pickle"
    remove_profiling_args(args)
    args.data_parallel_size = args.world_size = args.gtp_weight_remat_size = 1
    args.consumed_train_samples = args.skipped_train_samples = 0
    args.train_iters = 1
    timers = Mock()
    timers.return_value.elapsed.return_value = 1.0
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", lambda: timers)
    for getter in (
        "get_tensorboard_writer",
        "get_wandb_writer",
        "get_one_logger",
        "get_energy_monitor",
        "get_telemetry",
    ):
        monkeypatch.setattr(training, getter, lambda: None)
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 1)
    monkeypatch.setattr(training, "safe_get_rank", lambda: rank)
    monkeypatch.setattr(training, "one_logger_utils", Mock())
    monkeypatch.setattr(training, "num_floating_point_operations", lambda *a, **kw: 1)
    monkeypatch.setattr(training, "reduce_max_stat_across_model_parallel_group", lambda x, **kw: x)
    monkeypatch.setattr(training, "print_rank_last", Mock())
    monkeypatch.setattr(training.torch.distributed, "get_backend", lambda: backend)
    dump = Mock()
    monkeypatch.setattr(training.torch.cuda.memory, "_dump_snapshot", dump)
    training.training_log({}, {}, 0.01, 1, 1.0, False, 0, None, None, None, None)
    if expected:
        dump.assert_called_once_with(f"owned_{rank}.pickle")
    else:
        dump.assert_not_called()
