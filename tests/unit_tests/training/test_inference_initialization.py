# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Inference startup must not depend on training batch or progress state."""

import importlib.util
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.training import arguments, global_vars, initialize


@pytest.mark.parametrize("load_checkpoint", [False, True])
@pytest.mark.parametrize("conditional", [False, True])
def test_generate_samples_uses_inference_startup(monkeypatch, load_checkpoint, conditional):
    # This legacy example still imports the removed generation module. Isolate
    # that unrelated dependency so this test exercises its actual startup path.
    generation = ModuleType("megatron.inference.text_generation")
    generation.generate_and_post_process = Mock()
    monkeypatch.setitem(sys.modules, generation.__name__, generation)
    monkeypatch.setattr(sys, "path", list(sys.path))
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/academic_paper_scripts/detxoify_lm/generate_samples_gpt.py"
    )
    spec = importlib.util.spec_from_file_location("generate_samples_gpt", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    args = Namespace(
        load="checkpoint" if load_checkpoint else None,
        sample_input_file="prompts.jsonl" if conditional else None,
    )
    model = Mock()
    cfg = Mock()
    monkeypatch.setattr(script, "parse_and_validate_args", Mock(return_value=args))
    monkeypatch.setattr(script, "get_args", lambda: args)
    monkeypatch.setattr(script, "inference_cfg_container_from_args", Mock(return_value=cfg))
    calls = {}
    for name in (
        "set_run_config",
        "initialize_runtime_services",
        "initialize_megatron",
        "load_checkpoint",
        "generate_and_write_samples_conditional",
        "generate_and_write_samples_unconditional",
    ):
        calls[name] = Mock()
        monkeypatch.setattr(script, name, calls[name])
    monkeypatch.setattr(script, "get_model", Mock(return_value=[model]))

    script.main()

    calls["set_run_config"].assert_called_once_with(cfg)
    calls["initialize_runtime_services"].assert_called_once_with(args)
    calls["initialize_megatron"].assert_called_once_with()
    if load_checkpoint:
        calls["load_checkpoint"].assert_called_once_with([model], None, None)
    else:
        calls["load_checkpoint"].assert_not_called()
    selected = "conditional" if conditional else "unconditional"
    other = "unconditional" if conditional else "conditional"
    calls[f"generate_and_write_samples_{selected}"].assert_called_once_with(model)
    calls[f"generate_and_write_samples_{other}"].assert_not_called()


@pytest.mark.parametrize("restored_samples", [0, 8, 32])
def test_resume_updates_microbatches_before_setup_validation(monkeypatch, restored_samples):
    from megatron.training import training

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.skip_train = True  # Evaluation-only setup must also restore the calculator.
    args.load = "checkpoint"
    args.data_parallel_size = 1
    args.micro_batch_size = 1
    args.consumed_train_samples = 0
    model = [SimpleNamespace()]
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_timers", Mock())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(training, "is_gtp_remat_active", lambda args: False)
    monkeypatch.setattr(training, "get_model", Mock(return_value=model))
    monkeypatch.setattr(training, "unwrap_model", lambda value: value)
    events = []

    def restore(*unused_args, **unused_kwargs):
        assert unused_kwargs["restore_training_state"] is True
        args.consumed_train_samples = restored_samples
        events.append("load")
        return 7, 0

    def update(consumed_samples, verbose):
        assert consumed_samples == restored_samples
        assert verbose
        events.append("update")

    class ReachedBatchValidation(Exception):
        pass

    def validate():
        assert events == ["load", "update"]
        raise ReachedBatchValidation

    monkeypatch.setattr(training, "load_checkpoint", restore)
    monkeypatch.setattr(training, "update_num_microbatches", update)
    monkeypatch.setattr(training, "get_num_microbatches", validate)
    with pytest.raises(ReachedBatchValidation):
        training.setup_model_and_optimizer(Mock(), model_provider_func=Mock())


@pytest.mark.parametrize("build_tokenizer", [False, True])
@pytest.mark.parametrize("enable_runtime_flags", [False, True])
@pytest.mark.parametrize("training_kwargs", [{}, {"training": False}, {"training": True}])
def test_runtime_services_initialize_training_only_when_requested(
    monkeypatch, build_tokenizer, enable_runtime_flags, training_kwargs
):
    args = Namespace(
        enable_experimental=enable_runtime_flags, disable_jit_fuser=enable_runtime_flags
    )
    experimental = Mock()
    jit = Mock()
    monkeypatch.setattr(global_vars, "set_experimental_flag", experimental)
    monkeypatch.setattr(global_vars, "disable_jit_fuser", jit)
    services = {}
    for name in ("_build_tokenizer", "_set_wandb_writer", "_set_telemetry"):
        services[name] = Mock()
        monkeypatch.setattr(global_vars, name, services[name])
    training_services = Mock()
    monkeypatch.setattr(global_vars, "initialize_training_runtime_services", training_services)
    for name in (
        "init_num_microbatches_calculator",
        "_set_tensorboard_writer",
        "_set_timers",
        "_set_energy_monitor",
        "_set_one_logger",
        "_set_adlr_autoresume",
        "_set_train_state",
        "_set_signal_handler",
    ):
        monkeypatch.setattr(global_vars, name, Mock(side_effect=AssertionError(name)))

    global_vars.initialize_runtime_services(
        args, build_tokenizer=build_tokenizer, **training_kwargs
    )

    assert services["_build_tokenizer"].call_count == int(build_tokenizer)
    services["_set_wandb_writer"].assert_called_once_with(args)
    services["_set_telemetry"].assert_called_once_with(
        args, include_training=training_kwargs.get("training", False)
    )
    if training_kwargs.get("training", False):
        training_services.assert_called_once_with(args)
    else:
        training_services.assert_not_called()
    if enable_runtime_flags:
        experimental.assert_called_once_with(True)
        jit.assert_called_once_with()
    else:
        experimental.assert_not_called()
        jit.assert_not_called()
    assert vars(args) == {
        "enable_experimental": enable_runtime_flags,
        "disable_jit_fuser": enable_runtime_flags,
    }


def test_inference_telemetry_does_not_read_training_fields(monkeypatch):
    class InferenceArgs(Namespace):
        def __getattribute__(self, name):
            if name in {"micro_batch_size", "global_batch_size", "train_iters"}:
                raise AssertionError(f"Inference read training field {name}")
            return super().__getattribute__(name)

    monkeypatch.setattr(global_vars, "_detect_gpu_identity", Mock(return_value={}))
    attrs = global_vars.build_telemetry_resource_attrs(
        InferenceArgs(local_rank=0), include_training=False
    )
    assert "dl.batch_size" not in attrs
    assert "megatron.micro_batch_size" not in attrs
    assert "megatron.train_iters" not in attrs


@pytest.mark.parametrize("training_kwargs", [{}, {"training": False}])
def test_inference_distributed_initialization_without_training_fields(
    monkeypatch, training_kwargs, run_config
):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    for name in (
        "micro_batch_size",
        "global_batch_size",
        "async_save",
        "use_persistent_ckpt_worker",
        "rerun_mode",
        "error_injection_rate",
        "error_injection_type",
        "result_rejected_tracker_filename",
    ):
        delattr(args, name)
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize, "setup_logging", Mock())
    monkeypatch.setattr(initialize, "print_rank_0", Mock())
    monkeypatch.setattr(initialize, "set_default_log_ranks", Mock())
    distributed = Mock()
    seeds = Mock()
    monkeypatch.setattr(initialize, "_initialize_distributed", distributed)
    monkeypatch.setattr(initialize, "_set_random_seed", seeds)
    for name in (
        "init_persistent_async_worker",
        "initialize_rerun_state_machine",
        "_init_autoresume",
        "_compile_dependencies",
        "_initialize_tp_communicators",
    ):
        monkeypatch.setattr(initialize, name, Mock(side_effect=AssertionError(name)))

    initialize.initialize_megatron(allow_no_cuda=True, **training_kwargs)

    distributed.assert_called_once()
    seeds.assert_called_once()
    assert seeds.call_args.args[0] == run_config.rng.seed


def test_inference_rejects_training_tp_overlap_before_initialization(monkeypatch, run_config):
    monkeypatch.setattr(initialize, "get_args", lambda: Namespace(tp_comm_overlap=True))
    distributed = Mock()
    monkeypatch.setattr(initialize, "_initialize_distributed", distributed)
    with pytest.raises(ValueError, match="fixed training user buffers"):
        initialize.initialize_megatron(allow_no_cuda=True)
    distributed.assert_not_called()


def test_distributed_training_services_require_explicit_opt_in(monkeypatch, run_config):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    args.async_save = True
    args.use_persistent_ckpt_worker = True
    args.tp_comm_overlap = True
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize, "setup_logging", Mock())
    monkeypatch.setattr(initialize, "print_rank_0", Mock())
    monkeypatch.setattr(initialize, "set_default_log_ranks", Mock())
    events = []
    for name in (
        "init_persistent_async_worker",
        "initialize_rerun_state_machine",
        "_initialize_distributed",
        "_set_random_seed",
        "_init_autoresume",
        "_compile_dependencies",
        "_initialize_tp_communicators",
    ):
        monkeypatch.setattr(
            initialize,
            name,
            Mock(side_effect=lambda *a, service=name, **kw: events.append(service)),
        )

    initialize.initialize_megatron(allow_no_cuda=True, training=True)

    assert events == [
        "init_persistent_async_worker",
        "initialize_rerun_state_machine",
        "_initialize_distributed",
        "_set_random_seed",
        "_init_autoresume",
        "_compile_dependencies",
        "_initialize_tp_communicators",
    ]


@pytest.mark.parametrize("is_vlm,is_mimo", [(False, False), (True, False), (True, True)])
def test_dynamic_server_uses_inference_checkpoint_loader(monkeypatch, is_vlm, is_mimo):
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
        vlm_dynamic_inference,
    )

    args = Namespace(load="checkpoint", inference_ckpt_non_strict=False)
    if is_mimo:
        args.mimo_checkpoint_prefix_map = {"language_model.": "language_model.module.module."}
    model = Mock()
    provider_module = ModuleType("model")
    provider_module.model_provider = Mock()
    monkeypatch.setitem(sys.modules, "model", provider_module)
    monkeypatch.setitem(sys.modules, "model_provider", provider_module)
    monkeypatch.setitem(sys.modules, "mimo_checkpoint_model", provider_module)
    gpt_builders = ModuleType("gpt_builders")
    gpt_builders.gpt_builder = Mock()
    monkeypatch.setitem(sys.modules, "gpt_builders", gpt_builders)
    monkeypatch.setattr(vlm_dynamic_inference, "get_args", lambda: args)
    monkeypatch.setattr(vlm_dynamic_inference, "_get_model", Mock(return_value=[model]))
    load = Mock()
    monkeypatch.setattr(vlm_dynamic_inference, "load_checkpoint", load)

    def check_mimo(received_args, received_model):
        load.assert_called_once_with(
            ddp_model=[model], optimizer=None, opt_param_scheduler=None, strict=True
        )
        assert received_args is args
        assert received_model is model
        model.eval.assert_not_called()

    check = Mock(side_effect=check_mimo)
    monkeypatch.setattr(vlm_dynamic_inference, "_check_mimo_checkpoint_fully_loaded", check)

    assert vlm_dynamic_inference.get_model(is_vlm=is_vlm) is model
    load.assert_called_once_with(
        ddp_model=[model], optimizer=None, opt_param_scheduler=None, strict=True
    )
    if is_mimo:
        check.assert_called_once_with(args, model)
    else:
        check.assert_not_called()
    model.eval.assert_called_once()
