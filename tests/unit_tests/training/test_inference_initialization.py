# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Inference startup must not depend on training batch or progress state."""

import sys
from argparse import ArgumentParser, Namespace
from types import ModuleType
from unittest.mock import Mock

import pytest

from megatron.inference import initialize as inference_initialize
from megatron.training import arguments, global_vars, initialize


@pytest.mark.parametrize("build_tokenizer", [False, True])
@pytest.mark.parametrize("enable_runtime_flags", [False, True])
def test_inference_services_without_training_arguments(
    monkeypatch, build_tokenizer, enable_runtime_flags
):
    args = Namespace(
        enable_experimental=enable_runtime_flags, disable_jit_fuser=enable_runtime_flags
    )
    experimental = Mock()
    jit = Mock()
    monkeypatch.setattr(inference_initialize, "set_experimental_flag", experimental)
    monkeypatch.setattr(inference_initialize, "disable_jit_fuser", jit)
    services = {}
    for name in ("_build_tokenizer", "_set_wandb_writer", "_set_telemetry"):
        services[name] = Mock()
        monkeypatch.setattr(global_vars, name, services[name])
    for name in (
        "initialize_runtime_services",
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

    inference_initialize.initialize_runtime_services_for_inference(
        args, build_tokenizer=build_tokenizer
    )

    assert services["_build_tokenizer"].call_count == int(build_tokenizer)
    services["_set_wandb_writer"].assert_called_once_with(args)
    services["_set_telemetry"].assert_called_once_with(args, include_training=False)
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


def test_inference_distributed_initialization_without_training_fields(monkeypatch):
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

    initialize.initialize_megatron(allow_no_cuda=True, inference=True)

    distributed.assert_called_once()
    seeds.assert_called_once()
    assert seeds.call_args.args[0] == args.seed


def test_inference_rejects_training_tp_overlap_before_initialization(monkeypatch):
    monkeypatch.setattr(initialize, "get_args", lambda: Namespace(tp_comm_overlap=True))
    distributed = Mock()
    monkeypatch.setattr(initialize, "_initialize_distributed", distributed)
    with pytest.raises(ValueError, match="fixed training user buffers"):
        initialize.initialize_megatron(allow_no_cuda=True, inference=True)
    distributed.assert_not_called()


@pytest.mark.parametrize("is_vlm", [False, True])
def test_dynamic_server_uses_inference_checkpoint_loader(monkeypatch, is_vlm):
    from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
        vlm_dynamic_inference,
    )

    args = Namespace(load="checkpoint", inference_ckpt_non_strict=False)
    model = Mock()
    provider_module = ModuleType("model")
    provider_module.model_provider = Mock()
    monkeypatch.setitem(sys.modules, "model", provider_module)
    monkeypatch.setitem(sys.modules, "model_provider", provider_module)
    gpt_builders = ModuleType("gpt_builders")
    gpt_builders.gpt_builder = Mock()
    monkeypatch.setitem(sys.modules, "gpt_builders", gpt_builders)
    monkeypatch.setattr(vlm_dynamic_inference, "get_args", lambda: args)
    monkeypatch.setattr(vlm_dynamic_inference, "_get_model", Mock(return_value=[model]))
    load = Mock()
    monkeypatch.setattr(vlm_dynamic_inference, "load_checkpoint_for_inference", load)

    assert vlm_dynamic_inference.get_model(is_vlm=is_vlm) is model
    load.assert_called_once_with([model], strict=True)
    model.eval.assert_called_once()
