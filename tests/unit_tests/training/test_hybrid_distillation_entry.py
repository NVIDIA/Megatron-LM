# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Entry-point contracts retained across the main/dev config migration."""

import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import torch

import pretrain_hybrid
from megatron import training
from megatron.training import argument_utils, arguments, global_vars, utils


def test_hybrid_cached_logits_loss_uses_reader_and_keeps_reporting(monkeypatch):
    args = SimpleNamespace(
        logits_load_dir="teacher-cache",
        logits_load_decode_threads=2,
        logits_load_prefetch_factor=3,
        logits_load_msc_prefetch_depth=4,
        logits_load_kd_loss_alpha=0.5,
        logits_load_ignore_errors=False,
        check_for_nan_in_loss_and_grad=False,
        check_for_spiky_loss=False,
    )
    expected = (torch.tensor(12.0), torch.tensor(3), {"kd loss": torch.tensor([12.0, 3.0])})
    reader = Mock(return_value=expected)
    factory = Mock(return_value=reader)
    monkeypatch.setattr(pretrain_hybrid, "get_args", lambda: args)
    monkeypatch.setattr(pretrain_hybrid, "get_rerun_state_machine", Mock())
    monkeypatch.setattr(pretrain_hybrid, "_build_cached_logits_loss_func", factory)
    mask, output, model = torch.ones(3), torch.randn(3), object()
    result = pretrain_hybrid.loss_func(mask, output, model=model)
    reader.assert_called_once_with(mask, output, model=model)
    factory.assert_called_once_with(
        logprobs_dir="teacher-cache",
        decode_threads=2,
        prefetch_factor=3,
        msc_prefetch_depth=4,
        kd_loss_alpha=0.5,
        ignore_errors=False,
    )
    assert all(actual is wanted for actual, wanted in zip(result, expected))


def test_multimodal_dev_entry_initializes_runtime_before_pretrain(monkeypatch):
    args = SimpleNamespace(pipeline_model_parallel_size=1, padded_vocab_size=None)
    config = SimpleNamespace(profiling=None)
    events = []
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(arguments, "parse_and_validate_args", lambda **kwargs: args)
    monkeypatch.setattr(argument_utils, "pretrain_cfg_container_from_args", lambda value: config)
    monkeypatch.setattr(
        global_vars, "set_run_config", lambda value: events.append(("config", value))
    )

    def initialize(value, training):
        assert value is args and training
        args.padded_vocab_size = 128
        events.append(("runtime", value))

    monkeypatch.setattr(global_vars, "initialize_runtime_services", initialize)
    monkeypatch.setattr(
        argument_utils,
        "resolve_tokenizer_vocab_size",
        lambda value, size: events.append(("vocab", value, size)),
    )
    monkeypatch.setattr(utils, "start_memory_history_recording", lambda value: None)

    def pretrain(value, *rest, **kwargs):
        assert value is config
        assert events == [("config", config), ("runtime", args), ("vocab", config, 128)]
        events.append(("pretrain", value))

    monkeypatch.setattr(training, "pretrain", pretrain)
    runpy.run_module("examples.multimodal_dev.pretrain_multimodal", run_name="__main__")
    assert events[-1] == ("pretrain", config)
