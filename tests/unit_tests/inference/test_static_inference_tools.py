# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the static-engine setup in the inference tools."""

from argparse import ArgumentParser
from types import SimpleNamespace

import pytest

import tools.run_inference_performance_test as performance_test_tool
import tools.run_text_generation_server as text_generation_server_tool
from megatron.training.arguments import _add_inference_args


@pytest.mark.parametrize("tool", [text_generation_server_tool, performance_test_tool])
def test_static_engine_reads_inference_args(tool, monkeypatch):
    """The static context must be sized from the args that `_add_inference_args` registers."""
    captured = {}

    def fake_static_context(max_batch_size, max_sequence_length):
        captured["context_args"] = (max_batch_size, max_sequence_length)
        return "context"

    monkeypatch.setattr(tool, "get_tokenizer", lambda: "tokenizer")
    monkeypatch.setattr(tool, "StaticInferenceContext", fake_static_context)
    monkeypatch.setattr(tool, "GPTInferenceWrapper", lambda model, context: SimpleNamespace())
    monkeypatch.setattr(tool, "TextGenerationController", lambda **kwargs: "controller")
    monkeypatch.setattr(
        tool, "StaticInferenceEngine", lambda text_generation_controller: "static_engine"
    )
    monkeypatch.setattr(
        tool,
        "mpu",
        SimpleNamespace(is_pipeline_first_stage=lambda: True, is_pipeline_last_stage=lambda: True),
    )

    parser = _add_inference_args(ArgumentParser())
    args = parser.parse_args(["--inference-max-requests", "3", "--inference-max-seq-length", "512"])
    args.engine_type = "static"

    assert tool.get_inference_engine(args, model="model") == "static_engine"
    assert captured["context_args"] == (3, 512)
