# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for configurable sampling defaults on inference server CLIs."""

from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import examples.inference.launch_inference_server as high_level_server
import tools.run_dynamic_text_generation_server as legacy_server
from examples.inference.launch_inference_server import add_serve_args
from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from tools.run_dynamic_text_generation_server import add_text_generation_server_args


@pytest.mark.parametrize(
    ("add_args", "required_args"),
    [
        (add_serve_args, []),
        (
            add_text_generation_server_args,
            ["--language-model-type", "placeholder", "--tokenizer-prompt-format", "mistral"],
        ),
    ],
)
@pytest.mark.parametrize(
    ("serve_args", "expected_defaults"),
    [
        ([], (1.0, 1.0, 0, False)),
        (
            [
                "--default-temperature",
                "0.4",
                "--default-top-p",
                "0.8",
                "--default-top-k",
                "5",
                "--eval-mode",
            ],
            (0.4, 0.8, 5, True),
        ),
    ],
)
def test_sampling_default_flags(add_args, required_args, serve_args, expected_defaults):
    parser = add_args(ArgumentParser())
    args = parser.parse_args([*required_args, *serve_args])

    assert (
        args.default_temperature,
        args.default_top_p,
        args.default_top_k,
        args.eval_mode,
    ) == expected_defaults


@pytest.mark.asyncio
async def test_high_level_runner_builds_sampling_serve_config(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

        async def serve(self, serve_config, blocking):
            captured["serve_config"] = serve_config
            captured["blocking"] = blocking

    monkeypatch.setattr(high_level_server, "MegatronAsyncLLM", FakeLLM)

    args = Namespace(
        coordinator_host="coordinator.example.com",
        coordinator_port=5555,
        host="127.0.0.1",
        port=4321,
        parsers=["json"],
        verbose=True,
        frontend_replicas=2,
        default_temperature=0.4,
        default_top_p=0.8,
        default_top_k=5,
        eval_mode=True,
    )
    model = object()
    tokenizer = object()
    inference_config = object()

    await high_level_server._serve(args, model, tokenizer, inference_config)

    assert captured["llm_kwargs"] == {
        "model": model,
        "tokenizer": tokenizer,
        "inference_config": inference_config,
        "use_coordinator": True,
        "coordinator_host": "coordinator.example.com",
        "coordinator_port": 5555,
    }
    assert captured["serve_config"] == high_level_server.ServeConfig(
        host="127.0.0.1",
        port=4321,
        parsers=["json"],
        verbose=True,
        frontend_replicas=2,
        default_temperature=0.4,
        default_top_p=0.8,
        default_top_k=5,
        eval_mode=True,
    )
    assert captured["blocking"] is True


@pytest.mark.asyncio
async def test_legacy_runner_forwards_sampling_config_and_stops_frontend(monkeypatch):
    captured = {}
    tokenizer = object()
    multimodal_prompt_config = object()

    async def finished_engine_loop():
        return None

    engine = SimpleNamespace(
        start_listening_to_data_parallel_coordinator=AsyncMock(return_value="tcp://coord:5555"),
        controller=SimpleNamespace(
            tokenizer=tokenizer,
            inference_wrapped_model=SimpleNamespace(
                multimodal_prompt_config=multimodal_prompt_config
            ),
        ),
        engine_loop_task=finished_engine_loop(),
        # The frontend hashes on the engine's block boundaries, so the runner reads
        # both off the engine's context rather than taking them as flags.
        context=SimpleNamespace(
            block_size_tokens=256,
            prefix_caching_coordinator_policy=PrefixCachingCoordinatorPolicy.LONGEST_PREFIX,
        ),
    )

    monkeypatch.setattr(legacy_server.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        legacy_server,
        "args",
        Namespace(parsers=["json"], inference_text_gen_server_logging=True),
        raising=False,
    )
    monkeypatch.setattr(
        legacy_server, "start_text_gen_server", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(
        legacy_server, "stop_text_gen_server", lambda: captured.update(frontend_stopped=True)
    )

    await legacy_server.run_text_generation_server(
        engine=engine,
        coordinator_port=5555,
        server_port=4321,
        hostname="127.0.0.1",
        chat_template="template",
        default_temperature=0.4,
        default_top_p=0.8,
        default_top_k=5,
        eval_mode=True,
    )

    engine.start_listening_to_data_parallel_coordinator.assert_awaited_once_with(
        inference_coordinator_port=5555, launch_inference_coordinator=True, hostname="127.0.0.1"
    )
    assert captured == {
        "coordinator_addr": "tcp://coord:5555",
        "tokenizer": tokenizer,
        "parsers": ["json"],
        "rank": 0,
        "server_port": 4321,
        "verbose": True,
        "hostname": "127.0.0.1",
        "chat_template": "template",
        "multimodal_prompt_config": multimodal_prompt_config,
        "default_temperature": 0.4,
        "default_top_p": 0.8,
        "default_top_k": 5,
        "eval_mode": True,
        # Read off the engine's context so the frontend hashes on the same block
        # boundaries the engine caches on.
        "block_size_tokens": 256,
        "prefix_caching_coordinator_policy": PrefixCachingCoordinatorPolicy.LONGEST_PREFIX,
        "frontend_stopped": True,
    }
