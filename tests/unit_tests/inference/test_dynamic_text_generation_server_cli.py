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
from megatron.core.transformer import TransformerConfig
from megatron.training import argument_utils, arguments, global_vars
from megatron.training.models import GPTModelConfig, HybridModelConfig, LLaVAModelConfig
from tools.run_dynamic_text_generation_server import add_text_generation_server_args


@pytest.mark.parametrize("family", ["gpt", "hybrid", "llava", "mimo"])
@pytest.mark.parametrize("padded_vocab_size", [None, 768])
def test_deferred_bootstrap_resolves_detected_model(monkeypatch, family, padded_vocab_size):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.num_layers, args.hidden_size, args.num_attention_heads = 2, 128, 4
    args.max_position_embeddings = 32
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 261
    args.pad_vocab_size = True
    args.padded_vocab_size = padded_vocab_size
    args.make_vocab_size_divisible_by = 128
    for name in ("_GLOBAL_RUN_CONFIG", "_GLOBAL_TOKENIZER"):
        monkeypatch.setattr(global_vars, name, None)
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    monkeypatch.setattr(legacy_server, "parse_and_validate_args", lambda **kwargs: args)
    monkeypatch.setattr(legacy_server.sys, "argv", ["server"])
    monkeypatch.setattr(legacy_server, "_print_resolved_args", lambda *args: None)
    monkeypatch.setattr(legacy_server.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        argument_utils,
        "core_transformer_config_from_args",
        lambda args: TransformerConfig(num_layers=2, hidden_size=128, num_attention_heads=4),
    )
    for name in ("_set_wandb_writer", "_set_telemetry"):
        monkeypatch.setattr(global_vars, name, lambda *args, **kwargs: None)

    def initialize_without_tokenizer(*, build_tokenizer):
        assert not build_tokenizer
        global_vars.initialize_runtime_services(args, build_tokenizer=False)
        if padded_vocab_size is None:
            with pytest.raises(ValueError, match="Model vocabulary must be resolved"):
                _ = global_vars.get_run_config().model.effective_vocab_size

    def detect(args, **kwargs):
        assert global_vars._GLOBAL_TOKENIZER is None
        if family in ("hybrid", "mimo"):
            args.hybrid_layer_pattern = "M*"
        if family == "mimo":
            args.mimo_checkpoint_prefix_map = {"language_model.": "model.llm."}
        return family in ("llava", "mimo")

    monkeypatch.setattr(legacy_server, "initialize_megatron", initialize_without_tokenizer)
    monkeypatch.setattr(legacy_server, "_detect_vlm_from_checkpoint", detect)
    _, is_vlm = legacy_server.parse_args_and_detect_vlm(
        extra_args_provider=None, args_defaults={}, user_passed_attrs=set()
    )
    cfg = global_vars.get_run_config()
    expected_type = {
        "gpt": GPTModelConfig,
        "hybrid": HybridModelConfig,
        "llava": LLaVAModelConfig,
        "mimo": HybridModelConfig,
    }[family]
    assert type(cfg.model) is expected_type
    assert is_vlm == (family in ("llava", "mimo"))
    assert global_vars.get_tokenizer().vocab_size == 261
    assert (
        cfg.model.effective_vocab_size
        == cfg.tokenizer.padded_vocab_size
        == (384 if padded_vocab_size is None else padded_vocab_size)
    )
    if family == "mimo":
        assert args.mimo_checkpoint_prefix_map == {"language_model.": "model.llm."}


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
        # Both CLIs leave these unset so the model's generation_config.json can
        # apply; a hardcoded default here would make that tier unreachable.
        ([], (None, None, None, False)),
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
        # The runner sizes the frontend from the engine's DP group. A None group
        # reads as size 1, so this exercises the floor on the replica count.
        pg_collection=SimpleNamespace(dp=None),
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
        "num_replicas": 4,
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
