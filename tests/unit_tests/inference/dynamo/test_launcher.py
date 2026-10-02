# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("dynamo")

from megatron.core.inference.disaggregation.engine import StateHandoffDynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.inference.integrations.dynamo import engine_service
from megatron.inference.integrations.dynamo.args import parse_args
from megatron.inference.integrations.dynamo.dynamic_engine import DynamoDynamicInferenceEngine
from megatron.inference.integrations.dynamo.llm_engine import MegatronLLMEngine
from megatron.inference.integrations.dynamo.main import main


def _argv():
    return [
        "--role",
        "aggregated",
        "--model",
        "model-meta",
        "--nproc-per-node",
        "2",
        "--",
        "--load",
        "/checkpoints/model",
        "--tensor-model-parallel-size",
        "2",
    ]


def _external_argv():
    return [
        "--role",
        "aggregated",
        "--model",
        "model-meta",
        "--engine-launch-mode",
        "external",
        "--parent-event-host",
        "node-0",
        "--parent-event-port",
        "5556",
        "--",
        "--load",
        "/checkpoints/model path",
    ]


def test_parse_args_splits_dynamo_and_megatron_arguments():
    config = parse_args(_argv())
    assert config.component == "backend"
    assert config.endpoint_types == "chat,completions"
    assert config.nproc_per_node == 2
    assert config.megatron_argv == [
        "--load",
        "/checkpoints/model",
        "--tensor-model-parallel-size",
        "2",
    ]


def test_local_launch_requires_process_count():
    with pytest.raises(SystemExit):
        parse_args(["--model", "model-meta", "--", "--load", "/checkpoint"])


def test_external_launch_requires_fixed_parent_event_port():
    argv = _external_argv()
    del argv[argv.index("--parent-event-port") : argv.index("--parent-event-port") + 2]

    with pytest.raises(SystemExit):
        parse_args(argv)


def test_external_launch_accepts_deployment_managed_engine():
    config = parse_args(_external_argv())

    assert config.engine_launch_mode == "external"
    assert config.nproc_per_node is None
    assert config.parent_event_host == "node-0"
    assert config.parent_event_port == 5556


def test_disaggregated_role_requires_coordinator_address():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--role",
                "prefill",
                "--model",
                "model-meta",
                "--nproc-per-node",
                "1",
                "--",
                "--load",
                "/checkpoint",
            ]
        )


def test_public_entrypoint_uses_common_runner():
    with patch("megatron.inference.integrations.dynamo.main.run") as run:
        main()
    run.assert_called_once_with(MegatronLLMEngine)


def test_owned_engine_command_targets_megatron_only_service():
    config = parse_args(_argv())
    engine = MegatronLLMEngine(config)
    command = engine._engine_command("tcp://127.0.0.1:5556")

    assert command[1:4] == ["-m", "torch.distributed.run", "--standalone"]
    assert "--nproc-per-node=2" in command
    assert "megatron.inference.integrations.dynamo.engine_service" in command
    assert command[command.index("--dynamo-parent-event-address") + 1] == ("tcp://127.0.0.1:5556")
    assert "dynamo.megatron" not in command
    assert command[-4:] == ["--load", "/checkpoints/model", "--tensor-model-parallel-size", "2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role,backend",
    [
        ("aggregated", "nccl"),
        ("prefill", "nccl"),
        ("decode", "nccl"),
        ("prefill", "nixl"),
        ("decode", "nixl"),
    ],
)
async def test_engine_service_validates_handoff_and_log_probs_before_construction(
    monkeypatch, role, backend
):
    args = SimpleNamespace(
        return_log_probs=False,
        skip_prompt_log_probs=False,
        role=role,
        disagg_kv_transport_backend=backend,
    )

    def build_engine(*, engine_class):
        assert engine_class is not None
        assert args.return_log_probs
        assert args.skip_prompt_log_probs
        raise RuntimeError("configuration observed")

    monkeypatch.setattr(engine_service, "get_args", lambda: args)
    monkeypatch.setattr(engine_service, "get_dynamic_inference_engine", build_engine)

    if role != "aggregated" and backend == "nccl":
        with pytest.raises(ValueError, match="require --disagg-kv-transport-backend nixl"):
            await engine_service._serve()
    else:
        with pytest.raises(RuntimeError, match="configuration observed"):
            await engine_service._serve()


@pytest.mark.asyncio
async def test_from_args_resolves_only_registration_metadata(monkeypatch):
    async def fail_create_subprocess(*args, **kwargs):
        raise AssertionError("from_args must not start a process")

    monkeypatch.setattr("asyncio.create_subprocess_exec", fail_create_subprocess)
    fetch_model = AsyncMock(return_value="/cache/model-meta")
    monkeypatch.setattr(
        "megatron.inference.integrations.dynamo.llm_engine.fetch_model", fetch_model
    )
    engine, worker = await MegatronLLMEngine.from_args(_argv())

    assert engine._process is None
    assert engine.client is None
    assert worker.component == "backend"
    assert worker.model_name == "/cache/model-meta"
    assert engine.registration_model == "/cache/model-meta"
    fetch_model.assert_awaited_once_with("model-meta", ignore_weights=True)


@pytest.mark.asyncio
async def test_from_args_preserves_local_registration_model(tmp_path, monkeypatch):
    fetch_model = AsyncMock()
    monkeypatch.setattr(
        "megatron.inference.integrations.dynamo.llm_engine.fetch_model", fetch_model
    )
    argv = _argv()
    argv[argv.index("model-meta")] = str(tmp_path)
    argv[argv.index("--nproc-per-node") : argv.index("--nproc-per-node")] = [
        "--endpoint-types",
        "completions",
    ]

    engine, worker = await MegatronLLMEngine.from_args(argv)

    assert worker.model_name == str(tmp_path.resolve())
    assert worker.endpoint_types == "completions"
    assert engine.registration_model == str(tmp_path.resolve())
    fetch_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_readiness_reports_early_child_failure():
    engine = MegatronLLMEngine(parse_args(_argv()))
    engine._process = SimpleNamespace(returncode=17)
    with pytest.raises(RuntimeError, match="exited before readiness.*17"):
        await engine._wait_for_readiness()


@pytest.mark.asyncio
async def test_external_readiness_does_not_require_child_process():
    engine = MegatronLLMEngine(parse_args(_external_argv()))
    expected = {"coordinator_address": "tcp://127.0.0.1:5000"}

    async def report_ready():
        await asyncio.sleep(0)
        engine._on_engine_event("ready", expected)

    task = asyncio.create_task(report_ready())
    assert await engine._wait_for_readiness() == expected
    await task


def test_progress_requires_successful_running_engine_scheduling():
    engine = object.__new__(DynamoDynamicInferenceEngine)
    engine.rank = 0
    engine.state = EngineState.RUNNING
    engine._last_progress_time = 0.0
    report = MagicMock()
    engine.set_progress_callback(report)
    with patch.object(StateHandoffDynamicInferenceEngine, "schedule_requests", return_value=3):
        assert engine.schedule_requests() == 3
        report.assert_called_once()
        engine.state = EngineState.PAUSED
        engine.schedule_requests()
        report.assert_called_once()
    engine.state = EngineState.RUNNING
    with patch.object(
        StateHandoffDynamicInferenceEngine, "schedule_requests", side_effect=RuntimeError("stalled")
    ):
        with pytest.raises(RuntimeError, match="stalled"):
            engine.schedule_requests()
    report.assert_called_once()
