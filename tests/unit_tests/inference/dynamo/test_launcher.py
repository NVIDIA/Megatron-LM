from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("dynamo")

from megatron.inference.integrations.dynamo.args import parse_args
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


def _slurm_argv():
    return [
        "--role",
        "aggregated",
        "--model",
        "model-meta",
        "--launcher",
        "slurm",
        "--nnodes",
        "2",
        "--nproc-per-node",
        "4",
        "--master-addr",
        "node-0",
        "--master-port",
        "29500",
        "--slurm-nodelist",
        "node-[0-1]",
        "--parent-event-host",
        "10.0.0.10",
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


@pytest.mark.parametrize(
    "argv",
    [
        _argv()[:-6] + ["--nnodes", "2", "--", "--load", "/checkpoint"],
        [
            "--model",
            "model-meta",
            "--launcher",
            "slurm",
            "--nnodes",
            "2",
            "--nproc-per-node",
            "4",
            "--master-addr",
            "node-0",
            "--master-port",
            "29500",
            "--",
            "--load",
            "/checkpoint",
        ],
    ],
)
def test_multi_node_launcher_configuration_is_validated(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


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


def test_slurm_engine_command_launches_one_torchrun_agent_per_node():
    engine = MegatronLLMEngine(parse_args(_slurm_argv()))
    command = engine._engine_command("tcp://10.0.0.10:5556")

    assert command[:5] == [
        "srun",
        "--nodes=2",
        "--ntasks=2",
        "--ntasks-per-node=1",
        "--gpus-per-node=4",
    ]
    assert "--kill-on-bad-exit=1" in command
    assert "--nodelist=node-[0-1]" in command
    assert command[-3:-1] == ["bash", "-c"]
    payload = command[-1]
    assert "--nnodes=2" in payload
    assert "--nproc-per-node=4" in payload
    assert '--node-rank="${SLURM_NODEID}"' in payload
    assert "--master-addr=node-0" in payload
    assert "--master-port=29500" in payload
    assert "megatron.inference.integrations.dynamo.engine_service" in payload
    assert "'/checkpoints/model path'" in payload


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
async def test_readiness_message_is_received():
    engine = MegatronLLMEngine(parse_args(_argv()))
    expected = {"coordinator_address": "tcp://127.0.0.1:5000"}
    engine._process = SimpleNamespace(returncode=None)
    engine._on_engine_event("ready", expected)
    assert await engine._wait_for_readiness() == expected
