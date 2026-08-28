# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the benchmark example."""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from megatron.lite.runtime.contracts.config import ParallelConfig
from megatron.lite.runtime.contracts.data import ForwardResult, ModelOutputs
from megatron.lite.runtime.contracts.handle import ModelHandle

pytestmark = pytest.mark.optional

_LITE_ROOT = str(Path(__file__).resolve().parents[3])
sys.path = [path for path in sys.path if path != _LITE_ROOT]
sys.path.insert(0, _LITE_ROOT)


def test_bench_builds_mlite_runtime_config_with_model_hook():
    from examples.bench.bench import BenchCliConfig, build_runtime_config
    from megatron.lite.model.qwen3_5.config import Qwen35Config
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig

    cfg = BenchCliConfig(
        backend="mlite",
        hf_path="/tmp/hf",
        model_name="qwen3_5",
        use_thd=True,
        truncate_layers=2,
        disable_mtp=True,
    )

    runtime_cfg = build_runtime_config(cfg)

    assert runtime_cfg.backend == "mlite"
    assert isinstance(runtime_cfg.backend_cfg, MegatronLiteConfig)
    assert runtime_cfg.backend_cfg.impl_cfg["use_thd"] is True
    assert callable(runtime_cfg.backend_cfg.model_config_hook)

    model_cfg = runtime_cfg.backend_cfg.model_config_hook(Qwen35Config())
    assert model_cfg.num_hidden_layers == 2
    assert len(model_cfg.layer_types) == 2
    assert model_cfg.num_nextn_predict_layers == 0


def test_bench_mlite_deterministic_mounts_native_vision_not_mbridge(monkeypatch):
    from examples.bench.bench import BenchCliConfig, build_runtime_config

    monkeypatch.setenv("MEGATRON_LITE_DETERMINISTIC", "1")

    runtime_cfg = build_runtime_config(
        BenchCliConfig(backend="mlite", hf_path="/tmp/hf", model_name="qwen3_5")
    )

    impl_cfg = runtime_cfg.backend_cfg.impl_cfg
    assert impl_cfg["mount_vision_model"] is True
    assert ("mount_" + "mbridge_vision_model") not in impl_cfg


def test_bench_builds_bridge_dry_run_plan_without_bridge_import():
    from examples.bench.bench import BenchCliConfig, build_dry_run_plan

    plan = build_dry_run_plan(
        BenchCliConfig(
            backend="bridge",
            hf_path="/tmp/hf",
            model_name="qwen3_5",
            truncate_layers=2,
            override_transformer_json='{"attention_backend": "unfused"}',
            dry_run=True,
        )
    )

    assert plan["dry_run"] is True
    assert plan["runtime"]["backend"] == "bridge"
    backend_cfg = plan["runtime"]["backend_cfg"]
    assert backend_cfg["model_name"] == "qwen3_5"
    assert backend_cfg["override_transformer_config"] == {"attention_backend": "unfused"}
    assert backend_cfg["bridge_post_init"].startswith("<callable:")


def test_bench_builds_mbridge_dry_run_plan_without_mbridge_import():
    from examples.bench.bench import BenchCliConfig, build_dry_run_plan

    plan = build_dry_run_plan(
        BenchCliConfig(
            backend="mbridge",
            hf_path="/tmp/hf",
            model_name="qwen3_5",
            truncate_layers=2,
            override_transformer_json='{"attention_backend": "unfused"}',
            dry_run=True,
        )
    )

    assert plan["dry_run"] is True
    assert plan["runtime"]["backend"] == "mbridge"
    backend_cfg = plan["runtime"]["backend_cfg"]
    assert backend_cfg["model_name"] == "qwen3_5"
    assert backend_cfg["override_transformer_config"] == {"attention_backend": "unfused"}
    assert backend_cfg["bridge_post_init"].startswith("<callable:")


def test_qwen35_lite_sources_use_native_vision_not_mbridge_anchor():
    root = Path(__file__).resolve().parents[3]
    protocol = root / "megatron/lite/model/qwen3_5/lite/protocol.py"
    model = root / "megatron/lite/model/qwen3_5/lite/model.py"

    protocol_text = protocol.read_text(encoding="utf-8")
    model_text = model.read_text(encoding="utf-8")

    assert "mount_vision_model" in protocol_text
    assert "_build_native_vision_model" in model_text
    forbidden = (
        "mount_" + "mbridge_vision_model",
        "_build_" + "mbridge_for_vision_anchor",
        "mbridge_" + "bridge",
        "from mbridge import",
        "megatron.bridge",
    )
    for item in forbidden:
        assert item not in protocol_text
        assert item not in model_text


class _FakeRuntime:
    def __init__(self):
        self.loss = 0

    def train_mode(self, handle):
        return nullcontext()

    def zero_grad(self, handle) -> None:
        pass

    def forward_backward(
        self,
        handle,
        data,
        loss_fn,
        *,
        num_microbatches: int = 1,
        forward_only: bool = False,
    ):
        self.loss += 1
        return ForwardResult(model_output=ModelOutputs(loss=torch.tensor(float(self.loss))))

    def optimizer_step(self, handle):
        return True, 3.5, 0

    def lr_scheduler_step(self, handle):
        return 0.0


def test_pretrain_session_runs_with_fake_runtime_on_cpu():
    from examples.bench.session import PretrainSessionConfig, run_pretrain_session

    handle = ModelHandle(
        model=object(),
        optimizer=object(),
        parallel_state=None,
        config=type(
            "Cfg", (), {"model_name": "fake", "impl": "lite", "parallel": ParallelConfig()}
        )(),
        _extras={"optimizer_backend": "fake"},
    )

    result = run_pretrain_session(
        _FakeRuntime(),
        handle,
        PretrainSessionConfig(steps=3, warmup=1, device="cpu", seq_len=4),
        data_iter=iter([{}, {}, {}]),
    )

    assert result.backend == "mlite"
    assert result.seq_len == 4
    assert result.num_microbatches == 1
    assert len(result.step_traces) == 2
    assert [trace.loss for trace in result.step_traces] == [2.0, 3.0]
    assert result.step_traces[0].grad_norm == 3.5
    assert result.step_traces[0].peak_allocated_bytes == 0
    assert result.step_traces[0].post_reserved_bytes == 0
    assert result.metadata["memory_gate"] == {
        "all_rank_max": True,
        "max_steady_peak_growth": 0.0,
        "limit": 0.02,
        "passed": True,
        "empty_cache_between_steps": False,
    }


def test_no_optimizer_grad_norm_reports_real_gradients_without_mutation():
    from examples.bench.session import _global_grad_norm_without_step

    model = torch.nn.Linear(3, 2, bias=False)
    model.weight.grad = torch.full_like(model.weight, 2.0)
    original = model.weight.grad.clone()
    handle = ModelHandle(model=model, _extras={"model_chunks": [model]})

    assert _global_grad_norm_without_step(handle) == pytest.approx(24.0**0.5)
    torch.testing.assert_close(model.weight.grad, original, rtol=0, atol=0)


def test_bench_main_writes_dry_run_output_json(tmp_path):
    from examples.bench.bench import main

    output_path = tmp_path / "dry_run.json"

    artifact = main(
        [
            "--backend",
            "mlite",
            "--hf-path",
            "/tmp/hf",
            "--model-name",
            "qwen3_5",
            "--truncate-layers",
            "2",
            "--disable-mtp",
            "--dry-run",
            "--output-json",
            str(output_path),
        ]
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text()) == artifact


def test_bench_main_writes_output_json_only_on_rank_zero(tmp_path, monkeypatch):
    from examples.bench.bench import main

    output_path = tmp_path / "rank_one.json"
    monkeypatch.setenv("RANK", "1")

    artifact = main(
        [
            "--backend",
            "mlite",
            "--hf-path",
            "/tmp/hf",
            "--model-name",
            "qwen3_5",
            "--dry-run",
            "--output-json",
            str(output_path),
        ]
    )

    assert artifact["dry_run"] is True
    assert not output_path.exists()


def test_benchmark_snapshot_records_semantic_and_profiling_configuration(monkeypatch):
    from examples.bench.bench import BenchCliConfig, _benchmark_config_snapshot

    monkeypatch.setenv("MLITE_VLLM_BATCHED_GROUPED_WEIGHT_QUANT", "0")
    monkeypatch.setenv("MLITE_PROFILE_SYNC_PHASES", "1")
    snapshot = _benchmark_config_snapshot(
        BenchCliConfig(
            model_name="deepseek_v4",
            impl="vllm",
            ep=8,
            steps=10,
            warmup=5,
            seq_len=16384,
            skip_load_hf_weights=True,
            trace_fingerprints=True,
            impl_cfg_json=(
                '{"optimizer":"fsdp2","recompute":["full"],'
                '"cache_deployment_weights":false}'
            ),
        )
    )

    assert snapshot["load_hf_weights"] is False
    assert snapshot["optimizer_backend"] == "fsdp2"
    assert snapshot["parallel"]["ep"] == 8
    assert snapshot["schedule"]["warmup"] == 5
    assert snapshot["recompute"] == ["full"]
    assert snapshot["cache_deployment_weights"] is False
    assert snapshot["correctness"]["trace_fingerprints"] is True
    assert snapshot["fp8"]["batched_grouped_weight_quant"] == "0"
    assert snapshot["profiling"]["sync_phases"] is True


def test_result_summary_records_allocated_reserved_and_active_memory():
    from examples.bench.results import RunResult, StepTrace

    result = RunResult(
        backend="mlite",
        model_name="deepseek_v4",
        impl="vllm",
        optimizer_backend="fsdp2",
        tp=1,
        etp=None,
        ep=4,
        pp=1,
        vpp=1,
        cp=1,
        seq_len=16,
        num_microbatches=1,
        step_traces=[
            StepTrace(
                step=0,
                loss=1.0,
                grad_norm=2.0,
                step_ms=3.0,
                peak_reserved_bytes=12_000_000_000,
                post_allocated_bytes=10_000_000_000,
                post_reserved_bytes=11_000_000_000,
                active_bytes=9_000_000_000,
            )
        ],
        peak_mem_gb=8.0,
    )

    summary = result.summary_dict()
    assert summary["peak_mem_gb"] == 8.0
    assert summary["peak_reserved_gb"] == 12.0
    assert summary["post_allocated_gb"] == 10.0
    assert summary["post_reserved_gb"] == 11.0
    assert summary["active_gb"] == 9.0


def test_result_artifact_summary_and_trace_compare(tmp_path):
    from examples.bench.results import compare_step_traces, load_result_artifact, result_summary

    baseline = {
        "summary": {
            "backend": "mlite",
            "avg_step_ms": 10.0,
            "tok_per_s": 3200.0,
            "steps_measured": 2,
        },
        "result": {
            "step_traces": [
                {"step": 0, "loss": 1.0, "grad_norm": 2.0, "step_ms": 10.0},
                {"step": 1, "loss": 1.5, "grad_norm": 2.5, "step_ms": 10.0},
            ]
        },
    }
    candidate = {
        "summary": {
            "backend": "bridge",
            "avg_step_ms": 11.0,
            "tok_per_s": 2900.0,
            "steps_measured": 2,
        },
        "result": {
            "step_traces": [
                {"step": 0, "loss": 1.00001, "grad_norm": 2.00001, "step_ms": 11.0},
                {"step": 1, "loss": 1.49999, "grad_norm": 2.49999, "step_ms": 11.0},
            ]
        },
    }
    baseline_path = tmp_path / "mlite.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    loaded = load_result_artifact(baseline_path)

    assert result_summary(loaded)["backend"] == "mlite"
    assert compare_step_traces(baseline, candidate, atol=1e-3, rtol=0.0)["passed"] is True


def test_result_trace_compare_reports_metric_level_failures():
    from examples.bench.results import compare_step_traces

    baseline = {
        "result": {"step_traces": [{"step": 0, "loss": 1.0, "grad_norm": 2.0, "step_ms": 10.0}]}
    }
    candidate = {
        "result": {"step_traces": [{"step": 0, "loss": 1.00001, "grad_norm": 3.0, "step_ms": 10.0}]}
    }

    comparison = compare_step_traces(baseline, candidate, atol=1e-3, rtol=0.0)

    assert comparison["passed"] is False
    assert comparison["loss_passed"] is True
    assert comparison["grad_norm_passed"] is False


def test_result_trend_compare_allows_offsets_but_rejects_opposite_direction():
    from examples.bench.results import compare_step_trends

    def artifact(losses, grad_norms):
        return {
            "result": {
                "step_traces": [
                    {
                        "step": step,
                        "loss": loss,
                        "grad_norm": grad_norm,
                        "step_ms": 1.0,
                    }
                    for step, (loss, grad_norm) in enumerate(
                        zip(losses, grad_norms, strict=True)
                    )
                ]
            }
        }

    baseline = artifact(
        [20.0, 19.8, 19.7, 19.4, 19.2],
        [10.0, 10.2, 10.1, 10.5, 10.8],
    )
    aligned = artifact(
        [21.0, 20.7, 20.55, 20.1, 19.8],
        [8.0, 8.3, 8.15, 8.75, 9.2],
    )
    opposite = artifact(
        [21.0, 21.2, 21.3, 21.6, 21.8],
        [8.0, 7.8, 7.9, 7.5, 7.2],
    )

    assert compare_step_trends(baseline, aligned)["passed"] is True
    assert compare_step_trends(baseline, opposite)["passed"] is False


def test_correctness_compare_requires_bitwise_fields():
    from examples.bench.results import compare_correctness_artifacts

    baseline = {
        "eval_logits": {"sha256": "a", "shape": [1], "dtype": "torch.bfloat16"},
        "steps": [
            {
                "loss": {"value": 1.0, "float_hex": (1.0).hex()},
                "logits": {"sha256": "b"},
                "grad_fingerprint": {"sha256": "c", "tensor_count": 1},
                "grad_norm": {"value": 2.0, "float_hex": (2.0).hex()},
                "update_successful": True,
                "num_zeros": 0,
                "post_step_weights": {"sha256": "d", "tensor_count": 1},
            }
        ],
    }
    candidate = json.loads(json.dumps(baseline))

    assert compare_correctness_artifacts(baseline, candidate)["passed"] is True

    candidate["steps"][0]["grad_norm"] = {"value": 2.5, "float_hex": (2.5).hex()}
    comparison = compare_correctness_artifacts(baseline, candidate)

    assert comparison["passed"] is False
    assert comparison["max_grad_norm_abs"] == 0.5

    candidate = json.loads(json.dumps(baseline))
    candidate["steps"][0]["grad_fingerprint"]["sha256"] = "different"
    comparison = compare_correctness_artifacts(baseline, candidate)

    assert comparison["passed"] is False
    assert any(
        mismatch["step"] == 0 and mismatch["field"] == "grad_fingerprint"
        for mismatch in comparison["mismatches"]
    )


def test_correctness_compare_supports_explicit_numeric_tolerances():
    from examples.bench.results import compare_correctness_artifacts

    tensor_a = {"sha256": "a", "shape": [2], "values": [1.0, 2.0]}
    tensor_b = {"sha256": "b", "shape": [2], "values": [1.001, 1.998]}
    baseline = {
        "eval_logits": tensor_a,
        "steps": [
            {
                "loss": {"value": 1.0},
                "logits": tensor_a,
                "grad_norm": {"value": 2.0},
                "post_step_weights": None,
                "update_successful": True,
                "num_zeros": 0,
            }
        ],
    }
    candidate = {
        "eval_logits": tensor_b,
        "steps": [
            {
                "loss": {"value": 1.000001},
                "logits": tensor_b,
                "grad_norm": {"value": 2.00001},
                "post_step_weights": None,
                "update_successful": True,
                "num_zeros": 0,
            }
        ],
    }

    comparison = compare_correctness_artifacts(
        baseline,
        candidate,
        loss_atol=1e-5,
        grad_atol=1e-4,
        tensor_atol=3e-3,
    )

    assert comparison["passed"] is True
    assert abs(comparison["max_tensor_abs"] - 2e-3) < 1e-12


def test_correctness_fixed_routes_are_deterministic_and_unique():
    from types import SimpleNamespace

    from examples.bench.correctness import _fixed_route_batches
    from megatron.lite.runtime.contracts.data import PackedBatch

    batch = PackedBatch(
        input_ids=torch.arange(8),
        labels=torch.arange(8),
        seq_lens=torch.tensor([3, 5]),
    )
    handle = SimpleNamespace(
        _extras={
            "model_cfg": SimpleNamespace(
                num_hidden_layers=4,
                num_experts_per_tok=6,
                n_routed_experts=256,
            )
        }
    )

    routed_batch = next(_fixed_route_batches(iter([batch]), handle))
    rows = list(routed_batch.routed_experts.unbind())

    assert [tuple(row.shape) for row in rows] == [(3, 4, 6), (5, 4, 6)]
    assert all(torch.equal(row, row.remainder(256)) for row in rows)
    assert all(torch.all(row[..., 1:] != row[..., :-1]) for row in rows)
    assert routed_batch.r3_replay_mask.tolist() == [True] * 8
