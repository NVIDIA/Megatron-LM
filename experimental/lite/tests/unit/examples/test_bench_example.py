# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for the benchmark example."""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

from megatron.lite.runtime.contracts.config import ParallelConfig
from megatron.lite.runtime.contracts.data import ForwardResult
from megatron.lite.runtime.contracts.handle import ModelHandle

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

    def forward_backward(self, handle, data, loss_fn, *, num_microbatches: int = 1):
        self.loss += 1
        return ForwardResult(metrics={"loss": float(self.loss)})

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
