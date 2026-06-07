"""Unit tests for the benchmark example."""

from __future__ import annotations

from contextlib import nullcontext

from megatron.lite.runtime.contracts.config import ParallelConfig
from megatron.lite.runtime.contracts.data import ForwardResult
from megatron.lite.runtime.contracts.handle import ModelHandle


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


def test_bench_builds_bridge_dry_run_plan_without_mbridge_import():
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
            "Cfg",
            (),
            {
                "model_name": "fake",
                "impl": "lite",
                "parallel": ParallelConfig(),
            },
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
