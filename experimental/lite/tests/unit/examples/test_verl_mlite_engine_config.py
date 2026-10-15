# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))


from megatron.lite.runtime.contracts import LossContext

pytestmark = pytest.mark.optional


@pytest.fixture(autouse=True)
def _require_verl() -> None:
    pytest.importorskip("verl", reason="VERL is required for this optional example test.")


def _optimizer_config(**override_optimizer_config) -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="adam",
        lr=1e-6,
        min_lr=None,
        min_lr_ratio=None,
        clip_grad=1.0,
        weight_decay=0.1,
        lr_warmup_steps_ratio=0.0,
        total_training_steps=10,
        lr_warmup_steps=0,
        lr_warmup_init=0.0,
        lr_decay_steps=None,
        lr_decay_style="constant",
        weight_decay_incr_style="constant",
        lr_wsd_decay_style="exponential",
        lr_wsd_decay_steps=None,
        use_checkpoint_opt_param_scheduler=False,
        betas=(0.9, 0.95),
        override_optimizer_config=override_optimizer_config,
    )


def _engine(*, engine_config: object, optimizer_config: SimpleNamespace | None = None):
    from verl_mlite.engine.mlite_engine import MegatronLiteEngine

    return MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path="/tmp/qwen35", hf_config={"model_type": "qwen3_5_moe"}, mtp=None
        ),
        engine_config=engine_config,
        optimizer_config=optimizer_config or _optimizer_config(),
        checkpoint_config={},
    )


def _engine_config(**kwargs):
    from verl_mlite.engine.config import MegatronLiteEngineConfig

    values = {"custom_backend_module": None, "impl_cfg": {"use_thd": True}}
    values.update(kwargs)
    return MegatronLiteEngineConfig(**values)


def test_compile_cache_isolation_prefers_global_rank_and_preserves_roots(
    tmp_path, monkeypatch, capsys
) -> None:
    from verl_mlite.engine.mlite_engine import _isolate_compile_cache_per_rank

    roots = {
        "TORCHINDUCTOR_CACHE_DIR": tmp_path / "torchinductor",
        "TRITON_CACHE_DIR": tmp_path / "triton",
    }
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "0")
    for name, root in roots.items():
        monkeypatch.setenv(name, str(root))
        monkeypatch.delenv(f"VERL_MLITE_BASE_{name}", raising=False)

    _isolate_compile_cache_per_rank()
    _isolate_compile_cache_per_rank()
    for name, root in roots.items():
        assert Path(os.environ[name]) == root / "rank_7"

    monkeypatch.setenv("RANK", "8")
    _isolate_compile_cache_per_rank()
    for name, root in roots.items():
        assert Path(os.environ[name]) == root / "rank_8"

    output = capsys.readouterr().out
    assert "[compile-cache] rank=7" in output
    assert "[compile-cache] rank=8" in output


def test_compile_cache_isolation_ignores_unconfigured_cache_vars(monkeypatch, capsys) -> None:
    from verl_mlite.engine.mlite_engine import _isolate_compile_cache_per_rank

    monkeypatch.setenv("RANK", "3")
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(f"VERL_MLITE_BASE_{name}", raising=False)

    _isolate_compile_cache_per_rank()

    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        assert name not in os.environ
        assert f"VERL_MLITE_BASE_{name}" not in os.environ
    assert "TORCHINDUCTOR_CACHE_DIR: None -> None" in capsys.readouterr().out


@pytest.mark.parametrize("num_microbatches", [1, 4])
def test_verl_loss_hook_preserves_gradient_and_micro_outputs(num_microbatches):
    engine = _engine(engine_config=_engine_config())
    weight = torch.nn.Parameter(torch.tensor(1.0))
    outputs = []
    engine._build_verl_model_output = lambda **_kwargs: {"log_probs": weight * 3}
    engine.get_data_parallel_group = lambda: None

    hook = engine._make_runtime_loss_fn(
        lambda model_output, **_kwargs: (model_output["log_probs"] / num_microbatches, {}),
        num_microbatches=num_microbatches,
        output_lst=outputs,
    )
    for _ in range(num_microbatches):
        loss, _ = hook({}, object(), LossContext(source_batch=object()))
        (loss / num_microbatches).backward()

    torch.testing.assert_close(weight.grad, torch.tensor(3.0))
    assert [output["loss"] for output in outputs] == [3.0 / num_microbatches] * num_microbatches


def test_optimizer_offload_enables_full_optimizer_state_offload_by_default() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=True),
        optimizer_config=_optimizer_config(
            use_precision_aware_optimizer=True, decoupled_weight_decay=True
        ),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 1.0
    assert optimizer.use_precision_aware_optimizer is True
    assert optimizer.decoupled_weight_decay is True
    assert optimizer.adam_beta1 == 0.9
    assert optimizer.adam_beta2 == 0.95


def test_explicit_optimizer_offload_fraction_overrides_engine_default() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=True),
        optimizer_config=_optimizer_config(offload_fraction=0.25),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 0.25


def test_optimizer_cpu_offload_alias_maps_to_full_offload_fraction() -> None:
    engine = _engine(
        engine_config=_engine_config(optimizer_offload=False),
        optimizer_config=_optimizer_config(optimizer_cpu_offload=True),
    )

    optimizer = engine._build_mlite_optimizer_config()

    assert optimizer.offload_fraction == 1.0


def test_mlite_config_threads_rl_parallel_and_impl_settings() -> None:
    engine = _engine(
        engine_config=_engine_config(
            tp=2,
            ep=8,
            etp=1,
            pp=1,
            cp=1,
            optimizer_offload=True,
            attention_backend_override="flash",
            impl_cfg={"use_thd": True, "deterministic": False},
        )
    )

    config = engine._build_mlite_config()

    assert config.model_name == "qwen3_5"
    assert config.impl == "lite"
    assert config.parallel.tp == 2
    assert config.parallel.ep == 8
    assert config.parallel.etp == 1
    assert config.optimizer.offload_fraction == 1.0
    assert config.attention_backend_override == "flash"
    assert config.impl_cfg["use_thd"] is True
    assert config.impl_cfg["deterministic"] is False


def test_online_weight_export_requests_gpu_resident_bounded_streaming() -> None:
    engine = _engine(engine_config=_engine_config(export_dtype="bfloat16"))
    captured = {}

    class Runtime:
        @staticmethod
        def export_weights(handle, **kwargs):
            captured["handle"] = handle
            captured["kwargs"] = kwargs
            return iter(())

    engine.runtime = Runtime()
    engine.handle = object()
    engine._initial_sync_cache_cleared = True

    weights, metadata = engine.get_per_tensor_param(limit=3)

    assert list(weights) == []
    assert metadata is None
    assert captured == {
        "handle": engine.handle,
        "kwargs": {
            "buffer_max_size_bytes": 2 * 1024**3,
            "cpu": False,
            "export_dtype": "bfloat16",
            "limit": 3,
            "target": "vllm",
        },
    }


def test_cpu_resident_weight_export_stages_only_export_tensors(monkeypatch) -> None:
    engine = _engine(
        engine_config=_engine_config(param_offload=True, export_dtype="bfloat16")
    )
    captured = {}

    class Runtime:
        @staticmethod
        def export_weights(handle, **kwargs):
            captured["kwargs"] = kwargs
            return iter(())

    engine.runtime = Runtime()
    engine.handle = object()
    engine._initial_sync_cache_cleared = True
    to_calls = []
    monkeypatch.setattr(engine, "to", lambda *args, **kwargs: to_calls.append((args, kwargs)))

    weights, _ = engine.get_per_tensor_param()

    assert list(weights) == []
    assert to_calls == [(("cuda",), {"model": True, "optimizer": False, "grad": False})]
    assert "staging_device" not in captured["kwargs"]

    monkeypatch.setattr(engine, "to", lambda *args, **kwargs: pytest.fail("loaded full model"))
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.get_device_name", lambda: "cuda")
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.get_device_id", lambda: 3)
    engine.handle = SimpleNamespace(_extras={"optimizer_backend": "fsdp2"})

    weights, metadata = engine.get_per_tensor_param(cpu_resident=True)

    assert list(weights) == []
    assert metadata is None
    assert captured["kwargs"]["cpu"] is False
    assert captured["kwargs"]["buffer_max_size_bytes"] == 2 * 1024**3
    assert captured["kwargs"]["staging_device"] == torch.device("cuda", 3)

    for optimizer_backend in ("dist_opt", None):
        engine.handle = SimpleNamespace(
            _extras={"optimizer_backend": optimizer_backend}
        )
        with pytest.raises(NotImplementedError, match="yet"):
            engine.get_per_tensor_param(cpu_resident=True)


def test_local_lr_scheduler_warmup_decay_and_state_roundtrip() -> None:
    from verl_mlite.engine.mlite_engine import _build_lr_scheduler

    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0, "weight_decay": 0.1}])
    opt = SimpleNamespace(
        total_training_steps=4,
        lr_warmup_steps=1,
        lr_warmup_steps_ratio=0.0,
        lr_warmup_init=0.0,
        lr=1.0,
        min_lr=0.1,
        lr_decay_steps=4,
        lr_decay_style="linear",
        weight_decay=0.1,
        weight_decay_incr_style="constant",
        lr_wsd_decay_steps=None,
        lr_wsd_decay_style="exponential",
    )

    scheduler = _build_lr_scheduler(optimizer, opt)

    assert optimizer.param_groups[0]["lr"] == 0.0
    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == 1.0
    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.7)

    state = scheduler.state_dict()
    scheduler.step(10)
    scheduler.load_state_dict(state)

    assert scheduler.state_dict() == state
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.7)
