# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    assert hook.runtime_output_loss_scale == 1 / num_microbatches
    for _ in range(num_microbatches):
        loss, _ = hook({}, object(), LossContext(source_batch=object()))
        (loss / num_microbatches).backward()

    torch.testing.assert_close(weight.grad, torch.tensor(3.0))
    assert [output["loss"] for output in outputs] == [3.0 / num_microbatches] * num_microbatches


def test_verl_loss_hook_has_no_strong_self_reference():
    engine = _engine(engine_config=_engine_config())
    hook = engine._make_runtime_loss_fn(None, num_microbatches=1)

    assert all(cell.cell_contents is not hook for cell in (hook.__closure__ or ()))


def test_verl_loss_hook_honors_runtime_output_collector(monkeypatch):
    engine = _engine(engine_config=_engine_config())
    outputs = []
    engine._build_verl_model_output = lambda **_kwargs: {"log_probs": torch.tensor(1.0)}
    hook = engine._make_runtime_loss_fn(None, num_microbatches=1, output_lst=outputs)

    # This is a CPU contract test for the collector branch.  The production
    # training hook runs on the runtime-selected accelerator; pin that lookup
    # to CPU here so an empty loss does not turn this test into a CUDA probe.
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.get_device_id", lambda: "cpu")
    hook.runtime_collects_outputs = True
    hook({}, object(), LossContext(source_batch=object()))

    assert outputs == []


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


def test_mlite_config_preserves_protocol_owned_non_thd_layout() -> None:
    engine = _engine(
        engine_config=_engine_config(
            model_name="deepseek_v4",
            impl="vllm",
            impl_cfg={"use_thd": False, "use_deepep": True},
        )
    )

    config = engine._build_mlite_config()

    assert config.impl == "vllm"
    assert config.impl_cfg["use_thd"] is False
    assert config.impl_cfg["use_deepep"] is True


@pytest.mark.parametrize("zero_grad_on_exit, expected_calls", [(True, 1), (False, 0)])
def test_train_mode_clears_grads_before_context_offload(
    zero_grad_on_exit, expected_calls
) -> None:
    from verl_mlite.engine.mlite_engine import _MegatronLiteModeCtx

    events = []

    class RuntimeContext:
        def __enter__(self):
            events.append("runtime_enter")

        def __exit__(self, *_args):
            events.append("runtime_exit")

    engine = SimpleNamespace(
        mode=None,
        handle=object(),
        runtime=SimpleNamespace(train_mode=lambda _handle: RuntimeContext()),
        is_param_offload_enabled=True,
        is_optimizer_offload_enabled=True,
        optimizer_zero_grad=lambda: events.append("zero_grad"),
        to=lambda **kwargs: events.append(("to", kwargs["device"])),
    )

    with _MegatronLiteModeCtx(
        engine, mode="train", zero_grad_on_exit=zero_grad_on_exit
    ):
        events.append("body")

    assert events.count("zero_grad") == expected_calls
    assert events.index("runtime_exit") < events.index(("to", "cpu"))
    if zero_grad_on_exit:
        assert events.index("zero_grad") < events.index(("to", "cpu"))


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


def test_online_weight_export_preserves_model_dtypes_by_default() -> None:
    engine = _engine(engine_config=_engine_config())
    captured = {}

    class Runtime:
        @staticmethod
        def export_weights(handle, **kwargs):
            captured["kwargs"] = kwargs
            return iter(
                (
                    ("bf16_weight", torch.tensor([1.0], dtype=torch.bfloat16)),
                    (
                        "fp32_coefficient",
                        torch.tensor([1.0001], dtype=torch.float32),
                    ),
                )
            )

    engine.runtime = Runtime()
    engine.handle = object()
    engine._initial_sync_cache_cleared = True

    weights, _ = engine.get_per_tensor_param()
    exported = dict(weights)

    assert "export_dtype" not in captured["kwargs"]
    assert exported["bf16_weight"].dtype == torch.bfloat16
    assert exported["fp32_coefficient"].dtype == torch.float32
    assert exported["fp32_coefficient"].item() != float(
        exported["fp32_coefficient"].bfloat16().float().item()
    )


def test_qwen3_moe_online_weight_export_does_not_pass_unsupported_target() -> None:
    engine = _engine(engine_config=_engine_config())
    engine.model_config.hf_config = {"model_type": "qwen3_moe"}
    captured = {}

    class Runtime:
        @staticmethod
        def export_weights(handle, **kwargs):
            captured["kwargs"] = kwargs
            return iter(())

    engine.runtime = Runtime()
    engine.handle = object()
    engine._initial_sync_cache_cleared = True

    weights, metadata = engine.get_per_tensor_param()

    assert list(weights) == []
    assert metadata is None
    assert "target" not in captured["kwargs"]


def test_online_qat_export_uses_mlite_owned_exporter(monkeypatch) -> None:
    engine = _engine(
        engine_config=_engine_config(
            qat={
                "enable": True,
                "apply_modelopt_fake_quant": False,
                "mode": "mxfp4",
                "group_size": 32,
                "ignore_patterns": ["lm_head"],
            }
        )
    )
    source_weights = iter(
        [("model.layers.0.mlp.experts.0.gate_proj.weight", torch.ones(2, 32))]
    )
    captured = {}

    class Runtime:
        @staticmethod
        def export_weights(handle, **kwargs):
            return source_weights

    def fake_export(weights, qat_config):
        captured.update(weights=weights, qat_config=qat_config)
        return iter([("packed.weight", torch.ones(2, 16, dtype=torch.uint8))])

    monkeypatch.setattr("verl_mlite.qat_export.export_qat_weights", fake_export)
    engine.runtime = Runtime()
    engine.handle = object()
    engine._initial_sync_cache_cleared = True

    weights, metadata = engine.get_per_tensor_param()

    assert [name for name, _ in weights] == ["packed.weight"]
    assert metadata is None
    assert captured["weights"] is source_weights
    assert captured["qat_config"] == engine.engine_config.qat


def test_qat_export_rejects_native_resync_format_double_quantization() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _engine_config(qat={"enable": True, "mode": "mxfp4"}, resync_format="mxfp4")


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
