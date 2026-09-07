# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from megatron.lite.runtime import create_runtime
from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
from megatron.lite.runtime.backends.mlite.runtime import (
    MegatronLiteRuntime,
    _apply_attention_backend_env,
    _build_impl_cfg,
    _pipeline_callbacks,
    _reset_parameters,
)
from megatron.lite.runtime.contracts.config import OptimizerConfig, ParallelConfig, RuntimeConfig
from megatron.lite.runtime.contracts.handle import ModelHandle
from megatron.lite.runtime.contracts.loss import LossContext, get_loss_context, use_loss_context


def test_runtime_returns_loss_separately_from_microbatch_metrics():
    model = nn.Linear(1, 1, bias=False)
    handle = ModelHandle(
        model=model,
        parallel_state=types.SimpleNamespace(pp_size=1),
        _extras={"forward_step": lambda module, batch: {"value": module(batch["x"])}},
    )
    batches = iter([{"x": torch.ones(1, 1), "micro": i} for i in range(2)])
    result = MegatronLiteRuntime.__new__(MegatronLiteRuntime).forward_backward(
        handle,
        batches,
        lambda out, batch: (out["value"].sum(), {"micro": batch["micro"]}),
        num_microbatches=2,
    )

    assert result.metrics == {"micro": [0, 1]}
    assert result.model_output.loss is not None


def test_zero_grad_without_optimizer_clears_parameter_gradients():
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    model = nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    handle = ModelHandle(
        model=model,
        optimizer=None,
        _extras={"model_chunks": [model]},
    )

    runtime.zero_grad(handle)

    assert model.weight.grad is None


def test_post_step_scale_invalidation_requires_a_successful_optimizer_update():
    calls = []
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    model = nn.Linear(1, 1, bias=False)

    no_optim = ModelHandle(
        model=model,
        optimizer=None,
        _extras={"post_optimizer_step_hook": lambda: calls.append("no_optim")},
    )
    assert runtime.optimizer_step(no_optim) == (True, 0.0, 0)
    assert calls == []

    class Optimizer:
        def __init__(self):
            self.success = False

        def step(self):
            return self.success, torch.tensor(1.5), 0

    optimizer = Optimizer()
    fsdp2 = ModelHandle(
        model=model,
        optimizer=optimizer,
        _extras={"post_optimizer_step_hook": lambda: calls.append("fsdp2")},
    )
    assert runtime.optimizer_step(fsdp2) == (False, 1.5, 0)
    assert calls == []
    optimizer.success = True
    assert runtime.optimizer_step(fsdp2) == (True, 1.5, 0)
    assert calls == ["fsdp2"]


def test_pipeline_callbacks_accept_wrapped_and_presplit_context():
    context = LossContext(source_batch="source")
    seen = []
    forward, loss = _pipeline_callbacks(
        lambda _model, batch: seen.append((batch, get_loss_context()))
        or {"loss": torch.tensor(1.0)},
        lambda out, batch, ctx: (out["loss"], {"batch": batch, "source": ctx.source_batch}),
    )

    output = forward(None, ("wrapped", context))
    with use_loss_context(context):
        forward(None, "presplit")
    _, metrics = loss(output, "presplit", context)

    assert seen == [("wrapped", context), ("presplit", context)]
    assert metrics == {"batch": "presplit", "source": "source"}


def test_runtime_config_defaults_to_mlite_backend():
    cfg = RuntimeConfig()

    assert cfg.backend == "mlite"
    assert cfg.hf_path == ""
    assert isinstance(cfg.backend_cfg, dict)


def test_runtime_config_accepts_mlite_backend_cfg():
    cfg = RuntimeConfig(
        backend="mlite",
        hf_path="/models/Qwen3",
        backend_cfg={"model_name": "qwen3", "impl": "lite", "tp": 2, "ep": 4},
    )

    assert cfg.backend == "mlite"
    assert cfg.backend_cfg["model_name"] == "qwen3"
    assert cfg.backend_cfg["tp"] == 2


def test_mlite_config_defaults_and_parallel_fields():
    cfg = MegatronLiteConfig(
        model_name="qwen3_moe", parallel=ParallelConfig(tp=4, etp=1, ep=8, pp=2, vpp=2, cp=2)
    )

    assert cfg.model_name == "qwen3_moe"
    assert cfg.impl == "lite"
    assert cfg.parallel.tp == 4
    assert cfg.parallel.ep == 8
    assert cfg.parallel.pp == 2
    assert cfg.parallel.cp == 2


def test_mlite_config_impl_cfg_optimizer_and_load_gate():
    hook = lambda cfg: cfg  # noqa: E731
    cfg = MegatronLiteConfig(
        model_name="qwen3_moe",
        impl_cfg={"recompute": "full"},
        optimizer=OptimizerConfig(lr=1e-4, weight_decay=0.1, adam_beta1=0.9),
        load_hf_weights=False,
        model_config_hook=hook,
    )

    assert cfg.impl_cfg["recompute"] == "full"
    assert cfg.optimizer.lr == 1e-4
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.load_hf_weights is False
    assert cfg.model_config_hook is hook


def test_mlite_config_from_dict_accepts_optimizer_override_config():
    cfg = MegatronLiteConfig.from_dict(
        "/models/Qwen3",
        {
            "optimizer": {
                "override_optimizer_config": {
                    "fsdp2_use_fp32_master": False,
                    "offload_fraction": 1.0,
                }
            }
        },
    )

    assert cfg.optimizer.override_optimizer_config == {
        "fsdp2_use_fp32_master": False,
        "offload_fraction": 1.0,
    }


def test_mlite_config_from_dict_rejects_num_microbatches():
    with pytest.raises(ValueError, match="num_microbatches"):
        MegatronLiteConfig.from_dict(
            "/models/Qwen3", {"model_name": "qwen3", "tp": 4, "num_microbatches": 2}
        )


@dataclass
class _FakeImplConfig:
    parallel: object
    hf_path: str = ""
    optimizer_config: object = None
    attention_backend_override: str | None = None


def test_build_impl_cfg_backfills_top_level_hf_path_and_runtime_fields():
    proto = type("Proto", (), {"ImplConfig": _FakeImplConfig})
    cfg = MegatronLiteConfig(
        model_name="qwen3",
        hf_path="/models/top",
        load_hf_weights=False,
        attention_backend_override="local",
    )

    impl_cfg = _build_impl_cfg(proto, cfg)

    assert impl_cfg.parallel is cfg.parallel
    assert impl_cfg.hf_path == "/models/top"
    assert impl_cfg.optimizer_config is cfg.optimizer
    assert impl_cfg.attention_backend_override == "local"


def test_reset_parameters_helper_reinitializes_each_module_via_apply():
    calls = []

    class Resettable(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def reset_parameters(self):
            calls.append(self.name)

    model = nn.Sequential(Resettable("first"), Resettable("second"))
    model.apply(_reset_parameters)

    assert calls == ["first", "second"]


def test_reset_parameters_helper_preserves_replaced_parameter_identity():
    class ReplacingReset(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0]))

        def reset_parameters(self):
            self.weight = nn.Parameter(torch.tensor([3.0]))

    model = ReplacingReset()
    original = model.weight
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.apply(_reset_parameters)

    assert optimizer.param_groups[0]["params"][0] is model.weight
    assert model.weight is original
    torch.testing.assert_close(model.weight, torch.tensor([3.0]))


@pytest.mark.parametrize(
    "load_hf_weights", [True, False], ids=["checkpoint", "random-reset"]
)
def test_runtime_meta_init_refreshes_fsdp2_master_after_weights_are_ready(
    monkeypatch, load_hf_weights
):
    from megatron.lite.primitive.bundle import ModelBundle
    from megatron.lite.primitive.optimizers.fsdp2.adamw import build_adamw_optimizer
    from megatron.lite.primitive.optimizers.fsdp2.optimizer import FSDP2Optimizer

    loaded_value = torch.tensor([1.5, -2.25], dtype=torch.bfloat16)

    class Chunk(nn.Module):
        def __init__(self, *, device):
            super().__init__()
            self.weight = nn.Parameter(
                torch.empty(2, dtype=torch.bfloat16, device=device)
            )

        def reset_parameters(self):
            if not self.weight.is_meta:
                with torch.no_grad():
                    self.weight.copy_(loaded_value)

    def make_optimizer(param):
        inner = build_adamw_optimizer(
            [{"params": [param], "weight_decay": 0.0}],
            all_params=[param],
            lr=0.1,
            weight_decay=0.0,
            betas=(0.9, 0.99),
            eps=1.0e-8,
            foreach=False,
            use_fp32_master=True,
            cpu_update=False,
            model_param_dtypes={id(param): torch.bfloat16},
            opt=types.SimpleNamespace(),
        )
        return FSDP2Optimizer(inner, [param], clip_grad=100.0)

    chunk = Chunk(device="meta")

    class Protocol:
        ImplConfig = _FakeImplConfig

        @staticmethod
        def build_model_config(_hf_path):
            return types.SimpleNamespace()

        @staticmethod
        def build_model(_model_cfg, *, impl_cfg):
            del impl_cfg

            def post_model_load_hook():
                chunk.to_empty(device="cpu")
                with torch.no_grad():
                    chunk.weight.zero_()
                return {"optimizer": make_optimizer(chunk.weight)}

            return ModelBundle(
                chunks=[chunk],
                parallel_state=types.SimpleNamespace(),
                forward_step=lambda *_args: None,
                extras={"post_model_load_hook": post_model_load_hook},
            )

        @staticmethod
        def load_hf_weights(model, _path, _model_cfg, _ps):
            with torch.no_grad():
                model.weight.copy_(loaded_value)

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda _seed: None)

    cfg = MegatronLiteConfig(
        model_name="qwen3_moe",
        hf_path="/known" if load_hf_weights else "",
        load_hf_weights=load_hf_weights,
    )
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime._cfg = cfg
    runtime._hf_path = cfg.hf_path
    runtime._load_protocol = lambda _cfg: Protocol
    handle = runtime.build_model()

    eager_param = nn.Parameter(loaded_value.clone())
    eager_optimizer = make_optimizer(eager_param)
    grad = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
    chunk.weight.grad = grad.clone()
    eager_param.grad = grad.clone()

    assert handle._optimizer.step()[0]
    assert eager_optimizer.step()[0]
    torch.testing.assert_close(chunk.weight, eager_param, rtol=0, atol=0)


def test_build_impl_cfg_preserves_explicit_impl_hf_path():
    proto = type("Proto", (), {"ImplConfig": _FakeImplConfig})
    cfg = MegatronLiteConfig(
        model_name="qwen3", hf_path="/models/top", impl_cfg={"hf_path": "/models/impl"}
    )

    impl_cfg = _build_impl_cfg(proto, cfg)

    assert impl_cfg.hf_path == "/models/impl"


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("auto", ("1", "1", "1")),
        ("flash", ("1", "0", "0")),
        ("fused", ("0", "1", "0")),
        ("unfused", ("0", "0", "1")),
        ("local", ("0", "0", "0")),
        ("magi", ("1", "1", "1")),
    ],
)
def test_attention_backend_override_sets_expected_env(monkeypatch, backend, expected):
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(name, raising=False)

    _apply_attention_backend_env(backend, tag="unit")

    assert (
        os.environ["NVTE_FLASH_ATTN"],
        os.environ["NVTE_FUSED_ATTN"],
        os.environ["NVTE_UNFUSED_ATTN"],
    ) == expected


def test_attention_backend_override_rejects_unknown_backend():
    with pytest.raises(ValueError, match="attention_backend_override"):
        _apply_attention_backend_env("invalid", tag="unit")


class HookedOptimizer:
    def __init__(self):
        self.calls: list[str] = []

    def offload_state_to_cpu(self):
        self.calls.append("offload")

    def load_state_to_device(self):
        self.calls.append("load")


def test_runtime_to_prefers_optimizer_specific_offload_hooks():
    optimizer = HookedOptimizer()
    handle = ModelHandle(model=nn.Linear(2, 2), optimizer=optimizer, _extras={"model_chunks": []})
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    runtime.to(handle, "cpu", model=False, optimizer=True, grad=False)
    runtime.to(handle, "cuda", model=False, optimizer=True, grad=False)

    assert optimizer.calls == ["offload", "load"]


def test_training_transfer_parks_optimizer_and_releases_scratch(monkeypatch):
    events = []

    class Chunk:
        def release_export_scratch(self):
            events.append("release-scratch")

    class Optimizer:
        def offload_state_to_cpu(self):
            events.append("offload-optimizer")

        def load_state_to_device(self):
            events.append("load-optimizer")

    import megatron.lite.runtime.megatron_utils as megatron_utils

    monkeypatch.setattr(
        megatron_utils,
        "offload_model_to_cpu",
        lambda chunks: events.append("offload-model"),
    )
    monkeypatch.setattr(
        megatron_utils,
        "load_model_to_gpu",
        lambda chunks, load_grad: events.append("load-model"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(
        "megatron.lite.runtime.backends.mlite.runtime.gc.collect",
        lambda: events.append("collect"),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty-cache"))
    chunk = Chunk()
    handle = ModelHandle(
        model=chunk,
        optimizer=Optimizer(),
        _extras={"model_chunks": [chunk]},
    )
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    runtime.to(handle, "cpu", model=True, optimizer=False, grad=True)
    runtime.to(handle, "cuda", model=True, optimizer=False, grad=True)

    assert events == [
        "offload-model",
        "offload-optimizer",
        "release-scratch",
        "synchronize",
        "collect",
        "empty-cache",
        "load-model",
        "load-optimizer",
    ]


def test_export_transfer_does_not_move_optimizer_or_release_scratch(monkeypatch):
    events = []

    class Chunk:
        def release_export_scratch(self):
            events.append("release-scratch")

    import megatron.lite.runtime.megatron_utils as megatron_utils

    monkeypatch.setattr(
        megatron_utils,
        "offload_model_to_cpu",
        lambda chunks: events.append("offload-model"),
    )
    chunk = Chunk()
    handle = ModelHandle(
        model=chunk,
        optimizer=HookedOptimizer(),
        _extras={"model_chunks": [chunk]},
    )

    MegatronLiteRuntime.__new__(MegatronLiteRuntime).to(
        handle, "cpu", model=True, optimizer=False, grad=False
    )

    assert events == ["offload-model"]


class _FakeStorage:
    def __init__(self, size: int):
        self._size = size
        self.resize_calls: list[int] = []

    def size(self):
        return self._size

    def resize_(self, size: int):
        self.resize_calls.append(size)
        self._size = size
        return self


class _FakeBufferData:
    def __init__(self, size: int):
        self._storage = _FakeStorage(size)
        self.cpu_called = False
        self.pinned = False
        self.copied_from = None
        self.copy_non_blocking = None
        self.zero_calls = 0

    @property
    def data(self):
        return self

    def cpu(self):
        self.cpu_called = True
        return self

    def pin_memory(self):
        self.pinned = True
        return self

    def storage(self):
        return self._storage

    def copy_(self, other, *, non_blocking: bool):
        self.copied_from = other
        self.copy_non_blocking = non_blocking
        return self

    def zero_(self):
        self.zero_calls += 1
        return self


class _FakeBuffer:
    def __init__(self):
        self.param_data = _FakeBufferData(3)
        self.grad_data = _FakeBufferData(5)


class _FakeModule:
    def parameters(self):
        return []


class _FakeMegatronDDP:
    def __init__(self):
        self.buffer = _FakeBuffer()
        self.buffers = [self.buffer]
        self.expert_parallel_buffers = []
        self.module = _FakeModule()
        self.to_calls: list[str] = []

    def to(self, device):
        self.to_calls.append(device)
        raise AssertionError("DDP model chunks must use the buffer offload path")


class _FakeMegatronDDPSubclass(_FakeMegatronDDP):
    pass


class _FakeNativeModel:
    def __init__(self):
        self.calls: list[str] = []

    def to(self, device):
        self.calls.append(device)
        return self


def _install_fake_megatron_ddp(monkeypatch) -> None:
    core = types.ModuleType("megatron.core")
    distributed = types.ModuleType("megatron.core.distributed")
    distributed.DistributedDataParallel = _FakeMegatronDDP
    core.distributed = distributed
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.distributed", distributed)


def test_megatron_ddp_detection_accepts_ddp_and_subclasses(monkeypatch):
    from megatron.lite.runtime.megatron_utils import _is_megatron_ddp

    _install_fake_megatron_ddp(monkeypatch)

    assert _is_megatron_ddp(_FakeMegatronDDP()) is True
    assert _is_megatron_ddp(_FakeMegatronDDPSubclass()) is True
    assert _is_megatron_ddp(_FakeNativeModel()) is False


@pytest.mark.parametrize("model_cls", [_FakeMegatronDDP, _FakeMegatronDDPSubclass])
def test_megatron_ddp_model_move_helpers_use_buffer_path(monkeypatch, model_cls):
    import megatron.lite.runtime.megatron_utils as megatron_utils
    from megatron.lite.runtime.megatron_utils import load_model_to_gpu, offload_model_to_cpu

    _install_fake_megatron_ddp(monkeypatch)
    model = model_cls()
    buffer = model.buffer
    pinned_copies = []

    def fake_pinned_cpu_copy(tensor):
        pinned_copies.append(tensor)
        return tensor.cpu().pin_memory()

    monkeypatch.setattr(
        megatron_utils,
        "_pinned_cpu_copy",
        fake_pinned_cpu_copy,
    )

    offload_model_to_cpu([model])

    assert model.to_calls == []
    assert buffer.param_data.cpu_called is True
    assert buffer.param_data.pinned is True
    assert buffer.param_data_size == 3
    assert buffer.grad_data_size == 5
    assert buffer.param_data.storage().size() == 0
    assert buffer.grad_data.storage().size() == 0

    load_model_to_gpu([model])

    assert model.to_calls == []
    assert buffer.param_data.storage().size() == 3
    assert buffer.grad_data.storage().size() == 5
    assert buffer.param_data.copied_from is buffer.param_data.cpu_data
    assert buffer.param_data.copy_non_blocking is True
    assert buffer.grad_data.zero_calls == 1

    cpu_data = buffer.param_data.cpu_data
    offload_model_to_cpu([model])

    assert pinned_copies == [buffer.param_data]
    assert buffer.param_data.cpu_data is cpu_data
    assert cpu_data.copied_from is buffer.param_data
    assert cpu_data.copy_non_blocking is False


def test_native_model_move_helpers_do_not_require_megatron_core(monkeypatch):
    from megatron.lite.runtime.megatron_utils import load_model_to_gpu, offload_model_to_cpu

    monkeypatch.setitem(sys.modules, "megatron.core", None)
    monkeypatch.setitem(sys.modules, "megatron.core.distributed", None)
    model = _FakeNativeModel()

    offload_model_to_cpu([model])
    load_model_to_gpu([model])

    assert model.calls == ["cpu", "cuda"]


def test_native_model_offload_reshards_fsdp2_before_cpu_move(monkeypatch):
    import megatron.lite.runtime.megatron_utils as megatron_utils

    model = _FakeNativeModel()
    events = []

    monkeypatch.setattr(
        megatron_utils,
        "_reshard_fsdp2_modules",
        lambda model_chunk: events.append(("reshard", model_chunk)),
    )
    original_to = model.to

    def tracked_to(device):
        events.append(("to", device))
        return original_to(device)

    model.to = tracked_to
    megatron_utils.offload_model_to_cpu([model])

    assert events == [("reshard", model), ("to", "cpu")]


def test_model_handle_dp_defaults():
    handle = ModelHandle(model=MagicMock())

    assert handle.dp_rank == 0
    assert handle.dp_size == 1
    assert handle.dp_group is None


def test_model_handle_dp_from_parallel_state():
    ps = MagicMock()
    ps.dp_rank = 3
    ps.dp_size = 8
    ps.dp_group = "fake_group"

    handle = ModelHandle(model=MagicMock(), parallel_state=ps)

    assert handle.dp_rank == 3
    assert handle.dp_size == 8
    assert handle.dp_group == "fake_group"


def test_model_handle_cp_range_and_config_properties():
    cfg = {"tp": 8, "ep": 4}
    default_handle = ModelHandle(model=MagicMock())
    configured_handle = ModelHandle(model=MagicMock(), config=cfg, _extras={"cp_range": (1, 8)})

    assert default_handle.cp_range == (1, 1)
    assert configured_handle.cp_range == (1, 8)
    assert configured_handle.config is cfg


def test_runtime_dispatch_creates_mlite_backend():
    with patch("megatron.lite.runtime.backends.mlite.create") as mock_create:
        backend = MagicMock()
        mock_create.return_value = backend

        runtime = create_runtime(
            RuntimeConfig(
                backend="mlite", hf_path="/models/test", backend_cfg={"model_name": "qwen3"}
            )
        )

    assert runtime is backend
    mock_create.assert_called_once_with("/models/test", {"model_name": "qwen3"})


def test_runtime_dispatch_unknown_backend_raises():
    with pytest.raises(KeyError):
        create_runtime(RuntimeConfig(backend="nonexistent"))
