# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


def _build_fsdp2(monkeypatch, transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite import protocol

    seen = []

    class Model(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            seen.append(self.weight.device.type)

    monkeypatch.setattr(protocol, "Qwen3MoEModel", Model)
    monkeypatch.setattr(protocol, "init_parallel", lambda _p: SimpleNamespace())
    monkeypatch.setattr(protocol, "normalize_lora_config", lambda _cfg: SimpleNamespace(enabled=False))
    monkeypatch.setattr(protocol, "parse_recompute_spec", lambda _cfg: [])
    monkeypatch.setattr(protocol, "set_cross_entropy_fusion", lambda *_args: None)
    monkeypatch.setattr(protocol, "apply_qat_to_chunks", lambda *_args: None)
    monkeypatch.setattr(nn.Module, "cuda", lambda self: self)
    cfg = SimpleNamespace(num_nextn_predict_layers=0)
    bundle = protocol.build_model(cfg, impl_cfg=protocol.ImplConfig(optimizer="fsdp2"))
    return seen, [param.device.type for param in bundle.chunks[0].parameters()]


def test_fsdp2_always_builds_on_meta(monkeypatch, transformer_engine_import_stub) -> None:
    assert _build_fsdp2(monkeypatch, transformer_engine_import_stub) == (
        ["meta"],
        ["meta"],
    )


def test_meta_parameter_check_reports_explicit_cuda_module(
    transformer_engine_import_stub,
) -> None:
    transformer_engine_import_stub()
    from megatron.lite.model.qwen3_moe.lite.protocol import _validate_meta_parameters

    class ExplicitCudaModule(nn.Module):
        def named_parameters(self, *args, **kwargs):
            del args, kwargs
            parameter = SimpleNamespace(
                device=torch.device("cuda"),
                numel=lambda: 8,
                element_size=lambda: 2,
            )
            return iter((("weight", parameter),))

    with pytest.raises(RuntimeError) as exc_info:
        _validate_meta_parameters(ExplicitCudaModule())

    message = str(exc_info.value)
    assert "weight" in message
    assert "ExplicitCudaModule" in message
    assert "device=cuda" in message
    assert "bytes=16" in message


def test_te_parameter_modules_share_central_constructor_wrapper(
    transformer_engine_import_stub,
) -> None:
    transformer_engine_import_stub()
    from megatron.lite.primitive import transformer_engine as central_te
    from megatron.lite.primitive.modules import experts, gqa, mtp
    from megatron.lite.primitive.parallel import linear

    assert {experts.te, gqa.te, mtp.te, linear.te} == {central_te}


def test_te_constructor_imports_are_centralized() -> None:
    source_root = Path(__file__).parents[3] / "megatron" / "lite"
    direct_import = "import transformer_engine.pytorch as te"

    assert not [
        path.relative_to(source_root)
        for path in source_root.rglob("*.py")
        if direct_import in path.read_text()
    ]


def test_future_te_parameter_module_uses_central_device_policy(
    monkeypatch, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    from megatron.lite.primitive import transformer_engine as central_te

    seen = []

    class FutureLinear(nn.Module):
        def __init__(self, *, device):
            super().__init__()
            seen.append(device.type)
            self.weight = nn.Parameter(torch.empty(3, dtype=torch.bfloat16, device="cpu"))

    monkeypatch.setattr(central_te._TE, "FutureLinear", FutureLinear, raising=False)
    with torch.device("meta"):
        module = central_te.FutureLinear()

    assert seen == ["meta"]
    assert (module.weight.device.type, module.weight.dtype, tuple(module.weight.shape)) == (
        "meta",
        torch.bfloat16,
        (3,),
    )


def test_fully_sharded_meta_model_supports_to_empty(tmp_path) -> None:
    from megatron.lite.runtime.backends.mlite.runtime import _reset_parameters

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        with torch.device("meta"):
            model = nn.Linear(8, 2).to(torch.bfloat16)
        fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
        model.to_empty(device="cpu")
        model.apply(_reset_parameters)
        assert model.weight.to_local().device.type == "cpu"
        assert model.weight.dtype == torch.bfloat16
        assert torch.isfinite(model.weight.to_local()).all()
    finally:
        dist.destroy_process_group()


def test_custom_parameter_initializers_are_repeatable(
    transformer_engine_import_stub,
) -> None:
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import lora
    from megatron.lite.primitive.modules.attention.hca import HyperConnection
    from megatron.lite.primitive.modules.attention.mhc import (
        MultiHeadHyperConnectionHead,
    )
    from megatron.lite.primitive.parallel.linear import _VanillaColLinear

    hca = HyperConnection(hidden_size=4, hc_mult=2, sinkhorn_iters=2, eps=1e-6)
    mhc = MultiHeadHyperConnectionHead(hidden_size=4, hc_mult=2, eps=1e-6)
    modules = [
        _VanillaColLinear(4, 8, SimpleNamespace(tp_group=None, tp_size=1)),
        hca,
        mhc,
        lora.LinearLoRA(4, 8, 2),
        lora.GroupedLinearLoRA(2, 4, 8, 2),
        lora.SharedGroupedLinearLoRA(2, 4, 8, 2),
    ]
    with torch.no_grad():
        for module in modules:
            for param in module.parameters(recurse=False):
                param.fill_(float("nan"))
            module.reset_parameters()
            assert all(torch.isfinite(param).all() for param in module.parameters(recurse=False))

    assert torch.equal(hca.base, torch.zeros_like(hca.base))
    assert torch.equal(hca.scale, torch.ones_like(hca.scale))
    assert torch.equal(mhc.hc_base, torch.zeros_like(mhc.hc_base))
    assert torch.equal(mhc.hc_scale, torch.ones_like(mhc.hc_scale))
    for module in modules[-3:]:
        assert torch.equal(module.lora_b, torch.zeros_like(module.lora_b))


def test_expert_grouped_linears_follow_meta_context(
    monkeypatch, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    from megatron.lite.primitive import transformer_engine as central_te
    from megatron.lite.primitive.modules import experts

    class GroupedLinear(nn.Module):
        def __init__(self, groups, in_features, out_features, *, device, **_kwargs):
            super().__init__()
            del device
            self.weight = nn.Parameter(
                torch.empty(
                    groups,
                    out_features,
                    in_features,
                    dtype=torch.bfloat16,
                    device="cpu",
                )
            )

    monkeypatch.setattr(central_te._TE, "GroupedLinear", GroupedLinear, raising=False)
    monkeypatch.setattr(
        experts,
        "normalize_lora_config",
        lambda _cfg: SimpleNamespace(enabled=False),
    )
    ps = SimpleNamespace(
        ep_size=1,
        etp_size=1,
        etp_group=None,
        tp_size=1,
        tp_group=None,
    )
    config = SimpleNamespace(
        num_experts=8,
        hidden_size=16,
        moe_intermediate_size=8,
    )

    with torch.device("meta"):
        module = experts.Experts(config, ps)

    assert {
        (param.device.type, param.dtype, tuple(param.shape)) for param in module.parameters()
    } == {
        ("meta", torch.bfloat16, (8, 16, 16)),
        ("meta", torch.bfloat16, (8, 16, 8)),
    }


def test_te_parallel_linears_follow_meta_context(
    monkeypatch, transformer_engine_import_stub
) -> None:
    transformer_engine_import_stub()
    from megatron.lite.primitive import transformer_engine as central_te
    from megatron.lite.primitive.parallel import linear

    seen = []

    class TELinear(nn.Module):
        def __init__(self, in_features, out_features, *, device, **_kwargs):
            super().__init__()
            seen.append(device.type)
            self.weight = nn.Parameter(
                torch.empty(out_features, in_features, dtype=torch.bfloat16, device="cpu")
            )

        def forward(self, x):
            return x

    monkeypatch.setattr(central_te._TE, "Linear", TELinear, raising=False)
    monkeypatch.setattr(central_te._TE, "LayerNormLinear", TELinear, raising=False)
    ps = SimpleNamespace(tp_size=1, tp_rank=0, tp_group=None)
    with torch.device("meta"):
        modules = [
            linear.ColumnParallelLinear(4, 8, ps),
            linear.ColumnParallelLinear(4, 8, ps, normalization="RMSNorm"),
            linear.RowParallelLinear(4, 8, ps),
        ]

    assert seen == ["meta", "meta", "meta"]
    assert all(
        {(param.device.type, param.dtype) for param in module.parameters()}
        == {("meta", torch.bfloat16)}
        for module in modules
    )


def test_dispatcher_metadata_stays_materialized_in_meta_context() -> None:
    from megatron.lite.primitive.modules.dispatcher import TokenDispatcher

    with torch.device("meta"):
        dispatcher = TokenDispatcher(
            num_experts=4,
            hidden_size=8,
            ps=SimpleNamespace(ep_size=2),
            use_deepep=False,
        )

    assert dispatcher._sort_by_experts == [0, 2, 1, 3]
    assert dispatcher._restore_by_ranks == [0, 2, 1, 3]
