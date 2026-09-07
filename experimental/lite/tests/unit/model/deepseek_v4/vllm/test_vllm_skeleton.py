from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.moe import DeepseekV4MoE as LiteDeepseekV4MoE
from megatron.lite.model.deepseek_v4.vllm import protocol
from megatron.lite.model.deepseek_v4.vllm.primitive.moe.communication import (
    VLLMAlignedNormalDeepEPDispatcher,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.moe.module import (
    DeepseekV4MoE,
    _batch_invariant_gate_logits,
)
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig

pytestmark = pytest.mark.gpus(1)


def _tiny_config(*, layers: int = 2) -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=16,
        hidden_size=8,
        moe_intermediate_size=4,
        num_hidden_layers=layers,
        num_attention_heads=2,
        head_dim=4,
        qk_rope_head_dim=2,
        q_lora_rank=8,
        o_lora_rank=4,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_hash_layers=1,
        compress_ratios=[0] * layers,
        hc_mult=2,
        num_nextn_predict_layers=0,
    )


def test_vllm_owns_alignment_without_changing_lite_dispatcher() -> None:
    assert LiteDeepseekV4MoE.dispatcher_cls is TokenDispatcher
    assert DeepseekV4MoE.dispatcher_cls is VLLMAlignedNormalDeepEPDispatcher


@pytest.mark.parametrize("tokens", [1, 17])
def test_gate_logits_use_one_batch_invariant_gemm(tokens: int) -> None:
    hidden = torch.randn(tokens, 8, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4, 8, dtype=torch.bfloat16, device="cuda")
    actual = _batch_invariant_gate_logits(hidden, weight)
    if tokens <= 16:
        from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
            is_available,
            ll_bf16_gemm,
        )

        expected = (
            ll_bf16_gemm(hidden, weight)
            if is_available()
            else torch.mm(hidden, weight.T, out_dtype=torch.float32)
        )
    else:
        expected = torch.mm(hidden, weight.T, out_dtype=torch.float32)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("optimizer", "override", "expected"),
    [
        ("dist_opt", None, False),
        ("fsdp2", None, False),
        ("fsdp2", True, True),
        ("dist_opt", False, False),
        (None, None, False),
    ],
)
def test_deployment_weight_cache_policy_tracks_optimizer(
    optimizer, override, expected
) -> None:
    config = protocol.ImplConfig(optimizer=optimizer, cache_deployment_weights=override)
    assert protocol._deployment_weight_cache_enabled(config) is expected


def test_forward_context_carries_dynamic_ep_token_counts(monkeypatch) -> None:
    import contextlib
    import vllm.forward_context as forward_context

    observed = {}

    class FakeDPMetadata:
        def __init__(self, counts):
            self.num_tokens_across_dp_cpu = counts

    def fake_all_gather(outputs, local, *, group):
        assert group == "ep"
        outputs[0].copy_(local)
        outputs[1].fill_(7)

    def fake_create_forward_context(_attn_metadata, config, *, dp_metadata):
        observed["config"] = config
        observed["counts"] = dp_metadata.num_tokens_across_dp_cpu.clone()
        return "dynamic-context"

    @contextlib.contextmanager
    def fake_override(value):
        observed["context"] = value
        yield

    monkeypatch.setattr(protocol.dist, "all_gather", fake_all_gather)
    monkeypatch.setattr(forward_context, "DPMetadata", FakeDPMetadata)
    monkeypatch.setattr(
        forward_context, "create_forward_context", fake_create_forward_context
    )
    monkeypatch.setattr(forward_context, "override_forward_context", fake_override)
    batch = SimpleNamespace(
        input_ids=torch.arange(5, device="cuda"),
        total_tokens=5,
    )
    parallel = SimpleNamespace(ep_group="ep", ep_size=2)
    config = object()

    with protocol._vllm_forward_context(batch, parallel, config):
        assert observed["context"] == "dynamic-context"

    assert observed["config"] is config
    assert torch.equal(observed["counts"], torch.tensor([5, 7], dtype=torch.int32))


def test_cp_forward_inputs_reuse_lite_contiguous_layout() -> None:
    model = nn.Module()
    model.ps = ParallelState(cp_size=2, cp_rank=1, tp_size=1)
    batch = PackedBatch(
        input_ids=torch.arange(7),
        labels=torch.arange(10, 17),
        loss_mask=torch.ones(7),
        seq_lens=torch.tensor([3, 4]),
    )
    inputs, packed = protocol._prepare_cp_forward_inputs(model, batch)
    assert {name: value.shape for name, value in inputs.items()} == {
        "input_ids": (4,),
        "position_ids": (4,),
        "labels": (4,),
        "loss_mask": (4,),
    }
    assert packed is not None


def test_cp1_forward_inputs_use_shared_packing_and_roll_targets() -> None:
    model = nn.Module()
    model.ps = ParallelState(cp_size=1, cp_rank=0, tp_size=1)
    batch = PackedBatch(
        input_ids=torch.arange(5),
        labels=torch.tensor([11, 12, 21, 22, 23]),
        loss_mask=torch.ones(5),
        seq_lens=torch.tensor([2, 3]),
    )
    inputs, _ = protocol._prepare_cp_forward_inputs(model, batch)
    assert torch.equal(inputs["labels"], torch.tensor([12, 0, 22, 23, 0]))


def test_forward_builds_model_owned_training_metadata(monkeypatch) -> None:
    import contextlib

    captured = {}
    built = {}

    class Builder:
        def build(self, positions, packed_seq_params):
            built["positions"] = positions
            built["packed"] = packed_seq_params
            return "metadata"

    monkeypatch.setattr(protocol, "init_batch_invariance", lambda: None)
    monkeypatch.setattr(
        protocol, "init_parallel", lambda _cfg: ParallelState(ep_size=1, ep_rank=0)
    )
    monkeypatch.setattr(
        protocol,
        "_vllm_forward_context",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        protocol,
        "build_attention_metadata_builders",
        lambda *_args, **_kwargs: {0: Builder()},
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.workspace",
        SimpleNamespace(
            init_workspace_manager=lambda *_args, **_kwargs: None,
            is_workspace_manager_initialized=lambda: True,
            reset_workspace_manager=lambda: None,
        ),
    )
    monkeypatch.setattr(
        protocol,
        "_forward_step",
        lambda _model, _batch, **kwargs: captured.update(kwargs)
        or {"loss": torch.tensor(0.0)},
    )
    bundle = protocol.build_model(
        _tiny_config(layers=1),
        impl_cfg=protocol.ImplConfig(parallel=ParallelConfig(ep=1), hf_path="/unused"),
    )
    batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3]),
        labels=torch.tensor([4, 5, 6]),
        loss_mask=torch.ones(3),
        seq_lens=torch.tensor([3]),
    )
    bundle.forward_step(bundle.chunks[0], batch)
    assert captured["attention_metadata"] == {0: "metadata"}
    assert torch.equal(built["positions"], torch.arange(built["positions"].numel()))
    assert built["packed"] is not None


def test_build_model_returns_dist_opt_wrapped_chunks(monkeypatch) -> None:
    """Keep the Lite backend's in-place DDP ownership contract."""

    class WrappedModel(nn.Module):
        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

    captured = {}

    def fake_build_training_backend(chunks, *_args, **_kwargs):
        captured["raw"] = chunks[0]
        chunks[0] = WrappedModel(chunks[0])
        captured["wrapped"] = chunks[0]
        return None, None, None, "dist_opt"

    monkeypatch.setattr(protocol, "init_batch_invariance", lambda: None)
    monkeypatch.setattr(
        protocol, "init_parallel", lambda _cfg: ParallelState(ep_size=1, ep_rank=0)
    )
    monkeypatch.setattr(protocol, "build_training_backend", fake_build_training_backend)
    monkeypatch.setitem(
        sys.modules, "vllm.config", SimpleNamespace(VllmConfig=lambda: object())
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.workspace",
        SimpleNamespace(
            init_workspace_manager=lambda *_args, **_kwargs: None,
            is_workspace_manager_initialized=lambda: True,
            reset_workspace_manager=lambda: None,
        ),
    )

    bundle = protocol.build_model(
        _tiny_config(layers=1),
        impl_cfg=protocol.ImplConfig(parallel=ParallelConfig(ep=1), hf_path="/unused"),
    )

    assert bundle.chunks == [captured["wrapped"]]
    assert bundle.chunks[0].module is captured["raw"]
    assert bundle.extras["optimizer_backend"] == "dist_opt"


def test_post_optimizer_step_invalidates_all_deployment_weights() -> None:
    class CacheOwner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cleared = 0

        def clear_deployment_weight_cache(self) -> None:
            self.cleared += 1

    model = nn.Sequential(CacheOwner(), CacheOwner())
    model._fp8_source_scales_valid = True
    model._fp8_source_scales_by_name = {"weight": torch.ones(1)}
    model[0]._fp8_source_scales_by_parameter = {"weight": torch.ones(1)}

    protocol._post_optimizer_step(model)

    assert model._fp8_source_scales_valid is False
    assert model._fp8_source_scales_by_name == {}
    assert model[0]._fp8_source_scales_by_parameter == {}
    assert [module.cleared for module in model] == [1, 1]
