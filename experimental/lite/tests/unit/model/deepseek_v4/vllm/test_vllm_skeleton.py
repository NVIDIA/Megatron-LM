from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.model import registry
from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.moe import DeepseekV4MoE as LiteDeepseekV4MoE
from megatron.lite.model.deepseek_v4.vllm import protocol
from megatron.lite.model.deepseek_v4.vllm.primitive.moe.dispatcher import (
    VLLMAlignedNormalDeepEPDispatcher,
)
from megatron.lite.model.deepseek_v4.vllm.primitive.moe.module import (
    DeepseekV4MoE,
    _batch_invariant_gate_logits,
)
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction
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


def test_registry_exposes_vllm_training_runtime() -> None:
    assert registry.resolve_runtime_model_name("deepseek_v4", "vllm") == "deepseek_v4_vllm"
    assert registry.TRAIN_RUNTIME_MODULES["deepseek_v4_vllm"].endswith(".vllm.protocol")


def test_vllm_owns_alignment_without_changing_lite_dispatcher() -> None:
    assert LiteDeepseekV4MoE.dispatcher_cls is TokenDispatcher
    assert DeepseekV4MoE.dispatcher_cls is VLLMAlignedNormalDeepEPDispatcher


@pytest.mark.parametrize("tokens", [1, 17])
def test_gate_logits_use_one_batch_invariant_gemm(tokens: int) -> None:
    hidden = torch.randn(tokens, 8, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4, 8, dtype=torch.bfloat16, device="cuda")
    actual = _batch_invariant_gate_logits(hidden, weight)
    expected = torch.mm(hidden, weight.T, out_dtype=torch.float32)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_deployment_weight_cache_policy_tracks_optimizer() -> None:
    policy = protocol._deployment_weight_cache_enabled
    assert policy(protocol.ImplConfig(optimizer="dist_opt")) is True
    assert policy(protocol.ImplConfig(optimizer="fsdp2")) is False
    assert policy(
        protocol.ImplConfig(
            optimizer="fsdp2", cache_deployment_weights=True
        )
    ) is True
    assert policy(
        protocol.ImplConfig(
            optimizer="dist_opt", cache_deployment_weights=False
        )
    ) is False


def test_r3_uses_contiguous_cp_layout_and_live_actor_weights() -> None:
    model = nn.Module()
    model.ps = SimpleNamespace(tp_size=1, tp_rank=0, cp_size=2, cp_rank=1, cp_group=None)
    batch = PackedBatch(
        input_ids=torch.arange(8),
        labels=torch.arange(8),
        seq_lens=torch.tensor([4, 4]),
        r3_replay_mask=torch.tensor([1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.bool),
    )
    routes = torch.arange(8).view(2, 4, 1, 1)
    assert torch.equal(protocol.pack_routed_experts(model, batch, routes)[0], routes[1].view(4, 1))
    assert torch.equal(protocol.pack_r3_replay_mask(model, batch), batch.r3_replay_mask[4:])

    config = _tiny_config()
    moe = DeepseekV4MoE(config, ParallelState(), layer_idx=1)
    replay = RouterReplay()
    moe.gate.router_replay = replay
    logits = torch.tensor([[-2.0, -0.5, 0.5, 2.0], [1.5, -1.0, 0.25, -0.25]])
    replay.target_topk_idx = torch.tensor([[3, 2], [0, 1]])
    replay.target_replay_mask = torch.tensor([True, False])
    replay.router_replay_action = RouterReplayAction.REPLAY_FORWARD
    weights, ids = moe._replay_route(
        logits,
        torch.full((2, 2), -1.0),
        torch.tensor([[0, 1], [2, 3]]),
    )
    expected_ids = torch.tensor([[3, 2], [2, 3]])
    dense = torch.sqrt(torch.nn.functional.softplus(logits))
    expected = dense.gather(-1, expected_ids)
    expected = expected / expected.sum(-1, keepdim=True) * config.routed_scaling_factor
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)


@pytest.mark.parametrize("parallel", [ParallelConfig(tp=2), ParallelConfig(etp=2), ParallelConfig(vpp=2)])
def test_parallel_contract_rejects_unsupported_dimensions(parallel: ParallelConfig) -> None:
    with pytest.raises(NotImplementedError):
        protocol._validate_contract(
            _tiny_config(), protocol.ImplConfig(parallel=parallel)
        )


def test_parallel_contract_accepts_pp2_cp2_ep4() -> None:
    protocol._validate_contract(
        _tiny_config(),
        protocol.ImplConfig(
            parallel=ParallelConfig(pp=2, cp=2, ep=4),
            recompute=("full",),
        ),
    )


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
    captured = {}
    built = {}

    class Builder:
        def build(self, positions, packed_seq_params):
            built["positions"] = positions
            built["packed"] = packed_seq_params
            return "metadata"

    monkeypatch.setattr(protocol, "init_batch_invariance", lambda: None)
    monkeypatch.setattr(protocol, "init_parallel", lambda _cfg: ParallelState(ep_size=1, ep_rank=0))
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
        lambda _model, _batch, **kwargs: captured.update(kwargs) or {"loss": torch.tensor(0.0)},
    )
    bundle = protocol.build_model(
        _tiny_config(layers=1),
        impl_cfg=protocol.ImplConfig(
            parallel=ParallelConfig(ep=1), hf_path="/unused"
        ),
    )
    batch = PackedBatch(
        input_ids=torch.tensor([1, 2, 3]),
        labels=torch.tensor([4, 5, 6]),
        loss_mask=torch.ones(3),
        seq_lens=torch.tensor([3]),
    )
    bundle.forward_step(bundle.chunks[0], batch)
    assert captured["attention_metadata"] == {0: "metadata"}
    assert torch.equal(
        built["positions"], torch.arange(built["positions"].numel())
    )
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
    monkeypatch.setattr(protocol, "init_parallel", lambda _cfg: ParallelState(ep_size=1, ep_rank=0))
    monkeypatch.setattr(protocol, "build_training_backend", fake_build_training_backend)
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(VllmConfig=lambda: object()))
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
        impl_cfg=protocol.ImplConfig(
            parallel=ParallelConfig(ep=1), hf_path="/unused"
        ),
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

    protocol._post_optimizer_step(model)

    assert model._fp8_source_scales_valid is False
    assert model._fp8_source_scales_by_name == {}
    assert [module.cleared for module in model] == [1, 1]
