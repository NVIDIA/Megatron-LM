from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.model import registry
from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.vllm import protocol
from megatron.lite.model.deepseek_v4.vllm.moe import DeepseekV4MoE
from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.runtime.contracts import PackedBatch, ParallelConfig


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
            _tiny_config(), protocol.ImplConfig(parallel=parallel, use_deepep=True)
        )


def test_parallel_contract_accepts_pp2_cp2_ep4() -> None:
    protocol._validate_contract(
        _tiny_config(),
        protocol.ImplConfig(
            parallel=ParallelConfig(pp=2, cp=2, ep=4),
            use_deepep=True,
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
    inputs, padded_lengths, packed = protocol._prepare_cp_forward_inputs(model, batch)
    assert padded_lengths == [4, 4]
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
    inputs, lengths, _ = protocol._prepare_cp_forward_inputs(model, batch)
    assert lengths == [2, 3]
    assert torch.equal(inputs["labels"], torch.tensor([12, 0, 22, 23, 0]))


def test_forward_reuses_caller_owned_ephemeral_metadata(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(protocol, "initialize_ds4_vllm_batch_invariance", lambda: None)
    monkeypatch.setattr(protocol, "init_parallel", lambda _cfg: ParallelState(ep_size=1, ep_rank=0))
    monkeypatch.setattr(
        protocol.DS4SparseIndexerCompressorMetadataAdapter,
        "from_hf",
        lambda *_args, **_kwargs: pytest.fail("rebuilt caller-owned metadata"),
    )
    monkeypatch.setattr(protocol, "ds4_vllm_forward_context", lambda *_args, **_kwargs: nullcontext())
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
    monkeypatch.setattr(
        protocol,
        "_forward_step",
        lambda _model, _batch, **kwargs: captured.update(kwargs) or {"loss": torch.tensor(0.0)},
    )
    bundle = protocol.build_model(
        _tiny_config(layers=1),
        impl_cfg=protocol.ImplConfig(
            parallel=ParallelConfig(ep=1), use_deepep=True, hf_path="/unused"
        ),
    )
    attention_metadata = {0: object()}
    moe_metadata = {0: object()}
    batch = SimpleNamespace(
        input_ids=torch.tensor([1, 2, 3]),
        attention_metadata=attention_metadata,
        moe_metadata=moe_metadata,
    )
    bundle.forward_step(bundle.chunks[0], batch)
    assert captured["attention_metadata"] is attention_metadata
    assert captured["moe_metadata"] is moe_metadata


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

    monkeypatch.setattr(protocol, "initialize_ds4_vllm_batch_invariance", lambda: None)
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
            parallel=ParallelConfig(ep=1), use_deepep=True, hf_path="/unused"
        ),
    )

    assert bundle.chunks == [captured["wrapped"]]
    assert bundle.chunks[0].module is captured["raw"]
    assert bundle.extras["optimizer_backend"] == "dist_opt"
