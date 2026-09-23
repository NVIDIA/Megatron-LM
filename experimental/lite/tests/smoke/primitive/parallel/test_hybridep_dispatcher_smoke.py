# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""HybridEP token dispatcher on 2 GPUs: bitwise against mcore's _HybridEPManager, sanity against all-to-all."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.runtime.contracts import ParallelConfig

pytestmark = [
    pytest.mark.gpus(2, min_architecture="blackwell"),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
]

HIDDEN, EXPERTS, TOPK = 256, 8, 2


@pytest.fixture(scope="module", autouse=True)
def _dist():
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("run with torchrun --nproc-per-node=2")
    pytest.importorskip("deep_ep")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def ps():
    return init_parallel(ParallelConfig(tp=1, ep=2, pp=1, cp=1))


def _mcore_config():
    return SimpleNamespace(
        moe_router_topk=TOPK, moe_permute_fusion=False, moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False, moe_hybridep_pad_variable_tokens=True,
        moe_hybridep_routing_map_mode="indices", moe_expert_rank_capacity_factor=None,
        moe_flex_dispatcher_num_sms=None, moe_hybridep_num_blocks_permute=None,
        moe_hybridep_num_blocks_unpermute=None, moe_permute_fusion_into_hybridep=False,
        moe_hybridep_num_sms_preprocessing=108, use_transformer_engine_op_fuser=False,
        moe_use_grouped_tensor=False, fp8=False, fp4=False,
    )


def _inputs(num_tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(num_tokens, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    logits = torch.rand(num_tokens, EXPERTS, device="cuda", generator=g)
    scores, indices = torch.topk(logits, TOPK, dim=-1)
    scores = torch.sigmoid(scores)
    return hidden, scores, indices


def _expert_fn(dispatched: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    return (dispatched.float() * (probs.unsqueeze(1) + 0.5)).to(dispatched.dtype)


def _run_lite(ps, hidden, scores, indices, backend: str):
    from megatron.lite.primitive.modules.dispatcher import TokenDispatcher

    hidden = hidden.clone().requires_grad_(True)
    scores = scores.clone().requires_grad_(True)
    d = TokenDispatcher(EXPERTS, HIDDEN, ps, dispatch_backend=backend)
    dispatched, tpe, probs = d.dispatch(hidden, scores, indices)
    combined = d.combine(_expert_fn(dispatched, probs))
    (combined.float().sum() + probs.sum()).backward()
    return dict(dispatched=dispatched.detach(), tpe=tpe, probs=probs.detach(), combined=combined.detach(),
                grad_hidden=hidden.grad, grad_scores=scores.grad)


def _run_mcore(ps, hidden, scores, indices):
    from megatron.core.transformer.moe.token_dispatcher import _HybridEPManager

    hidden = hidden.clone().requires_grad_(True)
    scores = scores.clone().requires_grad_(True)
    routing_map = torch.zeros(hidden.size(0), EXPERTS, dtype=torch.bool, device="cuda").scatter_(1, indices, True)
    probs_2d = torch.zeros(hidden.size(0), EXPERTS, dtype=scores.dtype, device="cuda").scatter_add(1, indices, scores)
    manager = _HybridEPManager(ps.tp_ep_group, EXPERTS // ps.ep_size, EXPERTS, _mcore_config(), router_topk=TOPK)
    manager.setup_metadata(routing_map, probs_2d)
    dispatched = manager.dispatch(hidden)
    probs = manager.dispatched_probs
    combined = manager.combine(_expert_fn(dispatched, probs))
    (combined.float().sum() + probs.sum()).backward()
    return dict(dispatched=dispatched.detach(), tpe=manager.get_number_of_tokens_per_expert(), probs=probs.detach(),
                combined=combined.detach(), grad_hidden=hidden.grad, grad_scores=scores.grad)


def _assert_bitwise(lite, ref, case):
    for k in ("dispatched", "probs", "combined", "grad_hidden", "grad_scores"):
        assert lite[k].dtype == ref[k].dtype and lite[k].shape == ref[k].shape, (case, k, lite[k].shape, ref[k].shape)
        assert torch.equal(lite[k], ref[k]), (case, k, (lite[k].float() - ref[k].float()).abs().max().item())
    assert torch.equal(lite["tpe"].cpu().long(), ref["tpe"].cpu().long()), (case, lite["tpe"], ref["tpe"])


@pytest.mark.parametrize("tokens", [(128, 128), (96, 64), (200, 8)], ids=["equal", "unequal", "ragged"])
def test_hybridep_bitwise_matches_mcore(ps, tokens):
    num_tokens = tokens[ps.ep_rank]
    hidden, scores, indices = _inputs(num_tokens, seed=100 + ps.ep_rank)
    lite = _run_lite(ps, hidden, scores, indices, "hybridep")
    ref = _run_mcore(ps, hidden, scores, indices)
    _assert_bitwise(lite, ref, f"tokens={tokens}")
    if dist.get_rank() == 0:
        print(f"\nhybridep bitwise vs mcore ok: tokens={tokens} tpe={lite['tpe'].tolist()}", flush=True)


def test_hybridep_matches_alltoall(ps):
    hidden, scores, indices = _inputs(96 if ps.ep_rank == 0 else 64, seed=7 + ps.ep_rank)
    hep = _run_lite(ps, hidden, scores, indices, "hybridep")
    a2a = _run_lite(ps, hidden, scores, indices, "alltoall")
    assert torch.equal(hep["tpe"].cpu(), a2a["tpe"].cpu())
    assert torch.equal(hep["dispatched"].float().sum(0), a2a["dispatched"].float().sum(0)) or torch.allclose(
        hep["dispatched"].float().sum(0), a2a["dispatched"].float().sum(0), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(hep["combined"].float(), a2a["combined"].float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(hep["grad_hidden"].float(), a2a["grad_hidden"].float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(hep["grad_scores"], a2a["grad_scores"], rtol=1e-4, atol=1e-4)
