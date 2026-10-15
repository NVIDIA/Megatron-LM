# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 lite on FSDP2 (Magi, 2 GPUs): per DP2/EP2/CP2, step-0 loss matches dist_opt, two steps stay finite,
and the frozen indexer is unchanged."""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.gpus(2, min_architecture="blackwell"),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
    pytest.mark.optional,
]

S, CHUNK = 4096, 512
LOSS_REL = 1e-3
_CASES = [("dp2", dict(tp=1, ep=1, etp=1, pp=1, cp=1)), ("ep2", dict(tp=1, ep=2, etp=1, pp=1, cp=1)),
          ("cp2", dict(tp=1, ep=1, etp=1, pp=1, cp=2))]


def _deps(dist):
    from megatron.lite.primitive.optimizers.fsdp2 import fsdp2_available

    pytest.importorskip("megatron.core", reason="dist_opt reference needs megatron-core")
    pytest.importorskip("magi_attn_extensions.MSA")
    pytest.importorskip("msa_v1")
    if not fsdp2_available():
        pytest.skip("torch has no FSDP2 fully_shard")
    if dist.get_world_size() != 2:
        pytest.skip("fsdp2 smoke expects exactly 2 ranks")


def _step(tools, handle, cfg, step, *, update):
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.zero_grad(handle)
    res = runtime.forward_backward(handle, iter([tools.batch(cfg, step, S)]), None, num_microbatches=1)
    loss = float(res.model_output.loss.item())
    gnorm = None
    if update:
        ok, gnorm, _ = runtime.optimizer_step(handle)
        assert ok, f"optimizer step {step} failed"
        runtime.zero_grad(handle)
    return loss, gnorm


def _max_over_ranks(value, dist):
    t = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


@pytest.mark.parametrize("name,parallel", _CASES, ids=[c[0] for c in _CASES])
def test_fsdp2_step_matches_dist_opt_loss(magi_cfg, magi_source, dist, runtime_tools, name, parallel):
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
    from megatron.lite.primitive.optimizers.fsdp2 import FSDP2Optimizer
    from megatron.lite.runtime.contracts.config import ParallelConfig

    _deps(dist)
    ref = runtime_tools.build_handle(magi_cfg, magi_source, ParallelConfig(**parallel), chunk_size=CHUNK)
    ref_loss, _ = _step(runtime_tools, ref, magi_cfg, 0, update=False)
    ps = ref._parallel_state
    del ref
    runtime_tools.reset_parallel_state(ps)

    handle = runtime_tools.build_handle(magi_cfg, magi_source, ParallelConfig(**parallel), chunk_size=CHUNK, optimizer="fsdp2")
    assert handle._extras["optimizer_backend"] == "fsdp2"
    assert isinstance(handle._optimizer, FSDP2Optimizer)
    chunk = unwrap_model(handle._model)
    assert isinstance(chunk, FSDPModule)  # fully_shard mutates the chunk in place: Magi attributes stay attached
    assert chunk.magi_settings is not None and chunk.magi_msa_config is not None and hasattr(chunk, "has_dense_attention")
    sharded = [n for n, p in chunk.named_parameters() if isinstance(p, DTensor)]
    assert sharded, "no parameter was sharded by FSDP2"
    for n, p in chunk.named_parameters():
        if ".indexer." in n:
            assert not p.requires_grad, n
    w0 = runtime_tools.indexer_weights(handle)

    loss0, gnorm0 = _step(runtime_tools, handle, magi_cfg, 0, update=True)
    loss1, gnorm1 = _step(runtime_tools, handle, magi_cfg, 1, update=True)
    rel = _max_over_ranks(abs(loss0 - ref_loss) / max(abs(ref_loss), 1e-6), dist)
    bitwise = _max_over_ranks(float(loss0 != ref_loss), dist) == 0.0
    for n, w in runtime_tools.indexer_weights(handle).items():
        assert torch.equal(w, w0[n]), f"indexer weight {n} changed"
    for n, p in chunk.named_parameters():
        if ".indexer." in n:
            assert p.grad is None, n
    if dist.get_rank() == 0:
        print(f"\nfsdp2 {name}: step-0 loss fsdp2 {loss0:.6f} vs dist_opt {ref_loss:.6f} (rel {rel:.2e}, bitwise={'yes' if bitwise else 'no'}) "
              f"| gnorm {gnorm0:.4f} -> step-1 loss {loss1:.6f} gnorm {gnorm1:.4f} | {len(sharded)} sharded params", flush=True)
    assert all(map(torch.isfinite, map(torch.tensor, (loss0, loss1, gnorm0, gnorm1))))
    assert rel < LOSS_REL, (name, loss0, ref_loss)
    ps = handle._parallel_state
    del handle
    runtime_tools.reset_parallel_state(ps, keep_groups=True)
