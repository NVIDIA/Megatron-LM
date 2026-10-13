# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Training curves of MiniMax-M3 lite through the real runtime (Magi protocol, bf16), 2 GPUs.

Every rank sees the same 4096-token stream, so the dp=2 run equals a single-GPU run. Each parallel
layout (EP2, PP2, CP2) and optimizer backend (dist_opt, fsdp2) starts from the same HF-format weights
and must track the dist_opt DP baseline loss curve: mean relative |delta| over the steps and over the
last 10 steps both < 1%. The indexer weights must be bitwise unchanged after training on every layout
(frozen selector; ``kl_loss_coeff=0`` alone would not guarantee this).

Optional: ``ImplConfig.deterministic=True`` must reproduce loss and grad-norm bitwise across two runs (all-MSA model).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.gpus(2, min_architecture="blackwell"),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
    pytest.mark.optional,
]

S, STEPS, LR, CHUNK = 4096, 50, 1e-3, 512
CURVE_REL = 1e-2
_DP2, _EP2, _PP2, _CP2 = (dict(tp=1, ep=1, etp=1, pp=1, cp=1), dict(tp=1, ep=2, etp=1, pp=1, cp=1),
                          dict(tp=1, ep=1, etp=1, pp=2, cp=1), dict(tp=1, ep=1, etp=1, pp=1, cp=2))
_CASES = [("ep2", _EP2, "alltoall", "dist_opt"), ("pp2", _PP2, "alltoall", "dist_opt"), ("cp2", _CP2, "alltoall", "dist_opt"),
          ("ep2_hybridep", _EP2, "hybridep", "dist_opt"),
          ("fsdp2_dp2", _DP2, "alltoall", "fsdp2"), ("fsdp2_ep2", _EP2, "alltoall", "fsdp2"),
          ("fsdp2_pp2", _PP2, "alltoall", "fsdp2"), ("fsdp2_cp2", _CP2, "alltoall", "fsdp2")]


def _deps(dist):
    pytest.importorskip("megatron.core", reason="dist_opt needs megatron-core")
    pytest.importorskip("magi_attn_extensions.MSA")
    pytest.importorskip("msa_v1")
    if dist.get_world_size() != 2:
        pytest.skip("training-curve smoke expects exactly 2 ranks")


def _all_msa_config(magi_hf_kwargs):
    """Every layer through the MSA path (random init; no dense-layer calc_attn)."""
    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    cfg = MiniMaxM3Config._from_hf_dict(magi_hf_kwargs)
    cfg.layer_types = ["minimax_m3_sparse"] * cfg.num_hidden_layers
    return cfg


def _train(tools, cfg, src, parallel, *, steps=STEPS, seq_len=S, load=True, deterministic=False, moe_dispatcher="alltoall",
           optimizer="dist_opt"):
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    handle = tools.build_handle(cfg, src, parallel, optimizer=optimizer, lr=LR, chunk_size=CHUNK, load=load,
                                deterministic=deterministic, moe_dispatcher=moe_dispatcher)
    ps = handle._parallel_state
    w0 = tools.indexer_weights(handle)
    losses, norms, step_times = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        runtime.zero_grad(handle)
        res = runtime.forward_backward(handle, iter([tools.batch(cfg, step, seq_len)]), None, num_microbatches=1)
        ok, gnorm, _ = runtime.optimizer_step(handle)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        assert ok, f"optimizer step {step} failed"
        loss = res.model_output.loss
        losses.append(float(loss.item()) if loss is not None else float("nan"))
        norms.append(gnorm)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    for n, w in tools.indexer_weights(handle).items():
        assert torch.equal(w, w0[n]), f"indexer weight {n} changed during training"
    runtime.zero_grad(handle)
    del handle, runtime
    tools.reset_parallel_state(ps, keep_groups=optimizer == "fsdp2")
    stats = SimpleNamespace(step_ms=1e3 * sum(step_times[2:]) / max(len(step_times) - 2, 1), peak_gib=peak_gib)
    return losses, norms, stats


def _curve_delta(losses, base, dist):
    d, b = torch.tensor(losses, device="cuda"), torch.tensor(base, device="cuda")
    rel = (d - b).abs() / b.abs().clamp_min(1e-6)
    stats = torch.stack([rel.mean(), rel[-10:].mean()])
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    return float(stats[0]), float(stats[1])


@pytest.fixture(scope="module")
def baseline(magi_cfg, magi_source, dist, runtime_tools):
    from megatron.lite.runtime.contracts.config import ParallelConfig

    _deps(dist)
    losses, norms, st = _train(runtime_tools, magi_cfg, magi_source, ParallelConfig(**_DP2))
    if dist.get_rank() == 0:
        print(f"\ntrain_curve baseline dp2: loss[0]={losses[0]:.4f} loss[{STEPS // 2}]={losses[STEPS // 2]:.4f} "
              f"loss[-1]={losses[-1]:.4f} gnorm[0]={norms[0]:.3f} gnorm[-1]={norms[-1]:.3f} | {st.step_ms:.0f} ms/step, "
              f"peak {st.peak_gib:.2f} GiB", flush=True)
    assert all(torch.isfinite(torch.tensor(losses)))
    return losses


@pytest.mark.parametrize("name,parallel,moe_dispatcher,optimizer", _CASES, ids=[c[0] for c in _CASES])
def test_train_curve_tracks_dp_baseline(magi_cfg, magi_source, baseline, dist, runtime_tools, name, parallel, moe_dispatcher,
                                        optimizer):
    from megatron.lite.runtime.contracts.config import ParallelConfig

    if moe_dispatcher == "hybridep":
        pytest.importorskip("deep_ep")
    if optimizer == "fsdp2":
        from megatron.lite.primitive.optimizers.fsdp2 import fsdp2_available

        if not fsdp2_available():
            pytest.skip("torch has no FSDP2 fully_shard")
    losses, norms, st = _train(runtime_tools, magi_cfg, magi_source, ParallelConfig(**parallel), moe_dispatcher=moe_dispatcher,
                               optimizer=optimizer)
    mean_rel, tail_rel = _curve_delta(losses, baseline, dist)
    if dist.get_rank() == 0:
        print(f"train_curve {name}: loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f} | mean rel|delta| {mean_rel:.3e} "
              f"last-10 {tail_rel:.3e} | gnorm[-1]={norms[-1]:.3f} | {st.step_ms:.0f} ms/step, peak {st.peak_gib:.2f} GiB",
              flush=True)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert mean_rel < CURVE_REL, (name, mean_rel)
    assert tail_rel < CURVE_REL, (name, tail_rel)


def _first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1


@pytest.mark.optional
@pytest.mark.parametrize("cp", [1, 2])
def test_deterministic_mode_reproduces_bitwise(magi_hf_kwargs, magi_source, dist, runtime_tools, cp):
    from megatron.lite.runtime.contracts.config import ParallelConfig

    _deps(dist)
    steps, cfg = 20, _all_msa_config(magi_hf_kwargs)
    (l0, n0, st0), (l1, n1, st1) = [
        _train(runtime_tools, cfg, magi_source, ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=cp), steps=steps, load=False,
               deterministic=True)
        for _ in range(2)
    ]
    same = torch.tensor([int(l0 == l1 and n0 == n1)], device="cuda")
    dist.all_reduce(same, op=dist.ReduceOp.MIN)
    if dist.get_rank() == 0:
        print(f"\ndeterministic all_msa cp{cp} ({steps} steps): bitwise={'yes' if same.item() else 'NO'} "
              f"(first divergence loss {_first_divergence(l0, l1)} / gnorm {_first_divergence(n0, n1)}) "
              f"{(st0.step_ms + st1.step_ms) / 2:.0f} ms/step", flush=True)
    assert same.item(), cp
