# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Training curves of MiniMax-M3 lite through the real runtime (Magi protocol, dist_opt, bf16), 2 GPUs.

Every rank sees the same 4096-token stream, so the dp=2 run equals a single-GPU run. Each parallel
layout (EP2, PP2, CP2) starts from the same HF-format weights and must track the DP baseline loss curve:
mean relative |delta| over the steps and over the last 10 steps both < 1%. The indexer weights must be
bitwise unchanged after training on every layout (frozen selector; ``kl_loss_coeff=0`` alone would not
guarantee this).

Optional (explicit selection only):
* ``test_bench_step_time_and_memory`` records step time / peak memory at 16K tokens; the ``all_msa`` variant
  routes every layer through MSA to attribute the cost to the MSA path alone.
* ``test_deterministic_mode_reproduces_bitwise``: ``ImplConfig.deterministic=True`` (ordered msa_v1
  backward) must reproduce loss and grad-norm bitwise across two identical runs; ``deterministic=False``
  is recorded for reference. Runs on the all-MSA model because the dense layers' fa4 backward is unordered.
"""

from __future__ import annotations

import gc
import time
from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.gpus(2, min_architecture="blackwell"),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
    pytest.mark.timeout(seconds=1800),
]

S, STEPS, LR, CHUNK = 4096, 50, 1e-3, 512
CURVE_REL = 1e-2
_CASES = [("ep2", dict(tp=1, ep=2, etp=1, pp=1, cp=1)), ("pp2", dict(tp=1, ep=1, etp=1, pp=2, cp=1)),
          ("cp2", dict(tp=1, ep=1, etp=1, pp=1, cp=2))]


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


def _build_handle(cfg, src, parallel, *, load=True, deterministic=False):
    from megatron.lite.model.minimax_m3.lite import protocol
    from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
    from megatron.lite.runtime.contracts.config import OptimizerConfig
    from megatron.lite.runtime.contracts.handle import ModelHandle

    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer="dist_opt",
        optimizer_config=OptimizerConfig(optimizer="adam", lr=LR, weight_decay=0.1, clip_grad=1.0),
        deterministic=deterministic,
        magi_chunk_size=CHUNK,
    )
    torch.manual_seed(1)
    bundle = protocol.build_model(cfg, impl_cfg=impl_cfg)
    if load:
        for chunk in bundle.chunks:
            protocol.load_hf_weights(unwrap_model(chunk), src, cfg, bundle.parallel_state)
    reload = getattr(bundle.optimizer, "reload_model_params", None)
    if callable(reload):
        reload()
    extras = dict(bundle.extras)
    extras.update(model_chunks=bundle.chunks, forward_step=bundle.forward_step, finalize_grads=bundle.finalize_grads,
                  protocol=protocol, model_cfg=cfg)
    return ModelHandle(
        model=bundle.chunks[0] if len(bundle.chunks) == 1 else bundle.chunks,  # as MegatronLiteRuntime.build_model
        optimizer=bundle.optimizer,
        parallel_state=bundle.parallel_state,
        config=SimpleNamespace(parallel=parallel),
        _extras=extras,
    )


def _batch(cfg, step, seq_len):
    from megatron.lite.runtime.contracts.data import PackedBatch

    g = torch.Generator(device="cuda").manual_seed(100_000 + step)  # identical stream on every rank
    ids = torch.randint(0, cfg.vocab_size, (seq_len,), device="cuda", generator=g)
    labels = torch.randint(0, cfg.vocab_size, (seq_len,), device="cuda", generator=g)
    return PackedBatch(input_ids=ids, labels=labels, seq_lens=torch.tensor([seq_len], dtype=torch.int64, device="cuda"))


def _indexer_weights(handle):
    from megatron.lite.primitive.ckpt.hf_weights import unwrap_model

    chunks = handle._model if isinstance(handle._model, list) else [handle._model]
    return {n: p.detach().clone() for c in chunks for n, p in unwrap_model(c).named_parameters() if ".indexer." in n}


_PS_GROUP_ATTRS = ("tp_group", "ep_group", "etp_group", "cp_group", "pp_group", "pp_cpu_group", "dp_group", "dp_cp_group",
                   "tp_ep_group", "ep_dp_group")


def _reset_parallel_state(ps):
    """dist_opt initialises mcore's global parallel state per topology; tear it (and lite's groups) down between cases."""
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    if mpu.is_initialized():
        mpu.destroy_model_parallel()
    gc.collect()
    for attr in _PS_GROUP_ATTRS:
        group = getattr(ps, attr, None)
        if group is not None:
            try:
                dist.destroy_process_group(group)
            except Exception:
                pass
    torch.cuda.empty_cache()


def _train(cfg, src, parallel, *, steps=STEPS, seq_len=S, load=True, deterministic=False):
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    handle = _build_handle(cfg, src, parallel, load=load, deterministic=deterministic)
    ps = handle._parallel_state
    w0 = _indexer_weights(handle)
    losses, norms, step_times = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        runtime.zero_grad(handle)
        res = runtime.forward_backward(handle, iter([_batch(cfg, step, seq_len)]), None, num_microbatches=1)
        ok, gnorm, _ = runtime.optimizer_step(handle)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        assert ok, f"optimizer step {step} failed"
        loss = res.model_output.loss
        losses.append(float(loss.item()) if loss is not None else float("nan"))
        norms.append(gnorm)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    for n, w in _indexer_weights(handle).items():
        assert torch.equal(w, w0[n]), f"indexer weight {n} changed during training"
    runtime.zero_grad(handle)
    del handle, runtime
    _reset_parallel_state(ps)
    stats = SimpleNamespace(step_ms=1e3 * sum(step_times[2:]) / max(len(step_times) - 2, 1), peak_gib=peak_gib)
    return losses, norms, stats


def _curve_delta(losses, base, dist):
    d, b = torch.tensor(losses, device="cuda"), torch.tensor(base, device="cuda")
    rel = (d - b).abs() / b.abs().clamp_min(1e-6)
    stats = torch.stack([rel.mean(), rel[-10:].mean()])
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    return float(stats[0]), float(stats[1])


@pytest.fixture(scope="module")
def baseline(magi_cfg, magi_source, dist):
    from megatron.lite.runtime.contracts.config import ParallelConfig

    _deps(dist)
    losses, norms, st = _train(magi_cfg, magi_source, ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1))
    if dist.get_rank() == 0:
        print(f"\ntrain_curve baseline dp2: loss[0]={losses[0]:.4f} loss[{STEPS // 2}]={losses[STEPS // 2]:.4f} "
              f"loss[-1]={losses[-1]:.4f} gnorm[0]={norms[0]:.3f} gnorm[-1]={norms[-1]:.3f} | {st.step_ms:.0f} ms/step, "
              f"peak {st.peak_gib:.2f} GiB", flush=True)
    assert all(torch.isfinite(torch.tensor(losses)))
    return losses


@pytest.mark.parametrize("name,parallel", _CASES, ids=[c[0] for c in _CASES])
def test_train_curve_tracks_dp_baseline(magi_cfg, magi_source, baseline, dist, name, parallel):
    from megatron.lite.runtime.contracts.config import ParallelConfig

    losses, norms, st = _train(magi_cfg, magi_source, ParallelConfig(**parallel))
    mean_rel, tail_rel = _curve_delta(losses, baseline, dist)
    if dist.get_rank() == 0:
        print(f"train_curve {name}: loss[0]={losses[0]:.4f} loss[-1]={losses[-1]:.4f} | mean rel|delta| {mean_rel:.3e} "
              f"last-10 {tail_rel:.3e} | gnorm[-1]={norms[-1]:.3f} | {st.step_ms:.0f} ms/step, peak {st.peak_gib:.2f} GiB",
              flush=True)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert mean_rel < CURVE_REL, (name, mean_rel)
    assert tail_rel < CURVE_REL, (name, tail_rel)


@pytest.mark.optional
@pytest.mark.parametrize("layers", ["full", "all_msa"])
@pytest.mark.parametrize("cp", [1, 2])
def test_bench_step_time_and_memory(magi_cfg, magi_hf_kwargs, magi_source, dist, layers, cp):
    from megatron.lite.runtime.contracts.config import ParallelConfig

    _deps(dist)
    seq_len = 16384
    cfg = magi_cfg if layers == "full" else _all_msa_config(magi_hf_kwargs)
    _, _, st = _train(cfg, magi_source, ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=cp), steps=6, seq_len=seq_len,
                      load=layers == "full")
    if dist.get_rank() == 0:
        print(f"BENCH {layers} S={seq_len} cp{cp}: {st.step_ms:.0f} ms/step, peak {st.peak_gib:.2f} GiB", flush=True)


def _first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1


@pytest.mark.optional
@pytest.mark.parametrize("cp", [1, 2])
def test_deterministic_mode_reproduces_bitwise(magi_hf_kwargs, magi_source, dist, cp):
    from megatron.lite.runtime.contracts.config import ParallelConfig

    _deps(dist)
    steps, cfg = 20, _all_msa_config(magi_hf_kwargs)
    runs = {}
    for det in (True, False):
        (l0, n0, st0), (l1, n1, st1) = [
            _train(cfg, magi_source, ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=cp), steps=steps, load=False, deterministic=det)
            for _ in range(2)
        ]
        same = torch.tensor([int(l0 == l1 and n0 == n1)], device="cuda")
        dist.all_reduce(same, op=dist.ReduceOp.MIN)
        runs[det] = SimpleNamespace(same=bool(same.item()), loss_div=_first_divergence(l0, l1),
                                    norm_div=_first_divergence(n0, n1), step_ms=(st0.step_ms + st1.step_ms) / 2)
    d, nd = runs[True], runs[False]
    if dist.get_rank() == 0:
        print(f"\ndeterministic all_msa cp{cp} ({steps} steps): det=True bitwise={'yes' if d.same else 'NO'} "
              f"(first divergence {d.loss_div}/{d.norm_div}) {d.step_ms:.0f} ms/step | det=False bitwise="
              f"{'yes' if nd.same else 'no'} (first divergence {nd.loss_div}/{nd.norm_div}) {nd.step_ms:.0f} ms/step | "
              f"det/nondet step-time x{d.step_ms / nd.step_ms:.2f}", flush=True)
    assert d.same, (cp, d)
