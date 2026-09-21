# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""100-step training curves of the Magi-only Proxy-M3 runtime under CP/EP, bf16, 8 GPUs.

Every rank sees the same 4096-token stream. Baseline: Magi CP1. Cases: CP1 and CP2 through the
real runtime (``MegatronLiteRuntime.forward_backward`` -> Magi dispatch). Acceptance: mean
relative |delta| over 100 steps < 1% (plan curve criterion), and the indexer weights are bitwise unchanged after
training on every configuration (frozen selector; ``kl_loss_coeff=0`` alone would not guarantee this).

Set ``MLITE_MAGI_BENCH=1`` to also record step time / peak memory at 16K and 32K tokens (flex vs magi, cp1/cp2).
Run: torchrun --nproc-per-node=8 -m pytest -s <file> with MLITE_TEST_HARNESS=1, MAGI_ATTENTION_KERNEL_BACKEND=sdpa_ol.
"""

from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import pytest
import torch

_LITE = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.insert(0, os.path.join(_LITE, "ref", "minimax_m3"))
sys.path.insert(0, os.path.dirname(__file__))

pytestmark = [
    pytest.mark.gpus(8, min_architecture="blackwell"),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
    pytest.mark.timeout(seconds=3600),
]

S, STEPS, LR, CHUNK = 4096, 100, 1e-3, 512


def _init_dist_or_skip():
    import torch.distributed as dist

    if not torch.cuda.is_available() or "RANK" not in os.environ:
        pytest.skip("run with torchrun on GPUs")
    pytest.importorskip("megatron.core", reason="dist_opt needs megatron-core")
    pytest.importorskip("magi_attn_extensions.MSA")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if dist.get_world_size() != 8:
        pytest.skip("training-curve smoke expects exactly 8 ranks")
    return dist


def _proxy_config(all_msa: bool = False):
    from proxy_config import hf_proxy_text_config_kwargs

    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    hf = dict(hf_proxy_text_config_kwargs(magi=True))
    hf["model_type"] = "minimax_m3_vl_text"
    cfg = MiniMaxM3Config._from_hf_dict(hf)
    if all_msa:  # attribution bench: every layer through the MSA path (no dense-layer calc_attn / TE)
        cfg.layer_types = ["minimax_m3_sparse"] * cfg.num_hidden_layers
    return cfg


@pytest.fixture(scope="module")
def source_weights(tmp_path_factory):
    from megatron.lite.model.minimax_m3.lite.checkpoint import save_hf_weights
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import ParallelState

    dist = _init_dist_or_skip()
    cfg = _proxy_config()
    ps0 = ParallelState()
    tc = SimpleNamespace(tp=1, ep=1, etp=1, pp=1, cp=1, vpp=None, use_deepep=False, fp8=False, recompute_modules=[], deterministic=True)
    torch.manual_seed(20260914)
    ref = MiniMaxM3Model(cfg, tc, ps0, msa_backend="flex").to(torch.bfloat16).cuda()
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if n.endswith("norm.weight") or n.endswith("layer_norm_weight"):
                p.normal_(std=0.1)
        for layer in ref.layers:
            if layer.moe is not None:
                layer.moe.router.expert_bias.normal_(std=0.05)
    src = [str(tmp_path_factory.mktemp("hf_proxy_magi_train")) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(src, src=0)
    save_hf_weights(ref, src[0], cfg, ps0)
    del ref
    torch.cuda.empty_cache()
    return SimpleNamespace(cfg=cfg, src=src[0])


def _build_handle(cfg, src, parallel, *, load=True):
    from megatron.lite.model.minimax_m3.lite import protocol
    from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
    from megatron.lite.runtime.contracts.config import OptimizerConfig
    from megatron.lite.runtime.contracts.handle import ModelHandle

    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer="dist_opt",
        optimizer_config=OptimizerConfig(optimizer="adam", lr=LR, weight_decay=0.1, clip_grad=1.0),
        deterministic=False,
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
        model=bundle.chunks[0] if len(bundle.chunks) == 1 else bundle.chunks,
        optimizer=bundle.optimizer,
        parallel_state=bundle.parallel_state,
        config=SimpleNamespace(parallel=parallel),
        _extras=extras,
    )


def _batch(cfg, step, ps, seq_len=S):
    from megatron.lite.runtime.contracts.data import PackedBatch

    g = torch.Generator(device="cuda").manual_seed(100_000 + step)  # identical stream on every rank
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device="cuda", generator=g)
    labels = torch.randint(0, cfg.vocab_size, (1, seq_len), device="cuda", generator=g)
    ids, labels = ids.reshape(-1).contiguous(), labels.reshape(-1).contiguous()
    return PackedBatch(input_ids=ids, labels=labels, seq_lens=torch.tensor([ids.numel()], dtype=torch.int64, device="cuda"))


def _indexer_weights(handle):
    from megatron.lite.primitive.ckpt.hf_weights import unwrap_model

    chunks = handle._model if isinstance(handle._model, list) else [handle._model]
    return {n: p.detach().clone() for c in chunks for n, p in unwrap_model(c).named_parameters() if ".indexer." in n}


def _train(cfg, src, parallel, *, steps=STEPS, seq_len=S, bench=False, load=True):
    from test_minimax_m3_train_curve_smoke import _reset_parallel_state

    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    handle = _build_handle(cfg, src, parallel, load=load)
    ps = handle._parallel_state
    w0 = _indexer_weights(handle)
    losses, norms, step_times = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        runtime.zero_grad(handle)
        res = runtime.forward_backward(handle, iter([_batch(cfg, step, ps, seq_len)]), None, num_microbatches=1)
        ok, gnorm, _ = runtime.optimizer_step(handle)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        assert ok, f"optimizer step {step} failed"
        loss = res.model_output.loss
        losses.append(float(loss.item()) if loss is not None else float("nan"))
        norms.append(gnorm)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    for n, w in _indexer_weights(handle).items():  # frozen selector: bitwise unchanged after training
        assert torch.equal(w, w0[n]), f"indexer weight {n} changed during Magi training"
    runtime.zero_grad(handle)
    del handle, runtime
    _reset_parallel_state(ps)
    stats = SimpleNamespace(step_ms=1e3 * sum(step_times[2:]) / max(len(step_times) - 2, 1), peak_gib=peak_gib)
    return losses, norms, stats


def _curve_delta(losses, base):
    import torch.distributed as dist

    d, b = torch.tensor(losses, device="cuda"), torch.tensor(base, device="cuda")
    rel = (d - b).abs() / b.abs().clamp_min(1e-6)
    stats = torch.stack([rel.mean(), rel[-10:].mean()])
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    return float(stats[0]), float(stats[1])


@pytest.fixture(scope="module")
def baseline(source_weights):
    import torch.distributed as dist

    from megatron.lite.runtime.contracts.config import ParallelConfig

    losses, norms, st = _train(source_weights.cfg, source_weights.src, ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1))
    if dist.get_rank() == 0:
        print(f"\nmagi curve baseline cp1: loss[0]={losses[0]:.4f} loss[49]={losses[49]:.4f} loss[99]={losses[99]:.4f} "
              f"gnorm[0]={norms[0]:.3f} gnorm[99]={norms[99]:.3f} | {st.step_ms:.0f} ms/step, peak {st.peak_gib:.2f} GiB", flush=True)
    assert all(torch.isfinite(torch.tensor(losses)))
    return losses


_CASES = [("magi_cp1", dict(tp=1, ep=1, etp=1, pp=1, cp=1)), ("magi_cp2", dict(tp=1, ep=1, etp=1, pp=1, cp=2)),
          ("magi_cp2_ep2", dict(tp=1, ep=2, etp=1, pp=1, cp=2))]


@pytest.mark.parametrize("name,parallel", _CASES, ids=[c[0] for c in _CASES])
def test_magi_train_curve_tracks_cp1_baseline(source_weights, baseline, name, parallel):
    import torch.distributed as dist

    from megatron.lite.runtime.contracts.config import ParallelConfig

    losses, norms, st = _train(source_weights.cfg, source_weights.src, ParallelConfig(**parallel))
    mean_rel, tail_rel = _curve_delta(losses, baseline)
    if dist.get_rank() == 0:
        print(f"magi curve {name}: loss[0]={losses[0]:.4f} loss[49]={losses[49]:.4f} loss[99]={losses[99]:.4f} | "
              f"mean rel|delta| {mean_rel:.3e} last-10 {tail_rel:.3e} | gnorm[99]={norms[99]:.3f} | "
              f"{st.step_ms:.0f} ms/step, peak {st.peak_gib:.2f} GiB", flush=True)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert mean_rel < 1e-2, (name, mean_rel)
    assert tail_rel < 1e-2, (name, tail_rel)


@pytest.mark.skipif(os.environ.get("MLITE_MAGI_BENCH") != "1", reason="set MLITE_MAGI_BENCH=1")
@pytest.mark.parametrize("layers", ["full", "all_msa"])
@pytest.mark.parametrize("seq_len", [16384, 32768])
@pytest.mark.parametrize("cp", [1, 2])
def test_magi_bench_step_time_and_memory(source_weights, seq_len, cp, layers):
    """Magi step time / peak memory. ``all_msa`` attributes the cost to the MSA path alone (the dense
    layers' native calc_attn runs on the pure-torch sdpa_ol backend and dominates at small S)."""
    import torch.distributed as dist

    from megatron.lite.runtime.contracts.config import ParallelConfig

    cfg = source_weights.cfg if layers == "full" else _proxy_config(all_msa=True)
    _, _, st = _train(
        cfg,
        source_weights.src,
        ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=cp),
        steps=6,
        seq_len=seq_len,
        bench=True,
        load=layers == "full",
    )
    if dist.get_rank() == 0:
        print(f"BENCH {layers} S={seq_len} cp{cp} magi: {st.step_ms:.0f} ms/step, peak {st.peak_gib:.2f} GiB", flush=True)
