# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""P3: 100-step Magi training curves of Proxy-M3 under EP/PP/CP (dist_opt, bf16) vs DP.

Every rank sees the same token stream, so the dp=8 run equals a single-GPU run; each parallel
configuration starts from the same HF-format weights and must track the baseline loss curve
(mean relative |delta| over the 100 steps < 1%, per the P3 acceptance criteria). bf16 training
numbers are recorded; the 1% band is the plan's curve criterion, not the fp32 module gate.

Run: torchrun --nproc-per-node=8 -m pytest -s <file> with MLITE_TEST_HARNESS=1 (megatron-core needed).
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_LITE = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
sys.path.insert(0, os.path.join(_LITE, "ref", "minimax_m3"))

pytestmark = [
    pytest.mark.gpus(8, min_architecture="blackwell"),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
    pytest.mark.timeout(seconds=1800),
]

S, STEPS, LR = 1024, 100, 1e-3


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


def _proxy_config():
    from proxy_config import hf_proxy_text_config_kwargs

    from megatron.lite.model.minimax_m3.config import MiniMaxM3Config

    hf = dict(hf_proxy_text_config_kwargs(magi=True))
    hf["model_type"] = "minimax_m3_vl_text"
    return MiniMaxM3Config._from_hf_dict(hf)


@pytest.fixture(scope="module")
def source_weights(tmp_path_factory):
    """One bf16 Proxy-M3 saved in HF format so every configuration starts from identical weights."""
    from megatron.lite.model.minimax_m3.lite.checkpoint import save_hf_weights
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import ParallelState

    dist = _init_dist_or_skip()
    cfg = _proxy_config()
    ps0 = ParallelState()
    tc = SimpleNamespace(tp=1, ep=1, etp=1, pp=1, cp=1, vpp=None, use_deepep=False, fp8=False, recompute_modules=[], deterministic=True)
    torch.manual_seed(20260911)
    ref = MiniMaxM3Model(cfg, tc, ps0, msa_backend="flex").to(torch.bfloat16).cuda()
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if n.endswith("norm.weight") or n.endswith("layer_norm_weight"):
                p.normal_(std=0.1)
        for layer in ref.layers:
            if layer.moe is not None:
                layer.moe.router.expert_bias.normal_(std=0.05)
    src = [str(tmp_path_factory.mktemp("hf_proxy_train")) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(src, src=0)
    save_hf_weights(ref, src[0], cfg, ps0)
    del ref
    torch.cuda.empty_cache()
    return SimpleNamespace(cfg=cfg, src=src[0])


def _build_handle(cfg, src, parallel):
    from megatron.lite.model.minimax_m3.lite import protocol
    from megatron.lite.primitive.ckpt.hf_weights import unwrap_model
    from megatron.lite.runtime.contracts.config import OptimizerConfig
    from megatron.lite.runtime.contracts.handle import ModelHandle

    impl_cfg = protocol.ImplConfig(
        parallel=parallel,
        optimizer="dist_opt",
        optimizer_config=OptimizerConfig(optimizer="adam", lr=LR, weight_decay=0.0, clip_grad=1.0),
        deterministic=False,
    )
    torch.manual_seed(1)
    bundle = protocol.build_model(cfg, impl_cfg=impl_cfg)
    for chunk in bundle.chunks:
        protocol.load_hf_weights(unwrap_model(chunk), src, cfg, bundle.parallel_state)
    reload = getattr(bundle.optimizer, "reload_model_params", None)
    if callable(reload):
        reload()
    extras = dict(bundle.extras)
    extras.update(
        model_chunks=bundle.chunks,
        forward_step=bundle.forward_step,
        finalize_grads=bundle.finalize_grads,
        protocol=protocol,
        model_cfg=cfg,
    )
    return ModelHandle(
        model=bundle.chunks[0] if len(bundle.chunks) == 1 else bundle.chunks,  # as MegatronLiteRuntime.build_model
        optimizer=bundle.optimizer,
        parallel_state=bundle.parallel_state,
        config=SimpleNamespace(parallel=parallel),
        _extras=extras,
    )


def _batch(cfg, step, ps):
    from megatron.lite.runtime.contracts.data import PackedBatch

    g = torch.Generator(device="cuda").manual_seed(100_000 + step)  # identical stream on every rank
    ids = torch.randint(0, cfg.vocab_size, (1, S), device="cuda", generator=g)
    labels = torch.randint(0, cfg.vocab_size, (1, S), device="cuda", generator=g)
    # Magi plans and dispatches the identical global packed batch on every CP rank.
    ids = ids.reshape(-1).contiguous()
    labels = labels.reshape(-1).contiguous()
    return PackedBatch(input_ids=ids, labels=labels, seq_lens=torch.tensor([ids.numel()], dtype=torch.int64, device="cuda"))


def _train(cfg, src, parallel) -> tuple[list[float], list[float]]:
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    handle = _build_handle(cfg, src, parallel)
    ps = handle._parallel_state
    losses, norms = [], []
    for step in range(STEPS):
        runtime.zero_grad(handle)
        res = runtime.forward_backward(handle, iter([_batch(cfg, step, ps)]), None, num_microbatches=1)
        ok, gnorm, _ = runtime.optimizer_step(handle)
        assert ok, f"optimizer step {step} failed"
        loss = res.model_output.loss
        losses.append(float(loss.item()) if loss is not None else float("nan"))
        norms.append(gnorm)
    runtime.zero_grad(handle)
    del handle, runtime
    _reset_parallel_state(ps)
    return losses, norms


_PS_GROUP_ATTRS = ("tp_group", "ep_group", "etp_group", "cp_group", "pp_group", "pp_cpu_group", "dp_group", "dp_cp_group", "tp_ep_group", "ep_dp_group")


def _reset_parallel_state(ps) -> None:
    """dist_opt initialises mcore's global parallel state per topology; tear it (and lite's groups) down between cases."""
    import gc

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


def _curve_delta(losses, base) -> tuple[float, float]:
    import torch.distributed as dist

    d = torch.tensor(losses, device="cuda")
    b = torch.tensor(base, device="cuda")
    rel = ((d - b).abs() / b.abs().clamp_min(1e-6))
    stats = torch.stack([rel.mean(), rel[-10:].mean()])
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    return float(stats[0]), float(stats[1])


_CASES = [
    ("pp2", dict(tp=1, ep=1, etp=1, pp=2, cp=1)),
    ("ep2", dict(tp=1, ep=2, etp=1, pp=1, cp=1)),
    ("cp2", dict(tp=1, ep=1, etp=1, pp=1, cp=2)),
    ("ep2_cp2", dict(tp=1, ep=2, etp=1, pp=1, cp=2)),
]


@pytest.fixture(scope="module")
def baseline(source_weights):
    from megatron.lite.runtime.contracts.config import ParallelConfig

    losses, norms = _train(source_weights.cfg, source_weights.src, ParallelConfig(tp=1, ep=1, etp=1, pp=1, cp=1))
    import torch.distributed as dist

    if dist.get_rank() == 0:
        print(f"\nP3 curve baseline dp8: loss[0]={losses[0]:.4f} loss[49]={losses[49]:.4f} loss[99]={losses[99]:.4f} "
              f"gnorm[0]={norms[0]:.3f} gnorm[99]={norms[99]:.3f}", flush=True)
    assert all(torch.isfinite(torch.tensor(losses)))
    return losses


@pytest.mark.parametrize("name,parallel", _CASES, ids=[c[0] for c in _CASES])
def test_train_curve_tracks_dp_baseline(source_weights, baseline, name, parallel):
    import torch.distributed as dist

    from megatron.lite.runtime.contracts.config import ParallelConfig

    cfg, src = source_weights.cfg, source_weights.src
    losses, norms = _train(cfg, src, ParallelConfig(**parallel))
    mean_rel, tail_rel = _curve_delta(losses, baseline)
    if dist.get_rank() == 0:
        print(f"P3 curve {name}: loss[0]={losses[0]:.4f} loss[49]={losses[49]:.4f} loss[99]={losses[99]:.4f} | "
              f"mean rel|delta| {mean_rel:.3e} last-10 {tail_rel:.3e} | gnorm[99]={norms[99]:.3f}", flush=True)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert mean_rel < 1e-2, (name, mean_rel)
    assert tail_rel < 1e-2, (name, tail_rel)
