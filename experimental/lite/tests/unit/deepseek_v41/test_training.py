# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Random small full-model gates; real Core/TE/FHT/FlashMLA, never HF weights."""
import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _ds41_fixture import tiny_config


def _initialize(model, ps):
    with torch.no_grad():
        for name, value in model.named_parameters():
            if value.ndim > 1:
                key = re.sub(
                    r"(experts.linear_fc[12].weight)(\d+)",
                    lambda m: m[1] + str(int(m[2]) + ps.ep_rank * (8 // ps.ep_size)),
                    name,
                )
                generator = torch.Generator().manual_seed(
                    int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                )
                value.copy_(torch.randn(value.shape, generator=generator) * 0.02)
        for name, module in model.named_modules():
            lookup = getattr(module, "lookup", None)
            if lookup is not None:
                generator = torch.Generator().manual_seed(
                    int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
                )
                full = (
                    torch.randn(lookup.boundaries[-1], module.weight.shape[1], generator=generator)
                    * 0.02
                )
                a, b = lookup.boundaries[lookup.group.rank() : lookup.group.rank() + 2]
                module.weight.copy_(full[a:b])


def _worker(rank, root, world, ep, cp, pp, recompute, sft):
    from megatron.lite.runtime import create_runtime
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.contracts import LossContext, PackedBatch, ParallelConfig
    from megatron.lite.runtime.contracts.config import RuntimeConfig

    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    dist.init_process_group("nccl", f"file://{root}/store", rank=rank, world_size=world)
    torch.use_deterministic_algorithms(not sft, warn_only=True)
    options = dict(
        token_map=list(range(128)),
        deterministic_mode=not sft,
        attention_backend_override="cudnn",
        optimizer="adam" if sft else None,
        optimizer_config=dict(lr=0.001, min_lr=0.001, weight_decay=0.0),
    )
    if pp > 1:
        options["pipeline_split_layer"] = [3] if pp == 2 else [1, 3]
    options["recompute"] = recompute
    cfg = MegatronLiteConfig(
        model_name="deepseek_v41",
        impl="lite",
        hf_path=root,
        parallel=ParallelConfig(ep=ep, cp=cp, pp=pp),
        load_hf_weights=False,
        attention_backend_override=None,
        impl_cfg=options,
    )
    rt = create_runtime(RuntimeConfig(backend="mlite", hf_path=root, backend_cfg=cfg))
    handle = rt.build_model()
    model, ps = handle._model, handle._parallel_state
    while hasattr(model, "module"):
        model = model.module
    if not sft:
        _initialize(model, ps)
        assert all(m.config.deterministic_mode for m in model.modules() if hasattr(m, "config"))
    captured, losses = [], []

    def loss_fn(output, batch, ctx=None):
        mask = output["valid_mask"]
        captured.append(
            (output["rows"][mask].detach().cpu(), output["log_probs"][mask].detach().cpu())
        )
        return -(output["log_probs"] * mask).sum() / mask.sum().clamp_min(1), {}

    record = {}
    for step in range(24 if sft else 2):
        captured.clear()
        ids = (torch.arange(81, device="cuda") + (step % 8 if sft else 0)) % 127 + 1
        batch = PackedBatch(ids, ids, torch.tensor([5, 1, 75], device="cuda"))
        enabled = sft or bool(step)
        with rt.train_mode(handle), torch.set_grad_enabled(enabled):
            rt.zero_grad(handle)
            result = rt.forward_backward(
                handle,
                iter([(batch, LossContext(return_log_probs=True))]),
                loss_fn=loss_fn,
                forward_only=not enabled,
            )
        if enabled:
            grads = {
                n: p.main_grad if hasattr(p, "main_grad") else p.grad
                for n, p in model.named_parameters()
                if p.requires_grad
            }
            assert all(g is not None and torch.isfinite(g).all() for g in grads.values()), [
                n for n, g in grads.items() if g is None
            ]
            names = "linear_kv_proj.weight linear_q_up_proj.weight compressor.linear_wkv.weight main_projection.weight".split()
            main = [g for n, g in grads.items() if any(k in n for k in names)]
            assert main and all(g.abs().sum() > 0 for g in main)
        if sft:
            accepted, norm, _ = rt.optimizer_step(handle)
            assert accepted and torch.isfinite(torch.as_tensor(norm))
            losses.append(float(result.model_output.loss))
        elif ps.pp_is_last and ps.dp_rank == 0:
            record[str(step)] = [
                (rows.tolist(), values.float().tolist()) for rows, values in captured
            ]
    if sft:
        assert sum(losses[-7:]) <= sum(losses[:7]) * 0.8, losses
        record["losses"] = losses
    record["peak_bytes"] = torch.cuda.max_memory_allocated()
    (Path(root) / f"rank-{rank}.json").write_text(json.dumps(record))
    dist.destroy_process_group()


def _run(root, shape, sft=False):
    root.mkdir()
    (root / "config.json").write_text(json.dumps(asdict(tiny_config())))
    ep, cp, pp, recompute = shape
    world = max(ep, cp * pp)
    mp.start_processes(
        _worker, (str(root), world, ep, cp, pp, recompute, sft), nprocs=world, start_method="spawn"
    )
    records = [json.loads((root / f"rank-{r}.json").read_text()) for r in range(world)]
    if sft:
        return records
    phases = []
    for phase in ("0", "1"):
        rows = [
            (i, value)
            for rec in records
            for ids, vals in rec.get(phase, [])
            for i, value in zip(ids, vals)
        ]
        assert len(rows) == len({i for i, _ in rows}) == 78  # exclude all three document ends
        phases.append(torch.tensor([v for _, v in sorted(rows)]))
    assert torch.equal(*phases), "grad-mode changed forward log-probabilities"
    return phases[0]


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((e, c, p, m), marks=pytest.mark.gpus(max(e, c * p)))
        for e, c, p, m in [
            (2, 1, 1, None),
            (1, 2, 1, None),
            (1, 1, 2, None),
            (1, 1, 3, None),
            (1, 2, 2, None),
        ]
        + [(1, 1, 1, m) for m in ("full", "moe", "attn")]
    ],
)
def test_small_model_parallel_and_grad_mode_are_bitwise(tmp_path, monkeypatch, shape):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    reference = _run(tmp_path / "reference", (1, 1, 1, None))
    actual = _run(tmp_path / "parallel", shape)
    if shape[1] == 1:
        assert torch.equal(reference, actual)
    else:
        (tmp_path / "cp-report.json").write_text(
            json.dumps({"report_only_max_abs": (reference - actual).abs().max().item()})
        )


@pytest.mark.gpus(1)
def test_small_model_sft_loss_decreases(tmp_path):
    _run(tmp_path / "sft", (1, 1, 1, None), sft=True)
