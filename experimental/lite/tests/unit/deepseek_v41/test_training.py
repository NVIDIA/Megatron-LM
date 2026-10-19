# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Random small full-model gates; real Core/TE/FHT/FlashMLA, synthetic release-format weights, no downloaded weights."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _ds41_fixture import tiny_config


def _initialize(model, ps, root):
    from safetensors.torch import save_file

    from megatron.lite.model.deepseek_v41.lite.checkpoint import (
        _expert_sources,
        load_hf_weights,
        source_name,
    )

    tensors = {}

    def put(key, value, table=False):
        generator = torch.Generator().manual_seed(
            int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        )
        if value.ndim == 2 and key.endswith(".weight"):
            rows, cols = value.shape
            fp4 = ".ffn.experts." in key
            dense = key in ("embed.weight", "head.weight") or ".ffn.gate." in key
            if not dense:
                value = (
                    torch.randint(0, 256, (rows, cols // 2), generator=generator).to(torch.int8)
                    if fp4
                    else torch.randn(rows, cols, generator=generator).to(torch.float8_e4m3fn)
                )
                scale_rows = rows if fp4 or table else -(-rows // 32)
                scale = torch.full(
                    (scale_rows, -(-cols // 32)), 119 if fp4 else 121, dtype=torch.uint8
                )
                tensors[key[:-7] + ".scale"] = scale.view(torch.float8_e8m0fnu)
            else:
                value = (torch.randn(rows, cols, generator=generator) * 0.02).bfloat16()
        elif key.endswith("_fn"):
            value = torch.randn(value.shape, generator=generator) * 0.02
        tensors[key] = value.contiguous()

    state = list(model.named_parameters()) + [
        (n, t)
        for n, t in model.named_buffers()
        if n.endswith(("expert_bias", "tid2eid", "embed.embedding.weight"))
    ]
    for name, tensor in state:
        if name.endswith(".ape"):
            continue
        value = tensor.detach().cpu()
        keys = _expert_sources(name, model.release, ps)
        if name.startswith("engrams."):
            _, layer, suffix = name.split(".", 2)
            if suffix == "embed.embedding.weight":
                value = torch.empty(model.engrams[layer].global_rows, value.shape[1])
                suffix = "embed.weight"
            put(f"layers.{layer}.engram.{suffix}", value, table=suffix == "embed.weight")
        elif keys:
            for key, chunk in zip(keys, value.chunk(len(keys))):
                put(key, chunk)
        elif name.endswith(("alpha_pre", "alpha_post", "alpha_res")):
            slot = ("alpha_pre", "alpha_post", "alpha_res").index(name.rsplit(".", 1)[1])
            tensors.setdefault(source_name(name), torch.empty(3))[slot] = value.reshape(())
        else:
            put(source_name(name), value)
    path = Path(root) / f"release-rank{dist.get_rank()}"
    path.mkdir(exist_ok=True)
    save_file(tensors, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors")})
    )
    load_hf_weights(model, str(path), model.release, ps)


def _build(root, ep=1, cp=1, pp=1, recompute=None, sft=False):
    from megatron.lite.runtime import create_runtime
    from megatron.lite.runtime.backends.mlite.config import MegatronLiteConfig
    from megatron.lite.runtime.contracts import ParallelConfig
    from megatron.lite.runtime.contracts.config import RuntimeConfig

    torch.manual_seed(42)
    options = dict(
        token_map=list(range(128)),
        deterministic_mode=not sft,
        attention_backend_override="cudnn",
        optimizer="adam" if sft else None,
        optimizer_config=dict(lr=0.001, min_lr=0.001, weight_decay=0.0),
    )
    if pp > 1:
        options["pipeline_split_layer"] = [9] if pp == 2 else [3, 9]
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
    return rt, handle, model, ps


def _worker(rank, root, world, ep, cp, pp, recompute, sft):
    import importlib.abc
    import sys

    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "vllm" or fullname.startswith("vllm."):
                raise AssertionError("Lite execution requested vllm: " + fullname)

    assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)
    sys.meta_path.insert(0, Guard())
    from megatron.lite.runtime.contracts import LossContext, PackedBatch

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", f"file://{root}/store", rank=rank, world_size=world)
    torch.use_deterministic_algorithms(not sft, warn_only=True)
    rt, handle, model, ps = _build(root, ep, cp, pp, recompute, sft)
    _initialize(model, ps, root)
    if sft:
        handle._optimizer.reload_model_params()
    if not sft:
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
def test_small_model_parallel_contract_and_grad_mode(tmp_path, monkeypatch, shape):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    reference = _run(tmp_path / "reference", (1, shape[1] if shape[2] > 1 else 1, 1, None))
    actual = _run(tmp_path / "parallel", shape)
    if shape[1] == 1 or shape[2] > 1:
        error = (reference - actual).abs()
        assert error.max() <= 0.125 and torch.quantile(error, 0.99) <= 0.03
        torch.testing.assert_close(reference.mean(), actual.mean(), atol=0.02, rtol=0.005)
    else:
        (tmp_path / "cp-report.json").write_text(
            json.dumps({"report_only_max_abs": (reference - actual).abs().max().item()})
        )


@pytest.mark.gpus(1)
def test_small_model_sft_loss_decreases(tmp_path):
    _run(tmp_path / "sft", (1, 1, 1, None), sft=True)
