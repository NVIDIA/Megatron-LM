# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""P2 real-weight alignment: Truncated-M3 (embed + L0..L5 + norm + head, 25.7B params) lite vs HF.

Optional (needs the sliced checkpoint and ~110 GB of GPU memory for fp32):
    MINIMAX_M3_TRUNCATED_DIR=/path/MiniMax-M3-trunc6 NVIDIA_TF32_OVERRIDE=0 MLITE_TEST_HARNESS=1 \
        python -m pytest tests/unit/model/test_minimax_m3_truncated_optional.py -s

Runs HF (fp32) first and keeps per-layer dumps on CPU, frees it, then runs lite (fp32) loaded
through the weight spec (HF-module spelling with packed experts -> candidates path).
fp32 gate: per-layer rel < 1e-5, logits KL < 1e-4, MSA top-k sets equal (margin-aware report).
"""

from __future__ import annotations

import gc
import os
import sys
from types import SimpleNamespace

import pytest
import torch

_LITE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(_LITE, "ref", "minimax_m3"))
sys.path.insert(0, os.path.join(_LITE, "tools", "minimax_m3"))

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]
DEV = "cuda"
CKPT = os.environ.get("MINIMAX_M3_TRUNCATED_DIR", "")
CKPT_GRAD = os.environ.get("MINIMAX_M3_TRUNCATED_GRAD_DIR", "")  # smaller slice (4 layers) for fp32 fwd+bwd


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _build_fp32_flex_oracle(cfg):
    from megatron.lite.model.minimax_m3.lite.model import MiniMaxM3Model
    from megatron.lite.primitive.parallel import ParallelState

    ps = ParallelState()
    train_cfg = SimpleNamespace(
        tp=1, ep=1, etp=1, pp=1, cp=1, vpp=None, use_deepep=False,
        fp8=False, recompute_modules=[], deterministic=True,
    )
    return MiniMaxM3Model(cfg, train_cfg, ps, msa_backend="flex").float().cuda(), ps


@pytest.mark.skipif(not CKPT, reason="set MINIMAX_M3_TRUNCATED_DIR")
@pytest.mark.parametrize("S", [int(os.environ.get("MINIMAX_M3_TRUNCATED_SEQ", "2048"))])
def test_truncated_m3_matches_hf_fp32(S):
    from align_harness import LayerDumper, logits_metrics
    from minimax_m3_text import load_truncated
    from thresholds import FP32, assert_fp32_env

    from megatron.lite.model.minimax_m3.lite import protocol as P

    assert_fp32_env()
    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29539")
        dist.init_process_group("nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)

    torch.manual_seed(1234)
    # ---------------- HF fp32
    hf_cfg, hf = load_truncated(CKPT, dtype=torch.float32, device=DEV, attn="eager")
    ids = torch.randint(0, hf_cfg.vocab_size, (1, S), device=DEV)
    d = LayerDumper(hf, patterns=(r"model\.layers\.\d+", r"model\.layers\.\d+\.self_attn\.indexer"))
    with torch.no_grad():
        hf_logits = hf(input_ids=ids, use_cache=False).logits.float().cpu()
    d.remove()
    hf_dump = dict(d.dump)
    n_layers = hf_cfg.num_hidden_layers
    del hf, d
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- lite fp32
    cfg = P.build_model_config(CKPT)
    lite, ps = _build_fp32_flex_oracle(cfg)
    P.load_hf_weights(lite, CKPT, cfg, ps)
    lite.eval()
    d = LayerDumper(lite, patterns=(r"layers\.\d+",))
    with torch.no_grad():
        lite_logits = lite(input_ids=ids)["logits"].float().cpu()
    d.remove()

    rows = []
    for i in range(n_layers):
        r = _rel(d.dump[f"layers.{i}"].transpose(0, 1), hf_dump[f"model.layers.{i}"])
        rows.append((i, r))
    lm = logits_metrics(lite_logits, hf_logits)
    flips = {}
    for i, layer in enumerate(lite.layers):
        if layer.is_sparse_attention:
            l_idx = layer.attn.last_block_indices.sort(-1).values.cpu()
            h_idx = hf_dump[f"model.layers.{i}.self_attn.indexer"].sort(-1).values.to(torch.int32)
            flips[i] = (l_idx != h_idx).any(-1).float().mean().item()
    print(f"\nTruncated-M3 fp32 S={S}: layer rel {[(i, f'{r:.2e}') for i, r in rows]} | logits {lm} | MSA set flips {flips}")
    for i, r in rows:
        assert r < FP32.module_rel, (i, r)
    assert lm["kl_mean"] < FP32.logits_kl and lm["logprob_mean_abs_delta"] < FP32.logits_logprob_delta, lm
    for i, f in flips.items():
        assert f == 0.0, f"layer {i}: {f:.3%} MSA rows differ"


def _hf_grad_lookup(hf_grads: dict, mod: str):
    import re

    g = hf_grads.get(mod)
    if g is not None:
        return g
    em = re.match(r"^(.*\.experts)\.(\d+)\.(w1|w2|w3)\.weight$", mod)
    if em:
        e = int(em.group(2))
        if em.group(3) == "w2":
            return hf_grads[f"{em.group(1)}.down_proj"][e]
        gate, up = hf_grads[f"{em.group(1)}.gate_up_proj"][e].chunk(2, dim=0)
        return gate if em.group(3) == "w1" else up
    fm = re.match(r"^(.*)\.(gate|up)_proj\.weight$", mod)
    if fm:
        gate, up = hf_grads[f"{fm.group(1)}.gate_up_proj.weight"].chunk(2, dim=0)
        return gate if fm.group(2) == "gate" else up
    raise KeyError(mod)


@pytest.mark.skipif(not CKPT_GRAD, reason="set MINIMAX_M3_TRUNCATED_GRAD_DIR (4-layer slice)")
def test_truncated_m3_grads_match_hf_fp32():
    """fp32 fwd+bwd on real weights (4-layer slice): every parameter gradient rel < 1e-5."""
    from minimax_m3_text import load_truncated
    from thresholds import FP32, assert_fp32_env

    from megatron.lite.model.minimax_m3.lite import protocol as P
    from megatron.lite.model.minimax_m3.lite.checkpoint import (
        MiniMaxM3WeightSpec,
        disk_to_module_name,
    )

    assert_fp32_env()
    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29539")
        dist.init_process_group("nccl", rank=0, world_size=1)
        torch.cuda.set_device(0)
    S = int(os.environ.get("MINIMAX_M3_TRUNCATED_GRAD_SEQ", "1024"))
    torch.manual_seed(4321)

    hf_cfg, hf = load_truncated(CKPT_GRAD, dtype=torch.float32, device=DEV, attn="eager")
    hf.train()
    ids = torch.randint(0, hf_cfg.vocab_size, (1, S), device=DEV)
    labels = torch.randint(0, hf_cfg.vocab_size, (1, S), device=DEV)
    logits = hf(input_ids=ids, use_cache=False).logits
    hf_loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1))
    hf_loss.backward()
    hf_grads = {n: p.grad.detach().cpu() for n, p in hf.named_parameters() if p.grad is not None}
    hf_loss_val = hf_loss.item()
    del hf, logits, hf_loss
    gc.collect()
    torch.cuda.empty_cache()

    cfg = P.build_model_config(CKPT_GRAD)
    lite, ps = _build_fp32_flex_oracle(cfg)
    P.load_hf_weights(lite, CKPT_GRAD, cfg, ps)
    lite.train()
    out = lite(input_ids=ids, labels=labels)
    out["loss"].backward()
    assert abs(out["loss"].item() - hf_loss_val) / hf_loss_val < 1e-5, (out["loss"].item(), hf_loss_val)

    spec = MiniMaxM3WeightSpec(cfg)
    worst, checked, worst_name = 0.0, 0, ""
    for name, p in lite.named_parameters():
        if p.grad is None or "indexer" in name or ".router.expert_bias" in name:
            continue
        for hf_name, piece in spec.native_to_hf(name, p.grad):
            g = _hf_grad_lookup(hf_grads, disk_to_module_name(hf_name)).to(DEV)
            if piece.shape != g.shape:
                g = torch.nn.functional.pad(g, (0, 0, 0, piece.shape[0] - g.shape[0]))
            r = _rel(piece, g)
            if r > worst:
                worst, worst_name = r, hf_name
            checked += 1
            assert r < FP32.grad_rel, (name, hf_name, r)
    print(f"\nTruncated-M3 (4 layers) fp32 grads: {checked} tensors, worst rel {worst:.2e} ({worst_name}); loss {hf_loss_val:.6f}")
    assert checked > 100
