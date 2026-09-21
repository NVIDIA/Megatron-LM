# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Layer-wise alignment harness (P0 tool, non-product).

* ``LayerDumper``: forward hooks on selected modules -> ``{name: tensor}`` (fp32, detached).
* ``compare_dumps``: max-abs / rel-to-max / mean-abs per module, in module order.
* ``logits_metrics``: per-token logprob mean|Δ| and mean KL for two logits tensors.
* ``first_failure``: first module whose metric exceeds a threshold.

CLI:

    python align_harness.py noise-floor  --runs 3 --batch 2 --seq 2048 --dtype bf16 [--attn eager|sdpa]
    python align_harness.py bf16-vs-fp32 --batch 2 --seq 2048 [--attn eager|sdpa] [--out x.json]

``noise-floor`` runs the HF Proxy-M3 (random init, fixed seed) ``--runs`` times
on identical input and reports the per-layer run-to-run spread. Measured 0.0
(bitwise) on B200 for both eager and sdpa, so it cannot serve as a threshold.

``bf16-vs-fp32`` runs the same weights/input in bf16 and fp32 and reports the
per-layer gap. This gap is the **bf16 precision scale** used by later stages:
mlite-bf16 vs HF-bf16 per-module max-abs must stay within ``k x`` this gap
(plan revision 2026-09-10, k = 2 unless review_threshold says otherwise).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
def _first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for o in out:
            t = _first_tensor(o)
            if t is not None:
                return t
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if hasattr(out, "logits"):
        return out.logits
    return None


class LayerDumper:
    """Collect module outputs by name. ``patterns`` are regexes matched with ``re.fullmatch``."""

    DEFAULT_PATTERNS = (
        r".*embed_tokens",
        r".*layers\.\d+",
        r".*layers\.\d+\.self_attn",
        r".*layers\.\d+\.self_attn\.indexer",
        r".*layers\.\d+\.mlp",
        r".*\bnorm",
        r".*lm_head",
    )

    def __init__(self, model: nn.Module, patterns=DEFAULT_PATTERNS, to_cpu: bool = True):
        self.dump: dict[str, torch.Tensor] = {}
        self.order: list[str] = []
        self._handles = []
        self.to_cpu = to_cpu
        for name, mod in model.named_modules():
            if any(re.fullmatch(p, name) for p in patterns):
                self._handles.append(mod.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(_mod, _inp, out):
            t = _first_tensor(out)
            if t is None:
                return
            t = t.detach()
            if t.dtype not in (torch.int32, torch.int64, torch.bool):
                t = t.float()
            self.dump[name] = t.cpu() if self.to_cpu else t.clone()
            if name not in self.order:
                self.order.append(name)
        return fn

    def clear(self):
        self.dump, self.order = {}, []

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


@dataclass
class Diff:
    name: str
    max_abs: float
    rel_to_max: float
    mean_abs: float
    ref_max: float
    bitwise: bool


def compare_dumps(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], order=None) -> list[Diff]:
    names = order or [n for n in a if n in b]
    out = []
    for n in names:
        if n not in a or n not in b:
            continue
        x, y = a[n], b[n]
        if x.shape != y.shape:
            out.append(Diff(n, math.inf, math.inf, math.inf, float("nan"), False))
            continue
        if x.dtype in (torch.int32, torch.int64, torch.bool):
            # Discrete selections (e.g. top-k block ids): order within a row is irrelevant, compare as sets.
            # max_abs = fraction of rows whose selected SET differs; rel_to_max = fraction of sorted slots that differ;
            # mean_abs = fraction of raw (unsorted) slots that differ; bitwise = all sets identical.
            xs, ys = x.sort(-1).values, y.sort(-1).values
            row = (xs != ys).any(-1).float().mean().item()
            slot_sorted = (xs != ys).float().mean().item()
            slot_raw = (x != y).float().mean().item()
            out.append(Diff(n, row, slot_sorted, slot_raw, 0.0, row == 0.0))
            continue
        d = (x - y).abs()
        ref_max = y.abs().max().item()
        out.append(Diff(n, d.max().item(), d.max().item() / (ref_max + 1e-30), d.mean().item(), ref_max, bool(torch.equal(x, y))))
    return out


def logits_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """``a``/``b``: ``[..., vocab]``. Returns mean |Δ logprob| (per-token, over the argmax-of-b token) and mean KL(b || a)."""
    la = torch.log_softmax(a.float(), -1)
    lb = torch.log_softmax(b.float(), -1)
    tok = lb.argmax(-1, keepdim=True)
    dlp = (la.gather(-1, tok) - lb.gather(-1, tok)).abs().mean().item()
    kl = (lb.exp() * (lb - la)).sum(-1).mean().item()
    return {"logprob_mean_abs_delta": dlp, "kl_mean": kl}


def first_failure(diffs: list[Diff], threshold: float, key: str = "rel_to_max") -> Diff | None:
    for d in diffs:
        if getattr(d, key) > threshold:
            return d
    return None


def print_table(diffs: list[Diff], title: str = ""):
    if title:
        print(title)
    print(f"  {'module':48s} {'max_abs':>11s} {'rel_to_max':>11s} {'mean_abs':>11s} {'ref_max':>10s} bitwise")
    for d in diffs:
        print(f"  {d.name:48s} {d.max_abs:11.3e} {d.rel_to_max:11.3e} {d.mean_abs:11.3e} {d.ref_max:10.3e} {d.bitwise}")


# --------------------------------------------------------------------------- #
def build_hf_proxy(dtype: torch.dtype, attn: str, device: str, seed: int = 0):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref", "minimax_m3"))
    from proxy_config import hf_proxy_text_config  # noqa: E402
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLForCausalLM

    cfg = hf_proxy_text_config()
    cfg._attn_implementation = attn
    torch.manual_seed(seed)
    model = MiniMaxM3VLForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    return cfg, model


def cmd_noise_floor(args):
    device = "cuda"
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[args.dtype]
    cfg, model = build_hf_proxy(dtype, args.attn, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Proxy-M3: {n_params/1e6:.1f}M params, dtype={dtype}, attn={args.attn}, layers={cfg.num_hidden_layers}")
    torch.manual_seed(1234)
    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)
    dumper = LayerDumper(model)
    runs = []
    for r in range(args.runs):
        dumper.clear()
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        d = dict(dumper.dump)
        d["__logits__"] = out.logits.detach().float().cpu()
        runs.append((d, list(dumper.order) + ["__logits__"]))
    order = runs[0][1]
    worst: dict[str, Diff] = {}
    for i in range(1, len(runs)):
        diffs = compare_dumps(runs[i][0], runs[0][0], order)
        print_table(diffs, f"\nrun{i} vs run0")
        lm = logits_metrics(runs[i][0]["__logits__"], runs[0][0]["__logits__"])
        print(f"  logits: {lm}")
        for d in diffs:
            if d.name not in worst or d.max_abs > worst[d.name].max_abs:
                worst[d.name] = d
    summary = {"config": {"batch": args.batch, "seq": args.seq, "dtype": args.dtype, "attn": args.attn, "runs": args.runs},
               "noise_floor": {k: asdict(v) for k, v in worst.items()}}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
        print("wrote", args.out)
    print("\nnoise floor (max over runs) — max_abs per module:")
    for k, v in worst.items():
        print(f"  {k:48s} {v.max_abs:.3e}  bitwise={v.bitwise}")


def cmd_bf16_vs_fp32(args):
    device = "cuda"
    if args.truncated:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ref", "minimax_m3"))
        from minimax_m3_text import load_truncated  # noqa: E402

        cfg, model = load_truncated(args.truncated, dtype=torch.float32, device=device, attn=args.attn)
        print("Truncated-M3 fp32 loaded; mem GB", torch.cuda.memory_allocated() / 1e9)
    else:
        cfg, model = build_hf_proxy(torch.float32, args.attn, device)
    torch.manual_seed(1234)
    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)
    dumps = {}
    orders = {}
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        model.to(dtype)
        dumper = LayerDumper(model)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        d = dict(dumper.dump)
        d["__logits__"] = out.logits.detach().float().cpu()
        dumps[name], orders[name] = d, list(dumper.order) + ["__logits__"]
        dumper.remove()
    diffs = compare_dumps(dumps["bf16"], dumps["fp32"], orders["fp32"])
    tag = "Truncated-M3" if args.truncated else "Proxy-M3"
    print_table(diffs, f"\nHF bf16 vs HF fp32 ({tag}, attn={args.attn}, B={args.batch}, S={args.seq})")
    lm = logits_metrics(dumps["bf16"]["__logits__"], dumps["fp32"]["__logits__"])
    print(f"  logits: {lm}")
    summary = {"config": {"batch": args.batch, "seq": args.seq, "attn": args.attn, "truncated": args.truncated},
               "bf16_vs_fp32": {d.name: asdict(d) for d in diffs}, "logits": lm}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
        print("wrote", args.out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    nf = sub.add_parser("noise-floor")
    nf.add_argument("--runs", type=int, default=3)
    nf.add_argument("--batch", type=int, default=2)
    nf.add_argument("--seq", type=int, default=2048)
    nf.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    nf.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    nf.add_argument("--out", default="")
    nf.set_defaults(fn=cmd_noise_floor)
    bv = sub.add_parser("bf16-vs-fp32")
    bv.add_argument("--batch", type=int, default=2)
    bv.add_argument("--seq", type=int, default=2048)
    bv.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    bv.add_argument("--out", default="")
    bv.add_argument("--truncated", default="", help="Truncated-M3 dir (real weights); default: Proxy-M3")
    bv.set_defaults(fn=cmd_bf16_vs_fp32)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
