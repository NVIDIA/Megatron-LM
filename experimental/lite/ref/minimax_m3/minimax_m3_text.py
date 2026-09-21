# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Text-only Hugging Face reference for MiniMax-M3 (P0 golden).

transformers 5.16.1 ships a native ``minimax_m3_vl`` package whose
``MiniMaxM3VLForCausalLM`` + ``MiniMaxM3VLTextConfig`` already form a text-only
model (no vision tower / projector). This module pins that entry point and adds
the two loaders P0/P2 need:

* ``load_truncated(dir)``  – Truncated-M3 produced by ``tools/minimax_m3/slice_ckpt.py``.
* ``build_proxy()``        – Proxy-M3 from ``proxy_config.py`` (random init).

Both return ``(config, model)`` in eval mode. ``attn`` selects the HF attention
path; ``eager``/``sdpa`` materialise the MSA block mask densely (exact reference,
O(S^2) memory) and are the only paths that honour the sparse indexer.

Pinned reference: transformers==5.16.1 (``modeling_minimax_m3_vl.py`` md5
5df5ffe4cb1f6355a9987d1092006465); text-model code identical to GitHub main as
of 2026-09-09 (diff limited to the vision RoPE). Checkpoint revision
``MiniMaxAI/MiniMax-M3@f0e1c1e04d40177e4673a22097036854f536e9c0``.
"""

from __future__ import annotations

import os
import sys

import torch

PINNED_TRANSFORMERS = "5.16.1"
PINNED_CHECKPOINT_REVISION = "f0e1c1e04d40177e4673a22097036854f536e9c0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _check_transformers():
    import transformers

    if transformers.__version__ != PINNED_TRANSFORMERS:
        print(f"[minimax_m3_text] WARNING: transformers {transformers.__version__} != pinned {PINNED_TRANSFORMERS}")
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as m  # noqa: F401

    return m


def load_truncated(path: str, *, dtype=torch.bfloat16, device="cuda", attn="eager"):
    m = _check_transformers()
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLTextConfig

    cfg = MiniMaxM3VLTextConfig.from_pretrained(path)
    cfg._attn_implementation = attn
    model = m.MiniMaxM3VLForCausalLM.from_pretrained(path, config=cfg, dtype=dtype, device_map=device).eval()
    return cfg, model


def build_proxy(*, dtype=torch.bfloat16, device="cuda", attn="eager", seed=0):
    m = _check_transformers()
    from proxy_config import hf_proxy_text_config

    cfg = hf_proxy_text_config()
    cfg._attn_implementation = attn
    torch.manual_seed(seed)
    model = m.MiniMaxM3VLForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    return cfg, model


def describe(model) -> str:
    n = sum(p.numel() for p in model.parameters())
    layers = model.model.layers
    kinds = []
    for i, layer in enumerate(layers):
        a = "msa" if getattr(layer.self_attn, "indexer", None) is not None else "full"
        mlp = "moe" if hasattr(layer.mlp, "experts") else "dense"
        kinds.append(f"L{i}:{a}/{mlp}")
    return f"{n/1e9:.2f}B params; " + " ".join(kinds)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--truncated", default="", help="dir from slice_ckpt.py; omit to build the proxy")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--attn", default="eager")
    args = ap.parse_args()
    if args.truncated:
        cfg, model = load_truncated(args.truncated, attn=args.attn)
    else:
        cfg, model = build_proxy(attn=args.attn)
    print(describe(model))
    ids = torch.randint(0, cfg.vocab_size, (1, args.seq), device="cuda")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    print("logits", tuple(out.logits.shape), out.logits.dtype, "finite:", bool(torch.isfinite(out.logits).all()))
    print("max mem GB", torch.cuda.max_memory_allocated() / 1e9)
