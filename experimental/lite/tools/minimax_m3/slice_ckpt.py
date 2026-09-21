# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Slice a Truncated-M3 text-only checkpoint out of the real MiniMax-M3 shards (P0 tool).

Input : HF repo dir containing ``model.safetensors.index.json`` and at least the
        shards that hold ``embed_tokens``, layers ``0..L-1``, ``norm`` and ``lm_head``
        (for L=6 that is ``model-00001..00005-of-00059``).
Output: a directory loadable by
        ``transformers.MiniMaxM3VLForCausalLM.from_pretrained(out_dir)`` with
        ``num_hidden_layers = L``. Tensor names are written in the HF *module*
        spelling (the on-disk checkpoint uses the sglang spelling; the renames
        below mirror ``transformers/conversion_mapping.py`` for ``minimax_m3_vl``):

    language_model.model.X                          -> model.X
    language_model.lm_head                          -> lm_head
    .block_sparse_moe.                              -> .mlp.
    .e_score_correction_bias                        -> .gate.e_score_correction_bias
    .self_attn.index_{q,k}_{proj,norm}              -> .self_attn.indexer.{q,k}_{proj,norm}
    .experts.E.w1 / w3  (E=0..127)                  -> .experts.gate_up_proj  [E, 2*I, H]  (cat(w1, w3) on dim 1)
    .experts.E.w2                                   -> .experts.down_proj     [E, H, I]
    .gate_proj / .up_proj (dense MLP, shared exp.)  -> .gate_up_proj          [2*I, H]     (cat(gate, up) on dim 0)

Vision tower / projector / patch-merge tensors are dropped (prefix filter).

    python slice_ckpt.py --src /path/MiniMax-M3-partial --dst /path/MiniMax-M3-trunc6 --layers 6
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_LAYER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.")


def wanted(name: str, layers: int) -> bool:
    if not name.startswith("language_model."):
        return False
    m = _LAYER_RE.match(name)
    return int(m.group(1)) < layers if m else True


def rename_simple(name: str) -> str:
    name = re.sub(r"^language_model\.lm_head", "lm_head", name)
    name = re.sub(r"^language_model\.model\.", "model.", name)
    name = name.replace(".block_sparse_moe.", ".mlp.")
    name = name.replace(".mlp.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")
    name = re.sub(r"\.self_attn\.index_(q|k)_(proj|norm)\.", r".self_attn.indexer.\1_\2.", name)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--shard-gb", type=float, default=10.0, help="max output shard size")
    args = ap.parse_args()

    with open(os.path.join(args.src, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    with open(os.path.join(args.src, "config.json")) as f:
        full_cfg = json.load(f)
    tcfg = dict(full_cfg["text_config"])

    names = [n for n in weight_map if wanted(n, args.layers)]
    shards = sorted({weight_map[n] for n in names})
    print(f"{len(names)} tensors from {len(shards)} shards: {shards}")
    for s in shards:
        if not os.path.exists(os.path.join(args.src, s)):
            raise FileNotFoundError(f"missing shard {s}")

    # --- gather, rename, fuse
    out: dict[str, torch.Tensor] = {}
    experts: dict[tuple[str, str], dict[int, torch.Tensor]] = {}  # (prefix, w1|w2|w3) -> {E: tensor}
    gate_up: dict[str, dict[str, torch.Tensor]] = {}  # prefix -> {gate|up: tensor}
    n_experts = tcfg["num_local_experts"]
    for s in shards:
        with safe_open(os.path.join(args.src, s), framework="pt") as f:
            for n in f.keys():
                if not wanted(n, args.layers):
                    continue
                t = f.get_tensor(n)
                nn_ = rename_simple(n)
                m = re.match(r"^(.*\.experts)\.(\d+)\.(w[123])\.weight$", nn_)
                if m:
                    experts.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = t
                    continue
                m = re.match(r"^(.*)\.(gate|up)_proj\.weight$", nn_)
                if m:
                    gate_up.setdefault(m.group(1), {})[m.group(2)] = t
                    continue
                out[nn_] = t
    for (prefix, w), d in list(experts.items()):
        if w == "w2":
            continue
        assert len(d) == n_experts and len(experts[(prefix, "w3")]) == n_experts, prefix
        w1 = torch.stack([d[e] for e in range(n_experts)])  # [E, I, H]
        w3 = torch.stack([experts[(prefix, "w3")][e] for e in range(n_experts)])
        out[f"{prefix}.gate_up_proj"] = torch.cat([w1, w3], dim=1).contiguous()  # [E, 2I, H]
    for (prefix, w), d in experts.items():
        if w == "w2":
            out[f"{prefix}.down_proj"] = torch.stack([d[e] for e in range(n_experts)]).contiguous()  # [E, H, I]
    for prefix, d in gate_up.items():
        out[f"{prefix}.gate_up_proj.weight"] = torch.cat([d["gate"], d["up"]], dim=0).contiguous()

    total = sum(t.numel() * t.element_size() for t in out.values())
    print(f"{len(out)} output tensors, {total/1e9:.1f} GB")

    # --- write shards
    os.makedirs(args.dst, exist_ok=True)
    limit = int(args.shard_gb * 1e9)
    cur, cur_size, idx, wm = {}, 0, 0, {}
    items = sorted(out.items())
    def flush():
        nonlocal cur, cur_size, idx
        if not cur:
            return
        fn = f"model-{idx:05d}.safetensors"
        save_file(cur, os.path.join(args.dst, fn), metadata={"format": "pt"})
        for k in cur:
            wm[k] = fn
        print("wrote", fn, f"{cur_size/1e9:.1f} GB")
        cur, cur_size, idx = {}, 0, idx + 1
    for k, t in items:
        sz = t.numel() * t.element_size()
        if cur and cur_size + sz > limit:
            flush()
        cur[k] = t
        cur_size += sz
    flush()
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": wm}, f, indent=1)

    # --- text-only config
    L = args.layers
    tcfg["num_hidden_layers"] = L
    tcfg["moe_layer_freq"] = tcfg["moe_layer_freq"][:L]
    sac = dict(tcfg["sparse_attention_config"])
    for k in ("sparse_attention_freq", "sparse_disable_index_value"):
        sac[k] = sac[k][:L]
    tcfg["sparse_attention_config"] = sac
    tcfg["architectures"] = ["MiniMaxM3VLForCausalLM"]
    tcfg["model_type"] = "minimax_m3_vl_text"
    tcfg["torch_dtype"] = full_cfg.get("torch_dtype", "bfloat16")
    tcfg["_source"] = {"repo": "MiniMaxAI/MiniMax-M3", "sliced_layers": L}
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(tcfg, f, indent=1)
    for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
               "vocab.json", "merges.txt", "chat_template.jinja", "generation_config.json"):
        p = os.path.join(args.src, fn)
        if os.path.exists(p):
            shutil.copy(p, args.dst)
    print("done ->", args.dst)


if __name__ == "__main__":
    main()
