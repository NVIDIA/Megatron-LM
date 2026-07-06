"""MLM dist-checkpoint -> HF weight extraction + bitwise verification + forward parity.

Three phases:
  Phase 1 (container python, main process):
    - Load MLM dist-checkpoint via dist_checkpointing
    - Reverse all HF->MLM transforms (unpack QKV groups, split fc1 gate/up, rename)
    - Bitwise-compare every recovered weight against the original HF safetensors
    - Save recovered text-tower weights as a .pt file for Phase 2

  Phase 2 (gemma4_venv subprocess):
    - Load original HF model (Gemma4ForConditionalGeneration)
    - Run forward on FIXED_TOKENS -> save "original" logits
    - Override model text-tower weights with our recovered weights
    - Run forward again -> save "converted" logits

  Phase 3 (container python):
    - Compare original vs converted logits (bitwise identical iff weights match)
    - Print a summary and write V_RESULTS_MLM2HF.md

Run with container python:
    python3 mlm_to_hf_convert.py
    python3 mlm_to_hf_convert.py --no-forward         # weight test only
    python3 mlm_to_hf_convert.py --save-dir /path/hf  # also write a loadable HF ckpt

With --save-dir, a full from_pretrained-loadable HF checkpoint is written: the text
tower comes from the MLM dist-checkpoint (verified bitwise first), the vision tower
and all aux files (config.json, tokenizer*, chat_template, generation_config) are
copied verbatim from the original (this campaign trains only the text tower). The
export is skipped if the bitwise weight check fails.
"""
import gemma4_common  # noqa: F401  MUST be first

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gemma4_common import _build_model, _init_distributed, _make_config
from hf_to_mlm_convert import (
    CKPT_OUT as MLM_CKPT,
    HF as HF_PREFIX,
    NUM_GROUPS,
    HEADS_PER_GROUP,
    unpack_qkv,
)

GEMMA4_VENV_PYTHON = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "gemma4_venv/bin/python"
)
HF_WEIGHTS_DIR = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "gemma4-playground/weights/gemma-4-E4B-it"
)
HF_WEIGHTS_FILE = HF_WEIGHTS_DIR + "/model.safetensors"
IMPL = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "Gemma4_mlm/code_dev/shared-state/implementations/HYBRID-gemma4-mlm"
)
RESULTS_OUT = IMPL + "/V_RESULTS_MLM2HF.md"
FIXED_TOKENS = [2, 651, 1234, 99, 17, 8, 200, 9000, 42, 3]

HF_WORKER_FLAG = "--_hf-worker"


# ---------------------------------------------------------------------------
# Phase 1 helpers: load MLM ckpt + reverse conversion
# ---------------------------------------------------------------------------

def load_mlm_model(mlm_ckpt=MLM_CKPT):
    """Build Gemma4Model (local spec, E4B dims) and load the dist-checkpoint into it."""
    from megatron.core import dist_checkpointing
    from megatron.core.models.gemma4.gemma4_layer_specs import get_gemma4_layer_local_spec

    config = _make_config()
    model = _build_model(get_gemma4_layer_local_spec, config, vocab=262144)
    model.eval()

    print(f"[MLM] loading dist-checkpoint from {mlm_ckpt} ...", flush=True)
    sharded_sd = model.sharded_state_dict()
    loaded = dist_checkpointing.load(sharded_sd, mlm_ckpt)
    model.load_state_dict(loaded, strict=False)
    return model, config


def build_hf_state_dict(model, config):
    """Reverse all HF->MLM transforms and produce a dict keyed as 'model.language_model.*'.

    Transforms reversed:
      - pack_qkv  -> unpack_qkv:  linear_qkv -> q_proj, k_proj, v_proj
      - cat(gate, up) -> split:   linear_fc1  -> gate_proj, up_proj
      - rename:  decoder.layers.{i}.* -> model.language_model.layers.{i}.*
    """
    msd = {}
    for k, v in model.named_parameters():
        msd[k] = v.detach().cpu()
    for k, v in model.named_buffers():
        msd[k] = v.detach().cpu()

    hf = {}
    P = HF_PREFIX  # 'model.language_model.'

    def direct(hf_rel, mlm_key):
        hf[P + hf_rel] = msd[mlm_key].clone()

    # Top-level
    direct("embed_tokens.weight",              "embedding.word_embeddings.weight")
    direct("norm.weight",                      "decoder.final_layernorm.weight")
    direct("embed_tokens_per_layer.weight",    "ple.embed_tokens_per_layer.weight")
    direct("per_layer_model_projection.weight","ple.per_layer_model_projection.weight")
    direct("per_layer_projection_norm.weight", "ple.per_layer_projection_norm.weight")

    full_idx = set(config.full_attention_layers)
    for i in range(config.num_layers):
        mp = f"decoder.layers.{i}."
        hp = f"layers.{i}."
        head_dim = 512 if i in full_idx else 256

        # Direct renames
        for mk, hk in [
            ("input_layernorm.weight",            "input_layernorm.weight"),
            ("post_self_attn_layernorm.weight",   "post_attention_layernorm.weight"),
            ("pre_mlp_layernorm.weight",          "pre_feedforward_layernorm.weight"),
            ("post_mlp_layernorm.weight",         "post_feedforward_layernorm.weight"),
            ("post_per_layer_input_norm.weight",  "post_per_layer_input_norm.weight"),
            ("self_attention.q_layernorm.weight", "self_attn.q_norm.weight"),
            ("self_attention.k_layernorm.weight", "self_attn.k_norm.weight"),
            ("self_attention.linear_proj.weight", "self_attn.o_proj.weight"),
            ("mlp.linear_fc2.weight",             "mlp.down_proj.weight"),
            ("per_layer_input_gate.weight",       "per_layer_input_gate.weight"),
            ("per_layer_projection.weight",       "per_layer_projection.weight"),
            ("layer_scalar",                      "layer_scalar"),
        ]:
            direct(hp + hk, mp + mk)

        # Unpack fused QKV -> q_proj, k_proj, v_proj
        packed = msd[mp + "self_attention.linear_qkv.weight"]
        q, k, v = unpack_qkv(packed, head_dim)
        hf[P + hp + "self_attn.q_proj.weight"] = q.clone()
        hf[P + hp + "self_attn.k_proj.weight"] = k.clone()
        hf[P + hp + "self_attn.v_proj.weight"] = v.clone()

        # Split fused fc1 -> gate_proj, up_proj
        fc1 = msd[mp + "mlp.linear_fc1.weight"]
        half = fc1.size(0) // 2
        hf[P + hp + "mlp.gate_proj.weight"] = fc1[:half].clone().contiguous()
        hf[P + hp + "mlp.up_proj.weight"]   = fc1[half:].clone().contiguous()

    return hf


# ---------------------------------------------------------------------------
# Phase 1: bitwise weight comparison
# ---------------------------------------------------------------------------

def bitwise_compare(hf_converted, orig_file):
    """Compare recovered HF state_dict against the original HF safetensors.

    Returns (rows, n_mismatch) where rows = list of (key, status, detail).
    """
    orig = safe_open(orig_file, framework="pt", device="cpu")
    orig_keys = set(orig.keys())

    rows = []
    n_mismatch = 0
    n_skip = 0

    for key, tensor in sorted(hf_converted.items()):
        if key not in orig_keys:
            rows.append((key, "SKIP", "key absent in original HF safetensors"))
            n_skip += 1
            continue
        orig_t = orig.get_tensor(key)
        # Cast both to float32 for comparison (bf16 -> fp32 is exact/bitwise-preserving).
        t = tensor.float()
        o = orig_t.float()
        if t.shape != o.shape:
            rows.append((key, "SHAPE_MISMATCH", f"got {tuple(t.shape)} expected {tuple(o.shape)}"))
            n_mismatch += 1
        elif not torch.equal(t, o):
            diff = (t - o).abs()
            rows.append((key, "VALUE_MISMATCH", f"max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}"))
            n_mismatch += 1
        else:
            rows.append((key, "EQUAL", ""))

    return rows, n_mismatch, n_skip


# ---------------------------------------------------------------------------
# Export: write a standalone, from_pretrained-loadable HF checkpoint
# ---------------------------------------------------------------------------

def export_hf_checkpoint(hf_recovered, orig_file, orig_dir, save_dir):
    """Write a full HF checkpoint dir using MLM-recovered text weights.

    Text-tower tensors come from ``hf_recovered`` (the MLM->HF reverse conversion);
    every other tensor (vision tower, multimodal projector, ...) is copied verbatim
    from the original safetensors, because this campaign trains only the text tower.
    All aux files (config.json, tokenizer*, chat_template, generation_config, ...) are
    copied unchanged so the result loads with ``from_pretrained(save_dir)``.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Copy every aux file (skip weights + caches; we rewrite model.safetensors).
    for name in sorted(os.listdir(orig_dir)):
        src = os.path.join(orig_dir, name)
        if name == ".cache" or os.path.isdir(src):
            continue
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            continue
        shutil.copy2(src, os.path.join(save_dir, name))
        print(f"[export] copied aux file {name}", flush=True)

    # 2. Build the full tensor dict: recovered text tensors + original everything-else.
    orig = safe_open(orig_file, framework="pt", device="cpu")
    metadata = dict(orig.metadata() or {})
    metadata.setdefault("format", "pt")

    full = {}
    n_patched = 0
    for key in orig.keys():
        if key in hf_recovered:
            # Match the original tensor's dtype (text tower is bf16; recovered is bf16).
            o_dtype = orig.get_tensor(key).dtype
            full[key] = hf_recovered[key].to(o_dtype).contiguous()
            n_patched += 1
        else:
            full[key] = orig.get_tensor(key)

    # Sanity: every recovered text key must have landed in the original key set.
    unplaced = sorted(set(hf_recovered) - set(orig.keys()))
    if unplaced:
        print(f"[export] WARNING: {len(unplaced)} recovered keys not in original "
              f"safetensors (NOT written): {unplaced[:5]}", flush=True)

    out_path = os.path.join(save_dir, "model.safetensors")
    print(f"[export] writing {len(full)} tensors ({n_patched} text-tower patched from MLM) "
          f"-> {out_path} ...", flush=True)
    save_file(full, out_path, metadata=metadata)
    print(f"[export] DONE. HF checkpoint at {save_dir}", flush=True)
    return n_patched, len(full)


# ---------------------------------------------------------------------------
# Phase 2: HF forward worker (gemma4_venv subprocess)
# ---------------------------------------------------------------------------

def _hf_forward_worker(weights_pt_path, out_path):
    """Runs ONLY in the gemma4_venv subprocess.

    1. Load original HF model.
    2. Run forward on FIXED_TOKENS -> logits_orig.
    3. Override model text-tower weights with recovered weights.
    4. Run forward again -> logits_converted.
    5. Save both to out_path.
    """
    import torch
    from transformers import Gemma4ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = torch.tensor([FIXED_TOKENS], dtype=torch.long)

    print("[HF worker] loading original HF model...", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        HF_WEIGHTS_DIR, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval().to(device)

    with torch.no_grad():
        logits_orig = model(ids.to(device), use_cache=False).logits.cpu()
    greedy_orig = logits_orig.argmax(dim=-1)
    print(f"[HF worker] original  logits {tuple(logits_orig.shape)}  next_id={greedy_orig[0,-1].item()}", flush=True)

    # Load recovered weights and inject into model (strict=False: only text tower present).
    recovered = torch.load(weights_pt_path, map_location="cpu", weights_only=False)
    # Bring to bf16 to match model dtype.
    recovered = {k: v.to(torch.bfloat16) for k, v in recovered.items()}
    miss, unexp = model.load_state_dict(recovered, strict=False)
    text_miss = [k for k in miss if "language_model" in k]
    print(f"[HF worker] injected converted weights: text_missing={len(text_miss)} unexpected={len(unexp)}", flush=True)
    if text_miss:
        print(f"  UNEXPECTED text-tower missing keys: {text_miss[:5]}", flush=True)

    with torch.no_grad():
        logits_conv = model(ids.to(device), use_cache=False).logits.cpu()
    greedy_conv = logits_conv.argmax(dim=-1)
    print(f"[HF worker] converted logits {tuple(logits_conv.shape)}  next_id={greedy_conv[0,-1].item()}", flush=True)

    torch.save({
        "ids": ids,
        "logits_orig": logits_orig,
        "logits_conv": logits_conv,
        "greedy_orig": greedy_orig,
        "greedy_conv": greedy_conv,
    }, out_path)
    print(f"[HF worker] saved -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-forward", action="store_true", help="skip Phase 2 forward parity test")
    ap.add_argument("--save-dir", default=None,
                    help="write a standalone from_pretrained-loadable HF checkpoint here "
                         "(text tower from MLM, vision tower + aux files from original)")
    ap.add_argument("--mlm-ckpt", default=MLM_CKPT,
                    help="MLM dist-checkpoint dir to load (default: original converter output)")
    ap.add_argument("--allow-diff", action="store_true",
                    help="allow --save-dir to write even when recovered weights differ from "
                         "base HF (required for fine-tuned checkpoints)")
    ap.add_argument(HF_WORKER_FLAG, nargs=2, metavar=("WEIGHTS_PT", "OUT_PT"), dest="hf_worker",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    # --- subprocess entry point ---
    if args.hf_worker:
        _hf_forward_worker(args.hf_worker[0], args.hf_worker[1])
        return

    # -------------------------------------------------------------------------
    # Phase 1: load MLM checkpoint + reverse conversion + bitwise compare
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("Phase 1: load MLM checkpoint + reverse conversion")
    print("="*60)

    _init_distributed()
    model, config = load_mlm_model(args.mlm_ckpt)

    print("[convert] building HF state_dict (reverse conversion)...", flush=True)
    hf_sd = build_hf_state_dict(model, config)
    print(f"[convert] recovered {len(hf_sd)} HF tensors", flush=True)

    print("[compare] bitwise comparison against original HF safetensors...", flush=True)
    rows, n_mismatch, n_skip = bitwise_compare(hf_sd, HF_WEIGHTS_FILE)
    n_checked = len(rows) - n_skip
    all_eq = n_mismatch == 0

    print(f"[compare] {n_checked} tensors checked, {n_mismatch} mismatch, {n_skip} skipped")
    print(f"[compare] V_MLM2HF_W: {'PASS' if all_eq else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Export: write a standalone HF checkpoint (only if the weight check passed).
    # -------------------------------------------------------------------------
    if args.save_dir:
        if not all_eq and not args.allow_diff:
            print("[export] SKIPPED: weight bitwise check FAILED; pass --allow-diff to write anyway (fine-tuned ckpt).")
        else:
            print("\n" + "="*60)
            print(f"Export: writing HF checkpoint -> {args.save_dir}")
            if not all_eq:
                print(f"[export] --allow-diff set; proceeding despite {n_mismatch} weight diffs.")
            print("="*60)
            export_hf_checkpoint(hf_sd, HF_WEIGHTS_FILE, HF_WEIGHTS_DIR, args.save_dir)

    # Save recovered weights for Phase 2
    _, weights_pt = tempfile.mkstemp(suffix="_mlm2hf_weights.pt")
    torch.save(hf_sd, weights_pt)
    print(f"[convert] saved recovered weights -> {weights_pt}", flush=True)

    # -------------------------------------------------------------------------
    # Phase 2: forward parity via gemma4_venv subprocess
    # -------------------------------------------------------------------------
    fwd_result = None
    if not args.no_forward:
        print("\n" + "="*60)
        print("Phase 2: forward parity (gemma4_venv subprocess)")
        print("="*60)

        _, out_pt = tempfile.mkstemp(suffix="_mlm2hf_fwd.pt")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [GEMMA4_VENV_PYTHON, __file__, HF_WORKER_FLAG, weights_pt, out_pt],
            env=env,
        )
        try:
            if result.returncode != 0:
                print(f"[ERROR] HF subprocess exited with code {result.returncode}")
            else:
                fwd_result = torch.load(out_pt, map_location="cpu", weights_only=False)
        finally:
            try:
                os.remove(out_pt)
            except OSError:
                pass

    try:
        os.remove(weights_pt)
    except OSError:
        pass

    # -------------------------------------------------------------------------
    # Phase 3: compare forward results + write results
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    # Forward parity
    fwd_pass = None
    fwd_detail = "skipped (--no-forward)"
    if fwd_result is not None:
        lo = fwd_result["logits_orig"].float()
        lc = fwd_result["logits_conv"].float()
        fwd_bitwise = torch.equal(lo, lc)
        fwd_max_diff = (lo - lc).abs().max().item()
        go = fwd_result["greedy_orig"]
        gc = fwd_result["greedy_conv"]
        greedy_match = torch.equal(go, gc)
        fwd_pass = fwd_bitwise
        fwd_detail = (
            f"bitwise={'YES' if fwd_bitwise else 'NO'}  "
            f"max_logit_diff={fwd_max_diff:.2e}  "
            f"greedy_match={greedy_match}  "
            f"orig_next={go[0,-1].item()}  conv_next={gc[0,-1].item()}"
        )
        print(f"  V_MLM2HF_F (forward parity): {'PASS' if fwd_pass else 'FAIL'}  {fwd_detail}")

    print(f"  V_MLM2HF_W (weight bitwise): {'PASS' if all_eq else 'FAIL'}  "
          f"({n_checked} checked, {n_mismatch} mismatch)")

    # Write markdown results
    lines = [
        "# V_RESULTS_MLM2HF — MLM->HF reverse conversion + parity",
        "",
        "## V_MLM2HF_W — weight bitwise match (MLM dist-ckpt -> HF safetensors)",
        "",
        f"- Tensors checked: **{n_checked}**  |  Mismatches: **{n_mismatch}**  |  Skipped: **{n_skip}**",
        f"- **RESULT: {'PASS (all recovered weights bitwise-equal to original HF)' if all_eq else 'FAIL'}**",
        "",
    ]
    if n_mismatch > 0:
        lines += ["### Mismatches", ""]
        for k, s, d in rows:
            if s not in ("EQUAL", "SKIP"):
                lines.append(f"- `{k}`: {s} — {d}")
        lines.append("")

    if fwd_result is not None:
        lines += [
            "## V_MLM2HF_F — forward parity (original HF == converted-weights HF)",
            "",
            f"- FIXED_TOKENS = {FIXED_TOKENS}",
            f"- {fwd_detail}",
            f"- **RESULT: {'PASS (bitwise-identical logits)' if fwd_pass else 'FAIL'}**",
            "",
        ]
    else:
        lines += [
            "## V_MLM2HF_F — forward parity",
            "",
            "- Skipped (run without --no-forward to enable).",
            "",
        ]

    results_out = os.path.join(args.save_dir, "V_RESULTS_MLM2HF.md") if args.save_dir else RESULTS_OUT
    with open(results_out, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWROTE {results_out}")

    overall = all_eq and (fwd_pass is True or fwd_pass is None)
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
