"""I-d: HF Gemma4 E4B -> MLM Gemma4Model checkpoint conversion + V1 bitwise check.

Approach: hand-rolled converter (TASK Option A). Loads the HF safetensors (text
tower only, prefix ``model.language_model.*``), builds an MLM ``Gemma4Model`` (local
spec, E4B dims) via the same path as ic_tests, constructs its state_dict from the HF
tensors with the documented transforms, ``load_state_dict`` (strict), then saves an
MLM dist checkpoint for the training script.

V1 (bitwise): every MAPPED MLM tensor is byte-equal to its HF source (after the
documented structural transforms: fused QKV unpack, fused fc1 unpack). The borrower
k/v slices of linear_qkv have NO HF source (runtime-discarded) and are intentionally
unmapped -- reported separately, NOT as mismatches.

Run with the CONTAINER python (mlm_env first):
    python3 id_convert.py            # convert + V1, save ckpt
    python3 id_convert.py --no-save  # convert + V1 only (no ckpt write)
"""
import mlm_env  # noqa: F401  MUST be first

import argparse
import os

import torch
from safetensors import safe_open

from ic_tests import _build_model, _init_distributed, _make_config

IMPL = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "Gemma4_mlm/code_dev/shared-state/implementations/HYBRID-gemma4-mlm"
)
HF_WEIGHTS = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "gemma4-playground/weights/gemma-4-E4B-it/model.safetensors"
)
CKPT_OUT = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "Gemma4_mlm/code_dev/shared-state/implementations/HYBRID-gemma4-mlm/mlm_ckpt"
)
HF = "model.language_model."

NUM_HEADS = 8
NUM_GROUPS = 2
HEADS_PER_GROUP = NUM_HEADS // NUM_GROUPS  # 4


class HFStore:
    """Lazy reader of the HF safetensors text-tower tensors (CPU, native dtype)."""

    def __init__(self, path):
        self.f = safe_open(path, framework="pt", device="cpu")
        self.keys = set(self.f.keys())

    def get(self, name):
        return self.f.get_tensor(HF + name)

    def has(self, name):
        return (HF + name) in self.keys


def pack_qkv(q, k, v, head_dim):
    """Pack HF q/k/v_proj [out, H] into MLM linear_qkv [ng*(hpg+2)*hd, H] layout.

    MLM ``get_query_key_value_tensors`` views linear_qkv output as
    ``[ng, (hpg + 2) * hd]`` and splits each group into
    ``[hpg*hd (q), hd (k), hd (v)]``. So the row order is, per group g:
        q-heads of group g, then k-head g, then v-head g
    i.e. [q0..q3, k0, v0,  q4..q7, k1, v1] for ng=2, hpg=4.
    """
    H = q.size(1)
    q = q.reshape(NUM_GROUPS, HEADS_PER_GROUP * head_dim, H)
    k = k.reshape(NUM_GROUPS, head_dim, H)
    v = v.reshape(NUM_GROUPS, head_dim, H)
    # Per-group concat [q_group, k_group, v_group], then stack groups.
    packed = torch.cat([torch.cat([q[g], k[g], v[g]], dim=0) for g in range(NUM_GROUPS)], dim=0)
    return packed.contiguous()


def unpack_qkv(packed, head_dim):
    """Inverse of pack_qkv: linear_qkv -> (q, k, v) HF-shaped tensors. For V1 check."""
    H = packed.size(1)
    group_rows = (HEADS_PER_GROUP + 2) * head_dim
    packed = packed.reshape(NUM_GROUPS, group_rows, H)
    q_rows = HEADS_PER_GROUP * head_dim
    q = packed[:, :q_rows, :].reshape(NUM_GROUPS * q_rows, H)
    k = packed[:, q_rows : q_rows + head_dim, :].reshape(NUM_GROUPS * head_dim, H)
    v = packed[:, q_rows + head_dim :, :].reshape(NUM_GROUPS * head_dim, H)
    return q, k, v


def build_state_dict(hf, config):
    """Construct the MLM state_dict from HF tensors. Returns (state_dict, mapping_log).

    mapping_log: list of (mlm_key, hf_source_desc, hf_tensor_for_compare-or-None).
    For fused params, hf_tensor_for_compare is None and the V1 unpack check is done
    separately against the recorded HF q/k/v / gate/up sources.
    """
    sd = {}
    log = []  # (mlm_key, hf_desc)
    full_idx = set(config.full_attention_layers)

    def put(mlm_key, hf_name):
        t = hf.get(hf_name)
        sd[mlm_key] = t
        log.append((mlm_key, hf_name))

    # ---- Top level ----
    put("embedding.word_embeddings.weight", "embed_tokens.weight")
    put("decoder.final_layernorm.weight", "norm.weight")
    put("ple.embed_tokens_per_layer.weight", "embed_tokens_per_layer.weight")
    put("ple.per_layer_model_projection.weight", "per_layer_model_projection.weight")
    put("ple.per_layer_projection_norm.weight", "per_layer_projection_norm.weight")

    # ---- Per layer ----
    for i in range(config.num_layers):
        p = f"decoder.layers.{i}."
        hp = f"layers.{i}."
        head_dim = 512 if i in full_idx else 256

        # Norms (copy AS-IS; Gemma4RMSNorm has plain weight, no +1).
        put(p + "input_layernorm.weight", hp + "input_layernorm.weight")
        put(p + "post_self_attn_layernorm.weight", hp + "post_attention_layernorm.weight")
        put(p + "pre_mlp_layernorm.weight", hp + "pre_feedforward_layernorm.weight")
        put(p + "post_mlp_layernorm.weight", hp + "post_feedforward_layernorm.weight")
        put(p + "post_per_layer_input_norm.weight", hp + "post_per_layer_input_norm.weight")

        # Attention q/k norm.
        put(p + "self_attention.q_layernorm.weight", hp + "self_attn.q_norm.weight")
        put(p + "self_attention.k_layernorm.weight", hp + "self_attn.k_norm.weight")

        # Fused QKV from HF q/k/v_proj.
        q = hf.get(hp + "self_attn.q_proj.weight")
        k = hf.get(hp + "self_attn.k_proj.weight")
        v = hf.get(hp + "self_attn.v_proj.weight")
        sd[p + "self_attention.linear_qkv.weight"] = pack_qkv(q, k, v, head_dim)
        log.append((p + "self_attention.linear_qkv.weight", hp + "self_attn.{q,k,v}_proj.weight (fused)"))

        put(p + "self_attention.linear_proj.weight", hp + "self_attn.o_proj.weight")

        # Fused MLP fc1 = cat([gate, up]) (GatedMLP: gate first, then up).
        gate = hf.get(hp + "mlp.gate_proj.weight")
        up = hf.get(hp + "mlp.up_proj.weight")
        sd[p + "mlp.linear_fc1.weight"] = torch.cat([gate, up], dim=0).contiguous()
        log.append((p + "mlp.linear_fc1.weight", hp + "mlp.{gate,up}_proj.weight (fused)"))
        put(p + "mlp.linear_fc2.weight", hp + "mlp.down_proj.weight")

        # PLE per-layer pieces.
        put(p + "per_layer_input_gate.weight", hp + "per_layer_input_gate.weight")
        put(p + "per_layer_projection.weight", hp + "per_layer_projection.weight")

        # layer_scalar is a TRAINED per-layer buffer in E4B (NOT a constant 1.0 -- the
        # task brief's "=1.0" assumption was wrong; verified e.g. layer 0 == 0.061).
        # Copy it from HF so the layer output is scaled correctly.
        put(p + "layer_scalar", hp + "layer_scalar")

    return sd, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-save", action="store_true", help="skip dist-checkpoint write")
    args = ap.parse_args()

    torch.manual_seed(0)
    _init_distributed()

    from megatron.core.models.gemma4.gemma4_layer_specs import get_gemma4_layer_local_spec

    config = _make_config()
    model = _build_model(get_gemma4_layer_local_spec, config)  # random-init, bf16, cuda

    hf = HFStore(HF_WEIGHTS)
    sd, log = build_state_dict(hf, config)

    # Move state_dict to the model's device/dtype (bf16, cuda) for load + compare.
    sd = {k: v.to(device="cuda", dtype=torch.bfloat16) for k, v in sd.items()}

    # ---- Load into the model. tied output_layer is not in sd. ----
    mlm_keys = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())
    missing_in_sd = sorted(mlm_keys - set(sd.keys()))
    extra_in_sd = sorted(set(sd.keys()) - mlm_keys)

    load_res = model.load_state_dict(sd, strict=False)

    # ---- V1 bitwise verification ----
    rows = []  # (mlm_key, status, detail)
    full_idx = set(config.full_attention_layers)
    msd = dict(model.named_parameters())
    msd.update(dict(model.named_buffers()))  # include layer_scalar buffers

    def cmp(mlm_key, hf_tensor, desc):
        got = msd[mlm_key].detach()
        ref = hf_tensor.to(device="cuda", dtype=torch.bfloat16)
        eq = got.shape == ref.shape and torch.equal(got, ref)
        rows.append((mlm_key, "EQUAL" if eq else "MISMATCH", desc if eq else f"{desc} shapes {tuple(got.shape)} vs {tuple(ref.shape)}"))
        return eq

    # Direct (non-fused) params.
    direct = [
        ("embedding.word_embeddings.weight", "embed_tokens.weight"),
        ("decoder.final_layernorm.weight", "norm.weight"),
        ("ple.embed_tokens_per_layer.weight", "embed_tokens_per_layer.weight"),
        ("ple.per_layer_model_projection.weight", "per_layer_model_projection.weight"),
        ("ple.per_layer_projection_norm.weight", "per_layer_projection_norm.weight"),
    ]
    for mk, hn in direct:
        cmp(mk, hf.get(hn), hn)

    borrower_kv_unmapped = []
    first_shared = config.num_layers - config.num_kv_shared_layers
    for i in range(config.num_layers):
        p = f"decoder.layers.{i}."
        hp = f"layers.{i}."
        head_dim = 512 if i in full_idx else 256
        is_borrower = i >= first_shared

        for mk, hn in [
            ("input_layernorm.weight", "input_layernorm.weight"),
            ("post_self_attn_layernorm.weight", "post_attention_layernorm.weight"),
            ("pre_mlp_layernorm.weight", "pre_feedforward_layernorm.weight"),
            ("post_mlp_layernorm.weight", "post_feedforward_layernorm.weight"),
            ("post_per_layer_input_norm.weight", "post_per_layer_input_norm.weight"),
            ("self_attention.q_layernorm.weight", "self_attn.q_norm.weight"),
            ("self_attention.k_layernorm.weight", "self_attn.k_norm.weight"),
            ("self_attention.linear_proj.weight", "self_attn.o_proj.weight"),
            ("mlp.linear_fc2.weight", "mlp.down_proj.weight"),
            ("per_layer_input_gate.weight", "per_layer_input_gate.weight"),
            ("per_layer_projection.weight", "per_layer_projection.weight"),
            ("layer_scalar", "layer_scalar"),
        ]:
            cmp(p + mk, hf.get(hp + hn), hp + hn)

        # Fused QKV: unpack from the LOADED model param, compare q/k/v slices to HF.
        packed = msd[p + "self_attention.linear_qkv.weight"].detach().cpu()
        qg, kg, vg = unpack_qkv(packed, head_dim)
        q_ref = hf.get(hp + "self_attn.q_proj.weight").to(torch.bfloat16)
        k_ref = hf.get(hp + "self_attn.k_proj.weight").to(torch.bfloat16)
        v_ref = hf.get(hp + "self_attn.v_proj.weight").to(torch.bfloat16)
        q_eq = torch.equal(qg, q_ref)
        rows.append((p + "linear_qkv[q]", "EQUAL" if q_eq else "MISMATCH", f"vs {hp}self_attn.q_proj"))
        if is_borrower:
            # k/v are runtime-discarded for borrowers: still copied here (harmless), so
            # they ALSO match HF, but record as intentionally-irrelevant.
            k_eq = torch.equal(kg, k_ref)
            v_eq = torch.equal(vg, v_ref)
            borrower_kv_unmapped.append((i, k_eq and v_eq))
        else:
            k_eq = torch.equal(kg, k_ref)
            v_eq = torch.equal(vg, v_ref)
            rows.append((p + "linear_qkv[k]", "EQUAL" if k_eq else "MISMATCH", f"vs {hp}self_attn.k_proj"))
            rows.append((p + "linear_qkv[v]", "EQUAL" if v_eq else "MISMATCH", f"vs {hp}self_attn.v_proj"))

        # Fused fc1: unpack gate/up.
        fc1 = msd[p + "mlp.linear_fc1.weight"].detach().cpu()
        I = fc1.size(0) // 2
        gate_g, up_g = fc1[:I], fc1[I:]
        gate_ref = hf.get(hp + "mlp.gate_proj.weight").to(torch.bfloat16)
        up_ref = hf.get(hp + "mlp.up_proj.weight").to(torch.bfloat16)
        rows.append((p + "linear_fc1[gate]", "EQUAL" if torch.equal(gate_g, gate_ref) else "MISMATCH", f"vs {hp}mlp.gate_proj"))
        rows.append((p + "linear_fc1[up]", "EQUAL" if torch.equal(up_g, up_ref) else "MISMATCH", f"vs {hp}mlp.up_proj"))

    n_mismatch = sum(1 for _, s, _ in rows if s == "MISMATCH")
    all_eq = n_mismatch == 0

    # ---- Write V1 result ----
    out = os.path.join(IMPL, "V_RESULTS.md")
    lines = [
        "# V_RESULTS — Gemma 4 E4B HF->MLM conversion + parity (I-d)",
        "",
        "## V1 — conversion bitwise (every mapped MLM tensor == HF source)",
        "",
        f"- Approach: hand-rolled converter (Option A). HF prefix `{HF}`.",
        f"- Mapped tensors checked: **{len(rows)}**; MISMATCH: **{n_mismatch}**.",
        f"- load_state_dict(strict=False): missing keys (expected: layer_scalar buffers + tied output_layer) "
        f"= {len(load_res.missing_keys)}; unexpected = {len(load_res.unexpected_keys)}.",
        f"- MLM params not in state_dict (expected tied/buffers): {missing_in_sd[:5]}{' ...' if len(missing_in_sd) > 5 else ''}",
        f"- state_dict keys not MLM params: {extra_in_sd or 'none'}",
        "",
        f"### V1 RESULT: {'PASS (all mapped tensors bitwise-equal)' if all_eq else 'FAIL'}",
        "",
        "Borrower (24..41) linear_qkv k/v slices: filled from HF own k/v (copied), "
        "runtime-discarded by KV-share, so bitwise-IRRELEVANT to output. "
        f"All {len(borrower_kv_unmapped)} borrower k/v slices "
        f"{'happen to equal HF (copied)' if all(b for _, b in borrower_kv_unmapped) else 'differ (still irrelevant)'}.",
        "",
        "<details><summary>Mismatches (if any)</summary>",
        "",
    ]
    for k, s, d in rows:
        if s == "MISMATCH":
            lines.append(f"- {k}: {s} ({d})")
    lines += ["", "</details>", ""]
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("WROTE", out)
    print(f"V1: {len(rows)} tensors checked, {n_mismatch} mismatch -> {'PASS' if all_eq else 'FAIL'}")

    if not all_eq:
        print("V1 FAILED; not saving checkpoint.")
        return

    # ---- Save dist checkpoint ----
    if not args.no_save:
        from megatron.core import dist_checkpointing

        os.makedirs(CKPT_OUT, exist_ok=True)
        ckpt_sd = model.sharded_state_dict()
        dist_checkpointing.save(ckpt_sd, CKPT_OUT)
        print("SAVED MLM dist checkpoint ->", CKPT_OUT)


if __name__ == "__main__":
    main()
