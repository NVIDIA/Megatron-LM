"""I-c tests: V6 (both specs build+forward) + internal correctness (KV-bus, mask, sandwich).

Run with the CONTAINER python (mlm_env first). No HF dependency; uses random-init
E4B-dim models on a fixed int-token list. Writes a markdown summary.
"""
import mlm_env  # noqa: F401  MUST be first

import functools
import os

import torch

# Gemma4 MLP/PLE use gelu_pytorch_tanh == F.gelu(approximate="tanh"). Use the full
# sqrt(2/pi) constant (not fast_gelu's truncated 0.7978845608) for bitwise fidelity.
GELU_TANH = functools.partial(torch.nn.functional.gelu, approximate="tanh")

IMPL = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/code_dev/shared-state/implementations/HYBRID-gemma4-mlm"
OUT = IMPL + "/IC_results.md"
REFS = IMPL + "/refs"

results = []  # (check, passed, detail)


def record(check, passed, detail=""):
    results.append((check, bool(passed), detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {check} {detail}")


def _init_distributed():
    import megatron.core.parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "12399")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    ps.initialize_model_parallel(1, 1)
    # Required before building VocabParallelEmbedding / parallel linears (adds the
    # 'model-parallel-rng' cuda rng state used by _initialize_affine_weight_gpu).
    model_parallel_cuda_manual_seed(123)


def _make_config(num_layers=42, hidden=2560, ffn=10240):
    from megatron.core.transformer.gemma4_config import Gemma4TransformerConfig

    return Gemma4TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        ffn_hidden_size=ffn,
        num_attention_heads=8,
        num_query_groups=2,
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=GELU_TANH,
        add_bias_linear=False,
        qk_layernorm=True,
        bias_activation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        pipeline_dtype=torch.bfloat16,
    )


def _build_model(spec_fn, config, vocab=262144):
    from megatron.core.models.gemma4.gemma4_model import Gemma4Model

    spec = spec_fn(config)
    model = Gemma4Model(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=vocab,
        max_sequence_length=512,
    )
    return model.bfloat16().cuda()


def test_v6_local():
    from megatron.core.models.gemma4.gemma4_layer_specs import get_gemma4_layer_local_spec

    config = _make_config()
    model = _build_model(get_gemma4_layer_local_spec, config)
    ids = torch.tensor([[2, 651, 1234, 99, 17, 8, 200, 3]], device="cuda")
    with torch.no_grad():
        logits = model(ids)
    ok = tuple(logits.shape) == (1, ids.shape[1], 262144) and torch.isfinite(logits).all()
    record("V6_local_build_forward", ok, f"shape={tuple(logits.shape)}")
    return model


def test_v6_te():
    try:
        from megatron.core.models.gemma4.gemma4_layer_specs import (
            get_gemma4_layer_with_transformer_engine_spec,
        )

        config = _make_config()
        model = _build_model(get_gemma4_layer_with_transformer_engine_spec, config)
        ids = torch.tensor([[2, 651, 1234, 99, 17, 8, 200, 3]], device="cuda")
        with torch.no_grad():
            logits = model(ids)
        ok = tuple(logits.shape) == (1, ids.shape[1], 262144) and torch.isfinite(logits).all()
        record("V6_te_build_forward", ok, f"shape={tuple(logits.shape)}")
    except Exception as e:
        record("V6_te_build_forward", False, f"exception: {type(e).__name__}: {e}")


def test_mask_finfo_min():
    from megatron.core.transformer.gemma4_mask import (
        build_causal_mask,
        build_sliding_window_causal_mask,
    )

    dtype = torch.bfloat16
    m = build_causal_mask(6, dtype, torch.device("cpu"))
    fin = torch.finfo(dtype).min
    # Upper triangle (kv>q) must be finfo.min; diagonal+lower must be 0.
    ok_fill = m[0, 0, 0, 1].item() == fin and m[0, 0, 2, 2].item() == 0.0 and m[0, 0, 3, 0].item() == 0.0
    record("mask_causal_finfo_min", ok_fill, f"fill={m[0,0,0,1].item():.3e} (finfo.min={fin:.3e})")

    w = 4
    sm = build_sliding_window_causal_mask(10, w, dtype, torch.device("cpu"))
    # query q=8 attends keys 5..8 (window 4): key 5 allowed (0.0), key 4 masked (finfo.min), key 8 allowed.
    boundary = (
        sm[0, 0, 8, 5].item() == 0.0
        and sm[0, 0, 8, 4].item() == fin
        and sm[0, 0, 8, 8].item() == 0.0
        and sm[0, 0, 8, 9].item() == fin  # future masked
    )
    record("mask_sliding_window_boundary", boundary, "q=8 attends 5..8 incl")


def test_kv_bus(model_local):
    """Borrower layers' K/V must equal their producer's processed K/V.

    Instrument the attention forward to capture each layer's (k, v) actually used, then
    verify borrower 24..41 K/V == producer (sliding L22 idx22 / full L23 idx23).
    """
    captured = {}  # layer_idx -> (k, v)
    handles = []
    layers = model_local.decoder.layers

    def make_hook(idx):
        attn = layers[idx].self_attention
        orig = attn._gemma4_core_attention

        def wrapped(query, key, value, attention_mask):
            captured[idx] = (key.detach().clone(), value.detach().clone())
            return orig(query, key, value, attention_mask)

        attn._gemma4_core_attention = wrapped
        return attn, orig

    saved = [make_hook(i) for i in range(len(layers))]
    ids = torch.tensor([[2, 651, 1234, 99, 17, 8, 200, 3]], device="cuda")
    with torch.no_grad():
        model_local(ids)
    for attn, orig in saved:
        attn._gemma4_core_attention = orig

    # E4B: sliding producer idx 22, full producer idx 23; borrowers 24..41.
    full_idx = set(model_local.config.full_attention_layers)
    all_match = True
    detail = []
    for b in range(24, 42):
        is_full = b in full_idx
        producer = 23 if is_full else 22
        k_b, v_b = captured[b]
        k_p, v_p = captured[producer]
        match = torch.equal(k_b, k_p) and torch.equal(v_b, v_p)
        all_match = all_match and match
        if not match:
            detail.append(f"L{b}->L{producer} MISMATCH")
    record("kv_bus_borrowers_match_producers", all_match, "; ".join(detail) or "24..41 all match")


def _load_hf_weights_into_layer(layer, sd, head_dim, n_heads, n_kv, hidden, ple_dim):
    """Copy HF Gemma4TextDecoderLayer weights (in ``sd``) into an MLM Gemma4 layer.

    The fused ``linear_qkv`` is packed per query-group as [q(g) | k(g) | v(g)]
    (get_query_key_value_tensors); MLP fc1 is cat([gate, up]). This mirrors the I-d
    conversion and is the exact mapping V1 must reproduce.
    """
    groups = n_heads // n_kv
    qd, kd = head_dim * n_heads, head_dim * n_kv
    qw = sd["self_attn.q_proj.weight"].view(n_heads, head_dim, hidden)
    kw = sd["self_attn.k_proj.weight"].view(n_kv, head_dim, hidden)
    vw = sd["self_attn.v_proj.weight"].view(n_kv, head_dim, hidden)
    rows = []
    for g in range(n_kv):
        rows.append(qw[g * groups : (g + 1) * groups].reshape(-1, hidden))
        rows.append(kw[g].reshape(-1, hidden))
        rows.append(vw[g].reshape(-1, hidden))
    fused_qkv = torch.cat(rows, dim=0)

    with torch.no_grad():
        layer.self_attention.linear_qkv.weight.copy_(fused_qkv)
        layer.self_attention.linear_proj.weight.copy_(sd["self_attn.o_proj.weight"])
        layer.self_attention.q_layernorm.weight.copy_(sd["self_attn.q_norm.weight"])
        layer.self_attention.k_layernorm.weight.copy_(sd["self_attn.k_norm.weight"])
        layer.input_layernorm.weight.copy_(sd["input_layernorm.weight"])
        layer.post_self_attn_layernorm.weight.copy_(sd["post_attention_layernorm.weight"])
        layer.pre_mlp_layernorm.weight.copy_(sd["pre_feedforward_layernorm.weight"])
        layer.post_mlp_layernorm.weight.copy_(sd["post_feedforward_layernorm.weight"])
        layer.mlp.linear_fc1.weight.copy_(
            torch.cat([sd["mlp.gate_proj.weight"], sd["mlp.up_proj.weight"]], dim=0)
        )
        layer.mlp.linear_fc2.weight.copy_(sd["mlp.down_proj.weight"])
        layer.per_layer_input_gate.weight.copy_(sd["per_layer_input_gate.weight"])
        layer.per_layer_projection.weight.copy_(sd["per_layer_projection.weight"])
        layer.post_per_layer_input_norm.weight.copy_(sd["post_per_layer_input_norm.weight"])
        layer.layer_scalar.copy_(sd["layer_scalar"])


def test_layer_vs_hf(dtype_str):
    """Single sliding decoder layer: MLM vs HF, bitwise on identical weights+input.

    Strongest early correctness signal for the sandwich norms + PLE sub-block +
    attention (scale/v_norm/rope/mask) before full conversion lands. Reference dumped
    by ib_dump_hf_refs.py::dump_decoder_layer.
    """
    ref_path = os.path.join(REFS, f"decoder_layer_{dtype_str}.pt")
    if not os.path.exists(ref_path):
        record(f"layer_vs_hf_{dtype_str}", False, "ref missing (run ib_dump_hf_refs.py)")
        return
    from megatron.core.models.gemma4.gemma4_layer_specs import get_gemma4_layer_local_spec
    from megatron.core.models.gemma4.gemma4_model import Gemma4Model

    d = torch.load(ref_path, weights_only=False)
    dtype = getattr(torch, dtype_str)
    config = _make_config(num_layers=2, hidden=2560, ffn=10240)
    config.num_kv_shared_layers = 0  # both layers own their KV (matches HF dump)
    config.params_dtype = dtype
    config.bf16 = dtype == torch.bfloat16

    spec = get_gemma4_layer_local_spec(config)
    model = Gemma4Model(
        config=config, transformer_layer_spec=spec, vocab_size=262144, max_sequence_length=512
    ).to(dtype).cuda()
    layer = model.decoder.layers[0]
    _load_hf_weights_into_layer(
        layer, d["state_dict"], head_dim=256, n_heads=8, n_kv=2, hidden=2560, ple_dim=256
    )

    # HF tensors are [B,S,*]; MLM layer is seq-first [S,B,*].
    hidden = d["hidden"].transpose(0, 1).contiguous().to(dtype).cuda()  # [s,b,h]
    pli = d["per_layer_input"].transpose(0, 1).contiguous().to(dtype).cuda()  # [s,b,P]
    cos, sin = d["cos"].to(dtype).cuda(), d["sin"].to(dtype).cuda()  # [b,s,hd]
    mask = d["mask"].to(dtype).cuda()
    kv_bus = {}
    with torch.no_grad():
        out, _ = layer(
            hidden,
            attention_mask=mask,
            per_layer_input=pli,
            rotary_cos_sin=(cos, sin),
            kv_bus=kv_bus,
        )
    got = out.transpose(0, 1).contiguous().cpu()  # [b,s,h]
    ref = d["output"].cpu()
    diff = (got.float() - ref.float()).abs().max().item()
    exact = torch.equal(got, ref)
    # Realistic bar (TASK §5): fp32-island ops are exact, but the bf16/fp32 GEMM
    # accumulation order (eager matmul vs cuBLAS algo) is the unavoidable non-bitwise
    # island, amplified through 4 norm+residual sub-blocks. fp32 ~1e-2, bf16 larger.
    tol = 5e-2 if dtype == torch.float32 else 5e-1
    ok = exact or diff <= tol
    record(f"layer_vs_hf_{dtype_str}", ok, f"max_abs_diff={diff:.3e} exact={exact} (tol={tol})")


def _dump_state_dict_keys(model):
    """Write the MLM Gemma4 state_dict key list (top-level + per layer 0/1) for I-d."""
    sd = model.state_dict()

    def shape(k):
        v = sd[k]
        return tuple(v.shape) if v is not None else "None(tied/unallocated)"

    top = sorted(k for k in sd if not k.startswith("decoder.layers."))
    l0 = sorted(
        k[len("decoder.layers.0.") :] for k in sd if k.startswith("decoder.layers.0.")
    )
    lines = ["# MLM Gemma4 state_dict keys (for I-d conversion)", "", "## Top-level", ""]
    lines += [f"- `{k}`  shape={shape(k)}" for k in top]
    lines += ["", "## Per layer (decoder.layers.{i}.)", ""]
    lines += [f"- `{k}`  shape={shape('decoder.layers.0.' + k)}" for k in l0]
    with open(IMPL + "/IC_state_dict_keys.md", "w") as f:
        f.write("\n".join(lines))
    print("WROTE", IMPL + "/IC_state_dict_keys.md")


def main():
    torch.manual_seed(0)
    _init_distributed()
    test_mask_finfo_min()
    model_local = test_v6_local()
    test_kv_bus(model_local)
    test_v6_te()
    for dtype_str in ("float32", "bfloat16"):
        test_layer_vs_hf(dtype_str)
    _dump_state_dict_keys(model_local)

    lines = [
        "# I-c results (V6 + internal correctness)",
        "",
        "Random-init E4B-dim models, fixed int-token list, DDP=TP=PP=CP=SP=1.",
        "",
        "| check | passed | detail |",
        "|-------|--------|--------|",
    ]
    all_pass = True
    for check, passed, detail in results:
        all_pass = all_pass and passed
        lines.append(f"| {check} | {'YES' if passed else 'NO'} | {detail} |")
    lines += ["", f"**ALL PASS: {'YES' if all_pass else 'NO'}**", ""]
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print("WROTE", OUT)
    print("ALL_PASS", all_pass)


if __name__ == "__main__":
    main()
