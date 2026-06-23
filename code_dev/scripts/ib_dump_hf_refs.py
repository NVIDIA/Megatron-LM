"""Dump HF Gemma4 per-op references for bitwise I-b unit tests.

Run with the gemma4_venv python (transformers 5.11):
    env -u PYTHONPATH /lustre/fsw/.../gemma4_venv/bin/python ib_dump_hf_refs.py

Instantiates the HF module classes (Gemma4RMSNorm, Gemma4TextRotaryEmbedding, the
PLE pieces) with KNOWN random weights/inputs, runs them, and saves input+output
tensors to .pt under refs/. The MLM unit test (ib_unit_tests.py) loads these,
copies the SAME weights into the MLM modules, and asserts byte-equality.

This script is reusable for I-c: --full also dumps per-layer hidden states /
logits / greedy for a fixed token list (stubbed; wire in I-c).
"""
import argparse
import os
import sys

import torch

REFS = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/Gemma4_mlm/code_dev/shared-state/implementations/HYBRID-gemma4-mlm/refs"
GEMMA4_SRC = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/gemma4-playground/gemma4_src"

# E4B verified dims.
H = 2560
L = 42
PLE_DIM = 256
VOCAB_PLE = 262144
EPS = 1e-6
SLIDING_HD = 256
FULL_HD = 512


def _load_hf():
    # modeling_gemma4.py uses package-relative imports (from ... import initialization),
    # so it must be imported via the wired transformers package, not as a bare file.
    from transformers.models.gemma4 import modeling_gemma4 as m

    return m


def dump_rmsnorm(m, dtype):
    """Gemma4RMSNorm with and without scale, over head_dim=256."""
    torch.manual_seed(0)
    dim = SLIDING_HD
    x = torch.randn(2, 4, 8, dim, dtype=dtype)

    norm = m.Gemma4RMSNorm(dim, eps=EPS, with_scale=True)
    norm.weight.data = torch.randn(dim)  # non-trivial weight (plain, no +1)
    out_scale = norm(x)

    norm_ns = m.Gemma4RMSNorm(dim, eps=EPS, with_scale=False)
    out_noscale = norm_ns(x)

    torch.save(
        {
            "x": x,
            "weight": norm.weight.data.clone(),
            "eps": EPS,
            "out_scale": out_scale,
            "out_noscale": out_noscale,
        },
        os.path.join(REFS, f"rmsnorm_{dtype}.pt".replace("torch.", "")),
    )
    print(f"[rmsnorm {dtype}] scale out {tuple(out_scale.shape)}, noscale out {tuple(out_noscale.shape)}")


def dump_rope(m, dtype):
    """Both layer types' cos/sin vs HF Gemma4TextRotaryEmbedding.forward."""
    torch.manual_seed(0)
    seq = 16
    position_ids = torch.arange(seq).unsqueeze(0)  # [1, S]

    # Build a minimal config carrying the rope params for both layer types.
    from transformers import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        hidden_size=H,
        num_hidden_layers=L,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=SLIDING_HD,
        global_head_dim=FULL_HD,
        vocab_size=VOCAB_PLE,
    )
    rotary = m.Gemma4TextRotaryEmbedding(cfg)

    x = torch.zeros(1, seq, 8, FULL_HD, dtype=dtype)  # dtype carrier only
    cos_s, sin_s = rotary(x, position_ids, layer_type="sliding_attention")
    cos_f, sin_f = rotary(x, position_ids, layer_type="full_attention")

    torch.save(
        {
            "position_ids": position_ids,
            "dtype": str(dtype),
            "cos_sliding": cos_s,
            "sin_sliding": sin_s,
            "cos_full": cos_f,
            "sin_full": sin_f,
            "sliding_inv_freq": rotary.sliding_attention_inv_freq.clone(),
            "full_inv_freq": rotary.full_attention_inv_freq.clone(),
        },
        os.path.join(REFS, f"rope_{dtype}.pt".replace("torch.", "")),
    )
    print(
        f"[rope {dtype}] sliding cos {tuple(cos_s.shape)} full cos {tuple(cos_f.shape)} "
        f"sliding_nonzero_invfreq={int((rotary.sliding_attention_inv_freq != 0).sum())} "
        f"full_nonzero_invfreq={int((rotary.full_attention_inv_freq != 0).sum())}"
    )


def dump_ple(m, dtype):
    """PLE per_layer_inputs [B,S,L,P] vs HF get/project_per_layer_inputs.

    PLE math is dimension-agnostic, so we use a SMALL config (few layers, tiny
    vocab) to build the HF model fast on CPU. The MLM test derives dims from the
    saved weight shapes, so this stays a faithful bitwise check of the PLE logic.
    Keep H and P at E4B values (2560/256) for representativeness.
    """
    torch.manual_seed(0)
    from transformers import Gemma4TextConfig

    small_L = 4
    small_vocab = 2000
    cfg = Gemma4TextConfig(
        hidden_size=H,
        num_hidden_layers=small_L,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=SLIDING_HD,
        global_head_dim=FULL_HD,
        vocab_size=small_vocab,
        hidden_size_per_layer_input=PLE_DIM,
        vocab_size_per_layer_input=small_vocab,
    )
    model = m.Gemma4TextModel(cfg).to(dtype).eval()

    b, s = 2, 6
    input_ids = torch.randint(0, 1000, (b, s))
    inputs_embeds = torch.randn(b, s, H, dtype=dtype)

    with torch.no_grad():
        per_layer_token = model.get_per_layer_inputs(input_ids, inputs_embeds=None)
        per_layer_inputs = model.project_per_layer_inputs(inputs_embeds, per_layer_token)

    torch.save(
        {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "dtype": str(dtype),
            "embed_tokens_per_layer_weight": model.embed_tokens_per_layer.weight.data.clone(),
            "per_layer_model_projection_weight": model.per_layer_model_projection.weight.data.clone(),
            "per_layer_projection_norm_weight": model.per_layer_projection_norm.weight.data.clone(),
            "per_layer_inputs": per_layer_inputs,
        },
        os.path.join(REFS, f"ple_{dtype}.pt".replace("torch.", "")),
    )
    print(f"[ple {dtype}] per_layer_inputs {tuple(per_layer_inputs.shape)}")


# Full E4B (google) weights for the --full per-layer / logits / greedy dump.
E4B_WEIGHTS = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "gemma4-playground/weights/gemma-4-E4B-it"
)
# Fixed int-token input shared by the MLM eval harness (V2/V3/V5). BOS=2, EOS=1.
FIXED_TOKENS = [2, 651, 1234, 99, 17, 8, 200, 9000, 42, 3]


def dump_full_e4b():
    """Dump FULL E4B per-layer hidden states, final logits, and greedy tokens.

    Loads the real google E4B model and runs a TEXT-ONLY forward with
    ``output_hidden_states=True`` on FIXED_TOKENS. Used (after I-d conversion lands)
    by the MLM eval harness for V2 (per-layer hidden states), V3 (final logits), and
    V5 (greedy tokens). Saved to refs/full_e4b.pt.

    NOTE: the E4B checkpoint is a VLM (weights under ``model.language_model.*``), so it
    MUST be loaded as ``Gemma4ForConditionalGeneration``. Loading it as
    ``Gemma4ForCausalLM`` silently drops every language-model weight (prefix mismatch:
    expects ``model.*``) and runs RANDOM weights -- which produced a garbage reference
    in the first I-d parity run. Text-only ``input_ids`` exercises exactly the language
    model + PLE + lm_head + final softcap, the same path the MLM model implements.
    """
    from transformers import Gemma4ForConditionalGeneration

    # TEST A: full fp32 forward (OPDIFF_DTYPE=float32) -> full_e4b_fp32.pt.
    # TEST B: full fp64 CPU forward (OPDIFF_DTYPE=float64) -> full_e4b_fp64.pt.
    full_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }.get(os.environ.get("OPDIFF_DTYPE", "bfloat16"), torch.bfloat16)
    full_out = os.environ.get("OPDIFF_FULL_OUT", "full_e4b.pt")
    print(f"[full_e4b] dtype={full_dtype} out={full_out}")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        E4B_WEIGHTS, torch_dtype=full_dtype, attn_implementation="eager"
    ).eval()
    ids = torch.tensor([FIXED_TOKENS])
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
    # hidden_states: tuple(L+1) of [B, S, H] (embedding output + each layer output).
    hidden_states = [h.detach().clone() for h in out.hidden_states]
    logits = out.logits.detach().clone()
    greedy = logits.argmax(dim=-1).detach().clone()

    # Sanity: real (non-random) weights -> a trained LM does NOT predict ~uniform logits
    # and the embed_tokens weight is not tiny-normal-init. Fail loud if weights are random.
    ew = model.model.language_model.embed_tokens.weight
    assert ew.float().abs().mean().item() > 1e-3, (
        f"embed_tokens looks random-init (abs mean {ew.float().abs().mean().item():.2e}); "
        "weights were NOT loaded -- wrong model class / prefix."
    )

    torch.save(
        {
            "input_ids": ids,
            "hidden_states": hidden_states,
            "logits": logits,
            "greedy": greedy,
            "final_logit_softcapping": model.config.get_text_config().final_logit_softcapping,
        },
        os.path.join(REFS, full_out),
    )
    print(
        f"[full_e4b] {len(hidden_states)} hidden states {tuple(hidden_states[0].shape)}, "
        f"logits {tuple(logits.shape)}, greedy {greedy.tolist()}"
    )


def dump_decoder_layer(m, dtype):
    """Dump a single SLIDING ``Gemma4TextDecoderLayer`` (own KV) with random weights.

    Captures all layer weights, the input hidden state, per-layer input, position
    embeddings (cos/sin), and the output, for the MLM layer-vs-HF bitwise check
    (ic_tests.py::test_layer_vs_hf). Uses E4B dims (H=2560, head_dim=256, 8q/2kv) but
    a SMALL num_hidden_layers so the layer is an own-KV (non-shared) sliding layer.
    """
    torch.manual_seed(0)
    from transformers import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        hidden_size=H,
        intermediate_size=10240,
        num_hidden_layers=2,  # small -> layer 0 is own-KV sliding
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=SLIDING_HD,
        global_head_dim=FULL_HD,
        vocab_size=VOCAB_PLE,
        hidden_size_per_layer_input=PLE_DIM,
        vocab_size_per_layer_input=VOCAB_PLE,
        num_kv_shared_layers=0,
        rms_norm_eps=EPS,
        attention_bias=False,
        hidden_activation="gelu_pytorch_tanh",
        final_logit_softcapping=30.0,
    )
    layer = m.Gemma4TextDecoderLayer(cfg, layer_idx=0).to(dtype).eval()
    rotary = m.Gemma4TextRotaryEmbedding(cfg).to(dtype)

    b, s = 1, 8
    hidden = torch.randn(b, s, H, dtype=dtype)
    per_layer_input = torch.randn(b, s, PLE_DIM, dtype=dtype)
    position_ids = torch.arange(s).unsqueeze(0)
    pos_emb = rotary(hidden, position_ids, layer_type="sliding_attention")

    # Additive sliding causal mask, finfo.min fill (HF eager mask).
    min_val = torch.finfo(dtype).min
    q = torch.arange(s)
    causal = q[:, None] >= q[None, :]
    in_win = q[None, :] > (q[:, None] - cfg.sliding_window)
    mask = torch.where(causal & in_win, 0.0, min_val).to(dtype).view(1, 1, s, s)

    with torch.no_grad():
        out = layer(
            hidden,
            per_layer_input=per_layer_input,
            shared_kv_states={},
            position_embeddings=pos_emb,
            attention_mask=mask,
            position_ids=position_ids,
        )

    sd = {k: v.detach().clone() for k, v in layer.state_dict().items()}
    torch.save(
        {
            "dtype": str(dtype),
            "hidden": hidden,
            "per_layer_input": per_layer_input,
            "cos": pos_emb[0],
            "sin": pos_emb[1],
            "mask": mask,
            "output": out.detach().clone(),
            "state_dict": sd,
        },
        os.path.join(REFS, f"decoder_layer_{dtype}.pt".replace("torch.", "")),
    )
    print(f"[decoder_layer {dtype}] output {tuple(out.shape)}; {len(sd)} weights")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also dump full E4B per-layer states/logits/greedy")
    args = ap.parse_args()

    os.makedirs(REFS, exist_ok=True)
    m = _load_hf()
    torch.set_grad_enabled(False)

    for dtype in (torch.float32, torch.bfloat16):
        dump_rmsnorm(m, dtype)
        dump_rope(m, dtype)
        dump_ple(m, dtype)
        dump_decoder_layer(m, dtype)

    if args.full:
        dump_full_e4b()

    print("DUMP DONE ->", REFS)


if __name__ == "__main__":
    main()
