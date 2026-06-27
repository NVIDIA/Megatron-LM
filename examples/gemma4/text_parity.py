"""Live text parity: HF Gemma4 vs MLM Gemma4 on real text, SINGLE PROCESS.

Tokenizes the text with the E4B chat template, loads BOTH the HF and the MLM
(local-spec) model, runs a forward pass for each, reports the logits, then drops
into a breakpoint() so you can inspect the logits yourself. At the breakpoint the
following are in scope: ids, tok, hf_logits, hf_greedy, mlm_logits, mlm_greedy, diff.

Run with a container python that has BOTH HF transformers (5.x, Gemma4) AND the
MLM / TransformerEngine deps in one env (e.g. nemo.26.06):
    python3 text_parity.py
    python3 text_parity.py --text "Explain gravity in one sentence."

HF is loaded first and freed before MLM is built, so peak GPU memory is ~one model.
"""
import argparse
import os
import sys

HF_WEIGHTS = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "gemma4-playground/weights/gemma-4-E4B-it"
)
MLM_SRC = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "Gemma4_mlm/Megatron-LM"
)
MLM_CKPT = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "Gemma4_mlm/code_dev/shared-state/implementations/HYBRID-gemma4-mlm/mlm_ckpt"
)
DEFAULT_TEXT = "What is the capital of France? Answer in one word."


class _RawDecoder:
    """Minimal decode shim wrapping a tokenizers.Tokenizer (for the report)."""

    def __init__(self, t):
        self._t = t

    def decode(self, ids):
        return self._t.decode(list(ids))

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

def _tokenize(text):
    """Chat-template tokenize with the E4B tokenizer -> (decoder, ids[1, S]).

    For a parity test only the SAME ids reaching both models matters, so if the
    high-level AutoTokenizer trips a transformers/tokenizer-config bug (e.g.
    ``extra_special_tokens`` provided as a list), fall back to the raw
    ``tokenizers`` lib reading tokenizer.json + a manual Gemma chat template.
    """
    import torch

    # Preferred path: the real tokenizer + its chat template.
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(HF_WEIGHTS)
        messages = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"] if isinstance(ids, dict) and "input_ids" in ids else torch.tensor(ids)
        ids = ids.long()
        print(f"[tok] AutoTokenizer: {ids.shape[1]} tokens: {ids[0].tolist()}", flush=True)
        return tok, ids
    except Exception as e:
        print(f"[tok] AutoTokenizer failed ({type(e).__name__}: {e}); "
              "falling back to raw tokenizers + manual Gemma template", flush=True)

    # Fallback: raw fast tokenizer + standard Gemma chat format. Same ids feed
    # both models, so parity is unaffected by template byte-exactness.
    from tokenizers import Tokenizer

    t = Tokenizer.from_file(os.path.join(HF_WEIGHTS, "tokenizer.json"))
    prompt = f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
    seq = t.encode(prompt).ids
    if not seq or seq[0] != 2:  # ensure leading <bos> (id 2)
        seq = [2] + seq
    ids = torch.tensor([seq], dtype=torch.long)
    print(f"[tok] raw: {ids.shape[1]} tokens: {ids[0].tolist()}", flush=True)
    return _RawDecoder(t), ids


def _run_hf(ids):
    """HF Gemma4 forward (eager attn, bf16). Returns (logits[1,S,V] fp32 cpu, greedy)."""
    import torch
    from transformers import Gemma4ForConditionalGeneration

    print("[HF] loading model ...", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        HF_WEIGHTS, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval().cuda()
    with torch.no_grad():
        out = model(ids.cuda(), use_cache=False)
    logits = out.logits.float().cpu()
    greedy = logits.argmax(dim=-1)
    print(f"[HF] logits {tuple(logits.shape)}  next_id={greedy[0, -1].item()}", flush=True)
    # Free the HF model before building MLM so peak GPU memory stays at ~one model.
    del model, out
    torch.cuda.empty_cache()
    return logits, greedy


def _run_mlm(ids):
    """MLM Gemma4 (local spec) forward from the converted dist-checkpoint."""
    if MLM_SRC not in sys.path:
        sys.path.insert(0, MLM_SRC)
    # nvrx version shim: the container's nvidia_resiliency_ext may lack __version__
    # or report a dev pre-release of 0.6.0 (< 0.6.0), which fails MLM's >=0.6.0 guard.
    try:
        import nvidia_resiliency_ext as _nvrx
        from packaging.version import Version as _V
        _cur = getattr(_nvrx, "__version__", None)
        if _cur is None or _V(str(_cur)) < _V("0.6.0"):
            _nvrx.__version__ = "0.6.0"
    except Exception:
        pass

    import torch

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from megatron.core import dist_checkpointing
    from megatron.core.models.gemma4.gemma4_layer_specs import get_gemma4_layer_local_spec

    print("[MLM] init distributed ...", flush=True)
    _init_distributed()
    config = _make_config()
    model = _build_model(get_gemma4_layer_local_spec, config, vocab=262144)
    model.eval()

    print(f"[MLM] loading dist-checkpoint from {MLM_CKPT} ...", flush=True)
    sharded_sd = model.sharded_state_dict()
    loaded = dist_checkpointing.load(sharded_sd, MLM_CKPT)
    model.load_state_dict(loaded, strict=False)

    with torch.no_grad():
        logits = model(ids.cuda()).float().cpu()  # [1, S, V]
    greedy = logits.argmax(dim=-1)
    print(f"[MLM] logits {tuple(logits.shape)}  next_id={greedy[0, -1].item()}", flush=True)
    return logits, greedy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEFAULT_TEXT)
    args = ap.parse_args()

    import torch

    print(f"\n{'=' * 60}")
    print(f"TEXT  : {args.text!r}")
    print(f"{'=' * 60}")

    tok, ids = _tokenize(args.text)

    print("\n--- HF forward ---")
    hf_logits, hf_greedy = _run_hf(ids)

    print("\n--- MLM forward ---")
    mlm_logits, mlm_greedy = _run_mlm(ids)

    diff = (hf_logits - mlm_logits).abs()
    exact = torch.equal(hf_greedy, mlm_greedy)
    match_count = (hf_greedy == mlm_greedy).sum().item()

    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"  logits max-abs-diff : {diff.max().item():.4e}")
    print(f"  logits mean-abs-diff: {diff.mean().item():.4e}")
    print(f"  greedy exact match  : {exact}  ({match_count}/{ids.shape[1]} tokens)")
    print(f"  HF  greedy : {hf_greedy[0].tolist()}")
    print(f"  MLM greedy : {mlm_greedy[0].tolist()}")
    print(f"  HF  next token id={hf_greedy[0, -1].item()}  decoded={tok.decode([hf_greedy[0, -1].item()])!r}")
    print(f"  MLM next token id={mlm_greedy[0, -1].item()}  decoded={tok.decode([mlm_greedy[0, -1].item()])!r}")
    print(f"{'=' * 60}\n")

    # Inspect the logits yourself. In scope: ids, tok, hf_logits, hf_greedy,
    # mlm_logits, mlm_greedy, diff. E.g. diff[0, pos].max(), hf_logits[0, pos].topk(5).
    breakpoint()

    return hf_logits, mlm_logits


if __name__ == "__main__":
    main()
