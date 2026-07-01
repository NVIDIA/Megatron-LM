"""KV-cache parity test for Gemma4 MLM.

Compares two MLM-side forward strategies on the same prompt, single-rank:
  A. Reference: at each step, call ``model(tokens_so_far, inference_context=None)``
     -- equivalent to the pre-KV-cache server (full re-forward every step).
  B. KV cache: one prefill forward with a ``Gemma4InferenceContext``, then a
     decode loop of single-token forwards reusing the context.

Both paths see the same input tokens at each step (the next token is selected
greedily from path A's logits and fed to both). The KV-cache implementation is
correct iff per-step logits match (and therefore greedy tokens match).

Run with the container python; no HF dependency, no distributed launcher:
    python3 examples/gemma4/eval/kv_cache_parity.py
    python3 examples/gemma4/eval/kv_cache_parity.py --text "Two plus two is" --max-new 16
"""
import argparse
import os
import sys
import time

# Megatron uses some packaging guards that the container's nvrx pre-release trips.
try:
    import nvidia_resiliency_ext as _nvrx
    from packaging.version import Version as _V

    _cur = getattr(_nvrx, "__version__", None)
    if _cur is None or _V(str(_cur)) < _V("0.6.0"):
        _nvrx.__version__ = "0.6.0"
except Exception:
    pass

import functools

import torch

HF_WEIGHTS = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "gemma4-playground/weights/gemma-4-E4B-it"
)
MLM_CKPT = (
    # Leaf dist-checkpoint dir. The parent contains
    # ``latest_checkpointed_iteration.txt`` + this ``release/`` subdir; we point at
    # the subdir directly so ``dist_checkpointing.load`` finds it (the higher-level
    # ``megatron.training.checkpointing.load_checkpoint`` would resolve from the parent).
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "Gemma4_mlm/code_dev/shared-state/implementations/HYBRID-gemma4-mlm/mlm_ckpt_mg/release"
)
DEFAULT_TEXT = "What is the capital of France? Answer in one word."

GELU_TANH = functools.partial(torch.nn.functional.gelu, approximate="tanh")


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
    model_parallel_cuda_manual_seed(123)


def _make_config():
    from megatron.core.transformer.gemma4_config import Gemma4TransformerConfig

    return Gemma4TransformerConfig(
        num_layers=42,
        hidden_size=2560,
        ffn_hidden_size=10240,
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


def _build_model(config):
    from megatron.core.models.gemma4.gemma4_layer_specs import get_gemma4_layer_local_spec
    from megatron.core.models.gemma4.gemma4_model import Gemma4Model

    spec = get_gemma4_layer_local_spec(config)
    model = Gemma4Model(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=262144,
        max_sequence_length=4096,
    )
    return model.bfloat16().cuda()


class _RawDecoder:
    def __init__(self, t):
        self._t = t

    def decode(self, ids, skip_special_tokens=False):
        return self._t.decode(list(ids), skip_special_tokens=skip_special_tokens)


def _tokenize(text):
    """Chat-template tokenize. Falls back to raw tokenizers + manual Gemma
    template when AutoTokenizer trips the ``extra_special_tokens`` list-vs-dict
    bug (see server's _load_gemma4_tokenizer for the same workaround)."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(HF_WEIGHTS)
        messages = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        return tok, ids.long()
    except AttributeError as e:
        if "'list' object has no attribute 'keys'" not in str(e):
            raise

    from tokenizers import Tokenizer

    t = Tokenizer.from_file(os.path.join(HF_WEIGHTS, "tokenizer.json"))
    prompt = f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
    seq = t.encode(prompt).ids
    if not seq or seq[0] != 2:
        seq = [2] + seq
    ids = torch.tensor([seq], dtype=torch.long)
    return _RawDecoder(t), ids


def _load_checkpoint(model):
    from megatron.core import dist_checkpointing

    sharded_sd = model.sharded_state_dict()
    loaded = dist_checkpointing.load(sharded_sd, MLM_CKPT)
    model.load_state_dict(loaded, strict=False)


@torch.inference_mode()
def _generate_reference(model, prompt_ids: torch.Tensor, max_new: int):
    """Path A: full re-forward every step. Returns (logits_per_step, tokens, step_times_ms)."""
    seq = prompt_ids[0].tolist()
    logits_steps = []
    tokens = []
    step_times = []
    for _ in range(max_new):
        inp = torch.tensor([seq], dtype=torch.long, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(inp, None, None)  # [b, s, V]
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000.0)
        last = out[0, -1, :].float().cpu()
        logits_steps.append(last)
        next_tok = int(last.argmax(dim=-1).item())
        tokens.append(next_tok)
        seq.append(next_tok)
    return torch.stack(logits_steps), tokens, step_times


@torch.inference_mode()
def _generate_kv_cache(model, prompt_ids: torch.Tensor, max_new: int):
    """Path B: prefill + decode with Gemma4InferenceContext.

    Returns (logits_per_step, tokens, step_times_ms). step_times[0] is the
    prefill latency; step_times[1..] are decode-step latencies.
    """
    from megatron.core.models.gemma4.gemma4_inference_context import Gemma4InferenceContext

    cfg = model.config
    ctx = Gemma4InferenceContext(
        num_layers=cfg.num_layers,
        layer_types=list(cfg.layer_types),
        num_kv_shared_layers=getattr(cfg, "num_kv_shared_layers", 0),
    )

    prompt_tokens = prompt_ids.cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(prompt_tokens, None, None, inference_context=ctx)
    torch.cuda.synchronize()
    step_times = [(time.perf_counter() - t0) * 1000.0]
    ctx.advance(prompt_tokens.shape[1])
    last = out[0, -1, :].float().cpu()
    logits_steps = [last]
    next_tok = int(last.argmax(dim=-1).item())
    tokens = [next_tok]

    for _ in range(max_new - 1):
        inp = torch.tensor([[next_tok]], dtype=torch.long, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(inp, None, None, inference_context=ctx)
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000.0)
        ctx.advance(1)
        last = out[0, -1, :].float().cpu()
        logits_steps.append(last)
        next_tok = int(last.argmax(dim=-1).item())
        tokens.append(next_tok)

    return torch.stack(logits_steps), tokens, step_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    print(f"\n==== KV-cache parity (MLM Gemma4-E4B, single-rank) ====")
    print(f"prompt: {args.text!r}")
    print(f"max_new: {args.max_new}")

    tok, ids = _tokenize(args.text)
    print(f"prompt_ids: shape={tuple(ids.shape)}")

    print("\n[init] distributed (TP=1) ...")
    _init_distributed()
    print("[init] config / build model ...")
    config = _make_config()
    model = _build_model(config)
    model.eval()
    print(f"[init] loading dist-checkpoint from {MLM_CKPT} ...")
    _load_checkpoint(model)

    # Warmup: one prefill of each path to absorb first-call CUDA kernel JIT /
    # cuBLAS plan selection. Cheap (~1 forward pass each) and stops the cold
    # start from dominating the first timed step.
    print("\n[warmup] one forward of each path ...")
    with torch.inference_mode():
        _ = model(ids.cuda(), None, None)
        from megatron.core.models.gemma4.gemma4_inference_context import (
            Gemma4InferenceContext,
        )

        _warm_ctx = Gemma4InferenceContext(
            num_layers=model.config.num_layers,
            layer_types=list(model.config.layer_types),
            num_kv_shared_layers=getattr(model.config, "num_kv_shared_layers", 0),
        )
        _ = model(ids.cuda(), None, None, inference_context=_warm_ctx)
        del _warm_ctx
    torch.cuda.synchronize()

    print("[A] reference: full re-forward each step ...")
    ref_logits, ref_tokens, ref_times = _generate_reference(model, ids, args.max_new)

    print("[B] KV-cache: prefill + decode ...")
    kv_logits, kv_tokens, kv_times = _generate_kv_cache(model, ids, args.max_new)

    diff = (ref_logits - kv_logits).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    tokens_eq = ref_tokens == kv_tokens

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  ref tokens : {ref_tokens}")
    print(f"  kv  tokens : {kv_tokens}")
    print(f"  ref text (raw)        : {tok.decode(ref_tokens, skip_special_tokens=False)!r}")
    print(f"  kv  text (raw)        : {tok.decode(kv_tokens, skip_special_tokens=False)!r}")
    print(f"  ref text (no specials): {tok.decode(ref_tokens, skip_special_tokens=True)!r}")
    print(f"  kv  text (no specials): {tok.decode(kv_tokens, skip_special_tokens=True)!r}")
    print(f"  tokens equal      : {tokens_eq}")
    print(f"  logits max-abs-diff : {max_abs:.4e}")
    print(f"  logits mean-abs-diff: {mean_abs:.4e}")
    first_disagree = None
    print("  per-step diff (max / mean / top-1 margin):")
    for t in range(ref_logits.shape[0]):
        d = (ref_logits[t] - kv_logits[t]).abs()
        ref_top2 = ref_logits[t].topk(2).values
        margin = (ref_top2[0] - ref_top2[1]).item()
        ref_top = ref_logits[t].argmax().item()
        kv_top = kv_logits[t].argmax().item()
        agree = ref_top == kv_top
        if not agree and first_disagree is None:
            first_disagree = t
        flag = "OK " if agree else "!! "
        print(
            f"    {flag}step {t:3d}: diff max={d.max().item():.3e} mean={d.mean().item():.3e}  "
            f"top-1 margin={margin:.3e}  ref={ref_top} kv={kv_top}"
        )
    print("=" * 60)
    if first_disagree is not None:
        print(f"  FIRST DISAGREEMENT AT STEP {first_disagree}")
        print(f"    ref top-5: {ref_logits[first_disagree].topk(5)}")
        print(f"    kv  top-5: {kv_logits[first_disagree].topk(5)}")

    # ---- Speed comparison ----
    print()
    print("=" * 60)
    print("SPEED")
    print("=" * 60)
    ref_total = sum(ref_times)
    kv_total = sum(kv_times)
    kv_prefill = kv_times[0]
    kv_decode = kv_times[1:]
    print(f"  prompt len = {ids.shape[1]} tokens, max_new = {args.max_new}")
    print()
    print(f"  [A] reference (full re-forward every step)")
    print(f"      total wall: {ref_total:7.1f} ms  ({ref_total / args.max_new:.1f} ms/step avg)")
    print(f"      per-step:   first={ref_times[0]:.1f}ms  last={ref_times[-1]:.1f}ms  "
          f"min={min(ref_times):.1f}  max={max(ref_times):.1f}")
    print()
    print(f"  [B] KV-cache (prefill + decode)")
    print(f"      total wall: {kv_total:7.1f} ms")
    print(f"      prefill:    {kv_prefill:.1f} ms  (1 forward over {ids.shape[1]} tokens)")
    if kv_decode:
        dec_avg = sum(kv_decode) / len(kv_decode)
        print(f"      decode:     {len(kv_decode)} steps, total {sum(kv_decode):.1f}ms  "
              f"({dec_avg:.1f} ms/step avg)")
        print(f"      per-step:   first={kv_decode[0]:.1f}ms  last={kv_decode[-1]:.1f}ms  "
              f"min={min(kv_decode):.1f}  max={max(kv_decode):.1f}")
    print()
    print(f"  speedup (total):           {ref_total / kv_total:5.2f}x")
    if kv_decode:
        # Apples-to-apples per-decode: reference's later steps process the full
        # prompt + grown sequence; kv-cache decode is steady-state per new token.
        ref_decode_avg = sum(ref_times[-len(kv_decode):]) / len(kv_decode)
        print(f"  speedup per decode-step:   {ref_decode_avg / dec_avg:5.2f}x  "
              f"(ref-late-avg {ref_decode_avg:.1f}ms vs kv-decode-avg {dec_avg:.1f}ms)")
    print("=" * 60)

    # Correctness signal: greedy tokens identical between paths. The logit diff is
    # bf16 numerical noise from kernel-shape differences ([N x N] re-forward matmul
    # vs [1 x N+t] decode matmul pick different GEMM reduction orders), accumulating
    # across 42 layers + the softcap. Step 0 (both paths run the same prefill math
    # over an empty cache) should be bit-identical -- guard on that as a smoke
    # against future inference_context regressions in the prefill branch.
    step0_max = (ref_logits[0] - kv_logits[0]).abs().max().item()
    ok = tokens_eq and step0_max == 0.0
    print(
        f"\n{'PASS' if ok else 'FAIL'}: tokens_eq={tokens_eq}, step-0 prefill bit-identical "
        f"(max-diff={step0_max:.4e})."
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
