"""Live text parity: HF Gemma4 vs MLM Gemma4 on real text with chat template.

Single command, run with the container python3 (has TE/CUDA deps):
    python3 text_parity.py
    python3 text_parity.py --text "Explain gravity in one sentence."

Internally spawns a subprocess using gemma4_venv python to tokenize + run HF
(needs transformers 5.x), saves a temp ref, then loads it and runs the MLM
model in this process (needs TE/container deps). Prints a comparison at the end.
"""
import argparse
import os
import subprocess
import sys
import tempfile

GEMMA4_VENV_PYTHON = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "gemma4_venv/bin/python"
)
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

# ---------------------------------------------------------------------------
# HF worker — this code runs ONLY in the gemma4_venv subprocess.
# Invoked as:  gemma4_venv/bin/python text_parity.py --_hf-worker <text> <ref_path>
# ---------------------------------------------------------------------------
HF_WORKER_FLAG = "--_hf-worker"


def _hf_worker(text: str, ref_path: str):
    import torch
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(HF_WEIGHTS)
    messages = [{"role": "user", "content": text}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"] if "input_ids" in ids else torch.tensor(ids)
    ids = ids.long()
    print(f"[HF] {ids.shape[1]} tokens: {ids[0].tolist()}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HF] loading model on {device} ...", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        HF_WEIGHTS, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval().to(device)

    with torch.no_grad():
        out = model(ids.to(device), use_cache=False)
    logits = out.logits.cpu()
    greedy = logits.argmax(dim=-1)
    print(f"[HF] logits {tuple(logits.shape)}  next_id={greedy[0,-1].item()}  "
          f"decoded={tok.decode([greedy[0,-1].item()])!r}", flush=True)

    torch.save({"ids": ids, "hf_logits": logits, "hf_greedy": greedy, "text": text}, ref_path)
    print(f"[HF] saved ref -> {ref_path}", flush=True)


# ---------------------------------------------------------------------------
# MLM forward — runs in the main (container) process.
# ---------------------------------------------------------------------------

def _run_mlm(ids):
    if MLM_SRC not in sys.path:
        sys.path.insert(0, MLM_SRC)
    try:
        import nvidia_resiliency_ext as _nvrx
        if not hasattr(_nvrx, "__version__"):
            _nvrx.__version__ = "0.6.0"
    except Exception:
        pass

    import torch

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from gemma4_common import _build_model, _init_distributed, _make_config
    from megatron.core.models.gemma4.gemma4_layer_specs import get_gemma4_layer_local_spec
    from megatron.core import dist_checkpointing

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
        logits = model(ids.cuda()).cpu()  # [1, S, V]
    greedy = logits.argmax(dim=-1)
    print(f"[MLM] logits {tuple(logits.shape)}  next_id={greedy[0,-1].item()}", flush=True)
    return logits, greedy


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--ref",  default=None,
                    help="path for the HF ref .pt (default: auto temp file)")
    # internal flag used by the subprocess — not for users
    ap.add_argument(HF_WORKER_FLAG, nargs=2, metavar=("TEXT", "REF"), dest="hf_worker",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    # If we're running as the HF subprocess, do HF work and exit.
    if args.hf_worker:
        _hf_worker(args.hf_worker[0], args.hf_worker[1])
        return

    import torch

    # Step 1: spawn HF subprocess using gemma4_venv python.
    use_tmp = args.ref is None
    ref_path = args.ref or tempfile.mktemp(suffix="_text_parity_ref.pt")

    print(f"\n{'='*60}")
    print(f"TEXT  : {args.text!r}")
    print(f"{'='*60}")
    print("\n--- Step 1: HF forward (gemma4_venv subprocess) ---")

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)  # same as env -u PYTHONPATH
    result = subprocess.run(
        [GEMMA4_VENV_PYTHON, __file__, HF_WORKER_FLAG, args.text, ref_path],
        env=env,
    )
    if result.returncode != 0:
        print(f"\n[ERROR] HF subprocess exited with code {result.returncode}")
        sys.exit(1)

    # Step 2: load ref + run MLM in this process.
    print("\n--- Step 2: MLM forward (container python) ---")
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    ids       = ref["ids"]
    hf_logits = ref["hf_logits"]
    hf_greedy = ref["hf_greedy"]

    mlm_logits, mlm_greedy = _run_mlm(ids)

    # Clean up temp file.
    if use_tmp:
        try:
            os.remove(ref_path)
        except OSError:
            pass

    # Compare.
    diff = (hf_logits.float() - mlm_logits.float()).abs()
    exact = torch.equal(hf_greedy, mlm_greedy)
    match_count = (hf_greedy == mlm_greedy).sum().item()

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  logits max-abs-diff : {diff.max().item():.4e}")
    print(f"  logits mean-abs-diff: {diff.mean().item():.4e}")
    print(f"  greedy exact match  : {exact}  ({match_count}/{ids.shape[1]} tokens)")
    print(f"  HF  greedy : {hf_greedy[0].tolist()}")
    print(f"  MLM greedy : {mlm_greedy[0].tolist()}")
    print(f"  HF  next token id={hf_greedy[0,-1].item()}")
    print(f"  MLM next token id={mlm_greedy[0,-1].item()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
