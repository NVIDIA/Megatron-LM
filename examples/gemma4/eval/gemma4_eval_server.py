# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""KV-cache OpenAI-compatible inference server for Gemma4 (MLM).

Forward path
------------
``Gemma4Model.forward`` accepts an optional :class:`Gemma4InferenceContext`
(megatron/core/models/gemma4/gemma4_inference_context.py). The server creates
a fresh context per request, runs one prefill forward over the full prompt
(populates the per-layer KV cache; borrower layers alias their producer's
slot), then runs a decode loop of single-token forwards. Decode-step
position_ids continue from the cache offset, full-layer attention attends
over the entire cached K/V, and sliding-layer attention clips the cache to
the last ``sliding_window`` keys.

Migration to MCore's ``GPTInferenceWrapper`` / ``DynamicInferenceEngine`` is
a follow-up; the context surface here (``is_decode``, ``get_kv``,
``append_kv``, ``advance``) is intentionally small so the swap is mechanical.

Architecture
------------
``torch.distributed.run`` launches WORLD_SIZE = TP ranks on one node. All
ranks build the same Gemma4 model and participate in TP collectives inside
``model.forward``. Rank 0 also runs a Flask HTTP server on a background
thread. A global model lock serializes requests on rank 0 (single-flight).

Control protocol -- all broadcasts originate from rank 0 over the default
process group. Non-zero ranks sit in a loop blocked on broadcasts:

    rank0 -> broadcast header [cmd, n]
        cmd == PREFILL : also broadcast tokens [1, n]; all ranks create a
                         fresh Gemma4InferenceContext and run the prefill
                         forward; cache step is advanced by n.
        cmd == DECODE  : also broadcast tokens [1, 1]; all ranks run a
                         single-token forward against the existing context;
                         cache step is advanced by 1.
        cmd == END     : non-zero ranks drop the current context and loop
                         back to waiting for the next request.
        cmd == SHUTDOWN: non-zero ranks exit.

Endpoints
---------
    GET  /v1/health, /health        -- {"status": "ok"} once model is loaded
    POST /v1/completions            -- subset of OpenAI completions
    POST /v1/chat/completions       -- subset of OpenAI chat (chat template
                                       applied via HF AutoTokenizer)
"""

# Megatron-LM container vs nvrx version shim. Must run before any megatron
# import (mirrors pretrain_gemma4.py).
import os as _os
import sys as _sys
import time as _time

_PROGRAM_START_TIME = _time.time()

_REPO_ROOT = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), _os.path.pardir, _os.path.pardir, _os.path.pardir)
)
_GEMMA4_EXAMPLES = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.path.pardir))
for _p in (_REPO_ROOT, _GEMMA4_EXAMPLES):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
from functools import partial  # noqa: E402
from typing import List, Optional  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# Trigger the nvrx shim defined in pretrain_gemma4.py before any megatron import.
import pretrain_gemma4  # noqa: F401, E402  -- module import runs _apply_nvrx_version_shim()
from megatron.core.models.gemma4.gemma4_inference_context import (  # noqa: E402
    Gemma4InferenceContext,
)
from megatron.training import get_args, get_model, initialize_megatron  # noqa: E402
from megatron.training.arguments import parse_and_validate_args  # noqa: E402
from megatron.training.checkpointing import load_checkpoint  # noqa: E402
from model_provider import model_provider  # noqa: E402
from pretrain_gemma4 import gemma4_builder  # noqa: E402

logger = logging.getLogger("gemma4_eval_server")

# Control protocol opcodes.
CMD_PREFILL = 0
CMD_DECODE = 1
CMD_END = 2
CMD_SHUTDOWN = 3


def add_server_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="Gemma4 eval server")
    group.add_argument("--host", type=str, default="0.0.0.0")
    group.add_argument("--port", type=int, default=5000)
    group.add_argument(
        "--gemma4-hf-tokenizer",
        type=str,
        required=True,
        help="Path or HF id of the original Gemma4 release. Used for tokenization "
        "and chat-template application on rank 0 only; Megatron-side tokenizer "
        "stays NullTokenizer.",
    )
    group.add_argument("--max-new-tokens", type=int, default=512)
    group.add_argument("--default-temperature", type=float, default=0.0)
    group.add_argument("--default-top-p", type=float, default=1.0)
    return parser


def _broadcast_header(cmd: int, n: int) -> None:
    """Rank 0 sends a 2-int control header to all ranks."""
    header = torch.tensor([cmd, n], dtype=torch.long, device="cuda")
    dist.broadcast(header, src=0)


def _recv_header() -> tuple:
    header = torch.empty(2, dtype=torch.long, device="cuda")
    dist.broadcast(header, src=0)
    return int(header[0].item()), int(header[1].item())


def _new_inference_context(model) -> Gemma4InferenceContext:
    """Build a fresh per-request KV-cache context from the model's config."""
    cfg = model.config
    return Gemma4InferenceContext(
        num_layers=cfg.num_layers,
        layer_types=list(cfg.layer_types),
        num_kv_shared_layers=getattr(cfg, "num_kv_shared_layers", 0),
    )


@torch.inference_mode()
def _forward_step(
    model, tokens: torch.Tensor, inference_context: Gemma4InferenceContext
) -> Optional[torch.Tensor]:
    """All ranks run model.forward against the shared inference context.

    Returns last-position logits on rank 0 (float32), None elsewhere. The
    caller is responsible for advancing ``inference_context.step`` after this
    call.
    """
    out = model(tokens, None, None, inference_context=inference_context)
    if dist.get_rank() == 0:
        # out has shape [b, s, V] (Gemma4Model.forward transposes back).
        return out[:, -1, :].float()
    return None


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Greedy if temperature == 0; else temperature + top-p (nucleus)."""
    if temperature <= 0.0:
        return int(logits.argmax(dim=-1).item())
    probs = F.softmax(logits / temperature, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_p, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_p, dim=-1)
        # Keep the smallest prefix whose mass >= top_p.
        keep = cum <= top_p
        keep[..., 0] = True  # always keep top-1.
        sorted_p = sorted_p * keep
        sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        idx_in_sorted = torch.multinomial(sorted_p, num_samples=1)
        return int(sorted_idx.gather(-1, idx_in_sorted).item())
    return int(torch.multinomial(probs, num_samples=1).item())


def _generate(model, prompt_ids: List[int], max_new: int, temperature: float,
              top_p: float, eos_ids: List[int]) -> List[int]:
    """Rank-0 driver. Issues control broadcasts; returns the generated token list.

    One prefill forward over the full prompt seeds the KV cache, then a decode
    loop of single-token forwards each appends one new K/V per layer.
    """
    assert dist.get_rank() == 0
    ctx = _new_inference_context(model)

    # Prefill.
    prompt_tokens = torch.tensor([list(prompt_ids)], dtype=torch.long, device="cuda")
    n_prompt = prompt_tokens.shape[1]
    _broadcast_header(CMD_PREFILL, n_prompt)
    dist.broadcast(prompt_tokens, src=0)
    logits = _forward_step(model, prompt_tokens, ctx)
    ctx.advance(n_prompt)
    next_tok = _sample(logits[0], temperature, top_p)

    generated: List[int] = [next_tok]
    if next_tok in eos_ids or max_new <= 1:
        _broadcast_header(CMD_END, 0)
        return generated

    # Decode loop.
    for _ in range(max_new - 1):
        tok = torch.tensor([[next_tok]], dtype=torch.long, device="cuda")
        _broadcast_header(CMD_DECODE, 1)
        dist.broadcast(tok, src=0)
        logits = _forward_step(model, tok, ctx)
        ctx.advance(1)
        next_tok = _sample(logits[0], temperature, top_p)
        generated.append(next_tok)
        if next_tok in eos_ids:
            break
    _broadcast_header(CMD_END, 0)
    return generated


def _worker_loop(model) -> None:
    """Non-zero ranks: receive control headers and execute forwards in lockstep.

    A fresh inference context is created on each CMD_PREFILL and reused for
    subsequent CMD_DECODE steps until CMD_END.
    """
    ctx: Optional[Gemma4InferenceContext] = None
    while True:
        cmd, n = _recv_header()
        if cmd == CMD_SHUTDOWN:
            return
        if cmd == CMD_END:
            ctx = None
            continue
        if cmd == CMD_PREFILL:
            ctx = _new_inference_context(model)
            tokens = torch.empty((1, n), dtype=torch.long, device="cuda")
            dist.broadcast(tokens, src=0)
            _ = _forward_step(model, tokens, ctx)
            ctx.advance(n)
        elif cmd == CMD_DECODE:
            assert ctx is not None, "CMD_DECODE arrived before CMD_PREFILL"
            tokens = torch.empty((1, n), dtype=torch.long, device="cuda")
            dist.broadcast(tokens, src=0)
            _ = _forward_step(model, tokens, ctx)
            ctx.advance(n)
        else:
            raise RuntimeError(f"unknown control cmd {cmd}")


def _load_gemma4_tokenizer(path: str):
    """Load the Gemma4 HF tokenizer, working around a transformers >=4.55-ish
    schema change for ``extra_special_tokens``.

    The Gemma4 release ships ``tokenizer_config.json`` with
    ``extra_special_tokens: ["<|video|>"]`` (list). transformers expects a
    ``Dict[str, str]`` and crashes in ``_set_model_specific_special_tokens``
    with ``AttributeError: 'list' object has no attribute 'keys'``. We stage a
    tempdir of symlinks to the release, rewrite that one key to
    ``{"video": "<|video|>"}``, and load from the staged dir.
    """
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except AttributeError as e:
        if "'list' object has no attribute 'keys'" not in str(e):
            raise
        logger.warning("patching extra_special_tokens (list -> dict) for %s", path)
    cfg_name = "tokenizer_config.json"
    src_cfg = os.path.join(path, cfg_name)
    cfg = json.load(open(src_cfg))
    extras = cfg.get("extra_special_tokens", [])
    if isinstance(extras, list):
        cfg["extra_special_tokens"] = {
            tok.strip("<|>").strip() or f"extra_{i}": tok for i, tok in enumerate(extras)
        }
    staged = tempfile.mkdtemp(prefix="gemma4_tk_")
    for entry in os.listdir(path):
        src = os.path.join(path, entry)
        if not os.path.isfile(src):
            continue
        if entry == cfg_name:
            continue
        os.symlink(src, os.path.join(staged, entry))
    with open(os.path.join(staged, cfg_name), "w") as f:
        json.dump(cfg, f)
    return AutoTokenizer.from_pretrained(staged, trust_remote_code=True)


def _build_app(model, hf_tokenizer, args, model_lock: threading.Lock) -> Flask:
    app = Flask(__name__)
    eos_ids = [tok for tok in [hf_tokenizer.eos_token_id, hf_tokenizer.pad_token_id] if tok is not None]

    @app.get("/health")
    @app.get("/v1/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/v1/completions")
    def completions():
        body = request.get_json(force=True)
        prompt = body["prompt"]
        if isinstance(prompt, list):
            prompt = prompt[0]
        max_new = int(body.get("max_tokens", args.max_new_tokens))
        temperature = float(body.get("temperature", args.default_temperature))
        top_p = float(body.get("top_p", args.default_top_p))
        prompt_ids = hf_tokenizer(prompt, add_special_tokens=False).input_ids
        with model_lock:
            out_ids = _generate(model, prompt_ids, max_new, temperature, top_p, eos_ids)
        text = hf_tokenizer.decode(out_ids, skip_special_tokens=True)
        return jsonify({
            "id": "cmpl-gemma4-mlm",
            "object": "text_completion",
            "model": "gemma4-mlm",
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(out_ids),
                "total_tokens": len(prompt_ids) + len(out_ids),
            },
        })

    @app.post("/v1/chat/completions")
    def chat_completions():
        body = request.get_json(force=True)
        messages = body["messages"]
        max_new = int(body.get("max_tokens", args.max_new_tokens))
        temperature = float(body.get("temperature", args.default_temperature))
        top_p = float(body.get("top_p", args.default_top_p))
        prompt_ids = hf_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        with model_lock:
            out_ids = _generate(model, prompt_ids, max_new, temperature, top_p, eos_ids)
        text = hf_tokenizer.decode(out_ids, skip_special_tokens=True)
        return jsonify({
            "id": "chatcmpl-gemma4-mlm",
            "object": "chat.completion",
            "model": "gemma4-mlm",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(out_ids),
                "total_tokens": len(prompt_ids) + len(out_ids),
            },
        })

    return app


def main():
    parse_and_validate_args(
        extra_args_provider=add_server_args,
        args_defaults={"no_load_rng": True, "no_load_optim": True},
    )
    initialize_megatron()
    args = get_args()

    # Build the model via the existing Gemma4 builder. wrap_with_ddp=False
    # mirrors megatron/inference/utils.py:90 for the inference path.
    model = get_model(partial(model_provider, gemma4_builder), wrap_with_ddp=False)
    assert args.load is not None, "--load is required"
    args.exit_on_missing_checkpoint = True
    load_checkpoint(
        ddp_model=model,
        optimizer=None,
        opt_param_scheduler=None,
        strict=not getattr(args, "inference_ckpt_non_strict", False),
    )
    assert len(model) == 1, "virtual pipeline parallelism not supported in eval server"
    model = model[0]
    model.eval()

    rank = dist.get_rank()
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logger.info("loading HF tokenizer from %s", args.gemma4_hf_tokenizer)
        hf_tokenizer = _load_gemma4_tokenizer(args.gemma4_hf_tokenizer)
        if hf_tokenizer.chat_template is None:
            logger.warning("HF tokenizer has no chat_template; /v1/chat/completions will fail")
        model_lock = threading.Lock()
        app = _build_app(model, hf_tokenizer, args, model_lock)
        # Run Flask on a background thread so the main thread can drive control
        # broadcasts inside request handlers (the handlers run on Flask's thread
        # but acquire model_lock; the broadcasts happen there, not here).
        from werkzeug.serving import make_server
        server = make_server(args.host, args.port, app, threaded=False)
        logger.info("gemma4 eval server listening on %s:%d", args.host, args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("shutting down")
        finally:
            # Tell other ranks to exit their control loop.
            _broadcast_header(CMD_SHUTDOWN, 0)
    else:
        _worker_loop(model)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
