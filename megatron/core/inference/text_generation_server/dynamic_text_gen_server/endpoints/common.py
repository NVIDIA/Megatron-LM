# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import threading

import torch

GENERATE_NUM = 0
LOCK = threading.Lock()


def send_do_generate():
    """Broadcasts a message to perform a generation to all tensor parallel ranks."""
    choice = torch.tensor([GENERATE_NUM], dtype=torch.long, device=torch.cuda.current_device())
    torch.distributed.broadcast(choice, 0)


def resolve_sampling_default(app_config, gen_defaults, key, config_key, hardcoded):
    """Resolve one sampling default: app config > generation_config > hardcoded.

    Precedence for a field the request omits:

    1. an explicitly configured server default (`config_key` present in the app
       config) -- an operator who set it should not be overridden by a model file;
    2. the model's own `generation_config.json` declaration;
    3. the previous hardcoded fallback, so a model without a generation_config and a
       server without configured defaults behave exactly as before.

    Key presence is what distinguishes "operator set 1.0" from "unset", so
    `.get(config_key, default)` is deliberately not used here.
    """
    if config_key in app_config:
        return app_config[config_key]
    if key in gen_defaults:
        return gen_defaults[key]
    return hardcoded


def generation_config_sampling_defaults(tokenizer):
    """Sampling defaults declared by the model's `generation_config.json`.

    HF models ship sampling defaults (`temperature`, `top_p`, `top_k`, `do_sample`)
    in `generation_config.json`, and vLLM applies them when a request omits the
    field. These endpoints did not consult that file, so a client that omitted
    `top_p` got 1.0 here but the model-declared value (e.g. 0.95) under vLLM -- a
    silent per-engine difference in the sampling tail.

    Returns a dict with only the keys the config actually declares, so callers can
    fall back to their own defaults for the rest. Request values always win; this
    only supplies defaults.
    """
    gen_cfg = getattr(tokenizer, "generation_config", None)
    if not isinstance(gen_cfg, dict):
        return {}
    defaults = {}
    for key in ("temperature", "top_p", "top_k"):
        value = gen_cfg.get(key)
        # bool is an int subclass; reject it explicitly.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            defaults[key] = value
    # Greedy decoding is expressed here as top_k=1, matching the temperature==0 path.
    if gen_cfg.get("do_sample") is False:
        defaults["top_k"] = 1
    return defaults


_LOGGED_SAMPLING_DEFAULTS = False


def log_sampling_defaults_once(tokenizer, resolved):
    """Log the resolved sampling defaults once per process.

    Mirrors the startup log for the EOS token set: it makes it possible to confirm
    from the server log that `generation_config.json` was found and applied, rather
    than inferring it from sampled output (which is hopeless on peaked
    distributions, where top_p barely truncates).
    """
    global _LOGGED_SAMPLING_DEFAULTS
    if _LOGGED_SAMPLING_DEFAULTS:
        return
    _LOGGED_SAMPLING_DEFAULTS = True

    gen_cfg = getattr(tokenizer, "generation_config", None)
    logging.info(
        "Sampling defaults: generation_config=%s -> defaults=%s; first request "
        "resolved to temperature=%s top_p=%s top_k=%s",
        (
            {k: gen_cfg.get(k) for k in ("temperature", "top_p", "top_k", "do_sample")}
            if isinstance(gen_cfg, dict)
            else None
        ),
        generation_config_sampling_defaults(tokenizer),
        resolved.get("temperature"),
        resolved.get("top_p"),
        resolved.get("top_k"),
    )
