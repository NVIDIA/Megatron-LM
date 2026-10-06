# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import threading
from typing import TYPE_CHECKING, Iterable, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.inference.inference_client import InferenceClient

GENERATE_NUM = 0
LOCK = threading.Lock()

logger = logging.getLogger(__name__)


def abort_requests(client: "InferenceClient", request_ids: Iterable[int], reason: str) -> None:
    """Tell the coordinator to stop generating the given requests.

    Best-effort and never raises: it runs on paths that are already unwinding
    (a cancelled handler, or an error response), where letting a second failure
    escape would replace the real one. A request that has already finished is
    not an error to abort -- abort_request returns without recording an id it
    no longer holds local state for.

    Args:
        client: The InferenceClient the requests were submitted through.
        request_ids: Ids from add_request_with_id.
        reason: Logged so the abort can be told apart from a normal completion.
    """
    for request_id in request_ids:
        try:
            client.abort_request(request_id)
        except Exception:  # pylint: disable=broad-except
            logger.warning("Failed to abort request %s (%s)", request_id, reason, exc_info=True)
        else:
            logger.debug("Aborted request %s (%s)", request_id, reason)


def validate_offload_params(offload_params) -> Optional[str]:
    """Return an error message if client-supplied ``offload_params`` are malformed, else None.

    Top-level keys starting with ``_`` are engine-owned control fields (for example
    the prompt-preparation error the engine stamps on MP rank 0), so a client is not
    allowed to supply them. Called on the raw request value, so the ``dict`` check
    lives here too.
    """
    if offload_params is None:
        return None
    if not isinstance(offload_params, dict):
        return "'offload_params' must be an object"
    reserved = sorted(key for key in offload_params if isinstance(key, str) and key.startswith("_"))
    if reserved:
        return f"'offload_params' keys starting with '_' are reserved: {reserved}"
    return None


def collect_stage_metadata(response_metadata: dict, result: dict) -> None:
    """Fold one reply's ``payload_stage_metadata`` into the response-level dict.

    Every request in a batch goes through the same stager, so a key that already
    exists must carry the same value; a mismatch means the stager returned
    per-request metadata that cannot be represented once at the top level.
    """
    stage_metadata = result.get("payload_stage_metadata") or {}
    for key, value in stage_metadata.items():
        if key in response_metadata and response_metadata[key] != value:
            raise ValueError(f"payload stager returned conflicting response metadata for {key!r}")
        response_metadata[key] = value


def attach_stage_metadata(response: dict, response_metadata: dict) -> dict:
    """Merge the stager's response metadata into the top-level response body.

    The OpenAI-shaped fields already in ``response`` are reserved; a stager key that
    collides with one is a bug rather than something to overwrite silently.
    """
    overlap = set(response).intersection(response_metadata)
    if overlap:
        raise ValueError(
            f"payload stager response metadata collides with reserved fields: {sorted(overlap)}"
        )
    response.update(response_metadata)
    return response


def send_do_generate():
    """Broadcasts a message to perform a generation to all tensor parallel ranks."""
    choice = torch.tensor([GENERATE_NUM], dtype=torch.long, device=torch.cuda.current_device())
    torch.distributed.broadcast(choice, 0)


def apply_optional_sampling_default(app_config, config_key, value) -> None:
    """Set `app_config[config_key]` only when `value` was actually provided.

    Startup passes `default_temperature`/`default_top_p`/`default_top_k`
    through as `Optional`, `None` when the operator didn't configure one.
    Setting the key unconditionally (even to a hardcoded fallback like `1.0`)
    would make `resolve_sampling_default`'s `config_key in app_config` check
    always true, permanently hiding the model's own `generation_config.json`
    tier behind a value nobody actually asked for.
    """
    if value is not None:
        app_config[config_key] = value


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
