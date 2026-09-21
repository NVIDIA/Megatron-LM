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
