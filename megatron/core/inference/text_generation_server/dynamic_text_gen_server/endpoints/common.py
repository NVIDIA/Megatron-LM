# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import threading
from typing import TYPE_CHECKING, Iterable

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


def send_do_generate():
    """Broadcasts a message to perform a generation to all tensor parallel ranks."""
    choice = torch.tensor([GENERATE_NUM], dtype=torch.long, device=torch.cuda.current_device())
    torch.distributed.broadcast(choice, 0)


def openai_error(message, status, error_type=None, param=None, code=None):
    """Build an OpenAI-shaped error response for an endpoint to return.

    OpenAI's clients expect a JSON body of
    ``{"error": {"message", "type", "param", "code"}}`` and read the human-readable
    text out of ``error.message``. A bare string body makes them raise a parse
    error that hides the message entirely, so every endpoint error goes through
    here. ``message`` is passed through verbatim; callers that must preserve an
    exact wording for downstream consumers still get that wording, just nested
    under ``error.message``.

    Args:
        message (str): Human-readable description of what went wrong.
        status (int): HTTP status code.
        error_type (str | None): OpenAI error type. Defaults to
            ``invalid_request_error`` for 4xx and ``server_error`` otherwise.
        param (str | None): Offending request field, when one can be named.
        code (str | None): Machine-readable error code, when one applies.

    Returns:
        tuple[dict, int]: Body and status, in the form Quart returns directly.
    """
    if error_type is None:
        error_type = "invalid_request_error" if 400 <= status < 500 else "server_error"
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}, status
