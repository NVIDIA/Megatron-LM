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
