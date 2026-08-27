# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
import threading
from typing import Iterable

import torch

GENERATE_NUM = 0
LOCK = threading.Lock()

logger = logging.getLogger(__name__)


def abort_requests(client, request_ids: Iterable[int], reason: str) -> None:
    """Tell the coordinator to stop generating the given requests.

    Best-effort and never raises: it runs on paths that are already unwinding
    (a cancelled handler, or an error response), where letting a second failure
    escape would replace the real one. A request that has already finished is
    not an error to abort -- abort_request drops an unknown id.

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


def send_do_generate():
    """Broadcasts a message to perform a generation to all tensor parallel ranks."""
    choice = torch.tensor([GENERATE_NUM], dtype=torch.long, device=torch.cuda.current_device())
    torch.distributed.broadcast(choice, 0)
