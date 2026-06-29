# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import List

from ..nvshmem_types import MAX_SEGMENT_SIZE, ReceiveRequest, SendRequest

logger = logging.getLogger(__name__)

# Constants for ID encoding (from C++ implementation)
REQUEST_ID_BASE = 1000000000
SEGMENT_ID_MULTIPLIER = 1000
MAX_REQUESTS = 1000000
MAX_SEGMENTS_PER_REQUEST = 1000


class TaskSegmenter:
    """
    Splits large tasks (>256MB) into smaller segments to fit
    into the fixed-size communication slots.
    """

    def _encode_segment_id(self, task_id: int, segment_index: int) -> int:
        return REQUEST_ID_BASE + (task_id * SEGMENT_ID_MULTIPLIER) + segment_index

    def _calculate_num_segments(self, size: int) -> int:
        return (size + MAX_SEGMENT_SIZE - 1) // MAX_SEGMENT_SIZE

    def _validate_segmentation(self, task_id: int, size: int) -> bool:
        num_segments = self._calculate_num_segments(size)
        if num_segments > MAX_SEGMENTS_PER_REQUEST:
            logger.error(
                f"Error: Task {task_id} requires {num_segments} segments, "
                f"exceeds max {MAX_SEGMENTS_PER_REQUEST}"
            )
            return False
        if task_id >= MAX_REQUESTS:
            logger.error(f"Error: Task ID {task_id} exceeds max {MAX_REQUESTS}")
            return False
        return True

    def segment_send_request(self, req: SendRequest) -> List[SendRequest]:
        """
        Splits a single send request into multiple segments
        if larger than MAX_SEGMENT_SIZE.
        """
        if req.size <= MAX_SEGMENT_SIZE:
            return [req]

        if not self._validate_segmentation(req.task_id, req.size):
            raise ValueError(f"Task {req.task_id} validation failed")

        num_segments = self._calculate_num_segments(req.size)
        output_requests: List[SendRequest] = []

        for i in range(num_segments):
            segment_offset = i * MAX_SEGMENT_SIZE
            segment_size = min(MAX_SEGMENT_SIZE, req.size - segment_offset)
            segment_task_id = self._encode_segment_id(req.task_id, i)

            new_req = SendRequest(
                task_id=segment_task_id,
                src_tensor=req.src_tensor,
                src_pos=req.src_pos + segment_offset,
                size=segment_size,
                dest_pe=req.dest_pe,
            )
            output_requests.append(new_req)

        return output_requests

    def segment_receive_request(self, req: ReceiveRequest) -> List[ReceiveRequest]:
        """
        Splits a single receive request into multiple segments
        if larger than MAX_SEGMENT_SIZE.
        """
        if req.size <= MAX_SEGMENT_SIZE:
            return [req]

        if not self._validate_segmentation(req.task_id, req.size):
            raise ValueError(f"Task {req.task_id} validation failed")

        num_segments = self._calculate_num_segments(req.size)
        output_requests: List[ReceiveRequest] = []

        for i in range(num_segments):
            segment_offset = i * MAX_SEGMENT_SIZE
            segment_size = min(MAX_SEGMENT_SIZE, req.size - segment_offset)
            segment_task_id = self._encode_segment_id(req.task_id, i)

            new_req = ReceiveRequest(
                task_id=segment_task_id,
                dest_tensor=req.dest_tensor,
                dest_pos=req.dest_pos + segment_offset,
                size=segment_size,
                src_pe=req.src_pe,
            )
            output_requests.append(new_req)

        return output_requests
