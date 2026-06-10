# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import pytest

from megatron.core.resharding.nvshmem_copy_service.nvshmem_types import (
    MAX_SEGMENT_SIZE,
    ReceiveRequest,
    SendRequest,
)
from megatron.core.resharding.nvshmem_copy_service.planning.task_segmenter import TaskSegmenter


class TestTaskSegmenter:
    """Test suite for TaskSegmenter."""

    def test_segment_small_request(self):
        """Test segmenting a request smaller than max segment size."""
        segmenter = TaskSegmenter()

        # Request smaller than 256MB should not be segmented
        send_req = SendRequest(
            task_id=1, src_tensor=None, src_pos=0, size=500 * 1024, dest_pe=1
        )  # 500KB
        recv_req = ReceiveRequest(
            task_id=1, dest_tensor=None, dest_pos=0, size=500 * 1024, src_pe=0
        )

        send_segments = segmenter.segment_send_request(send_req)
        recv_segments = segmenter.segment_receive_request(recv_req)

        # Should produce exactly one segment (no splitting)
        assert len(send_segments) == 1
        assert send_segments[0].task_id == 1
        assert send_segments[0].size == 500 * 1024
        assert send_segments[0].dest_pe == 1

        assert len(recv_segments) == 1
        assert recv_segments[0].task_id == 1
        assert recv_segments[0].size == 500 * 1024

    def test_segment_large_request(self):
        """Test segmenting a request larger than max segment size."""
        segmenter = TaskSegmenter()

        # Request larger than 256MB should be segmented
        task_size = 3 * MAX_SEGMENT_SIZE  # 768MB
        send_req = SendRequest(task_id=1, src_tensor=None, src_pos=0, size=task_size, dest_pe=1)

        send_segments = segmenter.segment_send_request(send_req)

        # Should produce 3 segments
        assert len(send_segments) == 3
        for segment in send_segments:
            assert segment.size == MAX_SEGMENT_SIZE  # Each segment is max size
            assert segment.dest_pe == 1

    def test_segment_not_exact_multiple(self):
        """Test segmenting when size is not exact multiple of max segment size."""
        segmenter = TaskSegmenter()

        # 2.5 × 256MB = 640MB -> should produce 3 segments (256MB, 256MB, 128MB)
        task_size = int(2.5 * MAX_SEGMENT_SIZE)
        send_req = SendRequest(task_id=1, src_tensor=None, src_pos=0, size=task_size, dest_pe=1)

        send_segments = segmenter.segment_send_request(send_req)

        # Should produce 3 segments
        assert len(send_segments) == 3
        # First two segments are full size
        assert send_segments[0].size == MAX_SEGMENT_SIZE
        assert send_segments[1].size == MAX_SEGMENT_SIZE
        # Last segment is remainder
        assert send_segments[2].size == int(0.5 * MAX_SEGMENT_SIZE)

    def test_segment_send_and_receive_match(self):
        """Test that send and receive segmentation produces matching segments."""
        segmenter = TaskSegmenter()

        task_size = int(2.5 * MAX_SEGMENT_SIZE)
        send_req = SendRequest(task_id=1, src_tensor=None, src_pos=0, size=task_size, dest_pe=1)
        recv_req = ReceiveRequest(task_id=1, dest_tensor=None, dest_pos=0, size=task_size, src_pe=0)

        send_segments = segmenter.segment_send_request(send_req)
        recv_segments = segmenter.segment_receive_request(recv_req)

        # Should produce same number of segments
        assert len(send_segments) == len(recv_segments)

        # Sizes should match
        for send_seg, recv_seg in zip(send_segments, recv_segments):
            assert send_seg.size == recv_seg.size

    def test_segment_very_large_request(self):
        """Test segmenting a very large request."""
        segmenter = TaskSegmenter()

        # 10 × 256MB = 2.56GB
        task_size = 10 * MAX_SEGMENT_SIZE
        send_req = SendRequest(task_id=1, src_tensor=None, src_pos=0, size=task_size, dest_pe=1)

        send_segments = segmenter.segment_send_request(send_req)

        # Should produce 10 segments
        assert len(send_segments) == 10
        # All segments should be full size
        for segment in send_segments:
            assert segment.size == MAX_SEGMENT_SIZE

    def test_segment_zero_size_request(self):
        """Test handling of zero-size request."""
        segmenter = TaskSegmenter()

        send_req = SendRequest(task_id=1, src_tensor=None, src_pos=0, size=0, dest_pe=1)

        send_segments = segmenter.segment_send_request(send_req)

        # Should produce one segment with size 0
        assert len(send_segments) == 1
        assert send_segments[0].size == 0

    def test_segment_exactly_max_size(self):
        """Test segmenting request that is exactly max segment size."""
        segmenter = TaskSegmenter()

        # Exactly 256MB - should NOT be segmented
        send_req = SendRequest(
            task_id=1, src_tensor=None, src_pos=0, size=MAX_SEGMENT_SIZE, dest_pe=1
        )

        send_segments = segmenter.segment_send_request(send_req)

        # Should produce exactly 1 segment (no splitting needed)
        assert len(send_segments) == 1
        assert send_segments[0].size == MAX_SEGMENT_SIZE

    def test_segment_preserves_destination(self):
        """Test that segmentation preserves destination PE."""
        segmenter = TaskSegmenter()

        task_size = 2 * MAX_SEGMENT_SIZE
        send_req = SendRequest(
            task_id=1, src_tensor=None, src_pos=0, size=task_size, dest_pe=42
        )  # Non-standard PE

        send_segments = segmenter.segment_send_request(send_req)

        # All segments should have same destination
        for segment in send_segments:
            assert segment.dest_pe == 42

    def test_segment_position_offset(self):
        """Test that segments have correct position offsets."""
        segmenter = TaskSegmenter()

        task_size = int(2.5 * MAX_SEGMENT_SIZE)
        start_pos = 1000
        send_req = SendRequest(
            task_id=1, src_tensor=None, src_pos=start_pos, size=task_size, dest_pe=1
        )

        send_segments = segmenter.segment_send_request(send_req)

        # Check position offsets
        assert send_segments[0].src_pos == start_pos
        assert send_segments[1].src_pos == start_pos + MAX_SEGMENT_SIZE
        assert send_segments[2].src_pos == start_pos + 2 * MAX_SEGMENT_SIZE

    def test_segment_task_id_encoding(self):
        """Test that segments have encoded task IDs."""
        segmenter = TaskSegmenter()

        task_size = 2 * MAX_SEGMENT_SIZE
        original_task_id = 42
        send_req = SendRequest(
            task_id=original_task_id, src_tensor=None, src_pos=0, size=task_size, dest_pe=1
        )

        send_segments = segmenter.segment_send_request(send_req)

        # Segments should have encoded task IDs (different from original)
        # Based on the encoding: REQUEST_ID_BASE + (task_id * SEGMENT_ID_MULTIPLIER) + segment_index
        assert len(send_segments) == 2
        assert send_segments[0].task_id != original_task_id
        assert send_segments[1].task_id != original_task_id
        # Segment IDs should be different
        assert send_segments[0].task_id != send_segments[1].task_id
