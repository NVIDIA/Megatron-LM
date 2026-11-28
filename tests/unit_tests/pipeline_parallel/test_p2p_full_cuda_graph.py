# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Batched pipeline communication retains eager fences and supports graph replay."""

from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from tests.unit_tests.test_utilities import Utils


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("capturing", [False, True])
def test_batched_p2p_waits_requests_and_only_synchronizes_eagerly(capturing):
    """Skipping the capture-illegal device fence must retain the request waits."""
    communicator = P2PCommunicator.__new__(P2PCommunicator)
    communicator.config = ModelParallelConfig(batch_p2p_comm=True, batch_p2p_sync=True)
    communicator.pp_group = Mock()
    communicator.next_rank = 1
    communicator.prev_rank = 1
    calls = []
    request = Mock()
    request.wait.side_effect = lambda: calls.append("wait")
    with (
        patch(
            "megatron.core.pipeline_parallel.p2p_communication._batched_p2p_ops",
            return_value=[request],
        ),
        patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
        patch("torch.cuda.synchronize", side_effect=lambda: calls.append("synchronize")),
    ):
        _, _, requests = communicator._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=(8, 1, 32),
        )
    assert requests is None
    assert calls == (["wait"] if capturing else ["wait", "synchronize"])


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(Utils.world_size < 2, reason="Requires at least two GPUs")
def test_batched_p2p_full_cuda_graph_replays_changed_inputs():
    """Capture real NCCL sends/receives with the default eager race-condition fence enabled."""
    Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
    try:
        group = parallel_state.get_pipeline_model_parallel_group()
        config = ModelParallelConfig(
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            batch_p2p_comm=True,
            batch_p2p_sync=True,
        )
        communicator = P2PCommunicator(pp_group=group, config=config)
        inputs = torch.full((8, 1, 32), float(group.rank()), device="cuda")

        def exchange():
            received, _, _ = communicator._communicate(
                tensor_send_next=inputs,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=inputs.shape,
            )
            return received.square()

        for _ in range(3):
            exchange()
        torch.distributed.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = exchange()
        for iteration in range(4):
            inputs.fill_(group.rank() + iteration + 2)
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.full_like(output, float((1 - group.rank() + iteration + 2) ** 2))
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
