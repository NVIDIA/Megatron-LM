# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize('reduce,average', [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize('use_graph', [False, True], ids=['eager', 'graph'])
def test_mtp_logging_reduces_after_cleanup(monkeypatch, reduce, average, use_graph):
    """Logging must reduce metrics after cleanup in eager execution and graph replay."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    monkeypatch.setattr(MTPLossLoggingHelper, 'tracker', {})
    group = torch.distributed.group.WORLD
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert world_size > 1, 'This regression requires different metric values across ranks'
    loss = torch.tensor(float(rank + 1), device='cuda')
    correct = torch.tensor(float(rank + 1), device='cuda')
    total = torch.tensor(float(rank + 2), device='cuda')
    reduce_group = group if reduce else None
    avg_group = group if average else None

    def forward():
        MTPLossLoggingHelper.save_metrics_to_tracker(
            loss, correct, total, 0, 1, reduce_group=reduce_group, avg_group=avg_group
        )

    try:
        forward()
        MTPLossLoggingHelper.clean_metrics_in_tracker()
        if use_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            MTPLossLoggingHelper.clean_metrics_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        addresses = {
            key: tracker[key].data_ptr()
            for key in ('loss_values', 'correct_values', 'total_values')
        }
        for iteration in range(1, 4):
            loss.fill_(rank + iteration)
            if use_graph:
                graph.replay()
            else:
                forward()
            logged = {}
            MTPLossLoggingHelper.track_mtp_metrics(1.0, iteration, None, total_loss_dict=logged)
            expected_loss = (world_size - 1) / 2 + iteration
            if reduce:
                expected_loss *= world_size
            torch.testing.assert_close(
                logged['mtp_1 loss'], torch.tensor(expected_loss, device='cuda')
            )
            for key, address in addresses.items():
                assert tracker[key].data_ptr() == address
                torch.testing.assert_close(tracker[key], torch.zeros_like(tracker[key]))
            expected_total = world_size * (world_size + 3) / 2
            if reduce and average:
                expected_total *= world_size
            torch.testing.assert_close(
                tracker['cumulative_total_values'],
                torch.full((1,), expected_total * iteration, device='cuda'),
            )
    finally:
        Utils.destroy_model_parallel()
