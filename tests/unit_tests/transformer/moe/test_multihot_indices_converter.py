# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy

import pytest
import torch

from megatron.core import config
from megatron.core.device_utils import get_current_device
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
from tests.unit_tests.test_utilities import Utils


class PytorchIndicesToMultihot:
    def __init__(self, num_instances):
        self.num_instances = num_instances

    def _indices_to_multihot(self, indices, probs):
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_instances), dtype=torch.long, device=indices.device
        )
        multihot_probs = torch.zeros(
            (batch_size, self.num_instances), dtype=torch.float, device=indices.device
        )
        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
            mask.sum(dim=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs


# Unit test
@pytest.mark.experimental
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_indices_to_multihot():
    Utils.initialize_model_parallel()
    config.ENABLE_EXPERIMENTAL = True
    indices = torch.tensor(
        [
            [-1, -1, -1, -1, -1, 18, -1, -1],
            [7, -1, -1, -1, -1, -1, 3, -1],
            [-1, -1, -1, -1, -1, 3, 12, -1],
            [-1, -1, 25, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device=get_current_device(),
    )
    probs_indices = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0654, 0.0000, 0.0000],
            [0.1621, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0884, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0640, 0.0449, 0.0000],
            [0.0000, 0.0000, 0.1309, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        ],
        dtype=torch.float32,
        device=get_current_device(),
        requires_grad=True,
    )
    indices_pytorch = copy.deepcopy(indices)
    probs_indices_pytorch = copy.deepcopy(probs_indices)
    num_of_local_experts = 32

    # test forward
    multihot_indices, probs_in_multihot = fused_indices_to_multihot(
        indices, probs_indices, num_of_local_experts
    )
    pytorch_class = PytorchIndicesToMultihot(num_of_local_experts)
    multihot_indices_pytorch, probs_in_multihot_pytorch = pytorch_class._indices_to_multihot(
        indices_pytorch, probs_indices_pytorch
    )
    assert torch.allclose(multihot_indices, multihot_indices_pytorch)
    assert torch.allclose(probs_in_multihot, probs_in_multihot_pytorch)

    # test backward
    loss = (probs_in_multihot @ torch.transpose(probs_in_multihot, 0, 1)).sum() / 2
    loss.backward()
    loss_pytorch = (
        probs_in_multihot_pytorch @ torch.transpose(probs_in_multihot_pytorch, 0, 1)
    ).sum() / 2
    loss_pytorch.backward()
    assert torch.allclose(probs_indices.grad, probs_indices_pytorch.grad)
    Utils.destroy_model_parallel()
