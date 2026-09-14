# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import random

import pytest
import torch

from megatron.core import config
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot


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


class TestIndicesToMultihot:

    def setup_method(self, method):
        # enable experimental feature
        if config.ENABLE_EXPERIMENTAL is False:
            config.ENABLE_EXPERIMENTAL = True

    def teardown_method(self, method):
        # disable experimental feature
        if config.ENABLE_EXPERIMENTAL is True:
            config.ENABLE_EXPERIMENTAL = False

    @pytest.mark.experimental
    @pytest.mark.parametrize("num_of_token", [3, 5, 8, 128, 512])
    @pytest.mark.parametrize("topk", [2, 4, 6, 7, 8])
    @pytest.mark.parametrize("num_of_local_experts", [4, 7, 8, 12, 20, 30, 31, 32])
    def test_indices_to_multihot(self, num_of_token, topk, num_of_local_experts):
        # construct the indices and probs_indices
        indices = torch.full((num_of_token, topk), -1, dtype=torch.int32, device='cuda')
        probs_indices = torch.full((num_of_token, topk), 0, dtype=torch.float32, device='cuda')
        # Fill the indices with random values
        # There are 2 non-ordinary values in each row
        for i in range(num_of_token):
            positions = random.sample(range(indices.shape[1]), 2)
            values = random.sample(range(num_of_local_experts), 2)
            indices[i, positions[0]] = values[0]
            indices[i, positions[1]] = values[1]
        mask = indices != -1
        probs_indices[mask] = torch.rand(mask.sum(), device=indices.device)
        probs_indices.requires_grad = True

        indices_pytorch = copy.deepcopy(indices)
        probs_indices_pytorch = copy.deepcopy(probs_indices)

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
