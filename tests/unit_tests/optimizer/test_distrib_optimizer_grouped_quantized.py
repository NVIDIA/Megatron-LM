# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


class _FakeGroupedQuantizedTensor:
    def __init__(self, members, quantized_tensors=None):
        self._members = members
        self.quantized_tensors = quantized_tensors
        self.quantizer = object()

    def split_into_quantized_tensors(self):
        return self._members


def test_expand_quantized_param_shard_for_cast_splits_grouped_wrapper():
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    members = [torch.empty(3), torch.empty(5), torch.empty(2)]
    grouped_param = _FakeGroupedQuantizedTensor(members, quantized_tensors=members)
    shard_main_param = torch.arange(6)

    expanded_params, expanded_main_params, expanded_offsets = (
        optimizer._expand_quantized_param_shard_for_cast(
            grouped_param, shard_main_param, start_offset=2
        )
    )

    assert len(expanded_params) == len(members)
    assert all(expanded is member for expanded, member in zip(expanded_params, members))
    torch.testing.assert_close(expanded_main_params[0], torch.tensor([0]))
    torch.testing.assert_close(expanded_main_params[1], torch.tensor([1, 2, 3, 4, 5]))
    assert expanded_main_params[2] is None
    assert expanded_offsets == [2, 0, None]


def test_expand_quantized_param_shard_for_cast_keeps_plain_param_unchanged():
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    model_param = torch.empty(4)
    shard_main_param = torch.arange(4)

    expanded_params, expanded_main_params, expanded_offsets = (
        optimizer._expand_quantized_param_shard_for_cast(
            model_param, shard_main_param, start_offset=1
        )
    )

    assert len(expanded_params) == 1 and expanded_params[0] is model_param
    assert len(expanded_main_params) == 1 and expanded_main_params[0] is shard_main_param
    assert expanded_offsets == [1]


def test_grouped_quantized_tensor_detection_allows_lazy_split_members():
    grouped_param = _FakeGroupedQuantizedTensor([torch.empty(1)], quantized_tensors=None)

    assert DistributedOptimizer._is_grouped_quantized_tensor(grouped_param)
    assert DistributedOptimizer._is_distopt_quantized_param(grouped_param)
