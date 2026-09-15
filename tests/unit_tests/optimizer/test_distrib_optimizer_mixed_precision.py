# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.optimizer.optimizer_config import OptimizerConfig


@pytest.mark.internal
@pytest.mark.parametrize('low_precision_dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('fp32_buffer_first', [False, True])
def test_mixed_precision_checkpoint_state_roundtrip(low_precision_dtype, fp32_buffer_first):
    """Checkpoint each mixed-dtype parameter's own DP-local shard and Adam states."""
    parameters = [
        torch.nn.Parameter(torch.full((size,), value, dtype=dtype, device='cuda'))
        for size, value, dtype in [
            (9, 1.0, low_precision_dtype),
            (5, 2.0, torch.float32),
            (7, 3.0, low_precision_dtype),
            (3, 4.0, torch.float32),
        ]
    ]
    groups = [{'params': parameters[:2]}, {'params': parameters[2:]}]
    dtypes = [torch.float32, low_precision_dtype]
    if not fp32_buffer_first:
        dtypes.reverse()
    ranges = []
    for dtype in dtypes:
        param_map = {}
        offset = 0
        for parameter in parameters:
            if parameter.dtype != dtype:
                continue
            # A rank can own only part of a parameter; use distinct shard sizes.
            shard = Range(1, parameter.numel())
            param_map[parameter] = {
                'param': shard,
                'gbuf_local': Range(offset, offset + shard.size),
            }
            offset += shard.size
        ranges.append({(dtype, torch.float32): [{'param_map': param_map}]})

    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.config = OptimizerConfig()
    optimizer.gbuf_ranges = ranges
    optimizer.per_bucket_numel = []
    optimizer.per_bucket_numel_unpadded = []
    optimizer.model_param_group_index_map, group_ranges = (
        optimizer._build_optimizer_group_ranges(groups, ranges)
    )
    optimizer._build_model_and_main_param_groups(
        ranges, optimizer._build_model_param_gbuf_map(ranges), group_ranges, optimizer.config
    )
    optimizer.optimizer = torch.optim.Adam([group['orig_group'] for group in group_ranges])
    for group in optimizer.optimizer.param_groups:
        for parameter in group['params']:
            optimizer.optimizer.state[parameter] = {
                'step': torch.tensor(1.0),
                'exp_avg': parameter.detach().clone() + 10,
                'exp_avg_sq': parameter.detach().clone() + 20,
            }

    saved = {}
    for parameter in parameters:
        state = optimizer._get_main_param_and_optimizer_states(parameter)
        expected = parameter.detach()[1:].float().clone()
        torch.testing.assert_close(state['param'], expected)
        torch.testing.assert_close(state['exp_avg'], expected + 10)
        torch.testing.assert_close(state['exp_avg_sq'], expected + 20)
        saved[parameter] = {key: value.clone() for key, value in state.items()}

    # Restore through the same mapping used by checkpoint loading, then verify
    # actual optimizer tensors; a wrong index must not silently target a peer.
    for group in optimizer.optimizer.param_groups:
        for parameter in group['params']:
            parameter.zero_()
            for value in optimizer.optimizer.state[parameter].values():
                value.zero_()
    for parameter, state in saved.items():
        optimizer._set_main_param_and_optimizer_states(parameter, state)
    checkpoint = optimizer.get_parameter_state_dp_reshardable()
    for buffer_index, buffer in enumerate(ranges):
        for dtype, buckets in buffer.items():
            for parameter, state in zip(
                buckets[0]['param_map'], checkpoint[buffer_index][dtype][0]
            ):
                for key, expected in saved[parameter].items():
                    torch.testing.assert_close(state[key], expected)
