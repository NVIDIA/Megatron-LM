# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging
from unittest import mock

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.ssm.gated_delta_net import (
    _split_tensor_factory as gated_delta_split_tensor_factory,
)
from megatron.core.ssm.mamba_mixer import _split_tensor_factory as mamba_split_tensor_factory


@pytest.mark.parametrize(
    "factory_fn", [gated_delta_split_tensor_factory, mamba_split_tensor_factory]
)
@pytest.mark.internal
def test_split_tensor_factory_oom_is_handled(factory_fn, caplog):
    original_sh_ten = ShardedTensor.from_rank_offsets(
        'a', torch.arange(12, dtype=torch.float32).reshape(6, 2), (0, 0, 1)
    )
    factory = factory_fn(original_sh_ten, [2, 4], ['x', 'B'], 0)
    sub_state_dict = [torch.ones((2, 2), dtype=torch.float32), torch.full((4, 2), 2.0)]

    real_cat = torch.cat
    call_count = 0

    def fake_cat(tensors, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise torch.cuda.OutOfMemoryError('mock oom')
        return real_cat(tensors, *args, **kwargs)

    with (
        mock.patch('torch.cat', side_effect=fake_cat),
        mock.patch('gc.collect') as collect_mock,
        mock.patch('torch.cuda.empty_cache') as empty_cache_mock,
        caplog.at_level(logging.WARNING),
    ):
        merged = factory.merge_fn(sub_state_dict)

    assert torch.equal(merged, real_cat(sub_state_dict))
    assert merged.device.type == 'cpu'
    assert call_count == 2
    collect_mock.assert_called_once()
    empty_cache_mock.assert_called_once()
    assert "CUDA OutOfMemoryError encountered during tensors merging" in caplog.text
