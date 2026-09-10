# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from math import prod
from types import SimpleNamespace

import pytest
import torch

from megatron.training.utils import common_utils


class _FakeTensor:
    """Minimal CUDA-tensor stand-in for batch-broadcast control-flow tests."""

    def __init__(self, shape, dtype=None, value=None):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.value = value

    def cuda(self, non_blocking=False):
        return self

    def fill_(self, value):
        self.value = value
        return self

    def item(self):
        return self.value

    def numel(self):
        return prod(self.shape) if self.shape else 1


@pytest.fixture
def batch_broadcast_env(monkeypatch):
    args = SimpleNamespace(
        create_attention_mask_in_dataloader=False,
        cuda_graph_impl=None,
        dynamic_context_parallel=False,
        micro_batch_size=1,
        pipeline_model_parallel_size=1,
        seq_length=4,
        sft=False,
    )
    monkeypatch.setattr(common_utils, "get_args", lambda: args)
    monkeypatch.setattr(common_utils.mpu, "get_tensor_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(common_utils.mpu, "get_tensor_model_parallel_group", lambda: object())
    monkeypatch.setattr(common_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        common_utils.torch,
        "empty",
        lambda shape, dtype=None, device=None: _FakeTensor(shape, dtype=dtype),
    )

    broadcasts = []

    def broadcast(tensor, src, group):
        broadcasts.append(tensor)
        if tensor.shape == () and tensor.dtype == torch.int64:
            tensor.value = 0

    monkeypatch.setattr(common_utils.torch.distributed, "broadcast", broadcast)
    return broadcasts


def test_source_skips_empty_cu_seqlens_broadcast(monkeypatch, batch_broadcast_env):
    monkeypatch.setattr(common_utils.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    data = {
        "tokens": _FakeTensor((1, 4)),
        "labels": _FakeTensor((1, 4)),
        "loss_mask": _FakeTensor((1, 4)),
        "position_ids": _FakeTensor((1, 4)),
    }

    batch = common_utils.get_batch_on_this_tp_rank(iter([data]))

    assert batch["cu_seqlens"] is None
    assert all(tensor.numel() > 0 for tensor in batch_broadcast_env)


def test_receiver_skips_empty_cu_seqlens_broadcast(monkeypatch, batch_broadcast_env):
    monkeypatch.setattr(common_utils.mpu, "get_tensor_model_parallel_rank", lambda: 1)

    batch = common_utils.get_batch_on_this_tp_rank(None)

    assert batch["cu_seqlens"] is None
    assert all(tensor.numel() > 0 for tensor in batch_broadcast_env)
