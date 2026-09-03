# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon


def _make_gated_fc1_param():
    param = torch.nn.Parameter(torch.empty(8, 3))
    param.is_gated_fc1 = True
    param.partition_dim = 0
    param.partition_stride = 2
    param.pad_length = 0
    return param


def test_gated_fc1_orthogonalizes_gate_and_up_independently():
    param = _make_gated_fc1_param()
    grad = torch.arange(24, dtype=torch.float32).view(8, 3)
    optimizer = TensorParallelMuon.__new__(TensorParallelMuon)
    optimizer.pg_collection = None
    optimizer.tp_mode = "duplicated"
    optimizer.is_qkv_fn = lambda _: False

    calls = []

    def fake_orthogonalize(p, local_grad, tp_group, partition_dim):
        del p, tp_group, partition_dim
        calls.append(local_grad.clone())
        return local_grad + 1.0 if len(calls) == 1 else local_grad + 2.0

    optimizer.scaled_orthogonalize_fn_with_gtp_remat = fake_orthogonalize

    result = optimizer.orthogonalize(param, grad)

    assert len(calls) == 2
    assert torch.equal(calls[0], grad[:4])
    assert torch.equal(calls[1], grad[4:])
    assert torch.equal(result, torch.cat((grad[:4] + 1.0, grad[4:] + 2.0), dim=0))
