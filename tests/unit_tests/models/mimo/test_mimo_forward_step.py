# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for MIMO forward-step helpers."""

from __future__ import annotations

import pytest
import torch

from examples.mimo.training.step import loss_func, move_batch_to_cuda
from megatron.core.packed_seq_params import PackedSeqParams


def test_loss_func_returns_int_num_tokens_three_tuple():
    output = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

    loss_sum, num_tokens, loss_dict = loss_func(output, loss_mask=loss_mask)

    assert isinstance(num_tokens, torch.Tensor)
    assert not num_tokens.is_floating_point()
    assert num_tokens.dtype in (torch.int32, torch.int64, torch.int16)
    assert int(num_tokens.item()) == 3

    assert isinstance(loss_sum, torch.Tensor)
    assert loss_sum.shape == torch.Size([])
    assert torch.allclose(loss_sum, torch.tensor(1.0 + 2.0 + 4.0))

    assert set(loss_dict.keys()) == {"lm loss"}
    logged = loss_dict["lm loss"]
    assert logged.shape == torch.Size([2])
    assert torch.allclose(logged[0], loss_sum.detach())
    assert torch.allclose(logged[1], num_tokens.detach().float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_move_batch_to_cuda_recurses_dict_list_tuple():
    t_top = torch.tensor([1.0])
    t_in_list = torch.tensor([2.0])
    t_in_tuple = torch.tensor([3.0])
    t_nested = torch.tensor([4.0])

    batch = {
        "input_ids": t_top,
        "a_list": [t_in_list, "not a tensor", 7],
        "a_tuple": (t_in_tuple,),
        "nested": {"deep": t_nested},
        "scalar": 5,
    }

    out = move_batch_to_cuda(batch)

    assert isinstance(out, dict)
    assert isinstance(out["a_list"], list)
    assert isinstance(out["a_tuple"], tuple)
    assert out["scalar"] == 5
    assert out["a_list"][1] == "not a tensor"
    assert out["input_ids"].is_cuda
    assert out["a_list"][0].is_cuda
    assert out["a_tuple"][0].is_cuda
    assert out["nested"]["deep"].is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_move_batch_to_cuda_handles_packed_seq_params():
    cu_q = torch.tensor([0, 4, 8], dtype=torch.int32)
    cu_kv = torch.tensor([0, 4, 8], dtype=torch.int32)
    psp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=8,
        max_seqlen_kv=8,
    )

    batch = {"packing": psp}
    out = move_batch_to_cuda(batch)

    assert out["packing"] is psp
    assert psp.qkv_format == "thd"
    assert psp.max_seqlen_q == 8
    assert psp.cu_seqlens_q.is_cuda
    assert psp.cu_seqlens_kv.is_cuda
