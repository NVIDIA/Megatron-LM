# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the no-atomic cross-layer teacher accumulator."""

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact


def test_cli_target_accumulation_replays():
    pytest.importorskip("tilelang")
    from megatron.core.ops.attention.dsa.cli_target import accumulate_cli_attention_target

    torch.manual_seed(47)
    target = torch.rand(1, 3, 256, device="cuda")
    indices = torch.arange(256, device="cuda", dtype=torch.int32).expand(1, 3, -1).clone()
    indices[:, 1, 240:] = -1
    indices[:, 2] = -1

    def accumulate(value):
        state = accumulate_cli_attention_target(value, indices, None, None, 2)
        state = accumulate_cli_attention_target(value, indices, None, None, 2, state)
        return state.target_sum, state.teacher_entropy_sum

    assert_replays_bit_exact(
        accumulate,
        (target,),
        backward=False,
        replays=3,
        contention=True,
        what="DSA CLI teacher accumulation",
    )


def test_cli_attention_teacher_replays():
    """The new attention dispatch returns a deterministic teacher, independent of dO."""
    pytest.importorskip("flash_mla")
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    if not dsa_cudnn_kernels.supports_fused_cli_target_in_attention_backward():
        pytest.skip("SM90 and the cuDNN CLI teacher-target extension are required")
    torch.manual_seed(31)
    query = torch.randn(4, 1, 64, 576, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(256, 1, 1, 576, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(256, device="cuda", dtype=torch.int32).expand(1, 4, -1).clone()
    lengths = torch.full((1, 4), 256, device="cuda", dtype=torch.int32)

    def teacher(q, k):
        collected = []
        output = dsa_cudnn_kernels.run_fused_absorbed_sparse_attention(
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            indices,
            576**-0.5,
            512,
            lengths,
            cli_backward_callback=collected.append,
        )
        assert output is not None
        output.float().sum().backward()
        return collected[0]

    assert_replays_bit_exact(
        teacher,
        (query, key),
        backward=False,
        replays=3,
        contention=True,
        what="cuDNN DSA CLI teacher output",
    )
