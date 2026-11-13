# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The grouped projection boundary accepts packed modules without dense weights."""

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _build_attention,
    _make_config,
)


class GroupSlice(torch.nn.Module):
    def __init__(self, output_width):
        super().__init__()
        self.output_width = output_width
        self.gain = torch.nn.Parameter(torch.tensor(2.0, device="cuda", dtype=torch.bfloat16))

    def forward(self, hidden):
        return hidden[..., : self.output_width] * self.gain


def test_native_and_supplied_group_projection():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(42)
        cfg = _make_config(csa_compress_ratios=[0, 0, 0, 0])
        attn = _build_attention(
            cfg, layer_number=1, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        groups = cfg.output_projection_groups
        rank = cfg.output_projection_lora_rank
        width = cfg.v_head_dim * cfg.num_attention_heads // groups
        x = torch.randn(
            3, 1, groups, width, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        expected = torch.einsum(
            "...gd,grd->...gr", x, attn.linear_o_group_proj.view(groups, rank, width)
        )
        actual = attn._apply_grouped_output_projection(x)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        attn.group_output_projection = GroupSlice(rank)
        # A supplied projection must not read or allocate the dense grouped weight.
        attn.linear_o_group_proj = torch.nn.Parameter(
            torch.empty(0, device="cuda"), requires_grad=False
        )
        actual = attn._apply_grouped_output_projection(x)
        assert torch.equal(actual, x[..., :rank] * 2)
        actual.float().sum().backward()
        assert x.grad is not None and torch.equal(
            x.grad[..., :rank], torch.full_like(x.grad[..., :rank], 2)
        )
        assert attn.group_output_projection.gain.grad is not None
    finally:
        Utils.destroy_model_parallel()
