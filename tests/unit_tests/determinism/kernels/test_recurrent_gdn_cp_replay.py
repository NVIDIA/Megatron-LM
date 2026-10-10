# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Bit-exact replay of the recurrent CP core and its complete backward."""

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("fla.ops.gated_delta_rule.chunk_fwd")

from megatron.core.ssm.context_parallel.gdn_recurrent import gdn_recurrent_context_parallel
from tests.unit_tests.ssm.test_gdn_recurrent_cp import _agree, _inputs, distributed


@pytest.mark.parametrize("seq", [32, 264, 512])
def test_recurrent_cp_replay(seq):
    group = dist.group.WORLD
    full = _inputs(seq, 1, 128)
    length = seq // group.size()
    sl = slice(group.rank() * length, (group.rank() + 1) * length)
    torch.manual_seed(909)
    dy = torch.randn_like(full[2][:, sl]) / full[2].numel()
    snapshots = []
    for _ in range(3):
        inputs = [x[:, sl].detach().contiguous().requires_grad_() for x in full]
        y = gdn_recurrent_context_parallel(*inputs, scale=128**-0.5, cp_group=group)
        y.backward(dy)
        snapshots.append([y.detach().clone()] + [x.grad.clone() for x in inputs])
    for tensors in snapshots[1:]:
        for a, b in zip(tensors, snapshots[0]):
            _agree(torch.equal(a, b), "recurrent CP replay changed bits")


@pytest.mark.parametrize("width", [1, 2, 4])
def test_recurrent_module_forward_replay(width):
    from tests.unit_tests.ssm.test_gdn_chunkwise_cp import _config, _gdn, _model_groups

    with _model_groups() as pg:
        cfg = _config(
            pg.cp.size(), gdn_chunkwise_cp_state_mode="recurrent", linear_conv_kernel_dim=width
        )
        model = _gdn(cfg, pg)
        torch.manual_seed(144)
        x = torch.randn(16, 1, 128, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            values = [model(x, attention_mask=None)[0].clone() for _ in range(3)]
        for other in values[1:]:
            _agree(torch.equal(values[0], other), "GDN dispatch/conv forward replay changed bits")
