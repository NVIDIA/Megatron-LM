# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Real-NCCL tests for chunk-aligned recurrent GDN CP."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

pytest.importorskip("fla.ops.gated_delta_rule.chunk_fwd")

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from megatron.core.ssm.context_parallel.gdn_recurrent import gdn_recurrent_context_parallel
from tests.unit_tests.ssm.test_gdn_chunkwise_cp import (
    _assert_close,
    _config,
    _gdn,
    _model_groups,
    distributed,
    singleton_group,
)


def _inputs(seq, batch, dim):
    torch.manual_seed(8721)
    shape = (batch, seq, 4, dim)
    q = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(batch, seq, 4, device="cuda") * 0.04
    beta = torch.rand_like(g)
    return [x.detach().requires_grad_() for x in (q, k, v, g, beta)]


def _agree(ok, label):
    status = torch.tensor(int(ok), device="cuda")
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    assert status.item(), label


@pytest.mark.parametrize(
    "seq,batch,dim",
    [
        (32, 1, 32),
        (64, 1, 128),
        (128, 2, 32),
        (248, 1, 128),
        (256, 1, 128),
        (264, 2, 128),
        (512, 1, 128),
        (520, 1, 128),
        (2048, 1, 128),
    ],
)
@pytest.mark.parametrize("terminal_only", [False, True])
def test_core_against_native(seq, batch, dim, terminal_only):
    group = dist.group.WORLD
    length = seq // group.size()
    sl = slice(group.rank() * length, (group.rank() + 1) * length)
    full = _inputs(seq, batch, dim)
    dy = torch.randn_like(full[2]) / full[2].numel()
    if terminal_only:
        dy[:, :-1] = 0
    ref = chunk_gated_delta_rule(*full, use_qk_l2norm_in_kernel=False)[0]
    ref.backward(dy)
    local = [x[:, sl].detach().contiguous().requires_grad_() for x in full]
    y = gdn_recurrent_context_parallel(*local, scale=dim**-0.5, cp_group=group)
    y.backward(dy[:, sl].contiguous())
    _agree(torch.equal(y, ref[:, sl]), "core output must match native FLA exactly")
    for a, b in zip(local, full):
        _agree(torch.equal(a.grad, b.grad[:, sl]), "core gradient must match native FLA exactly")


@pytest.mark.parametrize("seq,recompute", [(64, False), (264, False), (512, True)])
def test_real_module(singleton_group, seq, recompute):
    from megatron.core import tensor_parallel
    from megatron.core.process_groups_config import ProcessGroupCollection

    with _model_groups() as pg:
        ref = _gdn(_config(1), ProcessGroupCollection(tp=pg.tp, cp=singleton_group))
        model = _gdn(_config(pg.cp.size(), gdn_chunkwise_cp_state_mode="recurrent"), pg)
        model.load_state_dict(copy.deepcopy(ref.state_dict()))
        torch.manual_seed(111)
        x = torch.randn(seq, 1, 128, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x) / x.numel()
        xr = x.clone().requires_grad_()
        yr, _ = ref(xr, attention_mask=None)
        yr.backward(dy)
        length = seq // pg.cp.size()
        sl = slice(pg.cp.rank() * length, (pg.cp.rank() + 1) * length)
        xx = x[sl].clone().requires_grad_()
        if recompute:
            yy = tensor_parallel.checkpoint(lambda a: model(a, attention_mask=None)[0], False, xx)
        else:
            yy, _ = model(xx, attention_mask=None)
        yy.backward(dy[sl])
        # Projection GEMM shapes can differ between CP1 and CPn; core equality
        # is checked independently above, full module uses established tolerances.
        _assert_close(yy, yr[sl], "recurrent layer output")
        _assert_close(xx.grad, xr.grad[sl], "recurrent layer dx")
        for (name, p), (rn, rp) in zip(model.named_parameters(), ref.named_parameters()):
            assert name == rn
            grad = p.grad.clone()
            dist.all_reduce(grad, group=pg.cp)
            _assert_close(grad, rp.grad, "recurrent layer " + name)
        with pytest.raises(ValueError, match="unpacked"):
            model._forward_chunkwise_cp(xx, SimpleNamespace(), None)


def test_config_guard():
    with pytest.raises(ValueError, match="state_mode"):
        _config(4, gdn_chunkwise_cp_state_mode="unknown")


def test_long_memory_terminal_gradient():
    """First-rank gradients remain nonzero for a loss only on the final token."""
    group = dist.group.WORLD
    seq, dim = 2048, 128
    full = _inputs(seq, 1, dim)
    with torch.no_grad():
        full[3].fill_(-1e-5)
        full[4].mul_(0.1)
    dy = torch.zeros_like(full[2])
    dy[:, -1] = torch.randn_like(dy[:, -1]) / dy.numel()
    reference = chunk_gated_delta_rule(*full, use_qk_l2norm_in_kernel=False)[0]
    reference.backward(dy)
    _agree(bool(full[2].grad[:, :64].abs().max() > 0), "long-range reference gradient vanished")
    length = seq // group.size()
    sl = slice(group.rank() * length, (group.rank() + 1) * length)
    local = [x[:, sl].detach().contiguous().requires_grad_() for x in full]
    output = gdn_recurrent_context_parallel(*local, scale=dim**-0.5, cp_group=group)
    output.backward(dy[:, sl].contiguous())
    _agree(torch.equal(output, reference[:, sl]), "long-memory output differs")
    for a, b in zip(local, full):
        _agree(torch.equal(a.grad, b.grad[:, sl]), "long-memory cross-rank gradient differs")
