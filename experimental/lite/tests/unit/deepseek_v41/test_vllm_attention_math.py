# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU adjoint check for the owned rotary VJP; no deployment forward claim."""
import torch

from megatron.lite.model.deepseek_v41.vllm.attention import _rotate, _SWAInsert


def test_zero_row_swa_has_empty_finite_gradients_without_importing_vllm(monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == 'vllm' or name.startswith('vllm.'):
            raise AssertionError('Empty SWA must not import or launch vLLM kernels')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)
    q = torch.empty(0, 72, 512, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.empty(0, 512, dtype=torch.bfloat16, requires_grad=True)
    output, decoded = _SWAInsert.apply(
        q, kv, torch.empty(0, dtype=torch.int64), torch.empty(1, 64), 1e-6
    )
    assert output.shape == q.shape and decoded.shape == kv.shape
    assert output.dtype == decoded.dtype == torch.bfloat16
    (output.float().sum() + decoded.float().sum()).backward()
    assert q.grad is not None and q.grad.shape == q.shape and torch.isfinite(q.grad).all()
    assert kv.grad is not None and kv.grad.shape == kv.shape and torch.isfinite(kv.grad).all()


def test_rotary_vjp_is_transpose_for_query_and_shared_kv():
    torch.manual_seed(413)
    positions = torch.tensor([2, 0, 3])
    angles = torch.randn(4, 4)
    cache = torch.cat((angles.cos(), angles.sin()), -1)
    for shape in ((3, 2, 12), (3, 12)):
        x = torch.randn(shape, requires_grad=True)
        grad = torch.randn(shape)
        y = _rotate(x, positions, cache)
        (actual,) = torch.autograd.grad(y, x, grad)
        expected = _rotate(grad, positions, cache, inverse=True)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(y[..., :4], x[..., :4], rtol=0, atol=0)


def test_ratio_two_geometry_and_open_group_adjoint():
    from megatron.lite.model.deepseek_v41.vllm.compressed import metadata, pool_vjp

    m = metadata(147, 2, 'cpu')
    assert m.lengths.tolist() == [73]
    assert m.slots[::2].eq(-1).all()
    assert m.slots[1::2].tolist() == list(range(73))
    assert m.table.tolist() == [[0, 1]]
    raw = torch.arange(5 * 1024, dtype=torch.float32).reshape(5, 1024) / 4096
    weight = torch.ones(512)
    dx, dw = pool_vjp(raw, weight, 2, torch.ones(5, 512), 1e-20)
    assert torch.isfinite(dx).all() and torch.isfinite(dw).all()
    assert dx[-1].eq(0).all() and dx[:-1].count_nonzero()
    assert dw.count_nonzero()
