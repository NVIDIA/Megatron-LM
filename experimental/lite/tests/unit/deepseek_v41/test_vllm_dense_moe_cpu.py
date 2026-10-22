# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""#260 MegaMoE backward math (the A/B alternative to the grouped VJP) on CPU."""
import torch

from megatron.lite.model.deepseek_v41.vllm import moe


def test_mlp_grads_match_autograd_through_rounded_operands():
    torch.manual_seed(81)
    a = torch.randn(5, 64).bfloat16().float()
    w1, w2 = torch.randn(64, 64).bfloat16(), torch.randn(64, 32).bfloat16()
    score, grad, limit = torch.rand(5), torch.randn(5, 64), 3.0
    da, ds, dw1, dw2 = moe._mlp_grads(a, score, grad, w1, w2, limit, (True,) * 4)

    leaves = [t.clone().requires_grad_() for t in (a, score, w1.float(), w2.float())]
    x, s, u, v = leaves
    h = (x.bfloat16().float() @ u.T).bfloat16().float()
    h = h + (x @ u.T - (x @ u.T).detach())  # BF16 rounding of the GEMM1 output: identity STE
    gate, up = h.chunk(2, -1)
    act = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit) * s[:, None]
    act = act + (moe._requant_l2(act) - act).detach()
    expected = torch.autograd.grad(act @ v.T, leaves, grad.bfloat16().float())
    for actual, reference in zip((da, ds, dw1, dw2), expected):
        cosine = torch.nn.functional.cosine_similarity(
            actual.double().flatten(), reference.double().flatten(), dim=0
        )
        assert cosine > 0.999


def test_requant_l2_is_a_power_of_two_e4m3_grid():
    torch.manual_seed(82)
    h = torch.randn(3, 64) * 10
    q = moe._requant_l2(h)
    groups = h.reshape(3, 2, 32)
    scale = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1, keepdim=True) / 448)))
    expected = ((groups / scale).to(torch.float8_e4m3fn).float() * scale).reshape(h.shape)
    torch.testing.assert_close(q, expected, rtol=0, atol=0)


def test_grouped_routed_vjp_matches_individual_reference_with_empty_expert(monkeypatch):
    counts = (3, 0, 5)
    ends = torch.tensor(counts, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    torch.manual_seed(87)
    a, score, grad = torch.randn(8, 64), torch.rand(8), torch.randn(8, 64)
    w1, w2 = torch.randn(3, 64, 64).bfloat16(), torch.randn(3, 64, 32).bfloat16()
    calls = []

    def grouped_cpu(left, right, splits, out_dtype=None):
        # Test oracle only; the production helper has one grouped CUDA call.
        assert not left.is_cuda and tuple(splits) == counts
        calls.append((tuple(left.shape), tuple(right.shape), out_dtype))
        left, right = left.bfloat16(), right.bfloat16()
        if right.ndim == 3:
            value = torch.cat([x.float() @ w.float() for x, w in zip(left.split(counts), right)])
        else:
            value = torch.stack(
                [x.float() @ y.float() for x, y in zip(left.split(counts, -1), right.split(counts))]
            )
        return value.to(out_dtype or left.dtype)

    with monkeypatch.context() as patcher:
        patcher.setattr(moe, '_native_grouped_mm', grouped_cpu)
        actual = moe._grouped_mlp_grads(a, score, grad, w1, w2, ends, 3.0, (True,) * 4)
    assert len(calls) == 5
    expected = [
        moe._mlp_grads(x, s, g, u, v, 3.0, (True,) * 4)
        for x, s, g, u, v in zip(a.split(counts), score.split(counts), grad.split(counts), w1, w2)
    ]
    for index, result in enumerate(actual):
        reference = (
            torch.cat([x[index] for x in expected])
            if index < 2
            else torch.stack([x[index] for x in expected])
        )
        torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-5)
    decoded = moe._mx(w1)
    assert torch.equal(
        decoded.view(torch.uint8), torch.stack([moe._mx(w) for w in w1]).view(torch.uint8)
    )


def test_batched_expert_grad_accumulation_and_fixed_slot_sum(monkeypatch):
    import sys
    from types import SimpleNamespace

    # Test-side placeholder only: production DDP provider requires TE.
    monkeypatch.setitem(
        sys.modules,
        "megatron.lite.primitive.parallel.linear",
        SimpleNamespace(_dummy_wgrad=lambda w: torch.empty_like(w)),
    )
    owners = [torch.nn.Parameter(torch.zeros(2, 3)) for _ in range(3)]
    for owner in owners:
        owner.main_grad = torch.zeros_like(owner)
    dw = torch.arange(18).reshape(3, 2, 3).float()
    for _ in range(2):
        result = moe._expert_weight_grads(owners, dw, (True, False, True))
    assert result[1] is None
    assert torch.equal(owners[0].main_grad, dw[0] * 2) and torch.equal(
        owners[2].main_grad, dw[2] * 2
    )
    assert not owners[1].main_grad.any()
    assert owners[0].grad_added_to_main_grad and owners[2].grad_added_to_main_grad
    position = torch.tensor([7, 0, 3, 5])
    gradients = torch.arange(16).reshape(4, 4).float()
    value = moe._sum_routes(gradients, position, 3, 3)
    expected = torch.stack((gradients[1], gradients[2] + gradients[3], gradients[0]))
    assert torch.equal(value.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(
        value.view(torch.int32), moe._sum_routes(gradients, position, 3, 3).view(torch.int32)
    )


def test_native_grouped_backend_uses_te_layouts_once(monkeypatch):
    import sys
    from types import SimpleNamespace

    counts, calls = (3, 0, 5), []

    def te_gemm(a, b, out, quantization_params, dtype, **kw):
        assert quantization_params == [None] * 3 and kw['grad']
        calls.append(kw['layout'])
        # CPU oracle only: native TE submits this list in one grouped invocation.
        values = []
        for x, y in zip(a, b):
            z = (
                y.float() @ x.float().T
                if kw['layout'] == 'TN'
                else (y.float() @ x.float() if kw['layout'] == 'NN' else y.float().T @ x.float())
            )
            values.append(z.to(dtype))
        if kw['single_output']:
            out[0].copy_(torch.cat(values))
        else:
            for dst, value in zip(out, values):
                dst.copy_(value)

    monkeypatch.setitem(
        sys.modules,
        'transformer_engine.pytorch.cpp_extensions',
        SimpleNamespace(general_grouped_gemm=te_gemm),
    )
    x, w = torch.randn(8, 32).bfloat16(), torch.randn(3, 64, 32).bfloat16()
    y = moe._native_grouped_mm(x, w.transpose(-1, -2), counts)
    expected = torch.cat([v.float() @ u.float().T for v, u in zip(x.split(counts), w)])
    torch.testing.assert_close(y, expected.bfloat16(), rtol=0, atol=0)
    dx = moe._native_grouped_mm(y, w, counts, torch.float32)
    dw = moe._native_grouped_mm(y.T, x, counts, torch.float32)
    torch.testing.assert_close(
        dx, torch.cat([v.float() @ u.float() for v, u in zip(y.split(counts), w)])
    )
    torch.testing.assert_close(
        dw, torch.stack([v.float().T @ u.float() for v, u in zip(y.split(counts), x.split(counts))])
    )
    assert calls == ['TN', 'NN', 'NT']


def test_expert_pack_batches_quantization_and_reads_updated_masters(monkeypatch):
    from megatron.lite.model.deepseek_v41.vllm import experts
    from megatron.lite.primitive.quantization import mxfp4

    torch.manual_seed(91)
    fc1 = [torch.randn(256, 128) for _ in range(3)]
    fc2 = [torch.randn(128, 128) for _ in range(3)]
    shared = (torch.randn(256, 128), torch.randn(128, 128))
    quantize, writes, calls = mxfp4.quantize_mxfp4, [], []
    write = experts._write_experts

    def quantized(value):
        calls.append(tuple(value.shape))
        return quantize(value)

    def checked(dst, src, shard, is_scale):
        sources = fc2 if shard == 'w2' else [w[:128] if shard == 'w1' else w[128:] for w in fc1]
        reference = torch.stack([quantize(w)[int(is_scale)] for w in sources])
        assert torch.equal(src.view(torch.uint8), reference.view(torch.uint8))
        # Check stacked layout against the same single-expert layout independently.
        expected = dst.clone()
        for e in range(len(sources)):
            write(expected[e : e + 1], reference[e : e + 1], shard, is_scale)
        write(dst, src, shard, is_scale)
        assert torch.equal(
            dst, expected
        )  # packed outputs are int8/int32, so equality is exact bytes
        writes.append((shard, is_scale))

    monkeypatch.setattr(mxfp4, 'quantize_mxfp4', quantized)
    monkeypatch.setattr(experts, '_write_experts', checked)
    # Shared-expert layout is unchanged and separately covered by the real CUDA gate.
    monkeypatch.setattr(experts, '_write', lambda *args: None)
    first = experts.kernel_weights(fc1, fc2, *shared)
    assert calls == [(3, 128, 128)] * 3 and len(writes) == 6
    fc1[0].data.mul_(-2)  # .data writes need not change Tensor._version.
    second = experts.kernel_weights(fc1, fc2, *shared)
    assert len(calls) == 6 and len(writes) == 12
    assert not torch.equal(first[0][0], second[0][0])
