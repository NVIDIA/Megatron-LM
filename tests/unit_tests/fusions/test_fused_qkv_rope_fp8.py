# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

try:
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.attention.rope import (
        RotaryPositionEmbedding,
        apply_fused_qkv_rotary_pos_emb,
    )

    from megatron.core.fusions.fused_qkv_rope_fp8 import HAVE_TRITON, fused_qkv_rope_fp8_quantize

    HAVE_DEPS = bool(HAVE_TRITON) and te.is_fp8_available() and torch.cuda.is_available()
except ImportError:
    HAVE_DEPS = False

pytestmark = pytest.mark.skipif(not HAVE_DEPS, reason="needs TE with FP8, Triton and a GPU")

HEAD_DIM = 128


def _quantizer(pow2=False, eps=0.0):
    return te.Float8CurrentScalingQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        device=torch.device("cuda"),
        rowwise=True,
        columnwise=False,
        force_pow_2_scales=pow2,
        amax_epsilon=eps,
    )


@pytest.mark.parametrize("seq_length", [128, 1024, 2048 + 8])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("groups, q_per_group", [(8, 4), (4, 1), (2, 8)])
@pytest.mark.parametrize("pow2", [False, True])
def test_matches_te_rope_then_quantize(seq_length, batch_size, groups, q_per_group, pow2):
    split = [q_per_group * HEAD_DIM, HEAD_DIM, HEAD_DIM]
    torch.manual_seed(0)
    shape = (seq_length, batch_size, groups, sum(split))
    qkv = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 4
    emb = RotaryPositionEmbedding(HEAD_DIM, 1.0)(seq_length + 3)

    q_ref, k_ref, v_ref = apply_fused_qkv_rotary_pos_emb(qkv, emb, emb, split)
    quantizer = _quantizer(pow2)
    ref = quantizer(torch.cat([t.reshape(-1) for t in (q_ref, k_ref, v_ref)]))

    q, k, v = fused_qkv_rope_fp8_quantize(qkv, emb, emb, split, quantizer)
    assert q.shape == q_ref.shape and k.shape == k_ref.shape and v.shape == v_ref.shape
    assert q.dtype == torch.bfloat16
    torch.testing.assert_close(q._scale_inv, ref._scale_inv, rtol=0, atol=0)
    assert k._scale_inv is q._scale_inv and v._scale_inv is q._scale_inv
    out = torch.cat([t._data.reshape(-1) for t in (q, k, v)])
    assert torch.equal(out.view(torch.uint8), ref._data.view(torch.uint8))


def test_backward_matches_te_rope():
    seq_length, batch_size, groups, q_per_group = 256, 2, 8, 4
    split = [q_per_group * HEAD_DIM, HEAD_DIM, HEAD_DIM]
    torch.manual_seed(0)
    qkv = torch.randn(
        seq_length, batch_size, groups, sum(split), dtype=torch.bfloat16, device="cuda"
    ).requires_grad_()
    emb = RotaryPositionEmbedding(HEAD_DIM, 1.0)(seq_length)

    ref = apply_fused_qkv_rotary_pos_emb(qkv, emb, emb, split)
    grads = [torch.randn_like(t) for t in ref]
    (grad_ref,) = torch.autograd.grad(ref, qkv, grads)

    out = fused_qkv_rope_fp8_quantize(qkv, emb, emb, split, _quantizer())
    (grad,) = torch.autograd.grad(out, qkv, grads)
    torch.testing.assert_close(grad, grad_ref, rtol=0, atol=0)


def test_unsupported_inputs_return_none():
    split = [4 * 64, 64, 64]  # head dim != 128
    qkv = torch.randn(16, 2, 8, sum(split), dtype=torch.bfloat16, device="cuda")
    emb = RotaryPositionEmbedding(64, 1.0)(16)
    assert fused_qkv_rope_fp8_quantize(qkv, emb, emb, split, _quantizer()) is None

    split = [4 * HEAD_DIM, HEAD_DIM, HEAD_DIM]
    qkv = torch.randn(16, 2, 8, sum(split), dtype=torch.float16, device="cuda")
    emb = RotaryPositionEmbedding(HEAD_DIM, 1.0)(16)
    assert fused_qkv_rope_fp8_quantize(qkv, emb, emb, split, _quantizer()) is None
