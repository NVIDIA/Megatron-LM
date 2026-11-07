# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-token codec choice, sign tags, and mixed-cache numerical behavior."""

import pytest
import torch

from megatron.training.distillation.reference_logits import (
    adaptive_logprobs,
    coarsen_logits,
    decode_adaptive_logits,
    decode_reference_logits,
    encode_adaptive_logits,
    encode_reference_logits,
    reference_logits_kl_div,
    stable_log_softmax,
    uses_reference_logits,
)


def _kl64(logits, target):
    log_q = torch.log_softmax(logits.double(), -1)
    log_p = torch.log_softmax(target.double(), -1)
    return torch.where(log_p.isfinite(), log_p.exp() * (log_p - log_q), 0).sum(-1)


def test_both_modes_without_more_bytes():
    logits = torch.zeros(2, 64)
    logits[1, 1:] = -30
    indices = logits.topk(2, -1).indices
    keep = torch.ones_like(indices, dtype=torch.bool)
    coarse, _ = coarsen_logits(logits, indices, keep)
    values = encode_adaptive_logits(coarse, keep)
    assert uses_reference_logits(values).tolist() == [False, True]
    assert values.shape == (2, 2) and values.dtype == torch.float16
    assert values.numel() * values.element_size() == 8
    assert torch.isfinite(adaptive_logprobs(values)).all()


@pytest.mark.parametrize("tag_value", [0.0, torch.inf])
def test_tag_preserves_zero_and_masked_gap(tag_value):
    values = torch.tensor([[-2.0, tag_value, -3.0]], dtype=torch.float16)
    assert uses_reference_logits(values).item()
    expected = values.clone()
    expected[..., 1] = -expected[..., 1].abs()
    torch.testing.assert_close(decode_adaptive_logits(values), decode_reference_logits(expected))
    assert torch.signbit(expected[..., 1]).item()


def test_k_one_stays_reference_encoded():
    logits = torch.tensor([[0.0, -30.0]])
    indices = torch.tensor([[0]])
    keep = torch.ones_like(indices, dtype=torch.bool)
    coarse, _ = coarsen_logits(logits, indices, keep)
    adaptive = encode_adaptive_logits(coarse, keep)
    fixed = encode_reference_logits(coarse, keep)
    torch.testing.assert_close(adaptive, fixed, atol=0, rtol=0)
    assert uses_reference_logits(adaptive).all()


@pytest.mark.parametrize("k", [2, 8, 32])
def test_kl_never_exceeds_either_valid_candidate(k):
    torch.manual_seed(804 + k)
    for sigma in (1.0, 4.0, 8.0):
        logits = torch.randn(32, 64) * sigma
        indices = logits.topk(k, -1).indices
        keep = torch.ones_like(indices, dtype=torch.bool)
        target, _ = coarsen_logits(logits, indices, keep)
        fixed = encode_reference_logits(target, keep)
        adaptive = encode_adaptive_logits(target, keep)
        a_kl = _kl64(decode_adaptive_logits(adaptive), target)
        r_kl = _kl64(decode_reference_logits(fixed), target)
        assert torch.all(a_kl <= r_kl + 2e-14)
        probability = stable_log_softmax(target)[..., :-1].half()
        probability[..., 1] = -probability[..., 1].abs()
        valid = probability.float().exp().sum(-1) <= 1
        p_kl = _kl64(decode_adaptive_logits(probability), target)
        assert torch.all(a_kl[valid] <= p_kl[valid] + 2e-14)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dtypes_and_top_p_masks(dtype):
    logits = torch.tensor([[0.0, -1.0, -2.0, -3.0]])
    indices = torch.tensor([[0, 1, 2]])
    coarse, _ = coarsen_logits(logits, indices, torch.ones_like(indices, dtype=torch.bool))
    keep = torch.tensor([[True, False, False]])
    values = encode_adaptive_logits(coarse, keep, dtype)
    decoded = decode_adaptive_logits(values)
    assert values.dtype == dtype and values.numel() == 3
    assert decoded[..., 1:3].isneginf().all()
    assert torch.isfinite(adaptive_logprobs(values)[..., [0, -1]]).all()


def test_mixed_full_vocab_gradients_match_fp64():
    logits = torch.zeros(2, 1, 64)
    logits[1, :, 1:] = -30
    indices = logits.topk(2, -1).indices
    keep = torch.ones_like(indices, dtype=torch.bool)
    target, _ = coarsen_logits(logits, indices, keep)
    values = encode_adaptive_logits(target, keep)
    student = (logits + torch.linspace(-0.2, 0.2, 64)).double().requires_grad_()
    reference = student.detach().clone().requires_grad_()
    actual = reference_logits_kl_div(student, values, indices, cache_format_version=4)
    selected = reference.gather(-1, indices)
    tail = reference.scatter(-1, indices, -torch.inf).logsumexp(-1, keepdim=True)
    expected = _kl64(torch.cat((selected, tail), -1), decode_adaptive_logits(values))
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(student.grad, reference.grad, atol=1e-12, rtol=1e-10)


def test_decoded_target_has_zero_gradient():
    for values in (
        torch.tensor([[[-2.0, 1.0]]], dtype=torch.float16),
        torch.tensor([[[-2.0, -1.0]]], dtype=torch.float16),
    ):
        student = decode_adaptive_logits(values).requires_grad_()
        loss = reference_logits_kl_div(
            student, values, torch.tensor([[[0, 1]]]), cache_format_version=4
        )
        loss.sum().backward()
        assert loss.eq(0).all()
        assert student.grad.eq(0).all()


def test_tail_metric_minimizes_decoded_tail_error():
    torch.manual_seed(801)
    for sigma in (1.0, 4.0, 8.0):
        logits = torch.randn(32, 128) * sigma
        indices = logits.topk(8, -1).indices
        keep = torch.ones_like(indices, dtype=torch.bool)
        target, _ = coarsen_logits(logits, indices, keep)
        adaptive = encode_adaptive_logits(target, keep, metric="tail")
        reference = encode_reference_logits(target, keep)
        probability = stable_log_softmax(target)[..., :-1].half()
        probability[..., 1] = -probability[..., 1].abs()
        true_tail = stable_log_softmax(target.double())[..., -1].exp()
        a_error = (adaptive_logprobs(adaptive)[..., -1].exp().double() - true_tail).abs()
        r_error = (
            stable_log_softmax(decode_reference_logits(reference))[..., -1].exp().double()
            - true_tail
        ).abs()
        p_error = (adaptive_logprobs(probability)[..., -1].exp().double() - true_tail).abs()
        valid = probability.float().exp().sum(-1) <= 1
        assert torch.all(a_error <= r_error + 1e-14)
        assert torch.all(a_error[valid] <= p_error[valid] + 1e-14)


def test_unknown_selection_metric_rejected():
    with pytest.raises(ValueError, match="selection metric"):
        encode_adaptive_logits(
            torch.tensor([[0.0, -1.0]]), torch.tensor([[True]]), metric="unknown"
        )
