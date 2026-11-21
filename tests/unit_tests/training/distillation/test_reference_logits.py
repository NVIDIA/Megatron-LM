# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Numerical tests independent of model construction and CUDA extensions."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.training.distillation.reference_logits import (
    coarsen_logits,
    decode_reference_logits,
    encode_reference_logits,
    reference_logits_kl_div,
    reference_logprobs,
    stable_log_softmax,
)


def _encode(logits, k, dtype=torch.float16):
    indices = logits.topk(k, -1).indices
    keep = torch.ones_like(indices, dtype=torch.bool)
    coarse, _ = coarsen_logits(logits, indices, keep)
    return encode_reference_logits(coarse, keep, dtype), indices


def _reference_loss(logits, values, indices):
    teacher = decode_reference_logits(values).double()
    valid = teacher[..., :-1].isfinite()
    local = indices.clamp(0, logits.size(-1) - 1)
    selected = logits.gather(-1, local).masked_fill(~valid, -torch.inf)
    # Reference tests use unique valid indices. Invalid ones scatter into
    # a separate column so their duplicate index cannot erase a real token.
    tail = torch.cat((logits, torch.full_like(logits[..., :1], -torch.inf)), -1)
    tail = tail.scatter(-1, torch.where(valid, local, logits.size(-1)), -torch.inf)
    student = torch.cat((selected, torch.logsumexp(tail[..., :-1], -1, keepdim=True)), -1)
    log_q = torch.log_softmax(student, -1)
    log_p = torch.log_softmax(teacher, -1)
    return torch.where(log_p.isfinite(), log_p.exp() * (log_p - log_q), 0).sum(-1)


@pytest.mark.parametrize("k", [1, 2, 8])
def test_roundtrip_and_storage_size(k):
    logits = torch.tensor([[[0.0, -1, -2, -3, -4, -5, -6, -7]]], dtype=torch.float64)
    values, indices = _encode(logits, k)
    assert values.dtype == torch.float16
    assert values.shape == indices.shape
    assert values.numel() * values.element_size() == 2 * k
    decoded = reference_logprobs(values).double().exp()
    p = logits.softmax(-1)
    selected = p.gather(-1, indices)
    tail = p.scatter(-1, indices, 0).sum(-1, keepdim=True)
    torch.testing.assert_close(decoded, torch.cat((selected, tail), -1), atol=3e-4, rtol=8e-3)
    if k == logits.size(-1):
        assert values[..., 0].isneginf().all()
        assert decoded[..., -1].eq(0).all()


def test_tiny_tail_survives_fp16_storage():
    logits = torch.tensor([[[0.0, -30.0, -31.0, -32.0]]], dtype=torch.float64)
    values, _ = _encode(logits, 1)
    actual = reference_logprobs(values)[..., -1].exp()
    expected = logits.softmax(-1)[..., 1:].sum(-1)
    assert 0 < actual.item() < 1e-12
    torch.testing.assert_close(actual.double(), expected, atol=0, rtol=0.002)


@pytest.mark.parametrize("k", [1, 3])
def test_documented_tail_reconstruction(k):
    logits = torch.tensor([[[0.0, -30.0, -31.0, -32.0]]])
    values, _ = _encode(logits, k)
    v = values.float()
    other_lse = torch.logsumexp(v[..., 1:], dim=-1)
    log_z = torch.nn.functional.softplus(torch.logaddexp(other_lse, v[..., 0]))
    tail = torch.exp(v[..., 0] - log_z)
    torch.testing.assert_close(tail, reference_logprobs(values)[..., -1].exp(), atol=0, rtol=1e-6)


def test_top_p_exclusions_are_added_to_tail_without_more_values():
    logits = torch.tensor([[[0.0, -1.0, -2.0, -3.0]]])
    indices = torch.tensor([[[0, 1, 2]]])
    coarse, _ = coarsen_logits(logits, indices, torch.ones_like(indices, dtype=torch.bool))
    keep = torch.tensor([[[True, False, False]]])
    values = encode_reference_logits(coarse, keep)
    assert values.shape == indices.shape
    assert values[..., 1:].isneginf().all()
    actual = reference_logprobs(values).exp()[..., -1]
    torch.testing.assert_close(actual, logits.softmax(-1)[..., 1:].sum(-1), atol=2e-5, rtol=1e-3)
    # The packed index sentinel wraps to the largest 17-bit index; value
    # validity must keep it from accidentally selecting a vocabulary entry.
    masked_indices = torch.tensor([[[0, 131071, 0]]])
    student = (logits + 0.2).requires_grad_()
    reference_logits_kl_div(student, values, masked_indices).sum().backward()
    assert torch.isfinite(student.grad).all()


@pytest.mark.parametrize("k", [1, 3, 7])
def test_loss_and_full_vocab_gradient_match_fp64(k):
    torch.manual_seed(801 + k)
    teacher = torch.randn(3, 2, 7, dtype=torch.float64)
    values, indices = _encode(teacher, k)
    logits = (teacher + torch.randn_like(teacher) * 0.3).requires_grad_()
    reference = logits.detach().clone().requires_grad_()
    actual_loss = reference_logits_kl_div(logits, values, indices)
    expected_loss = _reference_loss(reference, values, indices)
    weight = torch.rand_like(actual_loss)
    (actual_loss * weight).sum().backward()
    (expected_loss * weight).sum().backward()
    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(logits.grad, reference.grad, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_noncontiguous_logits_and_input_dtype(dtype):
    torch.manual_seed(801)
    teacher = torch.randn(2, 3, 8)
    values, indices = _encode(teacher, 4)
    logits = teacher.transpose(0, 1).contiguous().transpose(0, 1).to(dtype).requires_grad_()
    before = logits.detach().clone()
    reference_logits_kl_div(logits, values, indices).sum().backward()
    assert logits.grad.dtype == dtype
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.detach(), before, atol=0, rtol=0)


def test_equal_decoded_target_has_exact_zero_loss_and_gradient():
    values = torch.tensor([[[-2.0, -1.0]]], dtype=torch.float16)
    # One omitted token makes its logit equal to the tail log-sum-exp.
    logits = torch.tensor([[[0.0, -1.0, -2.0]]], requires_grad=True)
    indices = torch.tensor([[[0, 1]]])
    loss = reference_logits_kl_div(logits, values, indices)
    loss.sum().backward()
    assert loss.eq(0).all()
    assert logits.grad.eq(0).all()


def test_near_equality_gradient_matches_fp64():
    values = torch.tensor([[[-2.0, -1.0]]], dtype=torch.float16)
    indices = torch.tensor([[[0, 1]]])
    logits = torch.tensor([[[0.0, -1.0 + 1e-6, -2.0 - 1e-6]]], requires_grad=True)
    reference = logits.detach().double().requires_grad_()
    loss = reference_logits_kl_div(logits, values, indices)
    loss.sum().backward()
    _reference_loss(reference, values, indices).sum().backward()
    assert loss.ge(0).all()
    torch.testing.assert_close(logits.grad.double(), reference.grad, atol=1e-13, rtol=2e-5)


def test_dominant_logit_gradient_does_not_collapse():
    values = torch.tensor([[[-30.0]]], dtype=torch.float16)
    indices = torch.tensor([[[0]]])
    logits = torch.tensor([[[0.0, -29.0]]], requires_grad=True)
    reference_logits_kl_div(logits, values, indices).sum().backward()
    assert logits.grad[0, 0, 0] < 0
    assert logits.grad[0, 0, 1] > 0
    assert logits.grad.sum() == 0


def test_large_mismatch_does_not_overflow():
    values = torch.tensor([[[-100.0, -100.0]]], dtype=torch.float16)
    indices = torch.tensor([[[0, 1]]])
    logits = torch.tensor([[[-1000.0, 1000.0, 0.0]]], requires_grad=True)
    loss = reference_logits_kl_div(logits, values, indices)
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(logits.grad).all()


def test_gradcheck():
    values = torch.tensor([[[-2.0, -1.0]]], dtype=torch.float16)
    indices = torch.tensor([[[0, 1]]])
    logits = torch.tensor([[[0.0, -0.8, -2.1]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda z: reference_logits_kl_div(z, values, indices), (logits,)
    )


def _tp_worker(rank, world_size, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        torch.manual_seed(801)
        for k in (1, 4, 8):
            teacher = torch.randn(2, 1, 8)
            values, indices = _encode(teacher, k)
            full = (teacher + 0.2 * torch.randn_like(teacher)).requires_grad_()
            expected = reference_logits_kl_div(full, values, indices)
            expected.sum().backward()
            shard = full.detach().chunk(world_size, -1)[rank].contiguous().requires_grad_()
            actual = reference_logits_kl_div(
                shard, values, indices, world_size, rank, dist.group.WORLD
            )
            actual.sum().backward()
            torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
            torch.testing.assert_close(
                shard.grad, full.grad.chunk(world_size, -1)[rank], atol=2e-7, rtol=2e-5
            )
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_loss_and_gradient(tmp_path):
    if dist.is_initialized():
        pytest.skip("Standalone Gloo test; model-parallel CI already owns the process group.")
    mp.spawn(_tp_worker, args=(2, f"file://{tmp_path / 'rendezvous'}"), nprocs=2, join=True)
