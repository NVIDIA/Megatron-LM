# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for StudentLogitsCapture storage isolation from LM cross entropy and KD."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.ops.loss.cross_entropy import unfused_cross_entropy
from megatron.core.ops.loss.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.training.distillation import cached_logits_loss
from megatron.training.distillation.cached_logits_loss import StudentLogitsCapture, topk_kl_div


class _SingleRankTPGroup:
    """Minimal single-rank process group for exercising TP loss kernels without init."""

    def rank(self):
        return 0

    def size(self):
        return 1


class _OutputLayer(torch.nn.Linear):
    """Return the same ``(logits, bias)`` tuple as ``ColumnParallelLinear``."""

    def forward(self, hidden):
        return super().forward(hidden), None


class _Model(torch.nn.Module):
    """Minimal model binding ``LanguageModule.compute_language_model_loss``."""

    compute_language_model_loss = LanguageModule.compute_language_model_loss

    def __init__(self, cross_entropy, dtype, device):
        super().__init__()
        self.output_layer = _OutputLayer(16, 32, bias=False, dtype=dtype, device=device)
        self.tp_group = _SingleRankTPGroup()
        self.vocab_parallel_cross_entropy = cross_entropy


@pytest.fixture
def capture(monkeypatch):
    """Provide an isolated ``StudentLogitsCapture`` and clean up hooks after each test."""
    monkeypatch.setattr(cached_logits_loss, "_ACTIVE_STUDENT_LOGITS_CAPTURE", None)
    logits_capture = StudentLogitsCapture()
    yield logits_capture
    logits_capture.remove_hooks()


@pytest.mark.parametrize(
    "cross_entropy", [unfused_cross_entropy, fused_vocab_parallel_cross_entropy]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_capture_preserves_logits_and_joint_lm_kd_gradients(
    capture, monkeypatch, cross_entropy, dtype
):
    """Captured logits remain intact across native CE and match dense joint LM + KD grads."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Single-rank TP all-reduce is a no-op; keep the real CE forward and backward kernels.
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)

    generator = torch.Generator().manual_seed(0)
    model = _Model(cross_entropy, dtype, device)
    with torch.no_grad():
        weight_init = (0.1 * torch.randn(32, 16, generator=generator)).to(
            device=device, dtype=dtype
        )
        model.output_layer.weight.copy_(weight_init)
    capture.attach_hooks(model)

    hidden_base = torch.randn(4, 2, 16, generator=generator).to(device=device, dtype=dtype)
    hidden = hidden_base.clone().requires_grad_(True)
    labels = torch.randint(32, (2, 4), generator=generator).to(device=device)
    teacher_logprobs = torch.randn(4, 2, 32, generator=generator).log_softmax(dim=-1)
    teacher_values, teacher_indices = teacher_logprobs.topk(8, dim=-1)
    teacher_values = teacher_values.to(device=device)
    teacher_indices = teacher_indices.to(device=device)

    logits, _ = model.output_layer(hidden)
    expected = logits.detach().clone()
    lm = model.compute_language_model_loss(labels, logits).sum()
    student_logits = capture.pop()

    assert student_logits.data_ptr() != logits.data_ptr()
    torch.testing.assert_close(student_logits.detach(), expected, rtol=0, atol=0)
    if dtype == torch.float32:
        # Verify the native FP32 CE kernel actually overwrote its input buffer.
        assert not torch.equal(logits.detach(), expected)

    kd = topk_kl_div(
        student_logits, teacher_values, teacher_indices, 1, 0, model.tp_group, add_ghost_token=True
    ).sum()
    (lm + kd).backward()

    ref_weight = model.output_layer.weight.detach().clone().requires_grad_(True)
    ref_hidden = hidden_base.clone().requires_grad_(True)
    ref_logits = F.linear(ref_hidden, ref_weight)
    ref_lm = F.cross_entropy(
        ref_logits.transpose(0, 1).float().flatten(0, 1), labels.flatten(), reduction="sum"
    )
    ref_logprobs = ref_logits.float().log_softmax(dim=-1).gather(-1, teacher_indices)
    ref_student_residual = torch.log(
        (1.0 - ref_logprobs.exp().sum(dim=-1, keepdim=True)).clamp(min=1e-8)
    )
    ref_teacher_residual = torch.log(
        (1.0 - teacher_values.exp().sum(dim=-1, keepdim=True)).clamp(min=1e-8)
    )
    ref_student = torch.cat([ref_logprobs, ref_student_residual], dim=-1)
    ref_teacher = torch.cat([teacher_values, ref_teacher_residual], dim=-1)
    ref_kd = (ref_teacher.exp() * (ref_teacher - ref_student)).sum()
    (ref_lm + ref_kd).backward()

    torch.testing.assert_close(lm, ref_lm)
    torch.testing.assert_close(kd, ref_kd)
    torch.testing.assert_close(model.output_layer.weight.grad, ref_weight.grad)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_capture_isolates_in_place_mutating_cross_entropy(capture, dtype):
    """Backends that overwrite input logits in place (e.g., TE CE) cannot corrupt KD."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def mutating_cross_entropy(logits, labels, tp_group):
        del labels, tp_group
        return logits.zero_().sum(dim=-1)

    generator = torch.Generator().manual_seed(0)
    model = _Model(mutating_cross_entropy, dtype, device)
    with torch.no_grad():
        model.output_layer.weight.copy_(
            torch.randn(32, 16, generator=generator).to(device=device, dtype=dtype)
        )
    capture.attach_hooks(model)

    hidden = torch.randn(4, 2, 16, generator=generator).to(device=device, dtype=dtype)
    labels = torch.zeros(2, 4, dtype=torch.long, device=device)
    logits, _ = model.output_layer(hidden)
    expected = logits.detach().clone()
    model.compute_language_model_loss(labels, logits)
    student_logits = capture.pop()

    assert student_logits.data_ptr() != logits.data_ptr()
    assert torch.equal(logits.detach(), torch.zeros_like(expected))
    torch.testing.assert_close(student_logits.detach(), expected, rtol=0, atol=0)


def test_capture_supports_tensor_output_and_clears_on_pop(capture):
    """Bare tensor outputs are also cloned, and ``pop()`` clears the stored reference."""
    model = torch.nn.Module()
    model.output_layer = torch.nn.Linear(16, 32, bias=False)
    capture.attach_hooks(model)

    logits = model.output_layer(torch.ones(4, 2, 16))
    expected = logits.detach().clone()
    logits.zero_()
    student_logits = capture.pop()

    assert student_logits.data_ptr() != logits.data_ptr()
    torch.testing.assert_close(student_logits.detach(), expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="No student logits"):
        capture.pop()


def test_capture_is_skipped_in_eval_mode(capture):
    """Eval-mode forwards must not capture logits."""
    model = torch.nn.Module()
    model.output_layer = _OutputLayer(16, 32, bias=False)
    model.eval()
    capture.attach_hooks(model)
    model.output_layer(torch.zeros(4, 2, 16))

    with pytest.raises(RuntimeError, match="No student logits"):
        capture.pop()
