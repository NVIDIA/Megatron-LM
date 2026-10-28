# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Source license: BSD-3-Clause (NVIDIA terms reproduced in LICENSE).
"""Temperature-scaled teacher-to-student KL over tensor-parallel vocabulary shards."""

import math
import os

import torch
from torch import Tensor


def _compile_kernel(fn):
    if os.environ.get("MTP_KD_DISABLE_COMPILE", "0").lower() in ("1", "true"):
        return fn
    return torch.compile(fn, dynamic=False) if hasattr(torch, "compile") else fn


@_compile_kernel
def _local_sum_exp(logits: Tensor, maximum: Tensor, scale: float) -> Tensor:
    return (logits.float() * scale - maximum).exp().sum(-1, keepdim=True)


@_compile_kernel
def _local_kl(
    student: Tensor, teacher: Tensor, student_norm: Tensor, teacher_norm: Tensor, scale: float
) -> Tensor:
    log_p = teacher.float() * scale - teacher_norm
    log_q = student.float() * scale - student_norm
    p = log_p.exp()
    return (p * log_p).sum(-1) - (p * log_q).sum(-1)


@_compile_kernel
def _kl_grad(
    student: Tensor,
    teacher: Tensor,
    student_norm: Tensor,
    teacher_norm: Tensor,
    scale: float,
    weight: Tensor,
) -> Tensor:
    p = (teacher.float() * scale - teacher_norm).exp()
    q = (student.float() * scale - student_norm).exp()
    return ((q - p) * weight).to(student.dtype)


def vocab_parallel_log_normalizer(
    logits: Tensor, temperature: float, tp_group: torch.distributed.ProcessGroup | None = None
) -> Tensor:
    """Return FP32 logsumexp(logits / temperature) across the full vocabulary.

    Pass no group for gathered logits. Call under no-grad: this is metadata for
    the manually differentiated KL, not a differentiable distributed reduction.
    """
    scale = 1.0 / temperature
    maximum = (logits.amax(-1, keepdim=True).float() * scale).contiguous()
    if tp_group is not None and tp_group.size() > 1:
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    total = _local_sum_exp(logits, maximum, scale)
    if tp_group is not None and tp_group.size() > 1:
        torch.distributed.all_reduce(total, group=tp_group)
    return maximum + total.log()


class _VocabParallelKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, teacher, teacher_norm, temperature, tp_group):
        student_norm = vocab_parallel_log_normalizer(student, temperature, tp_group)
        loss = _local_kl(student, teacher, student_norm, teacher_norm, 1.0 / temperature)
        if tp_group is not None and tp_group.size() > 1:
            torch.distributed.all_reduce(loss, group=tp_group)
        ctx.save_for_backward(student, teacher, student_norm, teacher_norm)
        ctx.temperature = temperature
        return loss * temperature**2

    @staticmethod
    def backward(ctx, grad_output):
        student, teacher, student_norm, teacher_norm = ctx.saved_tensors
        grad = _kl_grad(
            student,
            teacher,
            student_norm,
            teacher_norm,
            1.0 / ctx.temperature,
            (grad_output * ctx.temperature).unsqueeze(-1),
        )
        return grad, None, None, None, None


def vocab_parallel_kl_div(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 1.0,
    tp_group: torch.distributed.ProcessGroup | None = None,
    teacher_log_normalizer: Tensor | None = None,
) -> Tensor:
    """Return per-token T² KL(teacher || student), with no teacher gradients.

    Logits have shape [sequence, batch, vocabulary shard]; both inputs must use
    the same shard. The FP32 result has shape [sequence, batch]. Backward
    recomputes probabilities, retaining no additional vocabulary-width activations.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("MTP KD temperature must be finite and positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have matching shapes")
    teacher_logits = teacher_logits.detach()
    if teacher_log_normalizer is None:
        with torch.no_grad():
            teacher_log_normalizer = vocab_parallel_log_normalizer(
                teacher_logits, temperature, tp_group
            )
    return _VocabParallelKL.apply(
        student_logits, teacher_logits, teacher_log_normalizer.detach(), temperature, tp_group
    )
