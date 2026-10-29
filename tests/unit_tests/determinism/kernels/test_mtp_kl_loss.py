# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of compiled MTP KD reductions and student gradients."""

import pytest
import torch

from megatron.core.ops.mtp_kl_loss import vocab_parallel_kl_div, vocab_parallel_log_normalizer
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cached_normalizer", [False, True])
def test_mtp_kl_loss_replay(dtype, cached_normalizer):
    seeded()
    student = torch.randn(128, 2, 8192, device="cuda", dtype=dtype, requires_grad=True)
    teacher = torch.randn_like(student)
    weight = torch.randn(128, 2, device="cuda")
    normalizer = vocab_parallel_log_normalizer(teacher, 2.0, None) if cached_normalizer else None

    def loss(student, teacher):
        return vocab_parallel_kl_div(student, teacher, 2.0, teacher_log_normalizer=normalizer)

    _, grads = assert_replays_bit_exact(
        loss, (student, teacher), grad_outputs={"out": weight}, contention=True, what="MTP KD"
    )
    assert len(grads) == 1
