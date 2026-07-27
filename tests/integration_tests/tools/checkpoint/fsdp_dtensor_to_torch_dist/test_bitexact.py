# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check 2 — bit-exact per-tensor diff.

The resume check *infers* equivalence from a scalar loss; this *proves* it. A
per-family worker loads the converted checkpoint into a real classic GPTModel +
DistributedOptimizer (built via the exact pretrain_gpt init path), immediately
re-saves it (no train step ⇒ no drift), and does a strict per-tensor diff of the
re-save against the converter output. A clean diff ⇒ bit-exact weights AND
optimizer (fp32 masters, exp_avg / exp_avg_sq, reconstructed param_groups). This
is the strongest single-GPU signal and is independent of iteration count.

FP8 (``dense_fp8``) is xfail: its amax/scale ``_extra_state`` is dropped by design,
so a few fp8 weight tensors re-quantize differently. Single-rank (1 GPU).
"""

import pytest
import torch

from tests.integration_tests.tools.checkpoint.fsdp_dtensor_to_torch_dist import (
    _cases,
    config,
    harness,
)


@pytest.mark.e2e
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="needs 1 GPU")
@pytest.mark.parametrize("family", _cases.bitexact_params())
def test_bitexact(family, family_runs):
    run = family_runs(family)
    iteration = config.CONVERT_ITERS[-1]  # the most-trained converted checkpoint
    verdict = harness.run_bitexact_worker(family, run.td[iteration], iteration)
    assert verdict.loaded_iteration == iteration
    assert verdict.verdict == "PASS", (
        f"[{family.name}] not bit-exact — "
        f"weights={verdict.weight_mismatches} "
        f"optimizer={verdict.optim_mismatches} "
        f"unexpected_extra={verdict.unexpected_extra}"
    )
