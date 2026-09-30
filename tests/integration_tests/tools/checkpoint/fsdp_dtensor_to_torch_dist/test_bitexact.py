# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check 2 — bit-exact against the FSDP source.

The resume check *infers* equivalence from a scalar loss; this *proves* it. A
per-family worker loads the converted checkpoint into a real classic GPTModel +
DistributedOptimizer (built via the exact pretrain_gpt init path) and compares what
that job now holds — every model tensor, fp32 master, ``exp_avg`` / ``exp_avg_sq``
and per-parameter group hyperparameter (``step``, ``betas``, ...) — against the
original fsdp_dtensor checkpoint with ``torch.equal``, keyed by the model's own
parameter names. Nothing the converter did is trusted: a transposed stack, swapped
moments, a lost master or a mis-grouped parameter all fail here.

FP8 (``dense_fp8``) is xfail: its GEMM weights are held re-quantized to fp8 with
fresh scaling state (the amax/scale ``_extra_state`` is not in the fsdp_dtensor
checkpoint), so they differ from the bf16 source; its fp32 masters and Adam moments
are still bit-exact. Single-rank (1 GPU).
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
    verdict = harness.run_bitexact_worker(family, run.fsdp_dir, run.td[iteration], iteration)
    harness.assert_bitexact(verdict, family.name, iteration)
