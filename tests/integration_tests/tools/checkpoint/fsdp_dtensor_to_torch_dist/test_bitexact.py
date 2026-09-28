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
    verdict = harness.run_bitexact_worker(family, run.fsdp_dir, run.td[iteration], iteration)
    assert verdict.loaded_iteration == iteration
    # Guard against a vacuous pass: every section must actually have been compared.
    assert all(n > 0 for n in verdict.counts.values()), verdict.counts
    assert not (verdict.mismatches or verdict.missing or verdict.unexpected), (
        f"[{family.name}] not bit-exact vs the FSDP source ({verdict.counts}): "
        f"mismatches={verdict.mismatches[:20]} missing={verdict.missing[:20]} "
        f"unexpected={verdict.unexpected[:20]}"
    )
