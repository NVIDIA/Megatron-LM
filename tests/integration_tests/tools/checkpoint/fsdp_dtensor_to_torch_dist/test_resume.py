# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check 1 — resume continuity.

Train real Megatron-FSDP, convert each configured checkpoint to torch_dist, resume
a classic (non-FSDP) job from each, and assert that the first post-load ``lm loss``
matches the FSDP reference within bf16 tolerance and the learning rate matches
exactly. A matching loss means the weights loaded; a matching LR means the optimizer
state and LR-scheduler bookkeeping converted correctly. Two independent load points
rule out a single lucky checkpoint. Single-rank (1 GPU).
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
@pytest.mark.parametrize("family", _cases.resume_params())
def test_resume_continuity(family, family_runs):
    run = family_runs(family)
    for iteration in config.CONVERT_ITERS:  # two independent load points
        log = run.root / f"resume_{iteration}.log"
        harness.check_resume(family, run.fsdp_metrics, run.td[iteration], iteration, log)
