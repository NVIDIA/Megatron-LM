# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check 3 — load-side resharding.

The converter's whole promise is that its full-global-shape torch_dist output
reshards into any target parallelism on load. This reuses the most-trained converted
checkpoint and loads it into a classic 2-GPU job under each target layout in the
family's ``reshard_cases`` (TP/PP/EP/TP-SP), asserting a clean load + first-post-load
continuity vs the FSDP reference.

EP>1 optimizer load is a documented limitation (ChainedOptimizer entry-count
mismatch) and is a strict xfail; the weights-only EP2 companion passes. Needs
>=2 GPUs.
"""

import pytest
import torch

from tests.integration_tests.tools.checkpoint.fsdp_dtensor_to_torch_dist import (
    _cases,
    config,
    harness,
)


@pytest.mark.e2e
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="reshard needs >=2 GPUs")
@pytest.mark.parametrize("family, reshard", _cases.reshard_params())
def test_reshard(family, reshard, family_runs):
    run = family_runs(family)
    iteration = config.CONVERT_ITERS[-1]  # the most-trained converted checkpoint
    nxt = iteration + 1
    mode = "optim" if reshard.with_optimizer else "weights"
    log = run.root / f"reshard_{reshard.layout}_{mode}.log"
    text = harness.reshard_load(family, run.td[iteration], reshard, log, iteration=iteration)
    harness.assert_loaded_at(text, iteration)
    resumed = harness.parse_iter_metrics(text)
    assert nxt in resumed, f"[{family.name}/{reshard.layout}] no iteration {nxt} in {log}"
    harness.assert_loss_lr(run.fsdp_metrics[nxt], resumed[nxt], loss_rtol=family.loss_rtol)
