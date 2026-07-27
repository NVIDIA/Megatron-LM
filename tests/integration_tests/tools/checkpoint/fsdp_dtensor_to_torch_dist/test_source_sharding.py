# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check 4 — source-side sharding.

The counterpart to reshard. Checks 1-3 start from a source trained single-rank (an
unsharded DCP store). This trains the FSDP source on >=2 GPUs under a real sharded
layout — DP2 (params + optimizer split across data-parallel ranks) or TP2 / EP2 —
then converts (single-process, unchanged) and resumes a classic single-rank job,
checking first-post-load continuity against the sharded run's own log. This exercises
the converter's source-side gather (multi-shard DTensor reassembly, TP-reshard
path, expert gather) that a single-rank source never produces.

PP2 is skipped: Megatron-FSDP + pipeline-parallel *training* fails at model build,
so no PP2 source can be produced (a training-side limitation, not the converter's).
Needs >=2 GPUs.
"""

import pytest
import torch

from tests.integration_tests.tools.checkpoint.fsdp_dtensor_to_torch_dist import (
    _cases,
    config,
    harness,
)


@pytest.mark.e2e
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="source sharding needs >=2 GPUs")
@pytest.mark.parametrize("family, source", _cases.source_shard_params())
def test_source_sharding(family, source, sharded_family_runs):
    run = sharded_family_runs(family, source.layout)
    for iteration in config.CONVERT_ITERS:
        nxt = iteration + 1
        log = run.root / f"resume_{iteration}.log"
        text = harness.resume_classic(
            family, run.td[iteration], iteration, log
        )  # classic single-rank
        harness.assert_loaded_at(text, iteration)
        resumed = harness.parse_iter_metrics(text)
        assert nxt in resumed, f"[{family.name}/{source.layout}] no iteration {nxt} in {log}"
        harness.assert_loss_lr(run.fsdp_metrics[nxt], resumed[nxt], loss_rtol=family.loss_rtol)
