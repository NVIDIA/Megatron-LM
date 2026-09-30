# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check 4 — source-side sharding.

The counterpart to reshard. The other checks start from a source trained single-rank
(an unsharded DCP store). This trains the FSDP source on 2 GPUs under a real sharded
layout — DP2 (params + optimizer split across data-parallel ranks) or TP2 / EP2 —
converts it (single-process, unchanged), and asserts the result is bit-exact against
that sharded source, then that a classic single-rank job resumes it. This exercises
the converter's source-side gather (multi-shard DTensor reassembly, TP-reshard path,
expert gather) that a single-rank source never produces.

The bit-exact check is the verdict here: the FSDP reference run used a different
parallel layout than the single-rank resume, so their losses agree only up to
numerics (loosened per case in the registry where routing amplifies it).

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
        if not family.bitexact_xfail:  # FP8 is not bit-exact by design (see test_bitexact)
            verdict = harness.run_bitexact_worker(
                family, run.fsdp_dir, run.td[iteration], iteration
            )
            harness.assert_bitexact(verdict, family.name, iteration)
        log = run.root / f"resume_{iteration}.log"  # classic single-rank
        harness.check_resume(
            family, run.fsdp_metrics, run.td[iteration], iteration, log, loss_rtol=source.loss_rtol
        )
