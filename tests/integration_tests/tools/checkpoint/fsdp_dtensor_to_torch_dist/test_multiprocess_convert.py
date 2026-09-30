# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check 5 — multi-process conversion.

For checkpoints larger than one host's RAM the converter runs under
``torchrun --nproc_per_node N``: it shards the tensors across the processes by
output group (SwiGLU halves, a layer's experts, a stacked parameter's layers and a
parameter's optimizer state stay on one process) and writes one torch_dist store
cooperatively. This re-converts each family's most-trained checkpoint with two
processes and asserts the result is identical — every tensor and ``common.pt`` — to
the single-process conversion the other checks already validated. CPU only.
"""

import pytest

from tests.integration_tests.tools.checkpoint.fsdp_dtensor_to_torch_dist import (
    _cases,
    config,
    harness,
)


@pytest.mark.e2e
@pytest.mark.parametrize("family", _cases.resume_params())
def test_multiprocess_convert(family, family_runs):
    run = family_runs(family)
    iteration = config.CONVERT_ITERS[-1]
    it_dir = f"iter_{iteration:07d}"
    out = harness.convert(
        run.fsdp_dir / it_dir, run.root / f"td{iteration}_nproc2", iteration, nproc=2
    )
    diffs = harness.diff_torch_dist(run.td[iteration] / it_dir, out / it_dir)
    assert not diffs, f"[{family.name}] 2-process conversion differs: {diffs[:20]}"
