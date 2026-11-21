# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed error handling for the MoE arm comparison."""

import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks/moe/token_imbalance"))
from compare_arms import _ArmValidationError, _run_step  # noqa: E402


def _check_step(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=15)
    )
    try:
        hidden = torch.ones(1)
        adapter = Mock()
        for failing in ((), (1,), (0, 1)):
            adapter.run_workload.return_value = {"elapsed_ms": 1.0}
            adapter.run_workload.side_effect = (
                AssertionError("nonfinite output") if rank in failing else None
            )
            kwargs = dict(learning_rate=0.0, collect=True, group=dist.group.WORLD)
            if failing:
                reason = "nonfinite output" if rank in failing else "failed on another rank"
                with pytest.raises(_ArmValidationError, match=reason):
                    _run_step(adapter, hidden, None, None, **kwargs)
            else:
                result = _run_step(adapter, hidden, None, None, **kwargs)
                times = [None, None]
                dist.all_gather_object(times, result["elapsed_ms"])
                assert times == [1.0, 1.0]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires the Gloo backend")
def test_post_step_validation_matches_collectives_across_ranks(tmp_path):
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    torch.multiprocessing.spawn(_check_step, args=((tmp_path / "init").as_uri(),), nprocs=2)
