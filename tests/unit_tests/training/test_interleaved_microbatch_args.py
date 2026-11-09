# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""validate_args checks the micro-batch groups of the interleaved pipeline schedule.

forward_backward_pipelining_with_interleaving checks them only when a step starts, so a
configuration that the schedule rejects used to pass validation and fail in the first training
or evaluation step.
"""

import sys

import pytest

from megatron.training.arguments import parse_args, validate_args

# Two pipeline stages with two virtual stages each, one sample per micro-batch, one DP rank.
VPP_ARGV = (
    "--num-layers 4 --hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 "
    "--seq-length 32 --max-position-embeddings 32 --pipeline-model-parallel-size 2 "
    "--num-layers-per-virtual-pipeline-stage 1"
).split()


def _validated(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["test_interleaved_microbatch_args.py", *VPP_ARGV, *extra])
    args = parse_args()
    # parse_args reads WORLD_SIZE from the environment; pin it to the pipeline size.
    args.world_size = 2
    return validate_args(args)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        # Without --global-batch-size a step has one micro-batch.
        ([], r"training micro-batches per step, but N=2, PP=2 and M=1"),
        (["--global-batch-size", "3"], r"training micro-batches per step \(M=3\).*remainder is 1"),
        (
            ["--global-batch-size", "4", "--microbatch-group-size-per-virtual-pipeline-stage", "5"],
            r"N=5, PP=2 and M=4",
        ),
        (
            ["--global-batch-size", "4", "--eval-iters", "1", "--eval-global-batch-size", "1"],
            r"evaluation micro-batches per step, but N=2, PP=2 and M=1",
        ),
    ],
)
def test_interleaved_schedule_rejects_misfit_microbatch_groups(monkeypatch, extra, message):
    with pytest.raises(ValueError, match=message):
        _validated(monkeypatch, extra)


@pytest.mark.parametrize(
    "extra",
    [
        ["--global-batch-size", "4"],
        ["--global-batch-size", "6"],
        # Evaluation does not run, so its batch size does not matter.
        ["--global-batch-size", "4", "--eval-iters", "0", "--eval-global-batch-size", "1"],
    ],
)
def test_interleaved_schedule_accepts_fitting_microbatch_groups(monkeypatch, extra):
    assert _validated(monkeypatch, extra).virtual_pipeline_model_parallel_size == 2
