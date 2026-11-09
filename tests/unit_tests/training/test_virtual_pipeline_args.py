# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""validate_args checks --num-virtual-stages-per-pipeline-rank like the other VPP options.

A wrong layer count used to pass validation and fail only when the model was built, and a value
below 1 built no model chunks at all.
"""

import sys

import pytest

from megatron.training.arguments import parse_args, validate_args

PP2_ARGV = (
    "--hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 --global-batch-size 8 "
    "--seq-length 32 --max-position-embeddings 32 --pipeline-model-parallel-size 2"
).split()


def _validated(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["test_virtual_pipeline_args.py", *PP2_ARGV, *extra])
    args = parse_args()
    # parse_args reads WORLD_SIZE from the environment; pin it to the pipeline size.
    args.world_size = 2
    return validate_args(args)


@pytest.mark.parametrize(
    ("num_layers", "num_virtual_stages", "message"),
    [
        (3, 2, "must be divisible by the pipeline-model-parallel size"),
        (3, 1, "must be divisible by the pipeline-model-parallel size"),
        (6, 2, r"per pipeline stage \(3\) must be divisible"),
        (4, 0, "must be at least 1"),
        (4, -1, "must be at least 1"),
    ],
)
def test_num_virtual_stages_per_pipeline_rank_is_checked(
    monkeypatch, num_layers, num_virtual_stages, message
):
    extra = [
        "--num-layers",
        str(num_layers),
        f"--num-virtual-stages-per-pipeline-rank={num_virtual_stages}",
    ]
    with pytest.raises(ValueError, match=message):
        _validated(monkeypatch, extra)


def test_num_virtual_stages_per_pipeline_rank_sets_the_virtual_pipeline_size(monkeypatch):
    args = _validated(
        monkeypatch, ["--num-layers", "4", "--num-virtual-stages-per-pipeline-rank", "2"]
    )
    assert args.virtual_pipeline_model_parallel_size == 2
