# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""validate_args rejects Megatron-FSDP v1 with pipeline parallelism.

The v1 device mesh covers only the data-, context-, expert- and tensor-parallel ranks, so with
more than one pipeline stage the run used to fail in einops while wrapping the model.
"""

import sys

import pytest

from megatron.training.arguments import parse_args, validate_args

PP2_FSDP_ARGV = (
    "--num-layers 4 --hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 "
    "--global-batch-size 8 --seq-length 32 --max-position-embeddings 32 "
    "--pipeline-model-parallel-size 2 --use-megatron-fsdp --ckpt-format fsdp_dtensor"
).split()


def test_megatron_fsdp_v1_rejects_pipeline_parallelism(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_megatron_fsdp_args.py", *PP2_FSDP_ARGV])
    args = parse_args()
    # parse_args reads WORLD_SIZE from the environment; pin it to the pipeline size.
    args.world_size = 2
    with pytest.raises(ValueError, match="Megatron-FSDP v1 does not support pipeline parallelism"):
        validate_args(args)
