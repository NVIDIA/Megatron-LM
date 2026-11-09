# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""validate_args checks the intervals of the parameter, activation and gradient dumps.

The dumps are written under --save. Without it, the first dumping iteration used to fail on an
assert in save_grads (or a TypeError for the tokens-per-expert dump), and an interval of 0
divided by zero in train_step.
"""

import sys

import pytest

from megatron.training.arguments import parse_args, validate_args

MODEL_ARGV = (
    "--num-layers 2 --hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 "
    "--global-batch-size 8 --seq-length 32 --max-position-embeddings 32"
).split()

SAVE_ARGV = ["--save", "/tmp/dumps", "--save-interval", "10"]

DUMP_FLAGS = [
    "--save-params-interval",
    "--save-activations-interval",
    "--save-tokens-per-expert-interval",
    "--save-wgrads-interval",
    "--save-dgrads-interval",
]


def _validated(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["test_dump_interval_args.py", *MODEL_ARGV, *extra])
    args = parse_args()
    # parse_args reads WORLD_SIZE from the environment; pin it so the test does not depend
    # on how the job was launched.
    args.world_size = 1
    return validate_args(args)


@pytest.mark.parametrize("flag", DUMP_FLAGS)
def test_dump_interval_requires_save(monkeypatch, flag):
    with pytest.raises(ValueError, match="which is not set"):
        _validated(monkeypatch, [flag, "1"])


@pytest.mark.parametrize("flag", DUMP_FLAGS)
@pytest.mark.parametrize("interval", ["0", "-3"])
def test_dump_interval_must_be_positive(monkeypatch, flag, interval):
    with pytest.raises(ValueError, match="must be positive"):
        _validated(monkeypatch, [*SAVE_ARGV, f"{flag}={interval}"])


@pytest.mark.parametrize("flag", DUMP_FLAGS)
def test_dump_interval_with_save_validates(monkeypatch, flag):
    args = _validated(monkeypatch, [*SAVE_ARGV, flag, "1"])
    assert args.save == "/tmp/dumps"
