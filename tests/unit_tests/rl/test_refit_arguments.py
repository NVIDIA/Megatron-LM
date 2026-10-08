# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from argparse import ArgumentParser

from megatron.training.arguments import _add_rl_args


def test_refit_execution_batch_bytes_argument():
    parser = _add_rl_args(ArgumentParser())

    assert parser.parse_args([]).refit_execution_batch_bytes is None
    assert (
        parser.parse_args(["--refit-execution-batch-bytes", "123"]).refit_execution_batch_bytes
        == 123
    )
