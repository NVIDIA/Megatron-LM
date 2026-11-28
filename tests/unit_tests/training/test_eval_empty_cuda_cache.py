# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse

from megatron.training.arguments import _add_validation_args
from megatron.training.config import ValidationConfig


def test_eval_empty_cuda_cache_defaults_off():
    assert ValidationConfig().eval_empty_cuda_cache is False


def test_eval_empty_cuda_cache_cli_flag():
    parser = argparse.ArgumentParser()
    _add_validation_args(parser)

    assert parser.parse_args([]).eval_empty_cuda_cache is False
    assert parser.parse_args(["--eval-empty-cuda-cache"]).eval_empty_cuda_cache is True
