# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys

import pytest

from megatron.training.arguments import parse_args, validate_args
from megatron.training.training import get_megatron_ddp_config


def _parse_minimal_args(monkeypatch, extra_args):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_megatron_fsdp_arguments.py",
            "--num-layers",
            "1",
            "--hidden-size",
            "128",
            "--num-attention-heads",
            "4",
            "--max-position-embeddings",
            "128",
            "--seq-length",
            "128",
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "1",
            "--train-iters",
            "1",
            "--lr",
            "1e-4",
            "--tokenizer-type",
            "NullTokenizer",
            "--vocab-size",
            "128",
            *extra_args,
        ],
    )
    return parse_args()


def test_fsdp_persistent_fallback_flag_maps_to_ddp_config(monkeypatch):
    args = _parse_minimal_args(
        monkeypatch,
        [
            "--use-megatron-fsdp",
            "--ckpt-format",
            "fsdp_dtensor",
            "--fsdp-db-use-persist-buf-on-alloc-fail",
        ],
    )

    assert get_megatron_ddp_config(args).fsdp_db_use_persist_buf_on_alloc_fail is True


def test_fsdp_persistent_fallback_flag_requires_megatron_fsdp(monkeypatch):
    args = _parse_minimal_args(monkeypatch, ["--fsdp-db-use-persist-buf-on-alloc-fail"])

    with pytest.raises(AssertionError, match="requires --use-megatron-fsdp"):
        validate_args(args)
