# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Keep NCCL UB CLI defaults specific to the selected MFSDP implementation."""

import sys

import pytest

from megatron.training.arguments import parse_args, validate_args
from megatron.training.training import get_megatron_ddp_config


@pytest.mark.parametrize(
    "version,ub,manual",
    [(1, False, False), (1, True, False), (2, False, False), (2, True, False), (2, True, True)],
)
def test_nccl_ub_registration_defaults(monkeypatch, version, ub, manual):
    argv = [
        "test_mfsdp_ub_arguments.py",
        "--enable-experimental",
        "--num-layers",
        "2",
        "--hidden-size",
        "128",
        "--num-attention-heads",
        "8",
        "--seq-length",
        "32",
        "--max-position-embeddings",
        "32",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "8",
        "--bf16",
        "--use-megatron-fsdp",
        "--megatron-fsdp-version",
        str(version),
        "--ckpt-format",
        "fsdp_dtensor",
        "--data-parallel-sharding-strategy",
        "optim_grads_params",
        "--no-gradient-accumulation-fusion",
    ]
    if ub:
        argv.append("--use-nccl-ub")
    if manual:
        argv.append("--fsdp-manual-registration")
    monkeypatch.setattr(sys, "argv", argv)

    args = validate_args(parse_args())
    config = get_megatron_ddp_config(args, use_torch_fsdp2=False)

    assert config.nccl_ub is ub
    assert config.fsdp_manual_registration is (manual or (ub and version == 1))
    assert config.fsdp_double_buffer is (ub and version == 1)
