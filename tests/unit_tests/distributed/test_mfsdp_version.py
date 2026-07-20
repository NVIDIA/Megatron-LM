# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for selecting the Megatron-FSDP implementation by configuration."""

from argparse import ArgumentParser, Namespace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp import mcore_fsdp_adapter
from megatron.training.arguments import add_megatron_arguments
from megatron.training.training import get_megatron_ddp_config


def test_fully_sharded_data_parallel_dispatches_by_config(monkeypatch):
    class V1:
        def __init__(self, *args):
            self.args = args

    class V2:
        def __init__(self, *args):
            self.args = args

    monkeypatch.setattr(mcore_fsdp_adapter, "FullyShardedDataParallelV1", V1)
    monkeypatch.setattr(mcore_fsdp_adapter, "FullyShardedDataParallelV2", V2)

    assert isinstance(
        mcore_fsdp_adapter.FullyShardedDataParallel(None, DistributedDataParallelConfig(), None), V1
    )
    assert isinstance(
        mcore_fsdp_adapter.FullyShardedDataParallel(
            None, DistributedDataParallelConfig(megatron_fsdp_version=2), None
        ),
        V2,
    )


def test_invalid_megatron_fsdp_version_raises_early():
    with pytest.raises(ValueError, match="megatron_fsdp_version must be either 1 or 2"):
        DistributedDataParallelConfig(megatron_fsdp_version=3)


def test_megatron_fsdp_version_cli_argument():
    parser = ArgumentParser()
    add_megatron_arguments(parser)

    assert parser.parse_args(["--megatron-fsdp-version", "2"]).megatron_fsdp_version == 2


def test_get_megatron_ddp_config_maps_megatron_fsdp_version():
    args = Namespace(
        use_torch_fsdp2=False,
        megatron_fsdp_version=2,
        accumulate_allreduce_grads_in_fp32=False,
        check_for_nan_in_loss_and_grad=False,
        check_for_large_grads=False,
        ddp_num_buckets=None,
        ddp_bucket_size=None,
        ddp_pad_buckets_for_high_nccl_busbw=False,
        ddp_reduce_scatter_with_fp32_accumulation=False,
        ddp_param_name_patterns_for_fp32_local_accumulation=[],
        ddp_average_in_collective=False,
        megatron_fsdp_main_params_dtype=torch.float32,
        megatron_fsdp_main_grads_dtype=None,
        megatron_fsdp_grad_comm_dtype=None,
        use_precision_aware_optimizer=False,
        use_megatron_fsdp=True,
        cuda_graph_impl="none",
    )

    ddp_config = get_megatron_ddp_config(args)

    assert ddp_config.megatron_fsdp_version == 2
    assert not ddp_config.fsdp_all_gather_in_start_param_sync
