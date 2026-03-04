# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import os

import pytest
import torch
from packaging import version

import megatron.core.nccl_allocator as nccl_allocator
from tests.unit_tests.test_utilities import Utils


class TestNCCLAllocator:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.7.0'),
        reason="Requires PyTorch 2.7.0 or later",
    )
    def test_nccl_allocator_init_sets_env_vars(self):
        nccl_allocator.init()
        assert os.environ.get("NCCL_NVLS_ENABLE") == "1"
        assert os.environ.get("TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK") == "0"

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.7.0'),
        reason="Requires PyTorch 2.7.0 or later",
    )
    @pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
    def test_nccl_nccl_mem_register_and_allreduce(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for NCCL allocator tests")

        world_size = torch.distributed.get_world_size()

        device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)

        # Default process group and backend
        pg = torch.distributed.new_group(ranks=list(range(world_size)), backend="nccl")

        nccl_allocator.init()

        # Create mempool via our allocator and register it around allocation
        pool = nccl_allocator.create_nccl_mem_pool()
        with nccl_allocator.nccl_mem(pool, group=pg):
            tensor = torch.ones([1], device=device)

        # Perform an all-reduce to ensure communication works with the pool registered
        torch.distributed.all_reduce(tensor, group=pg)
        torch.cuda.synchronize(device=device)
        assert tensor == torch.tensor([world_size], device=device)

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.7.0'),
        reason="Requires PyTorch 2.7.0 or later",
    )
    @pytest.mark.skipif(torch.cuda.device_count() != 8, reason="Requires 8 GPUs")
    @pytest.mark.skipif(
        torch.cuda.nccl.version() < (2, 27, 0), reason="Requires at least NCCL v2.27.0"
    )
    def test_ag_with_nccl_cta_policy(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for NCCL allocator tests")

        os.environ["NCCL_CTA_POLICY"] = "1"

        world_size = torch.distributed.get_world_size()

        if world_size != 8:
            pytest.skip("Requires 8 ranks")

        device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)

        pg = torch.distributed.new_group(ranks=list(range(world_size)), backend="nccl")

        nccl_allocator.init()

        pool = nccl_allocator.create_nccl_mem_pool()
        target_tensor_numel = 1000000
        with nccl_allocator.nccl_mem(pool, group=pg):
            tensor_shard = torch.ones([target_tensor_numel // world_size], device=device)
            tensor_unshard = torch.ones([target_tensor_numel], device=device)

        torch.distributed.all_gather_into_tensor(tensor_unshard, tensor_shard, group=pg)
        torch.cuda.synchronize(device=device)
