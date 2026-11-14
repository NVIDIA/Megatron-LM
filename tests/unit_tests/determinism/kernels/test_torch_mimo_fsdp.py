# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact replay of native FSDP shards, clipping, TE Adam and master copy-back."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import torch_mimo_fsdp as adapter
from megatron.core.distributed.torch_mimo_fsdp import TorchMimoFSDP, TorchMimoOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from tests.unit_tests.determinism.kernels.harness import bytes_equal
from tests.unit_tests.determinism.utils import RacingStreams
from tests.unit_tests.distributed.test_torch_mimo_fsdp import Stage
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    adapter.fully_shard is None, reason="Requires native FSDP2 mesh placement support."
)


def test_native_optimizer_step_replays():
    Utils.initialize_distributed()
    if dist.get_world_size() < 2:
        pytest.skip("Requires at least two ranks.")
    try:
        torch.manual_seed(192)
        initial = Stage().cuda().bfloat16()
        initial.scalar.data = initial.scalar.data.float()
        initial.block.routing.data = initial.block.routing.data.float()
        generator = torch.Generator(device="cuda").manual_seed(701 + dist.get_rank())
        inputs = [
            torch.randn(3, 8, device="cuda", dtype=torch.bfloat16, generator=generator)
            for _ in range(12)
        ]
        config = SimpleNamespace(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            gradient_accumulation_fusion=False,
            calculate_per_token_loss=True,
            fp16=False,
        )
        reference = None
        for replay in range(3):
            model = copy.deepcopy(initial)
            # Parameter deepcopy drops custom attributes, including expert ownership.
            model.block.experts.weight.allreduce = False
            wrapper = TorchMimoFSDP(
                config,
                DistributedDataParallelConfig(grad_reduce_in_fp32=True),
                model,
                SimpleNamespace(dp_cp=dist.group.WORLD, expt_dp=dist.group.WORLD),
                units=([model.block], [model.block.experts]),
            )
            optimizer = TorchMimoOptimizer(
                wrapper,
                OptimizerConfig(optimizer="adam", bf16=True, lr=1e-3),
                dist.group.WORLD,
                dist.group.WORLD,
            )
            norms = []
            with RacingStreams():
                for step in range(3):
                    optimizer.zero_grad()
                    for batch in inputs[step * 4 : (step + 1) * 4]:
                        wrapper(batch).float().square().sum().backward()
                    success, norm, _ = optimizer.step()
                    assert success
                    norms.append(norm.detach().clone())
            torch.cuda.synchronize()
            result = [p.to_local().detach().clone() for p in wrapper.parameters()]
            result.extend(norms)
            for master in optimizer.get_parameters():
                result.extend(
                    [
                        master.detach().clone(),
                        optimizer.state[master]["exp_avg"].clone(),
                        optimizer.state[master]["exp_avg_sq"].clone(),
                    ]
                )
            if reference is None:
                reference = result
            else:
                assert len(reference) == len(result)
                for expected, actual in zip(reference, result):
                    assert bytes_equal(
                        expected, actual
                    ), f"Optimizer trajectory differs on replay {replay}"
    finally:
        Utils.destroy_model_parallel()
