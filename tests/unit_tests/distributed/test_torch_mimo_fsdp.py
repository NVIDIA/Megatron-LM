# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Two-GPU numerical checks, independent of full-model data/checkpoints."""

import copy
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import torch_mimo_fsdp as adapter
from megatron.core.distributed.torch_mimo_fsdp import (
    TorchMimoFSDP,
    TorchMimoOptimizer,
    _backward_boundary,
)
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    adapter.fully_shard is None, reason="Requires native FSDP2 mesh placement support."
)


class Expert(nn.Linear):
    def __init__(self):
        super().__init__(8, 8, bias=False)
        self.weight.allreduce = False


class Shortcut(nn.Module):
    def __init__(self):
        super().__init__()
        self.compute = nn.Linear(8, 8, bias=False)
        self.experts = Expert()
        self.routing = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Like ShortcutMoEBlock, bypass the compute child's __call__.
        return self.compute.forward(x).sin() + self.experts(x).tanh() * self.routing.to(x.dtype)


class Stage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(8, 8, bias=False)
        self.block = Shortcut()
        # An uneven first dimension exercises empty local shards on rank 1.
        self.scalar = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # MIMO reads weights before entering the decoder's forward.
        x = torch.nn.functional.linear(x, self.embedding.weight)
        x = checkpoint(self.block, x, use_reentrant=False)
        # Shared MTP module called twice, tied embedding/output weight.
        x = self.block(x)
        return torch.nn.functional.linear(x, self.embedding.weight.T) * self.scalar


def full_grad(p):
    return None if p.grad is None else p.grad.full_tensor()


@pytest.fixture(autouse=True)
def distributed_stage():
    Utils.initialize_distributed()
    if dist.get_world_size() < 2:
        pytest.skip("Torch MIMO sharding tests require at least two ranks.")
    yield
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("expert_parallel", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize(
    "dtype,mixed", [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)]
)
def test_numerical_step(expert_parallel, per_token, dtype, mixed):
    rank = dist.get_rank()
    world = dist.get_world_size()
    ep = world if expert_parallel else 1
    singleton_groups = [dist.new_group([r]) for r in range(world)]
    expert_group = dist.group.WORLD if ep == 1 else singleton_groups[rank]
    torch.manual_seed(123)
    model = Stage().cuda().to(dtype=dtype)
    if mixed:
        model.scalar.data = model.scalar.data.float()
        model.block.routing.data = model.block.routing.data.float()
    original_dtypes = {name: p.dtype for name, p in model.named_parameters()}
    # All replicas of one expert must start equal; different experts need not.
    if ep > 1:
        with torch.no_grad():
            model.block.experts.weight.add_(rank * 0.05)
    reference = copy.deepcopy(model)
    cfg = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep,
        calculate_per_token_loss=per_token,
        gradient_accumulation_fusion=False,
        fp16=False,
    )
    wrapper = TorchMimoFSDP(
        cfg,
        DistributedDataParallelConfig(grad_reduce_in_fp32=True),
        model,
        SimpleNamespace(dp_cp=dist.group.WORLD, expt_dp=expert_group),
        units=([model.block], [model.block.experts]),
    )
    opt_cfg = OptimizerConfig(
        optimizer="adam",
        bf16=True,
        params_dtype=dtype,
        lr=1e-3,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.1,
        clip_grad=1.0,
        use_precision_aware_optimizer=False,
    )
    optimizer = TorchMimoOptimizer(wrapper, opt_cfg, dist.group.WORLD, dist.group.WORLD)
    reference_masters = [nn.Parameter(p.detach().float().clone()) for p in reference.parameters()]
    reference_opt = torch.optim.AdamW(
        [
            {"params": [master], "weight_decay": 0.0 if p.ndim == 1 else 0.1}
            for p, master in zip(reference.parameters(), reference_masters)
        ],
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    for iteration in range(3):
        optimizer.zero_grad()
        for p in reference.parameters():
            p.grad = None
        # Reduce each microbatch in FP32 and accumulate in parameter dtype,
        # matching native FSDP2 rather than all-reducing a BF16 accumulated grad.
        expected = {name: torch.zeros_like(p) for name, p in reference.named_parameters()}
        for microbatch in range(4):
            torch.manual_seed(700 + 40 * iteration + microbatch + 1000 * rank)
            x = torch.randn(3, 8, device="cuda", dtype=dtype)
            wrapper(x).float().square().sum().backward()
            reference(x).float().square().sum().backward()
            for name, p in reference.named_parameters():
                grad = p.grad.float()
                dist.all_reduce(grad, group=expert_group if "experts" in name else dist.group.WORLD)
                if not per_token:
                    grad.div_(world)
                expected[name].add_(grad.to(p.dtype))
                p.grad = None
        tolerance = 0.035 if dtype == torch.bfloat16 else 2e-6
        for name, p in model.named_parameters():
            assert p.dtype == original_dtypes[name]
            torch.testing.assert_close(full_grad(p), expected[name], rtol=tolerance, atol=tolerance)
        scale = 0.125
        wrapper.scale_gradients(scale)
        optimizer.prepare_grads()
        norm_sq = torch.zeros((), device="cuda")
        for (name, _), master in zip(reference.named_parameters(), reference_masters):
            master.grad = (expected[name] * scale).float()
            # Dense weights are duplicated in this unsharded reference.
            norm_sq += master.grad.square().sum() / (1 if "experts" in name and ep > 1 else world)
        dist.all_reduce(norm_sq)
        norm = optimizer.get_grad_norm()
        torch.testing.assert_close(norm, norm_sq.sqrt(), rtol=tolerance, atol=tolerance)
        clip = min(1.0, 1.0 / (float(norm) + 1e-6))
        ref_clip = min(1.0, 1.0 / (float(norm_sq.sqrt()) + 1e-6))
        for master in optimizer.get_parameters():
            if master.grad is not None:
                master.grad.mul_(clip)
        for master in reference_masters:
            master.grad.mul_(ref_clip)
        optimizer.step_with_ready_grads()
        reference_opt.step()
        with torch.no_grad():
            for p, master in zip(reference.parameters(), reference_masters):
                p.copy_(master)
        for (name, p), ref in zip(model.named_parameters(), reference.parameters()):
            torch.testing.assert_close(p.full_tensor(), ref, rtol=tolerance, atol=tolerance)
        # Nonempty shards retain FP32 optimizer parameters and Adam moments.
        for p in optimizer.get_parameters():
            assert p.dtype == torch.float32
            assert optimizer.state[p]["exp_avg"].dtype == torch.float32
        dist.barrier()
    if rank == 0:
        print(
            f"PASS EP={ep} token_sum={per_token} dtype={dtype} mixed={mixed}: 3 updates x 4 microbatches",
            flush=True,
        )


def test_backward_boundary_preserves_bridge_metadata():
    sample = torch.randn(4, device="cuda", requires_grad=True)
    sample._mimo_bridge_split_sizes = [1, 3]
    boundary = _backward_boundary(sample)
    assert boundary._mimo_bridge_split_sizes == [1, 3]
    boundary.sum().backward()
    torch.testing.assert_close(sample.grad, torch.ones_like(sample))
