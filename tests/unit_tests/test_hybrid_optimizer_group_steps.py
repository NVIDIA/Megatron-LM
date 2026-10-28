# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import copy

import pytest
import torch

from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from tests.unit_tests.test_utilities import Utils


def test_hybrid_fused_adam_group_steps_after_resume():
    """Resuming mixed CPU/GPU groups must preserve subsequent Adam updates."""
    fused_adam = pytest.importorskip("transformer_engine.pytorch.optimizers").FusedAdam
    Utils.initialize_distributed()

    def make_optimizer(params):
        return HybridDeviceOptimizer(
            [{"params": params[:1]}, {"params": params[1:3]}, {"params": params[3:]}],
            offload_fraction=0.5,
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=fused_adam,
            param_update_in_fp32=True,
            overlap_cpu_optimizer_d2h_h2d=False,
            lr=0.01,
            weight_decay=0.1,
        )

    def update(optimizer, params, step):
        optimizer.zero_grad()
        for index, param in enumerate(params):
            param.grad = (
                torch.linspace(-0.4, 0.7, param.numel(), device="cuda") * (step + index)
                + 0.1 * step
            ).to(param.dtype)
        # Scheduler changes must still propagate from HDO to every sub-optimizer.
        for group_id, group in enumerate(optimizer.param_groups):
            group["lr"] = 0.01 / (step + group_id)
            group["weight_decay"] = 0.02 * (step + group_id)
        optimizer.step()
        torch.cuda.synchronize()
        for inner_optimizer in optimizer.sub_optimizers:
            for inner_group in inner_optimizer.param_groups:
                original = optimizer.inner_param_to_orig_param[inner_group["params"][0]]
                parent_group = next(
                    group
                    for group in optimizer.param_groups
                    if any(param is original for param in group["params"])
                )
                assert inner_group["lr"] == parent_group["lr"]
                assert inner_group["weight_decay"] == parent_group["weight_decay"]

    params = [
        torch.nn.Parameter((torch.arange(16, device="cuda") / 32 + index).to(torch.bfloat16))
        for index in range(4)
    ]
    optimizer = make_optimizer(params)
    # Parent groups are CPU-only, mixed, and GPU-only respectively. Their indices
    # therefore cannot be zipped with the two GPU optimizer groups.
    assert len(optimizer.gpu_optimizer.param_groups) == 2
    assert params[0] in optimizer.gpu_params_map_cpu_copy
    assert params[1] in optimizer.gpu_params_map_cpu_copy
    assert params[2] not in optimizer.gpu_params_map_cpu_copy
    update(optimizer, params, 1)

    restored_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    checkpoint = copy.deepcopy(optimizer.state_dict())
    # DistributedOptimizer restores this metadata from the CPU Adam state.
    for group in checkpoint["param_groups"]:
        group["step"] = 1
    restored_optimizer = make_optimizer(restored_params)
    restored_optimizer.load_state_dict(checkpoint)

    for step in (2, 3):
        update(optimizer, params, step)
        update(restored_optimizer, restored_params, step)
        for param, restored_param in zip(params, restored_params):
            assert torch.equal(param, restored_param)
            state = optimizer.state[param]
            restored_state = restored_optimizer.state[restored_param]
            assert state.keys() == restored_state.keys()
            for key in state:
                assert torch.equal(state[key], restored_state[key]), key
        for candidate in (optimizer, restored_optimizer):
            assert [group["step"] for group in candidate.gpu_optimizer.param_groups] == [step] * 2
            assert [group["step"] for group in candidate.param_groups[1:]] == [step] * 2
