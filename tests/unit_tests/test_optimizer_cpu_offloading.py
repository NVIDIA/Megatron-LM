# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as GPUAdam
    from transformer_engine.pytorch.optimizers import FusedSGD as GPUSGD
except:
    # Handle environment where transformer_engine is not installed
    from torch.optim import SGD as GPUSGD
    from torch.optim import Adam as GPUAdam

from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from megatron.core.transformer.module import (
    convert_module_to_dtype_except_fp32_marked,
    mark_keep_in_fp32,
)


class Fp32MarkedToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.scale = mark_keep_in_fp32(nn.Parameter(torch.ones(4)))

    def forward(self, x):
        return self.proj(x) * self.scale


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BigNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 2048)
        self.fc2 = nn.Linear(2048, 8192)
        self.fc3 = nn.Linear(8192, 2048)
        self.fc4 = nn.Linear(2048, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ConditionalParameterModel(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.always = nn.Parameter(torch.tensor(1.0, device="cuda", dtype=dtype))
        self.optional = nn.Parameter(torch.tensor(1.0, device="cuda", dtype=dtype))

    def forward(self, use_optional):
        output = self.always
        if use_optional:
            output = output + self.optional
        return output


def setup_seed(seed):
    random.seed(seed)  # Set Python's built-in random seed
    np.random.seed(seed)  # Set NumPy's random seed
    torch.manual_seed(seed)  # Set PyTorch's CPU seed
    torch.cuda.manual_seed(seed)  # Set PyTorch's GPU seed (if using CUDA)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner for reproducibility


def test_load_state_dict_with_native_fp32_param():
    """Round-trip state for a BF16 toy net with a parameter marked to stay in FP32."""
    model = Fp32MarkedToyNet().cuda()
    convert_module_to_dtype_except_fp32_marked(model, torch.bfloat16)
    assert model.proj.weight.dtype == torch.bfloat16
    assert model.scale.dtype == torch.float32

    optimizer = HybridDeviceOptimizer(
        model.parameters(),
        offload_fraction=1.0,
        cpu_optimizer_cls=Adam,
        gpu_optimizer_cls=GPUAdam,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=False,
        lr=1e-3,
    )
    inputs = torch.ones(2, 4, device="cuda", dtype=torch.bfloat16)
    model(inputs).sum().backward()
    optimizer.step()

    restored_model = Fp32MarkedToyNet().cuda()
    convert_module_to_dtype_except_fp32_marked(restored_model, torch.bfloat16)
    restored_model.load_state_dict(model.state_dict())
    restored_optimizer = HybridDeviceOptimizer(
        restored_model.parameters(),
        offload_fraction=1.0,
        cpu_optimizer_cls=Adam,
        gpu_optimizer_cls=GPUAdam,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=False,
        lr=1e-3,
    )
    restored_optimizer.load_state_dict(optimizer.state_dict())

    assert set(restored_optimizer.state) == set(restored_model.parameters())
    assert restored_model.proj.weight in restored_optimizer.param_to_fp32_param
    assert restored_model.scale not in restored_optimizer.param_to_fp32_param
    assert torch.equal(
        restored_optimizer.param_to_fp32_param[restored_model.proj.weight],
        optimizer.param_to_fp32_param[model.proj.weight],
    )

    restored_model(inputs).sum().backward()
    restored_optimizer.step()


@pytest.mark.parametrize(
    "dtype,offload_fraction,param_update_in_fp32",
    [(torch.float32, 1.0, False), (torch.bfloat16, 0.0, True)],
    ids=["cpu-offloaded-param", "gpu-fp32-master-param"],
)
def test_conditional_parameter_matches_torch_sgd_reference(
    dtype, offload_fraction, param_update_in_fp32
):
    """HybridDeviceOptimizer must not reuse a previous grad when a param is inactive."""
    model = ConditionalParameterModel(dtype)
    reference_model = ConditionalParameterModel(dtype)
    optimizer = HybridDeviceOptimizer(
        model.parameters(),
        offload_fraction=offload_fraction,
        cpu_optimizer_cls=SGD,
        gpu_optimizer_cls=SGD,
        param_update_in_fp32=param_update_in_fp32,
        overlap_cpu_optimizer_d2h_h2d=False,
        lr=0.25,
    )
    reference_optimizer = SGD(reference_model.parameters(), lr=0.25)

    for use_optional in (True, False, True):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        model(use_optional).backward()
        reference_model(use_optional).backward()
        assert (model.optional.grad is None) == (reference_model.optional.grad is None)

        optimizer.step()
        reference_optimizer.step()
        torch.cuda.synchronize()

        torch.testing.assert_close(model.always, reference_model.always)
        torch.testing.assert_close(model.optional, reference_model.optional)


@pytest.mark.skipif(
    torch.__version__ < '2.3.0',
    reason=(
        "Requires PyTorch 2.3.0 or higher, lower versions of pytorch have "
        "misaligned optimizer accuracy for CPU and GPU."
    ),
)
@pytest.mark.parametrize('n_steps', [1, 10])
@pytest.mark.parametrize('overlap_cpu_optimizer_d2h_h2d', [False, True])
@pytest.mark.parametrize('offload_fraction', [0, 0.5, 1.0])
@pytest.mark.parametrize('optimizer', ['sgd', 'adam'])
@pytest.mark.parametrize('with_param_groups', [False, True])
def test_multi_device_hybrid_optimizer(
    with_param_groups, optimizer, offload_fraction, overlap_cpu_optimizer_d2h_h2d, n_steps
):
    setup_seed(42)
    net1 = Net().cuda()
    net2 = Net().cuda()
    net2.load_state_dict(net1.state_dict())
    base_lr = 1e-3
    params = list(net1.parameters())
    ref_params = list(net2.parameters())
    if with_param_groups:
        param_groups = [
            {"params": params[: len(params) // 2], "wd_mult": 1.0, "lr_mult": 1e-4},
            {"params": params[len(params) // 2 :], "wd_mult": 0.0, "lr_mult": 2e-4},
        ]
        params = param_groups
        ref_param_groups = [
            {"params": ref_params[: len(ref_params) // 2], "wd_mult": 1.0, "lr_mult": 1e-4},
            {"params": ref_params[len(ref_params) // 2 :], "wd_mult": 0.0, "lr_mult": 2e-4},
        ]
        ref_params = ref_param_groups

    if optimizer == 'adam':
        cls_kwargs = dict(cpu_optimizer_cls=Adam, gpu_optimizer_cls=GPUAdam)
    else:
        cls_kwargs = dict(cpu_optimizer_cls=SGD, gpu_optimizer_cls=GPUSGD)

    hdo = HybridDeviceOptimizer(
        params,
        offload_fraction=offload_fraction,
        lr=base_lr,
        overlap_cpu_optimizer_d2h_h2d=overlap_cpu_optimizer_d2h_h2d,
        **cls_kwargs,
    )

    ref_optimizer = cls_kwargs['gpu_optimizer_cls'](ref_params, lr=base_lr)

    # 1. run step on optimizer, make sure there is state generated
    assert len(hdo.state_dict()["state"]) == 0  # state is empty
    input = torch.randn(1, 3, 32, 32).cuda()
    output = net1(input)
    output.sum().backward()
    hdo.step()
    output = net2(input)
    output.sum().backward()
    ref_optimizer.step()
    # PyTorch SGD will not generate state
    if optimizer != 'sgd':
        assert len(hdo.state_dict()["state"]) != 0

    # 2. check the state is on right device
    if optimizer == 'adam':
        first_param_id = hdo.state_dict()["param_groups"][0]["params"][0]
        last_param_id = hdo.state_dict()["param_groups"][-1]["params"][-1]
        if offload_fraction > 0:
            assert not hdo.state_dict()["state"][first_param_id]["exp_avg"].is_cuda
        if offload_fraction < 1:
            assert hdo.state_dict()["state"][last_param_id]["exp_avg"].is_cuda

    # 3. check parameters allclose
    for _ in range(1, n_steps):
        input = torch.randn(1, 3, 32, 32).cuda()
        output = net1(input)
        output.sum().backward()
        hdo.step()
        output = net2(input)
        output.sum().backward()
        ref_optimizer.step()

    params = net1.state_dict()
    ref_params = net2.state_dict()
    for k, v in params.items():
        assert (v.isnan() == ref_params[k].isnan()).all()
        torch.nan_to_num_(v, 0)
        torch.nan_to_num_(ref_params[k], 0)
        assert torch.allclose(
            v, ref_params[k], atol=1e-03
        ), f"Weight {k} value mismatch, max error: {(v - ref_params[k]).abs().max()}"


@pytest.mark.skipif(
    torch.__version__ < '2.3.0',
    reason=(
        "Requires PyTorch 2.3.0 or higher, lower versions of pytorch have "
        "misaligned optimizer accuracy for CPU and GPU."
    ),
)
@pytest.mark.parametrize('n_steps', [1, 10])
@pytest.mark.parametrize('offload_fraction', [1, 0.5, 0])
@pytest.mark.parametrize('optimizer', ['adam', 'sgd'])
@pytest.mark.parametrize('with_param_groups', [False, True])
def test_overlap_cpu_optimizer_d2h_h2d_sync_correctness(
    with_param_groups, optimizer, offload_fraction, n_steps
):
    setup_seed(42)
    net1 = BigNet().cuda()
    net2 = BigNet().cuda()
    net2.load_state_dict(net1.state_dict())
    base_lr = 1e-3
    params = list(net1.parameters())
    ref_params = list(net2.parameters())
    if with_param_groups:
        param_groups = [
            {"params": params[: len(params) // 2], "wd_mult": 1.0, "lr_mult": 1e-4},
            {"params": params[len(params) // 2 :], "wd_mult": 0.0, "lr_mult": 2e-4},
        ]
        params = param_groups
        ref_param_groups = [
            {"params": ref_params[: len(ref_params) // 2], "wd_mult": 1.0, "lr_mult": 1e-4},
            {"params": ref_params[len(ref_params) // 2 :], "wd_mult": 0.0, "lr_mult": 2e-4},
        ]
        ref_params = ref_param_groups

    if optimizer == 'adam':
        cls_kwargs = dict(cpu_optimizer_cls=Adam, gpu_optimizer_cls=GPUAdam)
    else:
        cls_kwargs = dict(cpu_optimizer_cls=SGD, gpu_optimizer_cls=GPUSGD)

    hdo = HybridDeviceOptimizer(
        params,
        offload_fraction=offload_fraction,
        lr=base_lr,
        overlap_cpu_optimizer_d2h_h2d=True,
        **cls_kwargs,
    )

    ref_optimizer = cls_kwargs['gpu_optimizer_cls'](ref_params, lr=base_lr)

    # 1. run step on optimizer, make sure there is state generated
    assert len(hdo.state_dict()["state"]) == 0  # state is empty
    input = torch.randn(1, 3, 32, 32).cuda()
    output = net1(input)
    output.sum().backward()
    hdo.step()
    output = net2(input)
    output.sum().backward()
    ref_optimizer.step()
    # PyTorch SGD will not generate state
    if optimizer != 'sgd':
        assert len(hdo.state_dict()["state"]) != 0

    # 2. check the state is on right device
    if optimizer == 'adam':
        first_param_id = hdo.state_dict()["param_groups"][0]["params"][0]
        last_param_id = hdo.state_dict()["param_groups"][-1]["params"][-1]
        if offload_fraction > 0:
            assert not hdo.state_dict()["state"][first_param_id]["exp_avg"].is_cuda
        if offload_fraction < 1:
            assert hdo.state_dict()["state"][last_param_id]["exp_avg"].is_cuda

    inputs = [torch.randn(1, 3, 32, 32).cuda() for _ in range(1, n_steps)]
    for i in range(1, n_steps):
        output = net1(inputs[i - 1])
        output.sum().backward()
        hdo.step()

    for i in range(1, n_steps):
        output = net2(inputs[i - 1])
        output.sum().backward()
        ref_optimizer.step()

    params = net1.state_dict()
    ref_params = net2.state_dict()
    for k, v in params.items():
        assert (v.isnan() == ref_params[k].isnan()).all()
        torch.nan_to_num_(v, 0)
        torch.nan_to_num_(ref_params[k], 0)
        assert torch.allclose(
            v, ref_params[k], atol=1e-03
        ), f"Weight {k} value mismatch, max error: {(v - ref_params[k]).abs().max()}"
