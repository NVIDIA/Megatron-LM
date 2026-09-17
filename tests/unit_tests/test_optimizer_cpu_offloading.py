# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import random
from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW

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


@pytest.mark.parametrize('overlap', [False, True])
@pytest.mark.parametrize('gpu_dtype', [torch.bfloat16, torch.float32])
@pytest.mark.skipif(GPUAdam is Adam, reason='Requires TransformerEngine FusedAdam')
def test_hybrid_decoupled_gradient_and_missing_grad(overlap: bool, gpu_dtype: torch.dtype) -> None:
    """Route native FP32 gradients and skip absent gradients on both devices."""
    cpu_owned = nn.Parameter(torch.ones(64, device='cuda', dtype=torch.bfloat16))
    gpu_owned = nn.Parameter(torch.ones(64, device='cuda', dtype=gpu_dtype))
    reference = nn.Parameter(gpu_owned.detach().float().clone())
    optimizer = HybridDeviceOptimizer(
        [cpu_owned, gpu_owned],
        offload_fraction=0.5,
        cpu_optimizer_cls=AdamW,
        gpu_optimizer_cls=GPUAdam,
        param_update_in_fp32=True,
        overlap_cpu_optimizer_d2h_h2d=overlap,
        lr=0.1,
        weight_decay=0.01,
        fused=True,
    )
    reference_optimizer = GPUAdam([reference], lr=0.1, weight_decay=0.01)
    assert cpu_owned in optimizer.gpu_params_map_cpu_copy
    assert gpu_owned not in optimizer.gpu_params_map_cpu_copy
    assert (gpu_owned in optimizer.param_to_fp32_param) == (gpu_dtype != torch.float32)

    for value in (1.0, None, 0.5):
        before = [parameter.detach().clone() for parameter in (cpu_owned, gpu_owned)]
        gradient = None if value is None else torch.full_like(reference, value)
        cpu_owned.decoupled_grad = gradient
        gpu_owned.decoupled_grad = gradient
        reference.grad = None if gradient is None else gradient.clone()
        optimizer.step()
        reference_optimizer.step()
        torch.cuda.synchronize()
        assert torch.equal(gpu_owned, reference.to(gpu_dtype)), "GPU gradient was not consumed"
        assert gpu_owned.requires_grad, "An absent gradient must not freeze the model parameter"
        for old, parameter in zip(before, (cpu_owned, gpu_owned)):
            if value is None:
                assert torch.equal(old, parameter), "Absent gradient reused a stale child buffer"
            else:
                assert not torch.equal(
                    old, parameter
                ), "A supplied gradient did not update its owner"


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('overlap', [False, True])
@pytest.mark.skipif(GPUAdam is Adam, reason='Requires TransformerEngine FusedAdam')
def test_hybrid_state_dict_preserves_gpu_steps(dtype: torch.dtype, overlap: bool) -> None:
    """A native checkpoint must preserve more than the first resumed update."""
    initial = [
        torch.linspace(-0.5, 0.5, size, device='cuda').to(dtype) for size in (65, 93, 33, 101)
    ]

    def construct():
        parameters = [nn.Parameter(value.clone()) for value in initial]
        optimizer = HybridDeviceOptimizer(
            [{'params': parameters[:3]}, {'params': parameters[3:]}],
            offload_fraction=0.5,
            cpu_optimizer_cls=AdamW,
            gpu_optimizer_cls=GPUAdam,
            param_update_in_fp32=True,
            overlap_cpu_optimizer_d2h_h2d=overlap,
            lr=0.01,
            fused=True,
        )
        assert optimizer.cpu_optimizers and optimizer.gpu_optimizer is not None
        return parameters, optimizer

    def step(parameters, optimizer, index):
        for parameter in parameters:
            gradient = torch.full_like(parameter, index * 0.125, dtype=torch.float32)
            if dtype == torch.bfloat16:
                parameter.decoupled_grad = gradient
            else:
                parameter.grad = gradient
        optimizer.step()

    parameters, optimizer = construct()
    for index in (1, 2, 3):
        step(parameters, optimizer, index)
    torch.cuda.synchronize()
    checkpoint = deepcopy(optimizer.state_dict())
    restored_parameters, restored_optimizer = construct()
    for old, new in zip(parameters, restored_parameters):
        new.data.copy_(old.data)
    restored_optimizer.load_state_dict(checkpoint)

    for index in (4, 5):
        step(parameters, optimizer, index)
        step(restored_parameters, restored_optimizer, index)
        torch.cuda.synchronize()
        for child in (optimizer.gpu_optimizer, restored_optimizer.gpu_optimizer):
            assert all(group['step'] == index for group in child.param_groups)
        for left, right in zip(parameters, restored_parameters):
            assert torch.equal(left, right), "Native resume changed parameter bytes"
            left_state, right_state = optimizer.state[left], restored_optimizer.state[right]
            assert left_state.keys() == right_state.keys()
            for key in left_state:
                a, b = left_state[key], right_state[key]
                assert torch.equal(
                    a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
                ), f"Native resume changed {key} bytes"


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


def test_distributed_optimizer_with_cpu_offload_and_fp32_marked_param():
    """Test that DistributedOptimizer works with HybridDeviceOptimizer (CPU offloading)
    when the model has mark_keep_in_fp32 parameters without raising non-leaf Tensor ValueError.
    """
    import os

    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.transformer import TransformerConfig
    from tests.unit_tests.test_utilities import Utils
    from tests.unit_tests.test_utils import _init_distributed

    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    _init_distributed(world, rank)
    Utils.initialize_model_parallel()

    try:
        model = Fp32MarkedToyNet().cuda()
        convert_module_to_dtype_except_fp32_marked(model, torch.bfloat16)
        assert model.proj.weight.dtype == torch.bfloat16
        assert model.scale.dtype == torch.float32

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
        transformer_config = TransformerConfig(num_attention_heads=1, num_layers=1)
        ddp_model = DistributedDataParallel(transformer_config, ddp_config, model)

        optimizer_config = OptimizerConfig(
            optimizer='adam',
            lr=1e-3,
            bf16=True,
            use_distributed_optimizer=True,
            optimizer_cpu_offload=True,
            optimizer_offload_fraction=1.0,
        )

        optimizer = get_megatron_optimizer(optimizer_config, [ddp_model])

        inputs = torch.ones(2, 4, device="cuda", dtype=torch.bfloat16)
        output = ddp_model(inputs)
        loss = output.sum()
        loss.backward()

        update_successful, grad_norm, _ = optimizer.step()
        assert update_successful
    finally:
        Utils.destroy_model_parallel()
