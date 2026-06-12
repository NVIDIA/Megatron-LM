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


def setup_seed(seed):
    random.seed(seed)  # Set Python's built-in random seed
    np.random.seed(seed)  # Set NumPy's random seed
    torch.manual_seed(seed)  # Set PyTorch's CPU seed
    torch.cuda.manual_seed(seed)  # Set PyTorch's GPU seed (if using CUDA)
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner for reproducibility


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
@pytest.mark.parametrize('offload_fraction', [0.5, 1.0])
@pytest.mark.parametrize('param_update_in_fp32', [True, False])
def test_reload_model_params_reseeds_internal_copies(offload_fraction, param_update_in_fp32):
    """``reload_model_params`` must re-seed the optimizer's internal copies.

    Regression test for the bug where loading a checkpoint without the
    optimizer state (fine-tuning) left ``HybridDeviceOptimizer``'s pinned CPU
    clones (``gpu_params_map_cpu_copy``) and FP32 working copies
    (``param_to_fp32_param``) at their stale (random-init) values. The first
    ``step()`` would then copy those stale values back onto the model
    parameters, silently discarding the freshly loaded weights.
    See NVIDIA-NeMo/RL#2371.
    """
    setup_seed(42)
    # FP32 working copies only exist for non-FP32 params, so use BF16 when
    # exercising the ``param_update_in_fp32`` path.
    dtype = torch.bfloat16 if param_update_in_fp32 else torch.float32
    net = Net().cuda().to(dtype)
    params = list(net.parameters())

    hdo = HybridDeviceOptimizer(
        params,
        offload_fraction=offload_fraction,
        lr=1e-3,
        cpu_optimizer_cls=Adam,
        gpu_optimizer_cls=GPUAdam,
        param_update_in_fp32=param_update_in_fp32,
        overlap_cpu_optimizer_d2h_h2d=False,
    )

    # Run one step so the internal copies and optimizer state are populated.
    net(torch.randn(1, 3, 32, 32).cuda().to(dtype)).sum().backward()
    hdo.step()
    hdo.zero_grad()

    if param_update_in_fp32:
        assert len(hdo.param_to_fp32_param) > 0
    if offload_fraction > 0:
        assert len(hdo.gpu_params_map_cpu_copy) > 0

    def _matches_param(internal_copy, orig_param):
        return torch.allclose(
            internal_copy.to(device=orig_param.device, dtype=orig_param.dtype),
            orig_param,
            atol=1e-2,
        )

    # Simulate a fine-tune checkpoint load: overwrite the model parameters in
    # place with brand new values without touching the optimizer's copies.
    with torch.no_grad():
        for param in params:
            param.copy_(torch.randn_like(param))

    # Before reload, the internal copies are stale (do not match new params).
    for orig_param, cpu_copy in hdo.gpu_params_map_cpu_copy.items():
        assert not _matches_param(cpu_copy, orig_param)
    for orig_param, fp32_param in hdo.param_to_fp32_param.items():
        assert not _matches_param(fp32_param, orig_param)

    # Reload re-seeds every internal copy from the current parameter values.
    hdo.reload_model_params()

    for orig_param, cpu_copy in hdo.gpu_params_map_cpu_copy.items():
        assert _matches_param(cpu_copy, orig_param), "CPU clone not re-seeded on reload"
    for orig_param, fp32_param in hdo.param_to_fp32_param.items():
        assert _matches_param(fp32_param, orig_param), "FP32 copy not re-seeded on reload"


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
