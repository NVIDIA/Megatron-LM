import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from transformer_engine.pytorch.optimizers import FusedAdam, FusedSGD

from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from megatron.core.optimizer import _get_param_groups


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


@pytest.mark.parametrize('with_param_groups', [False, True])
def test_multi_device_hybrid_optimizer(with_param_groups):
    net = Net().cuda()

    params = list(net.parameters())
    if with_param_groups:
        param_groups = [
            {"params": params[: len(params) // 2],
            "wd_mult": 0.1,
            "lr_mult": 0.1,
            },
            {"params": params[len(params) // 2 :],
            "wd_mult": 0.2,
            "lr_mult": 0.2,},
        ]
        params = param_groups

    hdo = HybridDeviceOptimizer(
        params,
        offload_fraction=0.5,
        cpu_optimizer_cls=Adam,
        gpu_optimizer_cls=FusedAdam,
        lr=0.1,
    )

    # 1. run step on optimizer, make sure there is state generated
    assert len(hdo.state_dict()["state"]) == 0  # state is empty
    input = torch.randn(1, 3, 32, 32).cuda()
    output = net(input)
    output.sum().backward()
    hdo.step()
    assert len(hdo.state_dict()["state"]) != 0

    # 2. check the state is on right device
    first_param_id = hdo.state_dict()["param_groups"][0]["params"][0]
    last_param_id = hdo.state_dict()["param_groups"][-1]["params"][-1]
    assert not hdo.state_dict()["state"][first_param_id]["exp_avg"].is_cuda
    assert hdo.state_dict()["state"][last_param_id]["exp_avg"].is_cuda
