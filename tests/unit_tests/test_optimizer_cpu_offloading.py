import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from transformer_engine.pytorch.optimizers import FusedSGD, FusedAdam

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


def test_multi_device_hybrid_optimizer():
    net = Net().cuda()

    hdo = HybridDeviceOptimizer(
        list(net.parameters()),
        offload_ratio=0.5,
        cpu_optimizer_cls=Adam,
        gpu_optimizer_cls=FusedAdam,
        lr=0.1,
    )

    # Test the chained optimizer's state is a reference of the underlying optimizers' state
    # 1. run step on optimizers, make sure there is state
    assert len(hdo.state_dict()["state"]) == 0 # state is empty
    input = torch.randn(1, 3, 32, 32).cuda()
    output = net(input)
    output.sum().backward()
    hdo.step()
    assert len(hdo.state_dict()["state"]) != 0

    print(hdo.state_dict())

    # 2. check the state is a reference
    assert not hdo.state_dict()["state"][0]["exp_avg"].is_cuda
    assert hdo.state_dict()["state"][len(net.parameters()) - 1]["exp_avg"].is_cuda

    print(net.parameters())
