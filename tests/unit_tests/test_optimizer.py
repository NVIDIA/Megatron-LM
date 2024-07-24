import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

from megatron.core.device_utils import get_current_device
from megatron.core.optimizer import ChainedOptimizer


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


def test_chained_optimizer():
    net = Net()
    optimizer_1 = Adam(list(net.parameters())[:2], lr=0.01,)
    optimizer_2 = SGD(list(net.parameters())[2:], lr=0.1, momentum=0.9,)
    chained_optimizer = ChainedOptimizer([optimizer_1, optimizer_2])

    # Test the chained optimizer's param groups is a reference of the underlying optimizers' param groups
    assert optimizer_1.param_groups[0]["lr"] == 0.01
    chained_optimizer.param_groups[0]["lr"] = 0.02
    assert optimizer_1.param_groups[0]["lr"] == 0.02

    # Test the chained optimizer's state is a reference of the underlying optimizers' state
    # 1. run step on optimizers, make sure there is state
    assert len(chained_optimizer.state) == 0
    input = torch.randn(1, 3, 32, 32)
    output = net(input)
    output.sum().backward()
    optimizer_1.step()
    optimizer_2.step()
    assert len(chained_optimizer.state) != 0

    # 2. check the state is a reference
    exp_avg = list(optimizer_1.state.values())[0]["exp_avg"]
    momentum_buffer = list(optimizer_2.state.values())[0]["momentum_buffer"]
    assert not exp_avg.is_cuda and not exp_avg.is_xla
    assert not momentum_buffer.is_cuda and not momentum_buffer.is_xla

    def to_device(d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(device=get_current_device())
            elif isinstance(v, dict):
                to_device(v)
        return d

    for k, v in chained_optimizer.state.items():
        chained_optimizer.state[k] = to_device(v)

    exp_avg = list(optimizer_1.state.values())[0]["exp_avg"]
    momentum_buffer = list(optimizer_2.state.values())[0]["momentum_buffer"]
    assert exp_avg.is_cuda or exp_avg.is_xla
    assert momentum_buffer.is_cuda or momentum_buffer.is_xla
