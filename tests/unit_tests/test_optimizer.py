import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.test_utils import _deinit_distributed, _init_distributed


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
    optimizer_1 = Adam(list(net.parameters())[:2], lr=0.01)
    optimizer_2 = SGD(list(net.parameters())[2:], lr=0.1, momentum=0.9)
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
    assert not list(optimizer_1.state.values())[0]["exp_avg"].is_cuda
    assert not list(optimizer_2.state.values())[0]["momentum_buffer"].is_cuda

    def to_cuda(d):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to("cuda")
            elif isinstance(v, dict):
                to_cuda(v)
        return d

    for k, v in chained_optimizer.state.items():
        chained_optimizer.state[k] = to_cuda(v)

    assert list(optimizer_1.state.values())[0]["exp_avg"].is_cuda
    assert list(optimizer_2.state.values())[0]["momentum_buffer"].is_cuda


def test_precision_aware_fused_adam():
    try:
        from transformer_engine.pytorch.optimizers import FusedAdam
    except ImportError:
        # Older versions of TE don't have FusedAdam.
        return

    import inspect

    adam_args = inspect.signature(FusedAdam).parameters
    arg_names = ["master_weight_dtype", "exp_avg_dtype", "exp_avg_sq_dtype", "use_decoupled_grad"]
    for name in arg_names:
        if name not in adam_args:
            # Skip the test if TE doesn't support precision aware FusedAdam.
            return

    tensor = torch.rand(278011, dtype=torch.bfloat16).cuda()
    params_1 = [torch.nn.Parameter(tensor.float())]  # FP32 reference
    params_2 = [torch.nn.Parameter(tensor.clone())]  # BF16

    options = {"lr": 1, "betas": (0.1, 0.25), "eps": 1e-08, "weight_decay": 0, "amsgrad": False}

    optimizer_1 = FusedAdam(params_1, **options)
    optimizer_2 = FusedAdam(params_2, master_weights=True, use_decoupled_grad=True, **options)

    for _ in range(1000):
        for p_1, p_2 in zip(params_1, params_2):
            p_1.grad = torch.rand_like(p_1)
            p_2.decoupled_grad = p_1.grad.clone()

        optimizer_1.step()
        optimizer_2.step()

        master_params = [optimizer_2.get_unscaled_state(p, "master_param") for p in params_2]
        for p_1, p_2 in zip(params_1, master_params):
            bytes_1 = p_1.data.view(torch.uint8)
            bytes_2 = p_2.data.view(torch.uint8)
            # Make sure bit-wise matched
            assert torch.all(bytes_1 == bytes_2)

        for p_1, p_2 in zip(params_1, params_2):
            bytes_1 = p_1.data.bfloat16().view(torch.uint8)
            bytes_2 = p_2.data.view(torch.uint8)
            # Make sure bit-wise matched
            assert torch.all(bytes_1 == bytes_2)


@pytest.mark.skipif(
    not is_te_min_version("1.13.0"), reason="TE 1.13.0 is required for precision aware optimizer"
)
@pytest.mark.parametrize(
    "exp_avg_dtype", [torch.float32, torch.float16, torch.bfloat16, torch.uint8]
)
@pytest.mark.parametrize(
    "exp_avg_sq_dtype", [torch.float32, torch.float16, torch.bfloat16, torch.uint8]
)
def test_precision_aware_optimizer(exp_avg_dtype: str, exp_avg_sq_dtype: str):
    # Skip because bf16 optimizer states are not supported before TE 2.3.0
    if (
        exp_avg_dtype == torch.bfloat16 or exp_avg_sq_dtype == torch.bfloat16
    ) and not is_te_min_version("2.3.0"):
        pytest.skip("bfloat16 for exp_avg_dtype/exp_avg_sq_dtype requires TE >= 2.3.0")

    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    # Setup: distributed, model, mock_args.
    _init_distributed(world, rank)
    Utils.initialize_model_parallel()
    model = torch.nn.Linear(100, 100, bias=False, dtype=torch.bfloat16, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )
    for param in model.parameters():
        assert param.requires_grad

    optimizer_config = OptimizerConfig(
        optimizer='adam',
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        exp_avg_dtype=exp_avg_dtype,
        exp_avg_sq_dtype=exp_avg_sq_dtype,
    )
    optim = get_megatron_optimizer(optimizer_config, [model])

    # Run a forward and backward pass
    input = torch.randn(8, 100, dtype=torch.bfloat16, device='cuda')
    output = model(input)
    loss = output.sum()
    loss.backward()

    # Step the optimizer
    optim.step()

    # Save and reload state dict
    state_dict = optim.state_dict()
    optim.load_state_dict(state_dict)


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("precision", ['bf16', 'fp32'])
def test_optim_sharded_state_dict(use_distributed_optimizer: bool, precision: str):
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    # Setup: distributed, model, mock_args.
    _init_distributed(world, rank)
    Utils.initialize_model_parallel()
    model = torch.nn.Linear(100, 100, bias=False, dtype=torch.bfloat16, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=use_distributed_optimizer)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )
    for param in model.parameters():
        assert param.requires_grad

    if precision == 'bf16':
        optimizer_config = OptimizerConfig(
            optimizer='adam', bf16=True, use_distributed_optimizer=use_distributed_optimizer
        )
    elif precision == 'fp32':
        optimizer_config = OptimizerConfig(
            optimizer='adam',
            bf16=False,
            fp16=False,
            use_distributed_optimizer=use_distributed_optimizer,
        )
    optim = get_megatron_optimizer(optimizer_config, [model])

    model_sharded_state_dict = model.sharded_state_dict()
    sharded_state_dict = optim.sharded_state_dict(model_sharded_state_dict)

    if 'optimizer' in sharded_state_dict and 'state' in sharded_state_dict['optimizer']:
        assert (
            'common_step' not in sharded_state_dict['optimizer']['state']
            or sharded_state_dict['optimizer']['state']['common_step'] is not None
        ), "Found 'optimizer.state.common_step=None' in sharded state dict."
