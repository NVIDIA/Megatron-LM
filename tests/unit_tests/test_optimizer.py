import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

# FP8 recipe will be used to test precision-aware-optimizer.
from transformer_engine.pytorch.fp8 import fp8_autocast

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.test_utils import _deinit_distributed, _init_distributed

try:
    # Check if FP8 block scaling is available.
    from transformer_engine.pytorch.fp8 import check_fp8_block_scaling_support

    fp8_block_scaling_available, reason_for_no_fp8_block_scaling = check_fp8_block_scaling_support()
    from transformer_engine.common.recipe import Float8BlockScaling, Format
except:
    fp8_block_scaling_available = False
    reason_for_no_fp8_block_scaling = "FP8 block scaled GEMM requires Hopper and CUDA >= 12.9."
    try:
        from transformer_engine.common.recipe import DelayedScaling
    except:
        delayed_scaling_available = False


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
@pytest.mark.parametrize("precision", ['bf16', 'fp8'])
@pytest.mark.parametrize("main_params_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("main_grads_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    # use the same dtype for exp_avg and exp_avg_sq to reduce the number of tests
    "moment_dtype",
    [torch.float32, torch.float16, torch.bfloat16, torch.uint8],
)
def test_precision_aware_optimizer(
    precision: str,
    main_params_dtype: torch.dtype,
    main_grads_dtype: torch.dtype,
    moment_dtype: torch.dtype,
):
    # Skip because bf16 optimizer states are not supported before TE 2.3.0
    if (moment_dtype == torch.bfloat16) and not is_te_min_version("2.3.0"):
        pytest.skip("bfloat16 for moment_dtype requires TE >= 2.3.0")

    if precision == 'fp8':
        if not fp8_block_scaling_available:
            fp8_recipe = "delayed"
            fp8_recipe_settings = DelayedScaling()
        else:
            fp8_recipe = "blockwise"
            fp8_recipe_settings = Float8BlockScaling(fp8_format=Format.E4M3)
    else:
        fp8_recipe = None
        fp8_recipe_settings = None

    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    # Setup: distributed, model, mock_args.
    _init_distributed(world, rank)
    Utils.initialize_model_parallel()

    # First create baseline model with float32 optimizer states
    baseline_model = torch.nn.Linear(100, 100, bias=False, dtype=torch.bfloat16, device='cuda')
    baseline_model.requires_grad_(True)
    baseline_model.weight.data.fill_(1.0)
    baseline_ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    baseline_model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), baseline_ddp_config, baseline_model
    )
    baseline_optimizer_config = OptimizerConfig(
        optimizer='adam',
        lr=0.01,
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=False,
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    )
    baseline_optim = get_megatron_optimizer(baseline_optimizer_config, [baseline_model])

    # Create test model with specified dtypes for optimizer states
    test_model = torch.nn.Linear(100, 100, bias=False, dtype=torch.bfloat16, device='cuda')
    test_model.requires_grad_(True)
    test_model.weight.data.fill_(1.0)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    test_model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, test_model
    )
    test_optimizer_config = OptimizerConfig(
        optimizer='adam',
        lr=0.01,
        bf16=True,
        fp8_recipe=fp8_recipe,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        main_params_dtype=main_params_dtype,
        main_grads_dtype=main_grads_dtype,
        exp_avg_dtype=moment_dtype,
        exp_avg_sq_dtype=moment_dtype,
    )
    test_optim = get_megatron_optimizer(test_optimizer_config, [test_model])

    # Use same input for both models
    input = torch.randn(8, 100, dtype=torch.bfloat16, device='cuda')

    # Run model
    def run_model(model, input, optim, fp8_recipe, fp8_recipe_settings):
        if not fp8_recipe:
            output = model(input)
        else:
            with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe_settings):
                output = model(input)
        loss = output.sum()
        loss.backward()
        optim.step()
        return loss.item(), optim.get_grad_norm()

    # Run baseline model and test model
    baseline_loss, baseline_grad_norm = run_model(
        baseline_model, input, baseline_optim, fp8_recipe, fp8_recipe_settings
    )
    test_loss, test_grad_norm = run_model(
        test_model, input, test_optim, fp8_recipe, fp8_recipe_settings
    )

    rtol = 1e-3  # relative tolerance
    atol = 1e-5  # absolute tolerance

    # Compare grad norms - allow small difference due to precision
    rel_diff = abs(test_grad_norm - baseline_grad_norm) / (
        abs(baseline_grad_norm) + 1e-7  # avoid div by 0
    )
    abs_diff = abs(test_grad_norm - baseline_grad_norm)
    assert (
        rel_diff <= rtol or abs_diff <= atol
    ), f"Grad norm mismatch: baseline={baseline_grad_norm}, test={test_grad_norm}, rel_diff={rel_diff}, abs_diff={abs_diff}"

    # Compare losses - allow small difference due to precision
    loss_rel_diff = abs(test_loss - baseline_loss) / (abs(baseline_loss) + 1e-7)
    loss_abs_diff = abs(test_loss - baseline_loss)
    assert (
        loss_rel_diff <= rtol or loss_abs_diff <= atol
    ), f"Loss mismatch: baseline={baseline_loss}, test={test_loss}, rel_diff={loss_rel_diff}, abs_diff={loss_abs_diff}"

    # Save and reload state dict for the test model
    state_dict = test_optim.state_dict()
    test_optim.load_state_dict(state_dict)


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
