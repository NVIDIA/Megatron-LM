import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.muon import TensorParallelMuon, get_megatron_muon_optimizer
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.test_utils import _deinit_distributed, _init_distributed


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip muon optimizer for LTS test",
)
def test_muon_optimizer_smoke():
    """Smoke test for TensorParallelMuon optimizer."""
    # Create a simple linear model for testing
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Create TensorParallelMuon optimizer
    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum_beta=0.95,
        use_nesterov=True,
        weight_decay=0.01,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        pg_collection=None,
        mode="duplicated",
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test forward and backward pass
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original weight
    original_weight = model.weight.data.clone()

    # Test optimizer step
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # Test zero_grad
    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert 'state' in state_dict, "State dict should contain state"
    assert 'param_groups' in state_dict, "State dict should contain param_groups"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)


@pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip muon optimizer for LTS test",
)
def test_get_megatron_muon_optimizer_smoke():
    """Smoke test for get_megatron_muon_optimizer function."""
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    # Setup: distributed, model
    _init_distributed(world, rank)
    Utils.initialize_model_parallel()

    # Create a model with both linear and non-linear parameters
    model = Net().bfloat16().cuda()
    model.requires_grad_(True)

    # Wrap in DDP (required for Megatron optimizer)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )

    # Ensure all parameters require gradients
    for param in model.parameters():
        assert param.requires_grad, "All parameters should require gradients"

    # Create optimizer config for Muon
    optimizer_config = OptimizerConfig(
        optimizer='muon',  # This will be changed internally to 'adam' for non-linear params
        lr=0.01,
        weight_decay=0.01,
        bf16=True,
        use_distributed_optimizer=False,  # Muon doesn't support distributed optimizer
        muon_momentum=0.95,
        muon_use_nesterov=True,
        muon_fp32_matmul_prec="medium",
        muon_num_ns_steps=5,
        muon_scale_mode="spectral",
        muon_tp_mode="duplicated",
    )

    # Test creating the optimizer
    optimizer = get_megatron_muon_optimizer(
        config=optimizer_config,
        model_chunks=[model],
        use_gloo_process_groups=True,
        layer_wise_distributed_optimizer=False,
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
    assert len(optimizer.chained_optimizers) >= 1, "Should have at least one chained optimizer"

    # Test forward and backward pass
    input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original parameters
    original_params = {}
    for name, param in model.named_parameters():
        original_params[name] = param.data.clone()

    # Test optimizer step
    optimizer.step()

    # Verify at least some parameters were updated
    params_updated = 0
    for name, param in model.named_parameters():
        if not torch.equal(param.data, original_params[name]):
            params_updated += 1

    assert params_updated > 0, "At least some parameters should be updated after optimizer step"

    # Test zero_grad
    optimizer.zero_grad()
    for param in model.parameters():
        assert param.grad is None or torch.all(
            param.grad == 0
        ), f"Gradients should be zeroed for all parameters"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert isinstance(state_dict, list), "State dict should be a list"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)

    _deinit_distributed()


@pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip muon optimizer for LTS test",
)
def test_get_megatron_muon_optimizer_validation():
    """Test validation logic for get_megatron_muon_optimizer."""
    world = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))

    # Setup: distributed, model
    _init_distributed(world, rank)
    Utils.initialize_model_parallel()

    # Create a simple model
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
    model.requires_grad_(True)
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )

    # Test 1: Distributed optimizer should raise exception
    optimizer_config_dist = OptimizerConfig(
        optimizer='muon',
        lr=0.01,
        bf16=True,
        use_distributed_optimizer=True,  # This should cause an exception
    )

    with pytest.raises(Exception, match='muon with dist optimizer is not supported'):
        get_megatron_muon_optimizer(config=optimizer_config_dist, model_chunks=[model])

    # Test 2: FP16 should raise exception
    optimizer_config_fp16 = OptimizerConfig(
        optimizer='muon',
        lr=0.01,
        fp16=True,  # This should cause an exception
        use_distributed_optimizer=False,
    )

    with pytest.raises(Exception, match='muon with fp16 is not supported'):
        get_megatron_muon_optimizer(config=optimizer_config_fp16, model_chunks=[model])

    # Test 3: Invalid num_ns_steps should raise exception
    optimizer_config_invalid_ns = OptimizerConfig(
        optimizer='muon',
        lr=0.01,
        bf16=True,
        use_distributed_optimizer=False,
        muon_num_ns_steps=0,  # This should cause an exception
    )

    with pytest.raises(ValueError, match='num_ns_steps must be at least 1'):
        get_megatron_muon_optimizer(config=optimizer_config_invalid_ns, model_chunks=[model])

    _deinit_distributed()
