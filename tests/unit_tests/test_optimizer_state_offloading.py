# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for OptimizerStateOffloader."""

import pytest
import torch
import torch.nn as nn

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch.optimizers import FusedAdam  # noqa: F401

    TE_FUSED_ADAM_AVAILABLE = True
except ImportError:
    TE_FUSED_ADAM_AVAILABLE = False


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def create_model_and_optimizer(hidden_size=256, offload_optimizer_states=True, **optimizer_kwargs):
    """Helper to create model and optimizer for tests."""
    model = SimpleModel(hidden_size=hidden_size).bfloat16().cuda()
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )

    default_config = dict(
        optimizer='adam',
        bf16=True,
        lr=0.001,
        use_distributed_optimizer=True,
        offload_optimizer_states=offload_optimizer_states,
    )
    default_config.update(optimizer_kwargs)

    optimizer_config = OptimizerConfig(**default_config)
    optim = get_megatron_optimizer(optimizer_config, [model])
    return model, optim


def run_forward_backward_step(model, optim, hidden_size=256):
    """Run a single forward-backward-step cycle."""
    input_tensor = torch.randn(8, hidden_size, dtype=torch.bfloat16, device='cuda')
    output = model(input_tensor)
    output.sum().backward()
    optim.step()
    optim.zero_grad()


# =============================================================================
# Test 1: Basic OptimizerStateOffloader Initialization
# =============================================================================
@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_offloader_initialization():
    """Test that OptimizerStateOffloader initializes correctly."""
    Utils.initialize_model_parallel()
    model, optim = create_model_and_optimizer()
    dist_optim = optim.chained_optimizers[0]

    # Offloader is created in __init__ when offload_optimizer_states=True
    assert dist_optim._state_offloader is not None
    offloader = dist_optim._state_offloader

    # Verify offloader properties
    assert offloader.adam_optimizer is not None
    assert offloader._d2h_stream is not None
    assert offloader._h2d_stream is not None
    assert offloader._offloaded is False

    # Before first step, optimizer states are not initialized yet
    assert offloader._optimizer_states_initialized is False

    # Run one step to initialize optimizer states
    run_forward_backward_step(model, optim)

    # After first step, optimizer states should be marked as initialized
    assert offloader._optimizer_states_initialized is True
    Utils.destroy_model_parallel()


# =============================================================================
# Test 2: Early Master Weight Offloading Before First Step
# =============================================================================
@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_early_master_weight_offloading():
    """Test that master weights can be offloaded before the first optimizer step."""
    Utils.initialize_model_parallel()
    model, optim = create_model_and_optimizer()
    dist_optim = optim.chained_optimizers[0]

    # Offloader is created in __init__
    assert dist_optim._state_offloader is not None
    offloader = dist_optim._state_offloader

    # Before first step, optimizer states are not initialized
    assert offloader._optimizer_states_initialized is False

    # Capture original master weights before offload
    original_master_weights = []
    for group in dist_optim.shard_fp32_from_float16_groups:
        group_weights = [tensor.clone() for tensor in group]
        original_master_weights.append(group_weights)

    # Offload before first step - should only offload master weights
    offloader.offload()
    offloader.release_gpu_memory()
    torch.cuda.synchronize()

    # Verify master weights were offloaded (storage resized to 0)
    for group in dist_optim.shard_fp32_from_float16_groups:
        for tensor in group:
            assert tensor.untyped_storage().size() == 0, "Master weight should be offloaded"

    # Reload master weights
    offloader.reload()
    offloader.sync_before_step()

    # Verify master weights match after reload
    for group_idx, group in enumerate(dist_optim.shard_fp32_from_float16_groups):
        for param_idx, tensor in enumerate(group):
            original = original_master_weights[group_idx][param_idx]
            torch.testing.assert_close(
                tensor,
                original,
                msg=f"Master weight [{group_idx}][{param_idx}] mismatch after offload/reload",
            )

    # Now run a step and verify optimizer states can be offloaded after
    run_forward_backward_step(model, optim)
    assert offloader._optimizer_states_initialized is True

    Utils.destroy_model_parallel()


# =============================================================================
# Test 3: Offload and Reload Correctness
# =============================================================================
@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
@pytest.mark.parametrize("offload_optimizer_states", [True, False])
@pytest.mark.parametrize("offload_master_weights", [True, False])
def test_offload_reload_correctness(offload_optimizer_states, offload_master_weights):
    """Test that offload/reload preserves optimizer state values."""
    if not offload_optimizer_states and not offload_master_weights:
        pytest.skip("At least one offload type required")

    Utils.initialize_model_parallel()
    model, optim = create_model_and_optimizer()
    dist_optim = optim.chained_optimizers[0]

    # Run steps to build up optimizer state
    for _ in range(3):
        run_forward_backward_step(model, optim)

    offloader = dist_optim._state_offloader

    # Capture original states before offload
    original_states = {}
    for param, state in offloader.adam_optimizer.state.items():
        original_states[param] = {
            k: v.clone() for k, v in state.items() if isinstance(v, torch.Tensor)
        }

    # Offload
    offloader.offload(
        offload_optimizer_states=offload_optimizer_states,
        offload_master_weights=offload_master_weights,
    )

    # Release GPU memory
    offloader.release_gpu_memory()
    torch.cuda.synchronize()

    # Reload
    offloader.reload()
    offloader.sync_before_step()

    # Verify states match after reload
    for param, state in offloader.adam_optimizer.state.items():
        if param in original_states:
            for key, original_tensor in original_states[param].items():
                if key in state and isinstance(state[key], torch.Tensor):
                    reloaded_tensor = state[key]
                    assert reloaded_tensor.device.type == 'cuda', f"State {key} should be on GPU"
                    torch.testing.assert_close(
                        reloaded_tensor,
                        original_tensor,
                        msg=f"State {key} mismatch after offload/reload",
                    )
    Utils.destroy_model_parallel()


# =============================================================================
# Test 4: GPU Memory Release Verification
# =============================================================================
@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_gpu_memory_release():
    """Test that GPU memory is actually freed after release_gpu_memory()."""
    Utils.initialize_model_parallel()
    # Use larger model for measurable memory impact
    model, optim = create_model_and_optimizer(hidden_size=1024)
    dist_optim = optim.chained_optimizers[0]

    # Initialize optimizer states
    run_forward_backward_step(model, optim, hidden_size=1024)

    offloader = dist_optim._state_offloader

    # Measure memory before offload
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated()

    # Offload and release
    offloader.offload()
    offloader.release_gpu_memory()

    # Wait for async operations
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    memory_after = torch.cuda.memory_allocated()

    # Memory should decrease
    memory_freed = memory_before - memory_after
    assert memory_freed > 0, f"Expected memory to be freed, but got {memory_freed} bytes difference"
    Utils.destroy_model_parallel()


# =============================================================================
# Test 5: Multiple Offload/Reload Cycles
# =============================================================================
@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_multiple_offload_reload_cycles():
    """Test that multiple offload/reload cycles work correctly."""
    Utils.initialize_model_parallel()
    model, optim = create_model_and_optimizer()
    dist_optim = optim.chained_optimizers[0]

    # Initialize
    run_forward_backward_step(model, optim)

    offloader = dist_optim._state_offloader

    # Run multiple cycles
    for cycle in range(5):
        # Offload
        offloader.offload()
        offloader.release_gpu_memory()

        # Reload
        offloader.reload()
        offloader.sync_before_step()

        # Run optimizer step
        run_forward_backward_step(model, optim)

    # Verify model can still produce valid outputs
    input_tensor = torch.randn(8, 256, dtype=torch.bfloat16, device='cuda')
    output = model(input_tensor)
    assert not output.isnan().any(), "Model output contains NaN after multiple cycles"
    Utils.destroy_model_parallel()


# =============================================================================
# Test 6: Training Correctness with Offloading
# =============================================================================
@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_training_correctness_with_offloading():
    """Test that training with offloading produces same results as without."""
    Utils.initialize_model_parallel()
    torch.manual_seed(42)

    # Model 1: with offloading
    model1, optim1 = create_model_and_optimizer(offload_optimizer_states=True, lr=0.01)

    # Model 2: without offloading (reference)
    torch.manual_seed(42)
    model2, optim2 = create_model_and_optimizer(offload_optimizer_states=False, lr=0.01)

    # Train both models
    n_steps = 10
    torch.manual_seed(123)
    dist_optim1 = optim1.chained_optimizers[0]

    # Offloader is created in __init__ when offload_optimizer_states=True
    assert dist_optim1._state_offloader is not None
    offloader = dist_optim1._state_offloader

    for step in range(n_steps):
        input_tensor = torch.randn(8, 256, dtype=torch.bfloat16, device='cuda')

        # Model 1 with offloading
        # Offload states (master weights can be offloaded from the start,
        # optimizer states will be skipped until after first step)
        offloader.offload()
        offloader.release_gpu_memory()

        output1 = model1(input_tensor)
        loss1 = output1.sum()
        loss1.backward()

        offloader.reload()
        offloader.sync_before_step()
        optim1.step()
        optim1.zero_grad()

        # Model 2 without offloading
        output2 = model2(input_tensor)
        loss2 = output2.sum()
        loss2.backward()
        optim2.step()
        optim2.zero_grad()

    # Compare final model weights
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        torch.testing.assert_close(
            p1.data,
            p2.data,
            atol=1e-5,
            rtol=1e-4,
            msg=f"Parameter {n1} mismatch between offloaded and non-offloaded training",
        )
    Utils.destroy_model_parallel()
