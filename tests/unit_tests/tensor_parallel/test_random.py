# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.tensor_parallel.random import (
    BlockLevelCheckpointManager,
    CheckpointWithoutOutput,
    CudaRNGStatesTracker,
    checkpoint,
    convert_cuda_rng_state,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from tests.unit_tests.test_utilities import Utils


def test_cuda_rng_states_tracker():
    rng_tracker = CudaRNGStatesTracker()
    rng_tracker.set_states({"state1": 1234})
    assert rng_tracker.get_states()["state1"] == 1234
    rng_tracker.reset()
    assert rng_tracker.get_states() == {}
    seed = 1111
    rng_tracker.add("state2", seed)
    with pytest.raises(Exception):
        assert rng_tracker.add("state3", seed)
    with pytest.raises(Exception):
        assert rng_tracker.add("state2", 111)
    assert rng_tracker.get_states()['state2'] is not None
    with pytest.raises(Exception):
        assert ()

    rng_tracker.fork("state2")
    torch.cuda.manual_seed(seed)
    rng_state = torch.cuda.get_rng_state()
    assert torch.equal(rng_tracker.get_states()['state2'], rng_state)


@pytest.mark.parametrize("use_cudagraphable_rng", [True, False])
def test_double_fork_cuda_rng_states_tracker(use_cudagraphable_rng):
    rng_tracker = CudaRNGStatesTracker(use_cudagraphable_rng=use_cudagraphable_rng)
    rng_tracker.add("state1", 1234)
    rng_tracker.add("state2", 5678)
    randn_double_fork_1 = []
    randn_double_fork_2 = []
    with rng_tracker.fork("state1"):
        randn_double_fork_1.append(torch.randn(10, device="cuda"))
        with rng_tracker.fork("state2"):
            randn_double_fork_2.append(torch.randn(10, device="cuda"))
            with rng_tracker.fork("state1"):
                randn_double_fork_1.append(torch.randn(10, device="cuda"))
            randn_double_fork_2.append(torch.randn(10, device="cuda"))
        randn_double_fork_1.append(torch.randn(10, device="cuda"))
    if use_cudagraphable_rng:
        double_fork_state1 = rng_tracker.get_states()["state1"].get_state()
        double_fork_state2 = rng_tracker.get_states()["state2"].get_state()
    else:
        double_fork_state1 = rng_tracker.get_states()["state1"]
        double_fork_state2 = rng_tracker.get_states()["state2"]

    rng_tracker.reset()
    rng_tracker.add("state1", 1234)
    rng_tracker.add("state2", 5678)
    randn_single_fork_1 = []
    randn_single_fork_2 = []
    with rng_tracker.fork("state1"):
        randn_single_fork_1.append(torch.randn(10, device="cuda"))
        randn_single_fork_1.append(torch.randn(10, device="cuda"))
        randn_single_fork_1.append(torch.randn(10, device="cuda"))
    with rng_tracker.fork("state2"):
        randn_single_fork_2.append(torch.randn(10, device="cuda"))
        randn_single_fork_2.append(torch.randn(10, device="cuda"))
    if use_cudagraphable_rng:
        single_fork_state1 = rng_tracker.get_states()["state1"].get_state()
        single_fork_state2 = rng_tracker.get_states()["state2"].get_state()
    else:
        single_fork_state1 = rng_tracker.get_states()["state1"]
        single_fork_state2 = rng_tracker.get_states()["state2"]

    assert torch.equal(randn_double_fork_1[0], randn_single_fork_1[0])
    assert torch.equal(randn_double_fork_1[1], randn_single_fork_1[1])
    assert torch.equal(randn_double_fork_1[2], randn_single_fork_1[2])
    assert torch.equal(randn_double_fork_2[0], randn_single_fork_2[0])
    assert torch.equal(randn_double_fork_2[1], randn_single_fork_2[1])
    assert torch.equal(double_fork_state1, single_fork_state1)
    assert torch.equal(double_fork_state2, single_fork_state2)


def test_convert_cuda_rng_state():
    ## Get the default rng state
    torch.cuda.manual_seed(999)
    randn = torch.randn(10, device="cuda")
    rng_state = torch.cuda.get_rng_state()

    try:
        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker
    except ImportError:
        TECudaRNGStatesTracker = None

    ## from non-graphable RNG to graphable RNG
    # get state from non-graphable RNG
    tracker = CudaRNGStatesTracker(use_cudagraphable_rng=False)
    tracker.add("state1", 123)
    for i in range(3):
        with tracker.fork("state1"):
            randn = torch.randn(10, device="cuda")
    state = convert_cuda_rng_state(tracker.states_["state1"], to_graphable=True)
    rand_tensors = []
    for i in range(3):
        with tracker.fork("state1"):
            randn = torch.randn(10, device="cuda")
            rand_tensors.append(randn)

    # set state to local graph RNG
    cudagraphable_tracker = CudaRNGStatesTracker(use_cudagraphable_rng=True)
    cudagraphable_tracker.set_states({"state1": state.clone_state()})
    for i in range(3):
        with cudagraphable_tracker.fork("state1"):
            randn = torch.randn(10, device="cuda")
            assert torch.equal(randn, rand_tensors[i])

    # set state to TE RNG
    if TECudaRNGStatesTracker is not None:
        te_tracker = TECudaRNGStatesTracker()
        te_tracker.set_states({"state1": state})
        for i in range(3):
            with te_tracker.fork("state1"):
                randn = torch.randn(10, device="cuda")
                assert torch.equal(randn, rand_tensors[i])

    ## from graphable RNG to non-graphable RNG
    # get state from graphable RNG
    cudagraphable_tracker = CudaRNGStatesTracker(use_cudagraphable_rng=True)
    cudagraphable_tracker.add("state2", 123)
    for i in range(3):
        with cudagraphable_tracker.fork("state2"):
            randn = torch.randn(10, device="cuda")
    state = convert_cuda_rng_state(cudagraphable_tracker.states_["state2"], to_graphable=False)
    rand_tensors = []
    for i in range(3):
        with cudagraphable_tracker.fork("state2"):
            randn = torch.randn(10, device="cuda")
            rand_tensors.append(randn)

    # set state to non-graphable RNG
    tracker = CudaRNGStatesTracker(use_cudagraphable_rng=False)
    tracker.set_states({"state2": state})
    for i in range(3):
        with tracker.fork("state2"):
            randn = torch.randn(10, device="cuda")
            assert torch.equal(randn, rand_tensors[i])

    ## from TE RNG to non-graphable RNG
    if TECudaRNGStatesTracker is not None:
        # get state from TE RNG
        cudagraphable_tracker = TECudaRNGStatesTracker()
        cudagraphable_tracker.add("state3", 123)
        for i in range(3):
            with cudagraphable_tracker.fork("state3"):
                randn = torch.randn(10, device="cuda")
        state = convert_cuda_rng_state(cudagraphable_tracker.states_["state3"], to_graphable=False)
        rand_tensors = []
        for i in range(3):
            with cudagraphable_tracker.fork("state3"):
                randn = torch.randn(10, device="cuda")
                rand_tensors.append(randn)

        # set state to non-graphable RNG
        tracker = CudaRNGStatesTracker(use_cudagraphable_rng=False)
        tracker.set_states({"state3": state})
        for i in range(3):
            with tracker.fork("state3"):
                randn = torch.randn(10, device="cuda")
                assert torch.equal(randn, rand_tensors[i])

    ## After all tests, check if the default rng state is still the same.
    rng_state_final = torch.cuda.get_rng_state()
    assert torch.equal(rng_state, rng_state_final)


def test_model_parallel_cuda_manual_seed():
    Utils.initialize_model_parallel(4, 2)
    model_parallel_cuda_manual_seed(0, force_reset_rng=True)
    rng_tracker = get_cuda_rng_tracker()
    assert rng_tracker.get_states()['model-parallel-rng'] is not None
    Utils.destroy_model_parallel()


def test_checkpoint():
    def test_forward(*input):
        return input[0] + input[1]

    assert torch.equal(
        torch.ones(16) * 3, checkpoint(test_forward, None, torch.ones(16), torch.ones(16) * 2)
    )
    Utils.initialize_model_parallel()
    input1 = torch.ones((4, 4))
    checkpoint(test_forward, True, input1, torch.ones((4, 4)) * 2)
    assert torch.equal(torch.ones(input1.numel()).cuda(), input1)
    Utils.destroy_model_parallel()


def test_checkpoint_without_output():
    def normal_forward(input):
        x = torch.nn.functional.gelu(input)
        y = x * input
        return y

    def checkpoint_forward(input):
        checkpoint = CheckpointWithoutOutput()
        x = checkpoint.checkpoint(torch.nn.functional.gelu, input)
        y = x * input
        checkpoint.discard_output_and_register_recompute(y)
        return y

    Utils.initialize_model_parallel()

    input1 = torch.ones((4, 4))
    input1.requires_grad_(True)
    output1 = normal_forward(input1)
    input2 = torch.ones((4, 4))
    input2.requires_grad_(True)
    output2 = checkpoint_forward(input2)
    assert torch.equal(output1, output2)

    output1.backward(torch.ones((4, 4)), retain_graph=True)
    output2.backward(torch.ones((4, 4)), retain_graph=True)
    assert torch.equal(input1.grad, input2.grad)

    Utils.destroy_model_parallel()


def test_block_level_checkpoint_manager():
    """
    Test BlockLevelCheckpointManager with three sequential checkpoint functions.

    This test verifies that:
    1. The manager correctly handles sequential checkpoints where each function's
       output is the next function's input
    2. Recomputation happens in the correct order during backward
    3. Gradients are computed correctly and match the non-checkpointed version
    """

    # Define three simple functions that form a chain:
    # x -> func1 -> y1 -> func2 -> y2 -> func3 -> y3
    def func1(x):
        # Simple linear transformation
        return x * 2 + 1

    def func2(x):
        # Non-linear transformation
        return torch.nn.functional.gelu(x)

    def func3(x):
        # Another transformation
        return x * x + x

    Utils.initialize_model_parallel()

    # ========== Test 1: Basic forward and backward correctness ==========
    # Create input tensor
    input_ref = torch.randn(4, 4, device='cuda', requires_grad=True)
    input_ckpt = input_ref.detach().clone().requires_grad_(True)

    # Reference: normal forward without checkpoint
    y1_ref = func1(input_ref)
    y2_ref = func2(y1_ref)
    y3_ref = func3(y2_ref)
    loss_ref = y3_ref.sum()
    loss_ref.backward()
    grad_ref = input_ref.grad.clone()

    
    # With BlockLevelCheckpointManager
    manager = BlockLevelCheckpointManager()

    ckpt1 = CheckpointWithoutOutput()
    y1 = ckpt1.checkpoint(func1, input_ckpt)
    manager.add_checkpoint(ckpt1)

    ckpt2 = CheckpointWithoutOutput()
    y2 = ckpt2.checkpoint(func2, y1)
    manager.add_checkpoint(ckpt2)

    ckpt3 = CheckpointWithoutOutput()
    y3 = ckpt3.checkpoint(func3, y2)
    manager.add_checkpoint(ckpt3)

    loss_ckpt = y3.sum() 
    # Register unified recompute hook on the final output
    manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

    # Verify outputs are discarded (storage size is 0)
    assert y1.untyped_storage().size() == 0, "y1 storage should be released"
    assert y2.untyped_storage().size() == 0, "y2 storage should be released"
    assert y3.untyped_storage().size() == 0, "y3 storage should be released"

    # Compute Loss and Backward
    loss_ckpt.backward()
    grad_ckpt = input_ckpt.grad.clone()

    # print(grad_ckpt)
    # return 
    # Verify gradients match
    assert torch.allclose(grad_ckpt, grad_ref, atol=1e-6), (
        f"Gradients mismatch!\n"
        f"With manager: {grad_ckpt}\n"
        f"Reference: {grad_ref}"
    )

    # ========== Test 2: With randomness (dropout-like behavior) ==========
    def func_with_dropout(x):
        # Simulates dropout: random mask applied
        return torch.nn.functional.dropout(x, p=0.3, training=True)

    # Reset inputs
    input_ref2 = torch.randn(4, 4, device='cuda', requires_grad=True)
    input_ckpt2 = input_ref2.detach().clone().requires_grad_(True)

    # Set same random seed for both paths
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Reference forward
    y1_ref2 = func_with_dropout(input_ref2)
    y2_ref2 = func2(y1_ref2)
    loss_ref2 = y2_ref2.sum()
    loss_ref2.backward()
    grad_ref2 = input_ref2.grad.clone()

    # Reset random seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # With checkpoint manager
    manager2 = BlockLevelCheckpointManager()

    ckpt1_2 = CheckpointWithoutOutput()
    y1_2 = ckpt1_2.checkpoint(func_with_dropout, input_ckpt2)
    manager2.add_checkpoint(ckpt1_2)

    ckpt2_2 = CheckpointWithoutOutput()
    y2_2 = ckpt2_2.checkpoint(func2, y1_2)
    manager2.add_checkpoint(ckpt2_2)

    loss_ckpt2 = y2_2.sum() 

    manager2.discard_all_outputs_and_register_unified_recompute(loss_ckpt2)

    loss_ckpt2.backward()
    grad_ckpt2 = input_ckpt2.grad.clone()

    # Gradients should match because RNG state is restored during recompute
    assert torch.allclose(grad_ckpt2, grad_ref2, atol=1e-6), (
        f"Gradients with dropout mismatch!\n"
        f"With manager: {grad_ckpt2}\n"
        f"Reference: {grad_ref2}"
    )

    Utils.destroy_model_parallel()


def test_block_level_checkpoint_manager_with_multiple_outputs():
    """
    Test BlockLevelCheckpointManager with functions that return multiple outputs.
    """

    def func_multi_output(x):
        # Returns two outputs
        return x * 2, x + 1

    def func_combine(a, b):
        # Combines two inputs
        return a + b

    Utils.initialize_model_parallel()

    input_ref = torch.randn(4, 4, device='cuda', requires_grad=True)
    input_ckpt = input_ref.detach().clone().requires_grad_(True)

    # Reference
    y1a_ref, y1b_ref = func_multi_output(input_ref)
    y2_ref = func_combine(y1a_ref, y1b_ref)
    loss_ref = y2_ref.sum()
    loss_ref.backward()
    grad_ref = input_ref.grad.clone()

    # With manager
    manager = BlockLevelCheckpointManager()

    ckpt1 = CheckpointWithoutOutput()
    y1a, y1b = ckpt1.checkpoint(func_multi_output, input_ckpt)
    manager.add_checkpoint(ckpt1)

    ckpt2 = CheckpointWithoutOutput()
    y2 = ckpt2.checkpoint(func_combine, y1a, y1b)
    manager.add_checkpoint(ckpt2)

    loss_ckpt = y2.sum()  
    manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

    loss_ckpt.backward()
    grad_ckpt = input_ckpt.grad.clone()

    assert torch.allclose(grad_ckpt, grad_ref, atol=1e-6), (
        f"Gradients mismatch with multiple outputs!\n"
        f"With manager: {grad_ckpt}\n"
        f"Reference: {grad_ref}"
    )

    Utils.destroy_model_parallel()


def test_block_level_checkpoint_manager_error_handling():
    """
    Test error handling in BlockLevelCheckpointManager.
    """
    Utils.initialize_model_parallel()

    manager = BlockLevelCheckpointManager()

    # Test 1: Adding non-CheckpointWithoutOutput object should raise TypeError
    with pytest.raises(TypeError):
        manager.add_checkpoint("not a checkpoint")

    # Test 2: Adding checkpoint that hasn't called checkpoint() should raise ValueError
    ckpt = CheckpointWithoutOutput()
    with pytest.raises(ValueError):
        manager.add_checkpoint(ckpt)

    Utils.destroy_model_parallel()
