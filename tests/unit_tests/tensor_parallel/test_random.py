# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.tensor_parallel.random import (
    BlockLevelCheckpointManager,  # backward compat alias
    CheckpointWithoutOutput,
    CudaRNGStatesTracker,
    MHCBlockRecomputeManager,
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


def test_checkpoint_without_output_with_ckpt_manager_auto_register():
    """
    Test that CheckpointWithoutOutput auto-registers to manager when ckpt_manager is provided.
    """
    Utils.initialize_model_parallel()

    manager = MHCBlockRecomputeManager()

    def func(x):
        return x * 2 + 1

    input_t = torch.randn(4, 4, device='cuda', requires_grad=True)

    # Create checkpoint with ckpt_manager - should auto-register
    ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
    y = ckpt.checkpoint(func, input_t)

    # Verify auto-registration
    assert len(manager.checkpoints) == 1
    assert manager.checkpoints[0] is ckpt

    # Add another checkpoint
    ckpt2 = CheckpointWithoutOutput(ckpt_manager=manager)
    y2 = ckpt2.checkpoint(torch.nn.functional.gelu, y)

    # Verify both are registered
    assert len(manager.checkpoints) == 2
    assert manager.checkpoints[1] is ckpt2

    # Complete the forward and backward
    loss = y2.sum()
    manager.discard_all_outputs_and_register_unified_recompute(loss)
    loss.backward()

    assert input_t.grad is not None

    Utils.destroy_model_parallel()


def test_checkpoint_without_output_discard_is_noop_with_manager():
    """
    Test that discard_output_and_register_recompute is a NO-OP when ckpt_manager is set.
    The manager handles all discarding and hook registration.
    """
    Utils.initialize_model_parallel()

    manager = MHCBlockRecomputeManager()

    def func1(x):
        return x * 2

    def func2(x):
        return torch.nn.functional.gelu(x)

    # Reference without checkpoint
    input_ref = torch.randn(4, 4, device='cuda', requires_grad=True)
    y1_ref = func1(input_ref)
    y2_ref = func2(y1_ref)
    loss_ref = y2_ref.sum()
    loss_ref.backward()
    grad_ref = input_ref.grad.clone()

    # With ckpt_manager: discard_output_and_register_recompute is a no-op
    input_ckpt = input_ref.detach().clone().requires_grad_(True)

    ckpt1 = CheckpointWithoutOutput(ckpt_manager=manager)
    y1 = ckpt1.checkpoint(func1, input_ckpt)
    # This is a no-op when ckpt_manager is set
    ckpt1.discard_output_and_register_recompute(y1)

    ckpt2 = CheckpointWithoutOutput(ckpt_manager=manager)
    y2 = ckpt2.checkpoint(func2, y1)
    ckpt2.discard_output_and_register_recompute(y2)

    # Verify outputs are NOT discarded yet (discard_output_and_register_recompute is no-op)
    assert y1.untyped_storage().size() > 0, "y1 should NOT be discarded yet"
    assert y2.untyped_storage().size() > 0, "y2 should NOT be discarded yet"

    # Now use manager to discard all outputs and register unified hook
    loss_ckpt = y2.sum()
    manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

    # NOW outputs should be discarded
    assert y1.untyped_storage().size() == 0, "y1 should be discarded after manager call"
    assert y2.untyped_storage().size() == 0, "y2 should be discarded after manager call"

    loss_ckpt.backward()
    grad_ckpt = input_ckpt.grad.clone()

    assert torch.allclose(grad_ckpt, grad_ref, atol=1e-6)

    Utils.destroy_model_parallel()


def test_checkpoint_without_output_backward_compat():
    """
    Test backward compatibility: CheckpointWithoutOutput without ckpt_manager
    should work exactly as before.
    """
    Utils.initialize_model_parallel()

    def func(x):
        return torch.nn.functional.gelu(x)

    # Reference
    input_ref = torch.randn(4, 4, device='cuda', requires_grad=True)
    y_ref = func(input_ref)
    z_ref = y_ref * 2
    loss_ref = z_ref.sum()
    loss_ref.backward()
    grad_ref = input_ref.grad.clone()

    # Without ckpt_manager (backward compatible mode)
    input_ckpt = input_ref.detach().clone().requires_grad_(True)

    ckpt = CheckpointWithoutOutput()  # No ckpt_manager
    y = ckpt.checkpoint(func, input_ckpt)
    z = y * 2
    ckpt.discard_output_and_register_recompute(z)  # Should register individual hook

    # Verify output is discarded
    assert y.untyped_storage().size() == 0

    loss_ckpt = z.sum()
    loss_ckpt.backward()
    grad_ckpt = input_ckpt.grad.clone()

    assert torch.allclose(grad_ckpt, grad_ref, atol=1e-6)

    Utils.destroy_model_parallel()


def test_mhc_block_recompute_manager():
    """
    Test MHCBlockRecomputeManager with three sequential checkpoint functions.

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

    # With MHCBlockRecomputeManager using ckpt_manager parameter (simplified API)
    manager = MHCBlockRecomputeManager()

    # Using ckpt_manager parameter for auto-registration
    y1 = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func1, input_ckpt)
    y2 = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func2, y1)
    y3 = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func3, y2)

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

    # With checkpoint manager using ckpt_manager parameter
    manager2 = MHCBlockRecomputeManager()

    y1_2 = CheckpointWithoutOutput(ckpt_manager=manager2).checkpoint(func_with_dropout, input_ckpt2)
    y2_2 = CheckpointWithoutOutput(ckpt_manager=manager2).checkpoint(func2, y1_2)

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


def test_mhc_block_recompute_manager_with_multiple_outputs():
    """
    Test MHCBlockRecomputeManager with functions that return multiple outputs.
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

    # With manager using ckpt_manager parameter
    manager = MHCBlockRecomputeManager()

    y1a, y1b = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func_multi_output, input_ckpt)
    y2 = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func_combine, y1a, y1b)

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


def test_mhc_block_recompute_manager_error_handling():
    """
    Test error handling in MHCBlockRecomputeManager.
    """
    Utils.initialize_model_parallel()

    manager = MHCBlockRecomputeManager()

    # Test 1: Adding non-CheckpointWithoutOutput object should raise TypeError
    with pytest.raises(TypeError):
        manager.add_checkpoint("not a checkpoint")

    # Test 2: Adding checkpoint that hasn't called checkpoint() should raise ValueError
    ckpt = CheckpointWithoutOutput()
    with pytest.raises(ValueError):
        manager.add_checkpoint(ckpt)

    Utils.destroy_model_parallel()


def test_backward_compat_block_level_checkpoint_manager_alias():
    """
    Test that BlockLevelCheckpointManager alias works for backward compatibility.
    """
    # BlockLevelCheckpointManager should be an alias for MHCBlockRecomputeManager
    assert BlockLevelCheckpointManager is MHCBlockRecomputeManager

    Utils.initialize_model_parallel()

    # Should work the same way with ckpt_manager parameter
    manager = BlockLevelCheckpointManager()

    def func(x):
        return x * 2

    input_t = torch.randn(4, 4, device='cuda', requires_grad=True)

    y = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func, input_t)

    loss = y.sum()
    manager.discard_all_outputs_and_register_unified_recompute(loss)
    loss.backward()

    assert input_t.grad is not None

    Utils.destroy_model_parallel()


def test_mhc_block_recompute_manager_partial_checkpoint():
    """
    Test MHCBlockRecomputeManager with partial checkpointing.

    This test verifies the real-world scenario where only some operations
    are checkpointed while others are not.

    Computation chain:
        a --[f]--> b --[g]--> c --[h]--> d --[sum]--> loss
                   ^           ^
                   |           |
              checkpointed  checkpointed
              (ckpt_f)      (ckpt_h)

    Only f and h are wrapped with CheckpointWithoutOutput.
    g is a regular operation without checkpoint.

    This mimics the HyperConnection scenario where:
    - compute_mappings (checkpointed)
    - aggregate (not checkpointed, or checkpointed)
    - apply_h_res (checkpointed)
    - regular ops in between (not checkpointed)
    """

    # Define functions for the computation chain
    def func_f(x):
        """First checkpointed function: linear + nonlinear"""
        return torch.nn.functional.gelu(x * 2 + 1)

    def func_g(x):
        """Middle function without checkpoint: another transform"""
        return x * 3 - 2

    def func_h(x):
        """Second checkpointed function: nonlinear"""
        return torch.sigmoid(x) + x

    Utils.initialize_model_parallel()

    # ========== Reference: normal forward without any checkpoint ==========
    input_ref = torch.randn(4, 4, device='cuda', requires_grad=True)

    # a --[f]--> b --[g]--> c --[h]--> d
    b_ref = func_f(input_ref)  # a = input_ref, b = f(a)
    c_ref = func_g(b_ref)      # c = g(b)
    d_ref = func_h(c_ref)      # d = h(c)
    loss_ref = d_ref.sum()
    loss_ref.backward()
    grad_ref = input_ref.grad.clone()

    # ========== With MHCBlockRecomputeManager: partial checkpoint ==========
    input_ckpt = input_ref.detach().clone().requires_grad_(True)

    manager = MHCBlockRecomputeManager()

    # Step 1: f is checkpointed (using ckpt_manager for auto-registration)
    b = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func_f, input_ckpt)

    # Step 2: g is NOT checkpointed (regular operation)
    c = func_g(b)

    # Step 3: h is checkpointed (using ckpt_manager for auto-registration)
    d = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func_h, c)

    # Step 4: Compute loss and register unified recompute
    loss_ckpt = d.sum()
    manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

    # Verify checkpoint outputs are discarded
    assert b.untyped_storage().size() == 0, "b storage should be released"
    assert d.untyped_storage().size() == 0, "d storage should be released"
    # Note: c is not checkpointed, so its storage is NOT released
    assert c.untyped_storage().size() > 0, "c storage should NOT be released (not checkpointed)"

    # Backward
    loss_ckpt.backward()
    grad_ckpt = input_ckpt.grad.clone()

    # Verify gradients match
    assert torch.allclose(grad_ckpt, grad_ref, atol=1e-6), (
        f"Gradients mismatch with partial checkpoint!\n"
        f"With manager: {grad_ckpt}\n"
        f"Reference: {grad_ref}"
    )

    Utils.destroy_model_parallel()


def test_mhc_block_recompute_manager_partial_checkpoint_with_tuple_output():
    """
    Test MHCBlockRecomputeManager with partial checkpointing and tuple outputs.

    This more closely mimics HyperConnection's actual computation pattern:

    Computation chain:
        x --[compute_mappings]--> (h_pre, h_post, h_res)
                                      |         |
                                      v         |
        x, h_pre --[aggregate]-------> agg      |
                                        |       |
                                        v       v
                            (some ops)  y  h_res, residual
                                        |       |
                                        v       v
                    y, h_post --[apply_h_post]-> output
                                                |
                                h_res, residual --[apply_h_res]--> mixed
                                                |         |
                                                v         v
                                              final computations...

    In this test:
    - compute_mappings: checkpointed, returns tuple (h_pre, h_post, h_res)
    - aggregate: NOT checkpointed
    - apply_h_res: checkpointed
    - apply_h_post: checkpointed
    """

    def compute_mappings(x):
        """Mimics HyperConnection.compute_mappings, returns 3 tensors"""
        h_pre = torch.sigmoid(x.mean(dim=-1, keepdim=True).expand_as(x))
        h_post = torch.tanh(x.sum(dim=-1, keepdim=True).expand_as(x))
        h_res = torch.relu(x)
        return h_pre, h_post, h_res

    def aggregate(x, h_pre):
        """Mimics HyperConnection.aggregate"""
        return x * h_pre

    def apply_h_res(h_res, residual):
        """Mimics HyperConnection.apply_h_res"""
        return h_res + residual * 0.5

    def apply_h_post(y, h_post):
        """Mimics HyperConnection.apply_h_post"""
        return y * h_post + y

    Utils.initialize_model_parallel()

    # ========== Reference: normal forward ==========
    x_ref = torch.randn(4, 4, device='cuda', requires_grad=True)
    residual_ref = torch.randn(4, 4, device='cuda', requires_grad=True)

    h_pre_ref, h_post_ref, h_res_ref = compute_mappings(x_ref)
    agg_ref = aggregate(x_ref, h_pre_ref)
    # Simulate some intermediate computation
    y_ref = torch.nn.functional.gelu(agg_ref)
    mixed_ref = apply_h_res(h_res_ref, residual_ref)
    output_ref = apply_h_post(y_ref, h_post_ref)
    # Final output combines mixed and output
    final_ref = output_ref + mixed_ref
    loss_ref = final_ref.sum()
    loss_ref.backward()
    grad_x_ref = x_ref.grad.clone()
    grad_residual_ref = residual_ref.grad.clone()

    # ========== With MHCBlockRecomputeManager using ckpt_manager ==========
    x_ckpt = x_ref.detach().clone().requires_grad_(True)
    residual_ckpt = residual_ref.detach().clone().requires_grad_(True)

    manager = MHCBlockRecomputeManager()

    # Step 1: compute_mappings is checkpointed (returns tuple)
    h_pre, h_post, h_res = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
        compute_mappings, x_ckpt
    )

    # Step 2: aggregate is NOT checkpointed
    agg = aggregate(x_ckpt, h_pre)

    # Step 3: some intermediate computation (not checkpointed)
    y = torch.nn.functional.gelu(agg)

    # Step 4: apply_h_res is checkpointed
    mixed = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
        apply_h_res, h_res, residual_ckpt
    )

    # Step 5: apply_h_post is checkpointed
    output = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
        apply_h_post, y, h_post
    )

    # Step 6: Final output
    final = output + mixed
    loss_ckpt = final.sum()

    # Register unified recompute
    manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

    # Verify checkpoint outputs are discarded
    assert h_pre.untyped_storage().size() == 0, "h_pre storage should be released"
    assert h_post.untyped_storage().size() == 0, "h_post storage should be released"
    assert h_res.untyped_storage().size() == 0, "h_res storage should be released"
    assert mixed.untyped_storage().size() == 0, "mixed storage should be released"
    assert output.untyped_storage().size() == 0, "output storage should be released"

    # Non-checkpointed tensors should still have storage
    assert agg.untyped_storage().size() > 0, "agg storage should NOT be released"
    assert y.untyped_storage().size() > 0, "y storage should NOT be released"

    # Backward
    loss_ckpt.backward()
    grad_x_ckpt = x_ckpt.grad.clone()
    grad_residual_ckpt = residual_ckpt.grad.clone()

    # Verify gradients match
    assert torch.allclose(grad_x_ckpt, grad_x_ref, atol=1e-6), (
        f"Gradients for x mismatch!\n"
        f"With manager: {grad_x_ckpt}\n"
        f"Reference: {grad_x_ref}"
    )
    assert torch.allclose(grad_residual_ckpt, grad_residual_ref, atol=1e-6), (
        f"Gradients for residual mismatch!\n"
        f"With manager: {grad_residual_ckpt}\n"
        f"Reference: {grad_residual_ref}"
    )

    Utils.destroy_model_parallel()
