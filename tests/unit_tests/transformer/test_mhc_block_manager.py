# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.tensor_parallel.random import (
    CheckpointManager,
    CheckpointWithoutOutput,
    initialize_rng_tracker,
)
from tests.unit_tests.test_utilities import Utils


class TestCheckpointWithoutOutputManagerAPI:
    """Test CheckpointWithoutOutput integration with CheckpointManager."""

    def setup_method(self, method):
        Utils.initialize_model_parallel()
        initialize_rng_tracker(force_reset=True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_auto_register(self):
        """CheckpointWithoutOutput auto-registers to manager when ckpt_manager is provided."""
        manager = CheckpointManager()

        def func(x):
            return x * 2 + 1

        input_t = torch.randn(4, 4, device='cuda', requires_grad=True)

        ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
        y = ckpt.checkpoint(func, input_t)

        assert len(manager.checkpoints) == 1
        assert manager.checkpoints[0] is ckpt

        ckpt2 = CheckpointWithoutOutput(ckpt_manager=manager)
        y2 = ckpt2.checkpoint(torch.nn.functional.gelu, y)

        assert len(manager.checkpoints) == 2
        assert manager.checkpoints[1] is ckpt2

        loss = y2.sum()
        manager.discard_all_outputs_and_register_unified_recompute(loss)
        loss.backward()

        assert input_t.grad is not None

    def test_discard_is_noop_with_manager(self):
        """discard_output_and_register_recompute is a NO-OP when ckpt_manager is set."""
        manager = CheckpointManager()

        def func1(x):
            return x * 2

        def func2(x):
            return torch.nn.functional.gelu(x)

        input_ref = torch.randn(4, 4, device='cuda', requires_grad=True)
        y1_ref = func1(input_ref)
        y2_ref = func2(y1_ref)
        loss_ref = y2_ref.sum()
        loss_ref.backward()
        grad_ref = input_ref.grad.clone()

        input_ckpt = input_ref.detach().clone().requires_grad_(True)

        ckpt1 = CheckpointWithoutOutput(ckpt_manager=manager)
        y1 = ckpt1.checkpoint(func1, input_ckpt)
        ckpt1.discard_output_and_register_recompute(y1)

        ckpt2 = CheckpointWithoutOutput(ckpt_manager=manager)
        y2 = ckpt2.checkpoint(func2, y1)
        ckpt2.discard_output_and_register_recompute(y2)

        assert y1.untyped_storage().size() > 0, "y1 should NOT be discarded yet"
        assert y2.untyped_storage().size() > 0, "y2 should NOT be discarded yet"

        loss_ckpt = y2.sum()
        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

        assert y1.untyped_storage().size() == 0, "y1 should be discarded after manager call"
        assert y2.untyped_storage().size() == 0, "y2 should be discarded after manager call"

        loss_ckpt.backward()
        grad_ckpt = input_ckpt.grad.clone()

        assert torch.allclose(grad_ckpt, grad_ref, atol=1e-6)

    def test_backward_compat_without_manager(self):
        """CheckpointWithoutOutput without ckpt_manager should work exactly as before."""

        def func(x):
            return torch.nn.functional.gelu(x)

        input_ref = torch.randn(4, 4, device='cuda', requires_grad=True)
        y_ref = func(input_ref)
        z_ref = y_ref * 2
        loss_ref = z_ref.sum()
        loss_ref.backward()
        grad_ref = input_ref.grad.clone()

        input_ckpt = input_ref.detach().clone().requires_grad_(True)

        ckpt = CheckpointWithoutOutput()
        y = ckpt.checkpoint(func, input_ckpt)
        z = y * 2
        ckpt.discard_output_and_register_recompute(z)

        assert y.untyped_storage().size() == 0

        loss_ckpt = z.sum()
        loss_ckpt.backward()
        grad_ckpt = input_ckpt.grad.clone()

        assert torch.allclose(grad_ckpt, grad_ref, atol=1e-6)

    def test_error_handling(self):
        """CheckpointManager rejects invalid add_checkpoint calls."""
        manager = CheckpointManager()

        with pytest.raises(TypeError):
            manager.add_checkpoint("not a checkpoint")

        ckpt = CheckpointWithoutOutput()
        with pytest.raises(ValueError):
            manager.add_checkpoint(ckpt)


class TestCheckpointManagerSequentialChain:
    """Test CheckpointManager with sequential checkpoint chains."""

    def setup_method(self, method):
        Utils.initialize_model_parallel()
        initialize_rng_tracker(force_reset=True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_basic_sequential_chain(self):
        """Three sequential checkpoints: gradients match non-checkpointed version."""

        def func1(x):
            return x * 2 + 1

        def func2(x):
            return torch.nn.functional.gelu(x)

        def func3(x):
            return x * x + x

        input_ref = torch.randn(4, 4, device='cuda', requires_grad=True)
        input_ckpt = input_ref.detach().clone().requires_grad_(True)

        y1_ref = func1(input_ref)
        y2_ref = func2(y1_ref)
        y3_ref = func3(y2_ref)
        loss_ref = y3_ref.sum()
        loss_ref.backward()
        grad_ref = input_ref.grad.clone()

        manager = CheckpointManager()

        y1 = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func1, input_ckpt)
        y2 = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func2, y1)
        y3 = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func3, y2)

        loss_ckpt = y3.sum()
        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

        assert y1.untyped_storage().size() == 0, "y1 storage should be released"
        assert y2.untyped_storage().size() == 0, "y2 storage should be released"
        assert y3.untyped_storage().size() == 0, "y3 storage should be released"

        loss_ckpt.backward()
        grad_ckpt = input_ckpt.grad.clone()

        assert torch.allclose(
            grad_ckpt, grad_ref, atol=1e-6
        ), f"Gradients mismatch!\nWith manager: {grad_ckpt}\nReference: {grad_ref}"

    def test_sequential_chain_with_dropout(self):
        """RNG state is restored during recompute so dropout gradients match."""

        def func_with_dropout(x):
            return torch.nn.functional.dropout(x, p=0.3, training=True)

        def func2(x):
            return torch.nn.functional.gelu(x)

        input_ref = torch.randn(4, 4, device='cuda', requires_grad=True)
        input_ckpt = input_ref.detach().clone().requires_grad_(True)

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        y1_ref = func_with_dropout(input_ref)
        y2_ref = func2(y1_ref)
        loss_ref = y2_ref.sum()
        loss_ref.backward()
        grad_ref = input_ref.grad.clone()

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        manager = CheckpointManager()

        y1 = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func_with_dropout, input_ckpt)
        y2 = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func2, y1)

        loss_ckpt = y2.sum()
        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

        loss_ckpt.backward()
        grad_ckpt = input_ckpt.grad.clone()

        assert torch.allclose(
            grad_ckpt, grad_ref, atol=1e-6
        ), f"Gradients with dropout mismatch!\nWith manager: {grad_ckpt}\nReference: {grad_ref}"

    def test_multiple_outputs(self):
        """CheckpointManager handles functions that return multiple outputs."""

        def func_multi_output(x):
            return x * 2, x + 1

        def func_combine(a, b):
            return a + b

        input_ref = torch.randn(4, 4, device='cuda', requires_grad=True)
        input_ckpt = input_ref.detach().clone().requires_grad_(True)

        y1a_ref, y1b_ref = func_multi_output(input_ref)
        y2_ref = func_combine(y1a_ref, y1b_ref)
        loss_ref = y2_ref.sum()
        loss_ref.backward()
        grad_ref = input_ref.grad.clone()

        manager = CheckpointManager()

        y1a, y1b = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            func_multi_output, input_ckpt
        )
        y2 = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func_combine, y1a, y1b)

        loss_ckpt = y2.sum()
        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

        loss_ckpt.backward()
        grad_ckpt = input_ckpt.grad.clone()

        assert torch.allclose(grad_ckpt, grad_ref, atol=1e-6), (
            f"Gradients mismatch with multiple outputs!\n"
            f"With manager: {grad_ckpt}\nReference: {grad_ref}"
        )


class TestCheckpointManagerPartialCheckpoint:
    """Test CheckpointManager with partial checkpointing (some ops not checkpointed)."""

    def setup_method(self, method):
        Utils.initialize_model_parallel()
        initialize_rng_tracker(force_reset=True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_partial_checkpoint(self):
        """
        Only f and h are checkpointed; g is a regular operation.

        Computation chain:
            a --[f]--> b --[g]--> c --[h]--> d --[sum]--> loss
        """

        def func_f(x):
            return torch.nn.functional.gelu(x * 2 + 1)

        def func_g(x):
            return x * 3 - 2

        def func_h(x):
            return torch.sigmoid(x) + x

        input_ref = torch.randn(4, 4, device='cuda', requires_grad=True)

        b_ref = func_f(input_ref)
        c_ref = func_g(b_ref)
        d_ref = func_h(c_ref)
        loss_ref = d_ref.sum()
        loss_ref.backward()
        grad_ref = input_ref.grad.clone()

        input_ckpt = input_ref.detach().clone().requires_grad_(True)

        manager = CheckpointManager()

        b = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func_f, input_ckpt)
        c = func_g(b)
        d = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(func_h, c)

        loss_ckpt = d.sum()
        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

        assert b.untyped_storage().size() == 0, "b storage should be released"
        assert d.untyped_storage().size() == 0, "d storage should be released"
        assert c.untyped_storage().size() > 0, "c storage should NOT be released (not checkpointed)"

        loss_ckpt.backward()
        grad_ckpt = input_ckpt.grad.clone()

        assert torch.allclose(grad_ckpt, grad_ref, atol=1e-6), (
            f"Gradients mismatch with partial checkpoint!\n"
            f"With manager: {grad_ckpt}\nReference: {grad_ref}"
        )

    def test_partial_checkpoint_with_tuple_output(self):
        """
        Mimics HyperConnection's computation pattern with tuple outputs.

        - compute_mappings: checkpointed, returns tuple (h_pre, h_post, h_res)
        - aggregate: NOT checkpointed
        - apply_h_res: checkpointed
        - apply_h_post: checkpointed
        """

        def compute_mappings(x):
            h_pre = torch.sigmoid(x.mean(dim=-1, keepdim=True).expand_as(x))
            h_post = torch.tanh(x.sum(dim=-1, keepdim=True).expand_as(x))
            h_res = torch.relu(x)
            return h_pre, h_post, h_res

        def aggregate(x, h_pre):
            return x * h_pre

        def apply_h_res(h_res, residual):
            return h_res + residual * 0.5

        def apply_h_post(y, h_post):
            return y * h_post + y

        x_ref = torch.randn(4, 4, device='cuda', requires_grad=True)
        residual_ref = torch.randn(4, 4, device='cuda', requires_grad=True)

        h_pre_ref, h_post_ref, h_res_ref = compute_mappings(x_ref)
        agg_ref = aggregate(x_ref, h_pre_ref)
        y_ref = torch.nn.functional.gelu(agg_ref)
        mixed_ref = apply_h_res(h_res_ref, residual_ref)
        output_ref = apply_h_post(y_ref, h_post_ref)
        final_ref = output_ref + mixed_ref
        loss_ref = final_ref.sum()
        loss_ref.backward()
        grad_x_ref = x_ref.grad.clone()
        grad_residual_ref = residual_ref.grad.clone()

        x_ckpt = x_ref.detach().clone().requires_grad_(True)
        residual_ckpt = residual_ref.detach().clone().requires_grad_(True)

        manager = CheckpointManager()

        h_pre, h_post, h_res = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            compute_mappings, x_ckpt
        )
        agg = aggregate(x_ckpt, h_pre)
        y = torch.nn.functional.gelu(agg)
        mixed = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(
            apply_h_res, h_res, residual_ckpt
        )
        output = CheckpointWithoutOutput(ckpt_manager=manager).checkpoint(apply_h_post, y, h_post)

        final = output + mixed
        loss_ckpt = final.sum()

        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

        assert h_pre.untyped_storage().size() == 0, "h_pre storage should be released"
        assert h_post.untyped_storage().size() == 0, "h_post storage should be released"
        assert h_res.untyped_storage().size() == 0, "h_res storage should be released"
        assert mixed.untyped_storage().size() == 0, "mixed storage should be released"
        assert output.untyped_storage().size() == 0, "output storage should be released"

        assert agg.untyped_storage().size() > 0, "agg storage should NOT be released"
        assert y.untyped_storage().size() > 0, "y storage should NOT be released"

        loss_ckpt.backward()
        grad_x_ckpt = x_ckpt.grad.clone()
        grad_residual_ckpt = residual_ckpt.grad.clone()

        assert torch.allclose(
            grad_x_ckpt, grad_x_ref, atol=1e-6
        ), f"Gradients for x mismatch!\nWith manager: {grad_x_ckpt}\nReference: {grad_x_ref}"
        assert torch.allclose(grad_residual_ckpt, grad_residual_ref, atol=1e-6), (
            f"Gradients for residual mismatch!\n"
            f"With manager: {grad_residual_ckpt}\nReference: {grad_residual_ref}"
        )
