# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Ported from https://github.com/NVIDIA-NeMo/nv-mistralai-megatron
# Revision: f1856ec3ef887125dc7ad81f312c9ec837a3bb11
# License: BSD-3-Clause (see the source repository LICENSE).

"""Numerical and API checks for the opt-in TP1 tiled loss."""

import pytest
import torch

from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestTritonCrossEntropy:
    def setup_method(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("vocab", [127, 4097, 262272])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("shift", [0.0, 1000.0])
    def test_loss_and_weighted_gradients(self, vocab, dtype, shift):
        from megatron.core import parallel_state
        from megatron.core.ops.cross_entropy import triton_cross_entropy

        torch.manual_seed(1234)
        source = (torch.randn(17, 3, vocab, device="cuda") * 3 + shift).to(dtype)
        logits = source.clone().requires_grad_()
        reference = source.float().clone().requires_grad_()
        # Deliberately use noncontiguous labels and an unaligned number of rows.
        target = torch.randint(vocab, (3, 17), device="cuda").transpose(0, 1)
        weights = torch.randn_like(target, dtype=torch.float32)
        weights[::3] = 0
        actual = triton_cross_entropy(
            logits, target, parallel_state.get_tensor_model_parallel_group()
        )
        expected = torch.nn.functional.cross_entropy(
            reference.reshape(-1, vocab), target.reshape(-1), reduction="none"
        ).view_as(target)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(
            logits.grad,
            reference.grad.to(dtype),
            rtol=8e-3 if dtype == torch.bfloat16 else 2e-5,
            atol=1e-7,
        )
        assert torch.equal(logits.detach(), source), "The loss must not mutate its input"
        assert torch.count_nonzero(logits.grad[::3]) == 0

    def test_rejects_tensor_parallel_group(self):
        from megatron.core.ops.cross_entropy import triton_cross_entropy

        class LargerGroup:
            def size(self):
                return 2

        with pytest.raises(ValueError, match="requires TP1"):
            triton_cross_entropy(
                torch.zeros(2, 128, device="cuda", dtype=torch.bfloat16),
                torch.zeros(2, device="cuda", dtype=torch.int64),
                LargerGroup(),
            )

    def test_large_vocabulary_uses_64_bit_offsets(self):
        from megatron.core import parallel_state
        from megatron.core.ops.cross_entropy import triton_cross_entropy

        # Last rows start past 2**31 elements. Small numerical cases cannot
        # detect accidental int32 address arithmetic in this vocabulary.
        source = torch.randn(8193, 262272, device="cuda", dtype=torch.bfloat16)
        logits = source.requires_grad_()
        target = torch.randint(262272, (8193,), device="cuda")
        reference = source[-2:].float().detach().requires_grad_()
        actual = triton_cross_entropy(
            logits, target, parallel_state.get_tensor_model_parallel_group()
        )
        expected = torch.nn.functional.cross_entropy(reference, target[-2:], reduction="none")
        torch.testing.assert_close(actual[-2:], expected, rtol=2e-5, atol=2e-6)
        actual[-2:].sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(
            logits.grad[-2:], reference.grad.bfloat16(), rtol=8e-3, atol=1e-7
        )
        assert torch.count_nonzero(logits.grad[0]) == 0

    @pytest.mark.parametrize("hidden", [1536, 2688])
    def test_projection_loss_parameter_gradients(self, hidden):
        from megatron.core import parallel_state
        from megatron.core.ops.cross_entropy import triton_cross_entropy
        from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

        torch.manual_seed(2104)
        source = torch.randn(17, hidden, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(262272, hidden, device="cuda", dtype=torch.bfloat16) * 0.01
        ax, rx = source.clone().requires_grad_(), source.clone().requires_grad_()
        aw, rw = weights.clone().requires_grad_(), weights.clone().requires_grad_()
        target = torch.randint(262272, (17,), device="cuda")
        target[0], target[-1] = 0, 262271
        mask = torch.rand(17, device="cuda")
        mask[::4] = 0
        group = parallel_state.get_tensor_model_parallel_group()
        actual = triton_cross_entropy(torch.nn.functional.linear(ax, aw), target, group)
        expected = vocab_parallel_cross_entropy(
            torch.nn.functional.linear(rx, rw), target, tp_group=group
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        (actual * mask).mean().backward()
        (expected * mask).mean().backward()
        # Check the composed projection's parameter and input gradients, not
        # just the logits derivative in isolation.
        for observed, reference in ((ax.grad, rx.grad), (aw.grad, rw.grad)):
            relative_l2 = (observed.float() - reference.float()).norm() / reference.float().norm()
            assert relative_l2.item() < 5e-5, relative_l2.item()
