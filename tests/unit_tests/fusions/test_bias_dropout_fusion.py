# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_bias_dropout import _bias_dropout_add_func, get_bias_dropout_add

# ---------------------------------------------------------------------------
# Existing test: fused vs. unfused parity (same dtype)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("training", [True, False])
def test_bias_dropout_add(dtype, training):
    torch.manual_seed(42)
    device = "cuda"
    B, H = 16, 64

    # Initialize inputs
    x = torch.randn(B, H, dtype=dtype, device=device, requires_grad=training)
    residual = torch.randn(B, H, dtype=dtype, device=device, requires_grad=training)
    bias = torch.randn(H, dtype=dtype, device=device)

    # Run un-fused as reference
    torch.manual_seed(42)
    ref_fn = get_bias_dropout_add(training=training, fused=False)
    x_ref = x.clone().detach().requires_grad_(training)
    residual_ref = residual.clone().detach().requires_grad_(training)
    out_ref = ref_fn((x_ref, bias), residual_ref, prob=0.0)

    # Run fused
    torch.manual_seed(42)
    fused_fn = get_bias_dropout_add(training=training, fused=True)
    x_fused = x.clone().detach().requires_grad_(training)
    residual_fused = residual.clone().detach().requires_grad_(training)
    out_fused = fused_fn((x_fused, bias), residual_fused, prob=0.0)

    tols = dict(rtol=1e-6, atol=1e-6) if dtype is torch.float32 else dict(rtol=2e-2, atol=1e-2)

    assert out_fused.dtype == out_ref.dtype
    assert torch.allclose(out_fused, out_ref, **tols)

    if training:
        grad = torch.randn_like(out_ref)
        out_ref.backward(grad)
        out_fused.backward(grad)

        assert torch.allclose(x_ref.grad, x_fused.grad, **tols)
        assert torch.allclose(residual_ref.grad, residual_fused.grad, **tols)
    else:
        # In‑place check for inference
        assert out_fused.data_ptr() == x_fused.data_ptr()
        assert torch.allclose(out_fused, x_fused, **tols)


# ============================================================================
# Tests for fp32 residual connection fix
# ============================================================================
#
# The fix reverses the casting direction in _bias_dropout_add_func: when
# x is bf16/fp16 and residual is fp32, x (and bias) are upcast to fp32
# so that the residual stream stays in fp32.  The OLD (broken) code did the
# opposite: it downcast the fp32 residual to match x's dtype, destroying
# the fp32 residual stream from the very first layer.
# ============================================================================


class TestFp32ResidualPreservation:
    """Tests that _bias_dropout_add_func preserves fp32 residual dtype."""

    device = "cuda"
    B, H = 16, 64

    # -- helpers --------------------------------------------------------

    @staticmethod
    def _reference_bias_dropout_add(x, bias, residual, prob, training):
        """Manual reference computation in fp32 (no dropout for determinism)."""
        x_fp32 = x.float()
        r_fp32 = residual.float()
        if bias is not None:
            b_fp32 = bias.float()
            return r_fp32 + torch.nn.functional.dropout(x_fp32 + b_fp32, p=prob, training=training)
        return r_fp32 + torch.nn.functional.dropout(x_fp32, p=prob, training=training)

    # -- core: output dtype must follow residual, not x -----------------

    @pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_output_dtype_is_residual_dtype(self, x_dtype, training, has_bias):
        """The output tensor must have the same dtype as the residual (fp32),
        NOT x's dtype.  This is the central invariant of the fix."""
        x = torch.randn(self.B, self.H, dtype=x_dtype, device=self.device)
        residual = torch.randn(self.B, self.H, dtype=torch.float32, device=self.device)
        bias = torch.randn(self.H, dtype=x_dtype, device=self.device) if has_bias else None

        out = _bias_dropout_add_func((x, bias), residual, prob=0.0, training=training)

        assert (
            out.dtype == torch.float32
        ), f"Output dtype {out.dtype} must be fp32 (residual dtype), not {x_dtype}"

    # -- numerical correctness of the upcast path ----------------------

    @pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_numerical_correctness_with_fp32_residual(self, x_dtype, has_bias):
        """Forward result should match a reference computed entirely in fp32."""
        torch.manual_seed(7)
        x = torch.randn(self.B, self.H, dtype=x_dtype, device=self.device)
        residual = torch.randn(self.B, self.H, dtype=torch.float32, device=self.device)
        bias = torch.randn(self.H, dtype=x_dtype, device=self.device) if has_bias else None

        out = _bias_dropout_add_func((x, bias), residual, prob=0.0, training=True)

        ref = self._reference_bias_dropout_add(x, bias, residual, prob=0.0, training=True)
        # fp32 tolerance – the only imprecision is the upcast from bf16/fp16
        assert torch.allclose(
            out, ref, rtol=1e-5, atol=1e-5
        ), f"Max diff = {(out - ref).abs().max().item()}"

    # -- backward: gradients flow correctly through the upcast ---------

    @pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_backward_with_fp32_residual(self, x_dtype, has_bias):
        """Gradients should be computed for x and residual when dtypes differ."""
        x = torch.randn(self.B, self.H, dtype=x_dtype, device=self.device, requires_grad=True)
        residual = torch.randn(
            self.B, self.H, dtype=torch.float32, device=self.device, requires_grad=True
        )
        bias = (
            torch.randn(self.H, dtype=x_dtype, device=self.device, requires_grad=True)
            if has_bias
            else None
        )

        out = _bias_dropout_add_func((x, bias), residual, prob=0.0, training=True)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        assert x.grad is not None, "x.grad must not be None"
        assert residual.grad is not None, "residual.grad must not be None"
        assert x.grad.dtype == x_dtype, f"x.grad dtype should be {x_dtype}"
        assert residual.grad.dtype == torch.float32, "residual.grad must stay fp32"
        if has_bias:
            assert bias.grad is not None, "bias.grad must not be None"

    # -- same dtype: no regression (both fp32 or both bf16) ------------

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_same_dtype_no_regression(self, dtype, has_bias):
        """When x and residual share the same dtype, behaviour is unchanged."""
        torch.manual_seed(99)
        x = torch.randn(self.B, self.H, dtype=dtype, device=self.device)
        residual = torch.randn(self.B, self.H, dtype=dtype, device=self.device)
        bias = torch.randn(self.H, dtype=dtype, device=self.device) if has_bias else None

        out = _bias_dropout_add_func((x, bias), residual, prob=0.0, training=True)

        assert out.dtype == dtype
        ref = self._reference_bias_dropout_add(x, bias, residual, prob=0.0, training=True)
        ref = ref.to(dtype)
        tols = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=2e-2, atol=1e-2)
        assert torch.allclose(out, ref, **tols)

    # -- inference in-place optimisation still fires for same dtype ----

    def test_inplace_inference_same_dtype(self):
        """In eval mode with no grad, the in-place path should still work."""
        x = torch.randn(self.B, self.H, dtype=torch.bfloat16, device=self.device)
        residual = torch.randn(self.B, self.H, dtype=torch.bfloat16, device=self.device)
        bias = torch.randn(self.H, dtype=torch.bfloat16, device=self.device)

        x_ptr = x.data_ptr()
        out = _bias_dropout_add_func((x, bias), residual, prob=0.0, training=False)
        # In-place: output should reuse x's storage
        assert out.data_ptr() == x_ptr

    # -- inference with mixed dtypes should NOT be in-place ------------

    @pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float16])
    def test_no_inplace_when_dtypes_differ(self, x_dtype):
        """When x is low-precision but residual is fp32, the upcast creates
        a new tensor so the in-place optimisation must NOT fire."""
        x = torch.randn(self.B, self.H, dtype=x_dtype, device=self.device)
        residual = torch.randn(self.B, self.H, dtype=torch.float32, device=self.device)

        x_ptr = x.data_ptr()
        out = _bias_dropout_add_func((x, None), residual, prob=0.0, training=False)
        # The upcast `x = x.to(residual.dtype)` creates a new tensor,
        # so the output must NOT alias the original x buffer.
        assert out.dtype == torch.float32
        assert out.data_ptr() != x_ptr

    # -- dropout prob > 0 still works with mixed dtypes ----------------

    @pytest.mark.parametrize("has_bias", [True, False])
    def test_dropout_with_fp32_residual(self, has_bias):
        """Smoke test: non-zero dropout with mixed dtypes doesn't crash."""
        x = torch.randn(self.B, self.H, dtype=torch.bfloat16, device=self.device)
        residual = torch.randn(self.B, self.H, dtype=torch.float32, device=self.device)
        bias = torch.randn(self.H, dtype=torch.bfloat16, device=self.device) if has_bias else None

        out = _bias_dropout_add_func((x, bias), residual, prob=0.5, training=True)
        assert out.dtype == torch.float32
        # With dropout, some elements should be zeroed (before residual add)
        # so the output shouldn't be identical to the no-dropout case.
        out_no_drop = _bias_dropout_add_func((x, bias), residual, prob=0.0, training=True)
        # They *can* be equal with very low probability; just check dtypes
        assert out_no_drop.dtype == torch.float32

    # -- get_bias_dropout_add wrappers preserve fp32 residual ----------

    @pytest.mark.parametrize("training", [True, False])
    @pytest.mark.parametrize("fused", [True, False])
    def test_get_bias_dropout_add_fp32_residual(self, training, fused):
        """All four (training × fused) wrappers returned by get_bias_dropout_add
        should preserve fp32 residual dtype."""
        fn = get_bias_dropout_add(training=training, fused=fused)
        x = torch.randn(self.B, self.H, dtype=torch.bfloat16, device=self.device)
        residual = torch.randn(self.B, self.H, dtype=torch.float32, device=self.device)
        bias = torch.randn(self.H, dtype=torch.bfloat16, device=self.device)

        out = fn((x, bias), residual, prob=0.0)
        assert out.dtype == torch.float32, (
            f"get_bias_dropout_add(training={training}, fused={fused}) "
            f"returned {out.dtype}, expected fp32"
        )


# ============================================================================
# Tests simulating the multi-layer residual stream scenario
# ============================================================================


class TestFp32ResidualStreamAcrossLayers:
    """Simulates what happens across multiple transformer layers to ensure
    the fp32 residual stream is not degraded."""

    device = "cuda"

    def test_residual_stays_fp32_across_simulated_layers(self):
        """Simulate N layers of bias-dropout-add with bf16 x and fp32 residual.
        The residual should remain fp32 throughout — this is the scenario that
        was broken before the fix."""
        B, H = 4, 32
        num_layers = 8

        # Start with an fp32 residual (as the embedding layer would emit)
        residual = torch.randn(B, H, dtype=torch.float32, device=self.device)

        for layer_idx in range(num_layers):
            # Each layer produces bf16 output (as the attention/MLP would)
            x = torch.randn(B, H, dtype=torch.bfloat16, device=self.device)
            bias = torch.randn(H, dtype=torch.bfloat16, device=self.device)

            residual = _bias_dropout_add_func((x, bias), residual, prob=0.0, training=True)

            assert (
                residual.dtype == torch.float32
            ), f"Layer {layer_idx}: residual dtype degraded to {residual.dtype}"

    def test_residual_stays_fp32_no_bias(self):
        """Same multi-layer simulation but without bias tensors."""
        B, H = 4, 32
        num_layers = 8

        residual = torch.randn(B, H, dtype=torch.float32, device=self.device)

        for layer_idx in range(num_layers):
            x = torch.randn(B, H, dtype=torch.bfloat16, device=self.device)

            residual = _bias_dropout_add_func((x, None), residual, prob=0.0, training=True)

            assert (
                residual.dtype == torch.float32
            ), f"Layer {layer_idx}: residual dtype degraded to {residual.dtype}"

    def test_fp32_residual_precision_advantage(self):
        """Demonstrate that fp32 residuals accumulate more accurately than
        bf16 residuals over many additions — the whole point of the feature."""
        B, H = 2, 16
        num_layers = 50
        torch.manual_seed(42)

        # Ground truth: everything in fp64
        residual_fp64 = torch.randn(B, H, dtype=torch.float64, device=self.device)

        # Track fp32 and bf16 residual streams
        residual_fp32 = residual_fp64.float()
        residual_bf16 = residual_fp64.to(torch.bfloat16)

        for _ in range(num_layers):
            x_fp64 = torch.randn(B, H, dtype=torch.float64, device=self.device)
            x_bf16 = x_fp64.to(torch.bfloat16)

            # fp64 reference
            residual_fp64 = residual_fp64 + x_fp64

            # fp32 residual path (the fix)
            residual_fp32 = _bias_dropout_add_func(
                (x_bf16, None), residual_fp32, prob=0.0, training=True
            )

            # bf16 residual path (the old broken behaviour)
            residual_bf16 = _bias_dropout_add_func(
                (x_bf16, None), residual_bf16.to(torch.bfloat16), prob=0.0, training=True
            )

        err_fp32 = (residual_fp32.double() - residual_fp64).abs().mean().item()
        err_bf16 = (residual_bf16.double() - residual_fp64).abs().mean().item()

        # fp32 residual should be meaningfully more precise
        assert err_fp32 < err_bf16, (
            f"fp32 residual error ({err_fp32:.6e}) should be less than "
            f"bf16 residual error ({err_bf16:.6e})"
        )
