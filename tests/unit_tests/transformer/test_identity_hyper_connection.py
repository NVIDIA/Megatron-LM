# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for identity hyper-connections (iHC).

iHC keeps the mHC H_pre / H_post mappings but fixes the residual mixing matrix
H_res to the identity: nothing is projected for it and the Sinkhorn-Knopp step
is skipped, while the downstream H_res / H_post / BDA kernels are unchanged.
Numerical references use full FP32 matmul precision so identity mixing preserves
the residual even on devices where TF32 is selected for the native batched matmul.
"""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import (
    MHCCheckpointManager,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import HyperConnectionTransformerLayer
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    """Initialize model parallelism once for this module.

    Share TP=1/PP=1 groups because these tests use the same parallel layout and only
    need a fresh RNG seed per test. Utils.destroy_model_parallel releases the owned
    process groups and their watchdog threads when this module finishes.
    """
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()


@pytest.fixture(autouse=True)
def _reseed(_model_parallel):
    model_parallel_cuda_manual_seed(123)


@pytest.fixture(autouse=True)
def _full_fp32_matmul():
    """Match the precision assumed by the strict identity/reference assertions.

    CI containers can enable TF32 by default. On GB200 the native H_res batched
    matmul then rounds the residual even for an identity matrix, unlike the
    elementwise reference. Restore the original setting for subsequent tests.
    """
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)


@pytest.fixture(autouse=True)
def _run_eagerly():
    """Run every test here with torch.compile stanced to eager.

    iHC's constant H_res carries requires_grad=False, so letting these tests compile
    would add guard variants to the mHC kernels (native_h_post_bda, native_h_aggregate,
    ...) that every mHC test in this process shares. Under torch's default recompile
    limit of 8 those extra variants can push a later mHC eager-vs-graph comparison into
    the compiled-to-eager fallback on one side only (seen in CI on
    test_mhc_attention_graph_split). The eager stance leaves the shared compile caches
    exactly as the mHC tests expect them; the compiled kernels themselves are covered by
    the mHC tests.
    """
    with torch.compiler.set_stance("force_eager"):
        yield


def _make_config(identity, hidden_size=64, num_streams=4, **extra):
    """Build a hyper-connection TransformerConfig; ``identity`` selects iHC over mHC."""
    base = dict(
        num_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=4,
        use_cpu_initialization=True,
        enable_hyper_connections=True,
        enable_identity_hyper_connections=identity,
        num_residual_streams=num_streams,
        mhc_sinkhorn_iterations=5,
        mhc_init_gating_factor=0.01,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    base.update(extra)
    return TransformerConfig(**base)


def _identity_h_res(s, b, n, dtype):
    return torch.eye(n, dtype=dtype, device="cuda").expand(s, b, n, n)


class TestIdentityHyperConnectionConfig:
    def test_requires_hyper_connections(self):
        with pytest.raises(ValueError, match="enable_identity_hyper_connections requires"):
            TransformerConfig(
                num_layers=2,
                hidden_size=64,
                num_attention_heads=4,
                enable_identity_hyper_connections=True,
            )


class TestIdentityHyperConnectionModule:
    def _make_module(self, identity=True, hidden_size=64, num_streams=4, **extra):
        config = _make_config(identity, hidden_size, num_streams, **extra)
        module = HyperConnectionModule(config=config, layer_number=1)
        module.cuda()
        return module

    def test_parameter_layout_drops_h_res(self):
        n, C = 4, 64
        ihc = self._make_module(identity=True, num_streams=n, hidden_size=C)
        assert ihc.identity_h_res
        assert ihc.mapping_proj.weight.shape == (2 * n, n * C)
        assert ihc.bias.shape == (2 * n,)
        assert ihc.alpha_res is None
        assert {name for name, _ in ihc.named_parameters()} == {
            "mapping_proj.weight",
            "alpha_pre",
            "alpha_post",
            "bias",
        }
        assert ihc._sinkhorn_op is None
        assert ihc._proj_rms_compute_h_op is None

        mhc = self._make_module(identity=False, num_streams=n, hidden_size=C)
        assert not mhc.identity_h_res
        assert mhc.mapping_proj.weight.shape == (n * n + 2 * n, n * C)
        assert mhc.bias.shape == (n * n + 2 * n,)
        assert mhc.alpha_res is not None
        assert mhc._sinkhorn_op is not None

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_compute_mappings_returns_identity_h_res(self, dtype):
        s, b, n, C = 8, 2, 4, 64
        module = self._make_module(num_streams=n, hidden_size=C)
        x = torch.randn(s, b, n * C, device="cuda", dtype=dtype)

        h_pre, h_post, h_res = module.compute_mappings(x)

        assert h_pre.shape == (s, b, n) and h_pre.dtype == dtype
        assert h_post.shape == (s, b, n) and h_post.dtype == dtype
        assert h_res.shape == (s, b, n, n) and h_res.dtype == dtype
        assert h_res.is_contiguous()
        assert not h_res.requires_grad
        torch.testing.assert_close(h_res, _identity_h_res(s, b, n, dtype), atol=0.0, rtol=0.0)
        # H_pre / H_post keep their mHC ranges: sigmoid (+eps) and 2 * sigmoid.
        assert torch.all(h_pre > 0) and torch.all(h_pre <= 1 + 1e-3)
        assert torch.all(h_post >= 0) and torch.all(h_post <= 2)

    def test_forward_supports_empty_sequence(self):
        module = self._make_module(num_streams=2, hidden_size=4)
        hidden_states = torch.empty(0, 1, 8, device="cuda", requires_grad=True)

        aggregated, h_res, h_post, residual = module(hidden_states)

        assert aggregated.shape == (0, 1, 4)
        assert h_res.shape == (0, 1, 2, 2)
        assert h_post.shape == (0, 1, 2)
        assert residual.shape == (0, 1, 8)
        (aggregated.sum() + h_post.sum() + residual.sum()).backward()
        assert hidden_states.grad is not None
        assert module.mapping_proj.weight.grad is not None

    def test_output_is_residual_plus_h_post_expansion(self):
        """With H_res = I the layer output is x_l + H_post^T @ (F + bias), stream by stream."""
        s, b, n, C = 8, 2, 4, 64
        module = self._make_module(num_streams=n, hidden_size=C)
        hidden_states = torch.randn(s, b, n * C, device="cuda")
        layer_output = torch.randn(s, b, C, device="cuda")
        layer_bias = torch.randn(C, device="cuda")

        _, h_res, h_post, residual = module(hidden_states)
        output = module.fused_h_res_h_post_bda(
            h_res,
            residual,
            h_post,
            (layer_output, layer_bias),
            dropout_prob=0.0,
            training=True,
            fused=False,
        )

        expected = hidden_states.view(s, b, n, C) + h_post.unsqueeze(-1) * (
            layer_output + layer_bias
        ).unsqueeze(2)
        torch.testing.assert_close(output, expected.view(s, b, n * C))

    def test_matches_mhc_with_identity_residual_mixing(self):
        """iHC must reproduce mHC whose Sinkhorn output is forced to the identity."""
        s, b, n, C = 8, 2, 4, 64
        ihc = self._make_module(identity=True, num_streams=n, hidden_size=C)
        mhc = self._make_module(identity=False, num_streams=n, hidden_size=C)
        with torch.no_grad():
            mhc.mapping_proj.weight[: 2 * n].copy_(ihc.mapping_proj.weight)
            mhc.bias[: 2 * n].copy_(ihc.bias)
            mhc.alpha_pre.copy_(ihc.alpha_pre)
            mhc.alpha_post.copy_(ihc.alpha_post)

        def identity_sinkhorn(logits, num_iterations, eps):
            return torch.eye(n, dtype=logits.dtype, device=logits.device).expand_as(logits)

        mhc._sinkhorn_op = identity_sinkhorn

        hidden_states = torch.randn(s, b, n * C, device="cuda")
        layer_output = torch.randn(s, b, C, device="cuda")
        mix = torch.randn(s, b, C, device="cuda")

        def run(module):
            hs = hidden_states.clone().requires_grad_(True)
            aggregated, h_res, h_post, residual = module(hs)
            output = module.fused_h_res_h_post_bda(
                h_res,
                residual,
                h_post,
                (aggregated * mix + layer_output, None),
                dropout_prob=0.0,
                training=True,
                fused=False,
            )
            output.sum().backward()
            return output.detach(), hs.grad

        out_ihc, grad_ihc = run(ihc)
        out_mhc, grad_mhc = run(mhc)

        torch.testing.assert_close(out_ihc, out_mhc, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(grad_ihc, grad_mhc, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(
            ihc.mapping_proj.weight.grad,
            mhc.mapping_proj.weight.grad[: 2 * n],
            atol=1e-5,
            rtol=1e-4,
        )
        torch.testing.assert_close(ihc.bias.grad, mhc.bias.grad[: 2 * n], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(ihc.alpha_pre.grad, mhc.alpha_pre.grad, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(ihc.alpha_post.grad, mhc.alpha_post.grad, atol=1e-5, rtol=1e-4)

    def test_forward_with_checkpoint_matches_normal(self):
        s, b, n, C = 8, 2, 4, 64
        module = self._make_module(num_streams=n, hidden_size=C)
        hidden_states = torch.randn(s, b, n * C, device="cuda")
        layer_output = torch.randn(s, b, C, device="cuda")

        def run(manager):
            module.zero_grad(set_to_none=True)
            hs = hidden_states.clone().requires_grad_(True)
            aggregated, h_res, h_post, residual = module(hs, mhc_recompute_manager=manager)
            torch.testing.assert_close(
                h_res, _identity_h_res(s, b, n, hs.dtype), atol=0.0, rtol=0.0
            )
            output = module.fused_h_res_h_post_bda(
                h_res,
                residual,
                h_post,
                (aggregated + layer_output, None),
                dropout_prob=0.0,
                training=True,
                fused=False,
                manager=manager,
            )
            loss = output.sum()
            if manager is not None:
                manager.discard_all_outputs_and_register_unified_recompute(loss)
            loss.backward()
            return output.detach().clone(), hs.grad.clone(), module.mapping_proj.weight.grad.clone()

        out_ref, grad_ref, wgrad_ref = run(None)
        out_ckpt, grad_ckpt, wgrad_ckpt = run(MHCCheckpointManager())

        torch.testing.assert_close(out_ckpt, out_ref)
        torch.testing.assert_close(grad_ckpt, grad_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(wgrad_ckpt, wgrad_ref, atol=1e-5, rtol=1e-5)

    def test_use_fused_mhc_keeps_native_projection(self):
        """use_fused_mhc still fuses aggregate / H_post_bda; the projection stays native."""
        s, b, n, C = 8, 2, 4, 64
        module = self._make_module(num_streams=n, hidden_size=C, use_fused_mhc=True)
        assert module._proj_rms_compute_h_op is None
        assert module._sinkhorn_op is None

        hidden_states = torch.randn(
            s, b, n * C, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        aggregated, h_res, h_post, residual = module(hidden_states)
        torch.testing.assert_close(
            h_res, _identity_h_res(s, b, n, torch.bfloat16), atol=0.0, rtol=0.0
        )
        output = module.fused_h_res_h_post_bda(
            h_res,
            residual,
            h_post,
            (aggregated, None),
            dropout_prob=0.0,
            training=True,
            fused=False,
        )
        output.sum().backward()

        assert hidden_states.grad is not None
        assert module.mapping_proj.weight.grad is not None
        assert module.alpha_pre.grad is not None
        assert module.alpha_post.grad is not None


class TestIdentityHyperConnectionTransformerLayer:
    def _make_layer(self, hidden_size=64, num_streams=4, **extra):
        config = _make_config(True, hidden_size, num_streams, **extra)
        layer_spec = get_gpt_layer_with_transformer_engine_spec(enable_hyper_connection=True)
        layer = HyperConnectionTransformerLayer(config, layer_spec.submodules, layer_number=1)
        layer.cuda()
        layer.train()
        return layer

    def test_layer_builds_identity_hyper_connections(self):
        layer = self._make_layer()
        for module in (layer.self_attention_hyper_connection, layer.mlp_hyper_connection):
            assert isinstance(module, HyperConnectionModule)
            assert module.identity_h_res
            assert module.alpha_res is None
            assert module.mapping_proj.weight.shape[0] == 2 * layer.config.num_residual_streams

    def test_forward_backward(self):
        hidden_size, n, s, b = 64, 4, 8, 2
        layer = self._make_layer(hidden_size=hidden_size, num_streams=n)
        hidden_states = torch.randn(s, b, n * hidden_size, device="cuda", requires_grad=True)
        attention_mask = torch.ones((1, 1, s, s), dtype=bool, device="cuda")

        output, _ = layer(hidden_states=hidden_states, attention_mask=attention_mask)

        assert output.shape == (s, b, n * hidden_size)
        output.sum().backward()
        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == hidden_states.shape

    def test_forward_backward_with_mhc_recompute(self):
        hidden_size, n, s, b = 64, 4, 8, 2
        layer = self._make_layer(
            hidden_size=hidden_size,
            num_streams=n,
            recompute_granularity="selective",
            recompute_modules=["core_attn", "mhc"],
        )
        hidden_states = torch.randn(s, b, n * hidden_size, device="cuda", requires_grad=True)
        attention_mask = torch.ones((1, 1, s, s), dtype=bool, device="cuda")

        manager = MHCCheckpointManager()
        manager.is_last_layer_in_recompute_block = True
        output, _ = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            mhc_recompute_manager=manager,
        )

        assert output.shape == (s, b, n * hidden_size)
        manager.discard_all_outputs_and_register_unified_recompute(output)
        output.sum().backward()
        assert hidden_states.grad is not None
