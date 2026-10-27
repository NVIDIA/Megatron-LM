# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the gradient fan-in (moe_fuse_input_grad, fuse_attn_residual_grad)."""

from unittest import mock

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.grad_fanin import GradFanIn, attach_grad_fanin, grad_fanin_tap
from megatron.core.transformer.moe.moe_utils import router_gating_linear, router_input_grad_fanin
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine.pytorch as te

    HAVE_TE = True
    TE_HAS_GRAD_FANIN = getattr(te.Linear, "supports_grad_fanin", False) and getattr(
        te.LayerNormLinear, "supports_grad_fanin", False
    )
except ImportError:
    HAVE_TE = False
    TE_HAS_GRAD_FANIN = False

requires_te = pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required.")
requires_te_fanin = pytest.mark.skipif(
    not TE_HAS_GRAD_FANIN,
    reason="Transformer Engine Linear/LayerNormLinear without gradient fan-in support.",
)


def _rel_err(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


class _FinishSpy:
    """Records (accumulated, returned_sum) for every GradFanIn.finish call."""

    def __init__(self):
        self.calls = []
        self._orig = GradFanIn.finish

    def __enter__(self):
        spy = self

        def finish(fanin, grad, accumulated):
            out = spy._orig(fanin, grad, accumulated)
            spy.calls.append((accumulated, out is not None))
            return out

        self._patch = mock.patch.object(GradFanIn, "finish", finish)
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        return self._patch.__exit__(*exc)


class TestGradFanIn:
    """Unit tests for the GradFanIn bookkeeping."""

    def test_any_order_sums_and_last_consumer_returns(self):
        x = torch.zeros(8, 4, device="cuda")
        grads = [torch.randn(8, 4, device="cuda") for _ in range(3)]
        fanin = GradFanIn(x)
        for _ in grads:
            fanin.add_consumer()
        assert fanin.finish(grads[2].clone(), accumulated=False) is None
        assert fanin.finish(grads[0].clone(), accumulated=False) is None
        out = fanin.finish(grads[1].clone(), accumulated=False)
        torch.testing.assert_close(out, (grads[2] + grads[0]) + grads[1], rtol=0, atol=0)
        assert not fanin.matches(x)  # backward started: no new consumers

    def test_shared_gradient_is_not_written_in_place(self):
        """A contribution whose storage is shared is never handed out as an in-place target."""
        x = torch.zeros(8, 4, device="cuda")
        shared = torch.randn(8, 4, device="cuda")
        alias = shared.view(-1)  # keeps the storage shared
        snapshot = shared.clone()
        fanin = GradFanIn(x)
        fanin.add_consumer()
        fanin.add_consumer()
        fanin.finish(shared, accumulated=False)
        assert fanin.accum_target() is None
        assert fanin.partial_sum() is shared
        other = torch.randn(8, 4, device="cuda")
        out = fanin.finish(other, accumulated=False)
        torch.testing.assert_close(shared, snapshot, rtol=0, atol=0)
        torch.testing.assert_close(out, snapshot + other, rtol=0, atol=0)
        del alias

    def test_exclusive_gradient_is_writable(self):
        x = torch.zeros(8, 4, device="cuda")
        fanin = GradFanIn(x)
        fanin.add_consumer()
        fanin.add_consumer()
        fanin.finish(torch.randn(8, 4, device="cuda"), accumulated=False)
        assert fanin.accum_target() is not None

    def test_reshaped_gradient_is_writable(self):
        """The gradient of a reshape is a view; with no other user of its base it is writable."""
        x = torch.zeros(8, 4, device="cuda")
        fanin = GradFanIn(x)
        fanin.add_consumer()
        fanin.add_consumer()
        fanin.finish(torch.randn(32, device="cuda").view(8, 4), accumulated=False)
        assert fanin.accum_target() is not None

    def test_slice_of_larger_buffer_is_not_writable(self):
        x = torch.zeros(8, 4, device="cuda")
        buffer = torch.randn(64, 4, device="cuda")
        fanin = GradFanIn(x)
        fanin.add_consumer()
        fanin.add_consumer()
        fanin.finish(buffer[:8], accumulated=False)
        assert fanin.accum_target() is None

    def test_matches_requires_same_tensor(self):
        x = torch.zeros(4, device="cuda")
        fanin = GradFanIn(x)
        assert fanin.matches(x)
        assert not fanin.matches(x.view_as(x))
        assert not fanin.matches(x.clone())

    def test_attach_skips_modules_without_support(self):
        module = torch.nn.Linear(4, 4).cuda()
        x = torch.randn(2, 4, device="cuda", requires_grad=True)
        fanin = GradFanIn(x)
        with attach_grad_fanin(module, fanin) as attach:
            module(x)
        assert not attach.joined and fanin.num_consumers == 0

    def test_attach_undoes_unconsumed_registration(self):
        """A supporting module whose forward did not take the fan-in is unregistered."""

        class _Module(torch.nn.Linear):
            supports_grad_fanin = True

        module = _Module(4, 4).cuda()
        x = torch.randn(2, 4, device="cuda", requires_grad=True)
        fanin = GradFanIn(x)
        with attach_grad_fanin(module, fanin) as attach:
            module(x)
        assert not attach.joined and fanin.num_consumers == 0
        assert "_grad_fanin" not in module.__dict__

    def test_attach_skips_other_inputs_and_no_grad(self):
        class _Module(torch.nn.Linear):
            supports_grad_fanin = True

        module = _Module(4, 4).cuda()
        x = torch.randn(2, 4, device="cuda", requires_grad=True)
        fanin = GradFanIn(x)
        with attach_grad_fanin(module, fanin) as attach:
            module(x * 1.0)
        assert not attach.joined
        with attach_grad_fanin(module, fanin) as attach, torch.no_grad():
            module(x)
        assert not attach.joined and fanin.num_consumers == 0


@requires_te
class TestRouterGatingFanIn:
    """The router gating input-gradient GEMM accumulates into the fan-in."""

    S, H, E = 256, 512, 64

    def _run(self, x, weight, dy_branch, dy_logits, router_dtype, fused):
        x = x.detach().requires_grad_()
        fanin = GradFanIn(x) if fused else None
        with router_input_grad_fanin(fanin):
            logits = router_gating_linear(x, weight, None, router_dtype)
        branch = grad_fanin_tap(x, fanin) if fused else x
        torch.autograd.backward([branch * 2.0, logits], [dy_branch, dy_logits.to(router_dtype)])
        return x.grad

    @pytest.mark.parametrize("router_dtype", [torch.bfloat16, torch.float32])
    def test_matches_autograd(self, router_dtype):
        torch.manual_seed(0)
        x = torch.randn(self.S, 2, self.H, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(self.E, self.H, device="cuda", dtype=torch.bfloat16)
        weight.requires_grad_()
        dy_branch = torch.randn_like(x)
        dy_logits = torch.randn(self.S, 2, self.E, device="cuda")
        ref = self._run(x, weight, dy_branch, dy_logits, router_dtype, fused=False)
        ref_wgrad = weight.grad.clone()
        weight.grad = None
        with _FinishSpy() as spy:
            got = self._run(x, weight, dy_branch, dy_logits, router_dtype, fused=True)
        torch.testing.assert_close(weight.grad, ref_wgrad, rtol=0, atol=0)
        if router_dtype == torch.bfloat16:
            # Tap first, then the GEMM accumulates in its epilogue (one rounding instead of two).
            assert spy.calls == [(False, False), (True, True)]
            torch.testing.assert_close(got.float(), ref.float(), rtol=1.6e-2, atol=1e-2)
            assert _rel_err(got, ref) < 4e-3
        else:
            # FP32 router output: no epilogue accumulation; same bf16 add as autograd.
            assert spy.calls == [(False, False), (False, True)]
            torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_other_input_does_not_join(self):
        x = torch.randn(16, self.H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(self.E, self.H, device="cuda", dtype=torch.bfloat16)
        fanin = GradFanIn(x)
        with router_input_grad_fanin(fanin):
            router_gating_linear(x * 1.0, weight, None, torch.bfloat16)
        assert fanin.num_consumers == 0


class _LayerHarness:
    """Runs the same layer with a fan-in option off and on and compares the results."""

    @staticmethod
    def run(layers, seq_len=64, batch=2):
        hidden_size = layers[0].config.hidden_size
        torch.manual_seed(0)
        hidden_states = torch.randn(
            seq_len, batch, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        output_grad = torch.randn_like(hidden_states)
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device="cuda")
        results = []
        for layer in layers:
            layer_input = hidden_states.clone().requires_grad_(True)
            output, _ = layer(layer_input, attention_mask=attention_mask)
            output.backward(output_grad)
            grads = {n: p.grad for n, p in layer.named_parameters() if p.grad is not None}
            results.append((output.detach(), layer_input.grad, grads))
        return results


@requires_te
class TestTransformerLayerFanIn:
    """MLA (fused down projection) + MoE with a shared expert, options off vs on."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    @staticmethod
    def _build_layer(**options):
        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rope_type="rope",
            rotary_base=10000,
            original_max_position_embeddings=32,
            normalization="RMSNorm",
            mla_down_proj_fusion=True,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_ffn_hidden_size=256,
            moe_shared_expert_intermediate_size=256,
            moe_router_dtype=None,
            moe_token_dispatcher_type="alltoall",
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            **options,
        )
        submodules = get_gpt_layer_with_transformer_engine_submodules(
            num_experts=config.num_moe_experts,
            multi_latent_attention=True,
            mla_down_proj_fusion=True,
        )
        return TransformerLayer(config, submodules).cuda()

    def _compare(self, fused_options, expect_unchanged_prefixes):
        reference = self._build_layer()
        fused = self._build_layer(**fused_options)
        fused.load_state_dict(reference.state_dict())
        with _FinishSpy() as spy:
            (ref_out, ref_dx, ref_grads), (out, dx, grads) = _LayerHarness.run([reference, fused])
        # Forward is untouched.
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        assert grads.keys() == ref_grads.keys()
        for name, ref_grad in ref_grads.items():
            if name.startswith(expect_unchanged_prefixes):
                torch.testing.assert_close(grads[name], ref_grad, rtol=0, atol=0, msg=name)
            else:
                # Downstream of a fused sum: equal up to the rounding of that one add.
                assert _rel_err(grads[name], ref_grad) < 1e-2, name
        assert _rel_err(dx, ref_dx) < 1e-2
        return spy.calls

    def test_moe_input_fanin(self):
        calls = self._compare({"moe_fuse_input_grad": True}, ("mlp.",))
        # Dispatch tap + router gating (+ shared-expert fc1 with TE fan-in support).
        expected = 3 if TE_HAS_GRAD_FANIN else 2
        assert len(calls) == expected
        assert sum(returned for _, returned in calls) == 1
        assert any(accumulated for accumulated, _ in calls), calls

    @requires_te_fanin
    def test_attn_residual_fanin(self):
        # Every parameter gradient is computed before the fused add: all bitwise equal.
        calls = self._compare({"fuse_attn_residual_grad": True}, ("",))
        # Residual tap first, then the fused-norm down projection adds it (rmsnorm_bwd_add).
        assert calls == [(False, False), (True, True)]

    def test_attn_residual_fanin_without_te_support_is_noop(self):
        if TE_HAS_GRAD_FANIN:
            pytest.skip("Transformer Engine supports gradient fan-in.")
        reference = self._build_layer()
        fused = self._build_layer(fuse_attn_residual_grad=True)
        fused.load_state_dict(reference.state_dict())
        (ref_out, ref_dx, ref_grads), (out, dx, grads) = _LayerHarness.run([reference, fused])
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        torch.testing.assert_close(dx, ref_dx, rtol=0, atol=0)
        for name, ref_grad in ref_grads.items():
            torch.testing.assert_close(grads[name], ref_grad, rtol=0, atol=0, msg=name)

    def test_no_dangling_hooks_or_state(self):
        layer = self._build_layer(fuse_attn_residual_grad=True, moe_fuse_input_grad=True)
        _LayerHarness.run([layer])
        down_proj = layer.self_attention.linear_qkv_down_proj
        assert len(down_proj._forward_pre_hooks) == 0
        assert "_grad_fanin" not in down_proj.__dict__
        assert "_attn_residual_grad_fanin" not in layer.__dict__
        fc1 = layer.mlp.shared_experts.linear_fc1
        assert len(fc1._forward_pre_hooks) == 0
        assert "_grad_fanin" not in fc1.__dict__

    def test_eval_and_no_grad_skip_fanin(self):
        layer = self._build_layer(fuse_attn_residual_grad=True, moe_fuse_input_grad=True)
        x = torch.randn(16, 2, 256, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones((1, 1, 16, 16), dtype=bool, device="cuda")
        with torch.no_grad():
            layer(x, attention_mask=mask)
        layer.eval()
        layer(x.clone().requires_grad_(), attention_mask=mask)
        assert getattr(layer.mlp, "_input_grad_fanin_cur", None) is None
        assert getattr(layer, "_attn_residual_grad_fanin", None) is None


@requires_te
class TestDenseAttentionResidualFanIn:
    """GPT layer whose linear_qkv has the input norm fused (TELayerNormColumnParallelLinear)."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    @staticmethod
    def _build_layer(**options):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=512,
            normalization="RMSNorm",
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            **options,
        )
        return TransformerLayer(config, get_gpt_layer_with_transformer_engine_submodules()).cuda()

    @requires_te_fanin
    def test_matches_unfused(self):
        reference = self._build_layer()
        fused = self._build_layer(fuse_attn_residual_grad=True)
        fused.load_state_dict(reference.state_dict())
        with _FinishSpy() as spy:
            (ref_out, ref_dx, ref_grads), (out, dx, grads) = _LayerHarness.run([reference, fused])
        assert spy.calls == [(False, False), (True, True)]
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        # The fan-in only changes the layer input gradient; every parameter gradient is
        # computed before the fused add.
        for name, ref_grad in ref_grads.items():
            torch.testing.assert_close(grads[name], ref_grad, rtol=0, atol=0, msg=name)
        torch.testing.assert_close(dx.float(), ref_dx.float(), rtol=1.6e-2, atol=1e-2)
        assert _rel_err(dx, ref_dx) < 4e-3
