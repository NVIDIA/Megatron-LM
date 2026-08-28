# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for Attention Residuals (AttnRes, arXiv:2603.15031).

The aggregation-math and schedule-helper tests are device-agnostic and run on
CPU; they validate the custom autograd Function against a plain-autograd
reference implementation.
"""

import pytest
import torch

from megatron.core.transformer.attention_residual import (
    AttentionResidual,
    attn_res_final_num_sources,
    attn_res_num_payload_slices,
    attn_res_num_sources,
    is_attn_res_block_start,
    pack_attn_res_payload,
    unpack_attn_res_payload,
)


def _reference_attn_res(pseudo_query, key_norm_weight, eps, values):
    """Plain-autograd reference: RMSNorm keys, fp64 softmax over depth."""
    stacked = torch.stack([v.double() for v in values])  # [n, ..., h]
    keys = stacked * torch.rsqrt(stacked.pow(2).mean(dim=-1, keepdim=True) + eps)
    q = pseudo_query.double() * key_norm_weight.double()
    logits = (keys * q).sum(dim=-1)  # [n, ...]
    alpha = torch.softmax(logits, dim=0)
    return (alpha.unsqueeze(-1) * stacked).sum(dim=0)


class TestAttnResAggregationMath:

    def _make_inputs(self, n_sources, shape=(6, 2, 32), dtype=torch.float32, seed=1234):
        torch.manual_seed(seed)
        values = [
            (torch.randn(*shape, dtype=dtype) * (0.5 + i)).requires_grad_(True)
            for i in range(n_sources)
        ]
        pseudo_query = torch.randn(shape[-1], dtype=torch.float32).mul(0.05).requires_grad_(True)
        key_norm_weight = (
            torch.ones(shape[-1], dtype=torch.float32) + torch.randn(shape[-1]) * 0.1
        ).requires_grad_(True)
        return pseudo_query, key_norm_weight, values

    @pytest.mark.parametrize("n_sources", [1, 2, 5, 10])
    def test_forward_matches_reference(self, n_sources):
        from megatron.core.transformer.attention_residual import _AttnResAggregation

        eps = 1e-6
        pseudo_query, key_norm_weight, values = self._make_inputs(n_sources)
        out = _AttnResAggregation.apply(pseudo_query, key_norm_weight, eps, False, *values)
        ref = _reference_attn_res(pseudo_query, key_norm_weight, eps, values)
        torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("n_sources", [1, 3, 9])
    def test_backward_matches_reference(self, n_sources):
        from megatron.core.transformer.attention_residual import _AttnResAggregation

        eps = 1e-6
        pseudo_query, key_norm_weight, values = self._make_inputs(n_sources)
        grad_out = torch.randn(6, 2, 32)

        out = _AttnResAggregation.apply(pseudo_query, key_norm_weight, eps, False, *values)
        grads = torch.autograd.grad(out, [pseudo_query, key_norm_weight, *values], grad_out)

        ref = _reference_attn_res(pseudo_query, key_norm_weight, eps, values)
        ref_grads = torch.autograd.grad(
            ref, [pseudo_query, key_norm_weight, *values], grad_out.double()
        )

        for got, want in zip(grads, ref_grads):
            torch.testing.assert_close(got.double(), want.double(), rtol=1e-4, atol=1e-5)

    def test_zero_init_is_uniform_mean(self):
        """At zero pseudo-query the aggregation is the exact mean of the sources.

        Together with RMSNorm scale invariance this makes the network
        functionally identical to the PreNorm baseline at initialization.
        """
        torch.manual_seed(7)
        config_like_eps = 1e-6
        values = [torch.randn(4, 3, 16) * (1.0 + 3 * i) for i in range(5)]
        from megatron.core.transformer.attention_residual import _AttnResAggregation

        pseudo_query = torch.zeros(16, requires_grad=True)
        key_norm_weight = torch.ones(16, requires_grad=True)
        out = _AttnResAggregation.apply(
            pseudo_query, key_norm_weight, config_like_eps, False, *values
        )
        torch.testing.assert_close(out, torch.stack(values).mean(dim=0), rtol=1e-6, atol=1e-6)

    def test_zero_query_gradient_is_nonzero(self):
        """The pseudo-query must receive gradient at init so it can start learning."""
        from megatron.core.transformer.attention_residual import _AttnResAggregation

        torch.manual_seed(11)
        values = [torch.randn(4, 2, 16) for _ in range(3)]
        pseudo_query = torch.zeros(16, requires_grad=True)
        key_norm_weight = torch.ones(16, requires_grad=True)
        out = _AttnResAggregation.apply(pseudo_query, key_norm_weight, 1e-6, False, *values)
        out.sum().backward()
        assert pseudo_query.grad is not None and pseudo_query.grad.abs().sum() > 0

    def test_bf16_values_fp32_softmax(self):
        """bf16 sources stay bf16 on the output; softmax runs in fp32 internally."""
        from megatron.core.transformer.attention_residual import _AttnResAggregation

        torch.manual_seed(3)
        values = [torch.randn(4, 2, 16, dtype=torch.bfloat16, requires_grad=True) for _ in range(4)]
        pseudo_query = torch.randn(16).mul(0.01).requires_grad_(True)
        key_norm_weight = torch.ones(16, requires_grad=True)
        out = _AttnResAggregation.apply(pseudo_query, key_norm_weight, 1e-6, False, *values)
        assert out.dtype == torch.bfloat16
        out.float().sum().backward()
        for v in values:
            assert v.grad is not None and v.grad.dtype == torch.bfloat16

    def test_flat_token_layout(self):
        """Packed/THD layouts pass [t, 1, h] tensors; math must be layout-agnostic."""
        from megatron.core.transformer.attention_residual import _AttnResAggregation

        torch.manual_seed(5)
        values_3d = [torch.randn(8, 1, 16) for _ in range(3)]
        pseudo_query = torch.randn(16) * 0.1
        key_norm_weight = torch.ones(16)
        out = _AttnResAggregation.apply(pseudo_query, key_norm_weight, 1e-6, False, *values_3d)
        ref = _reference_attn_res(pseudo_query, key_norm_weight, 1e-6, values_3d)
        torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-5)


class TestAttnResCompileParity:
    """attn_res_impl='compile' must match the eager math bit-for-bit-ish."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="torch.compile parity checked on CUDA"
    )
    @pytest.mark.parametrize("n_sources", [1, 3, 6])
    def test_compile_matches_eager(self, n_sources):
        from megatron.core.transformer.attention_residual import _AttnResAggregation

        torch.manual_seed(21)
        eps = 1e-6

        def make_inputs():
            values = [
                (torch.randn(4, 2, 32, device='cuda') * (0.5 + i)).requires_grad_(True)
                for i in range(n_sources)
            ]
            pseudo_query = torch.randn(32, device='cuda').mul(0.05).requires_grad_(True)
            key_norm_weight = (
                torch.ones(32, device='cuda') + torch.randn(32, device='cuda') * 0.1
            ).requires_grad_(True)
            return pseudo_query, key_norm_weight, values

        torch.manual_seed(21)
        q_e, g_e, v_e = make_inputs()
        torch.manual_seed(21)
        q_c, g_c, v_c = make_inputs()
        grad_out = torch.randn(4, 2, 32, device='cuda')

        out_eager = _AttnResAggregation.apply(q_e, g_e, eps, False, *v_e)
        grads_eager = torch.autograd.grad(out_eager, [q_e, g_e, *v_e], grad_out)

        out_compiled = _AttnResAggregation.apply(q_c, g_c, eps, True, *v_c)
        torch.testing.assert_close(out_compiled, out_eager, rtol=1e-6, atol=1e-6)
        grads_compiled = torch.autograd.grad(out_compiled, [q_c, g_c, *v_c], grad_out)
        for got, want in zip(grads_compiled, grads_eager):
            # Parameter grads are chained fp32 token reductions; inductor's
            # reduction split (autotune-dependent) differs from the eager gemv
            # by a few ulps. A real math bug is orders of magnitude larger.
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


class TestAttnResSchedule:

    def test_block_start_and_source_counts_16_layers_k2(self):
        """16 layers, 2 layers/block: 8 blocks; final head sees 9 sources."""
        k = 2
        starts = [l for l in range(1, 17) if is_attn_res_block_start(l, k)]
        assert starts == [1, 3, 5, 7, 9, 11, 13, 15]
        assert attn_res_num_sources(1, k) == 1  # embedding only
        assert attn_res_num_sources(2, k) == 1
        assert attn_res_num_sources(3, k) == 2
        assert attn_res_num_sources(16, k) == 8
        assert attn_res_final_num_sources(16, k) == 9

    def test_kimi_linear_48b_layout(self):
        """27 layers, 3 layers/block: 9 blocks + embedding = 10 depth sources."""
        assert attn_res_final_num_sources(27, 3) == 10

    def test_full_attn_res_k1(self):
        """k=1 degenerates to one source per layer."""
        assert attn_res_num_sources(4, 1) == 4
        assert attn_res_final_num_sources(4, 1) == 5

    def test_payload_slices(self):
        # 16 layers, k=2, PP4 (4 layers/stage): boundaries after layers 4, 8, 12.
        assert attn_res_num_payload_slices(4, 2) == 3  # [emb, b1] + partial
        assert attn_res_num_payload_slices(8, 2) == 5  # [emb, b1, b2, b3] + partial
        assert attn_res_num_payload_slices(12, 2) == 7
        # Full AttnRes-like k=1: after layer 4, payload = 4 sources + partial.
        assert attn_res_num_payload_slices(4, 1) == 5

    def test_pack_unpack_roundtrip(self):
        torch.manual_seed(9)
        sources = [torch.randn(4, 2, 8) for _ in range(3)]
        partial = torch.randn(4, 2, 8)
        payload = pack_attn_res_payload([*sources, partial])
        assert payload.shape == (16, 2, 8)
        assert payload._base is None  # viewless: safe for deallocate_output_tensor
        got_sources, got_partial = unpack_attn_res_payload(payload, 4)
        assert len(got_sources) == 3
        for got, want in zip(got_sources, sources):
            torch.testing.assert_close(got, want)
        torch.testing.assert_close(got_partial, partial)

    def test_unpack_rejects_wrong_slice_count(self):
        payload = torch.randn(15, 2, 8)
        with pytest.raises(AssertionError):
            unpack_attn_res_payload(payload, 4)

    def test_payload_grad_flows_to_leaf(self):
        """Slice views route gradients back into the single recv leaf tensor."""
        payload = torch.randn(12, 2, 8, requires_grad=True)
        sources, partial = unpack_attn_res_payload(payload, 3)
        loss = (sources[0] * 1.0).sum() + (sources[1] * 2.0).sum() + (partial * 3.0).sum()
        loss.backward()
        assert payload.grad is not None
        torch.testing.assert_close(payload.grad[:4], torch.ones(4, 2, 8))
        torch.testing.assert_close(payload.grad[8:], torch.full((4, 2, 8), 3.0))


class TestAttentionResidualModule:

    def _make_config(self):
        from megatron.core.transformer.transformer_config import TransformerConfig

        return TransformerConfig(
            num_layers=4,
            hidden_size=16,
            num_attention_heads=4,
            enable_attention_residuals=True,
            attn_res_block_layers=2,
            layernorm_epsilon=1e-6,
        )

    def test_module_zero_init_and_fp32_params(self):
        config = self._make_config()
        module = AttentionResidual(config)
        assert torch.all(module.pseudo_query == 0)
        assert torch.all(module.key_norm_weight == 1)
        assert getattr(module.pseudo_query, 'keep_in_fp32', False)
        assert getattr(module.key_norm_weight, 'keep_in_fp32', False)
        # 1-D parameters: picked up by the generic no-weight-decay rule.
        assert module.pseudo_query.ndim == 1 and module.key_norm_weight.ndim == 1

    def test_module_forward_is_mean_at_init(self):
        config = self._make_config()
        module = AttentionResidual(config)
        values = [torch.randn(4, 2, 16) for _ in range(3)]
        out = module(values)
        torch.testing.assert_close(out, torch.stack(values).mean(dim=0), rtol=1e-6, atol=1e-6)


class TestAttnResInitEquivalence:
    """Zero-init AttnRes must be functionally identical to the PreNorm baseline.

    With zero pseudo-queries the aggregation is the exact MEAN of the depth
    sources, which partition the baseline residual SUM; every consumer of the
    aggregated state is a norm, and norms are scale-invariant up to their eps
    term. With a tiny eps the first forward therefore matches the baseline
    within float rounding — a one-shot oracle for the spec wiring, the
    block-boundary bookkeeping, and the final aggregation. (With production
    eps values the match is only approximate: eps breaks scale invariance
    when activations are small.)
    """

    def setup_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.destroy_model_parallel()

    def _make_model(self, enable_attn_res, block_layers=1, seed=123):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.transformer_config import TransformerConfig

        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        kwargs = dict(
            num_layers=4,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            add_bias_linear=False,
            # LayerNorm (not RMSNorm): apex's FusedLayerNorm — selected by the
            # local spec when apex is present — rejects RMSNorm, and LayerNorm
            # is equally scale-invariant so the equivalence oracle is unchanged.
            normalization="LayerNorm",
            layernorm_epsilon=1e-12,
            init_method_std=0.2,
            pipeline_dtype=torch.float32,
        )
        if enable_attn_res:
            kwargs.update(enable_attention_residuals=True, attn_res_block_layers=block_layers)
        config = TransformerConfig(**kwargs)
        spec = get_gpt_decoder_block_spec(config, use_transformer_engine=False)
        model = GPTModel(
            config=config,
            transformer_layer_spec=spec,
            vocab_size=97,
            max_sequence_length=32,
            position_embedding_type='rope',
            pre_process=True,
            post_process=True,
        )
        return model.cuda().eval()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    @pytest.mark.parametrize("block_layers", [1, 2])
    def test_first_forward_matches_baseline(self, block_layers):
        baseline = self._make_model(False)
        attnres = self._make_model(True, block_layers=block_layers)

        missing, unexpected = attnres.load_state_dict(baseline.state_dict(), strict=False)
        assert not unexpected, unexpected
        assert missing and all('attn_res' in k for k in missing), missing

        torch.manual_seed(7)
        input_ids = torch.randint(0, 97, (2, 16), device='cuda')
        position_ids = torch.arange(16, device='cuda').unsqueeze(0).expand(2, -1)

        with torch.no_grad():
            out_base = baseline(input_ids, position_ids, None)
            out_attn = attnres(input_ids, position_ids, None)

        torch.testing.assert_close(out_attn, out_base, rtol=1e-4, atol=1e-4)


class TestAttnResConfigValidation:

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            num_layers=4,
            hidden_size=16,
            num_attention_heads=4,
            enable_attention_residuals=True,
            attn_res_block_layers=2,
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid_config(self):
        from megatron.core.transformer.transformer_config import TransformerConfig

        TransformerConfig(**self._base_kwargs())

    def test_block_layers_required(self):
        from megatron.core.transformer.transformer_config import TransformerConfig

        with pytest.raises(ValueError, match="attn_res_block_layers"):
            TransformerConfig(**self._base_kwargs(attn_res_block_layers=None))

    def test_block_layers_without_enable_rejected(self):
        from megatron.core.transformer.transformer_config import TransformerConfig

        with pytest.raises(ValueError, match="enable_attention_residuals"):
            TransformerConfig(
                **self._base_kwargs(enable_attention_residuals=False, attn_res_block_layers=2)
            )

    def test_mutually_exclusive_with_mhc(self):
        from megatron.core.transformer.transformer_config import TransformerConfig

        with pytest.raises(ValueError, match="mutually"):
            TransformerConfig(**self._base_kwargs(enable_hyper_connections=True))

    @pytest.mark.parametrize(
        "overrides,match",
        [
            (
                dict(
                    recompute_granularity='full', recompute_method='uniform', recompute_num_layers=1
                ),
                "full",
            ),
            (dict(fused_residual_rmsnorm=True), "fused_residual_rmsnorm"),
            (dict(fp32_residual_connection=True), "fp32_residual_connection"),
            (dict(cpu_offloading=True, cpu_offloading_num_layers=1), "cpu_offloading"),
            (dict(heterogeneous_block_specs=True), "heterogeneous"),
        ],
    )
    def test_unsupported_combinations_rejected(self, overrides, match):
        from megatron.core.transformer.transformer_config import TransformerConfig

        with pytest.raises(ValueError, match=match):
            TransformerConfig(**self._base_kwargs(**overrides))


class TestAttnResHybridSchedule:
    """Hybrid payload-width math and hybrid-specific config validation."""

    def _hybrid_config(self, **overrides):
        from megatron.core.transformer.transformer_config import TransformerConfig

        kwargs = dict(
            num_layers=16,
            hidden_size=16,
            num_attention_heads=4,
            enable_attention_residuals=True,
            attn_res_block_layers=2,
            is_hybrid_model=True,
        )
        kwargs.update(overrides)
        return TransformerConfig(**kwargs)

    def test_pipe_segment_payload_slices(self):
        from megatron.core.transformer.attention_residual import attn_res_payload_slices_for_pp_rank

        config = self._hybrid_config(
            hybrid_layer_pattern="MM*-|M*--|**--|M-M-",
            pipeline_model_parallel_size=4,
            pipeline_dtype=torch.float32,
        )
        # Offsets: 0, 4, 8, 12 entries; k=2 -> slices = (L-1)//2 + 2.
        assert attn_res_payload_slices_for_pp_rank(config, 1) == 3
        assert attn_res_payload_slices_for_pp_rank(config, 2) == 5
        assert attn_res_payload_slices_for_pp_rank(config, 3) == 7

    def test_uneven_pipe_segments(self):
        from megatron.core.transformer.attention_residual import attn_res_payload_slices_for_pp_rank

        config = self._hybrid_config(
            num_layers=12,
            hybrid_layer_pattern="MM*|M*-M-|*-M-",
            pipeline_model_parallel_size=3,
            pipeline_dtype=torch.float32,
            attn_res_block_layers=1,
        )
        # Offsets: 0, 3, 8 entries; k=1 -> slices = L + 1.
        assert attn_res_payload_slices_for_pp_rank(config, 1) == 4
        assert attn_res_payload_slices_for_pp_rank(config, 2) == 9

    def test_pipe_free_even_split(self):
        from megatron.core.transformer.attention_residual import attn_res_payload_slices_for_pp_rank

        config = self._hybrid_config(
            hybrid_layer_pattern="MM*-M*--**--M-M-", pipeline_model_parallel_size=1
        )
        # Even split at pp=1 degenerates; exercise the helper via a fake rank
        # computation on a copy configured for pp=2.
        config2 = self._hybrid_config(
            hybrid_layer_pattern="MM*-M*--|**--M-M-",
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
        )
        assert attn_res_payload_slices_for_pp_rank(config2, 1) == (8 - 1) // 2 + 2
        del config

    def test_segment_count_mismatch_rejected(self):
        with pytest.raises(ValueError, match="pipe segments"):
            self._hybrid_config(
                hybrid_layer_pattern="MM*-|M*--|**--|M-M-",
                pipeline_model_parallel_size=2,
                pipeline_dtype=torch.float32,
            )

    def test_hybrid_mtp_rejected(self):
        with pytest.raises(ValueError, match="hybrid MTP"):
            self._hybrid_config(mtp_num_layers=1)

    def test_missing_pattern_with_pp_rejected(self):
        with pytest.raises(ValueError, match="hybrid_layer_pattern"):
            self._hybrid_config(pipeline_model_parallel_size=2, pipeline_dtype=torch.float32)


class TestAttnResVppSchedule:
    """Interleaved-VPP delta-payload widths, padded pack/unpack, and the grad tap."""

    def _vpp_config(self, **overrides):
        from megatron.core.transformer.transformer_config import TransformerConfig

        kwargs = dict(
            num_layers=16,
            hidden_size=16,
            num_attention_heads=4,
            enable_attention_residuals=True,
            attn_res_block_layers=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
        )
        kwargs.update(overrides)
        return TransformerConfig(**kwargs)

    def test_delta_slices_k_divides_chunk(self):
        """L=16, P=2, V=2, M=4, k=2: every boundary carries exactly W* = 3 slices."""
        from megatron.core.transformer.attention_residual import (
            attn_res_boundary_delta_slices,
            attn_res_uniform_payload_slices,
        )

        config = self._vpp_config()
        widths = [attn_res_boundary_delta_slices(config, s % 2, s // 2) for s in range(1, 4)]
        assert widths == [3, 3, 3]
        assert attn_res_uniform_payload_slices(config) == 3

    def test_delta_slices_k_not_dividing_chunk(self):
        """L=16, P=2, V=2, M=4, k=3: the FIRST-round boundary dominates W*."""
        from megatron.core.transformer.attention_residual import (
            attn_res_boundary_delta_slices,
            attn_res_uniform_payload_slices,
        )

        config = self._vpp_config(attn_res_block_layers=3)
        widths = [attn_res_boundary_delta_slices(config, s % 2, s // 2) for s in range(1, 4)]
        assert widths == [3, 2, 2]
        assert attn_res_uniform_payload_slices(config) == 3

    def test_delta_slices_pp4_vpp2(self):
        """L=16, P=4, V=2, M=2, k=2: steady width equals the closed form (P-1)M/k+1."""
        from megatron.core.transformer.attention_residual import (
            attn_res_boundary_delta_slices,
            attn_res_uniform_payload_slices,
        )

        config = self._vpp_config(pipeline_model_parallel_size=4)
        widths = [attn_res_boundary_delta_slices(config, s % 4, s // 4) for s in range(1, 8)]
        assert widths == [2, 3, 4, 4, 4, 4, 4]
        assert attn_res_uniform_payload_slices(config) == 4

    def test_v1_degenerates_to_full_prefix(self):
        """Without VPP the delta equals today's per-rank full-prefix width."""
        from megatron.core.transformer.attention_residual import (
            attn_res_boundary_delta_slices,
            attn_res_payload_slices_for_pp_rank,
        )

        config = self._vpp_config(
            pipeline_model_parallel_size=4, virtual_pipeline_model_parallel_size=None
        )
        for pp_rank in range(1, 4):
            assert attn_res_boundary_delta_slices(
                config, pp_rank, 0
            ) == attn_res_payload_slices_for_pp_rank(config, pp_rank)

    def test_hybrid_uneven_segment_delta_slices(self):
        """Hybrid pattern '|' segments map chunk-major; uneven lengths are allowed."""
        from megatron.core.transformer.attention_residual import (
            attn_res_boundary_delta_slices,
            attn_res_uniform_payload_slices,
        )
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=13,
            hidden_size=16,
            num_attention_heads=4,
            enable_attention_residuals=True,
            attn_res_block_layers=2,
            is_hybrid_model=True,
            hybrid_layer_pattern="MM*-|M*-|**|M-M-",
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
        )
        # Segments (chunk-major): 4, 3, 2, 4 entries; offsets 0, 4, 7, 9.
        widths = [attn_res_boundary_delta_slices(config, s % 2, s // 2) for s in range(1, 4)]
        assert widths == [3, 3, 2]
        assert attn_res_uniform_payload_slices(config) == 3

    def test_padded_pack_unpack_roundtrip(self):
        torch.manual_seed(11)
        sources = [torch.randn(4, 2, 8) for _ in range(2)]
        partial = torch.randn(4, 2, 8)
        payload = pack_attn_res_payload([*sources, partial], pad_to_slices=5)
        assert payload.shape == (20, 2, 8)
        assert payload._base is None
        torch.testing.assert_close(payload[12:], torch.zeros(8, 2, 8))
        got_sources, got_partial = unpack_attn_res_payload(payload, 3, padded_slices=5)
        assert len(got_sources) == 2
        for got, want in zip(got_sources, sources):
            torch.testing.assert_close(got, want)
        torch.testing.assert_close(got_partial, partial)

    def test_padded_unpack_divisible_trap(self):
        """padded_slices divisible by the real count must not silently mis-chunk."""
        torch.manual_seed(12)
        source = torch.randn(4, 2, 8)
        partial = torch.randn(4, 2, 8)
        payload = pack_attn_res_payload([source, partial], pad_to_slices=4)
        got_sources, got_partial = unpack_attn_res_payload(payload, 2, padded_slices=4)
        torch.testing.assert_close(got_sources[0], source)
        torch.testing.assert_close(got_partial, partial)

    def test_grad_tap_two_disjoint_graphs(self):
        """The cache leaf routes later-chunk grads into the producing chunk's graph.

        Emulates two virtual chunks whose graphs must stay disjoint
        (schedule backward runs with keep_graph=False): chunk A materializes a
        source, chunk B consumes the cache leaf; B's backward runs first.
        """
        from megatron.core.transformer.attention_residual import attn_res_tap_source

        x = torch.randn(4, 3, requires_grad=True)
        y = x * 3.0  # chunk A's locally formed source
        in_graph, leaf = attn_res_tap_source(y)
        out_a = (in_graph * 2.0).sum()  # chunk A's own consumption
        out_b = (leaf * 5.0).sum()  # chunk B consumes the cache leaf

        out_b.backward()  # later chunk's backward runs FIRST (1F1B ordering)
        assert leaf.grad is not None
        out_a.backward()  # tap backward drains leaf.grad into chunk A's graph
        torch.testing.assert_close(x.grad, torch.full((4, 3), 3.0 * (2.0 + 5.0)))
        assert leaf.grad is None  # drained exactly once

    def test_source_cache_hygiene(self):
        from megatron.core.transformer.attention_residual import (
            AttnResStageSources,
            attn_res_source_cache_reset,
            get_attn_res_source_cache,
        )

        attn_res_source_cache_reset()
        config = self._vpp_config()
        embedding = torch.randn(4, 2, 16)
        state, hidden = AttnResStageSources.enter(
            config,
            embedding,
            layers_before=0,
            pp_rank=0,
            vp_stage=0,
            microbatch_id=0,
            pre_process=True,
        )
        state.append_block_start(hidden)
        hidden = hidden * 2.0
        state.append_block_start(hidden)
        payload = state.exit_pack(hidden * 0.5)
        # W* = 3 for this config; rank 0 chunk 0 sends the full 2-source prefix.
        assert payload.shape[0] == 3 * 4
        assert 0 in get_attn_res_source_cache().sources
        # Re-entering the same microbatch at vp_stage 0 must trip the staleness assert.
        with pytest.raises(AssertionError, match="stale"):
            AttnResStageSources.enter(
                config,
                embedding,
                layers_before=0,
                pp_rank=0,
                vp_stage=0,
                microbatch_id=0,
                pre_process=True,
            )
        attn_res_source_cache_reset()
        assert not get_attn_res_source_cache().sources


class TestAttnResVppPipelineSimulation:
    """Single-process emulation of the full interleaved pipeline for one microbatch.

    Runs every (rank, chunk) stage through AttnResStageSources with per-rank
    cache swapping and detached requires-grad payload hand-offs (mimicking the
    p2p recv leaves), then runs the chunk backwards in reverse stage order
    (mimicking interleaved 1F1B) with keep_graph=False semantics. Values and
    input gradients must match a monolithic single-graph reference exactly.
    """

    def _run(self, pp_size, vp_size, num_layers, block_layers, seed=21):
        from megatron.core.transformer.attention_residual import (
            AttnResStageSources,
            attn_res_source_cache_reset,
            get_attn_res_source_cache,
            is_attn_res_block_start,
        )
        from megatron.core.transformer.transformer_config import TransformerConfig

        torch.manual_seed(seed)
        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=8,
            num_attention_heads=1,
            enable_attention_residuals=True,
            attn_res_block_layers=block_layers,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vp_size,
            pipeline_dtype=torch.float32,
        )
        num_stages = pp_size * (vp_size or 1)
        layers_per_stage = num_layers // num_stages
        embedding = torch.randn(4, 2, 8)
        # Per-layer coefficients; consuming ALL current sources at every layer
        # makes any wrong/missing/mixed-up source change the loss.
        a = [0.9 + 0.01 * l for l in range(num_layers)]
        b = [0.05 + 0.002 * l for l in range(num_layers)]

        def run_layer(l, partial, sources):
            return partial * a[l - 1] + sum(sources) * b[l - 1]

        # ---- Monolithic reference (single autograd graph). ----
        x_ref = embedding.clone().requires_grad_(True)
        sources_ref = []
        partial = x_ref
        for l in range(1, num_layers + 1):
            if is_attn_res_block_start(l, block_layers):
                sources_ref.append(partial)
            partial = run_layer(l, partial, sources_ref)
        loss_ref = sum((v * (i + 1)).sum() for i, v in enumerate([*sources_ref, partial]))
        loss_ref.backward()

        # ---- Interleaved pipeline emulation. ----
        attn_res_source_cache_reset()
        rank_caches = [dict() for _ in range(pp_size)]
        x_pipe = embedding.clone().requires_grad_(True)
        stage_inputs = []  # per stage: the recv leaf (None for stage 0)
        stage_outputs = []  # per stage: the sent payload (None for the last)
        carried = x_pipe
        loss_pipe = None
        for s in range(num_stages):
            pp_rank, vp_stage = s % pp_size, s // pp_size
            get_attn_res_source_cache().sources = rank_caches[pp_rank]
            if s == 0:
                recv = None
                stage_in = carried
            else:
                recv = carried.detach().requires_grad_(True)  # p2p recv leaf
                stage_in = recv
            stage_inputs.append(recv)
            state, partial = AttnResStageSources.enter(
                config,
                stage_in,
                layers_before=s * layers_per_stage,
                pp_rank=pp_rank,
                vp_stage=vp_stage if vp_size is not None else None,
                microbatch_id=0,
                pre_process=(s == 0),
            )
            for l in range(s * layers_per_stage + 1, (s + 1) * layers_per_stage + 1):
                if is_attn_res_block_start(l, block_layers):
                    state.append_block_start(partial)
                partial = run_layer(l, partial, state.graph_sources)
            if s == num_stages - 1:
                values = state.exit_aggregate_values(partial)
                loss_pipe = sum((v * (i + 1)).sum() for i, v in enumerate(values))
                stage_outputs.append(None)
            else:
                payload = state.exit_pack(partial)
                stage_outputs.append(payload)
                carried = payload
        # All per-rank caches must have been evicted by each rank's last chunk.
        for cache in rank_caches:
            assert not cache

        torch.testing.assert_close(loss_pipe, loss_ref)

        # Backwards in reverse stage order == interleaved 1F1B per-microbatch order.
        loss_pipe.backward()
        for s in range(num_stages - 2, -1, -1):
            grad = stage_inputs[s + 1].grad
            assert grad is not None, f"no grad reached the recv leaf of stage {s + 1}"
            torch.autograd.backward(stage_outputs[s], grad_tensors=grad)

        torch.testing.assert_close(x_pipe.grad, x_ref.grad)
        attn_res_source_cache_reset()

    def test_pp2_vpp2(self):
        self._run(pp_size=2, vp_size=2, num_layers=16, block_layers=2)

    def test_pp2_vpp2_k_not_dividing(self):
        self._run(pp_size=2, vp_size=2, num_layers=16, block_layers=3)

    def test_pp4_vpp2(self):
        self._run(pp_size=4, vp_size=2, num_layers=16, block_layers=2)

    def test_pp2_vpp4(self):
        self._run(pp_size=2, vp_size=4, num_layers=16, block_layers=2)

    def test_pp4_no_vpp_degeneration(self):
        self._run(pp_size=4, vp_size=None, num_layers=16, block_layers=2)


class TestAttnResVppConfigValidation:

    def _kwargs(self, **overrides):
        kwargs = dict(
            num_layers=16,
            hidden_size=16,
            num_attention_heads=4,
            enable_attention_residuals=True,
            attn_res_block_layers=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
        )
        kwargs.update(overrides)
        return kwargs

    def test_vpp_accepted(self):
        from megatron.core.transformer.transformer_config import TransformerConfig

        TransformerConfig(**self._kwargs())

    def test_variable_seq_lengths_rejected(self):
        from megatron.core.transformer.transformer_config import TransformerConfig

        with pytest.raises(ValueError, match="variable_seq_lengths"):
            TransformerConfig(**self._kwargs(variable_seq_lengths=True))

    def test_vpp_with_embedding_split_rejected(self):
        from megatron.core.transformer.transformer_config import TransformerConfig

        with pytest.raises(ValueError, match="account_for_embedding"):
            TransformerConfig(**self._kwargs(account_for_embedding_in_pipeline_split=True))

    def test_hybrid_segments_equal_pp_times_vpp_accepted(self):
        from megatron.core.transformer.transformer_config import TransformerConfig

        TransformerConfig(
            **self._kwargs(is_hybrid_model=True, hybrid_layer_pattern="M-M*|M-M*|M-M*|M-M*")
        )

    def test_hybrid_segment_count_mismatch_rejected(self):
        from megatron.core.transformer.transformer_config import TransformerConfig

        with pytest.raises(ValueError, match="pipe segments"):
            TransformerConfig(
                **self._kwargs(is_hybrid_model=True, hybrid_layer_pattern="M-M*|M-M*")
            )
