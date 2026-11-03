# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Selection, histogram, update, and serialization contracts for logit-space QB."""

import argparse
from unittest.mock import patch

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter
from megatron.core.transformer.moe.router_diagnostics import RouterDiagnosticChannel
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _config(**overrides):
    options = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_router_score_function="sigmoid",
        moe_router_load_balancing_type="quantile_balancing",
        moe_router_quantile_balancing_estimation_scope="global_batch",
        moe_router_qb_bias_space="logit",
        moe_aux_loss_coeff=0.0,
    )
    options.update(overrides)
    return TransformerConfig(**options)


@pytest.mark.parametrize("topk", [1, 2, 7])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("collect", [False, True])
def test_logit_selection_weights_gradients_and_histogram(topk, dense, collect):
    """Compare selection and gradients against a separate sigmoid-gather reference."""
    torch.manual_seed(243)
    logits = (torch.randn(23, 8) + torch.linspace(-24, 3, 8)).requires_grad_()
    reference_logits = logits.detach().clone().requires_grad_()
    bias = torch.linspace(24, -3, 8)
    hist = torch.zeros(8, 257, dtype=torch.int32)
    bounds = torch.tensor([-64.0, 64.0])
    span = torch.zeros(())
    kwargs = dict(qb_histogram=hist, qb_bin_bounds=bounds, qb_logit_span=span) if collect else {}
    probs, routes = moe_utils.topk_routing_with_score_function(
        logits,
        topk,
        score_function="sigmoid",
        expert_bias=bias,
        scaling_factor=3.16,
        qb_bias_space="logit",
        dense_output=dense,
        **kwargs,
    )
    ids = (reference_logits.detach() + bias).topk(topk, dim=1).indices
    selected = reference_logits.sigmoid().gather(1, ids)
    if topk > 1:
        selected = selected / (selected.sum(1, keepdim=True) + 1e-20)
    reference = torch.zeros_like(reference_logits).scatter(1, ids, 3.16 * selected)
    expected_mask = torch.zeros_like(logits, dtype=torch.bool).scatter(1, ids, True)
    if dense:
        mask = torch.zeros_like(expected_mask).scatter(1, routes, True)
        actual = torch.zeros_like(logits).scatter(1, routes, probs)
    else:
        mask, actual = routes, probs
    assert torch.equal(mask, expected_mask)
    torch.testing.assert_close(actual, reference)
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (reference * upstream).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    if collect:
        z = logits.detach()
        cutoff = (z + bias).topk(topk + 1, dim=1).values[:, -1:]
        bins = ((cutoff - z - bounds[0]) * (257 / (bounds[1] - bounds[0]))).floor().long()
        bins.clamp_(0, 256)
        expected = torch.stack([torch.bincount(col, minlength=257) for col in bins.T])
        assert torch.equal(hist.long(), expected)
        torch.testing.assert_close(span, (z.max(1).values - z.min(1).values).max())


def test_logit_histograms_pool_microbatches_and_ignore_common_offsets():
    """Repeated calls share one histogram and one range; token-wide shifts cancel."""
    torch.manual_seed(7)
    logits = torch.randn(32, 8)
    bias = torch.linspace(-2, 2, 8)

    def collect(chunks):
        hist = torch.zeros(8, 100, dtype=torch.int32)
        bounds, span = torch.tensor([-16.0, 16.0]), torch.zeros(())
        for chunk in chunks:
            moe_utils.topk_routing_with_score_function(
                chunk,
                2,
                score_function="sigmoid",
                expert_bias=bias,
                qb_bias_space="logit",
                qb_histogram=hist,
                qb_bin_bounds=bounds,
                qb_logit_span=span,
            )
        return hist, span

    whole, span = collect([logits])
    chunks, chunk_span = collect(logits.chunk(4))
    shifted, shifted_span = collect([logits + 100])
    assert torch.equal(chunks, whole)
    assert torch.equal(shifted, whole)
    torch.testing.assert_close(chunk_span, span)
    torch.testing.assert_close(shifted_span, span, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("span_value", [0.0, 3.0, 50.0])
def test_logit_update_adapts_range(span_value):
    """Histogram quantiles set bias; a pooled logit span sets the next bounds."""
    hist = torch.tensor([[0, 2, 2, 0], [2, 2, 0, 0]], dtype=torch.int32)
    span = torch.tensor(span_value)
    bias, bounds = moe_utils.get_updated_expert_bias_with_quantile(
        hist, torch.tensor([-1.0, 1.0]), torch.zeros(2), 1, logit_span=span
    )
    torch.testing.assert_close(bias, torch.tensor([0.25, -0.25]))
    width = 1.1 * max(span_value, 1e-3)
    torch.testing.assert_close(bounds, torch.tensor([-0.25 - width, 0.25 + width]))


@pytest.mark.parametrize("bad", [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_logit_preserves_bias_and_bounds(bad):
    """An invalid batch must not poison persistent histogram bounds."""
    z = torch.zeros(2, 4)
    z[0, 0] = bad
    hist = torch.zeros(4, 64, dtype=torch.int32)
    bounds, span, bias = torch.tensor([-16.0, 16.0]), torch.zeros(()), torch.arange(4).float()
    moe_utils.topk_routing_with_score_function(
        z,
        2,
        score_function="sigmoid",
        expert_bias=bias,
        qb_bias_space="logit",
        qb_histogram=hist,
        qb_bin_bounds=bounds,
        qb_logit_span=span,
    )
    assert torch.isinf(span)
    new_bias, new_bounds = moe_utils.get_updated_expert_bias_with_quantile(
        hist, bounds, bias, 2, logit_span=span
    )
    assert torch.equal(new_bias, bias)
    assert torch.equal(new_bounds, bounds)


def test_empty_logit_batch_preserves_state():
    """Empty token shards contribute neither histogram counts nor a logit span."""
    hist, span = torch.zeros(8, 10, dtype=torch.int32), torch.zeros(())
    bounds, bias = torch.tensor([-16.0, 16.0]), torch.arange(8).float()
    probs, mask = moe_utils.topk_routing_with_score_function(
        torch.empty(0, 8),
        2,
        score_function="sigmoid",
        expert_bias=bias,
        qb_bias_space="logit",
        qb_histogram=hist,
        qb_bin_bounds=bounds,
        qb_logit_span=span,
    )
    assert probs.shape == mask.shape == (0, 8)
    assert span == 0 and hist.sum() == 0
    new_bias, new_bounds = moe_utils.get_updated_expert_bias_with_quantile(
        hist, bounds, bias, 2, logit_span=span
    )
    assert torch.equal(new_bias, bias)
    assert torch.equal(new_bounds, bounds)


@pytest.mark.parametrize(
    "overrides",
    [
        {"moe_router_qb_bias_space": "invalid"},
        {"moe_router_load_balancing_type": "none"},
        {"moe_router_quantile_balancing_estimation_scope": "micro_batch"},
        {"moe_router_qb_logit_initial_bound": 0},
        {"moe_router_qb_logit_initial_bound": float('inf')},
        {"moe_router_qb_logit_initial_bound": float('nan')},
    ],
)
def test_invalid_logit_config(overrides):
    """Reject unsupported bias units, estimators, and initial bounds."""
    with pytest.raises(ValueError):
        _config(**overrides)


def test_logit_cli():
    """Expose the bias units and initial range through generated training arguments."""
    from megatron.training.arguments import add_megatron_arguments

    parser = argparse.ArgumentParser()
    add_megatron_arguments(parser)
    args = parser.parse_args(
        ['--moe-router-qb-bias-space', 'logit', '--moe-router-qb-logit-initial-bound', '32']
    )
    assert args.moe_router_qb_bias_space == 'logit'
    assert args.moe_router_qb_logit_initial_bound == 32


def test_old_te_rejected_before_logit_dispatch():
    """Do not silently route in score space with an older installed TE."""
    with (
        patch.object(moe_utils, 'HAVE_TE', True),
        patch.object(moe_utils, 'fused_topk_with_score_function', lambda **kwargs: None),
        patch.object(moe_utils, 'fused_topk_with_score_function_supports_logit_bias', False),
    ):
        with pytest.raises(ValueError, match='does not support logit-space'):
            moe_utils.topk_routing_with_score_function(
                torch.zeros(2, 8),
                2,
                score_function='sigmoid',
                expert_bias=torch.zeros(8),
                qb_bias_space='logit',
                fused=True,
            )


def test_fused_logit_dense_output_contract():
    """Adapt TE's full-width probabilities and dense IDs to Core's compact pair."""
    logits = torch.tensor([[0.0, -2.0, 1.0, 2.0]])
    bias = torch.tensor([3.0, 0.0, 0.0, 0.0])
    expected, indices = moe_utils.topk_routing_with_score_function(
        logits,
        2,
        score_function='sigmoid',
        expert_bias=bias,
        qb_bias_space='logit',
        dense_output=True,
    )

    def te_forward(**kwargs):
        assert kwargs['qb_bias_space'] == 'logit'
        ids = kwargs['topk_indices']
        ids.copy_(indices)
        return torch.zeros_like(logits).scatter(1, ids, expected), ids

    with (
        patch.object(moe_utils, 'HAVE_TE', True),
        patch.object(moe_utils, 'fused_topk_with_score_function', te_forward),
        patch.object(moe_utils, 'fused_topk_with_score_function_supports_logit_bias', True),
        patch.object(moe_utils, 'fused_topk_with_score_function_supports_topk_indices', True),
    ):
        actual, actual_ids = moe_utils.topk_routing_with_score_function(
            logits,
            2,
            score_function='sigmoid',
            expert_bias=bias,
            qb_bias_space='logit',
            dense_output=True,
            fused=True,
        )
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_ids, indices)


class TestLogitRouterState:
    """Exercise real router buffers and state-dict hooks on a model-parallel rank."""

    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def router(self, space='logit', inference=False):
        """Construct an unfused router with the requested checkpoint bias units."""
        cls = InferenceTopKRouter if inference else TopKRouter
        return cls(
            _config(moe_router_qb_bias_space=space),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        ).cuda()

    @pytest.mark.parametrize('strict', [False, True])
    def test_checkpoint_units(self, strict):
        """Round-trip logit state, but reject cross-space loads even with strict=False."""
        logit, score = self.router(), self.router('score')
        logit.expert_bias.copy_(torch.arange(8, device='cuda'))
        saved = logit.state_dict()
        assert 'logit_expert_bias' in saved and 'expert_bias' not in saved
        assert 'logit_qb_bin_bounds' in saved and 'qb_logit_span' not in saved
        clone = self.router()
        clone.load_state_dict(saved, strict=strict)
        torch.testing.assert_close(clone.expert_bias, logit.expert_bias)
        with pytest.raises(RuntimeError, match='score-space'):
            logit.load_state_dict(score.state_dict(), strict=strict)
        with pytest.raises(RuntimeError, match='logit-space'):
            score.load_state_dict(saved, strict=strict)

    def test_repeated_router_and_reset(self):
        """Repeated MTP-style calls accumulate once per grad-enabled invocation."""
        from megatron.core.distributed.finalize_model_grads import reset_model_temporary_tensors

        router = self.router()
        z = torch.randn(16, 8, device='cuda', requires_grad=True)
        router.quantile_balancing(z)
        router.quantile_balancing(z * 2)
        assert torch.all(router.qb_histogram.sum(-1) == 32)
        expected = 2 * (z.detach().max(-1).values - z.detach().min(-1).values).max()
        torch.testing.assert_close(router.qb_logit_span, expected)
        with torch.no_grad():
            router.quantile_balancing(z * 10)
        torch.testing.assert_close(router.qb_logit_span, expected)
        reset_model_temporary_tensors(router.config, [router])
        assert router.qb_histogram.sum() == 0 and router.qb_logit_span == 0

    def test_score_only_inference_kernel_is_bypassed(self):
        """A score-space-only inference backend must not handle logit bias."""
        router = self.router(inference=True)
        router.config.inference_moe_router_backend = 'triton'
        assert not router._uses_sigmoid_topk_kernel(torch.zeros(16, 8, device='cuda'), None)

    @pytest.mark.parametrize('space', ['score', 'logit'])
    def test_global_qb_diagnostics(self, space):
        """Global QB diagnostics report its additive bias without reading absent beta."""
        router = self.router(space)
        router.expert_bias.copy_(torch.arange(8, device='cuda'))
        target = 'megatron.core.transformer.moe.router'
        with (
            patch(target + '.is_observing_tensor', return_value=True),
            patch(target + '.observe_tensor') as observe,
        ):
            router.routing(torch.randn(4, 2, 8, device='cuda'))
        diagnostics = observe.call_args.args[3]
        torch.testing.assert_close(
            diagnostics[:, RouterDiagnosticChannel.EXPERT_BIAS], router.expert_bias.expand(2, -1)
        )

    @pytest.mark.parametrize('space', ['score', 'logit'])
    def test_global_qb_dense_output(self, space):
        """The global QB entry point honors the compact output contract."""
        router = self.router(space)
        z = torch.randn(16, 8, device='cuda')
        probs, ids = router.quantile_balancing(z, dense_output=True)
        full, mask = router.quantile_balancing(z)
        assert probs.shape == ids.shape == (16, 2)
        assert ids.dtype == torch.int64
        torch.testing.assert_close(probs, full.gather(1, ids))
        assert torch.equal(mask, torch.zeros_like(mask).scatter(1, ids, True))

    def test_hash_layer_checkpoint_without_bias(self):
        """Hash layers in a logit-QB model have no learned bias to serialize."""
        router = self.router()
        router.config.moe_num_hash_layers = 1
        router.config.hash_moe_vocab_size = 16
        router.set_layer_number(1)
        assert router.expert_bias is None
        saved = router.state_dict()
        assert 'logit_expert_bias' not in saved and 'expert_bias' not in saved
        router.load_state_dict(saved)

    @pytest.mark.parametrize('virtual_experts', [False, True])
    def test_training_dispatch_probability_layout(self, virtual_experts):
        """Flex transports consume full-width probabilities; the planner uses compact ones."""
        router = self.router()
        router.config.moe_virtual_expert_load_balance = virtual_experts
        with (
            patch.object(router, '_dense_route_indices_dtype', return_value=torch.int64),
            patch('megatron.core.transformer.moe.router.is_observing_tensor', return_value=False),
        ):
            probs, routes = router.routing(torch.randn(4, 2, 8, device='cuda'))
        width = 2 if virtual_experts else 8
        assert probs.shape == routes.shape == (8, width)
        assert routes.dtype == (torch.int64 if virtual_experts else torch.bool)
