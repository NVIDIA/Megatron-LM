# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Construction-time validation and kernel selection for ``moe_inference_training_forward``.

Not a parity test. These are the checks that fail at ``TransformerConfig``
construction rather than in arithmetic -- the backend allow-list, the recompute
requirement, the settings that would change which experts the value pass picks --
because they are cheap to get wrong and expensive to discover: a rejected setting
found ten minutes into a two-node job is the failure this file exists to move to
the first second. Also pins the canonical route order both generation and the
value pass hand the FlashInfer kernel.

Config-only and single-rank, so it needs no GPU. The bitwise question, whether the
training value pass reproduces generation, needs EP ranks and a generation forward
to compare against and lives in ``test_inference_training_forward.py``.
"""

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend

_MXFP8 = dict(fp8='hybrid', fp8_recipe='mxfp8', fp8_param=True)


def _config(**overrides):
    """A valid per-token NVFP4 parity config, which each test then breaks one way."""
    kwargs = dict(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=8,
        num_moe_experts=8,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_grouped_gemm=True,
        add_bias_linear=False,
        gated_linear_unit=False,
        activation_func=squared_relu,
        transformer_impl='inference_optimized',
        # Required by inference_optimized: a BF16 router would cost a per-decode
        # dtype conversion, and its expert module assumes RMSNorm. Unrelated to
        # what is tested here, but without them every test dies before reaching
        # its assertion.
        moe_router_dtype='fp32',
        normalization='RMSNorm',
        inference_grouped_gemm_backend='flashinfer',
        inference_moe_nvfp4_recipe='per_token',
        attention_backend=AttnBackend.flash,
        flash_attention_version=4,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        batch_invariant_backend='te_native',
        moe_inference_training_forward=True,
        # The value pass stores no activations, so the backward has to come from
        # a recompute pass. Validated, not assumed.
        recompute_granularity='selective',
        recompute_modules=['moe'],
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestBackendAllowList:
    @pytest.mark.parametrize("recipe", ['nvfp4', 'mxfp8'])
    def test_flashinfer_quantized_recipes_are_accepted(self, recipe):
        overrides = dict(inference_moe_nvfp4_recipe=None, **_MXFP8) if recipe == 'mxfp8' else {}
        assert _config(**overrides).moe_inference_training_forward

    @pytest.mark.parametrize("backend", ['vllm', 'torch'])
    def test_backends_that_already_match_training_are_rejected(self, backend):
        """Under batch_invariant_mode their kernels already match TE training."""
        with pytest.raises(ValueError, match="supports only inference_grouped_gemm_backend"):
            _config(inference_grouped_gemm_backend=backend, inference_moe_nvfp4_recipe=None)

    def test_flashinfer_bf16_is_rejected(self):
        """The FlashInfer BF16 kernel is not batch invariant, so nothing could match it."""
        # batch_invariant_mode rejects it first, with its own message naming the recipes.
        with pytest.raises(ValueError, match="inference_moe_nvfp4_recipe"):
            _config(inference_moe_nvfp4_recipe=None)

    def test_batch_invariant_mode_is_required(self):
        """Generation and the value pass launch the kernel on different token counts."""
        with pytest.raises(ValueError, match="requires batch_invariant_mode"):
            _config(batch_invariant_mode=False)

    def test_requires_the_inference_optimized_transformer_impl(self):
        """Only that spec builds the inference-capable expert module."""
        with pytest.raises(ValueError, match="inference_optimized"):
            _config(transformer_impl='transformer_engine')


class TestRecomputeRequirement:
    def test_recompute_is_required(self):
        """Without it there is no backward graph for the experts at all."""
        with pytest.raises(ValueError, match="recompute"):
            _config(recompute_granularity=None, recompute_modules=None)

    def test_moe_must_be_among_the_recomputed_modules(self):
        with pytest.raises(ValueError, match="recompute"):
            _config(recompute_modules=['layernorm'])

    def test_local_cuda_graphs_are_rejected(self):
        """They disable MoE-layer recompute, leaving the value pass with no backward."""
        with pytest.raises(ValueError, match="cuda_graph_impl"):
            _config(cuda_graph_impl='local')


class TestParityBreakingSettings:
    """Settings that make the training router pick different experts than generation.

    Each is rejected rather than tolerated, because the loss of parity they cause
    is otherwise silent: the layer runs, the loss moves, and only the KL shows it.
    """

    def test_overlapped_schedule_is_rejected(self):
        """It calls the layer's stages directly and never reaches the pass switch."""
        with pytest.raises(ValueError, match="overlap_moe_expert_parallel_comm"):
            _config(overlap_moe_expert_parallel_comm=True)

    @pytest.mark.parametrize("balancing", ["sinkhorn", "quantile_balancing"])
    def test_router_balancing_that_changes_selection_is_rejected(self, balancing):
        with pytest.raises(ValueError, match="moe_router_load_balancing_type"):
            _config(moe_router_load_balancing_type=balancing)

    def test_input_jitter_is_rejected(self):
        with pytest.raises(ValueError, match="moe_input_jitter_eps"):
            _config(moe_input_jitter_eps=0.01)

    def test_forced_load_balancing_is_rejected(self):
        with pytest.raises(ValueError, match="moe_router_force_load_balancing"):
            _config(moe_router_force_load_balancing=True)


class TestMaxTokensPerRank:
    def test_defaults_to_unset(self):
        assert _config().moe_inference_training_max_tokens_per_rank is None

    def test_accepts_a_positive_bound(self):
        config = _config(moe_inference_training_max_tokens_per_rank=4096)
        assert config.moe_inference_training_max_tokens_per_rank == 4096

    @pytest.mark.parametrize("bound", [0, -1])
    def test_rejects_a_non_positive_bound(self, bound):
        with pytest.raises(ValueError, match="must be positive"):
            _config(moe_inference_training_max_tokens_per_rank=bound)


class TestCanonicalRouteOrder:
    """Both generation and the value pass hand the kernel routes sorted by expert id.

    The kernel sums a token's routes in the order given. Generation reads its expert
    ids from ``torch.topk`` on the router scores, the value pass recovers them from
    the training router's boolean map, which has no order. Sorting both, with the
    probabilities permuted alongside, makes the two orders, and so the two sums, equal.
    """

    @staticmethod
    def _routed(monkeypatch, routing_map, probs):
        from types import SimpleNamespace

        from megatron.core.transformer.moe import experts

        seen = {}

        def fake_kernel(hidden_states, topk_ids, topk_weights, weights, **kwargs):
            seen["ids"], seen["probs"] = topk_ids, topk_weights
            return hidden_states

        monkeypatch.setattr(experts, "flashinfer_routed_nvfp4_moe", fake_kernel)
        stub = SimpleNamespace(
            num_local_experts=8,
            ep_group=SimpleNamespace(size=lambda: 1, rank=lambda: 0),
            _nvls_dispatcher=False,
            _flashinfer_activation_type=SimpleNamespace(value=None),
        )
        experts.InferenceGroupedMLP._flashinfer_routed_forward(
            stub, torch.zeros(2, 4), routing_map, probs, nvfp4_weights=object()
        )
        return seen["ids"], seen["probs"]

    def test_routes_reach_the_kernel_sorted_with_their_probabilities(self, monkeypatch):
        routing_map = torch.tensor([[5, 1, 3], [0, 7, 2]])
        probs = torch.tensor([[0.5, 0.1, 0.3], [0.0, 0.7, 0.2]])
        ids, sorted_probs = self._routed(monkeypatch, routing_map, probs)
        assert ids.tolist() == [[1, 3, 5], [0, 2, 7]]
        assert torch.equal(sorted_probs, torch.tensor([[0.1, 0.3, 0.5], [0.0, 0.2, 0.7]]))

    def test_order_of_the_input_does_not_matter(self, monkeypatch):
        routing_map = torch.tensor([[5, 1, 3]])
        probs = torch.tensor([[0.5, 0.1, 0.3]])
        a = self._routed(monkeypatch, routing_map, probs)
        b = self._routed(monkeypatch, routing_map.flip(-1), probs.flip(-1))
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


class TestMxfp8Recipe:
    """The flag under a model-level MXFP8 recipe.

    Needs no extra opt-in. The recompute runs TE, which is MXFP8 too, so the
    forward and the backward share a precision; the kernels still differ, which is
    the premise of the whole path rather than something specific to MXFP8.
    """

    def test_mxfp8_is_accepted(self):
        config = _config(inference_moe_nvfp4_recipe=None, **_MXFP8)
        assert config.moe_inference_training_forward
