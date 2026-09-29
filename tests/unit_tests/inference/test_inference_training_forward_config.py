# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Construction-time validation for ``moe_inference_training_forward``.

Not a parity test. These are the checks that fail at ``TransformerConfig``
construction rather than in arithmetic -- the backend allow-list, the recompute
requirement, the settings that would change which experts the value pass picks --
because they are cheap to get wrong and expensive to discover: a rejected setting
found ten minutes into a two-node job is the failure this file exists to move to
the first second.

Config-only and single-rank, so it needs no GPU. The bitwise question, whether the
training value pass reproduces generation, needs EP ranks and a generation forward
to compare against and lives in ``test_inference_training_forward.py``.
"""

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.transformer import TransformerConfig


def _config(**overrides):
    """A valid squared-ReLU parity config, which each test then breaks one way."""
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
        inference_grouped_gemm_backend='vllm',
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
    @pytest.mark.parametrize("backend", ['vllm', 'torch'])
    def test_backends_with_a_value_pass_are_accepted(self, backend):
        """Torch shares the value pass with vLLM: same weights, and under MXFP8 same kernel."""
        config = _config(inference_grouped_gemm_backend=backend)
        assert config.moe_inference_training_forward

    def test_unsupported_backend_is_rejected_rather_than_silently_wrong(self):
        """'flashinfer' has no training-side weight rebuild.

        Its routed kernel reads a shuffled Major-K copy derived from the
        parameters rather than a view of them, so without a per-forward rebuild
        it reads weights the optimizer has since moved -- a plausible number
        rather than an error, the worst failure mode for a parity feature.
        """
        with pytest.raises(ValueError, match="live parameters"):
            _config(inference_grouped_gemm_backend='flashinfer')

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


class TestMxfp8Recipe:
    """The flag under a model-level MXFP8 recipe.

    Needs no extra opt-in. The recompute runs TE, which is MXFP8 too, so the
    forward and the backward share a precision; the kernels still differ, but that
    is the premise of the whole path rather than something specific to MXFP8.
    """

    @pytest.mark.parametrize("backend", ['vllm', 'torch'])
    def test_mxfp8_is_accepted(self, backend):
        config = _config(
            inference_grouped_gemm_backend=backend,
            fp8='hybrid',
            fp8_recipe='mxfp8',
            # Required by inference_optimized with the MXFP8 recipe.
            fp8_param=True,
        )
        assert config.moe_inference_training_forward
