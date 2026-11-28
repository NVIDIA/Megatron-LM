# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GDP output-norm checkpointing preserves gradients across live microbatches."""

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from tests.unit_tests.ssm.gdp_test_utils import (
    HAVE_GDP_DEPS,
    _assert_replay_matches,
    _build_mixer,
    _make_config,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    "extra",
    [
        {"recompute_modules": ["gdp_norm_out"]},
        {"context_parallel_size": 2},
        {"tensor_model_parallel_size": 2},
        {"fp8": "hybrid"},
        {"cuda_graph_impl": "local"},
        {"fine_grained_activation_offloading": True},
        {"dynamic_context_parallel": True},
    ],
)
def test_unsupported_norm_recompute_is_rejected(extra):
    """Unsupported execution must fail instead of silently saving invalid views."""
    from dataclasses import replace

    config = _make_config(["gdp_in_proj"])
    options = {"recompute_modules": ["gdp_in_proj", "gdp_norm_out"], **extra}
    with pytest.raises(ValueError, match="gdp_norm_out requires"):
        replace(config, **options)


@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="requires GDP dependencies")
@pytest.mark.parametrize("recompute_qkv", [False, True])
def test_norm_recompute_preserves_live_microbatch_gradients(monkeypatch, recompute_qkv):
    """Two forwards before backward must retain independent checkpoint recipes."""
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    try:
        modules = ["gdp_in_proj"] + (["gdp_qkv"] if recompute_qkv else [])
        reference_config = _make_config(modules)
        candidate_config = _make_config(modules + ["gdp_norm_out"])
        reference = Float16Module(reference_config, _build_mixer(reference_config)).module
        candidate = Float16Module(
            candidate_config, _build_mixer(candidate_config, reference.state_dict())
        ).module
        assert reference.supports_two_stage_attention()
        assert not candidate.supports_two_stage_attention()
        sources = [torch.randn(128, 2, 256, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        gradients = [torch.randn_like(value) for value in sources]

        def run(module):
            module.zero_grad(set_to_none=True)
            inputs = [value.clone().requires_grad_() for value in sources]
            outputs = [module(value)[0] for value in inputs]
            for output, gradient in zip(reversed(outputs), reversed(gradients)):
                output.backward(gradient)
            return (
                torch.stack([value.detach() for value in outputs]),
                torch.stack([value.grad for value in inputs]),
                {name: value.grad.clone() for name, value in module.named_parameters()},
            )

        expected = run(reference)
        for _ in range(2):
            _assert_replay_matches(expected, run(candidate))
        candidate.eval()
        with torch.no_grad():
            actual, _ = candidate(sources[0])
            wanted, _ = reference(sources[0])
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
