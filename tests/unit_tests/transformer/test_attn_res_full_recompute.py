# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Full Hybrid checkpoints retain depth-source gradients across chunk boundaries."""

import os
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from megatron.core.recompute import checkpointed_attn_res_hybrid_forward
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention_residual import AttentionResidual
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _Entry(torch.nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.layer_number = index + 1
        self.attn_res_is_block_start = index % 3 == 0
        self.attn_res = AttentionResidual(config)
        self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, partial, *, attn_res_sources, **kwargs):
        sources = attn_res_sources if self.attn_res_is_block_start else (*attn_res_sources, partial)
        delta = torch.nn.functional.dropout(self.proj(self.attn_res(sources)), p=0.2, training=True)
        return delta if self.attn_res_is_block_start else partial + delta


@pytest.mark.parametrize(
    "method,size", [("uniform", 1), ("uniform", 2), ("uniform", 4), ("block", 3)]
)
@pytest.mark.parametrize("frozen_input", [False, True])
@pytest.mark.parametrize("distribute", [False, True])
def test_depth_sources_and_all_gradients(method, size, frozen_input, distribute):
    """Run under torchrun with two ranks to also exercise real activation sharding."""
    world = min(int(os.environ.get("WORLD_SIZE", 1)), 2)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=world, pipeline_model_parallel_size=1
    )
    try:
        from megatron.core import parallel_state

        config = TransformerConfig(
            num_layers=6,
            hidden_size=32,
            num_attention_heads=2,
            is_hybrid_model=True,
            enable_attention_residuals=True,
            attn_res_block_layers=3,
            attn_res_impl="torch",
            tensor_model_parallel_size=world,
            recompute_granularity="full",
            recompute_method=method,
            recompute_num_layers=size,
            distribute_saved_activations=distribute,
        )
        model_parallel_cuda_manual_seed(7391)
        torch.manual_seed(7391)
        layers = torch.nn.ModuleList([_Entry(config, i) for i in range(6)]).cuda()
        reference = deepcopy(layers)
        stack = SimpleNamespace(
            config=config,
            layers=layers,
            _execution_layer_config_list=[config] * 6,
            pg_collection=SimpleNamespace(tp=parallel_state.get_tensor_model_parallel_group()),
        )
        initial = torch.randn(8, 1, 32, device="cuda", requires_grad=not frozen_input)
        expected_input = initial.detach().clone().requires_grad_(not frozen_input)
        state = torch.cuda.get_rng_state()
        expected_partial, expected_sources = expected_input, []
        for layer in reference:
            if layer.attn_res_is_block_start:
                expected_sources.append(expected_partial)
            expected_partial = layer(expected_partial, attn_res_sources=tuple(expected_sources))
        expected = expected_partial.square().mean() + sum(
            v.square().mean() for v in expected_sources
        )
        expected.backward()
        torch.cuda.set_rng_state(state)
        partial, sources = checkpointed_attn_res_hybrid_forward(
            stack,
            initial,
            (),
            attention_mask=None,
            rotary_pos_emb=None,
            packed_seq_params=None,
            padding_mask=None,
            input_ids=None,
            use_inner_quantization_context=False,
        )
        assert len(sources) == len(expected_sources) == 2
        for actual, wanted in zip(
            (partial, *sources), (expected_partial, *expected_sources), strict=True
        ):
            assert actual.shape == (8, 1, 32)
            torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
        actual_loss = partial.square().mean() + sum(v.square().mean() for v in sources)
        actual_loss.backward()
        if frozen_input:
            assert initial.grad is None
        else:
            torch.testing.assert_close(initial.grad, expected_input.grad, atol=2e-6, rtol=2e-5)
        for (name, actual), (ref_name, wanted) in zip(
            layers.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert name == ref_name
            if wanted.grad is None:
                assert actual.grad is None, name
            else:
                torch.testing.assert_close(actual.grad, wanted.grad, atol=2e-6, rtol=2e-5, msg=name)
    finally:
        Utils.destroy_model_parallel()
