# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bitwise forward/backward checks of real core and skeleton entry points."""

import copy
import unittest

import torch

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.layer_interface import LayerInputs
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestLayerInputsEquivalence(unittest.TestCase):
    def test_real_cores_and_skeletons(self):
        for symbol in ("*", "-", "E", "M", "G", "+", "D"):
            for test_core in (False, True):
                with self.subTest(symbol=symbol, core=test_core):
                    check_equivalence(symbol, test_core)


def check_equivalence(symbol, test_core):
    """The new call must preserve outputs, input gradients, and all parameter gradients."""
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = MLATransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            hidden_dropout=0.1,
            attention_dropout=0.0,
            add_bias_linear=False,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            num_moe_experts=2,
            moe_router_topk=2,
            moe_shared_expert_intermediate_size=256,
            moe_token_dispatcher_type="alltoall",
            activation_func=torch.nn.functional.silu,
        )
        stack = (
            HybridStack(
                config,
                hybrid_stack_spec.submodules,
                layer_config_list=validate_segment_layers(symbol, config),
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                post_layer_norm=False,
                post_process=False,
            )
            .cuda()
            .train()
        )
        layer = stack.layers[0]
        module = layer
        if test_core:
            if symbol == 'M':
                module = layer.mixer
            elif symbol in ('-', 'E'):
                module = layer.mlp
            else:
                module = layer.self_attention
        x = torch.randn(64, 2, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones(2, 1, 64, 64, device='cuda', dtype=torch.bool).triu(1)
        inputs = LayerInputs(attention_mask=mask)
        cuda_state = torch.cuda.get_rng_state()
        tp_state = copy.deepcopy(get_cuda_rng_tracker().get_states())

        if test_core and symbol in ('M', '-', 'E'):
            legacy = module(x)
        else:
            legacy = module(x, attention_mask=mask)
        if test_core:
            output, bias = legacy
            if bias is not None:
                output = output + bias
        elif symbol == 'M':
            output = legacy
        else:
            output = legacy[0]
        output.float().square().sum().backward()
        expected_output = output.detach().clone()
        expected_x_grad = x.grad.clone()
        expected_grads = {
            name: p.grad.clone() for name, p in module.named_parameters() if p.grad is not None
        }
        module.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.set_rng_state(cuda_state)
        get_cuda_rng_tracker().set_states(tp_state)

        output = module(x, inputs)
        if test_core:
            addends = output
            output = addends[0]
            for addend in addends[1:]:
                output = output + addend
        output.float().square().sum().backward()
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, expected_x_grad, rtol=0, atol=0)
        actual_grads = {name: p.grad for name, p in module.named_parameters() if p.grad is not None}
        assert actual_grads.keys() == expected_grads.keys()
        for name in expected_grads:
            torch.testing.assert_close(
                actual_grads[name],
                expected_grads[name],
                rtol=0,
                atol=0,
                msg=lambda msg: f"{name}: {msg}",
            )
    finally:
        Utils.destroy_model_parallel()


if __name__ == "__main__":
    unittest.main()
