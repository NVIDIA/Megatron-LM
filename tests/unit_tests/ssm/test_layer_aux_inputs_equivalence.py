# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Packed calls preserve real core outputs and gradients."""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.layer_interface import AuxInputs
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.utils import (
    assert_bit_exact,
    capture_rng_state,
    restore_rng_state,
)
from tests.unit_tests.test_utilities import Utils


class TestAuxInputsEquivalence:
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "pattern,attribute", [("*", "self_attention"), ("-", "mlp"), ("E", "mlp"), ("M", "mixer")]
    )
    def test_core_forward_backward_matches_legacy(self, pattern, attribute):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            add_bias_linear=pattern != "E",
            num_moe_experts=2 if pattern == "E" else None,
            moe_router_topk=2,
            moe_shared_expert_intermediate_size=256 if pattern == "E" else None,
        )
        stack = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_config_list=validate_segment_layers(pattern, config),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            post_layer_norm=False,
            post_process=False,
        ).cuda()
        core = getattr(stack.layers[0], attribute)
        x = torch.randn(64, 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        kwargs = {"attention_mask": None} if pattern == "*" else {}
        rng = capture_rng_state()
        results = []
        for packed in (False, True):
            core.zero_grad(set_to_none=True)
            x.grad = None
            restore_rng_state(rng)
            output, bias = core(x, AuxInputs(**kwargs)) if packed else core(x, **kwargs)
            if packed:
                bias = bias.bias
            output = output if bias is None else output + bias
            output.float().square().sum().backward()
            grads = {
                name: parameter.grad.detach().clone()
                for name, parameter in core.named_parameters()
                if parameter.grad is not None
            }
            grads["input"] = x.grad.detach().clone()
            results.append((output.detach().clone(), grads))
        assert_bit_exact(*results[0], *results[1])
