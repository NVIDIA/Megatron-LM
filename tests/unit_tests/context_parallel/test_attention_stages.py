# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CP layout state must survive interleaved core/projection stages."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.context_parallel import CpPartitionModeConverter
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("variant", ["attention", "gdn", "gdn2"])
@pytest.mark.parametrize("tp_size,sequence_parallel", [(1, False), (2, True)])
def test_interleaved_attention_stages_preserve_cp_layout(variant, tp_size, sequence_parallel):
    """Compare split and atomic output/input/parameter gradients for two layouts.

    The second call takes the no-conversion path before the first projection runs;
    storing the return converter on the module would overwrite the first call's state.
    GDN variants use their deterministic reference recurrence so this tests the CP
    stage boundary independently of optional fused recurrence kernels.
    """
    if variant != "attention":
        pytest.importorskip("fla")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=1, context_parallel_size=2
    )
    try:
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=sequence_parallel,
            context_parallel_size=2,
            cp_partition_mode="contiguous",
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            gradient_accumulation_fusion=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            normalization="RMSNorm",
            activation_func=F.silu,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            experimental_attention_variant=None if variant == "attention" else variant,
            linear_attention_freq=None if variant == "attention" else [1],
            deterministic_mode=variant != "attention",
        )
        groups = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
            tp_cp=parallel_state.get_tensor_and_context_parallel_group(),
        )
        if variant == "attention":
            model = SelfAttention(
                config,
                get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                pg_collection=groups,
            )
        else:
            spec = get_experimental_attention_variant_module_spec(config=config)
            model = spec.module(
                config, submodules=spec.submodules, layer_number=1, pg_collection=groups
            )
        model = model.cuda().bfloat16()
        local_length = 16 // (tp_size if sequence_parallel else 1)
        inputs = [
            torch.randn(local_length, 1, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)
        ]

        def run(staged):
            model.zero_grad(set_to_none=True)
            xs = [x.detach().clone().requires_grad_() for x in inputs]
            results = []
            for x, layout in zip(xs, ("contiguous", "zigzag")):
                model._cp_input_partition_mode = layout
                if staged:
                    results.append(model.forward_pre_attn_and_core_attn(x, None))
                else:
                    results.append(model(x, None))
            if staged:
                assert isinstance(results[0], tuple)
                assert isinstance(results[0][1], CpPartitionModeConverter)
                assert results[0][1].target_partition_mode == "contiguous"
                assert isinstance(results[1], torch.Tensor)
                results = [model.forward_post_core_attn(state) for state in results]
            outputs = [out if bias is None else out + bias for out, bias in results]
            sum(out.float().square().mean() for out in outputs).backward()
            grads = {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            grads.update({f"input_{i}": x.grad.detach().clone() for i, x in enumerate(xs)})
            return [out.detach() for out in outputs], grads

        expected, reference_grads = run(staged=False)
        actual, grads = run(staged=True)
        for got, reference in zip(actual, expected):
            torch.testing.assert_close(got, reference, rtol=0, atol=0)
        assert grads.keys() == reference_grads.keys()
        for name in grads:
            torch.testing.assert_close(grads[name], reference_grads[name], rtol=1e-2, atol=1e-3)
    finally:
        Utils.destroy_model_parallel()
