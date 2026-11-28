# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Full packed Mamba/GDP forwards with runtime CP independent of construction."""

from copy import copy
from dataclasses import replace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_specs import gdp_stack_spec, hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import gated_delta_product, mamba_mixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.parametrize("variant", ["mamba", "gdp_headwise", "gdp_chunkwise"])
def test_runtime_cp_full_mixer_matches_static_cp(variant):
    """Compare output, input gradient and every CP-summed parameter gradient.

    The chunkwise case calls the full HybridStack so its runtime packed summary
    metadata is constructed from the selected microbatch group as in training.
    """
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("requires a world size divisible by two")
    if not mamba_mixer.HAVE_MAMBA_SSM:
        pytest.skip("requires mamba-ssm")
    if variant != "mamba" and not gated_delta_product.HAVE_FLA:
        pytest.skip("requires FLA GDP")
    if variant == "gdp_chunkwise" and not gated_delta_product.HAVE_FLA_GDP_CP:
        pytest.skip("requires FLA chunkwise GDP")

    Utils.initialize_model_parallel(dynamic_context_parallel=True)
    try:
        cp_group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=2)
        groups = ProcessGroupCollection.use_mpu_process_groups()
        chunkwise = variant == "gdp_chunkwise"
        # One-Householder GDP needs eight heads to keep its CP2 projected
        # convolution view aligned to causal-conv1d's eight-element stride.
        heads = max(4 if variant == "mamba" else 8, Utils.world_size)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            mamba_num_heads=heads,
            mamba_head_dim=64,
            mamba_num_groups=2,
            mamba_state_dim=128 if variant == "mamba" else 32,
            gdp_num_householder=1,
            params_dtype=torch.bfloat16,
            bf16=True,
            use_mamba_mem_eff_path=True,
            linear_cp_mode="chunkwise" if chunkwise else "headwise",
            linear_cp_layout="contiguous" if chunkwise else "zigzag",
            dynamic_context_parallel=True,
            calculate_per_token_loss=True,
            max_seqlen_per_dp_cp_rank=128,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )

        def build(model_config, model_groups):
            model_parallel_cuda_manual_seed(123)
            if chunkwise:
                return HybridStack(
                    model_config,
                    gdp_stack_spec.submodules,
                    layer_type_list=["M"],
                    post_layer_norm=False,
                    pg_collection=model_groups,
                ).cuda()
            spec = hybrid_stack_spec if variant == "mamba" else gdp_stack_spec
            mixer_spec = spec.submodules.mamba_layer.submodules.mixer
            return mixer_spec.module(
                model_config,
                mixer_spec.submodules,
                d_model=model_config.hidden_size,
                layer_number=1,
                pg_collection=model_groups,
            ).cuda()

        dynamic = build(config, groups)
        static_config = replace(
            config,
            context_parallel_size=2,
            dynamic_context_parallel=False,
            sequence_packing_scheduler=None,
        )
        static_groups = copy(groups)
        static_groups.cp = cp_group
        static = build(static_config, static_groups)
        static.load_state_dict(dynamic.state_dict())
        bounds = torch.tensor([0, 24, 64], dtype=torch.int32, device="cuda")
        sequence_ids = torch.tensor([0] * 24 + [1] * 40, device="cuda", dtype=torch.int32)[None]
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=bounds,
            cu_seqlens_kv=bounds,
            max_seqlen_q=40,
            max_seqlen_kv=40,
            seq_idx=sequence_ids,
        )
        runtime_packed = replace(packed, local_cp_size=2, cp_group=cp_group)
        torch.manual_seed(456 + cp_group.rank())
        hidden = torch.randn(32, 1, 128, device="cuda", dtype=torch.bfloat16) * 0.05
        output_gradient = torch.randn_like(hidden) * 0.05

        def run(module, metadata):
            x = hidden.detach().clone().requires_grad_()
            output = (
                module(x, attention_mask=None, packed_seq_params=metadata)
                if chunkwise
                else module(x, packed_seq_params=metadata)[0]
            )
            output.backward(output_gradient)
            gradients = {}
            for name, parameter in module.named_parameters():
                if parameter.grad is not None:
                    gradient = parameter.grad.detach().float().clone()
                    torch.distributed.all_reduce(gradient, group=cp_group)
                    gradients[name] = gradient
            return output.detach(), x.grad, gradients

        expected = run(static, packed)
        actual = run(dynamic, runtime_packed)
        torch.testing.assert_close(actual[0], expected[0], atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(actual[1], expected[1], atol=5e-3, rtol=5e-3)
        assert actual[2].keys() == expected[2].keys()
        assert actual[2], "full mixer must exercise learned parameters"
        for name in expected[2]:
            torch.testing.assert_close(
                actual[2][name], expected[2][name], atol=5e-3, rtol=5e-3, msg=name
            )
        assert dynamic.pg_collection.cp is groups.cp
        assert groups.cp.size() == 1
    finally:
        Utils.destroy_model_parallel()
