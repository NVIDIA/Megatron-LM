# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared fixtures and helpers for Gated Delta Net unit tests."""

import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA_GDN2
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-multi-rank-gpu-enable
# NVLS doesn't support one single GPU to be shared by multiple ranks, so disable this in test.
os.environ.update({"NCCL_NVLS_ENABLE": "0"})


def _unpack_sequence(x: torch.Tensor, cu_seqlens: torch.Tensor, dim=1) -> list[torch.Tensor]:
    unpacked_x = []
    cu_seqlens_list = cu_seqlens.tolist()
    num_seqs = len(cu_seqlens_list) - 1
    for i in range(num_seqs):
        idx_start = cu_seqlens_list[i]
        idx_end = cu_seqlens_list[i + 1]
        chunked_index = [slice(None)] * dim + [slice(idx_start, idx_end)]
        unpacked_x.append(x[tuple(chunked_index)])
    return unpacked_x


class GatedDeltaNetTestBase:
    """Shared model-parallel setup for parametrized GDN tests."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self, tp_size, sp, cp_size, use_gdn2):
        if use_gdn2 and not HAVE_FLA_GDN2:
            pytest.skip("FLA with GDN2 support is not installed.")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.sp_size = tp_size if sp else 1
        self.use_gdn2 = use_gdn2

        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        self.transformer_config = TransformerConfig(
            hidden_size=2048,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=16,
            num_query_groups=2,
            activation_func=F.silu,
            bf16=True,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=sp,
            context_parallel_size=cp_size,
            experimental_attention_variant="gdn2" if use_gdn2 else "gated_delta_net",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        gdn_spec = get_experimental_attention_variant_module_spec(config=self.transformer_config)

        self.gdn = gdn_spec.module(
            self.transformer_config,
            submodules=gdn_spec.submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            use_qk_l2norm=True,
            A_init_range=(1, 16),
            pg_collection=pg_collection,
        )
        self.gdn = self.gdn.cuda().bfloat16()

    def teardown_method(self):
        Utils.destroy_model_parallel()
