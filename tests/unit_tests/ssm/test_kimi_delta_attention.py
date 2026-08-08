# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.kimi_delta_attention import KimiDeltaAttention
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import fla

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

# NVLS doesn't support one single GPU shared by multiple ranks, so disable it in tests.
os.environ.update({"NCCL_NVLS_ENABLE": "0"})


@pytest.mark.parametrize(("tp_size", "sp"), [(1, False), (2, False), (2, True)])
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestKimiDeltaAttention:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, tp_size, sp):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        self.tp_size = tp_size
        self.sp_size = tp_size if sp else 1

        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        # Kimi-Linear-like config; only num_layers is shrunk.
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
            experimental_attention_variant="kimi_delta_attention",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        submodules = get_experimental_attention_variant_module_spec(
            config=self.transformer_config
        ).submodules

        self.kda = KimiDeltaAttention(
            self.transformer_config,
            submodules=submodules,
            layer_number=1,
            bias=False,
            conv_bias=False,
            conv_init=1.0,
            A_init_range=(1, 16),
            pg_collection=pg_collection,
        )
        self.kda = self.kda.cuda().bfloat16()

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        kda = self.kda
        micro_batch_size = 2
        seq_length = 64
        hidden_states = torch.ones(
            (seq_length // self.sp_size, micro_batch_size, kda.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )

        output, bias = kda(hidden_states, attention_mask=None)

        assert output.dim() == 3, f"Output too many dimensions ({output.shape=})"
        assert (
            output.shape[0] == seq_length // self.sp_size
        ), f"Output shape {output.shape[0]=} mismatch with {seq_length=} // {self.sp_size=}."
        assert (
            output.shape[1] == micro_batch_size
        ), f"Output shape {output.shape[1]=} mismatch with {micro_batch_size=}"
        assert (
            output.shape[2] == kda.config.hidden_size
        ), f"Output shape {output.shape[2]=} mismatch with {kda.config.hidden_size=}"
        assert (
            output.dtype == hidden_states.dtype
        ), f"Output dtype {output.dtype=} mismatch with {hidden_states.dtype=}"

    def test_sharded_state_dict(self):
        sharded_sd = self.kda.sharded_state_dict(prefix="")

        # The TP-sharded raw parameters and the depthwise conv must be sharded tensors.
        for key in ("A_log", "dt_bias", "conv1d.weight"):
            assert key in sharded_sd, f"Missing {key} in sharded state dict."
            assert isinstance(sharded_sd[key], ShardedTensor), f"{key} is not a ShardedTensor."

        # The child projections must contribute their own weights.
        for key in (
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "b_proj.weight",
            "f_proj_down.weight",
            "f_proj_up.weight",
            "g_proj_down.weight",
            "g_proj_up.weight",
            "out_proj.weight",
        ):
            assert key in sharded_sd, f"Missing {key} in sharded state dict."

        # A_log is per local value-head; dt_bias is per local gate channel (HV/tp * K).
        num_v_heads_local = 32 // self.tp_size
        assert sharded_sd["A_log"].local_shape == (num_v_heads_local,)
        assert sharded_sd["dt_bias"].local_shape == (num_v_heads_local * 128,)
