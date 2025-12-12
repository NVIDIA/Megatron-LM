# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import partial
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank as get_tensor_on_this_cp_rank,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness

try:
    import fla

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


@pytest.mark.parametrize(
    ("tp_size", "sp", "cp_size"),
    [(1, False, 1), (2, False, 1), (2, True, 1), (1, False, 2), (2, False, 2), (2, True, 2)],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
@pytest.mark.internal
class TestGatedDeltaNet:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, tp_size, sp, cp_size):
        # Initialize parallel and random seed
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.sp_size = tp_size if sp else 1

        # Get TP and CP process groups from device mesh
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)

        # Initialize model
        self.transformer_config = TransformerConfig(
            hidden_size=256,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=8,
            activation_func=F.silu,
            bf16=True,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=sp,
            context_parallel_size=cp_size,
        )
        gdn_submodules = get_gpt_layer_with_transformer_engine_spec(
            experimental_attention_variant="gated_delta_net", normalization="RMSNorm"
        ).submodules.self_attention.submodules

        self.gdn = GatedDeltaNet(
            self.transformer_config,
            submodules=gdn_submodules,
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

    def test_gpu_forward(self):
        gdn = self.gdn

        micro_batch_size = 2
        seq_length = 64
        hidden_states = torch.ones(
            (seq_length // self.sp_size // self.cp_size, micro_batch_size, gdn.config.hidden_size),
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
        )
        attention_mask = None

        output, bias = gdn(hidden_states, attention_mask)

        assert output.dim() == 3, f"Output too many dimensions ({output.shape=})"
        assert output.shape[0] == seq_length // self.sp_size // self.cp_size, (
            f"Output shape {output.shape[0]=} mismatch with "
            f" {seq_length=} // {self.sp_size=} // {self.cp_size=}."
        )
        assert (
            output.shape[1] == micro_batch_size
        ), f"Output shape {output.shape[1]=} mismatch with {micro_batch_size=}"
        assert (
            output.shape[2] == gdn.config.hidden_size
        ), f"Output shape {output.shape[2]=} mismatch with {gdn.config.hidden_size=}"
        assert (
            output.dtype == hidden_states.dtype
        ), f"Output dtype {output.dtype=} mismatch with {hidden_states.dtype=}"


@pytest.mark.parametrize(
    ("tp", "sp", "cp"),
    [
        (4, False, 1),  # TP w/o SP
        (4, True, 1),  # TP w/ SP
        (1, False, 2),  # CP
        (2, False, 2),  # TP w/o SP + CP
        (2, True, 2),  # TP w/ SP + CP
    ],
)
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
def test_parallel_gated_delta_net_correctness(tmp_path_dist_ckpt, tp, sp, cp):
    transformer_config = TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=8,
        activation_func=F.silu,
        bf16=True,
    )

    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        experimental_attention_variant="gated_delta_net", normalization="RMSNorm"
    )

    if cp:
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 5e-4, 5e-4

    _test_parallel_attention_correctness(
        transformer_config, transformer_layer_spec, tmp_path_dist_ckpt, tp, sp, cp
    )
