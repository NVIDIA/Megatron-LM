# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import copy

import pytest
import torch
from packaging import version

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec, get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.random import model_parallel_device_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.wrapped_process_group import WrappedProcessGroup
from tests.unit_tests.test_utilities import Utils
from megatron.core.device_utils import get_current_device, get_current_device_type

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

@pytest.mark.skip(reason="upstream issues")
class TestParallelAttention:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1,1)
        model_parallel_device_manual_seed(123)
        self.transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec() if HAVE_TE else get_gpt_layer_local_spec()
        self.parallel_attention = SelfAttention(self.transformer_config,
                                                transformer_layer_spec.submodules.self_attention.submodules,
                                                layer_number=1)


    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.parallel_attention, SelfAttention)
        assert self.parallel_attention.layer_number == 1

        num_weights = sum([p.numel() for p in self.parallel_attention.parameters()])
        assert num_weights == 648

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    def test_gpu_forward(self):

        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.parallel_attention.to(device=get_current_device())

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size))
        hidden_states = hidden_states.to(device=get_current_device())

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).to(device=get_current_device())

        output, bias = self.parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.flaky_in_dev
    def test_fused_rope_gpu_forward(self):
        self.parallel_attention.config.apply_rope_fusion = True
        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.parallel_attention.to(device=get_current_device())

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size))
        hidden_states = hidden_states.to(device=get_current_device())

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).to(device=get_current_device())
        rotary_pos_emb = torch.ones(sequence_length, 1, 1, self.parallel_attention.config.kv_channels).to(device=get_current_device())
        output, bias = self.parallel_attention(hidden_states, attention_mask, rotary_pos_emb=rotary_pos_emb)

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size
        self.parallel_attention.config.apply_rope_fusion = False

    @pytest.mark.flaky_in_dev
    def test_checkpointed_gpu_forward(self):
        transformer_config = self.transformer_config
        transformer_config.recompute_granularity = 'selective'
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec() if HAVE_TE else get_gpt_layer_local_spec()
        checkpointed_parallel_attention = SelfAttention(
            transformer_config,
            transformer_layer_spec.submodules.self_attention.submodules,
            layer_number=1,
        )
        config = checkpointed_parallel_attention.config

        sequence_length = 32
        micro_batch_size = 2

        checkpointed_parallel_attention.to(device=get_current_device())

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, checkpointed_parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.to(device=get_current_device())

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).to(device=get_current_device())

        output, bias = checkpointed_parallel_attention(hidden_states, attention_mask)

        assert config.recompute_granularity == 'selective'
        assert "core_attn" in config.recompute_modules
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


class TestSelfAttention:

    def setup_method(self, method):
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def run_self_attention(self, model_comm_pgs):
        self.transformer_config = TransformerConfig(
            num_layers=2, hidden_size=128, num_attention_heads=4, use_cpu_initialization=False
        )
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec() if HAVE_TE else get_gpt_layer_local_spec()
        self.self_attention = SelfAttention(
            self.transformer_config,
            transformer_layer_spec.submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            model_comm_pgs=model_comm_pgs,
        )

        config = self.self_attention.config
        sequence_length = 127
        micro_batch_size = 2

        self.self_attention.to(get_current_device())

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.self_attention.config.hidden_size),
            device=get_current_device(),
        )
        hidden_states_ref = copy.deepcopy(hidden_states)

        output, bias = self.self_attention(hidden_states, None)
        assert config.recompute_granularity is None
        # Check if output and bias have the correct shape
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.internal
    @pytest.mark.flaky
    def test_self_attention_mpu(self):

        tp_size = 4
        cp_size = 2
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_device_manual_seed(123)

        # Get TP and CP process groups from device mesh
        tp_group = parallel_state.get_tensor_model_parallel_group(wrapped=True)
        cp_group = parallel_state.get_context_parallel_group(wrapped=True)

        model_comm_pgs = ModelCommProcessGroups(tp=tp_group, cp=cp_group)

        self.run_self_attention(model_comm_pgs)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.flaky
    @pytest.mark.internal
    def test_self_attention_independent_pg_smoke(self):

        tp_size = 4
        cp_size = 2
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_device_manual_seed(123)

        # Create device mesh for TP and CP groups
        device_mesh = torch.distributed.init_device_mesh(
            get_current_device_type(), (tp_size, cp_size), mesh_dim_names=("tp", "cp")
        )
        # Get TP and CP process groups from device mesh
        tp_group = WrappedProcessGroup(device_mesh.get_group(mesh_dim="tp"))
        cp_group = WrappedProcessGroup(device_mesh.get_group(mesh_dim="cp"))

        model_comm_pgs = ModelCommProcessGroups(tp=tp_group, cp=cp_group)

        self.run_self_attention(model_comm_pgs)
