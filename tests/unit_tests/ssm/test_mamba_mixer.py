# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestMambaMixer:

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def get_mixer(self, tp_size=1, cp_size=1, use_mem_eff_path=True):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )
        modules = mamba_stack_spec.submodules.mamba_layer.submodules.mixer.submodules
        model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        mixer = MambaMixer(
            transformer_config,
            modules,
            transformer_config.hidden_size,
            layer_number=1,
            use_mem_eff_path=use_mem_eff_path,
            model_comm_pgs=model_comm_pgs,
        )
        mixer.cuda()
        return mixer

    @pytest.mark.parametrize(
        "tp_size,cp_size,use_mem_eff_path",
        [
            (1, 1, True),
            (1, 1, False),
            (8, 1, True),
            (4, 2, True),
            (2, 4, True),
            (1, 8, True),
            (1, 8, False),
        ],
    )
    def test_gpu_forward(self, tp_size, cp_size, use_mem_eff_path):
        mixer = self.get_mixer(1, 1, use_mem_eff_path)
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, mixer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, bias = mixer(hidden_states)
        assert mixer.config.mamba_num_heads == None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == mixer.config.hidden_size
        assert output.dtype == torch.float32

    def test_variable_batch_size_inference(self):
        mixer = self.get_mixer()

        # Test cases where batch size decreases, remains the same, and increases
        micro_batch_sizes = [4, 2, 2, 8]
        sequence_length = 32
        inference_context = StaticInferenceContext(
            max_batch_size=max(micro_batch_sizes), max_sequence_length=sequence_length
        )

        for micro_batch_size in micro_batch_sizes:
            inference_context.max_seqlen = inference_context.max_sequence_length
            inference_context.seqlen_offset = inference_context.sequence_len_offset
            hidden_states = torch.ones(
                (sequence_length, micro_batch_size, mixer.config.hidden_size)
            )
            hidden_states = hidden_states.cuda()
            output, bias = mixer(hidden_states, inference_context=inference_context)
            assert mixer.config.mamba_num_heads == None
            assert output.shape[0] == sequence_length
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == mixer.config.hidden_size
            assert output.dtype == torch.float32


class TestMambaMixerErrorChecks:

    @pytest.mark.parametrize(
        "hidden_size, ngroups, tp_size, expected_error_message",
        [
            (65, 8, 1, "d_inner must be evenly divisible by headdim"),
            (96, 8, 2, "nheads must be evenly divisble by tp_size"),  # nheads = 3
            (128, 2, 4, "ngroups must be evenly divisible by tp_size"),
            (128, 8, 4, "nheads must be evenly divisible by ngroups"),  # nheads = 4
        ],
    )
    def test_error_check(self, hidden_size, ngroups, tp_size, expected_error_message):
        Utils.initialize_model_parallel(tp_size)
        transformer_config = TransformerConfig(
            hidden_size=hidden_size,
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
            mamba_num_groups=ngroups,
        )
        submodules = mamba_stack_spec.submodules.mamba_layer.submodules.mixer.submodules
        model_comm_pgs = ModelCommProcessGroups.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        with pytest.raises(AssertionError, match=expected_error_message):
            MambaMixer(
                transformer_config,
                submodules,
                transformer_config.hidden_size,
                model_comm_pgs=model_comm_pgs,
            )
        Utils.destroy_model_parallel()
