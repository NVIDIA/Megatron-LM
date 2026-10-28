# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
from einops import rearrange

from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestMambaMixer:

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def get_mixer(
        self,
        tp_size=1,
        cp_size=1,
        use_mem_eff_path=True,
        batch_invariant=False,
        params_dtype=torch.float32,
    ):
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
            use_mamba_mem_eff_path=use_mem_eff_path,
            params_dtype=params_dtype,
        )
        # Only the mixer's decode caches depend on it; skip the full mode validation.
        transformer_config.batch_invariant_mode = batch_invariant
        assert isinstance(hybrid_stack_spec.submodules, HybridStackSubmodules)
        assert isinstance(hybrid_stack_spec.submodules.mamba_layer.submodules, MambaLayerSubmodules)
        assert isinstance(
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            MambaMixerSubmodules,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        mixer = MambaMixer(
            transformer_config,
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            transformer_config.hidden_size,
            layer_number=1,
            pg_collection=pg_collection,
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

        with InferenceMode.active():
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

    @pytest.mark.parametrize("refresh_on_eval", [False, True], ids=["direct", "eval"])
    def test_refreshes_decode_cache_in_place(self, refresh_on_eval):
        mixer = self.get_mixer()
        cache_ptr = mixer._A_neg_exp_cache.data_ptr()

        with torch.no_grad():
            mixer.A_log.add_(1.0)
        if refresh_on_eval:
            mixer.train()
            mixer.eval()
        else:
            mixer.refresh_cache()

        torch.testing.assert_close(mixer._A_neg_exp_cache, -torch.exp(mixer.A_log.float()))
        assert mixer._A_neg_exp_cache.data_ptr() == cache_ptr
        assert not mixer._A_neg_exp_cache_stale

    @pytest.mark.parametrize("batch_invariant", [False, True], ids=["default", "bi"])
    @pytest.mark.parametrize("refresh_on_eval", [False, True], ids=["direct", "eval"])
    def test_refreshes_fp32_param_caches_in_place(self, refresh_on_eval, batch_invariant):
        """Inference reads FP32 conv weight/bias (and, in batch-invariant mode, A and
        dt_bias) from caches that hold exactly the per-step values and refresh in place."""
        mixer = self.get_mixer(batch_invariant=batch_invariant, params_dtype=torch.bfloat16)
        caches = [mixer._conv_weight_fp32_cache, mixer._conv_bias_fp32_cache]
        if batch_invariant:
            caches += [mixer._A_neg_exp_cache, mixer._bi_dt_bias_cache]
        ptrs = [c.data_ptr() for c in caches]

        with torch.no_grad():
            for p in (mixer.A_log, mixer.dt_bias, mixer.conv1d_weight, mixer.conv1d_bias):
                p.add_(0.5)
        if refresh_on_eval:
            mixer.train()
            mixer.eval()
        else:
            mixer.refresh_cache()
            mixer.eval()  # already refreshed: eval() must not need to refresh again

        with torch.no_grad():
            got = list(
                mixer._get_conv_params(mixer.conv1d_weight, mixer.conv1d_bias, torch.float32)
            )
            bi = mixer._get_batch_invariant_decode_params()
        expected = [
            rearrange(mixer.conv1d_weight, "d 1 w -> d w").float(),
            mixer.conv1d_bias.float(),
        ]
        if batch_invariant:
            got += list(bi)
            expected += [-torch.exp(mixer.A_log.float()), mixer.dt_bias.float()]
        else:
            assert bi is None
        for g, e in zip(got, expected):
            assert torch.equal(g, e)
        assert [c.data_ptr() for c in got] == ptrs
        # Grad mode (and training) computes the values per call instead.
        weight, _ = mixer._get_conv_params(mixer.conv1d_weight, mixer.conv1d_bias, torch.float32)
        assert weight.data_ptr() != ptrs[0]
        assert mixer._get_batch_invariant_decode_params() is None

    def test_no_batch_invariant_caches_by_default(self):
        mixer = self.get_mixer().eval()
        assert mixer._bi_dt_bias_cache is None
        with torch.no_grad():
            assert mixer._get_batch_invariant_decode_params() is None
            # FP32 parameters need no cast, so the conv kernels read them directly.
            weight, bias = mixer._get_conv_params(
                mixer.conv1d_weight, mixer.conv1d_bias, torch.float32
            )
        assert weight.data_ptr() == mixer.conv1d_weight.data_ptr()
        assert bias.data_ptr() == mixer.conv1d_bias.data_ptr()


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
        assert isinstance(hybrid_stack_spec.submodules, HybridStackSubmodules)
        assert isinstance(hybrid_stack_spec.submodules.mamba_layer.submodules, MambaLayerSubmodules)
        assert isinstance(
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            MambaMixerSubmodules,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        with pytest.raises(AssertionError, match=expected_error_message):
            MambaMixer(
                transformer_config,
                hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
                transformer_config.hidden_size,
                pg_collection=pg_collection,
            )
        Utils.destroy_model_parallel()
