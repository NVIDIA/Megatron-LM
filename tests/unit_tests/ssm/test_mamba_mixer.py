# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext

import pytest
import torch

from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import mamba_mixer as mamba_mixer_module
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
            use_mamba_mem_eff_path=use_mem_eff_path,
        )
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

    @pytest.mark.parametrize("pip_conv", [True, False], ids=["causal_conv1d", "conv_fallback"])
    @pytest.mark.parametrize("graphed", [False, True], ids=["eager", "cuda_graph"])
    def test_single_token_sequences_match_each_token_alone(self, graphed, pip_conv, monkeypatch):
        """In inference, packed sequences of one token each, as an MTP draft step without a
        context packs its requests, give every token the output it has as a sequence of its
        own, with or without pip causal_conv1d. They run without host syncs, so a CUDA graph
        captures them."""
        if not pip_conv:
            monkeypatch.setattr(mamba_mixer_module, "causal_conv1d_fn", None)
        mixer = self.get_mixer()
        num_tokens = 6
        hidden_states = torch.randn((num_tokens, 1, mixer.config.hidden_size), device="cuda")
        cu_seqlens = torch.arange(num_tokens + 1, dtype=torch.int32, device="cuda")
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=1,
            max_seqlen_kv=1,
            seq_idx=cu_seqlens[:-1].unsqueeze(0),
        )

        def packed_forward():
            return mixer(hidden_states, packed_seq_params=packed_seq_params)[0]

        with torch.inference_mode(), InferenceMode.active():
            alone = torch.cat([mixer(hidden_states[i : i + 1])[0] for i in range(num_tokens)])
            packed = packed_forward()
            if graphed:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    packed_forward()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    packed = packed_forward()
                graph.replay()
        # Different GEMM shapes and scan tiles: fp32 rounding only.
        torch.testing.assert_close(packed, alone, rtol=0, atol=1e-3 * alone.abs().max().item())

    @pytest.mark.parametrize("inference", [False, True], ids=["training", "inference"])
    @pytest.mark.parametrize("seqlens", [[3, 5], [1, 1, 1, 1]], ids=["sequences", "single_tokens"])
    def test_unfused_path_takes_packed_input_only_as_single_tokens_in_inference(
        self, inference, seqlens
    ):
        """The unfused (non-memory-efficient) path takes packed input only as sequences of one
        token each in inference. Packed training runs the memory-efficient path, which also
        lays packed sequences out for context parallelism."""
        mixer = self.get_mixer(use_mem_eff_path=False)
        cu_seqlens = torch.tensor([0] + seqlens, dtype=torch.int32, device="cuda").cumsum(0)
        max_seqlen = max(seqlens)
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens.int(),
            cu_seqlens_kv=cu_seqlens.int(),
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )
        hidden_states = torch.randn((sum(seqlens), 1, mixer.config.hidden_size), device="cuda")
        with torch.no_grad(), InferenceMode.active() if inference else nullcontext():
            if inference and max_seqlen == 1:
                mixer(hidden_states, packed_seq_params=packed_seq_params)
            else:
                with pytest.raises(AssertionError, match="non-memory-efficient"):
                    mixer(hidden_states, packed_seq_params=packed_seq_params)


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
