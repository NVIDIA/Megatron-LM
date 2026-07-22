# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

import megatron.core.ssm.ops.ssd_combined as ssd_combined
from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.ssm.ops.ssd_combined import _cutedsl_ssd_enabled
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


@pytest.mark.skipif(
    not _cutedsl_ssd_enabled(),
    reason="CuteDSL SSD backend requires Blackwell (SM 10.0+) and the cutlass DSL runtime",
)
class TestMambaMixerCuteDSL:
    """The mamba layer's varlen SSM prefill must give near-identical results with the
    CuteDSL backend vs Triton. CuteDSL accelerates the case where each request length
    is a multiple of the chunk size; the mixer routes the varlen prefill through the
    ``mamba_chunk_scan_combined_varlen`` dispatcher (CuteDSL by default on Blackwell;
    forced per-backend here via the dispatcher's cached decision)."""

    def teardown_method(self, method):
        ssd_combined._CUTEDSL_SSD_ENABLED = None  # restore backend autodetection
        Utils.destroy_model_parallel()

    def _build_mixer(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=256, num_layers=1, num_attention_heads=1, use_cpu_initialization=True
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        mixer = MambaMixer(
            config,
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            config.hidden_size,
            layer_number=1,
            pg_collection=pg_collection,
        )
        return mixer.cuda()

    def _run_prefill(
        self, mixer, zxBCdt, seq_idx, cu_seqlens, batch_indices, num_requests, backend
    ):
        # Force the dispatcher's cached backend decision (no env knob anymore).
        ssd_combined._CUTEDSL_SSD_ENABLED = backend == "cutedsl"

        conv_dim = mixer.d_inner_local_tp + 2 * mixer.ngroups_local_tp * mixer.d_state
        conv_state = torch.zeros(num_requests, conv_dim, mixer.d_conv, device="cuda")
        ssm_state = torch.zeros(
            num_requests, mixer.nheads_local_tp, mixer.headdim, mixer.d_state, device="cuda"
        )
        # Prefill runs under inference (no autograd) in production; mirror that here
        # (the SSD scan returns plain state tensors, and cumsum(out=) needs no grad).
        # _ssm_prefill mutates zxBCdt/states in place, so feed fresh copies per backend.
        with torch.no_grad():
            y = mixer._ssm_prefill(
                zxBCdt=zxBCdt.clone(),
                conv_state=conv_state,
                ssm_state=ssm_state,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                batch_indices=batch_indices,
            )
        return y, ssm_state

    def test_prefill_cutedsl_matches_triton(self):
        torch.manual_seed(7)
        mixer = self._build_mixer()

        num_requests = 4  # extra padding slots; only request 0 is real
        real_seq_len = 256  # multiple of chunk_size -> exercises the CuteDSL kernel
        assert real_seq_len % mixer.chunk_size == 0

        dim_inputs = (
            mixer.d_inner_local_tp * 2
            + 2 * mixer.ngroups_local_tp * mixer.d_state
            + mixer.nheads_local_tp
        )
        zxBCdt = torch.randn(real_seq_len, 1, dim_inputs, device="cuda", dtype=torch.float32)
        seq_idx = torch.zeros((1, real_seq_len), dtype=torch.int32, device="cuda")
        cu_seqlens = torch.tensor([0, real_seq_len], dtype=torch.int32, device="cuda")
        batch_indices = torch.tensor([0], dtype=torch.long, device="cuda")

        y_tri, st_tri = self._run_prefill(
            mixer, zxBCdt, seq_idx, cu_seqlens, batch_indices, num_requests, "triton"
        )
        y_cute, st_cute = self._run_prefill(
            mixer, zxBCdt, seq_idx, cu_seqlens, batch_indices, num_requests, "cutedsl"
        )

        torch.testing.assert_close(y_cute, y_tri, rtol=2e-2, atol=0.25)
        torch.testing.assert_close(
            st_cute[batch_indices], st_tri[batch_indices], rtol=3e-2, atol=3e-2
        )
