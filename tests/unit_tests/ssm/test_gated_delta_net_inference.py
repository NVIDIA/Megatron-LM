# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Focused kernel and CUDA-graph tests for Gated DeltaNet dynamic inference."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import HAVE_FLA, GatedDeltaNet
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_FLA, reason="FLA is not installed.")
class TestGatedDeltaNetInference:
    """Validate state continuity and capture of the real FLA inference kernels."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        # Other tests in this multi-rank bucket may change the process's current
        # device. Raw CUDA-graph capture and Triton launches must use LOCAL_RANK's
        # device, matching Utils.initialize_distributed's initial assignment.
        torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=256,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            num_layers=1,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            layernorm_zero_centered_gamma=True,
            num_attention_heads=8,
            num_query_groups=2,
            activation_func=F.silu,
            bf16=True,
            tensor_model_parallel_size=1,
            experimental_attention_variant="gated_delta_net",
            gdn_kernel_backend="fla",
            linear_attention_freq=[1],
            transformer_impl="transformer_engine",
        )
        spec = get_experimental_attention_variant_module_spec(config=config)
        pg_collection = ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
        self.gdn = (
            spec.module(
                config,
                submodules=spec.submodules,
                layer_number=1,
                bias=False,
                conv_bias=False,
                use_qk_l2norm=True,
                pg_collection=pg_collection,
            )
            .cuda()
            .bfloat16()
            .eval()
        )
        yield
        Utils.destroy_model_parallel()

    @staticmethod
    def _prefill_context(cu_seqlens, batch_indices):
        return SimpleNamespace(
            is_chunked_prefill_enabled=lambda: False,
            mamba_metadata=SimpleNamespace(
                cu_seqlens=cu_seqlens, batch_indices_prefill=batch_indices
            ),
        )

    def _empty_states(self, slots=4):
        conv_shape, ssm_shape = self.gdn.mamba_state_shapes_per_request()
        conv_state = torch.zeros(
            slots, *conv_shape, device="cuda", dtype=self.gdn.conv1d.weight.dtype
        )
        ssm_state = torch.zeros(
            slots, *ssm_shape, device="cuda", dtype=self.gdn.in_proj.weight.dtype
        )
        return conv_state, ssm_state

    @torch.inference_mode()
    def test_prefill_decode_matches_full_forward(self):
        prompt_len, total_len = 9, 14
        hidden = torch.randn(
            total_len, 1, self.gdn.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        expected, _ = self.gdn(hidden, attention_mask=None)
        projected, _ = self.gdn.in_proj(hidden)
        conv_state, ssm_state = self._empty_states()
        slot = torch.tensor([2], device="cuda", dtype=torch.int32)
        cu_seqlens = torch.tensor([0, prompt_len], device="cuda", dtype=torch.long)

        outputs = [
            self.gdn.ssm_prefill(
                projected[:prompt_len],
                conv_state,
                ssm_state,
                self._prefill_context(cu_seqlens, slot),
            )
        ]
        for token in projected[prompt_len:]:
            outputs.append(
                self.gdn.ssm_decode(
                    token.view(1, 1, -1), conv_state, ssm_state, batch_indices=slot
                ).transpose(0, 1)
            )
        actual, _ = self.gdn.out_proj(torch.cat(outputs, dim=0))
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)

    @torch.inference_mode()
    def test_padding_index_does_not_modify_state(self):
        conv_state, ssm_state = self._empty_states()
        projected = torch.randn(
            2, 1, self.gdn.in_proj_dim // self.gdn.tp_size, device="cuda", dtype=torch.bfloat16
        )
        indices = torch.tensor([1, -1], device="cuda", dtype=torch.int32)
        conv_before = conv_state.clone()
        ssm_before = ssm_state.clone()

        self.gdn.ssm_decode(projected, conv_state, ssm_state, batch_indices=indices)

        assert not torch.equal(conv_state[1], conv_before[1])
        assert not torch.equal(ssm_state[1], ssm_before[1])
        torch.testing.assert_close(conv_state[[0, 2, 3]], conv_before[[0, 2, 3]], atol=0, rtol=0)
        torch.testing.assert_close(ssm_state[[0, 2, 3]], ssm_before[[0, 2, 3]], atol=0, rtol=0)

    @torch.inference_mode()
    def test_decode_cuda_graph_capture_and_replay(self):
        batch = 4
        projected = torch.randn(
            batch, 1, self.gdn.in_proj_dim // self.gdn.tp_size, device="cuda", dtype=torch.bfloat16
        )
        indices = torch.arange(batch, device="cuda", dtype=torch.int32)

        eager_conv, eager_ssm = self._empty_states(slots=batch)
        expected = self.gdn.ssm_decode(projected, eager_conv, eager_ssm, batch_indices=indices)

        graph_conv, graph_ssm = self._empty_states(slots=batch)
        static_projected = projected.clone()
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                graph_conv.zero_()
                graph_ssm.zero_()
                self.gdn.ssm_decode(static_projected, graph_conv, graph_ssm, batch_indices=indices)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graph_conv.zero_()
        graph_ssm.zero_()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = self.gdn.ssm_decode(
                static_projected, graph_conv, graph_ssm, batch_indices=indices
            )
        graph_conv.zero_()
        graph_ssm.zero_()
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, expected, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(graph_conv, eager_conv, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(graph_ssm, eager_ssm, atol=3e-2, rtol=3e-2)
