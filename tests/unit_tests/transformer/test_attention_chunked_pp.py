# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for SelfAttention with chunked pipeline model parallel."""

import copy
import pytest
import torch

from megatron.core.chunked_pipeline_parallel_utils import ChunkedPipelineParallelParams, KVCachePool
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("tp_size", [1, 2, 4])
class TestSelfAttentionChunkedPP:
    """Test SelfAttention with different chunked_pipeline_model_parallel_splits values."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, tp_size):
        self.tp_size = tp_size
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(42)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_chunked_pp_splits_consistency(self):
        """Test that different chunked_pipeline_model_parallel_splits produce consistent results.

        This test compares the output of SelfAttention with chunked_pp_splits=1 (baseline)
        against chunked_pp_splits=4 (chunked), using the same fixed input and model weights.
        """
        # Test parameters
        sequence_length = 256
        micro_batch_size = 2
        hidden_size = 128
        num_attention_heads = 8  # Must be divisible by tp_size
        seed = 42
        chunked_pp_splits_to_test = 4

        # Ensure num_attention_heads is divisible by tp_size
        assert num_attention_heads % self.tp_size == 0, (
            f"num_attention_heads ({num_attention_heads}) must be divisible by tp_size ({self.tp_size})"
        )

        # ===== Baseline: chunked_pp_splits = 1 =====
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)

        config_baseline = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            tensor_model_parallel_size=self.tp_size,
            sequence_parallel=False,  # Do not support sequence parallel in this test case
            context_parallel_size=1,  # Do not support context parallel in this test case
            use_cpu_initialization=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            chunked_pipeline_model_parallel_splits=1,
        )

        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        attn_layer_spec = layer_spec.submodules.self_attention.submodules

        attention_baseline = SelfAttention(
            config_baseline,
            attn_layer_spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).cuda()

        # Generate fixed input
        torch.manual_seed(seed + 100)  # Different seed for input
        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, hidden_size),
            device='cuda',
            dtype=torch.bfloat16,
        )
        hidden_states_baseline = hidden_states.clone().requires_grad_(True)

        # Forward pass baseline (no chunked PP)
        with torch.no_grad():
            output_baseline, bias_baseline = attention_baseline(
                hidden_states_baseline,
                attention_mask=None,
            )

        # ===== Test: chunked_pp_splits > 1 =====
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)

        config_chunked = copy.copy(config_baseline)
        config_chunked.chunked_pipeline_model_parallel_splits = chunked_pp_splits_to_test

        attention_chunked = SelfAttention(
            config_chunked,
            attn_layer_spec,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).cuda()

        # Copy weights from baseline to chunked model to ensure consistency
        attention_chunked.load_state_dict(attention_baseline.state_dict())

        # Calculate span sizes
        span_size = sequence_length // chunked_pp_splits_to_test
        spans = [span_size] * chunked_pp_splits_to_test

        # Forward pass with chunked PP (simulate chunked forward)
        output_chunks = []
        micro_batch_idx = 0
        kv_cache_pool = KVCachePool(config_chunked)
        with torch.no_grad():
            for span_idx in range(chunked_pp_splits_to_test):
                start_idx = span_idx * span_size
                end_idx = start_idx + span_size

                # Slice the input for this chunk
                hidden_states_chunk = hidden_states[start_idx:end_idx, :, :]

                # Create chunked PP params
                chunked_pp_params = ChunkedPipelineParallelParams(
                    micro_batch_idx=micro_batch_idx,
                    span_idx_in_micro=span_idx,
                    spans=spans,
                    kv_cache_pool=kv_cache_pool,
                )

                # Forward pass for this chunk
                output_chunk, bias_chunk = attention_chunked(
                    hidden_states_chunk,
                    attention_mask=None,
                    chunked_pp_params=chunked_pp_params,
                )
                output_chunks.append(output_chunk)

        # Concatenate all chunks
        output_chunked = torch.cat(output_chunks, dim=0)

        # ===== Compare results =====
        # Check shapes match
        assert output_baseline.shape == output_chunked.shape, (
            f"Shape mismatch: baseline {output_baseline.shape} vs chunked {output_chunked.shape}"
        )

        # Check values are close
        torch.testing.assert_close(
            output_baseline,
            output_chunked,
            atol=1e-2,
            rtol=1e-2,
            msg=lambda msg: f"Output mismatch between chunked_pp_splits=1 and "
                            f"chunked_pp_splits={chunked_pp_splits_to_test}: {msg}",
        )

        # Check bias consistency
        torch.testing.assert_close(
            bias_baseline,
            bias_chunk,  # Bias should be the same for all chunks
            atol=1e-5,
            rtol=1e-5,
            msg=lambda msg: f"Bias mismatch: {msg}",
        )
