# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import contextlib
import math
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import (
    DynamicInferenceContext,
    RequestOverflowError,
    TokenOverflowError,
)
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@contextlib.contextmanager
def rounder_override(n):
    original_token_rounder = DynamicInferenceContext.TOKEN_ROUNDER
    original_request_rounder = DynamicInferenceContext.REQUEST_ROUNDER
    try:
        DynamicInferenceContext.TOKEN_ROUNDER = n
        DynamicInferenceContext.REQUEST_ROUNDER = n
        yield
    finally:
        DynamicInferenceContext.TOKEN_ROUNDER = original_token_rounder
        DynamicInferenceContext.REQUEST_ROUNDER = original_request_rounder


class TestDynamicContext:

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def _setup_model_parallel_group(self, tensor_parallel_size, pipeline_parallel_size):
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
        )
        model_parallel_cuda_manual_seed(123)

    def _restore_model_parallel(self):
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def _get_dynamic_context(
        self,
        params_dtype,
        num_layers,
        kv_channels,
        num_attention_heads,
        max_sequence_length,
        buffer_size_gb,
        block_size_tokens,
        max_tokens,
        is_hybrid_model=False,
        layer_type_list=None,
        paused_buffer_size_gb=None,
        num_cuda_graphs=None,
        num_speculative_tokens=0,
        enable_chunked_prefill: bool = False,
        max_requests: int = None,
    ):
        if is_hybrid_model:
            if layer_type_list is None:
                layer_type_list = [Symbols.MAMBA, Symbols.MLP, Symbols.ATTENTION, Symbols.MLP]
            mamba_conv_states_shape = (544, 4)
            mamba_ssm_states_shape = (8, 64, 16)
            mamba_inference_state_config = MambaInferenceStateConfig(
                layer_type_list,
                mamba_conv_states_shape,
                mamba_ssm_states_shape,
                params_dtype,
                params_dtype,
            )
        else:
            mamba_inference_state_config = None

        dynamic_context = DynamicInferenceContext(
            model_config=TransformerConfig(
                params_dtype=params_dtype,
                num_layers=num_layers,
                kv_channels=kv_channels,
                num_attention_heads=num_attention_heads,
            ),
            inference_config=InferenceConfig(
                max_sequence_length=max_sequence_length,
                num_cuda_graphs=num_cuda_graphs,
                use_cuda_graphs_for_non_decode_steps=True,
                buffer_size_gb=buffer_size_gb,
                paused_buffer_size_gb=(
                    0.2 * buffer_size_gb if paused_buffer_size_gb is None else paused_buffer_size_gb
                ),
                block_size_tokens=block_size_tokens,
                max_tokens=max_tokens,
                num_speculative_tokens=num_speculative_tokens,
                mamba_inference_state_config=mamba_inference_state_config,
                use_flashinfer_fused_rope=None,  # default to using flash-infer if available
                # this is for compatibility with the LTS environment
                unified_memory_level=0,  # unit tests currently broken with UVM
                enable_chunked_prefill=enable_chunked_prefill,
                max_requests=max_requests,
            ),
        )
        return dynamic_context

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_initialize_dynamic_context(self, is_hybrid_model: bool):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            block_size_tokens=128,
            max_tokens=None,
            is_hybrid_model=is_hybrid_model,
        )

        if not is_hybrid_model:
            assert dynamic_context.kv_block_allocator.total_count == 491
            assert dynamic_context.kv_block_allocator.active_count == 392
            # We make max_requests divisible by the REQUEST_ROUNDER.
            assert dynamic_context.max_requests == 448
            assert dynamic_context.max_tokens == 16384
            assert dynamic_context.num_mamba_layers == 0
            assert dynamic_context.mamba_metadata is None
        else:
            assert dynamic_context.kv_block_allocator.total_count == 556
            assert dynamic_context.kv_block_allocator.active_count == 444
            assert dynamic_context.max_requests == 512
            assert dynamic_context.max_tokens == 16384
            assert dynamic_context.num_mamba_layers == 1
            assert dynamic_context.mamba_metadata is not None

        # Check initializations to -1
        assert torch.all(dynamic_context.request_ids == -1)

    @pytest.mark.internal
    def test_is_static_batching(self):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=8,
            max_sequence_length=512,
            buffer_size_gb=1.0,
            block_size_tokens=128,
            max_tokens=None,
        )
        assert not dynamic_context.is_static_batching()

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_is_memory_available(self, is_hybrid_model):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=8,
            max_sequence_length=512,
            buffer_size_gb=1.0,
            block_size_tokens=128,
            max_tokens=None,
            is_hybrid_model=is_hybrid_model,
        )
        dynamic_context.kv_block_allocator.total_avail = 10
        assert dynamic_context.kv_block_allocator.is_memory_available(10)
        assert not dynamic_context.kv_block_allocator.is_memory_available(11)

        assert dynamic_context.kv_block_allocator.is_memory_available(1)
        dynamic_context.kv_block_allocator.total_avail = 0
        assert not dynamic_context.kv_block_allocator.is_memory_available(1)

    @pytest.mark.internal
    @rounder_override(1)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_request_overflow(self, is_hybrid_model: bool):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=8,
            max_sequence_length=128,
            buffer_size_gb=0.01,
            block_size_tokens=32,
            max_tokens=None,
            is_hybrid_model=is_hybrid_model,
        )
        dynamic_context.max_requests //= 2
        with pytest.raises(RequestOverflowError):
            for i in range(dynamic_context.max_requests + 1):
                dynamic_context.add_request(
                    DynamicInferenceRequest(
                        request_id=i,
                        prompt_tokens=torch.zeros(10, device='cuda'),
                        sampling_params=SamplingParams(
                            num_tokens_to_generate=dynamic_context.max_tokens - 10
                        ),
                    )
                )  # Adding more than allowed requests

    @pytest.mark.internal
    @rounder_override(1)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_token_overflow_error(self, is_hybrid_model: bool):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=8,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=128,
            max_tokens=200,  # setting low, but >= context.max_requests.
            is_hybrid_model=is_hybrid_model,
        )

        with pytest.raises(TokenOverflowError):
            dynamic_context.add_request(
                DynamicInferenceRequest(
                    request_id=1,
                    prompt_tokens=torch.arange(0, 225, device='cuda'),
                    sampling_params=SamplingParams(
                        num_tokens_to_generate=dynamic_context.max_tokens - 25
                    ),
                )
            )  # Exceeding max token count

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_reset(self, is_hybrid_model: bool):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=8,
            max_sequence_length=128,
            buffer_size_gb=1.0,
            block_size_tokens=128,
            max_tokens=None,
            is_hybrid_model=is_hybrid_model,
        )

        # Initialize all variables
        dynamic_context.total_request_count = 10
        dynamic_context.active_token_count = 10
        dynamic_context.paused_request_count = 5
        dynamic_context.padded_active_token_count = 10
        dynamic_context.padded_active_request_count = 5
        dynamic_context.paused_tokens = torch.tensor([1, 2, 3], device='cuda')
        dynamic_context.request_ids.fill_(1)
        dynamic_context.request_query_lengths.fill_(1)
        dynamic_context.request_kv_length_offsets.fill_(1)
        dynamic_context.request_kv_block_counts.fill_(1)
        dynamic_context.request_last_kv_block_id.fill_(1)
        dynamic_context.request_last_kv_block_offset.fill_(1)
        dynamic_context.token_to_input_ids.fill_(1)
        dynamic_context.token_to_pos_ids.fill_(1)
        dynamic_context.token_to_request_idx.fill_(1)
        dynamic_context.token_to_position_in_request.fill_(1)
        dynamic_context.token_to_block_idx.fill_(1)
        dynamic_context.token_to_local_position_within_kv_block.fill_(1)
        dynamic_context.memory_buffer.fill_(1)
        dynamic_context.request_to_kv_block_ids.fill_(1)
        if is_hybrid_model:
            dynamic_context.mamba_conv_states.fill_(1)
            dynamic_context.mamba_ssm_states.fill_(1)

        # Call reset
        dynamic_context.reset()

        # Assert all variables are reset to zero or their default values
        assert dynamic_context.total_request_count == 0
        assert dynamic_context.active_token_count == 0
        assert dynamic_context.paused_request_count == 0
        assert dynamic_context.padded_active_token_count == 0
        assert dynamic_context.padded_active_request_count == 0
        assert dynamic_context.paused_tokens is None
        assert torch.all(dynamic_context.request_ids == -1)
        assert torch.all(dynamic_context.request_query_lengths == 0)
        assert torch.all(dynamic_context.request_kv_length_offsets == 0)
        assert torch.all(dynamic_context.request_kv_block_counts == 0)
        assert torch.all(dynamic_context.request_last_kv_block_id == -1)
        assert torch.all(dynamic_context.request_last_kv_block_offset == 0)
        assert torch.all(dynamic_context.token_to_input_ids == 0)
        assert torch.all(dynamic_context.token_to_pos_ids == 0)
        assert torch.all(dynamic_context.token_to_request_idx == -1)
        assert torch.all(dynamic_context.token_to_position_in_request == 0)
        assert torch.all(dynamic_context.token_to_block_idx == -1)
        assert torch.all(dynamic_context.token_to_local_position_within_kv_block == 0)
        if not is_hybrid_model:
            assert dynamic_context.kv_block_allocator.active_count == 819
            assert dynamic_context.kv_block_allocator.total_count == 1024
        else:
            assert dynamic_context.kv_block_allocator.active_count == 1517
            assert dynamic_context.kv_block_allocator.total_count == 1897
        assert torch.all(dynamic_context.request_to_kv_block_ids == -1)
        if is_hybrid_model:
            assert torch.all(dynamic_context.mamba_metadata.request_to_mamba_state_idx == -1)

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_allocate_and_release_memory_blocks(self, is_hybrid_model):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            block_size_tokens=128,
            max_tokens=None,
            is_hybrid_model=is_hybrid_model,
        )

        if is_hybrid_model:
            expected_memory_blocks = [551, 552, 553, 554]
        else:
            expected_memory_blocks = [486, 487, 488, 489]
        expected_block_count_avail = expected_memory_blocks[0]

        assert (
            dynamic_context.kv_block_allocator.allocate_memory_blocks(4)
            .cpu()
            .detach()
            .numpy()
            .tolist()
            == expected_memory_blocks
        )
        assert dynamic_context.kv_block_allocator.total_avail == expected_block_count_avail
        dynamic_context.kv_block_allocator.release_memory_blocks(
            torch.tensor(expected_memory_blocks[-2:], device='cuda')
        )
        assert dynamic_context.kv_block_allocator.total_avail == expected_block_count_avail + 2
        assert (
            dynamic_context.kv_block_allocator.allocate_memory_blocks(1).item()
            == expected_memory_blocks[-1]
        )
        assert dynamic_context.kv_block_allocator.total_avail == expected_block_count_avail + 1
        # Should return None since we allocate more blocks than what we have.
        assert (
            dynamic_context.kv_block_allocator.allocate_memory_blocks(
                dynamic_context.kv_block_allocator.total_avail + 100
            )
            == None
        )

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_add_request(self, is_hybrid_model: bool):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            block_size_tokens=128,
            max_tokens=None,
            is_hybrid_model=is_hybrid_model,
        )
        assert dynamic_context.block_size_tokens == 128
        context_length = 144
        dynamic_context.add_request(
            DynamicInferenceRequest(
                request_id=0,
                prompt_tokens=torch.arange(0, context_length, dtype=torch.long, device='cuda'),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=dynamic_context.max_tokens - context_length
                ),
            )
        )
        assert dynamic_context.total_request_count == 1
        assert dynamic_context.active_token_count == context_length
        assert dynamic_context.request_ids[0] == 0
        assert torch.all(dynamic_context.request_ids[1:] == -1)
        assert dynamic_context.request_query_lengths[0] == context_length
        assert dynamic_context.request_kv_length_offsets[0] == 0
        assert dynamic_context.request_kv_block_counts[0] == 2
        assert dynamic_context.request_last_kv_block_id[0].item() == (
            554 if is_hybrid_model else 489
        )
        assert dynamic_context.request_last_kv_block_offset[0].item() == 15
        assert torch.all(
            dynamic_context.token_to_pos_ids[0:context_length]
            == torch.arange(0, context_length, dtype=torch.long, device='cuda')
        )
        assert torch.all(
            dynamic_context.token_to_input_ids[0:context_length]
            == torch.arange(0, context_length, dtype=torch.long, device='cuda')
        )
        assert torch.all(
            dynamic_context.token_to_position_in_request[0:context_length]
            == torch.arange(0, context_length, dtype=torch.long, device='cuda')
        )

        # Verify token_to_block_idx and token_to_local_position_within_kv_block based on assigned blocks
        first_block_id = dynamic_context.request_to_kv_block_ids[0, 0]
        second_block_id = dynamic_context.request_to_kv_block_ids[0, 1]

        assert torch.all(
            dynamic_context.token_to_block_idx[0:context_length][
                0 : dynamic_context.block_size_tokens
            ]
            == first_block_id
        )
        assert torch.all(
            dynamic_context.token_to_block_idx[0:context_length][
                dynamic_context.block_size_tokens : context_length
            ]
            == second_block_id
        )
        assert torch.all(
            dynamic_context.token_to_local_position_within_kv_block[0:context_length]
            == torch.arange(0, context_length, dtype=torch.long, device='cuda')
            % dynamic_context.block_size_tokens
        )

    @pytest.mark.internal
    @rounder_override(64)
    def test_add_dummy_requests_parallel_populates_state(self):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=16,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.01,
            block_size_tokens=4,
            max_tokens=None,
        )

        requests = [
            DynamicInferenceRequest(
                request_id=100,
                prompt_tokens=torch.arange(0, 3, device='cuda'),
                sampling_params=SamplingParams(num_tokens_to_generate=2, termination_id=7),
            ),
            DynamicInferenceRequest(
                request_id=101,
                prompt_tokens=torch.arange(3, 9, device='cuda'),
                sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=8),
            ),
        ]

        lengths = [req.remaining_prompt_length for req in requests]
        total_tokens = sum(lengths)
        block_avail_before = dynamic_context.kv_block_allocator.total_avail

        dynamic_context.add_dummy_requests_parallel(requests, count_as_prefill=False)

        assert dynamic_context.active_token_count == total_tokens
        assert dynamic_context.total_request_count == len(requests)
        assert dynamic_context.num_prefill_requests == 0
        assert dynamic_context.kv_block_allocator.total_avail == block_avail_before

        expected_tokens = torch.cat(
            [torch.arange(0, 3, device='cuda'), torch.arange(3, 9, device='cuda')]
        )
        assert torch.equal(dynamic_context.token_to_input_ids[:total_tokens], expected_tokens)

        expected_positions = torch.tensor(
            [0, 1, 2, 0, 1, 2, 3, 4, 5], device='cuda', dtype=torch.long
        )
        assert torch.equal(
            dynamic_context.token_to_position_in_request[:total_tokens], expected_positions
        )
        assert torch.equal(dynamic_context.token_to_pos_ids[:total_tokens], expected_positions)

        expected_request_indices = torch.tensor(
            [0, 0, 0, 1, 1, 1, 1, 1, 1], device='cuda', dtype=torch.long
        )
        assert torch.equal(
            dynamic_context.token_to_request_idx[:total_tokens], expected_request_indices
        )

        expected_local = expected_positions % dynamic_context.block_size_tokens
        assert torch.equal(
            dynamic_context.token_to_local_position_within_kv_block[:total_tokens], expected_local
        )

        dummy_block_idx = dynamic_context.kv_block_allocator.dummy_block_idx
        assert torch.all(dynamic_context.token_to_block_idx[:total_tokens] == dummy_block_idx)

        assert torch.equal(
            dynamic_context.request_query_lengths[: len(requests)],
            torch.tensor(lengths, device='cuda', dtype=torch.int32),
        )
        assert torch.equal(
            dynamic_context.request_output_lengths[: len(requests)],
            torch.tensor([5, 7], device='cuda', dtype=torch.int32),
        )
        assert torch.equal(
            dynamic_context.request_kv_block_counts[: len(requests)],
            torch.tensor([1, 2], device='cuda', dtype=torch.int32),
        )
        assert torch.all(
            dynamic_context.request_to_kv_block_ids[0, :1] == dummy_block_idx
        ), "first request should use dummy block"
        assert torch.all(
            dynamic_context.request_to_kv_block_ids[1, :2] == dummy_block_idx
        ), "second request should use dummy blocks"
        assert torch.all(dynamic_context.request_to_kv_block_ids[:2, 2:] == -1)

        assert torch.all(dynamic_context.request_last_kv_block_id[:2] == dummy_block_idx)
        assert torch.equal(
            dynamic_context.request_last_kv_block_offset[:2],
            torch.tensor([2, 1], device='cuda', dtype=torch.int32),
        )

        assert torch.equal(
            dynamic_context.request_metadata["termination_id"][:2],
            torch.tensor([7.0, 8.0], device='cuda'),
        )

    @pytest.mark.internal
    @rounder_override(64)
    def test_add_dummy_requests_parallel_hybrid_allocates_mamba(self):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            block_size_tokens=8,
            max_tokens=None,
            is_hybrid_model=True,
            layer_type_list=[Symbols.MAMBA, Symbols.ATTENTION, Symbols.MLP, Symbols.ATTENTION],
        )

        request = DynamicInferenceRequest(
            request_id=55,
            prompt_tokens=torch.arange(0, 5, device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=4, termination_id=9),
        )

        dynamic_context.add_dummy_requests_parallel([request])

        mamba_idx = dynamic_context.mamba_metadata.request_to_mamba_state_idx[0].item()
        assert mamba_idx >= 0
        assert torch.all(dynamic_context.mamba_conv_states[:, mamba_idx] == 0)
        assert torch.all(dynamic_context.mamba_ssm_states[:, mamba_idx] == 0)

    @pytest.mark.internal
    @rounder_override(64)
    def test_add_dummy_requests_parallel_decode_does_not_count_as_prefill(self):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=256,
            buffer_size_gb=0.02,
            block_size_tokens=4,
            max_tokens=1_000_000,
        )

        request = DynamicInferenceRequest(
            request_id=5,
            prompt_tokens=torch.arange(0, 1, device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=2),
        )

        dynamic_context.num_prefill_requests = 0
        dynamic_context.add_dummy_requests_parallel([request], count_as_prefill=False)
        assert dynamic_context.num_prefill_requests == 0

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_update_request(self, is_hybrid_model: bool):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            block_size_tokens=128,
            max_tokens=None,
            is_hybrid_model=is_hybrid_model,
        )

        # This case should just reset and return since all requests are finished
        active_requests_mask = torch.Tensor([0, 0, 0])
        dynamic_context.paused_request_count = 0
        dynamic_context.total_request_count = 3
        dynamic_context.request_kv_block_counts[0:3] = 1
        new_block_ids = dynamic_context.kv_block_allocator.allocate_memory_blocks(3)
        dynamic_context.request_to_kv_block_ids[0:3, 0] = new_block_ids

        if is_hybrid_model:
            # Also initialize Mamba states for the dummy requests
            dynamic_context.mamba_conv_states[:, 0:3, :, :].fill_(1.0)
            dynamic_context.mamba_ssm_states[:, 0:3, :, :, :].fill_(1.0)

        dynamic_context.update_requests(
            active_requests_mask=active_requests_mask, new_tokens=torch.tensor([0, 1, 2])
        )
        assert dynamic_context.total_request_count == 0

        # This case would cover all cases
        # 1. Already there will be 2 paused requests
        # 2. Active request mask will have active and finished requests.
        # 3. The active requests will also have some requests that have to be paused because of reaching max token limit within block
        # 4. Some of these requests will be resumed.
        # Setup is as follows :
        # Request ids 0, 1 are paused
        # Request ids 2, 4, 9 are active requests
        # Request ids 3 7 8 have completed
        # Request ids 5 and 6 will require on more block later on because they finished their current block

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            block_size_tokens=128,
            max_tokens=None,
            is_hybrid_model=is_hybrid_model,
        )

        active_requests_mask = torch.Tensor([1, 0, 1, 1, 1, 0, 0, 1]).cuda().int()
        next_tokens = torch.arange(2, 10, device='cuda').int()
        dynamic_context.paused_request_count = 2
        dynamic_context.paused_tokens = torch.Tensor([0, 1]).cuda().int()
        dynamic_context.total_request_count = 5

        # Total req count should be equal to paused + num elements in active request mask.
        # So here it will raise an assertion error
        with pytest.raises(AssertionError) as error:
            dynamic_context.update_requests(
                active_requests_mask=active_requests_mask, new_tokens=next_tokens
            )

        total_request_count = 10
        dynamic_context.kv_block_allocator.total_avail -= 11  # We align 11 blocks to the 10 requests we have. 3rd request alone we setup like it requires 2 blocks
        dynamic_context.total_request_count = total_request_count

        dynamic_context.request_to_kv_block_ids[0:total_request_count, 0] = torch.arange(
            dynamic_context.kv_block_allocator.total_avail,
            dynamic_context.kv_block_allocator.total_avail + 10,
        )
        dynamic_context.request_to_kv_block_ids[3][
            1
        ] = dynamic_context.kv_block_allocator.total_avail  # Assign one extra block  to request 3.
        dynamic_context.request_kv_length_offsets[0:total_request_count] = 10
        # For 0, 1, 5, 6, the total number of tokens in last block is block size -1, so that they will all need extra blocks
        dynamic_context.request_kv_length_offsets[0:2] = dynamic_context.block_size_tokens - 1
        dynamic_context.request_kv_length_offsets[5:7] = dynamic_context.block_size_tokens - 1
        # For the 3rd request, its completed and required 2 blocks. So we add more tokens than block size
        dynamic_context.request_kv_length_offsets[3] = dynamic_context.block_size_bytes + 10
        dynamic_context.request_query_lengths[0:total_request_count] = (
            1  # Everything is in decode phase
        )

        dynamic_context.request_ids[0:total_request_count] = torch.arange(0, total_request_count)
        dynamic_context.request_kv_block_counts[0:total_request_count] = 1
        dynamic_context.request_kv_block_counts[3] = 2  # 3rd block alone requies 2 blocks
        dynamic_context.request_last_kv_block_id[0:total_request_count] = torch.arange(
            0, total_request_count
        )
        dynamic_context.request_last_kv_block_id[3] = 11
        dynamic_context.request_last_kv_block_offset[0:total_request_count] = 10
        # For the 3rd request, its completed and required 2 blocks. So we add more tokens than block size
        dynamic_context.request_last_kv_block_offset[0:2] = dynamic_context.block_size_tokens - 1
        dynamic_context.request_last_kv_block_offset[5:7] = dynamic_context.block_size_tokens - 1

        if is_hybrid_model:
            # Dummy fill for states to be non-zero before update
            for i in range(total_request_count):
                dynamic_context.mamba_metadata.request_to_mamba_state_idx[i] = i
            dynamic_context.mamba_metadata.mamba_state_free_slot_count -= total_request_count
            dynamic_context.mamba_conv_states[:, 0:total_request_count, :, :] = 1.0
            dynamic_context.mamba_ssm_states[:, 0:total_request_count, :, :, :] = 1.0

        dynamic_context.update_requests(
            active_requests_mask=active_requests_mask, new_tokens=next_tokens
        )

        # Then set up the test data
        dynamic_context.request_ids[0:10] = torch.tensor(
            [0, 1, 5, 6, 4, 2, 9, 7, 8, 9], device=torch.cuda.current_device()
        )

        # Now verify the values
        assert dynamic_context.request_ids[0:10].cpu().numpy().tolist() == [
            0,
            1,
            5,
            6,
            4,
            2,
            9,
            7,
            8,
            9,
        ]

        assert dynamic_context.paused_request_count == 0
        assert dynamic_context.total_request_count == 7
        assert dynamic_context.active_token_count == 7

        # The first four are zero because they have all obtained a new block
        assert dynamic_context.request_last_kv_block_offset[0:10].cpu().numpy().tolist() == [
            0,
            0,
            0,
            0,
            11,
            11,
            11,
            10,
            10,
            10,
        ]
        assert dynamic_context.token_to_input_ids[
            : dynamic_context.active_token_count
        ].cpu().numpy().tolist() == [0, 1, 5, 6, 4, 2, 9]

        assert dynamic_context.token_to_pos_ids[
            : dynamic_context.active_token_count
        ].cpu().numpy().tolist() == [128, 128, 128, 128, 11, 11, 11]

        # The first 4 requests will require an extra block.
        # Since 3 requests have finished, the last 3 rows should be all -1.
        if is_hybrid_model:
            assert torch.all(
                dynamic_context.request_to_kv_block_ids[0:10].cpu()
                == torch.tensor(
                    [
                        [544, 547, -1, -1],
                        [545, 544, -1, -1],
                        [549, 551, -1, -1],
                        [550, 552, -1, -1],
                        [548, -1, -1, -1],
                        [546, -1, -1, -1],
                        [553, -1, -1, -1],
                        [-1, -1, -1, -1],
                        [-1, -1, -1, -1],
                        [-1, -1, -1, -1],
                    ]
                )
            )
        else:
            assert torch.all(
                dynamic_context.request_to_kv_block_ids[0:10].cpu()
                == torch.tensor(
                    [
                        [479, 482, -1, -1],
                        [480, 479, -1, -1],
                        [484, 486, -1, -1],
                        [485, 487, -1, -1],
                        [483, -1, -1, -1],
                        [481, -1, -1, -1],
                        [488, -1, -1, -1],
                        [-1, -1, -1, -1],
                        [-1, -1, -1, -1],
                        [-1, -1, -1, -1],
                    ]
                )
            )

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_release_memory_blocks_for_finished_requests(self, is_hybrid_model):
        """Test that memory blocks are correctly released for finished requests."""

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            block_size_tokens=128,
            max_tokens=None,
            is_hybrid_model=is_hybrid_model,
        )

        # Set up the initial state with 5 requests
        # Allocate 5 blocks for 5 requests
        initial_blocks = dynamic_context.kv_block_allocator.allocate_memory_blocks(5)
        dynamic_context.total_request_count = 5
        dynamic_context.paused_request_count = 0

        # Record the available blocks before releasing memory
        initial_available_blocks = dynamic_context.kv_block_allocator.total_avail

        # Assign blocks to the requests (one block per request)
        for i in range(5):
            dynamic_context.request_to_kv_block_ids[i, 0] = initial_blocks[i]
            dynamic_context.request_query_lengths[i] = 1
            dynamic_context.request_ids[i] = i
            dynamic_context.request_last_kv_block_id[i] = initial_blocks[i]
            dynamic_context.request_last_kv_block_offset[i] = 0
            dynamic_context.request_kv_block_counts[i] = 1
            dynamic_context.request_in_prefill_status_tensor[i] = 0
            if is_hybrid_model:
                dynamic_context.mamba_conv_states[:, i, :, :].fill_(
                    float(i + 1)
                )  # Fill with distinct values
                dynamic_context.mamba_ssm_states[:, i, :, :, :].fill_(float(i + 1))
                dynamic_context.mamba_metadata.request_to_mamba_state_idx[i] = i
                dynamic_context.mamba_metadata.mamba_state_free_slot_count -= 1

        # Create an active_requests_mask where requests 0, 2, and 4 are finished (0),
        # and requests 1 and 3 are still active (1)
        active_requests_mask = torch.tensor([0, 1, 0, 1, 0], device=torch.cuda.current_device())

        # Call update_requests with these parameters
        dynamic_context.update_requests(
            active_requests_mask=active_requests_mask,
            new_tokens=torch.tensor([10, 11, 12, 13, 14], device=torch.cuda.current_device()),
        )

        # After the update, we should have released 3 blocks (for requests 0, 2, and 4)
        # and have 2 active requests (1 and 3)
        assert dynamic_context.total_request_count == 2
        assert dynamic_context.active_token_count == 2

        # Verify that 3 blocks were released by checking the available blocks
        assert dynamic_context.kv_block_allocator.total_avail == initial_available_blocks + 3

        if is_hybrid_model:
            # Request at position 3 now moves into finished request position 0
            # Request at position 1 remains active
            mamba_idx = {
                i: dynamic_context.mamba_metadata.request_to_mamba_state_idx[i] for i in range(5)
            }
            assert torch.all(dynamic_context.mamba_conv_states[:, mamba_idx[0], :, :] == 4.0)
            assert torch.all(dynamic_context.mamba_ssm_states[:, mamba_idx[0], :, :, :] == 4.0)
            assert torch.all(dynamic_context.mamba_conv_states[:, mamba_idx[1], :, :] == 2.0)
            assert torch.all(dynamic_context.mamba_ssm_states[:, mamba_idx[1], :, :, :] == 2.0)
            assert mamba_idx[2] == -1
            assert mamba_idx[3] == -1
            assert mamba_idx[4] == -1

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_finished_requests_with_multiple_blocks(self, is_hybrid_model):
        """Test that all memory blocks are correctly released for finished requests that use multiple blocks."""

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            block_size_tokens=128,
            max_tokens=None,
            is_hybrid_model=is_hybrid_model,
        )

        # Set up the initial state with 3 requests, where some use multiple blocks
        # Allocate 6 blocks in total for the requests
        initial_blocks = dynamic_context.kv_block_allocator.allocate_memory_blocks(6)
        dynamic_context.total_request_count = 3
        dynamic_context.paused_request_count = 0

        # Record the available blocks before releasing memory
        initial_available_blocks = dynamic_context.kv_block_allocator.total_avail

        # Assign blocks to the requests:
        # - Request 0: 1 block
        # - Request 1: 2 blocks
        # - Request 2: 3 blocks
        dynamic_context.request_to_kv_block_ids[0, 0] = initial_blocks[0]

        dynamic_context.request_to_kv_block_ids[1, 0] = initial_blocks[1]
        dynamic_context.request_to_kv_block_ids[1, 1] = initial_blocks[2]

        dynamic_context.request_to_kv_block_ids[2, 0] = initial_blocks[3]
        dynamic_context.request_to_kv_block_ids[2, 1] = initial_blocks[4]
        dynamic_context.request_to_kv_block_ids[2, 2] = initial_blocks[5]

        dynamic_context.request_kv_block_counts[0] = 1
        dynamic_context.request_kv_block_counts[1] = 2
        dynamic_context.request_kv_block_counts[2] = 3

        for i in range(3):
            dynamic_context.request_query_lengths[i] = 1
            dynamic_context.request_ids[i] = i
            dynamic_context.request_last_kv_block_id[i] = dynamic_context.request_to_kv_block_ids[
                i, dynamic_context.request_kv_block_counts[i] - 1
            ]
            dynamic_context.request_last_kv_block_offset[i] = 0
            dynamic_context.request_in_prefill_status_tensor[i] = 0
            if is_hybrid_model:
                dynamic_context.mamba_conv_states[:, i, :, :].fill_(float(i + 1))
                dynamic_context.mamba_ssm_states[:, i, :, :, :].fill_(float(i + 1))
                dynamic_context.mamba_metadata.request_to_mamba_state_idx[i] = i
                dynamic_context.mamba_metadata.mamba_state_free_slot_count -= 1

        # Create an active_requests_mask where all requests are finished
        active_requests_mask = torch.tensor([0, 0, 0], device=torch.cuda.current_device())

        # Call update_requests with these parameters
        dynamic_context.update_requests(
            active_requests_mask=active_requests_mask,
            new_tokens=torch.tensor([10, 11, 12], device=torch.cuda.current_device()),
        )

        # After the update, we should have released all 6 blocks and have 0 active requests
        assert dynamic_context.total_request_count == 0
        assert dynamic_context.active_token_count == 0

        # Verify that all 6 blocks were released by checking the available blocks
        assert dynamic_context.kv_block_allocator.total_avail == initial_available_blocks + 6

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_mamba_states_cache(self, is_hybrid_model: bool):

        if not is_hybrid_model:
            # If not hybrid, mamba_states_cache should fail
            dynamic_context = self._get_dynamic_context(
                params_dtype=torch.float32,
                num_layers=4,
                kv_channels=8,
                num_attention_heads=2,
                max_sequence_length=512,
                buffer_size_gb=0.03,
                block_size_tokens=128,
                max_tokens=None,
                is_hybrid_model=False,
            )
            with pytest.raises(AssertionError) as error:
                conv_state, ssm_state = dynamic_context.mamba_states_cache(layer_number=1)
            return

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            block_size_tokens=128,
            max_tokens=None,
            is_hybrid_model=is_hybrid_model,
            layer_type_list=[Symbols.MAMBA, Symbols.ATTENTION, Symbols.MAMBA, Symbols.ATTENTION],
        )

        # Add a request to populate states
        context_length = 10
        dynamic_context.add_request(
            DynamicInferenceRequest(
                request_id=0,
                prompt_tokens=torch.arange(0, context_length, dtype=torch.long, device='cuda'),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=dynamic_context.max_tokens - 10
                ),
            )
        )
        dynamic_context.initialize_attention_state()

        # Manually set some dummy values in mamba_conv_states and mamba_ssm_states
        # Mamba layers are at global indices 0 and 2 (mapped to local 0 and 1 via layer_map)
        # `layer_map` will map global layer index to the corresponding Mamba/Attention index.
        # For layer_type_list ["MAMBA", "ATTENTION", "MAMBA", "ATTENTION"],
        # global layer 1 (index 0) is MAMBA -> local mamba layer 0
        # global layer 3 (index 2) is MAMBA -> local mamba layer 1

        # Test for the first Mamba layer (global layer 1, local mamba layer 0)
        global_layer_1_mamba_local_idx = 0
        dynamic_context.mamba_conv_states[global_layer_1_mamba_local_idx] = 10.0
        dynamic_context.mamba_ssm_states[global_layer_1_mamba_local_idx] = 20.0

        # Test for the second Mamba layer (global layer 3, local mamba layer 1)
        global_layer_3_mamba_local_idx = 1
        dynamic_context.mamba_conv_states[global_layer_3_mamba_local_idx] = 30.0
        dynamic_context.mamba_ssm_states[global_layer_3_mamba_local_idx] = 40.0

        # Retrieve states using mamba_states_cache for global layer 1
        conv_state_layer1, ssm_state_layer1 = dynamic_context.mamba_states_cache(layer_number=1)
        assert torch.all(conv_state_layer1 == 10.0)
        assert torch.all(ssm_state_layer1 == 20.0)

        # Retrieve states using mamba_states_cache for global layer 3
        conv_state_layer3, ssm_state_layer3 = dynamic_context.mamba_states_cache(layer_number=3)
        assert torch.all(conv_state_layer3 == 30.0)
        assert torch.all(ssm_state_layer3 == 40.0)

    @pytest.mark.internal
    @rounder_override(64)
    def test_calculate_and_store_log_probs(self):

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            block_size_tokens=128,
            max_tokens=None,
        )

        # Add a few requests to the context
        request_data = {
            1001: {
                "tokens": torch.randint(0, 100, (10,), device='cuda'),
                "prefill_len": 10,
                "initial_token_offset": 0,
            },
            1002: {
                "tokens": torch.randint(0, 100, (5,), device='cuda'),
                "prefill_len": 5,
                "initial_token_offset": 10,
            },
            1003: {
                "tokens": torch.randint(0, 100, (7,), device='cuda'),
                "prefill_len": 7,
                "initial_token_offset": 15,
            },
        }

        current_token_idx = 0
        for req_id, data in request_data.items():
            dynamic_context.add_request(
                DynamicInferenceRequest(
                    request_id=req_id,
                    prompt_tokens=data["tokens"],
                    sampling_params=SamplingParams(
                        num_tokens_to_generate=dynamic_context.max_tokens - len(data["tokens"])
                    ),
                )
            )
            # Update the initial_token_offset as requests are added
            request_data[req_id]["initial_token_offset"] = current_token_idx
            current_token_idx += data["prefill_len"]

        # Simulate prefill step
        total_active_tokens = dynamic_context.active_token_count
        vocab_size = 50000
        # logits will have shape [1, total_active_tokens, vocab_size]
        prefill_logits = torch.randn(
            1, total_active_tokens, vocab_size, device='cuda', dtype=torch.float32
        )

        # New tokens from prefill (one token per active request)
        num_active_requests = (
            dynamic_context.total_request_count - dynamic_context.paused_request_count
        )
        prefill_new_tokens = torch.randint(0, 100, (num_active_requests,), device='cuda').long()

        # Call the function for prefill
        prefill_log_probs, _ = dynamic_context.calculate_log_probs(
            prefill_logits, prefill_new_tokens
        )

        # Calculate expected prefill log probs for the selected tokens
        expected_prefill_log_probs = (
            torch.nn.functional.log_softmax(prefill_logits.squeeze(0), dim=-1)
            .to(torch.float32)
            .cpu()
        )

        for i, (req_id, data) in enumerate(request_data.items()):
            req_len = data["tokens"].shape[0]
            initial_token_offset = data["initial_token_offset"]

            assert len(prefill_log_probs[i]) == req_len, len(prefill_log_probs[i])

            # Get the prompt tokens for this request and add the new sampled token
            request_tokens = data["tokens"][1:].tolist()
            request_tokens.append(prefill_new_tokens[i].item())

            for j, token in enumerate(request_tokens):
                assert (
                    prefill_log_probs[i][j]
                    == expected_prefill_log_probs[initial_token_offset + j, token].item()
                )

        # Simulate decode step
        # All requests are active, so the mask will be all ones for the current active requests
        active_requests_mask = torch.ones(dynamic_context.total_request_count, device='cuda').int()

        dynamic_context.update_requests(
            active_requests_mask=active_requests_mask, new_tokens=prefill_new_tokens
        )

        # Generate new logits for the decode step. Now each request contributes 1 token.
        decode_logits = torch.randn(
            1, num_active_requests, vocab_size, device='cuda', dtype=torch.float32
        )
        decode_new_tokens = torch.randint(0, 100, (num_active_requests,), device='cuda').long()
        decode_log_probs, _ = dynamic_context.calculate_log_probs(decode_logits, decode_new_tokens)

        # Verify the stored decode log probabilities
        expected_decode_log_probs = torch.nn.functional.log_softmax(
            decode_logits.squeeze(0), dim=-1
        ).to(torch.float32)

        for i, (req_id, data) in enumerate(request_data.items()):
            assert len(decode_log_probs[i]) == 1, len(decode_log_probs[i])

            token = decode_new_tokens[i].item()
            assert decode_log_probs[i][0] == expected_decode_log_probs[i, token].item()

        # Simulate mixed prefill and decode step (adding a new request to existing context)
        dynamic_context.update_requests(
            active_requests_mask=active_requests_mask, new_tokens=prefill_new_tokens
        )

        # Add a new prefill request to the existing context
        new_request_id = 1004
        new_request_tokens = torch.randint(0, 100, (12,), device='cuda').long()
        new_request_prefill_len = new_request_tokens.shape[0]
        initial_token_offset_new_request = dynamic_context.active_token_count
        dynamic_context.add_request(
            DynamicInferenceRequest(
                request_id=new_request_id,
                prompt_tokens=new_request_tokens,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=dynamic_context.max_tokens - len(new_request_tokens)
                ),
            )
        )
        request_data[new_request_id] = {
            "tokens": new_request_tokens,
            "prefill_len": new_request_prefill_len,
            "initial_token_offset": initial_token_offset_new_request,
        }

        # Simulate the step after adding the new prefill request.
        # This step will involve both prefill (for the new request) and decode (for existing requests).

        dynamic_context.initialize_attention_state()

        total_active_tokens_mixed_step = dynamic_context.active_token_count
        mixed_step_logits = torch.randn(
            1, total_active_tokens_mixed_step, vocab_size, device='cuda', dtype=torch.float32
        )

        num_active_requests_mixed_step = (
            dynamic_context.total_request_count - dynamic_context.paused_request_count
        )
        mixed_step_new_tokens = torch.randint(
            0, 100, (num_active_requests_mixed_step,), device='cuda'
        ).long()

        mixed_step_log_probs, _ = dynamic_context.calculate_log_probs(
            mixed_step_logits, mixed_step_new_tokens
        )

        expected_mixed_step_log_probs = (
            torch.nn.functional.log_softmax(mixed_step_logits.squeeze(0), dim=-1)
            .to(torch.float32)
            .cpu()
        )

        # Verify log probs for the mixed step
        current_global_token_offset = 0
        for i, (req_id, data) in enumerate(request_data.items()):

            # This logic needs to consider if the request was new (prefill) or existing (decode)
            if req_id == new_request_id:
                # This is the newly added prefill request
                expected_len = data["prefill_len"]
                assert len(mixed_step_log_probs[i]) == expected_len

                # For prefill, the log probs are for tokens[1:] + new_token
                prompt_tokens = data["tokens"][1:].tolist()
                new_sampled_token = mixed_step_new_tokens[i].item()

                for j in range(expected_len - 1):
                    # For prompt tokens
                    assert (
                        mixed_step_log_probs[i][j]
                        == expected_mixed_step_log_probs[
                            current_global_token_offset + j, prompt_tokens[j]
                        ].item()
                    )

                # For the newly sampled token
                assert (
                    mixed_step_log_probs[i][expected_len - 1]
                    == expected_mixed_step_log_probs[
                        current_global_token_offset + expected_len - 1, new_sampled_token
                    ].item()
                )

                current_global_token_offset += expected_len

            else:
                # These are existing requests, now in decode phase
                expected_len = 1
                assert len(mixed_step_log_probs[i]) == expected_len

                # For decode, the log prob is for the single new token
                new_sampled_token = mixed_step_new_tokens[i].item()
                assert (
                    mixed_step_log_probs[i][0]
                    == expected_mixed_step_log_probs[
                        current_global_token_offset, new_sampled_token
                    ].item()
                )

                current_global_token_offset += expected_len

    @pytest.mark.internal
    @rounder_override(64)
    def test_pipeline_parallel_uneven_layers(self):
        """
        Test that DynamicInferenceContext synchronizes the total block count across
        pipeline stages when they have unequal layer counts.
        """
        pp_size = 2
        self._setup_model_parallel_group(tensor_parallel_size=1, pipeline_parallel_size=pp_size)

        rank = parallel_state.get_pipeline_model_parallel_rank()

        mamba_conv_states_shape = (544, 4)
        mamba_ssm_states_shape = (8, 64, 16)
        params_dtype = torch.float32

        if rank == 0:
            mamba_inference_state_config = MambaInferenceStateConfig(
                [Symbols.MAMBA] + [Symbols.ATTENTION] * 4,
                mamba_conv_states_shape,
                mamba_ssm_states_shape,
                params_dtype,
                params_dtype,
            )
        else:
            mamba_inference_state_config = MambaInferenceStateConfig(
                [Symbols.MAMBA] * 4 + [Symbols.ATTENTION],
                mamba_conv_states_shape,
                mamba_ssm_states_shape,
                params_dtype,
                params_dtype,
            )

        context = DynamicInferenceContext(
            model_config=TransformerConfig(
                params_dtype=params_dtype,
                num_layers=10,
                kv_channels=64,
                num_attention_heads=8,
                pipeline_model_parallel_size=pp_size,
                tensor_model_parallel_size=1,
                pipeline_dtype=params_dtype,
            ),
            inference_config=InferenceConfig(
                max_sequence_length=128,
                buffer_size_gb=0.1,
                block_size_tokens=16,
                max_tokens=1024,
                unified_memory_level=0,
            ),
        )

        # Collect the total block counts on each rank
        local_total_blocks = torch.tensor(
            [context.kv_block_allocator.total_count], device='cuda', dtype=torch.long
        )
        gathered_block_counts = [torch.zeros_like(local_total_blocks) for _ in range(pp_size)]
        torch.distributed.all_gather(
            gathered_block_counts,
            local_total_blocks,
            group=parallel_state.get_pipeline_model_parallel_group(),
        )
        all_counts = [t.item() for t in gathered_block_counts]

        # Verify that there is only 1 unique value across all ranks
        unique_counts = set(all_counts)
        assert (
            len(unique_counts) == 1
        ), f"Block counts were not synchronized across ranks. Gathered: {all_counts}"

        self._restore_model_parallel()

    @pytest.mark.internal
    @pytest.mark.parametrize("ratio", [0.2, 0.4, 0.6, 0.8])
    @rounder_override(64)
    def test_mamba_memory_ratio_allocation(self, ratio):
        """
        Test that max_requests and block counts are partitioned correctly by mamba_memory_ratio.
        """

        buffer_gb = 0.05
        paused_gb = 0.01
        block_size = 256
        num_attention_heads = 8
        kv_channels = 64
        params_dtype = torch.float32

        layer_type_list = [Symbols.MAMBA, Symbols.ATTENTION]
        mamba_conv_states_shape = (544, 4)
        mamba_ssm_states_shape = (8, 64, 16)
        mamba_config = MambaInferenceStateConfig(
            layer_type_list,
            mamba_conv_states_shape,
            mamba_ssm_states_shape,
            params_dtype,
            params_dtype,
        )

        context = DynamicInferenceContext(
            model_config=TransformerConfig(
                params_dtype=params_dtype,
                num_layers=2,  # 1 Attn, 1 Mamba
                kv_channels=kv_channels,
                num_attention_heads=num_attention_heads,
            ),
            inference_config=InferenceConfig(
                max_sequence_length=512,
                buffer_size_gb=buffer_gb,
                paused_buffer_size_gb=paused_gb,
                block_size_tokens=block_size,
                max_tokens=2048,
                mamba_inference_state_config=mamba_config,
                mamba_memory_ratio=ratio,
                unified_memory_level=0,
            ),
        )

        dtype_size = torch.tensor([], dtype=params_dtype).element_size()

        mamba_mem_per_req = math.prod(mamba_conv_states_shape) + math.prod(mamba_ssm_states_shape)
        mamba_mem_per_req *= dtype_size

        kv_buffer_bytes = int(buffer_gb * 1024**3)
        kv_paused_bytes = int(paused_gb * 1024**3)
        total_mem_bytes = kv_buffer_bytes + kv_paused_bytes
        expected_mamba_mem = total_mem_bytes * ratio
        expected_mamba_max_reqs = int(expected_mamba_mem // mamba_mem_per_req)

        # KV block calculation with buffer size reduced by Mamba memory ratio
        kv_buffer_bytes = int(kv_buffer_bytes * (1.0 - ratio))
        kv_paused_bytes = int(kv_paused_bytes * (1.0 - ratio))

        kv_block_size_bytes = dtype_size * 2 * 1 * block_size * num_attention_heads * kv_channels

        expected_active_blocks = kv_buffer_bytes // kv_block_size_bytes
        expected_paused_blocks = kv_paused_bytes // kv_block_size_bytes
        expected_total_blocks = expected_active_blocks + expected_paused_blocks

        # Check that block allocator received the reduced block counts
        assert context.kv_block_allocator.total_count == expected_active_blocks
        assert context.kv_block_allocator.paused_count == expected_paused_blocks

        # max_requests should be limited by the Mamba calculation if mamba_max_requests is smaller
        # or the block count - 1 if that is smaller
        expected_limit = min(expected_total_blocks - 1, expected_mamba_max_reqs)

        # Apply rounding (REQUEST_ROUNDER = 64 in this test)
        expected_max_requests = (expected_limit // 64) * 64

        assert context.max_requests == expected_max_requests
        assert context.is_hybrid_model is True

    @pytest.mark.internal
    @rounder_override(64)
    def test_max_requests_less_than_tp_size(self):
        tp_size = 2
        self._setup_model_parallel_group(tensor_parallel_size=tp_size, pipeline_parallel_size=1)

        model_config = TransformerConfig(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=8,
            tensor_model_parallel_size=tp_size,
        )

        inference_config = InferenceConfig(
            max_sequence_length=512, buffer_size_gb=0.1, block_size_tokens=128, max_requests=1
        )

        with pytest.raises(AssertionError):
            DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        self._restore_model_parallel()

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    @pytest.mark.parametrize("num_cuda_graphs", [-1, 16, 32])
    @pytest.mark.parametrize("num_speculative_tokens", [0, 3])
    def test_add_dummy_requests_for_expert_parallel_step_matches_slow_path(
        self, is_hybrid_model: bool, num_cuda_graphs: int, num_speculative_tokens: int
    ):
        """The fast path (add_dummy_requests_for_expert_parallel_step) must leave
        the same observable state as the slow path
        (add_dummy_requests_for_cudagraph_capture(min(cuda_graph_dims))).
        """

        ctx = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            block_size_tokens=128,
            max_tokens=None,
            is_hybrid_model=is_hybrid_model,
            layer_type_list=(
                [Symbols.MAMBA, Symbols.ATTENTION, Symbols.MLP, Symbols.ATTENTION]
                if is_hybrid_model
                else None
            ),
            num_cuda_graphs=num_cuda_graphs,
            num_speculative_tokens=num_speculative_tokens,
        )

        smallest = min(ctx.cuda_graph_batch_dimensions_list)
        N = smallest.decode_req_count
        T = smallest.token_count  # N * (num_speculative_tokens + 1)
        assert smallest.prefill_req_count == 0, "smallest graph must be decode-only"

        # --- slow path (reference) ---
        ctx.add_dummy_requests_for_cudagraph_capture(smallest)

        slow_total_request_count = ctx.total_request_count
        slow_active_token_count = ctx.active_token_count
        slow_num_prefill_requests = ctx.num_prefill_requests
        slow_request_query_lengths = ctx.request_query_lengths[:N].clone()
        slow_request_kv_length_offsets = ctx.request_kv_length_offsets[:N].clone()
        slow_request_to_kv_block_ids_col0 = ctx.request_to_kv_block_ids[:N, 0].clone()
        slow_token_to_block_idx = ctx.token_to_block_idx[:T].clone()
        slow_token_to_local_pos = ctx.token_to_local_position_within_kv_block[:T].clone()
        if is_hybrid_model:
            slow_token_to_request_idx = ctx.token_to_request_idx[:T].clone()
            slow_mamba = ctx.mamba_metadata.request_to_mamba_state_idx[:N].clone()

        # --- reset and run fast path ---
        ctx.reset()
        ctx.add_dummy_requests_for_expert_parallel_step()

        # 1. Scalar counts
        assert ctx.total_request_count == slow_total_request_count
        assert ctx.active_token_count == slow_active_token_count
        assert ctx.num_prefill_requests == slow_num_prefill_requests

        # 2. Per-request MHA state
        assert torch.equal(ctx.request_query_lengths[:N], slow_request_query_lengths)
        assert torch.equal(ctx.request_kv_length_offsets[:N], slow_request_kv_length_offsets)
        assert torch.equal(ctx.request_to_kv_block_ids[:N, 0], slow_request_to_kv_block_ids_col0)

        # 3. Token-level state
        dummy_block_idx = ctx.kv_block_allocator.dummy_block_idx
        assert torch.all(ctx.token_to_block_idx[:T] == dummy_block_idx)
        assert torch.equal(ctx.token_to_block_idx[:T], slow_token_to_block_idx)
        assert torch.equal(ctx.token_to_local_position_within_kv_block[:T], slow_token_to_local_pos)

        if is_hybrid_model:
            # 4. token_to_request_idx
            assert torch.equal(ctx.token_to_request_idx[:T], slow_token_to_request_idx)

            # 5. Mamba state slots allocated (indices may differ, but must be valid and unique)
            fast_mamba = ctx.mamba_metadata.request_to_mamba_state_idx[:N]
            assert (fast_mamba >= 0).all(), "fast path should allocate valid mamba slots"
            assert (slow_mamba >= 0).all(), "slow path should allocate valid mamba slots"
            assert fast_mamba.unique().numel() == N, "fast path mamba slots must be unique"

    @pytest.mark.internal
    def test_gqa_high_tp_partition_heads(self):
        """Tests that TP > GQA results in 1 attention head per partition."""
        tp_size = 8
        self._setup_model_parallel_group(tensor_parallel_size=tp_size, pipeline_parallel_size=1)

        model_config = TransformerConfig(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=16,
            num_query_groups=2,  # GQA = 2
            tensor_model_parallel_size=tp_size,
        )

        # max_requests must be divisible by TP size (8) and REQUEST_ROUNDER
        inference_config = InferenceConfig(
            max_sequence_length=512, buffer_size_gb=0.1, block_size_tokens=128, max_requests=8
        )

        dynamic_context = DynamicInferenceContext(
            model_config=model_config, inference_config=inference_config
        )

        # With TP=8 and GQA=2, num_attention_heads_per_partition should be clamped to 1
        assert dynamic_context.num_attention_heads_per_partition == 1

        self._restore_model_parallel()

    @pytest.mark.internal
    @rounder_override(64)
    def test_chunked_prefill_state_preserved_across_decode_completions(self):
        """
        Tests that when a chunked prefill request is hidden, and active decode requests
        finish (causing the context boundary to shrink), the hidden chunked request
        is safely pulled to the new boundary so it doesn't lose its KV blocks or Mamba slot.
        """

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=128,
            buffer_size_gb=0.03,
            block_size_tokens=4,
            max_tokens=None,
            is_hybrid_model=True,
            layer_type_list=[Symbols.MAMBA, Symbols.ATTENTION],
            enable_chunked_prefill=True,
        )

        # Add 2 normal decode requests
        dynamic_context.add_request(
            DynamicInferenceRequest(
                request_id=10,
                prompt_tokens=torch.arange(0, 2, device='cuda'),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
            )
        )
        dynamic_context.add_request(
            DynamicInferenceRequest(
                request_id=11,
                prompt_tokens=torch.arange(0, 2, device='cuda'),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
            )
        )

        # Add Chunk 1 of the chunked prefill request
        req_999 = DynamicInferenceRequest(
            request_id=999,
            prompt_tokens=torch.arange(0, 8, device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(req_999, prefill_chunk_length=4)
        dynamic_context.chunked_prefill_request_id = 999

        # Capture the allocated state at index 2
        mamba_slot_before = dynamic_context.mamba_metadata.request_to_mamba_state_idx[2].item()
        kv_block_before = dynamic_context.request_to_kv_block_ids[2, 0].item()

        assert mamba_slot_before != -1
        assert kv_block_before != -1

        # Step 1: Forward pass for all 3 requests
        active_requests_mask = torch.tensor([1, 1, 1], dtype=torch.int32, device='cuda')
        new_tokens = torch.tensor([100, 101, 102], dtype=torch.int32, device='cuda')
        dynamic_context.update_requests(active_requests_mask, new_tokens)

        # At this point, req 999 is hidden at index 2. total_request_count is 2 (req 10, 11).
        assert dynamic_context.total_request_count == 2
        assert dynamic_context.request_ids[2].item() == 999

        # Step 2: Forward pass where req 10 finishes, req 11 continues. Req 999 is NOT scheduled.
        active_requests_mask = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
        new_tokens = torch.tensor([103, 104], dtype=torch.int32, device='cuda')
        dynamic_context.update_requests(active_requests_mask, new_tokens)

        # At this point, req 10 is evicted. Req 11 shifts to index 0. total_request_count becomes 1.
        # Req 999 should be pulled from index 2 down to index 1 (the new boundary).
        assert dynamic_context.total_request_count == 1
        assert dynamic_context.request_ids[0].item() == 11

        # Verify that the chunked request was correctly pulled to the boundary (index 1)
        assert dynamic_context.request_ids[1].item() == 999
        assert (
            dynamic_context.mamba_metadata.request_to_mamba_state_idx[1].item() == mamba_slot_before
        )
        assert dynamic_context.request_to_kv_block_ids[1, 0].item() == kv_block_before

        # Ensure the old index 2 was properly swapped during the pull
        assert dynamic_context.request_ids[2].item() == 11
        assert dynamic_context.mamba_metadata.request_to_mamba_state_idx[2].item() == -1
        assert dynamic_context.request_to_kv_block_ids[2, 0].item() == -1

        # Step 3: Add the next chunk. It should sit exactly at the boundary (index 1) and inherit the state.
        req_999.finished_chunk_token_count = 4
        dynamic_context.add_request(req_999, prefill_chunk_length=4)

        # Verify state at index 1 is active and its previous Mamba slot and KV blocks were inherited
        assert dynamic_context.total_request_count == 2
        assert dynamic_context.request_ids[1].item() == 999
        assert (
            dynamic_context.mamba_metadata.request_to_mamba_state_idx[1].item() == mamba_slot_before
        )
        assert dynamic_context.request_to_kv_block_ids[1, 0].item() == kv_block_before

    @pytest.mark.internal
    @rounder_override(4)
    def test_chunked_prefill_all_active_requests_finish_while_hidden(self):
        """
        Tests that update_requests does not crash when ALL active decode requests
        finish while a chunked prefill request is hidden (not scheduled this step).

        This exercises the scenario where:
        1. A chunked prefill completes a chunk and is hidden at total_request_count
        2. The next chunk is not scheduled (e.g., no token budget)
        3. All remaining active decode requests finish in the same step
        4. active_request_count becomes 0 — the code must not assert-fail
        """

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=128,
            buffer_size_gb=0.03,
            block_size_tokens=4,
            max_tokens=None,
            enable_chunked_prefill=True,
            max_requests=16,
        )

        # Add 2 normal decode requests
        dynamic_context.add_request(
            DynamicInferenceRequest(
                request_id=10,
                prompt_tokens=torch.arange(0, 2, device='cuda'),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
            )
        )
        dynamic_context.add_request(
            DynamicInferenceRequest(
                request_id=11,
                prompt_tokens=torch.arange(0, 2, device='cuda'),
                sampling_params=SamplingParams(num_tokens_to_generate=10),
            )
        )

        # Add Chunk 1 of a chunked prefill request
        req_999 = DynamicInferenceRequest(
            request_id=999,
            prompt_tokens=torch.arange(0, 8, device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        dynamic_context.add_request(req_999, prefill_chunk_length=4)
        dynamic_context.chunked_prefill_request_id = 999

        kv_block_before = dynamic_context.request_to_kv_block_ids[2, 0].item()
        assert kv_block_before != -1

        # Step 1: All 3 requests are active, process forward pass
        active_requests_mask = torch.tensor([1, 1, 1], dtype=torch.int32, device='cuda')
        new_tokens = torch.tensor([100, 101, 102], dtype=torch.int32, device='cuda')
        dynamic_context.update_requests(active_requests_mask, new_tokens)

        # Chunked prefill is now hidden at position 2, total_request_count = 2
        assert dynamic_context.total_request_count == 2
        assert dynamic_context.request_ids[2].item() == 999

        # Step 2: Both decode requests finish, chunked prefill NOT scheduled this step.
        # This must NOT crash even though active_request_count becomes 0.
        active_requests_mask = torch.tensor([0, 0], dtype=torch.int32, device='cuda')
        new_tokens = torch.tensor([103, 104], dtype=torch.int32, device='cuda')
        dynamic_context.update_requests(active_requests_mask, new_tokens)

        # total_request_count should be 0 (both finished, chunked prefill hidden)
        assert dynamic_context.total_request_count == 0
        assert dynamic_context.active_token_count == 0

        # The hidden chunked prefill should be pulled to position 0 (the new boundary)
        assert dynamic_context.request_ids[0].item() == 999
        assert dynamic_context.request_to_kv_block_ids[0, 0].item() == kv_block_before

        # Verify we can still add the next chunk at position 0
        req_999.finished_chunk_token_count = 4
        dynamic_context.add_request(req_999, prefill_chunk_length=4)

        assert dynamic_context.total_request_count == 1
        assert dynamic_context.request_ids[0].item() == 999
        assert dynamic_context.request_to_kv_block_ids[0, 0].item() == kv_block_before

    @pytest.mark.internal
    @rounder_override(64)
    def test_update_requests_speculative(self):
        """Test update_requests correctly interleaves sampled and speculative tokens."""

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=128,
            buffer_size_gb=0.01,
            block_size_tokens=256,
            num_speculative_tokens=2,
            unified_memory_level=0,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        # Setup 2 active decode requests
        ctx.total_request_count = 2
        ctx.paused_request_count = 0
        ctx.active_token_count = 2
        ctx.request_ids[:2] = torch.tensor([10, 11])
        ctx.request_query_lengths[:2] = 1
        ctx.request_kv_length_offsets[:2] = torch.tensor([5, 8])
        ctx.request_last_kv_block_offset[:2] = torch.tensor([5, 8])
        ctx.request_to_kv_block_ids[:2, 0] = torch.tensor([0, 1])
        ctx.request_last_kv_block_id[:2] = torch.tensor([0, 1])

        active_requests_mask = torch.tensor([1, 1], device='cuda')
        new_tokens = torch.tensor([99, 100], device='cuda')  # Sampled tokens
        new_speculative_tokens = torch.tensor(
            [[991, 1001], [992, 1002]], device='cuda'
        )  # Spec tokens

        ctx.update_requests(
            active_requests_mask=active_requests_mask,
            new_tokens=new_tokens,
            new_speculative_tokens=new_speculative_tokens,
        )

        # Each request generates 1 (sampled) + 2 (speculative) = 3 tokens.
        assert ctx.active_token_count == 6
        assert torch.equal(
            ctx.request_query_lengths[:2], torch.tensor([3, 3], dtype=torch.int32, device='cuda')
        )
        assert torch.equal(
            ctx.request_kv_length_offsets[:2],
            torch.tensor([6, 9], dtype=torch.int32, device='cuda'),
        )

        # Check interleaving: [sampled_1, spec1_1, spec2_1, sampled_2, spec1_2, spec2_2]
        expected_tokens = torch.tensor([99, 991, 992, 100, 1001, 1002], device='cuda')
        assert torch.equal(ctx.token_to_input_ids[:6], expected_tokens)

    @pytest.mark.internal
    @rounder_override(64)
    def test_speculative_boundary_crossing(self):
        """Test token block assignment when speculative tokens cross a KV block boundary."""

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=256,  # FA2-compatible block size to force boundary crossing
            num_speculative_tokens=2,
            unified_memory_level=0,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        # Setup 1 active decode request
        ctx.total_request_count = 1
        ctx.paused_request_count = 0
        ctx.active_token_count = 1

        ctx.request_ids[0] = 10
        ctx.request_query_lengths[0] = 1
        ctx.request_kv_block_counts[0] = 1

        # Length is 254, meaning existing tokens are at indices 0..253.
        # The last inserted token was at offset 253.
        # Adding 3 tokens places them at offsets 254, 255, and 256 (crosses block size of 256).
        ctx.request_kv_length_offsets[0] = 254
        ctx.request_last_kv_block_offset[0] = 253

        # Allocate one initial block manually
        blocks = ctx.kv_block_allocator.allocate_memory_blocks(1)
        first_block = blocks[0]
        ctx.request_to_kv_block_ids[0, 0] = first_block
        ctx.request_last_kv_block_id[0] = first_block

        active_requests_mask = torch.tensor([1], device='cuda')
        new_tokens = torch.tensor([50], device='cuda')
        new_speculative_tokens = torch.tensor([[51], [52]], device='cuda')

        # Run update_requests natively. It will automatically:
        # 1. Detect the boundary crossing and pause the request.
        # 2. Clone the prev_last_block_ids internally.
        # 3. Resume the request, allocating the new block.
        # 4. Map the 3 new tokens across the boundary.
        ctx.update_requests(
            active_requests_mask=active_requests_mask,
            new_tokens=new_tokens,
            new_speculative_tokens=new_speculative_tokens,
        )

        # Verify a new block was natively allocated by the resume logic
        assert ctx.request_kv_block_counts[0] == 2
        second_block = ctx.request_to_kv_block_ids[0, 1]
        assert second_block != -1
        assert second_block != first_block

        # Expected token mapping for the 3 generated tokens (sampled, spec1, spec2)
        # Token 0 (offset 2) -> first_block
        # Token 1 (offset 3) -> first_block
        # Token 2 (offset 4) -> second_block
        expected_blocks = torch.tensor(
            [first_block, first_block, second_block], dtype=torch.int, device='cuda'
        )

        assert torch.equal(ctx.token_to_block_idx[:3], expected_blocks)

    @pytest.mark.internal
    @rounder_override(64)
    def test_paused_speculative_tokens_tracking(self):
        """
        Test that speculative tokens are correctly saved and concatenated
        when requests are temporarily paused.
        """

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=1024,
            buffer_size_gb=0.1,
            block_size_tokens=256,
            num_speculative_tokens=2,
            unified_memory_level=0,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        # Setup 2 active requests. Request 0 is about to overflow its block.
        ctx.total_request_count = 2
        ctx.paused_request_count = 0
        ctx.active_token_count = 2
        ctx.request_ids[:2] = torch.tensor([10, 11])
        ctx.request_query_lengths[:2] = 1

        # Request 0 is at offset 254. Adding 1 sampled + 2 spec = 3 tokens will push it to 257,
        # which is >= block_size_tokens (256). It will require a new block.
        # Request 1 is at offset 5. It will not require a new block.
        ctx.request_kv_length_offsets[:2] = torch.tensor([254, 5])
        ctx.request_last_kv_block_offset[:2] = torch.tensor([254, 5])
        ctx.request_kv_block_counts[:2] = 1

        # Allocate blocks
        blocks = ctx.kv_block_allocator.allocate_memory_blocks(2)
        ctx.request_to_kv_block_ids[0, 0] = blocks[0]
        ctx.request_to_kv_block_ids[1, 0] = blocks[1]
        ctx.request_last_kv_block_id[:2] = blocks

        # Force the allocator to have no available blocks.
        # This guarantees request 0 stays paused and cannot immediately resume.
        ctx.kv_block_allocator.total_avail = 0
        ctx.kv_block_allocator.paused_count = 100  # Ensure it doesn't get completely evicted either

        active_requests_mask = torch.tensor([1, 1], device='cuda')
        new_tokens = torch.tensor([99, 100], device='cuda')  # Sampled
        new_speculative_tokens = torch.tensor(
            [[991, 1001], [992, 1002]], device='cuda'
        )  # Speculative

        # In update_requests, request 0 will be paused to allocate a new block.
        # Since total_avail is 0, it will stay paused and its tokens will be cached.
        ctx.update_requests(
            active_requests_mask=active_requests_mask,
            new_tokens=new_tokens,
            new_speculative_tokens=new_speculative_tokens,
        )

        # Verify paused state was populated correctly
        assert ctx.paused_tokens is not None
        assert ctx.paused_speculative_tokens is not None

        # Request 0 was the one paused, so its tokens should be shifted to
        # index 0 of the paused tensors.
        assert ctx.paused_request_count == 1
        assert ctx.total_request_count == 2

        assert ctx.paused_tokens[0].item() == 99
        assert torch.equal(
            ctx.paused_speculative_tokens[:, 0], torch.tensor([991, 992], device='cuda')
        )

    @pytest.mark.internal
    @rounder_override(64)
    def test_speculative_tokens_less_than_block_size_assert(self):

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=128,
            buffer_size_gb=0.01,
            block_size_tokens=256,
            num_speculative_tokens=256,
            unified_memory_level=0,
        )
        with pytest.raises(
            AssertionError, match="num_speculative_tokens.*must be < block_size_tokens"
        ):
            DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

    @pytest.mark.internal
    @rounder_override(64)
    def test_swap_book_keeping_tensors_with_speculative_tokens(self):

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=128,
            buffer_size_gb=0.01,
            block_size_tokens=256,
            num_speculative_tokens=2,
            unified_memory_level=0,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        ctx.request_ids[:2] = torch.tensor([10, 11])
        next_tokens = torch.tensor([99, 100], device='cuda')
        new_speculative_tokens = torch.tensor([[991, 1001], [992, 1002]], device='cuda')

        ctx._swap_book_keeping_tensors(
            src_idxs=torch.tensor([0]),
            dst_idxs=torch.tensor([1]),
            next_tokens=next_tokens,
            new_speculative_tokens=new_speculative_tokens,
        )

        assert torch.equal(ctx.request_ids[:2], torch.tensor([11, 10], device='cuda'))
        assert torch.equal(next_tokens[:2], torch.tensor([100, 99], device='cuda'))
        assert torch.equal(
            new_speculative_tokens[:, :2], torch.tensor([[1001, 991], [1002, 992]], device='cuda')
        )

    @pytest.mark.internal
    @rounder_override(64)
    def test_update_requests_with_finished_requests_and_speculative_tokens(self):

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=128,
            buffer_size_gb=0.01,
            block_size_tokens=32,
            num_speculative_tokens=2,
            unified_memory_level=0,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        # Setup 3 active requests: req0 (active), req1 (finished), req2 (active)
        ctx.total_request_count = 3
        ctx.paused_request_count = 0
        ctx.active_token_count = 3
        ctx.request_ids[:3] = torch.tensor([10, 11, 12])
        ctx.request_query_lengths[:3] = 1
        ctx.request_kv_length_offsets[:3] = torch.tensor([5, 8, 12])
        ctx.request_last_kv_block_offset[:3] = torch.tensor([5, 8, 12])
        ctx.request_to_kv_block_ids[:3, 0] = torch.tensor([0, 1, 2])
        ctx.request_last_kv_block_id[:3] = torch.tensor([0, 1, 2])
        ctx.request_kv_block_counts[:3] = 1

        active_requests_mask = torch.tensor([1, 0, 1], device='cuda')
        new_tokens = torch.tensor([99, 100, 101], device='cuda')
        new_speculative_tokens = torch.tensor([[991, 1001, 1011], [992, 1002, 1012]], device='cuda')

        ctx.update_requests(
            active_requests_mask=active_requests_mask,
            new_tokens=new_tokens,
            new_speculative_tokens=new_speculative_tokens,
        )

        # req1 is finished. req2 moves to req1's position.
        assert ctx.total_request_count == 2
        assert torch.equal(
            ctx.request_ids[:2], torch.tensor([10, 12], device='cuda', dtype=torch.int32)
        )

        # Check interleaving for req0 and req2
        # req0: [99, 991, 992]
        # req2: [101, 1011, 1012]
        expected_tokens = torch.tensor([99, 991, 992, 101, 1011, 1012], device='cuda')
        assert torch.equal(ctx.token_to_input_ids[:6], expected_tokens)

    @pytest.mark.internal
    @rounder_override(64)
    def test_chunked_prefill_hidden_state_prevents_token_bloat(self):
        """
        Test that hiding the chunked prefill request effectively prevents
        'dummy' speculative tokens from bloating the active_token_count, and that the
        next chunk seamlessly appends without needing legacy offset subtractions.
        """

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.05,
            block_size_tokens=128,
            max_requests=256,
            max_tokens=256,
            num_speculative_tokens=3,
            enable_chunked_prefill=True,
            unified_memory_level=0,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)
        ctx.reset_tensors()

        # 1. Add a standard decode request
        req_decode = DynamicInferenceRequest(
            request_id=10,
            prompt_tokens=torch.arange(0, 10, device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        ctx.add_request(req_decode)

        # 2. Add chunk 1 of a chunked prefill request
        req_chunked = DynamicInferenceRequest(
            request_id=42,
            prompt_tokens=torch.arange(0, 100, device='cuda'),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        ctx.chunked_prefill_request_id = 42
        ctx.add_request(req_chunked, prefill_chunk_length=50)

        # Verify initial active token count (10 from decode + 50 from prefill)
        assert ctx.active_token_count == 60

        # 3. Call update_requests
        active_requests_mask = torch.tensor([1, 1], dtype=torch.int32, device='cuda')
        new_tokens = torch.tensor([99, 199], dtype=torch.int32, device='cuda')
        new_spec = torch.tensor(
            [[100, 200], [101, 201], [102, 202]], dtype=torch.int32, device='cuda'
        )

        ctx.update_requests(
            active_requests_mask=active_requests_mask,
            new_tokens=new_tokens,
            new_speculative_tokens=new_spec,
        )

        # 4. Verify Hiding Invariants:
        # The chunked prefill request should be hidden safely out of bounds.
        # The active_token_count should ONLY contain the decode request's tokens
        # (1 base + 3 speculative = 4 tokens).
        assert ctx.total_request_count == 1
        assert ctx.active_token_count == 4
        assert ctx.request_ids[1].item() == 42

        # 5. Add chunk 2
        req_chunked.finished_chunk_token_count = 50
        ctx.add_request(req_chunked, prefill_chunk_length=50)

        # 6. Verify seamless append (no legacy offset math needed)
        # 4 active decode tokens + 50 new prefill tokens = 54
        assert ctx.active_token_count == 54
        assert ctx.total_request_count == 2

    @pytest.mark.internal
    @rounder_override(64)
    def test_chunked_prefill_swap_with_speculative_tokens(self):
        """Test that swapping a chunked prefill request to the end of the buffer
        correctly brings along the 2D speculative tokens for the other decode requests.
        """

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=128,
            buffer_size_gb=0.01,
            block_size_tokens=32,
            num_speculative_tokens=2,
            enable_chunked_prefill=True,
            unified_memory_level=0,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        # Setup 2 active requests in the WRONG order (violating the invariant)
        # Index 0: Chunked Prefill Request (ID 42)
        # Index 1: Standard Decode Request (ID 99)
        ctx.total_request_count = 2
        ctx.paused_request_count = 0
        ctx.active_token_count = 2

        ctx.chunked_prefill_request_id = 42
        ctx.request_ids[:2] = torch.tensor([42, 99])

        # Status: 1 = Prefill, 0 = Decode
        ctx.request_in_prefill_status_tensor[:2] = torch.tensor([1, 0])
        ctx.request_query_lengths[:2] = 1
        ctx.request_kv_length_offsets[:2] = torch.tensor([10, 20])
        ctx.request_last_kv_block_offset[:2] = torch.tensor([10, 20])
        ctx.request_to_kv_block_ids[:2, 0] = torch.tensor([0, 1])
        ctx.request_last_kv_block_id[:2] = torch.tensor([0, 1])
        ctx.request_kv_block_counts[:2] = 1

        active_requests_mask = torch.tensor([1, 1], device='cuda')

        # New base tokens: [100 (for prefill), 200 (for decode)]
        new_tokens = torch.tensor([100, 200], device='cuda')

        # New spec tokens: Col 0 for prefill (dummy), Col 1 for decode (real draft tokens)
        new_speculative_tokens = torch.tensor([[101, 201], [102, 202]], device='cuda')

        # Trigger update_requests.
        # It must detect ID 42 is at index 0, and swap it with index 1.
        ctx.update_requests(
            active_requests_mask=active_requests_mask,
            new_tokens=new_tokens,
            new_speculative_tokens=new_speculative_tokens,
        )

        # 1. Verify the IDs were swapped successfully
        assert torch.equal(
            ctx.request_ids[:2], torch.tensor([99, 42], dtype=torch.int32, device='cuda')
        )

        # 2. Verify the Decode request (now at Index 0) correctly flattened its
        #    base token (200) AND its specific speculative tokens (201, 202).
        # 3. Verify the Prefill request (now at Index 1) is hidden and does NOT
        #    flatten its dummy tokens.
        expected_flattened_tokens = torch.tensor(
            [200, 201, 202], device='cuda'  # Decode request (ID 99)
        )

        assert ctx.active_token_count == 3
        assert torch.equal(
            ctx.token_to_input_ids[:3], expected_flattened_tokens
        ), "Speculative tokens were not correctly flattened for the decode request!"

        # 4. Verify that the new_speculative_tokens tensor itself was swapped so that
        # the hidden state perfectly preserves the alignment for subsequent steps.
        expected_swapped_spec_tokens = torch.tensor([[201, 101], [202, 102]], device='cuda')
        assert torch.equal(
            new_speculative_tokens, expected_swapped_spec_tokens
        ), "new_speculative_tokens was not swapped in-place alongside the request metadata!"

    @pytest.mark.internal
    @rounder_override(64)
    def test_speculative_with_prefix_caching_shared_blocks(self):
        """Test that prefix caching correctly shares blocks when speculative decoding is enabled."""

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            num_speculative_tokens=2,
            enable_prefix_caching=True,
            unified_memory_level=0,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        bs = ctx.block_size_tokens
        # Use bs * 3 + 5 tokens so the prompt extends past the last full block.
        # This avoids the single-token-chunk clamp (effective_prefill >= 2) and
        # verifies that the prefix skip actually works.
        tail = 5
        prompt = torch.arange(bs * 3 + tail, device='cuda')

        # First request registers blocks.
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req1)
        # 3 full blocks are prefix-cacheable; the 4th (partial) block is not.
        first_full_blocks = [ctx.request_to_kv_block_ids[0][i].item() for i in range(3)]
        avail_after_first = ctx.kv_block_allocator.total_avail

        # Second request with same prefix should share the 3 full blocks.
        req2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req2)
        second_full_blocks = [ctx.request_to_kv_block_ids[1][i].item() for i in range(3)]

        # The 3 full blocks should be shared (same IDs).
        assert first_full_blocks == second_full_blocks

        # Only 1 new block allocated for the partial tail of the second request.
        assert ctx.kv_block_allocator.total_avail == avail_after_first - 1

        # Ref counts on the shared full blocks should be 2.
        for bid in first_full_blocks:
            assert ctx.kv_block_allocator.block_ref_counts[bid].item() == 2

        # Second request should skip the 3 full cached blocks (96 tokens),
        # leaving only the trailing tokens as the query.
        assert ctx.request_query_lengths[1].item() == tail

    @pytest.mark.internal
    @rounder_override(64)
    def test_speculative_with_prefix_caching_kv_offset(self):
        """Test that KV offset accounts for prefix skip when spec decoding is enabled."""

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            num_speculative_tokens=3,
            enable_prefix_caching=True,
            unified_memory_level=0,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        bs = ctx.block_size_tokens
        # Use bs * 2 + 5 tokens so the prompt extends past the last full block,
        # avoiding the single-token-chunk clamp while still testing the skip.
        tail = 5
        prompt = torch.arange(bs * 2 + tail, device='cuda')

        # First request.
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req1)

        # Second request with same prefix: should have kv_offset = prefix_skip_tokens.
        req2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req2)

        # 2 full blocks match → prefix_skip = 2 * bs = 64, query_length = tail.
        expected_skip = 2 * bs
        assert ctx.request_kv_length_offsets[1].item() == expected_skip
        assert ctx.request_query_lengths[1].item() == tail

    @pytest.mark.internal
    @rounder_override(64)
    def test_speculative_update_then_release_with_prefix_caching(self):
        """Test that update_requests with spec tokens + block release respects ref counts."""

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=4,
            num_speculative_tokens=2,
            enable_prefix_caching=True,
            unified_memory_level=0,
            max_requests=512,
            max_tokens=512,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        bs = ctx.block_size_tokens
        prompt = torch.arange(bs * 2, device='cuda')

        # Two requests sharing the same prefix.
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req1)
        shared_blocks = [ctx.request_to_kv_block_ids[0][i].item() for i in range(2)]

        req2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req2)

        # Verify initial ref counts are 2.
        for bid in shared_blocks:
            assert ctx.kv_block_allocator.block_ref_counts[bid].item() == 2

        # Release one request. Ref counts should decrement to 1.
        ctx.release_memory_blocks_from_request_indexes(torch.tensor([0]))
        for bid in shared_blocks:
            assert ctx.kv_block_allocator.block_ref_counts[bid].item() == 1

        # Blocks should still be discoverable via hash map.
        for bid in shared_blocks:
            h = ctx.kv_block_allocator.block_hashes[bid].item()
            assert h in ctx.kv_block_allocator.kv_hash_to_block_id

    @pytest.mark.internal
    @rounder_override(64)
    def test_speculative_boundary_crossing_with_prefix_caching(self):
        """Test block boundary crossing from speculative tokens does not corrupt shared blocks."""

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=4,
            num_speculative_tokens=2,
            enable_prefix_caching=True,
            unified_memory_level=0,
            max_tokens=512,
            max_requests=512,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        bs = ctx.block_size_tokens
        prompt = torch.arange(bs * 2, device='cuda')

        # Request 1: adds prefix blocks.
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req1)
        shared_b0 = ctx.request_to_kv_block_ids[0][0].item()
        shared_b1 = ctx.request_to_kv_block_ids[0][1].item()

        # Request 2: shares prefix, gets its own decode block.
        req2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req2)

        # Both requests share the same 2 blocks.
        assert ctx.request_to_kv_block_ids[1][0].item() == shared_b0
        assert ctx.request_to_kv_block_ids[1][1].item() == shared_b1

        # Set up request 0 for decode at offset that will cross block boundary.
        # Place at offset (block_size - 1) in last block so adding 3 tokens crosses.
        ctx.request_kv_length_offsets[0] = bs * 2 - 1  # one token from end of block 1
        # The local offset of index 6 is (6 % bs)
        ctx.request_last_kv_block_offset[0] = bs - 2
        ctx.request_query_lengths[0] = 1
        ctx.request_in_prefill_status_tensor[0] = 0
        ctx.active_token_count = 2

        active_mask = torch.tensor([1, 1], device='cuda', dtype=torch.int32)
        new_tokens = torch.tensor([50, 50], device='cuda')
        new_spec = torch.tensor([[51, 51], [52, 52]], device='cuda')

        ctx.update_requests(
            active_requests_mask=active_mask, new_tokens=new_tokens, new_speculative_tokens=new_spec
        )

        # A new block should have been allocated for the boundary crossing.
        assert ctx.request_kv_block_counts[0] == 3
        new_block = ctx.request_to_kv_block_ids[0][2].item()
        assert new_block != -1
        assert new_block != shared_b0
        assert new_block != shared_b1

        # Shared blocks should remain intact with ref count 2.
        assert ctx.kv_block_allocator.block_ref_counts[shared_b0].item() == 2
        assert ctx.kv_block_allocator.block_ref_counts[shared_b1].item() == 2

    @pytest.mark.internal
    @rounder_override(64)
    def test_chunked_prefill_prefix_caching_from_hidden_state(self):
        """Test prefix caching matching safely resolves from the hidden boundary state."""

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            max_requests=256,
            max_tokens=256,
            num_speculative_tokens=2,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            unified_memory_level=0,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)
        ctx.reset_tensors()

        bs = ctx.block_size_tokens

        # First request: register prefix blocks (bs * 3 tokens = 3 complete blocks).
        first_prompt = torch.arange(bs * 3, device='cuda')
        req_first = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=first_prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req_first)

        # Request 2: Chunked prefill sharing the same prefix
        req2 = DynamicInferenceRequest(
            request_id=42,
            prompt_tokens=first_prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.chunked_prefill_request_id = 42

        # Add chunk 1 (bs tokens)
        req2.finished_chunk_token_count = 0
        ctx.add_request(req2, prefill_chunk_length=bs)

        # Call update_requests to move req2 to the hidden state
        active_requests_mask = torch.tensor([1, 1], dtype=torch.int32, device='cuda')
        new_tokens = torch.tensor([99, 199], dtype=torch.int32, device='cuda')
        new_spec = torch.tensor([[100, 200], [101, 201]], dtype=torch.int32, device='cuda')
        ctx.update_requests(active_requests_mask, new_tokens, new_speculative_tokens=new_spec)

        # Capture active tokens before chunk 2 (which should just be the 3 tokens of req_first)
        tokens_before_chunk_2 = ctx.active_token_count
        assert tokens_before_chunk_2 == 3

        # Add chunk 2 (bs * 2 tokens)
        req2.finished_chunk_token_count = bs
        chunk_length = bs * 2
        ctx.add_request(req2, prefill_chunk_length=chunk_length)

        # Prefix match should find 2 matching blocks (blocks 1 and 2 from req_first).
        # With prefix match: 2 blocks matched -> skip (2*bs - 1) tokens
        # effective_chunk_length = chunk_length - prefix_skip_tokens
        # prefix_skip_tokens = min(2 * bs, chunk_length - 1) = 2 * bs - 1
        prefix_skip = 2 * bs - 1
        eff_chunk = chunk_length - prefix_skip

        (_, _, _, _, prefix_skip, eff_chunk) = ctx._compute_prefix_match(req2, chunk_length)
        expected_active = tokens_before_chunk_2 + eff_chunk
        assert ctx.active_token_count == expected_active

    @pytest.mark.internal
    @rounder_override(64)
    def test_prefix_caching_check_availability_with_speculative(self):
        """Test check_availability accounts for prefix match when spec decoding is enabled."""

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.01,
            block_size_tokens=32,
            num_speculative_tokens=3,
            enable_prefix_caching=True,
            unified_memory_level=0,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        bs = ctx.block_size_tokens
        prompt = torch.arange(bs * 2, device='cuda')

        # First request registers blocks.
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req1)

        # Exhaust the remaining pool.
        while ctx.kv_block_allocator.total_avail > 0:
            ctx.kv_block_allocator.allocate_memory_blocks(1)

        # A new request with the same prefix should still be schedulable
        # because prefix matching means 0 new blocks are needed from pool.
        req2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        _, _, kv_available = ctx.check_availability(req2)
        assert kv_available, "Matched blocks should not require pool allocation"

    @pytest.mark.internal
    @rounder_override(64)
    def test_prefix_match_exact_block_boundary(self):
        """Test prefix matching when the shared prefix is an exact multiple of the block size."""

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=16,
            enable_prefix_caching=True,
            unified_memory_level=0,
            max_tokens=512,
            max_requests=512,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        bs = ctx.block_size_tokens

        # req1: 32 tokens (exactly 2 complete blocks)
        prompt1 = torch.arange(bs * 2, device='cuda')
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt1,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req1)

        # req2: 35 tokens (first 32 tokens match req1)
        prompt2 = torch.arange(bs * 2 + 3, device='cuda')
        req2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt2,
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req2)

        # req2 should have 3 blocks total
        assert ctx.request_kv_block_counts[1].item() == 3

        # The first 2 blocks should be shared
        assert ctx.request_to_kv_block_ids[1, 0].item() == ctx.request_to_kv_block_ids[0, 0].item()
        assert ctx.request_to_kv_block_ids[1, 1].item() == ctx.request_to_kv_block_ids[0, 1].item()

        # The 3rd block should be a newly allocated pool block
        assert ctx.request_to_kv_block_ids[1, 2].item() != ctx.request_to_kv_block_ids[0, 1].item()

        # The offset points to the last token (index 34). In the 3rd block (indices 32-47), 34 is at offset 2.
        assert ctx.request_last_kv_block_offset[1].item() == 2

        # Effective query length should be 3 (35 total - 32 skipped)
        assert ctx.request_query_lengths[1].item() == 3

    @pytest.mark.internal
    @rounder_override(64)
    def test_eviction_with_shared_prefix_blocks(self):
        """Test that evicting a request drops ref counts correctly without destroying shared blocks."""

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=16,
            enable_prefix_caching=True,
            unified_memory_level=0,
            paused_buffer_size_gb=0.0,  # 0 paused capacity to force immediate eviction
            max_tokens=512,
            max_requests=512,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        bs = ctx.block_size_tokens
        prompt = torch.arange(bs * 2, device='cuda')

        # Add req1 and req2 with identical prompts
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req1)

        req2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req2)

        shared_b0 = ctx.request_to_kv_block_ids[0, 0].item()
        shared_b1 = ctx.request_to_kv_block_ids[0, 1].item()

        # Both blocks should be safely shared with ref count 2
        assert ctx.kv_block_allocator.block_ref_counts[shared_b0].item() == 2

        # Mock the state to make req1 paused and req2 active
        ctx.paused_request_count = 1
        ctx.total_request_count = 2
        ctx.request_ids[0] = 1
        ctx.request_ids[1] = 2
        ctx.request_kv_block_counts[0] = 2
        ctx.request_kv_block_counts[1] = 2

        # Exhaust the active block allocator
        ctx.kv_block_allocator.total_avail = 0

        # Trigger the eviction logic
        # next_tokens must be sized to total_request_count (1 paused + 1 active = 2)
        next_tokens = torch.tensor([50, 51], device='cuda')
        evicted_ids = ctx.evict_overflow_paused_requests(
            active_request_count=1, next_tokens=next_tokens
        )

        # req1 should be successfully evicted
        assert evicted_ids is not None
        assert evicted_ids[0].item() == 1

        # req2 remains active, so the shared blocks should drop to a ref count of 1
        assert ctx.kv_block_allocator.block_ref_counts[shared_b0].item() == 1
        assert ctx.kv_block_allocator.block_ref_counts[shared_b1].item() == 1

    @pytest.mark.internal
    @rounder_override(64)
    def test_oom_during_speculative_boundary_crossing(self):
        """Test boundary crossing with speculative tokens pauses the request gracefully when KV cache is full, keeping other requests active."""

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=128,
            buffer_size_gb=0.1,
            block_size_tokens=16,
            num_speculative_tokens=2,
            unified_memory_level=0,
            max_tokens=512,
            max_requests=512,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)
        bs = ctx.block_size_tokens

        # Setup 2 active requests.
        # Request 0 is exactly 1 token away from its boundary (will OOM).
        # Request 1 has plenty of space (will remain active).
        ctx.total_request_count = 2
        ctx.paused_request_count = 0
        ctx.active_token_count = 2

        ctx.request_ids[:2] = torch.tensor([10, 11], device='cuda')
        ctx.request_query_lengths[:2] = 1
        ctx.request_kv_block_counts[:2] = 1

        # Request 0 offset is 15. Adding 1 sampled + 2 spec = 3 tokens crosses the boundary (16).
        # Request 1 offset is 5. Adding 3 tokens = 8 (does not cross).
        ctx.request_kv_length_offsets[:2] = torch.tensor(
            [bs - 1, 5], device='cuda', dtype=torch.int32
        )
        ctx.request_last_kv_block_offset[:2] = torch.tensor(
            [bs - 1, 5], device='cuda', dtype=torch.int32
        )

        blocks = ctx.kv_block_allocator.allocate_memory_blocks(2)
        ctx.request_to_kv_block_ids[0, 0] = blocks[0]
        ctx.request_to_kv_block_ids[1, 0] = blocks[1]
        ctx.request_last_kv_block_id[:2] = blocks

        # Force OOM condition (no blocks left in the active pool)
        ctx.kv_block_allocator.total_avail = 0
        ctx.kv_block_allocator.paused_count = 100  # Prevent immediate eviction out of the system

        active_mask = torch.tensor([1, 1], device='cuda', dtype=torch.int32)
        new_tokens = torch.tensor([99, 88], device='cuda')
        new_spec = torch.tensor([[100, 200], [101, 201]], device='cuda')

        # Run update requests
        ctx.update_requests(
            active_requests_mask=active_mask, new_tokens=new_tokens, new_speculative_tokens=new_spec
        )

        # Request 0 should detect OOM, fail to allocate a new block, and pause.
        # Request 1 remains active, so active_request_count goes 2 -> 1, avoiding the deadlock assert.
        assert ctx.paused_request_count == 1
        assert ctx.total_request_count == 2

        # Request 1 generated 3 tokens (1 sampled + 2 spec)
        assert ctx.active_token_count == 3

        # Tokens must be cached in the paused buffers so Request 0 can resume cleanly later
        assert ctx.paused_tokens is not None
        assert ctx.paused_tokens[0].item() == 99

        assert ctx.paused_speculative_tokens is not None
        assert ctx.paused_speculative_tokens[0, 0].item() == 100
        assert ctx.paused_speculative_tokens[1, 0].item() == 101

    @pytest.mark.internal
    @rounder_override(64)
    def test_chunked_prefill_meets_prefix_caching(self):
        """Test that chunks in a chunked-prefill pipeline properly hit the prefix cache mid-flight."""

        model_config = TransformerConfig(
            params_dtype=torch.float32, num_layers=2, kv_channels=8, num_attention_heads=2
        )
        inference_config = InferenceConfig(
            max_sequence_length=512,
            buffer_size_gb=0.1,
            block_size_tokens=32,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            unified_memory_level=0,
            max_tokens=512,
            max_requests=512,
        )
        ctx = DynamicInferenceContext(model_config=model_config, inference_config=inference_config)

        bs = ctx.block_size_tokens
        prompt = torch.arange(128, device='cuda')

        # Cache req1 (fully processed)
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )
        ctx.add_request(req1)
        req1_blocks = [ctx.request_to_kv_block_ids[0, i].item() for i in range(4)]

        # Start chunked prefill for req2.
        req2 = DynamicInferenceRequest(
            request_id=2,
            prompt_tokens=prompt.clone(),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=bs,
            enable_prefix_caching=True,
        )

        # Add the first chunk (64 tokens)
        req2.finished_chunk_token_count = 0
        ctx.chunked_prefill_request_id = 2
        ctx.add_request(req2, prefill_chunk_length=64)

        # Assert the first chunk perfectly matched the first 2 cached blocks
        assert ctx.request_to_kv_block_ids[1, 0].item() == req1_blocks[0]
        assert ctx.request_to_kv_block_ids[1, 1].item() == req1_blocks[1]
        assert ctx.request_kv_block_counts[1].item() == 2

        # Simulate update_requests completing the chunk
        ctx.active_token_count += 1
        ctx.request_in_prefill_status_tensor[1] = 0
        ctx.total_request_count -= 1

        # Add the second chunk (64 tokens)
        req2.finished_chunk_token_count = 64
        ctx.add_request(req2, prefill_chunk_length=64)

        # It should correctly discover the remaining prefix blocks despite being mid-prefill
        assert ctx.request_to_kv_block_ids[1, 2].item() == req1_blocks[2]
        assert ctx.request_to_kv_block_ids[1, 3].item() == req1_blocks[3]
        assert ctx.request_kv_block_counts[1].item() == 4

        # Verify block references updated appropriately
        assert ctx.kv_block_allocator.block_ref_counts[req1_blocks[2]].item() == 2
        assert ctx.kv_block_allocator.block_ref_counts[req1_blocks[3]].item() == 2
