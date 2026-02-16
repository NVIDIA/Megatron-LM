# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import contextlib
import math

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

    def _setup_model_parallel_group(self, tensor_parallel_size, pipeline_parallel_size):

        self.pp_size = pipeline_parallel_size

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
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
    ):
        if is_hybrid_model:
            if layer_type_list is None:
                layer_type_list = [Symbols.MAMBA, Symbols.MLP, Symbols.ATTENTION, Symbols.MLP]
            mamba_conv_states_shape = (544, 4)
            mamba_ssm_states_shape = (8, 64, 16)
            mamba_inference_state_config = MambaInferenceStateConfig(
                layer_type_list, mamba_conv_states_shape, mamba_ssm_states_shape
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
                num_cuda_graphs=None,
                use_cuda_graphs_for_non_decode_steps=True,
                buffer_size_gb=buffer_size_gb,
                paused_buffer_size_gb=(
                    0.2 * buffer_size_gb if paused_buffer_size_gb is None else paused_buffer_size_gb
                ),
                block_size_tokens=block_size_tokens,
                max_tokens=max_tokens,
                mamba_inference_state_config=mamba_inference_state_config,
                use_flashinfer_fused_rope=None,  # default to using flash-infer if available
                # this is for compatibility with the LTS environment
                unified_memory_level=0,  # unit tests currently broken with UVM
            ),
        )
        return dynamic_context

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_initialize_dynamic_context(self, is_hybrid_model: bool):
        self._setup_model_parallel_group(1, 1)

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
            assert dynamic_context.block_allocator.total_count == 491
            assert dynamic_context.block_allocator.active_count == 392
            # We make max_requests divisible by the REQUEST_ROUNDER.
            assert dynamic_context.max_requests == 448
            assert dynamic_context.max_tokens == 16384
            assert dynamic_context.num_mamba_layers == 0
            assert dynamic_context.mamba_metadata is None
        else:
            assert dynamic_context.block_allocator.total_count == 556
            assert dynamic_context.block_allocator.active_count == 444
            assert dynamic_context.max_requests == 512
            assert dynamic_context.max_tokens == 16384
            assert dynamic_context.num_mamba_layers == 1
            assert dynamic_context.mamba_metadata is not None

        # Check initializations to -1
        assert torch.all(dynamic_context.request_ids == -1)

    @pytest.mark.internal
    def test_is_static_batching(self):
        self._setup_model_parallel_group(1, 1)
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
        self._setup_model_parallel_group(1, 1)
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
        dynamic_context.block_allocator.total_avail = 10
        assert dynamic_context.block_allocator.is_memory_available(10)
        assert not dynamic_context.block_allocator.is_memory_available(11)

        assert dynamic_context.block_allocator.is_memory_available(1)
        dynamic_context.block_allocator.total_avail = 0
        assert not dynamic_context.block_allocator.is_memory_available(1)

    @pytest.mark.internal
    @rounder_override(1)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_request_overflow(self, is_hybrid_model: bool):
        self._setup_model_parallel_group(1, 1)

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
        self._setup_model_parallel_group(1, 1)

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
        self._setup_model_parallel_group(1, 1)

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
            assert dynamic_context.block_allocator.active_count == 819
            assert dynamic_context.block_allocator.total_count == 1024
        else:
            assert dynamic_context.block_allocator.active_count == 1517
            assert dynamic_context.block_allocator.total_count == 1897
        assert torch.all(dynamic_context.request_to_kv_block_ids == -1)
        if is_hybrid_model:
            assert torch.all(dynamic_context.mamba_metadata.request_to_mamba_state_idx == -1)

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_allocate_and_release_memory_blocks(self, is_hybrid_model):
        self._setup_model_parallel_group(1, 1)
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
            dynamic_context.block_allocator.allocate_memory_blocks(4)
            .cpu()
            .detach()
            .numpy()
            .tolist()
            == expected_memory_blocks
        )
        assert dynamic_context.block_allocator.total_avail == expected_block_count_avail
        dynamic_context.block_allocator.release_memory_blocks(
            torch.tensor(expected_memory_blocks[-2:], device='cuda')
        )
        assert dynamic_context.block_allocator.total_avail == expected_block_count_avail + 2
        assert (
            dynamic_context.block_allocator.allocate_memory_blocks(1).item()
            == expected_memory_blocks[-1]
        )
        assert dynamic_context.block_allocator.total_avail == expected_block_count_avail + 1
        # Should return None since we allocate more blocks than what we have.
        assert (
            dynamic_context.block_allocator.allocate_memory_blocks(
                dynamic_context.block_allocator.total_avail + 100
            )
            == None
        )

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_add_request(self, is_hybrid_model: bool):
        self._setup_model_parallel_group(1, 1)

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
        self._setup_model_parallel_group(1, 1)

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
        block_avail_before = dynamic_context.block_allocator.total_avail

        dynamic_context.add_dummy_requests_parallel(requests, count_as_prefill=False)

        assert dynamic_context.active_token_count == total_tokens
        assert dynamic_context.total_request_count == len(requests)
        assert dynamic_context.num_prefill_requests == 0
        assert dynamic_context.block_allocator.total_avail == block_avail_before

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

        dummy_block_idx = dynamic_context.block_allocator.dummy_block_idx
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
        self._setup_model_parallel_group(1, 1)

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
        self._setup_model_parallel_group(1, 1)

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
        self._setup_model_parallel_group(1, 1)

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
        new_block_ids = dynamic_context.block_allocator.allocate_memory_blocks(3)
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
        dynamic_context.block_allocator.total_avail -= 11  # We align 11 blocks to the 10 requests we have. 3rd request alone we setup like it requires 2 blocks
        dynamic_context.total_request_count = total_request_count

        dynamic_context.request_to_kv_block_ids[0:total_request_count, 0] = torch.arange(
            dynamic_context.block_allocator.total_avail,
            dynamic_context.block_allocator.total_avail + 10,
        )
        dynamic_context.request_to_kv_block_ids[3][
            1
        ] = dynamic_context.block_allocator.total_avail  # Assign one extra block  to request 3.
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
        self._setup_model_parallel_group(1, 1)

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
        initial_blocks = dynamic_context.block_allocator.allocate_memory_blocks(5)
        dynamic_context.total_request_count = 5
        dynamic_context.paused_request_count = 0

        # Record the available blocks before releasing memory
        initial_available_blocks = dynamic_context.block_allocator.total_avail

        # Assign blocks to the requests (one block per request)
        for i in range(5):
            dynamic_context.request_to_kv_block_ids[i, 0] = initial_blocks[i]
            dynamic_context.request_query_lengths[i] = 1
            dynamic_context.request_ids[i] = i
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
        assert dynamic_context.block_allocator.total_avail == initial_available_blocks + 3

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
        self._setup_model_parallel_group(1, 1)

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
        initial_blocks = dynamic_context.block_allocator.allocate_memory_blocks(6)
        dynamic_context.total_request_count = 3
        dynamic_context.paused_request_count = 0

        # Record the available blocks before releasing memory
        initial_available_blocks = dynamic_context.block_allocator.total_avail

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
            if is_hybrid_model:
                dynamic_context.mamba_conv_states[:, i, :, :].fill_(float(i + 1))
                dynamic_context.mamba_ssm_states[:, i, :, :, :].fill_(float(i + 1))

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
        assert dynamic_context.block_allocator.total_avail == initial_available_blocks + 6

    @pytest.mark.internal
    @rounder_override(64)
    @pytest.mark.parametrize("is_hybrid_model", [False, True])
    def test_mamba_states_cache(self, is_hybrid_model: bool):
        self._setup_model_parallel_group(1, 1)

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
        self._setup_model_parallel_group(1, 1)
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

        if rank == 0:
            mamba_inference_state_config = MambaInferenceStateConfig(
                [Symbols.MAMBA] + [Symbols.ATTENTION] * 4,
                mamba_conv_states_shape,
                mamba_ssm_states_shape,
            )
        else:
            mamba_inference_state_config = MambaInferenceStateConfig(
                [Symbols.MAMBA] * 4 + [Symbols.ATTENTION],
                mamba_conv_states_shape,
                mamba_ssm_states_shape,
            )

        context = DynamicInferenceContext(
            model_config=TransformerConfig(
                params_dtype=torch.float32,
                num_layers=10,
                kv_channels=64,
                num_attention_heads=8,
                pipeline_model_parallel_size=pp_size,
                tensor_model_parallel_size=1,
                pipeline_dtype=torch.float32,
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
            [context.block_allocator.total_count], device='cuda', dtype=torch.long
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

    @pytest.mark.internal
    @pytest.mark.parametrize("ratio", [0.2, 0.4, 0.6, 0.8])
    @rounder_override(64)
    def test_mamba_memory_ratio_allocation(self, ratio):
        """
        Test that max_requests and block counts are partitioned correctly by mamba_memory_ratio.
        """
        self._setup_model_parallel_group(1, 1)

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
            layer_type_list, mamba_conv_states_shape, mamba_ssm_states_shape
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
        assert context.block_allocator.total_count == expected_active_blocks
        assert context.block_allocator.paused_count == expected_paused_blocks

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
