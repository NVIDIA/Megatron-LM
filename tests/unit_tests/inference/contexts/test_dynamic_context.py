import pytest
import torch

from megatron.core.inference.contexts.dynamic_context import (
    DynamicInferenceContext,
    RequestOverflowError,
    TokenOverflowError,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils


def set_rounder(value):
    """Utility function to set the DynamicInferenceContext rounder."""
    DynamicInferenceContext.ROUNDER = value  # For backwards compatibility
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


class TestDynamicContext:

    def _setup_model_parallel_group(self, tensor_parallel_size, pipeline_parallel_size):

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
        chunk_size_tokens,
        buffer_guarenteed_fraction,
        buffer_overflow_factor,
        max_requests_override,
        max_tokens_override,
    ):
        set_rounder(64)
        dynamic_context = DynamicInferenceContext(
            params_dtype=params_dtype,
            num_layers=num_layers,
            kv_channels=kv_channels,
            num_attention_heads=num_attention_heads,
            max_sequence_length=max_sequence_length,
            buffer_size_gb=buffer_size_gb,
            buffer_guaranteed_fraction=buffer_guarenteed_fraction,
            chunk_size_tokens=chunk_size_tokens,
            buffer_overflow_factor=buffer_overflow_factor,
            max_requests_override=max_requests_override,
            max_tokens_override=max_tokens_override,
        )
        return dynamic_context

    def teardown_method(self, method):
        set_rounder(64)
        Utils.destroy_model_parallel()

    @pytest.mark.experimental
    def test_initialize_dynamic_context(self):
        self._setup_model_parallel_group(1, 1)
        with pytest.raises(AssertionError) as error:

            dynamic_context = self._get_dynamic_context(
                params_dtype=torch.float32,
                num_layers=4,
                kv_channels=8,
                num_attention_heads=2,
                max_sequence_length=512,
                buffer_size_gb=0.01,
                buffer_guarenteed_fraction=0.1,
                chunk_size_tokens=128,
                max_requests_override=None,
                max_tokens_override=None,
                buffer_overflow_factor=None,
            )
        assert f'gtd_request_count (64) > max_requests (40).' in str(error.value)

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            buffer_guarenteed_fraction=0.1,
            chunk_size_tokens=128,
            max_requests_override=None,
            max_tokens_override=None,
            buffer_overflow_factor=None,
        )
        assert dynamic_context.gtd_chunk_count == 256
        assert dynamic_context.gtd_request_count == 64
        assert dynamic_context.chunk_allocator.chunk_count_total == 491
        assert dynamic_context.max_requests == 122
        assert dynamic_context.max_tokens == 62848

        # Check initializations to -1
        assert torch.all(dynamic_context.request_ids == -1)

    @pytest.mark.experimental
    def test_is_static_batching(self):
        self._setup_model_parallel_group(1, 1)
        dynamic_context = DynamicInferenceContext(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=8,
            max_sequence_length=512,
            buffer_size_gb=1.0,
            buffer_guaranteed_fraction=0.1,
            chunk_size_tokens=128,
        )
        assert not dynamic_context.is_static_batching()

    @pytest.mark.experimental
    def test_is_memory_available(self):
        self._setup_model_parallel_group(1, 1)
        dynamic_context = DynamicInferenceContext(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=8,
            max_sequence_length=512,
            buffer_size_gb=1.0,
            buffer_guaranteed_fraction=0.1,
            chunk_size_tokens=128,
        )
        dynamic_context.chunk_allocator.chunk_count_avail = 10
        assert dynamic_context.chunk_allocator.is_memory_available(10)
        assert not dynamic_context.chunk_allocator.is_memory_available(11)

        assert dynamic_context.chunk_allocator.is_memory_available(1)
        dynamic_context.chunk_allocator.chunk_count_avail = 0
        assert not dynamic_context.chunk_allocator.is_memory_available(1)

        dynamic_context.chunk_allocator.chunk_count_avail = 10
        dynamic_context.gtd_chunk_count = 5
        assert dynamic_context.chunk_allocator.is_memory_available(6)
        assert not dynamic_context.chunk_allocator.is_memory_available(6, safe=True)

    @pytest.mark.experimental
    def test_request_overflow(self):
        self._setup_model_parallel_group(1, 1)
        set_rounder(1)
        dynamic_context = DynamicInferenceContext(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=8,
            max_sequence_length=128,
            buffer_size_gb=0.01,
            buffer_guaranteed_fraction=0.1,
            chunk_size_tokens=32,
        )
        with pytest.raises(RequestOverflowError):
            for i in range(dynamic_context.max_requests + 1):
                dynamic_context.add_request(
                    i, torch.zeros(10, device='cuda')
                )  # Adding more than allowed requests

    @pytest.mark.experimental
    def test_token_overflow_error(self):
        self._setup_model_parallel_group(1, 1)
        set_rounder(1)
        dynamic_context = DynamicInferenceContext(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=8,
            max_sequence_length=512,
            buffer_size_gb=0.1,
            buffer_guaranteed_fraction=0.1,
            chunk_size_tokens=128,
            buffer_overflow_factor=1.0,
            max_requests_override=2,
            max_tokens_override=20,  # Setting a very low token limit
        )

        with pytest.raises(TokenOverflowError):
            dynamic_context.add_request(
                1, torch.arange(0, 25, device='cuda')
            )  # Exceeding max token count

    @pytest.mark.experimental
    def test_reset(self):
        self._setup_model_parallel_group(1, 1)
        dynamic_context = DynamicInferenceContext(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=8,
            max_sequence_length=128,
            buffer_size_gb=1.0,
            buffer_guaranteed_fraction=0.1,
            chunk_size_tokens=128,
        )

        # Initialize all variables
        dynamic_context.total_request_count = 10
        dynamic_context.active_token_count = 10
        dynamic_context.paused_request_count = 5
        dynamic_context.padded_active_token_count = 10
        dynamic_context.padded_active_sample_count = 5
        dynamic_context.paused_tokens = torch.tensor([1, 2, 3], device='cuda')
        dynamic_context.request_ids.fill_(1)
        dynamic_context.request_query_lengths.fill_(1)
        dynamic_context.request_kv_length_offsets.fill_(1)
        dynamic_context.request_kv_chunk_counts.fill_(1)
        dynamic_context.request_last_kv_chunk_id.fill_(1)
        dynamic_context.request_last_kv_chunk_offset.fill_(1)
        dynamic_context.token_to_input_ids.fill_(1)
        dynamic_context.token_to_pos_ids.fill_(1)
        dynamic_context.token_to_request_idx.fill_(1)
        dynamic_context.token_to_position_in_request.fill_(1)
        dynamic_context.token_to_chunk_idx.fill_(1)
        dynamic_context.token_to_local_position_within_kv_chunk.fill_(1)
        dynamic_context.chunk_allocator.chunk_count_avail = 5
        dynamic_context.memory_buffer.fill_(1)
        dynamic_context.request_to_kv_chunk_ids.fill_(1)

        # Call reset
        dynamic_context.reset()

        # Assert all variables are reset to zero or their default values
        assert dynamic_context.total_request_count == 0
        assert dynamic_context.active_token_count == 0
        assert dynamic_context.paused_request_count == 0
        assert dynamic_context.padded_active_token_count == 0
        assert dynamic_context.padded_active_sample_count == 0
        assert dynamic_context.paused_tokens is None
        assert torch.all(dynamic_context.request_ids == -1)
        assert torch.all(dynamic_context.request_query_lengths == 0)
        assert torch.all(dynamic_context.request_kv_length_offsets == 0)
        assert torch.all(dynamic_context.request_kv_chunk_counts == 0)
        assert torch.all(dynamic_context.request_last_kv_chunk_id == -1)
        assert torch.all(dynamic_context.request_last_kv_chunk_offset == 0)
        assert torch.all(dynamic_context.token_to_input_ids == 0)
        assert torch.all(dynamic_context.token_to_pos_ids == 0)
        assert torch.all(dynamic_context.token_to_request_idx == -1)
        assert torch.all(dynamic_context.token_to_position_in_request == 0)
        assert torch.all(dynamic_context.token_to_chunk_idx == -1)
        assert torch.all(dynamic_context.token_to_local_position_within_kv_chunk == 0)
        assert (
            dynamic_context.chunk_allocator.chunk_count_avail
            == dynamic_context.chunk_allocator.chunk_count_total - 1
        )
        assert torch.all(dynamic_context.request_to_kv_chunk_ids == -1)

    @pytest.mark.experimental
    def test_allocate_and_release_memory_chunks(self):
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            buffer_guarenteed_fraction=0.1,
            chunk_size_tokens=128,
            max_requests_override=None,
            max_tokens_override=None,
            buffer_overflow_factor=None,
        )

        assert dynamic_context.chunk_allocator.allocate_memory_chunks(
            4
        ).cpu().detach().numpy().tolist() == [486, 487, 488, 489]
        assert dynamic_context.chunk_allocator.chunk_count_avail == 486
        dynamic_context.chunk_allocator.release_memory_chunks(
            torch.tensor([488, 489], device='cuda')
        )
        assert dynamic_context.chunk_allocator.chunk_count_avail == 488
        assert dynamic_context.chunk_allocator.allocate_memory_chunks(1).item() == 489
        assert dynamic_context.chunk_allocator.chunk_count_avail == 487
        # Should return None since we allocate more chunks than what we have.
        assert (
            dynamic_context.chunk_allocator.allocate_memory_chunks(
                dynamic_context.chunk_allocator.chunk_count_avail + 100
            )
            == None
        )

    @pytest.mark.experimental
    def test_add_request(self):
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            buffer_guarenteed_fraction=0.1,
            chunk_size_tokens=128,
            max_requests_override=None,
            max_tokens_override=None,
            buffer_overflow_factor=None,
        )
        assert dynamic_context.chunk_size_tokens == 128
        context_length = 144
        dynamic_context.add_request(
            request_id=0, tokens=torch.arange(0, context_length, dtype=torch.long, device='cuda')
        )
        assert dynamic_context.total_request_count == 1
        assert dynamic_context.active_token_count == context_length
        assert dynamic_context.request_ids[0] == 0
        assert torch.all(dynamic_context.request_ids[1:] == -1)
        assert dynamic_context.request_query_lengths[0] == context_length
        assert dynamic_context.request_kv_length_offsets[0] == 0
        assert dynamic_context.request_to_kv_chunk_ids[0].cpu().detach().numpy().tolist() == [
            488,
            489,
            -1,
            -1,
        ]
        assert dynamic_context.request_kv_chunk_counts[0] == 2
        assert dynamic_context.request_last_kv_chunk_id[0] == 489
        assert dynamic_context.request_last_kv_chunk_offset[0].item() == 15
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
        assert torch.all(
            dynamic_context.token_to_chunk_idx[0:context_length][
                0 : dynamic_context.chunk_size_tokens
            ]
            == 488
        )
        assert torch.all(
            dynamic_context.token_to_chunk_idx[0:context_length][
                dynamic_context.chunk_size_tokens : context_length
            ]
            == 489
        )
        assert torch.all(
            dynamic_context.token_to_local_position_within_kv_chunk[0:context_length]
            == torch.arange(0, context_length, dtype=torch.long, device='cuda')
            % dynamic_context.chunk_size_tokens
        )

    @pytest.mark.experimental
    def test_update_request(self):
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            buffer_guarenteed_fraction=0.1,
            chunk_size_tokens=128,
            max_requests_override=None,
            max_tokens_override=None,
            buffer_overflow_factor=None,
        )

        # This case should just reset and return since all requests are finished
        active_requests_mask = torch.Tensor([0, 0, 0])
        dynamic_context.paused_request_count = 0
        dynamic_context.total_request_count = 3
        dynamic_context.request_kv_chunk_counts[0:3] = 1
        new_chunk_ids = dynamic_context.chunk_allocator.allocate_memory_chunks(3, safe=True)
        dynamic_context.request_to_kv_chunk_ids[0:3, 0] = new_chunk_ids
        dynamic_context.update_requests(
            active_requests_mask=active_requests_mask, new_tokens=torch.tensor([0, 1, 2])
        )
        assert dynamic_context.total_request_count == 0

        # This case would cover all cases
        # 1. Already there will be 2 paused requests
        # 2. Active request mask will have active and finished requests.
        # 3. The active requests will also have some requests that have to be paused because of reaching max token limit within chunk
        # 4. Some of these requests will be resumed.
        # Setup is as follows :
        # Request ids 0, 1 are paused
        # Request ids 2 , 4, 9 are active requests
        # Request ids 3 7 8 have completed
        # Request ids 5 and 6 will require on more chunk later on coz they finished their current chunk

        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            buffer_guarenteed_fraction=0.1,
            chunk_size_tokens=128,
            max_requests_override=None,
            max_tokens_override=None,
            buffer_overflow_factor=None,
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
        dynamic_context.chunk_allocator.chunk_count_avail -= 11  # We align 11 chunks to the 10 requests we have. 3rd request alone we setup like it requires 2 chunks
        dynamic_context.total_request_count = total_request_count

        dynamic_context.request_to_kv_chunk_ids[0:total_request_count, 0] = torch.arange(
            dynamic_context.chunk_allocator.chunk_count_avail,
            dynamic_context.chunk_allocator.chunk_count_avail + 10,
        )
        dynamic_context.request_to_kv_chunk_ids[3][
            1
        ] = (
            dynamic_context.chunk_allocator.chunk_count_avail
        )  # Assign one extra chunk  to request 3.
        dynamic_context.request_kv_length_offsets[0:total_request_count] = 10
        # For 0, 1, 5, 6, the total number of tokens in last chunk is chunk size -1, so that they will all need extra chunks
        dynamic_context.request_kv_length_offsets[0:2] = dynamic_context.chunk_size_tokens - 1
        dynamic_context.request_kv_length_offsets[5:7] = dynamic_context.chunk_size_tokens - 1
        # For the 3rd request, its completed and required 2 chunks. So we add more tokens than chunks size
        dynamic_context.request_kv_length_offsets[3] = dynamic_context.chunk_size_bytes + 10
        dynamic_context.request_query_lengths[0:total_request_count] = (
            1  # Everything is in decode phase
        )

        dynamic_context.request_ids[0:total_request_count] = torch.arange(0, total_request_count)
        dynamic_context.request_kv_chunk_counts[0:total_request_count] = 1
        dynamic_context.request_kv_chunk_counts[3] = 2  # 3rd chunk alone requies 2 chunks
        dynamic_context.request_last_kv_chunk_id[0:total_request_count] = torch.arange(
            0, total_request_count
        )
        dynamic_context.request_last_kv_chunk_id[3] = 11
        dynamic_context.request_last_kv_chunk_offset[0:total_request_count] = 10
        # For the 3rd request, its completed and required 2 chunks. So we add more tokens than chunks size
        dynamic_context.request_last_kv_chunk_offset[0:2] = dynamic_context.chunk_size_tokens - 1
        dynamic_context.request_last_kv_chunk_offset[5:7] = dynamic_context.chunk_size_tokens - 1

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

        # The first four are zero because they have all obtained a new chunk
        assert dynamic_context.request_last_kv_chunk_offset[0:10].cpu().numpy().tolist() == [
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

        # The first 4 requests will require an extra chunk.
        assert torch.all(
            dynamic_context.request_to_kv_chunk_ids[0:10].cpu()
            == torch.tensor(
                [
                    [479, 482, -1, -1],
                    [480, 479, -1, -1],
                    [484, 486, -1, -1],
                    [485, 487, -1, -1],
                    [483, -1, -1, -1],
                    [481, -1, -1, -1],
                    [488, -1, -1, -1],
                    [486, -1, -1, -1],
                    [487, -1, -1, -1],
                    [488, -1, -1, -1],
                ]
            )
        )

    @pytest.mark.experimental
    def test_release_memory_chunks_for_finished_requests(self):
        """Test that memory chunks are correctly released for finished requests."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            buffer_guarenteed_fraction=0.1,
            chunk_size_tokens=128,
            max_requests_override=None,
            max_tokens_override=None,
            buffer_overflow_factor=None,
        )

        # Set up the initial state with 5 requests
        # Allocate 5 chunks for 5 requests
        initial_chunks = dynamic_context.chunk_allocator.allocate_memory_chunks(5, safe=True)
        dynamic_context.total_request_count = 5
        dynamic_context.paused_request_count = 0

        # Record the available chunks before releasing memory
        initial_available_chunks = dynamic_context.chunk_allocator.chunk_count_avail

        # Assign chunks to the requests (one chunk per request)
        for i in range(5):
            dynamic_context.request_to_kv_chunk_ids[i, 0] = initial_chunks[i]
            dynamic_context.request_query_lengths[i] = 1
            dynamic_context.request_ids[i] = i

        # Create an active_requests_mask where requests 0, 2, and 4 are finished (0),
        # and requests 1 and 3 are still active (1)
        active_requests_mask = torch.tensor([0, 1, 0, 1, 0], device=torch.cuda.current_device())

        # Call update_requests with these parameters
        dynamic_context.update_requests(
            active_requests_mask=active_requests_mask,
            new_tokens=torch.tensor([10, 11, 12, 13, 14], device=torch.cuda.current_device()),
        )

        # After the update, we should have released 3 chunks (for requests 0, 2, and 4)
        # and have 2 active requests (1 and 3)
        assert dynamic_context.total_request_count == 2
        assert dynamic_context.active_token_count == 2

        # Verify that 3 chunks were released by checking the available chunks
        assert dynamic_context.chunk_allocator.chunk_count_avail == initial_available_chunks + 3

    @pytest.mark.experimental
    def test_finished_requests_with_multiple_chunks(self):
        """Test that all memory chunks are correctly released for finished requests that use multiple chunks."""
        self._setup_model_parallel_group(1, 1)
        dynamic_context = self._get_dynamic_context(
            params_dtype=torch.float32,
            num_layers=4,
            kv_channels=8,
            num_attention_heads=2,
            max_sequence_length=512,
            buffer_size_gb=0.03,
            buffer_guarenteed_fraction=0.1,
            chunk_size_tokens=128,
            max_requests_override=None,
            max_tokens_override=None,
            buffer_overflow_factor=None,
        )

        # Set up the initial state with 3 requests, where some use multiple chunks
        # Allocate 6 chunks in total for the requests
        initial_chunks = dynamic_context.chunk_allocator.allocate_memory_chunks(6, safe=True)
        dynamic_context.total_request_count = 3
        dynamic_context.paused_request_count = 0

        # Record the available chunks before releasing memory
        initial_available_chunks = dynamic_context.chunk_allocator.chunk_count_avail

        # Assign chunks to the requests:
        # - Request 0: 1 chunk
        # - Request 1: 2 chunks
        # - Request 2: 3 chunks
        dynamic_context.request_to_kv_chunk_ids[0, 0] = initial_chunks[0]

        dynamic_context.request_to_kv_chunk_ids[1, 0] = initial_chunks[1]
        dynamic_context.request_to_kv_chunk_ids[1, 1] = initial_chunks[2]

        dynamic_context.request_to_kv_chunk_ids[2, 0] = initial_chunks[3]
        dynamic_context.request_to_kv_chunk_ids[2, 1] = initial_chunks[4]
        dynamic_context.request_to_kv_chunk_ids[2, 2] = initial_chunks[5]

        dynamic_context.request_kv_chunk_counts[0] = 1
        dynamic_context.request_kv_chunk_counts[1] = 2
        dynamic_context.request_kv_chunk_counts[2] = 3

        for i in range(3):
            dynamic_context.request_query_lengths[i] = 1
            dynamic_context.request_ids[i] = i

        # Create an active_requests_mask where all requests are finished
        active_requests_mask = torch.tensor([0, 0, 0], device=torch.cuda.current_device())

        # Call update_requests with these parameters
        dynamic_context.update_requests(
            active_requests_mask=active_requests_mask,
            new_tokens=torch.tensor([10, 11, 12], device=torch.cuda.current_device()),
        )

        # After the update, we should have released all 6 chunks and have 0 active requests
        assert dynamic_context.total_request_count == 0
        assert dynamic_context.active_token_count == 0

        # Verify that all 6 chunks were released by checking the available chunks
        assert dynamic_context.chunk_allocator.chunk_count_avail == initial_available_chunks + 6
