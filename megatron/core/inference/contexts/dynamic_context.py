# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor

try:
    from nvidia_chunked_flash_attn.flash_attn_interface import _get_block_size
except ModuleNotFoundError:

    def _get_block_size(*args, **kwargs):
        raise Exception(
            "Install package `nvidia_chunked_flash_attn` to use " "inference dynamic batching."
        )


from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import divide as core_divide

from .base_context import BaseInferenceContext


class ContextOverflowError(Exception):
    '''Base exception for when a new request would not fit.'''

    pass


class RequestOverflowError(ContextOverflowError):
    '''Adding request would overflow max request count.'''

    pass


class TokenOverflowError(ContextOverflowError):
    '''Adding request would overflow max token count.'''

    pass


class ChunkOverflowError(ContextOverflowError):
    '''Adding request would overflow available memory chunks.'''

    pass


class DynamicInferenceContext(BaseInferenceContext):
    """Inference context that is passed to the main model in order
    to efficiently calculate and store the KV cache during inference.

    The dynamic inference context manages both: 1) in-flight batching, and 2) a
    memory buffer for the chunked KV cache. For in-flight batching, requests of
    arbitrary sequence length may be added, paused, or removed from the context
    at any step. The only constraint is the maximum number of requests or tokens
    that the context is defined to support. For the chunked KV cache, a memory
    buffer is allocated up front (size `buffer_size_gb`), that is divided into
    chunks and dynamically assigned to requests. At any given step, any unassigned
    chunks equate to unused space.

    Additionally, a fraction of the memory buffer (`gtd_request_fraction`, i.e.,
    the 'guaranteed' request fraction) is reserved for guaranteeing that a
    minimum number of active requests may continue to generate tokens on any step.
    The reason for this is that the context manages two pools of requests: 1)
    active requests, and 2) paused requests. Paused requests are requests where
    insufficient memory chunks remain for future assignment, and these requests
    are set aside until enough memory chunks are available. Active requests are
    requests that have sufficient memory chunks to proceed with their generations.

    The situation can arise where all requests eventually become paused due to all
    memory chunks being assigned. In this case, there are no active requests and
    thus no progress can be made. To handle this case, a fraction of the memory
    buffer is reserved that only allows active requests, and no paused requests.
    This fraction must be carefully tuned, as it can have an order of magnitude
    impact on overall latency.

    Args:
        params_dtype (torch.dtype): Dtype used for KV cache.
        num_layers (int): Number of layers.
        kv_channels (int): Hidden dimension per attention head.
        num_attention_heads (int): Number of attention heads.
        max_sequence_length (int): Max possible sequence length (prompt + output)
            that will occur.
        buffer_size_gb (float): Total buffer size (GB), shared by main and
            fallback contexts.
        buffer_guaranteed_fraction (float): Fraction of the memory buffer that is
            reserved to guarantee that one or more active requests are able to
            run to completion. Without reserving this memory, paused requests are
            able to fill the memory buffer and block execution of any requests.
        buffer_overflow_factor (Optional[float]): Scaling factor over the buffer
            size for auto computing `max_requests` and `max_tokens`. This scaling
            factor is used for fitting more requests and tokens in the memory
            buffer than it can safely hold, which in turn increases throughput.
        max_requests_override (Optional[int]): If set, overrides value computed
            from `buffer_overflow_factor`.
        max_tokens_override (Optional[int]): If set, overrides value computed
            from `buffer_overflow_factor`.
    """

    def __init__(
        self,
        *,
        params_dtype: torch.dtype,
        num_layers: int,
        kv_channels: int,
        num_attention_heads: int,
        max_sequence_length: int,
        buffer_size_gb: float,
        buffer_guaranteed_fraction: float,
        buffer_overflow_factor: Optional[float] = None,
        max_requests_override: Optional[int] = None,
        max_tokens_override: Optional[int] = None,
    ):

        super().__init__()

        # Per partition num heads and hidden size.
        projection_size = kv_channels * num_attention_heads
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        hidden_size_per_attention_head = core_divide(projection_size, num_attention_heads)
        num_attention_heads_per_partition = core_divide(num_attention_heads, world_size)

        # Chunk size tokens, bytes.
        dtype_size_bytes = params_dtype.itemsize
        self.chunk_size_tokens = _get_block_size("cuda", kv_channels, False, True)[1]
        self.chunk_size_bytes = (
            dtype_size_bytes
            * 2  # key, value
            * num_layers
            * self.chunk_size_tokens
            * num_attention_heads_per_partition
            * hidden_size_per_attention_head
        )

        # Adjust buffer to be a multiple of chunk size.
        buffer_size_bytes = int(buffer_size_gb * 1024**3)
        buffer_size_bytes_rem = buffer_size_bytes % self.chunk_size_bytes
        buffer_size_bytes = buffer_size_bytes - buffer_size_bytes_rem

        # Compute max_requets, max_tokens from buffer size and overflow factor.
        def bytes_to_max_requests_and_tokens(n_bytes):
            n_tokens = n_bytes / self.chunk_size_bytes * self.chunk_size_tokens
            n_requests = n_tokens / max_sequence_length
            return int(n_requests), int(n_tokens)

        self.max_requests, self.max_tokens = bytes_to_max_requests_and_tokens(buffer_size_bytes)

        if buffer_overflow_factor is not None:
            self.max_requests = self.round_up(int(self.max_requests * buffer_overflow_factor))
            self.max_tokens = self.round_up(int(self.max_tokens * buffer_overflow_factor / 50.0))

        if max_requests_override is not None:
            self.max_requests = self.round_up(max_requests_override)

        if max_tokens_override is not None:
            self.max_tokens = self.round_up(max_tokens_override)

        self.max_requests = min(self.max_requests, self.max_tokens)  # e.g., decode only.

        # Initialize context state.
        self.params_dtype = params_dtype
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length

        self.total_request_count = 0
        self.active_token_count = 0
        self.paused_request_count = 0
        self.padded_active_token_count = None
        self.padded_active_sample_count = None
        self.paused_tokens = None

        # Per-request state.
        self.request_ids = torch.full(
            (self.max_requests,), 0, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.request_query_lengths = torch.empty_like(self.request_ids)
        self.request_kv_length_offsets = torch.empty_like(self.request_ids)
        self.request_kv_chunk_counts = torch.empty_like(self.request_ids)
        self.request_last_kv_chunk_id = torch.empty_like(self.request_ids)
        self.request_last_kv_chunk_offset = torch.empty_like(self.request_ids)

        # Per-token state.
        self.token_input_ids = torch.full(
            (self.max_tokens,), 0, dtype=torch.long, device=torch.cuda.current_device()
        )
        self.token_pos_ids = torch.full_like(self.token_input_ids, 0)
        self.token_to_request_idx = torch.empty_like(self.token_input_ids)
        self.token_to_kv_seq_idx = torch.empty_like(self.token_input_ids)
        self.token_to_chunk_idx = torch.empty_like(self.token_input_ids)
        self.token_to_local_kv_seq_idx = torch.empty_like(self.token_input_ids)

        # Simulate a stack, by decrementing chunk_count_avail during allocations.
        # *Note: The last chunk idx (`dummy_chunk_idx`) is reserved for
        # decode-only inference steps. For these teps, `input_ids` and
        # `position_ids` are padded to length `max_requests`, regardless of the
        # number of valid active requests. This padding is to maintain a
        # consistent input shape when using cuda graphs for the decode-only steps.
        # All requests between the active request count and `max_requests` are
        # 'garbage', but must still point to valid memory within the flash
        # attention kernel. These garbage requests point to chunk
        # `dummy_chunk_idx`.
        # TODO: @lmcafee, abstract chunk allocation into separate class.
        self.chunk_count_total = buffer_size_bytes // self.chunk_size_bytes
        self.chunk_count_avail = self.chunk_count_total - 1
        self.dummy_chunk_idx = self.chunk_count_total - 1
        self.chunk_bag = torch.arange(
            self.chunk_count_total, dtype=torch.int32, device=torch.cuda.current_device()
        )

        # Memory buffer.
        self.memory_buffer = torch.full(
            (
                self.chunk_count_total,
                2,  # key and value
                self.num_layers,
                self.chunk_size_tokens,
                num_attention_heads_per_partition,
                hidden_size_per_attention_head,
            ),
            0,
            dtype=self.params_dtype,
            device=torch.cuda.current_device(),
        )

        # Precompute base pointers for all chunks and layers.
        chunk_idxs = torch.arange(self.chunk_count_total, device=torch.cuda.current_device())
        layer_idxs = torch.arange(self.num_layers, device=torch.cuda.current_device())
        row_idx = chunk_idxs.repeat_interleave(self.num_layers)
        col_idx = layer_idxs.repeat(self.chunk_count_total)

        memory_buffer_data_ptr = self.memory_buffer.data_ptr()
        dtype_size_bytes = params_dtype.itemsize

        self.key_memory_buffer_pointers = torch.empty(
            self.chunk_count_total,
            self.num_layers,
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        self.value_memory_buffer_pointers = torch.empty(
            self.chunk_count_total,
            self.num_layers,
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )

        self.key_memory_buffer_pointers[row_idx, col_idx] = dtype_size_bytes * (
            row_idx * self.memory_buffer[0].numel() + col_idx * self.memory_buffer[0][0][0].numel()
        )
        self.value_memory_buffer_pointers[row_idx, col_idx] = dtype_size_bytes * (
            row_idx * self.memory_buffer[0].numel()
            + col_idx * self.memory_buffer[0][0][0].numel()
            + self.memory_buffer[0][0].numel()
        )

        self.key_memory_buffer_pointers += memory_buffer_data_ptr
        self.value_memory_buffer_pointers += memory_buffer_data_ptr

        # Chunk ids.
        self.max_kv_chunk_count = math.ceil(self.max_sequence_length / self.chunk_size_tokens)
        self.request_kv_memory = torch.full(
            (self.max_requests, self.max_kv_chunk_count),
            0,
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )

        # `*_decode_only` tensors are for use with cuda graphs to maintain
        # consistent input shapes, which is required to use cuda graphs. Cuda
        # graphs are used only during decode-only steps (i.e., no requests are in
        # the prefill phases). During these decode-only steps, the `*_decode_only`
        # tensors are used, otherwise their same-name but un-suffixed
        # corresponding tensors are used.
        # TODO: @lmcafee, only use `_decode_only` tensors when both of the
        # following conditions are met: 1) decode-only step, and 2) cuda graphs
        # are enabled.
        self.curr_chunk_key_ptrs_decode_only = torch.full(
            (self.num_layers, self.max_requests * self.max_kv_chunk_count),
            0,
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )

        self.curr_chunk_value_ptrs_decode_only = torch.full(
            (self.num_layers, self.max_requests * self.max_kv_chunk_count),
            0,
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )

        self.query_seq_lengths_decode_only = torch.full(
            (self.max_requests,), 0, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.cu_query_seq_lengths_decode_only = torch.full(
            (self.max_requests + 1,), 0, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.kv_seq_lengths_decode_only = torch.full(
            (self.max_requests,), 0, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.cu_kv_seq_lengths_decode_only = torch.full(
            (self.max_requests + 1,), 0, dtype=torch.int32, device=torch.cuda.current_device()
        )

        # Guaranteed active requests.
        # * See details in the class docstring above. `gtd_request_fraction` is
        #   the fraction of the memory buffer that is reserved for guaranteeing
        #   that some number of active requests can always proceed with their
        #   generations. The number of bytes defined by `gtd_request_fraction *
        #   buffer_size_gb` is converted to a number of requests that this
        #   reserved space can handle (`gtd_request_count`), and rounded to be an
        #   exact multiple of `max_sequence_length`. This is then converted into
        #   the number of reserved chunks (`gtd_chunk_count`) and bytes
        #   (`gtd_byte_count`).
        gtd_byte_count = buffer_guaranteed_fraction * buffer_size_bytes
        gtd_request_count, _ = bytes_to_max_requests_and_tokens(gtd_byte_count)
        gtd_request_count = max(1, gtd_request_count)
        gtd_request_count = self.round_up(min(gtd_request_count, self.max_requests))
        gtd_chunk_count = gtd_request_count * self.max_kv_chunk_count

        assert (
            gtd_request_count <= self.max_requests
        ), "gtd_request_count (%d) > max_requests (%d)." % (gtd_request_count, self.max_requests)
        self.gtd_request_count = gtd_request_count
        self.gtd_chunk_count = gtd_chunk_count

        # Reset attention state.
        self.reset_attention_state()

    ROUNDER = 64

    @classmethod
    def round_up(cls, value):
        """Round up to nearest multiple of `ROUNDER` (above)."""
        return cls.ROUNDER * int(math.ceil(int(value) / cls.ROUNDER))

    def is_static_batching(self) -> bool:
        """Is static batching? False."""
        return False

    def is_decode_only(self) -> bool:
        """Test if all active requests are in decode phase."""
        return self.total_request_count - self.paused_request_count == self.active_token_count

    def has_unfinished_requests(self) -> bool:
        """Test if any requests remain."""
        return self.total_request_count > 0

    def cu_query_lengths(self) -> Tensor:
        """Cumulative query sequence lengths."""
        return self.cu_query_seq_lengths, self.max_seqlen_q

    def cu_kv_lengths(self) -> Tensor:
        """Cumulative key/value sequence lengths."""
        return self.cu_kv_seq_lengths, self.max_seqlen_k

    def get_active_sequence_lengths(self) -> Tensor:
        """Total sequence length (query + key) for active requests."""
        lengths = self.request_kv_length_offsets + self.request_query_lengths
        lengths = lengths[self.paused_request_count : self.total_request_count]
        return lengths

    def append_key_value_cache(self, layer_number: int, key: Tensor, value: Tensor) -> None:
        """Append to KV cache.

        Args:
            layer_number (int): Layer number.
            key (Tensor): Key tensor.
            value (Tensor): Value tensor.
        """

        chunk_idx = self.token_to_chunk_idx[: self.padded_active_token_count]
        local_kv_seq_idx = self.token_to_local_kv_seq_idx[: self.padded_active_token_count]
        assert key.size(1) == 1 and value.size(1) == 1
        key = key.squeeze(1)
        value = value.squeeze(1)

        self.memory_buffer[chunk_idx, 0, layer_number - 1, local_kv_seq_idx] = key[
            : self.padded_active_token_count
        ]
        self.memory_buffer[chunk_idx, 1, layer_number - 1, local_kv_seq_idx] = value[
            : self.padded_active_token_count
        ]

    def key_value_cache(self, layer_number: int) -> Tuple[Tensor, Tensor]:
        """Read from KV cache.

        Args:
            layer_number (int): Layer number.

        Return:
            (Tuple[Tensor, Tensor]) The key and value pointer tensors that point
            to chunks within the chunked memory buffer.
        """
        key_memory_ptrs = self.curr_chunk_key_ptrs[layer_number - 1]
        value_memory_ptrs = self.curr_chunk_value_ptrs[layer_number - 1]
        key_memory_ptrs = key_memory_ptrs.view(
            self.padded_active_sample_count, self.max_kv_chunk_count
        )
        value_memory_ptrs = value_memory_ptrs.view(
            self.padded_active_sample_count, self.max_kv_chunk_count
        )

        return key_memory_ptrs, value_memory_ptrs

    def apply_rotary_emb_query(
        self, query: Tensor, query_emb: Tensor, config: TransformerConfig, cu_seqlens_q: Tensor
    ) -> Tensor:
        """Apply rotary embedding to query tensor.

        Args:
            query (Tensor): Query tensor.
            query_emb (Tensor): Query rotary embeddings.
            config (TransformerConfig): Transformer config.
            cu_seqlens_q (Tensor): Cumulative sequence lengths.

        Return:
            (Tensor) Query tensor after applying rotary embeddings.
        """
        n = self.padded_active_token_count
        query_seq_idx = self.token_pos_ids[:n]
        query_emb = query_emb[query_seq_idx]
        query[:n] = apply_rotary_pos_emb(query[:n], query_emb[:n], config, cu_seqlens_q)
        return query

    def apply_rotary_emb_key(
        self, key: Tensor, key_emb: Tensor, config: TransformerConfig
    ) -> Tensor:
        """Apply rotary embedding to key tensor.

        Args:
            key (Tensor): Key tensor.
            key_emb (Tensor): Key rotary embeddings.
            config (TransformerConfig): Transformer config.

        Return:
            (Tensor) Key tensor after applying rotary embeddings.
        """
        n = self.padded_active_token_count
        key_seq_idx = self.token_to_kv_seq_idx[:n]
        key_emb = key_emb[key_seq_idx]
        if self.is_decode_only():
            assert key.shape[0] == n == self.max_requests
            key = apply_rotary_pos_emb(key[:n], key_emb[:n], config)
        else:
            key[:n] = apply_rotary_pos_emb(key[:n], key_emb[:n], config)
        return key

    def is_memory_available(self, num_chunks: int, safe: bool = False) -> bool:
        """Check if memory chunks are available.

        Use 'safe' to avoid all requests being blocked. As detailed in the class
        docstring above, a fraction of the KV cache memory buffer is reserved to
        guarantee that a minimum number of active requests can run on any given
        step. This is handled by not allocating the final `gtd_request_fraction`
        of the memory buffer to either newly-added or currently-paused requests.

        `gtd_chunk_count` is the total number of chunks reserved for this purpose,
        and it is calculated as `gtd_request_count` (number of guaranteed active
        requests) * `max_kv_chunk_count` (max number of chunks necessary to run a
        request to `max_sequence_length`).

        Args:
            num_chunks (int): Number of chunks to check.
            safe (bool): Include extra space for guaranteeing ability to run
                `gtd_request_count` to completion.

        Return:
            (bool) Is memory available?
        """
        if safe:
            return self.chunk_count_avail >= num_chunks + self.gtd_chunk_count
        else:
            return self.chunk_count_avail >= num_chunks

    def allocate_memory_chunks(self, num_chunks: int = 1, safe: bool = False) -> Optional[Tensor]:
        """Allocate memory chunks if available, else return None.

        TODO: @lmcafee, abstract chunk allocation into separate class.

        Args:
            num_chunks (int): Number of chunks to allocate.
            safe (bool): Include extra space for guaranteeing ability to run
                `gtd_request_count` to completion. See `is_memory_available()`
                for more details.

        Return:
            (Optional[Tensor]) Allocated chunk IDs.
        """
        if self.is_memory_available(num_chunks, safe):
            self.chunk_count_avail -= num_chunks
            return self.chunk_bag[self.chunk_count_avail : (self.chunk_count_avail + num_chunks)]
        else:
            return None

    def release_memory_chunks(self, chunks: Tensor) -> None:
        """Release memory chunks.

        TODO: @lmcafee, abstract chunk allocation into separate class.

        Args:
            chunks (Tensor): Chunk IDs to release.

        Return:
            None
        """
        num_chunks = chunks.size(dim=0)
        self.chunk_bag[self.chunk_count_avail : (self.chunk_count_avail + num_chunks)] = chunks
        self.chunk_count_avail += num_chunks

    def reset_attention_state(self) -> None:
        """Reset state used within attention, after each step."""
        self.max_seqlen_q = None
        self.max_seqlen_k = None
        self.cu_query_seq_lengths = None
        self.cu_query_seq_lengths_decode_only.fill_(0)
        self.query_seq_lengths_decode_only.fill_(0)
        self.cu_kv_seq_lengths = None
        self.cu_kv_seq_lengths_decode_only.fill_(0)
        self.kv_seq_lengths_decode_only.fill_(0)
        self.curr_chunk_ids = None
        self.curr_chunk_key_ptrs = None
        self.curr_chunk_value_ptrs = None

    def initialize_attention_state(self) -> None:
        """Initialize attention state so that every layer can use it"""

        self.padded_active_token_count = (
            self.max_requests if self.is_decode_only() else self.round_up(self.active_token_count)
        )
        self.padded_active_sample_count = (
            self.max_requests
            if self.is_decode_only()
            else (self.total_request_count - self.paused_request_count)
        )
        self.token_to_chunk_idx[self.active_token_count : self.padded_active_token_count] = (
            self.dummy_chunk_idx
        )
        self.token_to_local_kv_seq_idx[self.active_token_count : self.padded_active_token_count] = 0
        self.token_to_kv_seq_idx[self.active_token_count : self.padded_active_token_count] = 0

        query_lengths = self.request_query_lengths[
            self.paused_request_count : self.total_request_count
        ]
        if self.is_decode_only():
            self.query_seq_lengths_decode_only[
                self.paused_request_count : self.total_request_count
            ] = query_lengths
            cu_query_lengths_decode_only = torch.cumsum(self.query_seq_lengths_decode_only, dim=0)
            self.cu_query_seq_lengths_decode_only[1:] = cu_query_lengths_decode_only
            self.cu_query_seq_lengths = self.cu_query_seq_lengths_decode_only
            self.max_seqlen_q = 1
        else:
            cu_query_lengths = torch.cumsum(query_lengths, dim=0)
            self.cu_query_seq_lengths = torch.full(
                (self.total_request_count - self.paused_request_count + 1,),
                0,
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            self.cu_query_seq_lengths[1:] = cu_query_lengths
            self.max_seqlen_q = query_lengths.max().item()

        kv_lengths = self.request_kv_length_offsets + self.request_query_lengths
        kv_lengths = kv_lengths[self.paused_request_count : self.total_request_count]
        if self.is_decode_only():
            self.kv_seq_lengths_decode_only[
                self.paused_request_count : self.total_request_count
            ] = kv_lengths
            cu_kv_lengths_decode_only = torch.cumsum(self.kv_seq_lengths_decode_only, dim=0)
            self.cu_kv_seq_lengths_decode_only[1:] = cu_kv_lengths_decode_only
            self.cu_kv_seq_lengths = self.cu_kv_seq_lengths_decode_only
            self.max_seqlen_k = self.max_sequence_length
        else:
            self.cu_kv_seq_lengths = torch.full(
                (self.total_request_count - self.paused_request_count + 1,),
                0,
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            self.cu_kv_seq_lengths[1:] = torch.cumsum(kv_lengths, dim=0)
            self.max_seqlen_k = kv_lengths.max().item()

        if self.is_decode_only():
            self.curr_chunk_ids = self.request_kv_memory.flatten()
            self.curr_chunk_key_ptrs_decode_only.copy_(
                self.key_memory_buffer_pointers[self.curr_chunk_ids].t().contiguous()
            )
            self.curr_chunk_value_ptrs_decode_only.copy_(
                self.value_memory_buffer_pointers[self.curr_chunk_ids].t().contiguous()
            )
            self.curr_chunk_key_ptrs = self.curr_chunk_key_ptrs_decode_only
            self.curr_chunk_value_ptrs = self.curr_chunk_value_ptrs_decode_only
        else:
            self.curr_chunk_ids = self.request_kv_memory[
                self.paused_request_count : self.total_request_count
            ].view(-1)
            self.curr_chunk_key_ptrs = (
                self.key_memory_buffer_pointers[self.curr_chunk_ids].t().contiguous()
            )
            self.curr_chunk_value_ptrs = (
                self.value_memory_buffer_pointers[self.curr_chunk_ids].t().contiguous()
            )

    def reset(self) -> None:
        """Reset entire context.

        This method does:
        - Reset active/paused request/token counts to zero.
        - Reset available chunks to entire memory.
        - Reset other tensors to zeros (unncessary, just or sanity checking).

        This method is useful after cuda graph warmup iterations, where the
        context's memory buffer is referenced by the cuda graph system and
        cannot be deallocated.
        """

        # Reset request/token counts.
        self.total_request_count = 0
        self.active_token_count = 0
        self.paused_request_count = 0
        self.padded_active_token_count = 0
        self.padded_active_sample_count = 0
        self.paused_tokens = None

        # Reset request indexes.
        self.request_ids.fill_(0)
        self.request_query_lengths.fill_(0)
        self.request_kv_length_offsets.fill_(0)
        self.request_kv_chunk_counts.fill_(0)
        self.request_last_kv_chunk_id.fill_(0)
        self.request_last_kv_chunk_offset.fill_(0)

        # Reset token indexes.
        self.token_input_ids.fill_(0)
        self.token_pos_ids.fill_(0)
        self.token_to_request_idx.fill_(0)
        self.token_to_kv_seq_idx.fill_(0)
        self.token_to_chunk_idx.fill_(0)
        self.token_to_local_kv_seq_idx.fill_(0)

        # Reset available chunk count.
        self.reset_attention_state()
        self.chunk_count_avail = self.chunk_count_total - 1
        self.memory_buffer.fill_(0)
        self.request_kv_memory.fill_(0)

    def current_input_ids(self) -> Tensor:
        """Flattened input IDs for forward pass.

        Return:
            (Tensor) Flattened active input IDs.
        """
        return self.token_input_ids[: self.padded_active_token_count].unsqueeze(0)

    def current_position_ids(self) -> Tensor:
        """Flattened position IDs for forward pass.

        Return:
            (Tensor) Flattened active position IDs.
        """
        return self.token_pos_ids[: self.padded_active_token_count].unsqueeze(0)

    def last_token_logits(self, logits: Tensor) -> Tensor:
        """Last tokens of logits.

        Args:
            logits (Tensor): Output logits of forward pass.

        Return:
            (Tensor) Last token logits.
        """

        # todo: @lmcafee, remove these asserts?
        assert logits.size(0) == 1
        assert logits.size(1) == self.padded_active_token_count, (
            f"logits.size(1) ({tuple(logits.shape)}) != "
            f"padded_active_token_count ({self.padded_active_token_count})."
        )

        # Last token logits.
        logits = logits.squeeze(0)
        last_token_idxs = (
            torch.cumsum(
                self.request_query_lengths[self.paused_request_count : self.total_request_count],
                dim=0,
            )
            - 1
        )
        last_token_logits = logits[last_token_idxs, :]

        return last_token_logits

    def add_request(self, request_id: int, tokens: List[int]) -> None:
        """Add request to context.

        After a request is added, it will first do one prefill step, followed by
        an arbitrary number of decode steps.

        A request will failed to be added if one of the following is true:
        - Adding the request would overflow the max token count.
        - Adding the request would overflow the max request count.
        - Adding the request would overflow memory.

        todo: @lmcafee, cache non-added requests until there is space, for better
        user experience.

        Args:
            request_id (int): Unique ID of request.
            tokens (List[int]): Token IDs of request prompt.

        Return:
            None
        """

        # `context_length` here is the equal to prompt length, and does not
        # include output length.
        context_length = len(tokens)

        # Test for token and request overflow.
        if self.active_token_count + context_length > self.max_tokens:
            raise TokenOverflowError()
        if self.total_request_count >= self.max_requests:
            raise RequestOverflowError()

        # Preallocate chunks.
        num_chunks_needed = math.ceil(context_length / self.chunk_size_tokens)
        new_chunk_ids = self.allocate_memory_chunks(num_chunks_needed, safe=True)
        if new_chunk_ids is None:
            raise ChunkOverflowError()

        # Update request state.
        self.request_ids[self.total_request_count] = request_id
        self.request_query_lengths[self.total_request_count] = context_length
        self.request_kv_length_offsets[self.total_request_count] = 0
        self.request_kv_memory[self.total_request_count][:num_chunks_needed] = new_chunk_ids
        self.request_kv_chunk_counts[self.total_request_count] = num_chunks_needed
        self.request_last_kv_chunk_id[self.total_request_count] = new_chunk_ids[-1]
        self.request_last_kv_chunk_offset[self.total_request_count] = (
            context_length - 1
        ) % self.chunk_size_tokens

        # Update token state.
        arange_context_length = torch.arange(context_length, device=torch.cuda.current_device())

        self.token_pos_ids[self.active_token_count : (self.active_token_count + context_length)] = (
            arange_context_length
        )
        self.token_input_ids[
            self.active_token_count : (self.active_token_count + context_length)
        ] = tokens

        self.token_to_request_idx[
            self.active_token_count : (self.active_token_count + context_length)
        ] = self.total_request_count
        self.token_to_kv_seq_idx[
            self.active_token_count : (self.active_token_count + context_length)
        ] = arange_context_length
        self.token_to_chunk_idx[
            self.active_token_count : (self.active_token_count + context_length)
        ] = new_chunk_ids[arange_context_length // self.chunk_size_tokens]
        self.token_to_local_kv_seq_idx[
            self.active_token_count : (self.active_token_count + context_length)
        ] = (arange_context_length % self.chunk_size_tokens)

        # Increment request and token counts.
        self.total_request_count += 1
        self.active_token_count += context_length

    # TODO: see if we can compile this function
    def update_requests(self, active_requests: Tensor, next_tokens: Tensor) -> None:
        """Update context state after calling engine.step().

        This method is responsible for:
        - Update prefill requests to decode requests.
        - Persist decode requests as decode requests.
        - Terminate requests by length or termination id.

        *Note*: All bookkeeping tensors (i.e., `self.request_*`) are laid out
        contiguously, with a conceptual division between paused requests on the
        'left' (or, lower indices) and active requests on the 'right' (or, higher
        indices). The integers `paused_request_count` and `total_request_count`
        are used to track the boundaries between these two conceptual request
        groups. The reason for maintaining contiguous tensors rather than multiple
        smaller (e.g., per-group or per-request) tensors is for both 1) speed
        (avoid unnecessary tensor allocations), and 2) compatibility with the
        Flash Attention kernels, which packed contiguous tensors.

        Args:
            active_requests (Tensor): Mask tensor marking active requests.
            next_tokens (Tensor): Newly requestd tokens to append.

        Return:
            None
        """

        active_request_count = (active_requests == 1).sum().item()
        assert len(active_requests) + self.paused_request_count == self.total_request_count

        # Reset attention state.
        self.reset_attention_state()

        # Handle no requests (release chunks and reset counts).
        if active_request_count + self.paused_request_count == 0:

            # Release all KV memory.
            if self.total_request_count > 0:
                for idx in range(self.total_request_count):
                    length = self.request_kv_chunk_counts[idx]
                    self.release_memory_chunks(self.request_kv_memory[idx][:length])

            # Reset request/token counts.
            self.total_request_count = 0
            self.active_token_count = 0
            return

        # Concatenate paused + next tokens.
        if self.paused_request_count != 0:
            assert self.paused_tokens is not None
            next_tokens = torch.cat((self.paused_tokens, next_tokens))

        # Release KV cache of non-active requests. (Non-active requests were
        # previously marked as having either 1) outputted the termination token,
        # or 2) reach max_sequence_length.)
        if active_request_count + self.paused_request_count != self.total_request_count:
            non_active_idxs = (
                torch.nonzero(active_requests == 0, as_tuple=True)[0] + self.paused_request_count
            )
            lengths = self.request_kv_chunk_counts[non_active_idxs].long()
            row_idx = torch.repeat_interleave(non_active_idxs, lengths)

            begin_idxs = lengths.cumsum(dim=0).roll(1)
            begin_idxs[0] = 0
            col_idx = torch.arange(
                lengths.sum(), device=torch.cuda.current_device()
            ) - begin_idxs.repeat_interleave(lengths)
            self.release_memory_chunks(self.request_kv_memory[row_idx, col_idx])

        # Shift active requests left to maintain request order:
        # - Paused
        # - Active
        # - Inactive
        if (
            active_request_count > 0
            and active_request_count + self.paused_request_count != self.total_request_count
        ):

            # Destination & source active request indexes.
            dst_idxs = (
                torch.nonzero(active_requests[:active_request_count] == 0, as_tuple=True)[0]
                + self.paused_request_count
            )
            src_idxs = (
                torch.nonzero(active_requests[active_request_count:], as_tuple=True)[0]
                + active_request_count
                + self.paused_request_count
            )

            # Shift active requests left.
            self.request_kv_length_offsets[dst_idxs] = self.request_kv_length_offsets[src_idxs]
            self.request_query_lengths[dst_idxs] = self.request_query_lengths[src_idxs]
            self.request_ids[dst_idxs] = self.request_ids[src_idxs]
            next_tokens[dst_idxs] = next_tokens[src_idxs]

            self.request_kv_memory[dst_idxs] = self.request_kv_memory[src_idxs]

            self.request_kv_chunk_counts[dst_idxs] = self.request_kv_chunk_counts[src_idxs]
            self.request_last_kv_chunk_id[dst_idxs] = self.request_last_kv_chunk_id[src_idxs]
            self.request_last_kv_chunk_offset[dst_idxs] = self.request_last_kv_chunk_offset[
                src_idxs
            ]

        # Shift paused requests left to maintain request order:
        # - Paused
        # - Active
        # - Inactive
        if active_request_count > 0:
            local_paused_requests = (
                self.request_last_kv_chunk_offset[
                    self.paused_request_count : (active_request_count + self.paused_request_count)
                ]
                == self.chunk_size_tokens - 1
            ).byte()
            local_paused_request_count = (local_paused_requests == 1).sum().item()

            # Moved paused requests to the left.
            if (
                local_paused_request_count > 0
                and local_paused_request_count != active_request_count
            ):
                x_idxs = (
                    torch.nonzero(
                        local_paused_requests[:local_paused_request_count] == 0, as_tuple=True
                    )[0]
                    + self.paused_request_count
                )
                y_idxs = (
                    torch.nonzero(
                        local_paused_requests[local_paused_request_count:], as_tuple=True
                    )[0]
                    + local_paused_request_count
                    + self.paused_request_count
                )

                # Swap data.
                dst_idxs = torch.cat((x_idxs, y_idxs))
                src_idxs = torch.cat((y_idxs, x_idxs))

                self.request_kv_length_offsets[dst_idxs] = self.request_kv_length_offsets[src_idxs]
                self.request_query_lengths[dst_idxs] = self.request_query_lengths[src_idxs]
                self.request_ids[dst_idxs] = self.request_ids[src_idxs]
                next_tokens[dst_idxs] = next_tokens[src_idxs]

                self.request_kv_memory[dst_idxs] = self.request_kv_memory[src_idxs]
                self.request_kv_chunk_counts[dst_idxs] = self.request_kv_chunk_counts[src_idxs]
                self.request_last_kv_chunk_id[dst_idxs] = self.request_last_kv_chunk_id[src_idxs]
                self.request_last_kv_chunk_offset[dst_idxs] = self.request_last_kv_chunk_offset[
                    src_idxs
                ]

            self.paused_request_count += local_paused_request_count
            active_request_count -= local_paused_request_count

        # Assign released chunks to paused requests.
        # todo: @shanmugamr, un-pause requests using FIFO, rather than LIFO.
        if self.chunk_count_avail <= self.paused_request_count + self.gtd_chunk_count:
            if active_request_count < self.gtd_request_count:
                resume_request_count = min(
                    self.paused_request_count, self.gtd_request_count - active_request_count
                )
            else:
                resume_request_count = 0
        else:
            resume_request_count = min(self.paused_request_count, self.chunk_count_avail)

        self.paused_request_count -= resume_request_count
        active_request_count += resume_request_count
        assert active_request_count > 0, "active_request_count == %d." % active_request_count

        self.total_request_count = active_request_count + self.paused_request_count
        self.active_token_count = active_request_count

        self.token_input_ids[: self.active_token_count] = next_tokens[
            self.paused_request_count : self.total_request_count
        ]

        if self.paused_request_count > 0:
            self.paused_tokens = next_tokens[: self.paused_request_count]

        self.request_kv_length_offsets[self.paused_request_count : self.total_request_count].add_(
            self.request_query_lengths[self.paused_request_count : self.total_request_count]
        )
        self.request_query_lengths[self.paused_request_count : self.total_request_count].fill_(1)
        self.token_pos_ids[: self.active_token_count] = self.request_kv_length_offsets[
            self.paused_request_count : self.total_request_count
        ]

        self.request_last_kv_chunk_offset[self.paused_request_count : self.total_request_count] = (
            self.request_last_kv_chunk_offset[self.paused_request_count : self.total_request_count]
            + 1
        ) % self.chunk_size_tokens

        # Allocate new chunks for resumed requests.
        if resume_request_count > 0:
            assert torch.all(
                self.request_last_kv_chunk_offset[
                    self.paused_request_count : (self.paused_request_count + resume_request_count)
                ]
                == 0
            )
            chunk_ids = self.allocate_memory_chunks(resume_request_count)
            assert chunk_ids is not None
            row_idx = torch.arange(
                self.paused_request_count,
                self.paused_request_count + resume_request_count,
                device=torch.cuda.current_device(),
            )
            col_idx = self.request_kv_chunk_counts[
                self.paused_request_count : (self.paused_request_count + resume_request_count)
            ]
            self.request_kv_memory[row_idx, col_idx] = chunk_ids
            self.request_kv_chunk_counts[
                self.paused_request_count : (self.paused_request_count + resume_request_count)
            ] += 1
            self.request_last_kv_chunk_id[
                self.paused_request_count : (self.paused_request_count + resume_request_count)
            ] = chunk_ids

        # Update token indexes.
        self.token_to_request_idx[: self.active_token_count] = torch.arange(
            self.paused_request_count, self.total_request_count, device=torch.cuda.current_device()
        )
        self.token_to_kv_seq_idx[: self.active_token_count] = self.request_kv_length_offsets[
            self.paused_request_count : self.total_request_count
        ]

        self.token_to_chunk_idx[: self.active_token_count] = self.request_last_kv_chunk_id[
            self.paused_request_count : self.total_request_count
        ]
        self.token_to_local_kv_seq_idx[: self.active_token_count] = (
            self.request_last_kv_chunk_offset[self.paused_request_count : self.total_request_count]
        )
