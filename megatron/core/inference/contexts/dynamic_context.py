# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math
import warnings
from typing import Optional, Tuple

import torch
from packaging.version import Version as PkgVersion
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.package_info import __version__ as mcore_version
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import divide as core_divide

from .base_context import BaseInferenceContext
from .dynamic_chunk_allocator import ChunkAllocator


class ContextOverflowError(Exception):
    '''Base exception for when a new request would not fit.'''

    pass


class RequestOverflowError(ContextOverflowError):
    '''Adding request would overflow max request count.'''

    pass


class TokenOverflowError(ContextOverflowError):
    '''Adding request would overflow max token count.'''

    pass


class MaxSequenceLengthOverflowError(ContextOverflowError):
    '''Adding request would overflow max sequence length.'''

    pass


class ChunkOverflowError(ContextOverflowError):
    '''Adding request would overflow available memory chunks.'''

    pass


# pylint: disable=line-too-long
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
        chunk_size_tokens (int): Size of KV cache chunk size.
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
        chunk_size_tokens: int = 256,
        buffer_overflow_factor: Optional[float] = None,
        max_requests_override: Optional[int] = None,
        max_tokens_override: Optional[int] = None,
        tensor_model_parallel_size: Optional[int] = None,
    ):

        super().__init__(materialize_only_last_token_logits=True)
        # Per partition num heads and hidden size.
        projection_size = kv_channels * num_attention_heads
        if tensor_model_parallel_size is None:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
        else:
            tp_size = tensor_model_parallel_size
        hidden_size_per_attention_head = core_divide(projection_size, num_attention_heads)
        num_attention_heads_per_partition = core_divide(num_attention_heads, tp_size)

        # Chunk size tokens, bytes.
        dtype_size_bytes = params_dtype.itemsize
        self.chunk_size_tokens = chunk_size_tokens
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
            self.max_requests = self.round_up_requests(
                int(self.max_requests * buffer_overflow_factor)
            )
            self.max_tokens = self.round_up_tokens(
                int(self.max_tokens * buffer_overflow_factor / 50.0)
            )

        if max_requests_override is not None:
            self.max_requests = self.round_up_requests(max_requests_override)

        if max_tokens_override is not None:
            self.max_tokens = self.round_up_tokens(max_tokens_override)

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
            (self.max_requests,), -1, dtype=torch.int32, device=torch.cuda.current_device()
        )
        # request_query_lengths is the input prompt tokens length during prefill phase (1st step) and then 1 for the decode phase (i.e During generation)
        self.request_query_lengths = torch.empty_like(self.request_ids)
        # request_output_lengths is len(input_prompt_tokens) + num_tokens_to_generate
        self.request_output_lengths = torch.empty_like(self.request_ids)
        # request_kv_length_offsets is the same as query length during prefill phase (1st step) and then 1 for the decode phase (i.e During generation)
        self.request_kv_length_offsets = torch.empty_like(self.request_ids)
        self.request_kv_chunk_counts = torch.empty_like(self.request_ids)
        self.request_last_kv_chunk_id = torch.empty_like(self.request_ids)
        # request_last_kv_chunk_offset represents number of tokens in the last kv chunk
        self.request_last_kv_chunk_offset = torch.empty_like(self.request_ids)

        # Per-token state.
        self.token_to_input_ids = torch.full(
            (self.max_tokens,), 0, dtype=torch.long, device=torch.cuda.current_device()
        )
        self.token_to_pos_ids = torch.full_like(self.token_to_input_ids, 0)
        self.token_to_request_idx = torch.empty_like(self.token_to_input_ids)
        self.token_to_chunk_idx = torch.empty_like(self.token_to_input_ids)
        # i.e For a set of tokens A B C D E F ..  and chunk_size 4:
        # token_to_position_in_request is  [0, 1, 2, 3, 4, 5]
        # token_to_local_position_within_kv_chunk is [0 , 1, 2, 3, 0, 1, 2]
        self.token_to_position_in_request = torch.empty_like(self.token_to_input_ids)
        self.token_to_local_position_within_kv_chunk = torch.empty_like(self.token_to_input_ids)

        # Calculate the total number of chunks available in the buffer
        chunk_count_total = buffer_size_bytes // self.chunk_size_bytes

        # Memory buffer.
        self.memory_buffer = torch.full(
            (
                2,  # key and value
                self.num_layers,
                chunk_count_total,
                self.chunk_size_tokens,
                num_attention_heads_per_partition,
                hidden_size_per_attention_head,
            ),
            -1,
            dtype=self.params_dtype,
            device=torch.cuda.current_device(),
        )

        # Chunk ids.
        self.max_kv_chunk_count = math.ceil(self.max_sequence_length / self.chunk_size_tokens)
        self.request_to_kv_chunk_ids = torch.full(
            (self.max_requests, self.max_kv_chunk_count),
            -1,
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

        self.kv_memory_decode_only = torch.full(
            (self.max_requests, self.max_kv_chunk_count),
            0,
            dtype=torch.int,
            device=torch.cuda.current_device(),
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
        # Chunk ids.
        self.max_kv_chunk_count = math.ceil(self.max_sequence_length / self.chunk_size_tokens)
        gtd_byte_count = buffer_guaranteed_fraction * buffer_size_bytes
        gtd_request_count, _ = bytes_to_max_requests_and_tokens(gtd_byte_count)
        if buffer_guaranteed_fraction > 0:
            gtd_request_count = max(1, gtd_request_count)
        gtd_request_count = self.round_up_requests(min(gtd_request_count, self.max_requests))
        gtd_chunk_count = gtd_request_count * self.max_kv_chunk_count
        assert (
            gtd_request_count <= self.max_requests
        ), "gtd_request_count (%d) > max_requests (%d)." % (gtd_request_count, self.max_requests)
        self.gtd_request_count = gtd_request_count
        self.gtd_chunk_count = gtd_chunk_count

        # Initialize chunk allocator
        self.chunk_allocator = ChunkAllocator(
            chunk_count_total=chunk_count_total, gtd_chunk_count=self.gtd_chunk_count
        )

        # Store the dummy chunk idx reference for convenience
        self.dummy_chunk_idx = self.chunk_allocator.dummy_chunk_idx
        # Reset attention state.
        self.reset_attention_state()

    TOKEN_ROUNDER = 64
    REQUEST_ROUNDER = 4

    @classmethod
    def round_up_tokens(cls, value):
        """Round up to nearest multiple of `TOKEN_ROUNDER` (above)."""
        if PkgVersion(mcore_version) < PkgVersion("0.13"):
            return cls.round_up(value)
        return cls.TOKEN_ROUNDER * int(math.ceil(int(value) / cls.TOKEN_ROUNDER))

    @classmethod
    def round_up_requests(cls, value):
        """Round up to nearest multiple of `REQUEST_ROUNDER` (above)."""
        if PkgVersion(mcore_version) < PkgVersion("0.13"):
            return cls.round_up(value)
        return cls.REQUEST_ROUNDER * int(math.ceil(int(value) / cls.REQUEST_ROUNDER))

    @classmethod
    def round_up(cls, value):
        """Deprecated in favor of round_up_tokens and round_up_requests."""
        warnings.warn(
            "`round_up` is deprecated in favor of `round_up_tokens` or `round_up_requests` "
            "and will be removed in `megatron-core` 0.14."
        )
        ROUNDER = getattr(cls, "ROUNDER", 64)
        return ROUNDER * int(math.ceil(int(value) / ROUNDER))

    def is_static_batching(self) -> bool:
        """Is static batching? False."""
        return False

    def is_decode_only(self) -> bool:
        """Test if all active requests are in decode phase.

        For a request in prefill phase active_tokens = query length
        Once the request moves to decode phase active tokens is 1 for that request. So if all active requests are in decode phase, they will be equal to active token count.
        """
        total_active_requests = self.total_request_count - self.paused_request_count
        return total_active_requests == self.active_token_count

    def has_unfinished_requests(self) -> bool:
        """Test if any requests remain."""
        return self.total_request_count > 0

    def cu_query_lengths(self) -> Tensor:
        """Cumulative query sequence lengths."""
        return self.cu_query_seq_lengths, self.max_seqlen_q

    def cu_kv_lengths(self) -> Tensor:
        """Cumulative key/value sequence lengths."""
        return (
            self.cu_kv_seq_lengths,
            self.kv_seq_lengths,
            self.kv_seq_lengths_decode_only,
            self.max_seqlen_k,
        )

    def get_active_sequence_lengths(self) -> Tensor:
        """Total sequence length (query + key) for active requests."""
        lengths = self.request_kv_length_offsets + self.request_query_lengths
        lengths = lengths[self.paused_request_count : self.total_request_count]
        return lengths

    def get_max_sequence_lengths(self) -> Tensor:
        """Maximum sequence length for active requests."""
        return self.request_output_lengths[self.paused_request_count : self.total_request_count]

    def append_key_value_cache(self, layer_number: int, key: Tensor, value: Tensor) -> None:
        """Append to KV cache.

        Args:
            layer_number (int): Layer number.
            key (Tensor): Key tensor.
            value (Tensor): Value tensor.
        """

        chunk_idx = self.token_to_chunk_idx[: self.padded_active_token_count]
        local_kv_seq_idx = self.token_to_local_position_within_kv_chunk[
            : self.padded_active_token_count
        ]
        assert key.size(1) == 1 and value.size(1) == 1
        key = key.squeeze(1)
        value = value.squeeze(1)

        self.memory_buffer[0, layer_number - 1, chunk_idx, local_kv_seq_idx] = key[
            : self.padded_active_token_count
        ]
        self.memory_buffer[1, layer_number - 1, chunk_idx, local_kv_seq_idx] = value[
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
        return (
            self.memory_buffer[0, layer_number - 1],
            self.memory_buffer[1, layer_number - 1],
            self.block_table,
        )

    def apply_rotary_emb_query(
        self,
        query: Tensor,
        query_emb: Tensor,
        config: TransformerConfig,
        cu_seqlens_q: Tensor,
        cp_group: torch.distributed.ProcessGroup,
    ) -> Tensor:
        """Apply rotary embedding to query tensor.

        Args:
            query (Tensor): Query tensor.
            query_emb (Tensor): Query rotary embeddings.
            config (TransformerConfig): Transformer config.
            cu_seqlens_q (Tensor): Cumulative sequence lengths.
            cp_group (torch.distributed.ProcessGroup): Process group for context parallel.

        Return:
            (Tensor) Query tensor after applying rotary embeddings.
        """
        n = self.padded_active_token_count
        query_seq_idx = self.token_to_pos_ids[:n]
        query_emb = query_emb[query_seq_idx]
        query[:n] = apply_rotary_pos_emb(
            t=query[:n],
            freqs=query_emb[:n],
            config=config,
            cu_seqlens=cu_seqlens_q,
            cp_group=cp_group,
        )
        return query

    def apply_rotary_emb_key(
        self,
        key: Tensor,
        key_emb: Tensor,
        config: TransformerConfig,
        cp_group: torch.distributed.ProcessGroup,
    ) -> Tensor:
        """Apply rotary embedding to key tensor.

        Args:
            key (Tensor): Key tensor.
            key_emb (Tensor): Key rotary embeddings.
            config (TransformerConfig): Transformer config.
            cp_group (torch.distributed.ProcessGroup): Process group for context parallel.

        Return:
            (Tensor) Key tensor after applying rotary embeddings.
        """
        n = self.padded_active_token_count
        key_seq_idx = self.token_to_position_in_request[:n]
        key_emb = key_emb[key_seq_idx]
        if self.is_decode_only():
            assert key.shape[0] == n == self.max_requests
            key = apply_rotary_pos_emb(
                t=key[:n], freqs=key_emb[:n], config=config, cp_group=cp_group
            )
        else:
            key[:n] = apply_rotary_pos_emb(
                t=key[:n], freqs=key_emb[:n], config=config, cp_group=cp_group
            )
        return key

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
        self.kv_memory_decode_only.fill_(0)
        self.block_table = None

    def initialize_attention_state(self) -> None:
        """Initialize attention state so that every layer can use it"""

        self.padded_active_token_count = (
            self.max_requests
            if self.is_decode_only()
            else self.round_up_tokens(self.active_token_count)
        )
        self.padded_active_sample_count = (
            self.max_requests
            if self.is_decode_only()
            else (self.total_request_count - self.paused_request_count)
        )
        self.token_to_chunk_idx[self.active_token_count : self.padded_active_token_count] = (
            self.dummy_chunk_idx
        )
        self.token_to_local_position_within_kv_chunk[
            self.active_token_count : self.padded_active_token_count
        ] = 0
        self.token_to_position_in_request[
            self.active_token_count : self.padded_active_token_count
        ] = 0

        query_lengths = self.request_query_lengths[
            self.paused_request_count : self.total_request_count
        ]
        if self.is_decode_only():
            self.query_seq_lengths_decode_only[
                0 : self.total_request_count - self.paused_request_count
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

        kv_seq_lengths = self.request_kv_length_offsets + self.request_query_lengths
        self.kv_seq_lengths = kv_seq_lengths[self.paused_request_count : self.total_request_count]
        if self.is_decode_only():
            self.kv_seq_lengths_decode_only[
                0 : self.total_request_count - self.paused_request_count
            ] = self.kv_seq_lengths
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
            self.cu_kv_seq_lengths[1:] = torch.cumsum(self.kv_seq_lengths, dim=0)
            self.max_seqlen_k = self.kv_seq_lengths.max().item()

        kv_memory = self.request_to_kv_chunk_ids[
            self.paused_request_count : self.total_request_count
        ]
        if self.is_decode_only():
            self.kv_memory_decode_only[0 : self.total_request_count - self.paused_request_count] = (
                kv_memory
            )
            self.block_table = self.kv_memory_decode_only
        else:
            self.block_table = self.request_to_kv_chunk_ids[
                self.paused_request_count : self.total_request_count
            ]

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
        self.request_ids.fill_(-1)
        self.request_query_lengths.fill_(0)
        self.request_output_lengths.fill_(0)
        self.request_kv_length_offsets.fill_(0)
        self.request_kv_chunk_counts.fill_(0)
        self.request_last_kv_chunk_id.fill_(-1)
        self.request_last_kv_chunk_offset.fill_(0)
        self.request_to_kv_chunk_ids.fill_(-1)

        # Reset token indexes.
        self.token_to_input_ids.fill_(0)
        self.token_to_pos_ids.fill_(0)
        self.token_to_request_idx.fill_(-1)
        self.token_to_position_in_request.fill_(0)
        self.token_to_chunk_idx.fill_(-1)
        self.token_to_local_position_within_kv_chunk.fill_(0)

        # Reset available chunk count.
        self.reset_attention_state()
        self.chunk_allocator.reset()
        self.request_to_kv_chunk_ids.fill_(-1)

    def current_input_ids(self) -> Tensor:
        """Flattened input IDs for forward pass.

        Return:
            (Tensor) Flattened active input IDs.
        """
        return self.token_to_input_ids[: self.padded_active_token_count].unsqueeze(0)

    def current_position_ids(self) -> Tensor:
        """Flattened position IDs for forward pass.

        Return:
            (Tensor) Flattened active position IDs.
        """
        return self.token_to_pos_ids[: self.padded_active_token_count].unsqueeze(0)

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

    def add_request(
        self, request_id: int, tokens: torch.Tensor, num_tokens_to_generate: Optional[int] = None
    ) -> None:
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
            tokens (torch.Tensor): Token IDs of request prompt.
            num_tokens_to_generate (int): Number of tokens to generate for the request.

        Return:
            None
        """

        # `context_length` here is the equal to prompt length, and does not
        # include output length.
        context_length = len(tokens)

        # Test for token and request overflow.
        # TODO : Should move this into some waiting queue
        if self.active_token_count + context_length > self.max_tokens:
            raise TokenOverflowError()
        if self.total_request_count >= self.max_requests:
            raise RequestOverflowError()

        # Preallocate chunks.
        num_chunks_needed = math.ceil(context_length / self.chunk_size_tokens)
        new_chunk_ids = self.chunk_allocator.allocate_memory_chunks(num_chunks_needed, safe=True)
        if new_chunk_ids is None:
            raise ChunkOverflowError()

        if num_tokens_to_generate is None:
            num_tokens_to_generate = self.max_sequence_length - context_length
        elif context_length + num_tokens_to_generate > self.max_sequence_length:
            raise MaxSequenceLengthOverflowError()

        # Update request state.
        self.request_ids[self.total_request_count] = request_id
        self.request_query_lengths[self.total_request_count] = context_length
        self.request_output_lengths[self.total_request_count] = (
            context_length + num_tokens_to_generate
        )
        self.request_kv_length_offsets[self.total_request_count] = 0
        self.request_to_kv_chunk_ids[self.total_request_count][:num_chunks_needed] = new_chunk_ids
        self.request_kv_chunk_counts[self.total_request_count] = num_chunks_needed
        self.request_last_kv_chunk_id[self.total_request_count] = new_chunk_ids[-1]
        self.request_last_kv_chunk_offset[self.total_request_count] = (
            context_length - 1
        ) % self.chunk_size_tokens

        # Update token state.
        arange_context_length = torch.arange(context_length, device=torch.cuda.current_device())

        self.token_to_pos_ids[
            self.active_token_count : (self.active_token_count + context_length)
        ] = arange_context_length
        self.token_to_input_ids[
            self.active_token_count : (self.active_token_count + context_length)
        ] = tokens

        self.token_to_request_idx[
            self.active_token_count : (self.active_token_count + context_length)
        ] = self.total_request_count
        self.token_to_position_in_request[
            self.active_token_count : (self.active_token_count + context_length)
        ] = arange_context_length
        self.token_to_chunk_idx[
            self.active_token_count : (self.active_token_count + context_length)
        ] = new_chunk_ids[arange_context_length // self.chunk_size_tokens]
        self.token_to_local_position_within_kv_chunk[
            self.active_token_count : (self.active_token_count + context_length)
        ] = (arange_context_length % self.chunk_size_tokens)

        # Increment request and token counts.
        self.total_request_count += 1
        self.active_token_count += context_length

    def _swap_book_keeping_tensors(self, src_idxs, dst_idxs, next_tokens):
        """
        Swaps all the relevent booking tensors with src idxs to dst idxs
        """
        self.request_kv_length_offsets[dst_idxs] = self.request_kv_length_offsets[src_idxs]
        self.request_query_lengths[dst_idxs] = self.request_query_lengths[src_idxs]
        self.request_output_lengths[dst_idxs] = self.request_output_lengths[src_idxs]
        self.request_ids[dst_idxs] = self.request_ids[src_idxs]
        next_tokens[dst_idxs] = next_tokens[src_idxs]

        self.request_to_kv_chunk_ids[dst_idxs] = self.request_to_kv_chunk_ids[src_idxs]
        self.request_kv_chunk_counts[dst_idxs] = self.request_kv_chunk_counts[src_idxs]
        self.request_last_kv_chunk_id[dst_idxs] = self.request_last_kv_chunk_id[src_idxs]
        self.request_last_kv_chunk_offset[dst_idxs] = self.request_last_kv_chunk_offset[src_idxs]

    # TODO: see if we can compile this function
    def update_requests(self, active_requests_mask: Tensor, new_tokens: Tensor) -> None:
        """Update context state after calling engine.step().

        This method is responsible for:
        - Update prefill requests to decode requests.
        - Persist decode requests as decode requests.
        - Terminate requests by length or termination id.

        *Note*: All bookkeeping tensors (i.e., `self.request_*`) are laid out
        contiguously, with a conceptual division between paused requests on the
        'left' (or, lower indices) and active requests in the 'middle' (or, middle
        indices) and completed requests on the 'right' (or, higher indices). The integers
        `paused_request_count` and `total_request_count`  are used to track the boundaries
        between these request groups.
        - 0:paused_request_count -> paused requests
        - paused_request_count:total_request_count -> active requests
        - total_request_count:max_requests -> completed requests are moved here.
        The reason for maintaining contiguous tensors rather than multiple
        smaller (e.g., per-group or per-request) tensors is for both 1) speed
        (avoid unnecessary tensor allocations), and 2) compatibility with the
        Flash Attention kernels, which packed contiguous tensors.

        The following happens in this code :
        1. The active token mask tells us which requests are still active and which are completed
        2. If no paused requests are present and no active requests we release all memory and reset.
        3. Concatenate the paused tokens to the active tokens
        4. For the finished requests we release memory chunks and move them to the right
        5. We identify requests that require a new chunk and add them to the paused requests (i.e move them left)
        6. We determine how many requests we can resume and resume them
        7. We make changes to the request book keeping tesnsors and setup the tokens for next iteration
        8. We resume those requests by assigning chunks and updating bookkeeping tensors
        9. We make relevant changes to the token bookkeeping tensors

        Args:
            active_requests_mask (Tensor): 1D Mask tensor marking active requests.
            new_tokens (Tensor): Newly sampled tokens, with one token per active request.

        Return:
            None
        """
        # 1. The active token mask tells us which requests are still active and which are completed
        # active_request_count -> This corresponds to requests that have not reached EOD or max length
        # finished_request_count are requests that have reached the termination criterion
        active_request_count = (active_requests_mask == 1).sum().item()
        finished_request_count = (active_requests_mask == 0).sum().item()
        assert (
            active_request_count + finished_request_count + self.paused_request_count
            == self.total_request_count
        )

        # Reset attention state.
        self.reset_attention_state()

        # 2. If no paused requests are present and no active requests we release memory and reset.
        if active_request_count + self.paused_request_count == 0:
            if finished_request_count > 0:
                finished_idxs = (
                    torch.nonzero(active_requests_mask == 0, as_tuple=True)[0]
                    + self.paused_request_count
                )
                kv_chunks_assigned = self.request_to_kv_chunk_ids[finished_idxs]
                non_zero_values_in_kv_memory = kv_chunks_assigned[kv_chunks_assigned != -1]
                self.chunk_allocator.release_memory_chunks(non_zero_values_in_kv_memory)

            # Reset request/token counts.
            self.request_to_kv_chunk_ids.fill_(-1)
            self.total_request_count = 0
            self.active_token_count = 0
            return

        # 3. Concatenate the paused tokens to the active tokens if present.
        if self.paused_request_count != 0:
            assert self.paused_tokens is not None
            next_tokens = torch.cat((self.paused_tokens, new_tokens))
        else:
            next_tokens = new_tokens

        # 4. For the finished requests we release memory chunks and move them to the right:-
        #       a) Release all their memory
        #       b) Swap them to the right, so that we have this order [Paused, Active, Finished]
        if finished_request_count > 0:
            finished_idxs = (
                torch.nonzero(active_requests_mask == 0, as_tuple=True)[0]
                + self.paused_request_count
            )
            kv_chunks_asigned = self.request_to_kv_chunk_ids[finished_idxs]
            non_zero_values_in_kv_memory = kv_chunks_asigned[kv_chunks_asigned != -1]
            self.chunk_allocator.release_memory_chunks(non_zero_values_in_kv_memory)
            self.request_to_kv_chunk_ids[finished_idxs].fill_(-1)

            if active_request_count > 0:
                finished_idxs_on_left = (
                    torch.nonzero(active_requests_mask[:active_request_count] == 0, as_tuple=True)[
                        0
                    ]
                    + self.paused_request_count
                )
                active_idxs_on_right = (
                    torch.nonzero(active_requests_mask[active_request_count:], as_tuple=True)[0]
                    + active_request_count
                    + self.paused_request_count
                )

                self._swap_book_keeping_tensors(
                    src_idxs=active_idxs_on_right,
                    dst_idxs=finished_idxs_on_left,
                    next_tokens=next_tokens,
                )

        # 5. We identify requests that require a new chunk and add them to the paused requests (i.e move them left) :-
        #       a) Put requests that have filled their current chunk and  require a new one in a pause state temporarily
        #       b) Move the paused requests to the left, and active requets to the right
        #       c) Update the paused request count and active_request_count appropriately
        if active_request_count > 0:
            num_tokens_in_last_chunk = self.request_last_kv_chunk_offset[
                self.paused_request_count : (active_request_count + self.paused_request_count)
            ]
            active_requests_requiring_new_chunk = (
                num_tokens_in_last_chunk == self.chunk_size_tokens - 1
            ).byte()
            active_requests_requiring_new_chunk_count = (
                (active_requests_requiring_new_chunk == 1).sum().item()
            )

            # Swap unfinished active requests on the left side with paused requests on the right side
            # NOTE : We add paused request count because we concatenate
            # paused tokens to the left at the beginning of update requests
            if (
                active_requests_requiring_new_chunk_count > 0
                and active_requests_requiring_new_chunk_count != active_request_count
            ):
                active_request_ids_on_left = (
                    torch.nonzero(
                        active_requests_requiring_new_chunk[
                            :active_requests_requiring_new_chunk_count
                        ]
                        == 0,
                        as_tuple=True,
                    )[0]
                    + self.paused_request_count
                )
                paused_requests_idxs_on_right = (
                    torch.nonzero(
                        active_requests_requiring_new_chunk[
                            active_requests_requiring_new_chunk_count:
                        ],
                        as_tuple=True,
                    )[0]
                    + active_requests_requiring_new_chunk_count
                    + self.paused_request_count
                )
                dst_idxs = torch.cat((active_request_ids_on_left, paused_requests_idxs_on_right))
                src_idxs = torch.cat((paused_requests_idxs_on_right, active_request_ids_on_left))
                self._swap_book_keeping_tensors(
                    src_idxs=src_idxs, dst_idxs=dst_idxs, next_tokens=next_tokens
                )

            self.paused_request_count += active_requests_requiring_new_chunk_count
            active_request_count -= active_requests_requiring_new_chunk_count

        # 6. Now that we have the requests in following order [Paused, Active, Finished]
        # We determine how many requests we can resume and resume them
        # Assign released chunks to paused requests.
        # todo: @shanmugamr, un-pause requests using FIFO, rather than LIFO.
        if (
            self.chunk_allocator.chunk_count_avail
            <= self.paused_request_count + self.gtd_chunk_count
        ):
            if active_request_count < self.gtd_request_count:
                resume_request_count = min(
                    self.paused_request_count, self.gtd_request_count - active_request_count
                )
            else:
                # If there are more active requests than gtd requests and not enough
                # chunks available, no requests can be resumed
                resume_request_count = 0
        else:
            # If there are more available chunks than (paused + gtd requests), resume all paused requests
            resume_request_count = self.paused_request_count

        self.paused_request_count -= resume_request_count
        active_request_count += resume_request_count
        assert active_request_count > 0, "active_request_count == %d." % active_request_count

        # 7. We make changes to the request book keeping tesnsors and setup the tokens for next iteration
        self.total_request_count = active_request_count + self.paused_request_count
        # All these active requests are in decode phase, so they need only 1 token per request
        self.active_token_count = active_request_count
        # Always the first section of token input ids are only used.
        self.token_to_input_ids[: self.active_token_count] = next_tokens[
            self.paused_request_count : self.total_request_count
        ]

        if self.paused_request_count > 0:
            self.paused_tokens = next_tokens[: self.paused_request_count]

        self.request_kv_length_offsets[self.paused_request_count : self.total_request_count].add_(
            self.request_query_lengths[self.paused_request_count : self.total_request_count]
        )
        self.request_query_lengths[self.paused_request_count : self.total_request_count].fill_(1)
        self.token_to_pos_ids[: self.active_token_count] = self.request_kv_length_offsets[
            self.paused_request_count : self.total_request_count
        ]

        self.request_last_kv_chunk_offset[self.paused_request_count : self.total_request_count] = (
            self.request_last_kv_chunk_offset[self.paused_request_count : self.total_request_count]
            + 1
        ) % self.chunk_size_tokens

        # 8. We resume those requests by assigning chunks and updating bookkeeping tensors
        if resume_request_count > 0:
            assert torch.all(
                self.request_last_kv_chunk_offset[
                    self.paused_request_count : (self.paused_request_count + resume_request_count)
                ]
                == 0
            ), 'The request_last_kv_chunk_offset should be 0 for the requests that just got resumed this step. '

            chunk_ids = self.chunk_allocator.allocate_memory_chunks(resume_request_count)
            row_idx = torch.arange(
                self.paused_request_count,
                self.paused_request_count + resume_request_count,
                device=torch.cuda.current_device(),
            )
            col_idx = self.request_kv_chunk_counts[
                self.paused_request_count : (self.paused_request_count + resume_request_count)
            ]
            self.request_to_kv_chunk_ids[row_idx, col_idx] = chunk_ids
            self.request_kv_chunk_counts[
                self.paused_request_count : (self.paused_request_count + resume_request_count)
            ] += 1
            self.request_last_kv_chunk_id[
                self.paused_request_count : (self.paused_request_count + resume_request_count)
            ] = chunk_ids

        # 9. We make relevant changes to the token bookkeeping tensors
        self.token_to_request_idx[: self.active_token_count] = torch.arange(
            self.paused_request_count, self.total_request_count, device=torch.cuda.current_device()
        )
        self.token_to_position_in_request[: self.active_token_count] = (
            self.request_kv_length_offsets[self.paused_request_count : self.total_request_count]
        )

        self.token_to_chunk_idx[: self.active_token_count] = self.request_last_kv_chunk_id[
            self.paused_request_count : self.total_request_count
        ]
        self.token_to_local_position_within_kv_chunk[: self.active_token_count] = (
            self.request_last_kv_chunk_offset[self.paused_request_count : self.total_request_count]
        )
