# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import math
import warnings
from enum import Enum
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from packaging.version import Version as PkgVersion
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.package_info import __version__ as mcore_version
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import divide as core_divide

from .base_context import BaseInferenceContext
from .dynamic_chunk_allocator import ChunkAllocator

try:
    from packaging.version import Version as PkgVersion

    HAVE_PACKAGING = True
except:
    HAVE_PACKAGING = False


class ContextOverflowError(Exception):
    """Base exception for when a new request does not fit.

    Args:
        is_transient (bool): Flag marking whether error is transient (i.e., may
            work if we try again, but fails due to the current context state), or
            permanent (i.e., request will never fit in this context).
    """

    def __init__(
        self, request_id: Optional[int], message: Optional[str] = None, *, is_transient: bool = True
    ):
        request_str = '--' if request_id is None else str(request_id)
        message = "" if message is None else f" | {message}"
        super().__init__(f"request {request_str}{message}")
        self.is_transient = is_transient


class RequestOverflowError(ContextOverflowError):
    """Adding request would overflow max request count."""

    pass


class TokenOverflowError(ContextOverflowError):
    """Adding request would overflow max token count."""

    pass


class MaxSequenceLengthOverflowError(ContextOverflowError):
    """Adding request would overflow max sequence length."""

    def __init__(self, request_id, message: Optional[str] = None):
        super().__init__(request_id, message=message, is_transient=False)


class ChunkOverflowError(ContextOverflowError):
    """Adding request would overflow available memory chunks."""

    pass


class ActiveRequestCountOverflowError(ContextOverflowError):
    '''Used when `initialize_attention_state()` is called with
    `num_warmup_requests > max_requests.'''

    def __init__(self, max_request_count, active_request_count):
        assert active_request_count > max_request_count
        super().__init__(
            None,
            "active_request_count (%d) > max_request_count (%d)."
            % (active_request_count, max_request_count),
        )


class WarmupEngineMode(Enum):
    """Enumeration for warmup engine modes used during cuda graph capture."""

    DECODE = "decode"
    NON_DECODE = "non_decode"


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
        tensor_model_parallel_size (Optional[int]): Tensor model parallel size.
        num_cuda_graphs (Optional[int]): Maximum number of cuda graphs to capture,
            where the cuda graph batch sizes range from 1 to `max_requests` (as
            computed below). Due to rounding, the actual number of cuda graphs may
            not equal this argument.
        materialize_only_last_token_logits (bool): If True, only the last token logits
            are materialized in the context.
        use_cuda_graphs_for_non_decode_steps (bool): If True, use cuda graphs for non-decode
        engine steps.
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
        cache_mla_latent: bool = False,
        kv_lora_rank: Optional[int] = None,
        qk_pos_emb_head_dim: Optional[int] = None,
        num_cuda_graphs: Optional[int] = None,
        materialize_only_last_token_logits: bool = True,
        use_cuda_graphs_for_non_decode_steps: bool = True,
    ):
        super().__init__(materialize_only_last_token_logits=materialize_only_last_token_logits)

        self.cache_mla_latent = cache_mla_latent
        if self.cache_mla_latent:
            assert (
                chunk_size_tokens == 64
            ), "Flash MLA requires a block size of 64. Set --inference-dynamic-batching-chunk-size 64 to fix this assert"

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
        if self.cache_mla_latent:
            #   one vector  c_t  (rank)  +  optional RoPE phase slice
            kv_reduced_dim = kv_lora_rank + qk_pos_emb_head_dim
            self.kv_reduced_dim = kv_reduced_dim
            self.chunk_size_bytes = (
                dtype_size_bytes * num_layers * self.chunk_size_tokens * kv_reduced_dim
            )
        else:
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
            return self.round_up_requests(int(n_requests), tp_size=tp_size), self.round_up_tokens(
                int(n_tokens), tp_size=tp_size
            )

        self.max_requests, self.max_tokens = bytes_to_max_requests_and_tokens(buffer_size_bytes)
        if buffer_overflow_factor is not None:
            self.max_requests = self.round_up_requests(
                int(self.max_requests * buffer_overflow_factor), tp_size=tp_size
            )
            self.max_tokens = self.round_up_tokens(
                int(self.max_tokens * buffer_overflow_factor / 50.0), tp_size=tp_size
            )

        if max_requests_override is not None:
            self.max_requests = self.round_up_requests(max_requests_override, tp_size=tp_size)

        if max_tokens_override is not None:
            self.max_tokens = self.round_up_tokens(max_tokens_override, tp_size=tp_size)

        self.max_requests = min(self.max_requests, self.max_tokens)  # e.g., decode only.

        # Initialize context state.
        self.params_dtype = params_dtype
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length

        self.total_request_count = 0
        self.active_token_count = 0
        self.paused_request_count = 0
        self.padded_active_token_count = None
        self.padded_active_request_count = None
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
        if cache_mla_latent:
            self.memory_buffer = torch.full(
                (self.num_layers, chunk_count_total, self.chunk_size_tokens, kv_reduced_dim),
                -1,
                dtype=self.params_dtype,
                device=torch.cuda.current_device(),
            )
        else:
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

        # Cuda graph token-counts (i.e., token counts used by cuda-graph steps, both decode and non-decode).
        self.cuda_graph_token_counts = None
        if num_cuda_graphs is not None:

            # Ensure valid num_cuda_graphs.
            num_cuda_graphs = min(max(num_cuda_graphs, 1), self.max_requests)

            # Cuda graph step size.
            cuda_graph_rounder = 8
            self.cuda_graph_step_size = self.max_requests / num_cuda_graphs
            self.cuda_graph_step_size = (
                math.ceil(self.cuda_graph_step_size / cuda_graph_rounder) * cuda_graph_rounder
            )
            # Make sure divisble by TP size
            self.cuda_graph_step_size = math.ceil(self.cuda_graph_step_size / tp_size) * tp_size

            # Cuda graph token counts.
            if num_cuda_graphs == 1:
                self.cuda_graph_token_counts = [self.max_requests]
            else:
                self.cuda_graph_token_counts = list(
                    range(self.cuda_graph_step_size, self.max_requests, self.cuda_graph_step_size)
                )
                if self.cuda_graph_token_counts[-1] != self.max_requests:
                    self.cuda_graph_token_counts.append(self.max_requests)
                self.cuda_graph_token_counts.reverse()

            # Set used for validating active cuda graph token count.
            self.cuda_graph_token_counts_set = set(self.cuda_graph_token_counts)
            self.max_cuda_graph_token_count = max(self.cuda_graph_token_counts)

        # for backwards compatibility with legacy unit tests, we are keeping
        # self.cuda_graph_request_counts around.
        self.cuda_graph_request_counts = self.cuda_graph_token_counts

        self.non_decode_cuda_graphs = use_cuda_graphs_for_non_decode_steps and (
            num_cuda_graphs is not None
        )

        # `*_cudagraph_only` tensors are for use with cuda graphs to maintain
        # consistent input shapes, which is required to use cuda graphs.
        # During these steps, the `*_cudagraph_only`
        # tensors are used, otherwise their same-name but un-suffixed
        # corresponding tensors are used.

        self.query_seq_lengths_cudagraph_only = torch.full(
            (self.max_requests,), 0, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.cu_query_seq_lengths_cudagraph_only = torch.full(
            (self.max_requests + 1,), 0, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.kv_seq_lengths_cudagraph_only = torch.full(
            (self.max_requests,), 0, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.cu_kv_seq_lengths_cudagraph_only = torch.full(
            (self.max_requests + 1,), 0, dtype=torch.int32, device=torch.cuda.current_device()
        )

        self.request_to_kv_chunk_ids_cudagraph_only = torch.full(
            (self.max_requests, self.max_kv_chunk_count),
            0,
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )

        # Guaranteed active requests.
        # * See details in the class docstring above. `gtd_request_fraction` is
        #   the fraction of chunks in the memory buffer that are reserved for
        #   guaranteeing that some number of active requests can always proceed
        #   with their generations. The number of chunks defined by
        #   `buffer_guaranteed_fraction * chunk_count_total` is converted to a
        #   number of requests that this reserved space can safely handle
        #   (`gtd_request_count`).
        # * Note: computing the size of this guaranteed space from chunks rather
        #   than bytes is safer due to the non-linear impacts of a large
        #   `chunk_size_tokens` or `max_kv_chunk_count`. When computing from
        #   chunks, this space will always be less than `chunk_count_total`. When
        #   computing from bytes, this space can unexpectedly be much larger than
        #   `chunk_count_total`, resulting in stalled generations.
        gtd_chunk_count = int(buffer_guaranteed_fraction * chunk_count_total)
        gtd_chunk_count = min(gtd_chunk_count, chunk_count_total)
        self.gtd_request_count = max(1, gtd_chunk_count // self.max_kv_chunk_count)
        self.gtd_chunk_count = self.gtd_request_count * self.max_kv_chunk_count

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
    def round_up_tokens(cls, value, tp_size=None):
        """Round up to nearest multiple of `TOKEN_ROUNDER` (above) that is also divisible by tensor model parallel size."""
        if not HAVE_PACKAGING:
            raise ImportError(
                "`packaging` is required for this functionality, please install it with `pip install packaging`"
            )
        if PkgVersion(mcore_version) < PkgVersion("0.13"):
            return cls.round_up(value)

        # Make sure divisible by TP size
        if tp_size is None:
            # Check if parallel state is initialized before trying to get TP size
            if parallel_state.is_initialized():
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
            else:
                tp_size = 1
        token_rounder = math.ceil(cls.TOKEN_ROUNDER / tp_size) * tp_size

        return token_rounder * int(math.ceil(int(value) / token_rounder))

    @classmethod
    def round_up_requests(cls, value, tp_size=None):
        """Round up to nearest multiple of `REQUEST_ROUNDER` (above) that is also divisible by tensor model parallel size."""
        if not HAVE_PACKAGING:
            raise ImportError(
                "`packaging` is required for this functionality, please install it with `pip install packaging`"
            )
        if PkgVersion(mcore_version) < PkgVersion("0.13"):
            return cls.round_up(value)

        # Make sure divisible by TP size
        if tp_size is None:
            # Check if parallel state is initialized before trying to get TP size
            if parallel_state.is_initialized():
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
            else:
                tp_size = 1
        request_rounder = math.ceil(cls.REQUEST_ROUNDER / tp_size) * tp_size

        return request_rounder * int(math.ceil(int(value) / request_rounder))

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

    def cu_query_lengths(self) -> Tuple[Tensor, int]:
        """Cumulative query sequence lengths."""
        return self.cu_query_seq_lengths, self.max_seqlen_q

    def cu_kv_lengths(self) -> Tensor:
        """Cumulative key/value sequence lengths."""
        return (self.cu_kv_seq_lengths, self.kv_seq_lengths, self.max_seqlen_k)

    def get_active_sequence_lengths(self) -> Tensor:
        """Total sequence length (query + key) for active requests."""
        lengths = self.request_kv_length_offsets + self.request_query_lengths
        lengths = lengths[self.paused_request_count : self.total_request_count]
        return lengths

    def get_max_sequence_lengths(self) -> Tensor:
        """Maximum sequence length for active requests."""
        return self.request_output_lengths[self.paused_request_count : self.total_request_count]

    def get_active_request_count(self):
        """Returns the current number of active requests."""
        active_sequence_lengths = self.get_active_sequence_lengths()
        max_sequence_lengths = self.get_max_sequence_lengths()
        active_requests_mask = torch.less(active_sequence_lengths, max_sequence_lengths).byte()
        active_request_count = (active_requests_mask == 1).sum().item()
        return active_request_count

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

        if not self.cache_mla_latent:
            assert key.size(1) == 1 and value.size(1) == 1

        key = key.squeeze(1)
        # There is no value cache in FlashMLA/absorption
        if not self.cache_mla_latent:
            value = value.squeeze(1)

        if self.cache_mla_latent:
            # We pass the kv_concat as the key in cache_mla_latent
            kv_concat = key
            self.memory_buffer[layer_number - 1, chunk_idx, local_kv_seq_idx] = kv_concat[
                : self.padded_active_token_count
            ]
        else:
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
        if self.cache_mla_latent:
            return (self.memory_buffer[layer_number - 1], None, self.block_table)
        else:
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
        mscale: float = 1.0,
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
            mscale=mscale,
        )
        return query

    def apply_rotary_emb_key(
        self,
        key: Tensor,
        key_emb: Tensor,
        config: TransformerConfig,
        cp_group: torch.distributed.ProcessGroup,
        mscale: float = 1.0,
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
            assert key.shape[0] == n
            key = apply_rotary_pos_emb(
                t=key[:n], freqs=key_emb[:n], config=config, cp_group=cp_group, mscale=mscale
            )
        else:
            key[:n] = apply_rotary_pos_emb(
                t=key[:n], freqs=key_emb[:n], config=config, cp_group=cp_group, mscale=mscale
            )
        return key

    def reset_attention_state(self) -> None:
        """Reset state used within attention, after each step."""
        self.max_seqlen_q = None
        self.max_seqlen_k = None
        self.cu_query_seq_lengths = None
        self.cu_query_seq_lengths_cudagraph_only.fill_(0)
        self.query_seq_lengths_cudagraph_only.fill_(0)
        self.cu_kv_seq_lengths = None
        self.cu_kv_seq_lengths_cudagraph_only.fill_(0)
        self.kv_seq_lengths = None
        self.kv_seq_lengths_cudagraph_only.fill_(0)
        self.request_to_kv_chunk_ids_cudagraph_only.fill_(0)
        self.block_table = None

    def using_cuda_graph_this_step(self) -> bool:
        """Returns True if cuda graphs are being used for this step."""
        has_cuda_graphs = self.cuda_graph_token_counts is not None
        can_use_cuda_graphs = self.is_decode_only() or self.non_decode_cuda_graphs
        token_count_fits_cuda_graph = has_cuda_graphs and (
            self.active_token_count <= self.max_cuda_graph_token_count
        )
        return has_cuda_graphs and can_use_cuda_graphs and token_count_fits_cuda_graph

    def initialize_attention_state(
        self,
        *,
        num_warmup_tokens: Optional[int] = None,
        warmup_engine_mode: WarmupEngineMode = WarmupEngineMode.DECODE,
        num_warmup_requests: Optional[int] = None,
    ) -> None:
        """Initialize attention state so that every layer can use it.

        Args:
            num_warmup_tokens (Optional[int]): Number of tokens to use for
                warming up cuda graphs. Must be less than or equal to
                `max_requests`.
            warmup_engine_mode (WarmupEngineMode): Denote whether to setup
            for a decode or a non-decode cuda-graph warmup.
            num_warmup_requests (Optional[int]): [DEPRECATED] Use num_warmup_tokens instead.
            This argument is kept for backward compatibility with the legacy API.
        Return:
            None.
        """
        if num_warmup_requests is not None:
            warnings.warn(
                "The 'num_warmup_requests' argument is deprecated and will be removed in a future release. "
                "Please use 'num_warmup_tokens' instead.",
                DeprecationWarning,
            )
            # If num_warmup_tokens is not provided, use num_warmup_requests for backward compatibility
            if num_warmup_tokens is None:
                num_warmup_tokens = num_warmup_requests

        # warmup both decode and non-decode engine steps
        if num_warmup_tokens is not None:
            if num_warmup_tokens > self.max_requests:
                raise ActiveRequestCountOverflowError(self.max_requests, num_warmup_tokens)

            if warmup_engine_mode == WarmupEngineMode.NON_DECODE:
                assert self.non_decode_cuda_graphs, "Set non-decode cuda graphs to True"
                # Create a mismatch between self.active_token_count (0) and self.active_request_count
                # (which is 0 by default) so that self.is_decode() is False and we trigger
                # the non-decode attention kernel. The value of 1 is not special by any means, all
                # we need is for self.total_request_count to not be 0.
                self.total_request_count = 1

        active_token_count = (
            self.active_token_count if num_warmup_tokens is None else num_warmup_tokens
        )

        if self.using_cuda_graph_this_step():
            self.padded_active_token_count = (
                math.ceil(active_token_count / self.cuda_graph_step_size)
                * self.cuda_graph_step_size
            )
            self.padded_active_token_count = min(self.padded_active_token_count, self.max_requests)
            assert self.padded_active_token_count in self.cuda_graph_token_counts_set
            assert self.padded_active_token_count >= active_token_count
        else:
            self.padded_active_token_count = self.round_up_tokens(self.active_token_count)
            if self.is_decode_only():
                # For decode-only, the padded active token count cannot exceed max-requests.
                self.padded_active_token_count = min(
                    self.padded_active_token_count, self.max_requests
                )

        # How are we calculating the padded active request count?
        # Case 1: Using cuda graphs:
        #         It is always the same as padded_active_token_count, whether its decode or non-decode.
        #         We make the q and kv-lengths of extra padded requests to be 0.
        # Case 2: Not using cuda graphs:
        #         If decode - we set it to padded_active_token_count. Again q and kv lengths of extra padded requests will be 0.
        #         If non-decode - we set it to total_request_count - paused_request_count i.e. no padded requests.
        self.padded_active_request_count = (
            self.padded_active_token_count
            if self.using_cuda_graph_this_step() or self.is_decode_only()
            else (self.total_request_count - self.paused_request_count)
        )

        # Update token position indexes.
        self.token_to_chunk_idx[self.active_token_count : self.padded_active_token_count] = (
            self.dummy_chunk_idx
        )
        self.token_to_local_position_within_kv_chunk[
            self.active_token_count : self.padded_active_token_count
        ] = 0
        self.token_to_position_in_request[
            self.active_token_count : self.padded_active_token_count
        ] = 0

        # Update cu_query_seq_lengths, max_seqlen_q.
        query_lengths = self.request_query_lengths[
            self.paused_request_count : self.total_request_count
        ]
        if self.is_decode_only() or self.using_cuda_graph_this_step():
            self.query_seq_lengths_cudagraph_only[
                0 : self.total_request_count - self.paused_request_count
            ] = query_lengths
            if self.is_decode_only():
                self.cu_query_seq_lengths = None  # ensure no accidental use
                self.max_seqlen_q = 1
            else:
                self.cu_query_seq_lengths_cudagraph_only[
                    1 : self.padded_active_request_count + 1
                ] = torch.cumsum(
                    self.query_seq_lengths_cudagraph_only[: self.padded_active_request_count], dim=0
                )

                # The following will be passed to the FA kernel.
                self.cu_query_seq_lengths = self.cu_query_seq_lengths_cudagraph_only[
                    : (self.padded_active_request_count + 1)
                ]
                self.max_seqlen_q = self.padded_active_token_count
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
        if self.is_decode_only() or self.using_cuda_graph_this_step():
            # Re-assign `kv_seq_lengths` to be a view of the first
            # `active_cuda_graph_request_count` tokens of `kv_seq_lengths_decode_only`,
            # such that `kv_seq_lengths` has a static memory address and is therefore
            # cuda graph compatible. This allows `kv_seq_lengths` to transition between,
            # cuda graph sizes, which makes multi-batch-size cuda graphs possible.
            self.kv_seq_lengths_cudagraph_only[
                0 : self.total_request_count - self.paused_request_count
            ] = self.kv_seq_lengths
            self.kv_seq_lengths = self.kv_seq_lengths_cudagraph_only[
                : self.padded_active_request_count
            ]
            self.max_seqlen_k = self.max_sequence_length
            if self.is_decode_only():
                self.cu_kv_seq_lengths = None  # ensure no accidental use
            else:
                cu_kv_lengths = torch.cumsum(self.kv_seq_lengths, dim=0)
                # The following will be passed to the FA kernel.
                self.cu_kv_seq_lengths_cudagraph_only[1 : cu_kv_lengths.size(0) + 1] = cu_kv_lengths
                self.cu_kv_seq_lengths = self.cu_kv_seq_lengths_cudagraph_only[
                    : (self.padded_active_request_count + 1)
                ]
        else:
            self.cu_kv_seq_lengths = torch.full(
                (self.total_request_count - self.paused_request_count + 1,),
                0,
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            self.cu_kv_seq_lengths[1:] = torch.cumsum(self.kv_seq_lengths, dim=0)
            self.max_seqlen_k = self.kv_seq_lengths.max().item()

        # Update KV chunk IDs, block table.
        request_to_kv_chunk_ids = self.request_to_kv_chunk_ids[
            self.paused_request_count : self.total_request_count
        ]
        if self.is_decode_only() or self.using_cuda_graph_this_step():
            self.request_to_kv_chunk_ids_cudagraph_only[
                0 : self.total_request_count - self.paused_request_count
            ] = request_to_kv_chunk_ids
            self.block_table = self.request_to_kv_chunk_ids_cudagraph_only[
                : self.padded_active_request_count
            ]
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
        self.padded_active_request_count = 0
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

    def current_input_and_position_ids(
        self, *, num_warmup_tokens: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """Flattened input and position IDs for forward pass.

        Args:
            num_warmup_tokens (Optional[int]): Number of tokens to return for
                warming up cuda graphs. Must be less than or equal to
                `max_tokens`.

        Return:
            (Tuple[Tensor, Tensor]) Flattened active input and position IDs.
        """
        num_tokens = num_warmup_tokens or self.padded_active_token_count
        return (
            self.token_to_input_ids[:num_tokens].unsqueeze(0),
            self.token_to_pos_ids[:num_tokens].unsqueeze(0),
        )

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
        context_length = tokens.numel()
        if context_length > self.max_tokens:
            # **Note**: for `megatron-core >= 0.15`, this assert should be
            # `is_transient=False`. For backwards compatibility with legacy tests,
            # this must be `True` for one version cycle of `megatron-core`.
            raise TokenOverflowError(
                request_id,
                # is_transient=False, # use this in megatron-core 0.16
                is_transient=True,  # used temporarily for legacy tests
            )

        # Test for token and request overflow.
        # TODO : Should move this into some waiting queue
        if self.active_token_count + context_length > self.max_tokens:
            raise TokenOverflowError(request_id)
        if self.total_request_count >= self.max_requests:
            raise RequestOverflowError(request_id)

        # Preallocate chunks.
        num_chunks_needed = math.ceil(context_length / self.chunk_size_tokens)
        new_chunk_ids = self.chunk_allocator.allocate_memory_chunks(num_chunks_needed, safe=True)
        if new_chunk_ids is None:
            raise ChunkOverflowError(request_id)

        if num_tokens_to_generate is None:
            num_tokens_to_generate = self.max_sequence_length - context_length
        elif context_length + num_tokens_to_generate > self.max_sequence_length:
            raise MaxSequenceLengthOverflowError(request_id)

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

    def _move_book_keeping_tensors(self, src_idxs, dst_idxs, next_tokens):
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
    def update_requests(self, active_requests_mask: Tensor, new_tokens: Tensor) -> Tensor:
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
            (Tensor) Newly paused request IDs.
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

            # Reset the KV chunks for finished requests.
            # Note: do not use fill_() (or add_() and similar inplace ops) here.
            # The combinition of indexing with a tensor (like finished_idxs) and fill_()/add_() creates a clone
            # and updates it instead of the original tensor.
            self.request_to_kv_chunk_ids[finished_idxs] = -1

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

                self._move_book_keeping_tensors(
                    src_idxs=active_idxs_on_right,
                    dst_idxs=finished_idxs_on_left,
                    next_tokens=next_tokens,
                )

                # Reset chunk ids for recently moved requests.
                self.request_to_kv_chunk_ids[active_idxs_on_right] = -1

        # 5. We identify requests that require a new chunk and add them to the paused requests (i.e move them left) :-
        #       a) Put requests that have filled their current chunk and  require a new one in a pause state temporarily
        #       b) Move the paused requests to the left, and active requets to the right
        #       c) Update the paused request count and active_request_count appropriately
        newly_paused_request_ids = None
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
                self._move_book_keeping_tensors(
                    src_idxs=src_idxs, dst_idxs=dst_idxs, next_tokens=next_tokens
                )
                newly_paused_request_ids = self.request_ids[dst_idxs]

            self.paused_request_count += active_requests_requiring_new_chunk_count
            active_request_count -= active_requests_requiring_new_chunk_count

        # 6. Now that we have the requests in following order [Paused, Active, Finished]
        # We determine how many requests we can resume and resume them
        # Assign released chunks to paused requests.
        # todo: @shanmugamr, un-pause requests using FIFO, rather than LIFO.
        num_non_gtd_chunks = max(0, self.chunk_allocator.chunk_count_avail - self.gtd_chunk_count)
        if num_non_gtd_chunks:
            # if we have non-gtd chunks, use them. Do not dip into the gtd-chunk pool
            resume_request_count = min(num_non_gtd_chunks, self.paused_request_count)
        else:
            # only dip into the gtd-chunk pool if we have run out of non-gtd-chunks and the active
            # request count has fallen below a certain threshold.
            resume_request_count = min(
                max(self.gtd_request_count - active_request_count, 0), self.paused_request_count
            )

        self.paused_request_count -= resume_request_count
        active_request_count += resume_request_count
        assert active_request_count > 0, "active_request_count == %d." % active_request_count

        # Remove resumed requests from newly_paused_request_ids. We do this by
        # truncating the end of newly_paused_request_ids, which works because we
        # resume requests in LIFO order. If resume_request_count >
        # len(newly_paused_request_ids), this means that none of the paused
        # requests are newly paused during this update.
        if newly_paused_request_ids is not None and resume_request_count > 0:
            newly_paused_request_ids = newly_paused_request_ids[:-resume_request_count]

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

        # add_ and fill_ calls seems to work as intended with sliced indexing (i.e. x[3:5].add(...) or x[3:5].fill_)
        # but when another tensor is used for indexing, it does not work as expected (i.e. x[y] if x and y are torch tensors)
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
            ), "The request_last_kv_chunk_offset should be 0 for the requests that just got resumed this step. "

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

        return newly_paused_request_ids

    def calculate_log_probs(
        self, logits: Tensor, new_tokens: Tensor, only_last_token_logits: Optional[bool] = False
    ) -> List[List[float]]:
        """Calculate log probs for all active requests and return them.

        TODO: @wdykas support top-n log probs.

        Args:
            logits (Tensor): Raw model output logits with shape [1, sequence_length, vocab_size].
            new_tokens (Tensor): The newly sampled tokens.
            only_last_token_logits (bool): If set, the logits are from only the last token in each request

        Returns:
            List of lists where each inner list contains log probs for a request in the
            same order as the active requests (from paused_request_count to total_request_count).
        """
        # Calculate log_probs (sequence_length x vocab_size)
        log_probs = F.log_softmax(logits.squeeze(0).float(), dim=-1)

        if only_last_token_logits or self.is_decode_only():
            seq_idx = torch.arange(len(new_tokens), dtype=torch.int32, device=logits.device)
            selected_log_probs = log_probs[seq_idx, new_tokens]
            return [[lp] for lp in selected_log_probs.flatten().tolist()]

        # Get the selected token ids for all tokens.
        # We shift the active token window left by one to remove the first prompt token for
        # prefill requests and then set the token ids explicitly for the newly generated tokens.
        # This is necessary because we calculate the log probs *before* updating the request metadata.
        #
        # Example (decode & prefill mix):
        #
        #   active_query_lengths: [ 1 | 1 | 2 | 5 ]
        #
        #   new_tokens          : [ 52 | 12 | 3 | 86 ]
        #
        #   seq_idx             : [ 0 | 1 | 2 3 | 4 5 6 7 8 ]
        #
        #   new_token_idx       : [ 0 | 1 | 3 | 8 ]
        #
        #   active_token_ids before left shift:
        #                       : [ 31 | 75 | 45 16 | 90 12 72 24 88 ]
        #
        #   active_token_ids after shift:
        #                       : [ XX | XX | 16 XX | 12 72 24 88 XX ]   (XX = undefined)
        #
        #   active_token_ids[new_token_idx] = new_tokens
        #                       : [ 52 | 12 | 16  3 | 12 72 24 88 86 ]
        active_token_ids = self.token_to_input_ids[: self.active_token_count].roll(-1, 0)
        active_query_lengths = self.request_query_lengths[
            self.paused_request_count : self.total_request_count
        ]
        new_token_idx = active_query_lengths.cumsum(0) - 1
        active_token_ids[new_token_idx] = new_tokens

        # Extract the log probs for only the selected tokens.
        # (sequence_length x vocab_size) -> (sequence_length)
        seq_idx = torch.arange(self.active_token_count, device=log_probs.device)
        selected_log_probs = log_probs[seq_idx, active_token_ids]

        # Split the log probs across request boundaries
        selected_log_probs_list = selected_log_probs.cpu().split(
            active_query_lengths.tolist(), dim=0
        )

        # Convert each log prob tensor into a list
        return [lp.tolist() for lp in selected_log_probs_list]
