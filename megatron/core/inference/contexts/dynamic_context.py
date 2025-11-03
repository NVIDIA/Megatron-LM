# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
import math
import warnings
from contextlib import nullcontext
from enum import Enum, IntEnum
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from packaging.version import Version as PkgVersion
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.transformer.enums import AttnBackend
from megatron.core.inference.utils import CUDAGraphConfig
from megatron.core.inference.unified_memory import create_unified_mempool, has_unified_memory
from megatron.core.inference.utils import tensor_swap
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.package_info import __version__ as mcore_version
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import divide as core_divide
from megatron.core.inference.sampling_params import SamplingParams

from .attention_context.mha_metadata import GraphMHAMetadata, NonGraphMHAMetadata
from .attention_context.mha_splitpd_metadata import MHASplitPDMetadata
from .attention_context.mha_flashinfer_metadata import (
    GraphMHAFlashInferMetadata,
    NonGraphMHAFlashInferMetadata
)
from .base_context import BaseInferenceContext
from .context_metadata import SyncContextMetadata, AsyncContextMetadata
from .dynamic_block_allocator import BlockAllocator
from ..kv_cache import KVCacheBase, KVCacheLayout, MLACache, create_mhagqa_cache

try:
    from .fused_kv_append_kernel import triton_append_key_value_cache
except ImportError:
    triton_append_key_value_cache = None

logger = logging.getLogger(__name__)

try:
    from packaging.version import Version as PkgVersion

    HAVE_PACKAGING = True
except:
    HAVE_PACKAGING = False

try:
    import flashinfer  # pylint: disable=unused-import

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False


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


class BlockOverflowError(ContextOverflowError):
    """Adding request would overflow available memory blocks."""

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
    memory buffer for the block-level KV cache. For in-flight batching, requests of
    arbitrary sequence length may be added, paused, or removed from the context
    at any step. The only constraint is the maximum number of requests or tokens
    that the context is defined to support. For the block-level KV cache, a memory
    buffer is allocated up front (size `buffer_size_gb`), that is divided into
    blocks and dynamically assigned to requests. At any given step, any unassigned
    blocks equate to unused space.

    Additionally, a fraction of the memory buffer (`gtd_request_fraction`, i.e.,
    the 'guaranteed' request fraction) is reserved for guaranteeing that a
    minimum number of active requests may continue to generate tokens on any step.
    The reason for this is that the context manages two pools of requests: 1)
    active requests, and 2) paused requests. Paused requests are requests where
    insufficient memory blocks remain for future assignment, and these requests
    are set aside until enough memory blocks are available. Active requests are
    requests that have sufficient memory blocks to proceed with their generations.

    The situation can arise where all requests eventually become paused due to all
    memory blocks being assigned. In this case, there are no active requests and
    thus no progress can be made. To handle this case, a fraction of the memory
    buffer is reserved that only allows active requests, and no paused requests.
    This fraction must be carefully tuned, as it can have an order of magnitude
    impact on overall latency.

    Args:
        params_dtype (torch.dtype): Dtype used for KV cache.
        num_layers (int): Number of layers.
        kv_channels (int): Hidden dimension per attention head.
        num_attention_kv_heads (int): Number of key/value attention heads.
        num_attention_qo_heads (int): Number of query/output attention heads.
        max_sequence_length (int): Max possible sequence length (prompt + output)
            that will occur.
        buffer_size_gb (float): Total buffer size (GB), shared by main and
            fallback contexts.
        block_size_tokens (int): Size of KV cache block size.
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
        unified_memory_level (Optional[int]): Set unified memory usage within the
            dynamic inference context. The levels are: 0) no unified memory, 1)
            allocate `memory_buffer` in unified memory. Eventually, additional
            levels will be included to control other tensors within the context.
        use_flashinfer_fused_rope (bool): If True, use flashinfer's fused rope implementation.
        If None, defaults to using flash-infer if available.
        attention_backend (AttnBackend): Attention backend to use. Defaults to AttnBackend.flash.
    """

    def __init__(
        self,
        *,
        params_dtype: torch.dtype,
        num_layers: int,
        kv_channels: int,
        num_attention_kv_heads: int,
        num_attention_qo_heads: int,
        max_sequence_length: int,
        buffer_size_gb: float,
        buffer_guaranteed_fraction: float,
        block_size_tokens: int = 256,
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
        use_flashinfer_fused_rope: bool = False,
        unified_memory_level: Optional[int] = 0,
        cuda_graph_max_tokens: Optional[int] = None,
        cuda_graph_max_prefill_requests: Optional[int] = 16,
        attention_backend: AttnBackend = AttnBackend.flash,
        enable_async_scheduling: bool = False,
    ):
        super().__init__(materialize_only_last_token_logits=materialize_only_last_token_logits)

        # Store async scheduling flag
        self.enable_async_scheduling = enable_async_scheduling

        self.attention_backend = attention_backend
        self.cache_mla_latent = cache_mla_latent
        if self.cache_mla_latent:
            assert (
                block_size_tokens == 64
            ), "Flash MLA requires a block size of 64. Set --inference-dynamic-batching-block-size 64 to fix this assert"

        # Per partition num heads and hidden size.
        projection_size = kv_channels * num_attention_qo_heads
        if tensor_model_parallel_size is None:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
        else:
            tp_size = tensor_model_parallel_size
        hidden_size_per_attention_head = core_divide(projection_size, num_attention_qo_heads)
        num_attention_kv_heads_per_partition = core_divide(num_attention_kv_heads, tp_size)
        # Block size tokens, bytes.
        dtype_size_bytes = params_dtype.itemsize
        self.block_size_tokens = block_size_tokens
        if self.cache_mla_latent:
            #   one vector  c_t  (rank)  +  optional RoPE phase slice
            kv_reduced_dim = kv_lora_rank + qk_pos_emb_head_dim
            self.kv_reduced_dim = kv_reduced_dim
            self.block_size_bytes = (
                dtype_size_bytes * num_layers * self.block_size_tokens * kv_reduced_dim
            )
        else:
            self.block_size_bytes = (
                dtype_size_bytes
                * 2  # key, value
                * num_layers
                * self.block_size_tokens
                * num_attention_kv_heads_per_partition
                * hidden_size_per_attention_head
            )

        # Adjust buffer to be a multiple of block size.
        buffer_size_bytes = int(buffer_size_gb * 1024**3)
        buffer_size_bytes_rem = buffer_size_bytes % self.block_size_bytes
        buffer_size_bytes = buffer_size_bytes - buffer_size_bytes_rem

        # Calculate the total number of blocks available in the buffer
        block_count_total = buffer_size_bytes // self.block_size_bytes

        # Compute max_requets, max_tokens from buffer size and overflow factor.
        def bytes_to_max_requests_and_tokens(n_bytes):
            n_tokens = n_bytes / self.block_size_bytes * self.block_size_tokens
            n_requests = n_tokens / max_sequence_length
            n_requests = min(n_requests, block_count_total)
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
            self.max_requests = (
                max_requests_override
                if max_requests_override < self.REQUEST_ROUNDER
                else self.round_up_requests(max_requests_override, tp_size=tp_size)
            )

        if max_tokens_override is not None:
            self.max_tokens = max_tokens_override # self.round_up_tokens(max_tokens_override, tp_size=tp_size)

        self.max_requests = min(self.max_requests, self.max_tokens)  # e.g., decode only.

        # Initialize context state.
        self.params_dtype = params_dtype
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length

        # Store model parameters for FlashInfer metadata (needed for set_model_params)
        self.num_attention_qo_heads_per_partition = core_divide(num_attention_qo_heads, tp_size)
        self.num_attention_kv_heads_per_partition = num_attention_kv_heads_per_partition
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.unified_memory_level = unified_memory_level
        if unified_memory_level > 0:
            if not has_unified_memory and torch.distributed.get_rank() == 0:
                warnings.warn(
                    "Unified memory requested but not available; defaulting to GPU memory."
                )
                self.unified_memory_level = 0
            else:
                self.unified_memory_mempool = create_unified_mempool()

        # Initialize context metadata manager (handles request/token state)
        if enable_async_scheduling:
            self.context_metadata = AsyncContextMetadata()
        else:
            self.context_metadata = SyncContextMetadata()

        # Note: context_metadata.initialize() will be called after memory buffer
        # creation, as it needs block_allocator to be set up first.
        # Temporarily initialize placeholders for compatibility with memory buffer setup
        self.real_config = CUDAGraphConfig(
            token_count=0, prefill_req_count=0, decode_req_count=0, copy_id=0
        )
        self.padded_config = CUDAGraphConfig(
            token_count=0, prefill_req_count=0, decode_req_count=0, copy_id=0
        )

        # Memory buffer - now a list of cache objects, one per layer.
        # Determine layout based on backend
        if cache_mla_latent:
            layout = None  # MLA uses single canonical layout
        elif attention_backend in [AttnBackend.flashinfer_fa2,
                                   AttnBackend.flashinfer_fa3,
                                   AttnBackend.flashinfer_trt]:
            layout = KVCacheLayout.M_N2HCD  
        else:  # Flash backend
            layout = KVCacheLayout.M_2NCHD 

        ctx_manager = (
            torch.cuda.use_mem_pool(self.unified_memory_mempool)
            if self.unified_memory_level > 0
            else nullcontext()
        )

        self.memory_buffer: List[KVCacheBase] = []
        with ctx_manager:
            for layer_idx in range(self.num_layers):
                if cache_mla_latent:
                    cache = MLACache(
                        num_chunks=block_count_total,
                        chunk_size=self.block_size_tokens,
                        kv_reduced_dim=kv_reduced_dim,
                        dtype=self.params_dtype,
                        device=torch.cuda.current_device(),
                    )
                else:
                    cache = create_mhagqa_cache(
                        layout=layout,
                        num_chunks=block_count_total,
                        chunk_size=self.block_size_tokens,
                        num_kv_heads=num_attention_kv_heads_per_partition,
                        head_dim=hidden_size_per_attention_head,
                        dtype=self.params_dtype,
                        device=torch.cuda.current_device(),
                    )
                self.memory_buffer.append(cache)

        # Block ids.
        self.max_kv_block_count = math.ceil(self.max_sequence_length / self.block_size_tokens)

        # Note: Block mapping and attention metadata initialization moved to
        # context_metadata.initialize() which will be called after block_allocator setup

        # Guaranteed active requests.
        # * See details in the class docstring above. `gtd_request_fraction` is
        #   the fraction of blocks in the memory buffer that are reserved for
        #   guaranteeing that some number of active requests can always proceed
        #   with their generations. The number of blocks defined by
        #   `buffer_guaranteed_fraction * block_count_total` is converted to a
        #   number of requests that this reserved space can safely handle
        #   (`gtd_request_count`).
        # * Note: computing the size of this guaranteed space from blocks rather
        #   than bytes is safer due to the non-linear impacts of a large
        #   `block_size_tokens` or `max_kv_block_count`. When computing from
        #   blocks, this space will always be less than `block_count_total`. When
        #   computing from bytes, this space can unexpectedly be much larger than
        #   `block_count_total`, resulting in stalled generations.
        gtd_block_count = int(buffer_guaranteed_fraction * block_count_total)
        gtd_block_count = min(gtd_block_count, block_count_total)
        self.gtd_request_count = max(1, gtd_block_count // self.max_kv_block_count)
        self.gtd_block_count = self.gtd_request_count * self.max_kv_block_count

        # Initialize allocator for KV memory blocks
        self.block_allocator = BlockAllocator(
            block_count_total=block_count_total, gtd_block_count=self.gtd_block_count
        )

        # Initialize context metadata (request/token tensors and attention metadata)
        # This must be done after block_allocator is set up
        self.context_metadata.initialize(self)

        # CUDA graph config list
        self.populate_cudagraph_config_list(
            tp_size,
            num_cuda_graphs,
            cuda_graph_max_tokens,
            cuda_graph_max_prefill_requests,
            block_count_total - self.gtd_block_count,
            use_cuda_graphs_for_non_decode_steps = use_cuda_graphs_for_non_decode_steps,
        )

        self._using_cuda_graph_this_step = False

        # Store the dummy block idx reference for convenience
        self.dummy_block_idx = self.block_allocator.dummy_block_idx

        # Reset attention state.
        self.reset_attention_state()

        if use_flashinfer_fused_rope is True:
            assert HAVE_FLASHINFER, "flashinfer is not installed"
        elif use_flashinfer_fused_rope is None:
            use_flashinfer_fused_rope = HAVE_FLASHINFER
        self.use_flashinfer_fused_rope = use_flashinfer_fused_rope

    # ===== PROPERTY WRAPPERS FOR BACKWARD COMPATIBILITY =====
    # These delegate to context_metadata while maintaining the existing API

    # Helper method
    def _get_tensor(self, name: str):
        """Get tensor from active context metadata."""
        return self.context_metadata.get_active_tensors()[name]

    # Per-request tensor properties (7)
    @property
    def request_ids(self):
        return self._get_tensor('request_ids')

    @property
    def request_query_lengths(self):
        return self._get_tensor('request_query_lengths')

    @property
    def request_output_lengths(self):
        return self._get_tensor('request_output_lengths')

    @property
    def request_kv_length_offsets(self):
        return self._get_tensor('request_kv_length_offsets')

    @property
    def request_kv_block_counts(self):
        return self._get_tensor('request_kv_block_counts')

    @property
    def request_last_kv_block_id(self):
        return self._get_tensor('request_last_kv_block_id')

    @property
    def request_last_kv_block_offset(self):
        return self._get_tensor('request_last_kv_block_offset')

    @property
    def request_last_token_ids(self):
        return self._get_tensor('request_last_token_ids')

    # Per-token tensor properties (6)
    @property
    def token_to_input_ids(self):
        return self._get_tensor('token_to_input_ids')

    @property
    def token_to_pos_ids(self):
        return self._get_tensor('token_to_pos_ids')

    @property
    def token_to_request_idx(self):
        return self._get_tensor('token_to_request_idx')

    @property
    def token_to_block_idx(self):
        return self._get_tensor('token_to_block_idx')

    @property
    def token_to_position_in_request(self):
        return self._get_tensor('token_to_position_in_request')

    @property
    def token_to_local_position_within_kv_block(self):
        return self._get_tensor('token_to_local_position_within_kv_block')

    # Block mapping property (1)
    @property
    def request_to_kv_block_ids(self):
        return self._get_tensor('request_to_kv_block_ids')

    # Scalar count properties with setters (7)
    @property
    def total_request_count(self):
        return self.context_metadata.total_request_count

    @total_request_count.setter
    def total_request_count(self, value):
        self.context_metadata.total_request_count = value

    @property
    def active_token_count(self):
        return self.context_metadata.active_token_count

    @active_token_count.setter
    def active_token_count(self, value):
        self.context_metadata.active_token_count = value

    @property
    def paused_request_count(self):
        return self.context_metadata.paused_request_count

    @paused_request_count.setter
    def paused_request_count(self, value):
        self.context_metadata.paused_request_count = value

    @property
    def padded_active_token_count(self):
        return self.context_metadata.padded_active_token_count

    @padded_active_token_count.setter
    def padded_active_token_count(self, value):
        self.context_metadata.padded_active_token_count = value

    @property
    def padded_active_request_count(self):
        return self.context_metadata.padded_active_request_count

    @padded_active_request_count.setter
    def padded_active_request_count(self, value):
        self.context_metadata.padded_active_request_count = value

    @property
    def num_prefill_requests(self):
        return self.context_metadata.num_prefill_requests

    @num_prefill_requests.setter
    def num_prefill_requests(self, value):
        self.context_metadata.num_prefill_requests = value

    @property
    def paused_tokens(self):
        return self.context_metadata.paused_tokens

    @paused_tokens.setter
    def paused_tokens(self, value):
        self.context_metadata.paused_tokens = value

    @property
    def chunked_prefill_request_id(self):
        return self.context_metadata.chunked_prefill_request_id

    @chunked_prefill_request_id.setter
    def chunked_prefill_request_id(self, value):
        self.context_metadata.chunked_prefill_request_id = value

    @property
    def chunked_prefill_request_id_old(self):
        return self.context_metadata.chunked_prefill_request_id_old

    # Attention metadata properties
    @property
    def graph_attn_metadata(self):
        return self.context_metadata.get_active_attention_metadata()['graph']

    @property
    def non_graph_attn_metadata(self):
        return self.context_metadata.get_active_attention_metadata()['non_graph']

    # Note: active_attn_metadata is managed by existing reset_attention_state() logic
    # and points to either graph_attn_metadata or non_graph_attn_metadata

    TOKEN_ROUNDER = 64
    REQUEST_ROUNDER = 4

    def populate_cudagraph_config_list(
        self,
        tp_size,
        num_cuda_graphs,
        cuda_graph_max_tokens,
        cuda_graph_max_prefill_requests,
        block_avail,
        use_cuda_graphs_for_non_decode_steps,
    ):
        """
        Initialize the cudagraph config list.

        This function constructs CUDA graph configurations for different token counts and request patterns,
        then filters them based on resource constraints. The construction process involves:

        Construction Rules:
        1. Token count generation: Creates token counts from step_size to max_tokens, rounded to multiples of 8
        2. Tensor parallelism alignment: Ensures step_size is divisible by tensor parallel size
        3. Configuration creation: For each token count, creates three types of configs:
           - Decode-only: (token_count, 0, token_count) - all tokens used for decode requests
           - Mixed prefill+decode: (token_count, prefill_req_count, token_count - prefill_req_count)
           - Prefill-only: (token_count, max(prefill_req_count, ceil(token_count/(max_seq_len-1))), 0)

        Filtering Rules:
        1. Request limit: prefill_req_count + decode_req_count <= max_requests
        2. Non-negative counts: Both prefill_req_count and decode_req_count must be >= 0
        3. Token sufficiency: token_count >= prefill_req_count + decode_req_count
        4. Block availability: Total requests + required blocks <= available blocks

        Sorting Rules for Attention Metadata Construction:
        1. Configs are sorted by prefill token count (token_count - decode_req_count) in descending order

        """
        # Cuda graph token-counts (i.e., token counts used by cuda-graph steps, both decode and non-decode).
        self.cuda_graph_token_counts = None
        if num_cuda_graphs is not None:

            # Ensure valid num_cuda_graphs.
            if (
                cuda_graph_max_tokens is None
                or cuda_graph_max_tokens > self.max_tokens
                or cuda_graph_max_tokens < 0
            ):
                cuda_graph_max_tokens = self.max_tokens
            num_cuda_graphs = min(max(num_cuda_graphs, 1), cuda_graph_max_tokens)

            # Cuda graph step size.
            cuda_graph_rounder = 8
            self.cuda_graph_step_size = cuda_graph_max_tokens / num_cuda_graphs
            self.cuda_graph_step_size = cuda_graph_rounder * int(
                math.ceil(int(self.cuda_graph_step_size) / cuda_graph_rounder)
            )
            # Make sure divisble by TP size
            self.cuda_graph_step_size = math.ceil(self.cuda_graph_step_size / tp_size) * tp_size

            # Cuda graph token counts.
            if num_cuda_graphs == 1:
                self.cuda_graph_token_counts = [cuda_graph_max_tokens]
            else:
                self.cuda_graph_token_counts = list(
                    range(
                        self.cuda_graph_step_size, cuda_graph_max_tokens, self.cuda_graph_step_size
                    )
                )
                if self.cuda_graph_token_counts[-1] != cuda_graph_max_tokens:
                    self.cuda_graph_token_counts.append(cuda_graph_max_tokens)
                self.cuda_graph_token_counts.reverse()

        def _append_configs(token_count: int, prefill_count: int, decode_count: int):
            if self.enable_async_scheduling:
                for copy_id in (0, 1):
                    configs.append(
                        CUDAGraphConfig(token_count, prefill_count, decode_count, copy_id)
                    )
            else:
                configs.append(
                    CUDAGraphConfig(token_count, prefill_count, decode_count, copy_id=0)
                )

        configs: List[CUDAGraphConfig] = []
        if num_cuda_graphs is None:
            configs = []
        elif (
            not cuda_graph_max_prefill_requests or cuda_graph_max_prefill_requests <= 0 or not use_cuda_graphs_for_non_decode_steps
        ):  # decode only
            for size in self.cuda_graph_token_counts:
                _append_configs(size, 0, size)
        else:
            for size in self.cuda_graph_token_counts:
                _append_configs(size, 0, size)
                _append_configs(
                    size,
                    cuda_graph_max_prefill_requests,
                    size - cuda_graph_max_prefill_requests,
                )
                # We need to ensure the prefill requests are shorter than the max sequence length, considering the one decode token is used for prefill request construction
                prefill_only_minimal_num = max(
                    cuda_graph_max_prefill_requests,
                    math.ceil(size / max(1, self.max_sequence_length - 1)),
                )
                if prefill_only_minimal_num < self.max_requests:
                    _append_configs(
                        size,
                        max(prefill_only_minimal_num, min(self.max_requests, size)),
                        0,
                    )

        # filter out configurations that have too many requests or too many blocks
        filtered_cudagraph_config_list = []
        for config in configs:
            if config.prefill_req_count + config.decode_req_count > self.max_requests:
                continue
            if config.prefill_req_count < 0 or config.decode_req_count < 0:
                continue
            if config.token_count < config.prefill_req_count + config.decode_req_count:
                continue
            if (
                config.prefill_req_count
                + config.decode_req_count
                + math.ceil(config.token_count // self.block_size_tokens)
                > block_avail
            ):
                continue
            filtered_cudagraph_config_list.append(config)

        filtered_cudagraph_config_list.sort(
            key=lambda x: (x.token_count - x.decode_req_count), reverse=True
        )
        self.cudagraph_config_list = filtered_cudagraph_config_list


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
        """
        Return if this iteration we run decode only implementation.
        """
        return self.num_prefill_requests == 0


    def using_cuda_graph_this_step(self) -> bool:
        """Returns True if cuda graphs are being used for this step."""
        return self._using_cuda_graph_this_step

    def has_unfinished_requests(self) -> bool:
        """Test if any requests remain."""
        return self.total_request_count > 0

    def cu_query_lengths(self) -> Tuple[Tensor, int]:
        """Cumulative query sequence lengths."""
        return (
            self.active_attn_metadata["mha_metadata"].state_data["cu_query_seq_lengths"],
            self.active_attn_metadata["mha_metadata"].state_data["max_seqlen_q"],
        )

    def cu_kv_lengths(self) -> Tuple[Tensor, Tensor, int]:
        """Cumulative key/value sequence lengths."""
        return (
            self.active_attn_metadata["mha_metadata"].state_data["cu_kv_seq_lengths"],
            self.active_attn_metadata["mha_metadata"].state_data["kv_seq_lengths"],
            self.active_attn_metadata["mha_metadata"].state_data["max_seqlen_k"],
        )

    def get_active_sequence_lengths(self) -> Tensor:
        """Total sequence length (query + key) for active requests."""
        lengths = self.request_kv_length_offsets + self.request_query_lengths
        lengths = lengths[self.paused_request_count : self.total_request_count]
        return lengths

    def get_max_sequence_lengths(self) -> Tensor:
        """Maximum sequence length for active requests."""
        return self.request_output_lengths[self.paused_request_count : self.total_request_count]

    def get_active_sequence_lengths_old(self) -> Tensor:
        """Total sequence length (query + key) for active requests in OLD metadata slot.

        Used in async mode to check termination conditions for requests whose logits
        were computed in the previous iteration using the OLD metadata slot.
        """
        if not self.enable_async_scheduling:
            # In sync mode, there is no "old" slot, just return current
            return self.get_active_sequence_lengths()

        metadata = self.context_metadata
        old_tensors = metadata.get_old_tensors()
        old_paused_count = metadata.paused_request_count_old
        old_total_count = metadata.total_request_count_old

        lengths = old_tensors['request_kv_length_offsets'] + old_tensors['request_query_lengths']
        lengths = lengths[old_paused_count:old_total_count]
        return lengths

    def get_max_sequence_lengths_old(self) -> Tensor:
        """Maximum sequence length for active requests in OLD metadata slot.

        Used in async mode to check termination conditions for requests whose logits
        were computed in the previous iteration using the OLD metadata slot.
        """
        if not self.enable_async_scheduling:
            # In sync mode, there is no "old" slot, just return current
            return self.get_max_sequence_lengths()

        metadata = self.context_metadata
        old_tensors = metadata.get_old_tensors()
        old_paused_count = metadata.paused_request_count_old
        old_total_count = metadata.total_request_count_old

        return old_tensors['request_output_lengths'][old_paused_count:old_total_count]

    @property
    def request_ids_old(self):
        if not self.enable_async_scheduling:
            return self.request_ids
        return self.context_metadata.get_old_tensors()['request_ids']

    @property
    def request_last_token_ids_old(self):
        if not self.enable_async_scheduling:
            return self.request_last_token_ids
        return self.context_metadata.get_old_tensors()['request_last_token_ids']

    @property
    def chunked_prefill_request_id_post_process(self) -> int:
        if not self.enable_async_scheduling:
            return -1
        return getattr(
            self.context_metadata, "chunked_prefill_request_id_post_process", -1
        )

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
        cache = self.memory_buffer[layer_number - 1]

        # Use Triton kernel for M_2NCHD and S_NCHD layouts only
        if (triton_append_key_value_cache is not None
            and not self.cache_mla_latent
            and cache.supports_triton()):
            return triton_append_key_value_cache(
                key=key,
                value=value,
                cache=cache,
                padded_active_token_count=self.padded_active_token_count,
                token_to_block_idx=self.token_to_block_idx,
                token_to_local_position_within_kv_block=self.token_to_local_position_within_kv_block,
            )

        # Fallback: use cache's append method for all layouts
        cache.append(
            key=key,
            value=value,
            padded_active_token_count=self.padded_active_token_count,
            token_to_block_idx=self.token_to_block_idx,
            token_to_local_position_within_kv_block=self.token_to_local_position_within_kv_block,
        )

    def key_value_cache(self, layer_number: int) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """Read from KV cache.

        Args:
            layer_number (int): Layer number.

        Return:
            Tuple[Tensor, Optional[Tensor], Tensor]: The key cache, value cache (or None for MLA),
                and block table tensor.
                - For MLA: (kv_latent_cache, None, block_table)
                - For separate K/V caches (S_NCHD, S_NHCD): (k_cache, v_cache, block_table)
                - For merged caches (M_2NCHD, M_N2CHD, M_N2HCD): (k_view, v_view, block_table)
        """
        cache = self.memory_buffer[layer_number - 1]
        cache_content = cache.get_content()
        block_table = self.active_attn_metadata["mha_metadata"].state_data["block_table"]

        if isinstance(cache_content, tuple):
            # Separate K/V caches (S_NCHD, S_NHCD)
            k_cache, v_cache = cache_content
            return (k_cache, v_cache, block_table)
        elif self.cache_mla_latent:
            # MLA latent cache - return as single tensor with None for value
            return (cache_content, None, block_table)
        else:
            # Merged K/V cache - return views on K and V components
            # All merged layouts have the K/V dimension, we need to slice it
            from megatron.core.inference.kv_cache import KVCacheM2NCHD, KVCacheMN2CHD, KVCacheMN2HCD

            if isinstance(cache, KVCacheM2NCHD):
                # M_2NCHD layout: [2, N, C, H, D] - K/V is first dimension
                return (cache_content[0], cache_content[1], block_table)
            elif isinstance(cache, (KVCacheMN2CHD, KVCacheMN2HCD)):
                # M_N2CHD layout: [N, 2, C, H, D] - K/V is second dimension
                # M_N2HCD layout: [N, 2, H, C, D] - K/V is second dimension
                return (cache_content[:, 0], cache_content[:, 1], block_table)
            else:
                # Should not reach here, but fallback for unknown merged layout
                raise ValueError(f"Unknown cache type: {type(cache)}")

    def apply_fused_qk_rotary_emb(
        self, query: Tensor, key: Tensor, cos_sin_emb: Tensor, config: TransformerConfig
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary embedding to query and key tensors using flashinfer's fused rope.
        Args:
            query (Tensor): Query tensor.
            key (Tensor): Key tensor.
            cos_sin_emb (Tensor): Rotary embeddings.
            config (TransformerConfig): Transformer config.

        Return:
            (Tuple[Tensor, Tensor]) Query and Key tensors after applying rotary embeddings.
        """
        assert self.use_flashinfer_fused_rope, "flashinfer fused rope is not enabled"
        n = self.padded_active_token_count
        num_q_heads, head_size = query.shape[-2], query.shape[-1]
        num_k_heads = key.shape[-2]

        # use .view instead of .reshape to avoid extra transpose operations
        query_rope, key_rope = flashinfer.rope.apply_rope_with_cos_sin_cache(
            positions=self.token_to_pos_ids[:n],
            query=query[:n].reshape(n, num_q_heads * head_size),
            key=key[:n].reshape(n, num_k_heads * head_size),
            head_size=head_size,
            cos_sin_cache=cos_sin_emb,
            is_neox=not config.rotary_interleaved,
        )
        return query_rope.reshape(n, 1, num_q_heads, head_size), key_rope.reshape(
            n, 1, num_k_heads, head_size
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
            if key.shape[0] != n:
                raise AssertionError(
                    f"apply_rotary_emb_key: key.shape[0]={key.shape[0]} != n={n}; "
                    f"padded_active_request_count={self.padded_active_request_count}, "
                    f"active_token_count={self.active_token_count}, total_request_count={self.total_request_count}, "
                    f"paused_request_count={self.paused_request_count}"
                )
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
        # Attention metadata reset is now handled by MHAMetadata.reset()
        for attn_metadata in self.non_graph_attn_metadata.values():
            attn_metadata.reset()
        for attn_metadata in self.graph_attn_metadata.values():
            attn_metadata.reset()
        self.active_attn_metadata = None


    def add_dummy_requests_for_cudagraph_capture(self, graph_config: CUDAGraphConfig) -> None:
        """
        Adds dummy requests to reflect the number of prefill and decode requests in the graph config.
        These are using during cuda graph captures.
        """
        prefill_tokens = graph_config.token_count - graph_config.decode_req_count

        for i in range(graph_config.decode_req_count):
            self.add_request(
                DynamicInferenceRequest(
                    request_id=i,
                    prompt_tokens=torch.zeros(1, dtype=torch.long, device=torch.cuda.current_device()),
                    sampling_params=SamplingParams(num_tokens_to_generate=1),
                ),
            )
        if graph_config.prefill_req_count == 0:
            self.num_prefill_requests = 0
            return

        per_prefill_tokens = prefill_tokens // graph_config.prefill_req_count
        rem_prefill_tokens = prefill_tokens % graph_config.prefill_req_count
        prefill_token_counts = torch.full(
            (graph_config.prefill_req_count,),
            per_prefill_tokens,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        if rem_prefill_tokens > 0:
            prefill_token_counts[:rem_prefill_tokens] += 1
        assert per_prefill_tokens > 0
        for i in range(graph_config.prefill_req_count):
            self.add_request(
                DynamicInferenceRequest(
                    request_id=i + graph_config.decode_req_count,
                    prompt_tokens=torch.zeros(prefill_token_counts[i], dtype=torch.long, device=torch.cuda.current_device()),
                    sampling_params=SamplingParams(num_tokens_to_generate=1),
                ),
            )
        self.num_prefill_requests = graph_config.prefill_req_count

    def graph_matching(self, real_config: CUDAGraphConfig) -> Optional[CUDAGraphConfig]:
            """
            Matches the best graph for the given token count, prefill request count, and decode request count.
            """
            # first filter out graphs with smaller token count, prefill req count, or decode req count, as they are not valid
            graph_configs_valid = [
                graph_config
                for graph_config in self.cudagraph_config_list
                if graph_config.valid(real_config)
            ]
            if len(graph_configs_valid) == 0:
                return None
            # then find the best graph
            best_graph = min(graph_configs_valid)
            return best_graph

    @property
    def num_decode_requests(self) -> int:
        """
        Returns the number of decode requests.
        """
        return self.total_request_count - self.paused_request_count - self.num_prefill_requests

    def initialize_attention_state(
        self,
        *,
        construct_graph_config: Optional[CUDAGraphConfig] = None,
    ) -> None:
        """Initialize attention state so that every layer can use it.

        Args:
            construct_graph_config (Optional[CUDAGraphConfig]): The graph config to use for constructing the cuda graphs.
        Return:
            None.
        """
         # if in recording mode, add dummy requests for cuda graph capture
        if construct_graph_config is not None:
            self.reset()
            if self.enable_async_scheduling:
                self.context_metadata.activate_metadata(construct_graph_config.copy_id)

            if (
                construct_graph_config.prefill_req_count + construct_graph_config.decode_req_count
                > self.max_requests
            ):
                raise ActiveRequestCountOverflowError(
                    self.max_requests,
                    construct_graph_config.prefill_req_count
                    + construct_graph_config.decode_req_count,
                )
            self.add_dummy_requests_for_cudagraph_capture(construct_graph_config)

        real_config = CUDAGraphConfig(
            token_count=self.active_token_count,
            prefill_req_count=self.num_prefill_requests,
            decode_req_count=self.num_decode_requests,
            copy_id=self.context_metadata.active_id
        )
        self.real_config = real_config
        best_graph = self.graph_matching(real_config)
        self._using_cuda_graph_this_step = best_graph is not None
        if construct_graph_config is not None:
            assert (
                real_config == construct_graph_config == best_graph
            ), f"real_config: {real_config}, construct_graph_config: {construct_graph_config}, best_graph: {best_graph}"

        if self.using_cuda_graph_this_step():
            self.padded_config = best_graph
            self.padded_active_token_count = self.padded_config.token_count
            self.padded_active_request_count = self.padded_config.req_count
        else:
            self.padded_config = CUDAGraphConfig()
            self.padded_config.token_count = self.active_token_count
            if self.is_decode_only():
                self.padded_config.decode_req_count = self.padded_config.token_count
                self.padded_config.prefill_req_count = 0
            else:
                target_padding_req_count = self.round_up_requests(self.total_request_count - self.paused_request_count)
                self.padded_config.decode_req_count = self.num_decode_requests
                self.padded_config.prefill_req_count = target_padding_req_count - self.padded_config.decode_req_count
            self.padded_active_token_count = self.padded_config.token_count
            self.padded_active_request_count = self.padded_config.req_count

        # Update token position indexes.
        self.token_to_block_idx[self.active_token_count : self.padded_active_token_count] = (
            self.dummy_block_idx
        )
        self.token_to_local_position_within_kv_block[
            self.active_token_count : self.padded_active_token_count
        ] = 0
        self.token_to_position_in_request[
            self.active_token_count : self.padded_active_token_count
        ] = 0

        self.active_attn_metadata = (
            self.graph_attn_metadata
            if self.using_cuda_graph_this_step()
            else self.non_graph_attn_metadata
        )

        # Update cu_query_seq_lengths, max_seqlen_q.
        active_slice = slice(self.paused_request_count, self.total_request_count)
        query_lengths_view = self.request_query_lengths[active_slice]
        request_kv_length_offsets_view = self.request_kv_length_offsets[active_slice]
        request_to_kv_block_ids_view = self.request_to_kv_block_ids[active_slice]

        attn_config = real_config
        if real_config.decode_req_count > self.padded_config.decode_req_count:
            attn_config.prefill_req_count = attn_config.req_count - self.padded_config.decode_req_count
            attn_config.decode_req_count = self.padded_config.decode_req_count
            
        self.active_attn_metadata["mha_metadata"].update(
            request_query_lengths=query_lengths_view,
            request_kv_length_offsets=request_kv_length_offsets_view,
            request_to_kv_block_ids=request_to_kv_block_ids_view,
            real_config = attn_config,
            padded_config = self.padded_config,
        )
        # All attention metadata calculations are now handled by MHAMetadata.update()

    def reset(self) -> None:
        """Reset entire context.

        This method does:
        - Reset active/paused request/token counts to zero.
        - Reset available blocks to entire memory.
        - Reset other tensors to zeros (unncessary, just or sanity checking).

        This method is useful after cuda graph warmup iterations, where the
        context's memory buffer is referenced by the cuda graph system and
        cannot be deallocated.
        """

        # Reset request/token counts.
        self.total_request_count = 0
        self.active_token_count = 0
        self.paused_request_count = 0
        self.real_config = CUDAGraphConfig(
            token_count=0, prefill_req_count=0, decode_req_count=0, copy_id=self.context_metadata.active_id
        )
        self.padded_active_token_count = 0
        self.padded_active_request_count = 0
        self.paused_tokens = None

        # Reset request indexes.
        self.request_ids.fill_(-1)
        self.request_query_lengths.fill_(0)
        self.request_output_lengths.fill_(0)
        self.request_kv_length_offsets.fill_(0)
        self.request_kv_block_counts.fill_(0)
        self.request_last_kv_block_id.fill_(-1)
        self.request_last_kv_block_offset.fill_(0)
        self.request_last_token_ids.fill_(0)
        self.request_to_kv_block_ids.fill_(-1)

        # Reset token indexes.
        self.token_to_input_ids.fill_(0)
        self.token_to_pos_ids.fill_(0)
        self.token_to_request_idx.fill_(-1)
        self.token_to_position_in_request.fill_(0)
        self.token_to_block_idx.fill_(-1)
        self.token_to_local_position_within_kv_block.fill_(0)

        # Reset available block count.
        self.reset_attention_state()
        self.block_allocator.reset()
        self.request_to_kv_block_ids.fill_(-1)

        # Reset chunked prefill state
        self.chunked_prefill_request_id = -1
        self.num_prefill_requests = 0
        self._using_cuda_graph_this_step = False
        self.padded_config = CUDAGraphConfig(
            token_count=0, prefill_req_count=0, decode_req_count=0, copy_id=self.context_metadata.active_id
        )
        if hasattr(self.context_metadata, "chunked_prefill_request_id_post_process"):
            self.context_metadata.chunked_prefill_request_id_post_process = -1

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

    def last_token_logits_old(self, logits: Tensor) -> Tensor:
        """Last tokens of logits using OLD metadata slot.

        Used in async mode to extract last token logits when processing logits
        that were produced in the previous iteration using the OLD metadata slot.

        Args:
            logits (Tensor): Output logits of forward pass from previous iteration.

        Return:
            (Tensor) Last token logits for OLD active requests.
        """
        if not self.enable_async_scheduling:
            # In sync mode, delegate to regular method
            return self.last_token_logits(logits)

        metadata = self.context_metadata
        old_tensors = metadata.get_old_tensors()
        old_paused_count = metadata.paused_request_count_old
        old_total_count = metadata.total_request_count_old

        # Last token logits from OLD metadata
        logits_squeezed = logits.squeeze(0)
        old_query_lengths = old_tensors['request_query_lengths'][old_paused_count:old_total_count]
        last_token_idxs = torch.cumsum(old_query_lengths, dim=0) - 1
        last_token_logits = logits_squeezed[last_token_idxs, :]

        return last_token_logits

    def check_availability(
        self, req: DynamicInferenceRequest, safe: bool = False
    ) -> (bool, bool, bool):
        """
        Check if the request can be added to the context.
        """
        request_can_be_added = self.total_request_count < self.max_requests
        request_tokens_can_be_added = (
            self.active_token_count + req.remaining_prompt_length <= self.max_tokens
        )
        blocks = math.ceil(
            (req.remaining_prompt_length + req.finished_chunk_token_count) / self.block_size_tokens
        ) - math.ceil(req.finished_chunk_token_count / self.block_size_tokens)
        kv_cache_available = self.block_allocator.is_memory_available(blocks, safe=safe)
        return request_can_be_added, request_tokens_can_be_added, kv_cache_available

    def add_request(self, req: DynamicInferenceRequest, chunk_length: Optional[int] = None) -> None:
        """Add request to context. At this stage, we assume that the request is valid and can be added, as the checks are done in the schedule function.

        Args:
            req (DynamicInferenceRequest): Request to add.
            chunk_length (Optional[int]): Length of chunk to add. If None, the request will be fully added.

        Return:
            None
        """
        if chunk_length is None:
            chunk_length = req.remaining_prompt_length

        # req.finished_chunk_token_count > 0 means that the request is a scheduled chunked prefill request, and we are adding a chunk to it
        is_chunked_prefill = req.finished_chunk_token_count > 0

        assert chunk_length > 0, "Chunk length is 0"
        assert (
            chunk_length <= req.remaining_prompt_length
        ), "Chunk length is greater than remaining prompt length"
        if self.active_token_count + chunk_length > self.max_tokens:
            raise TokenOverflowError(req.request_id)

        # Use the remaining prompt tokens for this chunk
        this_round_tokens = req.remaining_prompt_tokens[:chunk_length]

        # only allocate new blocks
        already_allocated_blocks = (
            req.finished_chunk_token_count + self.block_size_tokens - 1
        ) // self.block_size_tokens  # ceiling division
        overall_required_blocks = (
            req.finished_chunk_token_count + chunk_length + self.block_size_tokens - 1
        ) // self.block_size_tokens  # ceiling division

        num_blocks_needed = overall_required_blocks - already_allocated_blocks

        if num_blocks_needed > 0:
            new_block_ids = self.block_allocator.allocate_memory_blocks(
                num_blocks_needed, safe=not is_chunked_prefill
            )
            if new_block_ids is None or len(new_block_ids) != num_blocks_needed:
                logger.warning(
                    "Unable to allocate %d KV blocks (safe=%s) for request %s; available=%d",
                    num_blocks_needed,
                    not is_chunked_prefill,
                    req.request_id,
                    self.block_allocator.get_available_block_count(),
                )
                raise BlockOverflowError(req.request_id)

        # when a request already starts chunked prefill, it is exactly the last request in the current system
        # (see dynamic_engine.py, schedule_chunked_prefill invariants)
        # no need to update count, as it is already here
        if is_chunked_prefill:
            current_id = self.total_request_count - 1
            self.active_token_count -= (
                1  # Overwrite the last token, which is the useless token from chunked prefill
            )
            assert (
                self.request_ids[current_id] == req.request_id
            ), "Continuation current_id mismatch"
        else:
            current_id = self.total_request_count

        if current_id >= self.max_requests:
            raise RequestOverflowError(req.request_id)

        if self.active_token_count + chunk_length > self.max_tokens:
            raise TokenOverflowError(req.request_id)

        self.request_ids[current_id] = req.request_id
        self.request_query_lengths[current_id] = chunk_length
        self.request_output_lengths[current_id] = (
            req.finished_chunk_token_count
            + chunk_length
            + req.sampling_params.num_tokens_to_generate
        )
        if num_blocks_needed > 0:
            self.request_to_kv_block_ids[current_id][
                already_allocated_blocks:overall_required_blocks
            ] = new_block_ids
        self.request_kv_length_offsets[current_id] = req.finished_chunk_token_count
        self.request_kv_block_counts[current_id] = overall_required_blocks
        self.request_last_kv_block_id[current_id] = self.request_to_kv_block_ids[current_id][
            overall_required_blocks - 1
        ]
        self.request_last_kv_block_offset[current_id] = (
            chunk_length + req.finished_chunk_token_count - 1
        ) % self.block_size_tokens
        
        token_offset_range = torch.arange(
            req.finished_chunk_token_count,
            req.finished_chunk_token_count + chunk_length,
            device=self.token_to_pos_ids.device,
        )
        self.token_to_pos_ids[self.active_token_count : self.active_token_count + chunk_length] = (
            token_offset_range
        )
        self.token_to_input_ids[
            self.active_token_count : self.active_token_count + chunk_length
        ] = this_round_tokens
        self.token_to_request_idx[
            self.active_token_count : self.active_token_count + chunk_length
        ] = current_id
        self.token_to_position_in_request[
            self.active_token_count : self.active_token_count + chunk_length
        ] = token_offset_range
        self.token_to_block_idx[
            self.active_token_count : self.active_token_count + chunk_length
        ] = self.request_to_kv_block_ids[current_id][token_offset_range // self.block_size_tokens]
        self.token_to_local_position_within_kv_block[
            self.active_token_count : self.active_token_count + chunk_length
        ] = (token_offset_range % self.block_size_tokens)
        self.active_token_count += chunk_length
        self.total_request_count += 0 if req.finished_chunk_token_count > 0 else 1
        self.num_prefill_requests += 0 if req.finished_chunk_token_count > 0 else 1

    def _move_book_keeping_tensors(self, src_idxs, dst_idxs, next_tokens):
        """
        Move all the relevent booking tensors with src idxs to dst idxs
        """
        self.request_kv_length_offsets[dst_idxs] = self.request_kv_length_offsets[src_idxs]
        self.request_query_lengths[dst_idxs] = self.request_query_lengths[src_idxs]
        self.request_output_lengths[dst_idxs] = self.request_output_lengths[src_idxs]
        self.request_ids[dst_idxs] = self.request_ids[src_idxs]
        next_tokens[dst_idxs] = next_tokens[src_idxs]

        self.request_to_kv_block_ids[dst_idxs] = self.request_to_kv_block_ids[src_idxs]
        self.request_kv_block_counts[dst_idxs] = self.request_kv_block_counts[src_idxs]
        self.request_last_kv_block_id[dst_idxs] = self.request_last_kv_block_id[src_idxs]
        self.request_last_kv_block_offset[dst_idxs] = self.request_last_kv_block_offset[src_idxs]
        self.request_last_token_ids[dst_idxs] = self.request_last_token_ids[src_idxs]

    def _swap_book_keeping_tensors(self, src_idxs, dst_idxs, next_tokens):
        """
        Swaps all the relevent booking tensors with src idxs to dst idxs
        """
        tensor_swap(self.request_kv_length_offsets, src_idxs, dst_idxs)
        tensor_swap(self.request_query_lengths, src_idxs, dst_idxs)
        tensor_swap(self.request_output_lengths, src_idxs, dst_idxs)
        tensor_swap(self.request_ids, src_idxs, dst_idxs)
        tensor_swap(next_tokens, src_idxs, dst_idxs)
        tensor_swap(self.request_to_kv_block_ids, src_idxs, dst_idxs)
        tensor_swap(self.request_kv_block_counts, src_idxs, dst_idxs)
        tensor_swap(self.request_last_kv_block_id, src_idxs, dst_idxs)
        tensor_swap(self.request_last_kv_block_offset, src_idxs, dst_idxs)
        tensor_swap(self.request_last_token_ids, src_idxs, dst_idxs)

    # TODO: see if we can compile this function
    def update_requests(self, active_requests_mask: Tensor, new_tokens: Tensor) -> Tensor:
        """Dispatch to sync or async implementations."""
        if self.enable_async_scheduling:
            return self._update_requests_async(active_requests_mask, new_tokens)
        return self._update_requests_sync(active_requests_mask, new_tokens)

    def _update_requests_sync(
        self, active_requests_mask: Tensor, new_tokens: Tensor
    ) -> Tensor:
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
        4. For the finished requests we release memory blocks and move them to the right
        5. We identify requests that require a new block and add them to the paused requests (i.e move them left)
        6. We determine how many requests we can resume and resume them
        7. We make changes to the request book keeping tesnsors and setup the tokens for next iteration
        8. We resume those requests by assigning blocks and updating bookkeeping tensors
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

        self.num_prefill_requests = 0  # all turns to decode
        if self.chunked_prefill_request_id != -1:
            active_requests_mask[-1] = (
                1  # must keep this, next iteration will add a new chunk to it
            )
        if hasattr(self.context_metadata, "chunked_prefill_request_id_post_process"):
            self.context_metadata.chunked_prefill_request_id_post_process = (
                self.chunked_prefill_request_id
            )

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
                kv_blocks_assigned = self.request_to_kv_block_ids[finished_idxs]
                non_zero_values_in_kv_memory = kv_blocks_assigned[kv_blocks_assigned != -1]
                self.block_allocator.release_memory_blocks(non_zero_values_in_kv_memory)

            # Reset request/token counts.
            self.request_to_kv_block_ids.fill_(-1)
            self.total_request_count = 0
            self.active_token_count = 0
            return

        # 3. Concatenate the paused tokens to the active tokens if present.
        if self.paused_request_count != 0:
            assert self.paused_tokens is not None
            next_tokens = torch.cat((self.paused_tokens, new_tokens))
        else:
            next_tokens = new_tokens

        # 4. For the finished requests we release memory blocks and move them to the right:-
        #       a) Release all their memory
        #       b) Swap them to the right, so that we have this order [Paused, Active, Finished]
        if finished_request_count > 0:
            finished_idxs = (
                torch.nonzero(active_requests_mask == 0, as_tuple=True)[0]
                + self.paused_request_count
            )
            kv_blocks_assigned = self.request_to_kv_block_ids[finished_idxs]
            non_zero_values_in_kv_memory = kv_blocks_assigned[kv_blocks_assigned != -1]
            self.block_allocator.release_memory_blocks(non_zero_values_in_kv_memory)

            # Reset the KV blocks for finished requests.
            # Note: do not use fill_() (or add_() and similar inplace ops) here.
            # The combinition of indexing with a tensor (like finished_idxs) and fill_()/add_() creates a clone
            # and updates it instead of the original tensor.
            self.request_to_kv_block_ids[finished_idxs] = -1

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

                # Reset block ids for recently moved requests.
                self.request_to_kv_block_ids[active_idxs_on_right] = -1

        # 5. We identify requests that require a new block and add them to the paused requests (i.e move them left) :-
        #       a) Put requests that have filled their current block and  require a new one in a pause state temporarily
        #       b) Move the paused requests to the left, and active requets to the right
        #       c) Update the paused request count and active_request_count appropriately
        newly_paused_request_ids = None
        if active_request_count > 0:
            num_tokens_in_last_block = self.request_last_kv_block_offset[
                self.paused_request_count : (active_request_count + self.paused_request_count)
            ]
            active_requests_requiring_new_block = (
                num_tokens_in_last_block == self.block_size_tokens - 1
            ).byte()

            if self.chunked_prefill_request_id != -1:
                # find the id in request_ids that is the chunked_prefill_request_id. Only one request should be chunked.
                pos = torch.where(self.request_ids == self.chunked_prefill_request_id)[0][0]
                active_requests_requiring_new_block[pos] = 0  # chunked prefill should not be paused

            active_requests_requiring_new_block_count = (
                (active_requests_requiring_new_block == 1).sum().item()
            )

            # Swap unfinished active requests on the left side with paused requests on the right side
            # NOTE : We add paused request count because we concatenate
            # paused tokens to the left at the beginning of update requests
            if (
                active_requests_requiring_new_block_count > 0
                and active_requests_requiring_new_block_count != active_request_count
            ):
                active_request_ids_on_left = (
                    torch.nonzero(
                        active_requests_requiring_new_block[
                            :active_requests_requiring_new_block_count
                        ]
                        == 0,
                        as_tuple=True,
                    )[0]
                    + self.paused_request_count
                )
                paused_requests_idxs_on_right = (
                    torch.nonzero(
                        active_requests_requiring_new_block[
                            active_requests_requiring_new_block_count:
                        ],
                        as_tuple=True,
                    )[0]
                    + active_requests_requiring_new_block_count
                    + self.paused_request_count
                )
                dst_idxs = torch.cat((active_request_ids_on_left, paused_requests_idxs_on_right))
                src_idxs = torch.cat((paused_requests_idxs_on_right, active_request_ids_on_left))
                self._move_book_keeping_tensors(
                    src_idxs=src_idxs, dst_idxs=dst_idxs, next_tokens=next_tokens
                )
                newly_paused_request_ids = self.request_ids[dst_idxs]

            self.paused_request_count += active_requests_requiring_new_block_count
            active_request_count -= active_requests_requiring_new_block_count

        # 6. Now that we have the requests in following order [Paused, Active, Finished]
        # We determine how many requests we can resume and resume them
        # Assign released blocks to paused requests.
        # todo: @shanmugamr, un-pause requests using FIFO, rather than LIFO.
        num_non_gtd_blocks = max(0, self.block_allocator.block_count_avail - self.gtd_block_count)
        if num_non_gtd_blocks:
            # if we have non-gtd blocks, use them. Do not dip into the gtd-block pool
            resume_request_count = min(num_non_gtd_blocks, self.paused_request_count)
        else:
            # only dip into the gtd-block pool if we have run out of non-gtd-blocks and the active
            # request count has fallen below a certain threshold.
            resume_request_count = min(
                max(self.gtd_request_count - active_request_count, 0), self.paused_request_count
            )

        self.paused_request_count -= resume_request_count
        active_request_count += resume_request_count
        assert active_request_count > 0, "active_request_count == %d." % active_request_count

        # finally, swap the chunked prefill to the end of the active requests to obey the invariant
        if self.chunked_prefill_request_id != -1:
            pos = torch.where(self.request_ids == self.chunked_prefill_request_id)[0][0]
            self._swap_book_keeping_tensors(
                src_idxs=torch.tensor([pos]),
                dst_idxs=torch.tensor([active_request_count + self.paused_request_count - 1]),
                next_tokens=next_tokens,
            )
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
            self.request_last_token_ids[: self.paused_request_count] = next_tokens[
                : self.paused_request_count
            ]

        if self.active_token_count > 0:
            self.request_last_token_ids[self.paused_request_count : self.total_request_count] = (
                next_tokens[self.paused_request_count : self.total_request_count]
            )

        # add_ and fill_ calls seems to work as intended with sliced indexing (i.e. x[3:5].add(...) or x[3:5].fill_)
        # but when another tensor is used for indexing, it does not work as expected (i.e. x[y] if x and y are torch tensors)
        self.request_kv_length_offsets[self.paused_request_count : self.total_request_count].add_(
            self.request_query_lengths[self.paused_request_count : self.total_request_count]
        )
        self.request_query_lengths[self.paused_request_count : self.total_request_count].fill_(1)
        self.token_to_pos_ids[: self.active_token_count] = self.request_kv_length_offsets[
            self.paused_request_count : self.total_request_count
        ]

        self.request_last_kv_block_offset[self.paused_request_count : self.total_request_count] = (
            self.request_last_kv_block_offset[self.paused_request_count : self.total_request_count]
            + 1
        ) % self.block_size_tokens

        # 8. We resume those requests by assigning blocks and updating bookkeeping tensors
        if resume_request_count > 0:
            assert torch.all(
                self.request_last_kv_block_offset[
                    self.paused_request_count : (self.paused_request_count + resume_request_count)
                ]
                == 0
            ), "The request_last_kv_block_offset should be 0 for the requests that just got resumed this step. "

            block_ids = self.block_allocator.allocate_memory_blocks(resume_request_count)
            if block_ids is None or block_ids.size(0) != resume_request_count:
                logger.warning(
                    "Unable to allocate %d KV blocks during sync resume; available=%d in_use=%d",
                    resume_request_count,
                    self.block_allocator.get_available_block_count(),
                    int(self.request_kv_block_counts[:self.total_request_count].sum().item()),
                )
                raise BlockOverflowError(None)
            row_idx = torch.arange(
                self.paused_request_count,
                self.paused_request_count + resume_request_count,
                device=torch.cuda.current_device(),
            )
            col_idx = self.request_kv_block_counts[
                self.paused_request_count : (self.paused_request_count + resume_request_count)
            ]
            self.request_to_kv_block_ids[row_idx, col_idx] = block_ids
            self.request_kv_block_counts[
                self.paused_request_count : (self.paused_request_count + resume_request_count)
            ] += 1
            self.request_last_kv_block_id[
                self.paused_request_count : (self.paused_request_count + resume_request_count)
            ] = block_ids

        # 9. We make relevant changes to the token bookkeeping tensors
        self.token_to_request_idx[: self.active_token_count] = torch.arange(
            self.paused_request_count, self.total_request_count, device=torch.cuda.current_device()
        )
        self.token_to_position_in_request[: self.active_token_count] = (
            self.request_kv_length_offsets[self.paused_request_count : self.total_request_count]
        )

        self.token_to_block_idx[: self.active_token_count] = self.request_last_kv_block_id[
            self.paused_request_count : self.total_request_count
        ]
        self.token_to_local_position_within_kv_block[: self.active_token_count] = (
            self.request_last_kv_block_offset[self.paused_request_count : self.total_request_count]
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

    def _update_requests_async(
        self, active_requests_mask: Tensor, new_tokens: Tensor
    ) -> Tensor:
        """Asynchronous update path with double-buffered metadata.

        This implements the 9-step algorithm with logical request reordering and double-buffered
        metadata slots. Requests are organized as: [Paused | Active] (Finished are removed).

        Args:
            active_requests_mask: Mask indicating which active requests are still running
            new_tokens: Newly sampled tokens for active requests

        Returns:
            Tensor of newly paused request IDs (or None)
        """
        metadata = self.context_metadata

        # =============================================================================
        # STEP 1: Create ext_active_mask and identify finished (BEFORE slot switch)
        # =============================================================================
        # Save chunked prefill state for post-processing
        metadata.chunked_prefill_request_id_post_process = metadata.chunked_prefill_request_id_old
        if metadata.chunked_prefill_request_id_old != -1:
            active_requests_mask[-1] = 1  # Keep chunked prefill request active

        # Validate counts
        active_request_count = (active_requests_mask == 1).sum().item()
        finished_request_count = (active_requests_mask == 0).sum().item()
        total_request_count_old = metadata.total_request_count_old
        paused_request_count_old = metadata.paused_request_count_old

        assert (
            active_request_count + finished_request_count + paused_request_count_old
            == total_request_count_old
        ), f"Request count mismatch: {active_request_count} + {finished_request_count} + {paused_request_count_old} != {total_request_count_old}"

        # Create extended active mask covering ALL old requests (paused + active)
        device = torch.cuda.current_device()
        ext_active_mask = torch.ones(
            total_request_count_old, dtype=torch.bool, device=device
        )
        ext_active_mask[paused_request_count_old:total_request_count_old] = active_requests_mask.to(device)

        # Use OLD select_map to reorder mask (this gives us the correct positions)
        if hasattr(metadata, 'select_map') and metadata.select_map is not None and metadata.select_map.numel() > 0:
            swapped_ext_active_mask = ext_active_mask[metadata.select_map]
        else:
            swapped_ext_active_mask = ext_active_mask

        # Identify finished requests in the NEW coordinate system
        finished_idxs = torch.nonzero(swapped_ext_active_mask == 0, as_tuple=True)[0]

        # =============================================================================
        # STEP 2: Switch metadata slot and copy counts
        # =============================================================================
        old_id = metadata.active_id
        metadata.activate_metadata(1 - old_id)
        metadata.copy_old_counts()

        # Get old tensors
        old_tensors = metadata.get_old_tensors()

        # =============================================================================
        # STEP 3: Reset attention and create request states
        # =============================================================================
        self.reset_attention_state()
        metadata.num_prefill_requests = 0  # All turn to decode

        # Define RequestState enum
        class RequestState(IntEnum):
            ACTIVE = 1
            PAUSED = 2
            FINISHED = 3

        # Create request_states tensor for the NEW slot
        request_states = torch.zeros(
            metadata.total_request_count, dtype=torch.int32, device=device
        )
        request_states[:metadata.paused_request_count] = RequestState.PAUSED.value
        request_states[metadata.paused_request_count:metadata.total_request_count] = RequestState.ACTIVE.value

        # =============================================================================
        # STEP 4: Release memory for finished requests and mark them
        # =============================================================================
        if finished_request_count > 0:
            kv_blocks_assigned = old_tensors['request_to_kv_block_ids'][finished_idxs]
            non_zero_values_in_kv_memory = kv_blocks_assigned[kv_blocks_assigned != -1]
            if non_zero_values_in_kv_memory.numel() > 0:
                self.block_allocator.release_memory_blocks(non_zero_values_in_kv_memory)

            # Mark finished in request_states
            request_states[finished_idxs] = RequestState.FINISHED.value

        active_request_count = request_states.eq(RequestState.ACTIVE.value).sum().item()

        # =============================================================================
        # STEP 5: Identify requests requiring new blocks → mark PAUSED
        # =============================================================================
        newly_paused_global_idxs = None
        if active_request_count > 0:
            num_tokens_in_last_block = old_tensors['request_last_kv_block_offset'][
                metadata.paused_request_count:metadata.total_request_count
            ]
            requests_requiring_new_block = (
                num_tokens_in_last_block == self.block_size_tokens - 1
            )

            # Don't pause chunked prefill request
            if metadata.chunked_prefill_request_id != -1:
                requests_requiring_new_block[-1] = False

            # Mark as PAUSED (only those not already finished)
            active_slice_states = request_states[metadata.paused_request_count:metadata.total_request_count]
            changing_to_paused_mask = requests_requiring_new_block & (active_slice_states != RequestState.FINISHED.value)

            if changing_to_paused_mask.any():
                newly_paused_global_idxs = torch.where(changing_to_paused_mask)[0] + metadata.paused_request_count
                request_states[newly_paused_global_idxs] = RequestState.PAUSED.value

        metadata.paused_request_count = request_states.eq(RequestState.PAUSED.value).sum().item()
        active_request_count = request_states.eq(RequestState.ACTIVE.value).sum().item()

        # =============================================================================
        # STEP 6: Determine resume count and mark resumed requests
        # =============================================================================
        # Calculate resume count based on available blocks
        num_non_gtd_blocks = max(0, self.block_allocator.get_available_block_count() - self.gtd_block_count)
        if num_non_gtd_blocks:
            resume_request_count = min(num_non_gtd_blocks, metadata.paused_request_count)
        else:
            resume_request_count = min(
                max(self.gtd_request_count - active_request_count, 0),
                metadata.paused_request_count
            )

        # Update counts
        metadata.paused_request_count -= resume_request_count
        active_request_count += resume_request_count

        # Resume paused requests (LIFO: last resume_request_count paused → active)
        paused_indices = torch.where(request_states == RequestState.PAUSED.value)[0]

        # Handle resume_request_count = 0 case explicitly (paused_indices[-0:] would return all elements!)
        if resume_request_count > 0:
            resume_indices = paused_indices[-resume_request_count:]
            request_states[resume_indices] = RequestState.ACTIVE.value
        else:
            resume_indices = torch.tensor([], dtype=torch.long, device=device)

        # Create resume tracking mask
        resume_mask = torch.zeros(metadata.total_request_count, dtype=torch.int32, device=device)
        if resume_request_count > 0:
            resume_mask[resume_indices] = 1

        # =============================================================================
        # STEP 7: Create select_map and tracking maps
        # =============================================================================
        paused_indices = torch.where(request_states == RequestState.PAUSED.value)[0]
        active_indices = torch.where(request_states == RequestState.ACTIVE.value)[0]
        finished_indices = torch.where(request_states == RequestState.FINISHED.value)[0]

        # Order: [Paused | Active] (Finished are dropped)
        select_map = torch.cat((paused_indices, active_indices))

        # Store tracking maps
        metadata.select_map = select_map
        metadata.active_map = active_indices
        metadata.paused_map = paused_indices
        metadata.finished_map = finished_indices

        # =============================================================================
        # STEP 8: Copy tensors and find resume positions in NEW coordinates
        # =============================================================================
        # Copy selected tensors from old to new slot
        metadata.select_old_tensors_to_active(select_map)

        # Find resumed requests in NEW coordinate system (after reordering)
        resume_idx_new = torch.where(resume_mask[select_map] == 1)[0]

        # Clear unused portion of block mapping
        new_tensors = metadata.get_active_tensors()
        new_tensors['request_to_kv_block_ids'][len(select_map):].fill_(-1)

        # Update final counts
        metadata.total_request_count = active_request_count + metadata.paused_request_count
        metadata.active_token_count = active_request_count  # All active are decode (1 token each)

        # Track newly paused request IDs
        newly_paused_request_ids = None
        if newly_paused_global_idxs is not None and newly_paused_global_idxs.numel() > 0:
            # Map newly paused indices from OLD to NEW coordinate system
            # Create inverse mapping: old_idx -> new_idx
            old_to_new = torch.full((len(request_states),), -1, dtype=torch.long, device=device)
            old_to_new[select_map] = torch.arange(len(select_map), device=device)

            # Get new positions of newly paused requests
            newly_paused_new_idxs = old_to_new[newly_paused_global_idxs]

            # Filter out any that weren't selected (shouldn't happen, but be safe)
            valid_mask = newly_paused_new_idxs >= 0
            newly_paused_new_idxs = newly_paused_new_idxs[valid_mask]

            # Extract request IDs
            if newly_paused_new_idxs.numel() > 0:
                newly_paused_request_ids = new_tensors['request_ids'][newly_paused_new_idxs]

                # Remove resumed requests (LIFO: they are at the end of paused section)
                if resume_request_count > 0 and newly_paused_request_ids.numel() > 0:
                    # Truncate the last resume_request_count entries
                    num_to_keep = max(0, newly_paused_request_ids.numel() - resume_request_count)
                    if num_to_keep > 0:
                        newly_paused_request_ids = newly_paused_request_ids[:num_to_keep]
                    else:
                        newly_paused_request_ids = None

        # =============================================================================
        # STEP 9: Update bookkeeping tensors and allocate blocks
        # =============================================================================
        # Update request bookkeeping tensors
        active_slice = slice(metadata.paused_request_count, metadata.total_request_count)

        new_tensors['request_kv_length_offsets'][active_slice].add_(
            new_tensors['request_query_lengths'][active_slice]
        )
        new_tensors['request_query_lengths'][active_slice].fill_(1)

        new_tensors['token_to_pos_ids'][:metadata.active_token_count] = (
            new_tensors['request_kv_length_offsets'][active_slice]
        )

        # Update last block offsets (increment and wrap)
        new_tensors['request_last_kv_block_offset'][active_slice] = (
            new_tensors['request_last_kv_block_offset'][active_slice] + 1
        ) % self.block_size_tokens

        # Allocate blocks for resumed requests (those with offset=0 after increment)
        if resume_request_count > 0:
            # Verify resumed requests have offset 0 (just wrapped around)
            assert torch.all(
                new_tensors['request_last_kv_block_offset'][resume_idx_new] == 0
            ), f"Resumed requests should have offset 0, got {new_tensors['request_last_kv_block_offset'][resume_idx_new]}"

            # Allocate new blocks
            block_ids = self.block_allocator.allocate_memory_blocks(resume_request_count)
            if block_ids is None or block_ids.size(0) != resume_request_count:
                logger.warning(
                    "Unable to allocate %d KV blocks for resumed requests; available=%d",
                    resume_request_count,
                    self.block_allocator.get_available_block_count(),
                )
                raise BlockOverflowError(None)

            # Assign blocks to resumed requests
            row_idx = resume_idx_new
            col_idx = new_tensors['request_kv_block_counts'][resume_idx_new]
            new_tensors['request_to_kv_block_ids'][row_idx, col_idx] = block_ids
            new_tensors['request_kv_block_counts'][resume_idx_new] += 1
            new_tensors['request_last_kv_block_id'][resume_idx_new] = block_ids

        # Update token bookkeeping tensors
        new_tensors['token_to_request_idx'][:metadata.active_token_count] = torch.arange(
            metadata.paused_request_count, metadata.total_request_count, device=device
        )
        new_tensors['token_to_position_in_request'][:metadata.active_token_count] = (
            new_tensors['request_kv_length_offsets'][active_slice]
        )
        new_tensors['token_to_block_idx'][:metadata.active_token_count] = (
            new_tensors['request_last_kv_block_id'][active_slice]
        )
        new_tensors['token_to_local_position_within_kv_block'][:metadata.active_token_count] = (
            new_tensors['request_last_kv_block_offset'][active_slice]
        )

        # Signal metadata ready for GPU consumption
        metadata.record_metadata_ready()

        return newly_paused_request_ids
