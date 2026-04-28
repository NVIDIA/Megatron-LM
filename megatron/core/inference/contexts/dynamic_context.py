# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import math
import operator
import warnings
from contextlib import nullcontext
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch import Tensor  # type: ignore

from megatron.core import parallel_state
from megatron.core.inference.batch_dimensions_utils import (
    CUDAGraphBatchDimensionBuilder,
    InferenceBatchDimensions,
)
from megatron.core.inference.config import (
    InferenceConfig,
    KVCacheManagementMode,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.unified_memory import (
    UnifiedMemoryUnsupportedError,
    create_unified_mempool,
)
from megatron.core.inference.utils import device_memory_summary, scatter_pack_valid, tensor_swap
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    get_layer_maps_from_layer_type_list,
)
from megatron.core.package_info import __version__ as mcore_version
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.utils import deprecate_args
from megatron.core.utils import divide as core_divide
from megatron.core.utils import get_pg_size, internal_api

from .attention_context.mamba_metadata import MambaMetadata, PrefixCachedMambaMetadata
from .base_context import BaseInferenceContext
from .kv_block_allocator import KVBlockAllocator
from .mamba_slot_allocator import MambaSlotAllocator
from .routing_metadata import RoutingMetadata

try:
    from .fused_kv_append_kernel import triton_append_key_value_cache
except ImportError:
    triton_append_key_value_cache = None

try:
    import flashinfer  # type: ignore # pylint: disable=unused-import

    HAVE_FLASHINFER = True
except ImportError:
    HAVE_FLASHINFER = False

try:
    from torch_memory_saver import torch_memory_saver

    torch_memory_saver.hook_mode = "torch"
    HAVE_TORCH_MEMORY_SAVER = True
except ImportError:
    HAVE_TORCH_MEMORY_SAVER = False

DEPRECATED_ARGS = [
    "params_dtype",
    "num_layers",
    "kv_channels",
    "num_attention_heads",
    "max_sequence_length",
    "buffer_size_gb",
    "paused_buffer_size_gb",
    "max_requests",
    "max_tokens",
    "block_size_tokens",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "pg_collection",
    "cache_mla_latent",
    "kv_lora_rank",
    "qk_pos_emb_head_dim",
    "num_cuda_graphs",
    "materialize_only_last_token_logits",
    "mamba_inference_state_config",
    "use_cuda_graphs_for_non_decode_steps",
    "use_flashinfer_fused_rope",
    "unified_memory_level",
    "cuda_graph_max_tokens",
    "cuda_graph_mixed_prefill_count",
    "metrics_writer",
    "request_metadata_types",
    "persist_cuda_graphs",
    "offload_kv_cache",
]


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
        _message = "" if message is None else f" | {message}"
        super().__init__(f"request {request_str}{_message}")
        self.request_id = request_id
        self.message = message
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


class TensorStateDeallocatedError(ContextOverflowError):
    """Context's tensor state is currently deallocated, such as when the engine
    has been suspended."""

    pass


class ContextErrorFactory:
    """Factory class for serializing/deserializing context errors."""

    @classmethod
    def serialize(cls, error: ContextOverflowError) -> dict:
        """Serialize error.

        Args:
            error (ContextOverflowError): Error.

        Returns:
            (dict) Serialized error data.
        """
        assert isinstance(error, ContextOverflowError)
        return {
            "type": type(error).__name__,
            "request_id": error.request_id,
            "message": error.message,
            "is_transient": error.is_transient,
        }

    @classmethod
    def deserialize(cls, obj: dict) -> ContextOverflowError:
        """Deserialize error.

        Args:
            obj (dict): Serialized error data.

        Returns:
            (ContextOverflowError) Deserialized error.
        """
        error_cls = {
            "ContextOverflowError": ContextOverflowError,
            "RequestOverflowError": RequestOverflowError,
            "TokenOverflowError": TokenOverflowError,
            "MaxSequenceLengthOverflowError": MaxSequenceLengthOverflowError,
            "BlockOverflowError": BlockOverflowError,
            "ActiveRequestCountOverflowError": ActiveRequestCountOverflowError,
        }[obj["type"]]
        error = ContextOverflowError(**{k: v for k, v in obj.items() if k != "type"})
        error.__class__ = error_cls  # todo (@lmcafee): better/safer alternative?
        return error


class UpdateRequestsSyncedCounters(NamedTuple):
    """Typed view of the once-per-step ``_ur.combined_sync`` GPU->CPU copy.

    Construct via :meth:`from_buffer` immediately after the ``.cpu().tolist()``.
    Field order matches the index constants
    ``DynamicInferenceContext._SYNC_*``; if those constants change, update
    :meth:`from_buffer` so it stays in sync — the buffer slot order is the
    only place this layout is asserted.
    """

    bucket0: int      # bucket_counts[0] (continuing)
    bucket1: int      # bucket_counts[1] (chunked)
    bucket2: int      # bucket_counts[2] (stayed-paused)
    bucket3: int      # bucket_counts[3] (newly paused)
    resume1: int      # resume_count from graph-1 resume
    resume2: int      # resume_count from graph-2 resume
    active_final: int  # new_active after evict+resume
    total_final: int   # new_total after evict+resume+chunked
    total_avail: int   # kv_block_allocator.total_avail_gpu
    mamba_free: int    # mamba free slot count
    evict_count: int   # eviction count

    @classmethod
    def from_buffer(cls, sv: List[int]) -> "UpdateRequestsSyncedCounters":
        """Wrap the int list returned by ``_ur.combined_sync.cpu().tolist()``."""
        # INVARIANT: positional construction; argument order must match the
        # _SYNC_* constants.
        return cls(*sv)


class UpdateRequestsScratch:
    """Static GPU scratch tensors and constants for the graphed update_requests path.

    All tensors are pre-allocated with static addresses required for CUDA graph
    capture. Lives as ``DynamicInferenceContext._ur`` and is constructed once
    by ``init_update_requests_state``.
    """

    def __init__(
        self,
        *,
        max_requests: int,
        max_tokens: int,
        num_speculative_tokens: int,
        max_release_per_step: int,
        kv_dummy_block_idx: int,
        sync_size: int,
        is_hybrid_model: bool,
    ):
        device = torch.cuda.current_device()

        # Constants used inside the body methods.
        self.num_generated_tokens = 1 + num_speculative_tokens
        self.max_allowed_active = min(
            max_requests, max_tokens // (num_speculative_tokens + 1)
        )

        # Classification scratch.
        self.sort_key = torch.full((max_requests,), 5, dtype=torch.int32, device=device)
        self.next_tokens = torch.zeros(max_requests, dtype=torch.int64, device=device)
        self.arange_long = torch.arange(max_requests, dtype=torch.int64, device=device)
        self.active_pre_gpu = torch.zeros(1, dtype=torch.int64, device=device)
        self.total_pre_gpu = torch.zeros(1, dtype=torch.int64, device=device)
        self.chunked_id_gpu = torch.full((1,), -1, dtype=torch.int32, device=device)
        self.active_mask = torch.zeros(max_requests, dtype=torch.uint8, device=device)
        self.bucket_counts = torch.zeros(5, dtype=torch.int64, device=device)
        self.swap_src = torch.empty(1, dtype=torch.int64, device=device)
        self.swap_dst = torch.empty(1, dtype=torch.int64, device=device)
        self.prev_last_block_ids = torch.empty(
            max_requests, dtype=torch.int32, device=device
        )
        # Read-only (max_requests,) view of a single 1. Used as the source for
        # ``bucket_counts.index_add_`` inside the classify graph body — the
        # scalar's stride-0 expansion avoids the per-call (max_requests * 8)
        # byte allocation a real ones-tensor would need.
        self.ones_long = torch.ones((), dtype=torch.int64, device=device).expand(
            max_requests
        )

        # Speculative tokens (None when num_speculative_tokens == 0).
        if num_speculative_tokens > 0:
            self.spec_tokens = torch.zeros(
                num_speculative_tokens, max_requests, dtype=torch.int64, device=device
            )
            # Pre-allocated constants for multi-token bookkeeping, avoiding
            # per-replay allocations inside the captured graph body.
            self.gen_arange = torch.arange(
                self.num_generated_tokens, dtype=torch.int64, device=device
            )
            self.pos_offset_pattern = self.gen_arange.repeat(max_requests)
            self.paused_spec_tokens_buf = torch.zeros(
                num_speculative_tokens, max_requests, dtype=torch.int64, device=device
            )
        else:
            self.spec_tokens = None
            self.gen_arange = None
            self.pos_offset_pattern = None
            self.paused_spec_tokens_buf = None

        # Scatter-pack scratch buffers for the graphed release path.
        # Size is max_release + 1: valid entries land in [0, num_valid),
        # invalid (-1 sentinel) entries all land at the sink slot at index
        # max_release and never get read. The +1 is the sink slot.
        self.release_pack_buf = torch.full(
            (max_release_per_step + 1,),
            kv_dummy_block_idx,
            dtype=torch.int32,
            device=device,
        )
        if is_hybrid_model:
            # Mamba release at most max_requests slots per call; +1 for sink.
            self.mamba_release_pack_buf = torch.zeros(
                max_requests + 1, dtype=torch.int32, device=device
            )
        else:
            self.mamba_release_pack_buf = None

        # GPU-resident boundaries consumed by the folded release inside the
        # classify graph and by the resume graph body. The classify graph
        # derives them from ``bucket_counts``:
        #   new_active_gpu = bucket[0] + bucket[1]  (continues + chunked)
        #   new_total_gpu  = new_active_gpu + bucket[2] + bucket[3]
        self.new_active_gpu = torch.zeros(1, dtype=torch.int64, device=device)
        self.new_total_gpu = torch.zeros(1, dtype=torch.int64, device=device)

        # Output scalar for the resume graph body: number of paused requests
        # brought back to active.
        self.resume_count_gpu = torch.zeros(1, dtype=torch.int64, device=device)

        # Eviction scratch: packed evict IDs buffer (+ 1 sink slot) and the
        # GPU-scalar evict count. Buffer dtype matches ``request_ids`` (int32)
        # because ``scatter_pack_valid`` writes via ``pack_buf[targets] =
        # values_flat`` which is an ``index_put_`` and rejects dtype mismatch.
        self.evict_ids_buf = torch.full(
            (max_requests + 1,), -1, dtype=torch.int32, device=device
        )
        self.evict_count_gpu = torch.zeros(1, dtype=torch.int64, device=device)

        # Newly-paused IDs capture buffer (+ 1 sink slot). Packed inside
        # graph 1 before eviction can modify request_ids. Same int32 dtype
        # constraint as ``evict_ids_buf``.
        self.newly_paused_ids_buf = torch.full(
            (max_requests + 1,), -1, dtype=torch.int32, device=device
        )

        # Static buffer for paused tokens, avoiding a per-step clone.
        # In _finalize_update_requests, the paused region of `next_tokens` is
        # copied here (GPU→GPU, no alloc) instead of cloned.
        self.paused_tokens_buf = torch.zeros(
            max_requests, dtype=torch.int64, device=device
        )

        # Combined sync buffer for the graph, read in a single .cpu().
        # Named offsets (DynamicInferenceContext._SYNC_*) index into it.
        self.combined_sync = torch.zeros(sync_size, dtype=torch.int64, device=device)


def get_mem_size_str(n_bytes: int) -> str:
    """Convert number of bytes to human-readable string."""
    if n_bytes == 0:
        return "0 bytes"
    for exp, suffix in ((4, "TB"), (3, "GB"), (2, "MB"), (3, "KB"), (0, "bytes")):
        nquery = int(1024**exp)
        if round(n_bytes / nquery) >= 1:
            return "%.3g %s" % (n_bytes / nquery, suffix)
    raise Exception(f"something went wrong, n_bytes={n_bytes}.")


@internal_api
# pylint: disable=line-too-long
class DynamicInferenceContext(BaseInferenceContext):
    """Inference context that is passed to the main model in order
    to efficiently calculate and store the KV cache during inference.

    The dynamic inference context manages both: 1) in-flight batching, and 2) a
    memory buffer for the block-level KV cache. For in-flight batching, requests of
    arbitrary sequence length may be added, paused, or removed from the context
    at any step. The only constraint is the maximum number of requests or tokens
    that the context is defined to support. For the block-level KV cache, a memory
    buffer is allocated up front (size `buffer_size_gb` if `unified_memory_level`
    == 0, or `buffer_size_gb + paused_buffer_size_gb` if `unified_memory_level` ==
    1), that is divided into blocks and dynamically assigned to requests. At any
    given step, any unassigned blocks equate to unused space.

    Args:
        model_config (TransformerConfig): Model config.
        inference_config (InferenceConfig): Inference config.
    """

    DEFAULT_MAX_TOKENS = 16384
    TOKEN_ROUNDER = 64
    REQUEST_ROUNDER = 4
    TMS_TAG = "inference_context"

    # Named offsets into ``_ur.combined_sync`` (the single GPU->CPU sync
    # buffer read once per ``update_requests`` call).
    # Graph 1 outputs:
    _SYNC_BUCKET0 = 0  # bucket_counts[0] (continuing)
    _SYNC_BUCKET1 = 1  # bucket_counts[1] (chunked)
    _SYNC_BUCKET2 = 2  # bucket_counts[2] (stayed-paused)
    _SYNC_BUCKET3 = 3  # bucket_counts[3] (newly paused)
    _SYNC_RESUME1 = 4  # resume_count from graph-1 resume
    # Graph 2 outputs:
    _SYNC_RESUME2 = 5  # resume_count from graph-2 resume
    _SYNC_ACTIVE_FINAL = 6  # new_active after evict+resume
    _SYNC_TOTAL_FINAL = 7  # new_total after evict+resume+chunked
    _SYNC_TOTAL_AVAIL = 8  # kv_block_allocator.total_avail_gpu
    _SYNC_MAMBA_FREE = 9  # mamba free slot count
    _SYNC_EVICT_COUNT = 10  # eviction count
    _SYNC_SIZE = 11

    @deprecate_args(
        *DEPRECATED_ARGS,
        message=(
            "Argument `{name}` has been deprecated. "
            "Only pass `model_config` and `inference_config`"
        ),
    )
    def __init__(self, model_config: TransformerConfig, inference_config: InferenceConfig):
        super().__init__(inference_config=inference_config)

        # Prefix caching configuration
        self.enable_prefix_caching = inference_config.enable_prefix_caching
        self.prefix_caching_eviction_policy = inference_config.prefix_caching_eviction_policy
        self.prefix_caching_coordinator_policy = inference_config.prefix_caching_coordinator_policy

        # Hyperparameter for choosing to prioritize prefix hit matches vs minimizing idle load
        self.prefix_caching_routing_alpha = inference_config.prefix_caching_routing_alpha

        # Prefix caching hit tracking (accumulated, reset by engine after logging).
        self.prefix_cache_hits = 0  # requests that matched at least one cached block
        self.prefix_cache_blocks_matched = 0  # total matched blocks across all requests

        # Engine step counter (used for logging, metrics, and event tracking)
        self.step_count = 0

        self.cache_mla_latent = (
            isinstance(model_config, MLATransformerConfig) and model_config.cache_mla_latents
        )
        if self.cache_mla_latent:
            assert (
                inference_config.block_size_tokens == 64
            ), "Flash MLA requires a block size of 64. Set --inference-dynamic-batching-block-size 64 to fix this assert"

        # Per partition num heads and hidden size.
        num_attention_heads = model_config.num_query_groups or model_config.num_attention_heads
        projection_size = model_config.kv_channels * num_attention_heads
        pg_collection = inference_config.pg_collection
        if pg_collection is not None:
            tp_size = get_pg_size(pg_collection.tp)
            pp_size = get_pg_size(pg_collection.pp)
        else:
            tp_size = model_config.tensor_model_parallel_size
            pp_size = model_config.pipeline_model_parallel_size
        self.hidden_size_per_attention_head = core_divide(projection_size, num_attention_heads)
        if num_attention_heads >= tp_size:
            self.num_attention_heads_per_partition = core_divide(num_attention_heads, tp_size)
        else:
            self.num_attention_heads_per_partition = 1

        self.num_speculative_tokens = inference_config.num_speculative_tokens
        assert self.num_speculative_tokens < inference_config.block_size_tokens, (
            f"num_speculative_tokens ({self.num_speculative_tokens}) must be < "
            f"block_size_tokens ({inference_config.block_size_tokens})"
        )

        # Cache the PP group we should use for PP collectives inside the context.
        # If the model provides a pg_collection with a pp group, prefer it.
        # Otherwise:
        # - for PP=1 we don't need a PP group at all
        # - for PP>1 we require Megatron parallel_state to be initialized
        if pg_collection is not None and get_pg_size(pg_collection.pp) > 1:
            self.pipeline_parallel_group = pg_collection.pp
        elif pp_size > 1:
            self.pipeline_parallel_group = parallel_state.get_pipeline_model_parallel_group()
        else:
            self.pipeline_parallel_group = None

        if pg_collection is not None:
            self.expert_model_parallel_group = pg_collection.ep
        elif parallel_state.get_expert_model_parallel_world_size() > 1:
            self.expert_model_parallel_group = parallel_state.get_expert_model_parallel_group()
        else:
            self.expert_model_parallel_group = None

        # Mamba states.
        mamba_inference_state_config = inference_config.mamba_inference_state_config
        self.is_hybrid_model = mamba_inference_state_config is not None
        if self.is_hybrid_model:
            self.mamba_conv_states_shape = mamba_inference_state_config.conv_states_shape
            self.mamba_ssm_states_shape = mamba_inference_state_config.ssm_states_shape
            self.mamba_conv_states_dtype = mamba_inference_state_config.conv_states_dtype
            self.mamba_ssm_states_dtype = mamba_inference_state_config.ssm_states_dtype
            self.mamba_chunk_size = mamba_inference_state_config.mamba_chunk_size

            # For hybrid models, the layer map converts the global layer index to the
            # corresponding attention layer index or Mamba layer index depending on the
            # layer type.
            attention_layer_map, dsa_layer_map, gdn_layer_map, mamba_layer_map = (
                operator.itemgetter(
                    Symbols.ATTENTION, Symbols.DS_ATTENTION, Symbols.GDN, Symbols.MAMBA
                )(get_layer_maps_from_layer_type_list(mamba_inference_state_config.layer_type_list))
            )

            if len(gdn_layer_map) > 0:
                raise NotImplementedError("GDN layers are not supported for inference.")

            self.num_attention_layers = len(attention_layer_map) + len(dsa_layer_map)
            self.num_mamba_layers = len(mamba_layer_map)
            self.layer_map = attention_layer_map | dsa_layer_map | mamba_layer_map
        else:
            # The layer map is the identity function for pure Transformer models.
            self.num_attention_layers = model_config.num_layers // pp_size
            self.num_mamba_layers = 0
            (self.mamba_conv_states_shape, self.mamba_ssm_states_shape) = (None, None)
            self.layer_map = {i: i for i in range(self.num_attention_layers)}

        if self.num_attention_layers == 0:
            raise NotImplementedError(
                f"Using `DynamicInferenceContext` with no attention is not supported."
            )

        # Block size tokens, bytes.
        kv_dtype_size_bytes = model_config.params_dtype.itemsize
        self.block_size_tokens = inference_config.block_size_tokens
        if self.cache_mla_latent:
            #   one vector  c_t  (rank)  +  optional RoPE phase slice
            self.kv_reduced_dim = model_config.kv_lora_rank + model_config.qk_pos_emb_head_dim
            self.block_size_bytes = (
                kv_dtype_size_bytes
                * self.num_attention_layers
                * self.block_size_tokens
                * self.kv_reduced_dim
            )
        else:
            self.block_size_bytes = (
                kv_dtype_size_bytes
                * 2  # key, value
                * self.num_attention_layers
                * self.block_size_tokens
                * self.num_attention_heads_per_partition
                * self.hidden_size_per_attention_head
            )
        assert self.block_size_bytes > 0

        mamba_states_memory_per_request = 0
        if self.is_hybrid_model:
            mamba_states_memory_per_request += (
                math.prod(self.mamba_conv_states_shape) * self.mamba_conv_states_dtype.itemsize
            )
            mamba_states_memory_per_request += (
                math.prod(self.mamba_ssm_states_shape) * self.mamba_ssm_states_dtype.itemsize
            )
            mamba_states_memory_per_request *= self.num_mamba_layers
            if self.num_speculative_tokens > 0:
                # Add memory for intermediate conv and SSM states
                intermediate_memory_per_request = (
                    math.prod(self.mamba_conv_states_shape) * self.mamba_conv_states_dtype.itemsize
                    + math.prod(self.mamba_ssm_states_shape) * self.mamba_ssm_states_dtype.itemsize
                )
                intermediate_memory_per_request *= self.num_mamba_layers
                intermediate_memory_per_request *= self.num_speculative_tokens + 1
                mamba_states_memory_per_request += intermediate_memory_per_request

        # Unified memory and general tensor management.
        self.unified_memory_level = inference_config.unified_memory_level
        self.static_kv_memory_pointers = inference_config.static_kv_memory_pointers
        self.kv_cache_management_mode = inference_config.kv_cache_management_mode

        if self.unified_memory_level != 0:
            try:
                self.unified_memory_mempool = create_unified_mempool()
            except UnifiedMemoryUnsupportedError:
                if torch.distributed.get_rank() == 0:
                    warnings.warn(
                        "Unified memory requested but not available; defaulting to GPU memory."
                    )
                self.unified_memory_level = 0
        # If we are in a mode that requires static KV memory pointers,
        # we must have either UVM or torch_memory_saver.
        if (
            self.static_kv_memory_pointers
            and self.kv_cache_management_mode != KVCacheManagementMode.PERSIST
        ):
            assert HAVE_TORCH_MEMORY_SAVER or self.unified_memory_level != 0, (
                "Static KV memory pointers require UVM or torch_memory_saver when not persisted. "
                "Use --rl-kv-cache-management-mode=persist, UVM, or install torch_memory_saver."
            )

        # When not using `torch_memory_saver`, we manually offload/restore tensors.
        # We use storage resize, similar to the logic in `core/distributed/param_and_grad_buffer.py`
        self._offloadable_tensor_names: set[str] = set()
        self._offloadable_cpu_backups: dict[str, torch.Tensor] = {}
        self._offloadable_storage_sizes: dict[str, int] = {}
        self._uses_torch_memory_saver: bool = False

        # Initialize block allocator.
        buffer_size_bytes = int(inference_config.buffer_size_gb * 1024**3)
        paused_buffer_size_bytes = (
            0
            if inference_config.paused_buffer_size_gb is None
            else int(inference_config.paused_buffer_size_gb * 1024**3)
        )

        mamba_max_requests = float('inf')

        if (mamba_memory_ratio := inference_config.mamba_memory_ratio) is not None:
            assert self.is_hybrid_model
            assert mamba_memory_ratio > 0 and mamba_memory_ratio < 1

            # Calculate total memory before partition
            total_memory = buffer_size_bytes + paused_buffer_size_bytes
            mamba_memory_bytes = total_memory * mamba_memory_ratio
            mamba_max_requests = int(mamba_memory_bytes // mamba_states_memory_per_request)

            # Reduce buffer sizes for KV cache
            buffer_size_bytes = int(buffer_size_bytes * (1.0 - mamba_memory_ratio))
            paused_buffer_size_bytes = int(paused_buffer_size_bytes * (1.0 - mamba_memory_ratio))

            block_count = buffer_size_bytes // self.block_size_bytes
            block_count = max(2, block_count)  # need >= 1 active block + 1 dummy block
            paused_block_count = paused_buffer_size_bytes // self.block_size_bytes
        elif self.is_hybrid_model and inference_config.max_requests is not None:
            # Auto-derive mamba/KV split from max_requests. Allocate exactly enough
            # mamba memory for max_requests, and give the rest to KV cache blocks.
            total_memory = buffer_size_bytes + paused_buffer_size_bytes
            mamba_memory_needed = inference_config.max_requests * mamba_states_memory_per_request
            assert mamba_memory_needed < total_memory, (
                f"Not enough memory for {inference_config.max_requests} mamba requests. "
                f"Need {mamba_memory_needed / 1024**3:.2f} GB for mamba states, "
                f"but total buffer is {total_memory / 1024**3:.2f} GB."
            )
            mamba_max_requests = inference_config.max_requests

            # Subtract mamba memory proportionally from active and paused buffers.
            mamba_memory_ratio = mamba_memory_needed / total_memory
            buffer_size_bytes = int(buffer_size_bytes * (1.0 - mamba_memory_ratio))
            paused_buffer_size_bytes = int(paused_buffer_size_bytes * (1.0 - mamba_memory_ratio))

            block_count = buffer_size_bytes // self.block_size_bytes
            block_count = max(2, block_count)  # need >= 1 active block + 1 dummy block
            paused_block_count = paused_buffer_size_bytes // self.block_size_bytes
        else:
            block_count = buffer_size_bytes // (
                self.block_size_bytes + mamba_states_memory_per_request
            )
            block_count = max(2, block_count)  # need >= 1 active block + 1 dummy block
            paused_block_count = paused_buffer_size_bytes // (
                self.block_size_bytes + mamba_states_memory_per_request
            )

        # If using pipeline parallelism synchronize the total block count in case the
        # pipeline stages have different layer allocations. Non-uniform block counts
        # can lead to some ranks pausing requests earlier than other ranks
        # (i.e., divergence in the scheduling behavior).
        if pp_size > 1:
            block_count_tensor = torch.tensor(
                (block_count, paused_block_count),
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(
                block_count_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=self.pipeline_parallel_group,
            )
            block_count = block_count_tensor[0].item()
            paused_block_count = block_count_tensor[1].item()

        # Initialize context state (needed before allocator for block_bag pre-sizing).
        self.params_dtype = model_config.params_dtype
        self.max_sequence_length = inference_config.max_sequence_length

        # Block ids. With speculative decoding, blocks are pre-allocated when the
        # last block offset >= block_size - 1 - num_speculative_tokens, so we may
        # need one extra block beyond what max_sequence_length alone requires.
        self.max_kv_block_count = math.ceil(self.max_sequence_length / self.block_size_tokens)
        if self.num_speculative_tokens > 0:
            self.max_kv_block_count += 1

        # Compute total_count and max_requests before constructing the allocator.
        total_count = (
            block_count if self.unified_memory_level == 0 else block_count + paused_block_count
        )
        if inference_config.max_requests is None:
            # Maximize compute utilization by defaulting to 1 block per request.
            self.max_requests = total_count

            # Adjust max_requests for Mamba memory constraints if necessary
            if self.is_hybrid_model and mamba_max_requests < self.max_requests:
                self.max_requests = int(mamba_max_requests)

            self.max_requests = self.max_requests // tp_size * tp_size
            self.max_requests = self.max_requests // self.REQUEST_ROUNDER * self.REQUEST_ROUNDER
        else:
            # User can control request overflow via max_requests.
            self.max_requests = inference_config.max_requests

        self.kv_block_allocator = KVBlockAllocator(
            context=self,
            total_count=total_count,
            paused_count=paused_block_count,
            max_kv_block_count=self.max_kv_block_count,
            max_requests=self.max_requests,
            enable_prefix_caching=self.enable_prefix_caching,
            prefix_caching_eviction_policy=self.prefix_caching_eviction_policy,
        )

        # Track request metadata.
        request_metadata_types = inference_config.request_metadata_types
        if request_metadata_types is None:
            request_metadata_types = DynamicInferenceRequest.get_metadata_types()
        self.request_metadata_types = request_metadata_types

        assert (
            self.max_requests % tp_size == 0
        ), f"max_requests must be divisible by tp_size ({tp_size}), but got {self.max_requests}"

        self.max_tokens = inference_config.max_tokens or self.DEFAULT_MAX_TOKENS

        min_tokens = self.max_requests * (self.num_speculative_tokens + 1)
        assert self.max_tokens >= min_tokens, (
            f"max_tokens ({self.max_tokens}) must be >= "
            f"max_requests * (num_speculative_tokens + 1) = {min_tokens}, "
            "to have consistency between cuda graph sizes and the block table size."
        )

        self.num_prefill_requests = 0

        # Per-step max sequence lengths consumed by the attention kernels. Set
        # by `prepare_attn_init` (graph mode, derived from padded dims) or by
        # `finalize_attn_init` (non-graph mode, computed via torch.max + .item()).
        self._max_seqlen_q: int = 0
        self._max_seqlen_k: int = 0

        # Hooks called at the tail of `run_attn_init_graph_body`.
        self._init_body_hooks: List[Callable[[bool], None]] = []

        self.moe_enable_routing_replay = model_config.moe_enable_routing_replay
        self.moe_routing_metadata = None
        if self.moe_enable_routing_replay:
            assert (
                model_config.num_moe_experts is not None
            ), "Router recording/replay requested but no MoE experts specified!"
            self.moe_routing_metadata = RoutingMetadata(self, model_config.moe_router_topk)

        # CUDA graph config list
        self.use_cuda_graphs_for_non_decode_steps = (
            inference_config.use_cuda_graphs_for_non_decode_steps
        )
        self.cuda_graph_batch_dimensions_list, self.cuda_graph_token_counts = (
            CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
                tp_size=tp_size,
                num_cuda_graphs=inference_config.num_cuda_graphs,
                cuda_graph_max_tokens=self.max_requests * (self.num_speculative_tokens + 1),
                cuda_graph_mixed_prefill_request_count=inference_config.cuda_graph_mixed_prefill_count,
                max_requests=self.max_requests,
                max_tokens=self.max_tokens,
                max_sequence_length=self.max_sequence_length,
                use_cuda_graphs_for_non_decode_steps=self.use_cuda_graphs_for_non_decode_steps,
                num_speculative_tokens=self.num_speculative_tokens,
            )
        )

        self.smallest_non_decode_cuda_graph_size = min(
            inference_config.cuda_graph_mixed_prefill_count, self.max_requests
        )

        # Deal with chunked prefill
        self.enable_chunked_prefill = inference_config.enable_chunked_prefill

        # FlashInfer.
        if inference_config.use_flashinfer_fused_rope is True:
            assert HAVE_FLASHINFER, "flashinfer is not installed"
        elif inference_config.use_flashinfer_fused_rope is None:
            inference_config.use_flashinfer_fused_rope = HAVE_FLASHINFER
        self.use_flashinfer_fused_rope = inference_config.use_flashinfer_fused_rope

        # Allocate GPU state.
        self.is_tensor_state_allocated = False
        self.initialize_all_tensors()

        # Allocate update_requests scratch tensors.  Pre-sizing block_bag and
        # mamba_state_free_slots in their constructors (above) avoids the
        # reallocations that used to shift the caching allocator's free-block
        # topology.  With no reallocations, these allocations are pure additions
        # that don't disturb subsequent CUDA graph captures.
        self.init_update_requests_state()

        # Print info.
        active_blocks = self.kv_block_allocator.active_count
        total_blocks = self.kv_block_allocator.total_count
        paused_blocks = self.kv_block_allocator.paused_count
        active_kv_bytes = active_blocks * self.block_size_bytes
        total_kv_bytes = total_blocks * self.block_size_bytes
        paused_kv_bytes = paused_blocks * self.block_size_bytes

        log_lines = [
            "DynamicInferenceContext: configuration summary",
            f"  max_requests:            {self.max_requests}",
            f"  max_tokens:              {self.max_tokens}",
            f"  max_sequence_length:     {self.max_sequence_length}",
            f"  block_size_tokens:       {self.block_size_tokens}",
            f"  max_kv_blocks_per_req:   {self.max_kv_block_count}",
            f"  KV cache:",
            f"    block_size_bytes:      {get_mem_size_str(self.block_size_bytes)}",
            f"    active_blocks:         {active_blocks} ({get_mem_size_str(active_kv_bytes)})",
            f"    paused_blocks:         {paused_blocks} ({get_mem_size_str(paused_kv_bytes)})",
            f"    total_blocks:          {total_blocks} ({get_mem_size_str(total_kv_bytes)})",
        ]

        if self.is_hybrid_model:
            mamba_conv_bytes = (
                math.prod(self.mamba_conv_states_shape)
                * self.mamba_conv_states_dtype.itemsize
                * self.num_mamba_layers
            )
            mamba_ssm_bytes = (
                math.prod(self.mamba_ssm_states_shape)
                * self.mamba_ssm_states_dtype.itemsize
                * self.num_mamba_layers
            )
            mamba_bytes_per_req = mamba_conv_bytes + mamba_ssm_bytes
            mamba_total_bytes = mamba_bytes_per_req * self.max_requests
            log_lines += [
                f"  Mamba states:",
                f"    num_mamba_layers:      {self.num_mamba_layers}",
                f"    conv_state_shape:      {self.mamba_conv_states_shape}",
                f"    ssm_state_shape:       {self.mamba_ssm_states_shape}",
                f"    per_request:           {get_mem_size_str(mamba_bytes_per_req)}",
                f"    total ({self.max_requests} requests):  {get_mem_size_str(mamba_total_bytes)}",
            ]

            if self.num_speculative_tokens > 0:
                spec_multiplier = self.num_speculative_tokens + 1
                spec_bytes_per_req = mamba_bytes_per_req * spec_multiplier
                spec_total_bytes = spec_bytes_per_req * self.max_requests
                log_lines += [
                    f"  Mamba speculative buffers (num_speculative_tokens={self.num_speculative_tokens}):",
                    f"    per_request:           {get_mem_size_str(spec_bytes_per_req)}",
                    f"    total ({self.max_requests} requests):  {get_mem_size_str(spec_total_bytes)}",
                ]

            if self.is_mamba_prefix_caching_enabled:
                prefix_cache_bytes = int(inference_config.prefix_caching_mamba_gb * 1024**3)
                prefix_cache_slots = prefix_cache_bytes // mamba_bytes_per_req
                log_lines += [
                    f"  Mamba prefix cache:",
                    f"    budget:                {get_mem_size_str(prefix_cache_bytes)}",
                    f"    slots:                 {prefix_cache_slots}",
                    f"    per_slot:              {get_mem_size_str(mamba_bytes_per_req)}",
                ]

        if inference_config._verbose and torch.distributed.get_rank() == 0:
            logging.info("\n".join(log_lines))

    def _allocate_memory_buffer(self):
        """Allocate the KV cache memory buffer."""
        if self.cache_mla_latent:
            self.memory_buffer = torch.empty(
                (
                    self.num_attention_layers,
                    self.kv_block_allocator.total_count,
                    self.block_size_tokens,
                    self.kv_reduced_dim,
                ),
                dtype=self.params_dtype,
                device=torch.cuda.current_device(),
            )
        else:
            self.memory_buffer = torch.empty(
                (
                    2,  # key and value
                    self.num_attention_layers,
                    self.kv_block_allocator.total_count,
                    self.block_size_tokens,
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                ),
                dtype=self.params_dtype,
                device=torch.cuda.current_device(),
            )
        if (
            self.kv_cache_management_mode == KVCacheManagementMode.OFFLOAD
            and not self._uses_torch_memory_saver
        ):
            assert self.unified_memory_level == 0
            self._offloadable_tensor_names.add("memory_buffer")
            self._offloadable_cpu_backups["memory_buffer"] = torch.empty_like(
                self.memory_buffer, device="cpu"
            ).pin_memory()

    def _allocate_mamba_states(self):
        """Allocate Mamba state buffers for hybrid models."""
        if self.is_hybrid_model:
            self.mamba_conv_states = torch.empty(
                (self.num_mamba_layers, self.max_requests) + self.mamba_conv_states_shape,
                dtype=self.mamba_conv_states_dtype,
                device=torch.cuda.current_device(),
            )
            self.mamba_ssm_states = torch.empty(
                (self.num_mamba_layers, self.max_requests) + self.mamba_ssm_states_shape,
                dtype=self.mamba_ssm_states_dtype,
                device=torch.cuda.current_device(),
            )
            if self.num_speculative_tokens > 0:
                self.mamba_intermediate_conv_states = torch.empty(
                    (
                        self.num_mamba_layers,
                        self.max_requests,
                        self.num_speculative_tokens + 1,
                        *self.mamba_conv_states_shape,
                    ),
                    dtype=self.mamba_conv_states_dtype,
                    device=torch.cuda.current_device(),
                )
                self.mamba_intermediate_ssm_states = torch.empty(
                    (
                        self.num_mamba_layers,
                        self.max_requests,
                        self.num_speculative_tokens + 1,
                        *self.mamba_ssm_states_shape,
                    ),
                    dtype=self.mamba_ssm_states_dtype,
                    device=torch.cuda.current_device(),
                )
            if (
                self.kv_cache_management_mode == KVCacheManagementMode.OFFLOAD
                and not self._uses_torch_memory_saver
            ):
                assert self.unified_memory_level == 0
                self._offloadable_tensor_names.add("mamba_conv_states")
                self._offloadable_cpu_backups["mamba_conv_states"] = torch.empty_like(
                    self.mamba_conv_states, device="cpu"
                ).pin_memory()
                self._offloadable_tensor_names.add("mamba_ssm_states")
                self._offloadable_cpu_backups["mamba_ssm_states"] = torch.empty_like(
                    self.mamba_ssm_states, device="cpu"
                ).pin_memory()
                if self.num_speculative_tokens > 0:
                    self._offloadable_tensor_names.add("mamba_intermediate_conv_states")
                    self._offloadable_cpu_backups["mamba_intermediate_conv_states"] = (
                        torch.empty_like(
                            self.mamba_intermediate_conv_states, device="cpu"
                        ).pin_memory()
                    )
                    self._offloadable_tensor_names.add("mamba_intermediate_ssm_states")
                    self._offloadable_cpu_backups["mamba_intermediate_ssm_states"] = (
                        torch.empty_like(
                            self.mamba_intermediate_ssm_states, device="cpu"
                        ).pin_memory()
                    )
        else:
            self.mamba_metadata = None

    def initialize_all_tensors(self) -> None:
        """Allocate all GPU state during initial construction."""
        # Mark allocated.
        if self.is_tensor_state_allocated:
            return
        self.is_tensor_state_allocated = True

        # Validate no tensors allocated prior to this method.
        for key in vars(self).keys():
            value = getattr(self, key)
            assert not isinstance(value, torch.Tensor), (
                "All tensors should be allocated within `initialize_all_tensors()`. "
                f"Please move tensor '{key}'."
            )

        # Per-request state.
        self.request_ids = torch.full(
            (self.max_requests,), -1, dtype=torch.int32, device=torch.cuda.current_device()
        )
        # request_query_lengths is the input prompt tokens length during prefill phase (1st step) and then 1 for the decode phase (i.e During generation)
        self.request_query_lengths = torch.empty_like(self.request_ids)
        # True only for a new request , then after a forward pass it is set to False
        self.request_in_prefill_status_tensor = torch.empty_like(self.request_ids)
        # request_output_lengths is len(input_prompt_tokens) + num_tokens_to_generate
        self.request_output_lengths = torch.empty_like(self.request_ids)
        # request_kv_length_offsets is the same as query length during prefill phase (1st step) and then 1 for the decode phase (i.e During generation)
        self.request_kv_length_offsets = torch.empty_like(self.request_ids)
        self.request_kv_block_counts = torch.empty_like(self.request_ids)
        self.request_last_kv_block_id = torch.empty_like(self.request_ids)
        # request_last_kv_block_offset represents number of tokens in the last kv block
        self.request_last_kv_block_offset = torch.empty_like(self.request_ids)
        self.request_to_kv_block_ids = torch.full(
            (self.max_requests, self.max_kv_block_count),
            -1,
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )

        # Track request metadata.
        self.request_metadata = {
            label: torch.empty(
                (self.max_requests,), dtype=dtype, device=torch.cuda.current_device()
            )
            for label, dtype in self.request_metadata_types
        }

        # Per-token state.
        self.token_to_input_ids = torch.full(
            (self.max_tokens,), 0, dtype=torch.long, device=torch.cuda.current_device()
        )
        self.token_to_pos_ids = torch.full_like(self.token_to_input_ids, 0)
        self.token_to_request_idx = torch.empty_like(self.token_to_input_ids)
        self.token_to_block_idx = torch.empty_like(self.token_to_input_ids)
        # i.e For a set of tokens A B C D E F ..  and block_size 4:
        # token_to_position_in_request is  [0, 1, 2, 3, 4, 5]
        # token_to_local_position_within_kv_block is [0 , 1, 2, 3, 0, 1, 2]
        self.token_to_position_in_request = torch.empty_like(self.token_to_input_ids)
        self.token_to_local_position_within_kv_block = torch.empty_like(self.token_to_input_ids)

        # Static tensor addresses of active slices to enable fast inference kernels.
        self.active_request_ids = torch.empty_like(self.request_ids, dtype=torch.int64)
        self.active_request_query_lengths = torch.empty_like(self.request_query_lengths)
        self.active_request_to_kv_block_ids = torch.empty_like(self.request_to_kv_block_ids)
        self.active_sequence_lengths = torch.empty_like(self.request_query_lengths)
        self.active_request_last_token_idxs = torch.empty_like(self.request_query_lengths)

        # Cumulative tensors consumed directly by attention kernels.
        self.cu_active_request_query_lengths = torch.zeros(
            (self.max_requests + 1,), dtype=torch.int32, device=torch.cuda.current_device()
        )
        self.cu_active_sequence_lengths = torch.zeros(
            (self.max_requests + 1,), dtype=torch.int32, device=torch.cuda.current_device()
        )

        # GPU scalars for freely-varying counts. Written from Python ints each
        # step; used by graphable ops via torch.where so slice shapes stay fixed.
        # Packed into one contiguous tensor for a single pinned H2D copy.
        self._context_op_metadata_gpu = torch.zeros(
            6, dtype=torch.int32, device=torch.cuda.current_device()
        )
        self._context_op_metadata_cpu = torch.zeros(6, dtype=torch.int32).pin_memory()
        self._real_request_count_gpu = self._context_op_metadata_gpu[0:1]
        self._real_token_count_gpu = self._context_op_metadata_gpu[1:2]
        self._real_decode_count_gpu = self._context_op_metadata_gpu[2:3]
        self._real_prefill_count_gpu = self._context_op_metadata_gpu[3:4]
        self._total_request_count_gpu = self._context_op_metadata_gpu[4:5]
        self._paused_request_count_gpu = self._context_op_metadata_gpu[5:6]

        # Pre-allocated index tensors for graphable ops (static addresses).
        self._arange_requests = torch.arange(
            self.max_requests, dtype=torch.int64, device=torch.cuda.current_device()
        )
        self._arange_tokens = torch.arange(
            self.max_tokens, dtype=torch.int64, device=torch.cuda.current_device()
        )

        if self.is_hybrid_model:
            self.active_mamba_indices = torch.empty_like(self.request_query_lengths)

        # NOTE: Need to build this outside the UVM / TMS context to avoid IMA.
        if self.is_hybrid_model:
            mamba_metadata_cls = (
                PrefixCachedMambaMetadata
                if self.is_mamba_prefix_caching_enabled
                else MambaMetadata
            )
            self.mamba_metadata = mamba_metadata_cls(
                max_requests=self.max_requests,
                max_tokens=self.max_tokens,
                mamba_chunk_size=self.mamba_chunk_size,
                d_conv=self.mamba_conv_states_shape[-1],
            )

        # Allocate large non-graphed buffers.
        need_static_addr = (
            self.static_kv_memory_pointers
            and self.kv_cache_management_mode != KVCacheManagementMode.PERSIST
        )

        ctx_manager = nullcontext()
        if self.unified_memory_level != 0:
            ctx_manager = torch.cuda.use_mem_pool(self.unified_memory_mempool)
        elif HAVE_TORCH_MEMORY_SAVER and need_static_addr:
            ctx_manager = torch_memory_saver.region(
                tag=self.TMS_TAG,
                enable_cpu_backup=(self.kv_cache_management_mode == KVCacheManagementMode.OFFLOAD),
            )
            self._uses_torch_memory_saver = True
        with ctx_manager:
            self._allocate_memory_buffer()
            self._allocate_mamba_states()

        # Allocate Mamba prefix cache if configured
        self.mamba_slot_allocator: Optional[MambaSlotAllocator] = None
        if self.is_hybrid_model and self.is_mamba_prefix_caching_enabled:
            self._allocate_mamba_cache(self.config.prefix_caching_mamba_gb)

        # Reset tensor-related metadata.
        self.reset_metadata()

    def reinitialize_inference_state_buffers(self):
        """Restore large tensors (KV cache, Mamba states) after a suspend.

        Called by the engine during `resume()`. Initial allocation is in `initialize_all_tensors()`.
        """
        if self.is_tensor_state_allocated:
            return
        self.is_tensor_state_allocated = True

        if self.kv_cache_management_mode == KVCacheManagementMode.PERSIST:
            return

        if self.unified_memory_level != 0 or self._uses_torch_memory_saver:
            # Need to bring back the memory block before we reset it.
            if self._uses_torch_memory_saver:
                tag = self.TMS_TAG
                if torch.distributed.get_rank() == 0:
                    logging.info(
                        "torch_memory_saver: resuming %s, before: %s", tag, device_memory_summary()
                    )
                torch_memory_saver.resume(tag)
                if torch.distributed.get_rank() == 0:
                    logging.info(
                        "torch_memory_saver: resumed  %s, after:  %s", tag, device_memory_summary()
                    )
            if self.kv_cache_management_mode == KVCacheManagementMode.RECOMPUTE:
                self.reset_metadata()
            return

        if self.kv_cache_management_mode == KVCacheManagementMode.OFFLOAD:
            for name, tensor in ((n, getattr(self, n)) for n in self._offloadable_tensor_names):
                tensor.storage().resize_(self._offloadable_storage_sizes[name])
                tensor.copy_(self._offloadable_cpu_backups[name], non_blocking=True)
        elif self.kv_cache_management_mode == KVCacheManagementMode.RECOMPUTE:
            self.is_tensor_state_allocated = False
            self.initialize_all_tensors()

    def deallocate_inference_state_buffers(self):
        """Deallocate large tensors (KV cache, Mamba states) during suspend.

        Called by the engine during `suspend()`. Mirror to `reinitialize_inference_state_buffers()`.
        """
        if not self.is_tensor_state_allocated:
            return
        self.is_tensor_state_allocated = False

        if self.kv_cache_management_mode == KVCacheManagementMode.PERSIST:
            return

        if self.unified_memory_level != 0:
            return

        if self._uses_torch_memory_saver:
            tag = self.TMS_TAG
            if torch.distributed.get_rank() == 0:
                logging.info(
                    "torch_memory_saver: pausing %s, before: %s", tag, device_memory_summary()
                )
            torch_memory_saver.pause(tag)
            if torch.distributed.get_rank() == 0:
                logging.info(
                    "torch_memory_saver: paused  %s, after:  %s", tag, device_memory_summary()
                )
            return

        if self.kv_cache_management_mode == KVCacheManagementMode.OFFLOAD:
            for name, tensor in ((n, getattr(self, n)) for n in self._offloadable_tensor_names):
                self._offloadable_storage_sizes[name] = tensor.storage().size()
                self._offloadable_cpu_backups[name].copy_(tensor, non_blocking=True)
                tensor.storage().resize_(0)
        elif self.kv_cache_management_mode == KVCacheManagementMode.RECOMPUTE:
            # TODO(@lmcafee): check that device == 'cuda'?
            for key in list(vars(self).keys()):
                value = getattr(self, key)
                if isinstance(value, torch.Tensor):
                    delattr(self, key)

    @classmethod
    def round_up_tokens(cls, value, tp_size=None):
        """Round up to nearest multiple of `TOKEN_ROUNDER` that is also divisible by tensor model parallel size."""
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
        """Round up to nearest multiple of `REQUEST_ROUNDER` that is also divisible by tensor model parallel size."""
        # Make sure divisible by TP size
        if tp_size is None:
            # Check if parallel state is initialized before trying to get TP size
            if parallel_state.is_initialized():
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
            else:
                tp_size = 1
        request_rounder = math.ceil(cls.REQUEST_ROUNDER / tp_size) * tp_size

        return request_rounder * int(math.ceil(int(value) / request_rounder))

    def is_static_batching(self) -> bool:
        """Is static batching? False."""
        return False

    def is_decode_only(self) -> bool:
        """
        Return whether this iteration uses the decode-only implementation.

        Reads ``padded_batch_dimensions`` — the dim was frozen by
        ``prepare_attn_init`` and tells the model which forward branch to
        take.  Reading ``num_prefill_requests`` instead is unsafe mid-step
        because ``_prepare_update_requests_metadata`` zeros it on the
        side stream before the forward, which would flip this from False
        to True after the dim has already been computed for mixed mode.

        Callers that run *inside* ``prepare_attn_init`` (i.e. before the
        new dim is set) must inline ``self.num_prefill_requests == 0``
        directly — the previous step's ``padded_batch_dimensions`` is
        stale at that point.
        """
        return self.padded_batch_dimensions.prefill_req_count == 0

    def using_cuda_graph_this_step(self) -> bool:
        """Returns True if cuda graphs are being used for this step."""
        return self._using_cuda_graph_this_step

    @property
    def is_mamba_prefix_caching_enabled(self) -> bool:
        """Whether Mamba prefix caching is configured for this context."""
        return (
            self.config.enable_prefix_caching
            and self.config.prefix_caching_mamba_gb is not None
            and self.config.prefix_caching_mamba_gb > 0
        )

    def has_unfinished_requests(self) -> bool:
        """Test if any requests remain."""
        return self.total_request_count > 0

    def cu_query_lengths(self) -> Tuple[Tensor, int]:
        """Cumulative query sequence lengths."""
        n = self.padded_active_request_count
        return self.cu_active_request_query_lengths[: n + 1], self._max_seqlen_q

    def cu_kv_lengths(self) -> Tuple[Tensor, Tensor, int]:
        """Cumulative key/value sequence lengths."""
        n = self.padded_active_request_count
        return (
            self.cu_active_sequence_lengths[: n + 1],
            self.active_sequence_lengths[:n],
            self._max_seqlen_k,
        )

    def get_active_sequence_lengths(self) -> Tensor:
        """Total sequence length (query + key) for active requests."""
        active_count = self.total_request_count - self.paused_request_count
        lengths = self.request_kv_length_offsets + self.request_query_lengths
        return lengths[:active_count]

    def get_max_sequence_lengths(self) -> Tensor:
        """Maximum sequence length for active requests."""
        active_count = self.total_request_count - self.paused_request_count
        return self.request_output_lengths[:active_count]

    def get_active_request_count(self):
        """Returns the current number of active requests."""
        return self.total_request_count - self.paused_request_count

    def active_slice(self) -> slice:
        """Slot range holding currently active requests in the canonical
        ``[active | paused | dead]`` layout."""
        return slice(0, self.total_request_count - self.paused_request_count)

    def paused_slice(self) -> slice:
        """Slot range holding currently paused requests in the canonical
        ``[active | paused | dead]`` layout."""
        return slice(
            self.total_request_count - self.paused_request_count, self.total_request_count
        )

    def build_active_slices(self):
        """Copy tensors into active buffers, and pad them with `torch.where`."""
        n = self.padded_active_request_count
        self.active_request_ids[:n].copy_(self.request_ids[:n])
        self.active_request_query_lengths[:n].copy_(self.request_query_lengths[:n])
        self.active_request_to_kv_block_ids[:n].copy_(self.request_to_kv_block_ids[:n])

        if self.is_hybrid_model:
            self.active_mamba_indices[:n].copy_(
                self.mamba_metadata.request_to_mamba_state_idx[:n]
            )

        # Request-level padding.
        arange_n = self._arange_requests[:n]
        real_req = self._real_request_count_gpu
        req_pad_mask = arange_n >= real_req

        self.active_request_to_kv_block_ids[:n] = torch.where(
            req_pad_mask.unsqueeze(1),
            self.kv_block_allocator.dummy_block_idx,
            self.active_request_to_kv_block_ids[:n],
        )

        # These query lengths are consumed by the attention kernel.
        # When we pad them, we must give the kernel correct q lengths.
        # One entry in the padded region must be an "absorber" that accounts for excess tokens.
        is_absorber = arange_n == real_req
        absorber_q_len = self.padded_active_token_count - self._real_token_count_gpu
        self.active_request_query_lengths[:n] = torch.where(
            req_pad_mask,
            torch.where(is_absorber, absorber_q_len, 0),
            self.active_request_query_lengths[:n],
        )

        # Token-level padding.
        padded_token_count = self.padded_active_token_count
        tok_pad_mask = self._arange_tokens[:padded_token_count] >= self._real_token_count_gpu
        self.token_to_block_idx[:padded_token_count] = torch.where(
            tok_pad_mask,
            self.kv_block_allocator.dummy_block_idx,
            self.token_to_block_idx[:padded_token_count],
        )
        self.token_to_local_position_within_kv_block[:padded_token_count] = torch.where(
            tok_pad_mask, 0, self.token_to_local_position_within_kv_block[:padded_token_count]
        )
        self.token_to_position_in_request[:padded_token_count] = torch.where(
            tok_pad_mask, 0, self.token_to_position_in_request[:padded_token_count]
        )

        # Cumsums and derived values must be computed after padding.
        torch.cumsum(
            self.active_request_query_lengths[:n],
            dim=0,
            out=self.cu_active_request_query_lengths[1 : n + 1],
        )
        self.active_request_last_token_idxs[:n].copy_(
            self.cu_active_request_query_lengths[1 : n + 1]
        )
        self.active_request_last_token_idxs[:n] -= 1

        # active_sequence_lengths = query + kv_offset for real requests.
        # For padding the K length matches the Q length: the absorber self-attends within dummies.
        torch.add(
            self.active_request_query_lengths[:n],
            self.request_kv_length_offsets[:n],
            out=self.active_sequence_lengths[:n],
        )
        self.active_sequence_lengths[:n] = torch.where(
            req_pad_mask,
            self.active_request_query_lengths[:n],
            self.active_sequence_lengths[:n],
        )
        torch.cumsum(
            self.active_sequence_lengths[:n],
            dim=0,
            out=self.cu_active_sequence_lengths[1 : n + 1],
        )

    def run_attn_init_graph_body(self, eager=False, cache_key=None):
        """Graphable portion of `initialize_attention_state`."""
        if not eager:
            # Set max_seqlens at capture time. On replay, this can be safely skipped.
            # Eager mode falls through to `finalize_attn_init`, which computes precise values.
            if self.padded_batch_dimensions.prefill_req_count == 0:
                self._max_seqlen_q = self.num_speculative_tokens + 1
            else:
                # Force the prefill kernel to launch for prefill graphs.
                self._max_seqlen_q = max(2, self.padded_batch_dimensions.token_count)
            self._max_seqlen_k = self.max_sequence_length
            self.padding_slice = slice(
                self.active_token_count, self.padded_active_token_count
            )

        self._context_op_metadata_gpu.copy_(self._context_op_metadata_cpu, non_blocking=True)
        self.build_active_slices()

        if self.is_hybrid_model:
            slot_alloc = self.mamba_slot_allocator
            self.mamba_metadata.update(
                self.active_mamba_indices,
                self.cu_active_request_query_lengths,
                real_decode_count_gpu=self._real_decode_count_gpu,
                real_prefill_count_gpu=self._real_prefill_count_gpu,
                arange_buf=self._arange_requests,
                padded_batch_dimensions=self.padded_batch_dimensions,
                intermediate_offsets_gpu=(
                    slot_alloc._intermediate_offsets_gpu if slot_alloc is not None else None
                ),
                intermediate_counts_gpu=(
                    slot_alloc._intermediate_counts_gpu if slot_alloc is not None else None
                ),
            )

        for hook in self._init_body_hooks:
            hook(not eager)

        return self._context_op_metadata_gpu

    def append_key_value_cache(self, layer_number: int, key: Tensor, value: Tensor) -> None:
        """Append to KV cache.

        Args:
            layer_number (int): Layer number.
            key (Tensor): Key tensor.
            value (Tensor): Value tensor.
        """
        attention_layer_number = self.layer_map[layer_number - 1]

        if triton_append_key_value_cache is not None and not self.cache_mla_latent:
            # currently does not support MLA latent cache
            return triton_append_key_value_cache(
                layer_number=attention_layer_number,
                key=key,
                value=value,
                memory_buffer=self.memory_buffer,
                padded_active_token_count=self.padded_active_token_count,
                token_to_block_idx=self.token_to_block_idx,
                token_to_local_position_within_kv_block=self.token_to_local_position_within_kv_block,
            )

        block_idx = self.token_to_block_idx[: self.padded_active_token_count]
        local_kv_seq_idx = self.token_to_local_position_within_kv_block[
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
            self.memory_buffer[attention_layer_number, block_idx, local_kv_seq_idx] = kv_concat[
                : self.padded_active_token_count
            ]
        else:
            self.memory_buffer[0, attention_layer_number, block_idx, local_kv_seq_idx] = key[
                : self.padded_active_token_count
            ]
            self.memory_buffer[1, attention_layer_number, block_idx, local_kv_seq_idx] = value[
                : self.padded_active_token_count
            ]

    def key_value_cache(self, layer_number: int) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """Read from KV cache.

        Args:
            layer_number (int): Layer number.

        Return:
            (Tuple[Tensor, Tensor, Tensor]) The key and value pointer tensors that point
            to blocks within the block-level memory buffer as well as the block table.
        """
        attention_layer_number = self.layer_map[layer_number - 1]

        block_table = self.active_request_to_kv_block_ids[: self.padded_active_request_count, :]
        if self.cache_mla_latent:
            return (self.memory_buffer[attention_layer_number], None, block_table)
        return (
            self.memory_buffer[0, attention_layer_number],
            self.memory_buffer[1, attention_layer_number],
            block_table,
        )

    def mamba_states_cache(
        self, layer_number: int, intermediate: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Returns the Mamba state tensors for the given layer."""
        assert self.is_hybrid_model, "Only hybrid models have Mamba state tensors"

        mamba_layer_number = self.layer_map[layer_number - 1]
        if intermediate:
            conv_state = self.mamba_intermediate_conv_states[mamba_layer_number]
            ssm_state = self.mamba_intermediate_ssm_states[mamba_layer_number]
        else:
            conv_state = self.mamba_conv_states[mamba_layer_number]
            ssm_state = self.mamba_ssm_states[mamba_layer_number]

        return (conv_state, ssm_state)

    # =========================================================================
    # Mamba prefix cache infrastructure
    # =========================================================================

    def _allocate_mamba_cache(self, mamba_gb: float) -> None:
        """Allocate the Mamba state cache for prefix caching.

        Args:
            mamba_gb: GPU memory budget in GB for the cache.
        """
        import math as _math

        conv_size = _math.prod(self.mamba_conv_states_shape) * self.mamba_conv_states_dtype.itemsize
        ssm_size = _math.prod(self.mamba_ssm_states_shape) * self.mamba_ssm_states_dtype.itemsize
        per_slot_bytes = self.num_mamba_layers * (conv_size + ssm_size)
        total_bytes = int(mamba_gb * 1024**3)
        max_slots = total_bytes // per_slot_bytes
        if max_slots < 1:
            logging.warning(
                "Mamba cache budget (%.3f GB) too small for even 1 slot "
                "(need %.3f GB per slot). Mamba caching disabled.",
                mamba_gb,
                per_slot_bytes / 1024**3,
            )
            return

        self.mamba_slot_allocator = MambaSlotAllocator(
            context=self,
            max_slots=max_slots,
            num_mamba_layers=self.num_mamba_layers,
            conv_states_shape=self.mamba_conv_states_shape,
            ssm_states_shape=self.mamba_ssm_states_shape,
            conv_states_dtype=self.mamba_conv_states_dtype,
            ssm_states_dtype=self.mamba_ssm_states_dtype,
        )
        self.kv_block_allocator.on_blocks_deregistered = (
            self.mamba_slot_allocator.on_kv_blocks_deregistered
        )

        logging.info(
            "Mamba prefix cache: %d slots (%.3f GB), per-slot %.1f KB",
            max_slots,
            max_slots * per_slot_bytes / 1024**3,
            per_slot_bytes / 1024,
        )

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
        # TODO: This `reset_varlen_metadata()` should not be necessary.
        if self.is_hybrid_model:
            self.mamba_metadata.reset_varlen_metadata()

    def reset_mamba_state(self) -> None:
        """Reset state used within Mamba layers."""
        if self.is_hybrid_model:
            self.mamba_metadata.reset()

    def add_dummy_requests_parallel(
        self, requests: Sequence[DynamicInferenceRequest], *, count_as_prefill: bool = True
    ) -> None:
        """Fast path to add dummy requests without allocating real KV blocks."""

        if not requests:
            return

        num_new_requests = len(requests)
        if self.total_request_count + num_new_requests > self.max_requests:
            raise RequestOverflowError(requests[-1].request_id)

        lengths: List[int] = []
        num_tokens_to_generate: List[int] = []
        request_ids: List[int] = []
        prompt_tokens: List[Tensor] = []
        metadata_cols: List[List] = [[] for _ in self.request_metadata_types]

        for req in requests:
            assert isinstance(
                req, DynamicInferenceRequest
            ), "add_dummy_requests_parallel expects DynamicInferenceRequest objects"
            assert (
                req.finished_chunk_token_count == 0
            ), "chunked requests are not supported in add_dummy_requests_parallel"
            assert req.remaining_prompt_tokens is not None, "request missing prompt tokens"
            assert req.sampling_params is not None, "request missing sampling params"
            prefill_chunk_length = req.remaining_prompt_length
            assert prefill_chunk_length > 0, "request without prompt tokens is not supported"
            lengths.append(prefill_chunk_length)
            num_tokens_to_generate.append(req.sampling_params.num_tokens_to_generate)
            request_ids.append(req.request_id)
            prompt_tokens.append(
                req.remaining_prompt_tokens.to(
                    device=self.token_to_input_ids.device, dtype=self.token_to_input_ids.dtype
                )
            )
            for i, m in enumerate(req.tracked_metadata):
                metadata_cols[i].append(m)

        total_new_tokens = sum(lengths)
        if self.active_token_count + total_new_tokens > self.max_tokens:
            raise TokenOverflowError(requests[-1].request_id)

        device = self.request_ids.device
        lengths_tensor = torch.tensor(
            lengths, dtype=self.request_query_lengths.dtype, device=device
        )
        tokens_to_generate_tensor = torch.tensor(
            num_tokens_to_generate, dtype=self.request_query_lengths.dtype, device=device
        )
        request_ids_tensor = torch.tensor(request_ids, dtype=self.request_ids.dtype, device=device)

        block_counts = torch.div(
            lengths_tensor + (self.block_size_tokens - 1),
            self.block_size_tokens,
            rounding_mode="floor",
        )

        start_request_idx = self.total_request_count
        end_request_idx = start_request_idx + num_new_requests
        request_slice = slice(start_request_idx, end_request_idx)

        self.request_ids[request_slice] = request_ids_tensor
        self.request_query_lengths[request_slice] = lengths_tensor
        self.request_in_prefill_status_tensor[request_slice] = 1
        self.request_output_lengths[request_slice] = lengths_tensor + tokens_to_generate_tensor
        self.request_kv_length_offsets[request_slice] = 0
        self.request_kv_block_counts[request_slice] = block_counts
        for i, (label, dtype) in enumerate(self.request_metadata_types):
            self.request_metadata[label][request_slice] = torch.tensor(
                metadata_cols[i], dtype=dtype, device=torch.cuda.current_device()
            )

        dummy_block_idx = self.kv_block_allocator.dummy_block_idx
        self.request_last_kv_block_id[request_slice] = dummy_block_idx
        self.request_last_kv_block_offset[request_slice] = torch.remainder(
            lengths_tensor - 1, self.block_size_tokens
        )

        kv_block_view = self.request_to_kv_block_ids[request_slice]
        kv_block_view.fill_(-1)
        block_counts_list = block_counts.tolist()
        for row, block_count in enumerate(block_counts_list):
            kv_block_view[row, :block_count] = dummy_block_idx

        token_start = self.active_token_count
        token_end = token_start + total_new_tokens
        token_slice = slice(token_start, token_end)

        concatenated_tokens = torch.cat(prompt_tokens, dim=0)
        assert concatenated_tokens.numel() == total_new_tokens
        self.token_to_input_ids[token_slice] = concatenated_tokens

        lengths_long = lengths_tensor.to(dtype=torch.long)
        request_indices = torch.arange(
            start_request_idx,
            end_request_idx,
            device=self.token_to_request_idx.device,
            dtype=self.token_to_request_idx.dtype,
        )
        token_request_indices = torch.repeat_interleave(
            request_indices.to(dtype=torch.long), lengths_long
        )
        self.token_to_request_idx[token_slice] = token_request_indices

        max_length = int(lengths_tensor.max().item())
        position_template = torch.arange(
            max_length,
            device=self.token_to_position_in_request.device,
            dtype=self.token_to_position_in_request.dtype,
        )
        expanded_positions = position_template.unsqueeze(0).expand(num_new_requests, -1)
        mask = position_template.unsqueeze(0) < lengths_long.unsqueeze(1)
        positions = expanded_positions[mask]
        assert positions.numel() == total_new_tokens
        self.token_to_position_in_request[token_slice] = positions
        self.token_to_pos_ids[token_slice] = positions
        self.token_to_local_position_within_kv_block[token_slice] = torch.remainder(
            positions, self.block_size_tokens
        )
        self.token_to_block_idx[token_slice] = dummy_block_idx

        if self.is_hybrid_model:
            for logical_idx, request_idx in enumerate(range(start_request_idx, end_request_idx)):
                mamba_idx = self.mamba_metadata.allocate_slot()
                if mamba_idx is None:
                    raise ContextOverflowError(
                        requests[logical_idx].request_id, "No Mamba slots available"
                    )
                self.mamba_conv_states[:, mamba_idx] = 0.0
                self.mamba_ssm_states[:, mamba_idx] = 0.0
                self.mamba_metadata.request_to_mamba_state_idx[request_idx] = mamba_idx

        self.active_token_count = token_end
        self.total_request_count = end_request_idx
        if count_as_prefill:
            self.num_prefill_requests += num_new_requests

    def add_dummy_requests_for_cudagraph_capture(
        self, graph_dimensions: InferenceBatchDimensions
    ) -> None:
        """
        Adds dummy requests to reflect the number of prefill and decode requests in the graph config.
        These are using during cuda graph captures.
        """
        prefill_tokens = graph_dimensions.token_count - (
            graph_dimensions.decode_req_count * (self.num_speculative_tokens + 1)
        )

        # Pre-construct shared objects (safe due to deep copy in DynamicInferenceRequest.__post_init__)
        shared_sampling_params = SamplingParams(num_tokens_to_generate=1, termination_id=-1)
        shared_decode_tokens = torch.zeros(
            self.num_speculative_tokens + 1, dtype=torch.long, device=torch.cuda.current_device()
        )

        decode_requests = [
            DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=shared_decode_tokens,
                sampling_params=shared_sampling_params,
            )
            for i in range(graph_dimensions.decode_req_count)
        ]
        self.add_dummy_requests_parallel(decode_requests, count_as_prefill=False)
        if graph_dimensions.prefill_req_count == 0:
            self.num_prefill_requests = 0
            return

        per_prefill_tokens = prefill_tokens // graph_dimensions.prefill_req_count
        rem_prefill_tokens = prefill_tokens % graph_dimensions.prefill_req_count

        # If there are remaining prefill tokens, we evenly distribute them to the prefill requests
        # starting from the first prefill request until we run out of remaining prefill tokens

        prefill_token_counts = [
            per_prefill_tokens + (1 if i < rem_prefill_tokens else 0)
            for i in range(graph_dimensions.prefill_req_count)
        ]

        assert per_prefill_tokens > 0
        # Create a single large tensor and slice from it for each prefill request
        max_prefill_tokens = per_prefill_tokens + (1 if rem_prefill_tokens > 0 else 0)
        shared_prefill_tokens = torch.zeros(
            max_prefill_tokens, dtype=torch.long, device=torch.cuda.current_device()
        )

        prefill_requests = [
            DynamicInferenceRequest(
                request_id=i + graph_dimensions.decode_req_count,
                prompt_tokens=shared_prefill_tokens[: prefill_token_counts[i]],
                sampling_params=shared_sampling_params,
            )
            for i in range(graph_dimensions.prefill_req_count)
        ]
        self.add_dummy_requests_parallel(prefill_requests)
        self.num_prefill_requests = graph_dimensions.prefill_req_count

    @property
    def num_decode_requests(self) -> int:
        """
        Returns the number of decode requests.
        """
        return self.total_request_count - self.paused_request_count - self.num_prefill_requests

    def add_dummy_requests_for_expert_parallel_step(
        self, graph_dimensions: InferenceBatchDimensions
    ) -> None:
        """Minimal context setup so an EP rank with no real requests can replay
        an already-captured cuda graph without crashing or corrupting memory.

        This is the fast alternative to add_dummy_requests_for_cudagraph_capture
        (which goes through the heavyweight add_dummy_requests_parallel path).

        We setup minimal state such that initialize_attention_state and the forward
        pass can run without error.

        Called AFTER the EP sync so graph_dimensions reflects the agreed-upon graph.
        """
        N_decode = graph_dimensions.decode_req_count
        N_prefill = graph_dimensions.prefill_req_count
        N = N_decode + N_prefill
        tokens_per_decode_request = self.num_speculative_tokens + 1
        T = graph_dimensions.token_count
        dummy_block_idx = self.kv_block_allocator.dummy_block_idx

        # 1. Request counts and token count.
        self.total_request_count = N
        self.active_token_count = T
        self.num_prefill_requests = N_prefill

        # 2. Per-request state consumed by the attention kernels.
        #    Decode requests come first, followed by prefill requests.
        self.request_query_lengths[0:N_decode].fill_(tokens_per_decode_request)
        if N_prefill > 0:
            prefill_tokens = T - N_decode * tokens_per_decode_request
            per_prefill_tokens = prefill_tokens // N_prefill
            rem_prefill_tokens = prefill_tokens % N_prefill
            self.request_query_lengths[N_decode:N].fill_(per_prefill_tokens)
            if rem_prefill_tokens > 0:
                self.request_query_lengths[N_decode : N_decode + rem_prefill_tokens] += 1

        self.request_kv_length_offsets[0:N].fill_(0)
        self.request_to_kv_block_ids[0:N, 0] = dummy_block_idx

        # 3. Token-level state consumed by the triton KV append kernel.
        self.token_to_block_idx[0:T] = dummy_block_idx
        # Compute per-request token positions: e.g. query_lengths [3,2] -> [0,1,2,0,1]
        query_lengths = self.request_query_lengths[0:N]
        starts = torch.cumsum(query_lengths, dim=0) - query_lengths
        # Per-token start offset: e.g. starts [0,3], query_lengths [3,2] -> [0,0,0,3,3]
        per_token_start = torch.repeat_interleave(starts, query_lengths)
        positions = torch.arange(T, device=query_lengths.device) - per_token_start
        self.token_to_local_position_within_kv_block[0:T] = torch.remainder(
            positions, self.block_size_tokens
        )

        if self.is_hybrid_model:
            # 4. token_to_request_idx: needed by hybrid model forward pass layers.
            self.token_to_request_idx[0:T] = torch.repeat_interleave(
                torch.arange(
                    0,
                    N,
                    device=self.token_to_request_idx.device,
                    dtype=self.token_to_request_idx.dtype,
                ),
                self.request_query_lengths[0:N],
            )

            # 5. Mamba state: allocate slots for dummy requests.
            self.mamba_metadata.request_to_mamba_state_idx[0:N] = (
                self.mamba_metadata.batch_allocate_slots(N)
            )

    def initialize_attention_state(
        self,
        *,
        construct_graph_dimensions: Optional[InferenceBatchDimensions] = None,
        is_expert_parallel_dummy_cuda_graph_step: bool = False,
    ) -> None:
        """Initialize attention state so that every layer can use it.

        Args:
            construct_graph_dimensions (Optional[InferenceBatchDimensions]):
                The graph config to use for constructing the cuda graphs.
            is_expert_parallel_dummy_cuda_graph_step (bool):
                Whether this is a dummy expert model parallel step.
        Return:
            None.
        """
        if self.prepare_attn_init(
            construct_graph_dimensions=construct_graph_dimensions,
            is_expert_parallel_dummy_cuda_graph_step=is_expert_parallel_dummy_cuda_graph_step,
        ):
            use_graph = self.using_cuda_graph_this_step()
            self.run_attn_init_graph_body(
                eager=not use_graph,
                cache_key=self.padded_batch_dimensions if use_graph else None,
            )
            self.finalize_attn_init()

    def prepare_attn_init(
        self,
        *,
        construct_graph_dimensions: Optional[InferenceBatchDimensions] = None,
        is_expert_parallel_dummy_cuda_graph_step: bool = False,
    ) -> bool:
        """Pre-graph phase of `initialize_attention_state`.

        Resolves batch/padded dimensions, sets graph-mode max_seqlen, and stages the real-count
        GPU scalars. Returns True if the body + finalize phases should still run, or False for
        the EP-dummy fast-path where no more work is needed.
        """
        self.is_creating_cuda_graphs = construct_graph_dimensions is not None
        assert not (
            self.is_creating_cuda_graphs and is_expert_parallel_dummy_cuda_graph_step
        ), "Dummy expert model parallel steps should not be creating cuda graphs."

        # If in CUDA graph creation mode, add dummy requests for CUDA graph capture.
        # EP dummy requests are added AFTER the EP sync below.
        if self.is_creating_cuda_graphs:
            self.add_dummy_requests_for_cudagraph_capture(construct_graph_dimensions)

        if is_expert_parallel_dummy_cuda_graph_step:
            # No real requests on this EP rank. Pass empty dimensions so the EP
            # all-reduce in match_graph_config picks up the real ranks' values.
            batch_dimensions = InferenceBatchDimensions(
                token_count=0, prefill_req_count=0, decode_req_count=0
            )
        else:
            batch_dimensions = InferenceBatchDimensions(
                token_count=self.active_token_count,
                prefill_req_count=self.num_prefill_requests,
                decode_req_count=self.num_decode_requests,
            )

        self.batch_dimensions = batch_dimensions

        best_graph = CUDAGraphBatchDimensionBuilder.match_graph_config(
            batch_dimensions,
            self.cuda_graph_batch_dimensions_list,
            smallest_non_decode_cuda_graph_size=self.smallest_non_decode_cuda_graph_size,
            strict=self.is_hybrid_model,
            decode_only_cuda_graphs=(not self.use_cuda_graphs_for_non_decode_steps),
            ep_group=self.expert_model_parallel_group,
            num_speculative_tokens=self.num_speculative_tokens,
        )
        self._using_cuda_graph_this_step = best_graph is not None

        if construct_graph_dimensions is not None:
            assert self._using_cuda_graph_this_step

        if is_expert_parallel_dummy_cuda_graph_step and not self.using_cuda_graph_this_step():
            # If we are here, this means that CUDAGraphBatchDimensionBuilder.match_graph_config
            # could not find a compatible cuda graph for the dummy forward step.
            # Now, we need not do the remaining setup. The controller
            # will directly call the model forward pass with a single token.
            for hook in self._init_body_hooks:
                hook(False)
            return False

        # Add dummy requests AFTER the EP sync so they match the resolved graph.
        if is_expert_parallel_dummy_cuda_graph_step:
            self.add_dummy_requests_for_expert_parallel_step(best_graph)
            batch_dimensions = InferenceBatchDimensions(
                token_count=self.active_token_count,
                prefill_req_count=self.num_prefill_requests,
                decode_req_count=self.num_decode_requests,
            )
            self.batch_dimensions = batch_dimensions

        if self.using_cuda_graph_this_step():
            self.padded_batch_dimensions = best_graph
        else:
            # Inline ``num_prefill_requests == 0`` rather than ``is_decode_only()``:
            # at this point the new ``padded_batch_dimensions`` hasn't been set
            # yet, and ``is_decode_only`` reads from it.
            if self.num_prefill_requests == 0:
                if self.num_speculative_tokens > 0:
                    padded_decode_req_count = min(
                        self.max_requests, self.round_up_requests(self.num_decode_requests)
                    )
                    padded_token_count = padded_decode_req_count * (self.num_speculative_tokens + 1)
                else:
                    padded_token_count = min(
                        self.max_tokens,
                        self.max_requests,
                        self.round_up_tokens(self.active_token_count),
                    )
                    padded_decode_req_count = padded_token_count
                padded_prefill_req_count = 0
            else:
                padded_token_count = self.round_up_tokens(self.active_token_count)
                target_padding_req_count = min(
                    self.max_requests,
                    self.round_up_requests(self.total_request_count - self.paused_request_count),
                )
                padded_decode_req_count = self.num_decode_requests
                padded_prefill_req_count = target_padding_req_count - padded_decode_req_count
            self.padded_batch_dimensions = InferenceBatchDimensions(
                token_count=padded_token_count,
                prefill_req_count=padded_prefill_req_count,
                decode_req_count=padded_decode_req_count,
            )

        self.padded_active_token_count = self.padded_batch_dimensions.token_count
        self.padded_active_request_count = self.padded_batch_dimensions.req_count

        # Stage GPU scalar values in the pinned CPU buffer.
        # The actual H2D transfer is captured inside `run_attn_init_graph_body`.
        self._context_op_metadata_cpu[0] = self.total_request_count - self.paused_request_count
        self._context_op_metadata_cpu[1] = self.active_token_count
        self._context_op_metadata_cpu[2] = batch_dimensions.decode_req_count
        self._context_op_metadata_cpu[3] = batch_dimensions.prefill_req_count
        self._context_op_metadata_cpu[4] = self.total_request_count
        self._context_op_metadata_cpu[5] = self.paused_request_count

        return True

    def finalize_attn_init(self) -> None:
        """Post-graph phase of `initialize_attention_state`.

        CPU-only bookkeeping that must happen after the graphable body,
        plus resolving deferred GPU / CPU syncs that were staged during the graphable body.
        """
        if self.moe_enable_routing_replay:
            if self.using_cuda_graph_this_step():
                self.moe_routing_metadata.enable_static_buffer_recording()
            else:
                self.moe_routing_metadata.disable_static_buffer_recording()

        # Non-graph mode: recompute max_seqlen from the just-populated buffers.
        if not self.using_cuda_graph_this_step():
            n = self.padded_active_request_count
            if n > 0:
                self._max_seqlen_q = torch.max(self.active_request_query_lengths[:n]).item()
                self._max_seqlen_k = torch.max(self.active_sequence_lengths[:n]).item()
            else:
                self._max_seqlen_q = self.num_speculative_tokens + 1
                self._max_seqlen_k = 1
            self.padding_slice = slice(
                self.active_token_count, self.padded_active_token_count
            )

    def reset_tensors(self) -> None:
        """Fill all GPU tensors with sentinel values."""

        # Reset request indexes.
        self.request_ids.fill_(-1)
        self.request_query_lengths.fill_(0)
        self.request_output_lengths.fill_(0)
        self.request_kv_length_offsets.fill_(0)
        self.request_kv_block_counts.fill_(0)
        self.request_last_kv_block_id.fill_(-1)
        self.request_last_kv_block_offset.fill_(0)
        self.request_to_kv_block_ids.fill_(-1)
        self.request_in_prefill_status_tensor.fill_(-1)

        # Reset request metadata.
        for metadata_tensor in self.request_metadata.values():
            metadata_tensor.fill_(0)

        # Reset token indexes.
        self.token_to_input_ids.fill_(0)
        self.token_to_pos_ids.fill_(0)
        self.token_to_request_idx.fill_(-1)
        self.token_to_position_in_request.fill_(0)
        self.token_to_block_idx.fill_(-1)
        self.token_to_local_position_within_kv_block.fill_(0)

    def reset_metadata(self) -> None:
        """Reset all bookkeeping state: counters, block allocator, attention/mamba state.

        This must be called after ``initialize_all_tensors()`` and after any
        suspend/resume cycle to bring the context back to a clean state.
        """

        # Reset request/token counts.
        self.total_request_count = 0
        self.active_token_count = 0
        self.lifetime_prefill_token_count = 0
        self.paused_request_count = 0
        self.batch_dimensions = InferenceBatchDimensions(
            token_count=0, prefill_req_count=0, decode_req_count=0
        )
        self.padded_batch_dimensions = InferenceBatchDimensions(
            token_count=0, prefill_req_count=0, decode_req_count=0
        )
        self.padded_active_token_count = 0
        self.padded_active_request_count = 0
        self.paused_tokens = None
        self.paused_speculative_tokens = None

        # Reset attention, mamba, and block allocator state.
        self.reset_attention_state()
        self.reset_mamba_state()
        self.kv_block_allocator.reset()
        self.request_to_kv_block_ids.fill_(-1)

        self.step_count = 0

        # Reset chunked prefill state
        self.chunked_prefill_request_id = -1
        self.num_prefill_requests = 0
        self._using_cuda_graph_this_step = False
        self.is_creating_cuda_graphs = False
        self.padded_batch_dimensions = InferenceBatchDimensions(
            token_count=0, prefill_req_count=0, decode_req_count=0
        )

    def reset(self) -> None:
        """Reset entire context.

        This method does:
        - Fill all GPU tensors with sentinel values.
        - Reset active/paused request/token counts to zero.
        - Reset available blocks to entire memory.

        This method is useful after cuda graph warmup iterations, where the
        context's memory buffer is referenced by the cuda graph system and
        cannot be deallocated.
        """
        self.reset_tensors()
        self.reset_metadata()

        # Reset Mamba cache state
        if self.mamba_slot_allocator is not None:
            self.mamba_slot_allocator.reset()

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
        assert num_tokens >= self.padded_batch_dimensions.decode_req_count * (
            self.num_speculative_tokens + 1
        )
        return (
            self.token_to_input_ids[:num_tokens].unsqueeze(0),
            self.token_to_pos_ids[:num_tokens].unsqueeze(0),
        )

    def speculative_required_logit_indices(self, device: torch.device) -> Tensor:
        """Token-level indices needed for speculative decode verification.

        Returns all decode token positions (base + speculative) concatenated
        with the last token position of each prefill request.

        Args:
            device (torch.device): Device on which to create the index tensor.

        Return:
            (Tensor) 1-D indices into the packed token sequence, length
            ``num_decode_requests * (num_speculative_tokens + 1) + num_prefill_requests``.
        """
        active_count = self.total_request_count - self.paused_request_count
        query_lengths = self.request_query_lengths[:active_count]
        num_decode = self.num_decode_requests

        decode_token_count = num_decode * (self.num_speculative_tokens + 1)
        decode_indices = torch.arange(decode_token_count, device=device)

        cumsum = torch.cumsum(query_lengths, dim=0)
        prefill_last_indices = cumsum[num_decode:].sub(1)

        return torch.cat([decode_indices, prefill_last_indices])

    @property
    def num_last_token_logits(self) -> int:
        """Number of rows produced by `last_token_logits` for the current step.

        Single source of truth for the bound: one row per request, with
        `(num_speculative_tokens + 1)` rows per decode request when MTP is active.
        """
        if self.num_speculative_tokens > 0:
            return (
                self.num_decode_requests * (self.num_speculative_tokens + 1)
                + self.num_prefill_requests
            )
        return self.total_request_count - self.paused_request_count

    def last_token_logits(self, logits: Tensor) -> Tensor:
        """Select the logit positions needed for token generation.

        When speculative decoding is active, decode requests need logits for all
        their tokens (base + speculative) for verification, while prefill requests
        only need the last token logit. This avoids materializing the full
        vocab-sized logits for every prefill token, which causes large memory
        spikes during prefill-heavy batches.

        Args:
            logits (Tensor): Output logits of forward pass, shape [1, S, H].

        Return:
            (Tensor) Selected logits, shape [N, H], where N == num_last_token_logits.
        """
        # todo: @lmcafee, remove these asserts?
        assert logits.size(0) == 1, f"logits.size(0) ({tuple(logits.shape)}) != 1"
        assert logits.size(1) == self.padded_active_token_count, (
            f"logits.size(1) ({tuple(logits.shape)}) != "
            f"padded_active_token_count ({self.padded_active_token_count})."
        )
        logits_2d = logits.squeeze(0)

        if self.num_speculative_tokens > 0:
            selected = self.speculative_required_logit_indices(logits.device)
            assert selected.numel() == self.num_last_token_logits
            return logits_2d[selected, :]

        active_request_count = self.total_request_count - self.paused_request_count
        last_token_idxs = self.active_request_last_token_idxs[:active_request_count]
        assert last_token_idxs.numel() == self.num_last_token_logits
        return logits_2d[last_token_idxs, :]

    def _compute_prefix_match(
        self, req: DynamicInferenceRequest, prefill_chunk_length: int
    ) -> Tuple[list, int, int, int, int, int]:
        """Compute prefix match results and skip counts for a request chunk.

        Shared by check_availability (budget checks) and add_request (execution).

        Returns:
            Tuple of (matched_block_ids, num_blocks_from_pool,
                      already_allocated_blocks, overall_required_blocks,
                      prefix_skip_tokens, effective_prefill_chunk_length).
        """
        finished = req.finished_chunk_token_count
        already_allocated_blocks = (finished + self.block_size_tokens - 1) // self.block_size_tokens
        overall_required_blocks = (
            finished + prefill_chunk_length + self.block_size_tokens - 1
        ) // self.block_size_tokens

        # Fast path: skip all prefix matching when disabled.
        if not self.enable_prefix_caching:
            num_blocks_from_pool = max(0, overall_required_blocks - already_allocated_blocks)
            return (
                [],
                num_blocks_from_pool,
                already_allocated_blocks,
                overall_required_blocks,
                0,
                prefill_chunk_length,
            )

        matched_block_ids, _ = self._find_kv_match_count(
            req, already_allocated_blocks, overall_required_blocks
        )
        num_matched = len(matched_block_ids)

        block_aligned = finished % self.block_size_tokens == 0
        if num_matched > 0 and block_aligned:
            prefix_skip_tokens = min(num_matched * self.block_size_tokens, prefill_chunk_length - 1)
        else:
            prefix_skip_tokens = 0

        # Hybrid models with Mamba caching: skip based on Mamba match count.
        # Only applies to the first chunk (finished == 0); continuation chunks
        # already had Mamba state restored during the first chunk.
        if self.is_hybrid_model and self.mamba_slot_allocator is not None and finished == 0:
            num_mamba_matched = getattr(req, '_mamba_num_matched_blocks', 0)
            assert (
                num_mamba_matched <= num_matched
            ), f"Mamba match ({num_mamba_matched}) > KV match ({num_matched})"
            if num_mamba_matched > 0 and block_aligned:
                raw_skip = num_mamba_matched * self.block_size_tokens
                if raw_skip >= prefill_chunk_length:
                    # Back off to previous block with cached Mamba state
                    mamba_map = self.mamba_slot_allocator.hash_to_block_id
                    backed_off_blocks = 0
                    for j in range(num_mamba_matched - 2, -1, -1):
                        if req.precomputed_block_hashes[j] in mamba_map:
                            backed_off_blocks = j + 1
                            break
                    prefix_skip_tokens = backed_off_blocks * self.block_size_tokens
                else:
                    prefix_skip_tokens = raw_skip
            else:
                prefix_skip_tokens = 0
        elif self.is_hybrid_model and finished == 0:
            prefix_skip_tokens = 0

        # Clamp so that effective_prefill_chunk_length >= 2 when possible.
        # A single-token prefill chunk (effective == 1) causes max_seqlen_q == 1,
        # which routes the batch into the flash-attention decode kernel and crashes.
        # Round down to a block boundary to keep block-table indexing consistent.
        if prefill_chunk_length - prefix_skip_tokens < 2 and prefill_chunk_length >= 2:
            max_skip = prefill_chunk_length - 2
            prefix_skip_tokens = (max_skip // self.block_size_tokens) * self.block_size_tokens

        effective_prefill_chunk_length = prefill_chunk_length - prefix_skip_tokens
        num_blocks_from_pool = max(
            0, overall_required_blocks - already_allocated_blocks - num_matched
        )

        return (
            matched_block_ids,
            num_blocks_from_pool,
            already_allocated_blocks,
            overall_required_blocks,
            prefix_skip_tokens,
            effective_prefill_chunk_length,
        )

    def check_availability(self, req: DynamicInferenceRequest) -> Tuple[bool, bool, bool]:
        """
        Check if the request can be added to the context.
        """
        # Note that for hybrid models checking the total request count is sufficient
        # because we allocate a single set of Mamba state tensors for each request
        request_can_be_added = (
            self.total_request_count < self.max_requests and self.paused_request_count == 0
        )

        (_, num_blocks_from_pool, _, _, _, effective_prefill_chunk_length) = (
            self._compute_prefix_match(req, req.remaining_prompt_length)
        )

        request_tokens_can_be_added = (
            self.active_token_count + effective_prefill_chunk_length <= self.max_tokens
        )
        kv_cache_available = self.kv_block_allocator.is_memory_available(num_blocks_from_pool)
        return request_can_be_added, request_tokens_can_be_added, kv_cache_available

    def _find_kv_match_count(
        self, req: DynamicInferenceRequest, start_block: int, end_block: int
    ) -> tuple[list[int], int]:
        """Find cached blocks matching a range of the prompt using precomputed hashes.

        Looks up hashes in req.precomputed_block_hashes[start_block:end_block] against
        the block allocator's hash-to-block mapping. Stops at the first non-match.

        Args:
            req: The inference request with precomputed_block_hashes set.
            start_block: First block index to match (inclusive).
            end_block: Last block index to match (exclusive); clamped to hash count.

        Returns:
            Tuple of:
            - List of matched block IDs (consecutive from start_block)
            - Parent hash of the last matched block (0 if no matches)
        """
        # Early return if prefix caching is disabled
        if not self.enable_prefix_caching:
            return [], 0

        # Early return if request has no precomputed hashes
        if not req.precomputed_block_hashes:
            return [], 0

        # Clamp end_block to the number of precomputed hashes (the trailing
        # partial block has no hash).
        end_block = min(end_block, len(req.precomputed_block_hashes))
        if start_block >= end_block:
            return [], 0

        hashes = req.precomputed_block_hashes[start_block:end_block]
        kv_hash_to_block = self.kv_block_allocator.kv_hash_to_block_id

        # Find longest KV prefix by iterating block hashes from end.
        # Parent-chained hashes guarantee: if hash at position N exists,
        # all hashes 0..N also exist. So first match from end = longest prefix.
        for i in range(len(hashes) - 1, -1, -1):
            if hashes[i] in kv_hash_to_block:
                num_matched = i + 1
                matched_blocks = [kv_hash_to_block[hashes[j]] for j in range(num_matched)]
                parent_hash = hashes[num_matched - 1]
                return matched_blocks, parent_hash

        return [], 0

    def add_request(
        self, req: DynamicInferenceRequest, prefill_chunk_length: Optional[int] = None
    ) -> None:
        """Add request to context. At this stage, we assume that the request is valid and can be added, as the checks are done in the schedule function.

        Args:
            req (DynamicInferenceRequest): Request to add.
            prefill_chunk_length (Optional[int]): Length of prefill chunk to add. If None, the request will be fully added.

        Return:
            None
        """
        # If tensor state is deallocated, do not add request.
        if not self.is_tensor_state_allocated:
            raise TensorStateDeallocatedError(req.request_id)

        # Prefill chunk length.
        if prefill_chunk_length is None:
            prefill_chunk_length = req.remaining_prompt_length

        assert prefill_chunk_length > 0, "Chunk length is 0"
        assert (
            prefill_chunk_length <= req.remaining_prompt_length
        ), "Prefill chunk length is greater than remaining prompt length"

        # =========================================================================
        # Block allocation + prefix matching + prefill skipping
        # =========================================================================
        (
            matched_block_ids,
            num_blocks_from_pool,
            already_allocated_blocks,
            overall_required_blocks,
            prefix_skip_tokens,
            effective_prefill_chunk_length,
        ) = self._compute_prefix_match(req, prefill_chunk_length)
        num_matched_blocks = len(matched_block_ids)
        effective_kv_offset = req.finished_chunk_token_count + prefix_skip_tokens

        # Track prefix cache hits.
        if num_matched_blocks > 0:
            self.prefix_cache_hits += 1
            self.prefix_cache_blocks_matched += num_matched_blocks

        # Slice tokens to skip matched prefix
        this_round_tokens = req.remaining_prompt_tokens[prefix_skip_tokens:prefill_chunk_length]

        new_block_ids = None
        if num_blocks_from_pool > 0:
            new_block_ids = self.kv_block_allocator.allocate_memory_blocks(num_blocks_from_pool)
            if new_block_ids is None or len(new_block_ids) != num_blocks_from_pool:
                raise BlockOverflowError(req.request_id)

        # Increment ref counts and update timestamps for matched (shared) blocks
        if num_matched_blocks > 0:
            matched_tensor = torch.tensor(
                matched_block_ids, dtype=torch.int32, device=torch.cuda.current_device()
            )
            self.kv_block_allocator.block_ref_counts[matched_tensor] += 1
            if self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.LRU:
                self.kv_block_allocator.update_timestamps(matched_tensor)

        if self.total_request_count >= self.max_requests:
            raise RequestOverflowError(req.request_id)

        if self.active_token_count + effective_prefill_chunk_length > self.max_tokens:
            raise TokenOverflowError(req.request_id)

        # In the [active | paused | dead] layout, new requests must land at right edge of active.
        # When no requests are paused, that landing point is `total_request_count`.
        # So writing naturally extends any hidden chunked-prefill record.
        # When paused requests exist, inserting at total would land past the paused region.
        # Instead we right-rotate, moving the chunked prefill record by one slot.
        active_count = self.total_request_count - self.paused_request_count
        if self.paused_request_count > 0:
            device = self.request_ids.device
            rotate_dst = torch.arange(active_count, self.total_request_count + 1, device=device)
            rotate_src = torch.roll(rotate_dst, shifts=1, dims=0)
            self._move_book_keeping_tensors(src_idxs=rotate_src, dst_idxs=rotate_dst)

        current_id = active_count

        self.request_ids[current_id] = req.request_id

        # Handle request metadata.
        assert (
            req.get_metadata_types() == self.request_metadata_types
        ), "Request added to context with invalid metadata types"
        metadata = req.tracked_metadata
        metadata_types = req.get_metadata_types()
        for m, m_type in zip(metadata, metadata_types):
            label, _ = m_type
            if not isinstance(m, torch.Tensor):
                m = torch.as_tensor(
                    m,
                    device=self.request_metadata[label].device,
                    dtype=self.request_metadata[label].dtype,
                )

            self.request_metadata[label][current_id] = m

        # Handle length and block assignments.
        self.request_query_lengths[current_id] = effective_prefill_chunk_length
        self.request_in_prefill_status_tensor[current_id] = 1
        self.request_output_lengths[current_id] = (
            req.finished_chunk_token_count
            + prefill_chunk_length
            + req.sampling_params.num_tokens_to_generate
        )

        # Assign blocks: matched blocks at [already_allocated, already_allocated + num_matched),
        # then newly allocated blocks after that.
        match_start = already_allocated_blocks
        new_block_start = already_allocated_blocks + num_matched_blocks
        if num_matched_blocks > 0:
            self.request_to_kv_block_ids[current_id][
                match_start : match_start + num_matched_blocks
            ] = matched_tensor
        if new_block_ids is not None:
            self.request_to_kv_block_ids[current_id][
                new_block_start : new_block_start + len(new_block_ids)
            ] = new_block_ids

        self.request_kv_length_offsets[current_id] = effective_kv_offset
        self.request_kv_block_counts[current_id] = overall_required_blocks
        self.request_last_kv_block_id[current_id] = self.request_to_kv_block_ids[current_id][
            overall_required_blocks - 1
        ]
        self.request_last_kv_block_offset[current_id] = (
            prefill_chunk_length + req.finished_chunk_token_count - 1
        ) % self.block_size_tokens

        token_offset_range = torch.arange(
            effective_kv_offset,
            effective_kv_offset + effective_prefill_chunk_length,
            device=self.token_to_pos_ids.device,
        )
        self.token_to_pos_ids[
            self.active_token_count : self.active_token_count + effective_prefill_chunk_length
        ] = token_offset_range
        self.token_to_input_ids[
            self.active_token_count : self.active_token_count + effective_prefill_chunk_length
        ] = this_round_tokens
        self.token_to_request_idx[
            self.active_token_count : self.active_token_count + effective_prefill_chunk_length
        ] = current_id
        self.token_to_position_in_request[
            self.active_token_count : self.active_token_count + effective_prefill_chunk_length
        ] = token_offset_range
        self.token_to_block_idx[
            self.active_token_count : self.active_token_count + effective_prefill_chunk_length
        ] = self.request_to_kv_block_ids[current_id][token_offset_range // self.block_size_tokens]
        self.token_to_local_position_within_kv_block[
            self.active_token_count : self.active_token_count + effective_prefill_chunk_length
        ] = (token_offset_range % self.block_size_tokens)

        # Register hashes for completely filled blocks (skip matched blocks).
        # Two disjoint ranges may need registration:
        #   Range 1: [previously_complete, min(already_allocated_blocks, num_complete_blocks))
        #       — the partial block from a prior chunk that this chunk's tokens completed
        #   Range 2: [already_allocated_blocks + num_matched_blocks, num_complete_blocks)
        #       — newly allocated blocks that are now complete
        if self.enable_prefix_caching and req.precomputed_block_hashes:
            total_tokens_after = req.finished_chunk_token_count + prefill_chunk_length
            num_complete_blocks = total_tokens_after // self.block_size_tokens
            previously_complete = req.finished_chunk_token_count // self.block_size_tokens

            def _register_range(start: int, end: int):
                if start >= end:
                    return
                block_ids_to_hash = self.request_to_kv_block_ids[current_id][start:end].tolist()
                block_hashes_slice = req.precomputed_block_hashes[start:end]
                self.kv_block_allocator.register_kv_block_hashes(
                    block_ids_to_hash, block_hashes_slice
                )

            # Range 1: prior-chunk partial block that this chunk just completed
            _register_range(previously_complete, min(already_allocated_blocks, num_complete_blocks))
            # Range 2: newly allocated (non-matched) blocks that are now complete
            _register_range(already_allocated_blocks + num_matched_blocks, num_complete_blocks)

        if self.is_hybrid_model and req.finished_chunk_token_count == 0:
            # Allocate a slot for Mamba states
            mamba_idx = self.mamba_metadata.allocate_slot()
            if mamba_idx is None:
                raise ContextOverflowError(req.request_id, "No Mamba slots available")
            self.mamba_metadata.request_to_mamba_state_idx[self.total_request_count] = mamba_idx

            # Restore Mamba state from the block corresponding to prefix_skip_tokens
            restore_block_count = prefix_skip_tokens // self.block_size_tokens
            restored = False
            if restore_block_count > 0 and self.mamba_slot_allocator is not None:
                restore_block_id = matched_block_ids[restore_block_count - 1]
                restored = self.mamba_slot_allocator.restore_to_live(
                    self.total_request_count, restore_block_id
                )
            if not restored:
                self.mamba_conv_states[:, mamba_idx] = 0.0
                self.mamba_ssm_states[:, mamba_idx] = 0.0

            # Compute intermediate offsets for state extraction during forward pass
            if self.mamba_slot_allocator is not None:
                self.mamba_slot_allocator.compute_and_store_offsets(
                    req,
                    current_id,
                    prefix_skip_tokens,
                    prefill_chunk_length,
                    num_matched_blocks,
                    matched_block_ids,
                    overall_required_blocks,
                )

        self.active_token_count += effective_prefill_chunk_length
        self.lifetime_prefill_token_count += effective_prefill_chunk_length
        self.total_request_count += 1
        self.num_prefill_requests += 1

    def _permute_book_keeping_tensors(
        self, perm: Tensor, next_tokens: Tensor, target: slice = slice(None)
    ) -> None:
        """
        Permute bookkeeping tensors according to the provided indices.

        Args:
            perm (Tensor): A 1-D long tensor of indices indicating the new order of requests.
            next_tokens (Tensor): An extra tensor to permute
            target (slice, optional): A slice object specifying the range of requests to permute.
                Defaults to slice(None), which means all requests.
        """
        p = perm
        s = target
        self.request_kv_length_offsets[s] = self.request_kv_length_offsets[p]
        self.request_query_lengths[s] = self.request_query_lengths[p]
        self.request_output_lengths[s] = self.request_output_lengths[p]
        self.request_ids[s] = self.request_ids[p]
        self.request_in_prefill_status_tensor[s] = self.request_in_prefill_status_tensor[p]
        if next_tokens is not None:
            next_tokens[s] = next_tokens[p]

        self.request_to_kv_block_ids[s] = self.request_to_kv_block_ids[p]
        self.request_kv_block_counts[s] = self.request_kv_block_counts[p]
        self.request_last_kv_block_id[s] = self.request_last_kv_block_id[p]
        self.request_last_kv_block_offset[s] = self.request_last_kv_block_offset[p]

        for metadata_tensor in self.request_metadata.values():
            metadata_tensor[s] = metadata_tensor[p]

        if self.is_hybrid_model:
            self.mamba_metadata.request_to_mamba_state_idx[s] = (
                self.mamba_metadata.request_to_mamba_state_idx[p]
            )

        if self._ur.spec_tokens is not None:
            self._ur.spec_tokens[:, s] = self._ur.spec_tokens[:, p]

        # Keep the prev-block-ID snapshot aligned with request_last_kv_block_id.
        # Eviction and chunked positioning permute slots after the snapshot is
        # taken; without this, token bookkeeping under speculative decoding
        # would read pre-permute prev IDs at post-permute slot positions.
        if self.num_speculative_tokens > 0 and getattr(self, "_ur", None) is not None:
            self._ur.prev_last_block_ids[s] = self._ur.prev_last_block_ids[p]

    def _move_book_keeping_tensors(
        self, src_idxs, dst_idxs, next_tokens=None, new_speculative_tokens=None
    ):
        """One-way move (``tensor[dst] = tensor[src]``) for every bookkeeping tensor.

        Used by the eager ``add_request`` rotation path to make room for a
        chunked-prefill continuation when paused requests exist. ``next_tokens`` /
        ``new_speculative_tokens`` are optional because the rotation runs
        outside ``update_requests`` and doesn't touch the sampled-token buffers.
        """
        self.request_kv_length_offsets[dst_idxs] = self.request_kv_length_offsets[src_idxs]
        self.request_in_prefill_status_tensor[dst_idxs] = self.request_in_prefill_status_tensor[
            src_idxs
        ]
        self.request_query_lengths[dst_idxs] = self.request_query_lengths[src_idxs]
        self.request_output_lengths[dst_idxs] = self.request_output_lengths[src_idxs]
        self.request_ids[dst_idxs] = self.request_ids[src_idxs]
        self.request_to_kv_block_ids[dst_idxs] = self.request_to_kv_block_ids[src_idxs]
        self.request_kv_block_counts[dst_idxs] = self.request_kv_block_counts[src_idxs]
        self.request_last_kv_block_id[dst_idxs] = self.request_last_kv_block_id[src_idxs]
        self.request_last_kv_block_offset[dst_idxs] = self.request_last_kv_block_offset[src_idxs]

        if next_tokens is not None:
            next_tokens[dst_idxs] = next_tokens[src_idxs]
        if new_speculative_tokens is not None:
            new_speculative_tokens[:, dst_idxs] = new_speculative_tokens[:, src_idxs]

        for metadata_tensor in self.request_metadata.values():
            metadata_tensor[dst_idxs] = metadata_tensor[src_idxs]

        if self.is_hybrid_model:
            self.mamba_metadata.request_to_mamba_state_idx[dst_idxs] = (
                self.mamba_metadata.request_to_mamba_state_idx[src_idxs]
            )

    def _swap_book_keeping_tensors(
        self, src_idxs, dst_idxs, next_tokens=None, new_speculative_tokens=None
    ):
        """
        Swaps all the relevent booking tensors with src idxs to dst idxs
        """
        tensor_swap(self.request_kv_length_offsets, src_idxs, dst_idxs)
        tensor_swap(self.request_query_lengths, src_idxs, dst_idxs)
        tensor_swap(self.request_in_prefill_status_tensor, src_idxs, dst_idxs)
        tensor_swap(self.request_output_lengths, src_idxs, dst_idxs)
        tensor_swap(self.request_ids, src_idxs, dst_idxs)
        tensor_swap(self.request_to_kv_block_ids, src_idxs, dst_idxs)
        tensor_swap(self.request_kv_block_counts, src_idxs, dst_idxs)
        tensor_swap(self.request_last_kv_block_id, src_idxs, dst_idxs)
        tensor_swap(self.request_last_kv_block_offset, src_idxs, dst_idxs)

        if next_tokens is not None:
            tensor_swap(next_tokens, src_idxs, dst_idxs)

        if new_speculative_tokens is not None:
            # new_speculative_tokens has request dimension as second dimension,
            # so swap on transposed view
            tensor_swap(new_speculative_tokens.t(), src_idxs, dst_idxs)

        for metadata_tensor in self.request_metadata.values():
            tensor_swap(metadata_tensor, src_idxs, dst_idxs)

        if self.is_hybrid_model:
            tensor_swap(self.mamba_metadata.request_to_mamba_state_idx, src_idxs, dst_idxs)

        # See _permute_book_keeping_tensors for rationale.
        if self.num_speculative_tokens > 0 and getattr(self, "_ur", None) is not None:
            tensor_swap(self._ur.prev_last_block_ids, src_idxs, dst_idxs)

    def get_index_of_chunked_prefill_request(self, safe: bool = True) -> int:
        """Get the slot index of the chunked prefill request.

        If ``safe`` is True, only ``request_ids[:total_request_count]`` is
        searched; a hidden chunked request (past the boundary) returns -1.
        Otherwise the search covers the full ``[0, max_requests)`` range.

        Uses a single ``.item()`` CPU sync via the any+argmax shape-stable
        pattern (no variable-length intermediates).

        Returns:
            (int) Slot index of the chunked prefill request, or -1 if none
            exists in the searched range.
        """
        if self.chunked_prefill_request_id == -1:
            return -1

        request_ids = self.request_ids
        if safe:
            request_ids = request_ids[: self.total_request_count]

        mask = request_ids == self.chunked_prefill_request_id
        any_match = mask.any()
        # argmax on a bool/int tensor returns the index of the first True
        # (or 0 if all False); disambiguate via where against a -1 sentinel.
        first_match = mask.to(torch.int32).argmax()
        idx_gpu = torch.where(any_match, first_match, torch.full_like(first_match, -1))
        return int(idx_gpu.item())

    def is_chunked_prefill_enabled(self) -> bool:
        """Returns whether chunked prefill is enabled."""
        return self.enable_chunked_prefill

    def release_memory_blocks_from_request_indexes(self, request_indexes) -> None:
        """Release KV blocks and mamba slots for the given request indexes.

        Dispatches to the shape-stable :meth:`_release_blocks_from_mask_gpu`.
        Scratch tensors are allocated in ``__init__`` so this is always safe.

        Args:
            request_indexes (torch.Tensor): Request indexes. (*Note*, NOT
                request ids.)
        """
        device = request_indexes.device
        released_mask = torch.zeros(self.max_requests, dtype=torch.bool, device=device)
        released_mask[request_indexes] = True
        self._release_blocks_from_mask_gpu(released_mask)
        # Eager release: we just enqueued possible dereg events GPU-side, so
        # arm the host flag that drain_pending_dereg uses to gate its sync,
        # then drain in the same call to keep the host dict consistent for
        # subsequent add_request calls.
        self.kv_block_allocator._pc_dereg_may_have_events = True
        self.kv_block_allocator.drain_pending_dereg()

    def _release_blocks_from_mask_gpu(self, released_mask: Tensor) -> None:
        """Shape-stable release driven by a full-size bool mask.

        The body is graph-safe: every intermediate has a shape that depends
        only on ``max_requests`` and ``max_kv_block_count``, and the stack
        pointer advances via GPU-scalar ops. Safe to call from inside a
        captured graph body (``update_requests`` Phase 2) or from host
        code (the ``release_memory_blocks_from_request_indexes`` wrapper).

        Args:
            released_mask: ``(max_requests,)`` bool tensor. True at each
                slot whose KV blocks and mamba slot should be released.
        """
        max_req = self.max_requests
        max_blk = self.max_kv_block_count
        max_release = max_req * max_blk

        # ── KV blocks: scatter-pack into the release buffer ──
        row_mask = released_mask.unsqueeze(1).expand(max_req, max_blk)
        kv_gathered = torch.where(
            row_mask,
            self.request_to_kv_block_ids,
            torch.full_like(self.request_to_kv_block_ids, -1),
        )
        kv_flat = kv_gathered.reshape(-1)
        num_valid_gpu = scatter_pack_valid(
            kv_flat,
            kv_flat != -1,
            self._ur.release_pack_buf,
            fill_value=self.kv_block_allocator.dummy_block_idx,
            sink_idx_value=max_release,
        )
        if (
            self.enable_prefix_caching
            and self.prefix_caching_eviction_policy == PrefixCachingEvictionPolicy.REF_ZERO
        ):
            # Prefix-aware release: per-block ref count decrement,
            # deregistration, hash clearing, and free-pool push. Host-side
            # dict cleanup is deferred to ``drain_pending_dereg`` called
            # after this graph body syncs.
            self.kv_block_allocator.release_memory_blocks_prefix_aware_gpu(
                self._ur.release_pack_buf[:max_release], num_valid_gpu
            )
        else:
            self.kv_block_allocator.release_memory_blocks_gpu(
                self._ur.release_pack_buf[:max_release], num_valid_gpu
            )
        # Invalidate released rows in the block ID matrix.
        self.request_to_kv_block_ids.masked_fill_(row_mask, -1)

        # ── Mamba slots: same pattern over a 1-D tensor ──
        if self.is_hybrid_model:
            mamba_idx_tensor = self.mamba_metadata.request_to_mamba_state_idx
            mamba_gathered = torch.where(
                released_mask, mamba_idx_tensor, torch.full_like(mamba_idx_tensor, -1)
            )
            m_num_valid_gpu = scatter_pack_valid(
                mamba_gathered,
                mamba_gathered != -1,
                self._ur.mamba_release_pack_buf,
                fill_value=0,
                sink_idx_value=max_req,
            )
            self.mamba_metadata.free_slots_gpu(
                self._ur.mamba_release_pack_buf[:max_req], m_num_valid_gpu
            )
            # Invalidate the released slot mappings.
            self.mamba_metadata.request_to_mamba_state_idx.masked_fill_(released_mask, -1)

        # ── Mamba slot allocator intermediate state ──
        # Mask-based resets so the op shapes never depend on the number of
        # released requests. 1-D tensors use the base mask; 2-D tensors
        # broadcast it over the second dim.
        if self.mamba_slot_allocator is not None:
            sa = self.mamba_slot_allocator
            sa._intermediate_counts_gpu.masked_fill_(released_mask, 0)
            sa._eos_cache_block_id_gpu.masked_fill_(released_mask, -1)
            row_mask_k = released_mask.unsqueeze(1).expand_as(sa._intermediate_offsets_gpu)
            sa._intermediate_offsets_gpu.masked_fill_(row_mask_k, 0)
            sa._intermediate_block_ids_gpu.masked_fill_(row_mask_k, -1)

    def init_update_requests_state(self) -> None:
        """Allocate the static GPU scratch tensors used by ``update_requests``.

        Called from ``__init__`` after ``initialize_all_tensors()``.  Pre-sizing
        ``block_bag`` and ``mamba_state_free_slots`` in their constructors
        eliminates the reallocations that used to disturb the CUDA caching
        allocator between graph captures, making this safe to run before
        engine warmup.
        """
        self._ur = UpdateRequestsScratch(
            max_requests=self.max_requests,
            max_tokens=self.max_tokens,
            num_speculative_tokens=self.num_speculative_tokens,
            max_release_per_step=self.kv_block_allocator.max_release_per_step,
            kv_dummy_block_idx=self.kv_block_allocator.dummy_block_idx,
            sync_size=self._SYNC_SIZE,
            is_hybrid_model=self.is_hybrid_model,
        )

    def _compute_resume_budget(
        self, is_active_now: Tensor, is_paused_now: Tensor, block_size_threshold: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the resume budget and per-slot block costs.

        Returns:
            active_avail_gpu (0-d int64): blocks available within the active
                budget = ``active_count - sum(active_blocks)``.
            needs_new_block_full ((max_req,) bool): per-slot flag for "the
                last KV block is full; resuming would cost +1 block".
            paused_blocks_i64 ((max_req,) int64): per-slot block cost for
                resuming (block_count + needs_new), masked to the paused
                region; zero elsewhere.
        """
        active_blocks = torch.where(
            is_active_now,
            self.request_kv_block_counts,
            torch.zeros_like(self.request_kv_block_counts),
        )
        active_used_gpu = active_blocks.to(torch.int64).sum()
        active_avail_gpu = self.kv_block_allocator.active_count - active_used_gpu

        needs_new_block_full = self.request_last_kv_block_offset >= block_size_threshold
        needs_new_block_cnt = needs_new_block_full.to(self.request_kv_block_counts.dtype)
        paused_blocks_full = torch.where(
            is_paused_now,
            self.request_kv_block_counts + needs_new_block_cnt,
            torch.zeros_like(self.request_kv_block_counts),
        )
        return active_avail_gpu, needs_new_block_full, paused_blocks_full.to(torch.int64)

    def _compute_fits(
        self,
        paused_blocks_i64: Tensor,
        is_paused_now: Tensor,
        active_avail_gpu: Tensor,
        active_cur: Tensor,
        total_cur: Tensor,
    ) -> Tensor:
        """Compute the resume count: how many leftmost paused requests fit.

        In ``[active | paused]`` layout, a forward cumsum over
        ``paused_blocks_i64`` accumulates from the leftmost paused. The fit
        count is paused_count minus the number of paused slots whose
        cumsum exceeds the active budget. The post-permute layout puts
        stayed-paused (bucket2) before newly-paused (bucket3), so a
        leftward sweep happens to resume stayed-paused first; the resume
        order is whichever direction makes the cumsum cheapest, not a
        FIFO/LIFO contract.

        Then clamps by the allocator pool (``total_avail_gpu`` plus LRU-
        evictable cached blocks under LRU prefix caching) and by the per-step
        ``max_allowed_active - active_cur`` slot budget.

        Returns the 0-d int64 resume count, clamped to non-negative.
        """
        forward_cumsum = paused_blocks_i64.cumsum(dim=0)
        # Only paused slots contribute: active slots have cumsum 0 which is
        # always <= budget, so we restrict the over-budget tally to the
        # paused region.
        over_budget_paused = is_paused_now & (forward_cumsum > active_avail_gpu)
        i_star = over_budget_paused.to(torch.int64).sum()
        paused_count = total_cur.view(()) - active_cur.view(())
        fits_gpu = paused_count - i_star

        total_pool = self.kv_block_allocator.total_avail_gpu.view(()).to(torch.int64)
        resume_by_pool = torch.minimum(fits_gpu, total_pool)
        room = torch.clamp(self._ur.max_allowed_active - active_cur.view(()), min=0)
        resume_count_gpu = torch.minimum(resume_by_pool, room)
        # Defensive: can't happen unless state is corrupted.
        return torch.clamp(resume_count_gpu, min=0)

    def _allocate_for_resume(
        self,
        slot_idx: Tensor,
        active_cur: Tensor,
        resume_count_gpu: Tensor,
        needs_new_block_full: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Allocate new blocks for the "needs_new" slots in the resumed range.

        The resumed range = ``[active_cur, active_cur + resume_count)`` —
        the leftmost paused slots, which become active.

        Returns:
            needs_new_in_resumed ((max_req,) bool): per-slot flag for
                "this slot was resumed and needs a new block now".
            allocated_per_slot ((max_req,) int32): per-slot block ID to
                write; valid only where ``needs_new_in_resumed``.
        """
        resumed_range_start = active_cur.view(())
        resumed_range_end = resumed_range_start + resume_count_gpu
        resumed_mask = (slot_idx >= resumed_range_start) & (slot_idx < resumed_range_end)
        needs_new_in_resumed = resumed_mask & needs_new_block_full
        needs_new_i64_mask = needs_new_in_resumed.to(torch.int64)
        num_new_blocks_gpu = needs_new_i64_mask.sum()

        allocated_full = self.kv_block_allocator.allocate_memory_blocks_gpu(num_new_blocks_gpu)

        prefix = torch.cumsum(needs_new_i64_mask, dim=0) - needs_new_i64_mask
        safe_prefix = prefix.clamp(max=self.max_requests - 1)
        allocated_per_slot = allocated_full[safe_prefix]
        return needs_new_in_resumed, allocated_per_slot

    def _writeback_resume(
        self,
        slot_idx: Tensor,
        needs_new_in_resumed: Tensor,
        allocated_per_slot: Tensor,
    ) -> None:
        """Apply the allocated blocks to the request bookkeeping tensors."""
        # 2D masked write: request_to_kv_block_ids[i, col_counts[i]] = allocated.
        col_idx_long = self.request_kv_block_counts.to(torch.int64).clamp(
            min=0, max=self.max_kv_block_count - 1
        )
        current_vals = self.request_to_kv_block_ids[slot_idx, col_idx_long]
        new_vals = torch.where(needs_new_in_resumed, allocated_per_slot, current_vals)
        self.request_to_kv_block_ids[slot_idx, col_idx_long] = new_vals

        # Increment per-slot block counts where needs_new_in_resumed.
        self.request_kv_block_counts.add_(
            needs_new_in_resumed.to(self.request_kv_block_counts.dtype)
        )

        new_last = torch.where(
            needs_new_in_resumed,
            allocated_per_slot.to(self.request_last_kv_block_id.dtype),
            self.request_last_kv_block_id,
        )
        self.request_last_kv_block_id.copy_(new_last)

    def _graphed_resume_body(self) -> None:
        """Graph-safe resume body: pure GPU tensor ops over max_requests buffers.

        Layout: ``[active | paused | dead]``.
        ``_ur.new_active_gpu`` = boundary between active and paused (= active count).
        ``_ur.new_total_gpu`` = boundary between paused and dead.

        Reads these boundaries, writes the updated ``_ur.new_active_gpu`` back
        in-place (incremented by the number of resumed requests), and leaves
        ``_ur.resume_count_gpu`` holding the count of requests actually resumed.
        # INVARIANT: ``_ur.new_active_gpu`` is mutated in-place across calls
        # — called twice (once before and once after eviction), each call
        # reads the current boundary and advances it. The pre-update value
        # is snapshotted via ``.clone()`` below so the resumed-range
        # computation in ``_allocate_for_resume`` reads the OLD boundary
        # instead of the just-published new one.
        """
        slot_idx = self._ur.arange_long
        # Snapshot the pre-update boundary. ``_ur.new_active_gpu`` is mutated
        # in-place below (the publish-before-allocate step), so a plain alias
        # would read the post-update value when ``_allocate_for_resume``
        # launches its kernels. The clone copies the storage at this stream
        # point, decoupling subsequent reads from the in-place write.
        active_cur = self._ur.new_active_gpu.clone()
        total_cur = self._ur.new_total_gpu
        block_size_threshold = self.block_size_tokens - 1 - self.num_speculative_tokens

        # Slot-role masks for [active | paused | dead].
        is_active_now = slot_idx < active_cur
        is_paused_now = (slot_idx >= active_cur) & (slot_idx < total_cur)

        active_avail_gpu, needs_new_block_full, paused_blocks_i64 = self._compute_resume_budget(
            is_active_now, is_paused_now, block_size_threshold
        )
        resume_count_gpu = self._compute_fits(
            paused_blocks_i64, is_paused_now, active_avail_gpu, active_cur, total_cur
        )

        # Publish boundary updates before allocate, so subsequent helpers
        # (including the prefix-aware allocator's LRU eviction) see them.
        self._ur.resume_count_gpu.copy_(resume_count_gpu.view(1))
        self._ur.new_active_gpu.copy_((active_cur.view(()) + resume_count_gpu).view(1))

        needs_new_in_resumed, allocated_per_slot = self._allocate_for_resume(
            slot_idx, active_cur, resume_count_gpu, needs_new_block_full
        )
        self._writeback_resume(slot_idx, needs_new_in_resumed, allocated_per_slot)

    def _graphed_chunked_positioning(self) -> None:
        """Graph-safe chunked-prefill positioning.

        Finds the chunked request slot via a GPU-scalar search, then moves
        it to its hidden slot via two cheap pairwise swaps:

        - **Case 1 (was active)**: swap chunked with the LAST RESUMED slot,
          then swap (now-chunked, at ``new_active - 1``) with the LAST
          PAUSED slot at ``new_total - 1``. After the boundary decrements
          below, slot ``chunked_idx`` ends up holding the LAST RESUMED
          (active state in the active region), slot ``new_active - 1``
          ends up holding the LAST PAUSED (paused state in the paused
          region — at the FIRST PAUSED position post-decrement, which
          reorders paused entries; left-to-right paused order is
          sacrificed for the cheaper swap path), and slot ``new_total - 1``
          holds the chunked, hidden by the boundary decrement.

        - **Case 2 (was hidden)**: a single swap from ``chunked_idx`` to
          ``total_cur``. Both slots are in the dead region, so this is
          purely bookkeeping with no active-state consequences. Swap 2
          collapses to a self-swap no-op in this case.

        # INVARIANT: Case 1 cannot use a single chunked_idx ↔ new_total - 1
        # swap. ``[chunked_idx, new_total)`` straddles the active|paused
        # boundary whenever ``resume_1 + resume_2 > 0`` (resumed slots sit
        # between the chunked and the remaining paused tail). A direct swap
        # would deposit the LAST PAUSED slot's data at ``chunked_idx``,
        # which stays inside the active region after the decrement —
        # leaving paused state (``last_kv_block`` full, paused
        # ``mamba_state_idx``) in the active region, and the next forward
        # overwrites position 0 of an already full block, eventually
        # producing NaN logits in the longest-lived leftmost slots.

        Adjusts ``_ur.new_total_gpu`` and ``_ur.new_active_gpu`` in-place
        (decrements both by 1 for Case 1 — hiding the active chunked request).
        """
        max_req = self.max_requests

        # -- Find the chunked request across the full buffer --
        chunked_mask = self.request_ids[:max_req] == self._ur.chunked_id_gpu
        has_any = chunked_mask.any()
        chunked_idx = chunked_mask.to(torch.int32).argmax().to(torch.int64)

        active_cur = self._ur.new_active_gpu.view(())  # 0-d int64
        total_cur = self._ur.new_total_gpu.view(())  # 0-d int64

        # Case 1: chunked was active (index < total).
        is_active_chunked = has_any & (chunked_idx < total_cur)
        case1_swap1_dst = (active_cur - 1).clamp(min=0)
        case1_swap2_dst = (total_cur - 1).clamp(min=0)
        # Skip swap 1 when chunked is already at the LAST RESUMED slot
        # (no resume happened, so chunked sits at ``active_cur - 1`` directly).
        case1_swap1_needed = is_active_chunked & (chunked_idx != case1_swap1_dst)
        # Swap 2 always runs in case 1, but degenerates to a self-swap when
        # there are no remaining paused (``new_total == new_active``).
        case1_swap2_needed = is_active_chunked & (case1_swap1_dst != case1_swap2_dst)

        # Case 2: chunked was hidden (index >= total) → swap chunked_idx ↔
        # total_cur. Clamp dst so the indexed write can never reach past
        # the buffer when total_cur somehow saturates to max_requests
        # (defensive; shouldn't occur because a hidden chunked implies at
        # least one slot remains beyond total_cur).
        is_hidden_chunked = has_any & ~is_active_chunked
        case2_dst = total_cur.clamp(max=self.max_requests - 1)
        case2_swap_needed = is_hidden_chunked & (chunked_idx != case2_dst)

        # Unified swap 1: chunked_idx ↔ {LAST RESUMED in case 1, total_cur in case 2}.
        # Defaults to (0, 0) self-swap when neither case fires.
        zero = torch.zeros_like(chunked_idx)
        swap1_src = torch.where(case1_swap1_needed | case2_swap_needed, chunked_idx, zero)
        swap1_dst = torch.where(
            case1_swap1_needed,
            case1_swap1_dst,
            torch.where(case2_swap_needed, case2_dst, zero),
        )
        self._ur.swap_src.copy_(swap1_src.view(1))
        self._ur.swap_dst.copy_(swap1_dst.view(1))
        self._swap_book_keeping_tensors(
            src_idxs=self._ur.swap_src,
            dst_idxs=self._ur.swap_dst,
            next_tokens=self._ur.next_tokens,
            new_speculative_tokens=self._ur.spec_tokens,
        )

        # Swap 2: in case 1, move chunked from ``new_active - 1`` to
        # ``new_total - 1`` (LAST PAUSED → LAST RESUMED's old slot). No-op
        # in case 2 or when no remaining paused.
        swap2_src = torch.where(case1_swap2_needed, case1_swap1_dst, zero)
        swap2_dst = torch.where(case1_swap2_needed, case1_swap2_dst, zero)
        self._ur.swap_src.copy_(swap2_src.view(1))
        self._ur.swap_dst.copy_(swap2_dst.view(1))
        self._swap_book_keeping_tensors(
            src_idxs=self._ur.swap_src,
            dst_idxs=self._ur.swap_dst,
            next_tokens=self._ur.next_tokens,
            new_speculative_tokens=self._ur.spec_tokens,
        )

        # Hide the active chunked request by removing it from both the active
        # and total counts.  The chunked slot was included in ``new_active`` at
        # the bucket-sum step (``bucket_counts[0] + bucket_counts[1]``), so
        # decrementing only ``new_total`` would leave ``new_active > new_total``
        # — the post-step layout would have a negative paused count and the
        # next forward would read past the end of valid token state.
        sub = is_active_chunked.to(torch.int64).view(1)
        self._ur.new_total_gpu.sub_(sub)
        self._ur.new_active_gpu.sub_(sub)

    def _graphed_eviction_body(self) -> None:
        """Graph-safe eviction: masked no-op when there is no overflow.

        Layout: ``[active | paused | dead]``.
        Reads ``_ur.new_active_gpu`` / ``_ur.new_total_gpu`` as the
        current boundaries (written by the preceding resume body within
        the same graph capture).  Computes the overflow and eviction
        count as GPU scalars, packs evicted request IDs into
        ``_ur.evict_ids_buf``, releases their blocks, and shrinks the
        total boundary.

        In ``[active | paused]`` layout the oldest paused requests are at
        the rightmost end of the paused region (just before dead slots).
        Eviction removes from the tail, so **no permutation is needed** --
        we just release and decrement ``_ur.new_total_gpu``.

        When ``overflow <= 0`` (the common case), ``evict_count == 0``
        and every masked operation is a no-op.
        """
        max_req = self.max_requests
        slot_idx = self._ur.arange_long
        active_cur = self._ur.new_active_gpu  # (1,) int64
        total_cur = self._ur.new_total_gpu  # (1,) int64
        paused_count_const = self.kv_block_allocator.paused_count  # Python int const

        # [active | paused]: paused at [active_cur, total_cur)
        is_paused = (slot_idx >= active_cur) & (slot_idx < total_cur)

        # -- Overflow --
        paused_blocks = torch.where(
            is_paused,
            self.request_kv_block_counts.to(torch.int64),
            torch.zeros_like(self.request_kv_block_counts, dtype=torch.int64),
        )
        paused_sum = paused_blocks.sum()
        overflow = paused_sum - paused_count_const
        has_overflow = overflow > 0

        # -- Evict count: smallest k from the RIGHT of the paused region
        #    whose cumulative block count >= overflow.
        # reverse_from_right[i] = sum of paused_blocks from i to end.
        # For paused slots: forward_cumsum gives left-to-right running sum.
        # reverse = total - exclusive_cumsum.
        forward_cumsum = paused_blocks.cumsum(dim=0)
        exclusive_cumsum = forward_cumsum - paused_blocks
        reverse_from_right = paused_sum - exclusive_cumsum

        # Slots where evicting from this slot rightward covers the overflow.
        sufficient = is_paused & (reverse_from_right >= overflow) & has_overflow
        sufficient_idx = torch.where(sufficient, slot_idx, torch.full_like(slot_idx, -1))
        rightmost_i = sufficient_idx.max()  # 0-d int64 — rightmost sufficient slot

        evict_count = torch.where(
            has_overflow, total_cur.view(()) - rightmost_i, torch.zeros_like(total_cur.view(()))
        )

        # -- Eviction mask: rightmost `evict_count` paused slots --
        evict_start = total_cur.view(()) - evict_count
        evict_mask = is_paused & (slot_idx >= evict_start)

        # -- Capture evicted IDs before release --
        scatter_pack_valid(
            self.request_ids[:max_req],
            evict_mask,
            self._ur.evict_ids_buf,
            fill_value=-1,
            sink_idx_value=max_req,
        )

        # -- Release blocks for evicted requests --
        self._release_blocks_from_mask_gpu(evict_mask)

        # -- Update boundaries --
        # In [active | paused], evicted slots are already at the tail.
        # No permutation needed — just shrink total. Active boundary unchanged.
        self._ur.new_total_gpu.sub_(evict_count.view(1))
        self._ur.evict_count_gpu.copy_(evict_count.view(1))

    def update_requests(
        self,
        active_requests_mask: Tensor,
        new_tokens: Tensor,
        new_speculative_tokens: Tensor = None,
    ) -> Dict:
        """Update context state after calling engine.step().

        Uses a **permutation-based reorder** to reclassify every request slot into
        one of five buckets and apply a single ``argsort``-derived permutation to
        every bookkeeping tensor.  This replaces the previous multi-pass
        swap/move approach, cutting kernel-launch count significantly.
        Steady-state GPU->CPU syncs are 1 per step: both graphs are
        queued back-to-back and a single ``.cpu()`` on the combined
        sync buffer reads all outputs.  Prefix caching adds a
        conditional drain sync after the graphs.

        *Note*: All bookkeeping tensors (i.e., ``self.request_*``) are laid out
        contiguously with ``[active | paused | dead]`` layout:

        - ``0:active_count`` -> active requests
        - ``active_count:total_request_count`` -> paused requests
        - ``total_request_count:max_requests`` -> dead/completed slots

        Bucket assignment (lower = further left in the final layout):

        ====== =============================== ================
        Key    Semantics                        Final region
        ====== =============================== ================
        0      Active -> continues               ``[0, new_active - chunked)``
        1      Active -> chunked prefill          rightmost active
        2      Paused -> stays paused             ``[new_active, new_active + stayed_paused)``
        3      Active -> newly paused (full blk)  ``[new_active + stayed_paused, new_total)``
        4      Active -> finished                 ``[new_total, total_pre)``
        ====== =============================== ================

        After the permutation, the layout is ``[active | paused | finished]``.
        The finished region is dead (blocks released, slots reusable).  The
        graphed helpers ``_graphed_resume_body``, ``_graphed_eviction_body``,
        and ``_graphed_chunked_positioning`` then adjust the active-paused
        boundary via boundary shifts and masked swaps within the
        already-sorted buffer.

        Args:
            active_requests_mask (Tensor): 1D byte mask over active requests
                (1 = continue, 0 = finished).
            new_tokens (Tensor): Newly sampled tokens, one per active request.
            new_speculative_tokens (Tensor): Newly sampled speculative tokens,
                with num_speculative tokens per active request.
                (num_speculative_tokens, active_request_length)

        Return:
            Dict with ``newly_paused_request_ids`` and ``evict_request_ids``.
        """
        # Eager (non-graphed) path: prepare, run both bodies, finalize.
        # The graphed path in TextGenerationController wraps each body method
        # in its own CudaGraphManager (capture-on-first-call, replay after);
        # this method is the backward-compatible entry point used by tests
        # and callers that don't need graph capture.
        self._prepare_update_requests_metadata()
        has_chunked = self._prepare_update_requests_new_tokens(
            active_requests_mask, new_tokens, new_speculative_tokens
        )
        self._classify_and_resume_body()
        self._evict_resume_chunked_tokens_body()
        return self._finalize_update_requests(has_chunked)

    def _prepare_update_requests_metadata(self) -> None:
        """Pre-forward prep: work that depends only on previous-step state.

        Reads GPU mirrors (set by prepare_attn_init or previous update_requests)
        to fill the graph body's input scalars.  Copies paused tokens and
        snapshots active_request_ids.
        """
        self.num_prefill_requests = 0
        self.request_in_prefill_status_tensor[self.request_in_prefill_status_tensor == 1] = 0

        # Graph body inputs: read from GPU mirrors (no CPU ints involved).
        self._ur.active_pre_gpu.copy_(self._real_request_count_gpu)
        self._ur.total_pre_gpu.copy_(self._total_request_count_gpu)
        self._ur.chunked_id_gpu.fill_(self.chunked_prefill_request_id)

        # Copy paused tokens into the static buffer.
        # This is a variable-length copy; CPU ints needed for slicing.
        active_count = int(self._real_request_count_gpu.item())
        total_count = int(self._total_request_count_gpu.item())
        paused_count = total_count - active_count
        if paused_count > 0:
            paused_slice = slice(active_count, total_count)
            self._ur.next_tokens[paused_slice].copy_(self.paused_tokens)
            if self._ur.spec_tokens is not None and self.paused_speculative_tokens is not None:
                self._ur.spec_tokens[:, paused_slice].copy_(self.paused_speculative_tokens)

        # Snapshot request IDs before update_requests permutes them.
        self.active_request_ids[:active_count].copy_(self.request_ids[:active_count])

    def _prepare_update_requests_new_tokens(
        self,
        active_requests_mask: Tensor,
        new_tokens: Tensor,
        new_speculative_tokens: Tensor = None,
    ) -> bool:
        """Post-sampling prep: copy mask and sampled tokens into static buffers.

        Returns whether a chunked prefill request is active.
        """
        # Source the active count from the caller's mask rather than syncing
        # ``_real_request_count_gpu``. The caller (controller post-sample path
        # or eager ``update_requests`` shim) sized ``new_tokens`` /
        # ``new_speculative_tokens`` to the same count, so this is the contract.
        # Avoids a CPU sync, and stays correct during the engine warmup path
        # where ``context.reset()`` leaves the GPU mirror stale.
        active_pre = active_requests_mask.numel()

        has_chunked = self.chunked_prefill_request_id != -1
        if has_chunked and active_pre > 0:
            active_requests_mask[-1] = 1

        self._ur.active_mask.zero_()
        self._ur.active_mask[:active_pre].copy_(active_requests_mask)

        self._ur.next_tokens[:active_pre].copy_(new_tokens)
        if new_speculative_tokens is not None and self._ur.spec_tokens is not None:
            self._ur.spec_tokens[:, :active_pre].copy_(new_speculative_tokens)

        return has_chunked

    def _compute_classification_masks(
        self, slot_idx: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute the per-slot classification masks for graph body 1.

        Returns ``(is_active, is_paused, finished_mask, nb_mask, chunked_mask)``
        where ``nb_mask`` is "needs new block" (would-pause) including any
        ``excess_mask`` slots forced over the active budget.
        """
        max_req = self.max_requests
        amask = self._ur.active_mask
        is_active = slot_idx < self._ur.active_pre_gpu
        is_paused = (slot_idx >= self._ur.active_pre_gpu) & (slot_idx < self._ur.total_pre_gpu)

        finished_mask = is_active & (amask == 0)
        nb_mask = (
            is_active
            & (
                self.request_last_kv_block_offset[:max_req]
                >= self.block_size_tokens - 1 - self.num_speculative_tokens
            )
            & ~finished_mask
        )
        chunked_mask = is_active & (self.request_ids[:max_req] == self._ur.chunked_id_gpu)
        nb_mask = nb_mask & ~chunked_mask

        # Force excess "would continue" requests over the per-step active
        # budget into the newly-paused bucket. Skipped when a chunked-prefill
        # request is in flight — it consumes the rightmost active slot.
        would_continue = is_active & (amask != 0) & ~nb_mask & ~chunked_mask
        continue_rank = would_continue.to(torch.int64).cumsum(dim=0)
        has_any_chunked = chunked_mask.any()
        excess_mask = (
            would_continue & (continue_rank > self._ur.max_allowed_active) & ~has_any_chunked
        )
        nb_mask = nb_mask | excess_mask
        return is_active, is_paused, finished_mask, nb_mask, chunked_mask

    def _build_sort_key_and_permute(
        self,
        is_active: Tensor,
        is_paused: Tensor,
        finished_mask: Tensor,
        nb_mask: Tensor,
        chunked_mask: Tensor,
    ) -> None:
        """Build the bucket sort key and apply the permutation to bookkeeping.

        Bucket keys: 0=continues, 1=chunked, 2=stayed-paused, 3=newly-paused,
        4=finished.  Later masked_fill_ wins, so effective precedence is
        chunked > finished > nb > active > paused > dead.
        # INVARIANT: bucket numeric ordering 0 < 1 < 2 < 3 < 4 IS the layout
        # contract — argsort below produces [active | paused | finished].
        # Adding a fractional/intermediate value silently breaks downstream.
        """
        sort_key = self._ur.sort_key
        sort_key.fill_(5)  # dead
        sort_key.masked_fill_(is_paused, 2)   # stays paused
        sort_key.masked_fill_(is_active, 0)   # continues active
        sort_key.masked_fill_(nb_mask, 3)     # newly paused (needs new block)
        sort_key.masked_fill_(finished_mask, 4)  # finished
        sort_key.masked_fill_(chunked_mask, 1)   # chunked (rightmost active)

        # INVARIANT: stable=True is required so requests within the same
        # bucket keep relative order — chunked-prefill positioning depends on
        # the chunked slot landing at the rightmost active position, and the
        # left-to-right paused order is what `_finalize_update_requests`'s
        # newly-paused-survivor formula reads to slice the buffer correctly.
        perm = sort_key.argsort(stable=True)
        self._permute_book_keeping_tensors(perm, self._ur.next_tokens)

        # Dead slots (key 5) are clamped to 4 so they share the finished
        # bucket, which is never read.  This keeps _ur.bucket_counts at 5.
        self._ur.bucket_counts.zero_()
        self._ur.bucket_counts.index_add_(
            0, sort_key.clamp(max=4).to(torch.int64), self._ur.ones_long
        )

        # [active | paused] output: active_count = bucket[0] + bucket[1]
        self._ur.new_active_gpu.copy_(self._ur.bucket_counts[0:1] + self._ur.bucket_counts[1:2])
        self._ur.new_total_gpu.copy_(
            self._ur.bucket_counts[0:1]
            + self._ur.bucket_counts[1:2]
            + self._ur.bucket_counts[2:3]
            + self._ur.bucket_counts[3:4]
        )

    def _capture_newly_paused_ids(self, slot_idx: Tensor) -> None:
        """Pack newly-paused request IDs into the static-address buffer.

        Newly-paused are at ``[new_active + stayed_paused, new_total)``;
        ``stayed_paused_count = bucket[2]``. Must run before eviction can
        modify ``request_ids``.
        """
        newly_paused_start = self._ur.new_active_gpu + self._ur.bucket_counts[2:3]
        newly_paused_mask = (slot_idx >= newly_paused_start) & (
            slot_idx < self._ur.new_total_gpu
        )
        scatter_pack_valid(
            self.request_ids[: self.max_requests],
            newly_paused_mask,
            self._ur.newly_paused_ids_buf,
            fill_value=-1,
            sink_idx_value=self.max_requests,
        )

    def _classify_and_resume_body(self) -> Tensor:
        """Graph body 1: classify, permute, release finished, resume.

        Returns ``_ur.combined_sync`` so ``CudaGraphManager`` has a tensor
        to register as the captured graph's output. Callers ignore the
        return value; the actual outputs are the in-place writes to
        bookkeeping tensors and the bucket/resume slots of
        ``_ur.combined_sync``.
        """
        slot_idx = self._ur.arange_long
        max_req = self.max_requests

        is_active, is_paused, finished_mask, nb_mask, chunked_mask = (
            self._compute_classification_masks(slot_idx)
        )
        self._build_sort_key_and_permute(
            is_active, is_paused, finished_mask, nb_mask, chunked_mask
        )

        # Release blocks from the now-finished region [new_total, total_pre).
        finished_region_mask = (slot_idx >= self._ur.new_total_gpu) & (
            slot_idx < self._ur.total_pre_gpu
        )
        self._release_blocks_from_mask_gpu(finished_region_mask)

        # Snapshot last-block IDs before resume reallocates them; needed for
        # multi-token bookkeeping in graph 2 to resolve which tokens go to
        # the previous block vs the newly-allocated one.
        if self.num_speculative_tokens > 0:
            self._ur.prev_last_block_ids[:max_req].copy_(self.request_last_kv_block_id[:max_req])

        self._capture_newly_paused_ids(slot_idx)
        self._graphed_resume_body()

        # Stage graph-1 outputs into the combined sync buffer.
        S = DynamicInferenceContext
        self._ur.combined_sync[S._SYNC_BUCKET0 : S._SYNC_BUCKET3 + 1].copy_(
            self._ur.bucket_counts[:4]
        )
        self._ur.combined_sync[S._SYNC_RESUME1 : S._SYNC_RESUME1 + 1].copy_(
            self._ur.resume_count_gpu
        )

        return self._ur.combined_sync

    def _writeback_request_offsets(
        self,
        slot_idx: Tensor,
        is_active_post: Tensor,
        is_alive: Tensor,
        old_offsets: Tensor,
    ) -> None:
        """Update the per-request offset tensors (kv_length, query, last_block).

        For active slots: kv_length advances by query_length, query resets to
        ``num_gen``, last_kv_block_offset advances by num_gen modulo block size.
        For non-alive (finished) slots: kv_length and output_length zero out;
        last_kv_block_offset zeroes too.

        # INVARIANT: query_length is preserved unchanged for paused slots.
        # Dead slots zero out, active slots reset to ``num_gen``, paused slots
        # retain whatever pre-writeback value they had (``num_gen`` for a
        # request paused mid-decode, ``chunk_length`` for one paused right
        # after prefill).  This pending value is consumed at the resume
        # step's writeback, where ``kv_length_offsets += query_length *
        # is_active_f`` compensates for the advancement that was skipped
        # when the request was newly paused (``is_active_post=False`` at
        # that step's writeback, so the multiply-by-0 dropped the
        # advancement).  Resetting query_length to 0 for paused would lose
        # that pending advancement forever; resetting to ``num_gen`` would
        # be correct mid-decode but wrong right after prefill (would
        # under-advance ``kv_length_offsets`` by ``chunk_length - num_gen``
        # tokens and the request would over-generate or read stale KV).
        # Either misstep would eventually trip the ``step_time / len(tokens)``
        # divide-by-zero in ``post_process_requests`` when the engine's trim
        # leaves the over-generated step's tokens list empty.
        """
        num_gen = self._ur.num_generated_tokens
        is_active_f = is_active_post.to(self.request_kv_length_offsets.dtype)
        is_alive_f = is_alive.to(self.request_kv_length_offsets.dtype)

        # ``addcmul_`` fuses the query_length * is_active_f mul with the add,
        # avoiding the temp the unfused ``add_(query * is_active_f)`` allocates
        # at every replay.
        self.request_kv_length_offsets.addcmul_(
            self.request_query_lengths.to(self.request_kv_length_offsets.dtype), is_active_f
        )
        self.request_query_lengths.mul_(is_alive_f.to(self.request_query_lengths.dtype))
        self.request_query_lengths.masked_fill_(is_active_post, num_gen)
        self.request_kv_length_offsets.mul_(is_alive_f)
        self.request_output_lengths.mul_(is_alive_f.to(self.request_output_lengths.dtype))
        self.request_last_kv_block_offset.copy_(
            torch.where(
                is_active_post,
                (old_offsets + num_gen) % self.block_size_tokens,
                old_offsets * is_alive.to(torch.int32),
            )
        )

    def _writeback_token_bookkeeping_single(self, slot_idx: Tensor) -> None:
        """Single-token-per-request token bookkeeping (decode without spec).

        In ``[active | paused]``, active requests start at index 0, so
        ``slot_idx`` is the identity mapping — no offset needed.
        """
        max_tok = self.max_requests
        self.token_to_input_ids[:max_tok].copy_(self._ur.next_tokens[:max_tok])
        self.token_to_request_idx[:max_tok].copy_(
            slot_idx.to(self.token_to_request_idx.dtype)
        )
        self.token_to_pos_ids[:max_tok].copy_(self.request_kv_length_offsets[:max_tok])
        self.token_to_position_in_request[:max_tok].copy_(
            self.request_kv_length_offsets[:max_tok]
        )
        self.token_to_block_idx[:max_tok].copy_(self.request_last_kv_block_id[:max_tok])
        self.token_to_local_position_within_kv_block[:max_tok].copy_(
            self.request_last_kv_block_offset[:max_tok]
        )

    def _writeback_token_bookkeeping_multi(
        self, slot_idx: Tensor, old_offsets: Tensor
    ) -> None:
        """Multi-token-per-request token bookkeeping (speculative decoding).

        Builds (max_req * num_gen) flat token records. The block-boundary
        crossing check selects between ``request_last_kv_block_id`` and the
        snapshot ``_ur.prev_last_block_ids`` so tokens that landed before
        the boundary still reference the previous block.
        """
        max_req = self.max_requests
        num_gen = self._ur.num_generated_tokens
        max_tok = max_req * num_gen

        base = self._ur.next_tokens[:max_req]
        spec = self._ur.spec_tokens[:, :max_req]
        interleaved = torch.cat([base.unsqueeze(0), spec], dim=0)
        # ``interleaved.T`` is non-contiguous; ``.reshape(-1)`` would force
        # a contig copy. Writing through a strided 2-D view of the output
        # buffer does the reorder in a single strided D2D copy instead.
        self.token_to_input_ids[:max_tok].view(max_req, num_gen).copy_(interleaved.T)

        kv_offsets = self.request_kv_length_offsets[:max_req]
        pos_base = kv_offsets.repeat_interleave(num_gen)
        pos_ids = pos_base + self._ur.pos_offset_pattern
        self.token_to_pos_ids[:max_tok].copy_(pos_ids)
        self.token_to_position_in_request[:max_tok].copy_(pos_ids)

        self.token_to_request_idx[:max_tok].copy_(
            slot_idx.repeat_interleave(num_gen).to(self.token_to_request_idx.dtype)
        )
        self.token_to_local_position_within_kv_block[:max_tok].copy_(
            self.token_to_pos_ids[:max_tok] % self.block_size_tokens
        )

        old_off = old_offsets[:max_req]
        raw_positions = old_off[:, None] + 1 + self._ur.gen_arange[None, :]
        crosses_boundary = raw_positions >= self.block_size_tokens
        current_ids = self.request_last_kv_block_id[:max_req]
        prev_ids = self._ur.prev_last_block_ids[:max_req]
        request_has_crossing = crosses_boundary.any(dim=1)
        use_prev = request_has_crossing[:, None] & ~crosses_boundary
        block_idx_2d = torch.where(
            use_prev,
            prev_ids[:, None].expand(-1, num_gen),
            current_ids[:, None].expand(-1, num_gen),
        )
        self.token_to_block_idx[:max_tok].copy_(block_idx_2d.reshape(-1))

    def _write_phase2_sync_outputs(self) -> None:
        """Pack the graph-2 outputs into the combined sync buffer.

        Slots written here are read alongside graph 1's outputs by the single
        ``.cpu()`` in ``_finalize_update_requests``.
        """
        S = DynamicInferenceContext
        self._ur.combined_sync[S._SYNC_RESUME2 : S._SYNC_RESUME2 + 1].copy_(
            self._ur.resume_count_gpu
        )
        self._ur.combined_sync[S._SYNC_ACTIVE_FINAL : S._SYNC_ACTIVE_FINAL + 1].copy_(
            self._ur.new_active_gpu
        )
        self._ur.combined_sync[S._SYNC_TOTAL_FINAL : S._SYNC_TOTAL_FINAL + 1].copy_(
            self._ur.new_total_gpu
        )
        self._ur.combined_sync[S._SYNC_TOTAL_AVAIL : S._SYNC_TOTAL_AVAIL + 1].copy_(
            self.kv_block_allocator.total_avail_gpu.to(torch.int64)
        )
        if self.is_hybrid_model:
            self._ur.combined_sync[S._SYNC_MAMBA_FREE : S._SYNC_MAMBA_FREE + 1].copy_(
                self.mamba_metadata._free_slot_count_gpu.to(torch.int64)
            )
        self._ur.combined_sync[S._SYNC_EVICT_COUNT : S._SYNC_EVICT_COUNT + 1].copy_(
            self._ur.evict_count_gpu
        )

    def _writeback_gpu_count_mirrors(self) -> None:
        """Refresh the bridge GPU scalars consumed by the next step.

        ``prepare_attn_init`` and ``_prepare_update_requests_metadata`` read
        these directly, avoiding a CPU sync between steps.
        """
        self._real_request_count_gpu.copy_(self._ur.new_active_gpu)
        self._real_token_count_gpu.copy_(
            self._ur.new_active_gpu * self._ur.num_generated_tokens
        )
        self._real_decode_count_gpu.copy_(self._ur.new_active_gpu)
        self._real_prefill_count_gpu.zero_()
        self._total_request_count_gpu.copy_(self._ur.new_total_gpu)
        self._paused_request_count_gpu.copy_(
            self._ur.new_total_gpu - self._ur.new_active_gpu
        )

    def _evict_resume_chunked_tokens_body(self) -> Tensor:
        """Graph body 2: evict, resume, chunked positioning, token bookkeeping.

        Returns ``_ur.combined_sync`` so ``CudaGraphManager`` has a tensor
        to register as the captured graph's output (same reason as
        :meth:`_classify_and_resume_body`).

        # INVARIANT: graph 1 (`_classify_and_resume_body`) must run before this
        # graph 2; graph 1 writes BUCKET*/RESUME1 into `_ur.combined_sync`,
        # and this body writes RESUME2/ACTIVE_FINAL/TOTAL_FINAL/etc. into
        # different slots. The single `.cpu()` in `_finalize_update_requests`
        # reads everything — re-ordering the bodies invalidates the buffer.
        """
        self._graphed_eviction_body()
        self._graphed_resume_body()
        self._graphed_chunked_positioning()

        slot_idx = self._ur.arange_long

        # [active | paused]: active at [0, _ur.new_active_gpu).
        is_active_post = slot_idx < self._ur.new_active_gpu
        is_alive = slot_idx < self._ur.new_total_gpu
        # Snapshot offsets before they're mutated; multi-token bookkeeping
        # below needs the pre-update value.
        old_offsets = self.request_last_kv_block_offset[: self.max_requests].clone()

        self._writeback_request_offsets(slot_idx, is_active_post, is_alive, old_offsets)

        if self._ur.num_generated_tokens == 1:
            self._writeback_token_bookkeeping_single(slot_idx)
        else:
            self._writeback_token_bookkeeping_multi(slot_idx, old_offsets)

        self._write_phase2_sync_outputs()
        self._writeback_gpu_count_mirrors()

        return self._ur.combined_sync

    def _finalize_update_requests(self, has_chunked: bool) -> Dict:
        """Phase 3: single GPU->CPU sync, then mirror state to host."""
        # THE sync: one .cpu().tolist() consumes everything the graphed bodies
        # wrote into _ur.combined_sync.
        sv = UpdateRequestsSyncedCounters.from_buffer(self._ur.combined_sync.cpu().tolist())

        # Pre-step total, captured before the writes below overwrite it.
        # The graphed bodies only release blocks for finished requests
        # (bucket4 = total_pre - sum(bucket0..3)) and evicted ones. Used
        # below to gate the prefix-caching dereg drain so it only fires
        # when the queue could plausibly hold events.
        total_pre = self.total_request_count

        self.reset_attention_state()

        if sv.total_final == 0 and not has_chunked:
            self.request_to_kv_block_ids.fill_(-1)
            self.total_request_count = 0
            self.active_token_count = 0
            self.paused_request_count = 0
            self.reset_mamba_state()
            self.kv_block_allocator.total_avail = sv.total_avail
            # Body 1 just released every active request's blocks. If
            # prefix caching is on, that release enqueued dereg events
            # we need to drain so the host-side hash dict stays in sync.
            if self.enable_prefix_caching and total_pre > 0:
                self.kv_block_allocator._pc_dereg_may_have_events = True
            self.kv_block_allocator.drain_pending_dereg()
            return {"newly_paused_request_ids": None, "evict_request_ids": None}

        # [active | paused]: paused_count = total - active.
        new_paused_final = sv.total_final - sv.active_final
        self.paused_request_count = new_paused_final
        self.total_request_count = sv.total_final
        active_request_count = sv.active_final
        num_gen = self._ur.num_generated_tokens
        self.active_token_count = active_request_count * num_gen
        self.kv_block_allocator.total_avail = sv.total_avail
        if self.is_hybrid_model:
            self.mamba_metadata.mamba_state_free_slot_count = sv.mamba_free

        # The pre-resume newly-paused buffer holds bucket3 entries in slot
        # order; resume eats stayed-paused (bucket2) first from the LEFT
        # of the paused region, and graphed eviction takes from the RIGHT
        # (newly-paused side). Tracking each pass's contribution exactly
        # gives the correct surviving slice — naive ``bucket3 - (resume1
        # + resume2)`` undercounts whenever bucket2 absorbed some resume.
        left_eaten_by_r1 = max(0, sv.resume1 - sv.bucket2)
        newly_after_r1 = sv.bucket3 - left_eaten_by_r1
        stayed_after_r1 = max(0, sv.bucket2 - sv.resume1)
        right_eaten_by_evict = min(sv.evict_count, newly_after_r1)
        newly_after_evict = newly_after_r1 - right_eaten_by_evict
        stayed_after_evict = stayed_after_r1 - (sv.evict_count - right_eaten_by_evict)
        left_eaten_by_r2 = max(0, sv.resume2 - stayed_after_evict)
        final_newly_paused = newly_after_evict - left_eaten_by_r2

        newly_paused_request_ids = None
        if final_newly_paused > 0:
            # Survivors live at the right of the original buffer's bucket3
            # range — eviction trimmed the rightmost end and the two
            # resume passes peeled off from the leftmost newly-paused.
            survivor_start = left_eaten_by_r1 + left_eaten_by_r2
            newly_paused_request_ids = self._ur.newly_paused_ids_buf[
                survivor_start : survivor_start + final_newly_paused
            ].clone()
        evict_request_ids = None
        if sv.evict_count > 0:
            evict_request_ids = self._ur.evict_ids_buf[: sv.evict_count].clone()

        # [active | paused]: paused tokens are at [active_final:total_final].
        if new_paused_final > 0:
            self._ur.paused_tokens_buf[:new_paused_final].copy_(
                self._ur.next_tokens[sv.active_final : sv.total_final]
            )
            self.paused_tokens = self._ur.paused_tokens_buf[:new_paused_final]
            if self._ur.spec_tokens is not None:
                self._ur.paused_spec_tokens_buf[:, :new_paused_final].copy_(
                    self._ur.spec_tokens[:, sv.active_final : sv.total_final]
                )
                self.paused_speculative_tokens = self._ur.paused_spec_tokens_buf[
                    :, :new_paused_final
                ]

        # bucket4 (finished count) = pre-step total minus the four kept
        # buckets; together with eviction it covers every release this
        # step. Skip the drain — and its .item() sync — when nothing was
        # released, even with prefix caching enabled.
        bucket4 = total_pre - sv.bucket0 - sv.bucket1 - sv.bucket2 - sv.bucket3
        if self.enable_prefix_caching and (bucket4 + sv.evict_count) > 0:
            self.kv_block_allocator._pc_dereg_may_have_events = True
        self.kv_block_allocator.drain_pending_dereg()

        if __debug__:
            assert active_request_count > 0 or self.chunked_prefill_request_id != -1, (
                "active_request_count == %d with no hidden chunked prefill." % active_request_count
            )

        return {
            "newly_paused_request_ids": newly_paused_request_ids,
            "evict_request_ids": evict_request_ids,
        }

    def calculate_log_probs(
        self, logits: Tensor, new_tokens: Tensor, only_last_token_logits: Optional[bool] = False
    ) -> Tuple[List[List[float]], Tensor]:
        """Calculate log probs for all active requests and return them.

        TODO: @wdykas support top-n log probs.

        Args:
            logits (Tensor): Raw model output logits with shape [1, sequence_length, vocab_size].
            new_tokens (Tensor): The newly sampled tokens.
            only_last_token_logits (bool): If set, the logits are from only the last token in each request

        Returns:
            List of lists where each inner list contains log probs for a request in the
            same order as the active requests (from 0 to active_count).
            log_probs (Tensor): Used to compute top n logprobs later if required.
        """

        # Calculate log_probs (sequence_length x vocab_size)
        logits_squeezed = logits.squeeze(0).float()

        if only_last_token_logits or self.is_decode_only():
            seq_idx = torch.arange(len(new_tokens), dtype=torch.int32, device=logits.device)
            log_probs = F.log_softmax(logits_squeezed[seq_idx], dim=-1)
            selected_log_probs = log_probs[seq_idx, new_tokens]
            return [[lp] for lp in selected_log_probs.tolist()], log_probs

        log_probs = F.log_softmax(logits_squeezed, dim=-1)
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
        active_count = self.total_request_count - self.paused_request_count
        active_query_lengths = self.request_query_lengths[:active_count]

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
        return [lp.tolist() for lp in selected_log_probs_list], log_probs

    def get_kvcache_utilization_stats(self) -> dict:
        """Compute KV cache buffer utilization stats for the current step.

        Returns a dictionary with counts and percentages for both allocated block
        usage (overall buffer occupancy) and active usage (blocks referenced by
        currently active requests this step).

        Return:
            {
            'total_blocks': int,
            'allocated_blocks': int,
            'active_unique_blocks': int,
            'allocated_utilization': float,
            'active_utilization': float,
            'active_request_count': int,
            'paused_request_count': int,
            'gtd_block_count': int,
            }
        """
        total_blocks = max(self.kv_block_allocator.total_count, 1)
        block_count_avail = int(self.kv_block_allocator.total_avail)

        # Overall allocated blocks in the buffer right now.
        allocated_blocks = self.kv_block_allocator.total_count - block_count_avail
        allocated_blocks = int(max(0, allocated_blocks))

        # Active unique blocks referenced by current active requests only.
        active_start = 0
        active_end = self.total_request_count - self.paused_request_count
        if active_end > active_start:
            active_rows = self.request_to_kv_block_ids[active_start:active_end]
            # Filter valid block ids (>= 0) and count unique ids.
            valid_ids = active_rows[active_rows >= 0]
            if valid_ids.numel() > 0:
                unique_ids = torch.unique(valid_ids)
                active_unique_blocks = int(unique_ids.numel())
            else:
                active_unique_blocks = 0
        else:
            active_unique_blocks = 0

        allocated_utilization = float(allocated_blocks) / float(total_blocks)
        active_utilization = float(active_unique_blocks) / float(total_blocks)

        # Diagnostic helpers
        total_request_count = int(self.total_request_count)
        return {
            'total_blocks': int(total_blocks),
            'allocated_blocks': int(allocated_blocks),
            'active_unique_blocks': int(active_unique_blocks),
            'allocated_utilization': allocated_utilization,
            'active_utilization': active_utilization,
            'active_request_count': int(self.get_active_request_count()),
            'paused_request_count': int(self.paused_request_count),
            'block_count_avail': int(block_count_avail),
            'active_token_count': int(self.active_token_count),
            'total_request_count': int(total_request_count),
            'max_requests': int(self.max_requests),
        }
