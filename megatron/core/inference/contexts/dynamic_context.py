# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import math
import warnings
from contextlib import nullcontext
from typing import List, Optional, Sequence, Tuple

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch import Tensor  # type: ignore

from megatron.core import parallel_state
from megatron.core.inference.batch_dimensions_utils import (
    CUDAGraphBatchDimensionBuilder,
    InferenceBatchDimensions,
)
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.unified_memory import (
    UnifiedMemoryUnsupportedError,
    create_unified_mempool,
)
from megatron.core.inference.utils import tensor_swap
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.package_info import __version__ as mcore_version
from megatron.core.ssm.mamba_hybrid_layer_allocation import get_layer_maps_from_layer_type_list
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.utils import deprecate_args
from megatron.core.utils import divide as core_divide
from megatron.core.utils import get_pg_size, internal_api

from .attention_context.mamba_metadata import MambaMetadata
from .attention_context.mha_metadata import GraphedMHAMetadata, NonGraphedMHAMetadata
from .base_context import BaseInferenceContext
from .dynamic_block_allocator import BlockAllocator
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


def get_mem_size_str(n_bytes: int) -> str:
    """Convert number of bytes to human-readable string."""
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

    @deprecate_args(
        *DEPRECATED_ARGS,
        message=(
            "Argument `{name}` has been deprecated. "
            "Only pass `model_config` and `inference_config`"
        ),
    )
    def __init__(self, model_config: TransformerConfig, inference_config: InferenceConfig):
        super().__init__(inference_config=inference_config)

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
        self.num_attention_heads_per_partition = core_divide(num_attention_heads, tp_size)

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
            mamba_conv_states_shape = mamba_inference_state_config.mamba_conv_states_shape
            mamba_ssm_states_shape = mamba_inference_state_config.mamba_ssm_states_shape
            assert (
                mamba_conv_states_shape is not None
            ), "`mamba_conv_states_shape` must be specified for hybrid models"
            assert (
                mamba_ssm_states_shape is not None
            ), "`mamba_ssm_states_shape` must be specified for hybrid models"

            # For hybrid models, the layer map converts the global layer index to the
            # corresponding attention layer index or Mamba layer index depending on the
            # layer type.
            attention_layer_map, mamba_layer_map, _, _ = get_layer_maps_from_layer_type_list(
                mamba_inference_state_config.layer_type_list
            )
            self.num_attention_layers = len(attention_layer_map)
            self.num_mamba_layers = len(mamba_layer_map)
            self.mamba_conv_states_shape = mamba_conv_states_shape
            self.mamba_ssm_states_shape = mamba_ssm_states_shape
            self.layer_map = attention_layer_map | mamba_layer_map
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
        dtype_size_bytes = model_config.params_dtype.itemsize
        self.block_size_tokens = inference_config.block_size_tokens
        if self.cache_mla_latent:
            #   one vector  c_t  (rank)  +  optional RoPE phase slice
            self.kv_reduced_dim = model_config.kv_lora_rank + model_config.qk_pos_emb_head_dim
            self.block_size_bytes = (
                dtype_size_bytes
                * self.num_attention_layers
                * self.block_size_tokens
                * self.kv_reduced_dim
            )
        else:
            self.block_size_bytes = (
                dtype_size_bytes
                * 2  # key, value
                * self.num_attention_layers
                * self.block_size_tokens
                * self.num_attention_heads_per_partition
                * self.hidden_size_per_attention_head
            )
        assert self.block_size_bytes > 0

        mamba_states_memory_per_request = 0
        if self.is_hybrid_model:
            mamba_states_memory_per_request += math.prod(self.mamba_conv_states_shape)
            mamba_states_memory_per_request += math.prod(self.mamba_ssm_states_shape)
            mamba_states_memory_per_request *= self.num_mamba_layers
            mamba_states_memory_per_request *= dtype_size_bytes

        # Unified memory.
        self.unified_memory_level = inference_config.unified_memory_level
        self.persist_cuda_graphs = inference_config.persist_cuda_graphs
        if self.unified_memory_level > 0:
            try:
                self.unified_memory_mempool = create_unified_mempool()
            except UnifiedMemoryUnsupportedError:
                if torch.distributed.get_rank() == 0:
                    warnings.warn(
                        "Unified memory requested but not available; defaulting to GPU memory."
                    )
                self.unified_memory_level = 0

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

        self.block_allocator = BlockAllocator(
            context=self,
            total_count=(
                block_count if self.unified_memory_level == 0 else block_count + paused_block_count
            ),
            paused_count=paused_block_count,
        )

        # Track request metadata.
        request_metadata_types = inference_config.request_metadata_types
        if request_metadata_types is None:
            request_metadata_types = DynamicInferenceRequest.get_metadata_types()
        self.request_metadata_types = request_metadata_types

        # Initialize context state.
        self.params_dtype = model_config.params_dtype
        self.max_sequence_length = inference_config.max_sequence_length

        # Request and token counts.
        self.total_request_count = 0
        self.active_token_count = 0
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

        # Block ids.
        self.max_kv_block_count = math.ceil(self.max_sequence_length / self.block_size_tokens)

        # Set max_requests, max_tokens.
        if inference_config.max_requests is None:
            # Maximize compute utilization by defaulting to 1 block per request.
            self.max_requests = self.block_allocator.total_count - 1  # -1 for dummy block

            # Adjust max_requests for Mamba memory constraints if necessary
            if self.is_hybrid_model and mamba_max_requests < self.max_requests:
                self.max_requests = int(mamba_max_requests)

            self.max_requests = self.max_requests // tp_size * tp_size
            self.max_requests = self.max_requests // self.REQUEST_ROUNDER * self.REQUEST_ROUNDER
        else:
            # User can control request overflow via max_requests.
            self.max_requests = inference_config.max_requests

        assert (
            self.max_requests % tp_size == 0
        ), f"max_requests must be divisible by tp_size ({tp_size}), but got {self.max_requests}"

        self.max_tokens = inference_config.max_tokens or self.DEFAULT_MAX_TOKENS

        assert self.max_tokens >= self.max_requests, (
            f"max_tokens ({self.max_tokens}) must be >= "
            f"max_requests ({self.max_requests}), "
            "to have consistency between cuda graph sizes and the block table size."
        )

        # Attention metadata initialization (tensors are now handled by MHAMetadata classes)

        self.num_prefill_requests = 0
        self.graph_attn_metadata = {}
        self.non_graph_attn_metadata = {}
        self.active_attn_metadata = None
        self.is_creating_cuda_graphs = False

        self.graph_attn_metadata["mha_metadata"] = GraphedMHAMetadata(
            block_count_total=self.block_allocator.total_count,
            max_kv_block_count=self.max_kv_block_count,
            max_requests=self.max_requests,
            block_size_tokens=self.block_size_tokens,
            max_seqlen=self.max_sequence_length,
        )

        self.non_graph_attn_metadata["mha_metadata"] = NonGraphedMHAMetadata(
            block_count_total=self.block_allocator.total_count,
            max_kv_block_count=self.max_kv_block_count,
            max_requests=self.max_requests,
            block_size_tokens=self.block_size_tokens,
            max_seqlen=self.max_sequence_length,
        )

        self.moe_enable_routing_replay = model_config.moe_enable_routing_replay
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
                cuda_graph_max_tokens=self.max_requests,
                cuda_graph_mixed_prefill_count=inference_config.cuda_graph_mixed_prefill_count,
                max_requests=self.max_requests,
                max_tokens=self.max_tokens,
                max_sequence_length=self.max_sequence_length,
                use_cuda_graphs_for_non_decode_steps=self.use_cuda_graphs_for_non_decode_steps,
            )
        )

        # Whether to offload the KV cache. Determines where the KV cache is allocated within memory.
        self.offload_kv_cache = inference_config.offload_kv_cache
        assert not (
            self.offload_kv_cache and self.unified_memory_level
        ), "The KV cache should not be instantiated in unified memory when it is offloaded during training."

        self._using_cuda_graph_this_step = False
        # Deal with chunked prefill
        self.enable_chunked_prefill = inference_config.enable_chunked_prefill
        self.chunked_prefill_request_id = -1

        # FlashInfer.
        if inference_config.use_flashinfer_fused_rope is True:
            assert HAVE_FLASHINFER, "flashinfer is not installed"
        elif inference_config.use_flashinfer_fused_rope is None:
            inference_config.use_flashinfer_fused_rope = HAVE_FLASHINFER
        self.use_flashinfer_fused_rope = inference_config.use_flashinfer_fused_rope

        # Allocate GPU state.
        self.is_tensor_state_allocated = False
        self.is_symmetric_memory_initialized = False
        self.allocate_all_tensors(is_init=True)

        # Print info.
        logging.info(
            "DynamicInferenceContext: allocated context with active buffer size %s (%d blocks)."
            % (
                get_mem_size_str(self.block_allocator.active_count * self.block_size_bytes),
                self.block_allocator.active_count,
            )
        )

    def allocate_all_tensors(self, *, is_init: bool) -> None:
        """Allocate GPU state.

        This method is used for both 1) initial allocation, and 2) resuming the
        GPU state after a suspend.

        Args:
            is_init (bool): True if this is being called from `__init__()`.
        """

        # Only allocate tensors when not using unified memory at all (level 0),
        # or for initial allocation during `__init__()`. For levels 1 and 2, we do
        # not perform any explicit allocations or deallocations after the initial
        # call to `__init__()`.
        if self.unified_memory_level != 0 and not is_init:
            return

        # Mark allocated.
        if self.is_tensor_state_allocated:
            return
        self.is_tensor_state_allocated = True

        # Validate no tensors allocated prior to this method.
        for key in vars(self).keys():
            value = getattr(self, key)
            assert not isinstance(value, torch.Tensor), (
                "All tensors should be allocated within `allocate_all_tensors()."
                f"Please move tensor '{key}'."
            )

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
            for label, dtype, _ in self.request_metadata_types
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

        # Memory buffer.
        def allocate_memory_buffer():
            """Allocate the memory buffer. This function is called below within
            `with ctx_manager:`."""
            if self.cache_mla_latent:
                self.memory_buffer = torch.empty(
                    (
                        self.num_attention_layers,
                        self.block_allocator.total_count,
                        self.block_size_tokens,
                        self.kv_reduced_dim,
                    ),
                    dtype=self.params_dtype,
                    device=torch.cuda.current_device(),
                )
            else:
                ctx = (
                    torch_memory_saver.region(tag="kv_cache", enable_cpu_backup=True)
                    if HAVE_TORCH_MEMORY_SAVER and self.offload_kv_cache
                    else nullcontext()
                )

                with ctx:
                    self.memory_buffer = torch.empty(
                        (
                            2,  # key and value
                            self.num_attention_layers,
                            self.block_allocator.total_count,
                            self.block_size_tokens,
                            self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head,
                        ),
                        dtype=self.params_dtype,
                        device=torch.cuda.current_device(),
                    )

        # Optional state tensors for hybrid models
        def allocate_mamba_states():
            """Allocate Mamba states. This function is called below within
            `with ctx_manager:`."""
            if self.is_hybrid_model:
                self.mamba_metadata = MambaMetadata(
                    max_requests=self.max_requests, max_tokens=self.max_tokens
                )
                self.mamba_conv_states = torch.empty(
                    (self.num_mamba_layers, self.max_requests) + self.mamba_conv_states_shape,
                    dtype=self.params_dtype,
                    device=torch.cuda.current_device(),
                )
                self.mamba_ssm_states = torch.empty(
                    (self.num_mamba_layers, self.max_requests) + self.mamba_ssm_states_shape,
                    dtype=self.params_dtype,
                    device=torch.cuda.current_device(),
                )

            else:
                self.mamba_metadata = None

        # Allocate `ctx_manager`-managed buffers. (For currently unknown reasons,
        # `ctx_manager` can only be used once.)
        ctx_manager = (
            torch.cuda.use_mem_pool(self.unified_memory_mempool)
            if self.unified_memory_level > 0
            else nullcontext()
        )
        with ctx_manager:
            allocate_memory_buffer()
            allocate_mamba_states()

        # Reset attention and Mamba state.
        self.reset_attention_state()
        self.reset_mamba_state()

    def deallocate_all_tensors(self):
        """Deallocate GPU state.

        This method is used for suspending the dynamic engine.
        """

        # Only deallocate tensors when not using unified memory at all (level 0).
        # For levels 1 and 2, we do not perform any explicit allocations or
        # deallocations after the initial call to `__init__()`.
        if self.unified_memory_level != 0:
            return

        # Mark deallocated.
        if not self.is_tensor_state_allocated:
            return
        self.is_tensor_state_allocated = False

        # Delete all tensor attributes.
        # TODO(@lmcafee): check that device == 'cuda'?
        keys = list(vars(self).keys())
        for key in keys:
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
        assert self.active_attn_metadata is not None
        return (
            self.active_attn_metadata["mha_metadata"].state_data["cu_query_seq_lengths"],
            self.active_attn_metadata["mha_metadata"].state_data["max_seqlen_q"],
        )

    def cu_kv_lengths(self) -> Tuple[Tensor, Tensor, int]:
        """Cumulative key/value sequence lengths."""
        assert self.active_attn_metadata is not None
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

    def get_active_request_count(self):
        """Returns the current number of active requests."""
        return self.total_request_count - self.paused_request_count

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

        assert self.active_attn_metadata is not None

        if self.cache_mla_latent:
            return (
                self.memory_buffer[attention_layer_number],
                None,
                self.active_attn_metadata["mha_metadata"].state_data["block_table"],
            )
        else:
            return (
                self.memory_buffer[0, attention_layer_number],
                self.memory_buffer[1, attention_layer_number],
                self.active_attn_metadata["mha_metadata"].state_data["block_table"],
            )

    def mamba_states_cache(self, layer_number: int) -> Tuple[Tensor, Tensor]:
        """Returns the Mamba state tensors for the given layer."""
        assert self.is_hybrid_model, "Only hybrid models have Mamba state tensors"

        mamba_layer_number = self.layer_map[layer_number - 1]
        conv_state = self.mamba_conv_states[mamba_layer_number]
        ssm_state = self.mamba_ssm_states[mamba_layer_number]

        return (conv_state, ssm_state)

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
            chunk_length = req.remaining_prompt_length
            assert chunk_length > 0, "request without prompt tokens is not supported"
            lengths.append(chunk_length)
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
        self.request_output_lengths[request_slice] = lengths_tensor + tokens_to_generate_tensor
        self.request_kv_length_offsets[request_slice] = 0
        self.request_kv_block_counts[request_slice] = block_counts
        for i, (label, dtype, _) in enumerate(self.request_metadata_types):
            self.request_metadata[label][request_slice] = torch.tensor(
                metadata_cols[i], dtype=dtype, device=torch.cuda.current_device()
            )

        dummy_block_idx = self.block_allocator.dummy_block_idx
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
        prefill_tokens = graph_dimensions.token_count - graph_dimensions.decode_req_count

        # Pre-construct shared objects (safe due to deep copy in DynamicInferenceRequest.__post_init__)
        shared_sampling_params = SamplingParams(num_tokens_to_generate=1, termination_id=-1)
        shared_decode_tokens = torch.zeros(1, dtype=torch.long, device=torch.cuda.current_device())

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

    def initialize_attention_state(
        self, *, construct_graph_dimensions: Optional[InferenceBatchDimensions] = None
    ) -> None:
        """Initialize attention state so that every layer can use it.

        Args:
            construct_graph_dimensions (Optional[InferenceBatchDimensions]): The graph config to use for constructing the cuda graphs.
        Return:
            None.
        """
        self.is_creating_cuda_graphs = construct_graph_dimensions is not None

        # If in CUDA graph creation mode, add dummy requests for CUDA graph capture
        if self.is_creating_cuda_graphs:
            self.add_dummy_requests_for_cudagraph_capture(construct_graph_dimensions)

        batch_dimensions = InferenceBatchDimensions(
            token_count=self.active_token_count,
            prefill_req_count=self.num_prefill_requests,
            decode_req_count=self.num_decode_requests,
        )
        self.batch_dimensions = batch_dimensions
        best_graph = CUDAGraphBatchDimensionBuilder.match_graph_config(
            batch_dimensions,
            self.cuda_graph_batch_dimensions_list,
            strict=self.is_hybrid_model,
            decode_only_cuda_graphs=(not self.use_cuda_graphs_for_non_decode_steps),
            explicit_chunked_prefill=self.is_chunked_prefill_enabled() and self.is_hybrid_model,
            ep_group=self.expert_model_parallel_group,
        )
        self._using_cuda_graph_this_step = best_graph is not None

        if self.using_cuda_graph_this_step():
            self.padded_batch_dimensions = best_graph
        else:
            padded_token_count = self.round_up_tokens(self.active_token_count)
            if self.is_decode_only():
                padded_token_count = min(
                    self.max_tokens,
                    self.max_requests,
                    self.round_up_tokens(self.active_token_count),
                )
                padded_decode_req_count = padded_token_count
                padded_prefill_req_count = 0
            else:
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
        self.padding_slice = slice(self.active_token_count, self.padded_active_token_count)

        # Update token position indexes.
        self.token_to_block_idx[self.active_token_count : self.padded_active_token_count] = (
            self.block_allocator.dummy_block_idx
        )
        self.token_to_local_position_within_kv_block[
            self.active_token_count : self.padded_active_token_count
        ] = 0
        self.token_to_position_in_request[
            self.active_token_count : self.padded_active_token_count
        ] = 0

        self.active_attn_metadata = (
            self.graph_attn_metadata  # type: ignore[assignment]
            if self.using_cuda_graph_this_step()
            else self.non_graph_attn_metadata  # type: ignore[assignment]
        )

        # Update cu_query_seq_lengths, max_seqlen_q.
        active_slice = slice(self.paused_request_count, self.total_request_count)
        query_lengths_view = self.request_query_lengths[active_slice]
        request_kv_length_offsets_view = self.request_kv_length_offsets[active_slice]
        request_to_kv_block_ids_view = self.request_to_kv_block_ids[active_slice]

        attn_dimensions = batch_dimensions
        if self.using_cuda_graph_this_step():
            # Treat some decode requests as prefill requests to fit the cuda graph batch dimension.
            if batch_dimensions.decode_req_count > self.padded_batch_dimensions.decode_req_count:
                total_req = batch_dimensions.req_count
                adjusted_decode_req_count = self.padded_batch_dimensions.decode_req_count
                adjusted_prefill_req_count = total_req - adjusted_decode_req_count
                attn_dimensions = InferenceBatchDimensions(
                    token_count=batch_dimensions.token_count,
                    prefill_req_count=adjusted_prefill_req_count,
                    decode_req_count=adjusted_decode_req_count,
                )

        assert self.active_attn_metadata is not None
        self.active_attn_metadata["mha_metadata"].update(
            request_query_lengths=query_lengths_view,
            request_kv_length_offsets=request_kv_length_offsets_view,
            request_to_kv_block_ids=request_to_kv_block_ids_view,
            batch_dimensions=attn_dimensions,
            padded_batch_dimensions=self.padded_batch_dimensions,
        )

        if self.is_hybrid_model:
            active_mamba_indices_view = self.mamba_metadata.request_to_mamba_state_idx[active_slice]
            token_to_request_idx_view = self.token_to_request_idx[: self.active_token_count]
            cu_seqlens = self.active_attn_metadata["mha_metadata"].state_data[
                "cu_query_seq_lengths"
            ]
            self.mamba_metadata.update(
                active_mamba_indices_view,
                token_to_request_idx_view,
                cu_seqlens,
                batch_dimensions=attn_dimensions,
                padded_batch_dimensions=self.padded_batch_dimensions,
                enable_chunked_prefill=self.is_chunked_prefill_enabled(),
            )

        if self.moe_enable_routing_replay:
            if self.using_cuda_graph_this_step():
                self.moe_routing_metadata.enable_static_buffer_recording()
            else:
                self.moe_routing_metadata.disable_static_buffer_recording()

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
        self.batch_dimensions = InferenceBatchDimensions(
            token_count=0, prefill_req_count=0, decode_req_count=0
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
        self.request_to_kv_block_ids.fill_(-1)

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

        # Reset available block count.
        self.reset_attention_state()
        self.reset_mamba_state()
        self.block_allocator.reset()
        self.request_to_kv_block_ids.fill_(-1)

        # Reset chunked prefill state
        self.chunked_prefill_request_id = -1
        self.num_prefill_requests = 0
        self._using_cuda_graph_this_step = False
        self.is_creating_cuda_graphs = False
        self.padded_batch_dimensions = InferenceBatchDimensions(
            token_count=0, prefill_req_count=0, decode_req_count=0
        )

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
        assert logits.size(0) == 1, f"logits.size(0) ({tuple(logits.shape)}) != 1"
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

    def check_availability(self, req: DynamicInferenceRequest) -> Tuple[bool, bool, bool]:
        """
        Check if the request can be added to the context.
        """
        # Note that for hybrid models checking the total request count is sufficient
        # because we allocate a single set of Mamba state tensors for each request
        request_can_be_added = (
            self.total_request_count < self.max_requests and self.paused_request_count == 0
        )

        request_tokens_can_be_added = (
            self.active_token_count + req.remaining_prompt_length <= self.max_tokens
        )
        blocks = math.ceil(
            (req.remaining_prompt_length + req.finished_chunk_token_count) / self.block_size_tokens
        ) - math.ceil(req.finished_chunk_token_count / self.block_size_tokens)
        kv_cache_available = self.block_allocator.is_memory_available(blocks)
        return request_can_be_added, request_tokens_can_be_added, kv_cache_available

    def add_request(self, req: DynamicInferenceRequest, chunk_length: Optional[int] = None) -> None:
        """Add request to context. At this stage, we assume that the request is valid and can be added, as the checks are done in the schedule function.

        Args:
            req (DynamicInferenceRequest): Request to add.
            chunk_length (Optional[int]): Length of chunk to add. If None, the request will be fully added.

        Return:
            None
        """

        # If tensor state is deallocated, do not add request.
        if not self.is_tensor_state_allocated:
            raise TensorStateDeallocatedError(req.request_id)

        # Chunk length.
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
            new_block_ids = self.block_allocator.allocate_memory_blocks(num_blocks_needed)
            if new_block_ids is None or len(new_block_ids) != num_blocks_needed:
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

        # Handle request metadata.
        assert (
            req.get_metadata_types() == self.request_metadata_types
        ), "Request added to context with invalid metadata types"
        metadata = req.tracked_metadata
        metadata_types = req.get_metadata_types()
        for m, m_type in zip(metadata, metadata_types):
            label, _, _ = m_type
            if not isinstance(m, torch.Tensor):
                m = torch.as_tensor(
                    m,
                    device=self.request_metadata[label].device,
                    dtype=self.request_metadata[label].dtype,
                )

            self.request_metadata[label][current_id] = m

        # Handle length and block assignments.
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

        if self.is_hybrid_model and not is_chunked_prefill:
            # Allocate a slot for Mamba states
            mamba_idx = self.mamba_metadata.allocate_slot()
            if mamba_idx is None:
                raise ContextOverflowError(req.request_id, "No Mamba slots available")

            # Initialize the allocated Mamba state
            self.mamba_conv_states[:, mamba_idx] = 0.0
            self.mamba_ssm_states[:, mamba_idx] = 0.0
            self.mamba_metadata.request_to_mamba_state_idx[self.total_request_count] = mamba_idx

        self.active_token_count += chunk_length
        self.total_request_count += 0 if req.finished_chunk_token_count > 0 else 1
        self.num_prefill_requests += 1

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

        for metadata_tensor in self.request_metadata.values():
            metadata_tensor[dst_idxs] = metadata_tensor[src_idxs]

        if self.is_hybrid_model:
            self.mamba_metadata.request_to_mamba_state_idx[dst_idxs] = (
                self.mamba_metadata.request_to_mamba_state_idx[src_idxs]
            )

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

        for metadata_tensor in self.request_metadata.values():
            tensor_swap(metadata_tensor, src_idxs, dst_idxs)

        if self.is_hybrid_model:
            tensor_swap(self.mamba_metadata.request_to_mamba_state_idx, src_idxs, dst_idxs)

    def get_index_of_chunked_prefill_request(self) -> int:
        """Get the index of the chunked prefill request in the context.

        Return:
            (int) Index of the chunked prefill request, or -1 if none exists.
        """
        return torch.where(self.request_ids == self.chunked_prefill_request_id)[0][0]

    def is_chunked_prefill_enabled(self) -> bool:
        """Returns whether chunked prefill is enabled."""
        if self.is_hybrid_model:
            return self.enable_chunked_prefill and not self.is_creating_cuda_graphs
        return self.enable_chunked_prefill

    def release_memory_blocks_from_request_indexes(self, request_indexes) -> None:
        """Release memory blocks used by the given request idxs.

        Args:
            request_indexes (torch.Tensor): Request indexes. (*Note*, NOT request
                ids.)
        """
        kv_blocks_assigned = self.request_to_kv_block_ids[request_indexes]
        non_zero_values_in_kv_memory = kv_blocks_assigned[kv_blocks_assigned != -1]
        self.block_allocator.release_memory_blocks(non_zero_values_in_kv_memory)

        # Reset the KV blocks for finished requests.
        # Note: do not use fill_() (or add_() and similar inplace ops) here.
        # The combinition of indexing with a tensor (like finished_idxs) and
        # fill_()/add_() creates a clone and updates it instead of the original
        # tensor.
        self.request_to_kv_block_ids[request_indexes] = -1

        # Free Mamba slots.
        if self.is_hybrid_model:
            self.mamba_metadata.free_slots(request_indexes)

    def resume_paused_requests(
        self,
        active_request_count: int,
        newly_paused_request_ids: torch.Tensor,
        next_tokens: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        """Resume as many paused requests as we have space for in the active buffer.

        Args:
            active_request_count (int): Number of active requests.
            newly_paused_request_ids (torch.Tensor): List of newly paused request ids.
            next_tokens (torch.Tensor): Sampled tokens.

        Returns:
            (tuple[int, torch.Tensor]) active_request_count, newly_paused_request_ids.
        """

        # Assign released blocks to paused requests.
        # todo: @shanmugamr, un-pause requests using FIFO, rather than LIFO.
        resume_request_count = 0
        if self.paused_request_count > 0:
            active_block_count_avail = self.block_allocator.get_active_avail()
            paused_block_counts = self.request_kv_block_counts[: self.paused_request_count]
            # Flip counts before cumsum, since paused requests are resumed from
            # the right-most index, so we must count resumed blocks starting from
            # the right side.
            paused_block_counts = paused_block_counts.flip(dims=[0])
            # Add +1 to all block counts, since any time a paused request is
            # resumed, it will be starting a new memory block. For background,
            # pausing happens after a request has generated the final token of a
            # memory block (i.e., token 256 of that block), which means the very
            # next token (whenever that request gets unpaused) will be in a new
            # block. So, when we resume a paused request, we have to account for
            # the fact that it will need an extra block beyond the ones that it
            # has already used.
            paused_block_counts += 1  # +1 for newly added block
            paused_block_counts_cumsum = paused_block_counts.cumsum(dim=0)
            resume_request_count = min(
                torch.nonzero(paused_block_counts_cumsum <= active_block_count_avail).numel(),
                self.block_allocator.total_avail,
            )

        self.paused_request_count -= resume_request_count
        active_request_count += resume_request_count

        # Resume requests by assigning blocks and updating bookkeeping tensors.
        if resume_request_count > 0:
            assert torch.all(
                self.request_last_kv_block_offset[
                    self.paused_request_count : (self.paused_request_count + resume_request_count)
                ]
                == self.block_size_tokens - 1
            ), "The request_last_kv_block_offset should be 0 for the requests that just got resumed this step."

            assert resume_request_count <= self.block_allocator.total_avail
            block_ids = self.block_allocator.allocate_memory_blocks(resume_request_count)
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

        # Remove resumed requests from newly_paused_request_ids. We do this by
        # truncating the end of newly_paused_request_ids, which works because we
        # resume requests in LIFO order. If resume_request_count >
        # len(newly_paused_request_ids), this means that none of the paused
        # requests are newly paused during this update.
        if newly_paused_request_ids is not None and resume_request_count > 0:
            newly_paused_request_ids = newly_paused_request_ids[:-resume_request_count]

        return active_request_count, newly_paused_request_ids

    def evict_overflow_paused_requests(
        self, active_request_count: int, next_tokens: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Evict requests that overflow the paused buffer.

        Args:
            active_request_count (int): Number of active requests.
            next_tokens (torch.Tensor): Sampled tokens.

        Returns:
            (torch.Tensor) Evicted request ids.
        """

        # Overflow paused block count.
        overflow_paused_block_count = (
            self.block_allocator.get_paused_used() - self.block_allocator.paused_count
        )

        # Nothing to evict?
        if overflow_paused_block_count <= 0:
            return None

        # Overflow paused block count.
        paused_block_counts = self.request_kv_block_counts[: self.paused_request_count]
        paused_block_counts_cumsum = paused_block_counts.cumsum(dim=0)
        valid_paused_request_count = torch.nonzero(
            paused_block_counts_cumsum <= self.block_allocator.paused_count
        ).numel()
        overflow_paused_request_count = self.paused_request_count - valid_paused_request_count

        # Nothing to evict? (Similar to checking overflow_paused_block_count
        # above, but here we allow up to one paused request to overflow into the
        # active buffer.
        if overflow_paused_request_count == 0:
            return None

        # Evict request count. (Flip paused_block_counts because evictions are
        # counted from the right-most paused requests.
        paused_block_counts = paused_block_counts[-overflow_paused_request_count:].flip(dims=[0])
        paused_block_counts_cumsum = paused_block_counts.cumsum(dim=0)
        remaining_paused_request_counts = torch.arange(
            overflow_paused_request_count - 1,
            -1,
            -1,
            dtype=paused_block_counts_cumsum.dtype,
            device=torch.cuda.current_device(),
        )
        net_block_counts = paused_block_counts_cumsum - remaining_paused_request_counts
        evict_request_count = torch.nonzero(net_block_counts >= 0)[0].item() + 1

        # Eviction index range.
        evict_start_idx = self.paused_request_count - evict_request_count
        evict_end_idx = self.paused_request_count
        evict_request_idxs = torch.arange(
            evict_start_idx, evict_end_idx, device=torch.cuda.current_device()
        )
        evict_request_ids = self.request_ids[evict_start_idx:evict_end_idx].clone()

        # Release memory.
        self.release_memory_blocks_from_request_indexes(evict_request_idxs)

        # Move evicted requests to the right of active requests, while minimizing
        # movement.
        if evict_request_count < active_request_count:
            # Swap all evicted requests with right-most active requests.
            src_idxs = torch.arange(
                self.paused_request_count - evict_request_count,
                self.paused_request_count,
                device=torch.cuda.current_device(),
            )
            dst_idxs = torch.arange(
                self.total_request_count - evict_request_count,
                self.total_request_count,
                device=torch.cuda.current_device(),
            )
        else:
            # Swap all active requests with left-most evicted requests.
            src_idxs = torch.arange(
                self.paused_request_count - evict_request_count,
                self.paused_request_count - evict_request_count + active_request_count,
                device=torch.cuda.current_device(),
            )
            dst_idxs = torch.arange(
                self.paused_request_count,
                self.paused_request_count + active_request_count,
                device=torch.cuda.current_device(),
            )

        # Swap evicted and active requests.
        self._swap_book_keeping_tensors(
            src_idxs=src_idxs, dst_idxs=dst_idxs, next_tokens=next_tokens
        )

        # Update tracking vars.
        self.paused_request_count -= evict_request_count
        self.total_request_count -= evict_request_count

        # Reset unused block ids.
        evict_slice = slice(
            self.total_request_count, self.total_request_count + evict_request_count
        )
        self.request_to_kv_block_ids[evict_slice] = -1
        if self.is_hybrid_model:
            self.mamba_metadata.request_to_mamba_state_idx[evict_slice] = -1

        return evict_request_ids

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
        4. For the finished requests we release memory blocks and move them to the right
        5. We identify requests that require a new block and add them to the paused requests (i.e move them left)
        6. Resume paused requests & evict overflowing paused requests.
        7. We make changes to the request book keeping tesnsors and setup the tokens for next iteration
        8. We make relevant changes to the token bookkeeping tensors

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

        active_request_count = (active_requests_mask == 1).sum().item()
        finished_request_count = (active_requests_mask == 0).sum().item()
        assert (
            active_request_count + finished_request_count + self.paused_request_count
            == self.total_request_count
        )

        # Reset attention state.
        self.reset_attention_state()

        # Update total_request_count.
        self.total_request_count = active_request_count + self.paused_request_count

        # 2. If no paused requests are present and no active requests we release memory and reset.
        if active_request_count + self.paused_request_count == 0:
            if finished_request_count > 0:
                finished_idxs = (
                    torch.nonzero(active_requests_mask == 0, as_tuple=True)[0]
                    + self.paused_request_count
                )
                self.release_memory_blocks_from_request_indexes(finished_idxs)

            # Reset request/token counts.
            self.request_to_kv_block_ids.fill_(-1)
            self.total_request_count = 0
            self.active_token_count = 0

            # Reset Mamba state.
            self.reset_mamba_state()
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
            self.release_memory_blocks_from_request_indexes(finished_idxs)

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
                self.request_to_kv_block_ids[active_idxs_on_right] = -1
                if self.is_hybrid_model:
                    self.mamba_metadata.request_to_mamba_state_idx[active_idxs_on_right] = -1

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
                active_requests_requiring_new_block[
                    self.get_index_of_chunked_prefill_request() - self.paused_request_count
                ] = 0  # chunked prefill should not be paused

            active_requests_requiring_new_block_count = (
                (active_requests_requiring_new_block == 1).sum().item()
            )

            if active_requests_requiring_new_block_count > 0:
                newly_paused_request_ids = self.request_ids[
                    torch.nonzero(active_requests_requiring_new_block) + self.paused_request_count
                ]

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

            self.paused_request_count += active_requests_requiring_new_block_count
            active_request_count -= active_requests_requiring_new_block_count

        # 6. Now that we have the requests in following order [Paused, Active, Finished]
        # We determine how many requests we can resume and resume them

        # 6.a. First, resume temporarily paused requests.
        active_request_count, newly_paused_request_ids = self.resume_paused_requests(
            active_request_count, newly_paused_request_ids, next_tokens
        )

        # 6.b. Evict requests that overflow the paused buffer.
        evict_request_ids = self.evict_overflow_paused_requests(active_request_count, next_tokens)

        # 6.c. Resume any additional requests.
        active_request_count, newly_paused_request_ids = self.resume_paused_requests(
            active_request_count, newly_paused_request_ids, next_tokens
        )

        assert active_request_count > 0, "active_request_count == %d." % active_request_count

        # 6.d. Swap the chunked prefill request to the end of the active requests
        # to obey the invariance.
        if self.chunked_prefill_request_id != -1:
            self._swap_book_keeping_tensors(
                src_idxs=torch.tensor([self.get_index_of_chunked_prefill_request()]),
                dst_idxs=torch.tensor([self.total_request_count - 1]),
                next_tokens=next_tokens,
            )

        # 7. We make changes to the request book keeping tesnsors and setup the tokens for next iteration
        assert self.total_request_count == active_request_count + self.paused_request_count

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

        self.request_last_kv_block_offset[self.paused_request_count : self.total_request_count] = (
            self.request_last_kv_block_offset[self.paused_request_count : self.total_request_count]
            + 1
        ) % self.block_size_tokens

        # 8. We make relevant changes to the token bookkeeping tensors
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
            same order as the active requests (from paused_request_count to total_request_count).
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
        # Total usable blocks exclude the reserved dummy block.
        total_blocks = max(self.block_allocator.total_count - 1, 1)
        block_count_avail = int(self.block_allocator.total_avail)

        # Overall allocated blocks in the buffer right now.
        allocated_blocks = (self.block_allocator.total_count - 1) - block_count_avail
        allocated_blocks = int(max(0, allocated_blocks))

        # Active unique blocks referenced by current active requests only.
        active_start = self.paused_request_count
        active_end = self.total_request_count
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

    def maybe_initialize_symmetric_memory(self):
        """
        Initializes symmetric memory for inference, if not already initialized
        """
        if not self.is_symmetric_memory_initialized:
            parallel_state._set_global_symmetric_memory_buffer()
            self.is_symmetric_memory_initialized = True
