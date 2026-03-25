# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_attr_wrapped_model


@dataclass
class MambaInferenceStateConfig:
    """
    Config for initializing Mamba model inference state tensors.

    Note that we maintain separate metadata for decode, regular prefill, and
    chunked prefill requests because the Mamba kernels do not yet support mixing
    these. Once the kernels have been updated we can simplify this code.
    """

    layer_type_list: List[str]
    """
    A list of strings that indicates the layer type (Mamba / Attention / MLP) for each layer.
    See `megatron/core/ssm/mamba_hybrid_layer_allocation.py` for the list of symbols.
    """

    conv_states_shape: Tuple[int]
    """Mamba conv states shape per request."""

    ssm_states_shape: Tuple[int]
    """Mamba SSM states shape per request."""

    conv_states_dtype: torch.dtype
    """The dtype to use for the Mamba conv state tensor. Defaults to the model dtype."""

    ssm_states_dtype: torch.dtype
    """The dtype to use for the Mamba SSM state tensor. Defaults to the model dtype."""

    mamba_chunk_size: int = 128
    """The chunk size used by the Mamba SSM Triton kernels."""

    @classmethod
    def from_model(
        cls,
        model: MegatronModule,
        conv_states_dtype: Optional[torch.dtype] = None,
        ssm_states_dtype: Optional[torch.dtype] = None,
    ) -> Optional["MambaInferenceStateConfig"]:
        """Returns Mamba inference state config from the model if it is a hybrid model."""
        from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols

        decoder = get_attr_wrapped_model(model, "decoder")
        layer_type_list = getattr(decoder, "layer_type_list", None)
        if layer_type_list is not None and Symbols.MAMBA in layer_type_list:
            (mamba_conv_states_shape, mamba_ssm_states_shape) = (
                decoder.mamba_state_shapes_per_request()
            )
            if conv_states_dtype is None:
                conv_states_dtype = model.config.params_dtype
            if ssm_states_dtype is None:
                ssm_states_dtype = model.config.params_dtype
            mamba_chunk_size = 128
            for layer_type, layer in zip(decoder.layer_type_list, decoder.layers):
                if layer_type == Symbols.MAMBA and hasattr(layer, 'mixer'):
                    mamba_chunk_size = layer.mixer.chunk_size
                    break
            return cls(
                layer_type_list=layer_type_list,
                conv_states_shape=mamba_conv_states_shape,
                ssm_states_shape=mamba_ssm_states_shape,
                conv_states_dtype=conv_states_dtype,
                ssm_states_dtype=ssm_states_dtype,
                mamba_chunk_size=mamba_chunk_size,
            )
        return None


class PrefixCachingEvictionPolicy(str, Enum):
    """Eviction policy for prefix caching blocks.

    Only applies when enable_prefix_caching is True.
    """

    REF_ZERO = "ref_zero"
    """Deregister blocks immediately when ref_count hits 0. No caching after release."""

    LRU = "lru"
    """Keep released blocks in hash table. Evict oldest ref=0 blocks when space is needed."""


class PrefixCachingCoordinatorPolicy(str, Enum):
    """Routing policy for the DP inference coordinator with prefix caching."""

    LONGEST_PREFIX = "longest_prefix"
    """Route to the rank with the longest consecutive prefix match."""

    FIRST_PREFIX_BLOCK = "first_prefix_block"
    """Route to the rank that has the first block hash cached. O(ranks) check."""

    ROUND_ROBIN = "round_robin"
    """Route requests to ranks in round-robin order, ignoring prefix affinity."""


class KVCacheManagementMode(str, Enum):
    """Mode for handling large tensors (KV cache, Mamba states) during suspend/resume."""

    PERSIST = "persist"
    """Do not deallocate and reallocate large tensors; keep them on GPU."""

    OFFLOAD = "offload"
    """Offload large tensors to CPU during deallocation; onload during allocation."""

    RECOMPUTE = "recompute"
    """Deallocate large tensors and recompute them from scratch during allocation."""


@dataclass
class InferenceConfig:
    """
    Config for inference.

    NOTE: Must remain mutually exclusive with the `TransformerConfig`.
    """

    # =================================
    # KV cache and Mamba states config
    # =================================
    block_size_tokens: int = 256
    """Size of KV cache block size."""

    buffer_size_gb: int = 20
    """
    Buffer size reserved on the GPU for the KV cache.
    If `unified_memory_level` >= 1, then CPU memory is additionally utilized, resulting in a total
    buffer size of `buffer_size_gb + paused_buffer_size_gb`.
    """

    paused_buffer_size_gb: Optional[int] = None
    """
    Portion of buffer reserved for paused requests. Active requests are paused when there are not
    enough active blocks available to continue generating a request. The total buffer size
    (active + paused) depends on `unified_memory_level` (uvm):
        - uvm 0: buffer_size_gb (paused buffer is inclusive)
        - uvm 1: buffer_size_gb + paused_buffer_size_gb
    """

    mamba_inference_state_config: Optional[MambaInferenceStateConfig] = None
    """The Mamba inference state config if the model is a hybrid model."""

    mamba_memory_ratio: Optional[float] = None
    """
    Percentage of memory buffer to allocate for Mamba states. If not specified, allocates Mamba
    state tensors for each KV cache block. Only used for hybrid models.
    """

    max_requests: Optional[int] = None
    """
    Max number of active requests to use for decode-only forward passes.
    This is primarily limited by the combination of `buffer_size_gb` and `max_sequence_length`.
    """

    max_tokens: Optional[int] = None
    """
    Max number of tokens to use for forward passes. This is primarily limited by prefill activation
    memory usage. (Defaults to 16384).
    """

    unified_memory_level: int = 0
    """
    Sets unified memory usage within the dynamic inference context.
    The levels are:
        0) no unified memory (default)
        1) allocate `memory_buffer` in unified memory.
    Eventually, additional levels will be included to control other tensors within the context.
    """

    kv_cache_management_mode: KVCacheManagementMode = KVCacheManagementMode.PERSIST
    """
    Mode used to determine how large tensors are handled by the allocate and deallocate methods.
    See `KVCacheManagementMode` for options.
    """

    # =================================
    # CUDA graph config
    # =================================
    num_cuda_graphs: Optional[int] = None
    """
    Maximum number of cuda graphs to capture, where the cuda graph batch sizes range from 1 to
    `max_requests`. Due to rounding, the actual number of cuda graphs may not equal this argument.
    """

    cuda_graph_mixed_prefill_count: Optional[int] = 16
    """ 
    The number of mixed prefill graphs to capture if mixed prefill/decode graphs are enabled.
    """

    use_cuda_graphs_for_non_decode_steps: bool = True
    """
    Whether to use CUDA graphs for non-decode steps.
    """

    static_kv_memory_pointers: bool = False
    """
    Whether the KV cache (and Mamba states) will reside at the same memory addresses
    after suspend/resume as before. When True, CUDA graphs that reference these buffers
    remain valid across suspend/resume cycles and do not need to be recaptured.
    Requires either UVM or `torch_memory_saver` when `kv_cache_management_mode` is not PERSIST.
    """

    # =================================
    # Model config
    # =================================
    max_sequence_length: int = 2560
    """Max possible sequence length (prompt + output) that will occur."""

    pg_collection: Optional[ProcessGroupCollection] = None
    """A `ProcessGroupCollection` for distributed execution."""

    use_flashinfer_fused_rope: Optional[bool] = False
    """
    If True, use flashinfer's fused rope implementation.
    If None, defaults to using flash-infer if available.
    """

    materialize_only_last_token_logits: bool = True
    """
    Whether to only materialize logits for the last token. This should be set to False
    if returning log probs.
    """

    # =================================
    # Engine config
    # =================================
    enable_chunked_prefill: bool = False
    """Whether to enable chunked prefill."""

    num_speculative_tokens: int = 0
    """The number of speculative tokens to generate for decode steps."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for KV cache block sharing."""

    prefix_caching_eviction_policy: PrefixCachingEvictionPolicy = (
        PrefixCachingEvictionPolicy.REF_ZERO
    )
    """Eviction policy for prefix caching blocks. See `PrefixCachingEvictionPolicy` for options.

    Only applies when enable_prefix_caching is True.
    """

    prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
        PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
    )
    """Routing policy for the DP inference coordinator. See
    `PrefixCachingCoordinatorPolicy` for options.

    Only applies when enable_prefix_caching is True and using a coordinator.
    """

    prefix_caching_mamba_gb: Optional[float] = None
    """GPU memory budget (in GB) for the Mamba state cache used by prefix caching
    on hybrid models. Each cache slot stores SSM and conv states for all Mamba layers
    at a single block boundary. When set, Mamba states at KV divergence and last-aligned
    block boundaries are cached and reused across requests with matching prefixes."""

    # =================================
    # Logging config
    # =================================
    track_paused_request_events: bool = False
    """
    Whether to track paused request events. If True, `add_event_pause()` is called on
    requests when they are paused during bookkeeping.
    """

    track_generated_token_events: bool = False
    """
    Whether to track per-token events with timestamps for each generated token.
    When enabled, each generated token creates a GENERATED_TOKEN event with a
    timestamp, useful for per-token latency analysis.
    """

    metrics_writer: Optional["WandbModule"] = None
    """Wandb module for writing metrics."""

    logging_step_interval: int = 0
    """
    The step interval at which to log inference metrics to wandb.
    Defaults to 0, which means no logging.
    """

    request_metadata_types: Optional[List[Tuple[str, torch.dtype, bool]]] = None
    """
    A list of the per-request metadata types to track. Each entry is a tuple
    consisting of the string label, the target dtype, and whether to store the data on GPU.
    """

    use_synchronous_zmq_collectives: bool = False
    """Whether to use synchronous ZMQ collectives for inference. If True, the
    all_reduce_max operation will be performed synchronously, which can help reduce
    performance variability for MoEs.
    """

    store_routing_indices_in_ray_block_store: bool = False
    """Whether to write MoE routing indices to a Ray-based distributed block store.
    Requires a DistributedBlockStore actor (from NeMo RL) running on the same node.
    When enabled, routing indices are written to the block store on request completion
    and a block_cache_key is returned in the response instead of inline data.
    """
