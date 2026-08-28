# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import List, Literal, Optional, Tuple

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_attr_wrapped_model


@dataclass
class MambaInferenceStateConfig:
    """
    Config for initializing recurrent mixer inference state tensors.

    Note that we maintain separate metadata for decode, regular prefill, and
    chunked prefill requests because the recurrent kernels do not yet support
    mixing these. Once the kernels have been updated we can simplify this code.
    """

    layer_type_list: List[str]
    """
    A list of strings that indicates the layer type (Mamba / GDN / Attention / MLP) for each layer.
    See `megatron/core/models/hybrid/hybrid_layer_allocation.py` for the list of symbols.
    """

    conv_states_shape: Tuple[int]
    """Recurrent mixer's conv state shape per request."""

    ssm_states_shape: Tuple[int]
    """Recurrent mixer state shape per request."""

    conv_states_dtype: torch.dtype
    """The dtype to use for the Mamba conv state tensor. Defaults to the model dtype."""

    ssm_states_dtype: torch.dtype
    """The dtype to use for Mamba SSM state. Batch-invariant mode requires FP32."""

    mamba_chunk_size: int = 128
    """The chunk size used by the Mamba SSM Triton kernels."""

    ssm_chunk_alignment: Optional[int] = None
    """Token quantum that a prefill chunk boundary must land on for the model's
    SSM mixers to see a clean chunk boundary. Defaults to `mamba_chunk_size`,
    which is correct for any Mamba-only model.

    This is the mixers' shared `ssm_inference_chunk_size`, which is not always
    their `chunk_size`: the forked Gated Delta Product prefill kernels run at a
    fixed 64 whatever `chunk_size` says. `from_model` asserts every SSM layer
    agrees rather than reconciling a mixed stack. Only the paths that genuinely
    require an aligned boundary consult it -- batch-invariant chunked prefill,
    which replays the partial tail at decode, and recurrent-state extraction for
    prefix caching, which can only snapshot at a chunk boundary. Ordinary
    chunked prefill splits anywhere, because each step re-chunks from its own
    slice start."""

    gdp_num_householder: int = 0
    """Number of Householder copies of the Gated Delta Product layers, or 0 if the
    model has none. Sizes the GDP chunk descriptors used by the forked prefill
    kernels, whose Householder-expanded token stream is this many times longer."""

    def __post_init__(self):
        if self.ssm_chunk_alignment is None:
            self.ssm_chunk_alignment = self.mamba_chunk_size

    @classmethod
    def from_model(
        cls,
        model: MegatronModule,
        conv_states_dtype: Optional[torch.dtype] = None,
        ssm_states_dtype: Optional[torch.dtype] = None,
    ) -> Optional["MambaInferenceStateConfig"]:
        """Return recurrent inference state config for a Mamba or GDN hybrid model."""
        from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
        from megatron.core.ssm.ssm_inference import ssm_chunking

        decoder = get_attr_wrapped_model(model, "decoder")
        layer_type_list = getattr(decoder, "layer_type_list", None)
        recurrent_symbols = (Symbols.MAMBA, Symbols.GDN)
        if layer_type_list is not None and any(
            symbol in layer_type_list for symbol in recurrent_symbols
        ):
            present_recurrent_symbols = {
                symbol for symbol in recurrent_symbols if symbol in layer_type_list
            }
            if len(present_recurrent_symbols) > 1:
                raise ValueError(
                    "Dynamic inference does not support mixing Mamba and GDN layers; "
                    "the recurrent-state cache and prefill metadata use one shared shape "
                    "and chunk size."
                )
            if (
                Symbols.GDN in present_recurrent_symbols
                and model.config.experimental_attention_variant == "gdn2"
            ):
                raise NotImplementedError("GDN2 does not support dynamic inference.")
            mamba_conv_states_shape, mamba_ssm_states_shape = (
                decoder.mamba_state_shapes_per_request()
            )
            if conv_states_dtype is None:
                conv_states_dtype = model.config.params_dtype
            if model.config.batch_invariant_mode:
                if ssm_states_dtype not in (None, torch.float32):
                    raise ValueError(
                        "batch_invariant_mode requires FP32 Mamba SSM states; "
                        f"got {ssm_states_dtype}."
                    )
                # State passing carries an unrounded FP32 boundary value across
                # chunks. Rounding the cache to BF16 changes the next transition.
                ssm_states_dtype = torch.float32
            elif ssm_states_dtype is None:
                ssm_states_dtype = model.config.params_dtype
            # `decoder.layers` is pipeline-local, so a stage holding no SSM
            # layer falls back to the Mamba defaults while a stage holding GDP
            # layers reports 64. Safe today because every consumer of a
            # disagreeing value is stage-local (bookkeeping-buffer sizing) or
            # gated off for GDP (batch-invariant chunk lengths, prefix-cache
            # extraction offsets). A future cross-rank consumer must reconcile
            # these across the PP group.
            chunking = ssm_chunking(decoder.layer_type_list, decoder.layers)
            if chunking is None:
                mamba_chunk_size = 128
                ssm_chunk_alignment = mamba_chunk_size
                gdp_num_householder = 0
            else:
                mamba_chunk_size = chunking.chunk_size
                ssm_chunk_alignment = chunking.inference_chunk_size
                gdp_num_householder = chunking.num_householder
            return cls(
                layer_type_list=layer_type_list,
                conv_states_shape=mamba_conv_states_shape,
                ssm_states_shape=mamba_ssm_states_shape,
                conv_states_dtype=conv_states_dtype,
                ssm_states_dtype=ssm_states_dtype,
                mamba_chunk_size=mamba_chunk_size,
                ssm_chunk_alignment=ssm_chunk_alignment,
                gdp_num_householder=gdp_num_householder,
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

    LOAD_BALANCED = "load_balanced"
    """Route to the rank with the fewest in-flight requests. Ignores prefix affinity."""


class MediaCacheCoordinatorPolicy(str, Enum):
    """Routing policy for the DP inference coordinator with media caching."""

    AFFINITY = "affinity"
    """Prefer ranks assigned the same media key when vision embeddings are cached."""

    LOAD_BALANCED = "load_balanced"
    """Ignore media affinity and route using prefix affinity and load."""


class KVCacheManagementMode(str, Enum):
    """Mode for handling large tensors (KV cache, Mamba states) during suspend/resume."""

    PERSIST = "persist"
    """Do not deallocate and reallocate large tensors; keep them on GPU."""

    OFFLOAD = "offload"
    """Offload large tensors to CPU during deallocation; onload during allocation."""

    RECOMPUTE = "recompute"
    """Deallocate large tensors and recompute them from scratch during allocation."""


class CudaGraphSizingDistribution(str, Enum):
    """How CUDA graph token-count sizes are spaced when generating the captured graphs.

    EXPONENTIAL (default) — token counts halve from `cuda_graph_max_tokens` down to `tp_size`,
    giving a log-spaced distribution. Bounded relative padding (~2x worst case) at every scale and
    `log2(max_tokens)` total graphs.

    LINEAR — Include size-1 and size-2 graphs where applicable, linear spacing up until 256, and
    sparser linear spacing past 256. e.g. `[1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)`.
    Higher graph density at the top end.
    """

    EXPONENTIAL = "exponential"
    LINEAR = "linear"


class AsyncScheduleMode(str, Enum):
    """Async scheduling mode for dynamic inference."""

    LEGACY = "legacy"
    """Resolve requests before preparing the next forward pass."""

    ASYNC = "async"
    """Overlap asynchronous scheduling phases by reordering them to prepare-before-resolve."""


@dataclass
class ImageProcessingConfig:
    """Configuration for converting raw images into model input tensors."""

    patch_dim: int
    dynamic_resolution: bool = False
    use_tiling: bool = False
    pixel_shuffle: bool = False
    spatial_merge_size: int = 1
    dynamic_resolution_min_patches: int = 1
    dynamic_resolution_max_patches: int = 128
    vision_model_type: str = "radio"
    pixel_mean: Optional[List[float]] = None
    pixel_std: Optional[List[float]] = None
    img_h: Optional[int] = None
    img_w: Optional[int] = None
    max_num_tiles: int = 1
    use_thumbnail: bool = False
    num_img_embeddings_per_tile: int = 0


@dataclass
class VideoProcessingConfig:
    """Configuration for decoding raw video bytes into model input tensors."""

    image_config: ImageProcessingConfig
    num_frames: int = 8
    temporal_patch_size: int = 1
    frame_manifest_magic: Optional[bytes] = None
    """Prefix for payloads encoded as ``magic + UTF-8 {"frame_paths": [...]}``."""
    video_maintain_aspect_ratio: bool = True


@dataclass(frozen=True)
class MediaPromptSpec:
    """Map one API media type to the model's prompt-token contract."""

    model_token: str = "<image>"
    prefix: str = ""
    suffix: str = ""
    input_marker: Optional[str] = None


@dataclass(frozen=True)
class MultimodalPromptConfig:
    """Prompt contracts used to lower structured image/video blocks."""

    image_spec: MediaPromptSpec = field(default_factory=MediaPromptSpec)
    video_spec: MediaPromptSpec = field(default_factory=MediaPromptSpec)

    def get_spec(self, modality: str) -> MediaPromptSpec:
        """Return the prompt specification for ``image`` or ``video``."""
        if modality == "image":
            return self.image_spec
        if modality == "video":
            return self.video_spec
        raise ValueError(f"Unsupported media modality: {modality!r}")

    @classmethod
    def from_dict(cls, value):
        """Build from image and video specs."""
        if not value:
            return cls()
        return cls(
            image_spec=MediaPromptSpec(**value.get("image_spec", {})),
            video_spec=MediaPromptSpec(**value.get("video_spec", {})),
        )


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
    On-GPU portion of the shared KV cache block pool.
    If `unified_memory_level` >= 1, then CPU memory is additionally utilized, resulting in a total
    buffer size of `buffer_size_gb + paused_buffer_size_gb`.
    """

    paused_buffer_size_gb: Optional[int] = None
    """
    Memory used to derive the paused-request block retention budget. This does not reserve blocks
    from active requests: active requests may use the entire shared pool of usable KV cache blocks.
    When the pool cannot satisfy new allocations, paused requests retain blocks only within this
    budget and excess paused requests may be evicted. The total buffer size depends on
    `unified_memory_level` (uvm):
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
    Maximum number of cuda graphs to capture.
    Graph token counts are spaced from 1 up to a per-graph-type budget:
      - Decode-only graphs are always bounded by `max_requests * (num_speculative_tokens + 1)`.
      - Prefill/mixed graphs are bounded by `cuda_graph_max_tokens` by default,
        or extend up to `max_tokens` when `cuda_graph_all_prefills` is set.
    Due to rounding, the actual number of cuda graphs may not equal this argument.
    """

    cuda_graph_mixed_prefill_count: Optional[int] = 16
    """
    The number of mixed prefill graphs to capture if mixed prefill/decode graphs are enabled.
    """

    cuda_graph_sizing_distribution: CudaGraphSizingDistribution = (
        CudaGraphSizingDistribution.EXPONENTIAL
    )
    """
    How CUDA graph token counts are spaced. EXPONENTIAL (default) halves from
    `cuda_graph_max_tokens` down to `tp_size` (log-spaced, ~log2(max_tokens) graphs).
    LINEAR uses a range of linear strides (includes small graphs + mid-range linearity + 
    a bigger step size at the top end).
    """

    use_cuda_graphs_for_non_decode_steps: bool = True
    """
    Whether to use CUDA graphs for non-decode steps.
    """

    cuda_graph_all_prefills: bool = False
    """
    Whether prefill/mixed CUDA graphs should span up to `max_tokens`.
    When False (default), prefill/mixed graphs are bounded by `cuda_graph_max_tokens`.
    When True, prefill/mixed graph capture is extended to cover the full `max_tokens` budget.
    """

    cuda_graph_max_tokens: int = 512
    """
    Token ceiling for the largest captured prefill/mixed CUDA graph.
    This is a raw token count (not scaled by speculative decoding). The effective ceiling is
    clamped to `[max_requests * (num_speculative_tokens + 1), max_tokens]` so it never falls
    below the decode bound nor exceeds the token budget. Ignored when `cuda_graph_all_prefills`
    is set, which extends capture to the full `max_tokens`.
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

    image_preprocessing_config: Optional[ImageProcessingConfig] = None
    """Configuration for preprocessing raw image payloads."""

    video_preprocessing_config: Optional[VideoProcessingConfig] = None
    """Configuration for decoding and preprocessing raw video payloads."""

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

    vision_embedding_cache_max_bytes: int = 0
    """Maximum GPU bytes retained for reusable vision embeddings.

    A value of zero disables the cache. Cache entries use an automatically
    generated media-content key and, unless ``allow_stale_multimodal_embeddings``
    is enabled, are discarded whenever the inference engine is suspended or its
    generation epoch changes.
    """

    prefix_caching_eviction_policy: PrefixCachingEvictionPolicy = (
        PrefixCachingEvictionPolicy.REF_ZERO
    )
    """Eviction policy for prefix caching blocks. See `PrefixCachingEvictionPolicy` for options.

    Only applies when enable_prefix_caching is True.
    """

    prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
        PrefixCachingCoordinatorPolicy.LOAD_BALANCED
    )
    """Routing policy for the DP inference coordinator. See
    `PrefixCachingCoordinatorPolicy` for options.

    Only applies when enable_prefix_caching is True and using a coordinator.
    """

    prefix_caching_routing_alpha: float = 0.5
    """Weight for prefix-aware scoring: score = alpha * match + (1 - alpha) * normalized_load.
    Higher alpha favors prefix cache hits; lower alpha favors load balance.
    Must be in [0, 1]. Only applies when enable_prefix_caching is True and using a coordinator.
    """

    media_cache_coordinator_policy: MediaCacheCoordinatorPolicy = (
        MediaCacheCoordinatorPolicy.AFFINITY
    )
    """Media-cache routing policy for the DP inference coordinator.

    Media affinity is active only when ``vision_embedding_cache_max_bytes`` is
    greater than zero. Media-salted prefix affinity is controlled separately by
    ``prefix_caching_coordinator_policy``.
    """

    media_cache_routing_weight: float = 1.0
    """Estimated vision-encoder reuse cost in compact-prompt block units.

    Multimodal coordinator routing combines this media-hit value with the number
    of matching routing-prefix blocks before blending cache affinity with load
    using ``prefix_caching_routing_alpha``. The engine independently uses
    post-expansion hashes for authoritative KV lookup. Must be non-negative.
    """

    prefix_caching_mamba_gb: Optional[float] = None
    """GPU memory budget (in GB) for the Mamba state cache used by prefix caching
    on hybrid models. Each cache slot stores SSM and conv states for all Mamba layers
    at a single block boundary. When set, Mamba states at KV divergence and last-aligned
    block boundaries are cached and reused across requests with matching prefixes.

    This budget covers both buffers allocated by MambaSlotAllocator: the durable cache
    (ssm_states/conv_states, max_slots slots reused across requests) and the per-step
    extraction scratch (intermediate_ssm_out/intermediate_conv_out). The scratch is
    sized to the tighter of two per-step bounds,
    ``min(ceil(max_tokens / block_size_tokens), 3 * max_requests)``, since a single
    engine step can extract at most one state per block_size_tokens of its token budget
    (and at most 3 per request). The scratch is reserved from this budget first, so a
    smaller ``max_tokens`` (or ``max_requests``) shrinks the scratch and leaves more
    durable cache slots."""

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

    sampling_backend: Literal['torch', 'flashinfer'] = 'torch'
    """Which sampling kernels to use during inference. Falls back to "torch" with a warning if
    "flashinfer" is requested but the package is not installed."""

    offset_sampling_seed_by_dp_rank: bool = True
    """
    If True, offset `inference_sampling_seed` by the data-parallel rank when seeding the
    sampling RNG. This gives each DP rank a unique generation seed so that the same prompt
    routed to different ranks produces different samples (important for RL training).
    If False (or `ModelParallelConfig.deterministic_mode` / `--deterministic-mode` is
    enabled), then all DP ranks share the same sampling / generation seed.
    """

    async_sched_mode: AsyncScheduleMode = AsyncScheduleMode.LEGACY
    """Mode used to schedule dynamic batching inference work."""

    logprobs_mode: Literal['raw_logprobs', 'processed_logprobs'] = 'raw_logprobs'
    """Whether returned log-probs are modified by the sampling parameters or not."""

    request_metadata_types: Optional[List[Tuple[str, torch.dtype]]] = None
    """
    A list of the per-request metadata types to track. Each entry is a tuple
    consisting of the string label and the target dtype.
    """

    use_synchronous_zmq_collectives: bool = False
    """Whether to use synchronous ZMQ collectives for inference. If True, the
    all_reduce_max operation will be performed synchronously, which can help reduce
    performance variability for MoEs.
    """

    disable_ep_consensus: bool = False
    """If True, the engine skips the EP-group consensus all-reduce in
    `run_engine_with_coordinator` and decides whether to step based on local
    state alone. The rank still calls `controller.dummy_forward()` whenever
    `local_pending == 0`, so EP collectives (NCCL all-to-all, etc.) stay in
    sync — without this, a peer running a real forward would deadlock waiting
    on this rank's all-to-all participation. Trades off the consensus
    all-reduce CPU cost for unconditional dummy_forwards on idle ranks.
    """

    ep_consensus_interval: int = 20
    """How many steps to skip between EP-consensus all-reduces when the engine
    has pending work. Consensus is always run immediately when there is no
    global work (to detect new arrivals quickly); this interval only applies
    to the busy case, where skipping avoids per-step all-reduce overhead.
    In the worst case, pausing is delayed by this many steps (~10–20 ms per
    step at typical decode throughput).
    """

    verbose: InitVar[bool] = False
    """Whether to log detailed context configuration at initialization.
    This is an InitVar and is not stored as a field on the config."""

    allow_stale_multimodal_embeddings: bool = False
    """Allow projected-media embeddings to survive weight-change boundaries.

    By default, suspend/resume and generation-epoch changes invalidate both the
    shared vision-embedding cache and request-local vision state. Enable this
    only when model weights are guaranteed not to change across those boundaries.
    """

    def __post_init__(self, verbose: bool):
        self._verbose = verbose
        self.async_sched_mode = AsyncScheduleMode(self.async_sched_mode)
        if not (0.0 <= self.prefix_caching_routing_alpha <= 1.0):
            raise ValueError(
                f"prefix_caching_routing_alpha must be in [0, 1], "
                f"got {self.prefix_caching_routing_alpha}"
            )
        if self.media_cache_routing_weight < 0:
            raise ValueError(
                "media_cache_routing_weight must be non-negative, "
                f"got {self.media_cache_routing_weight}"
            )

        if self.logprobs_mode not in ("raw_logprobs", "processed_logprobs"):
            raise ValueError(
                f"Unsupported logprobs_mode {self.logprobs_mode!r}. "
                "Supported modes: raw_logprobs, processed_logprobs."
            )

        # The speculative log-probs path does not yet apply processed-logprobs.
        if self.logprobs_mode == "processed_logprobs" and self.num_speculative_tokens > 0:
            raise ValueError(
                "logprobs_mode='processed_logprobs' is not yet supported with speculative decoding "
                "(num_speculative_tokens > 0)."
            )

        if self.sampling_backend == 'flashinfer':
            try:
                import flashinfer  # noqa: F401
            except ImportError:
                warnings.warn(
                    "sampling_backend='flashinfer' was requested but the flashinfer "
                    "package is not installed; falling back to sampling_backend='torch'."
                )
                self.sampling_backend = 'torch'
