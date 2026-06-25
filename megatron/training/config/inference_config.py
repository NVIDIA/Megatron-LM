# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Declarative configuration dataclass for Megatron inference entry points.

This module defines :class:`InferenceSetupConfig`, the inference counterpart to the
training-oriented config dataclasses (e.g. ``TrainingConfig``, ``OptimizerConfig``). It
holds the inference-specific knobs that today live as loose ``args.<attr>`` values
produced by ``_add_inference_args`` in ``megatron.training.arguments``. Field names mirror
the corresponding argparse ``dest`` names one-to-one, so an ``InferenceSetupConfig`` can be
built directly from an ``argparse.Namespace`` via ``_default_config_from_args``.

Layering note
-------------
``InferenceSetupConfig`` is the *declarative, serializable* layer (primitives/strings, safe
to YAML-serialize, built from args before the model or distributed groups exist). It is the
counterpart to ``megatron.training.models.GPTModelConfig``.

The *runtime engine* config consumed by the inference context/engine is
``megatron.core.inference.config.InferenceConfig`` -- it holds rich runtime objects
(``ProcessGroupCollection``, ``MambaInferenceStateConfig``, ``torch.dtype``, a wandb module)
and can only be built once the model and process groups exist.

Use :meth:`InferenceSetupConfig.to_inference_config` to produce the runtime engine config
from this declarative config plus the runtime artifacts. This mirrors the
``GPTModelConfig -> TransformerConfig`` relationship.
"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from megatron.core.inference.config import InferenceConfig
    from megatron.core.transformer.module import MegatronModule


@dataclass(kw_only=True)
class InferenceSetupConfig:
    """Declarative configuration settings for inference engines and the dynamic context.

    These fields correspond to the ``inference`` argument group defined by
    ``_add_inference_args`` in ``megatron/training/arguments.py``. They cover both
    the static and dynamic inference engines, the KV-cache memory buffer, CUDA graph
    capture during decode, prefix caching, and inference-time logging.

    This is the serializable, args-shaped layer. The runtime engine config consumed by
    the inference context/engine is ``megatron.core.inference.config.InferenceConfig``;
    build it via :meth:`to_inference_config`.
    """

    # ---------------- General inference settings ----------------

    inference_batch_times_seqlen_threshold: int = -1
    """If (batch-size * sequence-length) is smaller than this threshold then batches will not be
    split up for pipelining. Requires setting --pipeline-model-parallel-size > 1. Setting this to
    -1 indicates that batch pipelining is not used."""

    max_tokens_to_oom: int = 12000
    """Maximum number of tokens during inference (# in prompt + # to generate). Allows us to throw
    an error before OOM crashes server."""

    output_bert_embeddings: bool = False
    """Output Bert embeddings (via mean pooling) from model, rather than its binary head output or
    entire hidden batch."""

    bert_embedder_type: Literal["megatron", "huggingface"] = "megatron"
    """Select either Megatron or Huggingface as the Bert embedder."""

    cuda_graph_modules: list[str] = field(default_factory=list)
    """Selects capture coverage within per-layer CUDA graphs (local and transformer_engine
    implementations). An empty list means capturing the whole Transformer layer."""

    use_legacy_static_engine: bool = False
    """Use legacy static engine. (Current static engine uses dynamic engine under the hood.)"""

    inference_max_requests: int = 8
    """Maximum number of requests for inference."""

    inference_max_seq_length: int = 2560
    """Maximum sequence length expected for inference (prefill + decode)."""

    # ---------------- Dynamic batching ----------------

    inference_dynamic_batching: bool = False
    """Enable dynamic batching mode."""

    inference_dynamic_batching_buffer_size_gb: float = 40.0
    """Amount of on-GPU memory allocated for the KV cache. The total amount of memory allocated for
    the KV cache (CPU + GPU memory) depends on the value set for the unified virtual memory (UVM)
    level (via inference_dynamic_batching_unified_memory_level)."""

    inference_dynamic_batching_paused_buffer_size_gb: float | None = None
    """Amount of memory reserved for paused requests in the dynamic inference context. Active
    requests are paused when there are not enough active blocks available to continue generating a
    request."""

    inference_dynamic_batching_mamba_memory_ratio: float | None = None
    """Percentage of memory buffer to allocate for Mamba states. If not specified, allocates Mamba
    state tensors for each KV cache block. Only used for hybrid models."""

    inference_dynamic_batching_block_size: int = 256
    """KV cache block size. It should be a multiple of 256."""

    inference_dynamic_batching_max_requests: int | None = None
    """Override the inference context's `max_requests`. By default, `max_requests` is set to the
    number of blocks in the context's memory buffer."""

    inference_dynamic_batching_max_tokens: int | None = None
    """Override the inference context's default `max_tokens`."""

    inference_dynamic_batching_num_cuda_graphs: int = 16
    """Maximum number of cuda graphs to capture, where the cuda graph batch sizes range from 1 to
    `max_requests`. The user can also pass -1, in which case we automatically determine the number
    of graphs to capture based on the `max_requests`."""

    inference_dynamic_batching_track_paused_request_events: bool = False
    """Track paused request ids by adding 'paused' events to each request's event history. This has
    a very minor impact on latency."""

    inference_dynamic_batching_track_generated_token_events: bool = False
    """Track per-token events with timestamps for each generated token. When enabled, each generated
    token creates a GENERATED_TOKEN event with a timestamp, useful for per-token latency analysis."""

    inference_dynamic_batching_unified_memory_level: Literal[0, 1] = 0
    """Set unified memory usage within the dynamic inference context. The levels are: 0) no unified
    memory, 1) allocate `memory_buffer` in unified memory."""

    inference_dynamic_batching_cuda_graph_mixed_prefill_count: int = 16
    """Number of mixed prefill requests to capture in a cuda graph."""

    inference_dynamic_batching_cuda_graph_sizing_distribution: Literal["exponential", "linear"] = (
        "exponential"
    )
    """Spacing of CUDA graph token counts. "exponential" (default) halves from cuda_graph_max_tokens
    down to tp_size, giving a log-spaced distribution with bounded relative padding. "linear" uses
    varying linear strides across the range."""

    inference_dynamic_batching_sampling_backend: Literal["torch", "flashinfer"] = "torch"
    """Which sampling kernels to use during inference. Falls back to "torch" with a warning if
    "flashinfer" is requested but the package is not installed."""

    inference_dynamic_batching_async_sched_mode: Literal["legacy", "serial"] = "legacy"
    """Async scheduling mode for dynamic batching. "legacy" (default) preserves the
    existing resolve-before-prepare path. "serial" speculatively prepares and forwards decode-only
    greedy GPT steps before resolving finished requests."""

    inference_dynamic_batching_logprobs_mode: Literal["raw_logprobs", "processed_logprobs"] = (
        "raw_logprobs"
    )
    """How returned inference log-probs are computed engine-wide. "raw_logprobs" (default) uses the
    unmodified model logits; "processed_logprobs" uses temperature and filters by top-k/top-p."""

    # ---------------- CUDA graphs ----------------

    decode_only_cuda_graphs: bool = False
    """Only use cuda graphs for decode-only steps, not prefill and mixed steps."""

    inference_cuda_graph_all_prefills: bool = False
    """Extend prefill/mixed CUDA graph capture up to `max_tokens`. By default, all graphs are
    limited by the decode limit of `max_requests * (num_speculative_tokens + 1)`."""

    # ---------------- Chunked prefill / speculation ----------------

    enable_chunked_prefill: bool = False
    """Enable chunked prefill (disabled by default)."""

    num_speculative_tokens: int = 0
    """Number of speculative tokens generated during decode."""

    # ---------------- Prefix caching ----------------

    inference_dynamic_batching_enable_prefix_caching: bool = False
    """Enable/disable prefix caching for dynamic batching inference. When disabled, KV cache blocks
    cannot be shared between requests with identical prompt prefixes."""

    inference_dynamic_batching_prefix_caching_eviction_policy: Literal["ref_zero", "lru"] = "ref_zero"
    """Eviction policy for prefix caching blocks. "ref_zero" (default) immediately returns blocks to
    the free pool when ref_count hits 0. "lru" keeps blocks cached and evicts via LRU only when
    space is needed."""

    inference_dynamic_batching_prefix_caching_coordinator_policy: Literal[
        "longest_prefix", "first_prefix_block", "round_robin"
    ] = "first_prefix_block"
    """Coordinator routing policy for prefix caching. "first_prefix_block" (default) routes based on
    the first block hash only. "longest_prefix" routes to the rank with the longest matching prefix.
    "round_robin" ignores prefix affinity and cycles through ranks."""

    inference_dynamic_batching_prefix_caching_routing_alpha: float = 0.5
    """Weight for prefix-aware routing score: score = alpha * match + (1 - alpha) * normalized_load.
    Higher alpha favors prefix cache hits; lower alpha favors load balance."""

    inference_dynamic_batching_prefix_caching_mamba_gb: float | None = None
    """GPU memory budget (in GB) for the Mamba state cache used by prefix caching on hybrid models.
    When set, Mamba states at block boundaries are cached for reuse."""

    # ---------------- Logging ----------------

    inference_logging_step_interval: int = 0
    """Step interval for logging inference metrics. Default to 0 to disable inference logging."""

    inference_text_gen_server_logging: bool = False
    """Enable per-request logging in the inference text generation server."""

    inference_wandb_logging: bool = False
    """Enable inference wandb logging."""

    # ---------------- Coordinator / distributed ----------------

    inference_coordinator_port: int | None = None
    """This port will be used to setup the inference coordinator on node-0."""

    inference_use_synchronous_zmq_collectives: bool = False
    """Use synchronous ZMQ collectives for inference. Helps in reducing performance variability for
    MoEs."""

    inference_disable_ep_consensus: bool = False
    """Skip the EP-group consensus all-reduce in the inference engine control loop and step on local
    state only. Only safe when EP coordination is not required (e.g. ep_world_size == 1)."""

    # ---------------- Mamba inference state dtypes ----------------
    # NOTE: These are provided on the CLI as strings ("bf16"/"fp16"/"fp32") but are mapped to the
    # corresponding torch dtype during argument validation (see validate_args in arguments.py).

    mamba_inference_conv_states_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    """Dtype for the Mamba inference conv states tensor."""

    mamba_inference_ssm_states_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    """Dtype for the Mamba inference SSM states tensor."""

    # ---------------- Log-prob and RoPE knobs from _add_inference_args ----------------

    return_log_probs: bool = False
    """Return the log probabilities of the final output tokens. Mirrors ``--return-log-probs``.
    Controls ``materialize_only_last_token_logits`` (the engine must materialize all logits when
    log probs are requested, unless ``skip_prompt_log_probs`` is also True)."""

    skip_prompt_log_probs: bool = False
    """Skip prompt log probs. Mirrors ``--skip-prompt-log-probs``. When True, only the last
    token's logits are needed even if ``return_log_probs`` is True, so
    ``materialize_only_last_token_logits`` stays True."""

    use_flashinfer_fused_rope: bool = False
    """Use flashinfer's fused rope implementation. Mirrors ``--use-flashinfer-fused-rope``."""

    def to_inference_config(
        self,
        model: "MegatronModule",
        *,
        pg_collection: Any = None,
        kv_cache_management_mode: str = "persist",
        static_kv_memory_pointers: bool = False,
        enable_cuda_graphs: bool = True,
        metrics_writer: Any = None,
        verbose: bool = True,
    ) -> "InferenceConfig":
        """Build the runtime ``megatron.core.inference.config.InferenceConfig`` from this config.

        This is the bridge from the declarative inference settings to the runtime engine
        config consumed by the dynamic inference context/engine. It supplies the fields that
        depend on the built model (max sequence length, Mamba state config, process groups)
        and the cross-cutting values that do not live on this declarative config.

        Args:
            model: The (possibly wrapped) model to run inference with. Used to derive the
                effective max sequence length, the Mamba inference state config, and the
                process group collection when ``pg_collection`` is not provided.
            pg_collection: Process groups for distributed execution. Defaults to the
                model's ``pg_collection`` attribute when None.
            kv_cache_management_mode: How large tensors are handled on suspend/resume
                ("persist"/"offload"/"recompute"). Sourced from the RL arg
                ``rl_kv_cache_management_mode`` at the call site.
            static_kv_memory_pointers: Whether the KV cache stays at fixed addresses across
                suspend/resume. Sourced from the RL arg ``rl_persist_cuda_graphs`` (not part
                of the inference argument group).
            enable_cuda_graphs: When False, ``num_cuda_graphs`` is forced to None (no capture).
                Callers typically pass ``inference_cuda_graph_scope != none``; derived, not a
                 1:1 args field.
            metrics_writer: Optional wandb module for inference metric logging.
            verbose: Whether the context logs detailed configuration at initialization.

        Returns:
            A fully-populated runtime ``InferenceConfig``.
        """
        from megatron.core.inference.config import (
            AsyncScheduleMode,
            CudaGraphSizingDistribution,
            InferenceConfig,
            KVCacheManagementMode,
            MambaInferenceStateConfig,
            PrefixCachingCoordinatorPolicy,
            PrefixCachingEvictionPolicy,
        )
        from megatron.core.utils import get_attr_wrapped_model

        # Effective max sequence length depends on the model's position embedding type.
        position_embedding_type = get_attr_wrapped_model(model, "position_embedding_type")
        model_max_seq_len = get_attr_wrapped_model(model, "max_sequence_length")
        inf_max_seq_len = self.inference_max_seq_length
        max_batch_size = self.inference_dynamic_batching_max_requests

        if position_embedding_type == "learned_absolute":
            # The context's max_sequence_length must not exceed the model's, otherwise the
            # context's position_ids index past the position embedding table.
            if inf_max_seq_len:
                max_sequence_length = min(model_max_seq_len, inf_max_seq_len)
            else:
                max_sequence_length = model_max_seq_len
            assert max_batch_size is None or max_batch_size <= model_max_seq_len
        else:
            max_sequence_length = inf_max_seq_len
        if max_batch_size is not None:
            max_sequence_length = max(max_sequence_length, max_batch_size)

        mamba_inference_state_config = MambaInferenceStateConfig.from_model(
            model,
            conv_states_dtype=self.mamba_inference_conv_states_dtype,
            ssm_states_dtype=self.mamba_inference_ssm_states_dtype,
        )
        if pg_collection is None:
            pg_collection = get_attr_wrapped_model(model, "pg_collection")

        return InferenceConfig(
            verbose=verbose,
            block_size_tokens=self.inference_dynamic_batching_block_size,
            buffer_size_gb=self.inference_dynamic_batching_buffer_size_gb,
            paused_buffer_size_gb=self.inference_dynamic_batching_paused_buffer_size_gb,
            mamba_memory_ratio=self.inference_dynamic_batching_mamba_memory_ratio,
            num_cuda_graphs=(
                self.inference_dynamic_batching_num_cuda_graphs if enable_cuda_graphs else None
            ),
            max_requests=self.inference_dynamic_batching_max_requests,
            max_tokens=self.inference_dynamic_batching_max_tokens,
            unified_memory_level=self.inference_dynamic_batching_unified_memory_level,
            kv_cache_management_mode=KVCacheManagementMode(kv_cache_management_mode),
            cuda_graph_mixed_prefill_count=(
                self.inference_dynamic_batching_cuda_graph_mixed_prefill_count
            ),
            cuda_graph_sizing_distribution=CudaGraphSizingDistribution(
                self.inference_dynamic_batching_cuda_graph_sizing_distribution
            ),
            use_cuda_graphs_for_non_decode_steps=not self.decode_only_cuda_graphs,
            cuda_graph_all_prefills=self.inference_cuda_graph_all_prefills,
            static_kv_memory_pointers=static_kv_memory_pointers,
            max_sequence_length=max_sequence_length,
            mamba_inference_state_config=mamba_inference_state_config,
            pg_collection=pg_collection,
            use_flashinfer_fused_rope=self.use_flashinfer_fused_rope,
            materialize_only_last_token_logits=(
                not (self.return_log_probs and not self.skip_prompt_log_probs)
            ),
            track_generated_token_events=(
                self.inference_dynamic_batching_track_generated_token_events
            ),
            track_paused_request_events=self.inference_dynamic_batching_track_paused_request_events,
            enable_chunked_prefill=self.enable_chunked_prefill,
            enable_prefix_caching=self.inference_dynamic_batching_enable_prefix_caching,
            prefix_caching_eviction_policy=PrefixCachingEvictionPolicy(
                self.inference_dynamic_batching_prefix_caching_eviction_policy
            ),
            prefix_caching_coordinator_policy=PrefixCachingCoordinatorPolicy(
                self.inference_dynamic_batching_prefix_caching_coordinator_policy
            ),
            prefix_caching_routing_alpha=self.inference_dynamic_batching_prefix_caching_routing_alpha,
            prefix_caching_mamba_gb=self.inference_dynamic_batching_prefix_caching_mamba_gb,
            metrics_writer=metrics_writer,
            logging_step_interval=self.inference_logging_step_interval,
            num_speculative_tokens=self.num_speculative_tokens,
            use_synchronous_zmq_collectives=self.inference_use_synchronous_zmq_collectives,
            disable_ep_consensus=self.inference_disable_ep_consensus,
            sampling_backend=self.inference_dynamic_batching_sampling_backend,
            async_sched_mode=AsyncScheduleMode(
                self.inference_dynamic_batching_async_sched_mode
            ),
            logprobs_mode=self.inference_dynamic_batching_logprobs_mode,
        )
