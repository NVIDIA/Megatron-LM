# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Literal

from megatron.core.transformer.enums import InferenceCudaGraphScope


@dataclass(kw_only=True)
class InferenceScriptConfig:
    """Script/CLI configuration for Megatron inference entry points.

    Fields are populated from argparse via :func:`inference_cfg_from_args`. The argparse
    definitions in ``megatron.training.arguments._add_inference_args`` and
    ``megatron.inference.utils.add_inference_args`` remain the CLI source of truth during
    the gradual migration off global ``get_args()``.

    This is distinct from :class:`megatron.core.inference.config.InferenceConfig`, which
    is the runtime engine config consumed by ``DynamicInferenceContext`` and related core
    APIs. Use :func:`megatron.inference.utils.get_inference_config_from_model_and_args`
    to build the core config from this script config plus a loaded model.

    ``rl_kv_cache_management_mode`` and ``rl_persist_cuda_graphs`` are shared with RL
    training; they remain named after their RL CLI flags until RLConfig lands.
    """

    # Sampling and generation (inference script args)
    temperature: float = 1.0
    """Sampling temperature."""

    top_k: int = 1
    """Top-k sampling."""

    top_p: float = 0.0
    """Top-p sampling."""

    return_log_probs: bool = False
    """Return log probabilities of generated tokens."""

    prompts: list[str] | None = None
    """Input prompts from the command line."""

    num_tokens_to_prompt: list[int] = field(default_factory=lambda: [64, 1024])
    """Range for simulated prompt lengths (min, max)."""

    num_tokens_to_generate: int = 30
    """Number of tokens to generate per prompt."""

    num_tokens_from_file: bool = False
    """Use per-prompt ``num_tokens_to_generate`` from the prompt file."""

    top_n_logprobs: int = 0
    """Return top-n logprobs for generated tokens."""

    skip_prompt_log_probs: bool = False
    """Skip prompt log prob computation."""

    stop_words: list[str] | None = None
    """Stop words that terminate generation."""

    termination_id: int | None = None
    """Token ID that overrides ``tokenizer.eod`` for termination."""

    # Request scheduling (benchmark / dynamic inference scripts)
    incoming_requests_per_step: int | None = None
    """Deterministic number of requests added per engine step."""

    incoming_requests_per_sec: float = 100.0
    """Simulated request arrival rate."""

    incoming_requests_duration: float = 10.0
    """Duration over which simulated requests arrive."""

    suspend_resume_interval: int | None = None
    """Suspend/resume the dynamic engine every N requests."""

    suspend_timeout: float = 0.0
    """Seconds to sleep while the engine is suspended."""

    drain_between_batches: bool = False
    """Drain active requests between batches."""

    batch_boundaries: str | None = None
    """Comma-separated batch start indices (used with ``drain_between_batches``)."""

    inference_repeat_n: int = 1
    """Repeat inference iterations for benchmarking."""

    # Model provider and checkpoint loading
    model_provider: Literal["gpt", "hybrid", "mamba"] = "gpt"
    """Model builder provider (``mamba`` is deprecated; use ``hybrid``)."""

    load: str | None = None
    """Checkpoint load path."""

    exit_on_missing_checkpoint: bool = False
    """Exit when the requested checkpoint is missing."""

    inference_ckpt_non_strict: bool = False
    """Load checkpoint with ``strict=False``."""

    transformer_impl: str | None = None
    """Transformer implementation (e.g. ``inference_optimized``)."""

    fp8_recipe: str | None = None
    """FP8 recipe name."""

    inference_grouped_gemm_backend: str | None = None
    """Grouped GEMM backend for inference-optimized layers."""

    # Static engine limits
    inference_max_requests: int = 8
    """Maximum concurrent requests for static inference."""

    inference_max_seq_length: int = 2560
    """Maximum sequence length (prefill + decode) for static inference."""

    inference_batch_times_seqlen_threshold: int = -1
    """Batch pipelining threshold for inference."""

    max_tokens_to_oom: int = 12000
    """Maximum tokens (prompt + generation) before raising an OOM guard error."""

    use_legacy_static_engine: bool = False
    """Use the legacy static engine implementation."""

    output_bert_embeddings: bool = False
    """Return BERT embeddings instead of binary head output."""

    bert_embedder_type: Literal["megatron", "huggingface"] = "megatron"
    """BERT embedder backend."""

    inference_dynamic_batching: bool = False
    """Enable dynamic batching mode."""

    # Dynamic inference context / engine
    inference_dynamic_batching_buffer_size_gb: float = 40.0
    """GPU memory (GB) for the dynamic inference KV cache."""

    inference_dynamic_batching_paused_buffer_size_gb: float | None = None
    """Memory (GB) reserved for paused requests."""

    inference_dynamic_batching_mamba_memory_ratio: float | None = None
    """Fraction of buffer memory for Mamba states (hybrid models)."""

    inference_dynamic_batching_block_size: int = 256
    """KV cache block size in tokens."""

    inference_dynamic_batching_max_requests: int | None = None
    """Override dynamic context ``max_requests``."""

    inference_dynamic_batching_max_tokens: int | None = None
    """Override dynamic context ``max_tokens``."""

    inference_dynamic_batching_num_cuda_graphs: int = 16
    """Maximum number of CUDA graphs to capture (-1 for auto)."""

    inference_dynamic_batching_track_paused_request_events: bool = False
    """Track paused-request events in request history."""

    inference_dynamic_batching_track_generated_token_events: bool = False
    """Track per-token generation events with timestamps."""

    decode_only_cuda_graphs: bool = False
    """Capture CUDA graphs for decode-only steps only."""

    inference_cuda_graph_all_prefills: bool = False
    """Extend prefill/mixed CUDA graph capture up to ``max_tokens``."""

    inference_cuda_graph_scope: InferenceCudaGraphScope | None = None
    """Scope of inference CUDA graph capture."""

    inference_dynamic_batching_unified_memory_level: Literal[0, 1] = 0
    """Unified memory level for the dynamic inference context."""

    enable_chunked_prefill: bool = False
    """Enable chunked prefill."""

    num_speculative_tokens: int = 0
    """Number of speculative decode tokens."""

    inference_dynamic_batching_enable_prefix_caching: bool = False
    """Enable prefix caching for dynamic batching."""

    inference_dynamic_batching_prefix_caching_eviction_policy: Literal["ref_zero", "lru"] = (
        "ref_zero"
    )
    """Prefix cache block eviction policy."""

    inference_dynamic_batching_prefix_caching_coordinator_policy: Literal[
        "longest_prefix", "first_prefix_block", "round_robin"
    ] = "first_prefix_block"
    """Prefix cache coordinator routing policy."""

    inference_dynamic_batching_prefix_caching_routing_alpha: float = 0.5
    """Weight for prefix-aware coordinator routing."""

    inference_dynamic_batching_prefix_caching_mamba_gb: float | None = None
    """GPU memory budget (GB) for prefix-cached Mamba states."""

    inference_dynamic_batching_cuda_graph_mixed_prefill_count: int = 16
    """Number of mixed-prefill requests captured in a CUDA graph."""

    inference_dynamic_batching_cuda_graph_sizing_distribution: Literal["exponential", "linear"] = (
        "exponential"
    )
    """CUDA graph token-count spacing strategy."""

    inference_dynamic_batching_sampling_backend: Literal["torch", "flashinfer"] = "torch"
    """Sampling kernel backend during dynamic inference."""

    use_flashinfer_fused_rope: bool = False
    """Use FlashInfer fused RoPE."""

    inference_use_synchronous_zmq_collectives: bool = False
    """Use synchronous ZMQ collectives during inference."""

    inference_disable_ep_consensus: bool = False
    """Skip EP-group consensus in the inference engine control loop."""

    mamba_inference_conv_states_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    """Dtype for Mamba inference conv states."""

    mamba_inference_ssm_states_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    """Dtype for Mamba inference SSM states."""

    # Shared with RL (--rl-* CLI flags)
    rl_kv_cache_management_mode: Literal["persist", "offload", "recompute"] = "persist"
    """KV cache management mode (shared RL/inference flag)."""

    rl_persist_cuda_graphs: bool = False
    """Persist CUDA graph KV pointers across suspend/resume (shared RL/inference flag)."""

    # Logging and output
    inference_logging_step_interval: int = 0
    """Step interval for inference metric logging (0 disables)."""

    inference_wandb_logging: bool = False
    """Enable inference Weights & Biases logging."""

    inference_text_gen_server_logging: bool = False
    """Enable per-request logging in the text generation server."""

    inference_coordinator_port: int | None = None
    """Port for the inference coordinator on node 0."""

    record_throughput: bool = True
    """Include throughput stats in JSON output."""

    output_path: str | None = None
    """Path to save generations as JSON."""

    output_every_n_results: int = 1
    """Write every Nth result to the output file."""

    output_request_events: bool = False
    """Include request lifecycle events in JSON output."""

    prompt_file: str | None = None
    """Jsonl file containing input prompts."""

    prompt_file_num_truncate: int | None = None
    """Number of prompt-file samples to use."""

    throughput_check_only: bool = False
    """Run throughput check without verifying outputs."""

    coordinator_schedule_output_path: str | None = None
    """Path for coordinator scheduling decision JSON output."""
