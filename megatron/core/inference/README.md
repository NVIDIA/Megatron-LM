# Megatron Core Inference

This module implements an inference system for large language model serving intended for research purposes. It supports dynamic batching, KV cache management, distributed coordination, speculative decoding, prefix caching, and a variety of model architectures including GPT and Mamba hybrids.

## Directory Structure

```
inference/
├── config.py                              # InferenceConfig dataclass (all settings)
├── inference_request.py                   # Request classes and prefix hash computation
├── sampling_params.py                     # Per-request sampling parameters
├── common_inference_params.py             # Deprecated alias (CommonInferenceParams = SamplingParams)
├── scheduler.py                           # Request pool management (FIFO scheduling)
├── batch_dimensions_utils.py              # CUDA graph batch dimension management
├── data_parallel_inference_coordinator.py # Multi-rank DP coordinator (ZMQ)
├── inference_client.py                    # Async client for submitting requests
├── async_stream.py                        # Async streaming interface
├── headers.py                             # ZMQ message header definitions
├── communication_utils.py                 # Distributed communication helpers
├── unified_memory.py                      # CPU/GPU unified memory (UVM) support
├── symmetric_memory.py                    # Symmetric memory utilities
├── utils.py                               # Shared inference utilities
│
├── engines/
│   ├── abstract_engine.py                 # Engine interface (AbstractEngine)
│   ├── dynamic_engine.py                  # Primary engine for dynamic batching
│   ├── static_engine.py                   # Legacy fixed-batch engine (deprecated)
│   ├── mcore_engine.py                    # Alias: MCoreEngine = StaticInferenceEngine
│   └── async_zmq_communicator.py          # Low-level async ZMQ send/recv
│
├── contexts/
│   ├── base_context.py                    # Abstract context interface
│   ├── static_context.py                  # Fixed-size KV cache context (deprecated)
│   ├── dynamic_context.py                 # Block-allocated KV cache context
│   ├── kv_block_allocator.py              # KV cache block pool and prefix caching
│   ├── mamba_slot_allocator.py            # Mamba conv/SSM state caching
│   ├── fused_kv_append_kernel.py          # Triton kernel for fused KV cache append
│   ├── attention_context/
│   │   ├── metadata_base.py               # Base attention metadata
│   │   ├── mha_metadata.py                # Multi-head attention metadata
│   │   ├── mamba_metadata.py              # Mamba SSM layer metadata
│   │   └── triton/
│   │       └── tensor_ops.py              # Triton tensor operations for attention
│   └── routing_metadata.py               # MoE token routing metadata
│
├── model_inference_wrappers/
│   ├── abstract_model_inference_wrapper.py  # Model wrapper interface
│   ├── gpt/
│   │   └── gpt_inference_wrapper.py         # Causal decoder-only models (GPT)
│   ├── t5/
│   │   └── t5_inference_wrapper.py          # Encoder-decoder models (T5)
│   └── multimodal/
│       └── vlm_inference_wrapper.py         # Vision-Language models
│
├── text_generation_controllers/
│   ├── text_generation_controller.py                    # Main sampling loop
│   ├── encoder_decoder_text_generation_controller.py    # Encoder-decoder sampling
│   └── vlm_text_generation_controller.py                # VLM sampling
│
├── text_generation_server/
│   ├── text_generation_server.py            # Flask REST API (legacy)
│   ├── run_mcore_engine.py                  # Generation loop runner (param broadcast + execution)
│   ├── tokenization.py                      # Prompt tokenization and broadcast utilities
│   ├── endpoints/
│   │   ├── common.py                        # Shared endpoint utilities
│   │   └── completions.py                   # Completions endpoint
│   └── dynamic_text_gen_server/             # Modern async server (Quart, OpenAI-compatible)
│       ├── text_generation_server.py        # Quart app setup and routing
│       ├── tokenization.py                  # Tokenization utilities for dynamic server
│       └── endpoints/
│           ├── common.py                    # Shared request/response handling
│           ├── completions.py               # /v1/completions endpoint
│           ├── chat_completions.py          # /v1/chat/completions endpoint
│           └── health.py                    # /health and /v1/health endpoints
│
├── moe/
│   ├── fused_moe.py                         # Fused GEMM for expert computation
│   ├── activations.py                       # Expert activation functions
│   ├── permute.py                           # Token-to-expert permutation
│   └── pad.py                               # Expert padding for CUDA graphs
│
├── quantization/
│   ├── mxfp8_quantize.py                   # MXFP8 quantization logic
│   ├── mxfp8_tensor.py                     # FP8 tensor wrapper with scales
│   └── utils.py                            # Quantization helpers
│
└── communication/
    └── torch_symm_triton/                   # Symmetric memory communication kernels
        ├── barrier.py                       # Distributed barrier implementation
        ├── collectives.py                   # Collective operations (all-reduce, etc.)
        ├── fused_collectives.py             # Fused collective + compute kernels
        ├── multimem_asm.py                  # Multi-memory assembly operations
        └── utils.py                         # Communication utilities
```

## Architecture Overview

The inference system is organized around four layers:

```
Client / Server  ──>  Engine  ──>  Context + Scheduler  ──>  Model Wrapper + Sampling
```

1. **Clients and servers** submit requests and receive results.
2. **The engine** orchestrates the generation loop, manages request lifecycles, and coordinates across distributed ranks.
3. **The context** manages KV cache memory (block allocation, prefix caching, pause/resume).
4. **Model wrappers and the text generation controller** run the forward pass and sample tokens.

### Request Lifecycle

```
Client submits prompt + SamplingParams
        │
        ▼
[If multi-rank] Coordinator routes to DP rank (prefix-aware or round-robin)
        │
        ▼
Engine adds request ──> Scheduler places in active or waiting pool
        │
        ▼
Prefill: full prompt processed, KV cache blocks allocated
        │
        ▼
Decode loop: generate one token per step
  ├── Forward pass through model wrapper
  ├── Sample token via TextGenerationController
  ├── Update KV cache
  └── Check termination (EOS, max length, stop words)
        │
        ▼
Completed: detokenize, return results to client, release KV blocks
```

## Core Components

### Inference Engine (`engines/dynamic_engine.py`)

`DynamicInferenceEngine` is the primary engine. It supports:

- **Dynamic batching**: variable-sized requests processed together, with requests entering and leaving the batch at any time.
- **Request lifecycle management**: the engine transitions through states (RUNNING, PAUSED, SUSPENDED, STOPPED, and intermediate transition states) based on memory availability and control signals.
- **CUDA graph capture and replay**: pre-captured graphs for common batch dimensions eliminate CPU launch overhead.
- **Async processing**: an asyncio event loop handles concurrent request submission and result retrieval.
- **ZMQ-based coordination**: communicates with the `DataParallelInferenceCoordinator` for multi-rank data-parallel inference.
- **Speculative decoding**: generate multiple candidate tokens per step and verify them for faster throughput.
- **Chunked prefill**: split large prompts across multiple forward passes to control activation memory.

The legacy `StaticInferenceEngine` is deprecated and internally delegates to `DynamicInferenceEngine`.

### Inference Context (`contexts/dynamic_context.py`)

`DynamicInferenceContext` manages all per-request state:

- **Block-based KV cache**: memory is divided into fixed-size blocks (default 256 tokens). Blocks are allocated and released on demand via `KVBlockAllocator`.
- **Attention metadata**: tracks cumulative sequence lengths, block tables, and position IDs needed by the attention kernels.
- **Request tracking**: maintains counts of active, paused, and total requests.
- **Pause/resume**: when memory is insufficient, existing requests are paused (their blocks are freed or offloaded) and later resumed.
- **Mamba state management**: for hybrid Mamba/attention models, `MambaSlotAllocator` manages conv and SSM state tensors alongside the KV cache.

### KV Block Allocator (`contexts/kv_block_allocator.py`)

Manages the block pool for KV cache memory:

- Fixed-size block allocation and deallocation.
- Separate tracking for active and paused blocks.
- **Prefix caching**: blocks are hashed based on their token content. Requests sharing a common prefix can reuse cached blocks, avoiding redundant computation.
- **Eviction policies**:
  - `REF_ZERO`: release blocks immediately when no request references them.
  - `LRU`: keep released blocks cached; evict the least-recently-used block when space is needed.

### Scheduler (`scheduler.py`)

A simple FIFO scheduler that manages three pools:

- **Waiting pool**: requests queued for processing.
- **Active pool**: requests currently being generated.
- **Completed pool**: finished requests awaiting result retrieval.

Requests are promoted from waiting to active as memory and batch-size constraints allow.

### Text Generation Controller (`text_generation_controllers/text_generation_controller.py`)

Implements the token sampling loop:

- **Tokenization/detokenization** of prompts and generated text.
- **Sampling**: temperature scaling, top-k filtering, top-p (nucleus) sampling.
- **Log probability tracking**: optionally returns per-token log probabilities.
- **Stop word detection**: halts generation when a specified string is produced.
- **Speculative decoding support**: coordinates with Multi-Token Prediction (MTP) heads for draft-and-verify generation.

### Model Wrappers (`model_inference_wrappers/`)

Thin wrappers that adapt different model architectures for inference:

| Wrapper | Architecture | Key Behavior |
|---------|-------------|--------------|
| `GPTInferenceWrapper` | Causal decoder-only (GPT, LLaMA, etc.) | Attention mask + position ID construction, context-window slicing for prefill/decode |
| `T5InferenceWrapper` | Encoder-decoder (T5) | Separate encoder/decoder passes, padding-aware masking |
| `VLMInferenceWrapper` | Vision-Language (e.g., LLaVA) | Image embedding integration, tile-based image processing |

### Data-Parallel Coordinator (`data_parallel_inference_coordinator.py`)

Coordinates multi-rank data-parallel inference via ZMQ:

- **Worker registration**: DP ranks connect and register.
- **Request routing**: distributes incoming requests across ranks. Routing policies when prefix caching is enabled:
  - `LONGEST_PREFIX`: route to the rank with the best prefix match.
  - `FIRST_PREFIX_BLOCK`: O(1) lookup by first block hash.
  - `ROUND_ROBIN`: simple load balancing (ignores prefix affinity).
- **Prefix-aware scoring**: `score = alpha * match + (1 - alpha) * normalized_load`, controlled by `prefix_caching_routing_alpha`.
- **Control signals**: broadcast PAUSE, SUSPEND, RESUME, and STOP across all ranks.

### Inference Client (`inference_client.py`)

An async client for submitting requests to the coordinator:

```python
client = InferenceClient(inference_coordinator_address)
future = client.add_request(prompt, sampling_params)  # returns asyncio.Future
result = await future  # resolves when generation completes
```

## Configuration

All inference settings are centralized in `InferenceConfig` (`config.py`):

### KV Cache and Memory

| Parameter | Default | Description |
|-----------|---------|-------------|
| `block_size_tokens` | 256 | Number of tokens per KV cache block |
| `buffer_size_gb` | 20 | GPU memory reserved for KV cache |
| `paused_buffer_size_gb` | None | Memory reserved for paused request blocks |
| `max_requests` | None | Max concurrent active requests |
| `max_tokens` | None (16384) | Max tokens per forward pass |
| `unified_memory_level` | 0 | 0 = GPU only, 1 = GPU + CPU (UVM) |
| `kv_cache_management_mode` | PERSIST | How to handle KV cache on suspend: PERSIST, OFFLOAD, or RECOMPUTE |

### CUDA Graphs

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_cuda_graphs` | None | Max number of graphs to capture (batch sizes 1..max_requests) |
| `cuda_graph_mixed_prefill_count` | 16 | Number of mixed prefill/decode graphs |
| `use_cuda_graphs_for_non_decode_steps` | True | Use CUDA graphs for prefill steps |
| `static_kv_memory_pointers` | False | Keep KV buffers at fixed addresses across suspend/resume for graph validity |

### Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_sequence_length` | 2560 | Max total tokens (prompt + output) |
| `materialize_only_last_token_logits` | True | Only compute logits for the last token (memory optimization) |
| `use_flashinfer_fused_rope` | False | Use FlashInfer's fused RoPE kernel |

### Features

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_chunked_prefill` | False | Split large prompts across multiple forward passes |
| `num_speculative_tokens` | 0 | Number of speculative tokens per decode step |
| `enable_prefix_caching` | False | Enable KV block reuse for shared prefixes |
| `prefix_caching_eviction_policy` | REF_ZERO | Block eviction strategy: REF_ZERO or LRU |
| `prefix_caching_coordinator_policy` | FIRST_PREFIX_BLOCK | DP routing policy for prefix caching |
| `prefix_caching_routing_alpha` | 0.5 | Weight between prefix match (1.0) and load balance (0.0) |

### Mamba Hybrid Models

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mamba_inference_state_config` | None | Config for Mamba conv/SSM state tensors |
| `mamba_memory_ratio` | None | Fraction of memory buffer for Mamba states |
| `prefix_caching_mamba_gb` | None | GPU memory budget for Mamba state prefix cache |

### Logging

| Parameter | Default | Description |
|-----------|---------|-------------|
| `track_paused_request_events` | False | Record events when requests are paused |
| `track_generated_token_events` | False | Record per-token generation timestamps |
| `metrics_writer` | None | Wandb module for metrics |
| `logging_step_interval` | 0 | Step interval for wandb logging (0 = disabled) |

## Sampling Parameters

Per-request generation behavior is controlled via `SamplingParams` (`sampling_params.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 1.0 | Scales logits before sampling. Lower = more deterministic. |
| `top_k` | 0 | Keep only the k highest-probability tokens (0 = disabled) |
| `top_p` | 0.0 | Nucleus sampling: keep tokens with cumulative probability <= p (0.0 = disabled) |
| `num_tokens_to_generate` | None | Max tokens to generate |
| `termination_id` | None | Token ID that stops generation |
| `stop_words` | None | List of strings that stop generation when produced |
| `return_log_probs` | False | Include per-token log probabilities in output |
| `top_n_logprobs` | 0 | Return the top N log probabilities per generated token |
| `add_BOS` | False | Prepend a beginning-of-sequence token |

## Key Features

### Dynamic Batching
Requests of varying lengths are batched together. New requests enter the batch as existing ones complete, maximizing GPU utilization.

### Prefix Caching
When enabled, KV cache blocks are hashed based on token content. Requests with a shared prefix (e.g., a common system prompt) reuse already-computed KV blocks, skipping redundant prefill computation. See [KV Block Allocator](#kv-block-allocator-contextskv_block_allocatorpy) for eviction policies.

### Speculative Decoding
Multi-Token Prediction (MTP) heads generate draft tokens that are verified in a single forward pass. This reduces the number of sequential decode steps needed.

### Chunked Prefill
Large prompts are split into chunks and processed across multiple forward passes. This bounds activation memory usage and allows decode requests to make progress between chunks.

### Pause/Resume and Memory Management
When GPU memory is insufficient for new or growing requests, the engine pauses existing requests by releasing their KV cache blocks. Three modes control what happens to the data:
- **PERSIST**: blocks remain on GPU (fastest resume, highest memory use).
- **OFFLOAD**: blocks are moved to CPU memory via UVM.
- **RECOMPUTE**: blocks are discarded and recomputed from the prompt on resume.

### CUDA Graphs
Batch dimensions are pre-enumerated and CUDA graphs are captured for each. During inference, real batches are matched to the nearest captured graph, eliminating kernel launch overhead.

### MoE (Mixture of Experts)
Fused GEMM kernels (`moe/fused_moe.py`) handle expert computation efficiently. Routing metadata tracks token-to-expert assignments. Backend selection adapts based on model config, CUDA graph status, and quantization settings.

### Quantization
MXFP8 quantization (`quantization/`) reduces memory footprint and compute cost by representing weights and activations in 8-bit floating point with per-block scaling factors.

## Serving

### REST API

The `text_generation_server/` module provides HTTP endpoints for serving:

- **Legacy server** (`text_generation_server.py`): Flask-based, PUT `/generate` endpoint.
- **Dynamic server** (`dynamic_text_gen_server/`): Quart-based async server with OpenAI-compatible `/v1/completions` and `/v1/chat/completions` endpoints, plus `/health` for health checks.

### Entry Point

`text_generation_server/run_mcore_engine.py` provides the `run_mcore_engine` function that takes an already-instantiated engine, broadcasts sampling parameters across ranks, and runs the generation loop.

## Error Handling

The context layer defines a hierarchy of overflow errors:

| Error | Transient? | Meaning |
|-------|-----------|---------|
| `RequestOverflowError` | Yes | Too many concurrent requests |
| `TokenOverflowError` | Yes | Too many tokens in the current forward pass |
| `BlockOverflowError` | Yes | Not enough free KV cache blocks |
| `MaxSequenceLengthOverflowError` | No | Prompt + output exceeds `max_sequence_length` |

Transient errors trigger request pausing or queuing; non-transient errors fail the request immediately.

## ZMQ Communication Protocol

The coordinator and engines communicate via ZMQ messages with typed headers:

| Header | Direction | Purpose |
|--------|-----------|---------|
| `CONNECT` / `CONNECT_ACK` | Engine <-> Coordinator | Worker registration |
| `SUBMIT_REQUEST` | Client -> Coordinator | Submit a generation request |
| `ENGINE_REPLY` | Engine -> Coordinator -> Client | Return completed results |
| `PAUSE` / `UNPAUSE` | Coordinator -> Engine | Memory pressure signals |
| `SUSPEND` / `RESUME` | Coordinator -> Engine | Full suspend/resume cycle |
| `DISCONNECT` | Engine -> Coordinator | Worker disconnection |
| `STOP` / `SHUTDOWN` | Coordinator -> Engine | Graceful teardown |
| `SET_GENERATION_EPOCH` | Coordinator -> Engine | Checkpoint versioning |
