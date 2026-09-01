# Megatron Core Inference User Guide

A practical guide to running inference with Megatron Core (MCore) using the
dynamic inference path. This is the recommended and actively developed
inference stack in Megatron-LM.

The legacy static engine is deprecated. New work should target the dynamic path described here.

---

## Table of Contents

- [What Megatron Inference Is For](#what-megatron-inference-is-for)
- [Rollout Performance](#rollout-performance)
- [Supported Features](#supported-features)
- [Basic Usage: The High-Level API](#basic-usage-the-high-level-api)
   - [The two classes](#the-two-classes-megatronllm-and-megatronasyncllm)
   - [Direct Mode Compared to Coordinator (Indirect) Mode](#direct-mode-compared-to-coordinator-indirect-mode)
   - [Sync offline batch generation](#sync-offline-batch-generation)
   - [Async generation](#async-generation)
   - [Sampling parameters](#sampling-parameters)
   - [Engine configuration](#engine-configuration)
   - [Reading results](#reading-results)
   - [Lifecycle controls](#lifecycle-controls)
- [Async Scheduling](#async-scheduling)
- [Streaming](#streaming)
- [OpenAI-Compatible HTTP Server](#openai-compatible-http-server)
- [Weight Refit and Resharding for RL](#weight-refit-and-resharding-for-rl)
- [Multimodal (Vision-Language) Inference](#multimodal-vision-language-inference)
- [Disaggregated Prefill and Decode](#disaggregated-prefill-and-decode)
- [Customizing the Pipeline](#customizing-the-pipeline)
   - [Pipeline anatomy](#pipeline-anatomy)
   - [Customizing the TextGenerationController](#customizing-the-textgenerationcontroller)
   - [Customizing the DynamicInferenceContext](#customizing-the-dynamicinferencecontext)
   - [Driving the engine directly](#driving-the-engine-directly)
- [Examples Directory](#examples-directory)
- [Known Limitations](#known-limitations)
- [Roadmap and Future Work](#roadmap-and-future-work)
- [Additional Resources](#additional-resources)

---

## What Megatron Inference Is For

> **Scope.** Megatron Inference is an actively developed path designed primarily
> for RL and for workflows that require training/inference alignment. It is not
> currently positioned as a general-purpose production-serving replacement for
> vLLM, SGLang, or TensorRT-LLM. Performance varies by model, workload, and
> which consistency features you enable.

Megatron Inference is built primarily as the generation engine for
*reinforcement learning (RL)*, not as a standalone serving engine. Its design
center is the RL loop, where a model alternates between *training* and
*rollout* phases inside the same process. A rollout is typically generation
plus sandboxing or environment infrastructure. Megatron Inference provides the
*generation* portion.

This focus drives the major design benefits:

- **Consistency between training and inference.** RL is extremely sensitive to
  numerical mismatch between the framework that *trains* the policy and the one
  that *generates* rollouts. Running both in MCore removes the cross-framework
  portion of this gap and makes the remaining numerical mismatch far easier to
  control (refer to batch-invariant kernels below).
- **No model conversion.** Because generation runs on the same MCore model,
  there is *no Hugging Face to MCore conversion* step between training and
  generation. Architectures land in the inference stack close behind their
  training support, subject to the gaps in
  [Known Limitations](#known-limitations).
- **Inexpensive training to inference transitions.** This is because tight coupling enables
  in-place weight refit and shared memory management, drastically cutting
  re-initialization cost relative to standing up an external inference engine
  each rollout.
- **Colocated and non-colocated deployments.** Megatron Inference supports
  *weight refit and resharding between training and inference*, so the same
  weights can be moved between the two phases under different parallelism
  layouts. This covers both *colocated* setups (where training and inference share
  the same GPUs) and *non-colocated* setups (where training and inference run on
  separate resources), with the engine resharding weights to the inference-time
  parallel configuration during the swap.
- **First-class parallelism reuse.** Inference reuses Megatron Core's existing
  tensor parallelism (TP), expert parallelism (EP), and pipeline parallelism (PP) infrastructure directly.

---

## Rollout Performance

In an August 2026 Nemotron Ultra V3 SWE rollout benchmark, Megatron Inference
delivered **12.30 rollouts/GPU-hour versus 8.47 for vLLM (45% higher
throughput)**. It reduced mean wall-clock time by 31% and end-to-end trajectory
latency by 20% at p50, 21% at p90, and 18% at p99, while reward remained within
the observed run-to-run variation.

The comparison used the same Ultra V3 model, SWE validation set, 60-turn agent
limit, 64-request concurrency, and 16 GPUs per run; only the generation engine
differed. Results are averages over five runs and 1,600 trajectories per engine.
The batches ran six days apart on the same cluster, so node-level variation was
not controlled.

Performance maturity varies by model family. Most optimization work to date has
focused on hybrid models. In a representative Qwen 30B EP4 run on GB200
(batch size 256, output sequence length 256), Megatron Core reached about
24,000 generated tokens/s versus 34,000 tokens/s for vLLM, a gap of
approximately 29%. Additional Qwen optimizations are under active development
and are expected to narrow this gap as they are merged.

---

## Supported Features

| Area | Features |
|---|---|
| **Batching** | Dynamic or in-flight batching with vectorized bookkeeping, dynamic suspend and resume, and request eviction for high input-rate regimes. [Async scheduling](#async-scheduling), enabled by default, moves host-side bookkeeping off the critical path |
| **Chunked prefill** | Chunked-prefill scheduling with decode piggybacking, so long prompts don't stall in-flight decodes |
| **Attention and KV cache** | Optimized PagedAttention with prefix caching (LRU and ref-zero eviction, prefix-aware and load-aware coordinator routing, Mamba-state prefix caching for hybrid models). Sliding-window and sink attention are supported |
| **CUDA graphs** | Full-model CUDA graphs for prefill, decode, and mixed batches. Prefill and mixed steps up to `cuda_graph_max_tokens` (512 by default) get a graph instead of falling back to eager |
| **Speculative decoding** | Multi-Token Prediction (MTP)-based speculative decoding (with fused MTP bookkeeping and MTP CUDA graphs) |
| **Serving** | OpenAI-compatible HTTP server with chat templates, tool calling, and reasoning parsers. Server-sent-event [streaming](#streaming) of partial completions, including incremental tool-call deltas, plus health and profiling endpoints and prefix-cache hit reporting |
| **MoE** | Expert model parallelism with full CUDA-graph support, expert router replay, NVLS switch-multicast token dispatcher (notably faster than the all-to-all dispatchers other frameworks use) plus an allgatherv dispatcher optimized for multi-node NVLink, and shared-expert overlap with latent MoEs. Selectable grouped-GEMM backend (vLLM, torch, or FlashInfer) |
| **Parallelism** | Data-parallel coordinator with full multi-node support, tensor model parallelism with low-latency comm primitives, expert model parallelism, and pipeline parallelism |
| **Model families** | GPT-style dense models, MoE models, MLA models (for example DeepSeek-style, with `cache_mla_latents`), Mamba and hybrid (SSM and attention) models, Gated Delta Net and Gated Delta Product models, and vision-language models for image inputs. Refer to [Known Limitations](#known-limitations) for the per-family feature gaps |
| **Precision** | MXFP8 weight quantization through `--transformer-impl inference_optimized --fp8-recipe mxfp8`, using latency-optimized inference kernels. Configurable Mamba conv and SSM state dtypes |
| **RL** | [Weight refit and resharding](#weight-refit-and-resharding-for-rl) between training and inference over five transports, supporting both colocated (shared GPUs) and non-colocated (separate resources) deployments. Batch-invariant kernels for training and inference log-prob consistency. Per-DP-rank sampling seeds so the same prompt routed to different replicas yields different samples |
| **Sampling** | Temperature, top-k, top-p, stop words, log-probs, and top-N log-probs, with raw or post-processed log-prob semantics (`logprobs_mode`). Pluggable torch or FlashInfer sampling backend |
| **Disaggregation** | KV and SSM state handoff between prefill and decode engines over NIXL or NCCL, with resharding across mismatched TP/PP layouts. Refer to [Disaggregated Prefill and Decode](#disaggregated-prefill-and-decode) for what is and is not turnkey today |
| **Observability** | Per-request event tracking, wandb metrics, `/start_profile` and `/stop_profile` endpoints for `nsys --capture-range=cudaProfilerApi`, and MoE routing traces for load and predictability analysis |

> **Batch-invariant kernels (training and inference log-prob consistency).** Standard
> GEMM, attention, and norm kernels can produce slightly different numerics depending
> on batch composition, which shows up as log-prob mismatch between training and
> inference. This mismatch is a real source of error and instability in RL. Megatron Inference
> offers *batch-invariant kernels* whose outputs do not depend on how requests are
> batched, so per-token log-probs match between the training and inference forward
> passes.
>
> Enable it with `batch_invariant_mode` on the model's `TransformerConfig` — it
> is *not* an `InferenceConfig` field, so it must be set when you build the
> model, not on the engine config. Two companion fields tune it:
> `batch_invariant_backend` (`te_native` by default, or `deepgemm` / `triton`)
> selects the GEMM backend, and `batch_invariant_collective` (`ordered` by
> default, or `multimem`) selects the cross-rank expert-combine reduction.
>
> Both dense and MoE models are supported. Batch-invariant MoE is bf16-only,
> requires the unfused permute/unpermute path, and under
> `--transformer-impl inference_optimized` requires
> `inference_grouped_gemm_backend` of `vllm` or `torch` (plus the `nvls` token
> dispatcher when `EP > 1`). Some backend combinations additionally need DeepGEMM
> bf16 bindings: `uv pip install -e .[batch_invariant]`. Context parallelism and
> attention dropout are not supported in either case.

Many of these are toggled through `InferenceConfig`. Refer to the
[Engine configuration](#engine-configuration).

---

## Basic Usage: The High-Level API

The API lives in
[`megatron/core/inference/apis/`](../megatron/core/inference/apis/) and gives
you a *vLLM-style* `generate(prompts, sampling_params)` interface. It hides
the underlying pipeline (`DynamicInferenceContext` to `GPTInferenceWrapper` to
`TextGenerationController` to `DynamicInferenceEngine`) so that you do not have to
wire it up by hand.

```python
from megatron.core.inference.apis import (
    MegatronLLM,        # sync
    MegatronAsyncLLM,   # async
    SamplingParams,
    ServeConfig,
)
```

### The two classes: `MegatronLLM` and `MegatronAsyncLLM`

| Class | Use it when | Key methods |
|---|---|---|
| **`MegatronLLM`** | Synchronous offline batch generation (the common RL-rollout case). | `generate`, `pause`/`unpause`/`suspend`/`resume`, `serve(serve_config)`, `shutdown`/`wait_for_shutdown`; context manager (`with ... as llm:`) |
| **`MegatronAsyncLLM`** | Asyncio-native generation, and HTTP serving from inside an existing event loop. | `async generate`, async lifecycle controls, `serve(serve_config)`; async context manager (`async with ... as llm:`) |

Both expose the underlying building blocks as read-only properties. Use these for [advanced customization](#customizing-the-pipeline):

- `llm.engine`
- `llm.context`
- `llm.controller`
- `llm.is_primary_rank`

Both also expose `submit(coro)` and `run_sync(coro)`, which schedule a coroutine
on the engine's background runtime loop. Use these to reach the lower-level
async surface (for example `InferenceClient` streaming) without standing up your
own loop.

Constructor arguments worth knowing: `use_coordinator` (**defaults to `True`**),
`coordinator_host` / `coordinator_port`, and `inference_wrapper_cls` (defaults to
`GPTInferenceWrapper`; pass `VLMInferenceWrapper` for vision-language models).

**Caller responsibilities (before construction):**

- Call `initialize_megatron(...)` to perform full Megatron distributed setup.
- Build the model and call `model.eval()`. The API does *not* toggle model
  state.
- Have a tokenizer ready.

### Direct Mode Compared to Coordinator (Indirect) Mode

Megatron Inference supports two operating modes. Direct mode is simpler but limited. Coordinator mode adds a routing layer that enables serving, expert parallelism, and lifecycle controls. **Coordinator mode is the default** (`use_coordinator=True`); opt into direct mode explicitly.

#### Direct Mode (`use_coordinator=False`)

Direct mode is the simplest configuration for offline batch generation:

- *Every rank is treated as primary* and runs the engine synchronously.
- *You own data sharding*, which means that you decide the prompts that are assigned to which
  data-parallel replica and call `generate` on each.
- The simplest path for offline batch generation when you already shard the data
  yourself (typical for many RL rollout setups).
- Lifecycle controls (`pause`/`suspend`/...) are *not available* and raise
  `RuntimeError`. So are `submit`/`run_sync`, which need the background runtime
  loop.
- *Not allowed with expert parallelism* (`EP > 1`). This is because EP routing requires the
  coordinator.
- Text-only: `multi_modal_data` is rejected in direct mode.

```python
with MegatronLLM(
    model=model,
    tokenizer=tokenizer,
    inference_config=inference_config,
    use_coordinator=False,        # direct mode
) as llm:
    results = llm.generate(["Megatron inference is", "Hello, world"],
                           SamplingParams(num_tokens_to_generate=64))
    for r in results:
        print(r.generated_text)
```

#### Coordinator Mode (`use_coordinator=True`)

Coordinator mode adds a background routing layer and is required for serving and advanced features:

- A background data-parallel *coordinator routes requests across DP
  replicas* for you. An `InferenceClient` on *global rank 0* submits work.
- *Required* for: HTTP serving (`serve`), expert parallelism (`EP > 1`),
  multimodal inputs, streaming, and the lifecycle controls
  (`pause`/`unpause`/`suspend`/`resume`).
- `generate` may only be called on the *primary rank* (rank 0). Worker ranks
  block until shutdown propagates.
- Internally spins up a daemon-thread event loop so the engine's asyncio
  primitives don't collide with your loop.

```python
with MegatronLLM(
    model=model,
    tokenizer=tokenizer,
    inference_config=inference_config,
    use_coordinator=True,         # coordinator mode
) as llm:
    if llm.is_primary_rank:
        results = llm.generate(prompts, SamplingParams(num_tokens_to_generate=64))
```

> **Mode and class compatibility:** `MegatronAsyncLLM` *requires
> `use_coordinator=True`* (direct async is rejected at `__init__`).
> `MegatronLLM` supports both. So the three supported combinations are:
> sync+direct, sync+coordinator, async+coordinator.

| | Direct (`use_coordinator=False`) | Coordinator (`use_coordinator=True`, default) |
|---|---|---|
| Data sharding | You handle it | Coordinator routes across DP |
| `generate` callable on | Every rank | Primary rank (rank 0) only |
| HTTP `serve()` | ❌ | ✅ |
| Expert parallelism (EP > 1) | ❌ | ✅ |
| `pause`/`suspend`/`resume` | ❌ | ✅ |
| `submit` / `run_sync` | ❌ | ✅ |
| Streaming | ❌ | ✅ |
| `multi_modal_data` | ❌ | ✅ |
| `MegatronAsyncLLM` | ❌ | ✅ |

> **`serve()` is not async-only.** Both `MegatronLLM.serve(...)` and
> `MegatronAsyncLLM.serve(...)` start the HTTP frontend; what serving requires is
> *coordinator mode*, not the async class. Use `MegatronLLM.serve(...)` from a
> plain synchronous launcher script, and `MegatronAsyncLLM.serve(...)` when you
> are already inside an event loop.

### Sync Offline Batch Generation

The runnable end-to-end script is
[`examples/inference/offline_inference.py`](../examples/inference/offline_inference.py).
A minimal version:

```python
from megatron.core.inference.apis import MegatronLLM, SamplingParams

# Assumes that initialize_megatron(...) already ran and that model.eval() was called.
with MegatronLLM(
    model=model,
    tokenizer=tokenizer,
    inference_config=inference_config,
    use_coordinator=False,
) as llm:
    results = llm.generate(
        ["The capital of France is", "Write a haiku about GPUs"],
        SamplingParams(num_tokens_to_generate=128, temperature=0.8, top_p=0.95),
    )
    for r in results:
        print(r.generated_text)
```

`generate` accepts a single prompt or a batch, as *strings or pre-tokenized
token-id lists*:

- `"a single string"`: returns a 1-element list
- `["a", "b"]`: returns a list in input order
- `[1, 2, 3]`: a single token-id prompt
- `[[1, 2], [3, 4]]`: a batch of token-id prompts

`MegatronLLM.generate` *always* returns a `list[DynamicInferenceRequest]`,
even for single-prompt input.

### Async Generation

`MegatronAsyncLLM` mirrors the sync API with `await`. There is a deliberate
asymmetry:

* async `generate` returns a *single* request for single input
* *list* for batched input

```python
import asyncio
from megatron.core.inference.apis import MegatronAsyncLLM, SamplingParams

async def main():
    async with MegatronAsyncLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=inference_config,
        use_coordinator=True,     # async requires coordinator mode
    ) as llm:
        if llm.is_primary_rank:
            r = await llm.generate("Hello", SamplingParams(num_tokens_to_generate=32))
            print(r.generated_text)            # single input -> single result
            rs = await llm.generate(["a", "b"], SamplingParams(num_tokens_to_generate=32))
            print([x.generated_text for x in rs])  # batch input -> list

asyncio.run(main())
```

### Sampling Parameters

`SamplingParams` controls decoding behavior for each `generate` call:

| Field | Meaning |
|---|---|
| `num_tokens_to_generate` | Max new tokens to generate |
| `num_tokens_total` | Cap on prompt + generated length. Mutually exclusive with `num_tokens_to_generate` |
| `temperature` | Softmax temperature (`1.0` = unmodified) |
| `top_k` | Keep top-k logits (`0` = disabled) |
| `top_p` | Nucleus sampling threshold (`0.0` = disabled) |
| `termination_id` | Token id that stops generation (commonly the EOD token) |
| `stop_words` | List of strings that stop generation when produced |
| `detokenize_stop_sequence` | Keep the stop word and EOD in `generated_text` |
| `return_log_probs` | Return prompt and generated log-probs |
| `skip_prompt_log_probs` | Skip prompt log-probs (only generated) |
| `top_n_logprobs` | Return top-N log-probs per position |
| `return_segments` | Return per-token detokenized segments |
| `return_prompt_tokens` | Echo `prompt_tokens` back in the response. **Defaults to `False`** |
| `add_BOS` | Prepend BOS when tokenizing |
| `streaming`, `streaming_interval` | Emit incremental partial replies. Refer to [Streaming](#streaming) |
| `do_kv_handoff` | Pin KV blocks and publish handoff metadata for a peer decode engine. Refer to [Disaggregated Prefill and Decode](#disaggregated-prefill-and-decode) |

```python
sp = SamplingParams(
    num_tokens_to_generate=256,
    temperature=0.7,
    top_p=0.9,
    return_log_probs=True,        # needed for RL: importance weights / KL
)
```

> **`prompt_tokens` is no longer echoed by default.** The engine drops prompt
> token ids before serializing a finished request, which saves the transmission
> cost for long prompts. `prompt_length` is always reported. Set
> `return_prompt_tokens=True` if your client needs the ids.

> **RL note:** *Prompt* log-probs require every position's logits, so requesting
> `return_log_probs` without `skip_prompt_log_probs` also requires
> `InferenceConfig.materialize_only_last_token_logits=False`. The engine asserts
> on this combination. If you only need generated log-probs, set
> `skip_prompt_log_probs=True` and leave `materialize_only_last_token_logits` at
> its default `True`, which is cheaper.

`InferenceConfig.logprobs_mode` controls *which* log-probs you get:
`'raw_logprobs'` (default) returns the unmodified model log-probs, while
`'processed_logprobs'` returns log-probs after temperature, top-k, and top-p have
been applied. `'processed_logprobs'` is not yet supported with speculative
decoding.

### Engine Configuration

`InferenceConfig` configures the engine, KV-cache, and CUDA-graph behavior and is
where most features are turned on. Construct it directly, or derive it from
model and CLI args using the function
`megatron.inference.utils.get_inference_config_from_model_and_args`. Frequently
used fields:

| Field | Purpose |
|---|---|
| `max_sequence_length` | Max prompt and output length you expect |
| `buffer_size_gb` | On-GPU portion of the shared KV-cache block pool |
| `paused_buffer_size_gb` | Block-retention budget for paused requests |
| `block_size_tokens` | KV-cache block (page) size. MLA requires exactly `64` |
| `max_requests` / `max_tokens` | Caps on concurrent requests or tokens per forward pass |
| `enable_chunked_prefill` | Chunked prefill (piggybacking) |
| `enable_prefix_caching` | Prefix caching, with `prefix_caching_eviction_policy` (`ref_zero` / `lru`), `prefix_caching_coordinator_policy` (`load_balanced` / `longest_prefix` / `first_prefix_block`), and `prefix_caching_routing_alpha` to trade cache affinity against load balance |
| `prefix_caching_mamba_gb` | GPU budget for the Mamba-state prefix cache on hybrid models |
| `num_speculative_tokens` | MTP-based speculative decoding |
| `num_cuda_graphs`, `cuda_graph_max_tokens`, `cuda_graph_all_prefills`, `cuda_graph_sizing_distribution` | CUDA-graph capture controls. `cuda_graph_max_tokens` (512) bounds prefill and mixed graphs; `cuda_graph_all_prefills` extends capture to the full `max_tokens`; the sizing distribution is `exponential` (default) or `linear` |
| `async_sched_mode` | `'async'` (default) or `'legacy'`. Refer to [Async Scheduling](#async-scheduling) |
| `sampling_backend` | `'torch'` (default) or `'flashinfer'` |
| `logprobs_mode` | `'raw_logprobs'` (default) or `'processed_logprobs'` |
| `materialize_only_last_token_logits` | Set `False` when returning *prompt* log-probs |
| `mamba_inference_state_config`, `mamba_memory_ratio` | Hybrid, Mamba, GDN, or GDP model state. The state config also carries the conv and SSM state dtypes |
| `kv_cache_management_mode`, `unified_memory_level`, `static_kv_memory_pointers` | Suspend or resume memory handling (`persist` / `offload` / `recompute`). `static_kv_memory_pointers` keeps captured CUDA graphs valid across a suspend/resume cycle so they do not need recapture |
| `offset_sampling_seed_by_dp_rank` | Give each DP rank a distinct sampling seed (default `True`), so the same prompt routed to different replicas produces different samples |
| `image_preprocessing_config` | Image preprocessing for vision-language models |
| `use_flashinfer_fused_rope` | Use FlashInfer's fused RoPE kernel |
| `disable_ep_consensus`, `ep_consensus_interval`, `use_synchronous_zmq_collectives` | MoE and expert-parallel coordination tuning |
| `track_paused_request_events`, `track_generated_token_events`, `metrics_writer`, `logging_step_interval` | Observability |

```python
from megatron.core.inference.config import InferenceConfig

inference_config = InferenceConfig(
    max_sequence_length=4096,
    buffer_size_gb=40,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)
```

Some inference-relevant switches live on the model's `TransformerConfig` rather
than `InferenceConfig`, because they must be set when you build the model:
`batch_invariant_mode` (and `batch_invariant_backend` /
`batch_invariant_collective`), `inference_moe_token_dispatcher_type` (`nvls` by
default, or `nccl`), `inference_grouped_gemm_backend` (`vllm` by default, or
`torch` / `flashinfer`), `moe_enable_routing_replay`, and `window_size` for
sliding-window attention.

### Reading Results

`generate` returns `DynamicInferenceRequest` objects. The most commonly used fields are:

- `generated_text`: Decoded output string
- `generated_tokens`: Output token-ids
- `prompt`: Echoed prompt text
- `prompt_length`: Prompt length in tokens, always reported
- `prompt_tokens`: Prompt token ids, only when `SamplingParams.return_prompt_tokens=True`
- `prompt_log_probs`, `generated_log_probs`: Log-probs (when requested)
- `ttft`: Time-to-first-token (seconds)
- `status`: Terminal request status

### Lifecycle Controls

In *coordinator mode*, you can drive the engine's state machine. This is important
for the RL loop where you alternate generation and training:

- `pause()` / `unpause()` — halt and resume scheduling.
- `suspend()` / `resume()` — offload/reload GPU buffers (KV cache, Mamba
  states). Call `pause()` before `suspend()`.
- `shutdown()` / `wait_for_shutdown()` — tear down or block until the engine
  loop terminates.

These raise `RuntimeError` in direct mode. The context-manager exit calls
`shutdown()` for you.

`suspend()` / `resume()` are also the hook for *weight refit or resharding*
between training and inference: suspend the engine (optionally offloading the
KV cache), refit or reshard the updated weights into the inference parallel layout,
then resume. This is what enables both *colocated* (training and inference on
the same GPUs) and *non-colocated* (separate resources) RL deployments. Refer to
[Weight Refit and Resharding for RL](#weight-refit-and-resharding-for-rl) for the
refit call itself.

---

## Async Scheduling

Async scheduling reorders request processing to prepare the next forward pass
before resolving the previous one, overlapping host-side bookkeeping with GPU
work. It is enabled by default. Select legacy scheduling explicitly when a
feature such as MoE router replay requires it:

```python
from megatron.core.inference.config import AsyncScheduleMode, InferenceConfig

inference_config = InferenceConfig(
    max_sequence_length=4096,
    async_sched_mode=AsyncScheduleMode.LEGACY,   # or the string "legacy"
)
```

The equivalent command-line option is
`--inference-dynamic-batching-async-sched-mode legacy`.

The engine decides per step whether overlapping is profitable, so enabling the
mode does not force overlap on every step. Restrictions to be aware of:

- **MoE router replay is not supported.** Async scheduling and
  `moe_enable_routing_replay` are mutually exclusive and the engine raises if
  both are set.
- **Paused requests are not supported** by the overlapped ordering.
- **Speculative decoding needs one MTP depth per speculative token**, so
  `num_speculative_tokens` may not exceed the model's MTP depth count.

Prompt log-probs work under async scheduling but require materializing every
position's logits, which gives back much of the overlap benefit. The performance
recipes therefore pass `--skip-prompt-log-probs`.

---

## Streaming

Streaming emits partial results as tokens are produced instead of waiting for the
whole completion. It is a per-request setting on `SamplingParams` and requires
coordinator mode.

```python
sp = SamplingParams(num_tokens_to_generate=256, streaming=True, streaming_interval=4)
```

`streaming_interval` is the minimum number of unsent tokens per partial reply
(an integer >= 1); raise it to trade latency for fewer messages.

**Over HTTP**, pass the usual OpenAI field. Both `/v1/completions` and
`/v1/chat/completions` accept `"stream": true` and return an SSE stream
(`text/event-stream`), and both honor `"streaming_interval"` and
`"stream_options": {"include_usage": true}`. Chat streaming emits `content`,
`reasoning_content`, and incremental `tool_calls` deltas. HTTP streaming requires
a Hugging Face *fast* tokenizer; other tokenizers get an HTTP 400.

```bash
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "EMPTY", "stream": true,
       "messages": [{"role": "user", "content": "Write a haiku about GPUs"}]}'
```

**Programmatically**, stream through `InferenceClient.add_request_streaming`,
which returns an async iterator yielding `{"partial": {"request_id", "new_tokens"}}`
frames followed by exactly one `{"final": ...}` frame. `MegatronLLM` and
`MegatronAsyncLLM` do not yet expose a streaming `generate`, so reach the client
through the engine and drive it on the runtime loop with `llm.submit(...)` or
`llm.run_sync(...)`. `InferenceClient` also offers `abort_request(request_id)`.

---

## OpenAI-Compatible HTTP Server

Megatron Inference can serve requests over HTTP using the OpenAI API format. This section explains how to start the server and query it.

`serve(...)` — available on both `MegatronLLM` and `MegatronAsyncLLM` — starts
the HTTP frontend on the primary rank (global rank 0). Serving *requires
coordinator mode* and raises `ValueError` otherwise. The routes are:

| Route | Purpose |
|---|---|
| `/v1/completions` | Text completions, with optional SSE streaming |
| `/v1/chat/completions` | Chat completions, with chat templates, tool calling, reasoning parsers, image inputs, and optional SSE streaming |
| `/v1/health` | Readiness and liveness check |
| `/v1/start_profile`, `/v1/stop_profile` | Relay `cudaProfilerStart`/`cudaProfilerStop` to every engine, to pair with `nsys --capture-range=cudaProfilerApi` |

Each route is also served without the `/v1` prefix. Chat completions report
prefix-cache hits as `usage.prompt_tokens_details.cached_tokens`.

The runnable script is
[`examples/inference/launch_inference_server.py`](../examples/inference/launch_inference_server.py),
with the shell wrapper
[`examples/inference/run_inference_server.sh`](../examples/inference/run_inference_server.sh)
(packaged for a Nemotron-6 3B hybrid MoE config: TP 2, EP 8, PP 1).

```python
import asyncio
from megatron.core.inference.apis import MegatronAsyncLLM, ServeConfig

async def main():
    async with MegatronAsyncLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=inference_config,
        use_coordinator=True,
    ) as llm:
        await llm.serve(
            ServeConfig(host="0.0.0.0", port=5000),
            blocking=True,          # blocks until shutdown
        )

asyncio.run(main())
```

`ServeConfig` fields: `host` (`"0.0.0.0"`), `port` (`5000`), `parsers` (`[]` —
response/reasoning/tool parsers, named by their registry keys:
`deepseek-r1-reasoning`, `nemotron-v3-reasoning`, `qwen3-coder-tool`), `verbose`
(`False` — per-request logging), `frontend_replicas` (`4` — HTTP frontend
processes on the primary rank), and `sock` (`None` — an already-bound listening
socket to use instead of binding `host:port`).
`default_temperature` (`1.0`), `default_top_p` (`1.0`), and `default_top_k`
(`0`) provide sampling defaults for HTTP requests that omit those fields.
`eval_mode` (`False`) switches the frontend to evaluation-oriented response
defaults, avoiding prompt-token transmission unless a request opts in.

The same call works from a synchronous launcher:

```python
from megatron.core.inference.apis import MegatronLLM, ServeConfig

with MegatronLLM(model=model, tokenizer=tokenizer,
                 inference_config=inference_config) as llm:
    llm.serve(ServeConfig(port=5000), blocking=True)
```

To launch the server using the wrapper:

```bash
bash examples/inference/run_inference_server.sh \
    --hf-token <HF_TOKEN> \
    --hf-home /path/to/hf_home \
    --checkpoint /path/to/nemotron-3b-hybrid-moe
```

To verify that the server is ready, verify that you receive the following output:

```
INFO:root:Inference co-ordinator is ready to receive requests!
INFO:hypercorn.error:Running on http://0.0.0.0:5000 (CTRL + C to quit)
```

Then query it with any OpenAI-compatible client. Chat templates, tool calling,
and reasoning parsers are supported.

```bash
# Completions
curl http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "EMPTY", "prompt": "The capital of France is", "max_tokens": 32}'

# Chat completions
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "EMPTY", "messages": [{"role": "user", "content": "Hi!"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="EMPTY",                 # model field is not validated; pass anything
    messages=[{"role": "user", "content": "Write a haiku about GPUs"}],
)
print(resp.choices[0].message.content)
```

> The dynamic server returns `"model": "EMPTY"` and does *not*
> validate the request `model` field. You can pass anything you like. Refer to
> [Known Limitations](#known-limitations).

---

## Weight Refit and Resharding for RL

After a training step, the RL loop has to push the updated policy weights into
the inference engine — potentially across a different parallelism layout, and
potentially across a process boundary. Megatron Core does this through the
resharding module rather than through the inference API:

```python
from megatron.core.resharding.refit import prepare_swap_model_weights, swap_model_weights

# Once, to build and cache the transfer plan.
prepare_swap_model_weights(train_model, inference_model)

# Every rollout: pause + suspend the engine, refit, then resume.
llm.pause()
llm.suspend()
swap_model_weights(train_model, inference_model, refit_method)
llm.resume()
llm.unpause()
```

The transport is selected by `refit_method`, exposed on the command line as
`--refit-method`:

| Backend | Notes |
|---|---|
| `gloo` | Default. CPU-staged copy; the most portable |
| `nccl` | GPU collective copy |
| `nccl_m2n` | NCCL M2N for non-colocated refit, driven by a non-RL launcher |
| `nvshmem` | NVSHMEM copy service |
| `nixl` | NIXL copy service |

MXFP8 targets are handled transparently: when the destination model uses
`--transformer-impl inference_optimized` with `--fp8-recipe mxfp8`,
`prepare_swap_model_weights` installs a quantizing transform that later
`swap_model_weights` calls pick up. The built-in RL loop calls
`swap_model_weights(model, inference_model, args.refit_method)`; refer to
[`megatron/core/resharding/README.md`](../megatron/core/resharding/README.md) for
the plan-building and caching details.

---

## Multimodal (Vision-Language) Inference

Image inputs are supported on the dynamic-batching path. This requires
coordinator mode and a wrapper override, because the high-level API defaults to
`GPTInferenceWrapper`:

```python
from megatron.core.inference.model_inference_wrappers.multimodal.vlm_inference_wrapper import (
    VLMInferenceWrapper,
)

llm = MegatronLLM(
    model=model,
    tokenizer=tokenizer,
    inference_config=inference_config,      # set image_preprocessing_config
    inference_wrapper_cls=VLMInferenceWrapper,
)
results = llm.generate(prompt, sampling_params, multi_modal_data={"image": image_bytes})
```

`multi_modal_data` follows vLLM's modality-dictionary shape; `"image"` accepts
raw bytes, a list of raw bytes, or a preprocessed tensor dictionary. Batched
prompts take one dictionary per prompt.

Over HTTP, `/v1/chat/completions` accepts standard OpenAI multimodal content
parts: `image_url` blocks with either a base64 data URL or a remote `http(s)`
URL. Remote fetches refuse redirects and non-public addresses.

Supported models are LLaVA-style models and Nemotron Omni; the wrapped model must
implement `forward_lm_only`. As an alternative to wiring this up yourself,
`tools/run_dynamic_text_generation_server.py` auto-detects VLM versus GPT from
the checkpoint. Refer to [Known Limitations](#known-limitations) for what is not
yet covered — notably video, audio, and pipeline parallelism.

---

## Disaggregated Prefill and Decode

The building blocks for splitting prefill and decode across separate engines are
in tree. A prefill engine pins its KV blocks and publishes handoff metadata when
a request sets `SamplingParams.do_kv_handoff=True`; a decode engine imports that
state and continues generation. `DisaggDynamicInferenceEngine` is the engine
subclass for both roles, and `setup_kv_transfer(role, backend)` — with `role` of
`"prefill"` or `"decode"` and a `nixl` or `nccl` backend — wires up the
transport. It must be called collectively by every model-parallel rank.

KV state is resharded across mismatched TP and PP layouts between the two pools,
and hybrid models additionally hand off their recurrent conv and SSM state.
`--inference-shards` describes the partitioning, for example
`"tp=2,role=prefill+tp=1,dp=2,role=decode"`.

Two prerequisites: **prefix caching must be enabled on both the prefill and the
decode engine**, and a hybrid decode engine must not set
`prefix_caching_mamba_gb`.

> This is transfer plumbing plus a shard-layout spec, driven today by an external
> control plane (for example Dynamo). Megatron-LM does not yet ship a launcher
> that stands up a disaggregated deployment end to end, and handoff does not
> support log-probs.

---

## Customizing the Pipeline

`MegatronLLM` and `MegatronAsyncLLM` cover most use cases. For more control, you can assemble or subclass the underlying components directly. Common reasons to do this include:

- Implementing step-level scheduling control.
- Adding custom sampling or logit processing.
- Migrating an existing pipeline to Megatron Inference.

### Pipeline Anatomy

`MegatronLLM` and `MegatronAsyncLLM` build the following pipeline for you:

```
DynamicInferenceContext   # KV cache, paging, scheduling/bookkeeping state
        │
GPTInferenceWrapper       # model forward wrapper for inference
        │
TextGenerationController   # tokenize → forward → sample → detokenize
        │
DynamicInferenceEngine     # add_request / step loop, coordinator integration
```

You can reach any of these from a constructed `llm` through `llm.context`,
`llm.controller`, and `llm.engine`. Or build them explicitly, which is exactly
what `MegatronLLM` and `MegatronAsyncLLM` do internally:

```python
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.engines import DynamicInferenceEngine

context = DynamicInferenceContext(model.config, inference_config)
wrapped_model = GPTInferenceWrapper(model, context)
controller = TextGenerationController(wrapped_model, tokenizer)
engine = DynamicInferenceEngine(controller, context)
```

### Customizing the `TextGenerationController`

The `TextGenerationController` manages tokenization, the forward pass, sampling, and detokenization. To inject custom behavior, subclass it and pass your instance to the engine.

Override these methods to customize the pipeline:

- `sample_from_logits(...)`: custom sampling or logit processing (constrained
  decoding, custom penalties, grammar masks).
- `tokenize_prompt(...)` / `detokenize_generations(...)`: custom
  tokenization or detokenization.
- `generate_output_tokens_dynamic_batch(...)`: custom batch forward-step
  integration.

```python
class MyController(TextGenerationController):
    def sample_from_logits(self, last_token_logits, sampling_params, *args, **kwargs):
        # apply a custom logit bias, then defer to the base sampler
        last_token_logits = last_token_logits + my_logit_bias
        return super().sample_from_logits(last_token_logits, sampling_params, *args, **kwargs)

controller = MyController(wrapped_model, tokenizer)
engine = DynamicInferenceEngine(controller, context)
```

### Customizing the `DynamicInferenceContext`

The `DynamicInferenceContext` holds the KV cache, paging, and the
scheduling and bookkeeping state. For hybrid and SSM models it also manages the
recurrent state alongside the attention KV cache, that is sized using the
`mamba_inference_state_config` and `mamba_memory_ratio`. Mamba, Gated Delta Net,
and Gated Delta Product layers all share this one slot-indexed recurrent-state
cache, which is why a single model may not mix Mamba and GDN layers. Refer to
[Known Limitations](#known-limitations) for the per-mixer feature gaps.

For MLA models the context stores compressed latents rather than full K and V,
which is why `cache_mla_latents=True` and a block size of exactly 64 are
required.

Configure it through `InferenceConfig`, which controls buffer size, block size,
prefix caching, chunked prefill, CUDA graphs, suspend and resume memory mode,
and recurrent state. Refer to [Engine configuration](#engine-configuration).

To customize KV-cache layouts, eviction policies, or scheduling logic, subclass the context and pass it into the wrapper and engine.

### Driving the Engine Directly

For full step-level control, skip `generate` and drive the engine's
`add_request` and `step_modern` loops yourself. This is how you implement custom
arrival schedules, batch-drain modes, or suspend and resume policies:

```python
engine.add_request(request_id, prompt_text, sampling_params)
while engine.has_unfinished_requests():
    result = engine.step_modern()
    for record in result["finished_request_records"]:
        finished = record.merge()
        print(finished.request_id, finished.generated_text)
```

The fully worked manual-stepping example is
[`examples/inference/advanced/gpt_dynamic_inference.py`](../examples/inference/advanced/gpt_dynamic_inference.py).
It demonstrates arrival scheduling, batch-drain, suspend and resume, CUDA-graph
bucketing, log-probs, and JSON dumping. For explicit coordinator with `InferenceClient`
lifecycle management, refer to
[`gpt_dynamic_inference_with_coordinator.py`](../examples/inference/advanced/gpt_dynamic_inference_with_coordinator.py).

---

## Examples Directory

Everything above is runnable from
[`examples/inference/`](../examples/inference/):

| Path | Description |
|---|---|
| [`offline_inference.py`](../examples/inference/offline_inference.py) | Batched offline generation through the high-level API. Covers all three supported mode combinations using `--mode sync|async` and `--use-coordinator`. |
| [`run_offline_inference.sh`](../examples/inference/run_offline_inference.sh) | Shell wrapper for a Qwen 2.5-1.5B offline-inference config. |
| [`launch_inference_server.py`](../examples/inference/launch_inference_server.py) | OpenAI-compatible HTTP server using `MegatronAsyncLLM.serve(...)`. |
| [`run_inference_server.sh`](../examples/inference/run_inference_server.sh) | Shell wrapper for a Nemotron-6 3B hybrid-MoE server config. |
| [`utils.py`](../examples/inference/utils.py) | Shared helpers including `Request`, `build_requests`, output formatting, and JSON dump. |
| [`advanced/gpt_dynamic_inference.py`](../examples/inference/advanced/gpt_dynamic_inference.py) | Manual `add_request`/`step_modern` stepping. |
| [`advanced/gpt_dynamic_inference_with_coordinator.py`](../examples/inference/advanced/gpt_dynamic_inference_with_coordinator.py) | Explicit coordinator and `InferenceClient` lifecycle. |

Run the offline example across modes:

```bash
# sync + direct (defaults)
bash examples/inference/run_offline_inference.sh \
    --hf-token <HF_TOKEN> --checkpoint /path/to/qwen-1.5b

# sync + coordinator
bash examples/inference/run_offline_inference.sh \
    --hf-token <HF_TOKEN> --checkpoint /path/to/qwen-1.5b --use-coordinator

# async + coordinator
bash examples/inference/run_offline_inference.sh \
    --hf-token <HF_TOKEN> --checkpoint /path/to/qwen-1.5b --mode async --use-coordinator
```

All supported modes produce numerically identical generated text. Note that the
example script's `--use-coordinator` flag defaults to *off* (direct mode), which
is the opposite of the `MegatronLLM` constructor default.

---

## Known Limitations

**Model architecture gaps**

- **Vision-language models cover images only.** Video and audio have no supported
  preprocessing or modeling format and raise `NotImplementedError`. VLM dynamic
  inference also requires `PP=1` (pipeline and virtual pipeline parallelism both
  raise), and in-core static-tiling preprocessing was removed — clients needing
  static tiling must submit a preprocessed tensor payload.
- **MLA requires a specific configuration.** `cache_mla_latents=True`, a KV block
  size of exactly 64, and the `flash_mla` package. RoPE fusion, sliding-window
  attention, Flash Decoding, and the Triton KV-append fast path are all
  incompatible with the latent cache.
- **GDN2 is not supported.** `experimental_attention_variant='gdn2'` raises
  `NotImplementedError`. GDN is also dynamic-batching only; static batching
  raises.
- **A model may not mix Mamba and GDN layers**, because the recurrent-state cache
  and prefill metadata use one shared shape and chunk size.
- **Attention-free stacks are not supported.** A pipeline stage holding zero
  attention layers raises.
- **Prefix caching is limited for the Gated Delta variants.** GDN rejects
  `enable_prefix_caching` outright. GDP allows KV-block prefix caching but not
  recurrent-state caching, so setting `prefix_caching_mamba_gb` on a GDP model
  raises. GDN additionally does not support chunked prefill (GDP does), and
  neither supports speculative decoding, batch-invariant mode, or context
  parallelism.

**Feature interactions**

- **Sequence parallelism is rejected for dynamic batching** unless both `TP > 1`
  and `EP > 1`. This is easy to trip by reusing a training config that sets
  `--sequence-parallel`.
- **The FlashInfer sampling backend never runs under CUDA graphs.** Its kernel
  choice is data-dependent and it bakes the RNG state into a capture as a
  by-value constant, so the sampler always runs eagerly. That is a deliberate
  correctness trade-off against sampling-step latency.
- **`logprobs_mode='processed_logprobs'` is incompatible with speculative
  decoding.**
- **`sampling_backend='flashinfer'` silently falls back to `'torch'`** with a
  warning if FlashInfer is not installed.
- **Async scheduling excludes MoE router replay** and does not support paused
  requests. Refer to [Async Scheduling](#async-scheduling).
- **Batch-invariant MoE is bf16-only** and requires the unfused
  permute/unpermute path; batch-invariant mode in general excludes context
  parallelism and attention dropout.
- **MXFP8 fused quantization supports squared-ReLU only, not SwiGLU**, which
  falls back to bf16.
- **Disaggregated handoff does not support log-probs** (`return_log_probs` or
  `top_n_logprobs > 0` raises).

**Engine and serving**

- **`engine.reset()` is unsafe in coordinator mode.** It can deadlock (rebinds
  internal asyncio primitives that suspended waiters still reference) or
  silently re-route to direct-mode branches. The offline example therefore
  blocks `--inference-repeat-n > 1` together with `--use-coordinator`. Direct-mode
  reset is safe.
- **HTTP frontend is fixed to global rank 0.** There is no per-rank `role`
  override on `ServeConfig`. Control placement through the launcher (for example, torchrun
  rank-0 placement). `ServeConfig.sock` lets you pre-bind the listening socket,
  but it still only takes effect on rank 0.
- **Server returns `"model": "EMPTY"`.** The HTTP frontend doesn't echo or
  validate a configured model name and exposes no `GET /v1/models` endpoint.
  Clients may pass any `model` value. It is ignored.
- **HTTP streaming requires a Hugging Face fast tokenizer**; other tokenizers get
  an HTTP 400.
- **Streaming is not on the high-level API.** `MegatronLLM` and
  `MegatronAsyncLLM` have no streaming `generate`; stream through
  `InferenceClient` or the HTTP frontend.
- **Test coverage is uneven across the newer architectures.** The
  dynamic-inference functional suites cover GPT, MoE, and Mamba hybrid models.
  GDN and GDP have dynamic-inference unit tests only, and MLA dynamic inference
  has neither — its coverage is at the attention-layer level.

---

## Roadmap and Future Work

**API and serving:**

- **`megatron serve` CLI** — a single-binary launcher mirroring `vllm serve`,
  with single-node and multi-node or headless modes.
- **Config-based model construction** — `MegatronLLM(model="...")` with model
  recipes and checkpoint resolution. Use to remove manual model building.
- **Streaming on the high-level API** — a streaming `generate` on `MegatronLLM`
  and `MegatronAsyncLLM`, so streaming does not require dropping to
  `InferenceClient`.
- **Simplified inference API** overall.

**Models and performance:**

- **Turnkey disaggregated inference** — a launcher and control plane on top of
  the KV and SSM handoff primitives that exist today, plus log-prob support
  across a handoff.
- **FlashInfer integration** for attention and Mamba kernels (sampling, fused
  RoPE, and MoE grouped GEMM are already integrated).
- **All2Allv-based token dispatcher** for MoE.
- **Large-scale inference optimizations** (large models and long sequences).
- **Low-precision numerics** for KV cache and Mamba state. MXFP8 currently covers
  weights; the KV cache and recurrent state are still bf16, fp16, or fp32.
- **Broadening the newer architectures** — prefix caching and speculative
  decoding for Gated Delta variants, chunked prefill for GDN, pipeline
  parallelism for VLM, and functional-test coverage for MLA, VLM, GDN, and GDP.

---

## Additional Resources

- API reference and mental model documentation: [`megatron/core/inference/README.md`](../megatron/core/inference/README.md)
- Examples overview: [`examples/inference/README.md`](../examples/inference/README.md)
- Low-level engine source: [`megatron/core/inference/`](../megatron/core/inference/)
- High-level API source: [`megatron/core/inference/apis/`](../megatron/core/inference/apis/)
- Weight refit and resharding: [`megatron/core/resharding/README.md`](../megatron/core/resharding/README.md)
- MoE router replay: [`docs/api-guide/router_replay.md`](api-guide/router_replay.md)
- MoE routing trace tooling: [`examples/inference/README.md`](../examples/inference/README.md)
- Functional tests: `tests/functional_tests/test_cases/gpt/gpt_offline_inference_*`, `gpt_inference_server_smoke_*`, `gpt_dynamic_inference_*`, `tests/functional_tests/test_cases/hybrid/hybrid_dynamic_inference_*`
- Unit tests: `tests/unit_tests/inference/`
