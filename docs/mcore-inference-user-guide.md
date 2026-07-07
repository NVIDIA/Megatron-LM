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
- [OpenAI-Compatible HTTP Server](#openai-compatible-http-server)
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
  generation, providing *day-0 inference* for any model that is trainable in Megatron
  Core.
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

Megatron Inference is optimized for the generation (rollout) phase of the RL
loop. Its rollout performance is *on par with popular inference frameworks*,
so you get the training and inference consistency benefits of staying in MCore
without giving up generation speed.

The plots below show a sample comparison of decode step times against vLLM
during rollouts (lower is better). The two engines track each other closely
across batch sizes, with MCore comparable or slightly faster at larger batch
sizes:

<!-- TODO: These decode-step plots are pre-async-scheduling; refresh them once
async scheduling is merged, and add a prefill-perf analysis section. -->
<!-- TODO: Add a "Benchmark setup" note documenting the versions benched with
(vLLM version, MCore commit/version, GPU/hardware, model sizes). -->

<img src="images/inference_performance/ultra-performance.png" alt="Sample rollout decode step times — Nemotron 3 Ultra" width="600">

<img src="images/inference_performance/super-performance.png" alt="Sample rollout decode step times — Nemotron 3 Super" width="600">

You do not trade away rollout performance to gain the training and inference consistency benefits of MCore inference.

---

## Supported Features

| Area | Features |
|---|---|
| **Batching** | Dynamic or in-flight batching with vectorized bookkeeping, dynamic suspend and resume, and request eviction for high input-rate regimes |
| **Chunked prefill** | Chunked-prefill scheduling with decode piggybacking, so long prompts don't stall in-flight decodes |
| **Attention and KV cache** | Optimized PagedAttention with prefix caching (LRU and ref-zero eviction, prefix-aware coordinator routing) |
| **CUDA graphs** | Full-model CUDA graphs for prefill, decode, and mixed batches |
| **Speculative decoding** | Multi-Token Prediction (MTP)-based speculative decoding (with fused MTP bookkeeping and MTP CUDA graphs) |
| **Serving** | OpenAI-compatible HTTP server with chat templates, tool calling, and reasoning parsers |
| **MoE** | Expert model parallelism with full CUDA-graph support, expert router replay, NVLS switch-multicast token dispatcher (notably faster than the all-to-all dispatchers other frameworks use) plus an allgatherv dispatcher optimized for multi-node NVLink, and shared-expert overlap with latent MoEs |
| **Parallelism** | Data-parallel coordinator with full multi-node support, tensor model parallelism with low-latency comm primitives, and expert model parallelism |
| **Model families** | GPT-style dense models, MoE models, and Mamba and hybrid (SSM and attention) models |
| **Precision** | Low-precision functionality (for example, MXFP8) using latency-optimized inference kernels |
| **RL** | Weight refit and resharding between training and inference, supporting both colocated (shared GPUs) and non-colocated (separate resources) deployments. Batch-invariant kernels for training and inference log-prob consistency |
| **Sampling** | Temperature, top-k, top-p, stop words, log-probs, and top-N log-probs. Pluggable torch or FlashInfer sampling backend |

> **Batch-invariant kernels (training and inference log-prob consistency).** Standard
> GEMM, attention, and norm kernels can produce slightly different numerics depending
> on batch composition, which shows up as log-prob mismatch between training and
> inference. This mismatch is a real source of error and instability in RL. Megatron Inference
> offers *batch-invariant kernels* whose outputs do not depend on how requests are
> batched, so per-token log-probs match between the training and inference forward
> passes. *This is currently supported only for non-MoE (dense) models.* Note this
> is enabled through `batch_invariant_mode` on the model's `TransformerConfig` — it
> is *not* an `InferenceConfig` field, so it must be set when you build the model,
> not on the engine config.

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
    MegatronAsyncLLM,   # async + HTTP serving
    SamplingParams,
    ServeConfig,
)
```

### The two classes: `MegatronLLM` and `MegatronAsyncLLM`

| Class | Use it when | Key methods |
|---|---|---|
| **`MegatronLLM`** | Synchronous offline batch generation (the common RL-rollout case). | `generate`, `pause`/`unpause`/`suspend`/`resume`, `shutdown`/`wait_for_shutdown`; context manager (`with ... as llm:`) |
| **`MegatronAsyncLLM`** | Asyncio-native generation and *HTTP serving* through `serve(...)`. | `async generate`, async lifecycle controls, `serve(serve_config)`; async context manager (`async with ... as llm:`) |

Both expose the underlying building blocks as read-only properties. Use these for [advanced customization](#customizing-the-pipeline):

- `llm.engine`
- `llm.context`
- `llm.controller`
- `llm.is_primary_rank`

**Caller responsibilities (before construction):**

- Call `initialize_megatron(...)` to perform full Megatron distributed setup.
- Build the model and call `model.eval()`. The API does *not* toggle model
  state.
- Have a tokenizer ready.

### Direct Mode Compared to Coordinator (Indirect) Mode

Megatron Inference supports two operating modes. Direct mode is simpler but limited. Coordinator mode adds a routing layer that enables serving, expert parallelism, and lifecycle controls.

#### Direct Mode (`use_coordinator=False`)

Direct mode is the simplest configuration for offline batch generation:

- *Every rank is treated as primary* and runs the engine synchronously.
- *You own data sharding*, which means that you decide the prompts that are assigned to which
  data-parallel replica and call `generate` on each.
- The simplest path for offline batch generation when you already shard the data
  yourself (typical for many RL rollout setups).
- Lifecycle controls (`pause`/`suspend`/...) are *not available* and raise
  `RuntimeError`.
- *Not allowed with expert parallelism* (`EP > 1`). This is because EP routing requires the
  coordinator.

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
- *Required* for: HTTP serving (`serve`), expert parallelism (`EP > 1`), and
  the lifecycle controls (`pause`/`unpause`/`suspend`/`resume`).
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

| | Direct (`use_coordinator=False`) | Coordinator (`use_coordinator=True`) |
|---|---|---|
| Data sharding | You handle it | Coordinator routes across DP |
| `generate` callable on | Every rank | Primary rank (rank 0) only |
| HTTP `serve()` | ❌ | ✅ |
| Expert parallelism (EP > 1) | ❌ | ✅ |
| `pause`/`suspend`/`resume` | ❌ | ✅ |
| `MegatronAsyncLLM` | ❌ | ✅ |

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
| `temperature` | Softmax temperature (`1.0` = unmodified) |
| `top_k` | Keep top-k logits (`0` = disabled) |
| `top_p` | Nucleus sampling threshold (`0.0` = disabled) |
| `termination_id` | Token id that stops generation (commonly the EOD token) |
| `stop_words` | List of strings that stop generation when produced |
| `return_log_probs` | Return prompt and generated log-probs |
| `skip_prompt_log_probs` | Skip prompt log-probs (only generated) |
| `top_n_logprobs` | Return top-N log-probs per position |
| `add_BOS` | Prepend BOS when tokenizing |

```python
sp = SamplingParams(
    num_tokens_to_generate=256,
    temperature=0.7,
    top_p=0.9,
    return_log_probs=True,        # needed for RL: importance weights / KL
)
```

> **RL note:** For log-probs to be materialized correctly, set
> `InferenceConfig.materialize_only_last_token_logits=False` when you request
> `return_log_probs`.

### Engine Configuration

`InferenceConfig` configures the engine, KV-cache, and CUDA-graph behavior and is
where most features are turned on. Construct it directly, or derive it from
model and CLI args using the function
`megatron.inference.utils.get_inference_config_from_model_and_args`. Frequently
used fields:

| Field | Purpose |
|---|---|
| `max_sequence_length` | Max prompt and output length you expect |
| `buffer_size_gb` | GPU memory reserved for the KV cache |
| `block_size_tokens` | KV-cache block (page) size |
| `max_requests` / `max_tokens` | Caps on concurrent requests or tokens per forward pass |
| `enable_chunked_prefill` | Chunked prefill (piggybacking) |
| `enable_prefix_caching` | Prefix caching and `prefix_caching_eviction_policy` or `prefix_caching_coordinator_policy` |
| `num_speculative_tokens` | MTP-based speculative decoding |
| `num_cuda_graphs`, `cuda_graph_*` | CUDA-graph capture controls |
| `sampling_backend` | `'torch'` (default) or `'flashinfer'` |
| `mamba_inference_state_config`, `mamba_memory_ratio` | Hybrid or Mamba model state |
| `kv_cache_management_mode`, `unified_memory_level` | Suspend or resume memory handling (`persist` / `offload` / `recompute`) |

```python
from megatron.core.inference.config import InferenceConfig

inference_config = InferenceConfig(
    max_sequence_length=4096,
    buffer_size_gb=40,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)
```

### Reading Results

`generate` returns `DynamicInferenceRequest` objects. The most commonly used fields are:

- `generated_text`: Decoded output string
- `generated_tokens`: Output token-ids
- `prompt` / `prompt_tokens`: Echoed prompt text and token ids
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
the same GPUs) and *non-colocated* (separate resources) RL deployments.

---

## OpenAI-Compatible HTTP Server

Megatron Inference can serve requests over HTTP using the OpenAI API format. This section explains how to start the server and query it.

`MegatronAsyncLLM.serve(...)` starts the HTTP frontend on the primary rank
(global rank 0), exposing `/v1/completions` and `/v1/chat/completions`.
Serving *requires coordinator mode*.

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
response/reasoning/tool parsers), `verbose` (`False` — per-request logging),
`frontend_replicas` (`4` — HTTP frontend processes on the primary rank).

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
recurrent Mamba (SSM) state alongside the attention KV cache, that is sized using the
`mamba_inference_state_config` and `mamba_memory_ratio`.

Gated delta-net (GDN)
layers are not supported in inference. Refer to
[Known Limitations](#known-limitations).

Configure it through `InferenceConfig`, which controls buffer size, block size,
prefix caching, chunked prefill, CUDA graphs, suspend and resume memory mode,
and Mamba/SSM state. Refer to [Engine configuration](#engine-configuration).

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

All supported modes produce numerically identical generated text.

---

## Known Limitations

- **MLA models are not supported.**
- **Vision-language (VLM) / multimodal models are not supported.** Only
  text-in/text-out generation is supported today.
- **Gated delta-net (GDN) layers are not yet supported in inference.** The
  dynamic context raises `NotImplementedError` if a model contains GDN layers.
  Mamba (SSM) and attention hybrid layers are supported.
- **`engine.reset()` is unsafe in coordinator mode.** It can deadlock (rebinds
  internal asyncio primitives that suspended waiters still reference) or
  silently re-route to direct-mode branches. The offline example therefore
  blocks `--inference-repeat-n > 1` together with `--use-coordinator`. Direct-mode
  reset is safe.
- **HTTP frontend is fixed to global rank 0.** There is no per-rank `role`
  override on `ServeConfig`. Control placement through the launcher (for example, torchrun
  rank-0 placement).
- **Server returns `"model": "EMPTY"`.** The HTTP frontend doesn't echo or
  validate a configured model name and exposes no `GET /v1/models` endpoint.
  Clients may pass any `model` value. It is ignored.

Refer to [`megatron/core/inference/README.md`](../megatron/core/inference/README.md)
for the detailed root-cause notes behind each limitation.

---

## Roadmap and Future Work

**API and serving:**

- **Dynamic streaming** — offline streaming through `engine.async_step()` and HTTP
  streaming of partial outputs.
- **Weight-update APIs** — `suspend_for_refit()`,
  `update_weights_from_collective()`, and `resume_after_refit()` wrapping the
  resharding or refit primitives for RL weight swaps between rollout steps, across
  both colocated and non-colocated deployments.
- **`megatron serve` CLI** — a single-binary launcher mirroring `vllm serve`,
  with single-node and multi-node or headless modes.
- **Config-based model construction** — `MegatronLLM(model="...")` with model
  recipes and checkpoint resolution. Use to remove manual model building.
- **Simplified inference API** overall.

**Models and performance:**

- **Disaggregated inference** (prefill and decode separation).
- **FlashInfer integration** for attention and Mamba kernels (sampling is already
  integrated).
- **Async dynamic context update** — moves bookkeeping off the critical path.
- **All2Allv-based token dispatcher** for MoE.
- **Large-scale inference optimizations** (large models and long sequences).
- **Low-precision numerics** for KV cache and Mamba state.
- **Router-Replay** for reducing mismatch between inference and training for MoE models.

---

## Additional Resources

- API reference and mental model documentation: [`megatron/core/inference/README.md`](../megatron/core/inference/README.md)
- Examples overview: [`examples/inference/README.md`](../examples/inference/README.md)
- Low-level engine source: [`megatron/core/inference/`](../megatron/core/inference/)
- High-level API source: [`megatron/core/inference/apis/`](../megatron/core/inference/apis/)
- Functional tests: `tests/functional_tests/test_cases/gpt/gpt_offline_inference_*`, `gpt_inference_server_smoke_*`
- Unit tests: `tests/unit_tests/inference/high_level_api/`
