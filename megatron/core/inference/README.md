# Megatron Inference

Use `MegatronLLM` (sync) or `MegatronAsyncLLM` (async, with HTTP serving via `serve()`) for typical inference workflows. Both classes hide the underlying engine pipeline (`DynamicInferenceContext` + `GPTInferenceWrapper` + `TextGenerationController` + `DynamicInferenceEngine`) and provide a vLLM-style `generate(prompts, sampling_params)` API. Choose **direct mode** (`use_coordinator=False`) when you manage data sharding yourself; **coordinator mode** (`use_coordinator=True`) when you want the engine to route requests across data-parallel replicas (required for HTTP serving).

## Quickstart

### Offline batch (sync)

```python
from megatron.core.inference.apis import MegatronLLM, SamplingParams

# Caller owns initialize_megatron(...), model construction, and model.eval().
# See examples/inference/offline_inference.py for a runnable end-to-end script.
with MegatronLLM(
    model=model,
    tokenizer=tokenizer,
    inference_config=inference_config,
    use_coordinator=False,
) as llm:
    results = llm.generate(
        ["Megatron inference is", "Hello, world"],
        SamplingParams(num_tokens_to_generate=64),
    )
    for r in results:
        print(r.generated_text)
```

### OpenAI-compatible HTTP server

```python
import asyncio
from megatron.core.inference.apis import MegatronAsyncLLM, ServeConfig

async def main():
    async with MegatronAsyncLLM(
        model=model,
        tokenizer=tokenizer,
        inference_config=inference_config,
        use_coordinator=True,            # serve() requires coordinator mode
    ) as llm:
        await llm.serve(ServeConfig(host="0.0.0.0", port=5000))  # blocks until shutdown

asyncio.run(main())
```

## Public API

| Symbol | Purpose |
|---|---|
| `MegatronLLM` | Sync entry. Methods: `generate`, `pause`/`unpause`/`suspend`/`resume`, `shutdown`/`wait_for_shutdown`. Properties: `engine`, `context`, `controller`, `is_primary_rank`. Context-manager protocol. |
| `MegatronAsyncLLM` | Async-flavored equivalent. Adds `serve(serve_config, blocking=True)` for HTTP. |
| `ServeConfig` | Dataclass for the HTTP frontend. Fields: `host` (`"0.0.0.0"`), `port` (`5000`), `parsers` (`[]`), `verbose` (`False`), `frontend_replicas` (`4`). |
| `SamplingParams`, `DynamicInferenceRequest`, `DynamicInferenceRequestRecord` | Re-exports from `megatron.core.inference`. |

## Caller responsibilities

- Call `initialize_megatron(...)` (full Megatron distributed setup) BEFORE construction.
- Call `model.eval()` BEFORE construction. The class does not toggle model state.
- Lifecycle methods (`pause`/`unpause`/`suspend`/`resume`) require `use_coordinator=True`; they raise `RuntimeError` in direct mode.

## Future roadmap

Planned new features:

- **Dynamic streaming.** Offline streaming via `engine.async_step()`; HTTP streaming requires extending the coordinator / `InferenceClient` protocol to carry partial outputs (not just final request records).

- **Weight update APIs.** `suspend_for_refit()`, `update_weights_from_collective()`, `resume_after_refit()` wrapping the existing resharding/refit primitives for RL workflows where weights swap between rollout steps.

- **`megatron serve` CLI.** Single-binary launcher reusing `MegatronAsyncLLM.serve(...)`, with single-node and multi-node / headless modes — mirrors `vllm serve`.

- **Config-based model construction.** `MegatronLLM(model="...")` style with model recipes and checkpoint resolution, removing manual model building from caller responsibilities.

## Known limitations

- **`MegatronAsyncLLM` requires `use_coordinator=True`** -- constructing with `use_coordinator=False` raises `ValueError` at `__init__`. The underlying `DynamicInferenceEngine` caches its loop reference at construction time and binds internal asyncio primitives (`_cond`, `_state_events`) to it. Coordinator mode rebinds those to a dedicated daemon-thread loop via `start_listening_to_data_parallel_coordinator`; direct mode has no such rebinding, so the synchronous `engine.generate()` path collides with the caller's running asyncio loop and raises `RuntimeError: This event loop is already running`. Use `MegatronLLM` for sync direct/coordinator workflows. Tracked for an upstream `engine.async_generate(...)` (or engine loop-rebinding) fix that would let `MegatronAsyncLLM` support direct mode.

- **`llm.engine.reset()` is unsafe in coordinator mode.** Two failure modes, both upstream in `dynamic_engine.py`:
  - *Deadlock*: `reset()` *rebinds* (does not mutate in-place) `_cond` / `_state_events`. Any coroutine on the engine-loop task that is `await`ing one of those primitives holds a reference to the OLD object in its suspended frame. Subsequent `notify_all()` / `set()` calls hit the NEW objects, leaving the suspended waiter stranded; the next `generate()` hangs.
  - *Silent corruption*: `reset()` also sets `self.use_coordinator = False`, which silently re-routes failed-request handling, scheduling notification, and `suspend()`'s state machine to direct-mode branches. Outcome: not-a-hang but wrong behavior, harder to diagnose.
  - The example `offline_inference.py` blocks `--inference-repeat-n > 1` with `--use-coordinator` for these reasons. Direct-mode reset is safe.

- **Async-direct `generate()` is single-caller.** Concurrent `await llm.generate(...)` (e.g. via `asyncio.gather`) in direct mode raises `RuntimeError`. Pass batched prompts instead, or switch to coordinator mode.

- **HTTP frontend is fixed to global rank 0.** There is no per-rank `role` override on `ServeConfig` to host the HTTP server on a non-rank-0 rank or to opt a rank out of HTTP. Control placement via the launcher (e.g., torchrun rank-0 placement), mirroring how vLLM's `--headless` is invoked today.

- **Server returns `"model": "EMPTY"`.** The HTTP frontend doesn't expose a `ServeConfig.model_name` to echo in `/v1/completions` / `/v1/chat/completions` responses, doesn't validate the request `model` field against a configured name, and exposes no `GET /v1/models` discovery endpoint. Clients can still pass any `model` in their request body — the dynamic server ignores it.

## Low-level APIs

For step-level control, custom forward-step integration, or migration from existing pipelines, drop down to the building blocks in this directory: `DynamicInferenceEngine` (manual `add_request` / `step_modern` stepping), `DynamicInferenceContext`, `TextGenerationController`, and the model inference wrappers under `model_inference_wrappers/`. Runnable examples live in [`examples/inference/advanced/`](../../examples/inference/advanced/): `gpt_dynamic_inference.py` (manual stepping), `gpt_dynamic_inference_with_coordinator.py` (explicit coordinator + `InferenceClient` lifecycle), `gpt_static_inference.py` (static engine), and `simple_t5_batch_inference.py` (T5).

## See also

- Examples: [`examples/inference/offline_inference.py`](../../examples/inference/offline_inference.py) (4 modes via `--mode` / `--use-coordinator`), [`examples/inference/launch_inference_server.py`](../../examples/inference/launch_inference_server.py) (HTTP server).
