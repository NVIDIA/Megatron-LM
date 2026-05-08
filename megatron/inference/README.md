# Megatron Inference (High-Level API)

High-level entry points over the `megatron.core.inference` dynamic
engine. Hides `DynamicInferenceContext` + `GPTInferenceWrapper` +
`TextGenerationController` + `DynamicInferenceEngine` construction,
coordinator startup, and the per-instance background asyncio runtime behind
two top-level classes: `MegatronLLM` (sync) and `MegatronAsyncLLM` (async,
with HTTP serving via `serve()`).

Use this package when you want the typical `llm.generate(prompts, ...)`
ergonomic. Drop down to `megatron.core.inference` directly when you need
manual `add_request` / `step_modern` control or step-level scheduling.

## Quickstart

### Offline batch (sync)

```python
from megatron.inference import MegatronLLM, SamplingParams

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
from megatron.inference import MegatronAsyncLLM, ServeConfig

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

## Mental model

| Class × `use_coordinator` | Use case |
|---|---|
| `MegatronLLM`, direct (default) | Offline batch on ranks the caller manages (DP sharding owned by user). Blocking. |
| `MegatronLLM`, coordinator | Same offline workload with engine-managed DP routing + `pause`/`suspend`/`resume` lifecycle. |
| `MegatronAsyncLLM`, direct | Same as sync direct but `await`-able. Single-caller in direct mode (concurrent `generate` raises). |
| `MegatronAsyncLLM`, coordinator | Required for `serve()` and for RL-style persistent generators. |

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

## Known limitations

- **`MegatronAsyncLLM.generate()` blocks the caller's event loop in direct mode.** The engine call is sync and inline; it does not yield back while running. Acceptable for offline batched calls; degraded for server/RL workloads that interleave generation with other async work. Tracked for an upstream `engine.async_generate(...)`.

- **`llm.engine.reset()` is unsafe in coordinator mode.** Two failure modes, both upstream in `dynamic_engine.py`:
  - *Deadlock*: `reset()` *rebinds* (does not mutate in-place) `_cond` / `_state_events`. Any coroutine on the engine-loop task that is `await`ing one of those primitives holds a reference to the OLD object in its suspended frame. Subsequent `notify_all()` / `set()` calls hit the NEW objects, leaving the suspended waiter stranded; the next `generate()` hangs.
  - *Silent corruption*: `reset()` also sets `self.use_coordinator = False`, which silently re-routes failed-request handling, scheduling notification, and `suspend()`'s state machine to direct-mode branches. Outcome: not-a-hang but wrong behavior, harder to diagnose.
  - The example `offline_inference.py` blocks `--inference-repeat-n > 1` with `--use-coordinator` for these reasons. Direct-mode reset is safe.

- **Async-direct `generate()` is single-caller.** Concurrent `await llm.generate(...)` (e.g. via `asyncio.gather`) in direct mode raises `RuntimeError`. Pass batched prompts instead, or switch to coordinator mode.

## See also

- Examples: [`examples/inference/offline_inference.py`](../../examples/inference/offline_inference.py) (4 modes via `--mode` / `--use-coordinator`), [`examples/inference/launch_inference_server.py`](../../examples/inference/launch_inference_server.py) (HTTP server).
- Low-level engine: [`megatron/core/inference/`](../core/inference/).
