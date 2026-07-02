### Megatron Core Inference Examples

The runnable inference examples in this directory drive the high-level
inference API in `megatron/core/inference/apis/` (`MegatronLLM` for sync,
`MegatronAsyncLLM` for async + HTTP serving).

**For all documentation — supported features, basic and advanced usage, the
direct vs. coordinator modes, the OpenAI-compatible server, and how to run
these examples — see the user guide:**

➡️ [`docs/mcore-inference-user-guide.md`](../../docs/mcore-inference-user-guide.md)

#### What's in here

- **`offline_inference.py`** — batched offline generation (sync/async,
  direct/coordinator modes). Wrapper: `run_offline_inference.sh`.
- **`launch_inference_server.py`** — OpenAI-compatible HTTP server via
  `MegatronAsyncLLM.serve(...)`. Wrapper: `run_inference_server.sh`.
- **`utils.py`** — shared helpers used by the examples.
- **`advanced/`** — lower-level scripts that drive the
  `megatron.core.inference` APIs directly (manual stepping, explicit
  coordinator / `InferenceClient` lifecycle, the static engine, and T5).
