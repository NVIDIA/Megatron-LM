### Megatron Core Inference Documentation
This guide provides an example for Megatron Core for running model inference.

### Contents

- [What's in here](#whats-in-here)
- [Offline inference](#offline-inference)
- [OpenAI-compatible inference server](#openai-compatible-inference-server)
- [Legacy examples](#legacy-examples)
- [See also](#see-also)

### What's in here

These examples drive the high-level inference API in `megatron/inference/`
(`MegatronLLM` for sync, `MegatronAsyncLLM` for async + HTTP serving). For
the API surface and mental model see
[`megatron/inference/README.md`](../../megatron/inference/README.md).

The two top-level Python entrypoints cover all common workflows:

- **`offline_inference.py`** — batched offline generation. Supports the
  4 mode combinations (sync/async × direct/coordinator) via CLI flags.
  Replaces the legacy `gpt_dynamic_inference.py` and
  `gpt_dynamic_inference_with_coordinator.py` paths.
- **`launch_inference_server.py`** — OpenAI-compatible HTTP server using
  `MegatronAsyncLLM.serve(...)`. Replaces the legacy
  `tools/run_dynamic_text_generation_server.py` path.

`utils.py` holds shared helpers (`Request`, `build_requests`,
`build_dynamic_engine_setup_prefix`, output formatting, JSON dump) used by
both new examples and by the legacy scripts.

### Offline inference

`offline_inference.py` runs synthetic-load inference on a Megatron model and
prints a setup-prefix line, a "Unique prompts + outputs" table, and a
throughput summary. Optional JSON dump for regression testing via
`--output-path`.

The shell wrapper `run_offline_inference.sh` packages the typical Qwen
2.5-1.5B configuration. Required CLI args: `--hf-token`, `--checkpoint`.
Optional: `--mode sync|async` (default `sync`), `--use-coordinator` (default
off, i.e. direct mode), `--nproc <n>` (default `8`).

```bash
# sync + direct (defaults)
bash examples/inference/run_offline_inference.sh \
    --hf-token <HF_TOKEN> --checkpoint /path/to/qwen-1.5b

# sync + coordinator
bash examples/inference/run_offline_inference.sh \
    --hf-token <HF_TOKEN> --checkpoint /path/to/qwen-1.5b --use-coordinator

# async + direct
bash examples/inference/run_offline_inference.sh \
    --hf-token <HF_TOKEN> --checkpoint /path/to/qwen-1.5b --mode async

# async + coordinator
bash examples/inference/run_offline_inference.sh \
    --hf-token <HF_TOKEN> --checkpoint /path/to/qwen-1.5b --mode async --use-coordinator
```

All four modes produce numerically identical generated text. The high-level
API rejects `--use-coordinator` with `--inference-repeat-n > 1` (engine
reset is unsafe in coordinator mode — see
[`megatron/inference/README.md`](../../megatron/inference/README.md)).

### OpenAI-compatible inference server

`launch_inference_server.py` uses `MegatronAsyncLLM.serve(blocking=True)`
on a coordinator-backed engine. The HTTP frontend exposes
`/v1/completions` and `/v1/chat/completions` on global rank 0.

The shell wrapper `run_inference_server.sh` packages the Nemotron-6 3B
hybrid MoE configuration (TP 2, EP 8, PP 1). Required CLI args:
`--hf-token`, `--hf-home`, `--checkpoint`. Optional: `--nproc <n>` (default
`8`).

```bash
bash examples/inference/run_inference_server.sh \
    --hf-token <HF_TOKEN> \
    --hf-home /path/to/hf_home \
    --checkpoint /path/to/nemotron-3b-hybrid-moe
```

When the server is ready you'll see the readiness banner (~2 minutes after
launch on Nemotron-6 3B):

```
INFO:root:Inference co-ordinator is ready to receive requests!
INFO:hypercorn.error:Running on http://0.0.0.0:5000 (CTRL + C to quit)
```

Send requests with any OpenAI-compatible client. The dynamic server
currently returns `"model": "EMPTY"` and does not validate the request
`model` field — pass anything you like.

### Legacy examples

`legacy/` preserves the prior `gpt_dynamic_inference.py`,
`gpt_dynamic_inference_with_coordinator.py`, `gpt_static_inference.py`, and
`simple_t5_batch_inference.py` scripts as-is. They are still wired into
the existing recipes under `tests/test_utils/recipes/h100/{gpt,moe,mamba}-*-inference.yaml`
for backward compatibility. New work should target `offline_inference.py`
and `launch_inference_server.py`.

### See also

- API reference: [`megatron/inference/README.md`](../../megatron/inference/README.md)
- Low-level engine: [`megatron/core/inference/`](../../megatron/core/inference/)
- Functional tests: `tests/functional_tests/test_cases/gpt/gpt_offline_inference_*` + `gpt_inference_server_smoke_*`
- Unit tests: `tests/unit_tests/inference/high_level_api/`
