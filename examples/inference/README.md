### Megatron Core Inference Documentation
This guide provides an example for Megatron Core for running model inference.

### Contents

- [What's in here](#whats-in-here)
- [Offline inference](#offline-inference)
- [OpenAI-compatible inference server](#openai-compatible-inference-server)
- [Advanced examples](#advanced-examples)
- [See also](#see-also)

### What's in here

These examples drive the high-level inference API in `megatron/core/inference/apis/`
(`MegatronLLM` for sync, `MegatronAsyncLLM` for async + HTTP serving). For
the API surface and mental model see
[`megatron/core/inference/README.md`](../../megatron/core/inference/README.md).

The two top-level Python entrypoints cover all common workflows:

- **`offline_inference.py`** — batched offline generation. Supports the
  3 mode combinations (sync+direct, sync+coordinator, async+coordinator) via CLI flags.
  Replaces the `gpt_dynamic_inference.py` and
  `gpt_dynamic_inference_with_coordinator.py` paths.
- **`launch_inference_server.py`** — OpenAI-compatible HTTP server using
  `MegatronAsyncLLM.serve(...)`. Replaces the
  `tools/run_dynamic_text_generation_server.py` path.

`utils.py` holds shared helpers (`Request`, `build_requests`,
`build_dynamic_engine_setup_prefix`, output formatting, JSON dump) used by
both new examples and by the `advanced/` scripts.

### Offline inference

`offline_inference.py` runs synthetic-load inference on a Megatron model and
prints a setup-prefix line, a "Unique prompts + outputs" table, and a
throughput summary. Optional JSON dump for regression testing via
`--output-path`.

The shell wrapper `run_offline_inference.sh` packages the typical Qwen
2.5-1.5B configuration. Required CLI args: `--hf-token`, `--checkpoint`.
Optional: `--mode sync|async` (default `sync`), `--use-coordinator` (default
off, i.e. direct mode), `--nproc <n>` (default `8`). Currently async + direct is not supported.

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

All four modes produce numerically identical generated text. The high-level
API rejects `--use-coordinator` with `--inference-repeat-n > 1` (engine
reset is unsafe in coordinator mode — see
[`megatron/core/inference/README.md`](../../megatron/core/inference/README.md)).

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

### Advanced examples

`advanced/` contains scripts that drive the lower-level
`megatron.core.inference` APIs directly — manual `add_request` /
`step_modern` stepping, explicit coordinator / `InferenceClient`
lifecycle, the static engine, and T5 inference. Use these when you need
step-level scheduling control, custom forward-step / sampling
integration, or are migrating existing pipelines. For typical workflows,
prefer `offline_inference.py` and `launch_inference_server.py`. CI
recipes under `tests/test_utils/recipes/h100/{gpt,moe,mamba}-*-inference.yaml`
still target these scripts.

### MoE routing analysis

`tools/moe_routing/analyze_routing.py` and the `analyze_routing_*.py`scripts analyze per-layer top-K routing decisions from MoE models.  The same JSONL trace format and the same analysis scripts work for both training and inference.

#### Collecting traces

**During training**, enable these flags:

```bash
--moe-routing-trace-path /path/to/trace_dir   # enable tracing
--moe-routing-trace-max-iters 500             # optional: stop after N iters
--moe-routing-trace-capture-hidden-states     # required for distribution predictability
--moe-routing-trace-dump-weights              # required for distribution predictability
```

**During inference**, add these flags (e.g. to
`advanced/gpt_dynamic_inference_with_coordinator.py`):

```bash
--moe-routing-trace-path /path/to/trace_dir
--moe-routing-trace-max-steps 200
--moe-routing-trace-capture-hidden-states
--moe-routing-trace-dump-weights
```

Both write `router_trace_rank{N}.jsonl` into the specified directory (one file per rank).
`--moe-routing-trace-capture-hidden-states` also writes `hidden_states_rank{N}.bin` and
`--moe-routing-trace-dump-weights` writes `router_state_rank{N}.pt`; both are required by
`analyze_routing_predictability.py`.

#### Running analyses

```bash
python tools/moe_routing/analyze_routing.py /path/to/trace_dir --num-experts 512
```

The dispatcher runs these analyses in order:

| Script | Primary question | Role |
|--------|-----------------|------|
| `tools/moe_routing/analyze_routing_concentration.py` | How concentrated is routing? (hot-set size) | Hypothesis test: is per-layer static caching viable? High concentration (ratio > 2×) supports it; near-uniform rules it out. |
| `tools/moe_routing/analyze_routing_predictability.py` | How well do L_prev's hidden states predict L's routing distribution? | Affirmative signal: high cosine/Spearman means distributional routing is predictable one layer ahead. |

#### Interpreting the distribution predictability output

`analyze_routing_predictability.py` applies layer L's router weights to the hidden states
from L_prev and compares the resulting predicted per-expert token-count distribution to
what L actually routed.  The `cos` and `spearman` columns measure that agreement:

- `cos` ≥ 0.90 and `spearman` ≥ 0.70: the hidden-state signal from the previous MoE layer
  is sufficient to predict the aggregate expert load distribution of the next layer with high
  fidelity.
- Values near zero: weak cross-layer signal for this layer pair.

This is a distributional result: per-token assignment errors cancel in the aggregate count
histogram, so this measure is far more forgiving than per-token index overlap.

#### Adding new routing metrics

To add a new routing metric, put capture logic in `megatron/core/transformer/moe/router_trace.py`
(as part of the `RouterTracer` class) so it is available to both training and inference.  Avoid
adding bespoke logging flows to `megatron/training/activation_logging.py`
for routing metrics — that file handles lightweight count monitoring
(`tokens_per_expert`) and uses a different output format.

### See also

- API reference: [`megatron/core/inference/README.md`](../../megatron/core/inference/README.md)
- Low-level engine: [`megatron/core/inference/`](../../megatron/core/inference/)
- Functional tests: `tests/functional_tests/test_cases/gpt/gpt_offline_inference_*` + `gpt_inference_server_smoke_*`
- Unit tests: `tests/unit_tests/inference/high_level_api/`
