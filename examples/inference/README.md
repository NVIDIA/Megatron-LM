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
  `MegatronAsyncLLM.serve(...)` or `MegatronLLM.serve(...)`. Wrapper:
  `run_inference_server.sh`.
- **`utils.py`** — shared helpers used by the examples.
- **`advanced/`** — lower-level scripts that drive the
  `megatron.core.inference` APIs directly (manual stepping, explicit
  coordinator / `InferenceClient` lifecycle, the static engine, and T5).

### MoE routing analysis tooling

`tools/moe_routing/analyze_routing.py` and the `analyze_routing_*.py`scripts analyze per-layer top-K routing decisions from MoE models.  The JSONL trace format and the same analysis scripts work for both training and inference.

Inference can collect traces two ways.

| Path | Enable with | Captures | CUDA graphs |
|------|-------------|----------|------------|
| **Sink** | `--moe-enable-routing-replay` | top-K indices only | on |
| **Hook** | no replay + `--cuda-graph-impl none` | indices **+ hidden states + router weights** | must be off |

Only the hook path captures the hidden states and router weights that
`analyze_routing_predictability.py` needs. The sink fills an in-pipeline buffer that
holds indices only. Use the sink for concentration / load-balance (cheap and graph-safe);
use the hook when you need to save hidden states, weights, or other expensive data. 

#### Collecting traces

**Training** (hook path):

```bash
--moe-routing-trace-path /path/to/trace_dir   # enable tracing
--moe-routing-trace-max-training-iters 500    # optional: stop after N iters
--moe-routing-trace-capture-hidden-states     # for predictability
--moe-routing-trace-dump-weights              # for predictability
```

Forward hooks do not fire during CUDA graph replay. MoE cudagraphs must be disabled during training otherwise graph-captured layers are silently skipped.

**Inference — sink** (routing indices only, graphs on):

```bash
--moe-routing-trace-path /path/to/trace_dir
--moe-routing-trace-max-inference-steps 200
--moe-enable-routing-replay
```

**Inference — hook** (adds hidden states + weights for predictability):

```bash
--moe-routing-trace-path /path/to/trace_dir
--moe-routing-trace-max-inference-steps 200
--cuda-graph-impl none
--moe-routing-trace-capture-hidden-states
--moe-routing-trace-dump-weights
```

All write `router_trace_rank{N}.jsonl` (one file per rank).
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
what L actually routed.  This serves to provide an example on measuring whether the hidden-state signal from the previous MoE layer
  is sufficient to predict the aggregate expert load distribution of the next layer. Values near zero suggest weak cross-layer signal for this layer pair.

This is a distributional result: per-token assignment errors cancel in the aggregate count
histogram.

#### Adding new routing metrics

To add a new routing metric, put capture logic in `megatron/core/transformer/moe/router_trace.py`
(as part of the `RouterTracer` class) so it is available to both training and inference.  Avoid
adding bespoke logging flows to `megatron/training/activation_logging.py`
for routing metrics — that file handles lightweight count monitoring
(`tokens_per_expert`) and uses a different output format.
