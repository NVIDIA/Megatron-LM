# Metrics

Megatron emits two namespaces of metrics: training metrics (`megatron.training.*`) and inference metrics (`gen_ai.*`, following OTel GenAI semantic conventions).

All metrics are emitted **only on the export rank** (`is_exporting = True`). Non-exporting ranks don't create metric instruments.

For the general instrument pattern (weak-reference caching, None-skipping), see
[lens: metrics](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/metrics.md).

## Training metrics (`megatron.training.*`)

Training has no OTel standard, so Megatron uses a project-specific namespace. Emission is tied to `--log-interval` (same cadence as TensorBoard and W&B loggers).

| Metric | Type | Unit | Description |
|---|---|---|---|
| `megatron.training.step_duration_ms` | Histogram | ms | Duration of one training step in milliseconds |
| `megatron.training.loss` | Gauge | — | Training loss (last value per log interval) |
| `megatron.training.throughput_tflops` | Gauge | TFLOP/s | Training throughput in TFLOP/s/GPU |
| `megatron.training.tokens_per_sec` | Gauge | tokens/s | Training throughput in tokens per second |
| `megatron.training.grad_norm` | Gauge | — | Global gradient norm |
| `megatron.training.skipped_iters` | Counter | — | Optimizer steps skipped (NaN/inf loss) |
| `megatron.training.learning_rate` | Gauge | — | Current learning rate |
| `megatron.training.memory_allocated_gb` | Gauge | GB | Peak GPU memory allocated |

Loss, throughput, grad norm, and learning rate are **Gauges** (point-in-time value), not Histograms — this produces a Prometheus `gauge` which is semantically correct for a value that changes every log interval.

### Emission site

`megatron/training/training.py` calls `record_training_metrics()` from `megatron.core.telemetry.training_metrics` every `--log-interval` iterations. The instrument module caches per-Meter instruments using `WeakKeyDictionary` to avoid leaking on re-init.

## Inference metrics (GenAI)

Follows the [OTel GenAI metrics spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/). Emitted from `megatron/core/inference/text_generation_server/`.

| Metric | Type | Unit | Description |
|---|---|---|---|
| `gen_ai.server.request.duration` | Histogram | s | End-to-end request latency |
| `gen_ai.client.token.usage` | Histogram | `{token}` | Tokens per request, split by `gen_ai.token.type` |

All data points carry the required GenAI attributes:
- `gen_ai.operation.name = "text_completion"`
- `gen_ai.provider.name = "megatron"`
- `gen_ai.request.model = <model name>`

`gen_ai.client.token.usage` additionally has a `gen_ai.token.type` label with values `"input"` or `"output"` — query them separately in Prometheus/Grafana.

## Prometheus metric names

The OTel SDK may append a unit suffix when exporting to Prometheus.

| OTel instrument name | Prometheus metric (example) |
|---|---|
| `megatron.training.loss` | `megatron_training_loss` (Gauge) |
| `megatron.training.step_duration_ms` | `megatron_training_step_duration_ms_milliseconds` |
| `megatron.training.throughput_tflops` | `megatron_training_throughput_tflops` (Gauge) |
| `megatron.training.tokens_per_sec` | `megatron_training_tokens_per_sec` (Gauge) |
| `megatron.training.skipped_iters` | `megatron_training_skipped_iters_total` |
| `gen_ai.server.request.duration` | `gen_ai_server_request_duration_seconds` |
| `gen_ai.client.token.usage` | `gen_ai_client_token_usage_bucket` (+ `gen_ai_token_type` label) |

Dashboards use regex patterns (e.g. `{__name__=~"megatron_training_loss.*"}`) to match regardless of suffix. If a panel shows "No data", use **Explore → Prometheus → Metrics browser** to discover exact names on your SDK version.

## Filtering across runs

Metrics carry the `nemo.run.id` resource attribute on every data point. Use it to filter in Grafana:

```
{nemo_run_id="<id>", __name__=~"megatron_training_.*"}
```

Or to compare two runs:

```
{nemo_run_id=~"run-a|run-b", __name__="megatron_training_loss"}
```

## Metric vs span attribute

A recurring pitfall: putting training loss on a span attribute instead of a metric.

- **Loss** changes every iteration. Put it on `megatron.training.loss` metric. Prometheus stores each value; Grafana plots the series.
- **Iteration number** is categorical context for a specific span. Put it on `megatron.iteration` span attribute. Jaeger uses it for filtering.

Don't do it the other way. Loss on a span attribute produces no useful time series in Jaeger; it's wasted data. Iteration on a metric label produces one metric series per iteration — unbounded cardinality explosion.

See [lens: metrics — Metric vs span attribute vs resource attribute](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/metrics.md#metrics-vs-span-attributes-vs-resource-attributes).

## Adding custom metrics

To add Megatron-specific metrics, add a new file under `megatron/core/instruments/` following the pattern in `nemo.lens.instruments.inference`:

1. Declare a `WeakKeyDictionary` for per-Meter instrument caching.
2. Implement `_get_<domain>_instruments(meter)` that creates and caches instruments.
3. Implement `record_<domain>_metrics(meter, **kwargs)` that records only non-`None` values.

Use `megatron.<subsystem>.<metric>` naming for application-specific metrics, reserving the shared `dl.*` and `gen_ai.*` namespaces for cross-consumer or standard metrics.

See the existing `nemo.lens.instruments.inference` as a template.
