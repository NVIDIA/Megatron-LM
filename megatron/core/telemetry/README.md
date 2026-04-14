# Megatron-LM OpenTelemetry Instrumentation

This module adds first-class [OpenTelemetry](https://opentelemetry.io/) support to
Megatron-LM.  It emits **traces** (spans at training framework boundaries) and
**metrics** (loss, throughput, gradient norm, …) that can be exported to any OTLP-
compatible backend (Jaeger, Grafana Tempo, Honeycomb, Datadog, …).

Training spans use a Megatron-specific `megatron.*` namespace.  Inference spans and
metrics follow the [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

---

## 1. Enabling OTel

### CLI flags

| Flag | Type | Description |
|---|---|---|
| `--otel-enabled` | flag | Enable OTel telemetry |
| `--otel-service-name NAME` | string | Override `OTEL_SERVICE_NAME` |
| `--otel-span-groups SPEC` | string | Comma-separated span-group spec (see §3) |

### Environment variables (Megatron-specific)

| Variable | Default | Description |
|---|---|---|
| `MEGATRON_OTEL_ENABLED` | `0` | Master toggle; must be set to `1` to activate |
| `MEGATRON_OTEL_EXPORT_RANK` | `-1` | Which rank exports data (`-1` = last rank) |
| `MEGATRON_OTEL_TRACES_ENABLED` | `1` | Enable trace spans |
| `MEGATRON_OTEL_METRICS_ENABLED` | `1` | Enable metrics instruments |
| `MEGATRON_OTEL_SPAN_GROUPS` | `default` | Span granularity spec (see §3) |
| `MEGATRON_OTEL_EXPORTER` | `otlp` | Exporter backend: `otlp` or `console` |
| `NEMO_LENS_RUN_ID` | (auto) | Unique run identifier. Auto-detected from SLURM_JOB_ID or generated UUID. |
| `NEMO_LENS_USER` | (empty) | Optional user/team label for filtering. |

### Run Identification

Each training run is automatically assigned a unique `nemo.run.id` resource attribute that
flows to all backends (Jaeger, Prometheus/Grafana, Elasticsearch/Kibana).

**Priority order:**
1. `NEMO_LENS_RUN_ID` env var (explicit, highest priority)
2. `SLURM_JOB_ID` env var (auto-detected on SLURM clusters)
3. Auto-generated 12-character UUID (fallback)

All ranks in a distributed job share the same `run_id`. Each rank gets a unique
`service.instance.id` of `{run_id}-rank{rank}`.

**Filtering:**
- **Jaeger**: Search by tag `nemo.run.id=<value>`
- **Grafana**: Use the "Run ID" dropdown variable
- **Kibana**: Filter by `nemo.run.id` field in Discover

---

## 2. Standard OTel SDK Environment Variables

All standard OTel SDK environment variables are honoured automatically.

| Variable | Example | Description |
|---|---|---|
| `OTEL_SERVICE_NAME` | `megatron-training` | Service name reported to backend |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP collector endpoint |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` or `http/protobuf` | Wire protocol |
| `OTEL_EXPORTER_OTLP_HEADERS` | `api-key=<secret>` | Auth headers |
| `OTEL_TRACES_SAMPLER` | `parentbased_traceidratio` | Sampling strategy |
| `OTEL_TRACES_SAMPLER_ARG` | `0.1` | Sampler argument (e.g. ratio) |
| `OTEL_BSP_MAX_QUEUE_SIZE` | `2048` | BatchSpanProcessor queue |
| `OTEL_BSP_SCHEDULE_DELAY` | `5000` | BSP flush interval (ms) |
| `OTEL_SDK_DISABLED` | `true` | Disable the entire SDK |

---

## 3. Span Groups (Training)

Span granularity is controlled by a comma-separated **span-group spec** passed via
`MEGATRON_OTEL_SPAN_GROUPS` or `--otel-span-groups`.  The spec accepts preset
keywords, individual group names, or a mix.

### Preset keywords

| Keyword | Groups included | Overhead |
|---|---|---|
| `default` | `job`, `checkpoint`, `evaluate`, `inference` | < 0.1 % — safe for production |
| `per_step` | all of `default` + `model_init`, `load_checkpoint`, `step`, `forward_backward`, `optimizer`, `communication`, `data_loading` | < 1 % — use with sampling |
| `all` | everything including `microbatch`, `activation_offload` | > 1 % — dev/debug only |

### Individual group names

| Group | Spans emitted | Typical frequency |
|---|---|---|
| `job` | `megatron.pretrain`, `megatron.train` | once per job |
| `checkpoint` | `megatron.save_checkpoint` | every checkpoint |
| `evaluate` | `megatron.evaluate` | every eval interval |
| `model_init` | `megatron.model_init` | once at startup |
| `load_checkpoint` | `megatron.load_checkpoint` | once at startup |
| `step` | `megatron.train_step` | every iteration |
| `forward_backward` | `megatron.forward_backward` | every iteration |
| `optimizer` | `megatron.optimizer_step` | every iteration |
| `microbatch` | `megatron.microbatch.forward`, `megatron.microbatch.backward` | every microbatch |
| `communication` | `megatron.p2p.{recv,send}_{forward,backward}`, `megatron.grad_sync.{start,finish}` | every iteration |
| `activation_offload` | `megatron.activation.offload`, `megatron.activation.reload` | every microbatch |
| `data_loading` | (reserved for future use) | every iteration |
| `inference` | `text_completion {model}` | every inference request |

### Examples

```bash
# Coarse spans only — default
MEGATRON_OTEL_SPAN_GROUPS=default

# Include per-step spans
MEGATRON_OTEL_SPAN_GROUPS=per_step

# Default + microbatch only (skips step/optimizer overhead)
MEGATRON_OTEL_SPAN_GROUPS=default,microbatch

# Everything
MEGATRON_OTEL_SPAN_GROUPS=all

# Tag a run for easy filtering
NEMO_LENS_RUN_ID=gpt3-lr-sweep-42 MEGATRON_OTEL_ENABLED=1 torchrun ...
```

### Span hierarchy

```
megatron.pretrain                                    # job
  ├── megatron.model_init                            # model_init
  ├── megatron.load_checkpoint                       # load_checkpoint
  │     └── megatron.load_checkpoint.io_read         # load_checkpoint
  └── megatron.train                                 # job
        ├── megatron.train_step                      # step
        │     ├── megatron.forward_backward          # forward_backward
        │     │     ├── megatron.microbatch.forward   # microbatch (×N)
        │     │     ├── megatron.microbatch.backward  # microbatch (×N)
        │     │     ├── megatron.p2p.recv_forward     # communication
        │     │     ├── megatron.p2p.send_forward     # communication
        │     │     ├── megatron.p2p.recv_backward    # communication
        │     │     ├── megatron.p2p.send_backward    # communication
        │     │     ├── megatron.activation.offload   # activation_offload
        │     │     └── megatron.activation.reload    # activation_offload
        │     ├── megatron.grad_sync.start           # communication
        │     ├── megatron.grad_sync.finish           # communication
        │     └── megatron.optimizer_step             # optimizer
        ├── megatron.save_checkpoint                 # checkpoint
        │     ├── megatron.save_checkpoint.state_dict # checkpoint
        │     └── megatron.save_checkpoint.io_write   # checkpoint
        └── megatron.evaluate                        # evaluate
              └── megatron.evaluate.step              # evaluate (×N)

text_completion {model}                              # inference (GenAI semconv)
```

### Training span attributes

| Attribute | Type | Set on |
|---|---|---|
| `megatron.model_type` | str | `megatron.pretrain` |
| `megatron.train_iters` | int | `megatron.pretrain`, `megatron.train` |
| `megatron.global_batch_size` | int | `megatron.pretrain` |
| `megatron.iteration` | int | `megatron.train_step`, `megatron.save_checkpoint` |
| `megatron.loss` | float | `megatron.train_step` |
| `megatron.grad_norm` | float | `megatron.train_step`, `megatron.optimizer_step` |
| `megatron.num_microbatches` | int | `megatron.forward_backward` |
| `megatron.microbatch_id` | int | `megatron.microbatch.forward` |
| `megatron.eval_iters` | int | `megatron.evaluate` |
| `megatron.update_successful` | bool | `megatron.optimizer_step` |

---

## 4. Inference Spans (GenAI Semantic Conventions)

The inference server (`MegatronGenerate`) emits spans that follow the
[OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/).

### Span name

```
text_completion {gen_ai.request.model}
```

e.g. `text_completion gpt` or `text_completion llama`.

### Span attributes

| Attribute | Requirement | Value / source |
|---|---|---|
| `gen_ai.operation.name` | Required | `"text_completion"` |
| `gen_ai.provider.name` | Required | `"megatron"` |
| `gen_ai.request.model` | Conditionally Required | `args.model_type` |
| `gen_ai.request.max_tokens` | Recommended | `tokens_to_generate` from request |
| `gen_ai.request.temperature` | Recommended | `temperature` from request |
| `gen_ai.request.top_k` | Recommended | `top_k` if > 0 |
| `gen_ai.request.top_p` | Recommended | `top_p` if > 0.0 |
| `gen_ai.request.seed` | Recommended | `random_seed` if ≥ 0 |
| `gen_ai.usage.output_tokens` | Recommended | token count from response |
| `error.type` | Conditionally Required | set on error by `span_cm` |

W3C TraceContext is extracted from incoming HTTP headers so that upstream callers
(API gateway, client) can propagate trace context into the Megatron span.

---

## 5. Metrics Inventory

### Training metrics (`megatron.training.*`)

Training has no OTel standard, so a Megatron-specific namespace is used.
Metrics are emitted at the `--log-interval` cadence (same as tensorboard/wandb),
**only on the export rank** (`is_exporting = True`).

| Metric name | Type | Unit | Description |
|---|---|---|---|
| `megatron.training.step_duration_ms` | Histogram | ms | Mean step wall-clock time |
| `megatron.training.loss` | Gauge | — | Training loss (last value per log interval) |
| `megatron.training.throughput_tps` | Gauge | tokens/s | Training throughput |
| `megatron.training.grad_norm` | Gauge | — | Global gradient norm |
| `megatron.training.skipped_iters` | Counter | — | Optimizer steps skipped (NaN/inf loss) |

Loss, throughput, and grad norm are **Gauges** (point-in-time last value), not
Histograms — this produces a Prometheus `gauge` which is semantically correct.

### Inference metrics (GenAI semantic conventions)

Follows the [OTel GenAI metrics spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/).

| Metric name | Type | Unit | Description |
|---|---|---|---|
| `gen_ai.server.request.duration` | Histogram | s | End-to-end request latency |
| `gen_ai.client.token.usage` | Histogram | `{token}` | Tokens per request, split by `gen_ai.token.type` |

All inference metric data points carry the required GenAI attributes:
`gen_ai.operation.name = "text_completion"`, `gen_ai.provider.name = "megatron"`,
`gen_ai.request.model`.

The `gen_ai.client.token.usage` histogram has a `gen_ai.token.type` label with values
`"input"` or `"output"` — query them separately in Prometheus/Grafana.

---

## 6. Local Dev Setup (console exporter, no collector)

```bash
MEGATRON_OTEL_ENABLED=1 \
MEGATRON_OTEL_EXPORTER=console \
python examples/run_simple_mcore_train_loop.py
```

Span and metric data is printed to stdout.

---

## 7. Local Dev Setup with a Single Collector

```bash
docker run --rm -p 4317:4317 \
  otel/opentelemetry-collector-contrib:latest
```

```bash
MEGATRON_OTEL_ENABLED=1 \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
python examples/run_simple_mcore_train_loop.py
```

---

## 8. Full Observability Stack (Jaeger + Prometheus + Grafana)

A ready-to-use `docker-compose.otel.yml` is provided at the repository root.
It wires together:

- **OpenTelemetry Collector** — receives OTLP from Megatron, fans out to Jaeger and Prometheus
- **Jaeger** — distributed trace storage and search UI
- **Prometheus** — time-series storage for OTel metrics
- **Grafana** — unified dashboard pre-loaded with a Megatron training panel

### Start the stack

```bash
docker compose -f docker-compose.otel.yml up -d
```

### Configure Megatron

```bash
export MEGATRON_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # gRPC
# If Megatron runs on a different machine, replace localhost with the collector host IP.

# Choose span granularity:
export MEGATRON_OTEL_SPAN_GROUPS=default    # job/checkpoint/evaluate only (default)
export MEGATRON_OTEL_SPAN_GROUPS=per_step   # adds step/forward_backward/optimizer
export MEGATRON_OTEL_SPAN_GROUPS=all        # adds microbatch spans (high overhead)
```

### UIs

| Service | URL | Purpose |
|---|---|---|
| Grafana | http://localhost:3000 | Training metrics dashboard (Megatron folder) |
| Jaeger | http://localhost:16686 | Trace search and waterfall view |
| Prometheus | http://localhost:9090 | Ad-hoc PromQL queries |

Grafana opens with anonymous admin access — no login required for local development.

### File layout

```
docker-compose.otel.yml
observability/
├── otel-collector.yaml               # receiver → Jaeger + Prometheus pipeline
├── prometheus.yml                    # scrapes collector:8889
└── grafana/
    ├── provisioning/
    │   ├── datasources/              # Prometheus + Jaeger auto-wired
    │   └── dashboards/               # loads dashboards from /dashboards dir
    └── dashboards/
        └── megatron-training.json    # pre-built training + inference dashboard
```

### Prometheus metric names

The OTel SDK may append a unit suffix when exporting to Prometheus.

| OTel instrument name | Prometheus metric (example) |
|---|---|
| `megatron.training.loss` | `megatron_training_loss` (Gauge) |
| `megatron.training.step_duration_ms` | `megatron_training_step_duration_ms_milliseconds` |
| `megatron.training.skipped_iters` | `megatron_training_skipped_iters_total` |
| `gen_ai.server.request.duration` | `gen_ai_server_request_duration_seconds` |
| `gen_ai.client.token.usage` | `gen_ai_client_token_usage_bucket` (+ `gen_ai_token_type` label) |

The dashboard uses regex patterns (`{__name__=~"megatron_training_loss.*"}`) to match
regardless of suffix.  If panels show "No data", use **Explore → Prometheus → Metrics
browser** to discover exact names on your SDK version.

---

## 9. Production Setup

```bash
# Point at a remote collector
MEGATRON_OTEL_ENABLED=1 \
OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector-host>:4317 \
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer <token>" \
python pretrain_gpt.py ...
```

### Sampling to reduce overhead at per-step granularity

```bash
MEGATRON_OTEL_ENABLED=1 \
MEGATRON_OTEL_SPAN_GROUPS=per_step \
OTEL_TRACES_SAMPLER=parentbased_traceidratio \
OTEL_TRACES_SAMPLER_ARG=0.1 \
python pretrain_gpt.py ...
```

---

## 10. Overhead Guidance

| Span groups | Overhead | Recommendation |
|---|---|---|
| Disabled (`MEGATRON_OTEL_ENABLED=0`) | Zero | Default |
| `default` (job/checkpoint/evaluate) | < 0.1 % | Safe for all production runs |
| `per_step` | < 1 % | Use with `OTEL_TRACES_SAMPLER` |
| `all` (includes microbatch) | > 1 % | Development / profiling only |

Only the **export rank** sends data over the network.  All other ranks are set to
`frozenset()` span groups — `is_span_group_enabled()` returns `False` everywhere, so
**no span objects are created at all** on non-export ranks (true zero overhead, not just
no-op span recording).

---

## 11. Extending: Custom Spans in User Code

### Simple bounded block — `span_cm`

```python
from megatron.training.global_vars import get_telemetry
from nemo.lens.helpers import span_cm

telemetry = get_telemetry()
if telemetry is not None:
    with span_cm("my_custom_op", tracer=telemetry.tracer, param_count=1e9):
        ...  # your code here
```

### Group-gated block with exception safety — `managed_span`

`managed_span` is the preferred helper for new instrumentation.  It combines a
group-enabled check with `try/finally` lifecycle management — the span always ends
and exceptions are recorded with `ERROR` status:

```python
from nemo.lens.helpers import managed_span
from nemo.lens.groups import SpanGroup
from megatron.training.global_vars import get_telemetry

telemetry = get_telemetry()
tracer = telemetry.tracer if telemetry is not None else None

with managed_span(SpanGroup.STEP, "my_custom_step",
                  tracer=tracer, iteration=iteration) as span:
    result = do_work()
    if span is not None:
        span.set_attribute("my_custom.result", result)
# span is None when the group is disabled — no overhead, no exception
```

### Global tracer (no setup required)

```python
from nemo.lens.helpers import span_cm
from opentelemetry import trace
tracer = trace.get_tracer('megatron.core')

with span_cm("my_custom_op", tracer=get_tracer()):
    ...
```

### Checking group status before expensive preparation

```python
from nemo.lens.state import is_span_group_enabled
from nemo.lens.groups import SpanGroup

if is_span_group_enabled(SpanGroup.STEP):
    # only runs when step-level spans are active
    attrs = build_expensive_attributes()
    ...
```

### Checking export rank before expensive metric computation

`TelemetryHandle.is_exporting` is `True` only on the rank that sends data to the
collector.  Use it to skip metric value computation on all other ranks:

```python
from megatron.training.global_vars import get_telemetry

telemetry = get_telemetry()
if telemetry is not None and telemetry.is_exporting:
    from nemo.lens.instruments.training import record_training_metrics
    record_training_metrics(meter=telemetry.meter, loss=compute_loss())
```

---

## 12. Installation

```bash
# nemo-lens (already in dev/lts extras — provides opentelemetry-api)
pip install nemo-lens>=0.1.0

# Full SDK + OTLP exporters (required for actual export)
pip install 'megatron-core[otel]'
# which installs nemo-lens[sdk]:
#   opentelemetry-sdk, opentelemetry-exporter-otlp-proto-grpc/http
```
