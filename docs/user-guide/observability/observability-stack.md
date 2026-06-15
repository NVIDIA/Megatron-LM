# Demo Stack (PoC)

Megatron ships a `docker-compose.otel.yml` at the repository root as a **proof of concept** — one plausible observability pipeline you can spin up locally to try Megatron's telemetry without wiring up a production stack.

This is **not** a recommended production deployment and not something Megatron is opinionated about. Choosing an observability solution — retention, scale, auth, alerting, dashboards — is the user's decision, driven by your organisation's existing stack. The demo is here to help you get started, not to tell you what to run long-term.

For production, see [lens: sending telemetry to a backend](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/observability/backends.md).

## Start the stack

```bash
docker compose -f docker-compose.otel.yml up -d
```

## Components

| Service | Port | Purpose |
|---|---|---|
| OpenTelemetry Collector | 4317 / 4318 | Receives OTLP, fans out to Jaeger + Prometheus + Elasticsearch |
| Jaeger | 16686 | Trace search UI |
| Prometheus | 9090 | Metrics time-series storage |
| Grafana | 3000 | Training dashboards (pre-loaded) |
| Elasticsearch | 9200 | Trace + log storage |
| Kibana | 5601 | Log visualisation |
| DCGM Exporter | 9400 | GPU metrics |
| Node Exporter | 9100 | InfiniBand + network metrics |

For the general stack components and flow, see
[lens: observability stack](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/observability/stack.md).

## Configure Megatron

```bash
export MEGATRON_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # gRPC
# If Megatron runs on a different machine, replace localhost with the collector host IP.

export MEGATRON_OTEL_SPAN_GROUPS=default    # job/checkpoint/evaluate/inference only
# or:
export MEGATRON_OTEL_SPAN_GROUPS=per_step   # adds step/forward_backward/optimizer
# or:
export MEGATRON_OTEL_SPAN_GROUPS=all        # adds microbatch, layer, activation_offload
```

## UIs

| Service | URL | Purpose |
|---|---|---|
| Grafana | http://localhost:3000 | Training metrics dashboard (Megatron folder) |
| Jaeger | http://localhost:16686 | Trace search and waterfall view |
| Prometheus | http://localhost:9090 | Ad-hoc PromQL queries |
| Kibana | http://localhost:5601 | Log search (Discover tab) |

Grafana opens with anonymous admin — no login required for local dev.

## File layout

```
docker-compose.otel.yml
observability/
├── otel-collector.yaml               # receiver → Jaeger + Prometheus pipeline
├── prometheus.yml                    # scrapes collector:8889, dcgm, node-exporter
└── grafana/
    ├── provisioning/
    │   ├── datasources/              # Prometheus + Jaeger auto-wired
    │   └── dashboards/               # loads dashboards from /dashboards dir
    └── dashboards/
        └── megatron-training.json    # pre-built training + inference dashboard
```

## Typical workflow

1. Start the stack: `docker compose -f docker-compose.otel.yml up -d`
2. Run training with `MEGATRON_OTEL_ENABLED=1` and `MEGATRON_OTEL_SPAN_GROUPS=default`.
3. Watch Grafana's "Megatron Training" dashboard — loss, throughput, GPU utilisation.
4. If something looks off in a specific iteration, jump to Jaeger:
   - Find the `megatron.train_step` trace for that iteration.
   - Look at the waterfall — which phase took too long?
5. To drill deeper, restart training with `MEGATRON_OTEL_SPAN_GROUPS=per_step` (or `all` for microbatch-level view).

## Stopping the stack

```bash
docker compose -f docker-compose.otel.yml down

# Clear volumes too:
docker compose -f docker-compose.otel.yml down -v
```

## Running against a remote stack

If the stack runs on a different host:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://observability-host.internal:4317
```

Open port 4317 (gRPC) or 4318 (HTTP) on the observability host, or use an SSH tunnel.

## Production setup

For production runs, point at a hosted backend (W&B Weave, Grafana Cloud, Honeycomb, Datadog) or a shared internal collector instead of the docker-compose stack. See
[lens: observability backends](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/observability/backends.md)
for endpoint configuration for each backend.

## Grafana dashboard

The pre-loaded `megatron-training.json` dashboard shows:

- Training loss over time (per run, filterable by `nemo.run.id`)
- Step duration (p50/p95/p99)
- Throughput (tokens/second)
- Gradient norm
- Skipped iterations
- GPU utilisation (from DCGM exporter)
- InfiniBand traffic (from Node exporter)
- Inference request duration and token usage (if inference is enabled)

Panels use regex patterns for metric names (e.g. `{__name__=~"megatron_training_loss.*"}`) to tolerate OTel SDK suffix variation.

If a panel shows "No data", check the exact metric name in Prometheus's Metrics browser — SDK versions sometimes append unexpected suffixes.
