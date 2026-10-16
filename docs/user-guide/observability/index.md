<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Observability Overview

Megatron-LM is instrumented with [OpenTelemetry](https://opentelemetry.io/) via the [`nemo-lens`](https://github.com/NVIDIA-NeMo/Lens) library, emitting **traces** at training-framework boundaries and span events for training reports.

Telemetry exports to any OTLP-compatible backend (Jaeger, Grafana Tempo, W&B Weave, Honeycomb, Datadog, ...).

## What's in this section

- [Configuration](configuration.md)
- [Span Groups](span-groups.md)
- [Metrics](metrics.md)
- [Pipeline-Parallel Trace Correlation](pipeline-parallel.md)
- [Extending Instrumentation](extending.md)
- [Trace Lifecycle](trace-lifecycle.md)

## Scope

This documentation covers **Megatron-specific** usage: CLI flags, environment variables, span names, metric names, and the pipeline-parallel trace correlation integration.

For general concepts — span groups, instrumentation primitives, configuration model, custom exporters, resource detection — see the [lens documentation](https://github.com/NVIDIA-NeMo/Lens). This section links to lens docs when relevant rather than duplicating content.

## Quick start

```bash
export MEGATRON_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export MEGATRON_OTEL_SPAN_GROUPS=default   # coarse-grained; safe for production

torchrun --nproc_per_node=8 pretrain_gpt.py ...
```

With `default` span groups, training reports attach to recording coarse spans.
Switch to `per_step` for profiling individual steps, or `all` for fine-grained
debugging. See [Training Signals](metrics.md) for report cadence and field semantics.

## What gets instrumented

Training, checkpointing, evaluation, pipeline communication, and model layers
route instrumentation through `megatron/core/telemetry/telemetry.py`.
Namespaced [span groups](span-groups.md) control which operation boundaries
emit spans. The facade owns optional-dependency behavior and delegates the
instrumentation primitives to Lens.

## What gets exported

- **Traces**: Jaeger / Tempo / Honeycomb / etc. via OTLP.
- **Metrics**: Prometheus via the OTel Collector, or direct OTLP to Grafana Mimir / Datadog / etc.
- **Logs** (optional): via the OTel log bridge when `MEGATRON_OTEL_LOGS_ENABLED=1` — correlates `logging` records with the active span's trace ID.

Each enabled trainer rank exports its own telemetry. Resource attributes
identify the rank and inherited run; see [Configuration](configuration.md).

## Related

- Lens configuration model and env vars: [lens: configuration](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/configuration.md)
- Instrumentation primitives (`managed_span`, `trace_fn`, `span_cm`): [lens: instrumentation](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/instrumentation.md)
- Sending telemetry to a backend: [lens: backends](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/observability/backends.md)
