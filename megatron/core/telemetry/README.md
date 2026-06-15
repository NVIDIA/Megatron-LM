# Megatron-LM OpenTelemetry Instrumentation

This module contains Megatron's OpenTelemetry integration, built on top of [`nemo-lens`](https://github.com/NVIDIA-NeMo/Lens).

It emits **traces** at training framework boundaries (training loop, checkpointing, evaluation, P2P communication, pipeline parallel stages, inference) and **metrics** (loss, throughput, gradient norm) that export to any OTLP-compatible backend.

## Contents

```
megatron/core/telemetry/
├── span_groups.py    — MegatronSpanGroup: Megatron-specific span groups
├── _fallbacks.py     — No-op shims for when nemo-lens is not installed
└── __init__.py
```

Metric instruments, resource detection, and the instrumentation primitives themselves live in `nemo-lens`. This module is a thin integration layer.

## Quick start

```bash
export MEGATRON_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export MEGATRON_OTEL_SPAN_GROUPS=default

torchrun --nproc_per_node=8 pretrain_gpt.py ...
```

## Full documentation

See `docs/user-guide/observability/` in this repository for the full Observability guide:

| Topic | Doc |
|---|---|
| Overview | [docs/user-guide/observability/index.md](../../../docs/user-guide/observability/index.md) |
| Configuration (env vars, CLI flags) | [docs/user-guide/observability/configuration.md](../../../docs/user-guide/observability/configuration.md) |
| Span groups and span hierarchy | [docs/user-guide/observability/span-groups.md](../../../docs/user-guide/observability/span-groups.md) |
| Training and inference metrics | [docs/user-guide/observability/metrics.md](../../../docs/user-guide/observability/metrics.md) |
| Pipeline-parallel trace correlation | [docs/user-guide/observability/pipeline-parallel.md](../../../docs/user-guide/observability/pipeline-parallel.md) |
| Local docker-compose stack | [docs/user-guide/observability/observability-stack.md](../../../docs/user-guide/observability/observability-stack.md) |
| Adding new instrumentation | [docs/user-guide/observability/extending.md](../../../docs/user-guide/observability/extending.md) |

For the generic `nemo-lens` documentation (configuration model, instrumentation primitives, custom exporters, design decisions), see the lens docs at <https://github.com/NVIDIA-NeMo/Lens/tree/main/docs>.
