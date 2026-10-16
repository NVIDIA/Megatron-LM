# Megatron-LM OpenTelemetry Instrumentation

This module holds the building blocks for Megatron's OpenTelemetry integration,
built on top of [`nemo-lens`](https://github.com/NVIDIA-NeMo/Lens).

Once the call sites are instrumented, Megatron emits **traces** at training
framework boundaries (training loop, checkpointing, evaluation, P2P
communication, pipeline parallel stages, inference) and **metrics** (loss,
throughput, gradient norm) that export to any OTLP-compatible backend.

## Contents

```
megatron/core/telemetry/
├── telemetry.py         — Megatron's nemo-lens shim and span registry.
├── resource_attrs.py    — Megatron-owned trainer Resource mapping.
├── training_metrics.py  — OTel instruments for the training loop.
└── __init__.py
```

Use the Nemo Lens API for Resource parsing, composition, detection, and
instrumentation. Keep Megatron-specific Resource and signal mappings in
`megatron/core/telemetry/`.

## Optional dependencies

Nothing here requires `nemo-lens` or `opentelemetry` to import. Both are
optional: when neither is installed, `telemetry.py` exposes no-op decorators and
context managers. Metric recorders also become no-ops without OpenTelemetry.
Call sites can therefore import from this module unconditionally.

Install the real implementations with the `otel` extra:

```bash
pip install megatron-core[otel]
```

## Documentation

`docs/user-guide/observability/` holds the full Observability guide:

| Topic | Doc |
|---|---|
| Overview | [index.md](../../../docs/user-guide/observability/index.md) |
| Configuration (env vars, CLI flags) | [configuration.md](../../../docs/user-guide/observability/configuration.md) |
| Span groups and span hierarchy | [span-groups.md](../../../docs/user-guide/observability/span-groups.md) |
| Training and inference metrics | [metrics.md](../../../docs/user-guide/observability/metrics.md) |
| Pipeline-parallel trace correlation | [pipeline-parallel.md](../../../docs/user-guide/observability/pipeline-parallel.md) |
| Adding new instrumentation | [extending.md](../../../docs/user-guide/observability/extending.md) |

For the generic `nemo-lens` documentation (configuration model, instrumentation
primitives, custom exporters, design decisions), see the lens docs at
<https://github.com/NVIDIA-NeMo/Lens/tree/main/docs>.
