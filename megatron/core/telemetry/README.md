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
├── span_groups.py       — MegatronSpanGroup: Megatron-specific span groups.
├── training_metrics.py  — OTel instruments for the training loop.
├── fallbacks.py         — No-op shims for when nemo-lens is not installed.
└── __init__.py
```

Resource detection and the instrumentation primitives themselves live in
`nemo-lens`. This module is a thin integration layer.

## Optional dependencies

Nothing here requires `nemo-lens` or `opentelemetry` to import. Both are
optional: when neither is installed, `fallbacks` supplies no-op decorators and
context managers, `span_groups` falls back to a local `SpanGroup` stub, and
`record_training_metrics()` returns immediately. Call sites can therefore import
from this module unconditionally.

To pull in the real implementations, install
[`nemo-lens`](https://github.com/NVIDIA-NeMo/Lens) alongside Megatron. A
`megatron-core[otel]` extra that does this for you arrives with the
call-site instrumentation.

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

## Status

This module is the base layer. The call-site instrumentation (training loop,
checkpointing, pipeline schedules, inference server), the `otel` install extra,
the `--otel-*` CLI flags, the `MEGATRON_OTEL_*` environment variables, and the
`docs/index.md` toctree entry that publishes the guide above land separately;
see <https://github.com/NVIDIA/Megatron-LM/pull/6385>.
