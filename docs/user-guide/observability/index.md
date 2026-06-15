# Observability

Megatron-LM is instrumented with [OpenTelemetry](https://opentelemetry.io/) via the [`nemo-lens`](https://github.com/NVIDIA-NeMo/Lens) library, emitting **traces** at training-framework boundaries and **metrics** for loss, throughput, and gradient norm.

Telemetry exports to any OTLP-compatible backend (Jaeger, Grafana Tempo, W&B Weave, Honeycomb, Datadog, ...).

## What's in this section

```{toctree}
:maxdepth: 1

configuration
span-groups
metrics
pipeline-parallel
observability-stack
extending
```

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

With `default` span groups, Megatron emits a handful of coarse spans per iteration and a steady stream of training metrics. Switch to `per_step` for profiling individual steps, or `all` for fine-grained debugging.

## What gets instrumented

| Subsystem | File | Spans |
|---|---|---|
| Training loop | `megatron/training/training.py` | `megatron.pretrain`, `megatron.train`, `megatron.train_step`, `megatron.forward_backward`, `megatron.optimizer_step` |
| Pipeline schedules | `megatron/core/pipeline_parallel/schedules.py` | `megatron.microbatch.forward`, `megatron.microbatch.backward`, `megatron.pp.recv_forward.linked` |
| P2P communication | `megatron/core/pipeline_parallel/p2p_communication.py` | `megatron.p2p.{send,recv}_{forward,backward}` |
| Gradient sync (DDP) | `megatron/core/distributed/distributed_data_parallel.py` | `megatron.grad_sync.{start,finish}` |
| Checkpointing | `megatron/training/checkpointing.py` | `megatron.save_checkpoint.*`, `megatron.load_checkpoint.*` |
| Model init | `megatron/training/training.py` | `megatron.model_init` |
| Evaluation | `megatron/training/training.py` | `megatron.evaluate`, `megatron.evaluate.step` |
| Inference server | `megatron/core/inference/text_generation_server/` | `text_completion {model}` (GenAI semconv) |

Each span is tagged with a **span group** that controls whether it's emitted at runtime. See [Span Groups](span-groups.md).

## What gets exported

- **Traces**: Jaeger / Tempo / Honeycomb / etc. via OTLP.
- **Metrics**: Prometheus via the OTel Collector, or direct OTLP to Grafana Mimir / Datadog / etc.
- **Logs** (optional): via the OTel log bridge when `MEGATRON_OTEL_LOGS_ENABLED=1` — correlates `logging` records with the active span's trace ID.

By default, only **one rank** exports (the last rank). For multi-rank telemetry, see [Configuration — Export strategy](configuration.md#export-strategy).

## Related

- Lens configuration model and env vars: [lens: configuration](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/configuration.md)
- Instrumentation primitives (`managed_span`, `trace_fn`, `span_cm`): [lens: instrumentation](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/instrumentation.md)
- Observability stack (docker-compose): [Observability Stack](observability-stack.md)
