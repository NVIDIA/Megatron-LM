<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Configuration

## CLI flags

| Flag | Type | Description |
|---|---|---|
| `--otel-enabled` | flag | Enable OTel telemetry |
| `--otel-service-name NAME` | string | Override `OTEL_SERVICE_NAME` |

These flags are processed in `megatron/training/global_vars.py:_set_telemetry()`
and override the corresponding env vars. Span groups are configured through
`MEGATRON_OTEL_SPAN_GROUPS` or the shared fallback `NEMO_LENS_SPAN_GROUPS`.

## Megatron-specific environment variables

Each `MEGATRON_OTEL_*` variable is an **alias** for the corresponding [`NemoLensConfig` field](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/configuration.md) with `NEMO_LENS_*` as fallback — they are not independent settings. Setting `MEGATRON_OTEL_ENABLED=1` is equivalent to setting `NEMO_LENS_ENABLED=1`; they refer to the same underlying config. The prefix/fallback model lets Megatron scope its own env vars while still inheriting lens defaults from a shared environment.

| Variable | Default | Description |
|---|---|---|
| `MEGATRON_OTEL_ENABLED` | `0` | Master toggle; must be set to `1` to activate |
| `MEGATRON_OTEL_TRACES_ENABLED` | `1` | Enable trace spans |
| `MEGATRON_OTEL_METRICS_ENABLED` | `1` | Enable metrics instruments |
| `MEGATRON_OTEL_LOGS_ENABLED` | `0` | Enable OTel log bridge |
| `MEGATRON_OTEL_SPAN_GROUPS` | `default` | Span granularity spec (see [Span Groups](span-groups.md)) |
| `MEGATRON_OTEL_EXPORTER` | `otlp` | Exporter backend: `otlp` or `console` |
| `NEMO_LENS_USER_ID` | (empty) | Optional user/team label |

For the full config model, field semantics, and validation rules, see
[lens: configuration](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/configuration.md).

## Standard OTel SDK variables

All standard OTel SDK env vars are honoured by the SDK directly:

| Variable | Example |
|---|---|
| `OTEL_SERVICE_NAME` | `megatron-training` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc` or `http/protobuf` |
| `OTEL_EXPORTER_OTLP_HEADERS` | `Authorization=Bearer <token>` |
| `OTEL_TRACES_SAMPLER` | `parentbased_traceidratio` |
| `OTEL_TRACES_SAMPLER_ARG` | `0.1` |

## Run identification

`nv.dl.run.uuid` is the unique execution-instance identifier shared by the
processes and ranks participating in that execution. Megatron preserves an
inherited launcher value. When no value is inherited, Megatron telemetry setup
uses Lens to derive the attempt identity from Slurm or TorchElastic runtime
information. If neither scheduler nor local run identity is available, the
attribute is omitted. The submitted-job identity, `nv.dl.job.uuid`, comes from
inherited launcher data or Lens Slurm detection.

All ranks in one run should share `nv.dl.run.uuid`; each process is distinguished
by `nv.dl.rank`, `nv.dl.world_size`, and `nv.dl.local_rank`.

## Resource attributes

Megatron's `_set_telemetry()` supplies trainer Resource attributes so they
appear as process-level tags across every span in the run. It uses Lens's shared
dictionary composition once: trainer argument and locally detected GPU values
are defaults, decoded inherited launcher values are current, and Lens Slurm and
Kubernetes detector results plus trainer-owned identity are overrides. Later
layers win, including when a current value is empty. Megatron sets
`nv.dl.role=trainer`. Parent `host.name`, `process.pid`, and `host.gpu.count`
values are removed after composition so Lens detects them for the current
trainer process.

The service name resolves in this order: nonempty `--otel-service-name`,
non-whitespace `OTEL_SERVICE_NAME`, inherited `service.name`, then
`megatron-lm`.

When telemetry is enabled, Megatron passes this composed application map to
Lens's exact publication context for the trainer lifetime. Spawned workers
therefore inherit the same resolved input. Telemetry shutdown first shuts down
Lens and then exits the publication context, which restores the exact incoming
environment state, including the difference between an absent variable and an
empty one.

| Attribute | Source |
|---|---|
| `nv.dl.rank` | `args.rank` |
| `nv.dl.world_size` | `args.world_size` |
| `nv.dl.local_rank` | `args.local_rank` |
| `nv.dl.role` | `trainer` |
| `nv.dl.provider.name` | `mcore` |
| `nv.dl.run.uuid` | inherited launcher environment, otherwise Lens-derived attempt identity |
| `nv.dl.topology.size.tp` | `args.tensor_model_parallel_size` |
| `nv.dl.topology.size.pp` | `args.pipeline_model_parallel_size` |
| `nv.dl.topology.size.dp` | `args.data_parallel_size` |
| `nv.dl.training.config.global_batch_size` | `args.global_batch_size` |
| `nv.dl.training.config.micro_batch_size` | `args.micro_batch_size` |
| `nv.dl.training.config.sequence_length` | `args.seq_length` |
| `nv.dl.training.config.optimizer` | `args.optimizer` |
| `nv.dl.training.config.recompute_granularity` | `args.recompute_granularity` |
| `nv.dl.training.target.train_iters` | `args.train_iters` |
| `nv.dl.training.target.train_samples` | `args.train_samples`, or derived from iterations and batch size |
| `nv.dl.training.target.train_tokens` | `args.train_tokens`, or derived from samples and sequence length |
| `nv.dl.software.torch` | PyTorch version |
| `nv.dl.software.cuda` | PyTorch CUDA version |
| `nv.dl.software.nccl` | PyTorch NCCL version |
| `nv.dl.software.transformer_engine` | Transformer Engine package version |
| `nv.mcore.version` | Megatron Core package version |
| `nv.gpu.index` | inherited value or Lens GPU detector called by Megatron |
| `nv.gpu.model` | inherited value or Lens GPU detector called by Megatron |
| `nv.gpu.uuid` | inherited value or Lens GPU detector called by Megatron |
| `nv.gpu.serial` | inherited value or Lens GPU detector called by Megatron |
| `nv.gpu.pci_bus_id` | inherited value or Lens GPU detector called by Megatron |
| `nv.gpu.compute_capability` | inherited value or Lens GPU detector called by Megatron |
| `nv.gpu.memory_total` | inherited value or Lens GPU detector called by Megatron |
| `nv.gpu.driver_version` | inherited value or Lens GPU detector called by Megatron |

Lens Slurm detection can also add scheduler Resource attributes such as
`slurm.job.id`, `slurm.job.id.raw`, `slurm.array.job_id`,
`slurm.array.task_id`, `slurm.array.count`, `slurm.sluid`,
`slurm.array.sluid`, `slurm.cluster.name`, `slurm.partition`,
`slurm.nnodes`, `slurm.ntasks`, and `slurm.restart_count`.

## Typical configurations

### Local development with console exporter

```bash
export MEGATRON_OTEL_ENABLED=1
export MEGATRON_OTEL_EXPORTER=console
python examples/run_simple_mcore_train_loop.py
```

Spans and metrics print to stdout.

### Local collector

Point Megatron at an OTLP endpoint on localhost — an OpenTelemetry Collector, or
a backend such as Jaeger that accepts OTLP directly:

```bash
export MEGATRON_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
torchrun --nproc_per_node=8 pretrain_gpt.py ...
```

For a local stack to send this to, see
[lens: sending telemetry to a backend](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/observability/backends.md).

### Production with remote collector

```bash
export MEGATRON_OTEL_ENABLED=1
export MEGATRON_OTEL_SPAN_GROUPS=default
export OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector-host>:4317
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer <token>"
python pretrain_gpt.py ...
```

### Per-step granularity with trace sampling

```bash
export MEGATRON_OTEL_ENABLED=1
export MEGATRON_OTEL_SPAN_GROUPS=per_step
export OTEL_TRACES_SAMPLER=parentbased_traceidratio
export OTEL_TRACES_SAMPLER_ARG=0.1    # keep 10% of traces
```
