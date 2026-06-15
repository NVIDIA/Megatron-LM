# Configuration

## CLI flags

| Flag | Type | Description |
|---|---|---|
| `--otel-enabled` | flag | Enable OTel telemetry |
| `--otel-service-name NAME` | string | Override `OTEL_SERVICE_NAME` |
| `--otel-span-groups SPEC` | string | Comma-separated span-group spec (see [Span Groups](span-groups.md)) |

These flags are processed in `megatron/training/global_vars.py:_set_telemetry()` and override the corresponding env vars.

## Megatron-specific environment variables

Each `MEGATRON_OTEL_*` variable is an **alias** for the corresponding [`NemoLensConfig` field](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/configuration.md) with `NEMO_LENS_*` as fallback — they are not independent settings. Setting `MEGATRON_OTEL_ENABLED=1` is equivalent to setting `NEMO_LENS_ENABLED=1`; they refer to the same underlying config. The prefix/fallback model lets Megatron scope its own env vars while still inheriting lens defaults from a shared environment.

| Variable | Default | Description |
|---|---|---|
| `MEGATRON_OTEL_ENABLED` | `0` | Master toggle; must be set to `1` to activate |
| `MEGATRON_OTEL_RANK_STRATEGY` | `single_rank` | `single_rank`, `all_ranks`, `sampled`, `first_rank_per_node`, or any name registered via `register_rank_strategy()` |
| `MEGATRON_OTEL_EXPORT_RANK` | `-1` | For `single_rank`: which rank exports. `-1` = last rank |
| `MEGATRON_OTEL_EXPORT_SAMPLE_RATE` | `1.0` | For `sampled`: fraction in `[0.0, 1.0]` |
| `MEGATRON_OTEL_SAMPLING_STRATEGY` | (empty) | `rank_aware` or any name registered via `register_sampling_strategy()`. Empty leaves the OTel SDK default sampler in place. |
| `MEGATRON_OTEL_TRACES_ENABLED` | `1` | Enable trace spans |
| `MEGATRON_OTEL_METRICS_ENABLED` | `1` | Enable metrics instruments |
| `MEGATRON_OTEL_LOGS_ENABLED` | `0` | Enable OTel log bridge |
| `MEGATRON_OTEL_SPAN_GROUPS` | `default` | Span granularity spec (see [Span Groups](span-groups.md)) |
| `MEGATRON_OTEL_EXPORTER` | `otlp` | Exporter backend: `otlp` or `console` |
| `NEMO_LENS_RUN_ID` | (auto) | Unique run identifier. Auto-detected from `SLURM_JOB_ID` or generated UUID |
| `NEMO_LENS_USER_ID` | (empty) | Optional user/team label |

For the full config model, field semantics, and validation rules, see
[lens: configuration](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/configuration.md).

## Rank strategy

Controls which ranks actually send telemetry. Four strategies are available: `single_rank` (default), `all_ranks`, `sampled`, and `first_rank_per_node`, configured via `MEGATRON_OTEL_RANK_STRATEGY` above.

See [lens: sampling](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/sampling.md) for detailed semantics, when to use each, and how they compose with OTel SDK samplers.

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

## Run Identification

Each training run is automatically assigned a unique `nemo.run.id` resource attribute that flows to all backends.

**Priority order:**

1. `NEMO_LENS_RUN_ID` env var (explicit, highest priority)
2. `SLURM_JOB_ID` env var (auto-detected on SLURM clusters)
3. Auto-generated 12-character UUID (fallback)

All ranks in a distributed job share the same `run_id`. Each rank gets a unique `service.instance.id` of `{run_id}-rank{rank}`.

Filter by `nemo.run.id` in Jaeger, Grafana, Kibana to isolate a specific run.

## Resource attributes

Megatron's `_set_telemetry()` sets training-config attributes on the OTel `Resource` so they appear as Jaeger "Process" tags across every span in the run:

| Attribute | Megatron source |
|---|---|
| `dl.local_rank` | `args.local_rank` |
| `dl.tensor_parallel.size` | `args.tensor_model_parallel_size` |
| `dl.pipeline_parallel.size` | `args.pipeline_model_parallel_size` |
| `dl.data_parallel.size` | `args.data_parallel_size` |
| `dl.batch_size` | `args.global_batch_size` |
| `dl.sequence_length` | `args.seq_length` |
| `megatron.num_layers` | `args.num_layers` |
| `megatron.hidden_size` | `args.hidden_size` |
| `megatron.num_attention_heads` | `args.num_attention_heads` |
| `megatron.train_iters` | `args.train_iters` |
| `megatron.micro_batch_size` | `args.micro_batch_size` |
| `megatron.ckpt_format` | `args.ckpt_format` |
| `megatron.precision` | `fp16` / `bf16` / `fp32` |

Plus auto-detected attributes from lens's [resource detection](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/resources.md): hostname, PID, GPU count, SLURM metadata, Kubernetes metadata.

## Typical configurations

### Local development with console exporter

```bash
export MEGATRON_OTEL_ENABLED=1
export MEGATRON_OTEL_EXPORTER=console
python examples/run_simple_mcore_train_loop.py
```

Spans and metrics print to stdout.

### Local stack (Jaeger + Prometheus + Grafana)

```bash
docker compose -f docker-compose.otel.yml up -d

export MEGATRON_OTEL_ENABLED=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
torchrun --nproc_per_node=8 pretrain_gpt.py ...
```

UI: Jaeger at `:16686`, Grafana at `:3000`. See [Observability Stack](observability-stack.md).

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
