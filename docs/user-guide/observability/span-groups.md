# Span Groups

Span granularity in Megatron is controlled by the `MEGATRON_OTEL_SPAN_GROUPS` env var (or `--otel-span-groups` CLI flag). The spec accepts preset keywords, individual group names, or a mix.

For the general span-group mechanism see
[lens: span groups](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/span-groups.md).
This page covers Megatron's extensions and the complete span hierarchy.

## Preset keywords

| Preset | Groups included | Relative cost |
|---|---|---|
| `default` | `job`, `checkpoint`, `evaluate`, `inference` | Lowest — safe for production |
| `per_step` | `default` + `model_init`, `load_checkpoint`, `step`, `forward_backward`, `optimizer`, `communication`, `data_loading` | Moderate — use with sampling |
| `all` | everything including `microbatch`, `layer`, `activation_offload` | Highest — dev/debug only |

## `MegatronSpanGroup`

Defined in `megatron/core/telemetry/span_groups.py`. Extends lens's base `SpanGroup` with Megatron-specific groups:

| Group | Spans emitted | Typical frequency |
|---|---|---|
| `job` | `megatron.pretrain`, `megatron.train` | once per job |
| `checkpoint` | `megatron.save_checkpoint`, `megatron.save_checkpoint.state_dict`, `megatron.save_checkpoint.io_write` | every checkpoint |
| `evaluate` | `megatron.evaluate`, `megatron.evaluate.step` | every eval interval |
| `model_init` | `megatron.model_init` | once at startup |
| `load_checkpoint` | `megatron.load_checkpoint`, `megatron.load_checkpoint.io_read` | once at startup |
| `step` | `megatron.train_step` | every iteration |
| `forward_backward` | `megatron.forward_backward` | every iteration |
| `optimizer` | `megatron.optimizer_step` | every iteration |
| `microbatch` | `megatron.microbatch.forward`, `megatron.microbatch.backward` | every microbatch |
| `layer` | `megatron.layer.forward`, `megatron.layer.self_attention`, `megatron.layer.mlp` | every layer per microbatch |
| `communication` | `megatron.p2p.{recv,send}_{forward,backward}`, `megatron.grad_sync.{start,finish}`, `megatron.pp.recv_forward.linked` | every iteration |
| `activation_offload` | `megatron.activation.offload`, `megatron.activation.reload` | every microbatch |
| `data_loading` | (reserved for future use) | every iteration |
| `inference` | `text_completion {model}` | every inference request |

## Examples

```bash
# Coarse spans only — default
MEGATRON_OTEL_SPAN_GROUPS=default

# Include per-step spans
MEGATRON_OTEL_SPAN_GROUPS=per_step

# Default + microbatch only (skip step/optimizer groups)
MEGATRON_OTEL_SPAN_GROUPS=default,microbatch

# Everything
MEGATRON_OTEL_SPAN_GROUPS=all
```

## Span hierarchy

The full tree of spans Megatron can emit, with the controlling span group shown per span:

```
megatron.pretrain                                          # job
  ├── megatron.model_init                                  # model_init
  ├── megatron.load_checkpoint                             # load_checkpoint
  │     └── megatron.load_checkpoint.io_read               # load_checkpoint
  └── megatron.train                                       # job
        ├── megatron.train_step                            # step
        │     ├── megatron.forward_backward                # forward_backward
        │     │     ├── megatron.microbatch.forward        # microbatch (×N)
        │     │     │     └── megatron.layer.forward       # layer (×L per microbatch)
        │     │     │           ├── megatron.layer.self_attention
        │     │     │           └── megatron.layer.mlp
        │     │     ├── megatron.microbatch.backward       # microbatch (×N)
        │     │     ├── megatron.pp.recv_forward.linked    # communication — link to sender's context (PP > 1)
        │     │     ├── megatron.p2p.recv_forward          # communication
        │     │     ├── megatron.p2p.send_forward          # communication
        │     │     ├── megatron.p2p.recv_backward         # communication
        │     │     ├── megatron.p2p.send_backward         # communication
        │     │     ├── megatron.activation.offload        # activation_offload
        │     │     └── megatron.activation.reload         # activation_offload
        │     ├── megatron.grad_sync.start                 # communication
        │     ├── megatron.grad_sync.finish                # communication
        │     └── megatron.optimizer_step                  # optimizer
        ├── megatron.save_checkpoint                       # checkpoint
        │     ├── megatron.save_checkpoint.state_dict      # checkpoint
        │     └── megatron.save_checkpoint.io_write        # checkpoint
        └── megatron.evaluate                              # evaluate
              └── megatron.evaluate.step                   # evaluate (×N)

text_completion {model}                                    # inference (GenAI semconv)
```

## Span attributes

Key Megatron-specific span attributes:

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
| `dl.pipeline_parallel.rank` | int | `megatron.pp.recv_forward.linked` |
| `dl.microbatch_id` | int | `megatron.pp.recv_forward.linked` (warmup only) |

## Granularity guidance

| Span groups | Relative cost | Recommendation |
|---|---|---|
| Disabled (`MEGATRON_OTEL_ENABLED=0`) | None | Default for smoke tests |
| `default` | Lowest | Safe for all production runs |
| `per_step` | Moderate | Use with `OTEL_TRACES_SAMPLER` |
| `all` (includes microbatch, layer) | Highest | Development / profiling only |

Non-exporting ranks have `frozenset()` span groups — `is_span_group_enabled()` returns `False` everywhere, so **no span objects are created at all**. The disabled path is a frozenset lookup followed by an immediate return, not a no-op span that still allocates. See [lens: architecture](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/design/architecture.md).

## Inference spans (GenAI semconv)

The inference server (`MegatronGenerate`) emits spans that follow the [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/).

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

W3C TraceContext is extracted from incoming HTTP headers so upstream callers can propagate trace context into the Megatron span.
