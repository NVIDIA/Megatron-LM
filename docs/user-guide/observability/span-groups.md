<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Span Groups

Span granularity is controlled by `MEGATRON_OTEL_SPAN_GROUPS`, with
`NEMO_LENS_SPAN_GROUPS` as the shared fallback. The specification accepts
presets, individual namespaced group names, or a mix.

Megatron registers its groups through `megatron/core/telemetry/telemetry.py`
when that module imports. Lens owns resolution and the built-in `all`
wildcard; consumers do not maintain separate copies of Lens group classes.

## Presets

| Preset | Groups included |
| --- | --- |
| `default` | `megatron.job`, `megatron.ckpt`, `megatron.eval`, `megatron.inference` |
| `per_step` | `default` plus `megatron.train` |
| `profiling` | Every Megatron group, including `megatron.detail` |
| `all` | Every group registered in this process, including other libraries |

## Group ownership

| Group | Intended instrumentation scope |
| --- | --- |
| `megatron.job` | Coarse setup and training lifecycle |
| `megatron.train` | Trainer iterations and reporting |
| `megatron.ckpt` | Trainer-visible checkpoint work |
| `megatron.eval` | Evaluation |
| `megatron.inference` | Inference request and generation work |
| `megatron.detail` | High-cardinality MCore layers, microbatches, and communication |

A group's registration does not itself create spans. The instrumentation at a
call site determines its emitted name and attributes. Old unnamespaced names
such as `job`, `checkpoint`, `step`, and `microbatch` are not Megatron aliases.

## Examples

```bash
# Coarse production instrumentation
MEGATRON_OTEL_SPAN_GROUPS=default

# Add trainer-iteration visibility
MEGATRON_OTEL_SPAN_GROUPS=per_step

# Add high-cardinality MCore internals
MEGATRON_OTEL_SPAN_GROUPS=default,megatron.detail

# Include every imported library's registered groups
MEGATRON_OTEL_SPAN_GROUPS=all
```

When telemetry or a group is disabled, its group-gated helpers do not create
spans; the application body still executes normally. Keep expensive argument
construction inside an explicit group check. Use detailed instrumentation for
targeted profiling, not as a reason to change model computation.

## Training and checkpoint span contracts

| Operation | Span name | Group |
| --- | --- | --- |
| Attempted training iteration | `nv.dl.training.iteration` | `megatron.train` |
| Forward/backward wrapper | `nv.dl.training.iteration.forward_backward` | `megatron.train` |
| Optimizer update | `nv.dl.training.iteration.optimizer_step` | `megatron.train` |
| Reporting interval | `nv.mlm.train.iteration_report` | `megatron.train` |
| Parameter norm and ordinary logging | `nv.mlm.train.params_norm`, `nv.mlm.train.log` | `megatron.train` |
| First executed iteration's pre-hook enablement | `nv.mlm.train.forward_pre_hook` | `megatron.train` |
| Trainer-visible checkpoint save | `nv.dl.training.checkpoint.exposed_save` | `megatron.ckpt` |
| Save operation | `nv.dl.training.checkpoint.save` | `megatron.ckpt` |
| State dictionary and I/O dispatch | `nv.dl.training.checkpoint.save.state_dict`, `nv.dl.training.checkpoint.save.io_write` | `megatron.ckpt` |
| Queue finalization cost | `nv.dl.training.checkpoint.save.finalize` | `megatron.ckpt` |
| Checkpoint load and I/O | `nv.dl.training.checkpoint.load`, `nv.dl.training.checkpoint.load.io_read` | `megatron.ckpt` |
| Checkpoint reporting and heartbeat | `nv.mlm.checkpoint.report_memory`, `nv.mlm.checkpoint.timers_log`, `nv.mlm.checkpoint.ft_heartbeat` | `megatron.ckpt` |
| Evaluation and evaluation iteration | `nv.dl.training.evaluate`, `nv.dl.training.evaluate.step` | `megatron.eval` |
| Periodic GPU sniff | `nv.dl.resiliency.gpu_sniff.periodic` | `megatron.job` |
| CUDA graph capture | `nv.mcore.cuda_graph.capture` | `megatron.job` |
| Explicit memory reclaim | `nv.mcore.memory.reclaim` | `megatron.train` or `megatron.ckpt` |

`nv.dl.training.step` is the one-based training step. The first executed
iteration of each run has `nv.dl.training.iteration.is_first=true`.
`nv.dl.training.iteration.skipped` and
`nv.dl.training.optimizer.update_successful` distinguish attempted work from a
successful optimizer update.

## MCore detailed instrumentation

`megatron.detail` owns the canonical `nv.mcore.microbatch.*`, `nv.mcore.p2p.*`,
`nv.mcore.grad_sync.*`, and `nv.mcore.layer.*` operation spans. See
[Pipeline and MCore instrumentation](pipeline-parallel.md) for their names and
attributes.
