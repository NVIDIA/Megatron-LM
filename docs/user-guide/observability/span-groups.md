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
