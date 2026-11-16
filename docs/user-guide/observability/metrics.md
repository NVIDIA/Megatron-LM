<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Training Signals

Training reports use span events. Existing stdout, timer, TensorBoard,
W&B, and one-logger names and behavior are unchanged. The legacy OTel
loss/throughput/gradient metric emission is retired at this reporting boundary.

## Training report events

At the normal training-log cadence, the global last rank emits these events:

| Event | Contents |
|---|---|
| `nv.dl.training.objective` | One occurrence per objective key, with `nv.dl.training.step`, objective name and value, and `nv.dl.measurement.domain=dp_mean`. The optional weight is omitted unless its exact applied coefficient is already available. |
| `nv.dl.training.numerics` | One occurrence containing available host values such as learning rate, global batch size, consumed samples, parameter norm, gradient-zero and skipped/NaN counts, and the current loss scale when gradient scaling is enabled (FP16 or an explicit nonzero loss scale, including BF16). Its fields have heterogeneous reduction semantics, so the event has no measurement-domain attribute. |

`nv.dl.training.step` is the one-based training step. Report events attach to
the active `nv.mlm.train.log` span when the `megatron.train` group is enabled.
If that span is not recording, they fall back to the current recording
loop-pass root. If neither is recording, event emission is a no-op.

## Processed-token metric

`nv.dl.training.tokens.processed` is the only retained OTel training metric.
It is a monotonic counter with unit `{token}` and no metric attributes.

The counter reports the global processed-token value on each exporting rank.
Select one rank series per run rather than summing across ranks.

- Unpacked training records global batch size multiplied by sequence length.
- Packed training records the already-global real-token count from the existing
  sequence-length statistics path. Non-finite, fractional, or negative values
  are skipped with a warning.
- Configured rejected-data dummy skips, inference-only execution, exceptions,
  and exits before iteration commit do not increment the counter.

## Adding custom metrics

For a Megatron-specific metric, use a project-owned name such as
`megatron.my_subsystem.requests` and follow the weak-reference instrument-cache
and failure-isolation pattern in
`megatron/core/telemetry/training_metrics.py`. Shared `nv.dl.*`, `nv.mcore.*`,
and standard namespaces require a versioned schema definition; do not mint
ad hoc names in call sites.

See [lens: metrics](https://github.com/NVIDIA-NeMo/Lens/blob/main/docs/user-guide/metrics.md)
for general instrument guidance.
