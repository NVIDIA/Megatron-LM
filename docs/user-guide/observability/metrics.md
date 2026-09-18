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
