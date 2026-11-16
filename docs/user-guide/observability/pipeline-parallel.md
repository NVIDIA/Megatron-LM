<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Pipeline and MCore instrumentation

Detailed MCore instrumentation emits local-rank spans for microbatches,
point-to-point communication, gradient synchronization, and layer work.

## Canonical detailed spans

All spans below are controlled by `megatron.detail`:

| Production site | Span names |
| --- | --- |
| Pipeline schedules | `nv.mcore.microbatch.forward`, `nv.mcore.microbatch.backward` |
| P2P communicator | `nv.mcore.p2p.recv_forward`, `nv.mcore.p2p.recv_backward`, `nv.mcore.p2p.send_forward`, `nv.mcore.p2p.send_backward` |
| Schedule gradient-sync dispatch | `nv.mcore.grad_sync.start` |
| DDP gradient-sync completion | `nv.mcore.grad_sync.finish` |
| Transformer and Mamba layers | `nv.mcore.layer.forward` |
| Transformer attention and MLP | `nv.mcore.layer.self_attention`, `nv.mcore.layer.mlp` |
| Mamba mixer | `nv.mcore.layer.mamba` |

Whole-layer spans carry `nv.mcore.layer.number`.

## Gradient synchronization

The `nv.mcore.grad_sync.start.site` attribute identifies the dispatch:

- `interleaved_backward`
- `interleaved_cooldown`
- `non_interleaved_cooldown`

## Enabling detailed instrumentation

```bash
MEGATRON_OTEL_ENABLED=1
MEGATRON_OTEL_SPAN_GROUPS=default,megatron.detail
```

Use `profiling` to enable every Megatron group or `all` for every library's
registered groups. The `default` preset excludes `megatron.detail`.
