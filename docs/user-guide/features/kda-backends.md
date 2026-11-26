<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Kimi Delta Attention Backends

Megatron Core supports selectable recurrence backends for Kimi Delta Attention
(KDA). Set `TransformerConfig.kda_backend`; hybrid model construction copies it
to each `KDALayerConfig`.

| Backend | Configuration | Scope |
|---|---|---|
| Flash Linear Attention | `kda_backend="fla"` | Default backend; preserves existing behavior |
| cuDNN | `kda_backend="cudnn"` | Opt-in local and chunkwise context-parallel training backend |

Backend selection is strict. Megatron raises an error when the selected backend
or one of its requirements is unavailable; it does not silently change the
backend.

## Configure the cuDNN backend

The cuDNN backend requires:

- Megatron's existing Flash Linear Attention dependency for the shared KDA
  convolution, normalization, and packed-sequence preprocessing;
- `nvidia-cudnn-frontend[cutedsl]` 1.32.0 or newer (the development environment
  already pins 1.32.0);
- a Blackwell SM100-class GPU or newer;
- equal key and value head dimensions of 64 or 128; and
- `linear_cp_mode="chunkwise"` when the context-parallel size is greater than
  one.

The backend selection changes the recurrent KDA operator. It does not replace
the shared preprocessing and output-normalization components of the full KDA
layer.

For example:

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    # Other TransformerConfig and KDA fields are omitted.
    kda_backend="cudnn",
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_cp_mode="chunkwise",
)
```

With context parallelism, use packed sequence metadata or a fixed local token
partition from Megatron's chunkwise KDA path. The CP context must include CPU
sequence boundaries. They locate the first and last local sequence without a
device-to-host synchronization inside the cuDNN adapter.

The adapter summarizes only the sequence crossing each rank boundary. One
packed all-gather carries the state and transition summaries, and cuDNN FROST
state chains compose the state entering the local forward and leaving the local
backward. Sequences that start and finish on one rank remain local.

## Gate behavior

`kda_safe_gate=True` enables the bounded in-kernel gate and requires
`kda_lower_bound` in the range `[-5, 0)`. The same gate configuration is used
for local execution and context parallelism.

## Limitations

- The backend implements training forward and backward. It does not expose
  recurrent state outputs or intermediate states.
- Context parallelism supports the chunkwise layout only. Headwise KDA remains
  available through the FLA backend.
- KDA chunkwise CP is incompatible with `deterministic_mode` at the full-layer
  level because its convolution fallback cannot exchange history across ranks.
  Operator-level cuDNN KDA and FROST replay are nevertheless covered by the
  Blackwell bit-exact determinism test.
