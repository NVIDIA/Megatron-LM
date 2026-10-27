<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Mixture of Experts

```{toctree}
:maxdepth: 1
:caption: MoE Features

moe_a2a_overlap
multi_token_prediction
multi_latent_attention
../../api-guide/router_replay
```

```{include} ../../../megatron/core/transformer/moe/README.md
```


### HybridEP route precision for scheduler comparisons

HybridEP normally reduces expert contributions locally in FP32, rounds each rank
partial to BF16, and then combines the rank partials. Relocating an expert changes
these intermediate rounding points, including the input-gradient reduction.
Consequently, equal logical routing and expert weights need not produce identical
outputs or training trajectories across placement policies.

`--moe-hybridep-preserve-route-precision` keeps each top-k route separate through
HybridEP communication and performs one final FP32 reduction per logical token,
in both forward and input-gradient computation. Enable it on **both** the native
baseline and scheduler runs when testing placement-independent numerics. It does
not change replica gradient transport or expert computation, and it does not
promise bitwise determinism of GEMMs, optimizer updates, or complete training.

This mode requires flex/HybridEP with compact indices and does not support
pad-to-capacity routing. Token communication and metadata capacity can grow by up
to top-k; measure memory and throughput separately from the default fast path.
The default remains disabled. A baseline recorded without this option is not the
matching reference for a scheduler run with the option enabled.
