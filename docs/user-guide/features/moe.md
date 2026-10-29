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

multi_token_prediction
multi_latent_attention
../../api-guide/router_replay
```

```{include} ../../../megatron/core/transformer/moe/README.md
```


### Per-token router replay

`RouterReplay.set_target_indices(indices, replay_mask=mask)` can replay only a
subset of tokens. `indices` has shape `[num_tokens, topk]`; each true entry of
`mask` selects the supplied experts, while false entries use the router's native
selection (including group-limited or hash routing). Omitting `mask` retains
full replay. Scores remain differentiable for the selected experts.

Set targets before each `REPLAY_FORWARD` call. Both replayed and natively selected
forward routes are saved for the corresponding `REPLAY_BACKWARD` call, in FIFO
microbatch order. Recomputation reuses these resolved indices, so unmasked rows
do not make another routing decision. `clear_indices()` clears targets, masks,
and pending recomputation routes; callers should clear state on failed steps.
