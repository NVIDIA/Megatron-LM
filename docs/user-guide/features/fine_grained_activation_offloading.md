<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Fine-Grained Activation Offloading

Contributed in collaboration with RedNote.

Memory is often the limiting factor for very large sparse MoE models such as DeepSeek-V3 and Qwen3-235B. Fine-grained recomputation lowers activation memory at the cost of extra compute. Offloading can use host-device bandwidth so that reload overlaps compute and keeps overhead small in many setups. Fine-grained activation offloading moves activations at module granularity so you can tune how much activation memory leaves the device and adjust training throughput.

Supported offloading modules are `"attn_norm"`, `"core_attn"`, `"attn_proj"`, `"mlp_norm"`, `"expert_fc1"`, `"moe_act"`, and `"group_mlp"`. They can be combined with fine-grained recomputation to free almost all activations for a transformer layer on the device. Use `"group_mlp"` for TE op-fuser GroupedMLP, where FC1, activation, router-prob scaling, and FC2 are fused and therefore offloaded as a single group instead of separate `"expert_fc1"` and `"moe_act"` groups.

## Features

- Pipeline parallelism: PP=1, PP, and interleaved PP
- Compatible with fine-grained recomputation
- FP8 training
- MTP
- Mixed dense and MoE layers
- A2A overlap
- CUDA graphs
  - **Note:** A CUDA graph capture cannot include the offloading modules (temporary limitation).

## Usage

```bash
# Enable fine-grained activation offloading
--fine-grained-activation-offloading

# Modules whose inputs are offloaded (refer to your training script for list or delimiter syntax).
# Choices: "attn_norm", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "moe_act",
#          "group_mlp".
--offload-modules expert_fc1
```

`group_mlp` requires MoE, `--moe-grouped-gemm`, and `--use-transformer-engine-op-fuser`. It cannot be combined with `expert_fc1` or `moe_act` because the TE op-fuser path does not expose those two internal boundaries.
When the TE op fuser saves `GroupedTensor` activations, offloading moves the grouped tensor backing buffers such as row/column data, scales, and offsets independently and rebuilds the grouped wrapper on reload.
The minimum offload size is applied to each `GroupedTensor` backing buffer independently, so small scale or metadata buffers stay on GPU while large data buffers are offloaded.

## Compatible With Fine-Grained Recomputation

- For low-overhead modules such as LayerNorm or `moe_act`, use recomputation to save activation memory.
- For other modules, use offloading to save activation memory.
- Overlap offload and reload with compute when possible.

![Diagram comparing fine-grained activation offloading and fine-grained recomputation across a transformer layer](../../images/fine_grained_activation_offloading/offloading_and_recomputing.png)
