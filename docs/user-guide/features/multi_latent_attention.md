<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Multi-Latent Attention

## Multi-Latent Attention overview

Multi-Latent Attention ("MLA") is an innovative attention mechanism introduced by Deepseek team that enhances the efficiency of attention computation by leveraging multiple latent spaces. This approach is particularly beneficial for large language models (LLMs), as it reduces the computational burden associated with traditional attention mechanisms. According to Deepseek-V2 technical report, MLA achieves better performance compared to Multi-Head Attention (MHA) and requires smaller KV cache.

## Enabling Multi-Latent Attention

To enable MLA in Megatron-LM, set the following flags in command line:
- `--multi-latent-attention` to enable MLA in MLP.
- Set `MLATransformerConfig` to configure MLA.

