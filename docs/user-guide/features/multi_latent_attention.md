<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Multi-Latent Attention

## Multi-Latent Attention Overview

Multi-Latent Attention (MLA) is an attention variant from the DeepSeek team. It uses multiple latent spaces to change how attention is computed. That design often lowers cost for large language models (LLMs) compared with standard attention and can shrink the KV cache. The DeepSeek-V2 technical report compares MLA to Multi-Head Attention (MHA) on quality and cache size.

## Enabling Multi-Latent Attention

To enable MLA in Megatron-LM, set the following on the command line:

- `--multi-latent-attention` to turn on MLA.
- Use `MLATransformerConfig` for MLA-specific model settings when you build the training configuration.
