<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Language Models

Megatron Core supports the following language model architectures for large-scale training.

## Converting HuggingFace Models

Use [**Megatron Bridge**](https://github.com/NVIDIA-NeMo/Megatron-Bridge) to convert HuggingFace models to Megatron format. Megatron Bridge is the official standalone converter with support for an extensive list of models including LLaMA, Mistral, Mixtral, Qwen, DeepSeek, Gemma, Phi, Nemotron, and many more.

See the [Megatron Bridge supported models list](https://github.com/NVIDIA-NeMo/Megatron-Bridge?tab=readme-ov-file#supported-models) for the complete and up-to-date list.

## Decoder-Only Models

| Model | Description | Key Features |
|-------|-------------|--------------|
| **GPT** | Generative Pre-trained Transformer | Standard autoregressive LM, foundational architecture |
| **LLaMA** | Meta's LLaMA family | Efficient architecture with RoPE, SwiGLU, RMSNorm |
| **Mistral** | Mistral AI models | Sliding window attention, efficient inference |
| **Mixtral** | Sparse Mixture-of-Experts | 8x7B MoE architecture for efficient scaling |
| **Qwen** | Alibaba's Qwen series | HuggingFace integration, multilingual support |
| **Mamba** | State Space Model | Subquadratic sequence length scaling, efficient long context |

## Hybrid / DSA Models

GLM-5.2 and DeepSeek-V3.2-class models use DeepSeek Sparse Attention (DSA) with absorbed Multi-Latent Attention (`experimental_attention_variant="dsa"` / AbsorbedMLA). Training support is tracked in [#6392](https://github.com/NVIDIA/Megatron-LM/issues/6392).

In-framework Megatron text generation and refit-style eval are not supported for these checkpoints. Both the AbsorbedMLA forward path and the inference-optimized GPT layer spec assert when `experimental_attention_variant="dsa"` is set. See [#7106](https://github.com/NVIDIA/Megatron-LM/issues/7106).

The supported generation path today is:

1. Convert the Megatron checkpoint with [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge).
2. Serve the HuggingFace export with vLLM (`GlmMoeDsaForCausalLM`, as reported in [#7106](https://github.com/NVIDIA/Megatron-LM/issues/7106)).

Do not run Megatron or Megatron-Bridge in-framework generation against a DSA / AbsorbedMLA checkpoint. Those entry points hit the same limitation.

## Encoder-Only Models

| Model | Description | Key Features |
|-------|-------------|--------------|
| **BERT** | Bidirectional Encoder Representations | Masked language modeling, classification tasks |

## Encoder-Decoder Models

| Model | Description | Key Features |
|-------|-------------|--------------|
| **T5** | Text-to-Text Transfer Transformer | Unified text-to-text framework, sequence-to-sequence |

## Example Scripts

Training examples for these models can be found in the `examples/` directory:
- `examples/gpt3/` - GPT-3 training scripts
- `examples/llama/` - LLaMA training scripts
- `examples/mixtral/` - Mixtral MoE training
- `examples/mamba/` - Mamba training scripts
- `examples/bert/` - BERT training scripts
- `examples/t5/` - T5 training scripts

## Model Implementation

All language models are built using Megatron Core's composable transformer blocks, enabling:
- Flexible parallelism strategies (TP, PP, DP, EP, CP)
- Mixed precision training (FP16, BF16, FP8)
- Distributed checkpointing
- Efficient memory management
