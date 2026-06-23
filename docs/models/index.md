<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Supported Models

Megatron Core supports a wide range of language and multimodal models with optimized implementations for large-scale training.

## Model Conversion

For converting HuggingFace models to Megatron format, use [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), the official standalone converter. Megatron Bridge supports an extensive list of models including LLaMA, Mistral, Mixtral, Qwen, DeepSeek, Gemma, Phi, Nemotron, and many more.

See the [Megatron Bridge supported models list](https://github.com/NVIDIA-NeMo/Megatron-Bridge?tab=readme-ov-file#supported-models) for the complete and up-to-date list of supported models.

```{toctree}
:maxdepth: 1

llms
multimodal
../llama_mistral
```
