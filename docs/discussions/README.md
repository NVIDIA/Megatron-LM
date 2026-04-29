---
orphan: true
---

<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Megatron Discussions

This directory contains in-depth guides, tutorials, and discussions about optimizing and using Megatron for various use cases.

## Available Guides

### Training Guides

- **[Megatron-FSDP User Guide](megatron-fsdp-user-guide/megatron-fsdp-user-guide.md)**

  A practical guide to enable Megatron-FSDP training, including a quick-start example for DeepSeek-V3, required and recommended configurations, and instructions for checkpoint conversion from torch_dist to fsdp_dtensor.

## Previous News

- **[2025/05]** Megatron Core v0.11.0 brings new capabilities for multi-data center LLM training ([blog](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/)).
- **[2024/07]** Megatron Core v0.7 improves scalability and training resiliency and adds support for multimodal training ([blog](https://developer.nvidia.com/blog/train-generative-ai-models-more-efficiently-with-new-nvidia-Megatron-Core-functionalities/)).
- **[2024/06]** Megatron Core added supports for Mamba-based models. Check out our paper [An Empirical Study of Mamba-based Language Models](https://arxiv.org/pdf/2406.07887) and [code example](https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba).
- **[2024/01 Announcement]** NVIDIA has released the core capabilities in **Megatron-LM** into [**Megatron Core**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) in this repository. Megatron Core expands upon Megatron-LM's GPU-optimized techniques with more cutting-edge innovations on system-level optimizations, featuring composable and modular APIs.

## Contributing

If you'd like to contribute a guide or tutorial, please follow this structure:

1. Create a new directory: `docs/discussions/your-guide-name/`
2. Add your main guide: `docs/discussions/your-guide-name/your-guide-name.md`
3. Create an images directory: `docs/discussions/your-guide-name/images/`
4. Update this README.md with a link to your guide

Each guide should be self-contained with its own images and supporting files.