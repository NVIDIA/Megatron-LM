# Multimodal Models

Megatron Core supports multimodal models that combine language with vision, audio, and other modalities for comprehensive multimodal understanding.

## MIMO: Multimodal In/Out Framework

**MIMO (Multimodal In/Out Model)** is an experimental framework in Megatron Core that supports arbitrary combinations of modalities including vision, audio, and text. MIMO provides a flexible architecture for building custom multimodal models.

> **Note**: MIMO is experimental and under active development. The API may change in future releases.

**Key Features:**
- Arbitrary modality combinations (vision, audio, text, etc.)
- Flexible encoder architecture for different input modalities
- Unified embedding space across modalities
- Support for both vision-language and audio-vision-language models

See [examples/mimo](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/mimo) for training scripts and examples.

## Vision-Language Models

| Model | Description | Vision Encoder | Language Model |
|-------|-------------|----------------|----------------|
| **LLaVA** | Visual instruction tuning | CLIP ViT-L/14 | Mistral-7B / LLaMA |
| **NVLM** | NVIDIA Vision-Language Model | CLIP / Custom ViT | LLaMA-based |
| **LLaMA 3.1 Nemotron Nano VL** | Efficient multimodal model | Vision Transformer | LLaMA 3.1 8B |

## Vision Encoders

| Model | Description | Key Features |
|-------|-------------|--------------|
| **CLIP ViT** | OpenAI's CLIP Vision Transformer | Image-text alignment, multiple scales (L/14@336px) |
| **RADIO** | Resolution-Agnostic Dynamic Image Optimization | Flexible resolution handling, efficient vision encoding |

## Diffusion Models

For multimodal diffusion models (image generation, text-to-image, etc.), see [NeMo Diffusion Models](https://github.com/NVIDIA-NeMo/NeMo/tree/main/nemo/collections/diffusion). NeMo provides production-ready implementations of:
- Stable Diffusion variants
- Text-to-image generation
- Image-to-image translation
- ControlNet and other conditioning mechanisms

## Multimodal Features

- **Image-Text Alignment**: Pre-training on image-caption pairs
- **Visual Instruction Tuning**: Fine-tuning on instruction-following datasets
- **Flexible Vision Encoders**: Support for different ViT architectures and resolutions
- **Combined Checkpointing**: Unified checkpoints combining vision and language models
- **Efficient Training**: Full parallelism support (TP, PP, DP) for both vision and language components

## Example Scripts

Multimodal training examples can be found in the following directories:

**MIMO Framework:**
- `examples/mimo/` - Multimodal In/Out training with support for vision-language and audio-vision-language models

**Specific Multimodal Models:**
- `examples/multimodal/` - LLaVA-style training with Mistral + CLIP
- `examples/multimodal/nvlm/` - NVLM training scripts
- `examples/multimodal/llama_3p1_nemotron_nano_vl_8b_v1/` - Nemotron VL training
- `examples/multimodal/radio/` - RADIO vision encoder integration
