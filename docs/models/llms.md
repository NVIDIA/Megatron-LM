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
