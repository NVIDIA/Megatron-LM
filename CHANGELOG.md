# Changelog

## NVIDIA Megatron Core 0.9.0

- Uneven pipeline parallelism
  - Enable pipeline parallelism where first and last ranks have fewer transformer layers than the intermediate ranks
- Per layer CUDAGraph support for GPT training with Transformer Engine modules
- Enable different TP sizes for the vision encoder
- Enable pipeline parallelism for T5 & Llava models
- Support multi-tile multi-image input in Llava models
- MoE
  - FP8 support
  - Runtime upcycling support
  - Dispatcher implementation optimizations
  - Shared expert support with overlapping optimizations
    - Qwen Model support
- Known Issues
  - When using sequence parallel, during the transformer block forward pass, dropout is not using the appropriate rng context.


## NVIDIA Megatron Core 0.8.0

- Multimodal
  - Added initial support for training vision language models using the LLaVA architecture
  - Added initial support for inference with multimodal inputs
  - End-to-end multimodal example from data collection to training to evaluation is provided in examples/multimodal
- MoE
  - Context Parallel support.
  - Distributed checkpoint support for grouped GEMM.
- Mamba

## NVIDIA Megatron Core 0.7.0

- MoE
  - Token drop support
  - Several efficiency optimizations
  - Improved model parallelism
  - Memory optimizations
- Distributed checkpointing
  - Enabled for Retro
  - Asynchronous checkpoint saving
- Several minor bug fixes, speed improvements, and memory optimizations

## NVIDIA Megatron Core 0.6.0

- MoE (Mixture of Experts)
  - Performance optimization
    - Communication optimization for multi GPU and Single GPU 
    - 23% improvement (323 TFLOPS/GPU) over MCore 0.5.0 on Mixtral with Hopper BF16
    - GroupedMLP enhancement for Hopper
    - DP Overlapping. Support overlapping computation with gradient reduction and parameter gathering.
  - All-to-All based Token Dispatcher
  - Layer-wise logging for load balancing loss.
  - Improved expert parallel support including distributed optimizer.
- Distributed optimizer
- RETRO
  - Data processing
- BERT
  - Distributed checkpointing
- Dist checkpointing
  - PyTorch native distributed backend
  - Improved saving/loading speed
- TensorRT-LLM Export
  - Integration with TensorRT Model Optimizer Post-training quantization (PTQ)
  - Text generation driver to perform PTQ in Megatron-LM
  - Llama2 and Nemotron3-8b examples to use TensorRT-LLM unified build API to build engine after training.
- Several minor enhancements, bug fixes, and documentation updates

## NVIDIA Megatron Core 0.5.0

### Key Features and Enhancements

Megatron core documentation is now [live!](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html#quick-start)

### Model Features

- MoE (Mixture of Experts)
  - Support for Z-loss, Load balancing and Sinkhorn
  - Layer and communications refactor
  - Richer parallelism mappings and EP can be combined with other model parallel techniques for larger MoE variants, e.g. EP + TP + DP + SP + PP
  - Token dropless architecture with Top-K routing
  - Performance optimization with with GroupedGEMM when number of local experts is > 1
  - Distributed checkpointing
- Interleaved rotary embedding

### Datasets

- Masked WordPiece datasets for BERT and T5
- Raw and mock datasets

### Parallelism

### Performance

- Activation offloading to CPU
- Rope and Swiglu fusion
- Sliding window attention (via Transformer Engine)

### General Improvements

- Timers

## NVIDIA Megatron Core 0.4.0

### Key Features and Enhancements

#### Models

- BERT
- RETRO
- T5

#### Parallelism

- Mixture of Experts support for GPT
- Model parallel efficient Distributed Data Parallel (DDP)
- Context Parallel (2D Tensor Parallel) support

#### Datasets

- GPT Dataset
- Blended Dataset
