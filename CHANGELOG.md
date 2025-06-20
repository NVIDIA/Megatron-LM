# Changelog

## NVIDIA Megatron Core 0.13.0

- Hybrid Model
  - Add context parallel support for models with Mamba layers

## NVIDIA Megatron Core 0.12.0

- Add FP8 recipe selection to arguments (--fp8-recipe, --first-last-layers-bf16, --num-layers-at-start-in-bf16, --num-layers-at-end-in-bf16)
- Context parallel: fix loss scaling when calculate_per_token_loss=True
- Make the number of data parallel communication buckets configurable (--ddp-num-buckets, --ddp-pad-buckets-for-high-nccl-busbw)
- Inference
  - Support in-flight batching and chunked KV cache
  - Reduce memory usage,
    - by not materializing full attention mask
    - by only materializing logits for the last token during decode
    - by removing an obsolete tensor reference
- Hybrid Model
  - Inference
    - Add CUDA graph support
    - Change tools/run_mamba_text_generation_server.py to use megatron.core.inference
    - Fix a shape issue when materializing logits for Mamba model
  - Improve initialization of Mamba layers
  - Add configuration switches (--mamba-state-dim, --mamba-head-dim, --mamba-num-groups, --is-hybrid-model)
  - Make num_floating_point_operations work with hybrid model
  - Make hybrid_conversion.py work with mixer that uses TE linear
  - Add FP8 support
  - Fix Mamba dt_bias tensor parallelism
  - Support multimodal tokenizer
  - Improve data parallelism scaling
- MoE
  - Features:
    - DeepEP support, compatible with all the parallelisms and token drop / dropless
    - Important precision improvement: Enable FP32/FP64 routing and unpermutation using –moe-router-dtype. FP32 is recommended for all fine-grained MoE training
    - CUDA Graph support for MoE
    - Multi-Token Prediction (MTP) Support
    - Fused indices_to_multihot kernel for DeepEP dispatcher
  - Bug fixes:
    - Fix Hang Issue with MoE+Dense Hybrid models
    - Update theoretical memory and tflops estimation for MoE and MLA
    - Fix MoE Aux loss scaling for per token loss
    - Fixes for group limited routing and expert bias. We verified these fixes through dsv3 e2e verifications
  - Known issues:
    - The ckpt trained with Custom FSDP for MoE may not be compatible with 3D parallel training.

## NVIDIA Megatron Core 0.11.0

- Add multi datacenter training support though N/S connection
- MoE
  - Features
    - Support DeepSeek-V3 fine-tuning
      - Aux-loss-free load balancing strategy
      - Node-limited routing and Device-limited routing support.
      - Tensor Parallelism support for MLA and Sequence Auxiliary Loss
      - MTP (with TP and PP support) is coming soon.
    - Permutation / Unpermutation fusion kernel from TransformerEngine.
    - Uneven virtual pipeline parallel split support in first and last PP stage.
  - Bug fixes:
    - Fix the grad scale when TP != expert-TP and average_in_collective is enabled in DDP.
    - Fix TEGroupedMLP distckpt compatibility issue with FP8 padding/unpadding.
  - Known Issues:
    - When training the Dense+MoE hybrid model, the process will hang if any PP rank does not have expert params.
- Add MX-FP16 support for optimizer and master weights
- CUDA Graph memory optimizations
- Enable UCC backend for PP communication
- Optimizer CPU offload support for memory savings
- Models
  - Initial RADIO/CRADIO implementation
  - llama3.2 support
- Hybrid Model
  - Support quantization via TensorRT Model Optimizer

## NVIDIA Megatron Core 0.10.0

- Adding MLA to MCore
- Enable FP8 for GroupedMLP
- MoE Parallel Folding
- Enhance MoE Architecture: Support MoE Layer Frequency Patterns and Configurable MoE FFN Hidden Size
- Multimodal: NVLM training and evaluation support in MCore
- Mamba Hybrid
  - Increase performance and reduce memory footprint of Triton language/compiler distributed caching
  - Add more unit testing and fix bugs

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
- NVRx / Fault tolerance
  - fault and hang detection in addition to existing straggler detection
  - graceful exit and auto restart

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
