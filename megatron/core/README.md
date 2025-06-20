# Megatron-Core

Megatron-Core is an open-source PyTorch-based library that contains GPU-optimized techniques and cutting-edge system-level optimizations. It abstracts them into composable and modular APIs, allowing full flexibility for developers and model researchers to train custom transformers at-scale on NVIDIA accelerated computing infrastructure. This library is compatible with all NVIDIA Tensor Core GPUs, including FP8 acceleration support for [NVIDIA Hopper architectures](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/).

Megatron-Core offers core building blocks such as attention mechanisms, transformer blocks and layers, normalization layers, and embedding techniques. Additional functionality like activation re-computation, distributed checkpointing is also natively built-in to the library. The building blocks and functionality are all GPU optimized, and can be built with advanced parallelization strategies for optimal training speed and stability on NVIDIA Accelerated Computing Infrastructure. Another key component of the Megatron-Core library includes advanced model parallelism techniques (tensor, sequence, pipeline, context, and MoE expert parallelism).

Megatron-Core can be used with [NVIDIA NeMo](https://www.nvidia.com/en-us/ai-data-science/products/nemo/), an enterprise-grade AI platform. Alternatively, you can explore Megatron-Core with the native PyTorch training loop [here](https://github.com/NVIDIA/Megatron-LM/tree/main/examples). Visit [Megatron-Core documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html) to learn more.

## Quick links

- [Benchmark using NVIDIA NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html#performance-benchmarks)
- [Multimodal example (LLaVA training pipeline)](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/multimodal)
- [Mixture-of-Experts](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe)
- [Training Mamba-based Language Models](https://github.com/NVIDIA/Megatron-LM/tree/main/examples/mamba)
