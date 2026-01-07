# Megatron Core User Guide

**Megatron Core** is a GPU-optimized library for training large language models at scale. It provides modular, composable building blocks for creating custom training frameworks with state-of-the-art parallelism strategies and performance optimizations.

Megatron Core offers a flexible, reusable foundation for building large-scale transformer training systems. **Megatron-LM** serves as a reference implementation demonstrating how to use Megatron Core components to train models with billions to trillions of parameters across distributed GPU clusters.

## Key Features

* Composable transformer building blocks (attention, MLP, etc.)
* Advanced parallelism strategies (TP, PP, DP, EP, CP)
* Pipeline schedules and distributed optimizers
* Mixed precision support (FP16, BF16, FP8)
* GPU-optimized kernels and memory management
* High-performance dataloaders and dataset utilities
* Model architectures (LLaMA, Qwen, DeepSeek, GPT, Mamba, etc.)
