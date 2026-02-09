# Megatron In-Framework Inference

In-framework inference system for Megatron Core enabling training-inference consistency.

```{warning}
**Experimental Feature**: Megatron in-framework inference is under active development and should be considered experimental. Performance is **not expected to match** dedicated inference engines like vLLM or SGLang.
```

## Overview

[**Megatron in-framework inference**](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/inference) provides inference capabilities directly within Megatron Core, designed primarily to address numerical differences that arise when switching between separate training and inference backends.

This feature is intended for research teams exploring RL post-training and other workflows where numerical alignment between training and inference is critical. For production deployments with high throughput requirements, we recommend dedicated inference engines like [**vLLM**](https://github.com/vllm-project/vllm), [**SGLang**](https://github.com/sgl-project/sglang), or [**TensorRT-LLM**](https://github.com/NVIDIA/TensorRT-LLM).

## Key Features

- **Training-Inference Consistency** - Eliminates numerical differences by using the same kernels and parallelization for both training and inference
- **FP8 Consistency** - Maintains precision alignment for FP8 workflows across training and inference
- **Native Integration** - No framework handoffs or external dependencies required
- **RL Workflow Support** - Unified pipeline for training and rollout generation, usable with [NeMo RL](https://github.com/NVIDIA/NeMo-RL) and custom RL frameworks
- **Deterministic Generation** - Reproducible outputs for research and debugging

## Motivation

### The Numerical Consistency Problem

When using separate training and inference backends (e.g., Megatron for training + vLLM/SGLang for inference), numerical differences arise from:

- **Kernel differences** - Inference engines use optimized kernels that produce slightly different outputs than training kernels
- **Parallelization mismatches** - Different tensor/pipeline parallel implementations across frameworks
- **Precision handling** - FP8/BF16 quantization and computation differs between systems

### Impact on Post-Training Workflows

These differences cause significant problems in post-training:

- **Growing logprob errors** - Token probability errors increase throughout training, eventually causing crashes
- **Teacher-student inconsistencies** - Precision gaps in distillation when teacher (inference) and student (training) use different backends
- **Policy drift** - On-policy RL algorithms become effectively off-policy due to numerical mismatches
- **Non-reproducibility** - Run-to-run variation makes debugging and research difficult

## Use Cases

### Reinforcement Learning Post-Training

The primary use case is RL post-training where training-inference consistency is critical:

- **True on-policy RL** - Rollout generation with exact numerical alignment to training, eliminating the implicit off-policy shift caused by backend mismatches
- **GRPO/PPO workflows** - Prevent logprob errors from growing during training by ensuring actor inference matches training computations

### Knowledge Distillation

Consistent teacher-student computations for distillation workflows:

- **Teacher logit consistency** - Ensure teacher logits computed during inference match what the student sees during training
- **FP8 distillation** - Maintain precision consistency when both teacher and student use FP8

### Research and Debugging

- **Reproducibility studies** - Deterministic generation for controlled experiments
- **Numerical analysis** - Isolate and debug training vs inference differences
- **Algorithm development** - Prototype new RL methods without framework coupling

## Architecture

### Components

**Inference Engines**
- `DynamicInferenceEngine` - Primary engine with dynamic batching support
- `StaticInferenceEngine` - Legacy engine (deprecated, wraps DynamicInferenceEngine)

**Text Generation Controllers**
- Handle prompt preprocessing and output detokenization
- Support for GPT, T5, and VLM architectures

**Inference Contexts**
- Manage KV cache and inference state
- Support both static and dynamic memory allocation

## Limitations and Performance Expectations

Megatron in-framework inference prioritizes numerical consistency over throughput. **Performance is not expected to match dedicated inference engines**:

- **Lower throughput** than vLLM, SGLang, or TensorRT-LLM
- **No continuous batching optimizations** found in production inference engines
- **Limited kernel optimization** - uses training kernels rather than inference-optimized kernels
- **Research-focused** - designed for correctness, not production serving

This is an intentional design tradeoff: Megatron in-framework inference uses the same code paths as training to guarantee numerical consistency, at the cost of inference performance.

## Resources

- **[Megatron Core Inference](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/inference)** - Source code
- **[Megatron RL](megatron_rl.md)** - RL library using Megatron in-framework inference
- **[API Reference](../../apidocs/core/core.inference.md)** - Detailed API documentation
