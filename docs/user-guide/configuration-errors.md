<!---
   Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Configuration errors

Megatron Core validates parallelism settings while constructing a configuration.
This page maps common validation errors to the setting that violates the
corresponding invariant. The examples are deliberately small: replace the
values with ones appropriate for your model and launcher.

## Sequence parallelism requires tensor parallelism

Sequence parallelism shards sequence-dimension activations across tensor-parallel
ranks. It therefore needs more than one tensor-parallel rank.

```python
from megatron.core.transformer.transformer_config import TransformerConfig

TransformerConfig(
    num_layers=1,
    hidden_size=128,
    num_attention_heads=4,
    sequence_parallel=True,
)
```

This configuration raises:

```
Cannot use sequence parallelism without tensor parallelism
```

Either disable sequence parallelism for a single-rank setup, or use a
tensor-parallel size greater than one:

```python
TransformerConfig(
    num_layers=1,
    hidden_size=128,
    num_attention_heads=4,
    tensor_model_parallel_size=2,
    sequence_parallel=True,
)
```

## Attention heads must divide across tensor-parallel ranks

Each tensor-parallel rank owns an equal subset of attention heads. Set the
tensor-parallel size to a divisor of `num_attention_heads`.

```python
TransformerConfig(
    num_layers=1,
    hidden_size=128,
    num_attention_heads=4,
    tensor_model_parallel_size=3,
)
```

This configuration raises an error containing:

```
num_attention_heads (4) must be a multiple of tensor_model_parallel_size (3)
```

For example, use two tensor-parallel ranks:

```python
TransformerConfig(
    num_layers=1,
    hidden_size=128,
    num_attention_heads=4,
    tensor_model_parallel_size=2,
)
```

## Pipeline parallelism requires a communication dtype

Pipeline stages exchange activations. When `pipeline_model_parallel_size` is
greater than one, configure `pipeline_dtype` for those transfers.

```python
TransformerConfig(
    num_layers=1,
    hidden_size=128,
    num_attention_heads=4,
    pipeline_model_parallel_size=2,
)
```

This configuration raises:

```
When using pipeline parallelism, pipeline_dtype must be specified
```

Set a dtype matching the precision of the activations exchanged between stages:

```python
import torch

TransformerConfig(
    num_layers=1,
    hidden_size=128,
    num_attention_heads=4,
    pipeline_model_parallel_size=2,
    pipeline_dtype=torch.bfloat16,
)
```

## Tensor plus expert parallelism needs sequence parallelism

For training configurations that combine tensor parallelism and expert
parallelism, enable sequence parallelism to avoid an unsupported activation
layout.

```python
TransformerConfig(
    num_layers=1,
    hidden_size=128,
    num_attention_heads=4,
    tensor_model_parallel_size=2,
    expert_model_parallel_size=2,
    num_moe_experts=2,
)
```

This configuration emits a warning containing:

```
When using expert parallelism and tensor parallelism for training, sequence parallelism must be used
```

Enable sequence parallelism together with tensor parallelism:

```python
TransformerConfig(
    num_layers=1,
    hidden_size=128,
    num_attention_heads=4,
    tensor_model_parallel_size=2,
    expert_model_parallel_size=2,
    num_moe_experts=2,
    sequence_parallel=True,
)
```

## Context-parallel layout compatibility

Context parallelism currently does not support the `contiguous` attention
layout. Use the default `zigzag` layout when `context_parallel_size` is greater
than one.

```python
TransformerConfig(
    num_layers=1,
    hidden_size=128,
    num_attention_heads=4,
    context_parallel_size=2,
    attention_cp_layout="contiguous",
)
```

This configuration raises:

```
attention_cp_layout='contiguous' is not yet supported with context parallelism.
```

Remove the layout override or select `zigzag`:

```python
TransformerConfig(
    num_layers=1,
    hidden_size=128,
    num_attention_heads=4,
    context_parallel_size=2,
    attention_cp_layout="zigzag",
)
```
