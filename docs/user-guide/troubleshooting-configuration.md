<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Configuration Troubleshooting

`TransformerConfig` validates the interactions between tensor, pipeline, context, expert, and
sequence parallelism when it is constructed. Because the checks run at construction time, an
invalid configuration fails before any model is built or any GPU is touched, and the resulting
`ValueError` / `AssertionError` is the only diagnostic you get.

This page covers the validation failures that come up most often: what the error means, which
invariant it enforces, and how to fix the configuration. Every example below is exercised in
`tests/unit_tests/transformer/test_config_troubleshooting.py`, which asserts that the invalid
configuration raises and that the corrected one constructs cleanly.

| Symptom | Section |
|---------|---------|
| `num_attention_heads (N) must be a multiple of tensor_model_parallel_size (M)` | [Tensor parallel size must divide the attention heads](#tensor-parallel-size-must-divide-the-attention-heads) |
| `num_query_groups (N) must be a divisor of num_attention_heads (M)` | [Query groups must divide the attention heads](#query-groups-must-divide-the-attention-heads) |
| `Cannot use sequence parallelism without tensor parallelism` | [Sequence parallelism requires tensor parallelism](#sequence-parallelism-requires-tensor-parallelism) |
| `When using pipeline parallelism, pipeline_dtype must be specified` | [Pipeline parallelism requires a pipeline_dtype](#pipeline-parallelism-requires-a-pipeline-dtype) |
| `attn_logit_softcapping must be a positive finite value` | [Softcapping must be a positive finite value](#softcapping-must-be-a-positive-finite-value) |
| `pipeline_model_parallel_layout cannot be set with other pipeline layout arguments` | [The pipeline layout is exclusive](#the-pipeline-layout-is-exclusive) |
| `virtual_pipeline_model_parallel_size conflicts with pipeline_model_parallel_layout` | [Virtual pipeline size must match the layout](#virtual-pipeline-size-must-match-the-layout) |

## Tensor parallel size must divide the attention heads

**Invalid configuration**

```python
TransformerConfig(num_layers=4, hidden_size=512, num_attention_heads=8, tensor_model_parallel_size=3)
```

**Error**

```text
ValueError: num_attention_heads (8) must be a multiple of tensor_model_parallel_size (3).
```

**Invariant**

Tensor parallelism splits the attention heads evenly across the tensor-parallel ranks. Each rank
owns `num_attention_heads / tensor_model_parallel_size` heads, so that quotient has to be a whole
number. `hidden_size` is not checked separately: it is derived from the head configuration, so a
head count that divides cleanly also gives each rank a uniform `hidden_size` slice.

**Corrected configuration**

Pick a tensor-parallel size that divides 8, for example 2 or 4:

```python
TransformerConfig(num_layers=4, hidden_size=512, num_attention_heads=8, tensor_model_parallel_size=4)
```

## Query groups must divide the attention heads

**Invalid configuration**

```python
TransformerConfig(num_layers=4, hidden_size=512, num_attention_heads=8, num_query_groups=3)
```

**Error**

```text
ValueError: num_query_groups (3) must be a divisor of num_attention_heads (8).
```

**Invariant**

`num_query_groups` controls grouped-query attention: several query heads share one key/value head.
The sharing is only well defined when every key/value group covers the same number of query heads,
so `num_attention_heads % num_query_groups == 0`. A second check then ties the group count to
`tensor_model_parallel_size`: the groups must be either a multiple or a divisor of it, so that
groups are not split across ranks.

**Corrected configuration**

Use a divisor of the head count, here 2:

```python
TransformerConfig(num_layers=4, hidden_size=512, num_attention_heads=8, num_query_groups=2)
```

## Sequence parallelism requires tensor parallelism

**Invalid configuration**

```python
TransformerConfig(num_layers=4, hidden_size=512, num_attention_heads=8, sequence_parallel=True)
```

**Error**

```text
ValueError: Cannot use sequence parallelism without tensor parallelism
```

**Invariant**

Sequence parallelism splits the sequence dimension of the layer norms and dropout across
tensor-parallel ranks. With `tensor_model_parallel_size=1` there are no ranks to split across, so
the flag has no meaning and the layer would silently fall back to a different activation layout
than the one requested.

**Corrected configuration**

Either enable tensor parallelism, or drop the flag:

```python
TransformerConfig(
    num_layers=4, hidden_size=512, num_attention_heads=8,
    tensor_model_parallel_size=2, sequence_parallel=True,
)
```

## Pipeline parallelism requires a pipeline_dtype

**Invalid configuration**

```python
TransformerConfig(num_layers=4, hidden_size=512, num_attention_heads=8, pipeline_model_parallel_size=2)
```

**Error**

```text
ValueError: When using pipeline parallelism, pipeline_dtype must be specified
```

**Invariant**

Pipeline stages exchange activations, and the dtype of that traffic has to be known when the
config is built: it decides the size of the communication buffers and whether the inter-stage
send/receive matches the residual stream. Megatron Core does not infer it, because the right
answer depends on whether you train in BF16, FP16, or keep an FP32 residual connection.

**Corrected configuration**

```python
import torch
TransformerConfig(
    num_layers=4, hidden_size=512, num_attention_heads=8,
    pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16,
)
```

Note that setting `fp32_residual_connection=True` alongside a non-FP32 `pipeline_dtype` is not an
error: the config logs a warning and overrides `pipeline_dtype` to `torch.float` to match the
residual stream.

## Softcapping must be a positive finite value

**Invalid configuration**

```python
TransformerConfig(num_layers=4, hidden_size=512, num_attention_heads=8, attn_logit_softcapping=0.0)
```

**Error**

```text
ValueError: attn_logit_softcapping must be a positive finite value, got 0.0. Use None to disable
softcapping. A cap of 0.0 disables softcapping in TransformerEngine but collapses every logit to
zero in the local attention path, a negative cap is silently applied as its absolute value there
while FlashAttention ignores it entirely, and a non-finite cap produces NaN logits.
```

**Invariant**

The same value is consumed by three attention implementations that disagree about the edge cases,
so a non-positive or non-finite cap would give different numerics depending on which backend is
selected. Rejecting it up front keeps the backends interchangeable.

**Corrected configuration**

Use `None` to disable softcapping, or a positive finite value:

```python
TransformerConfig(num_layers=4, hidden_size=512, num_attention_heads=8, attn_logit_softcapping=None)
```

## The pipeline layout is exclusive

**Invalid configuration**

```python
TransformerConfig(
    num_layers=4, hidden_size=512, num_attention_heads=8,
    pipeline_model_parallel_size=2, pipeline_dtype=torch.float,
    pipeline_model_parallel_layout="E|tt|tt|L",
    num_layers_in_first_pipeline_stage=2,
)
```

**Error**

```text
ValueError: pipeline_model_parallel_layout cannot be set with other pipeline layout arguments.
self.num_layers_in_first_pipeline_stage=2, self.num_layers_in_last_pipeline_stage=None,
self.account_for_embedding_in_pipeline_split=False, self.account_for_loss_in_pipeline_split=False.
```

**Invariant**

`pipeline_model_parallel_layout` describes the whole pipeline, including where the embedding and
the loss live. The uneven-split arguments and the `account_for_*` flags describe the same thing in
a coarser way. Accepting both would leave it ambiguous which description wins.

**Corrected configuration**

Keep the layout string and drop the competing arguments:

```python
TransformerConfig(
    num_layers=4, hidden_size=512, num_attention_heads=8,
    pipeline_model_parallel_size=2, pipeline_dtype=torch.float,
    pipeline_model_parallel_layout="E|tt|tt|L",
)
```

## Virtual pipeline size must match the layout

**Invalid configuration**

```python
TransformerConfig(
    num_layers=4, hidden_size=512, num_attention_heads=8,
    pipeline_model_parallel_size=2, pipeline_dtype=torch.float,
    pipeline_model_parallel_layout="E|tt|tt|L",
    virtual_pipeline_model_parallel_size=5,
)
```

**Error**

```text
AssertionError: virtual_pipeline_model_parallel_size conflicts with pipeline_model_parallel_layout,
(self.virtual_pipeline_model_parallel_size=5,  detected_vpp_size=2)
```

**Invariant**

The layout string encodes the number of virtual stages per rank by repeating the stage group, so
it already determines the virtual pipeline size. Passing a different value as well would make the
schedule inconsistent with the layer placement.

**Corrected configuration**

Omit `virtual_pipeline_model_parallel_size` and let it be read from the layout:

```python
TransformerConfig(
    num_layers=4, hidden_size=512, num_attention_heads=8,
    pipeline_model_parallel_size=2, pipeline_dtype=torch.float,
    pipeline_model_parallel_layout="E|tt|tt|L",
)
```

## Diagnosing a failure that is not listed here

Every check lives in `TransformerConfig.__post_init__` in
`megatron/core/transformer/transformer_config.py`. The error text names the offending attribute,
so searching that file for the message string locates the exact invariant and the values it
compares. Constructing the config directly is enough to reproduce a validation failure; no model
or GPU is involved:

```python
from megatron.core.transformer.transformer_config import TransformerConfig
TransformerConfig(num_layers=4, hidden_size=512, num_attention_heads=8, tensor_model_parallel_size=3)
```
