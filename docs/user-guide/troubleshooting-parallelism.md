<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Troubleshooting Parallelism Configuration Errors

Megatron Core validates the relationships between parallelism dimensions—tensor
(TP), pipeline (PP), context (CP), expert (EP), and data (DP)—during
initialisation. The errors below are raised by configuration-construction code
and can be reproduced without a GPU.

---

## 1. World-size not divisible by the model-parallel product

**Error text**

```
RuntimeError: world_size (<W>) is not divisible by <TP × PP × CP>
```

**Violated invariant**

Every GPU must belong to exactly one combination of TP/PP/CP groups. The
product `TP × PP × CP` must therefore divide the total number of processes
evenly; the remainder becomes the data-parallel size.

**Invalid configuration**

```python
# 6 GPUs, TP=2, PP=2, CP=1  →  2×2×1 = 4, 6 % 4 ≠ 0
torchrun --nproc_per_node=6 pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2
```

**Corrected configuration**

```python
# 8 GPUs, TP=2, PP=2, CP=1  →  2×2×1 = 4, 8 % 4 = 0  (DP=2)
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2
```

**Rule of thumb**

```
Total GPUs  =  TP × PP × CP × DP
```

Choose any combination of TP, PP, CP whose product divides your total GPU
count; the quotient becomes DP automatically.

---

## 2. Virtual pipeline requires pipeline-parallel size > 1

**Error text**

```
RuntimeError: pipeline-model-parallel size should be greater than 1 with interleaved schedule
```

**Violated invariant**

The interleaved (virtual pipeline) schedule splits each pipeline stage into
multiple *virtual* stages to reduce the pipeline bubble. With only one physical
stage the schedule degenerates and the virtual-stage count is meaningless.

**Invalid configuration**

```python
from megatron.core import parallel_state

parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,          # only one stage
    virtual_pipeline_model_parallel_size=2,  # virtual stages require PP>1
)
```

**Corrected configuration**

```python
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=4,          # real stages now > 1
    virtual_pipeline_model_parallel_size=2,
)
```

**Tip**

`virtual_pipeline_model_parallel_size` is the number of model chunks *per
physical pipeline stage*. Total virtual stages = PP × virtual_pp_size. Leave
it `None` when PP = 1.

---

## 3. Sequence parallelism requires tensor parallelism > 1

**Error text**

```
ValueError: Cannot use sequence parallelism without tensor parallelism
```

**Violated invariant**

Sequence parallelism shards the sequence dimension across the TP group for
layer-norm and dropout layers. With TP = 1 there is only one rank in the group,
so there is nothing to shard.

**Invalid configuration**

```python
from megatron.core.model_parallel_config import ModelParallelConfig

config = ModelParallelConfig(
    tensor_model_parallel_size=1,  # no tensor parallelism
    sequence_parallel=True,        # but sequence_parallel is on
)
```

**Corrected configuration**

```python
config = ModelParallelConfig(
    tensor_model_parallel_size=4,  # TP >= 2
    sequence_parallel=True,
)
```

Or disable sequence parallelism:

```python
config = ModelParallelConfig(
    tensor_model_parallel_size=1,
    sequence_parallel=False,       # consistent with TP=1
)
```

---

## 4. Pipeline parallelism requires `pipeline_dtype` to be set

**Error text**

```
ValueError: When using pipeline parallelism, pipeline_dtype must be specified
```

**Violated invariant**

When tensors are passed between pipeline stages over the network, all ranks
must agree on their dtype. `pipeline_dtype` makes that contract explicit; it is
required whenever `pipeline_model_parallel_size > 1`.

**Invalid configuration**

```python
config = ModelParallelConfig(
    pipeline_model_parallel_size=4,
    # pipeline_dtype not set — defaults to None
)
```

**Corrected configuration**

```python
import torch
config = ModelParallelConfig(
    pipeline_model_parallel_size=4,
    pipeline_dtype=torch.bfloat16,   # must match training precision
)
```

For BF16 training use `torch.bfloat16`; for FP16 use `torch.float16`.

---

## 5. Attention-head count must be divisible by tensor-parallel size

**Error text**

```
ValueError: num_attention_heads (<H>) must be a multiple of tensor_model_parallel_size (<TP>).
```

**Violated invariant**

TP splits the attention heads across devices. Each TP rank gets
`num_attention_heads / TP` heads. If the division is not exact, heads cannot
be evenly assigned.

**Invalid configuration**

```python
from megatron.core.transformer.transformer_config import TransformerConfig

config = TransformerConfig(
    num_attention_heads=12,           # 12 heads
    tensor_model_parallel_size=8,    # 12 % 8 ≠ 0
    num_layers=24,
    hidden_size=1024,
)
```

**Corrected configuration**

```python
config = TransformerConfig(
    num_attention_heads=16,           # 16 % 8 = 0 ✓
    tensor_model_parallel_size=8,
    num_layers=24,
    hidden_size=1024,
)
```

Or reduce TP to a value that divides the current head count:

```python
config = TransformerConfig(
    num_attention_heads=12,
    tensor_model_parallel_size=4,    # 12 % 4 = 0 ✓
    num_layers=24,
    hidden_size=1024,
)
```

---

## 6. Expert parallelism requires `num_moe_experts` to be set

**Error text**

```
ValueError: num_moe_experts must be non None to use expert-parallel.
```

**Violated invariant**

Expert parallelism (EP) distributes experts across EP ranks. The system needs
to know how many experts the model has to build the EP process groups and
validate divisibility.

**Invalid configuration**

```python
config = TransformerConfig(
    expert_model_parallel_size=4,  # EP=4 but no experts declared
    # num_moe_experts not set
    num_layers=24,
    hidden_size=1024,
    num_attention_heads=16,
)
```

**Corrected configuration**

```python
config = TransformerConfig(
    expert_model_parallel_size=4,
    num_moe_experts=8,             # at least as many experts as EP ranks
    num_layers=24,
    hidden_size=1024,
    num_attention_heads=16,
)
```

**Note**: `num_moe_experts` should generally be a multiple of
`expert_model_parallel_size` to avoid load imbalance.

---

## 7. FP16 and BF16 cannot both be enabled

**Error text**

```
ValueError: Only one of self.fp16: True and self.bf16 True should be True.
```

**Violated invariant**

FP16 and BF16 are mutually exclusive mixed-precision modes; enabling both is
contradictory and would leave the training dtype undefined.

**Invalid configuration**

```python
config = TransformerConfig(
    fp16=True,
    bf16=True,   # cannot combine
    num_layers=24,
    hidden_size=1024,
    num_attention_heads=16,
)
```

**Corrected configuration**

Choose one:

```python
# BF16 (recommended for Ampere+ GPUs)
config = TransformerConfig(
    fp16=False,
    bf16=True,
    num_layers=24,
    hidden_size=1024,
    num_attention_heads=16,
)

# FP16
config = TransformerConfig(
    fp16=True,
    bf16=False,
    num_layers=24,
    hidden_size=1024,
    num_attention_heads=16,
)
```

---

## 8. Expert world-size not divisible by expert model-parallel product

**Error text**

```
RuntimeError: world_size (<W>) is not divisible by expert_tensor_model_pipeline_parallel size (<ETP × EP × PP>)
```

**Violated invariant**

Expert parallelism uses its own process-group layout: `expert_tensor_parallel_size × expert_model_parallel_size × pipeline_model_parallel_size` must divide the total world size exactly.

**Invalid configuration**

```python
# 8 GPUs: ETP=2, EP=3, PP=2  →  2×3×2=12, 8 % 12 ≠ 0
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    expert_model_parallel_size=3,
)
```

**Corrected configuration**

```python
# 8 GPUs: ETP=2, EP=2, PP=2  →  2×2×2=8, 8 % 8 = 0  (expert_dp=1)
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    expert_model_parallel_size=2,
)
```

---

## Quick reference

| Error excerpt | Likely cause | Key constraint |
|---|---|---|
| `world_size … is not divisible by` (model size) | Bad TP/PP/CP combo | `world_size % (TP×PP×CP) == 0` |
| `pipeline-model-parallel size should be greater than 1` | VP with PP=1 | PP > 1 when virtual_pp_size is set |
| `Cannot use sequence parallelism without tensor parallelism` | SP with TP=1 | TP > 1 when sequence_parallel=True |
| `pipeline_dtype must be specified` | PP > 1 but no dtype | Set pipeline_dtype when PP > 1 |
| `num_attention_heads … must be a multiple of tensor_model_parallel_size` | Heads not divisible by TP | `num_attention_heads % TP == 0` |
| `num_moe_experts must be non None to use expert-parallel` | EP without MoE config | Set num_moe_experts when EP > 1 |
| `Only one of … fp16 … and … bf16 … should be True` | Both precisions on | Enable exactly one of fp16 / bf16 |
| `world_size … is not divisible by expert_tensor_model_pipeline_parallel size` | Bad ETP/EP/PP combo | `world_size % (ETP×EP×PP) == 0` |
