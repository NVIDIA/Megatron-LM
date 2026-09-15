<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# Configuration Errors

Megatron Core validates many interactions between tensor, pipeline, context,
expert, and sequence parallelism at startup. When a combination is invalid, the
process exits during argument validation with a message that names the violated
invariant. This page collects the most common validation failures, the exact
error text to search for, and the corrected configuration.

The error excerpts below are stable messages raised by the configuration
dataclasses (`megatron.core.model_parallel_config.ModelParallelConfig` and
`megatron.core.transformer.transformer_config.TransformerConfig`); they do not
depend on line numbers.

## num_attention_heads must be a multiple of tensor_model_parallel_size

**Minimal invalid configuration**

```bash
python pretrain_gpt.py \
  --num-layers 12 --hidden-size 1024 \
  --num-attention-heads 12 --tensor-model-parallel-size 8 \
  ...
```

**Error text**

```
num_attention_heads (12) must be a multiple of tensor_model_parallel_size (8).
```

**Violated invariant**

Attention heads are sharded across the tensor-parallel ranks, so
`num_attention_heads` must be divisible by `tensor_model_parallel_size`.

**Corrected configuration**

```bash
python pretrain_gpt.py \
  --num-layers 12 --hidden-size 1024 \
  --num-attention-heads 16 --tensor-model-parallel-size 8 \
  ...
```

## num_query_groups must be a multiple or divisor of tensor_model_parallel_size

**Minimal invalid configuration**

```bash
python pretrain_gpt.py \
  --num-layers 12 --hidden-size 1024 \
  --num-attention-heads 16 --num-query-groups 3 \
  --tensor-model-parallel-size 8 \
  ...
```

**Error text**

```
num_query_groups (3) must be a multiple or divisor of tensor_model_parallel_size (8).
```

**Violated invariant**

The query groups must partition cleanly across the tensor-parallel ranks (either
each rank owns whole groups, or the groups are replicated evenly).

**Corrected configuration**

```bash
python pretrain_gpt.py \
  --num-layers 12 --hidden-size 1024 \
  --num-attention-heads 16 --num-query-groups 4 \
  --tensor-model-parallel-size 8 \
  ...
```

## fp16 and bf16 cannot both be enabled

**Minimal invalid configuration**

```bash
python pretrain_gpt.py \
  --num-layers 12 --hidden-size 1024 \
  --fp16 --bf16 \
  ...
```

**Error text**

```
Only one of self.fp16: True and self.bf16 True should be True.
```

**Violated invariant**

`fp16` and `bf16` are mutually exclusive precision modes; exactly one of the two
can be enabled.

**Corrected configuration**

```bash
python pretrain_gpt.py \
  --num-layers 12 --hidden-size 1024 \
  --bf16 \
  ...
```

## Sequence parallelism requires tensor parallelism

**Minimal invalid configuration**

```bash
python pretrain_gpt.py \
  --num-layers 12 --hidden-size 1024 \
  --tensor-model-parallel-size 1 --sequence-parallel \
  ...
```

**Error text**

```
Cannot use sequence parallelism without tensor parallelism
```

**Violated invariant**

Sequence parallelism shards the sequence dimension *within* a tensor-parallel
group, so it is meaningless (and rejected) when `tensor_model_parallel_size` is
1.

**Corrected configuration**

```bash
python pretrain_gpt.py \
  --num-layers 12 --hidden-size 1024 \
  --tensor-model-parallel-size 2 --sequence-parallel \
  ...
```

## Pipeline parallelism requires pipeline_dtype

**Minimal invalid configuration**

```bash
python pretrain_gpt.py \
  --num-layers 24 --hidden-size 1024 \
  --pipeline-model-parallel-size 2 \
  ...
```

**Error text**

```
When using pipeline parallelism, pipeline_dtype must be specified
```

**Violated invariant**

With more than one pipeline stage the inter-stage communication buffers need a
dtype that is chosen explicitly rather than defaulted.

**Corrected configuration**

```bash
python pretrain_gpt.py \
  --num-layers 24 --hidden-size 1024 \
  --pipeline-model-parallel-size 2 --pipeline-dtype bf16 \
  ...
```

## tensor_parallel_num_weight_shards must be divisible by tensor_model_parallel_size

**Minimal invalid configuration**

```bash
python pretrain_gpt.py \
  --num-layers 12 --hidden-size 1024 \
  --tensor-model-parallel-size 4 --tensor-parallel-num-weight-shards 6 \
  ...
```

**Error text**

```
tensor_parallel_num_weight_shards (6) must be divisible by tensor_model_parallel_size (4).
```

**Violated invariant**

`tensor_parallel_num_weight_shards` is the total number of shards each weight is
split into across the tensor-parallel and GTP axes; it must be an exact multiple
of the tensor-parallel size (and it must not be smaller than it).

**Corrected configuration**

```bash
python pretrain_gpt.py \
  --num-layers 12 --hidden-size 1024 \
  --tensor-model-parallel-size 4 --tensor-parallel-num-weight-shards 8 \
  ...
```

## P2P overlap warmup/flush is incompatible with batch_p2p_comm

**Minimal invalid configuration**

```bash
python pretrain_gpt.py \
  --num-layers 24 --hidden-size 1024 \
  --pipeline-model-parallel-size 2 \
  --overlap-p2p-communication-warmup-flush --batch-p2p-comm \
  ...
```

**Error text**

```
Pipeline parallel communication overlapping in warmup and flush is only compatible with overlap_p2p_comm but not batch_p2p_comm.
```

**Violated invariant**

Warmup/flush overlap only composes with the non-batched p2p communication path;
enabling both the overlap flag and batched p2p communication is contradictory.

**Corrected configuration**

```bash
python pretrain_gpt.py \
  --num-layers 24 --hidden-size 1024 \
  --pipeline-model-parallel-size 2 \
  --overlap-p2p-communication-warmup-flush \
  ...
```

## fp8 and fp4 cannot be combined

**Minimal invalid configuration**

```bash
python pretrain_gpt.py \
  --num-layers 12 --hidden-size 1024 \
  --fp8 --fp4 \
  ...
```

**Error text**

```
fp4 and fp8 cannot be used simultaneously. Please choose one.
```

**Violated invariant**

`fp8` and `fp4` are mutually exclusive low-precision modes. Relatedly, enabling
fp8 parameters or fp8 output projections requires the corresponding fp8 mode to
be active (`fp8_param must be used together with fp8 mode.`).

**Corrected configuration**

```bash
python pretrain_gpt.py \
  --num-layers 12 --hidden-size 1024 \
  --fp8 \
  ...
```
