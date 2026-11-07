<!---
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# tensor_parallel package

This package contains an implementation for tensor parallelism in transformer
models (see [Megatron-LM: Training Multi-Billion Parameter Language Models
Using Model Parallelism](https://arxiv.org/abs/1909.08053) and [Reducing
Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
for details).

## Vocabulary-parallel cross entropy

`vocab_parallel_cross_entropy` (and the fused `fused_vocab_parallel_cross_entropy`) computes the
loss with the vocabulary split across the tensor-parallel ranks: each rank reduces its own slice,
and the per-token maximum and sums are all-reduced.

For a confident token, the sum of the non-target terms is tiny next to the target's `exp(0) = 1`
and would be rounded away if added to it, which would zero both the loss and the target gradient
(see [A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642)). The implementation
therefore reduces the non-target terms separately and computes

- the loss as `softplus(log(sum_exp_others) - predicted_logit)`, and
- the target gradient as `-sum_exp_others / sum_exp_logits`,

which stay accurate at any confidence. Both sums come from one reduction per rank and are
all-reduced in the same call as before.
