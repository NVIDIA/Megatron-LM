# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Vocab-parallel cross entropy.

A target is called as ``loss = target(logits, labels, tp_group)`` with ``logits``
``[s, b, vocab / tp]`` and ``labels`` ``[s, b]``, and returns the per-token loss
``[s, b]``. The target owns every reduction over ``tp_group`` and consumes ``logits``
in place; ``backends.CrossEntropyTarget`` states the contract. Targets hold no
parameters or state.

* ``cross_entropy`` -- the eager custom-autograd implementation, with label smoothing,
  and the ``VocabParallelCrossEntropy`` steps the fused path shares.
* ``fused_cross_entropy`` -- the same computation compiled with ``jit_fuser``.
* ``backends`` -- ``select_cross_entropy``, which turns ``cross_entropy_loss_fusion``
  and ``cross_entropy_fusion_impl`` into the target a model binds once.

Transformer Engine's cross entropy stays in ``megatron.core.extensions.transformer_engine``.
Packages here do not re-export; import each name from the module that defines it.
"""
