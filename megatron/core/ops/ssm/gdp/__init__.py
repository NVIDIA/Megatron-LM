# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# The inference kernel modules in this package are forked from the Gated Delta Product
# kernels in flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention), licensed under the MIT
# license. See the LICENSE file in the repository root.

"""Gated Delta Product (GDP) operation and an inference-only fork of its kernels from
`flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_
(v0.5.1).

``mixer`` owns the operation's parameters, checkpoint mappings and execution;
``context_parallel`` implements its head-partitioned communication.

Training and the static-batching inference path call the pip
`flash-linear-attention` kernels, which own the backward pass. Only the
dynamic-batching decode and prefill steps route to the local kernels:

* `fused_recurrent.fused_recurrent_gated_delta_rule_update` -- decode (shared with
  GDN, whose gate and beta sigmoid it can apply in-kernel).
* `prepare.gdp_decode_prepare` -- the reshape/gating step feeding decode.
* `chunk.chunk_gated_delta_product_varlen` -- prefill.
* `prepare.gdp_prefill_prepare` -- the reshape/gating step feeding prefill.

All local kernel entry points are forward-only.
"""
