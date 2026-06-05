"""Runtime math primitives for Megatron Lite."""

from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.gated_delta_rule import l2norm, torch_chunk_gated_delta_rule
from megatron.lite.primitive.ops.logprob import (
    vocab_parallel_entropy,
    vocab_parallel_log_probs_from_logits,
)
from megatron.lite.primitive.ops.sp_ops import (
    AllGatherDim0,
    AllGatherDim0ForNonSPConsumer,
    ReduceScatterDim0,
    ScatterToSP,
)

__all__ = [
    "AllGatherDim0",
    "AllGatherDim0ForNonSPConsumer",
    "ReduceScatterDim0",
    "ScatterToSP",
    "l2norm",
    "torch_chunk_gated_delta_rule",
    "vocab_parallel_cross_entropy",
    "vocab_parallel_entropy",
    "vocab_parallel_log_probs_from_logits",
]
