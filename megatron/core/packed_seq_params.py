from dataclasses import dataclass

from torch import Tensor


@dataclass
class PackedSeqParams:
    # parameters to TEDotProductAttention and fused rope kernels for the `thd` (packed) sequence format,
    qkv_format: str = None
    cu_seqlens_q: Tensor = None
    cu_seqlens_kv: Tensor = None
    max_seqlen_q: Tensor = None
    max_seqlen_kv: Tensor = None
