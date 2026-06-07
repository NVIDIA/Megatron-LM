# MoT extension of PackedSeqParams.
#
# We intentionally avoid modifying megatron/core/packed_seq_params.py.
# A plain dataclass subclass adds the MoT-specific fields while remaining
# a drop-in replacement everywhere PackedSeqParams is accepted.

from dataclasses import dataclass, field
from typing import Optional, List

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams



@dataclass
class MoTPackedSeqParams(PackedSeqParams):
    """PackedSeqParams extended with MoT (Mixture-of-Transformers) fields.

    All new fields default to None so existing code that only reads the base
    fields is completely unaffected.

    Field naming:
        packed_*_token_indexes  — global arrays, replicated on all CP ranks
        local_*_token_indexes   — this rank's actual (unpadded) slice; len() == real token count
        padded_*_seqlen         — uniform per-rank chunk size for balanced A2A
                                  Lund = ceil(U / cp_size), Lgen = ceil(G / cp_size)
    """
    packed_vit_token_indexes: Optional[Tensor] = torch.tensor([]) # [V]  global vit token positions
    packed_vae_token_indexes: Optional[Tensor] = torch.tensor([])  # [G]  global vae token positions
    packed_text_indexes: Optional[Tensor] = torch.tensor([])  # [T]  global text token positions

    # global index arrays: replicated on every CP rank in the same DP group
    packed_und_token_indexes: Optional[Tensor] = torch.tensor([])  # [U]  global und positions
    packed_gen_token_indexes: Optional[Tensor] = torch.tensor([])  # [G]  global gen positions

    # local index arrays: actual (unpadded) slice for this CP rank
    # len(local_und_token_indexes) == actual und tokens on this rank (<= Lund)
    local_und_token_indexes: Optional[Tensor] = torch.tensor([])   # [actual_lund]
    local_gen_token_indexes: Optional[Tensor] = torch.tensor([])   # [actual_lgen]

    # padded per-rank chunk sizes (uniform across all ranks, required for A2A)
    padded_und_seqlen: Optional[int] = 0            # Lund = ceil(U / cp_size)
    padded_gen_seqlen: Optional[int] = 0            # Lgen = ceil(G / cp_size)

    # vit token seqlens before encoding
    # vit_token_seqlens: Optional[Tensor] = None  # [N_img]  vit token seqlens before encoding
    vit_tokens_encoded_per_cp: Optional[List[int]] = field(default_factory=list)  # [cp_size]  vit tokens encoded per cp