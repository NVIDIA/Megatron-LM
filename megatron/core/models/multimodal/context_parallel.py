# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""Multimodal Sequence Parallel (SP) and Context Parallel (CP) functionality."""

import torch

from megatron.core.packed_seq_params import PackedSeqParams


def get_padding(
    seq_len, cp_size, tp_size, has_sp, decoder_tp_comm_overlap=False, decoder_seq_len=None
):
    """Calculate padding needed for SP and/or CP.

    Args:
        seq_len (int): Model sequence length.
        cp_size (int): Context parallel size.
        tp_size (int): Tensor parallel size.
        has_sp (bool): Model uses sequence parallelism.
        decoder_tp_comm_overlap (bool): Decoder (LLM) uses tensor parallel communication overlap.
        decoder_seq_len (int): Decoder (LLM) maximum sequence length.

    Returns:
        padding (int): Padding needed given model configuration.
    """

    padding = 0
    # TP Comm overlap is performed with combined text+image embeddings.
    if has_sp and decoder_tp_comm_overlap:
        # If TP Comm Overlap is enabled for combined text+image embedding in LM backbone,
        # user needs to provide decoder_seq_len with any potential padding needed for SP+CP
        assert (
            decoder_seq_len is not None
        ), "Please provide decoder seq length when using TP comm overlap for LM backbone"
        padding = decoder_seq_len - seq_len
    elif has_sp or cp_size > 1:
        padding_factor = 1
        if has_sp and cp_size > 1:
            # Padding to multiple of tp_size * cp_size * 2 when using CP + SP.
            padding_factor = tp_size * cp_size * 2
        elif cp_size > 1:
            padding_factor = cp_size * 2
        elif has_sp:
            padding_factor = tp_size

        padding = int((seq_len + padding_factor - 1) // padding_factor * padding_factor) - seq_len

    return padding


def get_packed_seq_params(tokens, img_seq_len, padding_needed, cp_size, use_packed_sequence=False):
    """Get PackedSeqParams for CP.

    Args:
        tokens (torch.Tensor): [batch, seq_len] input tokens.
        img_seq_len (int): Image sequence length.
        padding_needed (int): Padding to add.
        cp_size (int): Context parallel size.
        use_packed_sequence (bool): Uses sequence packing.

    Returns:
        packed_seq_params (PackedSeqParams): Parameters to be sent to Transformer Engine.
    """
    batch_size = tokens.shape[0]
    # Calculate the valid token seq len that LM backbone should compute on
    combined_valid_seqlen = tokens.shape[1] + img_seq_len - padding_needed
    cu_seqlens = torch.arange(
        0,
        (batch_size + 1) * (combined_valid_seqlen),
        step=(combined_valid_seqlen),
        dtype=torch.int32,
        device=tokens.device,
    )
    # Calculate the total padded token seq len
    combined_padded_seqlen = tokens.shape[1] + img_seq_len
    cu_seqlens_padded = None
    qkv_format = 'sbhd'
    if cp_size > 1 and (padding_needed > 0 or use_packed_sequence):
        # Provide cu_seqlens_<q/kv>_padded for CP support
        cu_seqlens_padded = torch.arange(
            0,
            (batch_size + 1) * (combined_padded_seqlen),
            step=(combined_padded_seqlen),
            dtype=torch.int32,
            device=tokens.device,
        )
        # CP with padding mask type requires THD format
        qkv_format = 'thd'

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=combined_padded_seqlen,
        max_seqlen_kv=combined_padded_seqlen,
        qkv_format=qkv_format,
    )

    return packed_seq_params
