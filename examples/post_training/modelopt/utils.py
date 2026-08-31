# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Shared utilities for modelopt post-training scripts."""
import os
import sys
from typing import Any

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from megatron.core.utils import get_batch_on_this_cp_rank
from megatron.training import get_tokenizer
from megatron.training.utils import get_ltor_masks_and_position_ids


def get_hf_tokenizer():
    """Return the underlying HuggingFace tokenizer, unwrapping Megatron-Core nesting.

    Megatron-Core tokenizers are nested (e.g. get_tokenizer()._tokenizer may itself
    have a .tokenizer or ._tokenizer attribute holding the actual HF tokenizer).
    This helper unwraps one level of that nesting.
    """
    tokenizer = get_tokenizer()._tokenizer
    tok_attrs = ["tokenizer", "_tokenizer"]
    for attr in tok_attrs:
        if hasattr(tokenizer, attr):
            tokenizer = getattr(tokenizer, attr)
            break
    return tokenizer


def get_eos_token_id(hf_tokenizer=None):
    """Return the eos token id used for loss and position masking.

    Some tokenizers use eos tokens inside chat turns; this maps known chat eos strings
    to the token ids used when packing SFT samples.
    """
    if hf_tokenizer is None:
        hf_tokenizer = get_hf_tokenizer()

    if hf_tokenizer.eos_token == "<|eot_id|>":
        return 128001
    if hf_tokenizer.eos_token == "<|eot|>":
        return 200001
    if hf_tokenizer.eos_token == "<|im_end|>":
        return 151643
    if hf_tokenizer.eos_token == "<|return|>":
        return 199999

    return hf_tokenizer.eos_token_id


def build_lm_batch(
    input_ids: torch.Tensor,
    seq_length: int,
    *,
    sample_loss_mask: torch.Tensor | None = None,
    pad_attention_mask: torch.Tensor | None = None,
    eos_token_id: int | None = None,
    reset_position_ids: bool = False,
    reset_attention_mask: bool = False,
    eod_mask_loss: bool = False,
    pad_mask_loss: bool = False,
    cp_group: torch.distributed.ProcessGroup | None = None,
    is_hybrid_cp: bool = False,
) -> dict[str, torch.Tensor]:
    """Build causal-LM training tensors from packed or padded ``input_ids``.

    ``input_ids`` must contain ``seq_length + 1`` tokens per row so that ``tokens``
    and next-token ``labels`` both have length ``seq_length``.

    Args:
        input_ids: Token ids with an extra trailing token for the label shift.
        seq_length: Number of input tokens (excluding the extra label token).
        sample_loss_mask: Optional per-token mask aligned with ``input_ids``. When
            provided, only positions with a non-zero mask at the label positions
            contribute to ``loss_mask`` (SFT answer-only masking).
        pad_attention_mask: Optional HuggingFace-style attention mask aligned with
            ``input_ids``. When provided, padding positions are zeroed out in
            ``loss_mask`` using the label-aligned slice.
        eos_token_id: Eos token id for ``get_ltor_masks_and_position_ids``.
        reset_position_ids: Passed through to ``get_ltor_masks_and_position_ids``.
        reset_attention_mask: Passed through to ``get_ltor_masks_and_position_ids``.
        eod_mask_loss: Passed through to ``get_ltor_masks_and_position_ids``.
        pad_mask_loss: Passed through to ``get_ltor_masks_and_position_ids``.
        cp_group: When set, slice the batch for context parallelism.
        is_hybrid_cp: Passed through to ``get_batch_on_this_cp_rank``.

    Returns:
        Dict with ``tokens``, ``labels``, ``loss_mask``, ``attention_mask``, and
        ``position_ids`` ready for ``GPTModel.forward``.
    """
    if eos_token_id is None:
        eos_token_id = get_eos_token_id()

    tokens = input_ids[:, :seq_length].contiguous()
    labels = input_ids[:, 1 : seq_length + 1].contiguous()

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        eos_token_id,
        eos_token_id,
        reset_position_ids,
        reset_attention_mask,
        eod_mask_loss,
        pad_mask_loss,
    )

    if sample_loss_mask is not None:
        answer_only_loss_mask = sample_loss_mask[:, 1 : seq_length + 1].contiguous()
        loss_mask = loss_mask * answer_only_loss_mask.to(dtype=loss_mask.dtype)

    if pad_attention_mask is not None:
        pad_mask = pad_attention_mask[:, 1 : seq_length + 1].to(dtype=loss_mask.dtype)
        loss_mask = loss_mask * pad_mask

    batch = {
        "tokens": tokens,
        "labels": labels.contiguous(),
        "loss_mask": loss_mask.contiguous(),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    if cp_group is not None:
        batch = get_batch_on_this_cp_rank(batch, is_hybrid_cp=is_hybrid_cp, cp_group=cp_group)

    return batch


def build_lm_batch_from_input_ids(
    batch: dict[str, Any],
    *,
    seq_length: int | None = None,
    eos_token_id: int | None = None,
    reset_position_ids: bool = False,
    reset_attention_mask: bool = False,
    eod_mask_loss: bool = False,
    pad_mask_loss: bool = False,
    cp_group: torch.distributed.ProcessGroup | None = None,
    is_hybrid_cp: bool = False,
) -> dict[str, torch.Tensor]:
    """Build an LM batch dict from a dataloader batch containing ``input_ids``.

    Calibration and HF dataloaders provide ``input_ids`` of shape
    ``[batch, seq_length + 1]`` (or pass ``seq_length=input_ids.shape[1] - 1``).
    An optional ``attention_mask`` entry is used to mask padded label positions.
    """
    input_ids = batch["input_ids"]
    if seq_length is None:
        seq_length = input_ids.shape[1] - 1

    pad_attention_mask = batch.get("attention_mask")
    sample_loss_mask = batch.get("loss_mask")

    return build_lm_batch(
        input_ids,
        seq_length,
        sample_loss_mask=sample_loss_mask,
        pad_attention_mask=pad_attention_mask,
        eos_token_id=eos_token_id,
        reset_position_ids=reset_position_ids,
        reset_attention_mask=reset_attention_mask,
        eod_mask_loss=eod_mask_loss,
        pad_mask_loss=pad_mask_loss,
        cp_group=cp_group,
        is_hybrid_cp=is_hybrid_cp,
    )
