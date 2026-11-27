# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Encode prepared documents once, preserving eligible targets after lazy images."""

import operator

import torch

from megatron.energon import Cooker, DefaultTaskEncoder, Sample, stateless
from megatron.energon.edataclass import edataclass

from .binidx_token_mask import BinIdxTokenMaskDecoder
from .conversation_sample import ImageMedia
from .cookers.multimodal_interleaved_binidx import cook_multimodal_interleaved_binidx
from .image_processing import ImageTilingParams, ImageTilingStrategy
from .prepared_sample import InterleavedDocument, TokenizedText


@edataclass
class EncodedPreparedDocument(Sample):
    """Unbatched model positions and lazy plans, before sequence splitting/packing.

    ``labels[i]`` is the eligible next token predicted at ``input_ids[i]``.
    Image spans are half-open offsets into these expanded arrays, in plan order.
    ``loss_mask`` is float32; IDs and labels are int64. All tensors are on CPU.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    image_spans: list[tuple[int, int]]
    image_params: list[ImageTilingParams]


def _integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


class PreparedDocumentTaskEncoder(DefaultTaskEncoder):
    """Turn prepared IDs and typed images into next-token-aligned documents.

    Args:
        image_strategy: Existing image strategy whose ``num_embeddings`` counts
            actual visual embeddings only. Source end/break IDs remain in text;
            their overhead must not also be counted by this strategy.
        image_token_id: Structural sentinel repeated once per visual embedding.
            Text parts must not contain this ID.
        eos_token_id: Required last source token; never appended or substituted.
        vocab_size: Positive model vocabulary size, bounding all input IDs.
        image_token_budget: Positive per-image visual-position limit passed to
            the strategy and enforced on its returned plan. Documents may exceed
            this length; this encoder does not truncate, split, pad or pack them.
        ignore_index: Ignored label value, outside the model vocabulary (-100).

    No pixels are read during encoding. The consumer later materializes plans
    and owns batching, position/attention policy and sequence splitting. Source
    eligibility controls targets, including any source end marker; an ineligible
    input can still predict an eligible next token. Completely targetless
    documents are rejected, consistent with the prepared causal task contract.
    """

    cookers = (
        Cooker(
            cook=cook_multimodal_interleaved_binidx,
            has_tags={"cook": "multimodal_interleaved_binidx"},
        ),
    )
    decoder = BinIdxTokenMaskDecoder(image_decode="pil")

    def __init__(
        self,
        image_strategy: ImageTilingStrategy,
        *,
        image_token_id: int,
        eos_token_id: int,
        vocab_size: int,
        image_token_budget: int,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.image_strategy = image_strategy
        self.vocab_size = _integer(vocab_size, "vocab_size")
        self.image_token_budget = _integer(image_token_budget, "image_token_budget")
        self.image_token_id = _integer(image_token_id, "image_token_id")
        self.eos_token_id = _integer(eos_token_id, "eos_token_id")
        self.ignore_index = _integer(ignore_index, "ignore_index")
        if self.vocab_size <= 0 or self.image_token_budget <= 0:
            raise ValueError("vocab_size and image_token_budget must be positive")
        if (
            not 0 <= self.image_token_id < self.vocab_size
            or not 0 <= self.eos_token_id < self.vocab_size
            or self.image_token_id == self.eos_token_id
        ):
            raise ValueError("image_token_id and eos_token_id must be distinct vocabulary IDs")
        if 0 <= self.ignore_index < self.vocab_size:
            raise ValueError("ignore_index must be outside the vocabulary")

    @stateless
    def encode_sample(self, sample: InterleavedDocument) -> EncodedPreparedDocument:
        """Expand visual positions and shift source target eligibility exactly once."""
        tokens, eligible, spans, plans = [], [], [], []
        for part in sample.parts:
            if isinstance(part, TokenizedText):
                if any(
                    type(token) is not int or not 0 <= token < self.vocab_size
                    for token in part.tokens
                ):
                    raise ValueError("prepared text IDs must be integers in the vocabulary")
                if self.image_token_id in part.tokens:
                    raise ValueError("prepared text contains a structural image sentinel")
                flags = part.token_loss_mask
                if flags is not None and (
                    len(flags) != len(part.tokens) or any(type(flag) is not bool for flag in flags)
                ):
                    raise ValueError("token_loss_mask requires one boolean per source token")
                tokens.extend(part.tokens)
                eligible.extend([True] * len(part.tokens) if flags is None else flags)
            elif isinstance(part, ImageMedia):
                selected = self.image_strategy.compute_params([part], self.image_token_budget)
                if len(selected) != 1 or not isinstance(selected[0], ImageTilingParams):
                    raise ValueError("each image requires exactly one image plan")
                plan = selected[0]
                count = _integer(plan.num_embeddings, "image plan num_embeddings")
                if plan.media is not part or not 0 < count <= self.image_token_budget:
                    raise ValueError("image plan must retain its media and fit image_token_budget")
                start = len(tokens)
                tokens.extend([self.image_token_id] * count)
                eligible.extend([False] * count)
                spans.append((start, len(tokens)))
                plans.append(plan)
            else:
                raise TypeError(f"unsupported prepared part: {type(part).__name__}")
        if not tokens or tokens[-1] != self.eos_token_id:
            raise ValueError("prepared document must retain its configured terminal EOS")
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.full_like(input_ids, self.ignore_index)
        # Eligibility belongs to the target. The final visual position therefore
        # retains its next text target, even though visual targets are excluded.
        labels[:-1] = torch.where(
            torch.tensor(eligible[1:], dtype=torch.bool), input_ids[1:], self.ignore_index
        )
        loss_mask = (labels != self.ignore_index).float()
        if not torch.any(loss_mask):
            raise ValueError("prepared document has no eligible next-token targets")
        return EncodedPreparedDocument.derive_from(
            sample,
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            image_spans=spans,
            image_params=plans,
        )
