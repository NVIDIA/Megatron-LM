# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Causal document encoding and pull-based packing on Energon's stateful wrappers.

See MULTIMODAL_INTERLEAVED.md for source provenance and the consumer-adapter boundary.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Iterator

import torch

from megatron.energon import (
    Cooker,
    DefaultTaskEncoder,
    PackedSamplesOutput,
    PackingGroupConfig,
    PartialSample,
    stateless,
)

from .binidx_token_mask import BinIdxTokenMaskDecoder
from .conversation_sample import ImageMedia
from .cookers.multimodal_interleaved import cook_multimodal_interleaved
from .cookers.multimodal_interleaved_binidx import cook_multimodal_interleaved_binidx
from .image_processing import ImageTilingStrategy
from .multimodal_interleaved_samples import (
    EncodedDocument,
    InterleavedDocument,
    InterleavedPackedSample,
    TokenizedText,
)


@dataclass(frozen=True)
class InterleavedPackingConfig:
    """Model-token budget and explicit causal-pretraining token contract."""

    sequence_length: int
    """Packed length after image expansion, including padding."""
    image_token_id: int
    """Consumer's structural image token; literal image text is encoded separately."""
    eos_token_id: int
    """Append once at the original document boundary, never at arbitrary split points."""
    pad_token_id: int
    """Input padding token; padding targets are always ignored."""
    ignore_index: int = -100
    """Ignored text targets, including image slots and padding."""

    def __post_init__(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("interleaved packing needs at least two model tokens")
        if self.eos_token_id < 0 or self.pad_token_id < 0:
            raise ValueError("interleaved packing requires valid EOS and padding token IDs")
        if self.image_token_id in (self.eos_token_id, self.pad_token_id):
            raise ValueError("structural image token must differ from EOS and padding")


class MultimodalInterleavedTaskEncoder(
    DefaultTaskEncoder[InterleavedDocument, InterleavedPackedSample, dict, dict]
):
    """Share document semantics/packing; let consumer adapters implement ``batch``.

    Energon owns all cursors, carryover buffers and checkpoint replay. There is
    deliberately no mutable iterator or progress counter on this encoder.
    """

    cookers = [
        Cooker(
            cook_multimodal_interleaved_binidx, has_tags={"cook": "multimodal_interleaved_binidx"}
        ),
        Cooker(cook_multimodal_interleaved),
    ]
    decoder = BinIdxTokenMaskDecoder(image_decode="pil", av_decode="AVDecoder", guess_content=True)

    def _find_cooker(self, tags):
        # Native recipes may omit tags; filtered Cooker matching requires a dict.
        return super()._find_cooker({} if tags is None else tags)

    def __init__(
        self, tokenizer, image_strategy: ImageTilingStrategy, config: InterleavedPackingConfig
    ):
        super().__init__()
        if getattr(tokenizer, "use_gigatoken", False) or getattr(
            getattr(tokenizer, "_tokenizer", None), "use_gigatoken", False
        ):
            raise ValueError("interleaved ordinary-text encoding does not support GigaToken")
        self.tokenizer = tokenizer
        self.image_strategy = image_strategy
        self.config = config

    def _encode_text(self, text: str) -> list[int]:
        # Encode the complete span once, preserving normal BPE boundaries while
        # keeping all caller-controlled special-token spellings ordinary text.
        result = self.tokenizer.tokenize(text, add_special_tokens=False, split_special_tokens=True)
        if self.config.image_token_id in result:
            raise ValueError("text encoding produced a structural image ID")
        return result

    @stateless
    def preencode_sample(self, sample: InterleavedDocument) -> EncodedDocument:
        """Encode text once and compute image plans without reading pixels."""
        cfg = self.config
        tokens, pending_text, images, positions = [], [], [], []
        ownership = None
        prepared = False

        def flush_text():
            if pending_text:
                text_tokens = self._encode_text("".join(pending_text))
                tokens.extend(text_tokens)
                if ownership is not None:
                    ownership.extend([True] * len(text_tokens))
                pending_text.clear()

        for part in sample.parts:
            if isinstance(part, str):
                pending_text.append(part)
            elif isinstance(part, TokenizedText):
                flush_text()
                if any(type(token) is not int for token in part.tokens) or (
                    part.tokens
                    and (min(part.tokens) < 0 or max(part.tokens) >= self.tokenizer.vocab_size)
                ):
                    raise ValueError(
                        "prepared text IDs must be integers in the configured vocabulary"
                    )
                if cfg.image_token_id in part.tokens:
                    raise ValueError("prepared text contains an unmatched structural image ID")
                mask = part.token_loss_mask
                if mask is not None and (
                    len(mask) != len(part.tokens) or any(type(flag) is not bool for flag in mask)
                ):
                    raise ValueError("prepared token_loss_mask must contain one boolean per token")
                if mask is not None and ownership is None:
                    ownership = [True] * len(tokens)
                tokens.extend(part.tokens)
                if ownership is not None:
                    ownership.extend([True] * len(part.tokens) if mask is None else mask)
                prepared = True
            elif isinstance(part, ImageMedia):
                flush_text()
                positions.append(len(tokens))
                tokens.append(cfg.image_token_id)
                if ownership is not None:
                    ownership.append(False)
                # Give each image the same independent model budget on both
                # roles. Do not resize/drop it depending on the remaining pack.
                plans = self.image_strategy.compute_params([part], cfg.sequence_length)
                if len(plans) != 1 or plans[0].num_embeddings <= 0:
                    raise ValueError("interleaved images require exactly one nonempty image plan")
                images.append(plans[0])
            else:
                raise TypeError(f"unsupported interleaved document part: {type(part).__name__}")
        flush_text()
        if not tokens:
            raise ValueError(f"empty interleaved document {sample.__key__!r}")
        if prepared and tokens[-1] != cfg.eos_token_id:
            raise ValueError("prepared document must retain its configured terminal EOS")
        if tokens[-1] != cfg.eos_token_id:
            tokens.append(cfg.eos_token_id)
            if ownership is not None:
                ownership.append(True)

        token_tensor = torch.tensor(tokens, dtype=torch.long)
        labels = torch.full_like(token_tensor, cfg.ignore_index)
        labels[:-1] = token_tensor[1:]
        if ownership is not None:
            labels[:-1].masked_fill_(
                ~torch.tensor(ownership[1:], dtype=torch.bool), cfg.ignore_index
            )
        # Shift before slicing: a fragment's final text position keeps the
        # original next-token target. No target is shifted a second time.
        labels[(token_tensor == cfg.image_token_id) | (labels == cfg.image_token_id)] = (
            cfg.ignore_index
        )
        if not torch.any(labels != cfg.ignore_index):
            raise ValueError(
                f"interleaved document {sample.__key__!r} has no trainable text targets"
            )
        costs = [1] * len(tokens)
        for position, params in zip(positions, images, strict=True):
            costs[position] = int(params.num_embeddings)
        prefix = [0]
        for cost in costs:
            prefix.append(prefix[-1] + cost)
        return EncodedDocument.derive_from(
            sample,
            tokens=token_tensor,
            labels=labels,
            image_params=images,
            image_positions=positions,
            length_prefix=prefix,
        )

    @staticmethod
    def _window(sample):
        if isinstance(sample, PartialSample):
            return sample.sample, *sample.slice
        return sample, 0, len(sample.tokens)

    @stateless
    def select_next_pack(self, samples: Iterator[EncodedDocument | PartialSample]):
        """Fill one pack, pushing back an overflowing suffix with its full provenance."""
        remaining, pack, pushback = self.config.sequence_length, [], []
        for sample in samples:
            base, start, stop = self._window(sample)
            size = base.length_prefix[stop] - base.length_prefix[start]
            if size <= remaining:
                pack.append(sample)
                remaining -= size
            else:
                end = min(
                    stop,
                    bisect_right(base.length_prefix, base.length_prefix[start] + remaining) - 1,
                )
                # Keep an image and its following text together. If it cannot
                # fit an empty pack, fail explicitly instead of dropping it.
                while end > start and int(base.tokens[end - 1]) == self.config.image_token_id:
                    end -= 1
                if end > start:
                    pack.append(PartialSample(base, (start, end)))
                    pushback.append(PartialSample(base, (end, stop)))
                elif pack:
                    pushback.append(sample)
                else:
                    raise ValueError(
                        f"image span in {base.__key__!r} cannot fit interleaved pack "
                        f"of {self.config.sequence_length} tokens"
                    )
                break
            if remaining == 0:
                break
        return PackedSamplesOutput(packs=[pack] if pack else [], pushback=pushback)

    @stateless
    def postencode_sample(self, sample):
        """Resolve the selected token/image window without materializing pixels."""
        base, start, stop = self._window(sample)
        first = bisect_left(base.image_positions, start)
        last = bisect_left(base.image_positions, stop)
        length = base.length_prefix[stop] - base.length_prefix[start]
        result = InterleavedPackedSample.derive_from(
            base,
            tokens=base.tokens[start:stop],
            labels=base.labels[start:stop],
            image_params=base.image_params[first:last],
            cu_seqlens=torch.tensor([0, length], dtype=torch.int32),
            fragments=[(base.__key__, start, stop)],
        )
        return result

    @stateless
    def pack_selected_samples(
        self, samples: list[InterleavedPackedSample]
    ) -> InterleavedPackedSample:
        """Concatenate selected windows; boundaries use expanded model coordinates."""
        if not samples:
            raise ValueError("cannot produce an empty interleaved pack")
        cfg = self.config
        boundaries = [0]
        for sample in samples:
            boundaries.extend((sample.cu_seqlens[1:] + boundaries[-1]).tolist())
        padding = cfg.sequence_length - boundaries[-1]
        if padding < 0:
            raise ValueError("interleaved pack exceeds its model-token budget")
        tokens = torch.cat([sample.tokens for sample in samples])
        labels = torch.cat([sample.labels for sample in samples])
        if padding:
            tokens = torch.cat((tokens, torch.full((padding,), cfg.pad_token_id)))
            labels = torch.cat((labels, torch.full((padding,), cfg.ignore_index)))
            boundaries.append(cfg.sequence_length)
        if not torch.any(labels != cfg.ignore_index):
            raise ValueError("interleaved pack has no trainable targets")
        return InterleavedPackedSample.derive_from(
            samples[0],
            __key__="|".join(sample.__key__ for sample in samples),
            __restore_key__=(),  # Energon installs the final pack restore key.
            __sources__=tuple(source for sample in samples for source in (sample.__sources__ or ()))
            or None,
            tokens=tokens,
            labels=labels,
            image_params=[p for sample in samples for p in sample.image_params],
            cu_seqlens=torch.tensor(boundaries, dtype=torch.int32),
            fragments=[f for sample in samples for f in sample.fragments],
        )

    def build_packing_groups(self, datasets, packing_buffer_size, shuffle_buffer_size):
        """Pack each source independently unless an explicit packing_group joins it."""
        if packing_buffer_size != "stream":
            raise ValueError("interleaved pretraining supports only packing_buffer_size='stream'")
        groups = {}
        for index, loaded in enumerate(datasets):
            tags = getattr(loaded.dataset, "tags", None) or {}
            group = tags.get("packing_group")
            key = ("explicit", str(group)) if group is not None else ("source", index)
            groups.setdefault(key, []).append(loaded)
        return [
            PackingGroupConfig(
                datasets=group,
                packing_buffer_size="stream",
                shuffle_buffer_size=shuffle_buffer_size,
            )
            for group in groups.values()
        ]
