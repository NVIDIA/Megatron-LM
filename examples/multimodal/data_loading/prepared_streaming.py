# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Image-safe streaming packs for prepared causal documents."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from megatron.energon import PackedSamplesOutput, PackingGroupConfig, PartialSample, stateless
from megatron.energon.edataclass import edataclass
from megatron.energon.task_encoder.base import LoadedDataset

from .prepared_task_encoder import EncodedPreparedDocument, PreparedDocumentTaskEncoder, _integer


@edataclass
class PackedPreparedDocument(EncodedPreparedDocument):
    """A padded sequence with source windows and pack-relative image spans.

    ``fragment_offsets`` are cumulative packed positions, ending at
    ``valid_length`` and excluding padding. ``fragments`` records each source key
    and its half-open expanded-position window in that source document.
    """

    fragment_offsets: list[int]
    fragments: list[tuple[str, int, int]]
    valid_length: int


class PreparedStreamingTaskEncoder(PreparedDocumentTaskEncoder):
    """Split prepared documents only at image boundaries and stream fixed packs."""

    def __init__(
        self,
        image_strategy,
        *,
        image_token_id: int,
        eos_token_id: int,
        vocab_size: int,
        image_token_budget: int,
        sequence_length: int,
        pad_token_id: int,
        ignore_index: int = -100,
    ):
        if type(sequence_length) is not int or sequence_length <= 0:
            raise ValueError("sequence_length must be a positive built-in integer")
        pad_token_id = _integer(pad_token_id, "pad_token_id")
        super().__init__(
            image_strategy,
            image_token_id=image_token_id,
            eos_token_id=eos_token_id,
            vocab_size=vocab_size,
            image_token_budget=image_token_budget,
            ignore_index=ignore_index,
        )
        if not 0 <= pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside the vocabulary")
        if pad_token_id == self.image_token_id:
            raise ValueError("pad_token_id must differ from image_token_id")
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id

    def build_packing_groups(
        self, datasets: list[LoadedDataset], packing_buffer_size, shuffle_buffer_size: int | None
    ) -> list[PackingGroupConfig]:
        """Keep each source in its own streaming packer and retain its shuffle size."""
        if packing_buffer_size != "stream":
            raise ValueError("PreparedStreamingTaskEncoder requires packing_buffer_size='stream'")
        return [
            PackingGroupConfig(
                datasets=[dataset],
                packing_buffer_size="stream",
                shuffle_buffer_size=shuffle_buffer_size,
            )
            for dataset in datasets
        ]

    @staticmethod
    def _window(sample) -> tuple[EncodedPreparedDocument, int, int]:
        if isinstance(sample, PartialSample):
            base = sample.sample
            bounds = sample.slice
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(type(bound) is not int for bound in bounds)
            ):
                raise ValueError("prepared partial samples require integer (start, stop) windows")
            start, stop = bounds
        else:
            base = sample
            if not isinstance(base, EncodedPreparedDocument):
                raise ValueError("streaming packs require encoded prepared documents")
            start, stop = 0, len(base.input_ids)
        if not isinstance(base, EncodedPreparedDocument):
            raise ValueError("partial prepared samples must retain their encoded base document")
        length = len(base.input_ids)
        if not 0 <= start < stop <= length:
            raise ValueError("prepared source window must be non-empty and within its document")
        if base.input_ids.ndim != 1 or base.labels.ndim != 1 or base.loss_mask.ndim != 1:
            raise ValueError("prepared IDs, labels, and loss mask must be one-dimensional")
        if any(
            tensor.device.type != "cpu" for tensor in (base.input_ids, base.labels, base.loss_mask)
        ):
            raise ValueError("prepared streaming pack tensors must remain on CPU")
        if (
            base.labels.shape != base.input_ids.shape
            or base.loss_mask.shape != base.input_ids.shape
        ):
            raise ValueError("prepared IDs, labels, and loss mask must have matching shapes")
        if len(base.image_spans) != len(base.image_params):
            raise ValueError("prepared image spans and plans must have matching lengths")
        previous_stop = 0
        for image_start, image_stop in base.image_spans:
            if (
                type(image_start) is not int
                or type(image_stop) is not int
                or not previous_stop <= image_start < image_stop <= length
            ):
                raise ValueError("prepared image spans must be ordered non-empty document windows")
            if image_start < start < image_stop or image_start < stop < image_stop:
                raise ValueError("selected prepared window cuts through an image span")
            previous_stop = image_stop
        return base, start, stop

    def _next_boundary(
        self, base: EncodedPreparedDocument, start: int, stop: int, limit: int
    ) -> int:
        """Return the furthest legal cut no later than ``limit``."""
        boundary = limit
        for image_start, image_stop in base.image_spans:
            if image_stop - image_start > self.sequence_length:
                raise ValueError(
                    f"image span length {image_stop - image_start} exceeds sequence_length "
                    f"({self.sequence_length})"
                )
            if image_start < boundary < image_stop:
                boundary = image_start
                break
        return boundary if start < boundary <= stop else start

    @stateless
    def select_next_pack(self, samples: Iterator):
        """Pull only enough document windows to fill one pack, carrying exact suffixes."""
        pack = []
        pushback = []
        used = 0
        while used < self.sequence_length:
            try:
                sample = next(samples)
            except StopIteration:
                break

            base, start, stop = self._window(sample)
            limit = min(stop, start + self.sequence_length - used)
            boundary = self._next_boundary(base, start, stop, limit)
            if boundary == start:
                if pack:
                    pushback.append(sample)
                    break
                raise ValueError(
                    "no image-safe window fits the prepared document in an empty sequence"
                )

            pack.append(PartialSample(sample=base, slice=(start, boundary)))
            used += boundary - start
            if boundary < stop:
                pushback.append(PartialSample(sample=base, slice=(boundary, stop)))
                break

        if not pack:
            return []
        return PackedSamplesOutput(packs=[pack], pushback=pushback)

    @stateless
    def pack_selected_samples(self, samples: list[PartialSample | EncodedPreparedDocument]):
        """Concatenate selected windows without shifting labels across documents."""
        input_ids = []
        labels = []
        loss_masks = []
        image_spans = []
        image_params = []
        fragment_offsets = [0]
        fragments = []
        source_infos = []
        first_base = None
        valid_length = 0

        for sample in samples:
            base, start, stop = self._window(sample)
            if first_base is None:
                first_base = base
            window_length = stop - start
            if valid_length + window_length > self.sequence_length:
                raise ValueError("selected prepared windows exceed sequence_length")
            input_ids.append(base.input_ids[start:stop])
            labels.append(base.labels[start:stop])
            loss_masks.append(base.loss_mask[start:stop])
            fragments.append((base.__key__, start, stop))
            valid_length += window_length
            fragment_offsets.append(valid_length)

            for span_index, (image_start, image_stop) in enumerate(base.image_spans):
                if start <= image_start and image_stop <= stop:
                    image_spans.append(
                        (
                            valid_length - window_length + image_start - start,
                            valid_length - window_length + image_stop - start,
                        )
                    )
                    image_params.append(base.image_params[span_index])
            if base.__sources__ is not None:
                for source in base.__sources__:
                    if source not in source_infos:
                        source_infos.append(source)

        if first_base is None or valid_length == 0:
            raise ValueError("cannot pack an empty prepared sequence")
        if valid_length < self.sequence_length:
            padding = self.sequence_length - valid_length
            input_ids.append(torch.full((padding,), self.pad_token_id, dtype=torch.long))
            labels.append(torch.full((padding,), self.ignore_index, dtype=torch.long))
            loss_masks.append(torch.zeros(padding, dtype=torch.float32))

        packed_ids = torch.cat(input_ids)
        packed_labels = torch.cat(labels)
        packed_mask = torch.cat(loss_masks)
        if not torch.any(packed_mask):
            raise ValueError("completed prepared pack has no eligible next-token targets")
        return PackedPreparedDocument.derive_from(
            first_base,
            input_ids=packed_ids,
            labels=packed_labels,
            loss_mask=packed_mask,
            image_spans=image_spans,
            image_params=image_params,
            fragment_offsets=fragment_offsets,
            fragments=fragments,
            valid_length=valid_length,
            __sources__=tuple(source_infos) if source_infos else first_base.__sources__,
        )
