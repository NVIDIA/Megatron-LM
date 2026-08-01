# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Placeholder expansion for Nemotron Omni prompts.

This module owns the token-count contract: the number of `<image>` / `<so_embedding>` slots it
reserves must equal, exactly, the number of embedding rows the encoders produce. A mismatch is
not a graceful degradation -- it shifts every subsequent position, so the language model reads
image rows at text positions and vice versa. Both sides therefore derive their counts from the
same functions here, and `NemotronOmniEncoderStack` asserts the equality.

Two rules keep tokenization stable:

1. Every span component is tokenized independently -- the separator, `<img>`, each `<image>`,
   and `</img>` -- and the id lists are concatenated. Tokenizing the joined string instead
   would let the tokenizer merge across boundaries and silently change the count depending on
   how many `<image>` repetitions happened to be adjacent.
2. Frame timestamps use integer millisecond arithmetic (`int(1000.0 / fps)`). A float frame
   duration makes the rendered timestamp string differ between the host count and the device
   re-render, which changes the separator token count and produces a shape mismatch.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from megatron.core.inference.multimodal.nemotron_omni.config import NemotronOmniConfig


@dataclass
class MediaSpan:
    """One expanded placeholder span within the prompt.

    Attributes:
        modality: "image", "video", or "audio".
        item_index: Index of the media item within its modality list.
        embed_positions: Absolute positions in the expanded prompt that receive encoder rows.
    """

    modality: str
    item_index: int
    embed_positions: List[int] = field(default_factory=list)


@dataclass
class ExpandedPrompt:
    """Result of placeholder expansion."""

    token_ids: List[int]
    spans: List[MediaSpan]

    @property
    def embed_positions(self) -> List[int]:
        """All encoder-row positions, ascending.

        Spans are appended in prompt order and positions within a span are ascending, so a
        plain concatenation is already sorted. The context relies on that ordering to bisect
        for a prefill chunk's slice.
        """
        return [pos for span in self.spans for pos in span.embed_positions]


def calculate_timestamps(frame_indices: Sequence[int], frame_duration_ms: int) -> List[float]:
    """Frame timestamps in seconds.

    `frame_duration_ms` must already be the integer `int(1000.0 / fps)`; passing a float here
    reintroduces the host/device string mismatch this function exists to avoid.
    """
    return [int(i) * frame_duration_ms / 1000.0 for i in frame_indices]


def video_frame_separators(
    frame_indices: Sequence[int], frame_duration_ms: Optional[int], temporal_patch_size: int
) -> List[str]:
    """Build the human-readable separator that precedes each video tubelet.

    Tubelets group `temporal_patch_size` consecutive frames, and the separator names every
    frame in the group: "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.50 seconds: ".
    The first frame of a group is capitalized and later ones are not, and every group after the
    first is prefixed with a newline -- both match the training format.

    Args:
        frame_indices (Sequence[int]): Original frame indices, before tubelet grouping.
        frame_duration_ms (Optional[int]): Integer ms per frame, or None to omit timestamps.
        temporal_patch_size (int): Frames per tubelet.

    Return:
        (List[str]) One separator per tubelet.
    """
    num_frames = len(frame_indices)
    step = max(temporal_patch_size, 1)

    if frame_duration_ms is None:
        return [
            ("\n" if group_idx > 0 else "") + f"Frame {group_idx + 1}: "
            for group_idx in range(0, (num_frames + step - 1) // step)
        ]

    timestamps = calculate_timestamps(frame_indices, frame_duration_ms)
    separators = []
    for group_idx, start in enumerate(range(0, num_frames, step)):
        parts = []
        for offset in range(step):
            frame_idx = start + offset
            if frame_idx >= num_frames:
                # Padding frames (the tail tubelet repeats the last frame) are not named.
                break
            label = "Frame" if offset == 0 else "frame"
            parts.append(f"{label} {frame_idx + 1} sampled at {timestamps[frame_idx]:.2f} seconds")
        if parts:
            separator = " and ".join(parts) + ": "
            separators.append(("\n" if group_idx > 0 else "") + separator)
    return separators


class NemotronOmniPromptExpander:
    """Expands media placeholders in a prompt into concrete token spans."""

    def __init__(self, tokenizer: Any, config: NemotronOmniConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config

        self._img_start = self._encode(config.img_start_token)
        self._img_end = self._encode(config.img_end_token)
        self._img_context = self._encode(config.img_context_token)
        self._audio_start = self._encode(config.audio_start_token)
        self._audio_end = self._encode(config.audio_end_token)
        self._audio_context = self._encode(config.audio_context_token)

        assert (
            len(self._img_context) == 1
        ), f"{config.img_context_token!r} must be a single token, got {self._img_context}"
        assert (
            len(self._audio_context) == 1
        ), f"{config.audio_context_token!r} must be a single token, got {self._audio_context}"

    def _encode(self, text: str) -> List[int]:
        """Tokenize without special-token insertion."""
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            # Megatron tokenizers do not accept add_special_tokens.
            return self.tokenizer.tokenize(text)

    def _image_span(
        self, num_tokens: int, start: int, item_index: int
    ) -> Tuple[List[int], MediaSpan]:
        """`<img>` + `<image>` * num_tokens + `</img>`, with the context positions recorded."""
        token_ids = list(self._img_start)
        offset = start + len(token_ids)
        token_ids.extend(self._img_context * num_tokens)
        span = MediaSpan(
            modality="image",
            item_index=item_index,
            embed_positions=list(range(offset, offset + num_tokens)),
        )
        token_ids.extend(self._img_end)
        return token_ids, span

    def _audio_span(
        self, num_tokens: int, start: int, item_index: int
    ) -> Tuple[List[int], MediaSpan]:
        """`<so_start>` + `<so_embedding>` * num_tokens + `<so_end>`."""
        token_ids = list(self._audio_start)
        offset = start + len(token_ids)
        token_ids.extend(self._audio_context * num_tokens)
        span = MediaSpan(
            modality="audio",
            item_index=item_index,
            embed_positions=list(range(offset, offset + num_tokens)),
        )
        token_ids.extend(self._audio_end)
        return token_ids, span

    def _video_span(
        self,
        tokens_per_tubelet: Sequence[int],
        separators: Sequence[str],
        start: int,
        item_index: int,
    ) -> Tuple[List[int], MediaSpan]:
        """Separator + image span, repeated once per tubelet.

        The separators are ordinary text and are embedded through the language model's own
        table; only the `<image>` runs receive encoder rows.
        """
        assert len(separators) == len(
            tokens_per_tubelet
        ), f"{len(separators)} separators for {len(tokens_per_tubelet)} tubelets"
        # Batch-encode: the fast tokenizers' Rust backend parallelizes across the batch.
        try:
            encoded = self.tokenizer(
                list(separators), add_special_tokens=False, return_attention_mask=False
            )["input_ids"]
        except (TypeError, KeyError):
            encoded = [self._encode(sep) for sep in separators]

        token_ids: List[int] = []
        span = MediaSpan(modality="video", item_index=item_index)
        for separator_ids, num_tokens in zip(encoded, tokens_per_tubelet):
            token_ids.extend(separator_ids)
            token_ids.extend(self._img_start)
            offset = start + len(token_ids)
            token_ids.extend(self._img_context * num_tokens)
            span.embed_positions.extend(range(offset, offset + num_tokens))
            token_ids.extend(self._img_end)
        return token_ids, span

    def expand(
        self,
        prompt: str,
        image_token_counts: Sequence[int] = (),
        audio_token_counts: Sequence[int] = (),
        video_plans: Sequence[Dict[str, Any]] = (),
        add_bos: bool = False,
    ) -> ExpandedPrompt:
        """Replace each placeholder marker with its expanded span.

        Markers are consumed left to right across all modalities at once, so interleaved
        prompts ("<image> then <video> then <image>") keep their positional order.

        Args:
            prompt (str): Prompt text containing the placeholder markers.
            image_token_counts (Sequence[int]): Post-shuffle token count per image, in order.
            audio_token_counts (Sequence[int]): Encoder row count per audio item, in order.
            video_plans (Sequence[Dict[str, Any]]): Per video, a dict with
                `tokens_per_tubelet` and `separators`.
            add_bos (bool): Whether to prepend the tokenizer's BOS id.

        Return:
            (ExpandedPrompt) Expanded token ids plus the span metadata.
        """
        markers = {
            self.config.img_context_token: "image",
            self.config.video_token: "video",
            self.config.audio_context_token: "audio",
        }

        token_ids: List[int] = []
        if add_bos:
            token_ids.append(self.tokenizer.bos)

        spans: List[MediaSpan] = []
        counters = {"image": 0, "video": 0, "audio": 0}
        cursor = 0

        while cursor < len(prompt):
            # Find the earliest marker of any modality.
            next_pos = len(prompt)
            next_marker = None
            for marker in markers:
                found = prompt.find(marker, cursor)
                if found != -1 and found < next_pos:
                    next_pos, next_marker = found, marker

            if next_marker is None:
                token_ids.extend(self._encode(prompt[cursor:]))
                break

            if next_pos > cursor:
                token_ids.extend(self._encode(prompt[cursor:next_pos]))

            modality = markers[next_marker]
            index = counters[modality]
            counters[modality] += 1

            if modality == "image":
                assert index < len(image_token_counts), (
                    f"prompt has more {next_marker!r} markers than images "
                    f"({len(image_token_counts)})"
                )
                span_ids, span = self._image_span(image_token_counts[index], len(token_ids), index)
            elif modality == "audio":
                assert index < len(audio_token_counts), (
                    f"prompt has more {next_marker!r} markers than audio items "
                    f"({len(audio_token_counts)})"
                )
                span_ids, span = self._audio_span(audio_token_counts[index], len(token_ids), index)
            else:
                assert index < len(video_plans), (
                    f"prompt has more {next_marker!r} markers than videos " f"({len(video_plans)})"
                )
                plan = video_plans[index]
                span_ids, span = self._video_span(
                    plan["tokens_per_tubelet"], plan["separators"], len(token_ids), index
                )

            token_ids.extend(span_ids)
            spans.append(span)
            cursor = next_pos + len(next_marker)

        for modality, expected in (
            ("image", len(image_token_counts)),
            ("audio", len(audio_token_counts)),
            ("video", len(video_plans)),
        ):
            assert counters[modality] == expected, (
                f"{expected} {modality} item(s) supplied but the prompt has "
                f"{counters[modality]} placeholder(s)"
            )

        return ExpandedPrompt(token_ids=token_ids, spans=spans)
