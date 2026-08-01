# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Extract media from OpenAI-style chat content blocks.

The chat endpoint currently flattens multimodal content to text and drops the media, because
HF/Jinja chat templates want a plain string. This module splits that into two results instead:
the flattened text, with each media block replaced by the model's placeholder marker, and the
decoded media itself.

Kept model-agnostic -- the caller supplies the marker strings -- so a second multimodal model
reuses it without change.
"""

import base64
import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from megatron.core.inference.multimodal.types import MultimodalData

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]+)?(?P<b64>;base64)?,(?P<payload>.*)$", re.S)


@dataclass
class ExtractedContent:
    """Flattened prompt text plus the media it refers to."""

    text: str
    multimodal_data: MultimodalData = field(default_factory=MultimodalData)

    def is_multimodal(self) -> bool:
        """Whether any media was extracted."""
        return not self.multimodal_data.is_empty()


def _decode_data_url(url: str) -> bytes:
    """Decode a `data:` URL payload to raw bytes."""
    match = _DATA_URL_RE.match(url)
    if match is None:
        raise ValueError(
            "only inline data: URLs are supported for media; fetching remote URLs is the "
            "caller's responsibility"
        )
    payload = match.group("payload")
    if match.group("b64"):
        return base64.b64decode(payload)
    return payload.encode()


def _load_image(url: str) -> Any:
    """Decode an image `data:` URL into a PIL image."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("Pillow is required to decode image content blocks") from exc
    return Image.open(io.BytesIO(_decode_data_url(url)))


def _load_audio(block: Dict[str, Any], target_sample_rate: int) -> Tuple[Any, int]:
    """Decode an `input_audio` block into a `(waveform, sample_rate)` pair.

    Resampling happens here rather than in the audio front-end, which asserts the rate instead:
    a silent resample deep in the mel pipeline would change token counts without any signal at
    the API boundary.
    """
    try:
        import numpy as np
        import soundfile
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "soundfile and numpy are required to decode audio content blocks"
        ) from exc

    audio = block.get("input_audio", block)
    data = audio.get("data")
    assert data is not None, "input_audio block has no 'data' field"
    raw = base64.b64decode(data) if isinstance(data, str) else data

    waveform, sample_rate = soundfile.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)

    if sample_rate != target_sample_rate:
        # Linear resample. Adequate for speech at these rates; swap in a polyphase filter if
        # transcription quality on downsampled input turns out to matter.
        duration = waveform.shape[0] / sample_rate
        num_target = int(round(duration * target_sample_rate))
        source_positions = np.linspace(0, waveform.shape[0] - 1, num=num_target)
        waveform = np.interp(source_positions, np.arange(waveform.shape[0]), waveform).astype(
            "float32"
        )
        sample_rate = target_sample_rate

    return waveform, sample_rate


def extract_media_from_content(
    content: Any,
    *,
    image_marker: str,
    video_marker: str,
    audio_marker: str,
    audio_sample_rate: int = 16000,
) -> ExtractedContent:
    """Flatten one message's content, substituting placeholder markers for media.

    Args:
        content (Any): A string, a dict, or a list of OpenAI content blocks.
        image_marker (str): Marker to substitute for each image, e.g. `"<image>"`.
        video_marker (str): Marker to substitute for each video.
        audio_marker (str): Marker to substitute for each audio clip.
        audio_sample_rate (int): Rate to resample audio to.

    Return:
        (ExtractedContent) Flattened text and the decoded media, in prompt order.
    """
    if isinstance(content, str):
        return ExtractedContent(text=content)
    if isinstance(content, dict):
        return ExtractedContent(text=str(content.get("text", "")))
    if not isinstance(content, (list, tuple)):
        return ExtractedContent(text="" if content is None else str(content))

    data = MultimodalData()
    parts: List[str] = []

    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text" or (block_type is None and "text" in block):
            parts.append(str(block.get("text", "")))
        elif block_type == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            data.images.append(_load_image(url))
            parts.append(image_marker)
        elif block_type == "video_url":
            url = (block.get("video_url") or {}).get("url", "")
            # Videos need frame sampling and an fps, which the decoder here does not do.
            # Pass the raw bytes through and let the provider's video path own decoding.
            data.videos.append({"bytes": _decode_data_url(url)})
            parts.append(video_marker)
        elif block_type == "input_audio":
            data.audios.append(_load_audio(block, audio_sample_rate))
            parts.append(audio_marker)

    return ExtractedContent(text="".join(parts), multimodal_data=data)


def extract_media_from_messages(
    messages: Sequence[Dict[str, Any]],
    *,
    image_marker: str,
    video_marker: str,
    audio_marker: str,
    audio_sample_rate: int = 16000,
) -> Tuple[List[Dict[str, Any]], MultimodalData]:
    """Flatten a whole conversation, accumulating media across messages.

    Media order follows message order, then block order within a message, which is the order
    the placeholder markers appear in the rendered prompt. The provider pairs the two by
    position, so this ordering is the contract.

    Args:
        messages (Sequence[Dict[str, Any]]): Chat messages.
        image_marker (str): Marker to substitute for each image.
        video_marker (str): Marker to substitute for each video.
        audio_marker (str): Marker to substitute for each audio clip.
        audio_sample_rate (int): Rate to resample audio to.

    Return:
        (Tuple[List[Dict[str, Any]], MultimodalData]) Messages with string content, and the
        accumulated media.
    """
    combined = MultimodalData()
    flattened: List[Dict[str, Any]] = []

    for message in messages:
        if not isinstance(message, dict):
            flattened.append(message)
            continue
        extracted = extract_media_from_content(
            message.get("content"),
            image_marker=image_marker,
            video_marker=video_marker,
            audio_marker=audio_marker,
            audio_sample_rate=audio_sample_rate,
        )
        message_copy = dict(message)
        message_copy["content"] = extracted.text
        flattened.append(message_copy)

        combined.images.extend(extracted.multimodal_data.images)
        combined.videos.extend(extracted.multimodal_data.videos)
        combined.audios.extend(extracted.multimodal_data.audios)

    return flattened, combined
