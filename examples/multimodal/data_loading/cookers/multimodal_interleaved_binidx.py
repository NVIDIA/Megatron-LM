# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Join native BinIdx tokens with co-indexed, lazy image annotations."""

import numpy as np

from megatron.energon import CachePool, FileStore, basic_sample_keys, cooker, stateless

from ..binidx_token_mask import token_mask_key
from ..conversation_sample import ImageMedia
from ..prepared_sample import InterleavedDocument, TokenizedText
from ..token_mask import expand_token_mask


@stateless
@cooker(need_cache=True)
def cook_multimodal_interleaved_binidx(
    sample: dict,
    cache: CachePool,
    metadata: FileStore | None = None,
    images: FileStore | None = None,
    token_loss_mask: FileStore | None = None,
) -> InterleavedDocument:
    """Insert lazy images at original text-token boundaries without consuming text.

    Register this function explicitly in an Energon ``Cooker``. ``metadata`` is
    decoded JSON indexed by the primary source's original record number, using
    keys such as ``"7.json"``. Each row contains an ``images`` list with integer
    ``position``, ``width`` and ``height`` fields and an opaque image-store ``key``.
    Positions include any source BOS, must be ordered, and must precede the final
    text token. Equal positions preserve image order. Pixels remain lazy.

    The boolean ``text_only`` source tag permits omitting both image auxiliaries.
    By default all source IDs are retained. ``drop_bos_token_id`` explicitly
    verifies and removes one source BOS; images before it stay at the start.
    Tokenization, resizing, label shifting and packing belong to the consumer.

    Optional ``token_loss_mask`` supplies one eligibility flag per original token.
    It is selected by the primary source path and original record index, before
    BOS removal or image insertion. Flags are sliced with text IDs; masked IDs
    remain visible as inputs. No roles or loss policy are inferred here.
    """
    if "tokens" not in sample:
        raise ValueError(
            "cook: multimodal_interleaved_binidx requires native .bin/.idx token input"
        )
    tokens = sample["tokens"]
    if (
        not isinstance(tokens, np.ndarray)
        or tokens.ndim != 1
        or tokens.dtype.kind not in "iu"
        or len(tokens) == 0
        or np.any(tokens < 0)
    ):
        raise ValueError("BinIdx text must be a nonempty one-dimensional integer token array")
    sources = sample.get("__sources__") or ()
    if len(sources) != 1 or type(sources[0].index) is not int or sources[0].index < 0:
        raise ValueError("BinIdx text requires one primary source with its original document index")
    tags = sample.get("__tags__") or {}
    text_only = tags.get("text_only", False)
    if type(text_only) is not bool:
        raise ValueError("BinIdx text_only tag must be a boolean")
    if text_only and (metadata is not None or images is not None):
        raise ValueError("BinIdx text_only cannot be combined with image auxiliaries")
    if metadata is None and not text_only:
        raise ValueError("interleaved BinIdx requires co-indexed aux.metadata")
    row = {"images": []} if text_only else cache.get(metadata, f"{sources[0].index}.json")
    if not isinstance(row, dict) or not isinstance(row.get("images"), list):
        raise ValueError("BinIdx companion metadata requires an images list")
    ownership = None
    if token_loss_mask is not None:
        data = cache.get(token_loss_mask, token_mask_key(sources[0].dataset_path, sources[0].index))
        ownership = expand_token_mask(data, len(tokens))
    drop_bos = tags.get("drop_bos_token_id")
    cursor = 0
    if drop_bos is not None:
        if type(drop_bos) is not int or drop_bos < 0 or tokens[0] != drop_bos:
            raise ValueError("configured source BOS does not match the prepared token document")
        cursor = 1
    parts = []
    previous = 0
    for entry in row["images"]:
        position = entry["position"]
        if type(position) is not int or not previous <= position < len(tokens):
            raise ValueError(
                "image insertion positions must be ordered and precede the final token"
            )
        previous = position
        if images is None:
            raise ValueError("interleaved BinIdx images require aux.images")
        width, height, key = entry["width"], entry["height"], entry["key"]
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError("image width/height must be positive integers")
        if not isinstance(key, str) or not key:
            raise ValueError("image key must be a nonempty string")
        end = max(cursor, position)
        if end > cursor:
            parts.append(
                TokenizedText(
                    tokens=tokens[cursor:end].tolist(),
                    token_loss_mask=None if ownership is None else ownership[cursor:end],
                )
            )
        parts.append(
            ImageMedia(
                value=cache.get_lazy(images, key), metadata={"width": width, "height": height}
            )
        )
        cursor = end
    if cursor < len(tokens):
        parts.append(
            TokenizedText(
                tokens=tokens[cursor:].tolist(),
                token_loss_mask=None if ownership is None else ownership[cursor:],
            )
        )
    return InterleavedDocument(**basic_sample_keys(sample), parts=parts)
