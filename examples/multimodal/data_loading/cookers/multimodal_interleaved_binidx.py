# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Join native BinIdx text with co-indexed, lazy image annotations."""

import numpy as np

from megatron.energon import CachePool, FileStore, basic_sample_keys, cooker, stateless
from megatron.energon.flavors.binidx.token_mask import expand_token_mask

from ..binidx_token_mask import token_mask_key
from ..conversation_sample import ImageMedia
from ..multimodal_interleaved_samples import InterleavedDocument, TokenizedText


@stateless
@cooker(need_cache=True)
def cook_multimodal_interleaved_binidx(
    sample: dict,
    cache: CachePool,
    metadata: FileStore | None = None,
    images: FileStore | None = None,
    token_loss_mask: FileStore | None = None,
) -> InterleavedDocument:
    """Insert images at original text-token boundaries, without consuming text.

    Companion rows have ``images`` entries with ``position``, ``width``, ``height``
    and an opaque auxiliary-store ``key``. Positions include any source BOS.
    Equal positions preserve image order. The final token must remain text,
    retaining the terminal EOS. ``drop_bos_token_id`` explicitly verifies/removes
    a source BOS; images before that BOS remain at the document start.

    Optional ``token_loss_mask`` is a co-indexed companion selected by the primary
    source's original document index, including under subsets and shuffling.
    It supplies one ownership flag per original source token: True/1 is eligible
    for target loss; False/0 is excluded. Flags align before BOS removal, image
    insertion and next-token label shifting, including source BOS/EOS. Ownership
    is sliced with IDs when removing BOS or splitting text around images; image
    slots consume no source text ownership. Masked IDs remain visible as inputs.
    The encoder uses the target token's ownership, subject to existing image-input,
    image-target and padding exclusions. Without a companion, the current full-text
    objective and those exclusions are unchanged. Roles and delimiters are not
    inferred from prepared IDs.

    See "Optional prepared-token supervision masks" in
    ``../MULTIMODAL_INTERLEAVED.md`` for recipes and the binary v2 format.
    """
    if "tokens" not in sample:
        raise ValueError(
            "cook: multimodal_interleaved_binidx requires native .bin/.idx token input; "
            "use the default interleaved cooker for JSONL"
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
    # This is a companion annotation of the primary source, not another packed
    # data source. Its immutable path is captured by the resolved recipe.
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
