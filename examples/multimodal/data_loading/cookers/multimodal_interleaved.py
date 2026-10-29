# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent interleaved JSONL schema mapping with no sampling or image-transform policy."""

import json
from pathlib import PurePosixPath

from megatron.energon import CachePool, FileStore, basic_sample_keys, cooker, stateless

from ..conversation_sample import ImageMedia
from ..multimodal_interleaved_samples import InterleavedDocument


def _range_key(sample: dict, image: dict) -> str:
    """Describe a document-relative range; resolution belongs to the image store."""
    sources = sample.get("__sources__") or ()
    if not sources:
        raise ValueError("document-relative image needs JSONL source identity")
    source = sources[0]
    shard = PurePosixPath(str(source.shard_name or source.dataset_path))
    names = source.file_names or ()
    if not names or not PurePosixPath(names[0]).stem.isdecimal():
        raise ValueError("document-relative image needs its shard-local row index")
    extent = image["doc_tar_offset"]
    return json.dumps(
        {
            "path": str(shard.with_suffix(".tar")),
            "document_index": int(PurePosixPath(names[0]).stem),
            "offset": extent["offset"],
            "size": extent["size"],
            "tar_member": extent.get("new_tar_format", True),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@stateless
@cooker(need_cache=True)
def cook_multimodal_interleaved(
    sample: dict, cache: CachePool, images: FileStore | None = None
) -> InterleavedDocument:
    """Map plain text, repository text or ordered text/image JSONL records.

    Images require explicit dimensions and an auxiliary store. No image is
    silently dropped, resized by corpus metadata, or loaded while cooking.
    """
    if "json" not in sample:
        raise ValueError(
            "interleaved JSONL requires a decoded json record; BinIdx input requires "
            "recipe tags: {cook: multimodal_interleaved_binidx} and aux.metadata"
        )
    row = sample["json"]
    if "parts" in row:
        content = row["parts"]
    elif "multimodal_document" in row:
        content = row["multimodal_document"]
        if isinstance(content, dict):
            content = content["content"]
    elif "repo_texts" in row:
        files = []
        for item in row["repo_texts"]:
            if isinstance(item, str):
                files.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                name, text = item
                if not isinstance(name, str) or not isinstance(text, str):
                    raise ValueError("repository text entries require string names and content")
                files.append(name + "\n" + text)
            else:
                raise ValueError("invalid repository text entry")
        content = ["\n\n".join(files)]
    elif "text" in row:
        content = [row["text"]]
    else:
        raise ValueError(
            "interleaved document requires parts, multimodal_document, repo_texts or text"
        )
    if not isinstance(content, list):
        raise ValueError("interleaved document content must be an ordered list")
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError("interleaved document parts must be strings or image references")
        if images is None:
            raise ValueError("interleaved image reference requires an aux.images file store")
        width, height = item["width"], item["height"]
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError("image width/height must be positive integers")
        key = item.get("key")
        if key is None:
            key = _range_key(sample, item)
        if not isinstance(key, str) or not key:
            raise ValueError("image key must be a nonempty string")
        parts.append(
            ImageMedia(
                value=cache.get_lazy(images, key), metadata={"width": width, "height": height}
            )
        )
    return InterleavedDocument(**basic_sample_keys(sample), parts=parts)
