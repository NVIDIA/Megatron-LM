# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebDataset adapters for BAGEL's sample encoding and packing logic.

The BAGEL reference datasets read parquet or JSONL records directly. Energon
provides the same records as WebDataset dictionaries, so this module limits
itself to decoding those dictionaries and delegates the model-facing image
transforms, Edit sample construction, and sequence packing to BAGEL.
"""

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

from PIL import Image


def _load_bagel_dataset_base():
    """Load registry-backed BAGEL packing types after explicit data-root setup."""
    raw_path = os.environ.get("BAGEL_EXAMPLE_PATH")
    if not raw_path:
        raise RuntimeError(
            "BAGEL_EXAMPLE_PATH must name an existing directory before using "
            "DataConfig or BagelPacker"
        )
    data_path = Path(raw_path).expanduser()
    if not data_path.is_dir():
        raise RuntimeError(
            f"BAGEL_EXAMPLE_PATH must name an existing directory, got {raw_path!r}"
        )
    data_path = data_path.resolve()

    loaded_registry = sys.modules.get("bagel.data.dataset_info")
    loaded_raw_path = getattr(loaded_registry, "bagel_example_path", None)
    if loaded_raw_path is not None:
        loaded_path = Path(loaded_raw_path).expanduser().resolve()
        if loaded_path != data_path:
            raise RuntimeError(
                "BAGEL's dataset registry is already loaded for "
                f"{str(loaded_path)!r}; restart the process before changing "
                f"BAGEL_EXAMPLE_PATH to {str(data_path)!r}"
            )

    from bagel.data.dataset_base import DataConfig as ReferenceDataConfig
    from bagel.data.dataset_base import PackedDataset

    return ReferenceDataConfig, PackedDataset


class DataConfig:
    """Lazy constructor for BAGEL's reference ``DataConfig``."""

    def __new__(cls, *args, **kwargs):
        reference_data_config, _ = _load_bagel_dataset_base()
        return reference_data_config(*args, **kwargs)


class ImageTransform:
    """Lazy constructor for BAGEL's reference ``ImageTransform``."""

    def __new__(cls, *args, **kwargs):
        from bagel.data.transforms import ImageTransform as ReferenceImageTransform

        return ReferenceImageTransform(*args, **kwargs)


def _decode_json(value: Any) -> Dict[str, Any]:
    """Decode a WebDataset JSON field."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object, got {type(value).__name__}")
    return value


def _decode_text(value: Any) -> str:
    """Decode a WebDataset text field."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise TypeError(f"Expected text bytes or str, got {type(value).__name__}")
    return value


def _image_fields(sample: Dict[str, Any]) -> Iterable[tuple[str, bytes]]:
    """Yield image fields in WebDataset key order."""
    image_extensions = (".jpg", ".jpeg", ".png", ".webp")
    for key in sorted(sample):
        if key.lower() in {"jpg", "jpeg", "png", "webp"} or key.lower().endswith(
            image_extensions
        ):
            value = sample[key]
            if not isinstance(value, bytes):
                raise TypeError(f"Expected image bytes in field {key!r}")
            yield key, value


def _load_image(image_bytes: bytes) -> Image.Image:
    from bagel.data.data_utils import pil_img2rgb

    return pil_img2rgb(Image.open(io.BytesIO(image_bytes)))


class BagelT2ITaskEncoder:
    """Convert a ``{jpg, txt}`` WebDataset sample to BAGEL's packable form."""

    def __init__(self, tokenizer, vae_transform, vae_image_downsample=None):
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vae_image_downsample = vae_image_downsample or vae_transform.stride

    def encode_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        try:
            _, image_bytes = next(iter(_image_fields(sample)))
        except StopIteration as error:
            raise ValueError("T2I sample has no image field") from error

        if "txt" not in sample:
            raise ValueError("T2I sample has no txt field")

        image_tensor = self.vae_transform(_load_image(image_bytes))
        height, width = image_tensor.shape[1:]
        text_ids = self.tokenizer.encode(_decode_text(sample["txt"]))

        return {
            "image_tensor_list": [image_tensor],
            "text_ids_list": [text_ids],
            "sequence_plan": [
                {
                    "type": "text",
                    "enable_cfg": 1,
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
                {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
            ],
            "num_tokens": len(text_ids)
            + height * width // (self.vae_transform.stride**2),
            "data_indexes": {"dataset_name": "t2i"},
        }


class BagelEditTaskEncoder:
    """Decode an Edit WebDataset record and use BAGEL's Edit parser."""

    def __init__(self, tokenizer, vae_transform, vit_transform):
        from bagel.data.interleave_datasets.edit_dataset import UnifiedEditIterableDataset

        parser = object.__new__(UnifiedEditIterableDataset)
        parser.tokenizer = tokenizer
        parser.transform = vae_transform
        parser.vit_transform = vit_transform
        self._parser = parser

    def encode_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "json" not in sample:
            raise ValueError("Edit sample has no json field")
        metadata = _decode_json(sample["json"])

        images = [value for _, value in _image_fields(sample)]
        expected_num_images = metadata.get("num_images", len(images))
        if len(images) != expected_num_images:
            raise ValueError(
                f"Edit sample declares {expected_num_images} images but contains {len(images)}"
            )

        data = self._parser.parse_row(
            {
                "image_list": images,
                "instruction_list": metadata["instruction_list"],
            }
        )
        data["data_indexes"] = {"dataset_name": "edit"}
        return data


class BagelVLMTaskEncoder:
    """Convert a supported VLM WebDataset record to BAGEL's packable form."""

    def __init__(self, tokenizer, vit_transform, special_token_ids=None):
        from bagel.data.vlm_dataset import SftJSONLIterableDataset

        self.tokenizer = tokenizer
        self.vit_transform = vit_transform
        self.special_token_ids = special_token_ids or {}
        self._parser = object.__new__(SftJSONLIterableDataset)

    def encode_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "json" not in sample:
            raise ValueError("VLM sample has no json field")
        metadata = _decode_json(sample["json"])

        conversations = metadata["conversations"]
        for turn in conversations:
            role = turn["from"]
            if role not in {"human", "gpt"}:
                raise ValueError(f"Unsupported VLM conversation role: {role!r}")
            if role == "gpt" and "<image>" in turn["value"]:
                raise ValueError("VLM image placeholders are only supported in human turns")

        image_bytes_list = [value for _, value in _image_fields(sample) if value]
        placeholder_count = sum(
            turn["value"].count("<image>") for turn in conversations if turn["from"] == "human"
        )
        if placeholder_count != len(image_bytes_list):
            raise ValueError(
                "VLM image placeholder count must match supplied image count: "
                f"placeholders={placeholder_count}, images={len(image_bytes_list)}"
            )

        image_tensor_list = [
            self.vit_transform(_load_image(value), img_num=len(image_bytes_list))
            for value in image_bytes_list
        ]
        text_ids_list = []
        sequence_plan = []
        num_tokens = sum(
            tensor.shape[1] * tensor.shape[2] // (self.vit_transform.stride**2)
            for tensor in image_tensor_list
        )

        for element in self._parser.change_format(metadata, len(image_tensor_list)):
            if element["type"] == "text":
                text_ids = self.tokenizer.encode(element["text"])
                if not text_ids:
                    continue
                text_ids_list.append(text_ids)
                num_tokens += len(text_ids)
                sequence_plan.append(
                    {
                        "type": "text",
                        "enable_cfg": 0,
                        "loss": element["has_loss"],
                        "special_token_loss": 0,
                        "special_token_label": None,
                    }
                )
            elif element["type"] == "image":
                sequence_plan.append(
                    {
                        "type": "vit_image",
                        "enable_cfg": 0,
                        "loss": 0,
                        "special_token_loss": 0,
                        "special_token_label": None,
                    }
                )

        if not any(item["loss"] for item in sequence_plan):
            raise ValueError("VLM sample must contain at least one loss-bearing response")

        return {
            "image_tensor_list": image_tensor_list,
            "text_ids_list": text_ids_list,
            "sequence_plan": sequence_plan,
            "num_tokens": num_tokens,
            "data_indexes": {"dataset_name": "vlm"},
        }


class BagelPacker:
    """Expose BAGEL's packing methods without initializing dataset I/O."""

    def __init__(
        self,
        data_config,
        special_token_ids,
        max_num_tokens,
        *,
        interpolate_pos=False,
        use_flex=False,
    ):
        _, self._packed_dataset_type = _load_bagel_dataset_base()
        from bagel.data.data_utils import (
            get_flattened_position_ids_extrapolate,
            get_flattened_position_ids_interpolate,
        )

        self.data_config = data_config
        self.max_num_tokens = max_num_tokens
        self.use_flex = use_flex
        for name, value in special_token_ids.items():
            setattr(self, name, value)
        if interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

    def init_sequence_status(self):
        """Use the name expected by standalone task-encoder tests."""
        return self._packed_dataset_type.set_sequence_status(self)

    def pack_sequence(self, sample, sequence_status):
        return self._packed_dataset_type.pack_sequence(self, sample, sequence_status)

    def to_tensor(self, sequence_status):
        return self._packed_dataset_type.to_tensor(self, sequence_status)


__all__ = [
    "BagelEditTaskEncoder",
    "BagelPacker",
    "BagelT2ITaskEncoder",
    "BagelVLMTaskEncoder",
    "DataConfig",
    "ImageTransform",
]
