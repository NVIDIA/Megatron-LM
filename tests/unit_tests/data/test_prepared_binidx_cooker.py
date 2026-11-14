# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU contracts for cooking prepared IDs and lazy image companions."""

import json
import struct

import numpy as np
import pytest
from PIL import Image

from examples.multimodal.data_loading.conversation_sample import ImageMedia
from examples.multimodal.data_loading.cookers.multimodal_interleaved_binidx import (
    cook_multimodal_interleaved_binidx,
)
from examples.multimodal.data_loading.prepared_sample import TokenizedText
from megatron.energon import (
    Cooker,
    DefaultTaskEncoder,
    FileStore,
    NoCachePool,
    WorkerConfig,
    get_loader,
    get_train_dataset,
)
from megatron.energon.cache.file_store import SystemFileStore
from megatron.energon.decoders.sample_decoder import SampleDecoder
from megatron.energon.epathlib import EPath
from megatron.energon.source_info import SourceInfo


class _MemoryStore(FileStore):
    def __init__(self, values):
        self.values = values
        self.read_keys = []

    def __getitem__(self, key):
        self.read_keys.append(key)
        return self.values[key], SourceInfo(
            dataset_path=EPath("/synthetic"), index=0, shard_name="memory", file_names=(key,)
        )

    def get_path(self):
        return "synthetic"


class _PreparedEncoder(DefaultTaskEncoder):
    cookers = (
        Cooker(
            cook=cook_multimodal_interleaved_binidx,
            has_tags={"cook": "multimodal_interleaved_binidx"},
        ),
    )
    decoder = SampleDecoder(image_decode="pil")


@pytest.fixture
def sample(tmp_path):
    return {
        "tokens": np.array([3, 11, 12, 1], dtype=np.int32),
        "__key__": "subset-output-zero",
        "__restore_key__": ("source", 7),
        "__sources__": (
            SourceInfo(
                dataset_path=EPath(tmp_path / "tokens.bin"),
                index=7,
                shard_name="tokens.bin",
                file_names=("7.tokens",),
            ),
        ),
        "__tags__": {},
    }


def _text_ids(document):
    return [
        token for part in document.parts if isinstance(part, TokenizedText) for token in part.tokens
    ]


def _metadata(positions=()):
    return _MemoryStore(
        {
            "7.json": {
                "images": [
                    {"position": p, "width": 8, "height": 4, "key": f"image-{i}.png"}
                    for i, p in enumerate(positions)
                ]
            }
        }
    )


def test_prepared_ids_and_original_identity_are_preserved(sample):
    metadata = _metadata()
    document = cook_multimodal_interleaved_binidx(sample, NoCachePool(), metadata=metadata)
    assert _text_ids(document) == [3, 11, 12, 1]
    assert metadata.read_keys == ["7.json"]
    assert document.__key__ == sample["__key__"]
    assert document.__restore_key__ == sample["__restore_key__"]
    assert document.__sources__ == sample["__sources__"]
    assert sample["tokens"].tolist() == [3, 11, 12, 1]


@pytest.mark.parametrize("drop_bos", [False, True])
def test_equal_image_positions_are_ordered_and_pixels_stay_lazy(sample, drop_bos):
    sample["__tags__"] = {"drop_bos_token_id": 3} if drop_bos else {}
    images = _MemoryStore({f"image-{i}.png": Image.new("RGB", (8, 4), "red") for i in range(4)})
    document = cook_multimodal_interleaved_binidx(
        sample, NoCachePool(), metadata=_metadata((0, 1, 1, 3)), images=images
    )
    assert _text_ids(document) == ([11, 12, 1] if drop_bos else [3, 11, 12, 1])
    image_parts = [p for p in document.parts if isinstance(p, ImageMedia)]
    assert [p.value.fname for p in image_parts] == [f"image-{i}.png" for i in range(4)]
    assert images.read_keys == []
    assert all((p.width, p.height) == (8, 4) for p in image_parts)
    assert image_parts[0].value.get().getpixel((0, 0)) == (255, 0, 0)
    assert images.read_keys == ["image-0.png"]


def test_text_only_is_explicit_and_bos_removal_is_checked(sample):
    with pytest.raises(ValueError, match="aux.metadata"):
        cook_multimodal_interleaved_binidx(sample, NoCachePool())
    sample["__tags__"] = {"text_only": True}
    assert _text_ids(cook_multimodal_interleaved_binidx(sample, NoCachePool())) == [3, 11, 12, 1]
    with pytest.raises(ValueError, match="image auxiliaries"):
        cook_multimodal_interleaved_binidx(sample, NoCachePool(), metadata=_metadata())
    sample["__tags__"] = {"text_only": True, "drop_bos_token_id": 4}
    with pytest.raises(ValueError, match="BOS"):
        cook_multimodal_interleaved_binidx(sample, NoCachePool())


@pytest.mark.parametrize(
    "tokens",
    [np.array([]), np.array([3.0, 1.0]), np.array([True]), np.array([[3, 1]]), np.array([-1, 1])],
)
def test_invalid_prepared_ids_are_rejected(sample, tokens):
    sample["tokens"] = tokens
    with pytest.raises(ValueError, match="integer token array"):
        cook_multimodal_interleaved_binidx(sample, NoCachePool(), metadata=_metadata())


@pytest.mark.parametrize("positions", [(2, 1), (-1,), (4,), (True,)])
def test_invalid_image_boundaries_are_rejected(sample, positions):
    with pytest.raises(ValueError, match="insertion positions"):
        cook_multimodal_interleaved_binidx(
            sample, NoCachePool(), metadata=_metadata(positions), images=_MemoryStore({})
        )


@pytest.mark.parametrize("sources", [(), None])
def test_original_source_identity_is_required(sample, sources):
    sample["__sources__"] = sources
    with pytest.raises(ValueError, match="original document index"):
        cook_multimodal_interleaved_binidx(sample, NoCachePool(), metadata=_metadata())


def _write_native_recipe(root, *, drop_bos=False):
    documents = [[3, 10, 1], [3, 20, 21, 22, 1], [3, 30, 1]]
    sizes = np.array([len(ids) for ids in documents], dtype="<i4")
    np.concatenate([np.array(ids, dtype="<i4") for ids in documents]).tofile(root / "tokens.bin")
    offsets = np.concatenate(([0], np.cumsum(sizes[:-1], dtype=np.int64))) * 4
    with (root / "tokens.idx").open("wb") as stream:
        stream.write(b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 4, 3, 3))
        stream.write(sizes.tobytes())
        stream.write(offsets.astype("<i8").tobytes())
        stream.write(np.arange(3, dtype="<i8").tobytes())
    annotations = root / "annotations"
    annotations.mkdir()
    for index in range(3):
        images = (
            [{"position": 3, "width": 8, "height": 4, "key": "screen.png"}] if index == 1 else []
        )
        (annotations / f"{index}.json").write_text(json.dumps({"images": images}))
    image_root = root / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 4), "blue").save(image_root / "screen.png")
    bos_tag = "      drop_bos_token_id: 3\n" if drop_bos else ""
    recipe = root / "recipe.yaml"
    recipe.write_text(
        "__module__: megatron.energon\n__class__: Recipe\nsplits:\n  train:\n"
        "    path: tokens.bin\n    subset:\n      range: [1, 2]\n"
        "    tags:\n      cook: multimodal_interleaved_binidx\n"
        + bos_tag
        + "    aux:\n      metadata: filesystem://annotations\n      images: filesystem://images\n"
    )
    return recipe, image_root


@pytest.mark.parametrize("drop_bos", [False, True])
def test_native_recipe_loads_original_subset_record_with_lazy_image(
    tmp_path, monkeypatch, drop_bos
):
    recipe, image_root = _write_native_recipe(tmp_path, drop_bos=drop_bos)
    image_reads = []
    original_get = SystemFileStore.__getitem__

    def tracked_get(store, key):
        if str(store.get_path()) == str(image_root):
            image_reads.append(key)
        return original_get(store, key)

    monkeypatch.setattr(SystemFileStore, "__getitem__", tracked_get)
    dataset = get_train_dataset(
        recipe,
        worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0),
        task_encoder=_PreparedEncoder(),
        batch_size=None,
        packing_buffer_size=None,
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        repeat=False,
    )
    loader = get_loader(dataset, cache_pool=NoCachePool())
    try:
        documents = list(loader)
        assert len(documents) == 1
        document = documents[0]
        assert document.__sources__[0].index == 1
        assert _text_ids(document) == ([20, 21, 22, 1] if drop_bos else [3, 20, 21, 22, 1])
        assert image_reads == []
        image = next(p for p in document.parts if isinstance(p, ImageMedia))
        assert image.value.get().getpixel((0, 0)) == (0, 0, 255)
        assert image_reads == ["screen.png"]
    finally:
        loader.close()
