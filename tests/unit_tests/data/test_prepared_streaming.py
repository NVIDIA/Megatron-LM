# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Image-boundary and native restore contracts for prepared streaming packs."""

import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from examples.multimodal.data_loading.conversation_sample import ImageMedia
from examples.multimodal.data_loading.image_processing import ImageTilingParams
from examples.multimodal.data_loading.prepared_sample import InterleavedDocument, TokenizedText
from examples.multimodal.data_loading.prepared_streaming import (
    PackedPreparedDocument,
    PreparedStreamingTaskEncoder,
)
from examples.multimodal.data_loading.prepared_task_encoder import EncodedPreparedDocument
from megatron.energon import (
    NoCachePool,
    PackedSamplesOutput,
    PartialSample,
    WorkerConfig,
    get_savable_loader,
    get_train_dataset,
)
from megatron.energon.cache import MockLazy
from megatron.energon.cache.file_store import SystemFileStore
from megatron.energon.epathlib import EPath
from megatron.energon.errors import reraise_exception
from megatron.energon.source_info import SourceInfo


class _Strategy:
    def __init__(self, count=3):
        self.count = count

    def compute_params(self, media_list, num_tokens_available, **kwargs):
        return [ImageTilingParams(media=media_list[0], num_tiles=1, num_embeddings=self.count)]

    def apply_params(self, params, **kwargs):
        raise AssertionError("stream selection must not materialize image pixels")

    def stack(self, images):
        raise AssertionError("stream selection must not stack image pixels")


def _encoder(*, sequence_length=5, pad_token_id=0, strategy=None, **kwargs):
    options = dict(image_token_id=99, eos_token_id=1, vocab_size=128, image_token_budget=16)
    options.update(kwargs)
    return PreparedStreamingTaskEncoder(
        _Strategy() if strategy is None else strategy,
        sequence_length=sequence_length,
        pad_token_id=pad_token_id,
        **options,
    )


def _image(reads=None):
    def pixels(_):
        if reads is not None:
            reads.append("pixels")
        return Image.new("RGB", (64, 64), "blue")

    return ImageMedia(
        value=MockLazy(fname="screen.png", get_fn=pixels), metadata={"width": 64, "height": 64}
    )


def _document(parts, *, key="doc", index=7, source="fixture"):
    return InterleavedDocument(
        __key__=key,
        __restore_key__=("fixture", index),
        __sources__=(
            SourceInfo(
                dataset_path=EPath(f"/{source}/tokens.bin"),
                index=index,
                shard_name="tokens.bin",
                file_names=("tokens.bin",),
            ),
        ),
        __tags__={"packing_source": source},
        parts=parts,
    )


def _encoded(parts, *, sequence_length=5, key="doc", index=7, source="fixture", strategy=None):
    return _encoder(sequence_length=sequence_length, strategy=strategy).encode_sample(
        _document(parts, key=key, index=index, source=source)
    )


def _manual_encoded(*, key, ids, labels, spans=(), plans=(), source="fixture"):
    document = _document([TokenizedText(tokens=[3, 1])], key=key, source=source)
    return EncodedPreparedDocument.derive_from(
        document,
        input_ids=torch.tensor(ids, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        loss_mask=(torch.tensor(labels) != -100).float(),
        image_spans=list(spans),
        image_params=list(plans),
    )


def _select(encoder, samples):
    return encoder.select_next_pack(iter(samples))


def test_cut_at_image_end_is_valid_and_following_text_target_stays_with_visual_input():
    image = _image()
    encoded = _encoded([TokenizedText(tokens=[3, 10]), image, TokenizedText(tokens=[11, 1])])
    selected = _select(_encoder(), [encoded])
    assert isinstance(selected, PackedSamplesOutput)
    assert len(selected.pushback) == 1
    assert isinstance(selected.pushback[0], PartialSample)
    assert selected.pushback[0].sample is encoded
    assert selected.pushback[0].slice == (5, 7)
    packed = _encoder().pack_selected_samples(selected.packs[0])
    assert isinstance(packed, PackedPreparedDocument)
    assert packed.input_ids.tolist() == [3, 10, 99, 99, 99]
    assert packed.labels.tolist() == [10, -100, -100, -100, 11]
    assert packed.loss_mask.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert packed.image_spans == [(2, 5)]
    assert packed.image_params == [encoded.image_params[0]]
    assert packed.fragment_offsets == [0, 5]
    assert packed.fragments == [("doc", 0, 5)]
    assert packed.valid_length == 5


def test_consecutive_images_remain_whole_and_keep_order_across_packs():
    first, second = _image(), _image()
    encoded = _encoded(
        [TokenizedText(tokens=[3, 10]), first, second, TokenizedText(tokens=[11, 1])],
        sequence_length=5,
    )
    encoder = _encoder(sequence_length=5)
    first_selection = _select(encoder, [encoded])
    first_pack = encoder.pack_selected_samples(first_selection.packs[0])
    suffix_selection = _select(encoder, first_selection.pushback)
    second_pack = encoder.pack_selected_samples(suffix_selection.packs[0])
    assert first_pack.image_spans == [(2, 5)]
    assert second_pack.image_spans == [(0, 3)]
    assert first_pack.image_params == [encoded.image_params[0]]
    assert second_pack.image_params == [encoded.image_params[1]]
    assert first_pack.labels[-1].item() == -100
    assert second_pack.labels[2].item() == 11


def test_image_larger_than_sequence_is_rejected_without_resizing_or_dropping():
    encoded = _encoded(
        [TokenizedText(tokens=[3]), _image(), TokenizedText(tokens=[10, 1])],
        strategy=_Strategy(count=6),
    )
    with pytest.raises(ValueError, match="exceeds sequence_length"):
        _select(_encoder(), [encoded])


def test_selector_does_not_pull_another_source_after_filling_sequence():
    encoded = _encoded([TokenizedText(tokens=[3, 10, 11, 12, 1])])

    class CountingIterator:
        def __init__(self):
            self.count = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.count += 1
            if self.count == 1:
                return encoded
            raise AssertionError("selector over-pulled after the pack became full")

    source = CountingIterator()
    selected = _encoder().select_next_pack(source)
    assert source.count == 1
    assert not selected.pushback


def test_a_cut_inside_an_image_is_rejected_but_a_mixed_target_pack_is_valid():
    image = _image()
    base = _encoded([TokenizedText(tokens=[3, 10]), image, TokenizedText(tokens=[11, 1])])
    invalid = PartialSample(sample=base, slice=(3, 7))
    with pytest.raises(ValueError, match="cuts through an image span"):
        _select(_encoder(), [invalid])
    with pytest.raises(ValueError, match="cuts through an image span"):
        _encoder().pack_selected_samples([invalid])

    masked = _manual_encoded(key="masked", ids=[3, 99], labels=[-100, -100])
    eligible = _manual_encoded(key="eligible", ids=[10, 1], labels=[1, -100])
    packed = _encoder(sequence_length=4).pack_selected_samples(
        [PartialSample(sample=masked, slice=(0, 2)), PartialSample(sample=eligible, slice=(0, 2))]
    )
    assert packed.loss_mask.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert packed.fragments == [("masked", 0, 2), ("eligible", 0, 2)]


def test_completed_targetless_pack_is_rejected_and_padding_is_consistent():
    targetless = _manual_encoded(key="masked", ids=[3, 10], labels=[-100, -100])
    with pytest.raises(ValueError, match="no eligible next-token targets"):
        _encoder(sequence_length=4).pack_selected_samples(
            [PartialSample(sample=targetless, slice=(0, 2))]
        )

    encoded = _encoded([TokenizedText(tokens=[3, 10, 1])], sequence_length=5)
    packed = _encoder(sequence_length=5).pack_selected_samples(
        [PartialSample(sample=encoded, slice=(0, 3))]
    )
    assert packed.input_ids.tolist() == [3, 10, 1, 0, 0]
    assert packed.labels.tolist() == [10, 1, -100, -100, -100]
    assert packed.loss_mask.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert packed.fragment_offsets == [0, 3]
    assert packed.valid_length == 3
    assert packed.__sources__ == encoded.__sources__


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequence_length": True},
        {"sequence_length": 0},
        {"sequence_length": 4.0},
        {"pad_token_id": True},
        {"pad_token_id": -1},
        {"pad_token_id": 128},
        {"pad_token_id": 99},
    ],
)
def test_invalid_sequence_or_pad_configuration_is_rejected(kwargs):
    options = {"sequence_length": 5, "pad_token_id": 0}
    options.update(kwargs)
    with pytest.raises(ValueError):
        _encoder(**options)


def test_eos_only_final_suffix_is_explicitly_rejected():
    encoder = _encoder(sequence_length=4)
    encoded = encoder.encode_sample(_document([TokenizedText(tokens=[3, 10, 11, 12, 1])]))
    selection = _select(encoder, [encoded])
    assert encoder.pack_selected_samples(selection.packs[0]).labels.tolist() == [10, 11, 12, 1]
    suffix = _select(encoder, selection.pushback)
    with pytest.raises(ValueError, match="no eligible next-token targets"):
        encoder.pack_selected_samples(suffix.packs[0])


def test_pad_may_equal_eos_and_each_native_source_gets_its_own_stream_group():
    encoder = _encoder(pad_token_id=1)
    sources = [object(), object()]
    groups = encoder.build_packing_groups(sources, "stream", 7)
    assert len(groups) == 2
    assert [group.datasets for group in groups] == [[sources[0]], [sources[1]]]
    assert all(group.packing_buffer_size == "stream" for group in groups)
    assert all(group.shuffle_buffer_size == 7 for group in groups)
    with pytest.raises(ValueError, match="requires packing_buffer_size='stream'"):
        encoder.build_packing_groups(sources, 8, 7)


def _write_binidx_source(root, documents, *, image_index=None):
    root.mkdir()
    sizes = np.array([len(tokens) for tokens in documents], dtype="<i4")
    np.concatenate([np.array(tokens, dtype="<i4") for tokens in documents]).tofile(
        root / "tokens.bin"
    )
    offsets = np.concatenate(([0], np.cumsum(sizes[:-1], dtype=np.int64))) * 4
    with (root / "tokens.idx").open("wb") as stream:
        stream.write(b"MMIDIDX\0\0" + struct.pack("<QBQQ", 1, 4, len(documents), len(documents)))
        stream.write(sizes.tobytes())
        stream.write(offsets.astype("<i8").tobytes())
        stream.write(np.arange(len(documents), dtype="<i8").tobytes())
    (root / "annotations").mkdir()
    (root / "images").mkdir()
    for index in range(len(documents)):
        images = (
            [{"position": 3, "width": 64, "height": 64, "key": "screen.png"}]
            if index == image_index
            else []
        )
        (root / "annotations" / f"{index}.json").write_text(json.dumps({"images": images}))
    if image_index is not None:
        Image.new("RGB", (64, 64), "blue").save(root / "images" / "screen.png")


def _write_blended_recipe(tmp_path):
    _write_binidx_source(tmp_path / "source-a", [[3, 10, 11, 12, 1], [3, 30, 1]])
    _write_binidx_source(
        tmp_path / "source-b", [[3, 20, 21, 22, 1], [3, 40, 41, 42, 1]], image_index=0
    )
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "__module__: megatron.energon\n__class__: Recipe\nsplits:\n  train:\n"
        "    blend_epochized:\n"
        "      - repetitions: 1\n        path: source-a/tokens.bin\n"
        "        tags: {cook: multimodal_interleaved_binidx, packing_source: source-a}\n"
        "        aux:\n          metadata: filesystem://source-a/annotations\n"
        "          images: filesystem://source-a/images\n"
        "      - repetitions: 1\n        path: source-b/tokens.bin\n"
        "        tags: {cook: multimodal_interleaved_binidx, packing_source: source-b}\n"
        "        aux:\n          metadata: filesystem://source-b/annotations\n"
        "          images: filesystem://source-b/images\n"
    )
    return recipe


def _build_native_loader(recipe_path):
    encoder = _encoder()
    dataset = get_train_dataset(
        recipe_path,
        worker_config=WorkerConfig(
            rank=0,
            world_size=1,
            num_workers=0,
            global_error_handler=reraise_exception,
            restore_error_handler=reraise_exception,
        ),
        task_encoder=encoder,
        batch_size=None,
        packing_buffer_size="stream",
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        repeat=False,
    )
    return get_savable_loader(dataset, cache_pool=NoCachePool(), watchdog_timeout_seconds=None)


def _snapshot(sample):
    return {
        "input_ids": sample.input_ids.tolist(),
        "labels": sample.labels.tolist(),
        "loss_mask": sample.loss_mask.tolist(),
        "image_spans": [list(span) for span in sample.image_spans],
        "fragment_offsets": sample.fragment_offsets,
        "fragments": [list(fragment) for fragment in sample.fragments],
        "valid_length": sample.valid_length,
        "source": sample.__tags__["packing_source"],
    }


def _resume_native_in_fresh_process(recipe_path, state_path):
    loader = _build_native_loader(recipe_path)
    try:
        loader.restore_state_rank(torch.load(state_path, map_location="cpu", weights_only=False))
        return [_snapshot(sample) for sample in loader]
    finally:
        loader.close()


def test_native_binidx_streaming_blend_coverage_restore_and_lazy_image_suffix(
    tmp_path, monkeypatch
):
    recipe = _write_blended_recipe(tmp_path)
    image_reads = []
    original_get = SystemFileStore.__getitem__

    def tracked_get(store, key):
        if str(store.get_path()) == str(tmp_path / "source-b" / "images"):
            image_reads.append(key)
        return original_get(store, key)

    monkeypatch.setattr(SystemFileStore, "__getitem__", tracked_get)
    loader = _build_native_loader(recipe)
    try:
        outputs = list(loader)
        assert outputs
        assert {sample.__tags__["packing_source"] for sample in outputs} == {"source-a", "source-b"}
        source_a_outputs = [
            sample for sample in outputs if sample.__tags__["packing_source"] == "source-a"
        ]
        assert source_a_outputs[-1].valid_length == 3
        assert all(
            len({source.dataset_path for source in sample.__sources__}) == 1 for sample in outputs
        )
        # Independent document-window coverage catches source omissions and duplicates.
        windows = {"source-a": [], "source-b": []}
        for sample in outputs:
            windows[sample.__tags__["packing_source"]].extend(
                (source.index, start, stop)
                for (_, start, stop) in sample.fragments
                for source in sample.__sources__
            )
        assert sorted(windows["source-a"]) == [(0, 0, 5), (1, 0, 3)]
        assert sorted(windows["source-b"]) == [(0, 0, 3), (0, 3, 8), (1, 0, 5)]
        expected = {
            "source-a": [
                ([3, 10, 11, 12, 1], [10, 11, 12, 1, -100]),
                ([3, 30, 1, 0, 0], [30, 1, -100, -100, -100]),
            ],
            "source-b": [
                ([3, 20, 21, 0, 0], [20, 21, -100, -100, -100]),
                ([99, 99, 99, 22, 1], [-100, -100, 22, 1, -100]),
                ([3, 40, 41, 42, 1], [40, 41, 42, 1, -100]),
            ],
        }
        for source, oracle in expected.items():
            actual = [sample for sample in outputs if sample.__tags__["packing_source"] == source]
            assert [
                (sample.input_ids.tolist(), sample.labels.tolist()) for sample in actual
            ] == oracle
            assert [sample.loss_mask.tolist() for sample in actual] == [
                [float(label != -100) for label in labels] for _, labels in oracle
            ]
        assert image_reads == []
        assert loader.can_restore_sample()
        for sample in outputs:
            restored = loader.restore_sample(sample.__restore_key__)
            assert _snapshot(restored) == _snapshot(sample)
    finally:
        loader.close()


def test_native_fresh_process_restore_replays_pending_image_suffix(tmp_path):
    recipe = _write_blended_recipe(tmp_path)
    loader = _build_native_loader(recipe)
    iterator = iter(loader)
    selected_prefix = None
    state_path = tmp_path / "loader-state.pt"
    try:
        for sample in iterator:
            if sample.__tags__["packing_source"] == "source-b" and sample.valid_length == 3:
                selected_prefix = sample
                break
        assert selected_prefix is not None
        assert selected_prefix.image_spans == []
        torch.save(loader.save_state_rank(), state_path)
        uninterrupted = [_snapshot(sample) for sample in iterator]
    finally:
        iterator.close()
        loader.close()

    test_directory = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(test_directory, "../../.."))
    script = (
        "import json, sys; "
        f"sys.path.insert(0, {test_directory!r}); "
        "from test_prepared_streaming import _resume_native_in_fresh_process; "
        f"print(json.dumps(_resume_native_in_fresh_process({str(recipe)!r}, {str(state_path)!r})))"
    )
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (project_root, current_pythonpath) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True, env=environment
    )
    restored = json.loads(result.stdout.strip().splitlines()[-1])
    assert restored == uninterrupted


def test_native_finite_tail_checkpoint_does_not_restart_exhausted_source(tmp_path):
    _write_binidx_source(tmp_path / "finite", [[3, 10, 11, 1]])
    recipe = tmp_path / "finite.yaml"
    recipe.write_text(
        "__module__: megatron.energon\n__class__: Recipe\nsplits:\n  train:\n"
        "    path: finite/tokens.bin\n"
        "    tags: {cook: multimodal_interleaved_binidx, packing_source: finite}\n"
        "    aux:\n      metadata: filesystem://finite/annotations\n"
        "      images: filesystem://finite/images\n"
    )
    state_path = tmp_path / "tail-state.pt"
    loader = _build_native_loader(recipe)
    iterator = iter(loader)
    try:
        tail = next(iterator)
        assert tail.valid_length == 4
        assert tail.input_ids.tolist() == [3, 10, 11, 1, 0]
        torch.save(loader.save_state_rank(), state_path)
        assert list(iterator) == []
    finally:
        iterator.close()
        loader.close()
    assert _resume_native_in_fresh_process(recipe, state_path) == []
