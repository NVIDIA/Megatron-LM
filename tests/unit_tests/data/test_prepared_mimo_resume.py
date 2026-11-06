# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native prepared MIMO image-stream replay across fixed-P DP placements."""

import json
import struct
from argparse import Namespace
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from examples.mimo.training import prepared_binidx
from examples.mimo.training.energon_checkpoint import redistribute_energon_states
from examples.mimo.training.energon_provider import build_energon_iterator
from megatron.energon import NoCachePool
from megatron.energon.cache.file_store import SystemFileStore
from megatron.training.config.training_config import TokenizerConfig

_GLOBAL_WORKERS = 4
_GLOBAL_BATCH_SIZE = 4
_SEQUENCE_LENGTH = 8
_IMAGE_TOKEN_ID = 99
_IGNORE_INDEX = -100


def _write_binidx_source(root: Path) -> Path:
    root.mkdir()
    documents = [[3, 10 + index, 70, 10 + index, 80, 1] for index in range(32)]
    sizes = np.asarray([len(tokens) for tokens in documents], dtype="<i4")
    np.concatenate([np.asarray(tokens, dtype="<i4") for tokens in documents]).tofile(
        root / "tokens.bin"
    )
    offsets = np.concatenate(([0], np.cumsum(sizes[:-1], dtype=np.int64))) * 4
    with (root / "tokens.idx").open("wb") as stream:
        stream.write(b"MMIDIDX\0\0" + struct.pack("<QBQQ", 1, 4, len(documents), len(documents)))
        stream.write(sizes.tobytes())
        stream.write(offsets.astype("<i8").tobytes())
        stream.write(np.arange(len(documents), dtype="<i8").tobytes())

    annotations = root / "annotations"
    images = root / "images"
    annotations.mkdir()
    images.mkdir()
    for index in range(len(documents)):
        (annotations / f"{index}.json").write_text(
            json.dumps(
                {"images": [{"position": 3, "width": 64, "height": 64, "key": "screen.png"}]}
            )
        )
    Image.new("RGB", (64, 64), "blue").save(images / "screen.png")

    recipe = root.parent / "prepared-recipe.yaml"
    recipe.write_text(
        "__module__: megatron.energon\n"
        "__class__: Recipe\n"
        "splits:\n"
        "  train:\n"
        "    path: source/tokens.bin\n"
        "    tags: {cook: multimodal_interleaved_binidx}\n"
        "    aux:\n"
        "      metadata: filesystem://source/annotations\n"
        "      images: filesystem://source/images\n"
    )
    return recipe


def _args(recipe: Path, lanes: int, workers: int) -> Namespace:
    return Namespace(
        data_path=[str(recipe)],
        energon_data_config="examples.mimo.training.prepared_binidx:make_config",
        energon_dp_resume=True,
        energon_logical_workers=None,
        energon_image_token_budget=4,
        dynamic_resolution=True,
        disable_vision_class_token=True,
        patch_dim=16,
        pixel_shuffle=True,
        seq_length=_SEQUENCE_LENGTH,
        image_token_id=_IMAGE_TOKEN_ID,
        energon_eos_token_id=1,
        energon_pad_token_id=0,
        vocab_size=128,
        padded_vocab_size=128,
        global_batch_size=_GLOBAL_BATCH_SIZE,
        micro_batch_size=1,
        num_workers=workers,
        seed=17,
        mimo_llm_dp=lanes,
    )


def _build_streams(recipe, lanes, workers, checkpoint_paths=None):
    return [
        build_energon_iterator(
            _args(recipe, lanes, workers),
            lane_ids=(lane,),
            language_lanes=lanes,
            module_name=None,
            checkpoint_paths=checkpoint_paths,
        )
        for lane in range(lanes)
    ]


def _snapshot(batch):
    ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    loss_mask = batch["loss_mask"][0]
    assert ids.shape == labels.shape == loss_mask.shape == (_SEQUENCE_LENGTH,)
    assert torch.equal(loss_mask, (labels != _IGNORE_INDEX).float())
    assert torch.all((0 <= ids) & (ids < 128))

    index = 0
    found_image = False
    while index < len(ids):
        if ids[index] != _IMAGE_TOKEN_ID:
            index += 1
            continue
        found_image = True
        stop = index + 1
        while stop < len(ids) and ids[stop] == _IMAGE_TOKEN_ID:
            stop += 1
        assert stop - index == 4
        # The final visual input predicts the first ordinary token after it.
        # Keep this target eligible even when the image span ends at the row edge.
        assert loss_mask[stop - 1] == 1
        assert 10 <= labels[stop - 1] < 42
        index = stop

    return (tuple(ids.tolist()), tuple(labels.tolist()), tuple(loss_mask.tolist()), found_image)


def _next_global_batch(streams, workers):
    per_global_worker = {}
    for lane, stream in enumerate(streams):
        for local_worker in range(workers):
            per_global_worker[lane * workers + local_worker] = _snapshot(next(stream))
    assert set(per_global_worker) == set(range(_GLOBAL_WORKERS))
    return per_global_worker, Counter(per_global_worker.values())


def _streaming_carryover(state):
    if isinstance(state, dict):
        if state.get("__class__") == "StreamingPackingDataset":
            yield state["_carryover_buffer"]["_restore_keys"]
        else:
            for child in state.values():
                yield from _streaming_carryover(child)
    elif isinstance(state, (list, tuple)):
        for child in state:
            yield from _streaming_carryover(child)


def _save_and_reload_rank_states(streams, directory, *, committed_packs):
    directory.mkdir()
    states = []
    paths = {}
    for lane, stream in enumerate(streams):
        path = directory / f"train_dataloader_dprank{lane:03d}.pt"
        torch.save({"dataloader_state_dict": stream.save_state()}, path)
        paths[lane] = path
        states.append(
            torch.load(path, map_location="cpu", weights_only=False)["dataloader_state_dict"]
        )
        for checkpoint in states[-1]["energon"].worker_states:
            assert checkpoint.offset == 0
            assert checkpoint.state.sample_index == committed_packs
    return states, paths


def test_real_prepared_image_suffix_resumes_across_fixed_worker_dp(tmp_path, monkeypatch):
    # Model construction is outside this data-path regression; bind its real
    # configuration type so the selected public factory reads the same padded
    # vocabulary field used by the runtime model provider.
    tokenizer = TokenizerConfig(vocab_size=128, padded_vocab_size=128)
    monkeypatch.setattr(
        prepared_binidx,
        "get_run_config",
        lambda: SimpleNamespace(tokenizer=tokenizer),
        raising=False,
    )
    public_make_config = prepared_binidx.make_config

    def checkpoint_each_pack(args):
        config = public_make_config(args)
        return replace(
            config,
            loader_kwargs={
                **config.loader_kwargs,
                "checkpoint_every_sec": 0,
                "checkpoint_every_min_n_samples": 1,
            },
        )

    # Save the committed carryover itself, rather than an earlier checkpoint
    # plus a replay offset that would reconstruct the pending image later.
    monkeypatch.setattr(prepared_binidx, "make_config", checkpoint_each_pack)
    recipe = _write_binidx_source(tmp_path / "source")
    config = prepared_binidx.make_config(_args(recipe, lanes=2, workers=2))
    assert isinstance(config.loader_kwargs["cache_pool"], NoCachePool)

    image_directory = str((tmp_path / "source" / "images").resolve())
    original_getitem = SystemFileStore.__getitem__

    def reject_language_pixel_reads(store, key):
        if str(store.get_path()) == image_directory and key == "screen.png":
            raise AssertionError("language-lane replay must keep RADIO pixels lazy")
        return original_getitem(store, key)

    monkeypatch.setattr(SystemFileStore, "__getitem__", reject_language_pixel_reads)

    reference_streams = _build_streams(recipe, lanes=2, workers=2)
    try:
        first, first_multiset = _next_global_batch(reference_streams, workers=2)
        assert len(first_multiset) == _GLOBAL_WORKERS
        assert all(sample[3] for sample in first.values())
        assert all(sample[0][3:7] == (_IMAGE_TOKEN_ID,) * 4 for sample in first.values())

        second, _ = _next_global_batch(reference_streams, workers=2)
        assert all(not sample[3] for sample in second.values())
        # The second pack holds the previous document's two-token suffix and
        # the next document's three-token prefix. Its image-safe cut leaves the
        # next four-token image span in native streaming pushback.
        states, _ = _save_and_reload_rank_states(
            reference_streams, tmp_path / "checkpoint-0", committed_packs=2
        )
        for state in states:
            for checkpoint in state["energon"].worker_states:
                buffers = list(_streaming_carryover(checkpoint.state.dataset_state))
                assert len(buffers) == 1
                assert len(buffers[0]) == 1
                pending = buffers[0][0]
                assert pending[0] == "PartialSample"
                assert pending[2] == (3, 10)  # Starts at the pending four-token image.
        expected = [_next_global_batch(reference_streams, workers=2) for _ in range(3)]
        assert all(sample[3] for sample in expected[0][0].values())
    finally:
        for stream in reference_streams:
            stream.close()

    current_states = states
    for stage, (lanes, workers) in enumerate(((1, 4), (4, 1), (2, 2))):
        migrated = redistribute_energon_states(current_states, lanes)
        checkpoint_dir = tmp_path / f"checkpoint-{stage + 1}"
        checkpoint_dir.mkdir()
        checkpoint_paths = {}
        for lane, state in enumerate(migrated):
            path = checkpoint_dir / f"train_dataloader_dprank{lane:03d}.pt"
            torch.save({"dataloader_state_dict": state}, path)
            checkpoint_paths[lane] = path

        streams = _build_streams(recipe, lanes, workers, checkpoint_paths)
        try:
            actual = _next_global_batch(streams, workers)
            assert actual == expected[stage]
            if stage == 0:
                assert all(sample[3] for sample in actual[0].values())
            if stage < 2:
                current_states, _ = _save_and_reload_rank_states(
                    streams,
                    tmp_path / f"checkpoint-{stage + 1}-normal-save",
                    committed_packs=stage + 3,
                )
        finally:
            for stream in streams:
                stream.close()
