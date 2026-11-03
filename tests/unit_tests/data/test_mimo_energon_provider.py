# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU data-path checks using native Energon, independent of model/GPU initialization."""

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.mimo.training.energon_provider import (
    EnergonDataConfig,
    add_energon_data_args,
    build_energon_iterator,
    validate_energon_data_args,
)
from megatron.energon import Cooker, DefaultTaskEncoder, basic_sample_keys, stateless
from megatron.energon.flavors import CrudeJsonlDatasetFactory


@stateless
def cook_conversation(sample):
    return {**basic_sample_keys(sample), **sample["json"]}


@stateless
def cook_prepared_tokens(sample):
    tokens = sample["tokens"] if "tokens" in sample else sample["json"]["tokens"]
    return {**basic_sample_keys(sample), "tokens": tokens}


class ConversationEncoder(DefaultTaskEncoder):
    """Small SFT-style task with variable-length, non-image modality inputs."""

    cookers = [Cooker(cook_conversation)]

    @stateless
    def encode_sample(self, sample):
        user, assistant = sample["messages"]
        index = int(user["content"])
        tokens = [100 + index, *map(ord, user["content"] + assistant["content"]), 2]
        return {"tokens": tokens, "count": index % 3}

    @stateless
    def batch(self, samples):
        tokens = torch.tensor([sample["tokens"] for sample in samples])
        labels = torch.full_like(tokens, -100)
        labels[:, 3:6] = tokens[:, 4:7]
        lengths = torch.tensor([sample["count"] for sample in samples])
        values = torch.cat(
            [torch.full((sample["count"],), sample["tokens"][0]) for sample in samples]
        )
        return {
            "input_ids": tokens,
            "labels": labels,
            "loss_mask": (labels != -100).float(),
            "position_ids": torch.arange(tokens.shape[1]).expand_as(tokens),
            "cu_seqlens": torch.tensor([[0, tokens.shape[1]]] * len(samples)),
            "attention_mask": torch.ones(len(samples), 1, tokens.shape[1], dtype=torch.bool),
            "modality_token_indices": {"sound": torch.tensor([1])},
            "modality_inputs": {"sound": {"values": values, "lengths": lengths}},
        }


class PreparedTokenEncoder(DefaultTaskEncoder):
    """Already-tokenized data uses the same bridge without text or image policy."""

    cookers = [Cooker(cook_prepared_tokens)]

    @stateless
    def encode_sample(self, sample):
        return torch.tensor(sample["tokens"], dtype=torch.long)

    @stateless
    def batch(self, samples):
        tokens = torch.stack(samples)
        labels = torch.full_like(tokens, -100)
        labels[:, :-1] = tokens[:, 1:]
        labels[:, 1:3] = -100
        return {
            "input_ids": tokens,
            "labels": labels,
            "loss_mask": (labels != -100).float(),
            "position_ids": torch.arange(tokens.shape[1]).expand_as(tokens),
            "modality_inputs": {
                "frames": {"features": tokens[:, :1].float().repeat_interleave(2, dim=0)}
            },
        }


class RandomizedEncoder(ConversationEncoder):
    @stateless(restore_seeds=True)
    def encode_sample(self, sample):
        encoded = super().encode_sample(sample)
        # Exercise all native RNG families, without untracked main-process RNG.
        encoded["tokens"][1] = random.randrange(100) + int(np.random.randint(100))
        encoded["tokens"][2] = int(torch.randint(100, (1,)).item())
        return encoded


class EmptyModalityEncoder(ConversationEncoder):
    @stateless
    def encode_sample(self, sample):
        encoded = super().encode_sample(sample)
        encoded["count"] = 0
        return encoded


def merge_sound_batches(batches, *, lane_ids, module_name):
    assert len(lane_ids) == len(batches)
    if not any(batch["modality_inputs"]["sound"]["values"].numel() for batch in batches):
        return {"input_ids": torch.cat([batch["input_ids"] for batch in batches])}
    return {
        "input_ids": torch.cat([batch["input_ids"] for batch in batches]),
        "modality_inputs": {
            module_name: {
                module_name: {
                    "values": torch.cat([b["modality_inputs"]["sound"]["values"] for b in batches]),
                    "lengths": torch.cat(
                        [b["modality_inputs"]["sound"]["lengths"] for b in batches]
                    ),
                }
            }
        },
    }


def merge_frame_batches(batches, *, lane_ids, module_name):
    assert len(lane_ids) == len(batches)
    return {
        "input_ids": torch.cat([batch["input_ids"] for batch in batches]),
        "modality_inputs": {
            module_name: {
                module_name: {
                    "features": torch.cat(
                        [b["modality_inputs"]["frames"]["features"] for b in batches]
                    )
                }
            }
        },
    }


def conversation_config(args):
    return EnergonDataConfig(
        create_task_encoder=lambda *, split: ConversationEncoder(),
        checkpoint_identity=getattr(args, "application_identity", "conversation-fixture"),
        merge_encoder_batches=merge_sound_batches,
        dataset_kwargs={"parallel_shard_iters": 1, "max_samples_per_sequence": 1},
        loader_kwargs={"watchdog_timeout_seconds": None, "checkpoint_every_sec": 0},
    )


def prepared_config(args):
    return EnergonDataConfig(
        create_task_encoder=lambda *, split: PreparedTokenEncoder(),
        checkpoint_identity="prepared-token-fixture",
        merge_encoder_batches=merge_frame_batches,
        dataset_kwargs={"parallel_shard_iters": 1, "max_samples_per_sequence": 1},
        loader_kwargs={"watchdog_timeout_seconds": None, "checkpoint_every_sec": 0},
    )


def randomized_config(args):
    return EnergonDataConfig(
        create_task_encoder=lambda *, split: RandomizedEncoder(),
        checkpoint_identity="randomized-conversation-fixture",
        merge_encoder_batches=merge_sound_batches,
        dataset_kwargs={"parallel_shard_iters": 1, "max_samples_per_sequence": 1},
        loader_kwargs={"watchdog_timeout_seconds": None, "checkpoint_every_sec": 0},
    )


def empty_config(args):
    return replace(
        conversation_config(args),
        create_task_encoder=lambda *, split: EmptyModalityEncoder(),
        checkpoint_identity="empty-modality-fixture",
    )


def failing_config(args):
    def fail(*args, **kwargs):
        raise RuntimeError("application merger failed")

    return replace(conversation_config(args), merge_encoder_batches=fail)


def _args(path, config="conversation_config", **overrides):
    values = dict(
        dataset_provider="energon",
        energon_data_config=f"{__name__}:{config}",
        data_path=[str(path)],
        num_workers=1,
        micro_batch_size=2,
        seed=19,
        mimo_llm_tp=1,
        mimo_llm_pp=1,
        mimo_llm_cp=1,
        mimo_encoder_tp=1,
        mimo_encoder_dp=1,
        mimo_llm_dp=3,
        mimo_llm_only=False,
        mimo_encoder_prefetch=False,
        mimo_encoder_ddp_overlap=False,
        eval_iters=0,
        dataloader_save=None,
        save=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _jsonl(path, prepared=False):
    rows = []
    for i in range(96):
        rows.append(
            {"tokens": [100 + i, 7, 5, 23, 9, 11, 2]}
            if prepared
            else {
                "messages": [
                    {"role": "user", "content": f"{i:03d}"},
                    {"role": "assistant", "content": "OK"},
                ]
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    CrudeJsonlDatasetFactory.prepare_dataset(path)
    return path


def _build(args, lanes=(0,), module=None, **kwargs):
    return build_energon_iterator(
        args, lane_ids=lanes, language_lanes=args.mimo_llm_dp, module_name=module, **kwargs
    )


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    else:
        assert actual == expected


@pytest.mark.parametrize("prepared", [False, True])
def test_native_tasks_preserve_language_and_merge_unequal_lanes(tmp_path, prepared):
    args = _args(
        _jsonl(tmp_path / "data.jsonl", prepared),
        "prepared_config" if prepared else "conversation_config",
    )
    languages = [_build(args, (lane,)) for lane in range(3)]
    encoder = _build(args, (0, 1, 2), "arbitrary_encoder")
    try:
        for _ in range(3):
            batches = [next(language) for language in languages]
            actual = next(encoder)
            assert torch.equal(actual["input_ids"], torch.cat([b["input_ids"] for b in batches]))
            for batch in batches:
                assert "modality_inputs" not in batch
                assert torch.equal(batch["loss_mask"], (batch["labels"] != -100).float())
                if prepared:
                    assert torch.all(batch["labels"][:, 1:3] == -100)
                    assert torch.equal(batch["labels"][:, 3:6], batch["input_ids"][:, 4:7])
                else:
                    assert batch["loss_mask"].tolist() == [[0, 0, 0, 1, 1, 1, 0]] * 2
                    assert "attention_mask" in batch and "modality_token_indices" in batch
            if prepared:
                assert torch.equal(
                    actual["modality_inputs"]["arbitrary_encoder"]["arbitrary_encoder"]["features"],
                    actual["input_ids"][:, :1].float().repeat_interleave(2, dim=0),
                )
            else:
                values = actual["modality_inputs"]["arbitrary_encoder"]["arbitrary_encoder"]
                counts = (actual["input_ids"][:, 0] - 100) % 3
                assert torch.equal(values["lengths"], counts)
                assert torch.equal(
                    values["values"], actual["input_ids"][:, 0].repeat_interleave(counts)
                )
        assert encoder.save_state() is None
    finally:
        encoder.close()
        for language in languages:
            language.close()


def _resume_in_fresh_process(args, paths, output):
    """Module-level entrypoint so a new process restores the real native state."""
    encoder = _build(args, tuple(paths), "sound_encoder", checkpoint_paths=paths)
    languages = [_build(args, (lane,), checkpoint_paths=paths) for lane in paths]
    try:
        torch.save(
            {
                "encoder": [next(encoder) for _ in range(4)],
                "languages": [[next(language) for _ in range(4)] for language in languages],
            },
            output,
        )
    finally:
        encoder.close()
        for language in languages:
            language.close()


@pytest.mark.parametrize("workers", [1, 2])
def test_positive_workers_randomness_and_fresh_process_resume(tmp_path, workers):
    args = _args(
        _jsonl(tmp_path / "data.jsonl"), "randomized_config", mimo_llm_dp=2, num_workers=workers
    )
    languages = [_build(args, (lane,)) for lane in range(2)]
    encoder = _build(args, (0, 1), "sound_encoder")
    paths = {lane: tmp_path / f"lane{lane}.pt" for lane in range(2)}
    try:
        for _ in range(3):
            batches = [next(language) for language in languages]
            assert torch.equal(
                next(encoder)["input_ids"], torch.cat([b["input_ids"] for b in batches])
            )
        for lane, language in enumerate(languages):
            torch.save({"dataloader_state_dict": language.save_state()}, paths[lane])
        expected = {
            "encoder": [next(encoder) for _ in range(4)],
            "languages": [[next(language) for _ in range(4)] for language in languages],
        }
    finally:
        encoder.close()
        for language in languages:
            language.close()
    request, output = tmp_path / "request.pt", tmp_path / "continued.pt"
    torch.save((args, paths), request)
    code = (
        "import torch; "
        f"from {__name__} import _resume_in_fresh_process; "
        f"args, paths = torch.load({str(request)!r}, weights_only=False); "
        f"_resume_in_fresh_process(args, paths, {str(output)!r})"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), *sys.path])
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=90)
    actual = torch.load(output, weights_only=False)
    for key in expected:
        for left, right in zip(actual[key], expected[key], strict=True):
            if key == "languages":
                for a, b in zip(left, right, strict=True):
                    _equal(a, b)
            else:
                _equal(left, right)


def test_native_state_rejects_changed_application_and_seed(tmp_path):
    args = _args(_jsonl(tmp_path / "data.jsonl"), num_workers=0, mimo_llm_dp=1)
    iterator = _build(args)
    path = tmp_path / "state.pt"
    try:
        next(iterator)
        torch.save({"dataloader_state_dict": iterator.save_state()}, path)
    finally:
        iterator.close()
    for changed in (
        _args(args.data_path[0], num_workers=0, mimo_llm_dp=1, seed=20),
        _args(args.data_path[0], num_workers=0, mimo_llm_dp=1, application_identity="changed"),
    ):
        with pytest.raises(ValueError, match="incompatible"):
            _build(changed, checkpoint_paths={0: path})
    with pytest.raises(FileNotFoundError, match="missing"):
        _build(args, checkpoint_paths={0: tmp_path / "missing.pt"})
    fresh = _build(
        args, checkpoint_paths={0: tmp_path / "missing.pt"}, allow_missing_checkpoint=True
    )
    fresh.close()


def test_close_terminates_persistent_workers(tmp_path):
    iterator = _build(_args(_jsonl(tmp_path / "data.jsonl")))
    next(iterator)
    workers = list(iterator._loaders[0]._iterator._workers)
    assert all(worker.is_alive() for worker in workers)
    iterator.close()
    iterator.close()
    assert all(not worker.is_alive() for worker in workers)
    with pytest.raises(StopIteration):
        next(iterator)


def test_empty_modality_batches_keep_all_input_rows(tmp_path):
    args = _args(_jsonl(tmp_path / "data.jsonl"), "empty_config", mimo_llm_dp=2)
    encoder = _build(args, (0, 1), "sound_encoder")
    try:
        batch = next(encoder)
        assert batch["input_ids"].shape == (4, 7)
        assert "modality_inputs" not in batch
    finally:
        encoder.close()


def test_merger_failure_closes_all_workers_and_cannot_retry(tmp_path):
    args = _args(_jsonl(tmp_path / "data.jsonl"), "failing_config", mimo_llm_dp=2)
    encoder = _build(args, (0, 1), "sound_encoder")
    loaders = list(encoder._loaders)
    with pytest.raises(RuntimeError, match="merger failed"):
        next(encoder)
    assert all(not worker.is_alive() for loader in loaders for worker in loader._iterator._workers)
    with pytest.raises(StopIteration):
        next(encoder)


def test_second_lane_construction_failure_closes_first_loader(tmp_path, monkeypatch):
    import megatron.energon
    from examples.mimo.training import energon_provider

    args = _args(_jsonl(tmp_path / "data.jsonl"), mimo_llm_dp=2)
    original = megatron.energon.get_savable_loader
    created, closed = [], []

    def build(dataset, **kwargs):
        if created:
            raise RuntimeError("second lane failed")
        loader = original(dataset, **kwargs)
        created.append(loader)
        return loader

    original_close = energon_provider._close_loader

    def close(loader):
        closed.append(loader)
        original_close(loader)

    monkeypatch.setattr(megatron.energon, "get_savable_loader", build)
    monkeypatch.setattr(energon_provider, "_close_loader", close)
    with pytest.raises(RuntimeError, match="second lane failed"):
        _build(args, (0, 1), "sound_encoder")
    assert len(created) == 1 and closed == created


@pytest.mark.parametrize(
    "override",
    [
        {"mimo_llm_cp": 2},
        {"mimo_encoder_ddp_overlap": True},
        {"mimo_encoder_prefetch": True},
        {"eval_iters": 1},
        {"num_workers": 0},
        {"energon_data_config": "not_a_factory"},
        {"energon_dataloader_load": "/checkpoints"},
    ],
)
def test_unsupported_settings_fail_before_workers(override):
    with pytest.raises(ValueError):
        validate_energon_data_args(_args("unused.jsonl", **override))


def test_mock_args_are_untouched_and_help_explains_application_and_roots():
    args = argparse.Namespace(dataset_provider="mock")
    validate_energon_data_args(args)
    assert vars(args) == {"dataset_provider": "mock"}
    help_text = add_energon_data_args(argparse.ArgumentParser()).format_help()
    assert "MODULE:FUNCTION" in help_text and "not an iteration directory" in help_text
