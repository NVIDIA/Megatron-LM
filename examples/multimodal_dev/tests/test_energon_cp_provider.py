# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CP sharding tests for the qwen3.5 VL Energon provider."""

import os
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _install_energon_stubs():
    """Install minimal megatron.energon stubs when the package is absent."""
    if "megatron.energon" in sys.modules:
        return

    energon_mod = types.ModuleType("megatron.energon")

    class DefaultTaskEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def __class_getitem__(cls, item):
            return cls

    class VQASample:
        pass

    class WorkerConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def _not_used(*args, **kwargs):
        raise AssertionError("Energon dataset construction is not used in this unit test")

    energon_mod.DefaultTaskEncoder = DefaultTaskEncoder
    energon_mod.LimitDataset = object
    energon_mod.RepeatDataset = object
    energon_mod.VQASample = VQASample
    energon_mod.WorkerConfig = WorkerConfig
    energon_mod.get_loader = _not_used
    energon_mod.get_savable_loader = _not_used
    energon_mod.get_train_dataset = _not_used
    energon_mod.get_val_datasets = _not_used

    task_encoder_mod = types.ModuleType("megatron.energon.task_encoder")
    base_mod = types.ModuleType("megatron.energon.task_encoder.base")

    def stateless(fn=None, **_kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    base_mod.stateless = stateless
    sys.modules["megatron.energon"] = energon_mod
    sys.modules["megatron.energon.task_encoder"] = task_encoder_mod
    sys.modules["megatron.energon.task_encoder.base"] = base_mod


_install_energon_stubs()

from examples.multimodal_dev.data import energon
from examples.multimodal_dev.arguments import validate_multimodal_args


def test_dataloader_parallel_state_excludes_context_parallel(monkeypatch):
    no_cp_group = object()
    with_cp_group = object()
    calls = []

    def fake_rank(*, with_context_parallel=False):
        calls.append(("rank", with_context_parallel))
        return 7 if with_context_parallel else 3

    def fake_world_size(*, with_context_parallel=False):
        calls.append(("world", with_context_parallel))
        return 16 if with_context_parallel else 8

    def fake_group(*, with_context_parallel=False):
        calls.append(("group", with_context_parallel))
        return with_cp_group if with_context_parallel else no_cp_group

    monkeypatch.setattr(energon.parallel_state, "get_data_parallel_rank", fake_rank)
    monkeypatch.setattr(energon.parallel_state, "get_data_parallel_world_size", fake_world_size)
    monkeypatch.setattr(energon.parallel_state, "get_data_parallel_group", fake_group)

    rank, world_size, group = energon._dataloader_parallel_state()

    assert rank == 3
    assert world_size == 8
    assert group is no_cp_group
    assert calls == [("rank", False), ("world", False), ("group", False)]


def test_restore_loader_state_uses_cp_replicated_dp_rank(monkeypatch):
    captured = {}
    args = SimpleNamespace(load="/ckpt", dataloader_save="/data-state", iteration=42)

    monkeypatch.setattr(energon, "get_args", lambda: args)
    monkeypatch.setattr(energon, "_dataloader_parallel_state", lambda: (5, 8, object()))

    def fake_get_checkpoint_name(root, iteration, pipeline_rank=None, basename=None):
        captured["root"] = root
        captured["iteration"] = iteration
        captured["pipeline_rank"] = pipeline_rank
        captured["basename"] = basename
        return "/missing/dataloader.pt"

    monkeypatch.setattr(energon, "get_checkpoint_name", fake_get_checkpoint_name)
    monkeypatch.setattr(energon.os.path, "exists", lambda _path: False)

    energon._restore_loader_state_if_available(object())

    assert captured == {
        "root": "/data-state",
        "iteration": 42,
        "pipeline_rank": 0,
        "basename": "train_dataloader_dprank005.pt",
    }


def test_nonzero_context_parallel_rank_disables_dataloader_state_save(monkeypatch):
    args = SimpleNamespace(dataloader_save="/data-state")

    monkeypatch.setattr(energon.parallel_state, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(energon.parallel_state, "get_context_parallel_rank", lambda: 1)

    energon._disable_duplicate_dataloader_state_save_for_cp(args)

    assert args.dataloader_save is None


def test_context_parallel_rank_zero_keeps_dataloader_state_save(monkeypatch):
    args = SimpleNamespace(dataloader_save="/data-state")

    monkeypatch.setattr(energon.parallel_state, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(energon.parallel_state, "get_context_parallel_rank", lambda: 0)

    energon._disable_duplicate_dataloader_state_save_for_cp(args)

    assert args.dataloader_save == "/data-state"


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    all_special_ids = [0, 1, 2]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


def _make_task_encoder(monkeypatch, **overrides):
    args = SimpleNamespace(
        total_seq_length=8,
        seq_length=8,
        image_token_id=999,
        packing_seq_length=8,
        packing_pad_to_multiple=1,
        tensor_model_parallel_size=1,
        context_parallel_size=2,
        sequence_parallel=False,
        vision_spatial_merge_size=2,
        qwen_vl_min_pixels=None,
        qwen_vl_max_pixels=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    monkeypatch.setattr(energon, "get_args", lambda: args)
    return energon.Qwen35VLEnergonTaskEncoder(_FakeProcessor())


def _sample_with_len(length):
    return {
        "input_ids": torch.arange(length, dtype=torch.long),
        "labels": torch.arange(length, dtype=torch.long) + 100,
        "loss_mask": torch.ones(length, dtype=torch.float32),
        "pixel_values": torch.zeros(1, 4, dtype=torch.bfloat16),
        "image_grid_thw": torch.tensor([[2, 2, 2]], dtype=torch.int32),
    }


def _patch_encoder_inputs(monkeypatch, encoder, input_ids, answers):
    monkeypatch.setattr(encoder, "_build_conversation", lambda _sample: ([], answers))
    monkeypatch.setattr(
        encoder,
        "_processor_inputs",
        lambda _conversation: {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "pixel_values": torch.zeros(16, 4, dtype=torch.float32),
            "image_grid_thw": torch.tensor([[2, 2, 4]], dtype=torch.int32),
        },
    )


def test_validate_multimodal_args_rejects_packed_mbs_gt_one():
    with pytest.raises(ValueError, match="micro-batch-size 1"):
        validate_multimodal_args(
            SimpleNamespace(
                use_packed_sequence=True,
                micro_batch_size=2,
                packing_buffer_size=None,
            )
        )


def test_validate_multimodal_args_rejects_packing_buffer_without_thd():
    with pytest.raises(ValueError, match="packing-buffer-size requires"):
        validate_multimodal_args(
            SimpleNamespace(
                use_packed_sequence=False,
                micro_batch_size=1,
                packing_buffer_size=100,
            )
        )


def test_greedy_pack_accounts_for_forward_alignment_padding():
    samples = [_sample_with_len(5), _sample_with_len(3)]

    groups = energon._greedy_pack(samples, max_length=8, pad_to_multiple=4)

    assert [len(group) for group in groups] == [1, 1]


def test_pack_selected_samples_keeps_raw_segments_for_forward_padding(monkeypatch):
    encoder = _make_task_encoder(monkeypatch)
    samples = [_sample_with_len(5), _sample_with_len(3)]

    packed = encoder.pack_selected_samples(samples)

    assert [segment.numel() for segment in packed["input_ids"]] == [5, 3]
    assert [segment.numel() for segment in packed["labels"]] == [5, 3]
    assert [segment.numel() for segment in packed["loss_mask"]] == [5, 3]


def test_loss_mask_prefers_rightmost_answer_match(monkeypatch):
    encoder = _make_task_encoder(monkeypatch)
    text = "cat appears in prompt; assistant says cat"
    ids = torch.tensor([ord(ch) for ch in text], dtype=torch.long)

    mask = encoder._build_loss_mask(ids, ["cat"], "sample")

    first = text.find("cat")
    last = text.rfind("cat")
    assert mask[first : first + 3].sum().item() == 0
    assert mask[last : last + 3].sum().item() == 3


def test_loss_mask_raises_when_answer_span_is_missing(monkeypatch):
    encoder = _make_task_encoder(monkeypatch)
    ids = torch.tensor([ord(ch) for ch in "prompt only"], dtype=torch.long)

    with pytest.raises(ValueError, match="Assistant span not located"):
        encoder._build_loss_mask(ids, ["answer"], "sample")


def test_vision_alignment_accepts_matching_image_tokens_and_patches(monkeypatch):
    encoder = _make_task_encoder(monkeypatch)
    input_ids = torch.tensor([10, 999, 999, 999, 999, 11], dtype=torch.long)
    pixel_values = torch.zeros(16, 4, dtype=torch.bfloat16)
    image_grid_thw = torch.tensor([[2, 2, 4]], dtype=torch.int32)

    encoder._validate_vision_alignment(
        input_ids, pixel_values, image_grid_thw, "sample"
    )


def test_vision_alignment_rejects_truncated_image_token_block(monkeypatch):
    encoder = _make_task_encoder(monkeypatch)
    input_ids = torch.tensor([10, 999, 999, 999, 11], dtype=torch.long)
    pixel_values = torch.zeros(16, 4, dtype=torch.bfloat16)
    image_grid_thw = torch.tensor([[2, 2, 4]], dtype=torch.int32)

    with pytest.raises(ValueError, match="expects 4 merged vision tokens"):
        encoder._validate_vision_alignment(
            input_ids, pixel_values, image_grid_thw, "sample"
        )


def test_vision_alignment_rejects_pixel_grid_mismatch(monkeypatch):
    encoder = _make_task_encoder(monkeypatch)
    input_ids = torch.tensor([10, 999, 999, 999, 999, 11], dtype=torch.long)
    pixel_values = torch.zeros(15, 4, dtype=torch.bfloat16)
    image_grid_thw = torch.tensor([[2, 2, 4]], dtype=torch.int32)

    with pytest.raises(ValueError, match="expects 16 vision patches"):
        encoder._validate_vision_alignment(
            input_ids, pixel_values, image_grid_thw, "sample"
        )


def test_vision_alignment_rejects_unmerged_grid_dimensions(monkeypatch):
    encoder = _make_task_encoder(monkeypatch)
    input_ids = torch.tensor([10, 999, 999, 999, 999, 11], dtype=torch.long)
    pixel_values = torch.zeros(24, 4, dtype=torch.bfloat16)
    image_grid_thw = torch.tensor([[2, 3, 4]], dtype=torch.int32)

    with pytest.raises(ValueError, match="spatial dimensions must be divisible"):
        encoder._validate_vision_alignment(
            input_ids, pixel_values, image_grid_thw, "sample"
        )


def test_encode_sample_rejects_truncation_that_cuts_image_tokens(monkeypatch):
    encoder = _make_task_encoder(monkeypatch, total_seq_length=4, seq_length=4)
    input_ids = [10, 999, 999, 999, 999, 11] + [ord(ch) for ch in "answer"]
    _patch_encoder_inputs(monkeypatch, encoder, input_ids, ["answer"])

    with pytest.raises(ValueError, match="expects 4 merged vision tokens"):
        encoder.encode_sample(SimpleNamespace(__key__="sample"))


def test_encode_sample_rejects_truncation_that_removes_answer(monkeypatch):
    encoder = _make_task_encoder(monkeypatch, total_seq_length=10, seq_length=10)
    input_ids = [999, 999, 999, 999] + [ord(ch) for ch in "prompt answer"]
    _patch_encoder_inputs(monkeypatch, encoder, input_ids, ["answer"])

    with pytest.raises(ValueError, match="Assistant span not located"):
        encoder.encode_sample(SimpleNamespace(__key__="sample"))
