# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts for prepared Energon batches and heterogeneous MIMO data lanes."""

import argparse
from types import SimpleNamespace

import pytest
import torch


def test_collate_fixed_images():
    from examples.mimo.training.batch import collate_encoder_inputs

    first = torch.zeros(1, 3, 4, 4)
    second = torch.ones(2, 3, 4, 4)
    result = collate_encoder_inputs([{"x": first}, {"x": second}])
    assert set(result) == {"x"}
    assert torch.equal(result["x"], torch.cat((first, second)))


def test_collate_dynamic_images():
    from examples.mimo.training.batch import collate_encoder_inputs

    first = {"x": torch.zeros(1, 4, 12), "imgs_sizes": torch.tensor([[4, 4]])}
    second = {"x": torch.ones(1, 8, 12), "imgs_sizes": torch.tensor([[4, 8]])}
    result = collate_encoder_inputs([first, second])
    assert result["x"].shape == (1, 12, 12)
    assert torch.equal(result["x"][:, :4], first["x"])
    assert torch.equal(result["x"][:, 4:], second["x"])
    assert result["imgs_sizes"].device.type == "cpu"
    packed = result["packed_seq_params"]
    assert packed.qkv_format == "thd"
    assert torch.equal(packed.cu_seqlens_q, torch.tensor([0, 4, 12], dtype=torch.int32))
    assert torch.equal(packed.cu_seqlens_q, packed.cu_seqlens_kv)
    assert packed.max_seqlen_q == packed.max_seqlen_kv == 8


@pytest.mark.parametrize("dimensions", [[[3, 4]], [[0, 4]], [[4, 8]], [[4.0, 4.0]]])
def test_collate_rejects_invalid_image_metadata(dimensions):
    from examples.mimo.training.batch import collate_encoder_inputs

    with pytest.raises(ValueError):
        collate_encoder_inputs(
            [{"x": torch.zeros(1, 4, 12), "imgs_sizes": torch.tensor(dimensions)}]
        )


def test_mock_registration_and_configuration_unchanged():
    from examples.mimo.data.energon_provider import configure_energon_data
    from examples.mimo.training.data import add_mimo_data_args, add_mock_data_args

    original = vars(add_mock_data_args(argparse.ArgumentParser()).parse_args([]))
    current = add_mimo_data_args(argparse.ArgumentParser()).parse_args([])
    assert {name: getattr(current, name) for name in original} == original
    before = vars(current).copy()
    configure_energon_data(current)
    assert vars(current) == before
    assert not hasattr(current, "image_tag_type")


def _args(**overrides):
    values = dict(
        dataset_provider="energon",
        mimo_llm_tp=1,
        mimo_llm_pp=1,
        mimo_llm_cp=1,
        mimo_encoder_tp=1,
        gtp_weight_remat_size=1,
        micro_batch_size=1,
        num_workers=0,
        eval_iters=0,
        mimo_encoder_dp=1,
        mimo_llm_dp=2,
        mimo_llm_only=False,
        dynamic_resolution=True,
        disable_vision_class_token=True,
        patch_dim=16,
        pixel_shuffle=True,
        mimo_encoder_prefetch=False,
        tokenizer_type="MultimodalTokenizer",
        data_path=["recipe.yaml"],
        dataloader_save=None,
        save=None,
        vocab_size=None,
        seq_length=8,
        image_token_id=18,
        seed=17,
        multimodal_interleaved_min_image_patches=4,
        multimodal_interleaved_max_image_patches=4096,
        multimodal_interleaved_shuffle_buffer_documents=0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mimo_llm_cp", 2),
        ("micro_batch_size", 2),
        ("dynamic_resolution", False),
        ("pixel_shuffle", False),
        ("num_workers", -1),
        ("mimo_encoder_prefetch", True),
    ],
)
def test_unsupported_provider_settings_fail_before_tokenizer(field, value, monkeypatch):
    from examples.mimo.data.energon_provider import configure_energon_data
    from megatron.training import global_vars

    def unexpected(*args):
        pytest.fail("argument validation must not initialize a tokenizer")

    monkeypatch.setattr(global_vars, "rebuild_tokenizer", unexpected)
    with pytest.raises(ValueError):
        configure_energon_data(_args(**{field: value}))


def test_tokenizer_initialization_uses_public_config_signature(monkeypatch):
    from examples.mimo.data.energon_provider import (
        configure_energon_data,
        initialize_energon_tokenizer,
    )
    from megatron.training import global_vars

    args = _args()
    configure_energon_data(args)
    cfg = object()
    calls = []

    def rebuild(actual_cfg, actual_args):
        calls.append((actual_cfg, actual_args))
        return SimpleNamespace(vocab={"<image>": 18}, pad=0, vocab_size=64)

    monkeypatch.setattr(global_vars, "rebuild_tokenizer", rebuild)
    initialize_energon_tokenizer(cfg, args)
    assert calls == [(cfg, args)]
    assert (args.vocab_size, args.image_token_id, args.pad_token_id) == (64, 18, 0)


def test_language_batch_expansion_masks_and_boundaries():
    from examples.mimo.data.energon_provider import MimoInterleavedTaskEncoder

    encoder = MimoInterleavedTaskEncoder.__new__(MimoInterleavedTaskEncoder)
    encoder.config = SimpleNamespace(sequence_length=8, image_token_id=18, ignore_index=-100)
    encoder.role = "language"
    pack = SimpleNamespace(
        tokens=torch.tensor([9, 18, 10, 11, 0]),
        labels=torch.tensor([-100, -100, 11, -100, -100]),
        image_params=[SimpleNamespace(num_embeddings=4)],
        cu_seqlens=torch.tensor([0, 7, 8], dtype=torch.int32),
        __restore_key__=("source", 3),
    )
    result = encoder.batch([pack])
    assert result["input_ids"].tolist() == [[9, 18, 18, 18, 18, 10, 11, 0]]
    assert result["labels"].tolist() == [[-100, -100, -100, -100, -100, 11, -100, -100]]
    assert result["loss_mask"].tolist() == [[0, 0, 0, 0, 0, 1, 0, 0]]
    assert result["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5, 6, 0]]
    assert result["cu_seqlens"].tolist() == [[0, 7, 8]]
    assert result["max_seqlen"].tolist() == [7]
    pack.image_params = []
    with pytest.raises(ValueError, match="more image slots"):
        encoder.batch([pack])


class _Lane:
    def __init__(self, token):
        self.token = token
        self.closed = False

    def __next__(self):
        return {"input_ids": torch.tensor([[self.token]])}

    def close(self):
        self.closed = True


def test_encoder_lane_order_empty_images_and_non_authoritative_state():
    from examples.mimo.data.energon_provider import EncoderLaneIterator

    lanes = [_Lane(11), _Lane(21)]
    iterator = EncoderLaneIterator(lanes, "radio_encoder")
    assert next(iterator)["input_ids"].tolist() == [[11], [21]]
    assert iterator.save_state() is None
    iterator.close()
    assert all(lane.closed for lane in lanes)


def test_partial_encoder_lane_construction_closes_created_loaders(monkeypatch):
    import megatron.training
    from examples.mimo.data import energon_provider as provider

    tokenizer = SimpleNamespace(eod=2, pad=0)
    monkeypatch.setattr(megatron.training, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(provider, "get_pg_rank", lambda _: 0)
    created = []

    def build(*args, **kwargs):
        if kwargs["lane"] == 1:
            raise RuntimeError("second lane failed")
        lane = _Lane(11)
        created.append(lane)
        assert kwargs["lanes"] == 2 and not kwargs["checkpoint_enabled"]
        return lane

    monkeypatch.setattr(provider, "build_interleaved_iterator", build)
    monkeypatch.setattr(
        provider, "restore_interleaved_iterator", lambda iterator, *a, **k: iterator
    )
    topology = SimpleNamespace(
        grids={
            "language": SimpleNamespace(is_current_rank_in_grid=lambda: False),
            "radio_encoder": SimpleNamespace(is_current_rank_in_grid=lambda: True),
        },
        module_pgs={"radio_encoder": SimpleNamespace(dp=object())},
    )
    with pytest.raises(RuntimeError, match="second lane failed"):
        provider.build_energon_data_loaders(_args(), topology)
    assert len(created) == 1 and created[0].closed
