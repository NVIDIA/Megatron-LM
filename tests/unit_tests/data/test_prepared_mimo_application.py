# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prepared MIMO batches retain target ownership and realize pixels on encoder roles."""

from argparse import Namespace

import pytest
import torch
from PIL import Image

from examples.mimo.training import prepared_binidx
from examples.mimo.training.prepared_binidx import make_config
from examples.multimodal.data_loading.conversation_sample import ImageMedia
from examples.multimodal.data_loading.prepared_sample import InterleavedDocument, TokenizedText
from megatron.energon import NoCachePool
from megatron.energon.cache import MockLazy
from megatron.energon.cache.file_store import SystemFileStore
from megatron.energon.decoders.sample_decoder import SampleDecoder
from megatron.energon.epathlib import EPath


@pytest.fixture(autouse=True)
def runtime_tokenizer_config(monkeypatch):
    from megatron.training.config.training_config import TokenizerConfig

    config = TokenizerConfig(vocab_size=128, padded_vocab_size=128)
    monkeypatch.setattr(prepared_binidx, "get_run_config", lambda: Namespace(tokenizer=config))
    return config


def _args(**updates):
    options = dict(
        dynamic_resolution=True,
        disable_vision_class_token=True,
        patch_dim=16,
        pixel_shuffle=True,
        energon_image_token_budget=4,
        seq_length=16,
        image_token_id=99,
        energon_eos_token_id=1,
        energon_pad_token_id=0,
        vocab_size=128,
    )
    options.update(updates)
    return Namespace(**options)


def _pack(encoder, number, reads, *, with_image=True):
    parts = [TokenizedText(tokens=[3, number])]
    if with_image:
        image = ImageMedia(
            value=MockLazy(
                fname=f"{number}.png",
                get_fn=lambda _: (
                    reads.append(number) or Image.new("RGB", (64, 64), (number, 0, 0))
                ),
            ),
            metadata={"width": 64, "height": 64},
        )
        parts.append(image)
    parts.append(TokenizedText(tokens=[number + 1, 1]))
    doc = InterleavedDocument(
        __key__=str(number),
        __restore_key__=("fixture", number),
        __sources__=None,
        __tags__={},
        parts=parts,
    )
    encoded = encoder.encode_sample(doc)
    selection = encoder.select_next_pack(iter([encoded]))
    return encoder.pack_selected_samples(selection.packs[0])


def test_batch_stacks_once_shifted_targets_without_pixel_reads():
    config = make_config(_args())
    encoder = config.create_task_encoder(split="train")
    reads = []
    pack = _pack(encoder, 10, reads)
    batch = encoder.batch([pack])
    assert batch["input_ids"].shape == (1, 16)
    assert batch["input_ids"][0, :8].tolist() == [3, 10, 99, 99, 99, 99, 11, 1]
    assert batch["labels"][0, :8].tolist() == [10, -100, -100, -100, -100, 11, 1, -100]
    assert torch.equal(batch["labels"][0], pack.labels)
    assert torch.equal(batch["loss_mask"][0], pack.loss_mask)
    assert batch["position_ids"].tolist() == [list(range(16))]
    assert reads == []


def test_encoder_merger_realizes_lane_then_row_order_without_mutating_lazy_plans():
    config = make_config(_args())
    encoder = config.create_task_encoder(split="train")
    reads = []
    batches = [
        encoder.batch([_pack(encoder, n, reads), _pack(encoder, n + 2, reads)]) for n in (10, 20)
    ]
    original = [row["plans"][0] for b in batches for row in b["modality_inputs"]["prepared_images"]]
    values = [plan.media.value for plan in original]
    result = config.merge_encoder_batches(batches, lane_ids=(0, 1), module_name="radio_encoder")
    assert reads == [10, 12, 20, 22]
    assert result["input_ids"].shape == (4, 16)
    assert result["input_ids"][:, 1].tolist() == [10, 12, 20, 22]
    assert [plan.media.value for plan in original] == values
    assert all(isinstance(plan.media.value, MockLazy) for plan in original)
    vision = result["modality_inputs"]["radio_encoder"]["radio_encoder"]
    assert vision["x"].shape == (1, 64, 3 * 16 * 16)
    assert vision["imgs_sizes"].tolist() == [[64, 64]] * 4
    assert vision["imgs_sizes"].device.type == "cpu"
    packed = vision["packed_seq_params"]
    assert packed.cu_seqlens_q.tolist() == [0, 16, 32, 48, 64]
    assert packed.cu_seqlens_kv.tolist() == packed.cu_seqlens_q.tolist()
    assert packed.max_seqlen_q == 16
    assert int((result["input_ids"] == 99).sum()) == 16


def test_text_only_encoder_batch_has_no_modality_inputs():
    config = make_config(_args())
    encoder = config.create_task_encoder(split="train")
    batch = encoder.batch([_pack(encoder, 10, [], with_image=False)])
    result = config.merge_encoder_batches([batch], lane_ids=(0,), module_name="radio_encoder")
    assert "modality_inputs" not in result


def test_geometry_mismatch_rejects_before_encoder_forward():
    config = make_config(_args())
    encoder = config.create_task_encoder(split="train")
    batch = encoder.batch([_pack(encoder, 10, [])])
    batch["modality_inputs"]["prepared_images"][0]["spans"] = [(2, 5)]
    with pytest.raises(ValueError, match="geometry disagrees"):
        config.merge_encoder_batches([batch], lane_ids=(0,), module_name="radio_encoder")


@pytest.mark.parametrize(
    "changes",
    [
        {"dynamic_resolution": False},
        {"disable_vision_class_token": False},
        {"energon_eos_token_id": None},
        {"energon_pad_token_id": None},
        {"energon_image_token_budget": 0},
        {"energon_image_token_budget": 17},
    ],
)
def test_application_policy_fails_before_source_access(changes):
    with pytest.raises(ValueError):
        make_config(_args(**changes))


def test_checkpoint_identity_captures_task_geometry_and_objective_configuration():
    baseline = make_config(_args()).checkpoint_identity
    assert baseline == make_config(_args()).checkpoint_identity
    for changes in (
        {"seq_length": 32},
        {"energon_pad_token_id": 2},
        {"pixel_shuffle": False},
        {"patch_dim": 14},
        {"energon_image_token_budget": 8, "seq_length": 32},
    ):
        assert baseline != make_config(_args(**changes)).checkpoint_identity


def test_original_dimension_metadata_mismatch_rejects():
    config = make_config(_args())
    encoder = config.create_task_encoder(split="train")
    batch = encoder.batch([_pack(encoder, 10, [])])
    plan = batch["modality_inputs"]["prepared_images"][0]["plans"][0]
    plan.media.metadata = {"width": 64, "height": 32}
    with pytest.raises(ValueError, match="original width/height metadata"):
        config.merge_encoder_batches([batch], lane_ids=(0,), module_name="radio_encoder")


def test_native_direct_lazy_handle_stays_unmaterialized_in_original_batch(tmp_path):
    image_path = tmp_path / "actual.png"
    Image.new("RGB", (64, 64), "red").save(image_path)
    store = SystemFileStore(EPath(tmp_path))
    lazy = NoCachePool().get_lazy(store, "actual.png")
    # Keep the real DirectLazy cache owner; decode at the native file-store boundary.
    original = store.__class__.__getitem__

    class DecodedStore(SystemFileStore):
        def __getitem__(self, key):
            raw, source = original(self, key)
            return SampleDecoder(image_decode="pil").decode(key, raw), source

    lazy.ds = DecodedStore(EPath(tmp_path))
    config = make_config(_args())
    encoder = config.create_task_encoder(split="train")
    pack = _pack(encoder, 10, [])
    pack.image_params[0].media.value = lazy
    batch = encoder.batch([pack])
    assert lazy._data is None
    result = config.merge_encoder_batches([batch], lane_ids=(0,), module_name="radio_encoder")
    assert result["modality_inputs"]["radio_encoder"]["radio_encoder"]["x"].shape == (1, 16, 768)
    assert lazy._data is None


def test_factory_uses_initialized_runtime_vocabulary_instead_of_stale_cli(runtime_tokenizer_config):
    runtime_tokenizer_config.padded_vocab_size = 256
    config = make_config(_args(vocab_size=None))
    assert config.create_task_encoder(split="train").vocab_size == 256
