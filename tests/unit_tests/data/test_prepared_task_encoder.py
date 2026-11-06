# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent expanded-position target contracts for prepared image documents."""

import json
import struct
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from examples.multimodal.data_loading.conversation_sample import ImageMedia
from examples.multimodal.data_loading.image_processing import (
    DynamicResolutionImageTilingStrategy,
    ImageTilingParams,
    ImageTilingStrategy,
)
from examples.multimodal.data_loading.prepared_sample import InterleavedDocument, TokenizedText
from examples.multimodal.data_loading.prepared_task_encoder import PreparedDocumentTaskEncoder
from megatron.energon import NoCachePool, WorkerConfig, get_loader, get_train_dataset
from megatron.energon.cache import MockLazy
from megatron.energon.cache.file_store import SystemFileStore
from megatron.energon.epathlib import EPath
from megatron.energon.errors import reraise_exception
from megatron.energon.source_info import SourceInfo


class _Strategy(ImageTilingStrategy):
    def __init__(self, count=3):
        self.count = count
        self.budgets = []

    def compute_params(self, media_list, num_tokens_available, **kwargs):
        self.budgets.append(num_tokens_available)
        return [ImageTilingParams(media=media_list[0], num_tiles=1, num_embeddings=self.count)]

    def apply_params(self, params, **kwargs):
        raise AssertionError("encoding must not materialize pixels")

    def stack(self, images):
        raise AssertionError("encoding must not stack pixels")


def _encoder(strategy=None, **kwargs):
    options = dict(image_token_id=99, eos_token_id=1, vocab_size=128, image_token_budget=16)
    options.update(kwargs)
    return PreparedDocumentTaskEncoder(_Strategy() if strategy is None else strategy, **options)


def _document(parts):
    return InterleavedDocument(
        __key__="doc",
        __restore_key__=("fixture", 7),
        __sources__=(
            SourceInfo(
                dataset_path=EPath("/fixture/tokens.bin"),
                index=7,
                shard_name="tokens.bin",
                file_names=("tokens.bin",),
            ),
        ),
        __tags__={"source": "fixture"},
        parts=parts,
    )


def _image(reads=None):
    def pixels(_):
        if reads is not None:
            reads.append("pixels")
        return Image.new("RGB", (64, 64), "blue")

    return ImageMedia(
        value=MockLazy(fname="screen.png", get_fn=pixels), metadata={"width": 64, "height": 64}
    )


@pytest.mark.parametrize(
    "suffix,flags,expected_labels",
    [
        ([11, 1], None, [10, -100, -100, -100, 11, 1, -100]),
        ([11, 1], [False, True], [10, -100, -100, -100, -100, 1, -100]),
        ([8, 11, 1], None, [10, -100, -100, -100, 8, 11, 1, -100]),
        ([8, 11, 1], [False, True, True], [10, -100, -100, -100, -100, 11, 1, -100]),
        ([8, 11, 1], [True, False, True], [10, -100, -100, -100, 8, -100, 1, -100]),
    ],
)
def test_next_text_target_survives_optional_end_marker(suffix, flags, expected_labels):
    image = _image()
    source = _document(
        [TokenizedText(tokens=[3, 10]), image, TokenizedText(tokens=suffix, token_loss_mask=flags)]
    )
    encoded = _encoder().encode_sample(source)
    assert encoded.input_ids.tolist() == [3, 10, 99, 99, 99, *suffix]
    assert encoded.labels.tolist() == expected_labels
    assert encoded.loss_mask.tolist() == [float(x != -100) for x in expected_labels]
    assert encoded.image_spans == [(2, 5)]
    assert encoded.image_params[0].media is image
    assert encoded.__restore_key__ == source.__restore_key__
    assert encoded.__sources__ == source.__sources__
    assert encoded.__tags__ == source.__tags__


def test_adjacent_images_preserve_only_the_actual_next_text_target():
    encoded = _encoder().encode_sample(
        _document(
            [TokenizedText(tokens=[3, 10]), _image(), _image(), TokenizedText(tokens=[11, 1])]
        )
    )
    assert encoded.image_spans == [(2, 5), (5, 8)]
    assert encoded.input_ids.tolist() == [3, 10, *([99] * 6), 11, 1]
    assert encoded.labels.tolist() == [10, *([-100] * 6), 11, 1, -100]


@pytest.mark.parametrize("eos_eligible", [False, True])
def test_eos_immediately_after_image_retains_source_eligibility(eos_eligible):
    encoded = _encoder().encode_sample(
        _document(
            [
                TokenizedText(tokens=[3, 10]),
                _image(),
                TokenizedText(tokens=[1], token_loss_mask=[eos_eligible]),
            ]
        )
    )
    assert encoded.labels.tolist() == [10, -100, -100, -100, 1 if eos_eligible else -100, -100]


def test_default_flags_keep_all_ids_and_long_documents_untruncated():
    ids = [3, *range(10, 40), 1]
    plain = _encoder().encode_sample(_document([TokenizedText(tokens=ids)]))
    explicit = _encoder().encode_sample(
        _document([TokenizedText(tokens=ids, token_loss_mask=[True] * len(ids))])
    )
    assert plain.input_ids.tolist() == ids
    assert plain.labels.tolist() == ids[1:] + [-100]
    assert torch.equal(plain.labels, explicit.labels)
    assert torch.equal(plain.loss_mask, explicit.loss_mask)
    assert plain.image_spans == plain.image_params == []


def test_real_existing_image_strategy_plans_without_reading_pixels():
    reads = []
    image = _image(reads)

    def visual_embeddings(width, height, *, is_video):
        assert not is_video
        return (width // 16) * (height // 16) // 4

    strategy = DynamicResolutionImageTilingStrategy(
        vision_model_type="radio",
        min_num_patches=4,
        max_num_patches=16,
        patch_size=16,
        pixel_shuffle=True,
        get_num_embeddings=visual_embeddings,
        apply_data_augment=False,
    )
    encoded = _encoder(strategy).encode_sample(
        _document([TokenizedText(tokens=[3, 10]), image, TokenizedText(tokens=[11, 1])])
    )
    assert reads == []
    assert encoded.image_spans == [(2, 6)]
    assert encoded.labels.tolist() == [10, -100, -100, -100, -100, 11, 1, -100]
    assert encoded.image_params[0].media.value.get().getpixel((0, 0)) == (0, 0, 255)
    assert reads == ["pixels"]


def test_per_image_budget_is_independent_and_checked():
    strategy = _Strategy(np.int64(3))
    encoded = _encoder(strategy).encode_sample(
        _document(
            [TokenizedText(tokens=[3, 10]), _image(), _image(), TokenizedText(tokens=[11, 1])]
        )
    )
    assert strategy.budgets == [16, 16]
    assert encoded.image_spans == [(2, 5), (5, 8)]


@pytest.mark.parametrize("count", [0, -1, 17, 3.0, True])
def test_invalid_or_over_budget_image_plan_is_rejected(count):
    with pytest.raises(ValueError, match="image plan"):
        _encoder(_Strategy(count)).encode_sample(_document([_image(), TokenizedText(tokens=[1])]))


@pytest.mark.parametrize("plans", [[], [None], [None, None]])
def test_each_image_requires_one_typed_plan(plans):
    strategy = _Strategy()
    strategy.compute_params = lambda *args: plans
    with pytest.raises(ValueError, match="one image plan"):
        _encoder(strategy).encode_sample(_document([_image(), TokenizedText(tokens=[1])]))


def test_image_plan_must_keep_its_source_media():
    strategy = _Strategy()
    strategy.compute_params = lambda *args: [
        ImageTilingParams(media=_image(), num_tiles=1, num_embeddings=3)
    ]
    with pytest.raises(ValueError, match="retain its media"):
        _encoder(strategy).encode_sample(_document([_image(), TokenizedText(tokens=[1])]))


@pytest.mark.parametrize("tokens", [[3, -1, 1], [3, 128, 1], [3, True, 1], [3, 2.0, 1], [3, 99, 1]])
def test_bad_text_ids_are_rejected(tokens):
    with pytest.raises(ValueError, match="prepared text"):
        _encoder().encode_sample(_document([TokenizedText(tokens=tokens)]))


@pytest.mark.parametrize("flags", [[True], [True, 1, True]])
def test_malformed_source_ownership_is_rejected(flags):
    with pytest.raises(ValueError, match="token_loss_mask"):
        _encoder().encode_sample(
            _document([TokenizedText(tokens=[3, 10, 1], token_loss_mask=flags)])
        )


@pytest.mark.parametrize("parts", [[], [TokenizedText(tokens=[3, 10])], [_image()]])
def test_source_must_retain_terminal_eos(parts):
    with pytest.raises(ValueError, match="terminal EOS"):
        _encoder().encode_sample(_document(parts))


def test_completely_targetless_document_is_rejected():
    with pytest.raises(ValueError, match="no eligible"):
        _encoder().encode_sample(
            _document([TokenizedText(tokens=[3, 10, 1], token_loss_mask=[False, False, False])])
        )


@pytest.mark.parametrize(
    "options",
    [
        {"vocab_size": 0},
        {"image_token_budget": 0},
        {"image_token_budget": True},
        {"image_token_id": 1},
        {"image_token_id": -1},
        {"eos_token_id": 128},
        {"ignore_index": 0},
    ],
)
def test_invalid_encoding_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        _encoder(**options)


def test_actual_loss_gradients_and_eligible_denominator_follow_expected_targets():
    encoded = _encoder().encode_sample(
        _document([TokenizedText(tokens=[3, 10]), _image(), TokenizedText(tokens=[11, 1])])
    )
    expected = torch.tensor([1, 0, 0, 0, 1, 1, 0], dtype=torch.float32)
    assert torch.equal(encoded.loss_mask, expected)
    logits = torch.zeros(7, 128, requires_grad=True)
    per_token = F.cross_entropy(logits, encoded.labels, reduction="none", ignore_index=-100)
    per_token.retain_grad()
    loss = (per_token * encoded.loss_mask).sum() / encoded.loss_mask.sum()
    loss.backward()
    assert torch.allclose(loss.detach(), torch.log(torch.tensor(128.0)))
    assert torch.equal(per_token.grad, expected / 3)
    assert torch.equal(logits.grad.abs().sum(1) > 0, expected.bool())


def test_actual_mtp_heads_keep_shifted_post_image_target():
    from megatron.core.transformer import multi_token_prediction as mtp

    encoded = _encoder().encode_sample(
        _document([TokenizedText(tokens=[3, 10]), _image(), TokenizedText(tokens=[11, 1])])
    )
    targets, losses = [], []
    config = SimpleNamespace(
        mtp_num_layers=2,
        mtp_detach_heads=False,
        attention_cp_layout="zigzag",
        mtp_loss_scaling_factor=1.0,
        calculate_per_token_loss=False,
    )

    def loss(labels, logits):
        targets.append(labels.clone())
        return torch.ones_like(labels, dtype=torch.float32)

    def attach(hidden, head_loss):
        losses.append(head_loss.clone())
        return hidden

    with patch.object(mtp.MTPLossAutoScaler, "apply", side_effect=attach):
        mtp.process_mtp_loss(
            torch.zeros(21, 1, 2),
            encoded.labels[None],
            encoded.loss_mask[None],
            lambda hidden, **kwargs: (hidden, None),
            None,
            False,
            loss,
            config,
        )
    assert targets[0].tolist() == [[-100, -100, -100, 11, 1, -100, 0]]
    assert targets[1].tolist() == [[-100, -100, 11, 1, -100, 0, 0]]
    assert (losses[0] > 0).tolist() == [[False, False, False, True, True, False, False]]
    assert (losses[1] > 0).tolist() == [[False, False, True, True, False, False, False]]


def test_native_binidx_subset_yields_encoded_documents_without_pixel_reads(tmp_path, monkeypatch):
    docs = [[3, 10, 1], [3, 20, 21, 22, 1]]
    sizes = np.array([len(ids) for ids in docs], dtype="<i4")
    np.concatenate([np.array(ids, dtype="<i4") for ids in docs]).tofile(tmp_path / "tokens.bin")
    with (tmp_path / "tokens.idx").open("wb") as stream:
        stream.write(b"MMIDIDX\0\0" + struct.pack("<QBQQ", 1, 4, 2, 2))
        stream.write(sizes.tobytes())
        stream.write(np.array([0, 12], dtype="<i8").tobytes())
        stream.write(np.array([0, 1], dtype="<i8").tobytes())
    (tmp_path / "annotations").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations" / "1.json").write_text(
        json.dumps({"images": [{"position": 3, "width": 64, "height": 64, "key": "screen.png"}]})
    )
    Image.new("RGB", (64, 64), "blue").save(tmp_path / "images" / "screen.png")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "__module__: megatron.energon\n__class__: Recipe\nsplits:\n  train:\n"
        "    path: tokens.bin\n    subset: {range: [1, 2]}\n"
        "    tags: {cook: multimodal_interleaved_binidx}\n"
        "    aux:\n      metadata: filesystem://annotations\n      images: filesystem://images\n"
    )
    image_reads = []
    original_get = SystemFileStore.__getitem__

    def tracked_get(store, key):
        if str(store.get_path()) == str(tmp_path / "images"):
            image_reads.append(key)
        return original_get(store, key)

    monkeypatch.setattr(SystemFileStore, "__getitem__", tracked_get)
    dataset = get_train_dataset(
        recipe,
        worker_config=WorkerConfig(
            rank=0, world_size=1, num_workers=0, global_error_handler=reraise_exception
        ),
        task_encoder=_encoder(),
        batch_size=None,
        packing_buffer_size=None,
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        repeat=False,
    )
    loader = get_loader(dataset, cache_pool=NoCachePool())
    try:
        (document,) = list(loader)
        assert document.input_ids.tolist() == [3, 20, 21, 99, 99, 99, 22, 1]
        assert document.labels.tolist() == [20, 21, -100, -100, -100, 22, 1, -100]
        assert document.__sources__[0].index == 1
        assert document.image_spans == [(3, 6)]
        assert image_reads == []
        assert document.image_params[0].media.value.get().size == (64, 64)
        assert image_reads == ["screen.png"]
    finally:
        loader.close()
