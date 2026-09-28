# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Focused coverage for multimodal training metadata and context parallelism."""

import logging
import math
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core import _rank_utils
from megatron.core.models.multimodal import context_parallel, llava_model
from megatron.core.models.multimodal.llava_model import (
    IGNORE_INDEX,
    LLaVAModel,
    _precalculate_loss_weights,
)
from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def device():
    """Use the device selected by the distributed unit-test launcher."""
    return torch.device("cuda", torch.cuda.current_device())


class _RecordingLanguageModel(torch.nn.Module):
    """Expose the tensors delivered to the LM without running transformer layers."""

    def embedding(self, input_ids, position_ids):
        values = input_ids.float()
        return torch.stack((values, -values), dim=-1).transpose(0, 1).contiguous()

    def forward(self, **kwargs):
        self.inputs = kwargs
        return kwargs["decoder_input"]


class _PatchIdentityEncoder(torch.nn.Module):
    """Keep patch identity visible, including nonzero outputs for zero-valued dummies."""

    dynamic_resolution = True
    patch_dim = 2
    class_token_len = 0

    def forward(self, images, *, imgs_sizes, packed_seq_params):
        self.imgs_sizes = imgs_sizes.clone()
        return images[..., :2].contiguous() + 1000


def _make_model(cp_size=1):
    """Build only the state needed by the real forward and preprocessing methods."""
    model = object.__new__(LLaVAModel)
    torch.nn.Module.__init__(model)
    model.add_encoder = False
    model.add_decoder = True
    model.pre_process = True
    model.post_process = True
    model.encoder_hidden_state = None
    model.language_model = _RecordingLanguageModel()
    model.vision_model = None
    model.vision_projection = torch.nn.Identity()
    model.sound_model = None
    model.sound_projection = None
    model.image_token_index = -200
    model.sound_token_index = -300
    model.temporal_patch_dim = 1
    model.patch_dim = 2
    model.img_seq_len = 1
    model.dynamic_resolution = True
    model.context_parallel_lm = cp_size
    model.sequence_parallel_lm = False
    model.pg_collection = SimpleNamespace(tp=None)
    model.use_loss_scaling = False
    model._drop_vision_class_token = True
    model._pixel_shuffle = False
    model._conv_merging = False
    model._tile_tags = None
    model._max_num_tiles = 1
    model._language_max_sequence_length = 128
    model._language_is_pipeline_parallel = False
    model._vision_fp8 = False
    model._vision_fp8_recipe = None
    model._vision_projection_fp8 = False
    model._balance_vision_context_parallel_by_tokens = False
    model._profile_vision_context_parallel_partition = False
    # These tests compare metadata immediately before LM sharding. The actual
    # dynamic-resolution vision split/gather and preprocessing remain unmocked.
    model._process_embedding_token_parallel = mock.Mock(side_effect=lambda *args: args)
    return model


def _text_forward(model, labels, packed_seq_params=None):
    """Run the real forward path with text-only inputs and supplied target labels."""
    input_ids = torch.arange(labels.numel(), device=labels.device).reshape_as(labels)
    return model(
        images=None,
        input_ids=input_ids,
        position_ids=None,
        attention_mask=None,
        labels=labels,
        loss_mask=(labels != IGNORE_INDEX).float(),
        packed_seq_params=packed_seq_params,
    )


@pytest.mark.parametrize("unfreeze_router", [False, True])
@pytest.mark.parametrize("freeze_language_model", [False, True])
def test_freeze_preserves_router_trainability(device, unfreeze_router, freeze_language_model):
    model = _make_model()
    model.language_model = torch.nn.ModuleDict(
        {"router": torch.nn.Linear(2, 2), "experts": torch.nn.Linear(2, 2)}
    ).to(device)
    model.vision_model = torch.nn.Linear(2, 2).to(device)
    model.vision_projection = torch.nn.Linear(2, 2).to(device)

    model.freeze(
        freeze_language_model=freeze_language_model,
        freeze_vision_model=True,
        freeze_vision_projection=False,
        unfreeze_router=unfreeze_router,
    )

    for param in model.language_model["router"].parameters():
        assert param.requires_grad == (not freeze_language_model or unfreeze_router)
    for param in model.language_model["experts"].parameters():
        assert param.requires_grad == (not freeze_language_model)
    assert all(not param.requires_grad for param in model.vision_model.parameters())
    assert all(param.requires_grad for param in model.vision_projection.parameters())


@pytest.mark.parametrize("boundary_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    ("boundaries", "labels", "expected"),
    [
        (
            [0, 2, 6, 8],
            [1, -100, 2, 3, 4, 5, -100, 6],
            [0.25, 0, 0.125, 0.125, 0.125, 0.125, 0, 0.25],
        ),
        # An entirely ignored segment contributes neither weights nor normalization.
        ([0, 2, 4], [-100, -100, 1, 2], [0, 0, 0.5, 0.5]),
        # Truncated final segments and repeated boundaries are safe.
        ([0, 2, 9], [1, -100, 2], [0.5, 0, 0.5]),
        ([0, 0, 2, 3], [1, 2, -100], [0.5, 0.5, 0]),
    ],
)
def test_precalculate_loss_weights(device, boundary_dtype, boundaries, labels, expected):
    actual = _precalculate_loss_weights(
        torch.tensor(boundaries, dtype=boundary_dtype, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )
    torch.testing.assert_close(actual, torch.tensor(expected, device=device), rtol=1e-6, atol=0)
    torch.testing.assert_close(actual.sum(), actual.new_tensor(1.0))


def test_precalculate_loss_weights_rejects_all_ignored_labels(device):
    with pytest.raises(RuntimeError, match="no loss-bearing tokens"):
        _precalculate_loss_weights(
            torch.tensor([0, 2, 4], dtype=torch.int32, device=device),
            torch.full((4,), IGNORE_INDEX, dtype=torch.long, device=device),
        )


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("use_loss_scaling", [False, True])
@pytest.mark.parametrize("boundary_kind", ["none", "unpadded", "padded"])
def test_forward_loss_scaling_before_cp_sharding(device, cp_size, use_loss_scaling, boundary_kind):
    model = _make_model(cp_size)
    model.use_loss_scaling = use_loss_scaling
    labels = torch.tensor([[1, -100, 2, 3, 4, 5, -100, 6]], device=device)
    packed = None
    if boundary_kind != "none":
        cu = torch.tensor(
            [0, 1, 5, 7] if boundary_kind == "padded" else [0, 3, 8],
            dtype=torch.int32,
            device=device,
        )
        cu_padded = (
            torch.tensor([0, 2, 6, 8], dtype=torch.int32, device=device)
            if boundary_kind == "padded"
            else None
        )
        packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_q_padded=cu_padded)

    expected = (labels != IGNORE_INDEX).float()
    if cp_size > 1 and use_loss_scaling:
        if boundary_kind == "padded":
            values = [0.25, 0, 0.125, 0.125, 0.125, 0.125, 0, 0.25]
        elif boundary_kind == "unpadded":
            a = 1 / (math.sqrt(2) * (math.sqrt(2) + 2))
            b = 1 / (2 * (math.sqrt(2) + 2))
            values = [a, 0, a, b, b, b, 0, b]
        else:
            values = [1 / 6, 0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0, 1 / 6]
        expected = torch.tensor([values], device=device)

    def shard(embeddings, expanded_labels, weights, params):
        # Assert full-pack normalization is already applied when sharding starts.
        torch.testing.assert_close(weights, expected)
        assert params is packed
        return embeddings[:, :4], expanded_labels[:, :4], weights[:, :4], params

    model._process_embedding_token_parallel.side_effect = shard
    _, weights = _text_forward(model, labels, packed)
    torch.testing.assert_close(weights, expected[:, :4] if cp_size > 1 else expected)
    assert model._process_embedding_token_parallel.call_count == int(cp_size > 1)


def test_forward_cp_loss_scaling_requires_micro_batch_size_one(device):
    model = _make_model(cp_size=2)
    model.use_loss_scaling = True
    with pytest.raises(AssertionError, match="micro-batch-size 1"):
        _text_forward(model, torch.ones((2, 4), dtype=torch.long, device=device))
    model._process_embedding_token_parallel.assert_not_called()


def test_forward_cp_loss_scaling_rejects_all_ignored_labels(device):
    model = _make_model(cp_size=2)
    model.use_loss_scaling = True
    with pytest.raises(RuntimeError, match="no loss-bearing tokens"):
        _text_forward(model, torch.full((1, 4), IGNORE_INDEX, dtype=torch.long, device=device))
    model._process_embedding_token_parallel.assert_not_called()


@pytest.fixture
def vision_cp_group():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)
    try:
        yield
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
@pytest.mark.parametrize("balance_by_tokens", [False, True])
@pytest.mark.parametrize("dummy_rank", [False, True])
def test_dynamic_vision_cp_matches_unsharded_preprocessing(
    device, vision_cp_group, balance_by_tokens, dummy_rank
):
    sizes = [[4, 8]] if dummy_rank else [[4, 4], [4, 4], [4, 8]]
    frames = [1] if dummy_rank else [2, 1]
    expected_counts = [8] if dummy_rank else [8, 8]
    input_ids = torch.tensor(
        [[11, -200, 12] if dummy_rank else [11, -200, 12, -200, 13]], device=device
    )
    total_patches = sum(expected_counts)
    images = torch.arange(total_patches * 12, dtype=torch.float32, device=device).reshape(
        1, total_patches, 12
    )
    kwargs = dict(
        images=images,
        input_ids=input_ids,
        position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0),
        attention_mask=None,
        labels=input_ids.roll(-1, dims=1),
        loss_mask=torch.ones_like(input_ids, dtype=torch.float32),
        imgs_sizes=torch.tensor(sizes, dtype=torch.int32, device=device),
        num_frames=frames,
    )
    models = []
    outputs = []
    masks = []
    for cp_size in (1, 2):
        model = _make_model(cp_size)
        model.add_encoder = True
        model.vision_model = _PatchIdentityEncoder()
        model._balance_vision_context_parallel_by_tokens = balance_by_tokens
        model._preprocess_data = mock.Mock(wraps=model._preprocess_data)
        output, loss_mask = model(**kwargs)
        models.append(model)
        outputs.append(output if cp_size == 1 else output.transpose(0, 1))
        masks.append(loss_mask)
        preprocess = model._preprocess_data.call_args
        assert preprocess.kwargs["media_token_counts"].tolist() == expected_counts
        torch.testing.assert_close(
            preprocess.args[0], (images[..., :2] + 1000).transpose(0, 1), rtol=0, atol=0
        )

    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
    torch.testing.assert_close(masks[1], masks[0], rtol=0, atol=0)
    for key in ("labels", "input_ids"):
        torch.testing.assert_close(
            models[1].language_model.inputs[key],
            models[0].language_model.inputs[key],
            rtol=0,
            atol=0,
        )
    models[0]._process_embedding_token_parallel.assert_not_called()
    models[1]._process_embedding_token_parallel.assert_called_once()
    if dummy_rank and context_parallel.get_context_parallel_rank() == 1:
        assert models[1].vision_model.imgs_sizes.tolist() == [[2, 2]]


@pytest.mark.parametrize("profile_enabled", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_vision_profiling_gate_uses_model_tp_group(device, monkeypatch, profile_enabled, tp_rank):
    model = _make_model(cp_size=2)
    model.add_encoder = True
    model.vision_model = _PatchIdentityEncoder()
    model._profile_vision_context_parallel_partition = profile_enabled
    model.pg_collection.tp = object()
    rank_lookup = mock.Mock(return_value=tp_rank)
    monkeypatch.setattr(llava_model, "get_pg_rank", rank_lookup)
    splitter = mock.Mock(side_effect=RuntimeError("splitter reached"))
    monkeypatch.setattr(llava_model, "split_to_context_parallel_ranks_dynamic_res", splitter)

    with pytest.raises(RuntimeError, match="splitter reached"):
        model(
            images=torch.ones(1, 4, 12, device=device),
            input_ids=torch.tensor([[model.image_token_index]], device=device),
            position_ids=None,
            attention_mask=None,
            imgs_sizes=torch.tensor([[4, 4]], device=device),
        )
    assert splitter.call_args.kwargs["profile_partition"] == (profile_enabled and tp_rank == 0)
    if profile_enabled:
        rank_lookup.assert_called_once_with(model.pg_collection.tp)
    else:
        rank_lookup.assert_not_called()


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("profile_enabled", [False, True])
def test_partition_profile_logging_preserves_nonzero_global_rank(
    device, monkeypatch, caplog, cp_rank, profile_enabled
):
    monkeypatch.setattr(context_parallel, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(context_parallel, "get_context_parallel_rank", lambda: cp_rank)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 6)
    monkeypatch.setattr(_rank_utils, "safe_get_rank", lambda: 6)
    caplog.set_level(logging.INFO, logger=context_parallel.__name__)
    cu = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)

    context_parallel.split_to_context_parallel_ranks_dynamic_res(
        torch.ones(1, 8, 12, device=device),
        torch.tensor([[4, 4], [4, 4]], dtype=torch.int32, device=device),
        PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu),
        patch_dim=2,
        profile_partition=profile_enabled,
    )

    messages = [
        r.getMessage() for r in caplog.records if "VISION_CP_PARTITION_PROFILE" in r.message
    ]
    assert len(messages) == int(profile_enabled and cp_rank == 0)
    if messages:
        assert "global_rank=6" in messages[0]
        assert "loads=4,4" in messages[0]
