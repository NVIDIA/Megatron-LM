# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests for the heterogeneous MIMO mock-data path."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from examples.mimo.model_providers.radio_encoder import RADIO_ENCODER_MODULE_NAME
from megatron.core.packed_seq_params import PackedSeqParams


def _group(rank=0, size=1):
    return SimpleNamespace(rank=lambda: rank, size=lambda: size)


def _grid(contains_rank):
    return SimpleNamespace(is_current_rank_in_grid=lambda: contains_rank)


def _args():
    return argparse.Namespace(
        seed=123,
        dataset_provider="mock",
        micro_batch_size=2,
        llm_dp=2,
        encoder_dp=1,
        seq_length=8,
        image_seq_length=4,
        vocab_size=64,
        image_token_id=63,
        params_dtype=torch.float32,
        dynamic_resolution=False,
        patch_dim=2,
        img_h=4,
        img_w=4,
        pixel_shuffle=False,
        num_image_tiles=1,
        mock_dataset_size=16,
        disable_vision_class_token=True,
    )


def _topology(*, language_rank, encoder_rank=None):
    encoder = RADIO_ENCODER_MODULE_NAME
    grids = {"language": _grid(language_rank)}
    pgs = {"language": SimpleNamespace(pp=_group(size=3), dp=_group(rank=0, size=2))}
    if encoder_rank is not None:
        grids[encoder] = _grid(encoder_rank)
        pgs[encoder] = SimpleNamespace(pp=_group(), dp=_group(rank=1, size=2))
    return SimpleNamespace(grids=grids, module_pgs=pgs)


@pytest.fixture
def adapter(monkeypatch):
    from examples.mimo.training import data

    monkeypatch.setattr(data, "get_pg_rank", lambda pg: pg.rank())
    monkeypatch.setattr(data, "is_pp_first_stage", lambda pg: pg.rank() == 0)
    monkeypatch.setattr(data, "is_pp_last_stage", lambda pg: pg.rank() == pg.size() - 1)
    return data


def test_dynamic_radio_loader_emits_patchified_cpu_metadata(adapter):
    args = _args()
    args.micro_batch_size = 2
    args.llm_dp = 1
    args.seq_length = 24
    args.image_seq_length = 12
    args.params_dtype = torch.bfloat16
    args.dynamic_resolution = True
    args.pixel_shuffle = True
    args.patch_dim = 8
    args.img_h = 224
    args.img_w = 224
    args.num_image_tiles = 3
    loader = adapter.build_train_valid_test_data_loaders(
        args, _topology(encoder_rank=True, language_rank=False)
    )[0]

    inputs = next(iter(loader))["modality_inputs"][RADIO_ENCODER_MODULE_NAME][
        RADIO_ENCODER_MODULE_NAME
    ]
    assert inputs["x"].shape == (1, 96, 3 * 8 * 8)
    assert inputs["x"].dtype == torch.bfloat16
    assert inputs["imgs_sizes"].shape == (6, 2)
    assert inputs["imgs_sizes"].dtype == torch.int32
    assert inputs["imgs_sizes"].device.type == "cpu"
    assert torch.equal(inputs["imgs_sizes"], torch.full((6, 2), 32, dtype=torch.int32))

    packed = inputs["packed_seq_params"]
    assert isinstance(packed, PackedSeqParams)
    assert (packed.qkv_format, packed.max_seqlen_q, packed.max_seqlen_kv) == ("thd", 16, 16)
    assert packed.cu_seqlens_q.dtype == torch.int32
    assert packed.cu_seqlens_kv.dtype == torch.int32
    assert torch.equal(packed.cu_seqlens_q, torch.arange(0, 97, 16, dtype=torch.int32))
    assert torch.equal(packed.cu_seqlens_kv, packed.cu_seqlens_q)
    assert packed.cu_seqlens_q.device.type == "cpu"


def test_data_adapter_builds_independent_role_specific_loaders(adapter):
    language_loaders = adapter.build_train_valid_test_data_loaders(
        _args(), _topology(language_rank=True)
    )
    assert all(loader.batch_size == 2 for loader in language_loaders)
    assert all(loader.pin_memory for loader in language_loaders)
    assert len({id(loader.dataset) for loader in language_loaders}) == 3
    assert len({loader.dataset.seed for loader in language_loaders}) == 3
    language_batch = next(iter(language_loaders[0]))
    assert language_batch["input_ids"].shape == (2, 8)
    assert language_batch["modality_inputs"] == {}

    encoder_loaders = adapter.build_train_valid_test_data_loaders(
        _args(), _topology(encoder_rank=True, language_rank=False)
    )
    assert all(loader.batch_size == 4 for loader in encoder_loaders)
    assert all(loader.pin_memory for loader in encoder_loaders)
    encoder_batch = next(iter(encoder_loaders[0]))
    assert encoder_batch["input_ids"].shape == (4, 8)
    encoder_inputs = encoder_batch["modality_inputs"][RADIO_ENCODER_MODULE_NAME][
        RADIO_ENCODER_MODULE_NAME
    ]
    assert encoder_inputs["x"].shape == (4, 3, 4, 4)
