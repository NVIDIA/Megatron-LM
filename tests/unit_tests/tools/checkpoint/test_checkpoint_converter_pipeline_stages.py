# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The checkpoint converter builds the layers of each pipeline stage it loads or saves."""

import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import megatron.training.checkpointing as checkpointing
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.state import TrainState
from tests.unit_tests.test_utilities import Utils

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))

from loader_base import MegatronCheckpointLoaderBase
from saver_base import MegatronCheckpointSaverBase
from utils import (
    build_checkpoint_converter_process_groups,
    initialize_checkpoint_converter_fake_process_groups,
)

# Two pipeline stages with three decoder layers on the first and one on the last.
LAYER_SPLITS = {
    "layout": dict(pipeline_model_parallel_layout="Ettt|tL"),
    "uneven": dict(num_layers_in_first_pipeline_stage=3, num_layers_in_last_pipeline_stage=1),
}
STAGE_LAYER_NUMBERS = [[1, 2, 3], [4]]


def _decoder_provider(split):
    """A model provider that builds the decoder on the collection the converter passes."""
    config = TransformerConfig(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        use_cpu_initialization=True,
        perform_initialization=False,
        **LAYER_SPLITS[split],
    )

    def model_provider(pre_process=True, post_process=True, pg_collection=None):
        assert pg_collection is not None, "the converter must pass its process groups"
        return TransformerBlock(config, get_gpt_layer_local_spec(), pg_collection=pg_collection)

    return model_provider


def _recording_provider(calls):
    """A model provider that records the stage and the process groups of each model."""

    def model_provider(pre_process=True, post_process=True, pg_collection=None):
        calls.append(
            dict(
                pre_process=pre_process,
                post_process=post_process,
                pg_collection=pg_collection,
                pp_rank=None if pg_collection is None else pg_collection.pp.rank(),
                tp_rank=None if pg_collection is None else pg_collection.tp.rank(),
            )
        )
        return torch.nn.Linear(1, 1)

    return model_provider


def _set_up_converter_parallel_state(tensor_parallel_size=1, pipeline_parallel_size=2):
    """Set up the process groups as the converter does and return the collection."""
    parallel_state.destroy_model_parallel()
    return initialize_checkpoint_converter_fake_process_groups(
        parallel_state,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=1,
    )


def _make_loader(pg_collection, tensor_parallel_size=1, pipeline_parallel_size=2):
    loader = MegatronCheckpointLoaderBase(args=None, queue=None)
    loader.margs = SimpleNamespace(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_parallel_size,
        virtual_pipeline_model_parallel_size=None,
    )
    loader.pg_collection = pg_collection
    return loader


def _make_saver(pg_collection, model_provider, tensor_parallel_size=1, pipeline_parallel_size=2):
    saver = MegatronCheckpointSaverBase(
        args=SimpleNamespace(
            target_tensor_parallel_size=tensor_parallel_size,
            target_pipeline_parallel_size=pipeline_parallel_size,
            target_expert_parallel_size=1,
        ),
        queue=None,
    )
    saver.md = SimpleNamespace(iteration=1, params_dtype=torch.float32)
    saver.pg_collection = pg_collection
    saver.model_provider = model_provider
    saver.models = saver.initialize_models()
    return saver


def _load_model_shards(loader, model_provider, load_checkpoint=None):
    """Run the loader's model construction without reading any checkpoint."""
    with (
        mock.patch.object(checkpointing, "load_checkpoint", side_effect=load_checkpoint),
        mock.patch("megatron.training.get_train_state", return_value=TrainState()),
    ):
        all_models, _, _ = loader.load_model_shards(model_provider, torch.float32)
    return all_models


def _save_local_models(saver):
    """Run the saver's per-rank save loop and return the keyword arguments of each save."""
    saved = []

    def save_checkpoint(iteration, models, optimizer, opt_param_scheduler, **kwargs):
        saved.append(dict(kwargs, model=models[0]))

    with (
        mock.patch.object(checkpointing, "save_checkpoint", save_checkpoint),
        mock.patch("megatron.training.get_train_state", return_value=TrainState(iteration=1)),
    ):
        saver.save_local_models_to_checkpoint()
    return saved


def _loader_stage_layer_numbers(split, pg_collection):
    """Decoder layer numbers of each stage the loader builds, without loading any weights."""
    all_models = _load_model_shards(_make_loader(pg_collection), _decoder_provider(split))
    # all_models[pp_rank][vp_rank] lists the models of the stage's TP ranks.
    return [[layer.layer_number for layer in stage[0][0].layers] for stage in all_models]


def _saver_stage_layer_numbers(split, pg_collection):
    """Decoder layer numbers of each stage the saver builds and saves."""
    saver = _make_saver(pg_collection, _decoder_provider(split))
    saved_models = {save["pipeline_rank"]: save["model"] for save in _save_local_models(saver)}
    return [
        [layer.layer_number for layer in saved_models[pp].layers] for pp in sorted(saved_models)
    ]


@pytest.fixture
def converter_parallel_state():
    # As in the saver process, torch.distributed is initialized, so models see the fake groups.
    Utils.initialize_distributed()
    yield _set_up_converter_parallel_state()
    parallel_state.destroy_model_parallel()


@pytest.fixture
def no_global_process_groups():
    """No global process groups exist, so only the converter's collection can be used."""
    parallel_state.destroy_model_parallel()
    yield
    parallel_state.destroy_model_parallel()


@pytest.mark.parametrize("split", LAYER_SPLITS)
def test_loader_builds_each_pipeline_stage(converter_parallel_state, split):
    assert _loader_stage_layer_numbers(split, converter_parallel_state) == STAGE_LAYER_NUMBERS


@pytest.mark.parametrize("split", LAYER_SPLITS)
def test_saver_builds_each_pipeline_stage(converter_parallel_state, split):
    assert _saver_stage_layer_numbers(split, converter_parallel_state) == STAGE_LAYER_NUMBERS


def test_loader_builds_models_on_its_collection(no_global_process_groups):
    """The loader selects every rank on its collection and passes it to the model and loader."""
    pg_collection = build_checkpoint_converter_process_groups(
        tensor_parallel_size=2, pipeline_parallel_size=2, expert_parallel_size=1
    )
    loader = _make_loader(pg_collection, tensor_parallel_size=2, pipeline_parallel_size=2)
    calls, loads = [], []

    def load_checkpoint(model, optimizer, opt_param_scheduler, **kwargs):
        loads.append(dict(kwargs, pp_rank=pg_collection.pp.rank(), tp_rank=pg_collection.tp.rank()))

    _load_model_shards(loader, _recording_provider(calls), load_checkpoint)

    stages = [(c["pre_process"], c["post_process"], c["pp_rank"], c["tp_rank"]) for c in calls]
    assert stages == [
        (True, False, 0, 0),
        (True, False, 0, 1),
        (False, True, 1, 0),
        (False, True, 1, 1),
    ]
    assert all(c["pg_collection"] is pg_collection for c in calls)
    assert [(load["pp_rank"], load["tp_rank"]) for load in loads] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for load in loads:
        assert load["tp_group"] is pg_collection.tp
        assert load["pp_group"] is pg_collection.pp
        assert load["dp_cp_group"] is pg_collection.dp_cp_gtp_remat
        assert load["dp_group"] is pg_collection.dp
        assert load["expt_dp_group"] is pg_collection.expt_dp


def test_saver_builds_models_on_its_collection(no_global_process_groups):
    """The saver selects every stage on its collection and passes it to the model and saver."""
    pg_collection = build_checkpoint_converter_process_groups(
        tensor_parallel_size=2, pipeline_parallel_size=2, expert_parallel_size=1
    )
    calls = []
    saver = _make_saver(
        pg_collection, _recording_provider(calls), tensor_parallel_size=2, pipeline_parallel_size=2
    )

    saved = _save_local_models(saver)

    stages = [(c["pre_process"], c["post_process"], c["pp_rank"]) for c in calls]
    assert stages == [(True, False, 0), (True, False, 0), (False, True, 1), (False, True, 1)]
    assert all(c["pg_collection"] is pg_collection for c in calls)
    assert [(save["pipeline_rank"], save["tensor_rank"]) for save in saved] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for save in saved:
        assert save["tp_group"] is pg_collection.tp
        assert save["pp_group"] is pg_collection.pp
        assert save["cp_group"] is pg_collection.cp
        assert save["dp_cp_group"] is pg_collection.dp_cp_gtp_remat
        assert save["dp_group"] is pg_collection.dp
        assert save["dp_gtp_remat_group"] is pg_collection.dp_gtp_remat
        assert save["expt_dp_group"] is pg_collection.expt_dp


def test_loader_reads_each_legacy_checkpoint_file():
    """Legacy checkpoint files are named after the global ranks, which follow the collection."""
    pg_collection = _set_up_converter_parallel_state(tensor_parallel_size=2)
    loader = _make_loader(pg_collection, tensor_parallel_size=2, pipeline_parallel_size=2)
    checkpoint_names = []

    def load_checkpoint(model, optimizer, opt_param_scheduler, **kwargs):
        checkpoint_names.append(checkpointing.get_checkpoint_name("ckpt", 7))

    try:
        _load_model_shards(loader, _recording_provider([]), load_checkpoint)
    finally:
        parallel_state.destroy_model_parallel()

    assert checkpoint_names == [
        os.path.join("ckpt", "iter_0000007", f"mp_rank_{tp:02d}_{pp:03d}", "model_optim_rng.pt")
        for pp in range(2)
        for tp in range(2)
    ]


def test_loader_process_builds_each_pipeline_stage():
    """The command-line loader runs in a process where torch.distributed is not initialized."""
    script = "\n".join(
        [
            "import json",
            "import torch",
            f"import {__name__} as t",
            "from utils import initialize_checkpoint_converter_distributed",
            "assert not torch.distributed.is_initialized()",
            "initialize_checkpoint_converter_distributed()",
            "pg_collection = t._set_up_converter_parallel_state()",
            "print(json.dumps({s: t._loader_stage_layer_numbers(s, pg_collection) "
            "for s in t.LAYER_SPLITS}))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stdout + result.stderr
    stage_layer_numbers = json.loads(result.stdout.strip().splitlines()[-1])
    assert stage_layer_numbers == {split: STAGE_LAYER_NUMBERS for split in LAYER_SPLITS}
