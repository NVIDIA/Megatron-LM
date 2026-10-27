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

    def model_provider(pre_process=True, post_process=True, vp_stage=None, pg_collection=None):
        assert pg_collection is not None, "the converter must pass its process groups"
        return TransformerBlock(
            config, get_gpt_layer_local_spec(), pg_collection=pg_collection, vp_stage=vp_stage
        )

    return model_provider


def _interleaved_decoder_provider(calls):
    """A model provider for eight decoder layers on two pipeline stages of two virtual stages.

    It records the stage of each model chunk it builds.
    """
    config = TransformerConfig(
        num_layers=8,
        hidden_size=64,
        num_attention_heads=4,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
        use_cpu_initialization=True,
        perform_initialization=False,
    )

    def model_provider(pre_process=True, post_process=True, vp_stage=None, pg_collection=None):
        calls.append(
            dict(
                pp_rank=pg_collection.pp.rank(),
                vp_stage=vp_stage,
                pre_process=pre_process,
                post_process=post_process,
            )
        )
        return TransformerBlock(
            config, get_gpt_layer_local_spec(), pg_collection=pg_collection, vp_stage=vp_stage
        )

    return model_provider


def _recording_provider(calls):
    """A model provider that records the stage and the process groups of each model."""

    def model_provider(pre_process=True, post_process=True, vp_stage=None, pg_collection=None):
        calls.append(
            dict(
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
                pg_collection=pg_collection,
                ranks=None if pg_collection is None else _model_ranks(pg_collection),
            )
        )
        return torch.nn.Linear(1, 1)

    return model_provider


def _model_ranks(pg_collection):
    """The (pipeline, expert, tensor, expert tensor) ranks that the converter's groups report."""
    return (
        pg_collection.pp.rank(),
        pg_collection.ep.rank(),
        pg_collection.tp.rank(),
        pg_collection.expt_tp.rank(),
    )


def _set_up_converter_parallel_state(tensor_parallel_size=1, pipeline_parallel_size=2):
    """Set up the process groups as the converter does and return the collection."""
    parallel_state.destroy_model_parallel()
    return initialize_checkpoint_converter_fake_process_groups(
        parallel_state,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=1,
    )


def _make_loader(
    pg_collection,
    tensor_parallel_size=1,
    pipeline_parallel_size=2,
    virtual_pipeline_parallel_size=None,
):
    loader = MegatronCheckpointLoaderBase(args=None, queue=None)
    loader.margs = SimpleNamespace(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallel_size,
    )
    loader.pg_collection = pg_collection
    return loader


def _make_saver(
    pg_collection,
    model_provider,
    tensor_parallel_size=1,
    pipeline_parallel_size=2,
    expert_parallel_size=1,
):
    saver = MegatronCheckpointSaverBase(
        args=SimpleNamespace(
            target_tensor_parallel_size=tensor_parallel_size,
            target_pipeline_parallel_size=pipeline_parallel_size,
            target_expert_parallel_size=expert_parallel_size,
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
        saved.append(dict(kwargs, model=models[0], ranks=_model_ranks(saver.pg_collection)))

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


# The parallel_state setters that override the global ranks and sizes of the grid.
_RANK_AND_SIZE_OVERRIDES = (
    'set_tensor_model_parallel_rank',
    'set_pipeline_model_parallel_rank',
    'set_virtual_pipeline_model_parallel_rank',
    'set_expert_model_parallel_rank',
    'set_tensor_model_parallel_world_size',
    'set_pipeline_model_parallel_world_size',
    'set_virtual_pipeline_model_parallel_world_size',
    'set_expert_model_parallel_world_size',
)


@pytest.fixture
def no_rank_overrides(monkeypatch):
    """Fail if the converter overrides a global rank or size of parallel_state."""

    def forbidden(name):
        def override(*args, **kwargs):
            raise AssertionError(f"the converter must not call parallel_state.{name}")

        return override

    for name in _RANK_AND_SIZE_OVERRIDES:
        monkeypatch.setattr(parallel_state, name, forbidden(name))


@pytest.mark.parametrize("split", LAYER_SPLITS)
def test_loader_builds_each_pipeline_stage(converter_parallel_state, no_rank_overrides, split):
    assert _loader_stage_layer_numbers(split, converter_parallel_state) == STAGE_LAYER_NUMBERS


@pytest.mark.parametrize("split", LAYER_SPLITS)
def test_saver_builds_each_pipeline_stage(converter_parallel_state, no_rank_overrides, split):
    assert _saver_stage_layer_numbers(split, converter_parallel_state) == STAGE_LAYER_NUMBERS


def test_loader_builds_each_virtual_pipeline_chunk(converter_parallel_state, no_rank_overrides):
    """Each virtual pipeline chunk holds its own layers, and only the first chunk of the first
    stage and the last chunk of the last stage pre- and post-process."""
    calls = []
    loader = _make_loader(converter_parallel_state, virtual_pipeline_parallel_size=2)

    all_models = _load_model_shards(loader, _interleaved_decoder_provider(calls))

    # all_models[pp_rank][vp_stage] lists the model chunks of the stage's TP ranks.
    layer_numbers = [
        [[layer.layer_number for layer in chunk[0].layers] for chunk in stage]
        for stage in all_models
    ]
    assert layer_numbers == [[[1, 2], [5, 6]], [[3, 4], [7, 8]]]
    stages = [(c["pp_rank"], c["vp_stage"], c["pre_process"], c["post_process"]) for c in calls]
    assert stages == [
        (0, 0, True, False),
        (0, 1, False, False),
        (1, 0, False, False),
        (1, 1, False, True),
    ]


def test_loader_builds_models_on_its_collection(no_global_process_groups, no_rank_overrides):
    """The loader selects every model on its collection and passes it to the model and loader."""
    pg_collection = build_checkpoint_converter_process_groups(
        tensor_parallel_size=2, pipeline_parallel_size=2, expert_parallel_size=1
    )
    loader = _make_loader(pg_collection, tensor_parallel_size=2, pipeline_parallel_size=2)
    calls, loads = [], []

    def load_checkpoint(model, optimizer, opt_param_scheduler, **kwargs):
        loads.append(dict(kwargs, ranks=_model_ranks(pg_collection)))

    _load_model_shards(loader, _recording_provider(calls), load_checkpoint)

    # Ranks are (pipeline, expert, tensor, expert tensor).
    models = [(0, 0, 0, 0), (0, 0, 1, 1), (1, 0, 0, 0), (1, 0, 1, 1)]
    stages = [(c["pre_process"], c["post_process"], c["ranks"]) for c in calls]
    assert stages == [
        (True, False, models[0]),
        (True, False, models[1]),
        (False, True, models[2]),
        (False, True, models[3]),
    ]
    # Without a virtual pipeline, each stage is a single model built without a virtual stage.
    assert all(c["vp_stage"] is None for c in calls)
    assert all(c["pg_collection"] is pg_collection for c in calls)
    assert [load["ranks"] for load in loads] == models
    assert all(load["pg_collection"] is pg_collection for load in loads)


def test_saver_builds_models_on_its_collection(no_global_process_groups, no_rank_overrides):
    """The saver selects every model on its collection and passes it to the model and saver."""
    pg_collection = build_checkpoint_converter_process_groups(
        tensor_parallel_size=2, pipeline_parallel_size=2, expert_parallel_size=2
    )
    calls = []
    saver = _make_saver(
        pg_collection,
        _recording_provider(calls),
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        expert_parallel_size=2,
    )

    saved = _save_local_models(saver)

    # Ranks are (pipeline, expert, tensor, expert tensor), in the order the saver saves.
    models = [(pp, ep, tp, tp) for pp in range(2) for ep in range(2) for tp in range(2)]
    assert [(c["pre_process"], c["post_process"], c["ranks"]) for c in calls] == [
        (ranks[0] == 0, ranks[0] == 1, ranks) for ranks in models
    ]
    assert all(c["pg_collection"] is pg_collection for c in calls)
    assert [
        (save["pipeline_rank"], save["expert_rank"], save["tensor_rank"]) for save in saved
    ] == [ranks[:3] for ranks in models]
    assert [save["ranks"] for save in saved] == models
    assert all(save["pg_collection"] is pg_collection for save in saved)


def test_loader_reads_each_legacy_checkpoint_file(no_rank_overrides):
    """Legacy checkpoint files are named after the ranks the loader selects on its collection."""
    pg_collection = _set_up_converter_parallel_state(tensor_parallel_size=2)
    loader = _make_loader(pg_collection, tensor_parallel_size=2, pipeline_parallel_size=2)
    checkpoint_names = []

    def load_checkpoint(model, optimizer, opt_param_scheduler, **kwargs):
        # load_checkpoint() names the file of a legacy checkpoint after its collection.
        checkpoint_names.append(
            checkpointing.get_checkpoint_name("ckpt", 7, pg_collection=kwargs["pg_collection"])
        )

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
