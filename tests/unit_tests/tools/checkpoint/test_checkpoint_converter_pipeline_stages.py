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
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))

from loader_base import MegatronCheckpointLoaderBase
from saver_base import MegatronCheckpointSaverBase
from utils import initialize_checkpoint_converter_fake_process_groups

# Two pipeline stages with three decoder layers on the first and one on the last.
LAYER_SPLITS = {
    "layout": dict(pipeline_model_parallel_layout="Ettt|tL"),
    "uneven": dict(num_layers_in_first_pipeline_stage=3, num_layers_in_last_pipeline_stage=1),
}
STAGE_LAYER_NUMBERS = [[1, 2, 3], [4]]


def _decoder_provider(split):
    """A model provider that builds the decoder as GPTModel does when given no collection."""
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

    def model_provider(pre_process=True, post_process=True):
        return TransformerBlock(
            config,
            get_gpt_layer_local_spec(),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )

    return model_provider


def _set_up_converter_parallel_state():
    """Set up parallel_state as the converter does: rank overrides and fake groups, TP=1, PP=2."""
    parallel_state.destroy_model_parallel()
    parallel_state.set_tensor_model_parallel_world_size(1)
    parallel_state.set_pipeline_model_parallel_world_size(2)
    parallel_state.set_tensor_model_parallel_rank(0)
    parallel_state.set_pipeline_model_parallel_rank(0)
    initialize_checkpoint_converter_fake_process_groups(
        parallel_state, tensor_parallel_size=1, pipeline_parallel_size=2, expert_parallel_size=1
    )


def _loader_stage_layer_numbers(split):
    """Decoder layer numbers of each stage the loader builds, without loading any weights."""
    loader = MegatronCheckpointLoaderBase(args=None, queue=None)
    loader.margs = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=None,
    )
    with mock.patch.object(checkpointing, "load_checkpoint"):
        all_models, _, _ = loader.load_model_shards(_decoder_provider(split), torch.float32)
    # all_models[pp_rank][vp_rank] lists the models of the stage's TP ranks.
    return [[layer.layer_number for layer in stage[0][0].layers] for stage in all_models]


def _saver_stage_layer_numbers(split):
    """Decoder layer numbers of each stage the saver builds and saves."""
    saver = MegatronCheckpointSaverBase(
        args=SimpleNamespace(
            target_tensor_parallel_size=1,
            target_pipeline_parallel_size=2,
            target_expert_parallel_size=1,
        ),
        queue=None,
    )
    saver.md = SimpleNamespace(iteration=1, params_dtype=torch.float32)
    saver.model_provider = _decoder_provider(split)
    saver.models = saver.initialize_models()
    saved_models = {}

    def save_checkpoint(iteration, models, optimizer, opt_param_scheduler, **kwargs):
        saved_models[kwargs["pipeline_rank"]] = models[0]

    with mock.patch.object(checkpointing, "save_checkpoint", save_checkpoint):
        saver.save_local_models_to_checkpoint()
    return [
        [layer.layer_number for layer in saved_models[pp].layers] for pp in sorted(saved_models)
    ]


@pytest.fixture
def converter_parallel_state():
    # As in the saver process, torch.distributed is initialized, so models see the fake groups.
    Utils.initialize_distributed()
    _set_up_converter_parallel_state()
    yield
    parallel_state.destroy_model_parallel()


@pytest.mark.parametrize("split", LAYER_SPLITS)
def test_loader_builds_each_pipeline_stage(converter_parallel_state, split):
    assert _loader_stage_layer_numbers(split) == STAGE_LAYER_NUMBERS


@pytest.mark.parametrize("split", LAYER_SPLITS)
def test_saver_builds_each_pipeline_stage(converter_parallel_state, split):
    assert _saver_stage_layer_numbers(split) == STAGE_LAYER_NUMBERS


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
            "t._set_up_converter_parallel_state()",
            "print(json.dumps({s: t._loader_stage_layer_numbers(s) for s in t.LAYER_SPLITS}))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stdout + result.stderr
    stage_layer_numbers = json.loads(result.stdout.strip().splitlines()[-1])
    assert stage_layer_numbers == {split: STAGE_LAYER_NUMBERS for split in LAYER_SPLITS}
