# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The example scripts build their models on the process groups of the grid they set up.

The model providers take the collection that get_model() passes; one that is called without it
falls back to the global grid with a ProcessGroupFallbackWarning. The scripts that set up a grid
themselves take its collection from initialize_model_parallel(). The tests run without
torch.distributed, with stand-ins for the model classes and the grid.
"""

import importlib
import importlib.util
import pathlib
import sys
import types
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core import parallel_state, process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DETOXIFY_SCRIPT = (
    REPO_ROOT / "examples" / "academic_paper_scripts" / "detxoify_lm" / "generate_samples_gpt.py"
)


class _Stop(Exception):
    """Ends a model provider once it has resolved its process groups."""


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _fallback_warning(owner):
    return pytest.warns(
        ProcessGroupFallbackWarning, match=f"{owner} was called without `pg_collection`"
    )


def _import_example(monkeypatch, folder, name):
    """Import an example training script the way it runs, as a top-level module."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "examples" / folder))
    try:
        return importlib.import_module(name)
    except ImportError as error:
        # The BERT script builds a Transformer Engine layer spec when it is imported.
        pytest.skip(f"{name}.py cannot be imported here: {error}")


@pytest.mark.parametrize(
    "folder, name", [("bert", "pretrain_bert"), ("t5", "pretrain_t5")], ids=["bert", "t5"]
)
def test_model_provider_warns_without_a_collection(monkeypatch, folder, name):
    script = _import_example(monkeypatch, folder, name)
    global_groups = ProcessGroupCollection()
    shim = Mock(return_value=global_groups)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", shim)
    # The provider reads the arguments once it has its groups.
    monkeypatch.setattr(script, "get_args", Mock(side_effect=_Stop))

    with _fallback_warning(f"{name}.model_provider"), pytest.raises(_Stop):
        script.model_provider()

    shim.assert_called_once_with()


@pytest.fixture
def detoxify(monkeypatch):
    """The detoxification sampling script, with stand-ins for GPTModel and its arguments.

    The script imports megatron.inference.text_generation, which no longer exists; a stand-in
    module takes its place.
    """
    text_generation = types.ModuleType("megatron.inference.text_generation")
    text_generation.generate_and_post_process = None
    monkeypatch.setitem(sys.modules, "megatron.inference.text_generation", text_generation)
    spec = importlib.util.spec_from_file_location("generate_samples_gpt", DETOXIFY_SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    args = SimpleNamespace(
        spec=None,
        transformer_impl="local",
        num_experts=None,
        moe_grouped_gemm=False,
        padded_vocab_size=8,
        max_position_embeddings=8,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=False,
        position_embedding_type="learned_absolute",
        rotary_percent=1.0,
    )
    monkeypatch.setattr(script, "get_args", lambda: args)
    monkeypatch.setattr(script, "print_rank_0", lambda message: None)
    monkeypatch.setattr(script, "get_gpt_layer_local_spec", lambda **kwargs: "layer spec")
    monkeypatch.setattr(script, "GPTModel", lambda **kwargs: kwargs)
    return script


def test_detoxify_model_provider_builds_on_the_config_and_groups_of_get_model(
    detoxify, monkeypatch
):
    """get_model() passes config and pg_collection, and the model is built on both."""
    monkeypatch.setattr(
        detoxify, "core_transformer_config_from_args", Mock(side_effect=AssertionError)
    )
    monkeypatch.setattr(
        ProcessGroupCollection, "use_mpu_process_groups", Mock(side_effect=AssertionError)
    )
    config, pg_collection = object(), ProcessGroupCollection()

    with warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        model = detoxify.model_provider(
            pre_process=True, post_process=False, config=config, pg_collection=pg_collection
        )

    assert model["config"] is config
    assert model["pg_collection"] is pg_collection
    assert (model["pre_process"], model["post_process"]) == (True, False)


def test_detoxify_model_provider_warns_without_a_collection(detoxify, monkeypatch):
    config, global_groups = object(), ProcessGroupCollection()
    monkeypatch.setattr(detoxify, "core_transformer_config_from_args", lambda args: config)
    shim = Mock(return_value=global_groups)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", shim)

    with _fallback_warning("generate_samples_gpt.model_provider"):
        model = detoxify.model_provider()

    shim.assert_called_once_with()
    assert model["config"] is config
    assert model["pg_collection"] is global_groups


@pytest.fixture
def grid(monkeypatch):
    """A stand-in for initialize_model_parallel() that returns a collection of its own."""
    pg_collection = ProcessGroupCollection()
    initialize = Mock(return_value=pg_collection)
    monkeypatch.setattr(parallel_state, "initialize_model_parallel", initialize)
    monkeypatch.setattr(parallel_state, "destroy_model_parallel", lambda: None)
    monkeypatch.setattr(
        ProcessGroupCollection, "use_mpu_process_groups", Mock(side_effect=AssertionError)
    )
    for name, value in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0")):
        monkeypatch.setenv(name, value)
    return SimpleNamespace(pg_collection=pg_collection, initialize=initialize)


@pytest.mark.parametrize(
    "module, call",
    [
        ("examples.run_simple_mcore_train_loop", ((2, 1), {})),
        (
            "examples.export.trtllm_export.single_device_export.gpt_single_device_cpu_export",
            ((2, 1), {}),
        ),
        (
            "examples.export.trtllm_export.distributed_export.gpt_distributed_gpu_export",
            ((), {"tensor_model_parallel_size": 2, "pipeline_model_parallel_size": 1}),
        ),
    ],
    ids=["simple_train_loop", "trtllm_single_device", "trtllm_distributed"],
)
def test_scripts_take_the_collection_of_the_grid_they_build(grid, monkeypatch, module, call):
    script = importlib.import_module(module)
    monkeypatch.setattr(script.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(script.torch.distributed, "init_process_group", lambda **kwargs: None)

    pg_collection = script.initialize_distributed(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1
    )

    assert pg_collection is grid.pg_collection
    grid.initialize.assert_called_once_with(*call[0], **call[1])
