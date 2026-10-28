# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The inference entry points build their model on the groups that initialize_megatron registered.

get_model_for_inference() builds and loads the model on the collection it is given. The entry
points in this repository pass the registered one, parallel_state.get_registered_process_groups();
a caller that omits it falls back to the global grid with a ProcessGroupFallbackWarning. The
tests run without torch.distributed, with stand-ins for the model builder and the checkpoint
loader.
"""

import ast
import pathlib
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core import parallel_state, process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.inference import utils

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
# The folders whose code runs inference; tests are not scanned.
SCANNED_FOLDERS = ("megatron", "examples", "tools", "tasks")
# The groups that get_model_for_inference passes to load_checkpoint.
CHECKPOINT_GROUPS = {
    "tp_group": "tp",
    "pp_group": "pp",
    "dp_cp_group": "dp_cp",
    "dp_group": "dp",
    "expt_dp_group": "expt_dp",
}


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


class _Model:
    def eval(self):
        pass


def _groups():
    return ProcessGroupCollection(**{name: Mock() for name in CHECKPOINT_GROUPS.values()})


def _forbid_the_shim(monkeypatch):
    monkeypatch.setattr(
        ProcessGroupCollection,
        "use_mpu_process_groups",
        Mock(side_effect=AssertionError("the groups were rebuilt from the global grid")),
    )


@pytest.fixture
def inference(monkeypatch):
    """Stand-ins for the arguments, the model builder and the checkpoint loader."""
    args = SimpleNamespace(
        load="checkpoint",
        inference_ckpt_non_strict=False,
        transformer_impl="local",
        fp8_recipe=None,
        modelopt_enabled=False,
    )
    built, loaded = [], []

    class Builder:
        def build_distributed_models(self, pg_collection, wrap_with_ddp):
            built.append(pg_collection)
            return [_Model()]

    monkeypatch.setattr(utils, "get_args", lambda: args)
    monkeypatch.setattr(utils, "get_model_builder", lambda args, pg_collection: Builder())
    monkeypatch.setattr(utils, "load_checkpoint", lambda **kwargs: loaded.append(kwargs))
    monkeypatch.setattr(utils, "_get_checkpoint_model_modifier", lambda args, keys: None)
    return SimpleNamespace(args=args, built=built, loaded=loaded)


@pytest.fixture
def modelopt(inference, monkeypatch):
    """The ModelOpt path, with a stand-in for get_model; returns the groups it receives."""
    inference.args.modelopt_enabled = True
    monkeypatch.setattr(utils, "HAS_NVIDIA_MODELOPT", True)
    monkeypatch.setattr(utils, "modelopt_gpt_hybrid_builder", object(), raising=False)
    received = []

    def get_model(model_provider, wrap_with_ddp, pg_collection=None):
        received.append(pg_collection)
        return [_Model()]

    monkeypatch.setattr(utils, "_get_model", get_model)
    return received


def test_the_given_collection_builds_and_loads_the_model(inference, monkeypatch):
    _forbid_the_shim(monkeypatch)
    pg_collection = _groups()

    with warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        utils.get_model_for_inference(pg_collection=pg_collection)

    assert inference.built == [pg_collection]
    for argument, field in CHECKPOINT_GROUPS.items():
        assert inference.loaded[0][argument] is getattr(pg_collection, field)


def test_without_a_collection_it_warns_and_uses_the_global_grid(inference, monkeypatch):
    global_groups = _groups()
    shim = Mock(return_value=global_groups)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", shim)

    with pytest.warns(
        ProcessGroupFallbackWarning,
        match="get_model_for_inference was called without `pg_collection`",
    ):
        utils.get_model_for_inference()

    shim.assert_called_once_with()
    assert inference.built == [global_groups]


def test_modelopt_builds_on_the_registered_collection(inference, modelopt, monkeypatch):
    """The ModelOpt builder accepts the registered groups and loads with those of the model."""
    _forbid_the_shim(monkeypatch)
    registered = _groups()
    monkeypatch.setattr(parallel_state, "_REGISTERED_PROCESS_GROUPS", registered)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        utils.get_model_for_inference(pg_collection=registered)

    assert modelopt == [registered]
    assert all(inference.loaded[0][argument] is None for argument in CHECKPOINT_GROUPS)


def test_modelopt_rejects_other_collections(inference, modelopt, monkeypatch):
    monkeypatch.setattr(parallel_state, "_REGISTERED_PROCESS_GROUPS", _groups())

    with pytest.raises(ValueError, match="Custom inference process groups are not supported"):
        utils.get_model_for_inference(pg_collection=_groups())

    assert modelopt == []


def test_the_dynamic_engine_builds_its_model_on_the_registered_collection(monkeypatch):
    registered = _groups()
    monkeypatch.setattr(parallel_state, "_REGISTERED_PROCESS_GROUPS", registered)
    requested = []
    monkeypatch.setattr(utils, "get_args", lambda: SimpleNamespace())
    monkeypatch.setattr(
        utils, "get_model_for_inference", lambda **kwargs: requested.append(kwargs) or "model"
    )
    monkeypatch.setattr(utils, "build_tokenizer", lambda args: "tokenizer")
    monkeypatch.setattr(
        utils, "get_inference_config_from_model_and_args", lambda model, args: "config"
    )
    monkeypatch.setattr(utils, "build_dynamic_inference_engine", lambda **kwargs: kwargs)

    engine = utils.get_dynamic_inference_engine()

    assert requested == [{"pg_collection": registered}]
    assert engine["model"] == "model"


def _called_name(node):
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def test_in_tree_callers_pass_their_collection():
    """Every call of get_model_for_inference outside the tests passes pg_collection."""
    paths = [path for folder in SCANNED_FOLDERS for path in (REPO_ROOT / folder).rglob("*.py")]
    paths.extend(REPO_ROOT.glob("*.py"))
    calls, without_groups = [], []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) == "get_model_for_inference":
                site = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
                calls.append(site)
                if not any(keyword.arg == "pg_collection" for keyword in node.keywords):
                    without_groups.append(site)

    assert calls, "found no call of get_model_for_inference"
    assert without_groups == []
