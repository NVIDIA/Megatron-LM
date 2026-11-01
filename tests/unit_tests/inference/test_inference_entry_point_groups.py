# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""In-tree scripts build their model on the groups that initialize_megatron registered.

get_model_for_inference(), get_dynamic_inference_engine() and get_model() build the model on the
collection they are given. The inference entry points, and every other script in this repository
that calls megatron.training's get_model(), pass the registered one,
parallel_state.get_registered_process_groups(); a caller that omits it falls back to the global
grid with a ProcessGroupFallbackWarning. The tests run without torch.distributed, with stand-ins
for the model builder and the checkpoint loader.
"""

import ast
import pathlib
import sys
import warnings
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from megatron.core import parallel_state, process_groups_config
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.inference import utils

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
# The folders whose code builds models; tests are not scanned.
SCANNED_FOLDERS = ("megatron", "examples", "tools", "tasks")
# The modules that scripts import megatron.training's get_model() from.
GET_MODEL_MODULES = ("megatron.training", "megatron.training.training")
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
    monkeypatch.setattr(utils, "get_run_config", lambda: SimpleNamespace(tokenizer=Mock()))
    monkeypatch.setattr(
        utils, "get_model_builder", lambda args, tokenizer_config, pg_collection: Builder()
    )
    monkeypatch.setattr(utils, "load_checkpoint", lambda **kwargs: loaded.append(kwargs))
    monkeypatch.setattr(utils, "_get_checkpoint_model_modifier", lambda args, keys: None)
    return SimpleNamespace(args=args, built=built, loaded=loaded)


@pytest.fixture
def modelopt(inference, monkeypatch):
    """The ModelOpt path, with a stand-in for get_model that calls the provider as it does.

    Returns the groups get_model receives.
    """
    inference.args.modelopt_enabled = True
    monkeypatch.setattr(utils, "HAS_NVIDIA_MODELOPT", True)
    monkeypatch.setattr(utils, "modelopt_gpt_hybrid_builder", Mock(), raising=False)
    received = []

    def get_model(model_provider, wrap_with_ddp, pg_collection=None):
        received.append(pg_collection)
        model_provider(
            pre_process=True, post_process=True, config=None, pg_collection=pg_collection
        )
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
    utils.modelopt_gpt_hybrid_builder.assert_called_once_with(
        inference.args, pre_process=True, post_process=True, config=None, pg_collection=registered
    )
    assert all(inference.loaded[0][argument] is None for argument in CHECKPOINT_GROUPS)


def test_modelopt_rejects_other_collections(inference, modelopt, monkeypatch):
    monkeypatch.setattr(parallel_state, "_REGISTERED_PROCESS_GROUPS", _groups())

    with pytest.raises(ValueError, match="Custom inference process groups are not supported"):
        utils.get_model_for_inference(pg_collection=_groups())

    assert modelopt == []


@pytest.fixture
def engine_parts(monkeypatch):
    """Stand-ins for what get_dynamic_inference_engine() builds around the model."""
    monkeypatch.setattr(utils, "get_tokenizer", lambda: "tokenizer")
    monkeypatch.setattr(
        utils, "get_inference_config_from_model_and_args", lambda model, args: "config"
    )
    monkeypatch.setattr(utils, "build_dynamic_inference_engine", lambda **kwargs: kwargs)


def test_the_dynamic_engine_builds_its_model_on_the_given_collection(engine_parts, monkeypatch):
    pg_collection = _groups()
    requested = []
    monkeypatch.setattr(utils, "get_args", lambda: SimpleNamespace())
    monkeypatch.setattr(
        utils, "get_model_for_inference", lambda **kwargs: requested.append(kwargs) or "model"
    )

    engine = utils.get_dynamic_inference_engine(pg_collection=pg_collection)

    assert requested == [{"pg_collection": pg_collection}]
    assert engine["model"] == "model"


def test_the_dynamic_engine_without_groups_warns_and_uses_the_global_grid(
    inference, engine_parts, monkeypatch
):
    """It does not take the registered collection, which a framework may register partially."""
    monkeypatch.setattr(
        parallel_state, "_REGISTERED_PROCESS_GROUPS", ProcessGroupCollection(tp=Mock())
    )
    global_groups = _groups()
    monkeypatch.setattr(
        ProcessGroupCollection, "use_mpu_process_groups", Mock(return_value=global_groups)
    )

    with pytest.warns(
        ProcessGroupFallbackWarning,
        match="get_model_for_inference was called without `pg_collection`",
    ):
        utils.get_dynamic_inference_engine()

    assert inference.built == [global_groups]


@pytest.mark.parametrize("is_vlm,is_mimo", [(False, False), (True, False), (True, True)])
def test_the_vlm_server_builds_its_model_on_the_registered_collection(monkeypatch, is_vlm, is_mimo):
    from examples.inference.advanced import vlm_dynamic_inference

    _forbid_the_shim(monkeypatch)
    registered = _groups()
    monkeypatch.setattr(parallel_state, "_REGISTERED_PROCESS_GROUPS", registered)
    args = SimpleNamespace(load="checkpoint", inference_ckpt_non_strict=False)
    if is_mimo:
        args.mimo_checkpoint_prefix_map = {"language_model.": "language_model.module.module."}
    providers = ModuleType("model")
    providers.model_provider = Mock()
    for name in ("model", "model_provider", "mimo_checkpoint_model"):
        monkeypatch.setitem(sys.modules, name, providers)
    gpt_builders = ModuleType("gpt_builders")
    gpt_builders.gpt_builder = Mock()
    monkeypatch.setitem(sys.modules, "gpt_builders", gpt_builders)
    received = []

    def get_model(model_provider, wrap_with_ddp, pg_collection=None):
        received.append(pg_collection)
        return [_Model()]

    monkeypatch.setattr(vlm_dynamic_inference, "get_args", lambda: args)
    monkeypatch.setattr(vlm_dynamic_inference, "_get_model", get_model)
    monkeypatch.setattr(vlm_dynamic_inference, "load_checkpoint", Mock())
    monkeypatch.setattr(vlm_dynamic_inference, "_check_mimo_checkpoint_fully_loaded", Mock())

    vlm_dynamic_inference.get_model(is_vlm=is_vlm)

    assert received == [registered]


def _called_name(node):
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _calls(names, paths):
    """Return (site, call) for every call of a function in ``names`` in ``paths``."""
    calls = []
    for path in sorted(set(paths)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called_name(node) in names:
                calls.append((f"{path.relative_to(REPO_ROOT)}:{node.lineno}", node))
    return calls


def _sources():
    paths = [path for folder in SCANNED_FOLDERS for path in (REPO_ROOT / folder).rglob("*.py")]
    return paths + list(REPO_ROOT.glob("*.py"))


def _passes(call, *arguments):
    """Whether ``call`` passes one of ``arguments`` by keyword, as something other than None."""
    return any(
        keyword.arg in arguments
        and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)
        for keyword in call.keywords
    )


def test_in_tree_callers_pass_their_collection():
    """Every call of get_model_for_inference outside the tests passes pg_collection."""
    calls = _calls({"get_model_for_inference"}, _sources())

    assert calls, "found no call of get_model_for_inference"
    assert [site for site, call in calls if not _passes(call, "pg_collection")] == []


def test_in_tree_callers_of_the_dynamic_engine_pass_a_model_or_its_groups():
    """Every call of get_dynamic_inference_engine outside the tests passes model or groups."""
    calls = _calls({"get_dynamic_inference_engine"}, _sources())

    assert calls, "found no call of get_dynamic_inference_engine"
    assert [
        site for site, call in calls if not (call.args or _passes(call, "model", "pg_collection"))
    ] == []


def _dotted_name(node):
    """Return the dotted name of a name or attribute chain, such as ``megatron.training``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return prefix and f"{prefix}.{node.attr}"
    return None


def _imported_module(node, package):
    """Return the absolute name of the module that the ImportFrom ``node`` imports from.

    A relative import resolves against ``package``, the name parts of the importing file's package.
    """
    if not node.level:
        return node.module
    parts = package[: max(len(package) - node.level + 1, 0)]
    return ".".join(parts + ((node.module,) if node.module else ()))


def _get_model_calls(path, root=REPO_ROOT):
    """Return every call of megatron.training's get_model() in ``path``, in line order.

    A call matches under any name that ``path`` imports get_model() or its module as, such as
    ``get_training_model(...)`` or ``training.get_model(...)``, by an absolute import or by a
    relative one, which resolves against the package of ``path`` under ``root``; a function of
    another module that is also named get_model does not.
    """
    package = path.relative_to(root).parent.parts
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions, modules = set(), set(GET_MODEL_MODULES)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = _imported_module(node, package)
            for alias in node.names:
                if module in GET_MODEL_MODULES and alias.name == "get_model":
                    functions.add(alias.asname or alias.name)
                elif f"{module}.{alias.name}" in GET_MODEL_MODULES:
                    modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            modules.update(
                alias.asname
                for alias in node.names
                if alias.asname and alias.name in GET_MODEL_MODULES
            )
    if ".".join(package + (path.stem,)) == "megatron.training.training":
        functions.add("get_model")
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in functions)
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_model"
                and _dotted_name(node.func.value) in modules
            )
        )
    ]
    return sorted(calls, key=lambda call: call.lineno)


def test_the_get_model_check_sees_every_form_of_the_call(tmp_path):
    """Aliases and module attributes match, and passing None does not count as a collection."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import megatron.training\n"
        "import megatron.training.training as mtt\n"
        "from megatron import training\n"
        "from megatron.training import get_model as build\n"
        "from examples.post_training.modelopt.convert_model import get_model\n"
        "build(provider, pg_collection=groups)\n"
        "build(provider, pg_collection=None)\n"
        "megatron.training.get_model(provider)\n"
        "mtt.get_model(provider)\n"
        "training.get_model(provider)\n"
        "get_model(provider)\n",
        encoding="utf-8",
    )

    calls = _get_model_calls(probe, root=tmp_path)

    assert [(call.lineno, _passes(call, "pg_collection")) for call in calls] == [
        (6, True),
        (7, False),
        (8, False),
        (9, False),
        (10, False),
    ]


def test_the_get_model_check_resolves_relative_imports(tmp_path):
    """A module of the megatron.training package may import get_model() relatively."""
    package = tmp_path / "megatron" / "training"
    package.mkdir(parents=True)
    probe = package / "probe.py"
    probe.write_text(
        "from .training import get_model\n"
        "from . import training\n"
        "from .. import training as parent\n"
        "from .checkpointing import get_model as other\n"
        "get_model(provider)\n"
        "training.get_model(provider)\n"
        "parent.get_model(provider)\n"
        "other(provider)\n",
        encoding="utf-8",
    )
    defining = package / "training.py"
    defining.write_text("get_model(provider)\n", encoding="utf-8")

    assert [call.lineno for call in _get_model_calls(probe, root=tmp_path)] == [5, 6, 7]
    assert [call.lineno for call in _get_model_calls(defining, root=tmp_path)] == [1]


def test_in_tree_callers_pass_their_collection_to_get_model():
    """Every call of megatron.training's get_model() outside the tests passes pg_collection.

    The scripts that build their model after initialize_megatron() pass the collection it
    registered, so none of them falls back to the global grid.
    """
    calls = [
        (f"{path.relative_to(REPO_ROOT)}:{call.lineno}", call)
        for path in sorted(set(_sources()))
        for call in _get_model_calls(path)
    ]

    assert calls, "found no call of get_model"
    assert [site for site, call in calls if not _passes(call, "pg_collection")] == []
