# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""In-tree scripts build their model on the groups that initialize_megatron registered.

get_model_for_inference(), get_dynamic_inference_engine(), get_model() and
setup_model_and_optimizer() build the model on the collection they are given. The inference
entry points, and every other script in this repository that calls megatron.training's
get_model(), pass the registered one, parallel_state.get_registered_process_groups(); the callers
of setup_model_and_optimizer() pass the collection that pretrain() builds from it. A caller that
omits it falls back to the global grid with a ProcessGroupFallbackWarning. get_model() passes the
collection and the model config on to the model provider, so every provider takes both. The tests
run without torch.distributed, with stand-ins for the model builder and the checkpoint loader.
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
# The module that scripts import setup_model_and_optimizer() from.
SETUP_MODULES = ("megatron.training.training",)
# The keywords that get_model() passes to the model provider besides pre_process, post_process
# and, under virtual pipeline parallelism, vp_stage.
PROVIDER_KEYWORDS = ("config", "pg_collection")
# Scripts whose model providers once took neither keyword, so that get_model() failed to call
# them. No test runs them: two import a module that was removed, and the third needs the GPU
# container.
OWN_PROVIDER_SCRIPTS = (
    "examples/academic_paper_scripts/detxoify_lm/generate_samples_gpt.py",
    "examples/multimodal/model_converter/vision_model_tester.py",
    "examples/multimodal/run_text_generation.py",
)
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
    monkeypatch.setattr(utils, "build_tokenizer", lambda args: "tokenizer")
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


def _imported_calls(path, function, modules, root=REPO_ROOT, tree=None):
    """Return every call in ``path`` of ``function`` of megatron.training.training, in line order.

    ``modules`` are the modules that ``function`` can be imported from. A call matches under any
    name that ``path`` imports ``function`` or one of ``modules`` as, such as
    ``get_training_model(...)`` or ``training.get_model(...)``, by an absolute import or by a
    relative one, which resolves against the package of ``path`` under ``root``; a function of
    another module, or a method, that has the same name does not. The calls are nodes of
    ``tree``, the syntax tree of ``path``, which is parsed when it is not given.
    """
    package = path.relative_to(root).parent.parts
    if tree is None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    functions, names = set(), set(modules)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = _imported_module(node, package)
            for alias in node.names:
                if module in modules and alias.name == function:
                    functions.add(alias.asname or alias.name)
                elif f"{module}.{alias.name}" in modules:
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            names.update(
                alias.asname for alias in node.names if alias.asname and alias.name in modules
            )
    if ".".join(package + (path.stem,)) == "megatron.training.training":
        functions.add(function)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in functions)
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == function
                and _dotted_name(node.func.value) in names
            )
        )
    ]
    return sorted(calls, key=lambda call: call.lineno)


def _get_model_calls(path, root=REPO_ROOT, tree=None):
    """Return every call of megatron.training's get_model() in ``path``, in line order.

    See _imported_calls(), which matches them.
    """
    return _imported_calls(path, "get_model", GET_MODEL_MODULES, root, tree)


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


def test_the_setup_check_matches_the_calls_of_the_imported_function(tmp_path):
    """An alias and a module attribute match; a method of the same name does not."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from megatron.training.training import setup_model_and_optimizer as setup\n"
        "from megatron.training import training\n"
        "setup(model_type, provider)\n"
        "training.setup_model_and_optimizer(model_type, provider, pg_collection=groups)\n"
        "trainer.setup_model_and_optimizer()\n"
        "self.setup_model_and_optimizer()\n",
        encoding="utf-8",
    )

    calls = _imported_calls(probe, "setup_model_and_optimizer", SETUP_MODULES, root=tmp_path)

    assert [(call.lineno, _passes(call, "pg_collection")) for call in calls] == [
        (3, False),
        (4, True),
    ]


def test_in_tree_callers_pass_their_collection_to_setup_model_and_optimizer():
    """Every call of setup_model_and_optimizer() outside the tests passes pg_collection.

    pretrain(), the finetuning driver and the BERT embedding tool build their model on the
    collection that pretrain() builds from the registered groups, so none of them falls back to
    the global grid.
    """
    calls = [
        (f"{path.relative_to(REPO_ROOT)}:{call.lineno}", call)
        for path in sorted(set(_sources()))
        for call in _imported_calls(path, "setup_model_and_optimizer", SETUP_MODULES)
    ]

    assert calls, "found no call of setup_model_and_optimizer"
    assert [site for site, call in calls if not _passes(call, "pg_collection")] == []


def _module_file(module, path):
    """Return the file of ``module`` that the script ``path`` imports, next to it or at the root.

    Returns None for a module in neither place, such as one that the script imports from a folder
    that it adds to sys.path.
    """
    relative = pathlib.Path(*module.split("."))
    for folder in (path.parent, REPO_ROOT):
        for candidate in (folder / f"{relative}.py", folder / relative / "__init__.py"):
            if candidate.is_file():
                return candidate
    return None


# The nodes that open a scope: a name that they bind is not seen outside them.
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _parents(tree):
    """Return the parent of each node of ``tree``."""
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _blocks(node):
    """Return the statement lists of ``node``, such as the body and the else of an if."""
    blocks = (getattr(node, field, None) for field in ("body", "orelse", "finalbody", "handlers"))
    return [block for block in blocks if isinstance(block, list)]


def _nested_statements(statements):
    """Yield ``statements`` and the statements of their blocks, outside nested scopes."""
    for statement in statements:
        yield statement
        if not isinstance(statement, _SCOPES):
            for block in _blocks(statement):
                yield from _nested_statements(block)


def _binds(statement, name):
    """Whether ``statement`` binds ``name`` by a definition, an import or an assignment."""
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return statement.name == name
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        return any((alias.asname or alias.name.split(".")[0]) == name for alias in statement.names)
    if isinstance(statement, ast.Assign) or (
        isinstance(statement, ast.AnnAssign) and statement.value is not None
    ):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        return any(
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name
            for target in targets
            for node in ast.walk(target)
        )
    return False


def _reaching(statements, name):
    """Return the bindings of ``name`` in ``statements`` that may hold after them.

    Also returns whether one of them certainly holds. The last statement of ``statements`` that
    binds ``name`` itself hides the bindings before it; a binding in a block of a statement, such
    as one arm of an if, may not run, so it hides none. Loops are not followed.
    """
    bindings = []
    for statement in reversed(statements):
        if _binds(statement, name):
            return bindings + [statement], True
        if not isinstance(statement, _SCOPES):
            bindings += [
                nested
                for block in _blocks(statement)
                for nested in _nested_statements(block)
                if _binds(nested, name)
            ]
    return bindings, False


def _parameters(scope):
    """Return the names of the parameters of the function or lambda ``scope``."""
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return set()
    arguments = scope.args
    names = arguments.posonlyargs + arguments.args + arguments.kwonlyargs
    names += [argument for argument in (arguments.vararg, arguments.kwarg) if argument]
    return {argument.arg for argument in names}


def _body(scope):
    """Return the statements of the module, function or lambda ``scope``; a lambda has none."""
    return scope.body if isinstance(scope.body, list) else []


def _bindings(name, node, parents):
    """Return the statements that may bind ``name`` where the expression ``node`` reads it.

    As Python does, the name resolves in the innermost function that binds it, else in the
    module. In the function that reads it, a binding counts when it comes before the read in a
    block that holds the read, so a branch does not see the imports of another branch; in an
    enclosing function or in the module, which ran first, the bindings at its end count. Returns
    no statement for a parameter, which the caller binds, and for an unbound name.
    """
    bindings, child = [], node
    while True:
        parent = parents[child]
        for block in _blocks(parent):
            index = next((at for at, statement in enumerate(block) if statement is child), None)
            if index is not None:
                found, certain = _reaching(block[:index], name)
                bindings += found
                if certain:
                    return bindings
        if isinstance(parent, (ast.Module, *_SCOPES)):
            break
        child = parent
    scope = parent
    if bindings or isinstance(scope, ast.Module):
        return bindings
    if name in _parameters(scope) or any(
        _binds(statement, name) for statement in _nested_statements(_body(scope))
    ):
        return []
    while not isinstance(scope, ast.Module):
        scope = parents[scope]
        if isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            found, _ = _reaching(_body(scope), name)
            if found or name in _parameters(scope):
                return found
    return []


def _provider_definitions(provider, path, parents):
    """Return the functions that ``provider``, a model provider expression in ``path``, may be.

    ``parents`` maps each node of the syntax tree of ``path`` to its parent. A provider is a
    lambda, functools.partial() of a provider, or a name, which _bindings() resolves in the
    scope that reads it: to a function that ``path`` defines, to the value of an assignment, or
    to a function that ``path`` imports by name from a module that _module_file() finds and that
    defines it. Other providers are not found, such as one that the calling function receives
    from its own caller or imports from a folder that the script adds to sys.path.
    """
    if isinstance(provider, ast.Lambda):
        return [provider]
    if isinstance(provider, ast.Call) and _called_name(provider) == "partial" and provider.args:
        return _provider_definitions(provider.args[0], path, parents)
    if not isinstance(provider, ast.Name):
        return []
    definitions = []
    for binding in _bindings(provider.id, provider, parents):
        if isinstance(binding, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.append(binding)
        elif isinstance(binding, (ast.Assign, ast.AnnAssign)):
            definitions += _provider_definitions(binding.value, path, parents)
        elif isinstance(binding, ast.ImportFrom) and binding.module and not binding.level:
            module_file = _module_file(binding.module, path)
            for alias in binding.names:
                if module_file is not None and (alias.asname or alias.name) == provider.id:
                    module = ast.parse(module_file.read_text(encoding="utf-8"))
                    definitions += [
                        definition
                        for definition in _reaching(module.body, alias.name)[0]
                        if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
    return definitions


def _ignored_provider_keywords(provider):
    """Return the keywords of PROVIDER_KEYWORDS that the function ``provider`` rejects or ignores.

    A provider that names a keyword must read it, or it builds the model on other groups, or with
    another config, than get_model() passes. One that takes ``**kwargs`` accepts every keyword;
    whether it passes them on is not checked.
    """
    arguments = provider.args
    named = {argument.arg for argument in arguments.args + arguments.kwonlyargs}
    body = provider.body if isinstance(provider.body, list) else [provider.body]
    read = {
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return [
        keyword
        for keyword in PROVIDER_KEYWORDS
        if (keyword not in read if keyword in named else arguments.kwarg is None)
    ]


def _provider(call):
    """Return the model provider expression that the get_model() call ``call`` passes, if any."""
    if call.args:
        return call.args[0]
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == "model_provider_func"), None
    )


def test_the_provider_check_finds_providers_and_their_ignored_keywords(tmp_path):
    """Providers resolve in the scope of the call, and their ignored keywords are found.

    A defined, nested, imported, partial, lambda or assigned provider is found; a branch sees its
    own import, and a function of the same name in another scope does not count. A provider that
    is a parameter, that is passed in ``**kwargs`` or that the script imports from a module out
    of reach is not found.
    """
    (tmp_path / "providers.py").write_text(
        "def imported(pre_process, post_process, config=None, pg_collection=None):\n"
        "    return config, pg_collection\n"
        "def rejected(pre_process, post_process):\n"
        "    pass\n",
        encoding="utf-8",
    )
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from functools import partial\n"
        "from megatron.training import get_model\n"
        "from providers import imported\n"
        "def rejects(pre_process, post_process):\n"
        "    pass\n"
        "def unrelated():\n"
        "    def imported(pre_process, post_process):\n"
        "        pass\n"
        "def main(passed_by_the_caller, choice, **kwargs):\n"
        "    def ignores(pre_process, post_process, config=None, pg_collection=None):\n"
        "        return config\n"
        "    get_model(rejects)\n"
        "    get_model(ignores)\n"
        "    get_model(partial(imported))\n"
        "    get_model(lambda pre_process, post_process, **kwargs: None)\n"
        "    get_model(model_provider_func=passed_by_the_caller)\n"
        "    assigned = partial(rejects)\n"
        "    get_model(assigned)\n"
        "    get_model(**kwargs)\n"
        "    if choice:\n"
        "        from providers import rejected as chosen\n"
        "        get_model(chosen)\n"
        "    else:\n"
        "        from providers import imported as chosen\n"
        "        get_model(chosen)\n"
        "def out_of_reach():\n"
        "    from elsewhere import imported\n"
        "    get_model(imported)\n",
        encoding="utf-8",
    )
    tree = ast.parse(probe.read_text(encoding="utf-8"))
    parents = _parents(tree)

    found = [
        (
            call.lineno,
            [
                _ignored_provider_keywords(definition)
                for definition in _provider_definitions(_provider(call), probe, parents)
            ],
        )
        for call in _get_model_calls(probe, root=tmp_path, tree=tree)
    ]

    assert found == [
        (12, [["config", "pg_collection"]]),
        (13, [["pg_collection"]]),
        (14, [[]]),
        (15, [[]]),
        (16, []),
        (18, [["config", "pg_collection"]]),
        (19, []),
        (22, [["config", "pg_collection"]]),
        (25, [[]]),
        (28, []),
    ]


def test_the_providers_given_to_get_model_take_its_keywords():
    """get_model() calls the model provider with config= and pg_collection=.

    A provider that rejects them fails with a TypeError. One that names them must read them, or
    it builds the model on other groups, or with another config, than get_model() passes; one
    that takes them through ``**kwargs`` is only checked to accept them. The check covers the
    providers that _provider_definitions() finds, among them those of OWN_PROVIDER_SCRIPTS, and
    skips the others.
    """
    checked, ignored = set(), []
    for path in sorted(set(_sources())):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = _get_model_calls(path, tree=tree)
        parents = _parents(tree) if calls else {}
        for call in calls:
            for definition in _provider_definitions(_provider(call), path, parents):
                checked.add(str(path.relative_to(REPO_ROOT)))
                ignored += [
                    f"{path.relative_to(REPO_ROOT)}:{call.lineno} {keyword}"
                    for keyword in _ignored_provider_keywords(definition)
                ]

    assert ignored == []
    assert set(OWN_PROVIDER_SCRIPTS) <= checked
