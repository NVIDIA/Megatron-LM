# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the recipe loader behind the ``--model-recipe`` flag.

The loader accepts three spec forms (see :func:`load_recipe`):

1. Dotted Python path (``foo.bar``)
2. Dotted path with explicit function selection (``foo.bar:func_name``)
3. Filesystem path to a ``.py`` file (with optional ``:func`` suffix)

When ``:func`` is omitted, the loader calls the module's ``make_recipe()``
function. These tests use synthetic in-memory modules for dotted-path loading
and temporary files for filesystem-path loading.
"""

import sys
import textwrap
import types
from pathlib import Path

import pytest

from megatron.core.models.hybrid.common_layer_config import CommonLayerConfig
from megatron.core.models.hybrid.hybrid_model_config import CompiledRecipe, HybridModelConfig
from megatron.core.models.hybrid.layer_configs import (
    AttentionLayerConfig,
    CrossEntropyLayerConfig,
    EmbeddingLayerConfig,
    MambaLayerConfig,
)
from megatron.core.models.hybrid.layer_pattern import RECIPE_ENTRY_POINT, load_recipe


def _make_common(**overrides) -> CommonLayerConfig:
    base = dict(hidden_size=128, use_cpu_initialization=True)
    base.update(overrides)
    return CommonLayerConfig(**base)


def _make_valid_pattern(common):
    return [
        EmbeddingLayerConfig(
            common_config=common,
            vocab_size=1024,
            max_sequence_length=512,
            position_embedding_type="rope",
        ),
        AttentionLayerConfig(common_config=common, num_attention_heads=4),
        AttentionLayerConfig(common_config=common, num_attention_heads=4),
        CrossEntropyLayerConfig(),
    ]


def _register_synthetic_module(monkeypatch, name: str, **attrs) -> types.ModuleType:
    """Build a fresh :class:`types.ModuleType` with the requested attributes
    and inject into ``sys.modules`` for the test's lifetime."""
    module = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(module, k, v)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_recipe_entry_point_constant():
    """The canonical entry-point name is the documented public constant."""
    assert RECIPE_ENTRY_POINT == "make_recipe"


# ──────────────────────────────────────────────────────────────────────────
# Resolution form 1 — module:func explicit selection
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestExplicitFuncSelection:

    def test_module_colon_func_selects_named_function(self, monkeypatch):
        common = _make_common()
        recipe = HybridModelConfig(common_config=common, layer_pattern=_make_valid_pattern(common))

        def my_pretrain_recipe() -> HybridModelConfig:
            return recipe

        # Other (non-recipe) module attrs should be ignored.
        def helper(x):
            return x + 1

        _register_synthetic_module(
            monkeypatch,
            "tests._explicit_func_recipe",
            my_pretrain_recipe=my_pretrain_recipe,
            helper=helper,
        )
        compiled = load_recipe("tests._explicit_func_recipe:my_pretrain_recipe")
        assert isinstance(compiled, CompiledRecipe)

    def test_module_colon_func_missing_function(self, monkeypatch):
        _register_synthetic_module(monkeypatch, "tests._explicit_func_missing")
        with pytest.raises(AttributeError, match="missing_func"):
            load_recipe("tests._explicit_func_missing:missing_func")

    def test_module_colon_func_returning_wrong_type(self, monkeypatch):
        def bad() -> HybridModelConfig:
            return 42  # type: ignore[return-value]

        _register_synthetic_module(monkeypatch, "tests._explicit_func_wrong_type", bad=bad)
        with pytest.raises(TypeError, match="HybridModelConfig"):
            load_recipe("tests._explicit_func_wrong_type:bad")


# ──────────────────────────────────────────────────────────────────────────
# Resolution form 2 — canonical ``make_recipe``
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestMakeRecipeDefault:

    def test_make_recipe_used_when_no_other_resolution(self, monkeypatch):
        common = _make_common()
        recipe = HybridModelConfig(common_config=common, layer_pattern=_make_valid_pattern(common))

        def make_recipe():
            return recipe

        _register_synthetic_module(
            monkeypatch, "tests._make_recipe_convention", make_recipe=make_recipe
        )
        compiled = load_recipe("tests._make_recipe_convention")
        assert isinstance(compiled, CompiledRecipe)

    def test_make_recipe_used_even_when_other_functions_exist(self, monkeypatch):
        common = _make_common()
        default_recipe = HybridModelConfig(
            common_config=common,
            layer_pattern=_make_valid_pattern(common),
            tensor_model_parallel_size=2,
        )
        variant_recipe = HybridModelConfig(
            common_config=common,
            layer_pattern=_make_valid_pattern(common),
            tensor_model_parallel_size=4,
        )

        def make_recipe() -> HybridModelConfig:
            return default_recipe

        def debug_recipe() -> HybridModelConfig:
            return variant_recipe

        _register_synthetic_module(
            monkeypatch,
            "tests._make_recipe_default",
            make_recipe=make_recipe,
            debug_recipe=debug_recipe,
        )
        compiled = load_recipe("tests._make_recipe_default")
        assert compiled.config.tensor_model_parallel_size == 2
        explicit = load_recipe("tests._make_recipe_default:debug_recipe")
        assert explicit.config.tensor_model_parallel_size == 4

    def test_make_recipe_must_be_callable(self, monkeypatch):
        common = _make_common()
        recipe = HybridModelConfig(common_config=common, layer_pattern=_make_valid_pattern(common))
        _register_synthetic_module(monkeypatch, "tests._non_callable_recipe", make_recipe=recipe)
        with pytest.raises(TypeError, match="not callable"):
            load_recipe("tests._non_callable_recipe")

    def test_module_with_nothing_resolvable_errors(self, monkeypatch):
        _register_synthetic_module(monkeypatch, "tests._empty_recipe")
        with pytest.raises(AttributeError, match="no make_recipe"):
            load_recipe("tests._empty_recipe")

    def test_unimportable_module_raises(self):
        with pytest.raises(ImportError, match="could not be imported"):
            load_recipe("definitely.does.not.exist.module")


# ──────────────────────────────────────────────────────────────────────────
# Filesystem-path form
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestFilesystemPath:

    def _write_recipe_file(self, tmp_path: Path, body: str) -> Path:
        path = tmp_path / "my_recipe.py"
        path.write_text(textwrap.dedent(body))
        return path

    def test_file_path_with_make_recipe(self, tmp_path):
        path = self._write_recipe_file(
            tmp_path,
            """
            from megatron.core.models.hybrid import (
                AttentionLayerConfig, CommonLayerConfig, CrossEntropyLayerConfig,
                EmbeddingLayerConfig, HybridModelConfig,
            )
            def make_recipe() -> HybridModelConfig:
                common = CommonLayerConfig(hidden_size=128, use_cpu_initialization=True)
                return HybridModelConfig(
                    common_config=common,
                    layer_pattern=[
                        EmbeddingLayerConfig(common_config=common, vocab_size=1024,
                                             max_sequence_length=512, position_embedding_type="rope"),
                        AttentionLayerConfig(common_config=common, num_attention_heads=4),
                        CrossEntropyLayerConfig(),
                    ],
                )
            """,
        )
        compiled = load_recipe(str(path))
        assert isinstance(compiled, CompiledRecipe)

    def test_file_path_with_explicit_func(self, tmp_path):
        path = self._write_recipe_file(
            tmp_path,
            """
            from megatron.core.models.hybrid import (
                AttentionLayerConfig, CommonLayerConfig, CrossEntropyLayerConfig,
                EmbeddingLayerConfig, HybridModelConfig,
            )
            def my_pretrain_config() -> HybridModelConfig:
                common = CommonLayerConfig(hidden_size=128, use_cpu_initialization=True)
                return HybridModelConfig(
                    common_config=common,
                    layer_pattern=[
                        EmbeddingLayerConfig(common_config=common, vocab_size=1024,
                                             max_sequence_length=512, position_embedding_type="rope"),
                        AttentionLayerConfig(common_config=common, num_attention_heads=4),
                        CrossEntropyLayerConfig(),
                    ],
                )
            """,
        )
        compiled = load_recipe(f"{path}:my_pretrain_config")
        assert isinstance(compiled, CompiledRecipe)

    def test_file_path_does_not_exist(self, tmp_path):
        path = tmp_path / "nope.py"
        with pytest.raises(ImportError, match="does not exist"):
            load_recipe(str(path))


# ──────────────────────────────────────────────────────────────────────────
# Compile-time errors propagate
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.internal
class TestCompileErrorPropagation:

    def test_bad_pattern_propagates_typeerror(self, monkeypatch):
        common = _make_common()
        bad = HybridModelConfig(
            common_config=common,
            layer_pattern=[MambaLayerConfig(common_config=common), CrossEntropyLayerConfig()],
        )

        def make_recipe() -> HybridModelConfig:
            return bad

        _register_synthetic_module(monkeypatch, "tests._bad_pattern_recipe", make_recipe=make_recipe)
        with pytest.raises(TypeError, match="EmbeddingLayerConfig"):
            load_recipe("tests._bad_pattern_recipe")


@pytest.mark.internal
class TestRecipeArgProjection:
    """The launcher should learn model-shape/topology args from --model-recipe.

    This keeps recipe authors from passing dummy legacy shape flags whose
    values are ignored later by hybrid_builder.
    """

    def test_apply_model_recipe_to_args_sets_legacy_namespace_fields(self, monkeypatch):
        from megatron.training.arguments import _apply_model_recipe_to_args

        common = _make_common()
        recipe = HybridModelConfig(
            common_config=common,
            layer_pattern=_make_valid_pattern(common),
            tensor_model_parallel_size=2,
        )
        _register_synthetic_module(
            monkeypatch,
            "tests._args_projection_recipe",
            make_recipe=lambda: recipe,
        )
        args = types.SimpleNamespace(
            model_recipe="tests._args_projection_recipe",
            num_layers=None,
            hidden_size=None,
            num_attention_heads=None,
            max_position_embeddings=None,
            padded_vocab_size=None,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=None,
            position_embedding_type=None,
            rotary_percent=None,
            rotary_base=None,
            rotary_seq_len_interpolation_factor=None,
            untie_embeddings_and_output_weights=False,
            fp16_lm_cross_entropy=False,
            mtp_num_layers=None,
            encoder_num_layers=123,
            num_experts=None,
        )

        _apply_model_recipe_to_args(args)

        assert args._compiled_model_recipe.config.num_layers == 2
        assert args.num_layers == 2
        assert args.encoder_num_layers is None
        assert args.hidden_size == common.hidden_size
        assert args.num_attention_heads == 4
        assert args.max_position_embeddings == 512
        assert args.padded_vocab_size == 1024
        assert args.tensor_model_parallel_size == 2
