# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import operator
import sys
import warnings
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.extensions import transformer_engine as te_extension
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.quantization import custom_recipe, te_recipe
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

RECORDED_ROLES = []


def delayed_scaling_test_factory(role):
    """Pinned-TE-compatible factory used by real MCore execution tests."""
    from transformer_engine.common.recipe import Format
    from transformer_engine.pytorch.quantization import DelayedScalingRequest

    RECORDED_ROLES.append(role)
    return DelayedScalingRequest(fp8_format=Format.HYBRID)


TEST_FACTORY_PATH = f"{__name__}.delayed_scaling_test_factory"


def test_resolve_quantizer_factory_returns_callable():
    assert custom_recipe.resolve_quantizer_factory("operator.neg") is operator.neg


@pytest.mark.parametrize(
    ("path", "error"),
    [
        ("", "must be a non-empty string"),
        ("factory", "Expected 'package.module.callable'"),
        ("missing_package.module.factory", "Failed to import module"),
        ("operator.missing_factory", "Attribute 'missing_factory' not found"),
        ("operator.__doc__", "is not callable"),
    ],
)
def test_resolve_quantizer_factory_rejects_invalid_paths(path, error):
    with pytest.raises(ValueError, match=error):
        custom_recipe.resolve_quantizer_factory(path)


@pytest.mark.skipif(custom_recipe.te_recipe is None, reason="Transformer Engine is not installed")
def test_build_custom_recipe_forwards_attention_flags():
    factory = Mock()

    recipe = custom_recipe.build_custom_recipe(factory, fp8_dpa=True, fp8_mha=True)

    assert recipe.qfactory is factory
    assert recipe.fp8_dpa is True
    assert recipe.fp8_mha is True


@pytest.mark.skipif(custom_recipe.te_recipe is None, reason="Transformer Engine is not installed")
def test_build_custom_recipe_rejects_unsupported_attention_flags():
    class RecipeWithoutAttentionFlags:

        def __init__(self, qfactory):
            self.qfactory = qfactory

    with patch.object(custom_recipe.te_recipe, "CustomRecipe", RecipeWithoutAttentionFlags):
        with pytest.raises(ValueError, match="supports the 'fp8_dpa' constructor argument"):
            custom_recipe.build_custom_recipe(Mock(), fp8_dpa=True)


@pytest.mark.skipif(custom_recipe.te_recipe is None, reason="Transformer Engine is not installed")
@pytest.mark.parametrize(
    ("config_kwargs", "recipe_getter"),
    [
        (
            {"fp8": "hybrid", "fp8_recipe": "custom", "fp8_quantizer_factory": TEST_FACTORY_PATH},
            "fp8",
        ),
        (
            {"fp4": "e2m1", "fp4_recipe": "custom", "fp4_quantizer_factory": TEST_FACTORY_PATH},
            "fp4",
        ),
    ],
)
def test_legacy_custom_recipe_paths_forward_attention_flags(config_kwargs, recipe_getter):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        fp8_dot_product_attention=True,
        fp8_multi_head_attention=True,
        **config_kwargs,
    )
    if recipe_getter == "fp8":
        from megatron.core.fp8_utils import get_fp8_recipe

        recipe = get_fp8_recipe(config)
        cached_recipe = get_fp8_recipe(config)
    else:
        from megatron.core.fp4_utils import get_fp4_recipe

        recipe = get_fp4_recipe(config)
        cached_recipe = get_fp4_recipe(config)

    assert cached_recipe is recipe
    assert recipe.qfactory is delayed_scaling_test_factory
    assert recipe.fp8_dpa is True
    assert recipe.fp8_mha is True


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"fp8": "hybrid", "fp8_recipe": "custom", "fp8_quantizer_factory": TEST_FACTORY_PATH},
        {"fp4": "e2m1", "fp4_recipe": "custom", "fp4_quantizer_factory": TEST_FACTORY_PATH},
    ],
)
def test_legacy_custom_recipe_warns_once(config_kwargs):
    mode = "fp8" if "fp8" in config_kwargs else "fp4"
    custom_recipe._WARNED_LEGACY_CUSTOM_RECIPE_MODES.discard(mode)
    try:
        with pytest.warns(
            FutureWarning, match="Pass the factory path to --custom-recipe"
        ) as warnings:
            for _ in range(2):
                TransformerConfig(
                    num_layers=1, hidden_size=128, num_attention_heads=4, **config_kwargs
                )
        custom_recipe_warnings = [
            warning
            for warning in warnings
            if issubclass(warning.category, FutureWarning)
            and f"--{mode}-recipe custom" in str(warning.message)
        ]
        assert len(custom_recipe_warnings) == 1
    finally:
        custom_recipe._WARNED_LEGACY_CUSTOM_RECIPE_MODES.discard(mode)


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
def test_per_module_custom_recipe_uses_format_neutral_autocast():
    config = te_extension.TEQuantizationRecipe(
        fp8_quantization_recipe=Fp8Recipe.custom,
        fp8_format="hybrid",
        custom_recipe_factory=TEST_FACTORY_PATH,
        override_nonquantized_autocast=True,
    )
    expected_context = nullcontext()

    with (
        patch.object(te_extension.FP8GlobalStateManager, "is_fp8_enabled", return_value=False),
        patch.object(
            te_extension.te.pytorch, "autocast", return_value=expected_context
        ) as autocast,
    ):
        context = te_extension._get_fp8_autocast_for_quant_recipe(config)

    assert context is expected_context
    assert autocast.call_args.kwargs["recipe"].qfactory is delayed_scaling_test_factory


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
def test_per_module_custom_recipe_rejects_quantized_parameter_storage():
    with pytest.raises(ValueError, match="do not yet support quantized parameter storage"):
        te_extension.TEQuantizationRecipe.parse_from_config(
            {
                "fp8_quantization_recipe": Fp8Recipe.custom,
                "custom_recipe_factory": TEST_FACTORY_PATH,
                "fp8_param": True,
            }
        )


@pytest.mark.skipif(not te_recipe.HAVE_TE, reason="Transformer Engine is not installed")
def test_get_quantization_context_uses_format_neutral_te_api():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        custom_recipe=TEST_FACTORY_PATH,
        fp8_dot_product_attention=True,
        fp8_multi_head_attention=True,
    )
    expected_context = nullcontext()

    with patch.object(te_recipe.te, "autocast", return_value=expected_context) as autocast:
        context = te_recipe.get_quantization_context(config)

    assert context is expected_context
    call_kwargs = autocast.call_args.kwargs
    assert call_kwargs["enabled"] is True
    assert call_kwargs["amax_reduction_group"] is None
    assert call_kwargs["recipe"].qfactory is delayed_scaling_test_factory
    assert call_kwargs["recipe"].fp8_dpa is True
    assert call_kwargs["recipe"].fp8_mha is True


@pytest.mark.skipif(not te_recipe.HAVE_TE, reason="Transformer Engine is not installed")
def test_custom_recipe_init_context_is_noop_without_parameter_storage():
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, custom_recipe=TEST_FACTORY_PATH
    )

    with te_recipe.get_quantization_context(config, is_init=True):
        pass


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
def test_custom_recipe_is_materialized_once_per_config():
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, custom_recipe=TEST_FACTORY_PATH
    )

    first = te_recipe.get_te_quantization_recipe(config)
    second = te_recipe.get_te_quantization_recipe(config)

    assert first is second


CUSTOM_RECIPE_SPELLINGS = [
    pytest.param({"custom_recipe": TEST_FACTORY_PATH}, id="canonical"),
    pytest.param(
        {"fp8": "hybrid", "fp8_recipe": "custom", "fp8_quantizer_factory": TEST_FACTORY_PATH},
        id="legacy-fp8",
    ),
    pytest.param(
        {"fp4": "e2m1", "fp4_recipe": "custom", "fp4_quantizer_factory": TEST_FACTORY_PATH},
        id="legacy-fp4",
    ),
]


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.parametrize("config_kwargs", CUSTOM_RECIPE_SPELLINGS)
def test_custom_recipe_alignment_uses_declared_value_or_conservative_fallback(config_kwargs):
    from megatron.core.transformer.moe.moe_utils import get_align_size_for_quantization

    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, **config_kwargs
    )
    recipe = te_recipe.get_te_quantization_recipe(config)

    # Every custom spelling must use the recipe's declared alignment rather than the
    # 16-byte default of the legacy FP8 enum.
    expected = getattr(recipe, "quantization_alignment", 128)
    assert expected != 16
    assert te_recipe.get_quantization_alignment(config) == expected
    assert get_align_size_for_quantization(config) == expected
    with patch.object(te_recipe, "get_te_quantization_recipe", return_value=object()):
        assert te_recipe.get_quantization_alignment(config) == 128


@pytest.mark.parametrize(
    ("quant_kwargs", "num_local_experts", "expected"),
    [
        pytest.param({}, 1, False, id="bf16-single-local-expert"),
        pytest.param({}, 2, True, id="bf16-multi-local-expert"),
        pytest.param({"fp8": "hybrid", "fp8_recipe": "mxfp8"}, 1, True, id="builtin-fp8"),
        pytest.param({"custom_recipe": TEST_FACTORY_PATH}, 1, False, id="custom-single"),
        pytest.param({"custom_recipe": TEST_FACTORY_PATH}, 2, True, id="custom-multi"),
        pytest.param(
            {"fp8": "hybrid", "fp8_recipe": "custom", "fp8_quantizer_factory": TEST_FACTORY_PATH},
            1,
            False,
            id="legacy-custom-single",
        ),
    ],
)
def test_should_free_input_keeps_high_precision_rule_for_custom_recipes(
    quant_kwargs, num_local_experts, expected
):
    """Custom factories may return Identity quantizers that alias the MLP input.

    Only built-in FP8/FP4 recipes guarantee that TE saved a separate quantized copy, so
    only they may free the expert input when the dispatcher hands it over unchanged.
    """
    from megatron.core.models.common.utils import should_free_input

    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        moe_ffn_hidden_size=128,
        moe_token_dispatcher_type="alltoall",
        **quant_kwargs,
    )

    assert should_free_input("mlp", True, config, num_local_experts) is expected


@pytest.mark.skipif(not te_recipe.HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.parametrize(
    ("quant_kwargs", "expected"),
    [
        pytest.param({}, False, id="bf16"),
        pytest.param({"fp8": "hybrid", "fp8_recipe": "delayed"}, False, id="fp8-delayed"),
        pytest.param({"fp8": "hybrid", "fp8_recipe": "mxfp8"}, True, id="fp8-mxfp8"),
        pytest.param({"fp4": "e2m1", "fp4_recipe": "nvfp4"}, True, id="fp4-nvfp4"),
        pytest.param({"custom_recipe": TEST_FACTORY_PATH}, "te-validates", id="custom"),
        pytest.param(
            {"fp8": "hybrid", "fp8_recipe": "custom", "fp8_quantizer_factory": TEST_FACTORY_PATH},
            "te-validates",
            id="legacy-custom",
        ),
    ],
)
def test_supports_save_original_input_matches_recipe_family(quant_kwargs, expected):
    """Custom recipes rely on TE's per-quantizer validation of save_original_input."""
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4, **quant_kwargs)
    if expected == "te-validates":
        expected = te_recipe._te_validates_save_original_input()
        # Without TE's validation helper, custom recipes must stay conservative.
        with patch.dict(sys.modules, {"transformer_engine.pytorch.module._common": None}):
            assert te_recipe.supports_save_original_input(config) is False

    assert te_recipe.supports_save_original_input(config) is expected


def test_mfsdp_v2_rejects_custom_recipe_at_startup():
    from types import SimpleNamespace

    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV2

    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, custom_recipe=TEST_FACTORY_PATH
    )
    ddp_config = DistributedDataParallelConfig(data_parallel_sharding_strategy="optim_grads_params")
    pg_collection = SimpleNamespace(
        dp_cp=object(), tp=None, pp=None, cp=None, ep=None, expt_dp=None
    )

    with pytest.raises(ValueError, match="custom quantization recipes"):
        FullyShardedDataParallelV2._validate_config(
            config, ddp_config, torch.nn.Linear(2, 2), pg_collection, disable_bucketing=False
        )


def _skip_unless_fp8_available():
    availability = te_extension.te.pytorch.is_fp8_available()
    if isinstance(availability, tuple):
        fp8_available, reason = availability
    else:
        fp8_available, reason = availability, "FP8 execution is unavailable"
    if not fp8_available:
        pytest.skip(reason)


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_custom_recipe_dense_forward_backward_receives_semantic_role_names():
    _skip_unless_fp8_available()

    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        RECORDED_ROLES.clear()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            params_dtype=torch.bfloat16,
            custom_recipe=TEST_FACTORY_PATH,
            fp8_dot_product_attention=True,
        )
        block = TransformerBlock(
            config, get_gpt_layer_with_transformer_engine_spec(), name="decoder"
        )
        # The projection reuses the attention output saved by DPA instead of a quantized
        # copy; TE validates the factory's quantizer at runtime.
        linear_proj = block.layers[0].self_attention.linear_proj
        assert linear_proj.save_original_input is te_recipe.supports_save_original_input(config)
        hidden_states = torch.randn(
            16, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = block(hidden_states=hidden_states, attention_mask=None)
        output.float().square().mean().backward()
        if linear_proj.save_original_input:
            # The delayed-scaling test factory cannot requantize deterministically, so TE must
            # downgrade the optimization with a warning rather than fail or silently drift.
            assert any("save_original_input" in str(w.message) for w in caught)

        assert torch.isfinite(output).all()
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()
        role_names = {role.name for role in RECORDED_ROLES if role is not None}
        assert {
            "decoder.layers.0.self_attention.linear_qkv",
            "decoder.layers.0.self_attention.linear_proj",
            "decoder.layers.0.mlp.linear_fc1",
            "decoder.layers.0.mlp.linear_fc2",
            "decoder.layers.0.self_attention.core_attention",
        } <= role_names
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_custom_recipe_grouped_moe_checkpoint_extra_state_is_stateless():
    """Grouped experts must checkpoint without unpickling the CustomRecipe factory.

    TE stores the CustomRecipe object, including its quantizer factory, in the TE extra
    state, exactly as it does for the built-in recipes. MCore's own ``SafeUnpickler`` cannot
    decode an arbitrary factory, so the per-GEMM split used by distributed checkpointing
    stores an empty extra state. That is lossless for a stateless factory; this test uses a
    delayed-scaling factory, so it must also warn that amax history is being dropped.
    """
    _skip_unless_fp8_available()

    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            custom_recipe=TEST_FACTORY_PATH,
            num_moe_experts=2,
            moe_ffn_hidden_size=256,
            moe_grouped_gemm=True,
            moe_router_topk=2,
            moe_token_dispatcher_type="alltoall",
            moe_router_padding_for_quantization=True,
        )
        block = TransformerBlock(
            config,
            get_gpt_layer_with_transformer_engine_spec(num_experts=2, moe_grouped_gemm=True),
            name="decoder",
        )
        hidden_states = torch.randn(
            16, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        output = block(hidden_states=hidden_states, attention_mask=None)
        output.float().square().mean().backward()

        # Distributed-checkpoint save path: one extra state per expert GEMM. This factory
        # produces delayed-scaling quantizers, whose state cannot be split per GEMM, so the
        # save must say so rather than drop amax history silently.
        te_extension._warn_custom_recipe_extra_state_dropped.cache_clear()
        with pytest.warns(UserWarning, match="delayed-scaling state"):
            sharded_state_dict = block.sharded_state_dict()
        grouped_extra_states = {
            key: value
            for key, value in sharded_state_dict.items()
            if ".experts.linear_fc" in key and "_extra_state" in key
        }
        assert len(grouped_extra_states) == 4, sorted(grouped_extra_states)
        for sharded_object in grouped_extra_states.values():
            assert sharded_object.data.numel() == 0

        # Distributed-checkpoint load path: per-GEMM empty states are merged back by the
        # grouped linear's load pre-hook without decoding any pickled recipe.
        grouped_linear = block.layers[0].mlp.experts.linear_fc1
        grouped_state_dict = grouped_linear.state_dict()
        grouped_state_dict["_extra_state"] = torch.empty(0, dtype=torch.uint8)
        for gemm_idx in range(1, grouped_linear.num_gemms):
            grouped_state_dict[f"_extra_state{gemm_idx}"] = torch.empty(0, dtype=torch.uint8)
        grouped_linear.load_state_dict(grouped_state_dict)
    finally:
        Utils.destroy_model_parallel()


class _NameLessTELinear:
    """Stand-in for a Transformer Engine constructor that predates the ``name`` argument."""

    def __init__(self, in_features, out_features, bias=True):
        pass


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.parametrize(
    "config_kwargs",
    [
        pytest.param({}, id="bf16"),
        pytest.param({"fp8": "hybrid", "fp8_recipe": Fp8Recipe.mxfp8}, id="builtin_fp8"),
    ],
)
def test_te_name_kwarg_is_omitted_on_older_te_without_custom_recipe(config_kwargs):
    """``name`` must not be forwarded to a TE constructor that cannot accept it.

    MCore supports a wide range of TE versions and uses ``name`` for its own per-module
    matcher regardless of TE support. Forwarding it unconditionally raises TypeError on
    every BF16 and built-in FP8 run against a TE release without the keyword.
    """
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, **config_kwargs
    )
    assert (
        te_extension._te_name_kwarg(_NameLessTELinear, "decoder.layers.0.mlp.linear_fc1", config)
        == {}
    )
    # The stand-in constructor must actually reject the keyword we just declined to pass.
    with pytest.raises(TypeError):
        _NameLessTELinear(128, 128, name="decoder.layers.0.mlp.linear_fc1")


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
def test_te_name_kwarg_raises_for_custom_recipe_on_older_te():
    """A custom recipe may select quantizers by name, so a dropped name must not be silent."""
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, custom_recipe=TEST_FACTORY_PATH
    )
    with pytest.raises(RuntimeError, match="does not accept a 'name' argument"):
        te_extension._te_name_kwarg(_NameLessTELinear, "decoder.layers.0.mlp.linear_fc1", config)


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
def test_te_name_kwarg_is_forwarded_when_supported():
    config = TransformerConfig(
        num_layers=1, hidden_size=128, num_attention_heads=4, custom_recipe=TEST_FACTORY_PATH
    )
    name = "decoder.layers.0.mlp.linear_fc1"
    for te_class in (
        te_extension.te.pytorch.Linear,
        te_extension.te.pytorch.LayerNormLinear,
        te_extension.te.pytorch.GroupedLinear,
        te_extension.te.pytorch.DotProductAttention,
    ):
        if te_extension._te_constructor_accepts_name(te_class):
            assert te_extension._te_name_kwarg(te_class, name, config) == {"name": name}
    # A module with no semantic name never contributes the keyword.
    assert te_extension._te_name_kwarg(te_extension.te.pytorch.Linear, None, config) == {}


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
def test_safe_unpickler_allowlists_builtin_recipes_but_not_custom_recipe():
    """Pin why grouped custom-recipe extra state cannot be split per GEMM.

    This is a Megatron-side constraint, not a Transformer Engine one: TE serializes a
    ``CustomRecipe`` into the extra state exactly as it does the built-in recipes, but
    decoding one here would also have to resolve the factory's arbitrary callable, which is
    precisely what ``SafeUnpickler`` exists to prevent. If ``CustomRecipe`` is ever added to
    the allowlist, ``_split_extra_state`` can preserve delayed-scaling state instead of
    dropping it, and this test should be updated together with that change.
    """
    from megatron.core.safe_globals import SafeUnpickler

    recipe_module = "transformer_engine.common.recipe"
    for builtin in (
        "DelayedScaling",
        "Float8CurrentScaling",
        "Float8BlockScaling",
        "MXFP8BlockScaling",
        "NVFP4BlockScaling",
    ):
        assert (recipe_module, builtin) in SafeUnpickler._SAFE_CLASSES
    assert (recipe_module, "CustomRecipe") not in SafeUnpickler._SAFE_CLASSES


def _te_module_names(block):
    """Collect the semantic names Megatron actually handed to Transformer Engine."""
    return {
        module.name
        for module in block.modules()
        if isinstance(getattr(module, "name", None), str) and ".layers." in module.name
    }


def _layer_indices_in_names(names, suffix="self_attention.linear_qkv"):
    return sorted(
        int(name.split(".layers.")[1].split(".")[0]) for name in names if name.endswith(suffix)
    )


def _pp_naming_config(**kwargs):
    return TransformerConfig(
        num_layers=4,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        **kwargs,
    )


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_layer_names_are_global_across_pipeline_stages():
    """A layer's semantic name must identify the same layer regardless of the PP split.

    With stage-local indices, stage 1 of a 4-layer PP=2 model also emits ``decoder.layers.0``,
    so a factory selecting on ``.layers.0.`` silently keeps one layer per stage in a different
    precision instead of only the model's first layer.
    """
    Utils.initialize_model_parallel(1, 2)
    try:
        model_parallel_cuda_manual_seed(123)
        config = _pp_naming_config(pipeline_model_parallel_size=2)
        block = TransformerBlock(
            config, get_gpt_layer_with_transformer_engine_spec(), name="decoder"
        )
        indices = _layer_indices_in_names(_te_module_names(block))
        pp_rank = torch.distributed.get_rank()
        expected = [0, 1] if pp_rank == 0 else [2, 3]
        assert indices == expected, f"pp_rank={pp_rank} got {indices}, expected {expected}"
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not te_extension.HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_layer_names_are_global_across_virtual_pipeline_stages():
    """Virtual pipeline stages must not restart layer numbering either."""
    Utils.initialize_model_parallel(1, 2)
    try:
        model_parallel_cuda_manual_seed(123)
        config = _pp_naming_config(
            pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2
        )
        pp_rank = torch.distributed.get_rank()
        seen = {}
        for vp_stage in range(2):
            block = TransformerBlock(
                config,
                get_gpt_layer_with_transformer_engine_spec(),
                name="decoder",
                vp_stage=vp_stage,
            )
            seen[vp_stage] = _layer_indices_in_names(_te_module_names(block))
        # 4 layers over PP=2 x VPP=2 is one layer per (pp_rank, vp_stage) chunk, interleaved.
        assert seen[0] == [pp_rank], seen
        assert seen[1] == [2 + pp_rank], seen
        # Every emitted index is unique model-wide, which is the property factories rely on.
        flat = seen[0] + seen[1]
        assert len(set(flat)) == len(flat), seen
    finally:
        Utils.destroy_model_parallel()
