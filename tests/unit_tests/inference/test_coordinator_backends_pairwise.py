# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import partial

import pytest
from transformer_engine.pytorch.fp8 import check_mxfp8_support, check_nvfp4_support

from megatron.core.transformer import attention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.inference import coordinator_pairwise_utils as utils
from tests.unit_tests.inference import test_coordinator_features_pairwise as features
from tests.unit_tests.inference.engines import test_dynamic_engine as engines

# Decide hardware support at collection so every rank skips the same rows.
MXFP8, NVFP4 = check_mxfp8_support(), check_nvfp4_support()


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "version", "symbols"),
    [
        ("optimized", None, ()),
        ("fa2", 2, ("flash_attn_varlen_func", "flash_attn_with_kvcache")),
        ("fa4", 4, ("flash_attn4_varlen_func",)),
    ],
)
async def test_routed_dense_backends_match_direct(monkeypatch, backend, version, symbols):
    build = engines.DynamicInferenceEngineTestBase._build_test_env
    routed = partial(features.routed_model, engine_factory=lambda config: build(config).engine)
    monkeypatch.setattr(features, "routed_model", routed)
    instrument = features._instrument_scenario_runtime

    def instrument_native(env, scenario, runtime):
        instrument(env, scenario, runtime)
        for symbol in symbols:
            kernel = getattr(attention, symbol)

            def observed(*args, _kernel=kernel, **kwargs):
                result = _kernel(*args, **kwargs)
                phase = "decode" if env.engine.context.is_decode_only() else "prefill"
                runtime[f"{backend}-{phase}"] += 1
                return result

            monkeypatch.setattr(attention, symbol, observed)

    monkeypatch.setattr(features, "_instrument_scenario_runtime", instrument_native)
    options = dict(flash_attention_version=version or utils.FLASH_ATTENTION_VERSION)
    options["transformer_impl"] = "inference_optimized" if version is None else "local"
    if version == 2:
        options.update(context_block_size_tokens=256, context_max_tokens=256)
    if version == 4:
        options["hidden_size"] = 128
    signals = ("inference-optimized",) if version is None else ()
    required = (
        ("module-forward:inference-optimized",)
        if version is None
        else (f"{backend}-prefill", f"{backend}-decode")
    )
    await features.test_routed_model_kernels_match_direct(monkeypatch, options, signals, required)


@pytest.mark.internal
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "mla",
        pytest.param("mxfp8", marks=pytest.mark.skipif(not MXFP8[0], reason=MXFP8[1])),
        pytest.param("nvfp4", marks=pytest.mark.skipif(not NVFP4[0], reason=NVFP4[1])),
    ],
)
async def test_routed_native_model_modes_match_direct(monkeypatch, mode):
    """One-request physical batches preserve ordinary, non-BI precision semantics."""
    constructor = MLATransformerConfig if mode == "mla" else engines.TransformerConfig
    overrides = dict(
        flash_attention_version=utils.FLASH_ATTENTION_VERSION if mode == "mla" else 4,
        num_layers=2,
        add_bias_linear=False,
    )
    options = dict(hidden_size=256, context_max_requests=128, context_max_tokens=128)
    if mode == "mla":
        overrides.update(
            multi_latent_attention=True,
            cache_mla_latents=True,
            num_attention_heads=64,
            qk_layernorm=True,
        )
        options.update(context_block_size_tokens=64, position_embedding_type="rope")
        spec = engines.get_gpt_layer_with_transformer_engine_spec
        monkeypatch.setattr(
            engines,
            "get_gpt_layer_with_transformer_engine_spec",
            partial(spec, multi_latent_attention=True, qk_layernorm=True),
        )
    else:
        overrides.update(
            dict(fp8="hybrid", fp8_recipe="mxfp8")
            if mode == "mxfp8"
            else dict(fp4="e2m1", fp4_recipe="nvfp4")
        )
        # Both direct and routed GEMMs consume the same 128-row physical batch.
        for name in ("TOKEN_ROUNDER", "REQUEST_ROUNDER"):
            monkeypatch.setattr(engines.DynamicInferenceContext, name, 128)
        monkeypatch.setattr(engines, "set_rounder", lambda _: None)
    monkeypatch.setattr(engines, "TransformerConfig", lambda **kw: constructor(**(kw | overrides)))
    build = engines.DynamicInferenceEngineTestBase._build_test_env
    monkeypatch.setattr(
        features,
        "routed_model",
        partial(features.routed_model, engine_factory=lambda c: build(c).engine),
    )
    instrument = features._instrument_scenario_runtime

    def instrument_native(env, scenario, runtime):
        instrument(env, scenario, runtime)
        if mode == "mla":
            sources = [(attention, "flash_mla_with_kvcache")]
        else:
            from transformer_engine.pytorch.module import layernorm_linear, linear

            sources = [(linear, "general_gemm"), (layernorm_linear, "general_gemm")]
        for module, symbol in sources:
            kernel = getattr(module, symbol)

            def observed(*args, _kernel=kernel, **kwargs):
                result = _kernel(*args, **kwargs)
                if not features._active_target_ids(env):
                    return result
                context = env.engine.context
                if mode == "mla":
                    assert context.cache_mla_latent and args[1].shape[-1] == 576
                    assert args[4] == 512 and context.is_decode_only()
                else:
                    assert all(mode.upper() + "Tensor" in type(a).__name__ for a in args[:2])
                    assert context.padded_active_token_count == 128
                    assert any(a.size(0) == 128 for a in args[:2])
                phase = "decode" if context.is_decode_only() else "prefill"
                runtime[f"{mode}-{phase}"] += 1
                return result

            monkeypatch.setattr(module, symbol, observed)

    monkeypatch.setattr(features, "_instrument_scenario_runtime", instrument_native)
    required = (f"{mode}-decode",) if mode == "mla" else (f"{mode}-prefill", f"{mode}-decode")
    await features.test_routed_model_kernels_match_direct(monkeypatch, options, (), required)
