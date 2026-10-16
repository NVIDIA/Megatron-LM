# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Configuration gates for the independent dev-based GLM53 integration."""

import dataclasses
import functools
from argparse import ArgumentParser

import pytest
import torch
import torch.nn.functional as F

from megatron.core.enums import Fp8Recipe
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.core.utils import is_te_min_version


def test_hf_precision_flags_parse_through_real_training_cli():
    from megatron.training.arguments import add_megatron_arguments

    parser = add_megatron_arguments(ArgumentParser(allow_abbrev=False))
    defaults = parser.parse_args([])
    assert defaults.mtp_enable_hyper_connections is None
    assert defaults.mla_proj_use_quantization is True
    assert defaults.kda_conv_in_fp32 is False
    args = parser.parse_args(
        [
            "--kda-two-stage-gates",
            "--kda-disable-fp8",
            "--kda-conv-in-fp32",
            "--rmsnorm-cast-before-weight",
            "--mhc-norm-eps-inside-sqrt",
            "--mhc-keep-pre-in-fp32",
            "--no-mhc-learned-output-contract",
            "--no-mtp-enable-hyper-connections",
            "--dsa-indexer-kpool-key-dtype-compression",
            "--dsa-indexer-hf-scoring",
            "--no-dsa-indexer-qk-proj-use-quantization",
            "--no-dsa-indexer-weights-proj-use-quantization",
            "--no-mla-proj-use-quantization",
        ]
    )
    assert args.kda_two_stage_gates is args.kda_disable_fp8 is True
    assert args.kda_conv_in_fp32 is True
    assert args.rmsnorm_cast_before_weight is True
    assert args.mhc_norm_eps_inside_sqrt is args.mhc_keep_pre_in_fp32 is True
    assert args.mhc_keep_mappings_in_fp32 is False
    assert args.mhc_learned_output_contract is args.mtp_enable_hyper_connections is False
    assert args.dsa_indexer_kpool_key_dtype_compression is True
    assert args.dsa_indexer_hf_scoring is True
    assert args.dsa_indexer_kpool_use_quantization is False
    assert args.dsa_indexer_qk_proj_use_quantization is False
    assert args.dsa_indexer_weights_proj_use_quantization is False
    assert args.mla_proj_use_quantization is False


def _dsa_config(**kwargs):
    values = dict(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        experimental_attention_variant="dsa",
        add_bias_linear=False,
        dsa_indexer_topk=32,
        dsa_indexer_kpool=4,
        dsa_indexer_loss_coeff=0.0,
    )
    values.update(kwargs)
    return MLATransformerConfig(**values)


def test_defaults_do_not_change_existing_model_semantics():
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)
    assert config.kda_two_stage_gates is False
    assert config.kda_disable_fp8 is False
    assert config.kda_conv_in_fp32 is False
    assert config.dsa_indexer_kpool == 1
    assert config.dsa_indexer_kpool_key_dtype_compression is False
    assert config.dsa_indexer_hf_scoring is False
    assert config.dsa_indexer_qk_proj_use_quantization is True
    assert config.mhc_norm_eps_inside_sqrt is False
    assert config.mhc_keep_mappings_in_fp32 is False
    assert config.mhc_keep_pre_in_fp32 is False
    assert config.mhc_learned_output_contract is True
    assert config.mtp_enable_hyper_connections is None
    assert config.rmsnorm_cast_before_weight is False


@pytest.mark.parametrize("quantized_scores", [False, True])
def test_hf_pool_compression_is_independent_of_score_quantization(quantized_scores):
    config = _dsa_config(
        dsa_indexer_kpool_key_dtype_compression=True,
        dsa_indexer_kpool_use_quantization=quantized_scores,
    )
    assert config.dsa_indexer_kpool_key_dtype_compression is True
    assert config.dsa_indexer_kpool_use_quantization is quantized_scores


@pytest.mark.parametrize(
    "overrides",
    [
        {"dsa_indexer_kpool": 1},
        {
            "dsa_indexer_weights_proj_use_quantization": False,
            "dsa_indexer_weights_proj_output_dtype": "fp32",
        },
        {"dsa_indexer_scoring_relu": False},
    ],
)
def test_hf_scoring_rejects_unimplemented_precision_paths(overrides):
    with pytest.raises(ValueError, match="dsa_indexer_hf_scoring requires"):
        _dsa_config(dsa_indexer_hf_scoring=True, **overrides)


@pytest.mark.parametrize("pool", [0, -1])
def test_non_positive_pool_size_is_rejected(pool):
    with pytest.raises(ValueError, match="dsa_indexer_kpool must be positive"):
        _dsa_config(dsa_indexer_kpool=pool)


def test_pool_budget_must_be_divisible():
    with pytest.raises(ValueError, match="divisible"):
        _dsa_config(dsa_indexer_topk=31)


def test_trainable_pool_selector_is_not_silently_enabled():
    with pytest.raises(ValueError, match="indexer loss is not supported"):
        _dsa_config(dsa_indexer_loss_coeff=0.01)


def test_unquantized_kpool_weights_can_bypass_cudnn_indexer_dtype_contract(monkeypatch):
    import megatron.core.transformer.transformer_config as config_module

    monkeypatch.setattr(config_module, "_validate_dsa_kernel_backend_dependencies", lambda _: None)
    config = _dsa_config(
        dsa_kernel_backend="cudnn",
        dsa_indexer_weights_proj_use_quantization=False,
        dsa_indexer_weights_proj_output_dtype="fp32",
    )
    assert config.dsa_indexer_kpool == 4
    with pytest.raises(ValueError, match="per-token DSA"):
        _dsa_config(
            dsa_kernel_backend="cudnn",
            dsa_indexer_kpool=1,
            dsa_indexer_weights_proj_use_quantization=False,
            dsa_indexer_weights_proj_output_dtype="fp32",
        )


@pytest.mark.parametrize(
    "flag", ["mhc_norm_eps_inside_sqrt", "mhc_keep_mappings_in_fp32", "mhc_keep_pre_in_fp32"]
)
def test_fused_mhc_does_not_silently_ignore_precision_flags(flag):
    with pytest.raises(ValueError, match="use_fused_mhc does not implement"):
        TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            enable_hyper_connections=True,
            use_fused_mhc=True,
            **{flag: True},
        )


# Target GLM53 FP8 recipe: E4M3 blockwise scaling with FP8 primary weights (fp8_param_gather).
_BLOCKWISE_FP8_PARAM_GATHER = dict(fp8="e4m3", fp8_recipe="blockwise", fp8_param=True)

# Paths GLM53 keeps outside FP8 under that recipe: the KDA, MLA and DSA indexer projections,
# plus the KPool score inputs. The KDA, MLA and indexer modules read these flags when they
# build and run those paths, so a config that rewrote one would quantize a path the HF
# numerical profile keeps in BF16.
_GLM53_FP8_OPT_OUTS = {
    "kda_disable_fp8": True,
    "mla_proj_use_quantization": False,
    "dsa_indexer_qk_proj_use_quantization": False,
    "dsa_indexer_weights_proj_use_quantization": False,
    "dsa_indexer_kpool_use_quantization": False,
}


def _glm53_profile(**overrides):
    """GLM53 proxy model under the single-node TP1/PP1/CP1/EP8/ETP1 BF16 readiness geometry.

    Only width, depth and expert count are reduced; every precision and routing flag is the
    production value. The defaults are the BF16 profile, including post-FC2 expert weighting.
    """
    values = dict(
        num_layers=8,
        hidden_size=1024,
        ffn_hidden_size=3072,
        num_attention_heads=8,
        kv_channels=256,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        rmsnorm_cast_before_weight=True,
        activation_func=F.silu,
        gated_linear_unit=True,
        bias_activation_fusion=True,
        activation_func_clamp_value=10.0,
        q_lora_rank=512,
        kv_lora_rank=512,
        qk_head_dim=256,
        qk_pos_emb_head_dim=0,
        v_head_dim=256,
        qk_layernorm=True,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        kda_two_stage_gates=True,
        kda_disable_fp8=True,
        kda_conv_in_fp32=True,
        kda_core_in_fp32=True,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        transformer_impl="transformer_engine",
        attention_backend=AttnBackend.fused,
        dsa_kernel_backend="cudnn",
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=128,
        dsa_indexer_kpool=4,
        dsa_indexer_kpool_use_quantization=False,
        dsa_indexer_kpool_key_dtype_compression=True,
        dsa_indexer_hf_scoring=True,
        dsa_indexer_qk_proj_use_quantization=False,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_k_norm_epsilon=1e-6,
        dsa_indexer_weights_proj_use_quantization=False,
        dsa_indexer_weights_proj_output_dtype="bf16",
        dsa_indexer_loss_coeff=0.0,
        mla_proj_use_quantization=False,
        enable_hyper_connections=True,
        num_residual_streams=4,
        mhc_sinkhorn_iterations=20,
        mhc_init_gating_factor=1.0,
        mhc_norm_eps_inside_sqrt=True,
        mhc_keep_mappings_in_fp32=False,
        mhc_keep_pre_in_fp32=True,
        mhc_preserve_activation_dtype_rounding=True,
        mhc_learned_output_contract=False,
        num_moe_experts=32,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        moe_router_load_balancing_type="none",
        moe_router_topk=4,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        moe_router_topk_scaling_factor=2.5,
        moe_router_score_function="sigmoid",
        moe_router_dtype="fp32",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0.0,
        moe_aux_loss_coeff=0.0,
        moe_apply_probs_after_fc2=True,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=True,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        mtp_enable_hyper_connections=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        init_method_std=0.02,
        is_hybrid_model=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )
    values.update(overrides)
    return _dsa_config(**values)


@pytest.fixture
def glm53_profile(monkeypatch):
    """Return the GLM53 profile builder, validated without touching a CUDA device."""
    if not is_te_min_version("2.3.0.dev0"):
        pytest.skip("GLM53 blockwise FP8 requires Transformer Engine >= 2.3 (Float8BlockScaling)")
    import megatron.core.transformer.transformer_config as config_module

    def accept_cudnn_without_device_probe(backend):
        # The proxy selects the fused cuDNN DSA backend. Its dependency probe needs a CUDA
        # device and cuDNN frontend, neither of which takes part in precision validation.
        assert backend == "cudnn"

    monkeypatch.setattr(
        config_module,
        "_validate_dsa_kernel_backend_dependencies",
        accept_cudnn_without_device_probe,
    )
    return _glm53_profile


def _comparable(value):
    """Init methods are rebuilt per config as partials; compare what they compute."""
    if isinstance(value, functools.partial):
        return (value.func, value.args, tuple(sorted(value.keywords.items())))
    return value


def test_glm53_blockwise_fp8_param_config_rewrites_no_field(glm53_profile):
    """The config accepts the GLM53 blockwise fp8_param recipe, and __post_init__ rewrites no
    field, the FP8 opt-out flags included.

    This is a config-level property only. It does not check whether the modules honour the
    opt-outs when they build FP8 primary weights; that is a module-level property.
    """
    bf16_config = glm53_profile()
    fp8_config = glm53_profile(**_BLOCKWISE_FP8_PARAM_GATHER, moe_apply_probs_after_fc2=False)

    # The BF16 control proves the profile itself is valid with post-FC2 weighting.
    assert bf16_config.fp8 is None
    assert bf16_config.fp8_param is False
    assert bf16_config.moe_apply_probs_after_fc2 is True

    assert fp8_config.fp8 == "e4m3"
    assert fp8_config.fp8_recipe == Fp8Recipe.blockwise
    assert fp8_config.fp8_param is True
    assert fp8_config.moe_apply_probs_after_fc2 is False
    assert fp8_config.activation_func_fp8_input_store is False
    # Catches a rewrite that __post_init__ applies to both configs, which the diff below
    # cannot see.
    for name, expected in _GLM53_FP8_OPT_OUTS.items():
        assert getattr(fp8_config, name) is expected, name

    # Catches a rewrite that only the FP8 recipe triggers, on any field, opt-outs and the HF
    # numerical-profile flags included: the two configs differ in the knobs we set.
    changed = {
        field.name
        for field in dataclasses.fields(fp8_config)
        if _comparable(getattr(bf16_config, field.name))
        != _comparable(getattr(fp8_config, field.name))
    }
    assert changed == {"fp8", "fp8_recipe", "fp8_param", "moe_apply_probs_after_fc2"}


def test_glm53_blockwise_fp8_recipe_floors_backward_gradient_amax(glm53_profile):
    """The GLM53 blockwise recipe carries the #7523 backward-gradient amax_epsilon=1e-12 floor.

    Every other recipe setting keeps TE's Float8BlockScaling default, and the floor is set on
    this recipe instance, not on TE's class-wide default.
    """
    from transformer_engine.common import recipe as te_recipe

    from megatron.core.fp8_utils import get_fp8_recipe

    config = glm53_profile(**_BLOCKWISE_FP8_PARAM_GATHER, moe_apply_probs_after_fc2=False)
    block_scaling = te_recipe.Float8BlockScaling
    # TE declares the QParams without annotations, so they are class attributes rather than
    # dataclass fields, and recipe equality below ignores them. Snapshot the class defaults.
    default_qparams = {
        name: getattr(block_scaling, name)
        for name in ("fp8_quant_fwd_inp", "fp8_quant_fwd_weight", "fp8_quant_bwd_grad")
    }

    # Building a TE recipe is host-only; no CUDA kernel or arch-specific path is touched.
    recipe = get_fp8_recipe(config)

    assert type(recipe) is block_scaling
    # Dataclass equality covers every annotated field: format, block dims, GEMM params and
    # the DPA/MHA switches all stay at TE's defaults for an E4M3 recipe.
    assert recipe == block_scaling(fp8_format=te_recipe.Format.E4M3)
    # The identity checks rule out a class-level override; the forward equalities then rule
    # out an instance-level one that changes a forward QParams value.
    for name, default in default_qparams.items():
        assert getattr(block_scaling, name) is default, name
    assert recipe.fp8_quant_fwd_inp == default_qparams["fp8_quant_fwd_inp"]
    assert recipe.fp8_quant_fwd_weight == default_qparams["fp8_quant_fwd_weight"]
    # Megatron passes the literal 1e-12, so exact equality is the right comparison.
    assert recipe.fp8_quant_bwd_grad.amax_epsilon == 1e-12
    assert recipe.fp8_quant_bwd_grad == dataclasses.replace(
        default_qparams["fp8_quant_bwd_grad"], amax_epsilon=1e-12
    )


@pytest.mark.parametrize(
    "quantization",
    [
        pytest.param(_BLOCKWISE_FP8_PARAM_GATHER, id="fp8-param-gather"),
        pytest.param(dict(fp8="e4m3", fp8_recipe="blockwise"), id="fp8"),
        # fp8_param without fp8 is invalid on its own, but _validate_moe_probs_after_fc2 runs
        # right after ModelParallelConfig.__post_init__, ahead of the generic fp8_param check.
        pytest.param(dict(fp8_param=True), id="fp8-param-without-fp8"),
        pytest.param(dict(activation_func_fp8_input_store=True), id="fp8-activation-store"),
    ],
)
def test_glm53_post_fc2_probs_reject_quantized_expert_paths(glm53_profile, quantization):
    with pytest.raises(ValueError, match="supports non-quantized BF16/FP32 only"):
        glm53_profile(moe_apply_probs_after_fc2=True, **quantization)
