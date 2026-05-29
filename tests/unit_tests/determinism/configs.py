# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared config presets for parametrized determinism tests.

Each test in this package parametrizes its model factory with a list of
(name, overrides) pairs from below. Add a new entry here to widen the
determinism net to a new architecture variant — no changes needed in the
test files themselves.

Parallelism is expressed as a single composite dict like
``{"TP": 4, "FSDP": 2}`` or ``{"PP": 2, "VPP": 2, "EP": 4}``; see
``PARALLELISM_CONFIGS``. ``apply_parallelism`` translates the dict into
``Utils.initialize_model_parallel`` kwargs and returns flags for FSDP-wrap
and MoE auto-enable.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Base configs — everything below is shared across presets. Override fields
# in the per-preset dict only when they differ from the base.
# ---------------------------------------------------------------------------

_BASE_GPT = dict(
    num_layers=2,
    hidden_size=64,
    ffn_hidden_size=128,  # default is 4*hidden=256; halve for cheaper MLP
    num_attention_heads=8,
    use_cpu_initialization=True,
    bf16=True,
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    sequence_parallel=False,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    deterministic_mode=True,
)

# Hybrid / Mamba layers constrain hidden_size, so the base is wider.
# num_attention_heads must be ≥ max(TP) so the attention layer can shard
# evenly under TP=8 (otherwise: "heads must be divisible by GQA groups").
_BASE_HYBRID = dict(
    num_layers=3,
    # Mamba derives nheads = d_inner / head_dim = (hidden*expand) / 64
    # and requires nheads % ngroups (=8) == 0. hidden=256 → nheads=8 ✓.
    # Smaller hidden_size breaks the divisibility.
    hidden_size=256,
    num_attention_heads=8,
    use_cpu_initialization=True,
    bf16=True,
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    sequence_parallel=False,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    deterministic_mode=True,
)

# Overrides that turn a dense GPT preset into a MoE one. Applied automatically
# by tests when the chosen parallelism dict sets EP > 1.
_MOE_OVERRIDES = dict(
    num_moe_experts=4,
    moe_router_topk=2,
    moe_grouped_gemm=True,
    add_bias_linear=False,
)


def gpt_base() -> dict:
    return dict(_BASE_GPT)


def hybrid_base() -> dict:
    return dict(_BASE_HYBRID)


def moe_overrides(tp: int = 1, ep: int = 1) -> dict:
    """Return MoE overrides. When ``tp > 1`` we must also enable
    ``sequence_parallel`` (MoE+TP without SP raises in moe_layer.py) and
    propagate ``tensor_model_parallel_size`` into the config (otherwise
    the SP validator sees TP=1 in the config and rejects SP=True). When
    ``ep > 1`` we must propagate ``expert_model_parallel_size`` into the
    config — ``parallel_state`` initialising EP groups is not enough;
    ``ColumnParallelLinear``/``RowParallelLinear`` reads
    ``config.expert_model_parallel_size`` to decide whether expert weights
    use the expert tp_group or the dense tp_group."""
    overrides = dict(_MOE_OVERRIDES)
    if tp > 1:
        overrides["sequence_parallel"] = True
        overrides["tensor_model_parallel_size"] = tp
    if ep > 1:
        overrides["expert_model_parallel_size"] = ep
    return overrides


# ---------------------------------------------------------------------------
# Model presets — fed to @pytest.mark.parametrize. Each pytest.param's first
# arg is a dict of TransformerConfig overrides; the `id=` controls the test
# ID pytest prints (handy for `-k <preset-id>`).
# ---------------------------------------------------------------------------

GPT_CONFIGS = [
    # ``gpt-like``    — multi-head attention, LayerNorm, plain MLP (GPT-2 family).
    # ``llama-like``  — grouped-query attention, RMSNorm, gated linear unit
    #                   (Llama / modern-decoder family).
    # Perf coverage (det vs nondet breakdown) lives outside this file —
    # ``scripts/determinism/run_nsys_breakdown.sh`` wraps the actual training
    # entry point under ``nsys profile``. There is no pytest-side perf cell.
    pytest.param({}, id="gpt-like"),
    pytest.param(
        dict(
            num_query_groups=2,
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
        ),
        id="llama-like",
    ),
]


HYBRID_CONFIGS = [
    # mamba-attn-mlp covers Mamba + attention + MLP paths. pure-mamba is
    # dropped (Mamba path alone is already exercised here).
    pytest.param("M*-", {}, id="mamba-attn-mlp"),
]


# ---------------------------------------------------------------------------
# Composite parallelism configs.
#
# Each entry is a dict over the shortname keys below. ``apply_parallelism``
# normalises and forwards them to ``Utils.initialize_model_parallel``.
#
#   TP    tensor_model_parallel_size
#   PP    pipeline_model_parallel_size
#   VPP   virtual_pipeline_model_parallel_size
#   CP    context_parallel_size
#   EP    expert_model_parallel_size      (implies MoE preset)
#   FSDP  data-parallel sharding size     (wraps model with fully_shard_model)
#
# A test must skip an entry if Utils.world_size cannot host it; see
# ``required_world_size``.
# ---------------------------------------------------------------------------

PARALLELISM_CONFIGS = [
    # Pure TP.
    pytest.param({"TP": 4}, id="tp4"),
    pytest.param({"TP": 8}, id="tp8"),
    # MoE + EP.
    pytest.param({"EP": 2}, id="ep2"),
    # MoE + TP × EP composites.
    pytest.param({"TP": 2, "EP": 2}, id="tp2-ep2"),
    pytest.param({"TP": 2, "EP": 4}, id="tp2-ep4"),
    # FSDP — pure and EP composite.
    pytest.param({"FSDP": 8}, id="fsdp8"),
    pytest.param({"FSDP": 8, "EP": 4}, id="fsdp8-ep4"),
    # PP — verified via pipeline schedule + NaN-aware equality.
    pytest.param({"PP": 2}, id="pp2"),
    pytest.param({"PP": 4}, id="pp4"),
    pytest.param({"TP": 2, "PP": 2}, id="tp2-pp2"),
    # VPP — _build_gpt forwards vp_stage to GPTModel so each virtual chunk
    # gets the correct layer slice; runner uses num_layers = pp*vpp (one
    # layer per chunk; was bumped 2× before the vp_stage fix landed).
    pytest.param({"PP": 2, "VPP": 2}, id="pp2-vpp2"),
]


def parallelism_configs(*, exclude: tuple[str, ...] = ()) -> list:
    """``PARALLELISM_CONFIGS`` filtered by pytest-param id.

    Use this in a test file's parametrize when the test exercises a
    subset of the matrix — for example a hybrid-only test that doesn't
    cover TP=8, or a layer-only test that drops PP composites. Prefer
    this over runtime ``pytest.skip`` calls in the test body so the
    parametrize matrix reflects what actually runs.
    """
    excluded = set(exclude)
    return [p for p in PARALLELISM_CONFIGS if p.id not in excluded]


_SHORTNAME_TO_INIT_KWARG = {
    "TP": "tensor_model_parallel_size",
    "PP": "pipeline_model_parallel_size",
    "VPP": "virtual_pipeline_model_parallel_size",
    "CP": "context_parallel_size",
    "EP": "expert_model_parallel_size",
}


# ---------------------------------------------------------------------------
# FP8 / FP4 recipe coverage.
#
# Each cell that exercises a specific quantization recipe carries it as an
# explicit field in its TransformerConfig overrides — there is no global
# attention-backend toggle. The TE attention backend is whatever NVTE's
# default selection picks at first attention call; the deterministic-mode
# guard at megatron/training/determinism.py rejects ``--use-flash-attn``
# outright, so flash-attn is never reached under the determinism contract.
#
# FP8 recipe (specified per-cell in TransformerConfig overrides):
#   fp8='hybrid' / fp8='e4m3'  + fp8_recipe='tensorwise' | 'delayed' | 'mxfp8'
#
# FP4 recipe:
#   fp4='e2m1'  + fp4_recipe='nvfp4'
# ---------------------------------------------------------------------------


def apply_parallelism(parallelism: dict) -> tuple[dict, bool, bool]:
    """Translate a composite parallelism dict into init kwargs + flags.

    Returns:
        init_kwargs: kwargs for ``Utils.initialize_model_parallel``.
        needs_fsdp: True if the dict requests FSDP > 1.
        needs_moe:  True if the dict requests EP > 1 (caller should merge
                    ``moe_overrides()`` into the model config).
    """
    init_kwargs = {}
    for shortname, init_key in _SHORTNAME_TO_INIT_KWARG.items():
        if shortname in parallelism:
            init_kwargs[init_key] = parallelism[shortname]
    needs_fsdp = parallelism.get("FSDP", 1) > 1
    needs_moe = parallelism.get("EP", 1) > 1
    return init_kwargs, needs_fsdp, needs_moe


def required_world_size(parallelism: dict) -> int:
    """Total GPUs needed. FSDP and EP both shard along DP, so DP-size is
    ``max(FSDP, EP, 1)``; PP and CP and TP multiply in independently."""
    tp = parallelism.get("TP", 1)
    pp = parallelism.get("PP", 1)
    cp = parallelism.get("CP", 1)
    dp = max(parallelism.get("FSDP", 1), parallelism.get("EP", 1), 1)
    return tp * pp * cp * dp
