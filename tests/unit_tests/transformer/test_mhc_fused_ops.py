# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-operation fused mHC selection (``mhc_fused_ops``): config, CLI, binding and logging.

Every test only constructs configs and modules, apart from the CPU ``_projection_and_get_norm``
forward of the GLM projection pair's refusal test, so all of them run on CPU. CI runs each test
on 8 ranks at once, so nothing here assumes rank 0: the logging tests patch the rank helpers,
and an autouse fixture resets the module's once-per-process latches for every test.
"""

import copy
import json
import logging
import os
import subprocess
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core import _rank_utils
from megatron.core.fusions import fused_mhc_glm_projection as glm
from megatron.core.fusions import fused_mhc_kernels as fmk
from megatron.core.transformer import hyper_connection as hc
from megatron.core.transformer.mhc_fused_ops import MHC_FUSED_OPS, MHC_PRECISION_FLAGS
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.mhc_fused_op_spy import SPIED_ATTRIBUTES, spy_fused_mhc_ops

REPO_ROOT = Path(__file__).resolve().parents[3]

# The production GLM5Next mHC precision flags (test_glm53_hybrid_native_parity.py).
GLM_FLAGS = dict(
    mhc_norm_eps_inside_sqrt=True,
    mhc_normalize_before_projection=True,
    mhc_keep_mappings_in_fp32=False,
    mhc_keep_pre_in_fp32=True,
    mhc_preserve_activation_dtype_rounding=True,
)
CONFIG_PREFIX = "use_fused_mhc does not support this mhc_fused_ops selection: "
MODULE_PREFIX = "Fused mHC does not support this mhc_fused_ops selection: "
CONFIG_LEGACY_MESSAGE = (
    "use_fused_mhc does not support mhc_norm_eps_inside_sqrt, mhc_normalize_before_projection, "
    "mhc_keep_mappings_in_fp32, mhc_keep_pre_in_fp32 or mhc_preserve_activation_dtype_rounding; "
    "use the unfused mHC path."
)
MODULE_LEGACY_MESSAGE = (
    "Fused mHC does not support FP32 mixing, epsilon inside sqrt, normalize-before-projection or "
    "activation-dtype rounding preservation."
)
TRITON_SINKHORN = "triton(num_warps=1)"
# Every bound operation that the unfused binding holds as a module-level object (or None).
MODULE_LEVEL_OPS = (
    "_sinkhorn_op",
    "_h_aggregate_op",
    "_h_aggregate_into_op",
    "_h_post_bda_op",
    "_proj_rms_compute_h_op",
    "_glm_proj_rms_op",
    "_fused_add_3_op",
)


def _config(**overrides):
    values = dict(
        num_layers=1, hidden_size=32, num_attention_heads=4, enable_hyper_connections=True
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _per_op(ops, **overrides):
    return _config(use_fused_mhc=True, mhc_fused_ops=list(ops), **overrides)


def _module(config):
    return hc.HyperConnectionModule(config, layer_number=1)


def _assert_partial(op, func, **keywords):
    # functools.partial has no __eq__, and modules build fresh partials.
    assert isinstance(op, partial), op
    assert (op.func, op.args, op.keywords) == (func, (), keywords)


def _assert_same_binding(module, reference, exclude=()):
    """Assert that ``module`` binds the objects the unfused ``reference`` binds.

    The attributes in ``exclude`` are not compared; the caller checks them.
    """
    for name in MODULE_LEVEL_OPS:
        if name not in exclude:
            assert getattr(module, name) is getattr(reference, name), name
    _assert_partial(module._proj_rms_op, hc._proj_rms_impl, **reference._proj_rms_op.keywords)


def _run_python(script, env):
    """Run ``script`` in a fresh interpreter and return the JSON object it prints last."""
    pythonpath = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(env, PYTHONPATH=pythonpath),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


class _InfoLines(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines = []

    def emit(self, record):
        if record.levelno == logging.INFO:
            self.lines.append(record.getMessage())


@pytest.fixture(autouse=True)
def _fresh_process_state(monkeypatch):
    """Each test starts as a process that has logged nothing yet, without MHC_FORCE_BACKEND.

    The fused kernels read MHC_FORCE_BACKEND once, at import. Tests that need its effect set
    the state it leaves, or start a fresh interpreter.
    """
    monkeypatch.setattr(hc, "_FUSED_OPS_LINES_LOGGED", set())
    monkeypatch.setattr(hc, "_PER_OP_FALLBACK_WARNED", set())
    monkeypatch.setattr(fmk, "_MHC_FORCED_BACKEND", "auto")
    monkeypatch.setattr(fmk, "_MHC_BACKEND_VALIDATION_ERROR", None)
    monkeypatch.delenv("MHC_FORCE_BACKEND", raising=False)


@pytest.fixture
def warnings_logged(monkeypatch):
    """Record hyper_connection's single-rank WARNINGs: the real helper logs on rank 0 only."""
    records = []
    monkeypatch.setattr(
        hc,
        "log_single_rank",
        lambda logger, level, message, **kwargs: records.append((level, message)),
    )
    return records


@pytest.fixture
def fused_ops_lines(monkeypatch):
    """Record the per-rank fused-ops lines, logged as a fixed non-zero rank.

    The single-rank logging helpers read the rank too, so a line logged on rank 0 only would
    be missing here, on every rank.
    """
    monkeypatch.setattr(hc, "safe_get_rank", lambda: 5)
    monkeypatch.setattr(_rank_utils, "safe_get_rank", lambda: 5)
    monkeypatch.setattr(hc.logger, "disabled", False)
    handler = _InfoLines()
    level = hc.logger.level
    hc.logger.setLevel(logging.INFO)
    hc.logger.addHandler(handler)
    try:
        yield handler.lines
    finally:
        hc.logger.removeHandler(handler)
        hc.logger.setLevel(level)


@pytest.fixture
def triton_only_fmk(monkeypatch):
    """Pin the fused kernels' availability: Triton without cuTile."""
    monkeypatch.setattr(fmk, "is_triton_available", lambda: True)
    monkeypatch.setattr(fmk, "is_cutile_available", lambda: False)
    monkeypatch.setattr(fmk, "_TRITON_IMPLS", {name: object() for name in fmk._TRITON_IMPLS})


# --- TransformerConfig --------------------------------------------------------------------


def test_mhc_fused_ops_defaults():
    config = _config()
    assert config.mhc_fused_ops is None
    assert config.mhc_fused_tf32_rounding == "rn"


@pytest.mark.parametrize(
    "ops, flags",
    [
        (["proj_rms"], GLM_FLAGS),
        (["sinkhorn"], GLM_FLAGS),
        (["proj_rms", "sinkhorn"], GLM_FLAGS),
        (["sinkhorn", "proj_rms"], GLM_FLAGS),
        (["proj_rms", "sinkhorn"], dict(GLM_FLAGS, mhc_keep_mappings_in_fp32=True)),
    ],
    ids=["proj_rms", "sinkhorn", "proj_rms-sinkhorn", "sinkhorn-proj_rms", "keep_mappings_in_fp32"],
)
def test_glm_flags_accept_proj_rms_and_sinkhorn(ops, flags):
    config = _per_op(ops, **flags)
    assert config.mhc_fused_ops == ops


@pytest.mark.parametrize(
    "ops, flags, reasons",
    [
        (
            ["proj_rms"],
            dict(mhc_keep_mappings_in_fp32=True),
            ["proj_rms does not implement mhc_keep_mappings_in_fp32"],
        ),
        (
            ["proj_rms"],
            dict(mhc_keep_pre_in_fp32=True),
            ["proj_rms does not implement mhc_keep_pre_in_fp32"],
        ),
        # Either GLM norm flag alone: neither the projection-first op nor the GLM pair.
        (
            ["proj_rms"],
            dict(mhc_norm_eps_inside_sqrt=True),
            ["proj_rms does not implement mhc_norm_eps_inside_sqrt"],
        ),
        (
            ["proj_rms"],
            dict(mhc_normalize_before_projection=True),
            ["proj_rms does not implement mhc_normalize_before_projection"],
        ),
        (
            ["proj_rms"],
            dict(GLM_FLAGS, mhc_fused_backend="cutile"),
            [
                "proj_rms runs a Triton-only per-op implementation, which "
                "mhc_fused_backend='cutile' excludes"
            ],
        ),
        (
            ["sinkhorn"],
            dict(mhc_fused_backend="cutile"),
            [
                "sinkhorn runs a Triton-only per-op implementation, which "
                "mhc_fused_backend='cutile' excludes"
            ],
        ),
        (
            ["sinkhorn"],
            dict(GLM_FLAGS, mhc_fused_backend="cutile"),
            [
                "sinkhorn runs a Triton-only per-op implementation, which "
                "mhc_fused_backend='cutile' excludes"
            ],
        ),
        (
            ["h_aggregate"],
            dict(mhc_keep_pre_in_fp32=True),
            ["h_aggregate does not implement mhc_keep_pre_in_fp32"],
        ),
        (
            ["h_aggregate"],
            dict(mhc_keep_mappings_in_fp32=True),
            ["h_aggregate does not implement mhc_keep_mappings_in_fp32"],
        ),
        (
            ["h_post_bda"],
            dict(mhc_keep_mappings_in_fp32=True),
            ["h_post_bda does not implement mhc_keep_mappings_in_fp32"],
        ),
        (
            ["h_post_bda"],
            dict(mhc_preserve_activation_dtype_rounding=True),
            ["h_post_bda does not implement mhc_preserve_activation_dtype_rounding"],
        ),
        (
            list(MHC_FUSED_OPS),
            GLM_FLAGS,
            [
                "h_aggregate does not implement mhc_keep_pre_in_fp32",
                "h_post_bda does not implement mhc_preserve_activation_dtype_rounding",
            ],
        ),
    ],
    ids=[
        "proj_rms-keep_mappings",
        "proj_rms-keep_pre",
        "proj_rms-eps_inside_sqrt_alone",
        "proj_rms-normalize_before_projection_alone",
        "glm_proj_rms-cutile",
        "sinkhorn-cutile",
        "glm_sinkhorn-cutile",
        "h_aggregate-keep_pre",
        "h_aggregate-keep_mappings",
        "h_post_bda-keep_mappings",
        "h_post_bda-preserve_rounding",
        "glm-all_ops_listed",
    ],
)
def test_per_op_table_rejects(ops, flags, reasons):
    with pytest.raises(ValueError) as error:
        _per_op(ops, **flags)
    assert str(error.value) == CONFIG_PREFIX + "; ".join(reasons) + "."


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (
            dict(num_residual_streams=8),
            "num_residual_streams=8: per-op mode is validated at 4 only",
        ),
        (
            dict(mhc_recompute_attn_cuda_graph_split=True),
            "mhc_recompute_attn_cuda_graph_split is untested with per-op mode",
        ),
        (
            dict(cuda_graph_impl="transformer_engine"),
            "Transformer Engine per-layer CUDA graphs are untested with per-op mode",
        ),
        # The deprecated flag becomes cuda_graph_impl='transformer_engine' before the check.
        (
            dict(external_cuda_graph=True),
            "Transformer Engine per-layer CUDA graphs are untested with per-op mode",
        ),
    ],
    ids=["streams8", "attn_graph_split", "te_graphs", "external_cuda_graph"],
)
def test_per_op_v1_limits_reject(overrides, reason):
    with pytest.raises(ValueError) as error:
        _per_op(["proj_rms", "sinkhorn"], **GLM_FLAGS, **overrides)
    assert str(error.value) == CONFIG_PREFIX + reason + "."


@pytest.mark.parametrize(
    "overrides",
    [
        dict(recompute_granularity="selective", recompute_modules=["mhc"]),
        dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1),
        dict(cuda_graph_impl="local"),
        dict(
            cuda_graph_impl="full_iteration",
            recompute_granularity="selective",
            recompute_modules=["mhc"],
            hidden_dropout=0.0,
            attention_dropout=0.0,
        ),
    ],
    ids=["mhc_recompute", "full_recompute", "local_graphs", "full_iteration_mhc_recompute"],
)
def test_per_op_allows_recompute_and_full_iteration(overrides):
    config = _per_op(["proj_rms", "sinkhorn"], **GLM_FLAGS, **overrides)
    assert config.mhc_fused_ops == ["proj_rms", "sinkhorn"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(mhc_fused_ops=["sinkhorn"]), "mhc_fused_ops requires use_fused_mhc=True."),
        (
            dict(use_fused_mhc=True, mhc_fused_ops=[]),
            "mhc_fused_ops must name at least one operation; set use_fused_mhc=False to fuse none.",
        ),
        (
            dict(use_fused_mhc=True, mhc_fused_ops="sinkhorn"),
            "mhc_fused_ops must be a list of operation names, got str.",
        ),
        (
            dict(use_fused_mhc=True, mhc_fused_ops=["post"]),
            "unknown mhc_fused_ops entries ['post']; expected entries of "
            "('proj_rms', 'sinkhorn', 'h_aggregate', 'h_post_bda').",
        ),
        (
            dict(use_fused_mhc=True, mhc_fused_ops=["sinkhorn", "proj_rms", "sinkhorn"]),
            "mhc_fused_ops repeats ['sinkhorn'].",
        ),
    ],
    ids=["no_use_fused_mhc", "empty", "string", "unknown", "duplicates"],
)
def test_mhc_fused_ops_field_errors(overrides, message):
    with pytest.raises(ValueError) as error:
        _config(**overrides)
    assert str(error.value) == message


_RZ_REQUIRES_GLM_PAIR = (
    "mhc_fused_tf32_rounding='rz' requires use_fused_mhc=True and 'proj_rms' in mhc_fused_ops "
    "with mhc_norm_eps_inside_sqrt=True and mhc_normalize_before_projection=True."
)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            dict(mhc_fused_tf32_rounding="rd"),
            "unknown mhc_fused_tf32_rounding 'rd'; expected 'rn' or 'rz'.",
        ),
        (dict(mhc_fused_tf32_rounding="rz"), _RZ_REQUIRES_GLM_PAIR),
        (dict(use_fused_mhc=True, mhc_fused_tf32_rounding="rz"), _RZ_REQUIRES_GLM_PAIR),
        (
            dict(
                use_fused_mhc=True,
                mhc_fused_ops=["sinkhorn"],
                mhc_fused_tf32_rounding="rz",
                **GLM_FLAGS,
            ),
            _RZ_REQUIRES_GLM_PAIR,
        ),
        (
            dict(use_fused_mhc=True, mhc_fused_ops=["proj_rms"], mhc_fused_tf32_rounding="rz"),
            _RZ_REQUIRES_GLM_PAIR,
        ),
        (
            dict(
                use_fused_mhc=True,
                mhc_fused_ops=["proj_rms"],
                mhc_fused_tf32_rounding="rz",
                mhc_norm_eps_inside_sqrt=True,
            ),
            _RZ_REQUIRES_GLM_PAIR,
        ),
        (
            dict(
                use_fused_mhc=True,
                mhc_fused_ops=["proj_rms", "sinkhorn"],
                mhc_fused_tf32_rounding="rz",
                **GLM_FLAGS,
            ),
            None,
        ),
    ],
    ids=[
        "unknown",
        "unfused",
        "all_ops",
        "glm_sinkhorn_only",
        "proj_rms_eps_outside",
        "proj_rms_eps_inside_sqrt_alone",
        "glm_pair",
    ],
)
def test_tf32_rounding_requires_glm_pair(overrides, message):
    if message is None:
        assert _config(**overrides).mhc_fused_tf32_rounding == "rz"
        return
    with pytest.raises(ValueError) as error:
        _config(**overrides)
    assert str(error.value) == message


def test_all_ops_mode_keeps_both_legacy_messages():
    for flag in MHC_PRECISION_FLAGS:
        with pytest.raises(ValueError) as error:
            _config(use_fused_mhc=True, **{flag: True})
        assert str(error.value) == CONFIG_LEGACY_MESSAGE, flag

        # The module guard rechecks configs edited after validation.
        config = _config(use_fused_mhc=True)
        setattr(config, flag, True)
        with pytest.raises(ValueError) as error:
            _module(config)
        assert str(error.value) == MODULE_LEGACY_MESSAGE, flag


def test_module_guard_rechecks_the_table():
    config = _per_op(["sinkhorn"], **GLM_FLAGS)
    setattr(config, "mhc_fused_ops", ["h_post_bda"])
    with pytest.raises(ValueError) as error:
        _module(config)
    assert str(error.value) == (
        MODULE_PREFIX + "h_post_bda does not implement mhc_preserve_activation_dtype_rounding."
    )


@pytest.mark.parametrize(
    "dropped, kept",
    [
        ("mhc_normalize_before_projection", "mhc_norm_eps_inside_sqrt"),
        ("mhc_norm_eps_inside_sqrt", "mhc_normalize_before_projection"),
    ],
)
def test_module_guard_rejects_one_glm_norm_flag(dropped, kept):
    """A GLM selection edited after validation to drop one norm flag has no proj_rms
    implementation: the projection-first op implements neither flag, nor mhc_keep_pre_in_fp32."""
    config = _per_op(["proj_rms", "sinkhorn"], **GLM_FLAGS)
    setattr(config, dropped, False)
    with pytest.raises(ValueError) as error:
        _module(config)
    assert str(error.value) == (
        MODULE_PREFIX
        + f"proj_rms does not implement {kept}; "
        + "proj_rms does not implement mhc_keep_pre_in_fp32."
    )


@pytest.mark.parametrize("value, type_name", [("sinkhorn", "str"), (5, "int")])
def test_module_guard_checks_fields_before_the_table(value, type_name):
    """A malformed value gets the field error, never a table reason or a TypeError."""
    config = _per_op(["sinkhorn"], **GLM_FLAGS)
    setattr(config, "mhc_fused_ops", value)
    with pytest.raises(ValueError) as error:
        _module(config)
    assert str(error.value) == (
        MODULE_PREFIX + f"mhc_fused_ops must be a list of operation names, got {type_name}."
    )


def test_mhc_fused_ops_cli_flags():
    from megatron.training.arguments import add_megatron_arguments

    # ArgumentGroupFactory derives both flags from the TransformerConfig fields.
    parser = add_megatron_arguments(ArgumentParser(allow_abbrev=False))
    defaults = parser.parse_args([])
    assert defaults.mhc_fused_ops is None
    assert defaults.mhc_fused_tf32_rounding == "rn"
    args = parser.parse_args(
        ["--mhc-fused-ops", "proj_rms", "sinkhorn", "--mhc-fused-tf32-rounding", "rz"]
    )
    assert args.mhc_fused_ops == ["proj_rms", "sinkhorn"]
    assert args.mhc_fused_tf32_rounding == "rz"
    for unknown in (["--mhc-fused-ops", "post"], ["--mhc-fused-tf32-rounding", "rd"]):
        with pytest.raises(SystemExit) as error:
            parser.parse_args(unknown)
        assert error.value.code == 2


# --- HyperConnectionModule binding ----------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [{}, GLM_FLAGS, dict(use_fused_mhc=True), dict(use_fused_mhc=True, mhc_fused_backend="native")],
    ids=["plain", "glm_unfused", "all_ops", "all_ops_native"],
)
def test_unset_binds_todays_objects(overrides):
    """With mhc_fused_ops unset, a module binds what it bound before the per-op switch.

    Module-level objects and the cached emulate-casts compile compare by identity; the
    partials, which every module builds afresh, by their function, args and keywords.
    """
    config = _config(**overrides)
    module = _module(config)
    _assert_partial(
        module._proj_rms_op,
        hc._proj_rms_impl,
        eps_inside_sqrt=config.mhc_norm_eps_inside_sqrt,
        normalize_before_projection=config.mhc_normalize_before_projection,
    )
    assert module._fused_add_3_op is hc.native_fused_add_3
    assert hasattr(module._fused_add_3_op, "__wrapped__")
    assert module._glm_proj_rms_op is None
    if config.use_fused_mhc:
        backend = config.mhc_fused_backend
        _assert_partial(module._sinkhorn_op, fmk.fused_sinkhorn, backend=backend)
        _assert_partial(module._h_aggregate_op, fmk.fused_h_aggregate, backend=backend)
        _assert_partial(module._h_aggregate_into_op, fmk.fused_h_aggregate_into, backend=backend)
        _assert_partial(module._h_post_bda_op, fmk.fused_h_post_bda, backend=backend)
        _assert_partial(
            module._proj_rms_compute_h_op, fmk.fused_proj_rms_compute_h, backend=backend
        )
        return
    assert module._sinkhorn_op is hc.native_sinkhorn
    assert module._h_aggregate_op is hc.native_h_aggregate
    assert hasattr(module._h_aggregate_op, "__wrapped__")
    assert module._h_aggregate_into_op is hc.native_h_aggregate_into
    if config.mhc_preserve_activation_dtype_rounding:
        assert module._h_post_bda_op is hc._get_precision_preserving_h_post_bda()
    else:
        assert module._h_post_bda_op is hc.native_h_post_bda
    assert module._proj_rms_compute_h_op is None


# The attributes each operation rebinds in per-op mode without precision flags, and the
# function each binds: a partial of the fused kernels' function with the policy, or for
# sinkhorn the fixed-launch Triton Sinkhorn of fused_mhc_glm_projection itself.
_PER_OP_REBINDS = {
    "proj_rms": {"_proj_rms_compute_h_op": "fused_proj_rms_compute_h"},
    "sinkhorn": {"_sinkhorn_op": "fixed_launch_sinkhorn"},
    "h_aggregate": {
        "_h_aggregate_op": "fused_h_aggregate",
        "_h_aggregate_into_op": "fused_h_aggregate_into",
    },
    "h_post_bda": {"_h_post_bda_op": "fused_h_post_bda"},
}


@pytest.mark.parametrize(
    "op, policy",
    [(op, "auto") for op in MHC_FUSED_OPS]
    + [
        (op, policy)
        for policy in ("triton", "cutile")
        for op in ("proj_rms", "h_aggregate", "h_post_bda")
    ]
    + [("sinkhorn", "triton")],
)
def test_per_op_binds_only_listed_ops(monkeypatch, triton_only_fmk, warnings_logged, op, policy):
    """A listed operation binds its per-op implementation under the configured policy, and only it.

    sinkhorn binds the fixed-launch Triton Sinkhorn under auto and triton; cutile is rejected by
    the config.
    """
    if policy == "cutile":
        monkeypatch.setattr(fmk, "is_triton_available", lambda: False)
        monkeypatch.setattr(fmk, "is_cutile_available", lambda: True)
    module = _module(_per_op([op], mhc_fused_backend=policy))
    reference = _module(_config())
    rebinds = _PER_OP_REBINDS[op]
    for name in MODULE_LEVEL_OPS:
        if name not in rebinds:
            assert getattr(module, name) is getattr(reference, name), name
        elif op == "sinkhorn":
            # A module-level object, not a partial: the spy's identity check applies.
            assert getattr(module, name) is getattr(glm, rebinds[name]), name
        else:
            _assert_partial(getattr(module, name), getattr(fmk, rebinds[name]), backend=policy)
    _assert_partial(
        module._proj_rms_op,
        hc._proj_rms_impl,
        eps_inside_sqrt=False,
        normalize_before_projection=False,
    )

    labels = dict(module._fused_op_labels)
    if op == "sinkhorn":
        assert labels.pop(op) == TRITON_SINKHORN
    else:
        assert labels.pop(op).startswith(f"fmk({policy}:")
    assert warnings_logged == []
    assert labels == {
        name: label for name, label in reference._fused_op_labels.items() if name != op
    }


def test_per_op_glm_binding_keeps_the_parity_contract(triton_only_fmk, warnings_logged):
    """[proj_rms, sinkhorn] with the GLM flags keeps the objects the parity tests require.

    Only the Sinkhorn and the GLM projection pair change: sinkhorn binds the fixed-launch Triton
    Sinkhorn, and proj_rms the Triton GLM projection pair, which _projection_and_get_norm runs.
    """
    config = _per_op(["proj_rms", "sinkhorn"], **GLM_FLAGS)
    module = _module(config)
    # The projection runs inside _projection_and_get_norm, and the post is the emulate-casts
    # compile (test_glm53_hybrid_native_parity.py).
    assert module._proj_rms_compute_h_op is None
    assert module._glm_proj_rms_op is glm.glm_proj_rms_op("rn")
    assert module._h_post_bda_op is hc._get_precision_preserving_h_post_bda()
    assert module._proj_rms_op.keywords == {
        "eps_inside_sqrt": True,
        "normalize_before_projection": True,
    }
    assert module._sinkhorn_op is glm.fixed_launch_sinkhorn
    _assert_same_binding(
        module, _module(_config(**GLM_FLAGS)), exclude=("_sinkhorn_op", "_glm_proj_rms_op")
    )
    assert module._fused_op_labels == {
        "proj_rms": "glm-triton(tf32=rn)",
        "sinkhorn": TRITON_SINKHORN,
        "h_aggregate": "native",
        "h_post_bda": "native(emulate-casts)",
    }
    assert warnings_logged == []


@pytest.mark.parametrize("ops, flags", [(["proj_rms", "sinkhorn"], GLM_FLAGS)], ids=["glm"])
def test_per_op_explicit_triton_binds_the_glm_pair(triton_only_fmk, ops, flags):
    """An explicit triton policy constructs the GLM overlay and binds both Triton operations.

    The policy is enforced at call time: an input the pair does not support raises
    (test_per_op_glm_pair_refusal_policy).
    """
    module = _module(_per_op(ops, mhc_fused_backend="triton", **flags))
    assert module._glm_proj_rms_op is glm.glm_proj_rms_op("rn")
    assert module._sinkhorn_op is glm.fixed_launch_sinkhorn
    assert {op: module._fused_op_labels[op] for op in ops} == {
        "proj_rms": "glm-triton(tf32=rn)",
        "sinkhorn": TRITON_SINKHORN,
    }


@pytest.mark.parametrize(
    "ops, flags, labels",
    [
        (
            ["sinkhorn"],
            {},
            {"proj_rms": "native", "h_aggregate": "native", "h_post_bda": "native(compiled)"},
        ),
        (
            ["proj_rms", "sinkhorn"],
            GLM_FLAGS,
            {
                "proj_rms": "native(triton unavailable)",
                "h_aggregate": "native",
                "h_post_bda": "native(emulate-casts)",
            },
        ),
    ],
    ids=["plain", "glm"],
)
def test_per_op_sinkhorn_without_triton_keeps_native(
    monkeypatch, warnings_logged, ops, flags, labels
):
    """Under auto on a rank without Triton, the Triton-only operations keep their unfused objects.

    sinkhorn keeps the unfused Sinkhorn, and with the GLM flags proj_rms keeps the unfused
    projection; one warning names them, in table order.
    """
    monkeypatch.setattr(fmk, "is_triton_available", lambda: False)
    monkeypatch.setattr(fmk, "is_cutile_available", lambda: False)
    config = _per_op(ops, **flags)
    module = _module(config)
    assert module._sinkhorn_op is hc.native_sinkhorn
    _assert_same_binding(module, _module(_config(**flags)))
    assert module._fused_op_labels == dict(labels, sinkhorn="native(triton unavailable)")
    unavailable = ["proj_rms", "sinkhorn"] if "proj_rms" in ops else ["sinkhorn"]
    expected = [
        (
            logging.WARNING,
            f"[mHC] Falling back to the unfused path for mhc_fused_ops {unavailable}: "
            "Triton is unavailable.",
        )
    ]
    assert warnings_logged == expected
    # Once per process.
    _module(config)
    assert warnings_logged == expected


def test_per_op_sinkhorn_label_reads_the_num_warps_constant(monkeypatch, triton_only_fmk):
    """The label reads SINKHORN_NUM_WARPS when the module binds, as the launch does when it runs.

    test_fused_mhc_glm_projection.py checks the launch with the same patched constant (GPU).
    """
    monkeypatch.setattr(glm, "SINKHORN_NUM_WARPS", 2)
    module = _module(_per_op(["sinkhorn"]))
    assert module._sinkhorn_op is glm.fixed_launch_sinkhorn
    assert module._fused_op_labels["sinkhorn"] == "triton(num_warps=2)"


@pytest.mark.parametrize("rounding", ["rn", "rz"])
def test_per_op_glm_pair_label_reads_the_rounding(triton_only_fmk, rounding):
    """proj_rms with the GLM flags binds the op object of mhc_fused_tf32_rounding (one per
    rounding, so identity compares), and the label names the rounding."""
    module = _module(_per_op(["proj_rms"], mhc_fused_tf32_rounding=rounding, **GLM_FLAGS))
    assert module._glm_proj_rms_op is glm.glm_proj_rms_op(rounding)
    assert module._fused_op_labels["proj_rms"] == f"glm-triton(tf32={rounding})"


@pytest.mark.parametrize("policy", ["auto", "triton"])
def test_per_op_glm_pair_refusal_policy(triton_only_fmk, warnings_logged, policy):
    """The GLM projection pair refuses CPU inputs at call time, and the spy counts the refusal.

    auto warns once per reason and returns the unfused module's projection, bit for bit; an
    explicit triton policy raises instead of falling back.
    """
    module = _module(_per_op(["proj_rms", "sinkhorn"], mhc_fused_backend=policy, **GLM_FLAGS))
    reference = _module(_config(**GLM_FLAGS))
    reference.load_state_dict(module.state_dict())
    x = torch.randn(4, 2, 4 * 32, generator=torch.Generator().manual_seed(0)).bfloat16()
    message = (
        "the Triton GLM projection pair does not support this input: "
        "inputs on cpu and cpu, not one CUDA device"
    )
    with spy_fused_mhc_ops(module) as spy:
        if policy == "triton":
            with pytest.raises(RuntimeError) as error:
                module._projection_and_get_norm(x)
            assert str(error.value) == (
                f"mHC fused backend 'triton' was requested for ['proj_rms'], but {message}."
            )
            assert (spy.calls("_glm_proj_rms_op"), spy.unsupported("_glm_proj_rms_op")) == (0, 1)
            assert warnings_logged == []
            return
        actual = module._projection_and_get_norm(x)
        assert (spy.calls("_glm_proj_rms_op"), spy.unsupported("_glm_proj_rms_op")) == (0, 1)
        expected_warnings = [
            (
                logging.WARNING,
                "[mHC] Falling back to the unfused path for mhc_fused_ops ['proj_rms']: "
                f"{message}.",
            )
        ]
        assert warnings_logged == expected_warnings
        # Once per process.
        module._projection_and_get_norm(x)
        assert spy.unsupported("_glm_proj_rms_op") == 2
        assert warnings_logged == expected_warnings
    for name, got, want in zip(("proj", "norm"), actual, reference._projection_and_get_norm(x)):
        assert got.dtype == want.dtype == torch.float32, name
        assert torch.equal(got.detach().view(torch.int32), want.detach().view(torch.int32)), name


def test_per_op_does_not_log_fmk_backend_selection(monkeypatch):
    """That line describes operations a per-op module may not bind; the all-ops mode logs it."""

    def log_fused_mhc_backend_once(backend):
        raise AssertionError("per-op mode must not log the fused kernels' backend selection")

    monkeypatch.setattr(fmk, "log_fused_mhc_backend_once", log_fused_mhc_backend_once)
    _module(_per_op(["proj_rms", "h_aggregate", "h_post_bda"]))
    _module(_per_op(["proj_rms", "sinkhorn"], **GLM_FLAGS))
    with pytest.raises(AssertionError, match="must not log"):
        _module(_config(use_fused_mhc=True))


def test_fmk_op_labels_follow_fmk_resolution(monkeypatch, triton_only_fmk):
    """Labels carry the fused kernels' own per-op resolution, parsed from their status message.

    A changed message format fails here; at run time an entry the message lacks prints ``?``.
    """
    assert hc._fmk_op_labels("auto", MHC_FUSED_OPS) == {
        "proj_rms": "fmk(auto:native)",
        "sinkhorn": "fmk(auto:triton)",
        "h_aggregate": "fmk(auto:fwd:triton,bwd:triton)",
        "h_post_bda": "fmk(auto:fwd:triton,bwd:triton)",
    }
    assert hc._fmk_op_labels("native", ["h_aggregate"]) == {
        "h_aggregate": "fmk(native:fwd:native,bwd:native)"
    }
    monkeypatch.setattr(fmk, "is_triton_available", lambda: False)
    monkeypatch.setattr(fmk, "is_cutile_available", lambda: True)
    # auto never selects the cuTile h_aggregate kernels; only the explicit cutile policy does.
    assert hc._fmk_op_labels("auto", MHC_FUSED_OPS) == {
        "proj_rms": "fmk(auto:cutile)",
        "sinkhorn": "fmk(auto:cutile)",
        "h_aggregate": "fmk(auto:fwd:native,bwd:native)",
        "h_post_bda": "fmk(auto:fwd:cutile,bwd:cutile)",
    }
    assert hc._fmk_op_labels("cutile", MHC_FUSED_OPS) == {
        "proj_rms": "fmk(cutile:cutile)",
        "sinkhorn": "fmk(cutile:cutile)",
        "h_aggregate": "fmk(cutile:fwd:cutile,bwd:cutile)",
        "h_post_bda": "fmk(cutile:fwd:cutile,bwd:cutile)",
    }
    monkeypatch.setattr(
        fmk, "_mhc_backend_status", lambda backend: (f"policy={backend}; sinkhorn=triton", False)
    )
    assert hc._fmk_op_labels("auto", ["sinkhorn", "h_aggregate"]) == {
        "sinkhorn": "fmk(auto:triton)",
        "h_aggregate": "fmk(auto:?)",
    }


def test_policy_native_differs_by_mode():
    """The native policy binds different proj_rms arithmetic in the two modes, intentionally.

    Per-op mode rebinds an operation only for an accelerated implementation, so per-op native
    keeps today's unfused objects (_projection_and_get_norm, then _compute_h). The
    all-operations mode keeps binding the fused kernels' projection-first native op.
    """
    per_op = _module(_per_op(MHC_FUSED_OPS, mhc_fused_backend="native"))
    _assert_same_binding(per_op, _module(_config()))
    all_ops = _module(_config(use_fused_mhc=True, mhc_fused_backend="native"))
    _assert_partial(all_ops._proj_rms_compute_h_op, fmk.fused_proj_rms_compute_h, backend="native")


_FALLBACK_PROFILES = pytest.mark.parametrize(
    "ops, flags",
    [(list(MHC_FUSED_OPS), {}), (["proj_rms", "sinkhorn"], GLM_FLAGS)],
    ids=["all_ops_listed", "glm"],
)


@_FALLBACK_PROFILES
def test_forced_native_binds_todays_objects(monkeypatch, warnings_logged, ops, flags):
    """MHC_FORCE_BACKEND=native keeps every listed operation unfused, and nothing raises."""
    monkeypatch.setattr(fmk, "_MHC_FORCED_BACKEND", "native")
    monkeypatch.setattr(fmk, "_TRITON_AVAILABLE", False)
    monkeypatch.setattr(fmk, "_CUTILE_AVAILABLE", False)
    module = _module(_per_op(ops, **flags))
    _assert_same_binding(module, _module(_config(**flags)))
    assert {op: module._fused_op_labels[op] for op in ops} == {
        op: "native(MHC_FORCE_BACKEND=native)" for op in ops
    }
    assert warnings_logged == []
    # An explicit triton policy still raises, as it does in the all-operations mode.
    with pytest.raises(RuntimeError, match="Triton is unavailable"):
        _module(_per_op(ops, mhc_fused_backend="triton", **flags))


@_FALLBACK_PROFILES
def test_policy_native_binds_todays_objects(warnings_logged, ops, flags):
    module = _module(_per_op(ops, mhc_fused_backend="native", **flags))
    _assert_same_binding(module, _module(_config(**flags)))
    assert {op: module._fused_op_labels[op] for op in ops} == {
        op: "native(mhc_fused_backend=native)" for op in ops
    }
    assert warnings_logged == []


_FORCED_NATIVE_SCRIPT = """
import json
import sys
from functools import partial

from megatron.core.fusions import fused_mhc_kernels as fmk
from megatron.core.transformer import hyper_connection as hc
from megatron.core.transformer.mhc_fused_ops import MHC_FUSED_OPS
from megatron.core.transformer.transformer_config import TransformerConfig

GLM = dict(
    mhc_norm_eps_inside_sqrt=True,
    mhc_normalize_before_projection=True,
    mhc_keep_pre_in_fp32=True,
    mhc_preserve_activation_dtype_rounding=True,
)


def module(**overrides):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        enable_hyper_connections=True,
        **overrides,
    )
    return hc.HyperConnectionModule(config, layer_number=1)


def is_unfused(m):
    post = (
        hc._get_precision_preserving_h_post_bda()
        if m.config.mhc_preserve_activation_dtype_rounding
        else hc.native_h_post_bda
    )
    return (
        m._sinkhorn_op is hc.native_sinkhorn
        and m._h_aggregate_op is hc.native_h_aggregate
        and m._h_aggregate_into_op is hc.native_h_aggregate_into
        and m._h_post_bda_op is post
        and m._proj_rms_compute_h_op is None
        and m._glm_proj_rms_op is None
    )


per_op = module(use_fused_mhc=True, mhc_fused_ops=list(MHC_FUSED_OPS))
per_op_glm = module(use_fused_mhc=True, mhc_fused_ops=["proj_rms", "sinkhorn"], **GLM)
all_ops = module(use_fused_mhc=True)
# The modules return before binding sinkhorn, so they never import the fixed-launch file.
glm_imported_by_modules = "megatron.core.fusions.fused_mhc_glm_projection" in sys.modules
from megatron.core.fusions import fused_mhc_glm_projection as glm

try:
    glm._sinkhorn_kernels()
    glm_kernel_lookup = None
except Exception as error:
    glm_kernel_lookup = type(error).__name__
try:
    glm._glm_kernels()
    glm_pair_kernel_lookup = None
except Exception as error:
    glm_pair_kernel_lookup = type(error).__name__
all_ops_functions = {
    "_sinkhorn_op": fmk.fused_sinkhorn,
    "_h_aggregate_op": fmk.fused_h_aggregate,
    "_h_aggregate_into_op": fmk.fused_h_aggregate_into,
    "_h_post_bda_op": fmk.fused_h_post_bda,
    "_proj_rms_compute_h_op": fmk.fused_proj_rms_compute_h,
}
all_ops_bound = {}
for name, function in all_ops_functions.items():
    op = getattr(all_ops, name)
    all_ops_bound[name] = [
        isinstance(op, partial) and op.func is function,
        list(op.args),
        op.keywords,
    ]
print(
    json.dumps(
        {
            "forced_backend": fmk._MHC_FORCED_BACKEND,
            "triton_available": fmk.is_triton_available(),
            "triton_kernels_defined": hasattr(fmk, "_triton_sinkhorn_fwd_kernel"),
            "per_op_unfused": is_unfused(per_op),
            "per_op_glm_unfused": is_unfused(per_op_glm),
            "per_op_labels": per_op._fused_op_labels,
            "all_ops_bound": all_ops_bound,
            "glm_imported_by_modules": glm_imported_by_modules,
            "glm_num_warps": glm.SINKHORN_NUM_WARPS,
            "glm_kernel_lookup": glm_kernel_lookup,
            "glm_pair_kernel_lookup": glm_pair_kernel_lookup,
        }
    )
)
"""


def test_forced_native_environment_in_fresh_process():
    """The real MHC_FORCE_BACKEND=native, which the fused kernels read once, at import.

    fused_mhc_glm_projection imports there too, and its kernel lookups (the Sinkhorn's and the
    GLM projection pair's) fail closed.
    """
    result = _run_python(_FORCED_NATIVE_SCRIPT, dict(os.environ, MHC_FORCE_BACKEND="native"))
    assert result["forced_backend"] == "native"
    assert not result["triton_available"]
    assert not result["triton_kernels_defined"]
    assert result["per_op_unfused"] and result["per_op_glm_unfused"]
    assert not result["glm_imported_by_modules"]
    assert result["glm_num_warps"] == 1
    assert result["glm_kernel_lookup"] == "RuntimeError"
    assert result["glm_pair_kernel_lookup"] == "RuntimeError"
    assert result["per_op_labels"] == {
        op: "native(MHC_FORCE_BACKEND=native)" for op in MHC_FUSED_OPS
    }
    # The all-operations mode keeps its partials; the fused kernels dispatch natively at call time.
    assert result["all_ops_bound"] == {
        name: [True, [], {"backend": "auto"}]
        for name in (
            "_sinkhorn_op",
            "_h_aggregate_op",
            "_h_aggregate_into_op",
            "_h_post_bda_op",
            "_proj_rms_compute_h_op",
        )
    }


@pytest.mark.parametrize(
    "ops, flags",
    [
        (["h_aggregate"], {}),
        (["sinkhorn"], GLM_FLAGS),
        # The native policy binds nothing and looks up no label, but still validates.
        (["h_aggregate"], dict(mhc_fused_backend="native")),
    ],
    ids=["h_aggregate", "glm", "policy_native"],
)
def test_bogus_forced_backend_still_raises(monkeypatch, ops, flags):
    """A deferred MHC_FORCE_BACKEND error raises in per-op mode too, before anything binds."""
    monkeypatch.setattr(
        fmk, "_MHC_BACKEND_VALIDATION_ERROR", ValueError("Unsupported MHC_FORCE_BACKEND='bogus'")
    )
    with pytest.raises(ValueError, match="Unsupported MHC_FORCE_BACKEND='bogus'"):
        _module(_per_op(ops, **flags))


def test_explicit_triton_policy_raises_when_triton_unavailable(monkeypatch):
    monkeypatch.setattr(fmk, "is_triton_available", lambda: False)
    with pytest.raises(
        RuntimeError, match="mHC fused backend 'triton' was requested, but Triton is unavailable"
    ):
        _module(_per_op(["h_aggregate"], mhc_fused_backend="triton"))


# --- Spy ----------------------------------------------------------------------------------


class _FakeGLMOp:
    """Stands in for the GLM projection pair: refuses inputs that are not 2-D."""

    kind = "fake GLM op"

    def supported(self, x, weight):
        return x.dim() == 2

    def __call__(self, x, weight, eps):
        return x * eps, weight


def test_spy_counts_calls_and_unsupported_fallbacks():
    def sinkhorn(logits, iterations, eps):
        return logits

    def aggregate(x, h_pre, *, backend):
        return x

    glm_op = _FakeGLMOp()
    module = SimpleNamespace(
        _glm_proj_rms_op=glm_op,
        _proj_rms_compute_h_op=None,
        _sinkhorn_op=sinkhorn,
        _h_aggregate_op=partial(aggregate, backend="auto"),
        _h_aggregate_into_op=None,
        _h_post_bda_op=lambda *args: args,
    )
    x = torch.ones(2, 3)
    with spy_fused_mhc_ops([module]) as spy:
        # The caller asks the op through the attribute, then runs it or falls back.
        for inputs in (x, x, x.unsqueeze(0)):
            if module._glm_proj_rms_op.supported(inputs, None):
                module._glm_proj_rms_op(inputs, None, 0.5)
        module._sinkhorn_op(x, 20, 1e-6)
        module._h_aggregate_op(x, x)
        assert module._glm_proj_rms_op.kind == "fake GLM op"
        assert module._proj_rms_compute_h_op is None

        assert [record.module for record in spy.records] == ["#0"] * len(SPIED_ATTRIBUTES)
        assert (spy.calls("_glm_proj_rms_op"), spy.unsupported("_glm_proj_rms_op")) == (2, 1)
        assert spy.calls("_sinkhorn_op") == 1
        assert spy.calls("_h_post_bda_op") == 0
        assert spy.calls("_proj_rms_compute_h_op") == 0
        assert spy.targets("_sinkhorn_op") == {sinkhorn}
        spy.assert_ran("_glm_proj_rms_op", glm_op, min_calls=2)
        spy.assert_ran("_sinkhorn_op", sinkhorn)
        spy.assert_ran("_h_aggregate_op", aggregate)  # a partial of the target is bound to it
        with pytest.raises(AssertionError, match="refused 1 call"):
            spy.assert_no_unsupported()
        with pytest.raises(AssertionError, match="ran 0 time"):
            spy.assert_ran("_h_post_bda_op")
        with pytest.raises(AssertionError, match="ran 2 time"):
            spy.assert_ran("_glm_proj_rms_op", min_calls=3)
        with pytest.raises(AssertionError, match="is not bound to"):
            spy.assert_ran("_sinkhorn_op", aggregate)
        with pytest.raises(KeyError):
            spy.calls("_proj_rms_op")
        # A spied module can be copied, as a model can be while it is spied.
        assert copy.copy(module._glm_proj_rms_op).kind == "fake GLM op"
        assert copy.deepcopy(module)._glm_proj_rms_op.kind == "fake GLM op"
    assert module._glm_proj_rms_op is glm_op
    assert module._sinkhorn_op is sinkhorn


def test_spy_wraps_and_restores_module_attributes():
    root = torch.nn.ModuleDict(
        {
            "fused": _module(_per_op(["proj_rms", "h_aggregate", "h_post_bda"])),
            "glm": _module(_config(**GLM_FLAGS)),
        }
    )
    bound = {
        (name, attribute): getattr(module, attribute)
        for name, module in root.items()
        for attribute in SPIED_ATTRIBUTES
    }
    with spy_fused_mhc_ops(root) as spy:
        assert [(record.module, record.attribute) for record in spy.records] == list(bound)
        for (name, attribute), target in bound.items():
            spied = getattr(root[name], attribute)
            if target is None:
                assert spied is None, (name, attribute)
            else:
                assert spied is not target and spied.__wrapped__ is target, (name, attribute)
        assert spy.targets("_h_post_bda_op") == {
            bound["fused", "_h_post_bda_op"],
            bound["glm", "_h_post_bda_op"],
        }
        assert spy.calls("_sinkhorn_op") == 0
    for (name, attribute), target in bound.items():
        assert getattr(root[name], attribute) is target, (name, attribute)

    # Attributes are restored when the body raises too.
    with pytest.raises(RuntimeError, match="body failed"):
        with spy_fused_mhc_ops(root):
            raise RuntimeError("body failed")
    for (name, attribute), target in bound.items():
        assert getattr(root[name], attribute) is target, (name, attribute)
    with pytest.raises(ValueError, match="no mHC module"):
        with spy_fused_mhc_ops(torch.nn.Linear(2, 2)):
            pass


# --- Per-rank fused-ops line --------------------------------------------------------------


def _expected_precision_fields():
    """The precision state as torch reports it in this process, which never mixes TF32 APIs."""
    return [
        f"fp32_matmul_precision={torch.get_float32_matmul_precision()}",
        f"cuda.matmul.fp32_precision={torch.backends.cuda.matmul.fp32_precision}",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="
        + os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "unset"),
        f"NVIDIA_TF32_OVERRIDE={os.environ.get('NVIDIA_TF32_OVERRIDE', 'unset')}",
    ]


def test_fused_ops_line_once_per_process_with_rank_prefix(fused_ops_lines):
    config = _config(**GLM_FLAGS)
    _module(config)
    _module(config)
    assert len(fused_ops_lines) == 1
    assert fused_ops_lines[0].startswith("[rank 5] [mHC] fused ops: mode=unfused policy=auto ")
    # A block with a different config, such as an MTP block, adds its own line.
    _module(_config(**dict(GLM_FLAGS, mhc_preserve_activation_dtype_rounding=False)))
    assert len(fused_ops_lines) == 2


_PER_OP_FMK_OPS = ["proj_rms", "h_aggregate", "h_post_bda"]


@pytest.mark.parametrize(
    "overrides, forced_backend, expected",
    [
        (
            GLM_FLAGS,
            None,
            "mode=unfused policy=auto MHC_FORCE_BACKEND=unset proj_rms=native sinkhorn=native "
            "h_aggregate=native h_post_bda=native(emulate-casts)",
        ),
        (
            dict(use_fused_mhc=True, mhc_fused_ops=["proj_rms", "sinkhorn"], **GLM_FLAGS),
            None,
            "mode=per-op policy=auto MHC_FORCE_BACKEND=unset "
            "proj_rms=glm-triton(tf32=rn) sinkhorn=triton(num_warps=1) "
            "h_aggregate=native h_post_bda=native(emulate-casts)",
        ),
        (
            dict(use_fused_mhc=True, mhc_fused_ops=["sinkhorn"]),
            None,
            "mode=per-op policy=auto MHC_FORCE_BACKEND=unset proj_rms=native "
            "sinkhorn=triton(num_warps=1) h_aggregate=native h_post_bda=native(compiled)",
        ),
        (
            dict(use_fused_mhc=True, mhc_fused_ops=_PER_OP_FMK_OPS),
            None,
            "mode=per-op policy=auto MHC_FORCE_BACKEND=unset proj_rms=fmk(auto:native) "
            "sinkhorn=native h_aggregate=fmk(auto:fwd:triton,bwd:triton) "
            "h_post_bda=fmk(auto:fwd:triton,bwd:triton)",
        ),
        (
            dict(use_fused_mhc=True, mhc_fused_ops=_PER_OP_FMK_OPS, mhc_fused_backend="native"),
            None,
            "mode=per-op policy=native MHC_FORCE_BACKEND=unset "
            "proj_rms=native(mhc_fused_backend=native) sinkhorn=native "
            "h_aggregate=native(mhc_fused_backend=native) "
            "h_post_bda=native(mhc_fused_backend=native)",
        ),
        (
            dict(use_fused_mhc=True, mhc_fused_ops=_PER_OP_FMK_OPS),
            "native",
            "mode=per-op policy=auto MHC_FORCE_BACKEND=native "
            "proj_rms=native(MHC_FORCE_BACKEND=native) sinkhorn=native "
            "h_aggregate=native(MHC_FORCE_BACKEND=native) "
            "h_post_bda=native(MHC_FORCE_BACKEND=native)",
        ),
        (
            dict(use_fused_mhc=True),
            None,
            "mode=all-ops policy=auto MHC_FORCE_BACKEND=unset proj_rms=fmk(auto:native) "
            "sinkhorn=fmk(auto:triton) h_aggregate=fmk(auto:fwd:triton,bwd:triton) "
            "h_post_bda=fmk(auto:fwd:triton,bwd:triton)",
        ),
        (
            dict(use_fused_mhc=True, mhc_fused_backend="native"),
            None,
            "mode=all-ops policy=native MHC_FORCE_BACKEND=unset proj_rms=fmk(native:native) "
            "sinkhorn=fmk(native:native) h_aggregate=fmk(native:fwd:native,bwd:native) "
            "h_post_bda=fmk(native:fwd:native,bwd:native)",
        ),
    ],
    ids=[
        "unfused_glm",
        "per_op_glm",
        "per_op_sinkhorn",
        "per_op_fmk",
        "per_op_policy_native",
        "per_op_forced_native",
        "all_ops",
        "all_ops_policy_native",
    ],
)
def test_fused_ops_line_fields(
    monkeypatch, fused_ops_lines, triton_only_fmk, overrides, forced_backend, expected
):
    if forced_backend is not None:
        # The environment and the state the fused kernels derive from it at import.
        monkeypatch.setenv("MHC_FORCE_BACKEND", forced_backend)
        monkeypatch.setattr(fmk, "_MHC_FORCED_BACKEND", forced_backend)
    _module(_config(**overrides))
    expected = " ".join([expected] + _expected_precision_fields())
    assert fused_ops_lines == ["[rank 5] [mHC] fused ops: " + expected]


@pytest.mark.parametrize(
    "raising, accessor",
    [
        ("fp32_matmul_precision", (torch, "get_float32_matmul_precision")),
        ("cuda.matmul.fp32_precision", (torch._C, "_get_fp32_precision_getter")),
    ],
    ids=["fp32_matmul_precision", "cuda_matmul_fp32_precision"],
)
def test_fused_ops_line_survives_raising_accessors(monkeypatch, fused_ops_lines, raising, accessor):
    """Each accessor is read on its own: one that raises does not hide the others."""
    expected = [
        f"{raising}=<RuntimeError>" if field.startswith(f"{raising}=") else field
        for field in _expected_precision_fields()
    ]

    def mixed_state(*args, **kwargs):
        raise RuntimeError("mixed TF32 API state")

    monkeypatch.setattr(*accessor, mixed_state)
    _module(_config(mhc_norm_eps_inside_sqrt=True))
    (line,) = fused_ops_lines
    assert line.endswith(" " + " ".join(expected))


_MIXED_PRECISION_SCRIPT = """
import json
import logging

import torch

from megatron.core.transformer import hyper_connection as hc
from megatron.core.transformer.transformer_config import TransformerConfig

lines = []


class Lines(logging.Handler):
    def emit(self, record):
        lines.append(record.getMessage())


hc.logger.addHandler(Lines())
hc.logger.setLevel(logging.INFO)
# The new API alone, without the override at import: the legacy accessors now raise.
torch.backends.cuda.matmul.fp32_precision = "tf32"
try:
    torch.get_float32_matmul_precision()
    raised = None
except RuntimeError as error:
    raised = type(error).__name__
config = TransformerConfig(
    num_layers=1,
    hidden_size=32,
    num_attention_heads=4,
    enable_hyper_connections=True,
    mhc_norm_eps_inside_sqrt=True,
)
hc.HyperConnectionModule(config, layer_number=1)
print(json.dumps({"raised": raised, "lines": lines}))
"""


def test_fused_ops_line_in_mixed_precision_state():
    """The line survives the real mixed TF32 API state, in which torch's accessors raise.

    torch reads TORCH_ALLOW_TF32_CUBLAS_OVERRIDE once, at import. With it set, as run images
    set it, setting cuda.matmul.fp32_precision gives a consistent state, so the subprocess
    starts without it. This process never sets a precision API.
    """
    env = {k: v for k, v in os.environ.items() if k != "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"}
    # A non-zero rank, so that a line logged on rank 0 only is missing on every rank.
    result = _run_python(_MIXED_PRECISION_SCRIPT, dict(env, RANK="5"))
    assert result["raised"] == "RuntimeError"
    (line,) = result["lines"]
    assert line.startswith("[rank 5] [mHC] fused ops: mode=unfused ")
    assert " fp32_matmul_precision=<RuntimeError> " in line
    assert " cuda.matmul.fp32_precision=tf32 " in line
    assert " TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=unset " in line


def test_fused_ops_line_reads_tf32_environment(monkeypatch, fused_ops_lines):
    config = _config(**GLM_FLAGS)
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    _module(config)
    monkeypatch.delenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE")
    _module(config)
    assert len(fused_ops_lines) == 2
    assert fused_ops_lines[0].endswith(" TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 NVIDIA_TF32_OVERRIDE=0")
    assert fused_ops_lines[1].endswith(
        " TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=unset NVIDIA_TF32_OVERRIDE=unset"
    )


def test_fused_ops_line_skips_plain_unfused_mhc(fused_ops_lines):
    _module(_config())
    _module(_config(mhc_learned_output_contract=False))
    assert fused_ops_lines == []


def test_fused_ops_line_is_not_built_without_info(monkeypatch, fused_ops_lines):
    """Without INFO the rank is not looked up: that warns in processes without a known rank."""

    def safe_get_rank():
        raise AssertionError("the rank was looked up for a line that is not logged")

    config = _config(**GLM_FLAGS)
    monkeypatch.setattr(hc, "safe_get_rank", safe_get_rank)
    hc.logger.setLevel(logging.WARNING)
    _module(config)
    _module(_config(use_fused_mhc=True))
    assert fused_ops_lines == []
    # Nothing was latched, so the line is logged once INFO is enabled.
    monkeypatch.setattr(hc, "safe_get_rank", lambda: 5)
    hc.logger.setLevel(logging.INFO)
    _module(config)
    assert len(fused_ops_lines) == 1


def test_fused_ops_line_prints_in_all_ops_mode(fused_ops_lines):
    _module(_config(use_fused_mhc=True))
    (line,) = fused_ops_lines
    assert line.startswith("[rank 5] [mHC] fused ops: mode=all-ops policy=auto ")


def test_fused_ops_line_never_reads_allow_tf32(monkeypatch, fused_ops_lines):
    reads = []

    def get_cublas_allow_tf32():
        reads.append(True)
        raise RuntimeError("allow_tf32 was read")

    monkeypatch.setattr(torch._C, "_get_cublas_allow_tf32", get_cublas_allow_tf32)
    _module(_config(**GLM_FLAGS))
    _module(_per_op(["proj_rms", "sinkhorn"], **GLM_FLAGS))
    assert len(fused_ops_lines) == 2
    assert reads == []
