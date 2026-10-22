# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The deterministic-mode arg guard in ``megatron/training/determinism.py``.

``TransformerConfig.__post_init__`` resolves ``moe_router_aux_loss_fusion`` when it is
unset, but ``apply_determinism_to_args`` runs on the argparse Namespace before any config
exists, so it re-derives the same fallback. These tests pin the two derivations together:
the guard must reject exactly the arg combinations whose config ends up with the fusion on.
"""

import argparse
import ctypes
import os

import pytest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
from megatron.training.determinism import (
    ARG_VALUES_REQUIRED_FOR_DETERMINISM,
    DETERMINISM_ENV_VAR_DEFAULTS,
    apply_determinism_env,
    apply_determinism_to_args,
    assert_nccl_algo_still_empty,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def restore_torch_determinism():
    """apply_determinism_to_args flips a torch global on the paths that don't raise."""
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


def make_args(**kwargs):
    """A Namespace holding only what apply_determinism_to_args reads."""
    defaults = {
        **ARG_VALUES_REQUIRED_FOR_DETERMINISM,
        "moe_router_fusion": False,
        "moe_router_aux_loss_fusion": None,
    }
    return argparse.Namespace(**{**defaults, **kwargs})


@pytest.mark.internal
@pytest.mark.parametrize("router_fusion", [False, True])
@pytest.mark.parametrize("aux_loss_fusion", [None, False, True])
def test_guard_agrees_with_config_resolution(router_fusion, aux_loss_fusion):
    """The guard rejects exactly the args whose config resolves the fusion on.

    Unset is the case that matters: it inherits ``moe_router_fusion``, so dropping the
    fallback here would let ``--moe-router-fusion`` through in deterministic mode.
    """
    fusion_flags = dict(moe_router_fusion=router_fusion, moe_router_aux_loss_fusion=aux_loss_fusion)
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=1, num_moe_experts=4, **fusion_flags
    )

    if config.moe_router_aux_loss_fusion:
        with pytest.raises(AssertionError, match="moe_router_aux_loss_fusion"):
            apply_determinism_to_args(make_args(**fusion_flags))
    else:
        # Opting out explicitly keeps fused TopK routing available under determinism.
        apply_determinism_to_args(make_args(**fusion_flags))


@pytest.mark.internal
def test_other_required_args_still_checked():
    """The aux-loss branch is additive -- it must not shadow the dict-driven checks."""
    with pytest.raises(AssertionError, match="cross_entropy_loss_fusion"):
        apply_determinism_to_args(make_args(cross_entropy_loss_fusion=True))


@pytest.mark.parametrize("scan_mode", [None, "1", "0"])
def test_nccl_ep_scan_mode_is_required_before_group_creation(scan_mode):
    """Deterministic setup must reject atomic expert-slot assignment before changing env."""
    env = {} if scan_mode is None else {"NCCL_EP_HT_EM_AG_SCAN_MODE": scan_mode}
    if scan_mode == "0":
        with pytest.raises(AssertionError, match="NCCL_EP_HT_EM_AG_SCAN_MODE"):
            apply_determinism_env(env)
        assert env == {"NCCL_EP_HT_EM_AG_SCAN_MODE": "0"}
    else:
        apply_determinism_env(env)
        assert env["NCCL_EP_HT_EM_AG_SCAN_MODE"] == "1"


@pytest.mark.parametrize(
    "nccl_algo, error",
    [
        (None, None),
        ("", None),
        ("^NVLS", None),
        ("Ring,NVLS", None),
        ("NVLS,Ring", None),
        (" ", "only whitespace"),
        ("Tree", "each token"),
        ("NVLS", "needs Ring"),
    ],
)
def test_nccl_algo_is_kept_defaulted_or_rejected(nccl_algo, error):
    """Unset becomes Ring; "", ^NVLS and NVLS listed with Ring are kept; whitespace, Tree
    and a bare NVLS are rejected.

    The accepted cases still get the other defaults; a rejected value leaves env unchanged.
    """
    env = {} if nccl_algo is None else {"NCCL_ALGO": nccl_algo}
    if error:
        with pytest.raises(AssertionError, match=error):
            apply_determinism_env(env)
        assert env == {"NCCL_ALGO": nccl_algo}
    else:
        apply_determinism_env(env)
        assert env == {
            **DETERMINISM_ENV_VAR_DEFAULTS,
            "NCCL_ALGO": "Ring" if nccl_algo is None else nccl_algo,
        }


@pytest.mark.parametrize("rank", [0, 1])
def test_empty_nccl_algo_warning_is_issued_on_rank_0(rank, monkeypatch, recwarn):
    """Only rank 0 warns, and only for an empty NCCL_ALGO."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setattr("megatron.training.utils.get_device_arch_version", lambda: 10)
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    apply_determinism_to_args(make_args())
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    apply_determinism_to_args(make_args())
    monkeypatch.setenv("NCCL_ALGO", "Ring,NVLS")
    apply_determinism_to_args(make_args())
    assert not [w for w in recwarn if "NCCL_ALGO is empty" in str(w.message)]
    monkeypatch.setenv("NCCL_ALGO", "")
    apply_determinism_to_args(make_args())
    warned = any("NCCL_ALGO is empty" in str(w.message) for w in recwarn)
    assert warned == (rank == 0)


@pytest.mark.parametrize(
    "nccl_algo, arch, error",
    [
        ("", 9, "reproducible on Hopper"),
        ("Ring,NVLS", 9, "reproducible on Hopper"),
        ("", 10, None),
        ("Ring,NVLS", 10, None),
        ("^NVLS", 9, None),
        (None, 9, None),
    ],
)
def test_nvls_capable_nccl_algo_is_rejected_on_hopper(nccl_algo, arch, error, monkeypatch, recwarn):
    """A value that can let NCCL use NVLS ("" or a list with NVLS) stops a Hopper run before
    the warning; ^NVLS and unset (the Ring default) still pass there.
    """
    monkeypatch.setattr("megatron.training.utils.get_device_arch_version", lambda: arch)
    if nccl_algo is None:
        monkeypatch.delenv("NCCL_ALGO", raising=False)
    else:
        monkeypatch.setenv("NCCL_ALGO", nccl_algo)
    if error:
        with pytest.raises(AssertionError, match=error):
            apply_determinism_to_args(make_args())
        assert not [w for w in recwarn if "NCCL_ALGO is empty" in str(w.message)]
    else:
        apply_determinism_to_args(make_args())
        assert os.environ["NCCL_ALGO"] == ("Ring" if nccl_algo is None else nccl_algo)


def test_assert_nccl_algo_still_empty(monkeypatch):
    """Passes while libc still sees "", fails once a C++ write (PyTorch's) changes it."""
    monkeypatch.setenv("NCCL_ALGO", "")  # putenv, so libc agrees with os.environ
    assert_nccl_algo_still_empty()
    ctypes.CDLL(None).setenv(b"NCCL_ALGO", b"^NVLS", 1)  # what PyTorch does in C++
    with pytest.raises(AssertionError, match="NVLS exclusion applied anyway"):
        assert_nccl_algo_still_empty()


def test_assert_nccl_algo_still_empty_ignores_unset(monkeypatch):
    """Unset NCCL_ALGO is not checked, although PyTorch then writes ^NVLS."""
    monkeypatch.delenv("NCCL_ALGO", raising=False)
    libc = ctypes.CDLL(None)
    libc.setenv(b"NCCL_ALGO", b"^NVLS", 1)
    try:
        assert_nccl_algo_still_empty()
    finally:
        libc.unsetenv(b"NCCL_ALGO")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize(
    "nccl_algo, expected",
    [
        pytest.param(
            None,
            b"^NVLS",
            marks=pytest.mark.skipif(
                not is_torch_min_version("2.8.0"), reason="PyTorch writes ^NVLS from 2.8"
            ),
        ),
        ("", b""),
    ],
)
def test_pytorch_leaves_empty_nccl_algo_alone(nccl_algo, expected, monkeypatch):
    """Unset gets ^NVLS from PyTorch (the negative control); an empty NCCL_ALGO is left alone.

    Without the control, the "" case could pass because PyTorch stopped banning NVLS.
    """
    Utils.initialize_distributed()
    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_char_p
    libc.getenv.argtypes = [ctypes.c_char_p]
    if nccl_algo is None:
        monkeypatch.delenv("NCCL_ALGO", raising=False)
    else:
        monkeypatch.setenv("NCCL_ALGO", nccl_algo)
    try:
        torch.use_deterministic_algorithms(True)  # the autouse fixture restores it
        group = torch.distributed.new_group(list(range(torch.distributed.get_world_size())))
        assert libc.getenv(b"NCCL_ALGO") == expected
        assert_nccl_algo_still_empty()  # unset: a no-op even though libc holds ^NVLS
        torch.distributed.destroy_process_group(group)
    finally:
        libc.unsetenv(b"NCCL_ALGO")  # PyTorch's C++ write bypasses monkeypatch


@pytest.mark.internal
@pytest.mark.parametrize("name", sorted(ARG_VALUES_REQUIRED_FOR_DETERMINISM))
def test_each_required_arg_is_rejected_when_enabled(name):
    """Every option in ARG_VALUES_REQUIRED_FOR_DETERMINISM (cross entropy loss fusion,
    TP communication overlap, ...) is rejected on its own, and the error names it.
    """
    required = ARG_VALUES_REQUIRED_FOR_DETERMINISM[name]
    with pytest.raises(AssertionError, match=name) as exc_info:
        apply_determinism_to_args(make_args(**{name: not required}))
    assert "--disable-deterministic-mode" in str(exc_info.value)


@pytest.mark.internal
def test_all_incompatible_args_are_reported_together():
    """One failure lists every incompatible option, so users fix them in a single pass."""
    incompatible = {
        name: not required for name, required in ARG_VALUES_REQUIRED_FOR_DETERMINISM.items()
    }
    with pytest.raises(AssertionError) as exc_info:
        apply_determinism_to_args(
            make_args(**incompatible, moe_router_fusion=True, moe_router_aux_loss_fusion=True)
        )
    message = str(exc_info.value)
    for name in [*ARG_VALUES_REQUIRED_FOR_DETERMINISM, "moe_router_aux_loss_fusion"]:
        assert name in message


@pytest.mark.internal
def test_rejected_args_leave_global_state_untouched(monkeypatch):
    """An incompatible arg raises before env defaults are written or torch is switched."""
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", lambda *a, **k: pytest.fail("flipped")
    )
    for name in DETERMINISM_ENV_VAR_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(AssertionError):
        apply_determinism_to_args(make_args(cross_entropy_loss_fusion=True))
    assert not any(name in os.environ for name in DETERMINISM_ENV_VAR_DEFAULTS)


@pytest.mark.internal
@pytest.mark.parametrize(
    "name, value, error",
    [
        ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1", "NVTE_ALLOW_NONDETERMINISTIC_ALGO"),
        ("CUBLAS_WORKSPACE_CONFIG", ":8:8", "CUBLAS_WORKSPACE_CONFIG"),
        ("TRITON_CACHE_AUTOTUNING", "true", "TRITON_CACHE_AUTOTUNING"),
        ("MAMBA_DETERMINISTIC", "0", "MAMBA_DETERMINISTIC"),
        ("CAUSAL_CONV1D_DETERMINISTIC", "0", "CAUSAL_CONV1D_DETERMINISTIC"),
    ],
)
def test_env_var_with_nondeterministic_value_is_rejected(name, value, error):
    """A caller-set env var that breaks determinism fails hard and is left as the caller set it."""
    env = {name: value}
    with pytest.raises(AssertionError, match=error):
        apply_determinism_env(env)
    assert env == {name: value}


@pytest.mark.internal
@pytest.mark.parametrize(
    "name, value",
    [
        ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0"),
        ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        ("CUBLAS_WORKSPACE_CONFIG", ":16:8"),
        ("MAMBA_DETERMINISTIC", "1"),
        ("CAUSAL_CONV1D_DETERMINISTIC", "1"),
    ],
)
def test_env_var_with_deterministic_value_is_kept(name, value):
    """The negative control: an accepted caller value is kept rather than overwritten."""
    env = {name: value}
    apply_determinism_env(env)
    assert env[name] == value


@pytest.mark.internal
@pytest.mark.parametrize("cache_dir, error", [(None, "TRITON_CACHE_DIR"), ("/shared/cache", None)])
def test_triton_cache_autotuning_requires_shared_cache_dir(cache_dir, error):
    """Cached autotuning is only deterministic when every rank shares one cache."""
    env = {"TRITON_CACHE_AUTOTUNING": "1"}
    if cache_dir:
        env["TRITON_CACHE_DIR"] = cache_dir
    if error:
        with pytest.raises(AssertionError, match=error):
            apply_determinism_env(env)
    else:
        apply_determinism_env(env)


@pytest.mark.internal
@pytest.mark.parametrize(
    "env",
    [
        {"NCCL_ALGO": " "},
        {"NCCL_ALGO": "NVLS"},
        {"NCCL_ALGO": "Tree"},
        {"NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1"},
        {"CUBLAS_WORKSPACE_CONFIG": ":8:8"},
        {"NCCL_EP_HT_EM_AG_SCAN_MODE": "0"},
        {"TRITON_CACHE_AUTOTUNING": "true"},
        {"TRITON_CACHE_AUTOTUNING": "1"},
        {"MAMBA_DETERMINISTIC": "0"},
        {"CAUSAL_CONV1D_DETERMINISTIC": "0"},
    ],
    ids=lambda env: "-".join(f"{k}={v!r}" for k, v in env.items()),
)
def test_env_var_errors_point_to_disable_flag(env):
    """Every rejected env var tells the user how to opt out of deterministic mode."""
    with pytest.raises(AssertionError) as exc_info:
        apply_determinism_env(dict(env))
    message = str(exc_info.value)
    assert "--disable-deterministic-mode" in message
    assert "--deterministic-mode " not in message


@pytest.mark.internal
def test_hopper_nvls_error_points_to_disable_flag(monkeypatch):
    monkeypatch.setattr("megatron.training.utils.get_device_arch_version", lambda: 9)
    monkeypatch.setenv("NCCL_ALGO", "")
    with pytest.raises(AssertionError) as exc_info:
        apply_determinism_to_args(make_args())
    assert "--disable-deterministic-mode" in str(exc_info.value)


@pytest.mark.internal
def test_nccl_algo_still_empty_error_points_to_disable_flag(monkeypatch):
    monkeypatch.setenv("NCCL_ALGO", "")
    ctypes.CDLL(None).setenv(b"NCCL_ALGO", b"^NVLS", 1)
    with pytest.raises(AssertionError) as exc_info:
        assert_nccl_algo_still_empty()
    assert "--disable-deterministic-mode" in str(exc_info.value)


def _validated_training_args(monkeypatch, *extra_argv):
    """parse_args -> validate_args with the fused cross entropy loss that determinism rejects."""
    import sys

    from megatron.training.arguments import parse_args, validate_args

    monkeypatch.setattr(sys, "argv", ["test_determinism_args.py", *extra_argv])
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 4
    args.max_position_embeddings = 1024
    args.seq_length = 1024
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = "NullTokenizer"
    args.vocab_size = 1024
    args.cross_entropy_loss_fusion = True
    args.cross_entropy_fusion_impl = "native"
    return args, validate_args


@pytest.mark.internal
def test_validate_args_rejects_incompatible_args_by_default(monkeypatch):
    """Deterministic mode is on by default, so validate_args rejects fused cross entropy."""
    args, validate_args = _validated_training_args(monkeypatch)
    with pytest.raises(AssertionError, match="cross_entropy_loss_fusion"):
        validate_args(args)


@pytest.mark.internal
def test_validate_args_accepts_incompatible_args_when_mode_disabled(monkeypatch):
    """--disable-deterministic-mode skips the guard and leaves torch's global flag alone."""
    args, validate_args = _validated_training_args(monkeypatch, "--disable-deterministic-mode")
    torch.use_deterministic_algorithms(False)
    validated = validate_args(args)
    assert validated.cross_entropy_loss_fusion
    assert not torch.are_deterministic_algorithms_enabled()
