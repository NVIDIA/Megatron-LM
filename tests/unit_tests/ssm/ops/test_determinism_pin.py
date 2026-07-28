# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests for deterministic-mode Triton autotune pinning.

These exercise the selection logic directly, so they need neither a GPU nor a
working triton install.
"""

import pytest

from megatron.core.ssm.ops import determinism


class _Config:
    """Stand-in for ``triton.Config``."""

    def __init__(self, kwargs, num_warps=4, num_stages=1, pre_hook=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.pre_hook = pre_hook


class _Autotuner:
    """Stand-in for ``triton.runtime.autotuner.Autotuner``."""

    def __init__(self, configs, name="_fake_kernel"):
        self.configs = configs
        self.arg_names = ("x", "seqlen")
        self.keys = ("seqlen",)
        self.base_fn = type(
            "_Fn", (), {"__name__": name, "__module__": "mamba_ssm.ops.triton.fake"}
        )()


CHEAP = _Config({"BLOCK_SIZE_M": 32}, num_warps=4, num_stages=1)
FAST = _Config({"BLOCK_SIZE_M": 128}, num_warps=8, num_stages=2, pre_hook=lambda *_: None)


@pytest.fixture
def restore_state():
    saved_table = dict(getattr(determinism, "_untuned_kernels_warned", set()))
    determinism._untuned_kernels_warned.clear()
    yield
    determinism._untuned_kernels_warned.clear()
    determinism._untuned_kernels_warned.update(saved_table)


def test_falls_back_to_cheapest_when_untuned(restore_state, monkeypatch):
    """With no table entry the cheapest config wins - deterministic, never timed."""
    monkeypatch.setattr(determinism, "_lookup_tuned_config", lambda *a, **k: None)
    tuner = _Autotuner([FAST, CHEAP])
    with pytest.warns(UserWarning, match="no pre-tuned config"):
        chosen = determinism._deterministic_choice(tuner, tuner.configs, (None, 8192), {})
    assert chosen is CHEAP


def test_prefers_tuned_config_and_preserves_identity(restore_state, monkeypatch):
    """A table hit returns the ORIGINAL Config object, keeping pre_hook intact."""
    monkeypatch.setattr(determinism, "_lookup_tuned_config", lambda *a, **k: FAST)
    tuner = _Autotuner([FAST, CHEAP])
    chosen = determinism._deterministic_choice(tuner, tuner.configs, (None, 8192), {})
    assert chosen is FAST
    assert chosen.pre_hook is not None


def test_lookup_matches_against_candidates(monkeypatch):
    """Lookup returns the live candidate whose fields match the recorded entry."""
    table = {
        "sm100": {
            "_fake_kernel": {
                "*": {"kwargs": {"BLOCK_SIZE_M": 128}, "num_warps": 8, "num_stages": 2}
            }
        }
    }
    import megatron.core.ssm.ops.tuned_autotune_configs as tuned

    monkeypatch.setattr(tuned, "TUNED_AUTOTUNE_CONFIGS", table)
    monkeypatch.setattr(determinism, "_arch_tag", lambda: "sm100")
    assert determinism._lookup_tuned_config("_fake_kernel", "seqlen=8192", [CHEAP, FAST]) is FAST
    # An entry that matches no live candidate must miss rather than fabricate one.
    assert determinism._lookup_tuned_config("_other_kernel", "seqlen=8192", [CHEAP, FAST]) is None


def test_tuning_key_is_stable_and_shape_sensitive():
    tuner = _Autotuner([CHEAP])
    assert determinism._tuning_key(tuner, (None, 8192), {}) == determinism._tuning_key(
        tuner, (None, 8192), {}
    )
    assert determinism._tuning_key(tuner, (None, 8192), {}) != determinism._tuning_key(
        tuner, (None, 4096), {}
    )


def test_autotune_configs_pins_under_deterministic_mode():
    """The internal-kernel path pins regardless of TRITON_CACHE_AUTOTUNING."""
    determinism.set_deterministic_mode(True)
    try:
        assert determinism.autotune_configs([FAST, CHEAP]) == [CHEAP]
    finally:
        determinism.set_deterministic_mode(None)


def test_inert_outside_deterministic_mode():
    determinism.set_deterministic_mode(False)
    try:
        assert determinism.autotune_configs([FAST, CHEAP]) == [FAST, CHEAP]
    finally:
        determinism.set_deterministic_mode(None)
