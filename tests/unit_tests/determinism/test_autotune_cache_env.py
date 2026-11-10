# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cached Triton autotuning is opt-in, and the opt-in has to carry a shared cache path."""

import pytest

from megatron.training.determinism import apply_determinism_env


def test_cached_autotuning_is_not_turned_on_by_default():
    """Left alone, deterministic mode pins the cheapest config -- no cache to share.

    Defaulting it on would move every deterministic run onto the cached path, where
    determinism holds only while every rank reads one warm shared cache.
    """
    env = {}
    apply_determinism_env(env)
    assert "TRITON_CACHE_AUTOTUNING" not in env


def test_opting_in_without_a_cache_dir_is_rejected():
    with pytest.raises(AssertionError, match="TRITON_CACHE_DIR"):
        apply_determinism_env({"TRITON_CACHE_AUTOTUNING": "1"})


def test_opting_in_with_a_cache_dir_is_accepted():
    apply_determinism_env({"TRITON_CACHE_AUTOTUNING": "1", "TRITON_CACHE_DIR": "/shared/tc"})


def test_a_misspelled_opt_in_fails_instead_of_reading_as_opted_out():
    """The consumer tests ``== "1"``, so "true" would silently mean "off"."""
    with pytest.raises(AssertionError, match="TRITON_CACHE_AUTOTUNING"):
        apply_determinism_env({"TRITON_CACHE_AUTOTUNING": "true"})
