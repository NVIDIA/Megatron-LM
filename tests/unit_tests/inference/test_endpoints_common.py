# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.inference.text_generation_server.dynamic_text_gen_server.endpoints.common import (
    apply_optional_sampling_default,
    generation_config_sampling_defaults,
    resolve_sampling_default,
)


def test_apply_optional_sampling_default_skips_unset_value():
    # This is the exact bug wdykas flagged on PR #7191: if startup ever goes back
    # to setting the key unconditionally (even to a hardcoded default), tier 1 of
    # resolve_sampling_default always wins and generation_config.json is dead code.
    app_config = {}
    apply_optional_sampling_default(app_config, 'default_temperature', None)
    assert 'default_temperature' not in app_config


def test_apply_optional_sampling_default_sets_configured_value():
    app_config = {}
    apply_optional_sampling_default(app_config, 'default_temperature', 0.6)
    assert app_config['default_temperature'] == 0.6


def test_apply_optional_sampling_default_sets_falsy_configured_value():
    # 0 / 0.0 is a real, explicit configuration, not "unset". Only None means unset.
    app_config = {}
    apply_optional_sampling_default(app_config, 'default_top_k', 0)
    assert app_config['default_top_k'] == 0


def test_resolve_sampling_default_falls_through_to_generation_config_when_unset():
    # End-to-end regression for the reported bug: a server started with no
    # --default-temperature override (app_config never gets the key, per
    # apply_optional_sampling_default above) must let the model's own
    # generation_config.json value win, not silently fall to the hardcoded 1.0.
    app_config = {}
    gen_defaults = {"temperature": 0.6}
    assert (
        resolve_sampling_default(app_config, gen_defaults, "temperature", "default_temperature", 1.0)
        == 0.6
    )


def test_resolve_sampling_default_prefers_explicit_server_config():
    app_config = {"default_temperature": 0.2}
    gen_defaults = {"temperature": 0.6}
    assert (
        resolve_sampling_default(app_config, gen_defaults, "temperature", "default_temperature", 1.0)
        == 0.2
    )


def test_resolve_sampling_default_treats_configured_zero_as_explicit():
    # `config_key in app_config` is used instead of `.get(config_key, default)`
    # specifically so an operator-configured 0 (a real, valid top_k) is not
    # mistaken for "unset". This guards that distinction directly.
    app_config = {"default_top_k": 0}
    gen_defaults = {"top_k": 5}
    assert resolve_sampling_default(app_config, gen_defaults, "top_k", "default_top_k", 1) == 0


def test_resolve_sampling_default_falls_back_to_hardcoded_when_neither_set():
    assert resolve_sampling_default({}, {}, "temperature", "default_temperature", 1.0) == 1.0


def test_generation_config_sampling_defaults_missing_attr_returns_empty():
    class _Tokenizer:
        pass

    assert generation_config_sampling_defaults(_Tokenizer()) == {}


def test_generation_config_sampling_defaults_non_dict_returns_empty():
    class _Tokenizer:
        generation_config = "not-a-dict"

    assert generation_config_sampling_defaults(_Tokenizer()) == {}


def test_generation_config_sampling_defaults_extracts_numeric_fields():
    class _Tokenizer:
        generation_config = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "do_sample": True}

    assert generation_config_sampling_defaults(_Tokenizer()) == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }


def test_generation_config_sampling_defaults_rejects_bool_values():
    # bool is an int subclass in Python; a stray `"top_k": true` must not be
    # treated as a numeric sampling value.
    class _Tokenizer:
        generation_config = {"top_k": True}

    assert generation_config_sampling_defaults(_Tokenizer()) == {}


def test_generation_config_sampling_defaults_omits_non_numeric_fields():
    class _Tokenizer:
        generation_config = {"temperature": "warm", "top_p": 0.9}

    assert generation_config_sampling_defaults(_Tokenizer()) == {"top_p": 0.9}
