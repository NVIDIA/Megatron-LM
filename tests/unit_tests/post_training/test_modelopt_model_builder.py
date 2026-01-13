# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for model_provider integration with ModelOpt model_builder."""

from argparse import Namespace

import model_provider as mp


def _sentinel_builder(return_value, calls):
    """Create a builder stub that records invocation."""

    def _builder(args, pre_process, post_process, vp_stage, config=None, pg_collection=None):
        calls.append(
            {
                "args": args,
                "pre_process": pre_process,
                "post_process": post_process,
                "vp_stage": vp_stage,
                "config": config,
                "pg_collection": pg_collection,
            }
        )
        return return_value

    return _builder


def test_model_provider_switches_to_modelopt_builder(monkeypatch):
    """Ensure model_provider delegates to ModelOpt builder when enabled."""
    args = Namespace(record_memory_history=False, modelopt_enabled=True)
    modelopt_calls = []
    original_calls = []

    modelopt_result = object()
    original_result = object()

    # Force ModelOpt availability and stub builders.
    monkeypatch.setattr(mp, "has_nvidia_modelopt", True)
    monkeypatch.setattr(mp, "get_args", lambda: args)
    monkeypatch.setattr(
        mp, "modelopt_gpt_mamba_builder", _sentinel_builder(modelopt_result, modelopt_calls)
    )

    # original_builder should be ignored when ModelOpt is enabled.
    original_builder = _sentinel_builder(original_result, original_calls)

    returned = mp.model_provider(
        original_builder,
        pre_process=False,
        post_process=False,
        vp_stage=1,
        config="cfg",
        pg_collection="pg",
    )

    assert returned is modelopt_result
    assert modelopt_calls == [
        {
            "args": args,
            "pre_process": False,
            "post_process": False,
            "vp_stage": 1,
            "config": "cfg",
            "pg_collection": "pg",
        }
    ]
    assert len(original_calls) == 0
