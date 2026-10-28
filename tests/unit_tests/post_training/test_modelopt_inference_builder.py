# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The inference entry points build ModelOpt models with the arguments bound to the builder.

``modelopt_gpt_hybrid_builder`` takes ``args`` first, while ``megatron.training.get_model``
calls its provider with keyword arguments only. The tests run without torch.distributed, with
stand-ins for ``get_model`` and the builder.
"""

from argparse import Namespace

import pytest


class _Built(Exception):
    """Stops the entry point once the model provider has run."""


def _modelopt_builder(calls):
    """Stand-in with the signature of ``modelopt_gpt_hybrid_builder`` that records its call."""

    def modelopt_gpt_hybrid_builder(
        args,
        pre_process,
        post_process,
        vp_stage=None,
        config=None,
        pg_collection=None,
        *,
        disable_moe_grouped_gemm=False,
    ):
        calls.append((args, pre_process, post_process, vp_stage, config, pg_collection))
        raise _Built

    return modelopt_gpt_hybrid_builder


def _get_model(model_provider_func, wrap_with_ddp=True, config=None, pg_collection=None):
    """Call the provider the way ``get_model`` does without virtual pipeline stages."""
    return [
        model_provider_func(
            pre_process=True, post_process=True, config=config, pg_collection=pg_collection
        )
    ]


def test_get_model_for_inference_binds_args_to_the_modelopt_builder(monkeypatch):
    from megatron.inference import utils

    args = Namespace(modelopt_enabled=True)
    calls = []
    monkeypatch.setattr(utils, "HAS_NVIDIA_MODELOPT", True)
    monkeypatch.setattr(
        utils, "modelopt_gpt_hybrid_builder", _modelopt_builder(calls), raising=False
    )
    monkeypatch.setattr(utils, "get_args", lambda: args)
    monkeypatch.setattr(utils, "_get_model", _get_model)

    with pytest.raises(_Built):
        utils.get_model_for_inference()

    assert calls == [(args, True, True, None, None, None)]


def test_text_generation_server_binds_args_to_the_modelopt_builder(monkeypatch):
    import tools.run_text_generation_server as server

    args = Namespace(modelopt_enabled=True, num_layers_per_virtual_pipeline_stage=None, fp8=False)
    calls = []
    monkeypatch.setattr(server, "HAS_NVIDIA_MODELOPT", True)
    monkeypatch.setattr(
        server, "modelopt_gpt_hybrid_builder", _modelopt_builder(calls), raising=False
    )
    monkeypatch.setattr(server, "parse_and_validate_args", lambda **kwargs: args)
    monkeypatch.setattr(server, "inference_cfg_container_from_args", lambda *a, **kw: None)
    monkeypatch.setattr(server, "set_run_config", lambda run_config: None)
    monkeypatch.setattr(server, "initialize_megatron", lambda: None)
    monkeypatch.setattr(server, "get_args", lambda: args)
    monkeypatch.setattr(server, "get_model", _get_model)

    with pytest.raises(_Built):
        server.main()

    assert calls == [(args, True, True, None, None, None)]
