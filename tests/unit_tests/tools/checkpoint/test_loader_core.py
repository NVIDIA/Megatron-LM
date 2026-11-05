# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The core checkpoint loader builds GPT models through the GPT model builder."""

import os
import sys
from types import SimpleNamespace
from unittest import mock

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools', 'checkpoint'))

from loader_core import MegatronCheckpointLoaderLLM

import gpt_builders
import model_provider


def test_gpt_model_provider_uses_the_gpt_builder():
    """The provider the loader builds its models with is bound to the GPT model builder."""
    loader = MegatronCheckpointLoaderLLM(args=SimpleNamespace(model_type='GPT'), queue=None)
    margs = SimpleNamespace()
    pg_collection = object()
    gpt_model = object()
    gpt_builder = mock.Mock(return_value=gpt_model)
    run_config = SimpleNamespace(profiling=SimpleNamespace(record_memory_history=False))

    with (
        mock.patch.object(gpt_builders, 'gpt_builder', gpt_builder),
        mock.patch.object(model_provider, 'get_args', return_value=margs),
        mock.patch.object(model_provider, 'get_run_config', return_value=run_config),
    ):
        provider = loader.import_model_provider()
        # The loader passes every argument of the provider by keyword.
        model = provider(pre_process=True, post_process=False, pg_collection=pg_collection)

    assert model is gpt_model
    gpt_builder.assert_called_once_with(
        margs, True, False, None, config=None, pg_collection=pg_collection
    )
