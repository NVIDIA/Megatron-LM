# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for MegatronCheckpointLoaderLLM.import_model_provider in
tools/checkpoint/loader_core.py.
"""

import os
import sys
import types
from unittest import mock

# Add the tools/checkpoint directory to the path so we can import the module.
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'tools', 'checkpoint')
)

from loader_core import MegatronCheckpointLoaderLLM


def _install_fake_gpt_provider_modules():
    """Stub the 'model_provider' and 'gpt_builders' modules that
    import_model_provider() lazily imports for GPT checkpoints.

    The real model_provider() requires model_builder as its first positional
    argument (see model_provider.py); this fake mirrors that signature so the
    test exercises the same calling convention without pulling in the full
    megatron.core dependency chain.
    """
    fake_model_provider_mod = types.ModuleType('model_provider')

    def model_provider(
        model_builder,
        pre_process=True,
        post_process=True,
        vp_stage=None,
        config=None,
        pg_collection=None,
    ):
        return model_builder(pre_process, post_process)

    fake_model_provider_mod.model_provider = model_provider

    fake_gpt_builders_mod = types.ModuleType('gpt_builders')

    def gpt_builder(pre_process, post_process):
        return ('built', pre_process, post_process)

    fake_gpt_builders_mod.gpt_builder = gpt_builder

    return fake_model_provider_mod, fake_gpt_builders_mod


class TestImportModelProviderGPT:
    def test_returned_provider_is_callable_with_only_pre_post_process(self):
        """import_model_provider() must return a callable that already has
        gpt_builder bound, since load_model_shards() only ever calls it as
        model_provider(pre_process=..., post_process=...).
        """
        fake_model_provider_mod, fake_gpt_builders_mod = _install_fake_gpt_provider_modules()
        with mock.patch.dict(
            sys.modules,
            {'model_provider': fake_model_provider_mod, 'gpt_builders': fake_gpt_builders_mod},
        ):
            loader = MegatronCheckpointLoaderLLM(
                types.SimpleNamespace(model_type='GPT'), queue=None
            )
            returned_provider = loader.import_model_provider()

            # This mirrors the exact call site in
            # MegatronCheckpointLoaderBase.load_model_shards().
            result = returned_provider(pre_process=True, post_process=False)

        assert result == ('built', True, False)
