# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-level determinism check for GPTModel.

Adding a new parallelism cell is a one-line append to
``configs.PARALLELISM_CONFIGS`` — this file does not need to change.

The model factory + inputs + runner-builder live here (not in a separate
helpers file) because ``test_fp8_determinism.py`` is the only other
consumer and the natural home for "toy GPT model" is alongside the
canonical GPT determinism test. Importing from a test module is safe:
the module body only defines helpers + the ``RUNNER`` singleton (no test
side effects at import time).
"""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.bit_exact_runner import BitExactRunner
from tests.unit_tests.determinism.configs import GPT_CONFIGS, PARALLELISM_CONFIGS, gpt_base

SEQ_LEN = 32
MICRO_BATCH = 4
VOCAB_SIZE = 128


def build_gpt(overrides, pre_process=True, post_process=True, vp_stage=None, **_):
    """Toy GPT model factory matching the runner's ``build_model`` contract."""
    cfg_kwargs = gpt_base() | overrides
    return GPTModel(
        config=TransformerConfig(**cfg_kwargs),
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=cfg_kwargs.get("num_moe_experts")
        ),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    ).cuda()


def make_gpt_inputs():
    """Toy GPT inputs matching the runner's ``make_inputs`` contract."""
    return {
        "input_ids": torch.randint(
            0, VOCAB_SIZE, (MICRO_BATCH, SEQ_LEN), device="cuda", dtype=torch.long
        ),
        "position_ids": torch.arange(SEQ_LEN, device="cuda", dtype=torch.long)
        .unsqueeze(0)
        .repeat(MICRO_BATCH, 1),
        "attention_mask": torch.ones(
            MICRO_BATCH, 1, SEQ_LEN, SEQ_LEN, dtype=torch.bool, device="cuda"
        ),
    }


def make_gpt_runner(supports_pp: bool = True) -> BitExactRunner:
    """Configured ``BitExactRunner`` for the toy GPT model.

    ``supports_pp=True`` for tests that exercise the pipeline schedule;
    ``False`` for single-cell parametrize sweeps (FP8 recipes, etc).
    """
    return BitExactRunner(
        build_model=build_gpt,
        make_inputs=make_gpt_inputs,
        base_config=gpt_base,
        supports_pp=supports_pp,
        seq_len=SEQ_LEN,
        micro_batch=MICRO_BATCH,
    )


RUNNER = make_gpt_runner(supports_pp=True)


class TestGPTModelDeterminism:

    def setup_method(self, method):
        RUNNER.setup()

    def teardown_method(self, method):
        RUNNER.teardown()

    @pytest.mark.internal
    @pytest.mark.parametrize("parallelism", PARALLELISM_CONFIGS)
    @pytest.mark.parametrize("cfg_overrides", GPT_CONFIGS)
    def test_bit_exact_under_parallelism(self, cfg_overrides, parallelism):
        RUNNER.run(cfg_overrides, parallelism)
