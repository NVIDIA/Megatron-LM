# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared GPT model factory + inputs for determinism tests.

The GPT bit-exact harness (``test_gpt_model``) and the FP8 / FP4 recipe
sweep (``test_fp8_determinism``) both build the same toy GPT model with
the same toy inputs. Centralised here so each test file reduces to
"parametrize × call ``RUNNER.run(...)``".
"""

import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.bit_exact_runner import BitExactRunner
from tests.unit_tests.determinism.configs import gpt_base

SEQ_LEN = 32
MICRO_BATCH = 4
VOCAB_SIZE = 128


def build_gpt(overrides, pre_process=True, post_process=True, vp_stage=None, **_):
    """Toy GPT model factory matching the runner's ``build_model`` contract."""
    cfg_kwargs = gpt_base() | overrides
    return GPTModel(
        config=TransformerConfig(**cfg_kwargs),
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=cfg_kwargs.get("num_moe_experts"),
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
    ``False`` for single-cell parametrize sweeps (FP8 recipes, NCCL algos)
    that only need the naive path.
    """
    return BitExactRunner(
        build_model=build_gpt,
        make_inputs=make_gpt_inputs,
        base_config=gpt_base,
        supports_pp=supports_pp,
        seq_len=SEQ_LEN,
        micro_batch=MICRO_BATCH,
    )
