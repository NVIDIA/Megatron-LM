# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-level determinism check for a DeepSeek-V3-style model.

DeepSeek-V3 combines Multi-Latent Attention (MLA) with fine-grained MoE and
**group-limited (node-limited) routing**. That routing path
(``moe_utils.group_limited_topk`` and the ``scatter`` routing-map construction)
plus the MLA low-rank projections are not covered by ``test_gpt_model.py`` (dense
+ standard attention) — this file adds them.

The model is a scaled-down proxy: it is not numerically the real DSV3, but it
builds the same determinism-relevant kernels (MLA, sigmoid routing + expert bias,
group-limited top-k, seq-aux-loss balancing, grouped-GEMM experts). A determinism
test only asserts run-A == run-B, so proxy fidelity to the real recipe is
irrelevant — code-path coverage is what matters.

Adding a new parallelism cell is a one-line append to
``configs.DEEPSEEK_PARALLELISM_CONFIGS``; this file does not change.
"""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.determinism.bit_exact_runner import BitExactRunner
from tests.unit_tests.determinism.configs import (
    DEEPSEEK_CONFIGS,
    DEEPSEEK_PARALLELISM_CONFIGS,
    deepseek_base,
)

SEQ_LEN = 32
MICRO_BATCH = 4
VOCAB_SIZE = 128


def build_deepseek(overrides, pre_process=True, post_process=True, vp_stage=None, **_):
    """DeepSeek-V3-style GPTModel factory (MLA + MoE) matching the runner's
    ``build_model`` contract."""
    cfg_kwargs = deepseek_base() | overrides
    return GPTModel(
        config=MLATransformerConfig(**cfg_kwargs),
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=cfg_kwargs.get("num_moe_experts"),
            moe_grouped_gemm=cfg_kwargs.get("moe_grouped_gemm", False),
            multi_latent_attention=True,
            qk_layernorm=cfg_kwargs.get("qk_layernorm", False),
        ),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
        # MLA applies its own (YaRN) rotary internally; the GPTModel-level rope
        # matches the real DSV3 recipe (``--position-embedding-type rope``).
        position_embedding_type="rope",
    ).cuda()


def make_deepseek_inputs():
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


RUNNER = BitExactRunner(
    build_model=build_deepseek,
    make_inputs=make_deepseek_inputs,
    base_config=deepseek_base,
    supports_pp=True,
    seq_len=SEQ_LEN,
    micro_batch=MICRO_BATCH,
)


class TestDeepSeekModelDeterminism:

    def setup_method(self, method):
        RUNNER.setup()

    def teardown_method(self, method):
        RUNNER.teardown()

    @pytest.mark.internal
    @pytest.mark.parametrize("parallelism", DEEPSEEK_PARALLELISM_CONFIGS)
    @pytest.mark.parametrize("cfg_overrides", DEEPSEEK_CONFIGS)
    def test_bit_exact_under_parallelism(self, cfg_overrides, parallelism):
        RUNNER.run(cfg_overrides, parallelism)
