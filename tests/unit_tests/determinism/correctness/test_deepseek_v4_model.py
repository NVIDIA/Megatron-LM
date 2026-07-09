# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-level determinism check for a DeepSeek-V4-style model (DSA).

DeepSeek-V4 extends the V3 architecture (MLA + fine-grained MoE) with DeepSeek
Sparse Attention: a lightning indexer scores tokens and top-k selection
sparsifies core attention. Op-level DSA determinism (the three unique-index
``scatter_`` mask sites, the indexer-loss manual backward, and the unfused
sparse attention) is covered by ``test_dsa_paths.py`` at TP=1; this file adds
the missing **model-level** coverage — DSA inside a full ``GPTModel`` built
through the same spec path production uses
(``get_transformer_block_with_experimental_attention_variant_spec``), under
expert/tensor/pipeline parallelism. The ``tp2-ep2`` cell is the first
distributed bit-exact exercise of the DSA indexer-loss tensor-parallel
all-reduce.

Scope note: this branch carries DSA's pure-torch unfused path only. The dev
branch's DSV4 additions (fused DSA kernels, CSA, hyper-connections) are not
present here and are intentionally out of scope.

Adding a new parallelism cell is a one-line append to
``configs.DEEPSEEK_V4_PARALLELISM_CONFIGS``; this file does not change.
"""

import pytest
import torch

# The DSA indexer rotates activations with the fast_hadamard_transform CUDA
# extension (asserted at import-use time in dsa.py). Skip cleanly where the
# environment does not provide it — a determinism test must run the real
# kernel, not a mock.
pytest.importorskip("fast_hadamard_transform")

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.determinism.bit_exact_runner import BitExactRunner
from tests.unit_tests.determinism.configs import (
    DEEPSEEK_V4_CONFIGS,
    DEEPSEEK_V4_PARALLELISM_CONFIGS,
    deepseek_v4_base,
)

SEQ_LEN = 32
MICRO_BATCH = 4
VOCAB_SIZE = 128


def build_deepseek_v4(overrides, pre_process=True, post_process=True, vp_stage=None, **_):
    """DeepSeek-V4-style GPTModel factory (MLA + MoE + DSA) matching the
    runner's ``build_model`` contract. The block spec is produced by the same
    dispatch production uses when ``experimental_attention_variant`` is set."""
    cfg = MLATransformerConfig(**(deepseek_v4_base() | overrides))
    block_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=cfg, vp_stage=vp_stage
    )
    return GPTModel(
        config=cfg,
        transformer_layer_spec=block_spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQ_LEN,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
        position_embedding_type="rope",
    ).cuda()


def make_deepseek_v4_inputs():
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


# supports_pp=False: DSA's indexer-loss logging tracker is not PP-safe on this
# branch (PP>1 hangs on NCCL P2P setup because per-stage tracker sizes are not
# negotiated across pipeline stages — a functional bug fixed on the dev branch,
# not a determinism divergence). This makes any PP cell skip cleanly instead of
# hanging. See docs/developer/determinism/dsv4-assessment.md.
RUNNER = BitExactRunner(
    build_model=build_deepseek_v4,
    make_inputs=make_deepseek_v4_inputs,
    base_config=deepseek_v4_base,
    supports_pp=False,
    seq_len=SEQ_LEN,
    micro_batch=MICRO_BATCH,
)


class TestDeepSeekV4ModelDeterminism:

    def setup_method(self, method):
        RUNNER.setup()

    def teardown_method(self, method):
        RUNNER.teardown()

    @pytest.mark.internal
    @pytest.mark.parametrize("parallelism", DEEPSEEK_V4_PARALLELISM_CONFIGS)
    @pytest.mark.parametrize("cfg_overrides", DEEPSEEK_V4_CONFIGS)
    def test_bit_exact_under_parallelism(self, cfg_overrides, parallelism):
        RUNNER.run(cfg_overrides, parallelism)
