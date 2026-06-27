# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-level determinism check for a Nemotron-3-Ultra-style hybrid model.

Nemotron-3-Ultra is a hybrid stack of Mamba (``M``), attention (``*``) and MoE
(``E``) layers (real pattern ``MEMEMEM*...``). ``test_hybrid_model.py`` covers
the dense hybrid (Mamba + attention + MLP) but deliberately omits the MoE/EP
cells; this file adds **MoE-inside-hybrid** — grouped-GEMM experts in the MLP
slot of a hybrid stack — under expert parallelism, which is the configuration
where DSV3/Nemotron non-determinism empirically appears at scale.

The model is a scaled-down proxy (toy sizes), built with the same
determinism-relevant kernels as the real recipe (Mamba selective scan, TE
attention, sigmoid routing + expert bias, seq-aux-loss, grouped GEMM). A
determinism test asserts only run-A == run-B, so proxy fidelity is irrelevant.

Adding a new parallelism cell is a one-line append to
``configs.NEMOTRON_PARALLELISM_CONFIGS``.
"""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.bit_exact_runner import BitExactRunner
from tests.unit_tests.determinism.configs import (
    NEMOTRON_CONFIGS,
    NEMOTRON_PARALLELISM_CONFIGS,
    nemotron_hybrid_base,
)

_SEQ_LEN = 32
_MICRO_BATCH = 2
_VOCAB_SIZE = 128


def _nemotron_inputs() -> dict:
    return {
        "input_ids": torch.randint(
            0, _VOCAB_SIZE, (_MICRO_BATCH, _SEQ_LEN), device="cuda", dtype=torch.long
        ),
        "position_ids": (
            torch.arange(_SEQ_LEN, device="cuda", dtype=torch.long)
            .unsqueeze(0)
            .repeat(_MICRO_BATCH, 1)
        ),
        "attention_mask": torch.ones(
            _MICRO_BATCH, 1, _SEQ_LEN, _SEQ_LEN, dtype=torch.bool, device="cuda"
        ),
    }


# Lifecycle-only runner; its build_model lambda is never invoked. Per-test
# runners are built in the test body because HybridModel needs the layer_pattern
# from NEMOTRON_CONFIGS, which the parametrize feeds in.
_LIFECYCLE = BitExactRunner(
    build_model=lambda *a, **k: None,
    make_inputs=_nemotron_inputs,
    base_config=nemotron_hybrid_base,
    supports_pp=False,
)


class TestNemotronHybridModelDeterminism:

    def setup_method(self, method):
        _LIFECYCLE.setup()

    def teardown_method(self, method):
        _LIFECYCLE.teardown()

    @pytest.mark.internal
    @pytest.mark.parametrize("parallelism", NEMOTRON_PARALLELISM_CONFIGS)
    @pytest.mark.parametrize("layer_pattern, cfg_overrides", NEMOTRON_CONFIGS)
    def test_bit_exact_under_parallelism(self, layer_pattern, cfg_overrides, parallelism):
        # hybrid_layer_pattern length must be divisible by PP*VPP. Repeat the
        # base pattern until that holds (same logic as test_hybrid_model.py).
        pp = parallelism.get("PP", 1)
        vpp = parallelism.get("VPP", 1) or 1
        stages = pp * vpp
        if stages > 1 and len(layer_pattern) % stages != 0:
            reps = stages // len(layer_pattern) + 1
            layer_pattern = layer_pattern * reps
            trim = (len(layer_pattern) // stages) * stages
            layer_pattern = layer_pattern[:trim]
        # HybridModel requires explicit '|' separators for VPP so the layer
        # allocator knows the per-(pp,vpp)-stage boundary.
        if vpp > 1:
            seg = len(layer_pattern) // stages
            layer_pattern = "|".join(layer_pattern[i * seg : (i + 1) * seg] for i in range(stages))

        def build(overrides, pre_process=True, post_process=True, vp_stage=None, **_):
            cfg = TransformerConfig(**(nemotron_hybrid_base() | overrides))
            return HybridModel(
                config=cfg,
                hybrid_stack_spec=hybrid_stack_spec,
                vocab_size=_VOCAB_SIZE,
                max_sequence_length=_SEQ_LEN,
                hybrid_layer_pattern=layer_pattern,
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            ).cuda()

        runner = BitExactRunner(
            build_model=build,
            make_inputs=_nemotron_inputs,
            base_config=nemotron_hybrid_base,
            supports_pp=True,  # HybridModel inherits PP support
            seq_len=_SEQ_LEN,
            micro_batch=_MICRO_BATCH,
        )
        runner.run(cfg_overrides, parallelism)
