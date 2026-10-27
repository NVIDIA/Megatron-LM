# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-level determinism check for HybridModel (Mamba + attention).

Adding a new parallelism cell is a one-line append to
``configs.PARALLELISM_CONFIGS``. ``HYBRID_CONFIGS`` provides the
(layer_pattern, overrides) presets specific to this model class.
"""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.bit_exact_runner import BitExactRunner
from tests.unit_tests.determinism.configs import HYBRID_CONFIGS, hybrid_base

# Hybrid covers the cheap-and-valuable composites that exercise Mamba +
# parallelism interactions. The first cell pays a ~60s JIT tax (TE attention
# + Mamba selective_scan under the hybrid layer-spec); subsequent cells reuse
# the cache so they're ~5–10s each. Excluded:
#   * TP=4 / TP=8 — Mamba shard shape re-JIT costs ~25s/40s extra; TP=2
#     composites below already exercise the TP+Mamba sharding path.
#   * EP cells (ep2, tp2-ep2, tp2-ep4, fsdp8-ep4) — MoE-inside-hybrid grouped
#     GEMM compiles a new (E, K, N) shape that doesn't share with the dense
#     hybrid kernels (~60s extra JIT). GPT-model EP cells already cover MoE
#     + EP determinism; "MoE in the MLP slot of a hybrid pattern" is marginal
#     (Mamba layers have no MoE).
_HYBRID_PARALLELISM_CONFIGS = [
    pytest.param({"PP": 2}, id="pp2"),
    pytest.param({"PP": 4}, id="pp4"),
    pytest.param({"TP": 2, "PP": 2}, id="tp2-pp2"),
    pytest.param({"PP": 2, "VPP": 2}, id="pp2-vpp2"),
    pytest.param({"FSDP": 8}, id="fsdp8"),
]

_SEQ_LEN = 32
_MICRO_BATCH = 2
_VOCAB_SIZE = 128


def _hybrid_inputs() -> dict:
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


# Module-level lifecycle helper — its build_model lambda is never invoked;
# only setup/teardown (init / destroy model-parallel + cache flush) are used.
# Per-test runners are built inside the test body because HybridModel needs
# the layer_pattern from HYBRID_CONFIGS, which the parametrize feeds in.
_LIFECYCLE = BitExactRunner(
    build_model=lambda *a, **k: None,
    make_inputs=_hybrid_inputs,
    base_config=hybrid_base,
    supports_pp=False,
)


class TestHybridModelDeterminism:

    def setup_method(self, method):
        _LIFECYCLE.setup()

    def teardown_method(self, method):
        _LIFECYCLE.teardown()

    @pytest.mark.internal
    @pytest.mark.parametrize("parallelism", _HYBRID_PARALLELISM_CONFIGS)
    @pytest.mark.parametrize("layer_pattern, cfg_overrides", HYBRID_CONFIGS)
    def test_bit_exact_under_parallelism(self, layer_pattern, cfg_overrides, parallelism):
        # hybrid_layer_pattern length must be divisible by PP. Repeat the
        # base pattern until that holds.
        pp = parallelism.get("PP", 1)
        vpp = parallelism.get("VPP", 1) or 1
        stages = pp * vpp
        if stages > 1 and len(layer_pattern) % stages != 0:
            reps = stages // len(layer_pattern) + 1
            layer_pattern = layer_pattern * reps
            trim = (len(layer_pattern) // stages) * stages
            layer_pattern = layer_pattern[:trim]
        # HybridModel requires explicit '|' separators for VPP so the layer
        # allocator knows the per-(pp,vpp)-stage boundary. Split the pattern
        # evenly across stages.
        if vpp > 1:
            seg = len(layer_pattern) // stages
            layer_pattern = "|".join(layer_pattern[i * seg : (i + 1) * seg] for i in range(stages))

        def build(overrides, pre_process=True, post_process=True, vp_stage=None, **_):
            cfg = TransformerConfig(**(hybrid_base() | overrides))
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
            make_inputs=_hybrid_inputs,
            base_config=hybrid_base,
            supports_pp=True,  # HybridModel inherits PP support
            seq_len=_SEQ_LEN,
            micro_batch=_MICRO_BATCH,
        )
        runner.run(cfg_overrides, parallelism)
