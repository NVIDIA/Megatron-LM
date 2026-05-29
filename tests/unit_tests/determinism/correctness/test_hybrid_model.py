# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-level determinism check for HybridModel (Mamba + attention).

Adding a new parallelism cell is a one-line append to
``configs.PARALLELISM_CONFIGS``. ``HYBRID_CONFIGS`` provides the
(layer_pattern, overrides) presets specific to this model class.

``MambaModel`` is a deprecated alias for ``HybridModel`` — a single Hybrid
test covers both.
"""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.configs import HYBRID_CONFIGS, hybrid_base, parallelism_configs

# Hybrid skips TP=8: Mamba selective_scan / conv1d CUDA extensions JIT-compile
# per per-rank shard shape, and the second high-TP cell would re-JIT for a
# different shape (~10s extra). TP coverage at full degree is already exercised
# by ``test_gpt_model``; keeping TP=4 here gives one high-TP hybrid run that
# verifies the Mamba sharding path stays deterministic.
_HYBRID_PARALLELISM_CONFIGS = parallelism_configs(exclude=("tp8",))
from tests.unit_tests.determinism.bit_exact_runner import BitExactRunner

_SEQ_LEN = 32
_MICRO_BATCH = 2
_VOCAB_SIZE = 128


@pytest.fixture(scope="module", autouse=True)
def _prewarm_mamba_kernels():
    """Mamba's selective_scan / conv1d CUDA extensions JIT-compile on first
    use (~50s). Warm them once at module load so subsequent cells reuse
    the cached kernels.

    The warmup forward is wrapped in try/finally so a transient JIT/OOM
    failure still tears down ``parallel_state`` — otherwise subsequent
    tests hit ``setup_method`` with TP=1 still initialised and produce a
    cascade of secondary errors that masks the original failure.
    """
    import os

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from tests.unit_tests.test_utilities import Utils

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        cfg_kwargs = hybrid_base()
        cfg_kwargs["num_layers"] = 1
        cfg = TransformerConfig(**cfg_kwargs)
        warm = HybridModel(
            config=cfg,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=_VOCAB_SIZE,
            max_sequence_length=_SEQ_LEN,
            hybrid_layer_pattern="M",
        ).cuda()
        with torch.no_grad():
            warm(**_hybrid_inputs())
        del warm
    finally:
        Utils.destroy_model_parallel()
        torch.cuda.empty_cache()
    yield


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


class TestHybridModelDeterminism:

    def setup_method(self, method):
        # Use a placeholder runner during setup; the real one is built per-test
        # because HybridModel needs the layer_pattern from HYBRID_CONFIGS.
        BitExactRunner(
            build_model=lambda *a, **k: None,
            make_inputs=_hybrid_inputs,
            base_config=hybrid_base,
            supports_pp=False,
        ).setup()

    def teardown_method(self, method):
        BitExactRunner(
            build_model=lambda *a, **k: None,
            make_inputs=_hybrid_inputs,
            base_config=hybrid_base,
            supports_pp=False,
        ).teardown()

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
            layer_pattern = "|".join(
                layer_pattern[i * seg : (i + 1) * seg] for i in range(stages)
            )

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
