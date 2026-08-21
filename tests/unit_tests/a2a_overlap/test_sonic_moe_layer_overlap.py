# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fine-grained expert-parallel overlap tests for SonicMoE."""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.moe.sonic_moe_layer import (
    SonicMoELayer,
    replace_moe_layer_specs_with_sonic_moe,
)
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.test_schedule_layer_1f1b import (
    run_transformer_layer_a2a_overlap_with_capture,
    run_transformer_layer_ref_with_capture,
)
from tests.unit_tests.a2a_overlap.utils import (
    apply_flex_backend_kwargs,
    build_data,
    compare_captures,
    deterministic_mode,
    get_test_config,
    get_valid_dispatcher_configs,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("sonicmoe")


class TestSonicMoELayerOverlap:
    """Compare Sonic's fine-grained scheduler against serial layer execution."""

    def setup_method(self):
        """Initialize the EP=4 topology used by Megatron's overlap tests."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
            expert_tensor_parallel_size=1,
        )

    def teardown_method(self):
        """Destroy model-parallel groups after each overlap test."""
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9")
    @pytest.mark.parametrize("dispatcher_type,flex_backend", get_valid_dispatcher_configs())
    def test_transformer_layer_overlap(self, dispatcher_type, flex_backend, monkeypatch):
        """Require bitwise parity between serial and fine-grained Sonic execution."""
        if flex_backend in ("deepep", "deepepv2"):
            # Sonic recommends disabling QuACK autotuning for variable-shape EP. Besides avoiding
            # repeated tuning for per-rank token shapes, this makes the deterministic overlap
            # comparison use one fixed kernel configuration for reference and scheduled execution.
            monkeypatch.setenv("SONICMOE_QUACK_TUNED", "0")

        extra_kwargs = {"gated_linear_unit": True}
        apply_flex_backend_kwargs(extra_kwargs, dispatcher_type, flex_backend)
        config = get_test_config(extra_kwargs=extra_kwargs)
        microbatches = 4

        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=config, use_transformer_engine=True
            )
            assert replace_moe_layer_specs_with_sonic_moe(transformer_layer_spec) == 1
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            assert isinstance(model.decoder.layers[0].mlp, SonicMoELayer)

            params = reset_model(model)
            input_tensors = [build_data() for _ in range(microbatches)]
            reference_capture = run_transformer_layer_ref_with_capture(
                model, input_tensors, microbatches
            )

            reset_model(model, params)
            overlap_capture = run_transformer_layer_a2a_overlap_with_capture(
                model, input_tensors, microbatches
            )
            matches, message = compare_captures(reference_capture, overlap_capture, True)
            assert matches, f"[rank {torch.distributed.get_rank()}] {message}"
