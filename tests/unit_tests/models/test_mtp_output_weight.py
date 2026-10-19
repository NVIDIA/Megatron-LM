# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("model_type", ["gpt", "hybrid"])
@pytest.mark.parametrize("tied", [False, True])
def test_mtp_output_weight_receives_gradient(model_type, tied):
    """An MTP stage must not own an output weight bypassed by its tied embedding."""
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            mtp_num_layers=1,
            gradient_accumulation_fusion=False,
        )
        kwargs = dict(
            config=config,
            vocab_size=128,
            max_sequence_length=4,
            pre_process=False,
            post_process=True,
            share_embeddings_and_output_weights=tied,
        )
        if model_type == "gpt":
            layer_spec = get_gpt_layer_with_transformer_engine_spec()
            model = GPTModel(
                **kwargs,
                transformer_layer_spec=layer_spec,
                mtp_block_spec=get_gpt_mtp_block_spec(config, layer_spec, True),
            )
        else:
            model = HybridModel(
                **kwargs, hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern="-/-"
            )
        assert model.mtp_process

        # Use the same weight selection as model postprocessing, without running
        # the decoder or requiring a multi-stage training schedule.
        output_weight = model.shared_embedding_or_output_weight() if tied else None
        device = next(model.parameters()).device
        hidden = torch.randn(4, 1, config.hidden_size, device=device)
        logits, _ = model.output_layer(hidden, weight=output_weight)
        logits.square().mean().backward()

        if tied:
            assert output_weight is model.embedding.word_embeddings.weight
            assert output_weight.grad is not None
        # With the old allocation condition, tied MTP stages fail here: the
        # registered output_layer.weight is unused and has no gradient.
        missing = [
            name
            for name, parameter in model.output_layer.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert not missing, f"Output projection owns unused trainable parameters: {missing}"
        assert (model.output_layer.weight is None) == tied
    finally:
        Utils.destroy_model_parallel()
