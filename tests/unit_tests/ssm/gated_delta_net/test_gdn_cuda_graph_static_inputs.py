# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA-graph static inputs for layers whose attention has no attention mask."""

import copy

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gdn_layer_gets_no_attention_mask_static_input():
    """A GDN layer must not be handed a CUDA-graph static ``attention_mask``.

    ``GatedDeltaNet`` occupies the ``self_attention`` slot but accepts
    ``attention_mask`` only for signature compatibility and never reads it, and it has
    no ``attn_mask_type``. Before ``uses_attention_mask``, ``get_layer_static_inputs``
    probed ``self_attention.attn_mask_type`` unconditionally and raised AttributeError,
    or (with create_attention_mask_in_dataloader) allocated a persistent
    ``[mbs, 1, slen, seq]`` buffer the layer discards.

    Softmax attention in the same model must be unaffected.
    """
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_experimental_attention_variant_module_spec,
    )
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig
    from megatron.core.transformer.enums import CudaGraphModule
    from megatron.core.transformer.spec_utils import build_module
    from megatron.core.transformer.transformer_layer import TransformerLayer
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    try:
        seq_length, micro_batch_size = 128, 1
        config = TransformerConfig(
            num_layers=2,
            hidden_size=512,
            num_attention_heads=8,
            kv_channels=64,
            use_cpu_initialization=True,
            experimental_attention_variant="gated_delta_net",
            linear_attention_freq=2,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=[CudaGraphModule.attn],
            create_attention_mask_in_dataloader=True,
        )

        base_spec = get_gpt_layer_with_transformer_engine_spec()
        softmax_layer = build_module(base_spec, config=config, layer_number=1)
        softmax_inputs = softmax_layer.get_layer_static_inputs(seq_length, micro_batch_size)
        assert "attention_mask" in softmax_inputs

        gdn_spec = copy.deepcopy(base_spec)
        gdn_spec.submodules.self_attention = get_experimental_attention_variant_module_spec(
            config=config
        )
        gdn_layer = build_module(gdn_spec, config=config, layer_number=2)
        assert isinstance(gdn_layer, TransformerLayer)
        assert gdn_layer.self_attention.uses_attention_mask is False

        # Half one of the regression: no [mbs, 1, slen, seq] buffer the layer discards.
        gdn_inputs = gdn_layer.get_layer_static_inputs(seq_length, micro_batch_size)
        assert "attention_mask" not in gdn_inputs
        # hidden_states is still produced, i.e. the layer remains graphable.
        assert "hidden_states" in gdn_inputs

        # Half two: with the mask omitted from the dataloader, the pre-fix code probed
        # self_attention.attn_mask_type to decide whether to warn, and GDN has none.
        # This is the branch a real SBHD GDN capture run hits.
        config.create_attention_mask_in_dataloader = False
        assert not hasattr(gdn_layer.self_attention, "attn_mask_type")
        gdn_inputs = gdn_layer.get_layer_static_inputs(seq_length, micro_batch_size)
        assert "attention_mask" not in gdn_inputs
        softmax_inputs = softmax_layer.get_layer_static_inputs(seq_length, micro_batch_size)
        assert "attention_mask" not in softmax_inputs  # omitted by config, not by the flag
    finally:
        Utils.destroy_model_parallel()
