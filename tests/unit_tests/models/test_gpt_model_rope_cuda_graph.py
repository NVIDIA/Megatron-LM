# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The RoPE constants read by GPT block graphs must outlive the shared LRU cache."""

import gc
import types
import weakref

import pytest
import torch

from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("invalidate", ["clear", "evict"])
def test_block_graph_rotary_survives_lru_invalidation(invalidate):
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    model = None
    try:
        config = TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=4, use_cpu_initialization=True
        )
        model = (
            GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=128,
                max_sequence_length=128,
                position_embedding_type="rope",
            )
            .cuda()
            .eval()
        )
        # Capture the actual preprocessing that runs inside a block graph. The
        # attention/MLP arithmetic is not needed to exercise this ownership bug.
        config.cuda_graph_impl = "local"
        config.inference_cuda_graph_scope = InferenceCudaGraphScope.block
        context = types.SimpleNamespace(
            max_sequence_length=128,
            is_static_batching=lambda: False,
            is_dynamic_batching=lambda: True,
            use_flashinfer_fused_rope=False,
        )
        hidden = torch.zeros((16, 1, 32), device="cuda")
        InferenceMode.set_active()

        def rope():
            return model._preprocess(
                input_ids=None,
                position_ids=None,
                decoder_input=hidden,
                inference_context=context,
            )[1]

        with torch.no_grad():
            cached = rope()
            weak = weakref.ref(cached)
            shape = cached.shape
            expected = (cached + 1).cpu()
            del cached
            static_input = torch.ones(shape, device="cuda")
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = rope() + static_input
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = rope() + static_input
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output.cpu(), expected)

            if invalidate == "clear":
                model.rotary_pos_emb.forward.cache_clear()
            else:
                for length in range(1, 34):
                    model.rotary_pos_emb(length)
            gc.collect()
            # Fail before replay if the constant was freed, rather than reading
            # a dangling device pointer to demonstrate the old failure.
            assert weak() is not None, "block graph lost its cached RoPE constant"
            scratch = [torch.full(shape, -3.0, device="cuda") for _ in range(32)]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output.cpu(), expected)
            del graph, output, scratch
    finally:
        InferenceMode.unset_active()
        if model is not None:
            model.rotary_pos_emb.forward.cache_clear()
        Utils.destroy_model_parallel()
