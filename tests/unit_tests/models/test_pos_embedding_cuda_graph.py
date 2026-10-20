# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPT/Hybrid block graph constants must survive positional embedding cache cleanup."""

import gc
import types
import weakref

import pytest
import torch

from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _RotaryConsumer(torch.nn.Module):
    """Read the constant inside HybridModel.forward without attention/MLP kernels."""

    def forward(self, rotary_pos_emb, **kwargs):
        return rotary_pos_emb + 1


@pytest.mark.parametrize("model_type", ["gpt", "hybrid"])
@pytest.mark.parametrize("position_embedding_type", ["rope", "yarn"])
def test_block_graph_rotary_survives_cache_clear(model_type, position_embedding_type):
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    model = None

    try:
        config = TransformerConfig(
            num_layers=1 if model_type == "gpt" else 2,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        # Model-level YaRN options are attributes, not TransformerConfig fields.
        config.yarn_rotary_scaling_factor = 4.0
        config.yarn_original_max_position_embeddings = 64
        config.yarn_beta_fast = 32.0
        config.yarn_beta_slow = 1.0
        config.yarn_mscale = 1.0
        config.yarn_mscale_all_dim = 0.0
        config.yarn_correction_range_round_to_int = True
        if model_type == "gpt":
            model = GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=128,
                max_sequence_length=128,
                position_embedding_type=position_embedding_type,
            )
        else:
            model = HybridModel(
                config=config,
                hybrid_stack_spec=hybrid_stack_spec,
                hybrid_layer_pattern="*-",
                vocab_size=128,
                max_sequence_length=128,
                position_embedding_type=position_embedding_type,
                post_process=False,
            )
            model.decoder = _RotaryConsumer()
        model = model.cuda().eval()

        # Capture the actual preprocessing that runs inside a block graph. The
        # attention/MLP arithmetic is not needed to exercise this ownership bug.
        config.cuda_graph_impl = "local"
        config.inference_cuda_graph_scope = InferenceCudaGraphScope.block
        context = types.SimpleNamespace(
            max_sequence_length=128,
            is_static_batching=lambda: False,
            is_dynamic_batching=lambda: True,
            use_flashinfer_fused_rope=False,
            num_speculative_tokens=0,
        )
        hidden = torch.zeros((16, 1, 32), device="cuda")
        InferenceMode.set_active()

        def forward():
            if model_type == "hybrid":
                return model.forward(
                    input_ids=None,
                    position_ids=None,
                    attention_mask=None,
                    decoder_input=hidden,
                    inference_context=context,
                    runtime_gather_output=True,
                )
            return (
                model._preprocess(
                    input_ids=None,
                    position_ids=None,
                    decoder_input=hidden,
                    inference_context=context,
                )[1]
                + 1  # For hybrid, it's in the decoder
            )

        with torch.no_grad():
            refs = []
            graphs = []

            def record_rotary(module, args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                refs.append(weakref.ref(tensor))

            for seq_len in (64, 128):
                context.max_sequence_length = seq_len
                # Capture must retain the same constant used by warmup.
                expected = forward().clone()
                graph = torch.cuda.CUDAGraph()
                with model.rotary_pos_emb.register_forward_hook(record_rotary):
                    with torch.cuda.graph(graph):
                        output = forward()
                graphs.append((graph, output, expected))

            model.rotary_pos_emb.clear_cache()
            gc.collect()
            torch.cuda.empty_cache()

            assert len(refs) == 2
            for ref in refs:
                assert ref() is not None, "block graph lost its cached positional embedding"
                assert any(
                    tensor is ref() for tensor, _ in model.rotary_pos_emb._embedding_cache.values()
                )
            for graph, output, expected in graphs:
                graph.replay()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

    finally:
        InferenceMode.unset_active()
        Utils.destroy_model_parallel()
