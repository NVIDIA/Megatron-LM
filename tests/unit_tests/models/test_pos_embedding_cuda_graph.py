# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPT/Hybrid block graph constants must outlive the RoPE/YaRN LRU caches."""

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
@pytest.mark.parametrize("invalidate", ["clear", "evict"])
def test_block_graph_rotary_survives_lru_invalidation(
    model_type, position_embedding_type, invalidate
):
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
                + 1
            )

        with torch.no_grad():
            refs = []

            def record_rotary(module, args, output):
                tensor = output[0] if isinstance(output, tuple) else output
                refs.append(weakref.ref(tensor))

            handle = model.rotary_pos_emb.register_forward_hook(record_rotary)
            expected = forward().cpu()
            eager_weak = refs[-1]
            model.rotary_pos_emb.forward.cache_clear()
            gc.collect()
            assert eager_weak() is None, "eager-only RoPE constants must not be retained"
            expected = forward().cpu()
            handle.remove()
            weak = refs[-1]
            shape = weak().shape
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = forward()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = forward()
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

            # Recapture the same shape after the LRU has changed. Replacing the
            # model-owned constant here would invalidate the first graph.
            second_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(second_graph, stream=stream):
                second_output = forward()
            model.rotary_pos_emb.forward.cache_clear()
            gc.collect()
            assert weak() is not None, "recapture released the first graph's RoPE constant"
            graph.replay()
            second_graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output.cpu(), expected)
            assert torch.equal(second_output.cpu(), expected)
            del graph, second_graph, output, second_output, scratch
    finally:
        InferenceMode.unset_active()
        if model is not None:
            model.rotary_pos_emb.forward.cache_clear()
        Utils.destroy_model_parallel()
