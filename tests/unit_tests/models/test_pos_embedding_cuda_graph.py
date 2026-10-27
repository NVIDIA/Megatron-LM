# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GPT/Hybrid positional embedding caches must survive train/generation transitions."""

import gc
import types
import weakref

import pytest
import torch

from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    capture_constants,
    create_cudagraphs,
    delete_cuda_graphs,
    release_capture_constants,
)
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _RotaryConsumer(torch.nn.Module):
    """Expose HybridModel's embedding without attention/MLP kernels."""

    input_tensor = None

    def forward(self, rotary_pos_emb, **kwargs):
        return rotary_pos_emb


@pytest.mark.parametrize("model_type", ["gpt", "hybrid"])
@pytest.mark.parametrize("position_embedding_type", ["rope", "yarn"])
@pytest.mark.parametrize("clear_api", ["instance", "legacy_bound", "legacy_class"])
def test_block_graph_rotary_survives_train_generation_switch(
    model_type, position_embedding_type, clear_api
):
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
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
        if clear_api == "instance":
            clear_cache = model.rotary_pos_emb.clear_cache
        elif clear_api == "legacy_bound":
            clear_cache = model.rotary_pos_emb.forward.cache_clear
        else:
            clear_cache = type(model.rotary_pos_emb).forward.cache_clear

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

        def forward(inference_context=None):
            if model_type == "hybrid":
                return model.forward(
                    input_ids=None,
                    position_ids=None,
                    attention_mask=None,
                    decoder_input=hidden,
                    inference_context=inference_context,
                    runtime_gather_output=True,
                )
            return model._preprocess(
                input_ids=None,
                position_ids=None,
                decoder_input=hidden,
                inference_context=inference_context,
            )[1]

        with torch.inference_mode():
            graphs = []
            for seq_len in (64, 128, 64):
                context.max_sequence_length = seq_len
                # Capture must retain the same constant used by warmup.
                ref = weakref.ref(forward(context))
                expected = forward(context) + 1
                graph = torch.cuda.CUDAGraph()
                with capture_constants(graph), torch.cuda.graph(graph):
                    output = forward(context) + 1
                graphs.append((seq_len, ref, graph, output, expected))

        # Generation -> training: captured constants survive cleanup and can be
        # saved for backward even though generation populated them in inference mode.
        model.train()
        InferenceMode.unset_active()
        clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
        for seq_len, ref, *_ in graphs:
            assert ref() is not None, "block graph lost its cached positional embedding"
            hidden = torch.zeros((seq_len, 1, 32), device="cuda")
            embedding = forward()
            torch.testing.assert_close(embedding, ref(), rtol=0, atol=0)
            assert not embedding.is_inference()
            activations = torch.randn_like(embedding, requires_grad=True)
            (activations * embedding).sum().backward()
            torch.testing.assert_close(activations.grad, embedding, rtol=0, atol=0)
        del embedding

        # Training -> generation: discard ordinary training entries while preserving
        # the tensors and addresses consumed by the already-captured graphs.
        hidden = torch.zeros((16, 1, 32), device="cuda")
        training_ref = weakref.ref(forward())
        model.eval()
        InferenceMode.set_active()
        clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
        assert training_ref() is None
        with torch.inference_mode():
            for seq_len, ref, graph, output, expected in graphs:
                context.max_sequence_length = seq_len
                torch.testing.assert_close(forward(context), ref(), rtol=0, atol=0)
                graph.replay()
                torch.testing.assert_close(output, expected, rtol=0, atol=0)

        # Two graphs share the warmup tensor. Deleting one must not invalidate the other.
        clear_cache()
        first_graph = graphs[0][2]
        first_graph.reset()
        release_capture_constants(first_graph)
        assert graphs[0][1]() is not None
        graphs[2][2].replay()
        torch.testing.assert_close(graphs[2][3], graphs[2][4], rtol=0, atol=0)
        for _, _, graph, _, _ in graphs:
            graph.reset()
            release_capture_constants(graph)
        gc.collect()
        assert all(ref() is None for _, ref, *_ in graphs)

    finally:
        InferenceMode.unset_active()
        Utils.destroy_model_parallel()


class _RotaryGraphModule(MegatronModule):
    def __init__(self, config, embedding_type) -> None:
        super().__init__(config)
        self.rotary_pos_emb = embedding_type(32, 1.0)
        self.embedding_ref = None

    def project(self, x):
        embedding = self.rotary_pos_emb(x.shape[0])
        if isinstance(embedding, tuple):
            embedding = embedding[0]
        self.embedding_ref = weakref.ref(embedding)
        return x * embedding


@pytest.mark.parametrize("embedding_type", [RotaryEmbedding, YarnRotaryEmbedding])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("cache_hit", [False, True])
def test_rotary_constants_follow_managed_graph_lifetime(embedding_type, training, cache_hit):
    """Exercise capture misses, replay, graph deletion and recapture through the real manager."""
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(
        123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
    )
    try:
        config = TransformerConfig(
            num_layers=1, hidden_size=32, num_attention_heads=1, cuda_graph_impl="local"
        )
        module = _RotaryGraphModule(config, embedding_type).cuda().train(training)
        manager = CudaGraphManager(
            config,
            base_module=module,
            function_name="project",
            need_backward=training,
            inline_capture=not training,
            num_warmup_steps=0,
        )
        for _ in range(2):
            x = torch.randn(16, 1, 1, 32, device="cuda", requires_grad=training)
            expected = module.project(x, eager=True).detach().clone()
            if training:
                module.project(x).sum().backward()
                x.grad = None
            if not cache_hit:
                module.rotary_pos_emb.clear_cache()

            if training:
                create_cudagraphs()
            else:
                with torch.inference_mode():
                    module.project(x)
            ref = module.embedding_ref
            module.rotary_pos_emb.clear_cache()
            gc.collect()
            torch.cuda.empty_cache()
            assert ref() is not None

            with torch.set_grad_enabled(training):
                output = module.project(x)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                if training:
                    # Backward must own its saved constants independently of the forward graph.
                    torch.cuda.synchronize()
                    graph = manager.cudagraph_runners[0].fwd_graph
                    graph.reset()
                    release_capture_constants(graph)
                    assert ref() is not None
                    output.sum().backward()
                    torch.testing.assert_close(x.grad, ref(), rtol=0, atol=0)
            del output

            delete_cuda_graphs()
            gc.collect()
            assert all(
                r.fwd_graph is None and r.bwd_graph is None for r in manager.cudagraph_runners
            )
            assert ref() is None, "destroyed graph retained a positional embedding"
    finally:
        delete_cuda_graphs()
        torch.cuda.set_stream(torch.cuda.default_stream())
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("embedding_type", [RotaryEmbedding, YarnRotaryEmbedding])
def test_failed_capture_releases_rotary_constants(embedding_type):
    Utils.initialize_model_parallel(1, 1)
    try:
        module = embedding_type(32, 1.0)
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="abort capture"):
            with capture_constants(graph), torch.cuda.graph(graph):
                embedding = module(16)
                if isinstance(embedding, tuple):
                    embedding = embedding[0]
                ref = weakref.ref(embedding)
                output = embedding + 1
                del embedding
                raise RuntimeError("abort capture")
        graph.reset()
        del output
        gc.collect()
        assert ref() is None
    finally:
        Utils.destroy_model_parallel()


def _embedding_tensor(module, seq_len=16):
    result = module(seq_len)
    return result[0] if isinstance(result, tuple) else result


@pytest.mark.parametrize("embedding_type", [RotaryEmbedding, YarnRotaryEmbedding])
def test_rotary_cache_capacity_is_shared_across_instances(embedding_type):
    Utils.initialize_model_parallel(1, 1)
    try:
        first = embedding_type(32, 1.0, rotary_base=10000)
        second = embedding_type(32, 1.0, rotary_base=100000)
        first_tensor = _embedding_tensor(first)
        second_tensor = _embedding_tensor(second)
        assert not torch.equal(first_tensor, second_tensor)
        assert _embedding_tensor(first) is first_tensor
        assert _embedding_tensor(second) is second_tensor
        first_ref, second_ref = weakref.ref(first_tensor), weakref.ref(second_tensor)
        del first_tensor, second_tensor

        capacity = embedding_type.CACHE_CAPACITY_FOR_NON_CG_EMBS
        for seq_len in range(32, 32 + capacity - 2):
            second(seq_len)
        assert first_ref() is not None
        assert second_ref() is not None

        # Touching the first instance must update the shared LRU order.
        assert _embedding_tensor(first) is first_ref()
        second(32 + capacity)
        assert first_ref() is not None
        assert second_ref() is None
    finally:
        embedding_type.clear_cache()
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("embedding_type", [RotaryEmbedding, YarnRotaryEmbedding])
def test_class_cache_clear_scope_and_module_lifetime(embedding_type):
    Utils.initialize_model_parallel(1, 1)
    try:
        first = embedding_type(32, 1.0)
        second = embedding_type(32, 1.0)
        other_type = YarnRotaryEmbedding if embedding_type is RotaryEmbedding else RotaryEmbedding
        other = other_type(32, 1.0)

        other_ref = weakref.ref(_embedding_tensor(other))
        for forward in (None, first.forward, embedding_type.forward):
            first_ref = weakref.ref(_embedding_tensor(first))
            second_ref = weakref.ref(_embedding_tensor(second))
            if forward is None:
                first.clear_cache()
            elif hasattr(forward, "cache_parameters"):
                assert forward.cache_parameters() == {
                    "maxsize": embedding_type.CACHE_CAPACITY_FOR_NON_CG_EMBS,
                    "typed": False,
                }
                forward.cache_clear()
            assert first_ref() is None
            assert second_ref() is None
            assert other_ref() is not None

        first_refs = [weakref.ref(_embedding_tensor(first, seq_len)) for seq_len in (16, 32)]
        second_ref = weakref.ref(_embedding_tensor(second))
        module_ref = weakref.ref(first)
        del first
        gc.collect()
        assert module_ref() is None
        assert all(tensor_ref() is None for tensor_ref in first_refs)
        assert second_ref() is not None
        assert other_ref() is not None
    finally:
        RotaryEmbedding.clear_cache()
        YarnRotaryEmbedding.clear_cache()
        Utils.destroy_model_parallel()
