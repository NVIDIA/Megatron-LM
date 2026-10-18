# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import gc

import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    set_batch_invariant_mode,
    te_supports_batch_invariant_attention,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.models.test_gpt_model_batch_invariant import (
    _BIK_FA_VERSION,
    HAVE_FA3,
    HAVE_FA4,
    DummyTokenizer,
    _configure_flash_attention_env,
)
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def batch_invariant_mla():
    if not te_supports_batch_invariant_attention():
        pytest.skip("Requires batch-invariant Transformer Engine attention")
    if not (HAVE_FA3 or HAVE_FA4):
        pytest.skip("Requires FlashAttention-3 or -4")
    pytest.importorskip("flash_mla")
    Utils.initialize_model_parallel(1, 1)
    try:
        with set_batch_invariant_mode(True, backend="te_native"):
            yield
    finally:
        Utils.destroy_model_parallel()


def test_mla_request_arrival_batch_invariance(batch_invariant_mla):
    _configure_flash_attention_env()
    torch.manual_seed(321)
    model_parallel_cuda_manual_seed(321)
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=512,
        num_attention_heads=8,
        multi_latent_attention=True,
        qk_layernorm=True,
        cache_mla_latents=True,
        apply_rope_fusion=False,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_head_dim=128,
        v_head_dim=128,
        qk_pos_emb_head_dim=64,
        rope_type="rope",
        use_cpu_initialization=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        batch_invariant_mode=True,
        batch_invariant_backend="te_native",
        flash_attention_version=_BIK_FA_VERSION,
        normalization='RMSNorm',
        params_dtype=torch.bfloat16,
        attention_backend=AttnBackend.flash,
        fp32_residual_connection=False,
        nccl_all_reduce_for_prefill=False,
        bf16=True,
    )
    base = (
        GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                multi_latent_attention=True, qk_layernorm=True
            ),
            vocab_size=8192,
            max_sequence_length=4096,
            position_embedding_type='rope',
        )
        .cuda()
        .eval()
    )
    model = Float16Module(base.config, base).cuda().eval()
    tokenizer = DummyTokenizer(vocab_size=8192, eod=8191)
    prompts = {
        i: [(17 * i + j) % 8190 + 1 for j in range(length)]
        for i, length in enumerate((31, 257, 1025, 2049), 1)
    }
    sampling = SamplingParams(
        num_tokens_to_generate=8,
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        add_BOS=False,
        return_log_probs=True,
        termination_id=8191,
    )

    def run(ids, stagger=False):
        context = DynamicInferenceContext(
            model_config=base.config,
            inference_config=InferenceConfig(
                max_sequence_length=4096,
                max_tokens=4096,
                max_requests=8,
                enable_chunked_prefill=False,
                buffer_size_gb=0.125,
                block_size_tokens=64,
                num_cuda_graphs=None,
                materialize_only_last_token_logits=False,
                use_cuda_graphs_for_non_decode_steps=False,
                unified_memory_level=0,
            ),
        )
        controller = TextGenerationController(GPTInferenceWrapper(model, context), tokenizer)
        engine = DynamicInferenceEngine(controller=controller, context=context)
        results = {}
        steps = 0
        with torch.no_grad(), set_batch_invariant_mode(True, backend="te_native"):
            for rid in (ids[:-1] if stagger else ids):
                engine.add_request(rid, prompts[rid], sampling)
            for _ in range(64):
                if not engine.has_unfinished_requests():
                    break
                steps += 1
                if stagger and steps == 2:
                    engine.add_request(ids[-1], prompts[ids[-1]], sampling)
                for req in engine.step_modern()['finished_requests']:
                    results[req.request_id] = {
                        'tokens': req.generated_tokens,
                        'prompt_logprobs': req.prompt_log_probs,
                        'generated_logprobs': req.generated_log_probs,
                    }
            assert not engine.has_unfinished_requests(), 'Requests exceeded step limit'
        assert set(results) == set(ids)
        del engine, controller, context
        gc.collect()
        return results

    full = {}
    for rid in prompts:
        full.update(run([rid]))
    for reverse in (False, True):
        ids = list(reversed(prompts)) if reverse else list(prompts)
        assert run(ids, stagger=True) == full
