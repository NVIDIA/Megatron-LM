# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bitwise cache-hit parity, including chunked prefill, graph replay and refit."""

import gc

import pytest
import torch

from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    set_batch_invariant_mode,
    te_supports_batch_invariant_attention,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.inference.engines.ssm_test_helpers import (
    skip_if_sequence_packing_not_available,
)
from tests.unit_tests.models.test_gpt_model_batch_invariant import (
    HAVE_FA3,
    HAVE_FA4,
    DummyTokenizer,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
@pytest.mark.skipif(
    not te_supports_batch_invariant_attention() or not (HAVE_FA3 or HAVE_FA4),
    reason="Requires batch-invariant TE and FA3/4",
)
# Batch-invariant Mamba layers require the SSM kernels pinned; see assert_ssm_kernels_pinned.
@pytest.mark.usefixtures("pinned_autotune")
class TestBatchInvariantMambaPrefixCaching:
    def setup_method(self):
        skip_if_sequence_packing_not_available()
        Utils.initialize_model_parallel(1, 1)
        self._previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)

    def teardown_method(self):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()
        torch.use_deterministic_algorithms(self._previous_deterministic)

    @staticmethod
    def _run(*, prefix, graphs, chunked):
        model_parallel_cuda_manual_seed(321, inference_rng_tracker=True, force_reset_rng=True)
        torch.manual_seed(321)
        cfg = TransformerConfig(
            num_layers=3,
            hidden_size=128,
            num_attention_heads=4,
            is_hybrid_model=True,
            mamba_num_heads=8,
            mamba_head_dim=32,
            mamba_num_groups=1,
            mamba_state_dim=16,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            batch_invariant_mode=True,
            batch_invariant_backend="te_native",
            flash_attention_version=4 if HAVE_FA4 else 3,
            normalization="RMSNorm",
            params_dtype=torch.bfloat16,
            bf16=True,
            attention_backend=AttnBackend.flash,
            cuda_graph_impl="local" if graphs else "none",
            inference_rng_tracker=True,
        )
        model = (
            HybridModel(
                config=cfg,
                hybrid_stack_spec=hybrid_stack_spec,
                hybrid_layer_pattern="M*-",
                vocab_size=128,
                max_sequence_length=1024,
            )
            .cuda()
            .eval()
        )
        mamba_config = MambaInferenceStateConfig.from_model(model)
        assert mamba_config.ssm_states_dtype == torch.float32
        model = Float16Module(cfg, model).cuda().eval()
        context = DynamicInferenceContext(
            cfg,
            InferenceConfig(
                mamba_inference_state_config=mamba_config,
                prefix_caching_mamba_gb=0.125,
                max_sequence_length=1024,
                buffer_size_gb=0.125,
                block_size_tokens=256,
                max_requests=4,
                max_tokens=512 if chunked else 2048,
                num_cuda_graphs=4 if graphs else None,
                cuda_graph_max_tokens=256,
                materialize_only_last_token_logits=True,
                use_cuda_graphs_for_non_decode_steps=True,
                enable_chunked_prefill=chunked,
                enable_prefix_caching=prefix,
                use_flashinfer_fused_rope=False,
                unified_memory_level=0,
                sampling_backend="torch",
            ),
        )
        engine = DynamicInferenceEngine(
            TextGenerationController(
                GPTInferenceWrapper(model, context), DummyTokenizer(vocab_size=128, eod=127)
            ),
            context,
        )
        prompts = [[i % 120 + 1 for i in range(n)] for n in (600, 650, 512, 513, 768, 769)]
        outputs = {}
        graph_steps = 0
        for wave, order in enumerate(([0, 1], [1, 0], [2], [3], [4], [5], [0])):
            if wave == 6:
                # Old cached states must not be reused after a weight update.
                engine.suspend()
                for parameter in model.parameters():
                    parameter.mul_(1.03)
                engine.resume()
                skipped_before_refit_request = engine._prefill_tokens_skipped
            for i in order:
                engine.add_request(
                    wave * 10 + i + 1,
                    prompts[i],
                    SamplingParams(
                        seed=100 + i,
                        num_tokens_to_generate=8,
                        temperature=1.0,
                        top_k=0,
                        top_p=1.0,
                        return_log_probs=True,
                        skip_prompt_log_probs=True,
                        termination_id=-1,
                    ),
                )
            for _ in range(200):
                if not engine.has_unfinished_requests():
                    break
                result = engine.step_modern()
                graph_steps += int(context.using_cuda_graph_this_step())
                for request in result["finished_requests"]:
                    outputs[request.request_id] = (
                        list(request.generated_tokens),
                        list(request.generated_log_probs),
                    )
            assert not engine.has_unfinished_requests(), "Cached suffix made no scheduling progress"
            if wave == 6:
                assert engine._prefill_tokens_skipped == skipped_before_refit_request
        assert outputs[1] == outputs[11]
        assert outputs[2] == outputs[12]
        if prefix:
            assert engine._prefill_tokens_skipped > 0
        if graphs:
            assert graph_steps > 0
        del engine, context, model
        delete_cuda_graphs()
        gc.collect()
        return outputs

    @pytest.mark.parametrize("graphs", [False, True])
    @pytest.mark.parametrize("chunked", [False, True])
    @torch.inference_mode()
    def test_cache_matches_uncached_eager(self, graphs, chunked, monkeypatch):
        monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
        monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
        monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
        with set_batch_invariant_mode(True, backend="te_native"):
            baseline = self._run(prefix=False, graphs=False, chunked=False)
            cached = self._run(prefix=True, graphs=graphs, chunked=chunked)
        assert cached == baseline
