# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Parameterized test for prefix caching with CUDA graphs.

Verifies prefix caching correctness and CUDA graph usage across every
combination of model type (transformer, hybrid) and batch structure
(prefill, decode, mixed).

For each case, the same prefix-sharing scenario is run twice (with and
without CUDA graphs) and compared:
  1. Generated tokens match exactly (correctness).
  2. context.using_cuda_graph_this_step() returned True at expected steps.
"""

import os
import random
import types

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import (
    InferenceConfig,
    MambaInferenceStateConfig,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils

BLOCK_SIZE = 256
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 2048
NUM_TOKENS_TO_GENERATE = 8


def set_rounder(value):
    DynamicInferenceContext.ROUNDER = value
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


@pytest.mark.internal
@pytest.mark.skipif(not is_fa_min_version("2.7.3"), reason="need flash attn")
class TestPrefixCachingCudaGraphs:
    """Verify prefix caching + CUDA graph interaction across model types and batch structures."""

    def setup_method(self, method):
        Utils.initialize_model_parallel()
        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_model(self, model_type, num_cuda_graphs=None):
        """Create a model with optional CUDA graph support.

        Returns (model, mamba_config_or_none).
        """
        cuda_graph_impl = "local" if num_cuda_graphs else "none"

        if model_type == "transformer":
            config = TransformerConfig(
                params_dtype=torch.bfloat16,
                num_layers=4,
                hidden_size=32,
                num_attention_heads=4,
                use_cpu_initialization=True,
                cuda_graph_impl=cuda_graph_impl,
                inference_rng_tracker=True,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                pipeline_dtype=torch.bfloat16,
                add_bias_linear=True,
            )
            model = GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_layer_local_spec(),
                vocab_size=VOCAB_SIZE,
                max_sequence_length=MAX_SEQ_LEN,
                parallel_output=True,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            ).cuda()
            mamba_config = None
        else:  # hybrid
            config = TransformerConfig(
                params_dtype=torch.bfloat16,
                num_layers=3,
                hidden_size=256,
                mamba_num_heads=16,
                num_attention_heads=16,
                use_cpu_initialization=True,
                cuda_graph_impl=cuda_graph_impl,
                inference_rng_tracker=True,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                pipeline_dtype=torch.bfloat16,
                add_bias_linear=True,
                is_hybrid_model=True,
            )
            model = MambaModel(
                config=config,
                mamba_stack_spec=mamba_stack_spec,
                vocab_size=VOCAB_SIZE,
                max_sequence_length=MAX_SEQ_LEN,
                parallel_output=True,
                hybrid_layer_pattern="M*-",
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
            ).cuda()
            mamba_config = MambaInferenceStateConfig.from_model(model)

        for param in model.parameters():
            param.data = param.data.to(config.params_dtype)
        model.eval()
        return model, mamba_config

    def _reset_cuda_graph_state(self, model):
        """Reset all CUDA graph global and per-module state."""
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None
        for module in model.modules():
            if isinstance(module, CudaGraphManager):
                module.cudagraph_runners.clear()
                module.inference_cudagraphs_lookup_table.clear()

    def _build_engine(self, model, mamba_config, num_cuda_graphs):
        """Build an engine with prefix caching and optional CUDA graphs."""
        set_rounder(4)
        inference_config_kwargs = dict(
            max_sequence_length=MAX_SEQ_LEN,
            buffer_size_gb=0.5,
            block_size_tokens=BLOCK_SIZE,
            materialize_only_last_token_logits=False,
            enable_prefix_caching=True,
            prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
            unified_memory_level=0,
            num_cuda_graphs=num_cuda_graphs,
            use_cuda_graphs_for_non_decode_steps=True,
        )
        if mamba_config is not None:
            inference_config_kwargs.update(
                mamba_inference_state_config=mamba_config, prefix_caching_mamba_gb=0.05
            )
        context = DynamicInferenceContext(
            model_config=model.config, inference_config=InferenceConfig(**inference_config_kwargs)
        )
        wrapper = GPTInferenceWrapper(model, context)
        wrapper.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )
        controller = TextGenerationController(
            inference_wrapped_model=wrapper,
            tokenizer=types.SimpleNamespace(
                vocab_size=VOCAB_SIZE, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )
        self._reset_cuda_graph_state(model)
        return DynamicInferenceEngine(controller, context)

    def _create_prompts(self):
        """Build 4 prompts with overlapping prefixes.

        req 0: tokens[0:300] — seed, no prefix match
        req 1: tokens[0:300] — identical to req 0, full prefix match (1 block)
        req 2: tokens[0:256] + unique[256:500] — partial match, 1 block
        req 3: tokens[0:256] + unique[256:600] — partial match, 1 block shared + 1 unique
        """
        device = torch.cuda.current_device()
        base = torch.arange(0, 256, dtype=torch.int64, device=device)
        extra = torch.arange(256, 300, dtype=torch.int64, device=device)
        unique_2 = torch.arange(1000, 1244, dtype=torch.int64, device=device)
        unique_3 = torch.arange(2000, 2344, dtype=torch.int64, device=device)
        return [
            torch.cat([base, extra]),  # req 0: 300
            torch.cat([base, extra]),  # req 1: 300
            torch.cat([base, unique_2]),  # req 2: 500
            torch.cat([base, unique_3]),  # req 3: 600
        ]

    def _make_request(self, req_id, prompt):
        return DynamicInferenceRequest(
            request_id=req_id,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(
                num_tokens_to_generate=NUM_TOKENS_TO_GENERATE, termination_id=-1, top_k=1
            ),
            block_size_tokens=BLOCK_SIZE,
            enable_prefix_caching=True,
        )

    def _run_scenario(self, engine, batch_structure, prompts):
        """Run the prefix-sharing scenario with the given batch structure.

        Returns (outputs, step_log) where step_log is a list of
        (prefill_req_count, decode_req_count, using_cuda_graph) per step.
        """
        ctx = engine.context
        finished = {}
        step_log = []

        def _step_and_log():
            result = engine.step_modern()
            step_log.append(
                (
                    ctx.batch_dimensions.prefill_req_count,
                    ctx.batch_dimensions.decode_req_count,
                    ctx.using_cuda_graph_this_step(),
                )
            )
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

        if batch_structure in ("prefill", "decode"):
            # Add all 4 requests before first step.
            for i, prompt in enumerate(prompts):
                engine._add_request(self._make_request(i, prompt))
            while engine.has_unfinished_requests():
                _step_and_log()

        elif batch_structure == "mixed":
            # Add req 0+1 first, step until both are decoding, then add 2+3.
            for i in [0, 1]:
                engine._add_request(self._make_request(i, prompts[i]))
            reqs_added = False
            while engine.has_unfinished_requests():
                _step_and_log()
                if not reqs_added:
                    last_p, last_d, _ = step_log[-1]
                    if last_p == 0 and last_d > 0:
                        for i in [2, 3]:
                            engine._add_request(self._make_request(i, prompts[i]))
                        reqs_added = True

        return finished, step_log

    @pytest.mark.parametrize("model_type", ["transformer", "hybrid"])
    @pytest.mark.parametrize("batch_structure", ["prefill", "decode", "mixed"])
    @torch.inference_mode()
    def test_prefix_caching_cuda_graphs(self, model_type, batch_structure):
        """Verify correctness and CUDA graph usage for prefix caching."""
        if model_type == "hybrid":
            sequence_packing_available, reason = _check_mamba_sequence_packing_support()
            if not sequence_packing_available:
                pytest.skip(reason)

        # Create model with CUDA graph support (cuda_graph_impl="local").
        model, mamba_config = self._create_model(model_type, num_cuda_graphs=2)
        prompts = self._create_prompts()

        # Baseline: no CUDA graphs.
        baseline_engine = self._build_engine(model, mamba_config, num_cuda_graphs=None)
        baseline_outputs, _ = self._run_scenario(baseline_engine, batch_structure, prompts)

        # CG enabled.
        cg_engine = self._build_engine(model, mamba_config, num_cuda_graphs=2)
        cg_outputs, step_log = self._run_scenario(cg_engine, batch_structure, prompts)

        # 1. Correctness: generated tokens must match.
        for req_id in range(4):
            assert baseline_outputs[req_id] == cg_outputs[req_id], (
                f"req {req_id}: baseline {baseline_outputs[req_id]} != " f"cg {cg_outputs[req_id]}"
            )

        # 2. CUDA graph usage at expected batch types.
        if batch_structure == "prefill":
            assert any(
                p > 0 and d == 0 and cg for p, d, cg in step_log
            ), f"no prefill-only CG step found in {step_log}"

        elif batch_structure == "decode":
            decode_only = [(p, d, cg) for p, d, cg in step_log if p == 0 and d > 0]
            assert decode_only, f"no decode-only steps found in {step_log}"
            assert all(
                cg for _, _, cg in decode_only
            ), f"not all decode-only steps used CG: {decode_only}"

        elif batch_structure == "mixed":
            assert any(
                p > 0 and d > 0 and cg for p, d, cg in step_log
            ), f"no mixed CG step found in {step_log}"


@pytest.mark.internal
@pytest.mark.skipif(not is_fa_min_version("2.7.3"), reason="need flash attn")
class TestHybridChunkedPrefillIntermediateState:
    """Verify hybrid chunked prefill with concurrent Mamba state extraction and restoration.

    Scenario: one request is mid-chunk (Mamba intermediate state being extracted during
    forward pass) while another request has its Mamba state restored from the prefix cache.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        set_rounder(64)
        Utils.destroy_model_parallel()

    def _create_hybrid_model(self, num_cuda_graphs=None):
        """Create a hybrid (Mamba + attention) model."""
        cuda_graph_impl = "local" if num_cuda_graphs else "none"
        config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=3,
            hidden_size=256,
            mamba_num_heads=16,
            num_attention_heads=16,
            use_cpu_initialization=True,
            cuda_graph_impl=cuda_graph_impl,
            inference_rng_tracker=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            add_bias_linear=True,
            is_hybrid_model=True,
        )
        model = MambaModel(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=VOCAB_SIZE,
            max_sequence_length=MAX_SEQ_LEN,
            parallel_output=True,
            hybrid_layer_pattern="M*-",
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()
        for param in model.parameters():
            param.data = param.data.to(config.params_dtype)
        model.eval()
        return model

    def _reset_cuda_graph_state(self, model):
        """Reset all CUDA graph global and per-module state."""
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None
        for module in model.modules():
            if isinstance(module, CudaGraphManager):
                module.cudagraph_runners.clear()
                module.inference_cudagraphs_lookup_table.clear()

    def _build_engine(
        self,
        model,
        mamba_config,
        enable_prefix_caching,
        enable_chunked_prefill,
        max_tokens=None,
        num_cuda_graphs=None,
    ):
        """Build an engine with the given prefix caching / chunked prefill config."""
        set_rounder(4)
        inference_config_kwargs = dict(
            max_sequence_length=MAX_SEQ_LEN,
            buffer_size_gb=0.5,
            block_size_tokens=BLOCK_SIZE,
            mamba_inference_state_config=mamba_config,
            materialize_only_last_token_logits=False,
            unified_memory_level=0,
            num_cuda_graphs=num_cuda_graphs,
            use_cuda_graphs_for_non_decode_steps=True,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=enable_chunked_prefill,
            max_requests=128,
        )
        if enable_prefix_caching:
            inference_config_kwargs.update(
                prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
                prefix_caching_mamba_gb=0.05,
            )
        if max_tokens is not None:
            inference_config_kwargs["max_tokens"] = max_tokens
        context = DynamicInferenceContext(
            model_config=model.config, inference_config=InferenceConfig(**inference_config_kwargs)
        )
        wrapper = GPTInferenceWrapper(model, context)
        wrapper.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )
        controller = TextGenerationController(
            inference_wrapped_model=wrapper,
            tokenizer=types.SimpleNamespace(
                vocab_size=VOCAB_SIZE, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )
        self._reset_cuda_graph_state(model)
        return DynamicInferenceEngine(controller, context)

    def _make_request(self, req_id, prompt, enable_pc):
        return DynamicInferenceRequest(
            request_id=req_id,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(
                num_tokens_to_generate=NUM_TOKENS_TO_GENERATE, termination_id=-1, top_k=1
            ),
            block_size_tokens=BLOCK_SIZE if enable_pc else None,
            enable_prefix_caching=enable_pc,
        )

    @torch.inference_mode()
    def test_hybrid_chunked_prefill_intermediate_state(self):
        """Concurrent Mamba state extraction (mid-chunk) and restoration (prefix-cached).

        req0 (300 tokens): seeds the cache with 1 block of Mamba state.
        req1 (800 tokens): 256 shared prefix, 544 unique. With max_tokens=400 and 1
            Mamba match (skip 256, effective=544), this request is chunked across steps.
        req2 (300 tokens): identical to req0. Full prefix match, 1 Mamba match.

        In the critical step, req2 has Mamba state restored from cache while req1 has
        Mamba state being computed fresh with intermediate state extraction.
        """
        sequence_packing_available, reason = _check_mamba_sequence_packing_support()
        if not sequence_packing_available:
            pytest.skip(reason)

        # Clear NVTE env vars set by conftest set_env fixture.
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

        model = self._create_hybrid_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)

        device = torch.cuda.current_device()
        prompt0 = torch.arange(0, 300, dtype=torch.int64, device=device)
        prompt1 = torch.cat(
            [
                torch.arange(0, 256, dtype=torch.int64, device=device),
                torch.arange(5000, 5544, dtype=torch.int64, device=device),
            ]
        )  # 800 tokens: 256 shared + 544 unique
        prompt2 = torch.arange(0, 300, dtype=torch.int64, device=device)  # identical to prompt0

        # Baseline: no prefix caching, no chunked prefill.
        baseline_engine = self._build_engine(
            model, mamba_config, enable_prefix_caching=False, enable_chunked_prefill=False
        )
        for i, prompt in enumerate([prompt0, prompt1, prompt2]):
            baseline_engine._add_request(self._make_request(i, prompt, enable_pc=False))
        baseline_outputs = {}
        while baseline_engine.has_unfinished_requests():
            result = baseline_engine.step_modern()
            for record in result["finished_request_records"]:
                merged = record.merge()
                baseline_outputs[merged.request_id] = list(merged.generated_tokens)

        # Test: prefix caching + chunked prefill, max_tokens=400.
        test_engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_tokens=400,
        )
        ctx = test_engine.context

        test_outputs = {}

        def collect_finished(result):
            for record in result["finished_request_records"]:
                merged = record.merge()
                test_outputs[merged.request_id] = list(merged.generated_tokens)

        # Phase 1: run req0 to completion (seeds cache).
        req0 = self._make_request(0, prompt0, enable_pc=True)
        test_engine._add_request(req0)
        while test_engine.has_unfinished_requests():
            collect_finished(test_engine.step_modern())

        # Verify req0 cached its Mamba state.
        assert (
            len(ctx.mamba_slot_allocator.hash_to_block_id) > 0
        ), "req0 should have cached Mamba state"

        # Phase 2: add req1 + req2 simultaneously.
        req1 = self._make_request(1, prompt1, enable_pc=True)
        req2 = self._make_request(2, prompt2, enable_pc=True)
        test_engine._add_request(req1)
        test_engine._add_request(req2)

        while test_engine.has_unfinished_requests():
            collect_finished(test_engine.step_modern())

        # Verify Mamba state was restored for req2 (prefix-cached).
        assert (
            req2._mamba_num_matched_blocks == 1
        ), f"req2 should have 1 Mamba match, got {req2._mamba_num_matched_blocks}"

        # Verify prefix caching saved prefill tokens.
        assert ctx.lifetime_prefill_token_count < (300 + 800 + 300), (
            f"prefix caching should reduce total prefill tokens, "
            f"got {ctx.lifetime_prefill_token_count}"
        )

        # Correctness: generated tokens must match baseline.
        for req_id in range(3):
            assert baseline_outputs[req_id] == test_outputs[req_id], (
                f"req {req_id}: baseline {baseline_outputs[req_id]} != "
                f"test {test_outputs[req_id]}"
            )
