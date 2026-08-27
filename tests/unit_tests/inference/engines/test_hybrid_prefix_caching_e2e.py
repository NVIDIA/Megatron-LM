# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""End-to-end test for Mamba prefix caching with a real hybrid model.

This test exercises the 4 key indices within a Mamba prefill:

  1. num_mamba_matched — how many blocks have cached Mamba state.
     Determines how many tokens the prefill can skip.

  2. num_kv_matched — how many KV blocks are shared with prior
     requests. Can exceed num_mamba_matched, since KV blocks are
     always registered for every completed block, while Mamba
     state is only cached at divergence and last-aligned blocks.

  3. last_aligned_block — the last full-block boundary in the
     prompt: floor(prompt_len / block_size) * block_size. Mamba
     state is always cached here (if it falls within the
     effective prefill). This is the "end of the known prefix"
     state that future requests can restore from.

  4. end_of_sequence — the actual prompt length. When prompt_len
     is block-aligned (prompt_len == last_aligned), the final
     Mamba state is cached via the EOS path (copy from live
     buffer). When not aligned, there's a gap between
     last_aligned and end_of_sequence that doesn't get cached.

5 requests with overlapping prefixes are processed in a specific
order so that each request sees a different combination of these
indices. The test verifies both internal state (mamba cache
registration, skip counts) and output correctness (generated
tokens match between pc=off and pc=on).
"""

import os
import random
import types

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import (
    AsyncScheduleMode,
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
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils

BLOCK_SIZE = 256
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 2048
NUM_TOKENS_TO_GENERATE = 16
# multi-group uses 4x the requests (20 vs 5), creating larger batch
# composition differences between pc=off and pc=on. reduce decode steps
# to stay within the safe bf16 rounding margin.
MULTI_GROUP_TOKENS_TO_GENERATE = 8
NUM_GROUPS = 4
GROUP_TOKEN_STRIDE = 2000


def skip_if_mamba_sequence_packing_not_available():
    sequence_packing_available, reason = _check_mamba_sequence_packing_support()
    if not sequence_packing_available:
        pytest.skip(reason)


def set_rounder(value):
    DynamicInferenceContext.ROUNDER = value
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


@pytest.mark.internal
@pytest.mark.skipif(not is_fa_min_version("2.7.3"), reason="need flash attn")
class TestMambaPrefixCachingE2E:
    """End-to-end test for Mamba prefix caching with a real hybrid model."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()
        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    def setup_method(self, method):
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def _create_model(self, num_cuda_graphs=None):
        transformer_config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=3,
            hidden_size=256,
            mamba_num_heads=16,
            num_attention_heads=16,
            use_cpu_initialization=True,
            cuda_graph_impl="local" if num_cuda_graphs else "none",
            inference_rng_tracker=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            add_bias_linear=True,
            is_hybrid_model=True,
        )
        model = HybridModel(
            config=transformer_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=VOCAB_SIZE,
            max_sequence_length=MAX_SEQ_LEN,
            parallel_output=True,
            hybrid_layer_pattern="M*-",
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()
        for param in model.parameters():
            param.data = param.data.to(transformer_config.params_dtype)
        model.eval()
        return model

    def _create_prompts(self, offset=0):
        """Build 5 prompts with carefully designed prefix sharing.

        Each prompt uses disjoint token ID ranges for unique segments
        so that parent-chained block hashes differ where content differs.

        The prompts are designed so that each request hits a different
        combination of the 4 key indices:

        req 0 (300 tokens): seed request, no matches. last_aligned=256
        req 1 (800 tokens): 1 KV match, 1 Mamba match (block 0 from req 0)
        req 2 (800 tokens): 2 KV matches, but only 1 Mamba match
        req 3 (800 tokens): 2 KV matches, 2 Mamba matches
        req 4 (1100 tokens): 3 KV matches, 3 Mamba matches
        """
        device = torch.cuda.current_device()
        base = torch.arange(offset, offset + 256, dtype=torch.int64, device=device)
        seg_B = torch.arange(offset + 256, offset + 512, dtype=torch.int64, device=device)
        seg_1rest = torch.arange(offset + 512, offset + 800, dtype=torch.int64, device=device)
        seg_2rest = torch.arange(offset + 800, offset + 1088, dtype=torch.int64, device=device)
        seg_3rest = torch.arange(offset + 1088, offset + 1376, dtype=torch.int64, device=device)
        seg_4ext = torch.arange(offset + 1376, offset + 1708, dtype=torch.int64, device=device)
        extra_0 = torch.arange(offset + 1708, offset + 1752, dtype=torch.int64, device=device)

        prompts = [
            torch.cat([base, extra_0]),  # 300
            torch.cat([base, seg_B, seg_1rest]),  # 800
            torch.cat([base, seg_B, seg_2rest]),  # 800
            torch.cat([base, seg_B, seg_3rest]),  # 800
            torch.cat([base, seg_B, seg_1rest[:256], seg_4ext]),  # 1100
        ]
        assert [len(p) for p in prompts] == [300, 800, 800, 800, 1100]
        return prompts

    def _build_engine(
        self,
        model,
        mamba_config,
        enable_prefix_caching,
        buffer_size_gb=0.5,
        # max_requests is not capped, so it auto-derives from buffer_size_gb. The
        # Mamba cache budget must cover the per-step extraction scratch (which scales
        # with max_requests) on top of the durable cache, so it needs enough headroom.
        prefix_caching_mamba_gb=2.0,
        request_rounder=4,
        num_cuda_graphs=None,
        enable_chunked_prefill=False,
        max_tokens=None,
        max_requests=None,
        async_sched_mode=AsyncScheduleMode.LEGACY,
        prefix_caching_lease_epochs=0,
    ):
        set_rounder(request_rounder)
        inference_config_kwargs = dict(
            max_sequence_length=MAX_SEQ_LEN,
            buffer_size_gb=buffer_size_gb,
            block_size_tokens=BLOCK_SIZE,
            mamba_inference_state_config=mamba_config,
            materialize_only_last_token_logits=async_sched_mode == AsyncScheduleMode.ASYNC,
            enable_prefix_caching=enable_prefix_caching,
            enable_chunked_prefill=enable_chunked_prefill,
            unified_memory_level=0,
            num_cuda_graphs=num_cuda_graphs,
            sampling_backend='torch',
            async_sched_mode=async_sched_mode,
        )
        if max_tokens is not None:
            inference_config_kwargs["max_tokens"] = max_tokens
        if max_requests is not None:
            inference_config_kwargs["max_requests"] = max_requests
        if enable_prefix_caching:
            inference_config_kwargs.update(
                prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
                prefix_caching_mamba_gb=prefix_caching_mamba_gb,
                prefix_caching_lease_epochs=prefix_caching_lease_epochs,
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
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None
        for module in model.modules():
            if isinstance(module, CudaGraphManager):
                module.cudagraph_runners.clear()
                module.custom_cudagraphs_lookup_table.clear()
        return DynamicInferenceEngine(controller, context)

    def _make_request(self, req_id, prompt, enable_pc, num_tokens=NUM_TOKENS_TO_GENERATE):
        return DynamicInferenceRequest(
            request_id=req_id,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(
                num_tokens_to_generate=num_tokens, termination_id=-1, top_k=1
            ),
            block_size_tokens=BLOCK_SIZE if enable_pc else None,
            enable_prefix_caching=enable_pc,
        )

    def _run_simple(
        self,
        model,
        mamba_config,
        prompts,
        enable_pc,
        base_req_id=0,
        num_tokens=NUM_TOKENS_TO_GENERATE,
        **engine_kwargs,
    ):
        """Run all prompts with given pc setting, return (finished_dict, lifetime_prefill)."""
        engine = self._build_engine(
            model, mamba_config, enable_prefix_caching=enable_pc, **engine_kwargs
        )
        for i, prompt in enumerate(prompts):
            engine._add_request(self._make_request(base_req_id + i, prompt, enable_pc, num_tokens))
        finished = {}
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)
        return finished, engine.context.lifetime_prefill_token_count

    def _get_ref_count(self, alloc, block_hash):
        bid = alloc.kv_hash_to_block_id.get(block_hash)
        return 0 if bid is None else alloc.block_ref_counts[bid].item()

    def _assert_step(self, step, reqs_by_group, alloc, step_prefill, num_groups, ctx=None):
        """Shared per-step verification for single-group and multi-group runs."""
        G = num_groups
        if step == 1:
            for g in range(G):
                r = reqs_by_group[g]
                assert r[0]._mamba_num_matched_blocks == 0, f"step 1 group {g}"
                assert r[0].precomputed_block_hashes[0] in ctx.mamba_slot_allocator.hash_to_block_id
            assert len(ctx.mamba_slot_allocator.hash_to_block_id) == G
            assert step_prefill == G * 300, f"step 1: expected {G * 300}, got {step_prefill}"
            if G == 1:
                assert (
                    self._get_ref_count(alloc, reqs_by_group[0][0].precomputed_block_hashes[0]) == 1
                )

        elif step == 2:
            for g in range(G):
                r = reqs_by_group[g]
                assert r[1]._mamba_num_matched_blocks == 1, f"step 2 group {g}"
                assert r[1].precomputed_block_hashes[2] in ctx.mamba_slot_allocator.hash_to_block_id
            assert len(ctx.mamba_slot_allocator.hash_to_block_id) == G * 2
            assert step_prefill == G * 544, f"step 2: expected {G * 544}, got {step_prefill}"
            if G == 1:
                assert (
                    self._get_ref_count(alloc, reqs_by_group[0][0].precomputed_block_hashes[0]) == 2
                )

        elif step == 3:
            for g in range(G):
                r = reqs_by_group[g]
                assert r[2]._mamba_num_matched_blocks == 1, f"step 3 group {g} req 2"
                assert r[4]._mamba_num_matched_blocks == 3, f"step 3 group {g} req 4"
                assert r[2].precomputed_block_hashes[1] in ctx.mamba_slot_allocator.hash_to_block_id
                assert r[2].precomputed_block_hashes[2] in ctx.mamba_slot_allocator.hash_to_block_id
                assert r[4].precomputed_block_hashes[3] in ctx.mamba_slot_allocator.hash_to_block_id
                h0 = r[0].precomputed_block_hashes[0]
                h1 = r[1].precomputed_block_hashes[1]
                assert self._get_ref_count(alloc, h0) == 4, f"step 3 group {g}"
                assert self._get_ref_count(alloc, h1) == 3, f"step 3 group {g}"
            assert len(ctx.mamba_slot_allocator.hash_to_block_id) == G * 5
            assert step_prefill == G * (
                544 + 332
            ), f"step 3: expected {G * 876}, got {step_prefill}"

        elif step == 4:
            for g in range(G):
                r = reqs_by_group[g]
                assert r[3]._mamba_num_matched_blocks == 2, f"step 4 group {g}"
                assert r[3].precomputed_block_hashes[2] in ctx.mamba_slot_allocator.hash_to_block_id
                h0 = r[0].precomputed_block_hashes[0]
                h1 = r[1].precomputed_block_hashes[1]
                assert self._get_ref_count(alloc, h0) == 5, f"step 4 group {g}"
                assert self._get_ref_count(alloc, h1) == 4, f"step 4 group {g}"
            assert len(ctx.mamba_slot_allocator.hash_to_block_id) == G * 6
            assert step_prefill == G * 288, f"step 4: expected {G * 288}, got {step_prefill}"

    def _run_pc_on(self, model, mamba_config, prompts):
        """Run requests with prefix caching enabled, verifying per-step state."""
        engine = self._build_engine(model, mamba_config, enable_prefix_caching=True)
        alloc = engine.context.kv_block_allocator
        ctx = engine.context

        reqs = {i: self._make_request(i, p, True) for i, p in enumerate(prompts)}
        for i in [0, 1, 2, 4]:
            engine._add_request(reqs[i])

        step = 0
        req3_added = False
        finished = {}
        prev_prefill = 0
        reqs_by_group = [{k: reqs[k] for k in reqs}]

        while engine.has_unfinished_requests():
            result = engine.step_modern()
            step += 1
            step_prefill = ctx.lifetime_prefill_token_count - prev_prefill
            prev_prefill = ctx.lifetime_prefill_token_count
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

            if step <= 2 or (step == 3 and not req3_added) or (step == 4 and req3_added):
                self._assert_step(step, reqs_by_group, alloc, step_prefill, 1, ctx)
            if step == 3 and not req3_added:
                engine._add_request(reqs[3])
                req3_added = True

        return finished, ctx.lifetime_prefill_token_count

    def _run_multi_pc_on(self, model, mamba_config, all_prompts, num_cuda_graphs=None):
        """Run 4 groups with prefix caching enabled, verifying per-step state."""
        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            buffer_size_gb=2.0,
            # Large buffer auto-derives many max_requests, so the extraction scratch
            # is large; give the Mamba cache enough budget to cover it plus durable.
            prefix_caching_mamba_gb=4.0,
            num_cuda_graphs=num_cuda_graphs,
        )
        alloc = engine.context.kv_block_allocator
        ctx = engine.context

        reqs = []
        for g, prompts in enumerate(all_prompts):
            group_reqs = {}
            for lid, prompt in enumerate(prompts):
                rid = g * 5 + lid
                group_reqs[lid] = self._make_request(
                    rid, prompt, True, MULTI_GROUP_TOKENS_TO_GENERATE
                )
            reqs.append(group_reqs)

        for g in range(NUM_GROUPS):
            for lid in [0, 1, 2, 4]:
                engine._add_request(reqs[g][lid])

        step = 0
        req3_added = False
        finished = {}
        prev_prefill = 0

        while engine.has_unfinished_requests():
            result = engine.step_modern()
            step += 1
            step_prefill = ctx.lifetime_prefill_token_count - prev_prefill
            prev_prefill = ctx.lifetime_prefill_token_count
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

            if step <= 2 or (step == 3 and not req3_added) or (step == 4 and req3_added):
                self._assert_step(step, reqs, alloc, step_prefill, NUM_GROUPS, ctx)
            if step == 3 and not req3_added:
                for g in range(NUM_GROUPS):
                    engine._add_request(reqs[g][3])
                req3_added = True

        return finished, ctx.lifetime_prefill_token_count

    @torch.inference_mode()
    def test_mamba_prefix_caching_e2e(self):
        """Verify output tokens match between pc=off and pc=on."""
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_prompts()

        off_outputs, off_prefill = self._run_simple(model, mamba_config, prompts, False)
        on_outputs, on_prefill = self._run_pc_on(model, mamba_config, prompts)

        for req_id in range(5):
            assert (
                off_outputs[req_id] == on_outputs[req_id]
            ), f"req {req_id}: pc=off {off_outputs[req_id]} != pc=on {on_outputs[req_id]}"
        assert off_prefill == 3800 and on_prefill == 2008 and on_prefill < off_prefill

    @torch.inference_mode()
    def test_async_sched_mamba_prefix_caching_with_chunked_prefill_e2e(self):
        """Async combined chunking and Mamba prefix caching matches legacy output."""
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_prompts()[:3]

        legacy_outputs, legacy_prefill = self._run_simple(
            model, mamba_config, prompts, enable_pc=False
        )
        async_outputs, async_prefill = self._run_simple(
            model,
            mamba_config,
            prompts,
            enable_pc=True,
            enable_chunked_prefill=True,
            max_tokens=400,
            max_requests=4,
            async_sched_mode=AsyncScheduleMode.ASYNC,
        )

        assert async_outputs == legacy_outputs
        assert async_prefill < legacy_prefill

    @pytest.mark.parametrize("num_cuda_graphs", [None, 2])
    @torch.inference_mode()
    def test_mamba_prefix_caching_multi_group_e2e(self, num_cuda_graphs):
        """Verify multi-group prefix caching with 4 independent groups."""
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model(num_cuda_graphs=num_cuda_graphs)
        mamba_config = MambaInferenceStateConfig.from_model(model)
        all_prompts = [self._create_prompts(g * GROUP_TOKEN_STRIDE) for g in range(NUM_GROUPS)]

        _, off_prefill = self._run_simple(
            model,
            mamba_config,
            [p for group in all_prompts for p in group],
            False,
            num_tokens=MULTI_GROUP_TOKENS_TO_GENERATE,
            num_cuda_graphs=num_cuda_graphs,
            buffer_size_gb=2.0,
            prefix_caching_mamba_gb=0.2,
        )
        on_outputs, on_prefill = self._run_multi_pc_on(
            model, mamba_config, all_prompts, num_cuda_graphs=num_cuda_graphs
        )

        # verify per-group outputs match independent runs
        for g in range(NUM_GROUPS):
            ref_outputs, _ = self._run_simple(
                model,
                mamba_config,
                all_prompts[g],
                True,
                base_req_id=g * 5,
                num_tokens=MULTI_GROUP_TOKENS_TO_GENERATE,
                num_cuda_graphs=num_cuda_graphs,
            )
            for lid in range(5):
                rid = g * 5 + lid
                assert (
                    on_outputs[rid] == ref_outputs[rid]
                ), f"group {g} req {lid}: multi {on_outputs[rid]} != per-group {ref_outputs[rid]}"

        assert off_prefill == NUM_GROUPS * 3800
        assert on_prefill == NUM_GROUPS * 2008 and on_prefill < off_prefill

    def _create_block_aligned_prompts(self):
        """Build 4 prompts with block-aligned lengths for EOS path testing."""
        device = torch.cuda.current_device()
        seg_0 = torch.arange(8000, 8256, dtype=torch.int64, device=device)
        seg_1 = torch.arange(8256, 8512, dtype=torch.int64, device=device)
        prompts = [
            seg_0.clone(),
            seg_0.clone(),
            torch.cat([seg_0, seg_1]),
            torch.cat([seg_0, seg_1]),
        ]
        assert [len(p) for p in prompts] == [256, 256, 512, 512]
        return prompts

    def _run_eos_pc_on(self, model, mamba_config, prompts):
        """Run block-aligned prompts with pc=on, per-step assertions.

        Scheduling with pending_block_hashes coordination:
          - step 1: A scheduled (B, C, D deferred: h0 pending)
          - step 2: B + C co-scheduled (D deferred: h1 pending from C)
          - step 3: D scheduled
        """
        engine = self._build_engine(model, mamba_config, enable_prefix_caching=True)
        alloc = engine.context.kv_block_allocator
        ctx = engine.context

        reqs = {i: self._make_request(i, p, True) for i, p in enumerate(prompts)}
        for i in range(4):
            engine._add_request(reqs[i])

        step = 0
        finished = {}
        prev_prefill = 0

        while engine.has_unfinished_requests():
            result = engine.step_modern()
            step += 1
            step_prefill = ctx.lifetime_prefill_token_count - prev_prefill
            prev_prefill = ctx.lifetime_prefill_token_count
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

            if step == 1:
                assert reqs[0]._mamba_num_matched_blocks == 0, f"step 1"
                assert len(ctx.mamba_slot_allocator.hash_to_block_id) == 1
                assert (
                    reqs[0].precomputed_block_hashes[0] in ctx.mamba_slot_allocator.hash_to_block_id
                )
                assert step_prefill == 256
            elif step == 2:
                # B: 1 mamba match but raw_skip >= chunk_length, back off to 0 blocks, full recompute (256)
                # C: 1 mamba match, skip 256, effective 256
                assert reqs[1]._mamba_num_matched_blocks == 1, f"step 2 B"
                assert reqs[2]._mamba_num_matched_blocks == 1, f"step 2 C"
                assert len(ctx.mamba_slot_allocator.hash_to_block_id) == 2
                assert (
                    reqs[2].precomputed_block_hashes[1] in ctx.mamba_slot_allocator.hash_to_block_id
                )
                assert step_prefill == 512  # B=256 (back-off recompute) + C=256
            elif step == 3:
                # D: 2 mamba matches, raw_skip >= chunk_length, back off to block 0, skip 256, effective 256
                assert reqs[3]._mamba_num_matched_blocks == 2, f"step 3 D"
                assert len(ctx.mamba_slot_allocator.hash_to_block_id) == 2
                assert step_prefill == 256

        return finished, ctx.lifetime_prefill_token_count

    @torch.inference_mode()
    def test_mamba_block_aligned_eos_e2e(self):
        """Verify block-aligned EOS caching and recompute-based back-off."""
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_block_aligned_prompts()

        off_outputs, off_prefill = self._run_simple(model, mamba_config, prompts, False)
        on_outputs, on_prefill = self._run_eos_pc_on(model, mamba_config, prompts)

        for req_id in range(4):
            assert (
                off_outputs[req_id] == on_outputs[req_id]
            ), f"req {req_id}: pc=off {off_outputs[req_id]} != pc=on {on_outputs[req_id]}"
        assert off_prefill == 1536 and on_prefill == 1024 and on_prefill < off_prefill

    def _create_eviction_prompts(self):
        device = torch.cuda.current_device()
        return [
            torch.arange(8000, 8300, dtype=torch.int64, device=device),
            torch.arange(8300, 8600, dtype=torch.int64, device=device),
            torch.arange(8000, 8300, dtype=torch.int64, device=device),  # identical to E
        ]

    @torch.inference_mode()
    def test_mamba_lru_eviction_e2e(self):
        """Verify KV eviction invalidates mamba state via invalidate_mamba_state_for_block."""
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_eviction_prompts()

        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            buffer_size_gb=0.002,
            prefix_caching_mamba_gb=0.05,
            request_rounder=1,
        )
        alloc = engine.context.kv_block_allocator
        ctx = engine.context

        assert alloc.pool_size == 3, f"expected 3 total blocks, got {alloc.pool_size}"
        assert ctx.max_requests >= 1

        finished = {}

        def _run_one(req_id, prompt):
            # Use num_tokens_to_generate=2 so the request survives the prefill
            # step (commit_mamba_intermediate_states runs after update_requests)
            req = self._make_request(req_id, prompt, True, num_tokens=2)
            engine._add_request(req)
            while engine.has_unfinished_requests():
                result = engine.step_modern()
                for record in result["finished_request_records"]:
                    merged = record.merge()
                    finished[merged.request_id] = list(merged.generated_tokens)
            return req

        # E: seed request
        req_E = _run_one(0, prompts[0])
        h_E0 = req_E.precomputed_block_hashes[0]
        assert (
            h_E0 in ctx.mamba_slot_allocator.hash_to_block_id and h_E0 in alloc.kv_hash_to_block_id
        )
        assert len(ctx.mamba_slot_allocator.hash_to_block_id) == 1 and alloc.pool_avail == 1

        # F: disjoint prefix, forces eviction of E's cached block
        req_F = _run_one(1, prompts[1])
        assert req_F.precomputed_block_hashes[0] in ctx.mamba_slot_allocator.hash_to_block_id
        assert (
            h_E0 not in alloc.kv_hash_to_block_id
            and h_E0 not in ctx.mamba_slot_allocator.hash_to_block_id
        )
        assert len(ctx.mamba_slot_allocator.hash_to_block_id) == 1

        # G: identical to E, but E's state was evicted
        req_G = _run_one(2, prompts[2])
        assert req_G._mamba_num_matched_blocks == 0
        assert h_E0 in ctx.mamba_slot_allocator.hash_to_block_id
        assert finished[0] == finished[2]

    @torch.inference_mode()
    def test_mamba_chunked_prefill_unaligned_boundary_snapshot(self):
        """Chunked prefill snapshots Mamba state at the last block boundary.

        ``compute_and_store_offsets`` records a Mamba state snapshot at a KV-block
        boundary only when that boundary is a whole multiple of the SSM chunk size
        measured from the start of the current prefill chunk. Because the chunk
        start equals ``finished_chunk_token_count`` on continuation chunks, this
        holds exactly when every chunk boundary is block-aligned.

        Here ``max_tokens`` (300) is intentionally not a multiple of the block size
        (256), so the request spans several chunks and its last full-block boundary
        (token 768) lands in a continuation chunk. The scheduler keeps each chunk
        boundary block-aligned, so the final chunk begins at token 512 and the
        token-768 snapshot is extracted and committed. A second request sharing the
        768-token prefix then restores that state and skips those blocks.
        """
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)

        device = torch.cuda.current_device()
        # 800-token prompt -> 3 full blocks (256/512/768) + a 32-token tail.
        # The last full-block boundary (768) falls in the final continuation chunk.
        prompt = torch.arange(9000, 9800, dtype=torch.int64, device=device)
        assert len(prompt) == 800

        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_tokens=300,  # not a multiple of BLOCK_SIZE (256) -> forces unaligned cuts
            max_requests=4,
            request_rounder=4,
        )
        ctx = engine.context
        # Sanity: the prompt genuinely spans multiple prefill chunks.
        assert ctx.max_tokens < len(prompt)

        # --- Seed request: fills the cache, no prior matches. ---
        seed = self._make_request(0, prompt, enable_pc=True, num_tokens=4)
        engine._add_request(seed)
        while engine.has_unfinished_requests():
            engine.step_modern()
        # Seed has no prior cache, so no Mamba blocks are matched during its prefill.
        assert seed._mamba_num_matched_blocks == 0

        # block index 2 == the boundary at token 768 (768 // 256 - 1). The final
        # chunk begins block-aligned at token 512, so (768 - 512) % 128 == 0 and
        # the state at this boundary is extracted and committed.
        assert len(seed.precomputed_block_hashes) == 3
        last_block_hash = seed.precomputed_block_hashes[2]
        assert (
            last_block_hash in ctx.mamba_slot_allocator.hash_to_block_id
        ), "Mamba snapshot at the last block boundary (token 768) was not recorded."

        # --- Reuse request: shares the full 768-token prefix, should restore the
        # cached Mamba state and skip those blocks entirely. ---
        reuse_prompt = torch.cat(
            [prompt[:768], torch.arange(9800, 9900, dtype=torch.int64, device=device)]
        )
        reuse = self._make_request(1, reuse_prompt, enable_pc=True, num_tokens=4)
        engine._add_request(reuse)
        while engine.has_unfinished_requests():
            engine.step_modern()

        assert reuse._mamba_num_matched_blocks == 3, (
            "Reuse request should restore Mamba state from the token-768 snapshot "
            f"(3 matched blocks), got {reuse._mamba_num_matched_blocks}."
        )

    # ------------------------------------------------------------------
    # Bounded staleness (lease) eviction
    # ------------------------------------------------------------------

    def _assert_cache_consistent(self, engine):
        """Structural invariants the prefix cache must hold at any quiescent point.

        Checked after every epoch change in the lease tests, since expiry is the
        one path that tears entries down out of LRU order.
        """
        alloc = engine.context.kv_block_allocator
        cached_ids = set(alloc.kv_hash_to_block_id.values())

        # 1. Every cached block has its parent cached. _find_kv_match_count walks
        #    back from the longest match and assumes every ancestor resolves.
        for block_hash, block_id in alloc.kv_hash_to_block_id.items():
            parent_id = alloc.block_parent_id[block_id].item()
            if parent_id >= 0:
                assert parent_id in cached_ids, (
                    f"dangling child: block {block_id} (hash {block_hash}) has "
                    f"parent {parent_id}, which is no longer cached"
                )

        # 2. Hash map and per-block hashes agree in both directions.
        for block_hash, block_id in alloc.kv_hash_to_block_id.items():
            assert alloc.block_hashes[block_id].item() == block_hash

        # 3. Mamba state is only kept for blocks the KV cache still knows about,
        #    which is what keeps expiry from resurrecting stale SSM/conv state.
        mamba_map = engine.context.mamba_slot_allocator.hash_to_block_id
        assert set(mamba_map) <= set(alloc.kv_hash_to_block_id), (
            "mamba state cached for hashes the KV allocator has dropped: "
            f"{set(mamba_map) - set(alloc.kv_hash_to_block_id)}"
        )

        # 4. No block leaked or was double-freed: the free region holds distinct
        #    ids, and no free block is still registered.
        free_region = alloc.block_bag[: alloc.pool_avail].tolist()
        assert len(set(free_region)) == len(free_region), "duplicate id in the free pool"
        assert not (set(free_region) & cached_ids), "a cached block is also in the free pool"

        # 5. Lease stamps track registration exactly.
        registered = (alloc.block_lease_epoch >= 0).nonzero(as_tuple=True)[0].tolist()
        assert set(registered) == cached_ids, (
            f"lease stamps {sorted(set(registered))} disagree with the hash map "
            f"{sorted(cached_ids)}"
        )

    def _drain(self, engine, finished):
        """Step until the engine is idle, collecting generated tokens."""
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

    @torch.inference_mode()
    def test_prefix_cache_lease_expiry_e2e(self):
        """A one-epoch lease drops KV and Mamba state at the next weight update.

        The repeat of an identical prompt must then fully recompute rather than
        restore stale state, and must still generate the same tokens.
        """
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_eviction_prompts()

        engine = self._build_engine(
            model, mamba_config, enable_prefix_caching=True, prefix_caching_lease_epochs=1
        )
        ctx = engine.context
        alloc = ctx.kv_block_allocator
        pool_avail_empty = alloc.pool_avail
        finished = {}

        # Seed: caches both KV blocks and Mamba state under epoch 0.
        req_seed = self._make_request(0, prompts[0], True, num_tokens=2)
        engine._add_request(req_seed)
        self._drain(engine, finished)
        seed_hash = req_seed.precomputed_block_hashes[0]
        assert seed_hash in alloc.kv_hash_to_block_id
        assert seed_hash in ctx.mamba_slot_allocator.hash_to_block_id
        assert alloc.get_block_remaining_lease(alloc.kv_hash_to_block_id[seed_hash]) == 1
        self._assert_cache_consistent(engine)

        # New weights land: the one-epoch lease is up, so everything cached under
        # epoch 0 goes, KV and Mamba alike, and the memory comes back.
        num_expired = ctx.set_prefix_cache_epoch(1)
        assert num_expired > 0
        assert alloc.kv_hash_to_block_id == {}
        assert ctx.mamba_slot_allocator.hash_to_block_id == {}
        assert alloc.pool_avail == pool_avail_empty, "expired blocks did not return to the pool"
        self._assert_cache_consistent(engine)

        # The identical prompt cannot match anything, so it recomputes from
        # scratch -- and still produces the same tokens as the seed run.
        req_repeat = self._make_request(1, prompts[2], True, num_tokens=2)
        engine._add_request(req_repeat)
        self._drain(engine, finished)
        assert req_repeat._mamba_num_matched_blocks == 0, "restored state from an expired lease"
        assert finished[0] == finished[1]
        # It re-registers under the current epoch, starting a fresh lease.
        repeat_hash = req_repeat.precomputed_block_hashes[0]
        assert alloc.block_lease_epoch[alloc.kv_hash_to_block_id[repeat_hash]].item() == 1

    @torch.inference_mode()
    def test_prefix_cache_lease_survives_until_it_expires_e2e(self):
        """The lease is bounded staleness, not no staleness: entries stay usable
        for the configured number of epochs and are dropped only past that."""
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_eviction_prompts()

        engine = self._build_engine(
            model, mamba_config, enable_prefix_caching=True, prefix_caching_lease_epochs=3
        )
        ctx = engine.context
        alloc = ctx.kv_block_allocator
        finished = {}

        req_seed = self._make_request(0, prompts[0], True, num_tokens=2)
        engine._add_request(req_seed)
        self._drain(engine, finished)
        seed_hash = req_seed.precomputed_block_hashes[0]
        seed_block = alloc.kv_hash_to_block_id[seed_hash]

        # Two weight updates inside a three-epoch lease change nothing.
        for epoch in (1, 2):
            assert ctx.set_prefix_cache_epoch(epoch) == 0
            assert seed_hash in alloc.kv_hash_to_block_id
            assert seed_hash in ctx.mamba_slot_allocator.hash_to_block_id
            assert alloc.get_block_remaining_lease(seed_block) == 3 - epoch
            self._assert_cache_consistent(engine)

        # Still matchable two epochs in: the repeat restores cached Mamba state
        # rather than recomputing it. Its tokens are deliberately not compared
        # against the seed's -- the seed computed that state, this one restored
        # it, and the two are not bit-identical in bf16.
        req_hit = self._make_request(1, prompts[2], True, num_tokens=2)
        engine._add_request(req_hit)
        self._drain(engine, finished)
        assert req_hit._mamba_num_matched_blocks > 0, "a live lease failed to match"

        # Third epoch: the seed's lease is up.
        assert ctx.set_prefix_cache_epoch(3) > 0
        assert seed_hash not in alloc.kv_hash_to_block_id
        self._assert_cache_consistent(engine)

    @torch.inference_mode()
    def test_prefix_cache_lease_expiry_with_requests_in_flight_e2e(self):
        """An epoch can land mid-decode, while requests still hold cached blocks.

        Those blocks cannot go back to the free pool yet, so they are unregistered
        in place: unmatchable at once, reclaimed when their owner finishes. The
        in-flight requests must be unaffected and the pool must come back whole.
        """
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_prompts()

        num_tokens = MULTI_GROUP_TOKENS_TO_GENERATE

        engine = self._build_engine(
            model, mamba_config, enable_prefix_caching=True, prefix_caching_lease_epochs=1
        )
        ctx = engine.context
        alloc = ctx.kv_block_allocator
        pool_avail_empty = alloc.pool_avail
        finished = {}

        for i, prompt in enumerate(prompts):
            engine._add_request(self._make_request(i, prompt, True, num_tokens=num_tokens))

        # Bump the epoch every other step, so expiry repeatedly fires while
        # requests are mid-flight and holding registered blocks.
        epoch = 0
        step = 0
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)
            step += 1
            if step % 2 == 0:
                epoch += 1
                ctx.set_prefix_cache_epoch(epoch)
                self._assert_cache_consistent(engine)

        assert epoch > 0, "the run was too short to exercise a mid-flight epoch change"
        # Expiry under them did not disturb the in-flight requests: all of them
        # finished, with a full generation each.
        assert set(finished) == set(range(len(prompts)))
        for req_id in range(len(prompts)):
            assert (
                len(finished[req_id]) == num_tokens
            ), f"req {req_id} produced {len(finished[req_id])} tokens"

        # Blocks pinned by in-flight requests at expiry time were unregistered in
        # place; once their owners finished they must all be back in the pool.
        ctx.set_prefix_cache_epoch(epoch + 1)
        assert alloc.kv_hash_to_block_id == {}
        assert ctx.mamba_slot_allocator.hash_to_block_id == {}
        assert alloc.pool_avail == pool_avail_empty, "blocks leaked across lease expiry"

    @pytest.mark.parametrize("lease_epochs", [1, 2, 5])
    @torch.inference_mode()
    def test_prefix_cache_lease_stress_e2e(self, lease_epochs):
        """Stress the lease across many epochs of overlapping traffic.

        Each round admits a batch of prefix-sharing prompts and then advances the
        epoch, so registration and expiry interleave continuously and blocks are
        recycled through the pool many times over.

        The assertions here are the deterministic ones: cache structure, the
        lease bound, and pool accounting. Generated tokens are deliberately not
        compared against a prefix-caching-off run -- restoring cached Mamba state
        is not bit-identical to recomputing it in bf16, and greedy sampling turns
        that into a different token, which is why the multi-group test above
        compares pc=on against pc=on rather than against pc=off. Output
        correctness across an expiry is covered by
        test_prefix_cache_lease_expiry_e2e, where both sides recompute from an
        empty cache and the comparison is exact.
        """
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)

        num_rounds = 6
        # Disjoint token ranges per round, plus a repeat of an earlier round's
        # prompts, so each round mixes fresh prefixes with ones whose cache entry
        # is either still leased or long expired.
        round_prompts = [self._create_prompts(offset=r * GROUP_TOKEN_STRIDE) for r in range(3)]
        schedule = [round_prompts[r % 3] for r in range(num_rounds)]

        num_tokens = MULTI_GROUP_TOKENS_TO_GENERATE

        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            prefix_caching_lease_epochs=lease_epochs,
        )
        ctx = engine.context
        alloc = ctx.kv_block_allocator
        pool_avail_empty = alloc.pool_avail

        for round_idx, prompts in enumerate(schedule):
            finished = {}
            for i, prompt in enumerate(prompts):
                engine._add_request(
                    self._make_request(1000 * round_idx + i, prompt, True, num_tokens=num_tokens)
                )
            self._drain(engine, finished)

            # Every request completed and produced its full generation.
            for i in range(len(prompts)):
                req_id = 1000 * round_idx + i
                assert (
                    len(finished[req_id]) == num_tokens
                ), f"round {round_idx} req {i} produced {len(finished[req_id])} tokens"

            self._assert_cache_consistent(engine)

            # New weights: nothing older than the lease may survive.
            ctx.set_prefix_cache_epoch(round_idx + 1)
            self._assert_cache_consistent(engine)
            oldest_allowed = ctx.prefix_cache_epoch - lease_epochs
            for block_id in alloc.kv_hash_to_block_id.values():
                assert alloc.block_lease_epoch[block_id].item() > oldest_allowed, (
                    f"round {round_idx}: block {block_id} outlived its {lease_epochs}-epoch "
                    f"lease (stamped {alloc.block_lease_epoch[block_id].item()}, "
                    f"epoch {ctx.prefix_cache_epoch})"
                )

        # Draining the cache past the lease must return every block.
        ctx.set_prefix_cache_epoch(ctx.prefix_cache_epoch + lease_epochs)
        assert alloc.kv_hash_to_block_id == {}
        assert ctx.mamba_slot_allocator.hash_to_block_id == {}
        assert alloc.pool_avail == pool_avail_empty, "blocks leaked over the stress run"

    @torch.inference_mode()
    def test_prefix_cache_lease_expiry_mid_chunked_prefill_e2e(self):
        """A lease expiring between chunks of one request must not orphan a chain.

        Chunked prefill registers a request's blocks in several batches. When an
        epoch lands between them, the earlier batch is unregistered in place (the
        request still holds it), so the later batch arrives parented to a block
        that is no longer cached. Registering it anyway would leave hashes that a
        future request matches but cannot resolve back to the root, which is a
        hard failure inside the match walk rather than a silent miss.
        """
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)

        device = torch.cuda.current_device()
        # 800 tokens -> 3 full blocks plus a tail, spread over several chunks.
        long_prompt = torch.arange(6000, 6800, dtype=torch.int64, device=device)
        # Shares the first 768 tokens, so it probes whatever the first request
        # left in the cache.
        shared_prefix = torch.cat(
            [long_prompt[:768], torch.arange(6900, 6950, dtype=torch.int64, device=device)]
        )
        num_tokens = MULTI_GROUP_TOKENS_TO_GENERATE

        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_tokens=300,
            max_requests=4,
            prefix_caching_lease_epochs=1,
        )
        ctx = engine.context
        finished = {}

        # Drive the first request one step at a time, advancing the epoch after
        # the opening chunk so expiry lands between its registration batches.
        engine._add_request(self._make_request(0, long_prompt, True, num_tokens=num_tokens))
        step = 0
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)
            step += 1
            if step == 1:
                ctx.set_prefix_cache_epoch(1)
                self._assert_cache_consistent(engine)

        # The second request walks the cache the first one left behind. If any
        # orphaned chain survived, resolving the match raises here -- reaching the
        # end of this call at all is the assertion. Whether it restores or
        # recomputes depends on what survived expiry, so its tokens are not
        # compared against a no-cache run.
        engine._add_request(self._make_request(1, shared_prefix, True, num_tokens=num_tokens))
        self._drain(engine, finished)
        self._assert_cache_consistent(engine)

        assert set(finished) == {0, 1}
        for req_id in (0, 1):
            assert (
                len(finished[req_id]) == num_tokens
            ), f"req {req_id} produced {len(finished[req_id])} tokens"
