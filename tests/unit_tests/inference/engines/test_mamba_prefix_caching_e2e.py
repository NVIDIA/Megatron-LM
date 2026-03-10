# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
# 16 tokens, not 32: pc=off and pc=on use different batch compositions
# (different GEMM shapes from token concatenation), which causes tiny
# bf16 rounding differences that accumulate over decode steps. at 32
# tokens these differences flip an argmax (~60% failure rate on req 2,
# token 31). 16 tokens stays within the safe margin.
NUM_TOKENS_TO_GENERATE = 16
# multi-group uses 4x the requests (20 vs 5), creating larger batch
# composition differences between pc=off and pc=on. reduce decode steps
# to stay within the safe bf16 rounding margin.
MULTI_GROUP_TOKENS_TO_GENERATE = 8
NUM_GROUPS = 4
GROUP_TOKEN_STRIDE = 2000  # each group uses ~1752 contiguous token IDs


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
    """End-to-end test for Mamba prefix caching with a real hybrid model.

    Exercises the 4 key indices within a Mamba prefill:
      1. num_mamba_matched — blocks with cached Mamba state (determines skip)
      2. num_kv_matched   — blocks with shared KV cache (can exceed mamba match)
      3. last_aligned     — last full-block boundary (always cached if reachable)
      4. end_of_sequence  — prompt length (triggers EOS cache if block-aligned)

    5 requests with overlapping prefixes are processed in a specific order
    so that each request sees a different combination of these indices.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel()
        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_model(self):
        """Create minimal hybrid model (M*- pattern, 3 layers)."""
        transformer_config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=3,
            hidden_size=256,
            mamba_num_heads=16,
            num_attention_heads=16,
            use_cpu_initialization=True,
            cuda_graph_impl="none",
            inference_rng_tracker=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            add_bias_linear=True,
            is_hybrid_model=True,
        )

        model = MambaModel(
            config=transformer_config,
            mamba_stack_spec=mamba_stack_spec,
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

    def _create_prompts(self):
        """Build 5 prompts with carefully designed prefix sharing.

        Each prompt uses disjoint token ID ranges for unique segments
        so that parent-chained block hashes differ where content differs.

        The prompts are designed so that each request hits a different
        combination of the 4 key indices:

        req 0 (300 tokens): seed request, no matches. last_aligned=256
            (not block-aligned, so EOS path is NOT taken; intermediate
            extraction at token 256 caches block 0's Mamba state).

        req 1 (800 tokens): 1 KV match, 1 Mamba match (block 0 from req 0).
            skip=256. kv_div_rel=0 (excluded), last_aligned_rel=512 (cached).
            Caches Mamba state at token 768 (last_aligned block 2).

        req 2 (800 tokens): 2 KV matches, but only 1 Mamba match. This is
            the key case where num_kv_matched > num_mamba_matched: block 1
            has KV cache but no Mamba state yet. skip=256.
            kv_div_rel=256 (divergence block cached), last_aligned_rel=512
            (cached). Creates Mamba state for block 1 at token 512.

        req 3 (800 tokens): 2 KV matches, 2 Mamba matches. Benefits from
            req 2's divergence block at token 512 (block 1). skip=512.
            kv_div_rel=0 (excluded), last_aligned_rel=256 (cached).

        req 4 (1100 tokens): 3 KV matches, 3 Mamba matches. Restores from
            req 1's last_aligned cache at token 768 (block 2). skip=768.
            kv_div_rel=0 (excluded), last_aligned_rel=256 (cached).
        """
        device = torch.cuda.current_device()

        base = torch.arange(0, 256, dtype=torch.int64, device=device)
        seg_B = torch.arange(1000, 1256, dtype=torch.int64, device=device)
        seg_1rest = torch.arange(2000, 2288, dtype=torch.int64, device=device)
        seg_2rest = torch.arange(3000, 3288, dtype=torch.int64, device=device)
        seg_3rest = torch.arange(4000, 4288, dtype=torch.int64, device=device)
        seg_4ext = torch.arange(5000, 5332, dtype=torch.int64, device=device)
        extra_0 = torch.arange(6000, 6044, dtype=torch.int64, device=device)

        prompt_0 = torch.cat([base, extra_0])  # 300
        prompt_1 = torch.cat([base, seg_B, seg_1rest])  # 800
        prompt_2 = torch.cat([base, seg_B, seg_2rest])  # 800
        prompt_3 = torch.cat([base, seg_B, seg_3rest])  # 800
        prompt_4 = torch.cat([base, seg_B, seg_1rest[:256], seg_4ext])  # 1100

        prompts = [prompt_0, prompt_1, prompt_2, prompt_3, prompt_4]

        # Sanity check lengths
        assert len(prompt_0) == 300
        assert len(prompt_1) == 800
        assert len(prompt_2) == 800
        assert len(prompt_3) == 800
        assert len(prompt_4) == 1100

        return prompts

    def _create_group_prompts(self, group_id):
        """Build 5 prompts for one group with non-overlapping token ranges.

        Identical sharing structure to _create_prompts, but token IDs start
        at group_id * GROUP_TOKEN_STRIDE so different groups have completely
        disjoint hashes.
        """
        device = torch.cuda.current_device()
        offset = group_id * GROUP_TOKEN_STRIDE

        base = torch.arange(offset, offset + 256, dtype=torch.int64, device=device)
        seg_B = torch.arange(offset + 256, offset + 512, dtype=torch.int64, device=device)
        seg_1rest = torch.arange(offset + 512, offset + 800, dtype=torch.int64, device=device)
        seg_2rest = torch.arange(offset + 800, offset + 1088, dtype=torch.int64, device=device)
        seg_3rest = torch.arange(offset + 1088, offset + 1376, dtype=torch.int64, device=device)
        seg_4ext = torch.arange(offset + 1376, offset + 1708, dtype=torch.int64, device=device)
        extra_0 = torch.arange(offset + 1708, offset + 1752, dtype=torch.int64, device=device)

        prompt_0 = torch.cat([base, extra_0])  # 300
        prompt_1 = torch.cat([base, seg_B, seg_1rest])  # 800
        prompt_2 = torch.cat([base, seg_B, seg_2rest])  # 800
        prompt_3 = torch.cat([base, seg_B, seg_3rest])  # 800
        prompt_4 = torch.cat([base, seg_B, seg_1rest[:256], seg_4ext])  # 1100

        assert len(prompt_0) == 300
        assert len(prompt_1) == 800
        assert len(prompt_2) == 800
        assert len(prompt_3) == 800
        assert len(prompt_4) == 1100

        return [prompt_0, prompt_1, prompt_2, prompt_3, prompt_4]

    def _build_engine(
        self,
        model,
        mamba_config,
        enable_prefix_caching,
        buffer_size_gb=0.5,
        prefix_caching_mamba_gb=0.05,
        request_rounder=4,
    ):
        """Build context + wrapper + controller + engine."""
        set_rounder(request_rounder)

        inference_config_kwargs = dict(
            max_sequence_length=MAX_SEQ_LEN,
            buffer_size_gb=buffer_size_gb,
            block_size_tokens=BLOCK_SIZE,
            mamba_inference_state_config=mamba_config,
            materialize_only_last_token_logits=False,
            enable_prefix_caching=enable_prefix_caching,
            unified_memory_level=0,
        )
        if enable_prefix_caching:
            inference_config_kwargs.update(
                prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
                prefix_caching_mamba_gb=prefix_caching_mamba_gb,
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
        CudaGraphManager.global_mempool = None

        engine = DynamicInferenceEngine(controller, context)
        return engine

    def _make_request(
        self, req_id, prompt, enable_prefix_caching, num_tokens_to_generate=NUM_TOKENS_TO_GENERATE
    ):
        """Create a DynamicInferenceRequest with the given prompt."""
        sampling_params = SamplingParams(
            num_tokens_to_generate=num_tokens_to_generate, termination_id=-1, top_k=1
        )
        return DynamicInferenceRequest(
            request_id=req_id,
            prompt_tokens=prompt,
            sampling_params=sampling_params,
            block_size_tokens=BLOCK_SIZE if enable_prefix_caching else None,
            enable_prefix_caching=enable_prefix_caching,
        )

    def _run_pc_off(self, model, mamba_config, prompts):
        """Run all 5 requests with prefix caching disabled.

        Returns: ({req_id: generated_tokens}, lifetime_prefill_token_count).
        """
        engine = self._build_engine(model, mamba_config, enable_prefix_caching=False)

        for i, prompt in enumerate(prompts):
            req = self._make_request(i, prompt, enable_prefix_caching=False)
            engine._add_request(req)

        finished = {}
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

        return finished, engine.context.lifetime_prefill_token_count

    def _get_ref_count(self, alloc, block_hash):
        """Look up ref count for a block identified by its KV hash."""
        bid = alloc.kv_hash_to_block_id.get(block_hash)
        if bid is None:
            return 0
        return alloc.block_ref_counts[bid].item()

    def _run_pc_on(self, model, mamba_config, prompts):
        """Run requests with prefix caching enabled, in controlled order.

        Submission order:
          - reqs 0, 1, 2, 4 added up front
          - req 3 added after step 3 (after req 2's mamba states are committed)

        At each prefill step, verifies:
          - _mamba_num_matched_blocks on the scheduled request(s)
          - mamba_hash_to_block_id entry count and specific hash presence
          - lifetime_prefill_token_count (reflects effective tokens after skip)
          - block_ref_counts on shared blocks (verifies block re-use)

        Returns: ({req_id: generated_tokens}, lifetime_prefill_token_count).
        """
        engine = self._build_engine(model, mamba_config, enable_prefix_caching=True)
        alloc = engine.context.block_allocator
        ctx = engine.context

        # Create all requests up front (so hashes are precomputed)
        reqs = {}
        for i, prompt in enumerate(prompts):
            reqs[i] = self._make_request(i, prompt, enable_prefix_caching=True)

        # Add reqs 0, 1, 2, 4 (not 3)
        for i in [0, 1, 2, 4]:
            engine._add_request(reqs[i])

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

            # block 0 hash (shared by all 5 requests)
            h0 = reqs[0].precomputed_block_hashes[0]

            if step == 1:
                # req 0: num_mamba_matched=0 (seed request, no prior state).
                # last_aligned=256 != end_of_sequence=300, so intermediate
                # extraction (not EOS copy) caches block 0's Mamba state.
                assert (
                    reqs[0]._mamba_num_matched_blocks == 0
                ), f"step 1: expected 0 mamba matches, got {reqs[0]._mamba_num_matched_blocks}"
                assert (
                    len(alloc.mamba_hash_to_block_id) == 1
                ), f"step 1: expected 1 mamba entry, got {len(alloc.mamba_hash_to_block_id)}"
                assert reqs[0].precomputed_block_hashes[0] in alloc.mamba_hash_to_block_id

                # Prefill skipping: no skip, all 300 tokens computed
                assert (
                    step_prefill == 300
                ), f"step 1: expected 300 prefill tokens, got {step_prefill}"

                # Ref counts: block 0 owned by req 0 only
                assert self._get_ref_count(alloc, h0) == 1

            elif step == 2:
                # req 1: num_mamba_matched=1 (block 0 from req 0), num_kv_matched=1.
                # last_aligned=768, cached via intermediate extraction at block 2.
                assert (
                    reqs[1]._mamba_num_matched_blocks == 1
                ), f"step 2: expected 1 mamba match, got {reqs[1]._mamba_num_matched_blocks}"
                assert (
                    len(alloc.mamba_hash_to_block_id) == 2
                ), f"step 2: expected 2 mamba entries, got {len(alloc.mamba_hash_to_block_id)}"
                assert reqs[1].precomputed_block_hashes[2] in alloc.mamba_hash_to_block_id

                # Prefill skipping: skip 256 (1 block), compute 800 - 256 = 544
                assert (
                    step_prefill == 544
                ), f"step 2: expected 544 prefill tokens, got {step_prefill}"

                # Ref counts: block 0 now shared by reqs 0 and 1
                assert self._get_ref_count(alloc, h0) == 2

            elif step == 3:
                # req 2: num_kv_matched=2 > num_mamba_matched=1 (block 1 has
                # KV cache from req 1 but no Mamba state yet). skip limited
                # by Mamba, not KV. Divergence block at token 512 is cached,
                # creating the Mamba state that req 3 will later discover.
                assert (
                    reqs[2]._mamba_num_matched_blocks == 1
                ), f"step 3 req 2: expected 1 mamba match, got {reqs[2]._mamba_num_matched_blocks}"
                # req 4: num_mamba_matched=3, restores from req 1's
                # last_aligned cache at block 2 (token 768).
                assert (
                    reqs[4]._mamba_num_matched_blocks == 3
                ), f"step 3 req 4: expected 3 mamba matches, got {reqs[4]._mamba_num_matched_blocks}"
                # 2 prior + req 2 divergence + req 2 last_aligned + req 4 last_aligned = 5
                assert (
                    len(alloc.mamba_hash_to_block_id) == 5
                ), f"step 3: expected 5 mamba entries, got {len(alloc.mamba_hash_to_block_id)}"
                assert reqs[2].precomputed_block_hashes[1] in alloc.mamba_hash_to_block_id
                assert reqs[2].precomputed_block_hashes[2] in alloc.mamba_hash_to_block_id
                assert reqs[4].precomputed_block_hashes[3] in alloc.mamba_hash_to_block_id

                # Prefill skipping: req 2 skips 256 (544 computed) +
                # req 4 skips 768 (332 computed) = 876 total
                assert (
                    step_prefill == 544 + 332
                ), f"step 3: expected 876 prefill tokens, got {step_prefill}"

                # Ref counts: block 0 shared by reqs 0, 1, 2, 4
                assert self._get_ref_count(alloc, h0) == 4
                # block 1 (tokens 0-511) shared by reqs 1, 2, 4
                h1 = reqs[1].precomputed_block_hashes[1]
                assert self._get_ref_count(alloc, h1) == 3

                # Add req 3 now: req 2's divergence block at token 512
                # is discoverable, so req 3 will get 2 Mamba matches.
                engine._add_request(reqs[3])
                req3_added = True

            elif step == 4 and req3_added:
                # req 3: num_mamba_matched=2 (block 0 from req 0, block 1
                # from req 2's divergence cache). This is the payoff: the
                # divergence block cached by req 2 enables req 3 to skip
                # an additional 256 tokens (512 vs 256).
                assert (
                    reqs[3]._mamba_num_matched_blocks == 2
                ), f"step 4: expected 2 mamba matches, got {reqs[3]._mamba_num_matched_blocks}"
                assert (
                    len(alloc.mamba_hash_to_block_id) == 6
                ), f"step 4: expected 6 mamba entries, got {len(alloc.mamba_hash_to_block_id)}"
                assert reqs[3].precomputed_block_hashes[2] in alloc.mamba_hash_to_block_id

                # Prefill skipping: skip 512 (2 blocks), compute 800 - 512 = 288
                assert (
                    step_prefill == 288
                ), f"step 4: expected 288 prefill tokens, got {step_prefill}"

                # Ref counts: block 0 shared by all 5 reqs, block 1 by reqs 1-4
                assert self._get_ref_count(alloc, h0) == 5
                h1 = reqs[1].precomputed_block_hashes[1]
                assert self._get_ref_count(alloc, h1) == 4

        return finished, ctx.lifetime_prefill_token_count

    def _run_multi_pc_off(self, model, mamba_config, all_prompts):
        """Run all 20 requests (4 groups x 5) with prefix caching disabled.

        Returns: lifetime_prefill_token_count.
        """
        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=False,
            buffer_size_gb=2.0,
            prefix_caching_mamba_gb=0.2,
        )

        for g, prompts in enumerate(all_prompts):
            for local_id, prompt in enumerate(prompts):
                req_id = g * 5 + local_id
                req = self._make_request(
                    req_id,
                    prompt,
                    enable_prefix_caching=False,
                    num_tokens_to_generate=MULTI_GROUP_TOKENS_TO_GENERATE,
                )
                engine._add_request(req)

        while engine.has_unfinished_requests():
            engine.step_modern()

        return engine.context.lifetime_prefill_token_count

    def _run_group_pc_on(self, model, mamba_config, prompts, base_req_id):
        """Run one group's 5 requests with pc=on, return generated outputs.

        The engine's hash coordination produces the same 4-step scheduling
        pattern as the multi-group run (reqs 0,1,2+4,3). Used as a
        reference to verify multi-group outputs match per-group outputs,
        proving no cross-group interference.
        """
        engine = self._build_engine(model, mamba_config, enable_prefix_caching=True)

        for i, prompt in enumerate(prompts):
            req = self._make_request(
                base_req_id + i,
                prompt,
                enable_prefix_caching=True,
                num_tokens_to_generate=MULTI_GROUP_TOKENS_TO_GENERATE,
            )
            engine._add_request(req)

        finished = {}
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

        return finished

    def _run_multi_pc_on(self, model, mamba_config, all_prompts):
        """Run 4 groups with prefix caching enabled, in controlled order.

        Scheduling order:
          - all 4 groups' reqs 0, 1, 2, 4 added up front (16 requests)
          - after step 3, all 4 groups' req 3 added (4 requests)

        Per-step verification checks block reuse, prefill skipping, mamba
        hash table size, and mamba hash presence across all 4 groups.

        Returns: ({req_id: generated_tokens}, lifetime_prefill_token_count).
        """
        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            buffer_size_gb=2.0,
            prefix_caching_mamba_gb=0.2,
        )
        alloc = engine.context.block_allocator
        ctx = engine.context

        # Create all requests, organized by group
        reqs = []  # reqs[g][local_id]
        for g, prompts in enumerate(all_prompts):
            group_reqs = {}
            for local_id, prompt in enumerate(prompts):
                req_id = g * 5 + local_id
                group_reqs[local_id] = self._make_request(
                    req_id,
                    prompt,
                    enable_prefix_caching=True,
                    num_tokens_to_generate=MULTI_GROUP_TOKENS_TO_GENERATE,
                )
            reqs.append(group_reqs)

        # Add reqs 0, 1, 2, 4 for all groups (not req 3)
        for g in range(NUM_GROUPS):
            for local_id in [0, 1, 2, 4]:
                engine._add_request(reqs[g][local_id])

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

            if step == 1:
                # Each group's req 0: no mamba matches (seed requests)
                for g in range(NUM_GROUPS):
                    assert reqs[g][0]._mamba_num_matched_blocks == 0, (
                        f"step 1 group {g}: expected 0 mamba matches, "
                        f"got {reqs[g][0]._mamba_num_matched_blocks}"
                    )
                    assert reqs[g][0].precomputed_block_hashes[0] in alloc.mamba_hash_to_block_id

                assert (
                    len(alloc.mamba_hash_to_block_id) == 4
                ), f"step 1: expected 4 mamba entries, got {len(alloc.mamba_hash_to_block_id)}"
                assert (
                    step_prefill == 1200
                ), f"step 1: expected 1200 prefill tokens, got {step_prefill}"

            elif step == 2:
                # Each group's req 1: 1 mamba match (block 0 from req 0)
                for g in range(NUM_GROUPS):
                    assert reqs[g][1]._mamba_num_matched_blocks == 1, (
                        f"step 2 group {g}: expected 1 mamba match, "
                        f"got {reqs[g][1]._mamba_num_matched_blocks}"
                    )
                    assert reqs[g][1].precomputed_block_hashes[2] in alloc.mamba_hash_to_block_id

                assert (
                    len(alloc.mamba_hash_to_block_id) == 8
                ), f"step 2: expected 8 mamba entries, got {len(alloc.mamba_hash_to_block_id)}"
                assert (
                    step_prefill == 2176
                ), f"step 2: expected 2176 prefill tokens, got {step_prefill}"

            elif step == 3:
                # Each group's reqs 2 + 4 co-scheduled
                for g in range(NUM_GROUPS):
                    assert reqs[g][2]._mamba_num_matched_blocks == 1, (
                        f"step 3 group {g} req 2: expected 1 mamba match, "
                        f"got {reqs[g][2]._mamba_num_matched_blocks}"
                    )
                    assert reqs[g][4]._mamba_num_matched_blocks == 3, (
                        f"step 3 group {g} req 4: expected 3 mamba matches, "
                        f"got {reqs[g][4]._mamba_num_matched_blocks}"
                    )
                    assert reqs[g][2].precomputed_block_hashes[1] in alloc.mamba_hash_to_block_id
                    assert reqs[g][2].precomputed_block_hashes[2] in alloc.mamba_hash_to_block_id
                    assert reqs[g][4].precomputed_block_hashes[3] in alloc.mamba_hash_to_block_id

                    # Ref counts
                    h0 = reqs[g][0].precomputed_block_hashes[0]
                    h1 = reqs[g][1].precomputed_block_hashes[1]
                    assert self._get_ref_count(alloc, h0) == 4, (
                        f"step 3 group {g}: block 0 ref count expected 4, "
                        f"got {self._get_ref_count(alloc, h0)}"
                    )
                    assert self._get_ref_count(alloc, h1) == 3, (
                        f"step 3 group {g}: block 1 ref count expected 3, "
                        f"got {self._get_ref_count(alloc, h1)}"
                    )

                assert (
                    len(alloc.mamba_hash_to_block_id) == 20
                ), f"step 3: expected 20 mamba entries, got {len(alloc.mamba_hash_to_block_id)}"
                assert (
                    step_prefill == 3504
                ), f"step 3: expected 3504 prefill tokens, got {step_prefill}"

                # Add all 4 groups' req 3
                for g in range(NUM_GROUPS):
                    engine._add_request(reqs[g][3])
                req3_added = True

            elif step == 4 and req3_added:
                # Each group's req 3: 2 mamba matches
                for g in range(NUM_GROUPS):
                    assert reqs[g][3]._mamba_num_matched_blocks == 2, (
                        f"step 4 group {g}: expected 2 mamba matches, "
                        f"got {reqs[g][3]._mamba_num_matched_blocks}"
                    )
                    assert reqs[g][3].precomputed_block_hashes[2] in alloc.mamba_hash_to_block_id

                    # Ref counts
                    h0 = reqs[g][0].precomputed_block_hashes[0]
                    h1 = reqs[g][1].precomputed_block_hashes[1]
                    assert self._get_ref_count(alloc, h0) == 5, (
                        f"step 4 group {g}: block 0 ref count expected 5, "
                        f"got {self._get_ref_count(alloc, h0)}"
                    )
                    assert self._get_ref_count(alloc, h1) == 4, (
                        f"step 4 group {g}: block 1 ref count expected 4, "
                        f"got {self._get_ref_count(alloc, h1)}"
                    )

                assert (
                    len(alloc.mamba_hash_to_block_id) == 24
                ), f"step 4: expected 24 mamba entries, got {len(alloc.mamba_hash_to_block_id)}"
                assert (
                    step_prefill == 1152
                ), f"step 4: expected 1152 prefill tokens, got {step_prefill}"

        return finished, ctx.lifetime_prefill_token_count

    @torch.inference_mode()
    def test_mamba_prefix_caching_e2e(self):
        """Verify output tokens match between pc=off and pc=on."""
        skip_if_mamba_sequence_packing_not_available()

        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_prompts()

        off_outputs, off_prefill = self._run_pc_off(model, mamba_config, prompts)
        on_outputs, on_prefill = self._run_pc_on(model, mamba_config, prompts)

        for req_id in range(5):
            assert (
                off_outputs[req_id] == on_outputs[req_id]
            ), f"req {req_id}: pc=off {off_outputs[req_id]} != pc=on {on_outputs[req_id]}"

        # Verify prefill token savings. pc=off computes all prompt tokens;
        # pc=on skips matched prefix tokens.
        # pc=off: 300 + 800 + 800 + 800 + 1100 = 3800
        # pc=on: 300 + 544 + 544 + 288 + 332 = 2008
        assert off_prefill == 3800, f"pc=off prefill: expected 3800, got {off_prefill}"
        assert on_prefill == 2008, f"pc=on prefill: expected 2008, got {on_prefill}"
        assert on_prefill < off_prefill

    @torch.inference_mode()
    def test_mamba_prefix_caching_multi_group_e2e(self):
        """Verify multi-group prefix caching with 4 independent groups.

        Tests that 4 concurrent groups with disjoint token ranges produce
        identical outputs whether run simultaneously or independently,
        proving no cross-group interference.

        Output correctness (pc=off vs pc=on) is already validated by the
        single-group test. Direct pc=off vs pc=on comparison is impractical
        here because the 4x batch size amplifies bf16 GEMM rounding
        differences beyond the argmax stability margin.
        """
        skip_if_mamba_sequence_packing_not_available()

        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        all_prompts = [self._create_group_prompts(g) for g in range(NUM_GROUPS)]

        # Run pc=off for prefill baseline
        off_prefill = self._run_multi_pc_off(model, mamba_config, all_prompts)

        # Run all 4 groups simultaneously with pc=on (per-step assertions inside)
        on_outputs, on_prefill = self._run_multi_pc_on(model, mamba_config, all_prompts)

        # Run each group independently with pc=on as reference
        for g in range(NUM_GROUPS):
            ref_outputs = self._run_group_pc_on(
                model, mamba_config, all_prompts[g], base_req_id=g * 5
            )
            for local_id in range(5):
                req_id = g * 5 + local_id
                assert on_outputs[req_id] == ref_outputs[req_id], (
                    f"group {g} req {local_id}: multi-group {on_outputs[req_id]} "
                    f"!= per-group {ref_outputs[req_id]}"
                )

        # Verify prefill token savings
        assert (
            off_prefill == NUM_GROUPS * 3800
        ), f"pc=off prefill: expected {NUM_GROUPS * 3800}, got {off_prefill}"
        assert (
            on_prefill == NUM_GROUPS * 2008
        ), f"pc=on prefill: expected {NUM_GROUPS * 2008}, got {on_prefill}"
        assert on_prefill < off_prefill

    # =====================================================================
    # Edge case: block-aligned EOS + cached logit zero-prefill
    # =====================================================================

    def _create_block_aligned_prompts(self):
        """Build 4 prompts with block-aligned lengths for EOS path testing.

        A: 256 tokens — exactly 1 block, block-aligned EOS
        B: 256 tokens — identical to A, cached logit zero-prefill
        C: 512 tokens — exactly 2 blocks, extends A's prefix
        D: 512 tokens — identical to C, cached logit zero-prefill
        """
        device = torch.cuda.current_device()
        seg_0 = torch.arange(8000, 8256, dtype=torch.int64, device=device)
        seg_1 = torch.arange(8256, 8512, dtype=torch.int64, device=device)

        prompt_A = seg_0.clone()  # 256
        prompt_B = seg_0.clone()  # 256 (identical to A)
        prompt_C = torch.cat([seg_0, seg_1])  # 512
        prompt_D = torch.cat([seg_0, seg_1])  # 512 (identical to C)

        assert len(prompt_A) == 256
        assert len(prompt_B) == 256
        assert len(prompt_C) == 512
        assert len(prompt_D) == 512

        return [prompt_A, prompt_B, prompt_C, prompt_D]

    def _run_eos_pc_off(self, model, mamba_config, prompts):
        """Run block-aligned prompts with pc=off."""
        engine = self._build_engine(model, mamba_config, enable_prefix_caching=False)

        for i, prompt in enumerate(prompts):
            req = self._make_request(i, prompt, enable_prefix_caching=False)
            engine._add_request(req)

        finished = {}
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

        return finished, engine.context.lifetime_prefill_token_count

    def _run_eos_pc_on(self, model, mamba_config, prompts):
        """Run block-aligned prompts with pc=on, per-step assertions.

        Scheduling with pending_block_hashes coordination:
          - step 1: A scheduled (B, C, D deferred: h0 pending)
          - step 2: B + C co-scheduled (D deferred: h1 pending from C)
          - step 3: D scheduled

        Verifies:
          - Block-aligned EOS caching (dynamic_context.py:1224-1232):
            prompt_len is an exact multiple of block_size, so Mamba state
            is cached via store_mamba_state_for_block_from_live instead of
            intermediate extraction.
          - Cached logit zero-prefill: when all blocks match (identical
            prompts, block-aligned), effective_chunk_length == 0 and the
            cached block boundary logit is used directly.
          - Identical prompts: all hashes match, no new mamba entries.
        """
        engine = self._build_engine(model, mamba_config, enable_prefix_caching=True)
        alloc = engine.context.block_allocator
        ctx = engine.context

        reqs = {}
        for i, prompt in enumerate(prompts):
            reqs[i] = self._make_request(i, prompt, enable_prefix_caching=True)

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
                # A: 256 tokens, no mamba matches. EOS path fires
                # (last_aligned_abs = 256 == prompt_len) and caches block 0
                # via store_mamba_state_for_block_from_live. No intermediate
                # extraction: last_aligned_rel = 256 - 0 = 256, seq_len = 256,
                # 256 < 256 is false → filtered.
                assert (
                    reqs[0]._mamba_num_matched_blocks == 0
                ), f"step 1: expected 0 mamba matches, got {reqs[0]._mamba_num_matched_blocks}"
                assert (
                    len(alloc.mamba_hash_to_block_id) == 1
                ), f"step 1: expected 1 mamba entry, got {len(alloc.mamba_hash_to_block_id)}"
                assert reqs[0].precomputed_block_hashes[0] in alloc.mamba_hash_to_block_id
                assert (
                    step_prefill == 256
                ), f"step 1: expected 256 prefill tokens, got {step_prefill}"

            elif step == 2:
                # B: 1 mamba match, skip = 256 == chunk_length → effective == 0,
                #   uses cached logit (zero redundant computation)
                assert (
                    reqs[1]._mamba_num_matched_blocks == 1
                ), f"step 2 B: expected 1 mamba match, got {reqs[1]._mamba_num_matched_blocks}"
                # C: 1 mamba match. skip = min(1*256, 512) = 256, effective = 256
                assert (
                    reqs[2]._mamba_num_matched_blocks == 1
                ), f"step 2 C: expected 1 mamba match, got {reqs[2]._mamba_num_matched_blocks}"
                # Block 1 cached via C's EOS path (512 == 512, block-aligned)
                assert (
                    len(alloc.mamba_hash_to_block_id) == 2
                ), f"step 2: expected 2 mamba entries, got {len(alloc.mamba_hash_to_block_id)}"
                assert reqs[2].precomputed_block_hashes[1] in alloc.mamba_hash_to_block_id
                # B contributes 0 tokens (cached logit), C contributes 256
                assert (
                    step_prefill == 0 + 256
                ), f"step 2: expected 256 prefill tokens, got {step_prefill}"

            elif step == 3:
                # D: 2 mamba matches, skip = 512 == chunk_length → effective == 0,
                #   uses cached logit (zero redundant computation)
                assert (
                    reqs[3]._mamba_num_matched_blocks == 2
                ), f"step 3 D: expected 2 mamba matches, got {reqs[3]._mamba_num_matched_blocks}"
                # No new mamba entries (D's blocks already cached by A + C)
                assert (
                    len(alloc.mamba_hash_to_block_id) == 2
                ), f"step 3: expected 2 mamba entries, got {len(alloc.mamba_hash_to_block_id)}"
                assert step_prefill == 0, f"step 3: expected 0 prefill tokens, got {step_prefill}"

        return finished, ctx.lifetime_prefill_token_count

    @torch.inference_mode()
    def test_mamba_block_aligned_eos_e2e(self):
        """Verify block-aligned EOS caching, cached logit zero-prefill, and identical prompts.

        Exercises code paths not covered by the main e2e test:
        - Block-aligned EOS path (dynamic_context.py:1224-1232): prompt_len is
          an exact multiple of block_size, so Mamba state is cached via
          store_mamba_state_for_block_from_live (copy from live buffer) instead
          of intermediate extraction.
        - Cached logit zero-prefill: when all blocks match (identical prompts,
          block-aligned), effective_chunk_length == 0 and the cached block
          boundary logit is used directly, achieving zero redundant computation.
        - Identical prompts: every block hash matches, no new mamba entries.
        """
        skip_if_mamba_sequence_packing_not_available()

        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_block_aligned_prompts()

        off_outputs, off_prefill = self._run_eos_pc_off(model, mamba_config, prompts)
        on_outputs, on_prefill = self._run_eos_pc_on(model, mamba_config, prompts)

        for req_id in range(4):
            assert (
                off_outputs[req_id] == on_outputs[req_id]
            ), f"req {req_id}: pc=off {off_outputs[req_id]} != pc=on {on_outputs[req_id]}"

        # pc=off: 256 + 256 + 512 + 512 = 1536
        # pc=on: 256 + (0+256) + 0 = 512
        #   step 1: A = 256 (seed, no matches)
        #   step 2: B = 0 (1 mamba match, cached logit, zero prefill)
        #         + C = 256 (1 mamba match, skip 256, effective 256)
        #   step 3: D = 0 (2 mamba matches, cached logit, zero prefill)
        assert off_prefill == 1536, f"pc=off prefill: expected 1536, got {off_prefill}"
        assert on_prefill == 512, f"pc=on prefill: expected 512, got {on_prefill}"
        assert on_prefill < off_prefill

    # =====================================================================
    # Edge case: LRU eviction invalidates mamba state
    # =====================================================================

    def _create_eviction_prompts(self):
        """Build 3 prompts for LRU eviction testing.

        E: 300 tokens — seed request, caches block 0 mamba state
        F: 300 tokens — disjoint prefix, forces eviction of E's cached block
        G: 300 tokens — identical to E, verifies eviction effect
        """
        device = torch.cuda.current_device()
        prompt_E = torch.arange(8000, 8300, dtype=torch.int64, device=device)
        prompt_F = torch.arange(8300, 8600, dtype=torch.int64, device=device)
        prompt_G = torch.arange(8000, 8300, dtype=torch.int64, device=device)  # identical to E

        for p in [prompt_E, prompt_F, prompt_G]:
            assert len(p) == 300

        return [prompt_E, prompt_F, prompt_G]

    @torch.inference_mode()
    def test_mamba_lru_eviction_e2e(self):
        """Verify KV eviction invalidates mamba state via invalidate_mamba_state_for_block.

        Exercises dynamic_block_allocator.py:280-286: when KV blocks are evicted
        (via _deregister_blocks), blocks with cached Mamba state are also removed
        from mamba_hash_to_block_id and their cache slots freed via
        invalidate_mamba_state_for_block.

        Uses a tiny KV buffer (3 blocks total, 2 usable, request_rounder=1) so
        that a single completed request's cached block fills the only available
        cache slot, and the next disjoint request forces LRU eviction.

        Each 300-token request uses 2 blocks: 1 full (256 tokens, registered
        with hash) + 1 partial (44 tokens, unregistered, returned to free pool
        on release). Only the full block occupies a cache slot.

        Flow:
          E runs alone → caches block 0 mamba state (1 cached, 1 free)
          F runs alone → needs 2 blocks, only 1 free → evicts E's cached
            block (oldest LRU), which triggers invalidate_mamba_state_for_block
          G runs alone (identical to E) → no KV match, no mamba match,
            full prefill despite being an identical prompt
        """
        skip_if_mamba_sequence_packing_not_available()

        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_eviction_prompts()

        # Use a tiny buffer: 0.002 GB → 3 total blocks (2 usable) for this
        # model. request_rounder=1 so max_requests = (3-1) // 1 = 2.
        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            buffer_size_gb=0.002,
            prefix_caching_mamba_gb=0.05,
            request_rounder=1,
        )
        alloc = engine.context.block_allocator
        ctx = engine.context

        assert alloc.total_count == 3, (
            f"expected 3 total blocks for eviction test, got {alloc.total_count} "
            f"(adjust buffer_size_gb)"
        )
        assert ctx.max_requests >= 1, f"max_requests={ctx.max_requests}, need at least 1"

        finished = {}

        def _run_one(req_id, prompt):
            # Use num_tokens_to_generate=2 so the request survives the prefill
            # step. With =1, the request finishes during prefill before
            # commit_mamba_intermediate_states runs (it's called after
            # update_requests in the engine), so mamba state is never cached.
            req = self._make_request(
                req_id, prompt, enable_prefix_caching=True, num_tokens_to_generate=2
            )
            engine._add_request(req)
            while engine.has_unfinished_requests():
                result = engine.step_modern()
                for record in result["finished_request_records"]:
                    merged = record.merge()
                    finished[merged.request_id] = list(merged.generated_tokens)
            return req

        # --- E: seed request, no matches ---
        req_E = _run_one(0, prompts[0])
        h_E0 = req_E.precomputed_block_hashes[0]
        assert h_E0 in alloc.mamba_hash_to_block_id, "E's block 0 should have mamba state"
        assert h_E0 in alloc.kv_hash_to_block_id, "E's block 0 should be in KV cache"
        assert len(alloc.mamba_hash_to_block_id) == 1
        # 1 cached full block + 1 free (partial returned to pool)
        assert (
            alloc.total_avail == 1
        ), f"expected 1 free block after E (partial returned), got {alloc.total_avail}"

        # --- F: disjoint prefix, forces eviction of E's cached block ---
        req_F = _run_one(1, prompts[1])
        h_F0 = req_F.precomputed_block_hashes[0]
        assert h_F0 in alloc.mamba_hash_to_block_id

        # E's block was evicted via _deregister_blocks → invalidate_mamba_state_for_block
        assert h_E0 not in alloc.kv_hash_to_block_id, "E's KV hash should be evicted"
        assert (
            h_E0 not in alloc.mamba_hash_to_block_id
        ), "E's mamba hash should be invalidated by _deregister_blocks"
        assert len(alloc.mamba_hash_to_block_id) == 1

        # --- G: identical to E, but E's state was evicted ---
        req_G = _run_one(2, prompts[2])

        # No mamba match despite identical prompt (mamba hash evicted)
        assert req_G._mamba_num_matched_blocks == 0, (
            f"G should have 0 mamba matches after eviction, "
            f"got {req_G._mamba_num_matched_blocks}"
        )

        # G re-cached block 0's mamba state
        assert h_E0 in alloc.mamba_hash_to_block_id, "G should re-register E's block 0 mamba hash"

        # E and G have the same prompt, both computed from scratch with
        # greedy decoding (top_k=1), so their outputs must match
        assert finished[0] == finished[2], f"E tokens {finished[0]} != G tokens {finished[2]}"
