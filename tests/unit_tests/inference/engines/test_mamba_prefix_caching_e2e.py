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

    def setup_method(self, method):
        Utils.initialize_model_parallel()
        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    def teardown_method(self, method):
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
        prefix_caching_mamba_gb=0.05,
        request_rounder=4,
        use_triton_conv1d=False,
        num_cuda_graphs=None,
    ):
        set_rounder(request_rounder)
        inference_config_kwargs = dict(
            max_sequence_length=MAX_SEQ_LEN,
            buffer_size_gb=buffer_size_gb,
            block_size_tokens=BLOCK_SIZE,
            mamba_inference_state_config=mamba_config,
            materialize_only_last_token_logits=False,
            enable_prefix_caching=enable_prefix_caching,
            unified_memory_level=0,
            use_triton_conv1d=use_triton_conv1d,
            num_cuda_graphs=num_cuda_graphs,
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
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None
        for module in model.modules():
            if isinstance(module, CudaGraphManager):
                module.cudagraph_runners.clear()
                module.inference_cudagraphs_lookup_table.clear()
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
        use_triton_conv1d=False,
        **engine_kwargs,
    ):
        """Run all prompts with given pc setting, return (finished_dict, lifetime_prefill)."""
        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=enable_pc,
            use_triton_conv1d=use_triton_conv1d,
            **engine_kwargs,
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

    def _run_pc_on(self, model, mamba_config, prompts, use_triton_conv1d=False):
        """Run requests with prefix caching enabled, verifying per-step state."""
        engine = self._build_engine(
            model, mamba_config, enable_prefix_caching=True, use_triton_conv1d=use_triton_conv1d
        )
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

    def _run_multi_pc_on(
        self, model, mamba_config, all_prompts, use_triton_conv1d=False, num_cuda_graphs=None
    ):
        """Run 4 groups with prefix caching enabled, verifying per-step state."""
        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            buffer_size_gb=2.0,
            prefix_caching_mamba_gb=0.2,
            use_triton_conv1d=use_triton_conv1d,
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

    @pytest.mark.parametrize("use_triton_conv1d", [False, True])
    @torch.inference_mode()
    def test_mamba_prefix_caching_e2e(self, use_triton_conv1d):
        """Verify output tokens match between pc=off and pc=on."""
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_prompts()

        off_outputs, off_prefill = self._run_simple(
            model, mamba_config, prompts, False, use_triton_conv1d=use_triton_conv1d
        )
        on_outputs, on_prefill = self._run_pc_on(
            model, mamba_config, prompts, use_triton_conv1d=use_triton_conv1d
        )

        for req_id in range(5):
            assert (
                off_outputs[req_id] == on_outputs[req_id]
            ), f"req {req_id}: pc=off {off_outputs[req_id]} != pc=on {on_outputs[req_id]}"
        assert off_prefill == 3800 and on_prefill == 2008 and on_prefill < off_prefill

    @pytest.mark.parametrize("num_cuda_graphs", [None, 2])
    @pytest.mark.parametrize("use_triton_conv1d", [False, True])
    @torch.inference_mode()
    def test_mamba_prefix_caching_multi_group_e2e(self, use_triton_conv1d, num_cuda_graphs):
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
            use_triton_conv1d=use_triton_conv1d,
            num_cuda_graphs=num_cuda_graphs,
            buffer_size_gb=2.0,
            prefix_caching_mamba_gb=0.2,
        )
        on_outputs, on_prefill = self._run_multi_pc_on(
            model,
            mamba_config,
            all_prompts,
            use_triton_conv1d=use_triton_conv1d,
            num_cuda_graphs=num_cuda_graphs,
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
                use_triton_conv1d=use_triton_conv1d,
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

    def _run_eos_pc_on(self, model, mamba_config, prompts, use_triton_conv1d=False):
        """Run block-aligned prompts with pc=on, per-step assertions.

        Scheduling with pending_block_hashes coordination:
          - step 1: A scheduled (B, C, D deferred: h0 pending)
          - step 2: B + C co-scheduled (D deferred: h1 pending from C)
          - step 3: D scheduled
        """
        engine = self._build_engine(
            model, mamba_config, enable_prefix_caching=True, use_triton_conv1d=use_triton_conv1d
        )
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

    @pytest.mark.parametrize("use_triton_conv1d", [False, True])
    @torch.inference_mode()
    def test_mamba_block_aligned_eos_e2e(self, use_triton_conv1d):
        """Verify block-aligned EOS caching and recompute-based back-off."""
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_block_aligned_prompts()

        off_outputs, off_prefill = self._run_simple(
            model, mamba_config, prompts, False, use_triton_conv1d=use_triton_conv1d
        )
        on_outputs, on_prefill = self._run_eos_pc_on(
            model, mamba_config, prompts, use_triton_conv1d=use_triton_conv1d
        )

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

    @pytest.mark.parametrize("use_triton_conv1d", [False, True])
    @torch.inference_mode()
    def test_mamba_lru_eviction_e2e(self, use_triton_conv1d):
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
            use_triton_conv1d=use_triton_conv1d,
        )
        alloc = engine.context.kv_block_allocator
        ctx = engine.context

        assert alloc.total_count == 3, f"expected 3 total blocks, got {alloc.total_count}"
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
        assert len(ctx.mamba_slot_allocator.hash_to_block_id) == 1 and alloc.total_avail == 1

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
