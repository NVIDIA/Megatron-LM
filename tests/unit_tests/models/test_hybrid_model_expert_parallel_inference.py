# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for full HybridModel inference with expert-parallel batch dimension sync.

When expert parallelism > 1 with strict matching (hybrid models), batch
dimensions are MAX-reduced across EP ranks. Different EP ranks can be in
one of four request states:

  - NONE:    0 requests (dummy rank, uses is_expert_parallel_dummy_cuda_graph_step)
  - DECODE:  >0 decode requests, 0 prefill requests
  - PREFILL: 0 decode requests, >0 prefill requests
  - MIXED:   >0 decode requests, >0 prefill requests

These tests verify that the full HybridModel produces correct output shapes
for every combination of these states across EP ranks, using the real EP
synchronization path (strict matching + MAX-reduce on batch dimensions).
"""

import itertools

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.cuda_graphs import _CudagraphGlobalRecord, delete_cuda_graphs
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils

# Request state constants for parametrized tests.
NONE = "none"  # 0 requests (dummy rank)
DECODE = "decode"  # >0 decode, 0 prefill
PREFILL = "prefill"  # 0 decode, >0 prefill
MIXED = "mixed"  # >0 decode, >0 prefill
# Non-cuda-graphable states (token_count exceeds the cuda graph capacity
# of _SWEEP_MAX_REQUESTS, forcing eager fallback for all EP ranks).
PREFILL_AT_MAX_TOKENS = "prefill_max"  # prefill-only, token_count > graph capacity
MIXED_GIANT_PREFILL = "mixed_giant"  # mixed with a giant prefill, token_count > graph capacity

_NO_CUDA_GRAPH_STATES = {PREFILL_AT_MAX_TOKENS, MIXED_GIANT_PREFILL}
ALL_STATES = [NONE, DECODE, PREFILL, MIXED, PREFILL_AT_MAX_TOKENS, MIXED_GIANT_PREFILL]

# Fixed expert-parallel size. When world_size > _EP_SIZE the remaining
# ranks form data-parallel replicas, each running the same EP combo
# independently.
_EP_SIZE = 4

# Combinatorial sweep: unordered combinations with repetition of ALL_STATES
# across the EP ranks. Since rank assignment is symmetric (shuffling ranks
# with the same multiset of states is not a distinct configuration), we use
# combinations_with_replacement rather than the full Cartesian product.
_STATE_COMBOS = list(itertools.combinations_with_replacement(ALL_STATES, _EP_SIZE))

# Cap max_requests for the cross-product sweep so that the cuda graph
# capacity (max_requests * (num_speculative_tokens + 1) = 64) is small
# enough that PREFILL_AT_MAX_TOKENS / MIXED_GIANT_PREFILL token counts
# overflow it (forcing eager fallback), while the other states still fit.
_SWEEP_MAX_REQUESTS = 64

# Batch dimensions used to set up each non-dummy state via
# add_dummy_requests_for_cudagraph_capture. These are intentionally small
# to keep the tests fast while still exercising the EP padding logic.
_STATE_DIMS = {
    # 2 decode requests, 1 token each -> 2 tokens total
    DECODE: InferenceBatchDimensions(token_count=2, prefill_req_count=0, decode_req_count=2),
    # 2 prefill requests with 16 tokens each -> 32 tokens total
    PREFILL: InferenceBatchDimensions(token_count=32, prefill_req_count=2, decode_req_count=0),
    # 4 decode (4 tokens) + 2 prefill (60 tokens) = 64 tokens
    MIXED: InferenceBatchDimensions(token_count=64, prefill_req_count=2, decode_req_count=4),
    # 1 prefill of 128 tokens (> _SWEEP_MAX_REQUESTS=64, overflows every graph)
    PREFILL_AT_MAX_TOKENS: InferenceBatchDimensions(
        token_count=128, prefill_req_count=1, decode_req_count=0
    ),
    # 2 decode (2 tokens) + 1 prefill (126 tokens) = 128 tokens (overflows graphs)
    MIXED_GIANT_PREFILL: InferenceBatchDimensions(
        token_count=128, prefill_req_count=1, decode_req_count=2
    ),
}


@pytest.mark.internal
class TestDynamicInference:
    """Verify full HybridModel output shapes under EP strict matching scenarios."""

    HIDDEN_SIZE = 256
    NUM_ATTN_HEADS = 4
    MAX_SEQ_LEN = 512
    VOCAB_SIZE = 128

    def setup_method(self, method):
        available, reason = _check_mamba_sequence_packing_support(for_inference_not_training=True)
        if not available:
            pytest.skip(reason, allow_module_level=True)
        if not is_fa_min_version("2.7.3"):
            pytest.skip("need flash-attn >= 2.7.3 for dynamic batching", allow_module_level=True)
        if Utils.world_size < _EP_SIZE:
            pytest.skip(f"EP test requires at least {_EP_SIZE} GPUs", allow_module_level=True)
        if Utils.world_size % _EP_SIZE != 0:
            pytest.skip(
                f"world_size ({Utils.world_size}) must be divisible by EP size ({_EP_SIZE})",
                allow_module_level=True,
            )

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=_EP_SIZE,
        )

    def teardown_method(self, method):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()

    def _build_model(self):
        model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
        config = TransformerConfig(
            num_layers=3,
            mtp_hybrid_override_pattern="ME*",
            hidden_size=self.HIDDEN_SIZE,
            num_attention_heads=self.NUM_ATTN_HEADS,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            attention_backend=AttnBackend.fused,
            num_moe_experts=2,
            moe_token_dispatcher_type="alltoall",
            cuda_graph_impl="local",
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=self.VOCAB_SIZE,
            max_sequence_length=self.MAX_SEQ_LEN,
            hybrid_layer_pattern="M*",
        )
        model.cuda()
        model.eval()
        return model

    def _build_context(
        self,
        model,
        *,
        num_cuda_graphs=16,
        use_cuda_graphs_for_non_decode_steps=True,
        max_requests=None,
    ):
        mamba_config = MambaInferenceStateConfig.from_model(model)
        return DynamicInferenceContext(
            model_config=model.config,
            inference_config=InferenceConfig(
                max_sequence_length=self.MAX_SEQ_LEN,
                buffer_size_gb=1.0,
                block_size_tokens=256,
                materialize_only_last_token_logits=False,
                mamba_inference_state_config=mamba_config,
                num_cuda_graphs=num_cuda_graphs,
                use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
                max_requests=max_requests,
            ),
        )

    def _assert_dynamic_inference_shape(self, model, ctx, rank, state_label):
        """Run model.forward and assert the logits shape matches
        padded_batch_dimensions.token_count."""
        padded = ctx.padded_batch_dimensions
        input_ids = torch.randint(0, self.VOCAB_SIZE, (1, padded.token_count), device="cuda")
        out = model.forward(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=None,
            inference_context=ctx,
            runtime_gather_output=True,
        )
        assert out.shape == (1, padded.token_count, self.VOCAB_SIZE), (
            f"Rank {rank} (state={state_label}): expected output shape "
            f"(1, {padded.token_count}, {self.VOCAB_SIZE}), "
            f"got {tuple(out.shape)}"
        )

    @staticmethod
    def _assert_cuda_graphs_were_replayed(expect_replayed, rank, label):
        """Assert that CUDA graphs were (or were not) recorded and replayed
        during the preceding model.forward() call.

        The inference path in CudaGraphManager records each layer's runner
        into _CudagraphGlobalRecord.cudagraph_inference_record the first time
        a graph is captured.  A non-empty record with fwd_graph_recorded=True
        on every runner confirms the graph was both recorded and replayed.
        """
        record = _CudagraphGlobalRecord.cudagraph_inference_record
        if expect_replayed:
            assert len(record) > 0, (
                f"Rank {rank} ({label}): expected CUDA graphs to be recorded and "
                f"replayed, but cudagraph_inference_record is empty"
            )
            for runner, _graph_type, _args, _kwargs in record:
                assert runner.fwd_graph_recorded, (
                    f"Rank {rank} ({label}): CUDA graph runner for "
                    f"{runner.base_module.__class__.__name__} (layer "
                    f"{runner.base_module.layer_number}) was not recorded"
                )
        else:
            assert len(record) == 0, (
                f"Rank {rank} ({label}): expected no CUDA graph replay, "
                f"but cudagraph_inference_record has {len(record)} entries"
            )

    def _assert_dummy_forward_shape(self, model, rank):
        """Run model.forward with a single dummy token (no inference context),
        mirroring the real engine's dummy_forward fallback, and verify the
        logits shape."""
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        dummy_tokens = torch.zeros(1, tp_size, dtype=torch.long, device="cuda")
        position_ids = torch.zeros(1, tp_size, dtype=torch.long, device="cuda")
        out = model.forward(input_ids=dummy_tokens, position_ids=position_ids, attention_mask=None)
        expected = (1, tp_size, self.VOCAB_SIZE)
        assert out.shape == expected, (
            f"Rank {rank} (dummy bail-out): expected out shape "
            f"{expected}, got {tuple(out.shape)}"
        )

    # ------------------------------------------------------------------
    # test_ep_state_cross_product: combinatorial sweep with mixed CUDA graphs
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("rank_states", _STATE_COMBOS, ids=[",".join(s) for s in _STATE_COMBOS])
    @pytest.mark.internal
    @torch.inference_mode()
    def test_ep_state_cross_product(self, rank_states):
        """Test all combinatorial (unordered, with repetition) assignments of
        the request states across EP ranks.

        The context is built with use_cuda_graphs_for_non_decode_steps=True,
        so the CUDA graph list contains decode-only, mixed, and prefill-only
        graphs. For combos whose states all fit within the cuda graph capacity
        (max_requests * (num_speculative_tokens + 1) = _SWEEP_MAX_REQUESTS),
        every rank — including dummy ranks — must find a matching graph.

        For combos that include any non-cuda-graphable state
        (PREFILL_AT_MAX_TOKENS / MIXED_GIANT_PREFILL), match_graph_config
        returns None on every rank (eager fallback): dummy ranks bail out and
        run dummy_forward; non-dummy ranks run the eager
        padded_batch_dimensions path.
        """
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        my_state = rank_states[ep_rank]
        is_dummy = my_state == NONE
        expect_graph = not any(s in _NO_CUDA_GRAPH_STATES for s in rank_states)

        model = self._build_model()
        ctx = self._build_context(model, max_requests=_SWEEP_MAX_REQUESTS)

        # Phase 1: Set up each rank's request state directly.
        if not is_dummy:
            ctx.add_dummy_requests_for_cudagraph_capture(_STATE_DIMS[my_state])

        # Phase 2: Initialize attention state (EP collective).
        ctx.initialize_attention_state(is_expert_parallel_dummy_cuda_graph_step=is_dummy)

        # Phase 3: Verify CUDA graph match status agrees with the combo.
        used_graph = ctx.using_cuda_graph_this_step()
        assert used_graph == expect_graph, (
            f"EP rank {ep_rank} (state={my_state}, rank_states={rank_states}): "
            f"expected using_cuda_graph_this_step={expect_graph}, got {used_graph}"
        )

        if used_graph:
            # All EP ranks must agree on padded token count when a graph
            # was matched. (In eager mode dummy ranks bail out without
            # setting padded_batch_dimensions, so this only applies here.)
            padded = ctx.padded_batch_dimensions
            ep_group = parallel_state.get_expert_model_parallel_group()
            tc = torch.tensor([padded.token_count], dtype=torch.int32, device="cuda")
            tc_max = tc.clone()
            tc_min = tc.clone()
            dist.all_reduce(tc_max, op=dist.ReduceOp.MAX, group=ep_group)
            dist.all_reduce(tc_min, op=dist.ReduceOp.MIN, group=ep_group)
            assert tc_max.item() == tc_min.item(), (
                f"Padded token count mismatch across EP ranks: "
                f"min={tc_min.item()}, max={tc_max.item()} "
                f"(rank_states={rank_states})"
            )
            self._assert_dynamic_inference_shape(model, ctx, ep_rank, my_state)
        elif is_dummy:
            # Eager fallback, dummy rank: bailed out of
            # initialize_attention_state, run the dummy_forward path.
            self._assert_dummy_forward_shape(model, ep_rank)
        else:
            # Eager fallback, non-dummy rank: padded_batch_dimensions
            # set via the eager fallback path inside
            # initialize_attention_state.
            self._assert_dynamic_inference_shape(model, ctx, ep_rank, my_state)

        self._assert_cuda_graphs_were_replayed(
            used_graph, ep_rank, f"state={my_state}, rank_states={rank_states}"
        )

    # ------------------------------------------------------------------
    # test_dummy_bailout_with_decode_only_cuda_graphs: dedicated bail-out
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "peer_state", [PREFILL, MIXED], ids=[f"peer={s}" for s in [PREFILL, MIXED]]
    )
    @pytest.mark.internal
    @torch.inference_mode()
    def test_dummy_bailout_with_decode_only_cuda_graphs(self, peer_state):
        """Verify the dummy-rank bail-out path when only decode CUDA graphs
        are available.

        With use_cuda_graphs_for_non_decode_steps=False, the CUDA graph list
        contains only decode-only graphs. When any EP rank has prefill
        requests, adjust_batch_dims_for_expert_parallelism returns None
        (forcing eager mode), and match_graph_config returns None for all
        ranks. A dummy rank then bails out of initialize_attention_state
        early (padded_batch_dimensions is not set).

        This test verifies that:
          - The dummy rank correctly falls back to model.forward (the real
            engine's dummy_forward path).
          - The non-dummy rank still produces correct _dynamic_inference
            output (padded_batch_dimensions is computed via the non-graph
            fallback path).

        Even EP ranks are dummy; odd EP ranks have the parametrized peer_state.
        """
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        is_even = ep_rank % 2 == 0

        model = self._build_model()
        ctx = self._build_context(model, use_cuda_graphs_for_non_decode_steps=False)

        # Set up request state.
        if not is_even:
            ctx.add_dummy_requests_for_cudagraph_capture(_STATE_DIMS[peer_state])

        # Initialize attention state (EP collective).
        if is_even:
            ctx.initialize_attention_state(is_expert_parallel_dummy_cuda_graph_step=True)
        else:
            ctx.initialize_attention_state()

        # Verify: no rank should have matched a CUDA graph because the
        # peer has prefill but only decode graphs are available.
        assert not ctx.using_cuda_graph_this_step(), (
            f"EP rank {ep_rank}: expected no CUDA graph match with "
            f"decode-only graphs and peer_state={peer_state}"
        )

        if is_even:
            # Dummy rank bailed out — exercise the eager fallback.
            self._assert_dummy_forward_shape(model, ep_rank)
        else:
            # Non-dummy rank: padded_batch_dimensions is set via the
            # non-graph fallback path in initialize_attention_state.
            self._assert_dynamic_inference_shape(model, ctx, ep_rank, peer_state)
        self._assert_cuda_graphs_were_replayed(
            False, ep_rank, f"decode-only graphs, peer_state={peer_state}"
        )

    # ------------------------------------------------------------------
    # test_mixed_cuda_graphs_tokens_exceed_max_requests: eager fallback
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "peer_state", [PREFILL, MIXED], ids=[f"peer={s}" for s in [PREFILL, MIXED]]
    )
    @pytest.mark.internal
    @torch.inference_mode()
    def test_mixed_cuda_graphs_tokens_exceed_max_requests(self, peer_state):
        """Verify eager fallback when mixed CUDA graphs are allowed but
        a rank's token count exceeds the CUDA graph capacity.

        With use_cuda_graphs_for_non_decode_steps=True, the CUDA graph
        list includes mixed and prefill-only graphs. However, the
        maximum CUDA graph token capacity is bounded by max_requests
        (specifically, max_requests * (num_speculative_tokens + 1)).

        When one EP rank has a token count exceeding this capacity, no
        CUDA graph can accommodate the EP-adjusted dimensions.
        match_graph_config returns None for all ranks, forcing eager
        mode globally. This test verifies that:
          - No rank matches a CUDA graph (eager mode is forced).
          - Dummy ranks bail out and produce correct shapes via the
            eager dummy_forward path.
          - Non-dummy ranks produce correct shapes via the eager
            padded_batch_dimensions fallback.
        """
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        is_even = ep_rank % 2 == 0

        model = self._build_model()

        # Use a small max_requests so that the CUDA graph capacity
        # (max_requests tokens with no speculative decoding) is easily
        # exceeded by a prefill-heavy rank.
        small_max_requests = 16
        ctx = self._build_context(
            model, use_cuda_graphs_for_non_decode_steps=True, max_requests=small_max_requests
        )

        # Even EP ranks are dummy (no requests). Odd EP ranks get a state
        # whose token count exceeds small_max_requests.
        overflow_token_count = small_max_requests + 16  # 32 tokens > 16 capacity
        overflow_dims = {
            PREFILL: InferenceBatchDimensions(
                token_count=overflow_token_count, prefill_req_count=2, decode_req_count=0
            ),
            MIXED: InferenceBatchDimensions(
                token_count=overflow_token_count, prefill_req_count=1, decode_req_count=2
            ),
        }

        if not is_even:
            ctx.add_dummy_requests_for_cudagraph_capture(overflow_dims[peer_state])

        # Initialize attention state (EP collective).
        if is_even:
            ctx.initialize_attention_state(is_expert_parallel_dummy_cuda_graph_step=True)
        else:
            ctx.initialize_attention_state()

        # No rank should have matched a CUDA graph — the EP-adjusted
        # token count exceeds every graph's capacity.
        assert not ctx.using_cuda_graph_this_step(), (
            f"EP rank {ep_rank}: expected no CUDA graph match when token count "
            f"({overflow_token_count}) exceeds max_requests ({small_max_requests}), "
            f"peer_state={peer_state}"
        )

        if is_even:
            # Dummy rank bailed out — exercise the eager fallback.
            self._assert_dummy_forward_shape(model, ep_rank)
        else:
            # Non-dummy rank: padded_batch_dimensions is set via the
            # eager fallback path.  Verify shape correctness.
            self._assert_dynamic_inference_shape(model, ctx, ep_rank, peer_state)
        self._assert_cuda_graphs_were_replayed(
            False, ep_rank, f"overflow tokens, peer_state={peer_state}"
        )
