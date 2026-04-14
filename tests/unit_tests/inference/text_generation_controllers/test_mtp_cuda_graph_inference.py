# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for CUDA-graphed MTP (Multi-Token Prediction) inference.

Verifies that:
1. CUDA graph replay produces the same output as eager execution (no extra
   padding in the CUDA graphed case).
2. CUDA graphs work correctly with sequence parallelism (padding is applied
   to make batch sizes divisible by TP).
3. CUDA graphs work correctly with expert parallelism and dummy ranks.
"""

import itertools
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.cuda_graphs import (
    _set_capture_end,
    _set_capture_start,
    delete_cuda_graphs,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import unwrap_model
from tests.unit_tests.test_utilities import Utils

# --------------------------------------------------------------------------- #
#  TestMTPCudaGraphInference (TP = 2)
# --------------------------------------------------------------------------- #


class TestMTPCudaGraphInference:
    """Tests for MTP CUDA-graphed inference with tensor parallelism.

    All tests require at least 2 GPUs (TP = 2).
    """

    HIDDEN_SIZE = 32
    VOCAB_SIZE = 100
    MAX_SEQ_LEN = 64
    NUM_LAYERS = 4
    NUM_ATTN_HEADS = 4
    TP_SIZE = 2

    def setup_method(self, method):
        if Utils.world_size < self.TP_SIZE:
            pytest.skip(f"Need at least {self.TP_SIZE} GPUs")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.TP_SIZE, pipeline_model_parallel_size=1
        )

    def teardown_method(self, method):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()

    # ---- helpers ---------------------------------------------------------- #

    def _build_model(self, *, sequence_parallel=False, mtp_num_layers=2):
        """Build a GPT model with MTP layers and local CUDA graph support."""
        model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
        config = TransformerConfig(
            num_layers=self.NUM_LAYERS,
            hidden_size=self.HIDDEN_SIZE,
            num_attention_heads=self.NUM_ATTN_HEADS,
            use_cpu_initialization=True,
            attention_backend=AttnBackend.local,
            params_dtype=torch.float32,
            tensor_model_parallel_size=self.TP_SIZE,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.float32,
            mtp_num_layers=mtp_num_layers,
            sequence_parallel=sequence_parallel,
            cuda_graph_impl="local",
        )
        layer_spec = get_gpt_layer_local_spec()
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=layer_spec, use_transformer_engine=False
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=self.VOCAB_SIZE,
            max_sequence_length=self.MAX_SEQ_LEN,
            parallel_output=True,
            pre_process=True,
            post_process=True,
            mtp_block_spec=mtp_block_spec,
        ).cuda()
        model.eval()
        return model

    def _build_controller(
        self,
        *,
        sequence_parallel=False,
        mtp_num_layers=2,
        num_speculative_tokens=2,
        max_requests=None,
    ):
        """Build a model, DynamicInferenceContext, and TextGenerationController."""
        model = self._build_model(
            sequence_parallel=sequence_parallel, mtp_num_layers=mtp_num_layers
        )
        config = model.config
        if max_requests is None:
            max_requests = 16
        context = DynamicInferenceContext(
            model_config=config,
            inference_config=InferenceConfig(
                max_sequence_length=self.MAX_SEQ_LEN * 2,
                buffer_size_gb=0.2,
                materialize_only_last_token_logits=False,
                use_flashinfer_fused_rope=None,
                unified_memory_level=0,
                num_speculative_tokens=num_speculative_tokens,
                block_size_tokens=256,
                max_requests=max_requests,
            ),
        )
        wrapped = GPTInferenceWrapper(model, context)
        wrapped.model_is_pipeline_parallel = False
        mock_tokenizer = mock.Mock()
        ctrl = TextGenerationController(inference_wrapped_model=wrapped, tokenizer=mock_tokenizer)
        return model, context, ctrl

    def _warmup_mtp_graphs(self, model, batch_sizes, *, sp_enabled=False):
        """Warm up MTP CUDA graphs for the given batch sizes.

        Replicates the warmup logic from ``DynamicEngine._warmup_mtp_cuda_graphs``.
        """
        unwrapped = unwrap_model(model)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        device = torch.cuda.current_device()
        dtype = model.config.params_dtype
        hidden_size = model.config.hidden_size

        _set_capture_start()
        for bs in sorted(batch_sizes):
            dummy_hidden = torch.zeros((bs, 1, hidden_size), device=device, dtype=dtype)
            if sp_enabled:
                dummy_hidden = scatter_to_sequence_parallel_region(dummy_hidden, group=tp_group)
            dummy_token_ids = torch.zeros((1, bs), device=device, dtype=torch.long)
            dummy_position_ids = torch.zeros((1, bs), device=device, dtype=torch.int64)
            unwrapped.compute_mtp_single_step(
                hidden_states=dummy_hidden,
                next_token_ids=dummy_token_ids,
                position_ids=dummy_position_ids,
                depth=0,
            )
        _set_capture_end()

    @staticmethod
    def _set_mtp_cuda_graph_flag(model, enabled):
        """Set ``use_mtp_cuda_graphs`` on all MTP layers."""
        unwrapped = unwrap_model(model)
        for layer in unwrapped.mtp.layers:
            layer.use_mtp_cuda_graphs = enabled

    # ---- Test 1: graph output matches eager (no additional padding) ------- #

    @pytest.mark.parametrize("batch_size", [2, 4, 8])
    @torch.inference_mode()
    def test_cuda_graph_output_matches_eager(self, batch_size):
        """CUDA graph replay produces the same output as eager execution.

        The batch size exactly matches a warmed-up graph, so there is no
        additional padding in the CUDA graphed case.  Both paths must
        produce identical hidden states and logits.
        """
        model = self._build_model()
        unwrapped = unwrap_model(model)
        self._warmup_mtp_graphs(model, [batch_size])

        # Create identical random inputs on all TP ranks.
        hidden = torch.randn(batch_size, 1, self.HIDDEN_SIZE, device='cuda')
        dist.broadcast(hidden, src=0)
        token_ids = torch.randint(0, self.VOCAB_SIZE, (1, batch_size), device='cuda')
        dist.broadcast(token_ids, src=0)
        position_ids = torch.arange(batch_size, device='cuda', dtype=torch.int64).unsqueeze(0)

        # Graph path.
        self._set_mtp_cuda_graph_flag(model, True)
        h_graph, logits_graph = unwrapped.compute_mtp_single_step(
            hidden_states=hidden.clone(),
            next_token_ids=token_ids.clone(),
            position_ids=position_ids.clone(),
            depth=0,
        )
        # Clone immediately — CUDA graph output buffers are reused on next call.
        h_graph = h_graph.clone()
        logits_graph = logits_graph.clone()

        # Eager path.
        self._set_mtp_cuda_graph_flag(model, False)
        h_eager, logits_eager = unwrapped.compute_mtp_single_step(
            hidden_states=hidden.clone(),
            next_token_ids=token_ids.clone(),
            position_ids=position_ids.clone(),
            depth=0,
        )

        torch.testing.assert_close(h_graph, h_eager)
        torch.testing.assert_close(logits_graph, logits_eager)

    # ---- Test 2: graph matches eager with sequence parallelism ------------ #

    @pytest.mark.parametrize("batch_size", [2, 4])
    @torch.inference_mode()
    def test_cuda_graph_output_matches_eager_with_sp(self, batch_size):
        """CUDA graph replay matches eager with sequence parallelism.

        Hidden states are in scattered SP format ``[batch_size/TP, 1, H]``.
        Token/position IDs remain at full ``[1, batch_size]``.  Both paths
        must produce identical outputs.
        """
        model = self._build_model(sequence_parallel=True)
        unwrapped = unwrap_model(model)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        self._warmup_mtp_graphs(model, [batch_size], sp_enabled=True)

        # Create random inputs; scatter hidden for SP.
        hidden = torch.randn(batch_size, 1, self.HIDDEN_SIZE, device='cuda')
        dist.broadcast(hidden, src=0)
        hidden_sp = scatter_to_sequence_parallel_region(hidden, group=tp_group)

        token_ids = torch.randint(0, self.VOCAB_SIZE, (1, batch_size), device='cuda')
        dist.broadcast(token_ids, src=0)
        position_ids = torch.arange(batch_size, device='cuda', dtype=torch.int64).unsqueeze(0)

        # Graph path.
        self._set_mtp_cuda_graph_flag(model, True)
        h_graph, logits_graph = unwrapped.compute_mtp_single_step(
            hidden_states=hidden_sp.clone(),
            next_token_ids=token_ids.clone(),
            position_ids=position_ids.clone(),
            depth=0,
        )
        h_graph = h_graph.clone()
        logits_graph = logits_graph.clone()

        # Eager path.
        self._set_mtp_cuda_graph_flag(model, False)
        h_eager, logits_eager = unwrapped.compute_mtp_single_step(
            hidden_states=hidden_sp.clone(),
            next_token_ids=token_ids.clone(),
            position_ids=position_ids.clone(),
            depth=0,
        )

        torch.testing.assert_close(h_graph, h_eager)
        torch.testing.assert_close(logits_graph, logits_eager)

    # ---- Test 3: end-to-end _compute_serial_mtp_and_sample with SP ------- #

    @pytest.mark.parametrize("active_request_count", [2, 3, 4, 5])
    @torch.inference_mode()
    def test_cuda_graph_sp_padding_end_to_end(self, active_request_count):
        """Full ``_compute_serial_mtp_and_sample`` with CUDA graphs and SP.

        Active request counts that are not multiples of TP are padded.
        The MTP CUDA graph is pre-warmed for the padded batch size.
        Verifies that padding, SP scatter/gather, and MTP forward all
        work correctly through the CUDA graph path.
        """
        tp_size = self.TP_SIZE
        num_spec = 2
        # max_requests must accommodate the padded count.
        max_requests = ((active_request_count + tp_size - 1) // tp_size) * tp_size * 2
        model, ctx, ctrl = self._build_controller(
            sequence_parallel=True,
            mtp_num_layers=num_spec,
            num_speculative_tokens=num_spec,
            max_requests=max_requests,
        )
        unwrapped = unwrap_model(model)

        # Compute the padded batch size.
        padded_count = active_request_count
        padded_count += (tp_size - padded_count % tp_size) % tp_size

        # Warmup MTP CUDA graphs for the padded count.
        self._warmup_mtp_graphs(model, [padded_count], sp_enabled=True)
        ctrl._mtp_cuda_graph_batch_sizes = [padded_count]

        # Set up context state.
        ctx.total_request_count = active_request_count
        ctx.paused_request_count = 0
        ctx.request_kv_length_offsets[:active_request_count] = torch.arange(
            active_request_count, dtype=torch.int32, device='cuda'
        )
        ctx.request_query_lengths[:active_request_count] = torch.ones(
            active_request_count, dtype=torch.int32, device='cuda'
        )

        ctrl.num_speculative_tokens = num_spec
        ctrl.num_mtp_heads = num_spec
        ctrl._init_mtp_sampling_tensor()
        # Zero out buffers allocated with torch.empty to avoid garbage values
        # in padding positions causing out-of-bounds embedding lookups.
        ctrl._mtp_token_ids_buf.zero_()
        ctrl._mtp_position_ids_buf.zero_()
        ctrl._sampled_tokens_cuda[:active_request_count] = torch.remainder(
            torch.arange(active_request_count, device='cuda'), self.VOCAB_SIZE
        )

        # Build decoder hidden states cache in SP format.
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        pad = (tp_size - active_request_count % tp_size) % tp_size
        s_total = active_request_count + pad

        torch.manual_seed(42)
        full_hidden = torch.randn(s_total, 1, self.HIDDEN_SIZE, device='cuda', dtype=torch.float32)
        dist.broadcast(full_hidden, src=0)
        local_hidden = full_hidden.chunk(tp_size)[tp_rank].contiguous()
        unwrapped._decoder_hidden_states_cache = local_hidden

        ctrl._last_accepted_seq_indices = torch.arange(active_request_count, device='cuda')

        # Enable CUDA graphs for MTP.
        ctrl._mtp_resolved_padded_count = padded_count
        self._set_mtp_cuda_graph_flag(model, True)

        # Greedy sampling: top_k=1 selects argmax deterministically.
        ctrl._torch_sampling_buckets = [(list(range(active_request_count)), 1.0, 1, 0.0)]
        ctrl._torch_sampling_bucket_index_tensors = [
            torch.arange(active_request_count, device='cuda', dtype=torch.long)
        ]

        # Run MTP forward pass.
        ctrl._compute_serial_mtp_and_sample()

        # Verify sampled MTP tokens.
        for depth in range(num_spec):
            sampled = ctrl._sampled_mtp_tokens_cuda[depth, :active_request_count]
            assert sampled.shape == (active_request_count,)
            assert sampled.dtype == torch.int64
            assert torch.all(sampled >= 0) and torch.all(sampled < self.VOCAB_SIZE)

        # Verify decoder hidden states cache was cleaned up.
        assert not hasattr(unwrapped, '_decoder_hidden_states_cache')

    # ---- Test 4: SP padding graph vs eager produces same MTP tokens ------- #

    @pytest.mark.parametrize("active_request_count", [3, 5])
    @torch.inference_mode()
    def test_cuda_graph_sp_padding_matches_eager(self, active_request_count):
        """With SP padding, CUDA graph path produces the same MTP tokens as eager.

        Runs ``_compute_serial_mtp_and_sample`` twice — once through the
        CUDA graph path and once through the eager path — with identical
        inputs, and asserts the sampled MTP tokens match.
        """
        tp_size = self.TP_SIZE
        num_spec = 2
        padded_count = active_request_count
        padded_count += (tp_size - padded_count % tp_size) % tp_size
        max_requests = padded_count * 2

        def _run_mtp(use_cuda_graph):
            """Build fresh model+controller and run MTP, returning sampled tokens."""
            delete_cuda_graphs()
            model, ctx, ctrl = self._build_controller(
                sequence_parallel=True,
                mtp_num_layers=num_spec,
                num_speculative_tokens=num_spec,
                max_requests=max_requests,
            )
            unwrapped = unwrap_model(model)

            if use_cuda_graph:
                self._warmup_mtp_graphs(model, [padded_count], sp_enabled=True)
                ctrl._mtp_cuda_graph_batch_sizes = [padded_count]
                ctrl._mtp_resolved_padded_count = padded_count
                self._set_mtp_cuda_graph_flag(model, True)
            else:
                ctrl._mtp_resolved_padded_count = None
                self._set_mtp_cuda_graph_flag(model, False)

            ctx.total_request_count = active_request_count
            ctx.paused_request_count = 0
            ctx.request_kv_length_offsets[:active_request_count] = torch.arange(
                active_request_count, dtype=torch.int32, device='cuda'
            )
            ctx.request_query_lengths[:active_request_count] = torch.ones(
                active_request_count, dtype=torch.int32, device='cuda'
            )

            ctrl.num_speculative_tokens = num_spec
            ctrl.num_mtp_heads = num_spec
            ctrl._init_mtp_sampling_tensor()
            # Zero out buffers allocated with torch.empty to avoid garbage values
            # in padding positions causing out-of-bounds embedding lookups.
            ctrl._mtp_token_ids_buf.zero_()
            ctrl._mtp_position_ids_buf.zero_()
            ctrl._sampled_tokens_cuda[:active_request_count] = torch.remainder(
                torch.arange(active_request_count, device='cuda'), self.VOCAB_SIZE
            )

            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_group = parallel_state.get_tensor_model_parallel_group()
            pad = (tp_size - active_request_count % tp_size) % tp_size
            s_total = active_request_count + pad

            torch.manual_seed(42)
            full_hidden = torch.randn(
                s_total, 1, self.HIDDEN_SIZE, device='cuda', dtype=torch.float32
            )
            dist.broadcast(full_hidden, src=0)
            local_hidden = full_hidden.chunk(tp_size)[tp_rank].contiguous()
            unwrapped._decoder_hidden_states_cache = local_hidden

            ctrl._last_accepted_seq_indices = torch.arange(active_request_count, device='cuda')
            ctrl._torch_sampling_buckets = [(list(range(active_request_count)), 1.0, 1, 0.0)]
            ctrl._torch_sampling_bucket_index_tensors = [
                torch.arange(active_request_count, device='cuda', dtype=torch.long)
            ]

            ctrl._compute_serial_mtp_and_sample()

            return [
                ctrl._sampled_mtp_tokens_cuda[d, :active_request_count].clone()
                for d in range(num_spec)
            ]

        graph_tokens = _run_mtp(use_cuda_graph=True)
        eager_tokens = _run_mtp(use_cuda_graph=False)

        for depth in range(num_spec):
            assert torch.equal(graph_tokens[depth], eager_tokens[depth]), (
                f"Depth {depth}: graph tokens {graph_tokens[depth].tolist()} != "
                f"eager tokens {eager_tokens[depth].tolist()}"
            )

    # ---- Test 5: multiple MTP depths with CUDA graphs --------------------- #

    @torch.inference_mode()
    def test_cuda_graph_multi_depth(self):
        """Run multiple MTP depths with CUDA graphs enabled.

        Verifies that the hidden output from one depth feeds correctly into
        the next depth through the same CUDA graph, producing valid outputs
        at every depth.
        """
        batch_size = 4
        num_depths = 2
        model = self._build_model(mtp_num_layers=num_depths)
        unwrapped = unwrap_model(model)
        self._warmup_mtp_graphs(model, [batch_size])
        self._set_mtp_cuda_graph_flag(model, True)

        hidden = torch.randn(batch_size, 1, self.HIDDEN_SIZE, device='cuda')
        dist.broadcast(hidden, src=0)
        token_ids = torch.randint(0, self.VOCAB_SIZE, (1, batch_size), device='cuda')
        dist.broadcast(token_ids, src=0)
        position_ids = torch.arange(batch_size, device='cuda', dtype=torch.int64).unsqueeze(0)

        current_hidden = hidden.clone()
        for depth in range(num_depths):
            current_hidden, logits = unwrapped.compute_mtp_single_step(
                hidden_states=current_hidden,
                next_token_ids=token_ids.clone(),
                position_ids=position_ids.clone(),
                depth=depth,
            )
            # Clone — graph output buffers are reused.
            current_hidden = current_hidden.clone()

            assert current_hidden.shape == (batch_size, 1, self.HIDDEN_SIZE), (
                f"Depth {depth}: expected hidden shape ({batch_size}, 1, {self.HIDDEN_SIZE}), "
                f"got {current_hidden.shape}"
            )
            assert logits.shape == (batch_size, 1, self.VOCAB_SIZE), (
                f"Depth {depth}: expected logits shape ({batch_size}, 1, {self.VOCAB_SIZE}), "
                f"got {logits.shape}"
            )
            assert torch.all(
                torch.isfinite(logits)
            ), f"Depth {depth}: logits contain non-finite values"

    # ---- Test 6: eager fallback when no matching graph exists ------------- #

    @torch.inference_mode()
    def test_eager_fallback_no_matching_graph(self):
        """When ``use_mtp_cuda_graphs`` is True but no warmed graph matches the
        batch size, ``forward_single_position`` falls back to eager execution.
        The system should produce valid outputs without errors.
        """
        model = self._build_model()
        unwrapped = unwrap_model(model)
        # Warmup for batch_size=4 only.
        self._warmup_mtp_graphs(model, [4])
        self._set_mtp_cuda_graph_flag(model, True)

        # Run with batch_size=6 — no matching graph exists.
        batch_size = 6
        hidden = torch.randn(batch_size, 1, self.HIDDEN_SIZE, device='cuda')
        dist.broadcast(hidden, src=0)
        token_ids = torch.randint(0, self.VOCAB_SIZE, (1, batch_size), device='cuda')
        dist.broadcast(token_ids, src=0)
        position_ids = torch.arange(batch_size, device='cuda', dtype=torch.int64).unsqueeze(0)

        h_out, logits = unwrapped.compute_mtp_single_step(
            hidden_states=hidden.clone(),
            next_token_ids=token_ids.clone(),
            position_ids=position_ids.clone(),
            depth=0,
        )

        assert h_out.shape == (batch_size, 1, self.HIDDEN_SIZE)
        assert logits.shape == (batch_size, 1, self.VOCAB_SIZE)
        assert torch.all(torch.isfinite(logits))

    # ---- Test 7: graph flag propagation matches main model ---------------- #

    @torch.inference_mode()
    def test_mtp_graph_flag_propagation(self):
        """``use_mtp_cuda_graphs`` is correctly toggled via the helper and
        every MTP layer sees the same value.
        """
        model = self._build_model(mtp_num_layers=2)
        unwrapped = unwrap_model(model)

        self._set_mtp_cuda_graph_flag(model, True)
        for layer in unwrapped.mtp.layers:
            assert layer.use_mtp_cuda_graphs is True

        self._set_mtp_cuda_graph_flag(model, False)
        for layer in unwrapped.mtp.layers:
            assert layer.use_mtp_cuda_graphs is False


# --------------------------------------------------------------------------- #
#  TestMTPCudaGraphExpertParallel (EP = 2)
# --------------------------------------------------------------------------- #

_EP_SIZE = 2

# Request state constants for parametrized tests.
NONE = "none"
DECODE = "decode"
PREFILL = "prefill"
MIXED = "mixed"

ALL_STATES = [NONE, DECODE, PREFILL, MIXED]

# Combinatorial sweep: C(4+2-1, 2) = 10 test cases.
_STATE_COMBOS = list(itertools.combinations_with_replacement(ALL_STATES, _EP_SIZE))

# Batch dimensions for each non-dummy state.
_STATE_DIMS = {
    DECODE: InferenceBatchDimensions(token_count=2, prefill_req_count=0, decode_req_count=2),
    PREFILL: InferenceBatchDimensions(token_count=16, prefill_req_count=2, decode_req_count=0),
    MIXED: InferenceBatchDimensions(token_count=32, prefill_req_count=1, decode_req_count=2),
}


@pytest.mark.internal
class TestMTPCudaGraphExpertParallel:
    """Tests for MTP CUDA-graphed inference with expert parallelism.

    Follows the test pattern from ``test_mamba_model_expert_parallel_inference.py``.
    All tests require at least ``_EP_SIZE`` GPUs.
    """

    HIDDEN_SIZE = 32
    VOCAB_SIZE = 100
    MAX_SEQ_LEN = 128
    NUM_LAYERS = 2
    NUM_ATTN_HEADS = 4
    NUM_MOE_EXPERTS = 2

    def setup_method(self, method):
        if Utils.world_size < _EP_SIZE:
            pytest.skip(f"EP test requires at least {_EP_SIZE} GPUs")
        if Utils.world_size % _EP_SIZE != 0:
            pytest.skip(
                f"world_size ({Utils.world_size}) must be divisible by EP size ({_EP_SIZE})"
            )
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=_EP_SIZE,
        )

    def teardown_method(self, method):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()

    # ---- helpers ---------------------------------------------------------- #

    def _build_model(self):
        """Build a GPT model with MTP + MoE + local CUDA graphs."""
        model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
        config = TransformerConfig(
            num_layers=self.NUM_LAYERS,
            hidden_size=self.HIDDEN_SIZE,
            num_attention_heads=self.NUM_ATTN_HEADS,
            use_cpu_initialization=True,
            attention_backend=AttnBackend.local,
            params_dtype=torch.float32,
            expert_model_parallel_size=_EP_SIZE,
            num_moe_experts=self.NUM_MOE_EXPERTS,
            moe_token_dispatcher_type="alltoall",
            add_bias_linear=False,
            mtp_num_layers=2,
            cuda_graph_impl="local",
            moe_pad_experts_for_cuda_graph_inference=True,
        )
        layer_spec = get_gpt_layer_local_spec(num_experts=self.NUM_MOE_EXPERTS)
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=layer_spec, use_transformer_engine=False
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=self.VOCAB_SIZE,
            max_sequence_length=self.MAX_SEQ_LEN,
            parallel_output=True,
            pre_process=True,
            post_process=True,
            mtp_block_spec=mtp_block_spec,
        ).cuda()
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
        """Build a DynamicInferenceContext for the model."""
        return DynamicInferenceContext(
            model_config=model.config,
            inference_config=InferenceConfig(
                max_sequence_length=self.MAX_SEQ_LEN,
                buffer_size_gb=0.5,
                block_size_tokens=256,
                materialize_only_last_token_logits=False,
                num_cuda_graphs=num_cuda_graphs,
                use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
                max_requests=max_requests,
            ),
        )

    # ---- Test 1: all EP ranks run MTP eager forward ----------------------- #

    @pytest.mark.parametrize("batch_size", [2, 4, 8])
    @pytest.mark.internal
    @torch.inference_mode()
    def test_ep_mtp_eager_forward(self, batch_size):
        """All EP ranks can run MTP forward in eager mode.

        The MoE all-to-all collectives must match across EP ranks.  Verifies
        that all ranks complete without hanging and produce valid shapes.
        """
        model = self._build_model()
        unwrapped = unwrap_model(model)

        # Broadcast identical inputs so all EP ranks see the same data.
        hidden = torch.randn(batch_size, 1, self.HIDDEN_SIZE, device='cuda')
        dist.broadcast(hidden, src=0)
        token_ids = torch.randint(0, self.VOCAB_SIZE, (1, batch_size), device='cuda')
        dist.broadcast(token_ids, src=0)
        position_ids = torch.arange(batch_size, device='cuda', dtype=torch.int64).unsqueeze(0)

        h_out, logits = unwrapped.compute_mtp_single_step(
            hidden_states=hidden.clone(),
            next_token_ids=token_ids.clone(),
            position_ids=position_ids.clone(),
            depth=0,
        )

        assert h_out.shape == (batch_size, 1, self.HIDDEN_SIZE)
        assert logits.shape == (batch_size, 1, self.VOCAB_SIZE)
        assert torch.all(torch.isfinite(logits))

    # ---- Test 2: dummy ranks + real ranks in eager mode ------------------- #

    @pytest.mark.internal
    @torch.inference_mode()
    def test_ep_mtp_eager_dummy_and_real_ranks(self):
        """Even EP ranks run as dummy (with zeros), odd ranks run with real data.

        Both must issue matching MoE all-to-all collectives via the
        MTP eager forward to avoid hangs.
        """
        batch_size = 4
        model = self._build_model()
        unwrapped = unwrap_model(model)

        ep_rank = parallel_state.get_expert_model_parallel_rank()
        is_dummy = ep_rank % 2 == 0

        if is_dummy:
            hidden = torch.zeros(batch_size, 1, self.HIDDEN_SIZE, device='cuda')
            token_ids = torch.zeros(1, batch_size, device='cuda', dtype=torch.long)
        else:
            hidden = torch.randn(batch_size, 1, self.HIDDEN_SIZE, device='cuda')
            token_ids = torch.randint(0, self.VOCAB_SIZE, (1, batch_size), device='cuda')
        position_ids = torch.arange(batch_size, device='cuda', dtype=torch.int64).unsqueeze(0)

        # All ranks must complete without hanging.
        h_out, logits = unwrapped.compute_mtp_single_step(
            hidden_states=hidden, next_token_ids=token_ids, position_ids=position_ids, depth=0
        )

        assert h_out.shape == (batch_size, 1, self.HIDDEN_SIZE)
        assert logits.shape == (batch_size, 1, self.VOCAB_SIZE)

    # ---- Test 3: EP state cross product with DynamicInferenceContext ------- #

    @pytest.mark.parametrize("rank_states", _STATE_COMBOS, ids=[",".join(s) for s in _STATE_COMBOS])
    @pytest.mark.internal
    @torch.inference_mode()
    def test_ep_state_cross_product(self, rank_states):
        """Test combinatorial assignments of request states across EP ranks.

        Verifies that:
        - All EP ranks agree on CUDA graph usage (on or off).
        - When CUDA graphs are used, all ranks agree on the padded batch size
          (which would be used as the MTP batch dimension).
        """
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        my_state = rank_states[ep_rank]
        is_dummy = my_state == NONE

        model = self._build_model()
        ctx = self._build_context(model)

        # Phase 1: Set up each rank's request state.
        if not is_dummy:
            ctx.add_dummy_requests_for_cudagraph_capture(_STATE_DIMS[my_state])

        # Phase 2: Initialize attention state (EP collective).
        if is_dummy:
            ctx.initialize_attention_state(is_expert_parallel_dummy_cuda_graph_step=True)
        else:
            ctx.initialize_attention_state()

        # Phase 3: Verify EP agreement on CUDA graph usage.
        uses_graph = ctx.using_cuda_graph_this_step()
        ep_group = parallel_state.get_expert_model_parallel_group()
        uses_graph_t = torch.tensor([int(uses_graph)], device='cuda', dtype=torch.int32)
        graph_min = uses_graph_t.clone()
        graph_max = uses_graph_t.clone()
        dist.all_reduce(graph_min, op=dist.ReduceOp.MIN, group=ep_group)
        dist.all_reduce(graph_max, op=dist.ReduceOp.MAX, group=ep_group)
        assert graph_min.item() == graph_max.item(), (
            f"CUDA graph usage disagrees across EP ranks: "
            f"min={graph_min.item()}, max={graph_max.item()} "
            f"(rank_states={rank_states})"
        )

        if not uses_graph:
            return

        # Phase 4: Derive MTP padded batch size from EP-synced dimensions.
        mtp_padded = ctx.padded_batch_dimensions.req_count

        # Verify MTP padded count agrees across EP ranks.
        padded_t = torch.tensor([mtp_padded], dtype=torch.int32, device='cuda')
        padded_max = padded_t.clone()
        padded_min = padded_t.clone()
        dist.all_reduce(padded_max, op=dist.ReduceOp.MAX, group=ep_group)
        dist.all_reduce(padded_min, op=dist.ReduceOp.MIN, group=ep_group)
        assert padded_max.item() == padded_min.item(), (
            f"MTP padded batch size mismatch across EP ranks: "
            f"min={padded_min.item()}, max={padded_max.item()} "
            f"(rank_states={rank_states})"
        )

    # ---- Test 4: dummy EP rank bail-out with decode-only CUDA graphs ------ #

    @pytest.mark.parametrize(
        "peer_state", [PREFILL, MIXED], ids=[f"peer={s}" for s in [PREFILL, MIXED]]
    )
    @pytest.mark.internal
    @torch.inference_mode()
    def test_ep_dummy_bailout_with_decode_only_cuda_graphs(self, peer_state):
        """Verify the dummy-rank bail-out path when only decode CUDA graphs
        are available.

        With ``use_cuda_graphs_for_non_decode_steps=False``, only decode-only
        graphs exist. When any EP rank has prefill requests, no graph matches
        and all ranks fall back to eager mode.  The MTP forward for the dummy
        rank must use eager execution without hanging.
        """
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        is_even = ep_rank % 2 == 0

        model = self._build_model()
        ctx = self._build_context(model, use_cuda_graphs_for_non_decode_steps=False)

        # Even ranks are dummy; odd ranks have the peer_state.
        if not is_even:
            ctx.add_dummy_requests_for_cudagraph_capture(_STATE_DIMS[peer_state])

        if is_even:
            ctx.initialize_attention_state(is_expert_parallel_dummy_cuda_graph_step=True)
        else:
            ctx.initialize_attention_state()

        # No rank should match a CUDA graph.
        assert not ctx.using_cuda_graph_this_step(), (
            f"EP rank {ep_rank}: expected no CUDA graph match with "
            f"decode-only graphs and peer_state={peer_state}"
        )

        # MTP eager forward should still work on all ranks.
        unwrapped = unwrap_model(model)

        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        dummy_hidden = torch.zeros((tp_size, 1, self.HIDDEN_SIZE), device='cuda')
        dummy_tokens = torch.zeros((1, tp_size), device='cuda', dtype=torch.long)
        dummy_positions = torch.zeros((1, tp_size), device='cuda', dtype=torch.long)

        h_out, logits = unwrapped.compute_mtp_single_step(
            hidden_states=dummy_hidden,
            next_token_ids=dummy_tokens,
            position_ids=dummy_positions,
            depth=0,
        )

        assert h_out.shape == (tp_size, 1, self.HIDDEN_SIZE)
        assert logits.shape == (tp_size, 1, self.VOCAB_SIZE)
