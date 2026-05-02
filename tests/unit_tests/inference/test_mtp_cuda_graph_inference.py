# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for CUDA-graphed MTP (Multi-Token Prediction) inference.

Verifies that:
1. CUDA graph replay produces the same output as eager execution (no extra
   padding in the CUDA graphed case).
2. CUDA graphs work correctly with sequence parallelism (padding is applied
   to make batch sizes divisible by TP).
3. CUDA graphs work correctly with expert parallelism and dummy ranks.

Uses DynamicInferenceEngine for CUDA graph warmup so MTP graph capture
logic matches production code exactly.
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
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
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
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import unwrap_model
from tests.unit_tests.test_utilities import Utils

# --------------------------------------------------------------------------- #
#  TestMTPCudaGraphInference (TP = 2)
# --------------------------------------------------------------------------- #


class TestMTPCudaGraphInference:
    """Tests for MTP CUDA-graphed inference with tensor parallelism.

    All tests require at least 2 GPUs (TP = 2).  Uses DynamicInferenceEngine
    for CUDA graph warmup so MTP graph capture matches production code.
    """

    HIDDEN_SIZE = 32
    VOCAB_SIZE = 100
    MAX_SEQ_LEN = 64
    NUM_LAYERS = 4
    NUM_ATTN_HEADS = 4
    TP_SIZE = 2

    @classmethod
    def setup_class(cls):
        if Utils.world_size < cls.TP_SIZE:
            pytest.skip(f"Need at least {cls.TP_SIZE} GPUs")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=cls.TP_SIZE, pipeline_model_parallel_size=1
        )

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()

    def teardown_method(self):
        delete_cuda_graphs()

    # ---- helpers ---------------------------------------------------------- #

    def _build_model(
        self, *, sequence_parallel=False, mtp_num_layers=2, mtp_use_repeated_layer=False
    ):
        """Build a GPT model with MTP layers and local CUDA graph support."""
        model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
        config = TransformerConfig(
            num_layers=self.NUM_LAYERS,
            hidden_size=self.HIDDEN_SIZE,
            num_attention_heads=self.NUM_ATTN_HEADS,
            use_cpu_initialization=True,
            attention_backend=AttnBackend.local,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=self.TP_SIZE,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            mtp_num_layers=mtp_num_layers,
            mtp_use_repeated_layer=mtp_use_repeated_layer,
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
        for param in model.parameters():
            param.data = param.data.to(config.params_dtype)
        model.eval()
        return model

    def _build_engine(
        self,
        *,
        sequence_parallel=False,
        mtp_num_layers=2,
        mtp_use_repeated_layer=False,
        num_speculative_tokens=2,
        max_requests=16,
    ):
        """Build a DynamicInferenceEngine with automatic MTP CUDA graph warmup.

        The engine's `__init__` calls `create_cuda_graphs()` which captures
        both decoder and MTP CUDA graphs, matching production warmup exactly.
        """
        delete_cuda_graphs()
        model = self._build_model(
            sequence_parallel=sequence_parallel,
            mtp_num_layers=mtp_num_layers,
            mtp_use_repeated_layer=mtp_use_repeated_layer,
        )
        config = model.config
        context = DynamicInferenceContext(
            model_config=config,
            inference_config=InferenceConfig(
                max_sequence_length=self.MAX_SEQ_LEN,
                buffer_size_gb=0.5,
                materialize_only_last_token_logits=False,
                num_speculative_tokens=num_speculative_tokens,
                block_size_tokens=256,
                max_requests=max_requests,
                num_cuda_graphs=-1,
            ),
        )
        wrapped = GPTInferenceWrapper(model, context)
        wrapped.model_is_pipeline_parallel = False
        mock_tokenizer = mock.Mock()
        ctrl = TextGenerationController(inference_wrapped_model=wrapped, tokenizer=mock_tokenizer)
        engine = DynamicInferenceEngine(ctrl, context)
        return engine

    @staticmethod
    def _get_mtp_warmed_batch_sizes(engine):
        """Return the MTP batch sizes (padded req_counts) warmed by the engine.

        These are the `n` values for which MTP CUDA graphs were captured.
        Hidden states shape is `[n // tp, 1, H]` with SP, `[n, 1, H]` without.
        Token/position IDs are always `[1, n]`.
        """
        context = engine.context
        model_config = engine.controller.inference_wrapped_model.model.config
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        sp_enabled = model_config.sequence_parallel and tp_size > 1
        sizes = set()
        for dim in context.cuda_graph_batch_dimensions_list:
            n = dim.req_count
            if sp_enabled:
                n += (tp_size - n % tp_size) % tp_size
            if n > 0:
                sizes.add(n)
        return sorted(sizes)

    @staticmethod
    def _mtp_kwargs(use_graph, batch_size, mtp_depth):
        """Construct call-site kwargs that route `compute_mtp_single_step` to
        either CUDA graph replay or eager execution.

        The wrapped `compute_mtp_single_step` honors `eager=True` to bypass the
        manager and `cache_key=...` for O(1) runner lookup.
        """
        if use_graph:
            return {"cache_key": ("mtp", batch_size, mtp_depth)}
        return {"eager": True}

    @staticmethod
    def _assert_mtp_cuda_graphs_were_replayed(model, expect_replayed):
        """Assert that MTP CUDA graphs were (or were not) replayed.

        MTP runners are stored in the CudaGraphManager's lookup table
        rather than the global inference record.  A runner with
        `fwd_graph_recorded=True` confirms the graph was captured and
        replayed.
        """
        unwrapped = unwrap_model(model)
        manager = getattr(unwrapped, '_mtp_cudagraph_manager', None)
        if manager is None:
            assert not expect_replayed, "No MTP CudaGraphManager found on the model"
            return
        table = manager.custom_cudagraphs_lookup_table
        mtp_runners = [v for k, v in table.items() if isinstance(k, tuple) and k[0] == 'mtp']
        if expect_replayed:
            assert (
                len(mtp_runners) > 0
            ), "Expected MTP CUDA graphs to be replayed, but no MTP runners found"
            for runner in mtp_runners:
                assert runner.fwd_graph_recorded, (
                    "Expected MTP CUDA graph to be recorded and replayed, "
                    f"but runner for {runner.base_module.__class__.__name__} "
                    "has fwd_graph_recorded=False"
                )
        else:
            recorded = [r for r in mtp_runners if r.fwd_graph_recorded]
            assert len(recorded) == 0, (
                f"Expected no MTP CUDA graph replay, but {len(recorded)} "
                "runners have fwd_graph_recorded=True"
            )

    # ---- Test 1: graph output matches eager (no additional padding) ------- #

    @pytest.mark.parametrize("mtp_use_repeated_layer", [False, True])
    @torch.inference_mode()
    def test_cuda_graph_output_matches_eager(self, mtp_use_repeated_layer):
        """CUDA graph replay produces the same output as eager execution.

        The batch sizes exactly match warmed-up graphs (from the engine's
        CUDA graph warmup), so there is no additional padding.  Both paths
        must produce identical hidden states and logits.
        """
        engine = self._build_engine(mtp_use_repeated_layer=mtp_use_repeated_layer)
        model = engine.controller.inference_wrapped_model.model
        unwrapped = unwrap_model(model)
        batch_sizes = self._get_mtp_warmed_batch_sizes(engine)
        assert len(batch_sizes) > 0, "Engine did not warm up any MTP CUDA graphs"

        mtp_depth = None if unwrapped.mtp.mtp_use_repeated_layer else 0

        for batch_size in batch_sizes[:3]:
            hidden = torch.randn(
                batch_size, 1, self.HIDDEN_SIZE, device='cuda', dtype=torch.bfloat16
            )
            dist.broadcast(hidden, src=0)
            token_ids = torch.randint(0, self.VOCAB_SIZE, (1, batch_size), device='cuda')
            dist.broadcast(token_ids, src=0)
            position_ids = torch.arange(batch_size, device='cuda', dtype=torch.int64).unsqueeze(0)

            h_graph, logits_graph = unwrapped.compute_mtp_single_step(
                hidden_states=hidden.clone(),
                next_token_ids=token_ids.clone(),
                position_ids=position_ids.clone(),
                depth=mtp_depth,
                **self._mtp_kwargs(use_graph=True, batch_size=batch_size, mtp_depth=mtp_depth),
            )
            h_graph = h_graph.clone()
            logits_graph = logits_graph.clone()

            h_eager, logits_eager = unwrapped.compute_mtp_single_step(
                hidden_states=hidden.clone(),
                next_token_ids=token_ids.clone(),
                position_ids=position_ids.clone(),
                depth=mtp_depth,
                **self._mtp_kwargs(use_graph=False, batch_size=batch_size, mtp_depth=mtp_depth),
            )

            torch.testing.assert_close(
                h_graph, h_eager, msg=f"Hidden mismatch at batch_size={batch_size}"
            )
            torch.testing.assert_close(
                logits_graph, logits_eager, msg=f"Logits mismatch at batch_size={batch_size}"
            )

        self._assert_mtp_cuda_graphs_were_replayed(model, True)

    # ---- Test 2: graph matches eager with sequence parallelism ------------ #

    @pytest.mark.parametrize("mtp_use_repeated_layer", [False, True])
    @torch.inference_mode()
    def test_cuda_graph_output_matches_eager_with_sp(self, mtp_use_repeated_layer):
        """CUDA graph replay matches eager with sequence parallelism.

        Hidden states are in scattered SP format `[batch_size/TP, 1, H]`.
        Token/position IDs remain at full `[1, batch_size]`.  Both paths
        must produce identical outputs.
        """
        engine = self._build_engine(
            sequence_parallel=True, mtp_use_repeated_layer=mtp_use_repeated_layer
        )
        model = engine.controller.inference_wrapped_model.model
        unwrapped = unwrap_model(model)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        batch_sizes = self._get_mtp_warmed_batch_sizes(engine)
        assert len(batch_sizes) > 0, "Engine did not warm up any MTP CUDA graphs"

        mtp_depth = None if unwrapped.mtp.mtp_use_repeated_layer else 0

        for batch_size in batch_sizes[:3]:
            hidden = torch.randn(
                batch_size, 1, self.HIDDEN_SIZE, device='cuda', dtype=torch.bfloat16
            )
            dist.broadcast(hidden, src=0)
            hidden_sp = scatter_to_sequence_parallel_region(hidden, group=tp_group)

            token_ids = torch.randint(0, self.VOCAB_SIZE, (1, batch_size), device='cuda')
            dist.broadcast(token_ids, src=0)
            position_ids = torch.arange(batch_size, device='cuda', dtype=torch.int64).unsqueeze(0)

            h_graph, logits_graph = unwrapped.compute_mtp_single_step(
                hidden_states=hidden_sp.clone(),
                next_token_ids=token_ids.clone(),
                position_ids=position_ids.clone(),
                depth=mtp_depth,
                **self._mtp_kwargs(use_graph=True, batch_size=batch_size, mtp_depth=mtp_depth),
            )
            h_graph = h_graph.clone()
            logits_graph = logits_graph.clone()

            h_eager, logits_eager = unwrapped.compute_mtp_single_step(
                hidden_states=hidden_sp.clone(),
                next_token_ids=token_ids.clone(),
                position_ids=position_ids.clone(),
                depth=mtp_depth,
                **self._mtp_kwargs(use_graph=False, batch_size=batch_size, mtp_depth=mtp_depth),
            )

            torch.testing.assert_close(
                h_graph, h_eager, msg=f"Hidden mismatch at batch_size={batch_size}"
            )
            torch.testing.assert_close(
                logits_graph, logits_eager, msg=f"Logits mismatch at batch_size={batch_size}"
            )

        self._assert_mtp_cuda_graphs_were_replayed(model, True)

    # ---- Test 3: end-to-end _compute_serial_mtp_and_sample with SP ------- #

    @pytest.mark.parametrize("mtp_use_repeated_layer", [False, True])
    @torch.inference_mode()
    def test_cuda_graph_sp_padding_end_to_end(self, mtp_use_repeated_layer):
        """Full `_compute_serial_mtp_and_sample` with CUDA graphs and SP.

        Active request counts that are not multiples of TP are padded.
        The engine's CUDA graph warmup pre-captures MTP graphs for the
        padded batch sizes.  Verifies that padding, SP scatter/gather, and
        MTP forward all work correctly through the CUDA graph path.
        """
        tp_size = self.TP_SIZE
        num_spec = 2
        max_requests = 16
        engine = self._build_engine(
            sequence_parallel=True,
            mtp_num_layers=num_spec,
            mtp_use_repeated_layer=mtp_use_repeated_layer,
            num_speculative_tokens=num_spec,
            max_requests=max_requests,
        )
        ctrl = engine.controller
        context = engine.context
        model = ctrl.inference_wrapped_model.model
        unwrapped = unwrap_model(model)

        mtp_sizes = self._get_mtp_warmed_batch_sizes(engine)

        # Find active_request_counts whose TP-padded values match warmed MTP sizes.
        active_counts = []
        for n in mtp_sizes:
            for active in range(n, 0, -1):
                padded = active + (tp_size - active % tp_size) % tp_size
                if padded == n and active <= max_requests:
                    active_counts.append(active)
                    break
        assert len(active_counts) > 0, "No valid active request counts found"

        for active_request_count in active_counts[:4]:
            padded_count = (
                active_request_count + (tp_size - active_request_count % tp_size) % tp_size
            )

            context.reset()
            context.total_request_count = active_request_count
            context.paused_request_count = 0
            context.request_kv_length_offsets[:active_request_count] = torch.arange(
                active_request_count, dtype=torch.int32, device='cuda'
            )
            context.request_query_lengths[:active_request_count] = torch.ones(
                active_request_count, dtype=torch.int32, device='cuda'
            )

            ctrl.num_speculative_tokens = num_spec
            ctrl.num_mtp_heads = num_spec
            ctrl._init_mtp_sampling_tensors()
            ctrl._mtp_token_ids_buf.zero_()
            ctrl._mtp_position_ids_buf.zero_()
            ctrl._sampled_tokens_cuda[:active_request_count] = torch.remainder(
                torch.arange(active_request_count, device='cuda'), self.VOCAB_SIZE
            )

            tp_rank = parallel_state.get_tensor_model_parallel_rank()

            torch.manual_seed(42)
            full_hidden = torch.randn(
                padded_count, 1, self.HIDDEN_SIZE, device='cuda', dtype=torch.bfloat16
            )
            dist.broadcast(full_hidden, src=0)
            local_hidden = full_hidden.chunk(tp_size)[tp_rank].contiguous()
            unwrapped._decoder_hidden_states_cache = local_hidden

            ctrl._last_accepted_seq_indices = torch.arange(active_request_count, device='cuda')
            ctrl._mtp_resolved_padded_count = padded_count
            context._using_cuda_graph_this_step = True

            ctrl._torch_sampling_buckets = [(list(range(active_request_count)), 1.0, 1, 0.0)]
            ctrl._torch_sampling_bucket_index_tensors = [
                torch.arange(active_request_count, device='cuda', dtype=torch.long)
            ]

            ctrl._compute_serial_mtp_and_sample()

            for depth in range(num_spec):
                sampled = ctrl._sampled_mtp_tokens_cuda[depth, :active_request_count]
                assert sampled.shape == (
                    active_request_count,
                ), f"active={active_request_count}, depth={depth}"
                assert sampled.dtype == torch.int64
                assert torch.all(sampled >= 0) and torch.all(sampled < self.VOCAB_SIZE)

            assert not hasattr(unwrapped, '_decoder_hidden_states_cache')

        self._assert_mtp_cuda_graphs_were_replayed(model, True)

    # ---- Test 4: SP padding graph vs eager produces same MTP tokens ------- #

    @pytest.mark.parametrize("mtp_use_repeated_layer", [False, True])
    @torch.inference_mode()
    def test_cuda_graph_sp_padding_matches_eager(self, mtp_use_repeated_layer):
        """With SP padding, CUDA graph path produces the same MTP tokens as eager.

        Uses a single engine (shared model weights) and toggles the CUDA
        graph flag between runs.  Both paths receive identical inputs and
        must produce the same sampled MTP tokens.
        """
        tp_size = self.TP_SIZE
        num_spec = 2
        max_requests = 16
        engine = self._build_engine(
            sequence_parallel=True,
            mtp_num_layers=num_spec,
            mtp_use_repeated_layer=mtp_use_repeated_layer,
            num_speculative_tokens=num_spec,
            max_requests=max_requests,
        )
        ctrl = engine.controller
        context = engine.context
        model = ctrl.inference_wrapped_model.model

        mtp_sizes = self._get_mtp_warmed_batch_sizes(engine)

        # Find active counts that require TP padding (active % tp != 0).
        active_counts = []
        for n in mtp_sizes:
            for active in range(n, 0, -1):
                padded = active + (tp_size - active % tp_size) % tp_size
                if padded == n and active % tp_size != 0 and active <= max_requests:
                    active_counts.append(active)
                    break
        assert len(active_counts) > 0, "No active counts with TP padding found"

        for active_request_count in active_counts[:2]:
            padded_count = (
                active_request_count + (tp_size - active_request_count % tp_size) % tp_size
            )

            def _run_mtp(use_cuda_graph):
                """Set up state and run MTP, returning sampled tokens."""
                unwrapped = unwrap_model(model)
                context.reset()
                context.total_request_count = active_request_count
                context.paused_request_count = 0
                context.request_kv_length_offsets[:active_request_count] = torch.arange(
                    active_request_count, dtype=torch.int32, device='cuda'
                )
                context.request_query_lengths[:active_request_count] = torch.ones(
                    active_request_count, dtype=torch.int32, device='cuda'
                )

                ctrl.num_speculative_tokens = num_spec
                ctrl.num_mtp_heads = num_spec
                ctrl._init_mtp_sampling_tensors()
                ctrl._mtp_token_ids_buf.zero_()
                ctrl._mtp_position_ids_buf.zero_()
                ctrl._sampled_tokens_cuda[:active_request_count] = torch.remainder(
                    torch.arange(active_request_count, device='cuda'), self.VOCAB_SIZE
                )

                if use_cuda_graph:
                    ctrl._mtp_resolved_padded_count = padded_count
                    context._using_cuda_graph_this_step = True
                else:
                    ctrl._mtp_resolved_padded_count = None
                    context._using_cuda_graph_this_step = False

                tp_rank = parallel_state.get_tensor_model_parallel_rank()

                torch.manual_seed(42)
                full_hidden = torch.randn(
                    padded_count, 1, self.HIDDEN_SIZE, device='cuda', dtype=torch.bfloat16
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
            self._assert_mtp_cuda_graphs_were_replayed(model, True)
            eager_tokens = _run_mtp(use_cuda_graph=False)

            for depth in range(num_spec):
                assert torch.equal(graph_tokens[depth], eager_tokens[depth]), (
                    f"active={active_request_count}, depth={depth}: "
                    f"graph tokens {graph_tokens[depth].tolist()} != "
                    f"eager tokens {eager_tokens[depth].tolist()}"
                )

    # ---- Test 5: multiple MTP depths with CUDA graphs --------------------- #

    @pytest.mark.parametrize("mtp_use_repeated_layer", [False, True])
    @torch.inference_mode()
    def test_cuda_graph_multi_depth(self, mtp_use_repeated_layer):
        """Run multiple MTP depths with CUDA graphs enabled.

        Verifies that the hidden output from one depth feeds correctly into
        the next depth through the same CUDA graph, producing valid outputs
        at every depth.
        """
        num_depths = 2
        engine = self._build_engine(
            mtp_num_layers=num_depths, mtp_use_repeated_layer=mtp_use_repeated_layer
        )
        model = engine.controller.inference_wrapped_model.model
        unwrapped = unwrap_model(model)
        batch_sizes = self._get_mtp_warmed_batch_sizes(engine)
        assert len(batch_sizes) > 0, "Engine did not warm up any MTP CUDA graphs"

        use_repeated = unwrapped.mtp.mtp_use_repeated_layer

        batch_size = batch_sizes[0]

        hidden = torch.randn(batch_size, 1, self.HIDDEN_SIZE, device='cuda', dtype=torch.bfloat16)
        dist.broadcast(hidden, src=0)
        token_ids = torch.randint(0, self.VOCAB_SIZE, (1, batch_size), device='cuda')
        dist.broadcast(token_ids, src=0)
        position_ids = torch.arange(batch_size, device='cuda', dtype=torch.int64).unsqueeze(0)

        current_hidden = hidden.clone()
        for depth in range(num_depths):
            mtp_depth = None if use_repeated else depth
            current_hidden, logits = unwrapped.compute_mtp_single_step(
                hidden_states=current_hidden,
                next_token_ids=token_ids.clone(),
                position_ids=position_ids.clone(),
                depth=mtp_depth,
                **self._mtp_kwargs(use_graph=True, batch_size=batch_size, mtp_depth=mtp_depth),
            )
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

        self._assert_mtp_cuda_graphs_were_replayed(model, True)

    # ---- Test 6: caller-driven eager bypass for non-warmed shapes --------- #

    @pytest.mark.parametrize("mtp_use_repeated_layer", [False, True])
    @torch.inference_mode()
    def test_eager_bypass_for_non_warmed_shape(self, mtp_use_repeated_layer):
        """Passing `eager=True` runs `compute_mtp_single_step` outside the
        CudaGraphManager wrapper. This is the canonical caller-side fallback
        for a shape that warmup did not capture.
        """
        engine = self._build_engine(mtp_use_repeated_layer=mtp_use_repeated_layer)
        model = engine.controller.inference_wrapped_model.model
        unwrapped = unwrap_model(model)
        warmed_sizes = set(self._get_mtp_warmed_batch_sizes(engine))

        # Find a batch size with no matching CUDA graph.
        fallback_size = None
        for candidate in range(1, 32):
            if candidate not in warmed_sizes:
                fallback_size = candidate
                break
        assert fallback_size is not None, "Could not find a non-warmed batch size"

        mtp_depth = None if unwrapped.mtp.mtp_use_repeated_layer else 0

        hidden = torch.randn(
            fallback_size, 1, self.HIDDEN_SIZE, device='cuda', dtype=torch.bfloat16
        )
        dist.broadcast(hidden, src=0)
        token_ids = torch.randint(0, self.VOCAB_SIZE, (1, fallback_size), device='cuda')
        dist.broadcast(token_ids, src=0)
        position_ids = torch.arange(fallback_size, device='cuda', dtype=torch.int64).unsqueeze(0)

        h_out, logits = unwrapped.compute_mtp_single_step(
            hidden_states=hidden.clone(),
            next_token_ids=token_ids.clone(),
            position_ids=position_ids.clone(),
            depth=mtp_depth,
            eager=True,
        )

        assert h_out.shape == (fallback_size, 1, self.HIDDEN_SIZE)
        assert logits.shape == (fallback_size, 1, self.VOCAB_SIZE)
        assert torch.all(torch.isfinite(logits))

    # ---- Test 7: delete_cuda_graphs resets MTP runners -------------------- #

    @torch.inference_mode()
    def test_delete_cuda_graphs_resets_mtp_runners(self):
        """`delete_cuda_graphs()` resets MTP CUDA graph runners.

        MTP runners join the standard `cudagraph_inference_record`, so the
        standard cleanup loop resets their `fwd_graph_recorded` flag.
        """
        engine = self._build_engine()
        model = engine.controller.inference_wrapped_model.model

        self._assert_mtp_cuda_graphs_were_replayed(model, True)

        unwrapped = unwrap_model(model)
        manager = getattr(unwrapped, '_mtp_cudagraph_manager', None)
        assert manager is not None
        assert len(manager.cudagraph_runners) > 0
        assert all(r.fwd_graph_recorded for r in manager.cudagraph_runners)

        delete_cuda_graphs()

        assert all(not r.fwd_graph_recorded for r in manager.cudagraph_runners)
        assert all(r.fwd_graph is None for r in manager.cudagraph_runners)

    # ---- Test 8: last_token_logits under CUDA graph padding ---------------- #

    @torch.inference_mode()
    def test_last_token_logits_cuda_graph_padding(self):
        """num_last_token_logits returns padded count and last_token_logits
        produces the correct shape under CUDA graph padding.

        Uses add_request + update_requests to build real decode batches, then
        verifies that under CUDA graph matching:
        1. num_last_token_logits uses the padded decode count from the matched graph
        2. last_token_logits returns the padded number of rows
        3. The real (unpadded) index positions are sequential 0..N-1
        """
        num_spec = 2
        max_requests = 16
        engine = self._build_engine(num_speculative_tokens=num_spec, max_requests=max_requests)
        context = engine.context
        tokens_per_decode = num_spec + 1

        # Collect decode-only graph sizes to pick active counts that will match.
        decode_graph_sizes = sorted(
            {
                dim.decode_req_count
                for dim in context.cuda_graph_batch_dimensions_list
                if dim.prefill_req_count == 0 and dim.decode_req_count > 1
            }
        )
        assert len(decode_graph_sizes) > 0, "No decode-only graph dims found"

        # Use active counts 1 less than some graph sizes to guarantee padding.
        active_counts = [s - 1 for s in decode_graph_sizes if s >= 2][:3]
        assert len(active_counts) > 0, "No sub-capacity decode graph dims found"

        for active_decode_count in active_counts:
            context.reset()

            # Add prefill requests, then step them into decode state.
            prompt_length = 10
            for i in range(active_decode_count):
                req = DynamicInferenceRequest(
                    request_id=i,
                    prompt_tokens=torch.arange(prompt_length, device='cuda'),
                    sampling_params=SamplingParams(num_tokens_to_generate=100),
                )
                context.add_request(req)

            context.initialize_attention_state()

            active_mask = torch.ones(active_decode_count, device='cuda', dtype=torch.int32)
            new_tokens = torch.arange(active_decode_count, device='cuda')
            new_spec = torch.arange(num_spec * active_decode_count, device='cuda').reshape(
                num_spec, active_decode_count
            )
            context.update_requests(
                active_requests_mask=active_mask,
                new_tokens=new_tokens,
                new_speculative_tokens=new_spec,
            )

            # Now all requests are decode. initialize_attention_state should match a graph.
            context.initialize_attention_state()

            assert (
                context.using_cuda_graph_this_step()
            ), f"Expected CUDA graph for active={active_decode_count}"

            # Read the actually matched graph dimensions.
            matched = context.padded_batch_dimensions
            padded_decode = matched.decode_req_count
            padded_token_count = matched.token_count
            assert padded_decode >= active_decode_count

            expected_padded_logits = padded_decode * tokens_per_decode
            assert context.num_last_token_logits == expected_padded_logits, (
                f"active={active_decode_count}, padded={padded_decode}: "
                f"num_last_token_logits expected {expected_padded_logits}, "
                f"got {context.num_last_token_logits}"
            )

            # Verify the real decode indices are [0, 1, ..., real_token_count - 1].
            real_token_count = active_decode_count * tokens_per_decode
            real_slice = context.active_logit_idxs[:real_token_count]
            expected_real = torch.arange(real_token_count, dtype=torch.int32, device='cuda')
            assert torch.equal(
                real_slice, expected_real
            ), f"real decode indices: {real_slice.tolist()} vs {expected_real.tolist()}"

            # Padding indices should be zero (indexing into logits[0]).
            padding_count = expected_padded_logits - real_token_count
            if padding_count > 0:
                padding_slice = context.active_logit_idxs[real_token_count:expected_padded_logits]
                assert (
                    padding_slice.sum().item() == 0
                ), f"padding indices should be zero, got {padding_slice.tolist()}"

            # Verify last_token_logits produces a tensor with the padded row count.
            vocab_size = 64
            fake_logits = torch.randn(
                1, padded_token_count, vocab_size, device='cuda', dtype=torch.float32
            )
            result = context.last_token_logits(fake_logits)
            assert result.shape == (expected_padded_logits, vocab_size), (
                f"last_token_logits shape: expected ({expected_padded_logits}, {vocab_size}), "
                f"got {result.shape}"
            )


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

    Follows the test pattern from `test_mamba_model_expert_parallel_inference.py`.
    All tests require at least `_EP_SIZE` GPUs.
    """

    HIDDEN_SIZE = 32
    VOCAB_SIZE = 100
    MAX_SEQ_LEN = 128
    NUM_LAYERS = 2
    NUM_ATTN_HEADS = 4
    NUM_MOE_EXPERTS = 2

    @classmethod
    def setup_class(cls):
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

    @classmethod
    def teardown_class(cls):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()

    def teardown_method(self):
        delete_cuda_graphs()

    # ---- helpers ---------------------------------------------------------- #

    def _build_model(self, inference_moe_token_dispatcher_type='nccl'):
        """Build a GPT model with MTP + MoE + local CUDA graphs."""
        model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
        config = TransformerConfig(
            num_layers=self.NUM_LAYERS,
            hidden_size=self.HIDDEN_SIZE,
            num_attention_heads=self.NUM_ATTN_HEADS,
            use_cpu_initialization=True,
            attention_backend=AttnBackend.local,
            params_dtype=torch.bfloat16,
            expert_model_parallel_size=_EP_SIZE,
            num_moe_experts=self.NUM_MOE_EXPERTS,
            moe_token_dispatcher_type="alltoall",
            add_bias_linear=False,
            mtp_num_layers=2,
            cuda_graph_impl="local",
            moe_pad_experts_for_cuda_graph_inference=True,
            inference_moe_token_dispatcher_type=inference_moe_token_dispatcher_type,
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
        for param in model.parameters():
            param.data = param.data.to(config.params_dtype)
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
        hidden = torch.randn(batch_size, 1, self.HIDDEN_SIZE, device='cuda', dtype=torch.bfloat16)
        dist.broadcast(hidden, src=0)
        token_ids = torch.randint(0, self.VOCAB_SIZE, (1, batch_size), device='cuda')
        dist.broadcast(token_ids, src=0)
        position_ids = torch.arange(batch_size, device='cuda', dtype=torch.int64).unsqueeze(0)

        h_out, logits = unwrapped.compute_mtp_single_step(
            hidden_states=hidden.clone(),
            next_token_ids=token_ids.clone(),
            position_ids=position_ids.clone(),
            depth=0,
            eager=True,
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
            hidden = torch.zeros(
                batch_size, 1, self.HIDDEN_SIZE, device='cuda', dtype=torch.bfloat16
            )
            token_ids = torch.zeros(1, batch_size, device='cuda', dtype=torch.long)
        else:
            hidden = torch.randn(
                batch_size, 1, self.HIDDEN_SIZE, device='cuda', dtype=torch.bfloat16
            )
            token_ids = torch.randint(0, self.VOCAB_SIZE, (1, batch_size), device='cuda')
        position_ids = torch.arange(batch_size, device='cuda', dtype=torch.int64).unsqueeze(0)

        # All ranks must complete without hanging.
        h_out, logits = unwrapped.compute_mtp_single_step(
            hidden_states=hidden,
            next_token_ids=token_ids,
            position_ids=position_ids,
            depth=0,
            eager=True,
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
    def test_nccl_ep_dummy_bailout_with_decode_only_cuda_graphs(self, peer_state):
        """Verify the dummy-rank bail-out path when only decode CUDA graphs
        are available.

        With `use_cuda_graphs_for_non_decode_steps=False`, only decode-only
        graphs exist. When any EP rank has prefill requests, no graph matches
        and all ranks fall back to eager mode.  The MTP forward for the dummy
        rank must use eager execution without hanging.
        """
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        is_even = ep_rank % 2 == 0

        model = self._build_model(inference_moe_token_dispatcher_type='nccl')
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
        dummy_hidden = torch.zeros(
            (tp_size, 1, self.HIDDEN_SIZE), device='cuda', dtype=torch.bfloat16
        )
        dummy_tokens = torch.zeros((1, tp_size), device='cuda', dtype=torch.long)
        dummy_positions = torch.zeros((1, tp_size), device='cuda', dtype=torch.long)

        h_out, logits = unwrapped.compute_mtp_single_step(
            hidden_states=dummy_hidden,
            next_token_ids=dummy_tokens,
            position_ids=dummy_positions,
            depth=0,
            eager=True,
        )

        assert h_out.shape == (tp_size, 1, self.HIDDEN_SIZE)
        assert logits.shape == (tp_size, 1, self.VOCAB_SIZE)
