# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the MTP draft-KV bookkeeping on `DynamicInferenceContext`.

The MTP KV cache reserves one extra attention-layer slot in the shared KV buffer and drives its
own attention metadata (bypassing the coalesced CPU->GPU bookkeeping transfer) so the MTP draft
forwards never disturb the main step's state. The methods under test are pure GPU tensor
bookkeeping -- no model forward is involved -- so they are exercised directly against a real
context with hand-seeded per-request state:

  * `_mtp_begin_decode` -- enter MTP-forward mode for a draft loop
  * `_mtp_setup_decode_step` -- one draft depth (roll-by-one)
  * `_mtp_setup_prefill_step` -- the varlen commit pass

The lifecycle itself (`begin_decode_for_capture`, `advance_decode_step`, `end_forward`)
lives on `MTPMetadata` and is driven directly; only the steps that build metadata have a
wrapper here.

The invariants asserted are the ones the draft attention depends on: write position
`P_r = base_position_r - 1 + depth`, read length `kv_len = P_r + 1` (write-then-attend), and
padding rows that never index real KV.
"""

import types

import pytest
import torch

from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.disaggregation.decode_admission import admit_prefilled_decode
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# Small, fixed geometry: block_size_tokens=8 keeps block-boundary crossings reachable with
# two-digit positions, and max_sequence_length=64 gives max_kv_block_count=8 block columns.
BLOCK_SIZE_TOKENS = 8
MAX_SEQUENCE_LENGTH = 64
NUM_LAYERS = 4


def _make_context(
    num_speculative_tokens: int = 2,
    mtp_num_layers=1,
    mtp_use_repeated_layer: bool = True,
    mtp_layer_type_list=None,
    is_hybrid_model: bool = False,
    max_requests: int = 16,
    max_tokens: int = 256,
) -> DynamicInferenceContext:
    """Build a real `DynamicInferenceContext`, MTP-KV-enabled unless a gate is switched off."""
    if is_hybrid_model:
        mamba_inference_state_config = MambaInferenceStateConfig(
            layer_type_list=[Symbols.MAMBA, Symbols.MLP, Symbols.ATTENTION, Symbols.MLP],
            conv_states_shape=(544, 4),
            ssm_states_shape=(8, 64, 16),
            conv_states_dtype=torch.bfloat16,
            ssm_states_dtype=torch.bfloat16,
        )
    else:
        mamba_inference_state_config = None

    return DynamicInferenceContext(
        model_config=TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=NUM_LAYERS,
            kv_channels=16,
            num_attention_heads=4,
            mtp_num_layers=mtp_num_layers,
            mtp_use_repeated_layer=mtp_use_repeated_layer,
        ),
        inference_config=InferenceConfig(
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            use_cuda_graphs_for_non_decode_steps=True,
            buffer_size_gb=0.1,
            paused_buffer_size_gb=0.02,
            block_size_tokens=BLOCK_SIZE_TOKENS,
            max_tokens=max_tokens,
            max_requests=max_requests,
            num_speculative_tokens=num_speculative_tokens,
            mamba_inference_state_config=mamba_inference_state_config,
            mtp_layer_type_list=mtp_layer_type_list,
            use_flashinfer_fused_rope=None,
            unified_memory_level=0,  # unit tests currently broken with UVM
        ),
    )


def _seed_requests(context, block_rows, paused_request_count: int = 0):
    """Populate the per-request block table for `len(block_rows)` active requests.

    `block_rows[r]` is the list of KV block ids request r owns, in position order. Rows before
    `paused_request_count` are left at the -1 fill, so a test asserting on the active slice fails
    loudly if the implementation forgets to offset by the paused count.
    """
    num_active = len(block_rows)
    context.paused_request_count = paused_request_count
    context.total_request_count = paused_request_count + num_active
    for i, blocks in enumerate(block_rows):
        row = paused_request_count + i
        context.request_to_kv_block_ids[row, : len(blocks)] = torch.tensor(
            blocks, dtype=context.request_to_kv_block_ids.dtype
        )
    return num_active


class TestMtpKvCacheGating:
    """`enable_mtp_kv_cache` is derived, not configured, so it must track every gate exactly."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_enabled_reserves_one_extra_attention_slot(self):
        """The draft KV lives in a reserved slot appended after the main attention layers."""
        context = _make_context()
        assert context.enable_mtp_kv_cache is True
        # The slot index is the pre-increment attention-layer count, and the count grows by one
        # so the shared KV buffer is sized for it.
        assert context.mtp_kv_layer_slot == NUM_LAYERS
        assert context.num_attention_layers == NUM_LAYERS + 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_speculative_tokens": 0},
            {"mtp_num_layers": None},
            {"mtp_use_repeated_layer": False},
            {"mtp_layer_type_list": [Symbols.MAMBA, Symbols.MLP]},
            {"mtp_layer_type_list": [Symbols.ATTENTION, Symbols.ATTENTION]},
            {"mtp_layer_type_list": [Symbols.MAMBA]},
        ],
        ids=["no-drafts", "no-head", "per-depth", "recurrent-mlp", "two-attentions", "recurrent"],
    )
    def test_unsupported_config_has_no_draft_kv_plane(self, kwargs):
        """Head patterns also gate attention-only backbones, which have no Mamba metadata."""
        context = _make_context(**kwargs)
        assert context.enable_mtp_kv_cache is False
        assert context.mtp_kv_layer_slot is None
        assert context.num_attention_layers == NUM_LAYERS

    def test_enabled_for_hybrid_main_decoder_with_attention_mtp_head(self):
        """The gate is on the MTP head, not the main decoder: a hybrid backbone is fine."""
        context = _make_context(
            is_hybrid_model=True, mtp_layer_type_list=[Symbols.ATTENTION, Symbols.MLP]
        )
        assert context.enable_mtp_kv_cache is True


class TestMtpDecodeBookkeeping:
    """The per-depth draft write/read metadata."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_begin_decode_seeds_offsets_and_block_table(self):
        context = _make_context()
        _seed_requests(context, [[3, 4], [7, 9]])
        start_positions = torch.tensor([5, 11], device=torch.cuda.current_device())

        context._mtp_begin_decode(2, 2, start_positions)

        assert context.mtp_metadata.forward_active is True
        assert context.mtp_metadata.graphed is False
        assert context.mtp_metadata.active_request_count == 2
        assert context.mtp_metadata.padded_count == 2
        assert context.mtp_metadata.active_offsets.dtype == torch.int32
        assert context.mtp_metadata.active_offsets.cpu().tolist() == [5, 11]
        assert context.mtp_metadata.active_block_table[:, :2].cpu().tolist() == [[3, 4], [7, 9]]

    def test_begin_decode_clones_caller_start_positions(self):
        """`advance_decode_step` must not mutate the caller's `base_position - 1` tensor."""
        context = _make_context()
        _seed_requests(context, [[3], [7]])
        start_positions = torch.tensor([5, 11], device=torch.cuda.current_device())

        context._mtp_begin_decode(2, 2, start_positions)
        context.mtp_metadata.advance_decode_step()
        context.mtp_metadata.advance_decode_step()

        assert start_positions.cpu().tolist() == [
            5,
            11,
        ], "advancing the MTP write positions aliased and corrupted the caller's tensor"
        assert context.mtp_metadata.active_offsets.cpu().tolist() == [7, 13]

    def test_begin_decode_honors_paused_request_offset(self):
        """Active requests start at `paused_request_count`, not at row 0."""
        context = _make_context()
        _seed_requests(context, [[3, 4], [7, 9]], paused_request_count=2)
        start_positions = torch.tensor([5, 11], device=torch.cuda.current_device())

        context._mtp_begin_decode(2, 2, start_positions)

        # Rows 0-1 are paused (still the -1 fill); the block table must hold the ACTIVE rows.
        assert context.mtp_metadata.active_block_table[:, :2].cpu().tolist() == [[3, 4], [7, 9]]

    def test_begin_decode_reads_current_ownership_after_row_reuse(self):
        """A previous forward's table cannot survive compaction or request-slot reuse."""
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        positions = torch.tensor([9], device=torch.cuda.current_device())
        context._mtp_begin_decode(1, 1, positions)
        context.mtp_metadata.end_forward()
        context.request_to_kv_block_ids[0, :2] = torch.tensor([7, 9], dtype=torch.int32)
        context._mtp_begin_decode(1, 1, positions)
        assert context.mtp_metadata.active_block_table[0, :2].cpu().tolist() == [7, 9]

    def test_setup_decode_step_writes_roll_by_one_maps(self):
        """Depth 0 writes at `base_position - 1` and attends over `position + 1` keys."""
        context = _make_context()
        _seed_requests(context, [[3, 4], [7, 9]])
        # Positions 5 and 11: 5 is in block-column 0 (local 5); 11 is in block-column 1 (local 3).
        context._mtp_begin_decode(2, 2, torch.tensor([5, 11], device=torch.cuda.current_device()))

        context._mtp_setup_decode_step()

        gv = context.gpu_view
        assert gv.token_to_block_idx[:2].cpu().tolist() == [3, 9]
        assert gv.token_to_local_position_within_kv_block[:2].cpu().tolist() == [5, 3]
        assert gv.token_to_request_idx[:2].cpu().tolist() == [0, 1]
        assert gv.token_to_position_in_request[:2].cpu().tolist() == [5, 11]
        assert gv.token_to_pos_ids[:2].cpu().tolist() == [5, 11]
        # One query token per request; kv_len = P + 1 (write-then-attend).
        assert gv.mha_query_lengths[:2].cpu().tolist() == [1, 1]
        assert gv.mha_cu_query_seq_lengths[:3].cpu().tolist() == [0, 1, 2]
        assert gv.mha_kv_seq_lengths[:2].cpu().tolist() == [6, 12]
        assert gv.mha_cu_kv_seq_lengths[:3].cpu().tolist() == [0, 6, 18]

    def test_setup_decode_step_crosses_block_boundary(self):
        """A draft that steps past a block boundary must land in the NEXT block at local 0."""
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        # Start at the last slot of block-column 0, so depth 1 lands in block-column 1.
        context._mtp_begin_decode(
            1, 1, torch.tensor([BLOCK_SIZE_TOKENS - 1], device=torch.cuda.current_device())
        )
        gv = context.gpu_view

        context._mtp_setup_decode_step()
        assert int(gv.token_to_block_idx[0].item()) == 3
        assert int(gv.token_to_local_position_within_kv_block[0].item()) == BLOCK_SIZE_TOKENS - 1

        context.mtp_metadata.advance_decode_step()
        context._mtp_setup_decode_step()
        assert int(gv.token_to_block_idx[0].item()) == 4
        assert int(gv.token_to_local_position_within_kv_block[0].item()) == 0
        assert int(gv.mha_kv_seq_lengths[0].item()) == BLOCK_SIZE_TOKENS + 1

    def test_setup_decode_step_advances_by_exactly_one_per_depth(self):
        """Across D depths the write position must advance by 1 per depth and never skip."""
        context = _make_context()
        _seed_requests(context, [[3, 4, 5], [7, 9, 11]])
        context._mtp_begin_decode(2, 2, torch.tensor([5, 11], device=torch.cuda.current_device()))
        gv = context.gpu_view

        seen = []
        for _ in range(4):
            context._mtp_setup_decode_step()
            seen.append(gv.token_to_position_in_request[:2].cpu().tolist())
            context.mtp_metadata.advance_decode_step()

        assert seen == [[5, 11], [6, 12], [7, 13], [8, 14]]

    def test_setup_decode_step_neutralizes_padding_rows(self):
        """Padded slots must never index real KV or contribute query/key length."""
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        context._mtp_begin_decode(1, 4, torch.tensor([5], device=torch.cuda.current_device()))

        context._mtp_setup_decode_step()

        gv = context.gpu_view
        dummy = context.kv_block_allocator.dummy_block_idx
        assert gv.token_to_block_idx[1:4].cpu().tolist() == [dummy, dummy, dummy]
        assert gv.token_to_local_position_within_kv_block[1:4].cpu().tolist() == [0, 0, 0]
        assert gv.mha_query_lengths[1:4].cpu().tolist() == [0, 0, 0]
        assert gv.mha_kv_seq_lengths[1:4].cpu().tolist() == [0, 0, 0]
        # Cumulative lengths flat-line across the padding so the varlen kernel sees empty rows.
        assert gv.mha_cu_query_seq_lengths[1:5].cpu().tolist() == [1, 1, 1, 1]
        assert gv.mha_cu_kv_seq_lengths[1:5].cpu().tolist() == [6, 6, 6, 6]
        assert (gv.mha_block_table[1:4] == -1).all()

    def test_setup_decode_step_eager_routes_to_non_graph_metadata(self):
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        context._mtp_begin_decode(
            1, 1, torch.tensor([5], device=torch.cuda.current_device()), graphed=False
        )

        context._mtp_setup_decode_step()

        assert context.active_attn_metadata is context.non_graph_attn_metadata
        assert context._using_cuda_graph_this_step is False
        assert context.active_token_count == 1
        assert context.padded_active_token_count == 1
        mha = context.non_graph_attn_metadata["mha_metadata"]
        # Eager takes a tight per-step max: kv_len = 5 + 1.
        assert mha.state_data["max_seqlen_k"] == 6
        assert mha.state_data["max_seqlen_q"] == 1

    def test_setup_decode_step_graphed_routes_to_graph_metadata(self):
        """Graphed replay must use the fixed capture-time bound, not a per-step `.item()` sync."""
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        context._mtp_begin_decode(
            1, 4, torch.tensor([5], device=torch.cuda.current_device()), graphed=True
        )

        context._mtp_setup_decode_step()

        assert context.active_attn_metadata is context.graph_attn_metadata
        assert context._using_cuda_graph_this_step is True
        assert context.padded_active_token_count == 4
        mha = context.graph_attn_metadata["mha_metadata"]
        assert mha.state_data["max_seqlen_k"] == mha.max_seqlen == MAX_SEQUENCE_LENGTH
        assert mha.state_data["max_seqlen_q"] == 1
        # The metadata is sized by the PADDED count so the graph's launch bounds stay stable.
        assert mha.state_data["query_lengths"].shape[0] == 4

    def test_setup_decode_step_restores_the_graph_flag_clobbered_by_the_commit_pass(self):
        """The commit pass runs eager and leaves the live graph flag False.

        The depth loop reads `using_cuda_graph_this_step()` to pick `eager=` and `cache_key=`
        for each draft forward, so `_mtp_setup_decode_step` must put the flag back before the
        forward sees it. If it did not, a graphed step would run eager with `cache_key=None`
        and trigger an illegal CUDA-graph capture at runtime.
        """
        context = _make_context()
        device = torch.cuda.current_device()
        _seed_requests(context, [[3, 4]])

        # The commit pass: a varlen eager forward that clobbers the flag.
        context._mtp_setup_prefill_step(
            append_counts=torch.tensor([2], device=device),
            block_table_prefill=torch.full(
                (1, context.max_kv_block_count),
                3,
                dtype=context.gpu_view.mha_block_table.dtype,
                device=device,
            ),
        )
        context.mtp_metadata.end_forward()
        assert context.using_cuda_graph_this_step() is False

        context._mtp_begin_decode(1, 1, torch.tensor([5], device=device), graphed=True)
        context._mtp_setup_decode_step()

        assert context.using_cuda_graph_this_step() is True

    def test_graphed_bounds_are_position_independent(self):
        """Graphed replay must use the fixed capture-time bound, never a per-step max.

        A per-step `int(kv_len.max().item())` would both desync from the captured launch bounds
        and add a GPU->CPU sync inside the draft loop.
        """
        context = _make_context()
        device = torch.cuda.current_device()
        _seed_requests(context, [[3, 4, 5]])
        mha = context.graph_attn_metadata["mha_metadata"]

        context._mtp_begin_decode(1, 1, torch.tensor([2], device=device), graphed=True)
        context._mtp_setup_decode_step()
        bound_at_low_position = mha.state_data["max_seqlen_k"]

        context._mtp_begin_decode(1, 1, torch.tensor([20], device=device), graphed=True)
        context._mtp_setup_decode_step()
        bound_at_high_position = mha.state_data["max_seqlen_k"]

        assert bound_at_low_position == bound_at_high_position == MAX_SEQUENCE_LENGTH
        # The real per-request lengths still come from the GPU cu_kv tensors.
        assert int(context.gpu_view.mha_kv_seq_lengths[0].item()) == 21

    def test_capture_advances_positions_while_staying_on_scratch(self):
        """Warmup walks the same setup/advance sequence as a real step, on scratch memory only."""
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        dummy = context.kv_block_allocator.dummy_block_idx
        gv = context.gpu_view

        context.mtp_metadata.begin_decode_for_capture(2)
        seen_positions = []
        for _ in range(3):
            context._mtp_setup_decode_step()
            seen_positions.append(gv.token_to_position_in_request[:2].cpu().tolist())
            assert (gv.token_to_block_idx[:2] == dummy).all()
            context.mtp_metadata.advance_decode_step()
        context.mtp_metadata.end_forward()

        assert seen_positions == [[0, 0], [1, 1], [2, 2]]
        assert context.mtp_metadata.forward_active is False

    def test_capture_crossing_a_block_boundary_stays_on_scratch(self):
        """Even past a block boundary the capture must not touch a real block."""
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        dummy = context.kv_block_allocator.dummy_block_idx

        context.mtp_metadata.begin_decode_for_capture(1)
        for _ in range(BLOCK_SIZE_TOKENS + 2):
            context._mtp_setup_decode_step()
            assert int(context.gpu_view.token_to_block_idx[0].item()) == dummy
            context.mtp_metadata.advance_decode_step()

    def test_capture_uses_graph_metadata_at_the_padded_size(self):
        """Capture-time launch bounds must match the runtime graphed step's."""
        context = _make_context()
        _seed_requests(context, [[3, 4]])

        context.mtp_metadata.begin_decode_for_capture(4)
        context._mtp_setup_decode_step()

        assert context.active_attn_metadata is context.graph_attn_metadata
        assert context._using_cuda_graph_this_step is True
        mha = context.graph_attn_metadata["mha_metadata"]
        assert mha.state_data["max_seqlen_q"] == 1
        assert mha.state_data["max_seqlen_k"] == mha.max_seqlen
        assert mha.state_data["query_lengths"].shape[0] == 4
        # Every capture row is a real (non-padding) row, so nothing is sentinel-filled.
        assert context.gpu_view.mha_query_lengths[:4].cpu().tolist() == [1, 1, 1, 1]

    def test_begin_decode_for_capture_touches_only_scratch_kv(self):
        """Capture-time metadata must point every row at the scratch block."""
        context = _make_context()
        _seed_requests(context, [[3, 4]])

        context.mtp_metadata.begin_decode_for_capture(4)

        dummy = context.kv_block_allocator.dummy_block_idx
        assert context.mtp_metadata.graphed is True
        assert context.mtp_metadata.forward_active is True
        assert context.mtp_metadata.active_request_count == 4
        assert context.mtp_metadata.padded_count == 4
        assert (context.mtp_metadata.active_offsets == 0).all()
        assert (context.mtp_metadata.active_block_table == dummy).all()

        # A capture-time setup step must not write any real block id.
        context._mtp_setup_decode_step()
        assert (context.gpu_view.token_to_block_idx[:4] == dummy).all()

    def test_end_decode_leaves_mtp_forward_mode(self):
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        context._mtp_begin_decode(1, 1, torch.tensor([5], device=torch.cuda.current_device()))
        assert context.mtp_metadata.forward_active is True

        context.mtp_metadata.end_forward()

        assert context.mtp_metadata.forward_active is False

    def test_begin_decode_asserts_when_disabled(self):
        context = _make_context(num_speculative_tokens=0)
        with pytest.raises(AssertionError):
            context._mtp_begin_decode(1, 1, torch.tensor([5], device=torch.cuda.current_device()))


class TestMtpMainExecutionState:
    """Every MTP forward republishes the context's execution state; it must be put back."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_prefill_step_restores_the_main_forward_counts(self):
        """Log-prob computation runs after the MTP phase and reads these as the MAIN counts."""
        context = _make_context()
        device = torch.cuda.current_device()
        # Stand in for a 4-token prefill forward.
        context.active_attn_metadata = context.non_graph_attn_metadata
        context.active_token_count = 4
        context.padded_active_token_count = 8
        context._using_cuda_graph_this_step = False

        table = torch.full(
            (1, context.max_kv_block_count),
            3,
            dtype=context.gpu_view.mha_block_table.dtype,
            device=device,
        )
        with context._mtp_forward_phase():
            context.paused_request_count = 0
            context.total_request_count = 1
            context._mtp_setup_prefill_step(
                append_counts=torch.tensor([1], device=device), block_table_prefill=table
            )
            # The MTP forward has overwritten the counts with its own one-token geometry.
            assert context.active_token_count == 1

        assert context.active_token_count == 4
        assert context.padded_active_token_count == 8
        assert context._using_cuda_graph_this_step is False
        assert context.active_attn_metadata is context.non_graph_attn_metadata
        # Leaving the scope must also leave MTP-forward mode, or KV routing stays on the
        # draft plane and `is_decode_only` stays False for the rest of the run.
        assert context.mtp_metadata.forward_active is False

    def test_state_is_restored_when_a_draft_forward_raises(self):
        """A failed draft must not leave its counts behind for the log-prob code."""
        context = _make_context()
        context.active_attn_metadata = context.non_graph_attn_metadata
        context.active_token_count = 4
        context.padded_active_token_count = 8

        with pytest.raises(RuntimeError, match="draft blew up"):
            with context._mtp_forward_phase():
                context.active_token_count = 1
                context.padded_active_token_count = 1
                raise RuntimeError("draft blew up")

        assert context.active_token_count == 4
        assert context.padded_active_token_count == 8
        assert context.mtp_metadata.forward_active is False


class TestMtpPrefillBookkeeping:
    """The varlen commit pass: prompt seeding and the per-step committed-KV refresh."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @staticmethod
    def _block_table(context, block_rows):
        """Build the GPU block table argument the controller passes in.

        Also declares `len(block_rows)` active requests on the context: `_mtp_setup_prefill_step`
        reads `mtp_metadata.request_matched_prefix_blocks` over the active slice to keep writes
        out of inherited blocks, so the slice must be at least as long as the block table.
        """
        context.paused_request_count = 0
        context.total_request_count = len(block_rows)
        table = torch.full(
            (len(block_rows), context.max_kv_block_count),
            -1,
            dtype=context.gpu_view.mha_block_table.dtype,
            device=torch.cuda.current_device(),
        )
        for i, blocks in enumerate(block_rows):
            table[i, : len(blocks)] = torch.tensor(blocks, dtype=table.dtype, device=table.device)
        return table

    def test_prompt_seed_writes_positions_from_zero(self):
        """Prompt seeding (no start positions) writes each request's positions 0..count-1."""
        context = _make_context()
        device = torch.cuda.current_device()
        append_counts = torch.tensor([3, 2], device=device)
        block_table = self._block_table(context, [[3, 4], [7, 9]])

        context._mtp_setup_prefill_step(
            append_counts=append_counts, block_table_prefill=block_table
        )

        gv = context.gpu_view
        assert gv.token_to_position_in_request[:5].cpu().tolist() == [0, 1, 2, 0, 1]
        assert gv.token_to_request_idx[:5].cpu().tolist() == [0, 0, 0, 1, 1]
        assert gv.token_to_block_idx[:5].cpu().tolist() == [3, 3, 3, 7, 7]
        assert gv.token_to_local_position_within_kv_block[:5].cpu().tolist() == [0, 1, 2, 0, 1]
        # Fresh causal prefill: kv_length == query_length per request.
        assert gv.mha_query_lengths[:2].cpu().tolist() == [3, 2]
        assert gv.mha_kv_seq_lengths[:2].cpu().tolist() == [3, 2]
        assert gv.mha_cu_query_seq_lengths[:3].cpu().tolist() == [0, 3, 5]
        assert gv.mha_cu_kv_seq_lengths[:3].cpu().tolist() == [0, 3, 5]

    def test_commit_refresh_shifts_by_request_start_positions(self):
        """The decode refresh rewrites each request's own committed offset range."""
        context = _make_context()
        device = torch.cuda.current_device()
        append_counts = torch.tensor([2, 3], device=device)
        start_positions = torch.tensor([4, 9], device=device)
        block_table = self._block_table(context, [[3, 4], [7, 9]])

        context._mtp_setup_prefill_step(
            append_counts=append_counts,
            block_table_prefill=block_table,
            request_start_positions=start_positions,
        )

        gv = context.gpu_view
        assert gv.token_to_position_in_request[:5].cpu().tolist() == [4, 5, 9, 10, 11]
        # Request 0 stays in block-column 0; request 1's positions 9-11 are in block-column 1.
        assert gv.token_to_block_idx[:5].cpu().tolist() == [3, 3, 9, 9, 9]
        assert gv.token_to_local_position_within_kv_block[:5].cpu().tolist() == [4, 5, 1, 2, 3]

    def test_commit_refresh_crosses_block_boundary_mid_run(self):
        """A refreshed run that straddles a block boundary must switch blocks mid-run."""
        context = _make_context()
        device = torch.cuda.current_device()
        block_table = self._block_table(context, [[3, 4]])

        context._mtp_setup_prefill_step(
            append_counts=torch.tensor([3], device=device),
            block_table_prefill=block_table,
            request_start_positions=torch.tensor([BLOCK_SIZE_TOKENS - 1], device=device),
        )

        gv = context.gpu_view
        assert gv.token_to_block_idx[:3].cpu().tolist() == [3, 4, 4]
        assert gv.token_to_local_position_within_kv_block[:3].cpu().tolist() == [
            BLOCK_SIZE_TOKENS - 1,
            0,
            1,
        ]

    def test_prefill_step_neutralizes_token_and_request_padding(self):
        """SP token padding and request padding must never index real KV."""
        context = _make_context()
        device = torch.cuda.current_device()
        append_counts = torch.tensor([3, 2], device=device)
        block_table = self._block_table(context, [[3, 4], [7, 9]])

        context._mtp_setup_prefill_step(
            append_counts=append_counts,
            block_table_prefill=block_table,
            padded_token_count=8,
            padded_request_count=4,
        )

        gv = context.gpu_view
        dummy = context.kv_block_allocator.dummy_block_idx
        assert gv.token_to_block_idx[5:8].cpu().tolist() == [dummy, dummy, dummy]
        assert gv.token_to_local_position_within_kv_block[5:8].cpu().tolist() == [0, 0, 0]
        # Row 2 is the trailing pad request carrying the 3 SP pad tokens; it reads the dummy
        # block. Row 3 is request padding proper: zero-length and never indexed.
        assert gv.mha_query_lengths[2:4].cpu().tolist() == [3, 0]
        assert gv.mha_kv_seq_lengths[2:4].cpu().tolist() == [3, 0]
        assert gv.mha_cu_query_seq_lengths[2:5].cpu().tolist() == [5, 8, 8]
        assert gv.mha_cu_kv_seq_lengths[2:5].cpu().tolist() == [5, 8, 8]
        assert (gv.mha_block_table[2] == dummy).all()
        assert (gv.mha_block_table[3:4] == -1).all()
        assert context.active_token_count == 5
        assert context.padded_active_token_count == 8

    @pytest.mark.parametrize(
        "append_counts_list, padded_token_count",
        [([0, 1, 0], 4), ([3, 2], 8), ([2, 2], 4)],
        ids=["single_token_tp4", "five_tokens_pad8", "already_aligned"],
    )
    def test_prefill_step_query_metadata_covers_every_padded_token(
        self, append_counts_list, padded_token_count
    ):
        """`cu_seqlens_q` must describe the SP pad rows, not only the committed ones.

        The packed hidden is padded to a TP multiple before the sequence-parallel scatter, so
        after the MTP layer's internal gather the attention receives `padded_token_count` query
        rows. Varlen attention requires `q.shape[0] == cu_seqlens_q[-1]`, so every padded row
        has to be accounted for by some request in the metadata.

        `single_token_tp4` is the shape that arises whenever exactly one draft position is
        committed under TP=4: one real token padded up to four.
        """
        context = _make_context()
        device = torch.cuda.current_device()
        append_counts = torch.tensor(append_counts_list, device=device)
        num_requests = len(append_counts_list)
        total = int(append_counts.sum())
        pad_tokens = padded_token_count - total
        block_table = self._block_table(context, [[3, 4]] * num_requests)

        context._mtp_setup_prefill_step(
            append_counts=append_counts,
            block_table_prefill=block_table,
            padded_token_count=padded_token_count,
            padded_request_count=num_requests,
        )

        gv = context.gpu_view
        mha = context.non_graph_attn_metadata["mha_metadata"]
        padded_p = mha.state_data["cu_query_seq_lengths"].numel() - 1
        cu_q = gv.mha_cu_query_seq_lengths[: padded_p + 1].cpu().tolist()

        assert cu_q[0] == 0
        assert cu_q[-1] == padded_token_count, (
            f"cu_seqlens_q ends at {cu_q[-1]} but attention will be handed "
            f"{padded_token_count} query rows"
        )
        assert int(gv.mha_query_lengths[:padded_p].sum()) == padded_token_count
        # Causal prefill: the kv run matches the query run for every request, pad row included.
        assert gv.mha_cu_kv_seq_lengths[: padded_p + 1].cpu().tolist() == cu_q
        # The kernel's seqlen bound has to cover the pad run as well as the real requests.
        assert mha.state_data["max_seqlen_q"] >= max([*append_counts_list, pad_tokens])
        if pad_tokens > 0:
            # The pad request must read a real (dummy) block; -1 is not a valid page index.
            assert (
                gv.mha_block_table[num_requests] == context.kv_block_allocator.dummy_block_idx
            ).all()
        assert context.active_token_count == total
        assert context.padded_active_token_count == padded_token_count

    def test_prefill_step_pads_query_metadata_with_every_request_slot_taken(self):
        """A full batch leaves no spare request row, so the pad rows join the last request.

        `mha_query_lengths` and friends are sized to `max_requests`, so when every slot holds a
        real request the trailing-request form has nowhere to go. `cu_seqlens_q[-1]` must still
        equal the padded query-row count.
        """
        max_requests = 3
        context = _make_context(max_requests=max_requests)
        device = torch.cuda.current_device()
        append_counts = torch.tensor([0, 1, 0], device=device)
        block_table = self._block_table(context, [[3, 4]] * max_requests)
        padded_token_count = 4

        context._mtp_setup_prefill_step(
            append_counts=append_counts,
            block_table_prefill=block_table,
            padded_token_count=padded_token_count,
            padded_request_count=max_requests,
        )

        gv = context.gpu_view
        mha = context.non_graph_attn_metadata["mha_metadata"]
        assert gv.mha_query_lengths.numel() == max_requests, "test needs a fully occupied batch"
        cu_q = gv.mha_cu_query_seq_lengths[: max_requests + 1].cpu().tolist()
        assert cu_q[-1] == padded_token_count
        assert int(gv.mha_query_lengths[:max_requests].sum()) == padded_token_count
        # The pad rows joined the last real request, so its run grew by the pad amount.
        assert gv.mha_query_lengths[max_requests - 1].item() == 3
        assert gv.mha_kv_seq_lengths[max_requests - 1].item() == 3
        assert gv.mha_cu_kv_seq_lengths[: max_requests + 1].cpu().tolist() == cu_q
        assert mha.state_data["max_seqlen_q"] >= 3
        assert context.padded_active_token_count == padded_token_count

    def test_prefill_step_forces_varlen_path_on_a_pure_decode_step(self):
        """The commit pass declares itself varlen, so the ragged forward avoids the decode kernel.

        `num_prefill_requests` is left truthful: the flag states the mode directly rather than
        faking a prefill count to imply it.
        """
        context = _make_context()
        device = torch.cuda.current_device()
        context.num_prefill_requests = 0
        context._using_cuda_graph_this_step = True
        block_table = self._block_table(context, [[3], [7]])

        context._mtp_setup_prefill_step(
            append_counts=torch.tensor([2, 1], device=device),
            block_table_prefill=block_table,
            request_start_positions=torch.tensor([4, 6], device=device),
        )

        assert context.num_prefill_requests == 0  # untouched
        assert context.mtp_metadata.is_varlen_forward is True
        assert context.is_decode_only() is False
        assert context._using_cuda_graph_this_step is False
        assert context.mtp_metadata.forward_active is True
        assert context.active_attn_metadata is context.non_graph_attn_metadata
        mha = context.non_graph_attn_metadata["mha_metadata"]
        assert mha.state_data["max_seqlen_q"] == 2
        assert mha.state_data["max_seqlen_k"] == 2

    def test_finalize_clears_the_varlen_mode(self):
        """The commit pass must hand the step back exactly as it found it."""
        context = _make_context()
        device = torch.cuda.current_device()
        context.num_prefill_requests = 0
        block_table = self._block_table(context, [[3]])

        context._mtp_setup_prefill_step(
            append_counts=torch.tensor([2], device=device), block_table_prefill=block_table
        )
        assert context.mtp_metadata.is_varlen_forward is True
        assert context.is_decode_only() is False

        context.mtp_metadata.end_forward()

        assert context.mtp_metadata.is_varlen_forward is False
        assert context.mtp_metadata.forward_active is False
        # With the flag cleared, the step's own counts decide again.
        assert context.num_prefill_requests == 0
        assert context.is_decode_only() is True

    def test_prefill_step_asserts_when_disabled(self):
        context = _make_context(num_speculative_tokens=0)
        device = torch.cuda.current_device()
        with pytest.raises(AssertionError):
            context._mtp_setup_prefill_step(
                append_counts=torch.tensor([2], device=device),
                block_table_prefill=self._block_table(context, [[3]]),
            )


class TestMtpChunkBoundaryCarry:
    """The chunked-prefill boundary carry on `MTPMetadata`.

    The carry must survive BETWEEN steps. It is keyed by request id AND by the prompt position
    the hidden was computed at, so a carry left behind by a request that has since restarted at
    a different offset can never be consumed.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @staticmethod
    def _meta():
        """A freshly allocated context's MTPMetadata (`__init__` runs initialize_all_tensors)."""
        return _make_context().mtp_metadata

    @staticmethod
    def _hidden(meta, fill):
        return torch.full((1, 1, meta.hidden_size), fill, device="cuda", dtype=meta.hidden_dtype)

    def test_carry_then_take_at_the_recorded_position(self):
        meta = self._meta()
        hidden = self._hidden(meta, 7.0)
        meta.carry_chunk_boundary(hidden=hidden, req_id=42, position=3)

        taken = meta.take_chunk_boundary(req_id=42, seam_position=3)
        assert taken is not None
        assert taken.shape == (1, 1, meta.hidden_size)
        assert torch.equal(taken, hidden)

    def test_take_raises_on_a_position_mismatch(self):
        """A carry-holding continuation chunk takes no prefix skip, so its seam lands here.

        `_compute_prefix_match` gives up that chunk's ENTIRE match to guarantee it. A mismatch
        therefore means the back-off stopped holding, not that the carry is merely stale, and
        declining would leave a committed position unwritten.
        """
        meta = self._meta()
        meta.carry_chunk_boundary(hidden=self._hidden(meta, 1.0), req_id=42, position=3)
        with pytest.raises(AssertionError, match="sits at position 3"):
            meta.take_chunk_boundary(req_id=42, seam_position=9)

    def test_take_raises_on_a_request_mismatch(self):
        """The caller derives `req_id` from the carry itself, so a mismatch is a caller bug."""
        meta = self._meta()
        meta.carry_chunk_boundary(hidden=self._hidden(meta, 1.0), req_id=42, position=3)
        with pytest.raises(AssertionError, match="belongs to request 42"):
            meta.take_chunk_boundary(req_id=7, seam_position=3)

    def test_a_first_chunk_can_never_consume_a_carry(self):
        """off == 0 asks for seam position -1; a valid carry always records position >= 0."""
        meta = self._meta()
        meta.carry_chunk_boundary(hidden=self._hidden(meta, 1.0), req_id=42, position=0)
        with pytest.raises(AssertionError, match="sits at position 0"):
            meta.take_chunk_boundary(req_id=42, seam_position=-1)

    def test_invalidate_drops_the_carry_but_keeps_the_buffer(self):
        meta = self._meta()
        buf_before = meta.chunk_boundary_hidden
        meta.carry_chunk_boundary(hidden=self._hidden(meta, 1.0), req_id=42, position=3)
        meta.invalidate_chunk_boundary()

        assert not meta.chunk_boundary_valid
        assert meta.chunk_boundary_req_id == -1
        assert meta.chunk_boundary_position == -1
        with pytest.raises(AssertionError, match="no live chunk-boundary carry"):
            meta.take_chunk_boundary(req_id=42, seam_position=3)
        # The buffer address is stable across invalidation -- only the keys are cleared.
        assert meta.chunk_boundary_hidden is buf_before

    def test_reset_metadata_invalidates_the_carry(self):
        """The gap this closes: a request restarted after a reset must not match its old carry."""
        context = _make_context()
        meta = context.mtp_metadata
        meta.carry_chunk_boundary(
            hidden=torch.zeros((1, 1, meta.hidden_size), device="cuda", dtype=meta.hidden_dtype),
            req_id=42,
            position=3,
        )
        context.reset_metadata()

        assert not meta.chunk_boundary_valid
        assert context.chunked_prefill_request_id == -1
        with pytest.raises(AssertionError, match="no live chunk-boundary carry"):
            meta.take_chunk_boundary(req_id=42, seam_position=3)

    def test_deallocate_invalidates_the_carry(self):
        """This is what makes the carry safe: it dies with every other piece of MTP state."""
        meta = self._meta()
        meta.carry_chunk_boundary(hidden=self._hidden(meta, 1.0), req_id=42, position=3)
        meta.deallocate()

        assert not meta.chunk_boundary_valid
        assert meta.chunk_boundary_hidden is None
        assert meta.request_matched_prefix_blocks is None
        meta.allocate(device=torch.cuda.current_device())
        assert not meta.request_matched_prefix_blocks.any()
        with pytest.raises(AssertionError, match="no live chunk-boundary carry"):
            meta.take_chunk_boundary(req_id=42, seam_position=3)

    def test_carry_is_a_private_copy(self):
        """The producing step's activation buffer is reused; the carry must not alias it."""
        meta = self._meta()
        src = self._hidden(meta, 5.0)
        meta.carry_chunk_boundary(hidden=src, req_id=42, position=3)
        src.fill_(-1.0)

        taken = meta.take_chunk_boundary(req_id=42, seam_position=3)
        assert torch.all(taken == 5.0)

    def test_disabled_context_never_carries(self):
        context = _make_context(num_speculative_tokens=0)
        assert not context.enable_mtp_kv_cache
        meta = context.mtp_metadata
        meta.reset_request_rows()
        meta.move_request_rows(torch.tensor([0]), torch.tensor([1]))
        meta.swap_request_rows(torch.tensor([0]), torch.tensor([1]))
        assert meta.request_matched_prefix_blocks is None
        meta.carry_chunk_boundary(
            hidden=torch.zeros((1, 1, 8), device="cuda"), req_id=42, position=3
        )
        assert not meta.chunk_boundary_valid
        with pytest.raises(AssertionError, match="no live chunk-boundary carry"):
            meta.take_chunk_boundary(req_id=42, seam_position=3)


class TestMtpSpareBlockLifecycle:
    """Owned lookahead across prefill, verification, rewind, pause, and resume.

    Prefill reserves capacity for the first draft loop; decode scheduling also covers the
    draft writes after verification. The blocks are counted in `request_kv_block_counts`, but
    `request_last_kv_block_id` deliberately still names the last token-bearing block: the pointer
    drives the MAIN model's writes, whose local offsets are `position % block_size_tokens`, so
    entering the block early would send the next main token to the wrong block.

    Reserved capacity is derived from owned blocks and the committed main-token span.
    Scheduling must honor that capacity without separately maintaining a spare-block flag.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    # A prompt whose last token sits at offset 6 of an 8-token block, i.e. within
    # num_speculative_tokens (2) of the boundary, so the reserve triggers.
    SPARE_PROMPT_LENGTH = 7
    # Last token at offset 3: comfortably clear of the boundary, so no reserve.
    NO_SPARE_PROMPT_LENGTH = 4

    @staticmethod
    def _request(context, prompt_length, request_id=1):
        return DynamicInferenceRequest(
            request_id=request_id,
            prompt_tokens=torch.arange(prompt_length, device=torch.cuda.current_device()),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
            block_size_tokens=context.block_size_tokens,
            enable_prefix_caching=False,
        )

    @staticmethod
    def _step(context, active_mask_values):
        """Run one `update_requests` over the active rows, with all-new tokens.

        `update_requests` indexes `new_speculative_tokens` unconditionally once
        `num_speculative_tokens > 0`, and expects `[D, num_active]` in active-row order.
        """
        device = torch.cuda.current_device()
        num_active = len(active_mask_values)
        active_mask = torch.tensor(active_mask_values, device=device, dtype=torch.int32)
        new_tokens = torch.full((num_active,), 100, device=device, dtype=torch.int64)
        new_speculative_tokens = None
        if context.num_speculative_tokens > 0:
            new_speculative_tokens = torch.full(
                (context.num_speculative_tokens, num_active), 101, device=device, dtype=torch.int64
            )
        return context.update_requests(active_mask, new_tokens, new_speculative_tokens)

    @staticmethod
    def _row_blocks(context, row):
        count = int(context.request_kv_block_counts[row].item())
        return context.request_to_kv_block_ids[row, :count].tolist()

    @staticmethod
    def _reserved_blocks(context, rows):
        return context.request_kv_block_counts[rows] - context.get_committed_kv_block_counts(rows)

    @staticmethod
    def _rewind(context, accepted):
        controller = TextGenerationController.__new__(TextGenerationController)
        controller.inference_wrapped_model = types.SimpleNamespace(inference_context=context)
        controller.num_speculative_tokens = context.num_speculative_tokens
        released, mask = controller._rewind_kv_cache(accepted_counts_cpu=accepted)
        assert not mask.any(), "MTP lookahead must remain owned after rejection"
        context.kv_block_allocator.release_memory_blocks(released[mask])

    @staticmethod
    def _assert_main_token_mapping(context):
        n = context.active_token_count
        positions = context.token_to_pos_ids[:n].long()
        rows = context.token_to_request_idx[:n].long()
        columns = positions // context.block_size_tokens
        expected = context.request_to_kv_block_ids[rows, columns]
        assert (expected >= 0).all()
        torch.testing.assert_close(
            context.token_to_block_idx[:n], expected.to(context.token_to_block_idx.dtype)
        )
        active = slice(context.paused_request_count, context.total_request_count)
        last_columns = (
            context.request_kv_length_offsets[active] + context.request_query_lengths[active] - 1
        ) // context.block_size_tokens
        request_rows = torch.arange(context.paused_request_count, context.total_request_count)
        torch.testing.assert_close(
            context.request_last_kv_block_id[active],
            context.request_to_kv_block_ids[request_rows, last_columns.long()],
        )

    @staticmethod
    def _assert_draft_token_mapping(context, starts):
        count = starts.numel()
        owned = context.request_to_kv_block_ids
        with context._mtp_forward_phase():
            context._mtp_begin_decode(count, count, starts)
            for draft_depth in range(context.num_speculative_tokens):
                context._mtp_setup_decode_step()
                positions = (starts + draft_depth).cpu().long()
                expected = owned[torch.arange(count), positions // context.block_size_tokens]
                assert (expected >= 0).all(), "draft reached an unallocated block"
                torch.testing.assert_close(
                    context.gpu_view.token_to_block_idx[:count].cpu(),
                    expected.to(context.gpu_view.token_to_block_idx.dtype),
                )
                context.mtp_metadata.advance_decode_step()

    def test_mtp_row_annotations_follow_context_lifecycle(self):
        context = _make_context()
        for i, length in enumerate((16, 24, 32), start=1):
            context.add_request(self._request(context, length, request_id=i))
        meta = context.mtp_metadata
        annotations = meta.request_matched_prefix_blocks
        assert annotations.device.type == "cpu" and annotations.is_pinned()
        annotations[:3] = torch.tensor([0, 1, 2], dtype=annotations.dtype)
        next_tokens = torch.tensor([100, 101, 102])

        # Swap and overlapping movement use the same row indices as main KV state.
        context._swap_book_keeping_tensors(torch.tensor([0]), torch.tensor([2]), next_tokens)
        assert annotations[:3].tolist() == [2, 1, 0]
        context._move_book_keeping_tensors(torch.tensor([0, 1]), torch.tensor([1, 2]), next_tokens)
        assert annotations[:3].tolist() == [2, 2, 1]
        meta.reset_request_rows(torch.tensor([0]))
        assert annotations[:3].tolist() == [0, 2, 1]
        context.reset_tensors()
        assert not annotations.any()

    def test_async_compaction_clears_vacated_mtp_annotations(self):
        context = _make_context()
        for i, length in enumerate((16, 24, 32), start=1):
            context.add_request(self._request(context, length, request_id=i))
        self._step(context, [1, 1, 1])
        annotations = context.mtp_metadata.request_matched_prefix_blocks
        annotations[:3] = torch.tensor([0, 1, 2], dtype=annotations.dtype)
        context.resolve_requests(torch.tensor([0, 1, 1], dtype=torch.int32))
        # Hole filling moves the rightmost survivor into row 0; row 1 stays put.
        assert annotations[:3].tolist() == [2, 1, 0]

    @pytest.mark.parametrize("depth,accepted", [(d, a) for d in (1, 2, 4, 7) for a in range(d + 1)])
    @pytest.mark.parametrize("successor_path", ["update", "prepare"])
    def test_verification_and_drafting_own_every_write(self, depth, accepted, successor_path):
        """Sweep every block offset and acceptance count, including D=block_size-1.

        The first transition includes the original B=8,D=2,prompt=5 failure. A second
        verification/draft step exercises retained lookahead after rewind and both schedulers.
        """
        context = _make_context(num_speculative_tokens=depth)
        count = 2 * context.block_size_tokens
        for length in range(1, count + 1):
            context.add_request(self._request(context, length, request_id=length))
        # Admission owns the blocks before any decode scheduling has run. Check the initial
        # prefill's draft loop too, using real allocation rather than synthetic block IDs.
        self._assert_main_token_mapping(context)
        self._assert_draft_token_mapping(context, torch.arange(count, device="cuda"))
        self._step(context, [1] * count)
        self._assert_main_token_mapping(context)

        for acceptance in (accepted, depth - accepted):
            owned = context.request_to_kv_block_ids.clone()
            available = context.kv_block_allocator.get_allocatable_count()
            self._rewind(context, torch.full((count,), acceptance, dtype=torch.int64))
            torch.testing.assert_close(context.request_to_kv_block_ids, owned)
            assert context.kv_block_allocator.get_allocatable_count() == available
            starts = (
                context.request_kv_length_offsets[:count]
                + context.request_query_lengths[:count]
                - 1
            ).to(device="cuda", dtype=torch.int64)
            self._assert_draft_token_mapping(context, starts)

            if successor_path == "update":
                self._step(context, [1] * count)
            else:
                assert context.can_prepare_requests()
                context.prepare_requests()
            self._assert_main_token_mapping(context)

    def test_insufficient_draft_headroom_pauses_before_verification(self):
        context = _make_context()
        # Main positions 5..7 fit the first block, but the following draft can write 8.
        context.add_request(self._request(context, 5, request_id=1))
        context.add_request(self._request(context, 7, request_id=2))
        held = context.kv_block_allocator.allocate_memory_blocks(
            context.kv_block_allocator.pool_avail
        ).clone()
        self._step(context, [1, 1])
        assert context.paused_request_count == 1
        assert int(context.request_ids[0]) == 1
        assert context.request_kv_length_offsets[0] == 0, "paused before preparing verification"
        self._assert_main_token_mapping(context)

        context.kv_block_allocator.release_memory_blocks(held[:1])
        self._step(context, [1])
        assert context.paused_request_count == 0
        row = int(torch.nonzero(context.request_ids[:2] == 1)[0])
        assert context.request_kv_block_counts[row] == 2
        assert self._reserved_blocks(context, row)
        self._assert_main_token_mapping(context)
        context.kv_block_allocator.release_memory_blocks(held[1:])

    def test_async_prepare_checks_draft_headroom_before_mutating_state(self):
        context = _make_context()
        context.add_request(self._request(context, 4))
        self._step(context, [1])  # verification writes 4..6, draft fits through 7
        assert context.request_kv_block_counts[0] == 1
        held = context.kv_block_allocator.allocate_memory_blocks(
            context.kv_block_allocator.pool_avail
        ).clone()
        self._rewind(context, torch.tensor([0]))  # committed position 4; next draft reaches 8
        before = context.request_kv_length_offsets.clone()
        assert not context.can_prepare_requests()
        with pytest.raises(RuntimeError, match="cannot pause requests"):
            context.prepare_requests()
        torch.testing.assert_close(context.request_kv_length_offsets, before)
        context.kv_block_allocator.release_memory_blocks(held[:1])
        assert context.can_prepare_requests()
        context.prepare_requests()
        assert context.request_kv_block_counts[0] == 2
        self._assert_main_token_mapping(context)
        context.kv_block_allocator.release_memory_blocks(held[1:])

    @pytest.mark.parametrize("depth", [2, 7])
    def test_imported_decode_reserves_draft_headroom(self, depth):
        context = _make_context(num_speculative_tokens=depth)
        # Main verification fits through position 15; drafting reaches the next block.
        prompt_length = 2 * context.block_size_tokens - (depth + 1)
        request = self._request(context, prompt_length)
        blocks = context.kv_block_allocator.allocate_memory_blocks(3).tolist()
        prompt_blocks = (prompt_length + context.block_size_tokens - 1) // context.block_size_tokens
        admit_prefilled_decode(
            context,
            request,
            prompt_block_ids=blocks[:prompt_blocks],
            continuation_block_ids=blocks[prompt_blocks:],
            input_tokens=list(range(depth + 1)),
        )
        assert self._reserved_blocks(context, 0)
        assert int(context.request_last_kv_block_id[0]) == blocks[1]
        self._assert_main_token_mapping(context)

    def test_prefill_reserves_a_block_the_pointer_does_not_name(self):
        """The reserve is counted and owned, but the row has not entered it yet."""
        context = _make_context()
        context.add_request(self._request(context, self.SPARE_PROMPT_LENGTH))

        assert bool(self._reserved_blocks(context, 0))
        blocks = self._row_blocks(context, 0)
        # One block for the 7 prompt tokens, one reserved for the draft writes past them.
        assert len(blocks) == 2
        assert int(context.request_last_kv_block_id[0]) == blocks[0], (
            "the pointer must still name the last token-bearing block, or the next main token "
            "is written into the reserve at the wrong local offset"
        )

    def test_a_prompt_clear_of_the_boundary_reserves_nothing(self):
        context = _make_context()
        context.add_request(self._request(context, self.NO_SPARE_PROMPT_LENGTH))

        assert not bool(self._reserved_blocks(context, 0))
        blocks = self._row_blocks(context, 0)
        assert len(blocks) == 1
        assert int(context.request_last_kv_block_id[0]) == blocks[0]

    def test_without_speculative_decoding_nothing_is_ever_reserved(self):
        """The whole mechanism is inert at D=0, whatever the prompt length."""
        context = _make_context(num_speculative_tokens=0)
        for request_id, prompt_length in enumerate(range(1, 2 * BLOCK_SIZE_TOKENS + 1), start=1):
            context.add_request(self._request(context, prompt_length, request_id=request_id))
        assert not self._reserved_blocks(context, slice(0, context.total_request_count)).any()

    def test_update_requests_crosses_into_the_spare_without_pausing_or_allocating(self):
        """The row already owns the block it is crossing into, so nothing is drawn or paused."""
        context = _make_context()
        context.add_request(self._request(context, self.SPARE_PROMPT_LENGTH))
        blocks = self._row_blocks(context, 0)
        allocatable_before = context.kv_block_allocator.get_allocatable_count()

        self._step(context, [1])

        assert context.paused_request_count == 0, "a row holding a reserve has nothing to wait for"
        assert context.kv_block_allocator.get_allocatable_count() == allocatable_before
        assert not bool(self._reserved_blocks(context, 0)), "the reserve was consumed"
        assert int(context.request_last_kv_block_id[0]) == blocks[1]
        assert self._row_blocks(context, 0) == blocks, "no block was added past the reserve"

    def test_resume_keeps_the_spare_until_main_token_positions_are_prepared(self):
        """Resume grants capacity; preparing the next main inputs advances their pointer.

        A force-paused row already owns the block it will enter. Neither phase may
        allocate another block or route main inputs into the reserve before they reach it.
        """
        context = _make_context()
        context.add_request(self._request(context, self.SPARE_PROMPT_LENGTH))
        blocks = self._row_blocks(context, 0)

        # Park the row in the paused region without disturbing its block state, exactly as the
        # force-pause swap leaves it.
        context.paused_request_count = 1
        allocatable_before = context.kv_block_allocator.get_allocatable_count()

        active_request_count, _ = context.resume_paused_requests(0, None)

        assert active_request_count == 1
        assert (
            context.kv_block_allocator.get_allocatable_count() == allocatable_before
        ), "resume granted a second block to a row that already owned the one it crosses into"
        assert bool(self._reserved_blocks(context, 0))
        assert int(context.request_last_kv_block_id[0]) == blocks[0]
        assert self._row_blocks(context, 0) == blocks

        self._step(context, [1])
        assert not bool(self._reserved_blocks(context, 0))
        assert int(context.request_last_kv_block_id[0]) == blocks[1]
        assert context.kv_block_allocator.get_allocatable_count() == allocatable_before
        self._assert_main_token_mapping(context)

    def test_resume_still_grants_a_block_when_there_is_no_spare(self):
        """Control: the spare exclusion must not starve an ordinary boundary-crossing row."""
        context = _make_context(num_speculative_tokens=0)
        # Last token at offset 7 of 8, so the row needs a block to keep decoding.
        context.add_request(self._request(context, 2 * BLOCK_SIZE_TOKENS))
        assert not bool(self._reserved_blocks(context, 0))
        blocks = self._row_blocks(context, 0)

        context.paused_request_count = 1
        allocatable_before = context.kv_block_allocator.get_allocatable_count()

        context.resume_paused_requests(0, None)

        assert context.kv_block_allocator.get_allocatable_count() == allocatable_before - 1
        new_blocks = self._row_blocks(context, 0)
        assert len(new_blocks) == len(blocks) + 1
        assert int(context.request_last_kv_block_id[0]) == new_blocks[-1]

    def test_reserved_capacity_follows_the_row_when_a_force_pause_reorders_the_batch(self):
        """End-to-end: force-pause moves the spare holder, and resuming it allocates nothing.

        `max_tokens // (num_speculative_tokens + 1)` caps the active count, so the last rows are
        force-paused regardless of whether they hold a reserve. The spare holder is added last so
        it is the one evicted from the active window, which also exercises block ownership surviving
        `_move_book_keeping_tensors`.
        """
        max_allowed_active = 8
        context = _make_context(max_tokens=max_allowed_active * 3)
        num_filler = max_allowed_active
        for request_id in range(1, num_filler + 1):
            context.add_request(self._request(context, 1, request_id=request_id))
        spare_request_id = num_filler + 1
        context.add_request(
            self._request(context, self.SPARE_PROMPT_LENGTH, request_id=spare_request_id)
        )
        spare_blocks = self._row_blocks(context, num_filler)
        assert bool(self._reserved_blocks(context, num_filler))
        allocatable_before = context.kv_block_allocator.get_allocatable_count()

        # Step 1: the batch is one over the cap, so the spare holder is force-paused.
        self._step(context, [1] * (num_filler + 1))

        assert context.paused_request_count == 1
        paused_row = 0
        assert int(context.request_ids[paused_row]) == spare_request_id
        assert bool(
            self._reserved_blocks(context, paused_row)
        ), "the reserve is still owned; pausing does not release it"
        assert self._row_blocks(context, paused_row) == spare_blocks

        # Step 2: retire one filler so the paused row fits back in the active window.
        active_mask = [1] * (context.total_request_count - context.paused_request_count)
        active_mask[0] = 0
        self._step(context, active_mask)

        assert context.paused_request_count == 0, "the row should have resumed"
        resumed_row = context.request_ids[: context.total_request_count] == spare_request_id
        resumed_row = int(torch.nonzero(resumed_row)[0].item())
        assert not bool(self._reserved_blocks(context, resumed_row))
        assert (
            self._row_blocks(context, resumed_row) == spare_blocks
        ), "resume granted a block past the reserve instead of crossing into it"
        assert int(context.request_last_kv_block_id[resumed_row]) == spare_blocks[1]
        # The only pool movement is the finished filler's block going back.
        assert context.kv_block_allocator.get_allocatable_count() >= allocatable_before

    def test_reusing_a_vacated_row_does_not_inherit_reserved_capacity(self):
        """New admission replaces the prior row's ownership and committed-token span."""
        context = _make_context()
        context.add_request(self._request(context, self.NO_SPARE_PROMPT_LENGTH, request_id=1))
        context.add_request(self._request(context, self.SPARE_PROMPT_LENGTH, request_id=2))
        self._step(context, [0, 1])

        assert int(context.request_ids[0]) == 2
        context.add_request(self._request(context, self.NO_SPARE_PROMPT_LENGTH, request_id=3))
        assert self._reserved_blocks(context, 1) == 0
        assert len(self._row_blocks(context, 1)) == 1
        self._step(context, [1, 1])
        self._assert_main_token_mapping(context)

    def test_imported_decode_replaces_recycled_row_metadata(self):
        """Import derives capacity from the new request and clears its MTP annotations."""
        context = _make_context()
        input_token_count = context.num_speculative_tokens + 1
        # The imported main tokens end at offset 6, so its next step needs a block.
        request = self._request(context, 12, request_id=7)
        context.request_kv_block_counts[0] = 3
        context.request_kv_length_offsets[0] = 0
        context.request_query_lengths[0] = 1
        context.mtp_metadata.request_matched_prefix_blocks[0] = 2
        assert self._reserved_blocks(context, 0) == 2
        blocks = context.kv_block_allocator.allocate_memory_blocks(2).tolist()

        admit_prefilled_decode(
            context,
            request,
            prompt_block_ids=blocks,
            continuation_block_ids=[],
            input_tokens=list(range(input_token_count)),
        )

        assert self._reserved_blocks(context, 0) == 0
        assert context.mtp_metadata.request_matched_prefix_blocks[0] == 0
        assert bool(context._get_async_sched_rows_requiring_new_block()[0])

    def test_eviction_does_not_charge_a_spare_holder_for_a_block_it_will_not_draw(self):
        """A paused reserve holder resumes for free, so it must not push the evict count up.

        `evict_overflow_paused_requests` sizes the eviction from how many blocks the survivors
        need. Counting a spare holder there evicts a request that could have resumed without
        touching the pool, throwing away a full prefill under exactly the memory pressure the
        reserve exists to survive.
        """

        def _context_with_one_overflow_paused_row(has_spare):
            context = _make_context()
            context.add_request(self._request(context, self.SPARE_PROMPT_LENGTH))
            assert bool(self._reserved_blocks(context, 0))
            if not has_spare:
                reserve = context.request_to_kv_block_ids[0, 1:2].clone()
                context.kv_block_allocator.release_memory_blocks(reserve)
                context.request_to_kv_block_ids[0, 1] = -1
                context.request_kv_block_counts[0] -= 1
            context.paused_request_count = 1
            # Force a one-row overflow of the paused retention budget, and an empty pool, so the
            # decision turns purely on whether the row is charged for a block.
            context._get_paused_request_count_within_block_budget = lambda: 0
            context._get_releasable_block_counts = lambda start, end: [0, 1]
            context.kv_block_allocator.get_allocatable_count = lambda: 0
            return context

        device = torch.cuda.current_device()
        next_tokens = torch.full((1,), 100, device=device, dtype=torch.int64)

        spare_context = _context_with_one_overflow_paused_row(has_spare=True)
        assert (
            spare_context.evict_overflow_paused_requests(0, next_tokens) is None
        ), "a paused row that already owns its next block was evicted to free one"
        assert spare_context.paused_request_count == 1

        # Control: the identical row without a reserve genuinely needs a block, so it is evicted.
        plain_context = _context_with_one_overflow_paused_row(has_spare=False)
        evicted = plain_context.evict_overflow_paused_requests(0, next_tokens)
        assert evicted is not None and evicted.numel() == 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mtp_use_repeated_layer": False},
            {"mtp_layer_type_list": [Symbols.MAMBA, Symbols.MLP]},
            {"mtp_layer_type_list": [Symbols.ATTENTION, Symbols.ATTENTION]},
        ],
        ids=["per-depth-head", "recurrent-head", "multi-attention-head"],
    )
    def test_a_head_that_writes_no_draft_kv_reserves_nothing(self, kwargs):
        """The reserve exists only for draft KV writes, so it follows `enable_mtp_kv_cache`.

        These heads still draft (`num_speculative_tokens > 0`) but write no KV into the main
        block table, so the ordinary step-5 pause grants their blocks a step later as usual.
        Reserving would pin an extra block per boundary-adjacent prompt for its whole lifetime.
        """
        context = _make_context(**kwargs)
        assert context.num_speculative_tokens > 0 and not context.enable_mtp_kv_cache
        context.add_request(self._request(context, self.SPARE_PROMPT_LENGTH))

        assert not bool(self._reserved_blocks(context, 0))
        assert self._row_blocks(context, 0) == [int(context.request_last_kv_block_id[0])]
