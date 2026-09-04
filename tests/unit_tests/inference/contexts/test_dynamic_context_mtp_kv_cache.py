# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for the MTP draft-KV bookkeeping on `DynamicInferenceContext`.

The MTP KV cache reserves one extra attention-layer slot in the shared KV buffer and drives its
own attention metadata (bypassing the coalesced CPU->GPU bookkeeping transfer) so the MTP draft
forwards never disturb the main step's state. The methods under test are pure GPU tensor
bookkeeping -- no model forward is involved -- so they are exercised directly against a real
context with hand-seeded per-request state:

  * `_mtp_begin_decode` / `_mtp_begin_decode_for_capture` -- enter MTP-forward mode
  * `_mtp_setup_decode_step` / `_mtp_advance_decode_step` -- one draft depth (roll-by-one)
  * `_mtp_setup_prefill_step` / `_mtp_finalize_prefill_step` -- the varlen commit pass
  * `_mtp_snapshot_prerewind_block_table` -- the pre-rewind block-table snapshot
  * `_mtp_end_decode` -- leave MTP-forward mode

The invariants asserted are the ones the draft attention depends on: write position
`P_r = base_position_r - 1 + depth`, read length `kv_len = P_r + 1` (write-then-attend), and
padding rows that never index real KV.
"""

import pytest
import torch

from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
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
) -> DynamicInferenceContext:
    """Build a real `DynamicInferenceContext`, MTP-KV-enabled unless a gate is switched off."""
    if is_hybrid_model or mtp_layer_type_list is not None:
        mamba_inference_state_config = MambaInferenceStateConfig(
            layer_type_list=[Symbols.MAMBA, Symbols.MLP, Symbols.ATTENTION, Symbols.MLP],
            conv_states_shape=(544, 4),
            ssm_states_shape=(8, 64, 16),
            conv_states_dtype=torch.bfloat16,
            ssm_states_dtype=torch.bfloat16,
            mtp_layer_type_list=mtp_layer_type_list,
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
            max_tokens=256,
            max_requests=max_requests,
            num_speculative_tokens=num_speculative_tokens,
            mamba_inference_state_config=mamba_inference_state_config,
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

    def test_disabled_without_speculative_decoding(self):
        """No drafts means no draft KV to populate."""
        context = _make_context(num_speculative_tokens=0)
        assert context.enable_mtp_kv_cache is False
        assert context.mtp_kv_layer_slot is None
        assert context.num_attention_layers == NUM_LAYERS

    def test_disabled_without_mtp_layers(self):
        """A model with no MTP head has nothing to seed."""
        context = _make_context(mtp_num_layers=None)
        assert context.enable_mtp_kv_cache is False
        assert context.mtp_kv_layer_slot is None

    def test_disabled_for_per_depth_head(self):
        """A per-depth head has no single `mtp.layers[0]` to seed through."""
        context = _make_context(mtp_use_repeated_layer=False)
        assert context.enable_mtp_kv_cache is False
        assert context.mtp_kv_layer_slot is None

    def test_disabled_for_recurrent_mtp_head(self):
        """A Mamba MTP head has no KV to append, so the reserved slot would go unused."""
        context = _make_context(mtp_layer_type_list=[Symbols.MAMBA, Symbols.MLP])
        assert context.enable_mtp_kv_cache is False

    def test_disabled_for_multi_attention_mtp_head(self):
        """Two attention layers in the head would collide on the single reserved slot."""
        context = _make_context(mtp_layer_type_list=[Symbols.ATTENTION, Symbols.ATTENTION])
        assert context.enable_mtp_kv_cache is False

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

        assert context._mtp_forward_active is True
        assert context._mtp_graphed is False
        assert context._mtp_active_request_count == 2
        assert context._mtp_padded_count == 2
        assert context._mtp_offsets_gpu.dtype == torch.int32
        assert context._mtp_offsets_gpu.cpu().tolist() == [5, 11]
        assert context._mtp_block_table_gpu[:, :2].cpu().tolist() == [[3, 4], [7, 9]]

    def test_begin_decode_clones_caller_start_positions(self):
        """`_mtp_advance_decode_step` must not mutate the caller's `base_position - 1` tensor."""
        context = _make_context()
        _seed_requests(context, [[3], [7]])
        start_positions = torch.tensor([5, 11], device=torch.cuda.current_device())

        context._mtp_begin_decode(2, 2, start_positions)
        context._mtp_advance_decode_step()
        context._mtp_advance_decode_step()

        assert start_positions.cpu().tolist() == [
            5,
            11,
        ], "advancing the MTP write positions aliased and corrupted the caller's tensor"
        assert context._mtp_offsets_gpu.cpu().tolist() == [7, 13]

    def test_begin_decode_honors_paused_request_offset(self):
        """Active requests start at `paused_request_count`, not at row 0."""
        context = _make_context()
        _seed_requests(context, [[3, 4], [7, 9]], paused_request_count=2)
        start_positions = torch.tensor([5, 11], device=torch.cuda.current_device())

        context._mtp_begin_decode(2, 2, start_positions)

        # Rows 0-1 are paused (still the -1 fill); the block table must hold the ACTIVE rows.
        assert context._mtp_block_table_gpu[:, :2].cpu().tolist() == [[3, 4], [7, 9]]

    def test_begin_decode_prefers_prerewind_block_table(self):
        """Deep drafts extend past the accepted range into blocks rewind has since released."""
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        context._mtp_snapshot_prerewind_block_table()
        # Simulate `_rewind_kv_cache` releasing the second block and clearing it to -1.
        context.request_to_kv_block_ids[0, 1] = -1

        context._mtp_begin_decode(1, 1, torch.tensor([9], device=torch.cuda.current_device()))

        assert int(context._mtp_block_table_gpu[0, 1].item()) == 4, (
            "the draft loop read the post-rewind block table and would send deep drafts to "
            "block -1"
        )

    def test_begin_decode_falls_back_to_live_block_table(self):
        """With no snapshot taken (e.g. a pure-prefill step) the live table is used."""
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        assert getattr(context, "_mtp_prerewind_block_table", None) is None

        context._mtp_begin_decode(1, 1, torch.tensor([9], device=torch.cuda.current_device()))

        assert context._mtp_block_table_gpu[0, :2].cpu().tolist() == [3, 4]

    def test_snapshot_is_a_copy_not_an_alias(self):
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        context._mtp_snapshot_prerewind_block_table()
        context.request_to_kv_block_ids[0, 0] = 99

        assert int(context._mtp_prerewind_block_table[0, 0].item()) == 3

    def test_snapshot_is_a_noop_when_disabled(self):
        context = _make_context(num_speculative_tokens=0)
        context._mtp_snapshot_prerewind_block_table()
        assert getattr(context, "_mtp_prerewind_block_table", None) is None

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

        context._mtp_advance_decode_step()
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
            context._mtp_advance_decode_step()

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
        context._mtp_finalize_prefill_step()
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

        context._mtp_begin_decode_for_capture(2)
        seen_positions = []
        for _ in range(3):
            context._mtp_setup_decode_step()
            seen_positions.append(gv.token_to_position_in_request[:2].cpu().tolist())
            assert (gv.token_to_block_idx[:2] == dummy).all()
            context._mtp_advance_decode_step()
        context._mtp_end_decode()

        assert seen_positions == [[0, 0], [1, 1], [2, 2]]
        assert context._mtp_forward_active is False

    def test_capture_crossing_a_block_boundary_stays_on_scratch(self):
        """Even past a block boundary the capture must not touch a real block."""
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        dummy = context.kv_block_allocator.dummy_block_idx

        context._mtp_begin_decode_for_capture(1)
        for _ in range(BLOCK_SIZE_TOKENS + 2):
            context._mtp_setup_decode_step()
            assert int(context.gpu_view.token_to_block_idx[0].item()) == dummy
            context._mtp_advance_decode_step()

    def test_capture_uses_graph_metadata_at_the_padded_size(self):
        """Capture-time launch bounds must match the runtime graphed step's."""
        context = _make_context()
        _seed_requests(context, [[3, 4]])

        context._mtp_begin_decode_for_capture(4)
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

        context._mtp_begin_decode_for_capture(4)

        dummy = context.kv_block_allocator.dummy_block_idx
        assert context._mtp_graphed is True
        assert context._mtp_forward_active is True
        assert context._mtp_active_request_count == 4
        assert context._mtp_padded_count == 4
        assert (context._mtp_offsets_gpu == 0).all()
        assert (context._mtp_block_table_gpu == dummy).all()

        # A capture-time setup step must not write any real block id.
        context._mtp_setup_decode_step()
        assert (context.gpu_view.token_to_block_idx[:4] == dummy).all()

    def test_end_decode_leaves_mtp_forward_mode(self):
        context = _make_context()
        _seed_requests(context, [[3, 4]])
        context._mtp_begin_decode(1, 1, torch.tensor([5], device=torch.cuda.current_device()))
        assert context._mtp_forward_active is True

        context._mtp_end_decode()

        assert context._mtp_forward_active is False

    def test_begin_decode_asserts_when_disabled(self):
        context = _make_context(num_speculative_tokens=0)
        with pytest.raises(AssertionError):
            context._mtp_begin_decode(1, 1, torch.tensor([5], device=torch.cuda.current_device()))


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
        """Build the GPU block table argument the controller passes in."""
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
        assert gv.mha_query_lengths[2:4].cpu().tolist() == [0, 0]
        assert gv.mha_kv_seq_lengths[2:4].cpu().tolist() == [0, 0]
        assert gv.mha_cu_query_seq_lengths[2:5].cpu().tolist() == [5, 5, 5]
        assert gv.mha_cu_kv_seq_lengths[2:5].cpu().tolist() == [5, 5, 5]
        assert (gv.mha_block_table[2:4] == -1).all()
        assert context.active_token_count == 5
        assert context.padded_active_token_count == 8

    def test_prefill_step_forces_varlen_path_on_a_pure_decode_step(self):
        """`num_prefill_requests` is forced >= 1 so the ragged forward avoids the decode kernel."""
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

        assert context.num_prefill_requests == 2
        assert context.is_decode_only() is False
        assert context._using_cuda_graph_this_step is False
        assert context._mtp_forward_active is True
        assert context.active_attn_metadata is context.non_graph_attn_metadata
        mha = context.non_graph_attn_metadata["mha_metadata"]
        assert mha.state_data["max_seqlen_q"] == 2
        assert mha.state_data["max_seqlen_k"] == 2

    def test_finalize_restores_the_main_step_prefill_count(self):
        """The commit pass must hand the step back exactly as it found it."""
        context = _make_context()
        device = torch.cuda.current_device()
        context.num_prefill_requests = 0
        block_table = self._block_table(context, [[3]])

        context._mtp_setup_prefill_step(
            append_counts=torch.tensor([2], device=device), block_table_prefill=block_table
        )
        assert context.num_prefill_requests == 1

        context._mtp_finalize_prefill_step()

        assert context.num_prefill_requests == 0
        assert context._mtp_forward_active is False

    def test_prefill_step_asserts_when_disabled(self):
        context = _make_context(num_speculative_tokens=0)
        device = torch.cuda.current_device()
        with pytest.raises(AssertionError):
            context._mtp_setup_prefill_step(
                append_counts=torch.tensor([2], device=device),
                block_table_prefill=self._block_table(context, [[3]]),
            )
