# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GB200 gate for full-iteration CUDA Graph dynamic balanced routes.

Run with four GPUs so the production wrapper captures a real balanced CP4
two-hop all-to-all path::

    TEST_DIR=tests/unit_tests/transformer/experimental_attention_variant
    torchrun --nproc-per-node=4 -m pytest -xvs \
        "$TEST_DIR/test_full_iteration_dynamic_route_graph.py"

The test deliberately traverses the production full-iteration input path:
``FullCudaGraphWrapper`` runs the eager GPT prepare callback, stages all nine
fixed-address tensor owners, reconstructs ``PackedSeqParams`` inside capture,
and then replays one real fused DSV4 attention forward/backward graph while the
pack metadata changes outside the graph.
"""

import gc
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch
import torch.distributed as dist

import megatron.core.parallel_state as parallel_state
import pretrain_gpt
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.full_cuda_graph import (
    FullCudaGraphPreparedIterator,
    FullCudaGraphWrapper,
    StaticBufferLoader,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant import cp_balanced_indexer
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention_cp import (
    _DSV4_CP_FUSED_KERNELS_UNAVAILABLE_REASON,
    _DSV4_CP_GRAPH_FUSED_BF16_ATOL,
    _DSV4_CP_GRAPH_FUSED_FP32_ATOL,
    _DSV4_CP_GRAPH_FUSED_RTOL,
    _assert_cp_graph_bitwise_match,
    _assert_cp_graph_fused_grad_match,
    _build_attention,
    _copy_module_parameters,
    _dsv4_cp_fused_kernels_available,
    _make_dsv4_cp_config,
    _make_hidden_and_grad,
    _make_thd_packed_seq_params,
    _max_int_across_world,
    _run_dsv4_attention_forward_backward,
    _zero_existing_grads,
)

pytestmark = pytest.mark.launch_on_gb200

_CP_SIZE = 4
_REPLAY_COUNT = 30
_PACK_REAL_SEGMENTS = ((28, 88, 120, 248), (60, 56, 116, 240), (44, 72, 112, 232))
_PACK_PADDED_SEGMENTS = ((32, 96, 128, 256), (64, 64, 128, 256), (48, 80, 128, 256))
# Exercise restoration explicitly, rather than merely cycling A/B/C forever.
_PACK_SCHEDULE = (0, 1, 2, 0)
_ROUTE_OWNER_NAMES = ("layout_i32", "route_i64")
_PREPARED_ROUTE_OWNER_NAMES = (
    "dsa_cp_graph_layout_buffer",
    "dsa_cp_graph_route_buffer",
)
_PREPARED_OWNER_NAMES = (
    "tokens",
    "labels",
    "loss_mask",
    "position_ids",
    "padding_mask",
    "cu_seqlens",
    "cu_seqlens_padded",
) + _PREPARED_ROUTE_OWNER_NAMES


def _clear_cuda_state():
    """Release graph-private allocations between distributed GPU tests."""
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _phase(name):
    """Emit a flushed per-rank marker around long first-use fused-kernel phases."""
    rank = dist.get_rank() if dist.is_initialized() else -1
    print(f"[fulliter-route-toy rank={rank}] {name}", flush=True)


def _assert_fast_fused_grad_match(actual, expected, label):
    """Apply the production tolerance, deferring expensive FP64 diagnostics to failures."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all(), f"{label}: actual has non-finite values"
    assert torch.isfinite(expected).all(), f"{label}: expected has non-finite values"
    if actual.dtype == torch.bfloat16:
        atol = _DSV4_CP_GRAPH_FUSED_BF16_ATOL
    elif actual.dtype == torch.float32:
        atol = _DSV4_CP_GRAPH_FUSED_FP32_ATOL
    else:
        raise AssertionError(f"{label}: unsupported fused-gradient dtype {actual.dtype}")
    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=_DSV4_CP_GRAPH_FUSED_RTOL,
            atol=atol,
            msg=f"{label}: fused graph/eager gradient mismatch",
        )
    except AssertionError:
        # Reuse the shared FP64 cosine/tensor-similarity diagnostics only on a
        # real mismatch; doing those reductions for every large parameter on
        # all 30 successful replays makes this gate needlessly take minutes.
        _assert_cp_graph_fused_grad_match(actual, expected, label)


def _build_and_attach_route(packed, cp_group, capacity):
    """Build the real rank-local route and attach its two storage owners."""
    plan = cp_balanced_indexer.build_graph_dynamic_plan(
        packed.cu_seqlens_q_padded, cp_group, capacity
    )
    cp_balanced_indexer.attach_graph_dynamic_plan(packed, plan)
    return plan


def _reset_full_cuda_graph_wrapper_state():
    """Reset process-global wrapper inputs before this process-isolated gate."""
    for graph in FullCudaGraphWrapper.cuda_graph.values():
        if graph is not None and hasattr(graph, "reset"):
            graph.reset()
    FullCudaGraphWrapper.curr_iteration = {"training": 0, "validation": 0}
    FullCudaGraphWrapper.cuda_graph = {"training": None, "validation": None}
    FullCudaGraphWrapper.result = {"training": None, "validation": None}
    FullCudaGraphWrapper.prepared_run_signatures = {"training": None, "validation": None}
    StaticBufferLoader.static_buffers = {"training": [], "validation": []}
    StaticBufferLoader.reset_prepared_state()


def _make_local_prepared_batch(packed, cp_rank, capacity, marker):
    """Build one rank-local seven-value GPT batch for an underfilled THD pack."""
    real_cu = packed.cu_seqlens_q
    padded_cu = packed.cu_seqlens_q_padded
    full_padding_mask = torch.ones(capacity, dtype=torch.bool, device="cuda")
    full_position_ids = torch.zeros(capacity, dtype=torch.int64, device="cuda")
    for index in range(real_cu.numel() - 1):
        real_len = int((real_cu[index + 1] - real_cu[index]).item())
        physical_start = int(padded_cu[index].item())
        physical_end = int(padded_cu[index + 1].item())
        assert real_len <= physical_end - physical_start
        full_padding_mask[physical_start : physical_start + real_len] = False
        full_position_ids[physical_start : physical_start + real_len] = torch.arange(
            real_len, dtype=torch.int64, device="cuda"
        )

    local_rows = capacity // _CP_SIZE
    local_start = cp_rank * local_rows
    local_end = local_start + local_rows
    padding_mask = full_padding_mask[local_start:local_end].view(1, local_rows).contiguous()
    position_ids = full_position_ids[local_start:local_end].view(1, local_rows).contiguous()
    tokens = (
        torch.arange(local_start, local_end, dtype=torch.int64, device="cuda")
        + marker * capacity
    ).view(1, local_rows)
    tokens = tokens.masked_fill(padding_mask, 0).contiguous()
    labels = (tokens + 1).masked_fill(padding_mask, 0).contiguous()
    loss_mask = (~padding_mask).to(torch.float32).contiguous()
    return tokens, labels, loss_mask, None, position_ids, packed, padding_mask


def _prepared_source_snapshot(batch):
    """Return the exact nine tensor values expected in the wrapper owners."""
    packed = batch[5]
    plan = cp_balanced_indexer.get_graph_dynamic_plan(packed)
    assert plan is not None
    return {
        "tokens": batch[0],
        "labels": batch[1],
        "loss_mask": batch[2],
        "position_ids": batch[4],
        "padding_mask": batch[6],
        "cu_seqlens": packed.cu_seqlens_q,
        "cu_seqlens_padded": packed.cu_seqlens_q_padded,
        "dsa_cp_graph_layout_buffer": plan["layout_i32"],
        "dsa_cp_graph_route_buffer": plan["route_i64"],
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
class TestFullIterationDynamicBalancedRouteGraph:
    """One complete attention forward/backward graph accepts 30 changing packs."""

    @pytest.fixture(scope="class", autouse=True)
    def distributed_cp4(self, request):
        """Create the CP4 group required by the real two-hop route consumer."""
        if Utils.world_size < _CP_SIZE:
            pytest.skip("dynamic balanced route gate requires torchrun with four GPU ranks")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=_CP_SIZE,
        )
        torch.manual_seed(7801)
        model_parallel_cuda_manual_seed(7801)

        cls = request.cls
        cls.cp_rank = parallel_state.get_context_parallel_rank()
        cls.pg = ProcessGroupCollection.use_mpu_process_groups()
        cls.fused_kernels_available = _dsv4_cp_fused_kernels_available()
        cp_group = cls.pg.cp
        assert cp_group.size() == _CP_SIZE

        yield

        _clear_cuda_state()
        Utils.destroy_model_parallel()
        # The graph registers NCCL buffers against this test-owned subgroup.
        # Destroy it explicitly so later CUDA tests do not retain graph state.
        if cp_group is dist.group.WORLD:
            raise RuntimeError("dynamic route toy unexpectedly reused the world process group")
        if dist.distributed_c10d._world.pg_map.get(cp_group) is not None:
            dist.destroy_process_group(cp_group)
        cls.pg = None
        _clear_cuda_state()

    @pytest.fixture(autouse=True)
    def clean_case(self):
        _clear_cuda_state()
        _reset_full_cuda_graph_wrapper_state()
        DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        yield
        _reset_full_cuda_graph_wrapper_state()
        DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        _clear_cuda_state()

    def test_production_wrapper_replays_30_underfilled_dynamic_packs(self, monkeypatch):
        """Replay changing prepared owners through one real DSA forward/backward graph."""
        if not self.fused_kernels_available:
            pytest.skip(_DSV4_CP_FUSED_KERNELS_UNAVAILABLE_REASON)

        capacity = sum(_PACK_PADDED_SEGMENTS[0])
        assert all(sum(composition) == capacity for composition in _PACK_PADDED_SEGMENTS)
        assert all(
            length % (2 * _CP_SIZE) == 0
            for composition in _PACK_PADDED_SEGMENTS
            for length in composition
        )
        assert all(
            len(real) == len(padded) and all(r <= p for r, p in zip(real, padded))
            for real, padded in zip(_PACK_REAL_SEGMENTS, _PACK_PADDED_SEGMENTS)
        )
        # This gate must cover genuinely underfilled packs, not merely different
        # boundaries whose valid-token total still fills the graph capacity.
        real_totals = tuple(sum(composition) for composition in _PACK_REAL_SEGMENTS)
        total_padding = tuple(capacity - total for total in real_totals)
        final_sequence_padding = tuple(
            padded[-1] - real[-1]
            for real, padded in zip(_PACK_REAL_SEGMENTS, _PACK_PADDED_SEGMENTS)
        )
        assert real_totals == (484, 472, 460)
        assert total_padding == (28, 40, 52)
        assert final_sequence_padding == (8, 16, 24)

        source_packs = tuple(
            _make_thd_packed_seq_params(real, padded)
            for real, padded in zip(_PACK_REAL_SEGMENTS, _PACK_PADDED_SEGMENTS)
        )
        # The production prepared ABI pins max_seqlen to the global graph
        # capacity and retains the fixed CP group as a Python capture invariant.
        for packed in source_packs:
            packed.max_seqlen_q = capacity
            packed.max_seqlen_kv = capacity
            packed.cp_group = self.pg.cp
            packed.local_cp_size = None
            packed.total_tokens = None
            packed.pad_between_seqs = True
        source_plans = tuple(
            _build_and_attach_route(packed, self.pg.cp, capacity) for packed in source_packs
        )
        reference_plan = source_plans[0]

        assert all(
            plan["layout_i32"].shape == reference_plan["layout_i32"].shape
            for plan in source_plans
        )
        assert all(
            plan["route_i64"].shape == reference_plan["route_i64"].shape
            for plan in source_plans
        )
        assert all(
            plan["validated_cu"].shape == reference_plan["validated_cu"].shape
            for plan in source_plans
        )
        assert (
            _max_int_across_world(
                int(
                    any(
                        not torch.equal(source_plans[0][name], source_plans[index][name])
                        for index in (1, 2)
                        for name in _ROUTE_OWNER_NAMES
                    )
                )
            )
            == 1
        ), "A/B/C must produce different route metadata on at least one CP rank"

        config = _make_dsv4_cp_config(
            context_parallel_size=_CP_SIZE,
            num_layers=2,
            hidden_size=256,
            num_attention_heads=64,
            # FlashMLA's DSv4 sparse-prefill kernel is specialized for the
            # production h_q=64 and D_QK=D_V=512 contract. Keep the model and
            # low-rank widths compact, but do not shrink these kernel-facing
            # dimensions.
            v_head_dim=512,
            qk_pos_emb_head_dim=64,
            q_lora_rank=64,
            o_groups=8,
            o_lora_rank=64,
            csa_compress_ratios=[0, 4],
            csa_window_size=64,
            dsa_indexer_n_heads=64,
            dsa_indexer_head_dim=128,
            dsa_indexer_topk=128,
            dsa_indexer_loss_coeff=1.0e-2,
            dsa_indexer_use_sparse_loss=True,
            calculate_per_token_loss=True,
            use_fused_kernels=True,
            apply_rope_fusion=True,
            dsa_cp_balance_indexer=True,
            cuda_graph_impl="full_iteration",
            cuda_graph_modules=[],
            max_seqlen_per_dp_cp_rank=capacity // _CP_SIZE,
            pad_packed_seq_alignment="max",
            thd_max_packed_sequences=len(_PACK_PADDED_SEGMENTS[0]),
        )
        assert config.num_attention_heads == 64
        assert config.qk_head_dim + config.qk_pos_emb_head_dim == 512
        assert config.v_head_dim == 512
        assert (config.dsa_indexer_n_heads, config.dsa_indexer_head_dim) == (64, 128)
        assert config.dsa_indexer_topk == 128
        assert config.dsa_cp_balance_indexer_graph_dynamic_packs

        torch.manual_seed(7802)
        model_parallel_cuda_manual_seed(7802)
        graph_attention = _build_attention(config, layer_number=2, pg_collection=self.pg).cuda()
        eager_attention = _build_attention(config, layer_number=2, pg_collection=self.pg).cuda()
        graph_attention.train()
        eager_attention.train()
        # The production prepare callback asks the owning model chunk for this
        # optional VPP tag even though the supported FICG contract is PP1/VPP1.
        graph_attention.vp_stage = None
        _copy_module_parameters(graph_attention, eager_attention)
        DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(0.125, device="cuda"))

        local_rows = capacity // _CP_SIZE
        local_indices = torch.arange(
            self.cp_rank * local_rows, (self.cp_rank + 1) * local_rows, device="cuda"
        )
        full_hidden, full_grad = _make_hidden_and_grad(capacity, config.hidden_size)
        local_hidden = full_hidden.index_select(0, local_indices)
        local_grad = full_grad.index_select(0, local_indices)
        static_hidden = local_hidden.detach().clone().requires_grad_(True)
        static_grad = local_grad.detach().clone()

        source_batches = tuple(
            _make_local_prepared_batch(packed, self.cp_rank, capacity, pack_index + 1)
            for pack_index, packed in enumerate(source_packs)
        )
        source_snapshots = tuple(_prepared_source_snapshot(batch) for batch in source_batches)

        # Preserve the real prepared branch of GPT get_batch. Raw inputs are
        # already finalized seven-value batches in this focused gate, so only
        # that half of the dispatcher is replaced by next(raw_iterator).
        original_get_batch = pretrain_gpt.get_batch

        def get_batch_dispatch(
            data_iterator, vp_stage=None, *, config=None, pg_collection=None
        ):
            if isinstance(data_iterator, FullCudaGraphPreparedIterator):
                return original_get_batch(
                    data_iterator,
                    vp_stage,
                    config=config,
                    pg_collection=pg_collection,
                )
            assert config is graph_attention.config
            assert pg_collection is self.pg
            return next(data_iterator)

        monkeypatch.setattr(pretrain_gpt, "get_batch", get_batch_dispatch)
        monkeypatch.setattr(pretrain_gpt, "get_args", lambda: object())
        monkeypatch.setattr(
            pretrain_gpt,
            "core_transformer_config_from_args",
            lambda _args: pytest.fail("prepared path rebuilt TransformerConfig from global args"),
        )

        def full_iteration_forward_backward(**kwargs):
            batch = pretrain_gpt.get_batch(kwargs["data_iterator"][0])
            packed = batch[5]
            output, _ = kwargs["model"][0](
                hidden_states=static_hidden,
                attention_mask=None,
                packed_seq_params=packed,
            )
            output.backward(static_grad)
            return output

        def unused_forward_step(*_args, **_kwargs):
            raise AssertionError("focused full-iteration gate owns its forward/backward callable")

        wrapper = FullCudaGraphWrapper(
            full_iteration_forward_backward,
            cuda_graph_warmup_steps=3,
            batch_prepare_func=pretrain_gpt.prepare_full_cuda_graph_dynamic_packed_batch,
        )

        def run_wrapper(source_batch):
            return wrapper(
                forward_step_func=unused_forward_step,
                data_iterator=[iter((source_batch,))],
                model=[graph_attention],
                num_microbatches=1,
                seq_length=capacity,
                micro_batch_size=1,
                decoder_seq_length=None,
                forward_only=False,
                pg_collection=self.pg,
            )

        _phase("eager-baselines-begin")
        eager_results = tuple(
            _run_dsv4_attention_forward_backward(
                eager_attention,
                local_hidden.detach().clone().requires_grad_(True),
                local_grad,
                packed,
            )
            for packed in source_packs
        )
        _phase("eager-baselines-end")

        # Exercise the same three eager warmups used by the prior raw graph
        # gate, but through the wrapper so its owners and run signature are
        # established before capture.
        for warmup_index in range(3):
            _phase(f"warmup-{warmup_index + 1}-begin")
            _zero_existing_grads(graph_attention, static_hidden)
            run_wrapper(source_batches[0])
            torch.cuda.synchronize()
            _phase(f"warmup-{warmup_index + 1}-end")

        prepared_owners = StaticBufferLoader.prepared_static_buffers["training"][0][0]
        assert tuple(prepared_owners) == _PREPARED_OWNER_NAMES
        owner_ptrs = {name: tensor.data_ptr() for name, tensor in prepared_owners.items()}

        _phase("capture-begin")
        _zero_existing_grads(graph_attention, static_hidden)
        run_wrapper(source_batches[0])
        torch.cuda.synchronize()
        _phase("capture-end")
        assert FullCudaGraphWrapper.cuda_graph["training"] is not None
        assert {
            name: tensor.data_ptr() for name, tensor in prepared_owners.items()
        } == owner_ptrs

        first_graph_outputs = {}
        observed_nonzero_indexer_grad = False
        _phase("replays-begin")
        for replay_index in range(_REPLAY_COUNT):
            _phase(f"replay-{replay_index:02d}-metadata-begin")
            pack_index = _PACK_SCHEDULE[replay_index % len(_PACK_SCHEDULE)]
            _phase(f"replay-{replay_index:02d}-graph-begin")
            _zero_existing_grads(graph_attention, static_hidden)
            graph_output = run_wrapper(source_batches[pack_index])
            torch.cuda.synchronize()
            _phase(f"replay-{replay_index:02d}-graph-end")

            assert {
                name: tensor.data_ptr() for name, tensor in prepared_owners.items()
            } == owner_ptrs
            for name, expected in source_snapshots[pack_index].items():
                torch.testing.assert_close(prepared_owners[name], expected, rtol=0, atol=0)

            graph_result = (
                graph_output.detach().clone(),
                static_hidden.grad.detach().clone(),
                {
                    name: parameter.grad.detach().clone()
                    for name, parameter in graph_attention.named_parameters()
                    if parameter.grad is not None
                },
            )
            eager_result = eager_results[pack_index]
            label = f"full_graph_replay_{replay_index:02d}_pack_{pack_index}"
            _assert_cp_graph_bitwise_match(graph_result[0], eager_result[0], f"{label}:output")
            _assert_cp_graph_fused_grad_match(
                graph_result[1], eager_result[1], f"{label}:hidden_grad"
            )
            assert graph_result[2].keys() == eager_result[2].keys()
            observed_nonzero_indexer_grad = observed_nonzero_indexer_grad or any(
                "core_attention.indexer" in name and bool(torch.count_nonzero(grad).item())
                for name, grad in graph_result[2].items()
            )
            for name, graph_grad in graph_result[2].items():
                _assert_fast_fused_grad_match(
                    graph_grad, eager_result[2][name], f"{label}:param_grad:{name}"
                )

            # Replaying A after B/C must restore exactly A's graph result.
            if pack_index in first_graph_outputs:
                _assert_cp_graph_bitwise_match(
                    graph_result[0], first_graph_outputs[pack_index], f"{label}:repeat_restore"
                )
            else:
                first_graph_outputs[pack_index] = graph_result[0]
            _phase(f"replay-{replay_index:02d}-checked")
        _phase("replays-end")

        assert set(first_graph_outputs) == {0, 1, 2}
        assert observed_nonzero_indexer_grad, "nonzero auxiliary indexer gradients were not tested"
        assert (
            _max_int_across_world(
                int(not torch.equal(first_graph_outputs[0], first_graph_outputs[1]))
            )
            == 1
        ), "the real consumer must observe the changing route on at least one CP rank"

        wrapper.reset_cuda_graph("training")
        del graph_output, graph_attention, eager_attention
        del static_hidden, static_grad, full_hidden, full_grad, local_hidden, local_grad
        _clear_cuda_state()
