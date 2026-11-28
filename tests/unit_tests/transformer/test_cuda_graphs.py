# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import functools
import gc
import inspect
import logging
import math
import os
import sys
import warnings
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

import megatron.core.transformer.cuda_graphs as cuda_graphs_module
import megatron.core.transformer.moe.paged_stash as paged_stash_module
import megatron.core.transformer.transformer_config as transformer_config_module
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.extensions.te_cuda_graphs import set_te_cuda_graph_current_microbatch
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_block import HybridStack, HyperConnectionHybridLayer
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    CheckpointWithoutOutput,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graph_config import validate_moe_cuda_graph_support
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    TECudaGraphHelper,
    VisionTECudaGraphHelper,
    _chunk_cuda_graph_block_is_graphable,
    _CudagraphGlobalRecord,
    _CudagraphReplayNode,
    _CudaGraphRunner,
    _te_cuda_graph_layer_is_graphable,
    create_cudagraphs,
    delete_cuda_graphs,
    is_graph_warmup,
)
from megatron.core.transformer.enums import (
    AttnBackend,
    CudaGraphModule,
    CudaGraphScope,
    InferenceCudaGraphScope,
)
from megatron.core.transformer.experimental_attention_variant.dsa_logging import (
    DSAIndexerLossLoggingHelper,
    get_dsa_metric_tracker_size,
    initialize_dsa_metric_tracker,
)
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.moe.fused_a2a import reset_hybrid_ep_buffer
from megatron.core.transformer.spec_utils import ModuleSpec, get_submodules
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.utils import get_attr_wrapped_model, is_fa_min_version, is_te_min_version
from megatron.training import arguments as training_arguments
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    get_run_config,
    set_args,
    set_global_variables,
)
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.test_utilities import (
    Utils,
    is_nccl_ep_available,
    is_nccl_ep_fp8_dispatch_available,
)

fp8_available, _ = check_fp8_support()


def test_cuda_graph_runner_stream_pool_is_bounded(monkeypatch):
    created_streams = []

    class FakeStream:
        def __init__(self):
            self.cuda_stream = len(created_streams) + 1
            created_streams.append(self)

    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    monkeypatch.setattr(cuda_graphs_module, "_CUDA_GRAPH_STREAM_POOLS", None)
    monkeypatch.setattr(cuda_graphs_module, "_CUDA_GRAPH_STREAM_NEXT_SLOT", 0)

    pool_size = cuda_graphs_module._CUDA_GRAPH_STREAM_POOL_SIZE
    assigned = [cuda_graphs_module._get_cuda_graph_stream() for _ in range(2 * pool_size)]

    assert len(created_streams) == pool_size
    assert len({stream.cuda_stream for stream in created_streams}) == pool_size
    assert assigned[:pool_size] == assigned[pool_size:]


@pytest.mark.parametrize("has_te_modules", [False, True])
@pytest.mark.parametrize("failure_phase", ["before_capture", "capture", "warmup", None])
def test_failed_capture_restores_global_state(monkeypatch, has_te_modules, failure_phase):
    """Propagate capture errors while restoring only the state started by this call."""
    monkeypatch.setattr(_CudagraphGlobalRecord, "cudagraph_created", False)
    monkeypatch.setattr(_CudagraphGlobalRecord, "cudagraph_record", [object()])
    monkeypatch.setattr(cuda_graphs_module, "_IS_GRAPH_CAPTURING", False)
    monkeypatch.setattr(cuda_graphs_module, "_IS_GRAPH_WARMUP", False)
    monkeypatch.setattr(cuda_graphs_module, "HAVE_TE_GRAPHS", True)
    monkeypatch.setattr(_CudagraphGlobalRecord, "_te_capture_started", False)
    te_events = []
    monkeypatch.setattr(
        cuda_graphs_module, "te_set_capture_start", lambda: te_events.append("start")
    )
    monkeypatch.setattr(cuda_graphs_module, "te_set_capture_end", lambda: te_events.append("end"))
    # Configuration alone must not cause the outer cleanup to unfreeze GC.
    monkeypatch.setattr(cuda_graphs_module, "FREEZE_GC", True)
    gc_events = []
    monkeypatch.setattr(gc, "unfreeze", lambda: gc_events.append("unfreeze"))
    original_save = torch.autograd.function.FunctionCtx.save_for_backward
    capture_error = RuntimeError("injected capture failure")

    def capture(cls):
        cls._enable_saved_tensors_observer()
        if failure_phase == "before_capture":
            raise capture_error
        cuda_graphs_module._set_capture_start()
        if has_te_modules:
            cuda_graphs_module.te_set_capture_start()
            cls._te_capture_started = True
        if failure_phase == "warmup":
            cuda_graphs_module._set_warmup_start()
        if failure_phase is not None:
            raise capture_error
        cuda_graphs_module._set_capture_end()
        if has_te_modules:
            cuda_graphs_module.te_set_capture_end()
            cls._te_capture_started = False
        return "captured"

    monkeypatch.setattr(_CudagraphGlobalRecord, "_create_cudagraphs", classmethod(capture))
    if failure_phase is None:
        assert create_cudagraphs() == "captured"
    else:
        with pytest.raises(RuntimeError, match="injected capture failure") as exc_info:
            create_cudagraphs()
        assert exc_info.value is capture_error

    assert not cuda_graphs_module.is_graph_capturing()
    assert not cuda_graphs_module.is_graph_warmup()
    expected_te_events = (
        ["start", "end"] if has_te_modules and failure_phase != "before_capture" else []
    )
    assert te_events == expected_te_events
    assert not _CudagraphGlobalRecord._te_capture_started
    assert gc_events == []
    assert _CudagraphGlobalRecord._saved_tensors_observer is None
    assert torch.autograd.function.FunctionCtx.save_for_backward is original_save


class TestCudaGraphSavedTensorObserver:
    @pytest.fixture
    def record(self):
        # Exercise the real observer without sharing the process-global graph records.
        class Record(_CudagraphGlobalRecord):
            cudagraph_created = False
            cudagraph_record = []
            _saved_tensors_observer = None

        original_save = torch.autograd.function.FunctionCtx.save_for_backward
        try:
            yield Record
        finally:
            Record._disable_saved_tensors_observer()
            assert torch.autograd.function.FunctionCtx.save_for_backward is original_save

    @pytest.mark.parametrize("save_view", [False, True])
    def test_empty_create_preserves_observer_for_later_training(self, record, save_view):
        record._enable_saved_tensors_observer()
        observing_save = torch.autograd.function.FunctionCtx.save_for_backward

        # GRPO computes old/reference logprobs with forward-only schedules before
        # its first training forward records any CUDA graphs.
        for _ in range(2):
            assert record.create_cudagraphs() is None
            assert record._saved_tensors_observer is not None
            assert torch.autograd.function.FunctionCtx.save_for_backward is observing_save

        class SaveInput(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                # TE 2.20 saves an input view; TE 2.21 saves the input directly.
                ctx.save_for_backward(value.view(-1, 3) if save_view else value)
                return value * 2

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 2

        value = torch.ones(2, 3, requires_grad=True)
        output = SaveInput.apply(value)
        assert value.cg_buffer_metadata.is_saved_for_backward
        assert (output.grad_fn.saved_tensors[0] is value) is not save_view

    @pytest.mark.parametrize("capture_fails", [False, True])
    def test_capture_restores_observer_on_success_or_failure(
        self, record, monkeypatch, capture_fails
    ):
        original_save = torch.autograd.function.FunctionCtx.save_for_backward
        record._enable_saved_tensors_observer()
        observing_save = torch.autograd.function.FunctionCtx.save_for_backward
        record.cudagraph_record = [object()]
        capture_stats = {"time": 1.0}

        def capture(cls):
            assert torch.autograd.function.FunctionCtx.save_for_backward is observing_save
            value = torch.ones(2, 3)
            torch.autograd.function.FunctionCtx().save_for_backward(value)
            assert value.cg_buffer_metadata.is_saved_for_backward
            if capture_fails:
                raise RuntimeError("capture failed")
            cls.cudagraph_created = True
            cls.cudagraph_record = []
            return capture_stats

        monkeypatch.setattr(record, "_create_cudagraphs", classmethod(capture))
        if capture_fails:
            with pytest.raises(RuntimeError, match="capture failed"):
                record.create_cudagraphs()
        else:
            assert record.create_cudagraphs() is capture_stats

        assert record._saved_tensors_observer is None
        assert torch.autograd.function.FunctionCtx.save_for_backward is original_save


def _base_cuda_graph_config(**kwargs) -> TransformerConfig:
    return TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4, **kwargs)


def _te_whole_moe_paged_stash_config(**overrides) -> TransformerConfig:
    kwargs = {
        "cuda_graph_impl": "transformer_engine",
        "cuda_graph_modules": [CudaGraphModule.moe],
        "num_moe_experts": 4,
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "hybridep",
        "moe_expert_rank_capacity_factor": 1.2,
        "moe_paged_stash": True,
        "use_transformer_engine_op_fuser": True,
    }
    kwargs.update(overrides)
    return _base_cuda_graph_config(**kwargs)


def _validated_cuda_graph_cli_args(monkeypatch, cli_args=None, **overrides):
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    warning_messages = []
    print_messages = []

    monkeypatch.setattr(
        training_arguments, "warn_rank_0", lambda msg, *args, **kwargs: warning_messages.append(msg)
    )
    monkeypatch.setattr(
        training_arguments, "print_rank_0", lambda msg, *args, **kwargs: print_messages.append(msg)
    )
    monkeypatch.setattr(sys, "argv", ["test_cuda_graphs.py", *(cli_args or [])])

    args = parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1

    for key, value in overrides.items():
        setattr(args, key, value)

    args = validate_args(args)
    return args, warning_messages, print_messages


class TestCudaGraphConfigAndArguments:
    def test_initialize_dsa_tracker_finalizes_pp_agreed_capacity(self, monkeypatch):
        class MetricModule(torch.nn.Module):
            logs_dsa_indexer_loss = True

            def __init__(self):
                super().__init__()
                self.layer_number = 5
                self.config = SimpleNamespace(
                    dsa_indexer_loss_coeff=0.1, num_layers=2, mtp_num_layers=1
                )

        model = torch.nn.Module()
        model.add_module("metric", MetricModule())
        pp_group = object()
        model_pg_collection = SimpleNamespace(pp=pp_group)
        schedule_pg_collection = MultiModuleProcessGroupCollection(
            module_pgs={"language": model_pg_collection}, language_model_module_name="language"
        )
        calls = []
        tracker = {}
        monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

        def pp_max(tensor, op=None, group=None):
            calls.append((op, group))
            tensor.fill_(7)

        monkeypatch.setattr(torch.distributed, "all_reduce", pp_max)

        assert initialize_dsa_metric_tracker([model], schedule_pg_collection) == 7
        assert calls == [(torch.distributed.ReduceOp.MAX, pp_group)]
        assert tracker["values"].shape == (7,)
        assert tracker["agreed_size"] == 7

        fixed_storage = tracker["values"]
        assert initialize_dsa_metric_tracker([model], schedule_pg_collection) == 7
        assert tracker["values"] is fixed_storage
        assert len(calls) == 1

        with pytest.raises(RuntimeError, match="different PP group"):
            initialize_dsa_metric_tracker([model], SimpleNamespace(pp=object()))

        with pytest.raises(RuntimeError):
            DSAIndexerLossLoggingHelper.ensure_tracker_size(8)
        assert tracker["values"] is fixed_storage
        assert tracker["values"].shape == (7,)

    def test_initialize_dsa_tracker_skips_multi_module_rank_without_language_model(
        self, monkeypatch
    ):
        """An encoder-only MIMO rank neither scans nor joins the language PP collective."""
        values = torch.ones(2)
        tracker = {"values": values, "agreed_size": 2, "agreed_size_pp_group": object()}
        pg_collection = MultiModuleProcessGroupCollection(
            module_pgs={"encoder": SimpleNamespace(pp=object())}, language_model_module_name=None
        )

        class UnexpectedModel:
            def modules(self):
                pytest.fail("encoder-only rank scanned a model for DSA writers")

        original_tracker = tracker.copy()
        monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(
            torch.distributed,
            "all_reduce",
            lambda *args, **kwargs: pytest.fail("encoder-only rank joined DSA initialization"),
        )

        assert initialize_dsa_metric_tracker(UnexpectedModel(), pg_collection) == 0
        assert tracker.keys() == original_tracker.keys()
        assert all(tracker[key] is value for key, value in original_tracker.items())
        assert tracker["values"] is values

    def test_initialize_dsa_tracker_caches_zero_capacity(self, monkeypatch):
        """A writer-free language PP group finalizes zero without allocating storage."""
        pp_group = object()
        tracker = {}
        calls = []
        monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(
            torch.distributed,
            "all_reduce",
            lambda tensor, op=None, group=None: calls.append((op, group)),
        )

        pg_collection = SimpleNamespace(pp=pp_group)
        assert initialize_dsa_metric_tracker(torch.nn.Module(), pg_collection) == 0
        assert calls == [(torch.distributed.ReduceOp.MAX, pp_group)]
        assert tracker["agreed_size"] == 0
        assert "values" not in tracker

        assert initialize_dsa_metric_tracker(torch.nn.Module(), pg_collection) == 0
        assert calls == [(torch.distributed.ReduceOp.MAX, pp_group)]
        assert "values" not in tracker

        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker(num_layers=17, pp_group=pp_group)
        assert calls == [(torch.distributed.ReduceOp.MAX, pp_group)]
        assert "values" not in tracker

    def test_initialize_dsa_tracker_rejects_existing_zero_length_storage(self, monkeypatch):
        """Even empty lazy storage means a metric write preceded lifecycle initialization."""
        pp_group = object()
        tracker = {"values": torch.zeros(0)}
        monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

        with pytest.raises(RuntimeError, match="before its first metric write"):
            initialize_dsa_metric_tracker(torch.nn.Module(), SimpleNamespace(pp=pp_group))

        assert tracker["values"].shape == (0,)
        assert "agreed_size" not in tracker

    def test_initialize_dsa_tracker_empty_pp_stage_uses_peer_writer_size(self, monkeypatch):
        """A language PP stage without a local writer allocates the peer-agreed capacity."""
        pp_group = object()
        tracker = {}
        calls = []
        monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

        def pp_max(tensor, op=None, group=None):
            calls.append((op, group))
            tensor.fill_(4)

        monkeypatch.setattr(torch.distributed, "all_reduce", pp_max)

        assert initialize_dsa_metric_tracker(torch.nn.Module(), SimpleNamespace(pp=pp_group)) == 4
        assert calls == [(torch.distributed.ReduceOp.MAX, pp_group)]
        assert tracker["values"].shape == (4,)
        assert tracker["agreed_size"] == 4

    def test_dsa_tracker_size_ignores_disabled_metric_writers(self):
        """A DSA/CSA module with loss logging disabled cannot write tracker storage."""

        class DisabledMetricModule(torch.nn.Module):
            logs_dsa_indexer_loss = True

            def __init__(self):
                super().__init__()
                self.layer_number = 3
                self.config = SimpleNamespace(
                    dsa_indexer_loss_coeff=0.0, num_layers=4, mtp_num_layers=1
                )

        assert get_dsa_metric_tracker_size(DisabledMetricModule()) == 0

    @pytest.mark.parametrize(
        ("cuda_graph_modules", "expected"),
        [([], True), ([CudaGraphModule.attn], True), ([CudaGraphModule.mlp], False)],
    )
    def test_te_dsa_tracker_capture_follows_attention_scope(
        self, monkeypatch, cuda_graph_modules, expected
    ):
        """TE requires an initialized tracker only when its graph records DSA metric writes."""

        class MetricModule(torch.nn.Module):
            logs_dsa_indexer_loss = True

            def __init__(self):
                super().__init__()
                self.layer_number = 1
                self.config = SimpleNamespace(
                    dsa_indexer_loss_coeff=0.1, num_layers=1, mtp_num_layers=None
                )

        metric_module = MetricModule()

        def discover_layers(helper):
            helper.flattened_callables = [metric_module]

        from megatron.core.pipeline_parallel import p2p_communication

        monkeypatch.setattr(cuda_graphs_module, "HAVE_TE_GRAPHS", True)
        monkeypatch.setattr(TECudaGraphHelper, "_discover_layers", discover_layers)
        monkeypatch.setattr(p2p_communication, "P2PCommunicator", lambda **_kwargs: object())

        helper = TECudaGraphHelper(
            model=[torch.nn.Module()],
            config=SimpleNamespace(
                cuda_graph_impl="transformer_engine", cuda_graph_modules=cuda_graph_modules
            ),
            seq_length=1,
            micro_batch_size=1,
            pg_collection=SimpleNamespace(tp=object(), dp=object(), dp_cp=object(), pp=object()),
        )

        assert helper._captures_dsa_metric_writes is expected

    def test_te_reset_after_capture_cleans_dsa_tracker_in_place(self, monkeypatch):
        from importlib import import_module

        from megatron.core.transformer.moe import moe_logging

        finalize_model_grads_module = import_module(
            "megatron.core.distributed.finalize_model_grads"
        )

        reduce_group = object()
        avg_group = object()
        values = torch.tensor([7.0, 11.0])
        dsa_tracker = {
            'values': values,
            'reduce_group': reduce_group,
            'avg_group': avg_group,
            'agreed_size': 2,
        }
        calls = []

        class FakeModelChunk:
            @staticmethod
            def zero_grad_buffer():
                calls.append('zero_grad_buffer')

        class FakeOptimizer:
            @staticmethod
            def zero_grad():
                calls.append('zero_grad')

        helper = object.__new__(TECudaGraphHelper)
        helper.model = [FakeModelChunk()]
        helper.optimizers = [FakeOptimizer()]
        helper.config = SimpleNamespace()

        monkeypatch.setattr(DSAIndexerLossLoggingHelper, 'tracker', dsa_tracker)
        monkeypatch.setattr(
            moe_logging,
            'get_moe_metrics_tracker',
            lambda: SimpleNamespace(clear=lambda: calls.append('clear_moe_tracker')),
        )
        monkeypatch.setattr(
            finalize_model_grads_module,
            'reset_model_temporary_tensors',
            lambda config, model: calls.append(('reset_model_temporary_tensors', config, model)),
        )

        helper._reset_after_capture()

        assert calls[:3] == ['zero_grad_buffer', 'zero_grad', 'clear_moe_tracker']
        assert calls[3] == ('reset_model_temporary_tensors', helper.config, helper.model)
        assert dsa_tracker['values'] is values
        assert torch.equal(dsa_tracker['values'], torch.zeros(2))
        assert dsa_tracker['reduce_group'] is reduce_group
        assert dsa_tracker['avg_group'] is avg_group
        assert dsa_tracker['agreed_size'] == 2

    @pytest.mark.parametrize("cuda_graph_impl", ["transformer_engine", "chunk_wise"])
    def test_te_reset_after_capture_cleans_what_a_captured_block_accumulates(
        self, monkeypatch, cuda_graph_impl
    ):
        """TE's warm-up passes run the callables on the constant sample inputs. A chunk-wise
        block includes the post-process, which adds to the MTP loss tracker, and the MoE
        dispatch, where one route for every token can overflow a rank capacity and latch the
        dispatcher's over-budget flag. Both are cleared after the capture: the tracker tensors in
        place (the graphs write to them), the flag on every dispatcher of every model chunk.
        Per-layer graphs capture neither and leave both alone."""
        from importlib import import_module

        from megatron.core.transformer.moe import moe_logging
        from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

        finalize_model_grads_module = import_module(
            "megatron.core.distributed.finalize_model_grads"
        )

        class FakeDispatcher:
            resets = 0

            def reset_over_budget(self):
                self.resets += 1

        class FakeModelChunk:
            def __init__(self, token_dispatchers):
                self._modules = [
                    SimpleNamespace(),
                    *(SimpleNamespace(token_dispatcher=d) for d in token_dispatchers),
                    # Dispatchers without an over-budget flag (all-gather, plain all-to-all).
                    SimpleNamespace(token_dispatcher=object()),
                ]

            @staticmethod
            def zero_grad_buffer():
                pass

            def modules(self):
                return iter(self._modules)

        dispatchers = [FakeDispatcher() for _ in range(3)]
        loss_sums = torch.tensor([3.0, 5.0])
        num_tokens = torch.tensor([7.0, 11.0])
        acceptance_counts = torch.ones(2, 2)
        reduce_group = object()
        mtp_tracker = {
            "loss_sums": loss_sums,
            "num_tokens": num_tokens,
            "acceptance_counts": acceptance_counts,
            "reduce_group": reduce_group,
        }
        monkeypatch.setattr(MTPLossLoggingHelper, "tracker", mtp_tracker)
        monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", {})
        monkeypatch.setattr(
            moe_logging, "get_moe_metrics_tracker", lambda: SimpleNamespace(clear=lambda: None)
        )
        monkeypatch.setattr(
            finalize_model_grads_module, "reset_model_temporary_tensors", lambda config, model: None
        )

        helper = object.__new__(TECudaGraphHelper)
        helper.model = [FakeModelChunk(dispatchers[:2]), FakeModelChunk(dispatchers[2:])]
        helper.optimizers = []
        helper.config = SimpleNamespace(cuda_graph_impl=cuda_graph_impl)

        helper._reset_after_capture()

        is_chunk = cuda_graph_impl == "chunk_wise"
        assert mtp_tracker["loss_sums"] is loss_sums
        assert mtp_tracker["num_tokens"] is num_tokens
        assert mtp_tracker["acceptance_counts"] is acceptance_counts
        assert mtp_tracker["reduce_group"] is reduce_group
        if is_chunk:
            assert torch.count_nonzero(loss_sums) == 0
            assert torch.count_nonzero(num_tokens) == 0
            assert torch.count_nonzero(acceptance_counts) == 0
        else:
            assert torch.equal(loss_sums, torch.tensor([3.0, 5.0]))
            assert torch.equal(num_tokens, torch.tensor([7.0, 11.0]))
            assert torch.equal(acceptance_counts, torch.ones(2, 2))
        assert [dispatcher.resets for dispatcher in dispatchers] == [int(is_chunk)] * 3

    def test_te_capture_requires_initialized_dsa_tracker(self, monkeypatch):
        """A custom TE caller cannot lazily allocate tracker storage during capture."""
        helper = object.__new__(TECudaGraphHelper)
        helper.config = SimpleNamespace()
        helper._captures_dsa_metric_writes = True
        helper._start_capturing = lambda: pytest.fail("capture started before validation")

        monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", {})
        monkeypatch.setattr(
            cuda_graphs_module, "validate_moe_cuda_graph_support", lambda _cfg: None
        )

        with pytest.raises(RuntimeError, match="initialized before CUDA Graph capture"):
            helper.create_cudagraphs()

    def test_vision_te_reset_does_not_clear_language_dsa_tracker(self, monkeypatch):
        """Vision capture leaves cleanup to the language-model TE helper."""
        reduce_group = object()
        values = torch.tensor([7.0, 11.0])
        tracker = {"values": values, "reduce_group": reduce_group, "agreed_size": 2}
        monkeypatch.setattr(DSAIndexerLossLoggingHelper, "tracker", tracker)

        helper = object.__new__(VisionTECudaGraphHelper)
        helper._reset_after_capture()

        assert tracker["values"] is values
        assert torch.equal(tracker["values"], torch.tensor([7.0, 11.0]))
        assert tracker["reduce_group"] is reduce_group
        assert tracker["agreed_size"] == 2

    def test_local_impl_defaults_to_layer_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_impl='local')
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.layer

    def test_local_impl_allows_expert_activation_offload_scope(self):
        cfg = _base_cuda_graph_config(
            cuda_graph_impl='local',
            cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe_router],
            fine_grained_activation_offloading=True,
            offload_modules=['expert_fc1', 'moe_act'],
            num_moe_experts=4,
        )

        assert cfg.cuda_graph_impl == 'local'
        assert CudaGraphModule.attn in cfg.cuda_graph_modules
        assert CudaGraphModule.moe_router in cfg.cuda_graph_modules
        assert CudaGraphModule.moe_preprocess in cfg.cuda_graph_modules

    def test_local_impl_rejects_unsupported_activation_offload_scope(self):
        with pytest.raises(
            AssertionError,
            match=(
                "fine-grained activation offloading with cuda_graph_impl='local'.*"
                "Unsupported offload_modules: \\['qkv_linear'\\]"
            ),
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules=[CudaGraphModule.attn],
                fine_grained_activation_offloading=True,
                offload_modules=['qkv_linear'],
            )

    def test_local_impl_rejects_full_layer_graph_with_activation_offload(self):
        with pytest.raises(
            AssertionError, match="not supported with whole-layer CUDA graph capture"
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules=[],
                fine_grained_activation_offloading=True,
                offload_modules=['expert_fc1'],
            )

    def test_te_whole_layer_graph_with_activation_offload_warns(self):
        with pytest.warns(UserWarning, match="does not currently install.*offload event boundary"):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl='transformer_engine',
                cuda_graph_modules=[],
                fine_grained_activation_offloading=True,
                offload_modules=['core_attn'],
            )

        assert cfg.cuda_graph_modules == []

    @pytest.mark.parametrize(
        ('expert_module', 'use_transformer_engine_op_fuser'),
        [('expert_fc1', False), ('moe_act', False), ('fused_group_mlp', True)],
    )
    def test_hybrid_mhc_delayed_expert_offload_warns(
        self, expert_module, use_transformer_engine_op_fuser
    ):
        with pytest.warns(
            UserWarning, match="HybridModel mHC wrappers do not currently support"
        ) as warning_records:
            cfg = _base_cuda_graph_config(
                cuda_graph_impl='transformer_engine',
                cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.moe_router],
                fine_grained_activation_offloading=True,
                offload_modules=[expert_module],
                delay_offload_until_cuda_graph=True,
                is_hybrid_model=True,
                enable_hyper_connections=True,
                num_residual_streams=2,
                num_moe_experts=4,
                use_transformer_engine_op_fuser=use_transformer_engine_op_fuser,
            )

        assert cfg.delay_offload_until_cuda_graph
        assert any(expert_module in str(record.message) for record in warning_records)

    def test_delayed_expert_offload_logs_about_final_queue_drain(self, monkeypatch):
        log_records = []
        monkeypatch.setattr(
            transformer_config_module,
            'log_single_rank',
            lambda logger, level, message: log_records.append((logger, level, message)),
        )

        cfg = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine',
            cuda_graph_modules=[CudaGraphModule.moe_router],
            fine_grained_activation_offloading=True,
            offload_modules=['expert_fc1'],
            delay_offload_until_cuda_graph=True,
            num_moe_experts=4,
        )

        assert cfg.delay_offload_until_cuda_graph
        assert len(log_records) == 1
        assert log_records[0][0] is transformer_config_module.logger
        assert log_records[0][1] == logging.WARNING
        assert "final eager expert group can remain queued" in log_records[0][2]

    def test_local_impl_rejects_moe_router_graph_with_mlp_norm_offload(self):
        with pytest.raises(
            AssertionError,
            match=(
                "fine-grained activation offloading with cuda_graph_impl='local'.*"
                "Unsupported offload_modules: \\['mlp_norm'\\]"
            ),
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules=[CudaGraphModule.moe_router],
                fine_grained_activation_offloading=True,
                offload_modules=['mlp_norm'],
                num_moe_experts=4,
            )

    def test_local_explicit_moe_graph_rejects_dropless_moe(self):
        with pytest.raises(
            AssertionError, match="moe cuda graph is only supported with drop-padding MoE"
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local', cuda_graph_modules=[CudaGraphModule.moe], num_moe_experts=4
            )

    def test_local_inference_full_layer_graph_allows_dropless_moe(self):
        cfg = _base_cuda_graph_config(
            cuda_graph_impl='local',
            cuda_graph_modules=[],
            inference_cuda_graph_scope=InferenceCudaGraphScope.block,
            num_moe_experts=4,
        )

        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.block

    @pytest.mark.parametrize(
        "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
    )
    def test_te_whole_moe_graph_allows_sync_free_hybridep_paged_stash(
        self, monkeypatch, cuda_graph_modules
    ):
        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
        cfg = _te_whole_moe_paged_stash_config(
            cuda_graph_modules=cuda_graph_modules, cuda_graph_warmup_steps=2
        )
        validate_moe_cuda_graph_support(cfg)

        assert cfg.cuda_graph_modules == cuda_graph_modules

    @pytest.mark.parametrize(
        "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
    )
    def test_te_whole_moe_paged_stash_rejects_dynamic_microbatches(
        self, monkeypatch, cuda_graph_modules
    ):
        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
        with pytest.raises(AssertionError, match="require a fixed runtime microbatch schedule"):
            _te_whole_moe_paged_stash_config(
                cuda_graph_modules=cuda_graph_modules, cuda_graph_dynamic_microbatches=True
            )

    @pytest.mark.parametrize(
        "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
    )
    @pytest.mark.parametrize("warmup_steps", [0, 1])
    def test_te_whole_moe_paged_stash_requires_two_warmup_steps(
        self, monkeypatch, cuda_graph_modules, warmup_steps
    ):
        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
        with pytest.raises(AssertionError, match="require at least 2 cuda_graph_warmup_steps"):
            _te_whole_moe_paged_stash_config(
                cuda_graph_modules=cuda_graph_modules, cuda_graph_warmup_steps=warmup_steps
            )

    def test_full_iteration_paged_stash_requires_two_warmup_steps(self):
        kwargs = dict(
            cuda_graph_impl="full_iteration",
            cuda_graph_modules=[],
            moe_expert_capacity_factor=1.0,
            moe_pad_expert_input_to_capacity=True,
        )
        for warmup_steps in (0, 1):
            with pytest.raises(AssertionError, match="at least two eager warmup steps"):
                _te_whole_moe_paged_stash_config(cuda_graph_warmup_steps=warmup_steps, **kwargs)

        config = _te_whole_moe_paged_stash_config(cuda_graph_warmup_steps=2, **kwargs)
        assert config.cuda_graph_warmup_steps == 2

    @pytest.mark.parametrize(
        "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
    )
    def test_te_whole_moe_paged_stash_requires_minimum_te_version(
        self, monkeypatch, cuda_graph_modules
    ):
        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: False)

        with pytest.raises(ValueError, match=r"Transformer Engine >= 2\.19\.0"):
            _te_whole_moe_paged_stash_config(
                cuda_graph_modules=cuda_graph_modules, cuda_graph_warmup_steps=2
            )

    def test_chunk_wise_paged_stash_allows_dynamic_microbatches(self, monkeypatch):
        """Chunk graphs key the stash by the runtime (vp, layer, microbatch) coordinates, so
        the microbatch count may vary; the other whole-MoE paged-stash requirements hold."""
        chunk_wise = dict(cuda_graph_impl="chunk_wise", cuda_graph_modules=[])
        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
        # Packed sequences, whose pack count varies from step to step.
        packed = dict(
            sequence_packing_scheduler="dp_balanced",
            cuda_graph_dynamic_microbatches=True,
            pad_packed_seq_alignment="max",
            max_seqlen_per_dp_cp_rank=128,
            thd_max_packed_sequences=8,
        )
        cfg = _te_whole_moe_paged_stash_config(**chunk_wise, **packed, cuda_graph_warmup_steps=2)
        assert cfg.cuda_graph_dynamic_microbatches
        with pytest.raises(AssertionError, match="require at least 2 cuda_graph_warmup_steps"):
            _te_whole_moe_paged_stash_config(**chunk_wise, cuda_graph_warmup_steps=1)
        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: False)
        with pytest.raises(ValueError, match=r"Transformer Engine >= 2\.19\.0"):
            _te_whole_moe_paged_stash_config(**chunk_wise, cuda_graph_warmup_steps=2)

    def test_te_moe_router_paged_stash_still_allows_dynamic_microbatches(self):
        cfg = _te_whole_moe_paged_stash_config(
            cuda_graph_modules=[CudaGraphModule.moe_router], cuda_graph_dynamic_microbatches=True
        )

        assert cfg.cuda_graph_dynamic_microbatches

    @pytest.mark.parametrize(
        "cuda_graph_modules", [[CudaGraphModule.moe], []], ids=["explicit-moe", "full-layer"]
    )
    def test_te_whole_moe_graph_rejects_sync_free_hybridep_without_paged_stash(
        self, cuda_graph_modules
    ):
        with pytest.raises(
            AssertionError, match="sync-free HybridEP with rank capacity and paged stash"
        ):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=cuda_graph_modules,
                num_moe_experts=4,
                moe_token_dispatcher_type="flex",
                moe_flex_dispatcher_backend="hybridep",
                moe_expert_rank_capacity_factor=1.2,
                use_transformer_engine_op_fuser=True,
            )
            validate_moe_cuda_graph_support(cfg)

    @pytest.mark.parametrize(
        ("moe", "supported"),
        [
            (dict(moe_megakernel_backend="mok"), True),
            (dict(moe_expert_capacity_factor=1.0, moe_pad_expert_input_to_capacity=True), True),
            (
                dict(
                    moe_token_dispatcher_type="flex",
                    moe_flex_dispatcher_backend="hybridep",
                    moe_expert_rank_capacity_factor=1.2,
                ),
                True,
            ),
            (dict(num_moe_experts=None), True),
            (dict(), False),
            (dict(moe_expert_capacity_factor=1.0), False),
            (dict(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="hybridep"), False),
            (
                dict(
                    moe_token_dispatcher_type="flex",
                    moe_flex_dispatcher_backend="ncclep",
                    moe_expert_rank_capacity_factor=1.2,
                ),
                False,
            ),
        ],
        ids=[
            "megakernel",
            "drop-padding",
            "hybridep-rank-capacity",
            "dense",
            "dropless-alltoall",
            "token-drop-without-padding",
            "hybridep-without-rank-capacity",
            "ncclep-rank-capacity",
        ],
    )
    def test_chunk_wise_moe_graph_needs_static_token_shapes(self, moe, supported):
        """A chunk-wise graph holds the whole MoE: any dispatch with static token shapes can be
        captured (no paged stash or op fuser requirement, unlike a per-layer whole-MoE graph)."""
        config = SimpleNamespace(
            cuda_graph_impl="chunk_wise",
            cuda_graph_modules=[],
            num_moe_experts=4,
            moe_expert_capacity_factor=None,
            moe_pad_expert_input_to_capacity=False,
            moe_megakernel_backend=None,
            moe_token_dispatcher_type="alltoall",
            moe_flex_dispatcher_backend="deepep",
            moe_expert_rank_capacity_factor=None,
            moe_paged_stash=False,
            use_transformer_engine_op_fuser=False,
        )
        for name, value in moe.items():
            setattr(config, name, value)

        if supported:
            validate_moe_cuda_graph_support(config)
        else:
            with pytest.raises(
                AssertionError, match="chunk-wise CUDA graphs with MoE need static token shapes"
            ):
                validate_moe_cuda_graph_support(config)

    @pytest.mark.parametrize(
        "cuda_graph_impl, megakernel_backend, supported",
        [
            ("chunk_wise", None, False),
            ("chunk_wise", "mok", True),
            ("transformer_engine", None, True),
        ],
        ids=["chunk-wise", "chunk-wise-megakernel", "per-layer"],
    )
    def test_chunk_wise_drop_padding_moe_needs_a_dispatcher_without_host_reads(
        self, cuda_graph_impl, megakernel_backend, supported
    ):
        """Drop-padding gives every expert a static token count, but the allgather dispatcher
        still copies the counts to the host in every forward: a chunk-wise capture of the whole
        MoE stops there. alltoall and flex have a static path."""
        config = SimpleNamespace(
            cuda_graph_impl=cuda_graph_impl,
            cuda_graph_modules=[],
            num_moe_experts=4,
            moe_expert_capacity_factor=1.0,
            moe_pad_expert_input_to_capacity=True,
            moe_megakernel_backend=megakernel_backend,
            moe_token_dispatcher_type="allgather",
        )
        if supported:
            validate_moe_cuda_graph_support(config)
        else:
            with pytest.raises(AssertionError, match="need moe_token_dispatcher_type 'alltoall'"):
                validate_moe_cuda_graph_support(config)
        for dispatcher in ("alltoall", "flex"):
            config.moe_token_dispatcher_type = dispatcher
            validate_moe_cuda_graph_support(config)

    def test_full_iteration_impl_requires_empty_scope(self):
        with pytest.raises(
            AssertionError,
            match='cuda_graph_modules must be empty when cuda_graph_impl="full_iteration"',
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='full_iteration', cuda_graph_modules=[CudaGraphModule.attn]
            )

    def test_full_iteration_scope_string_in_config_migrated(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl='local', cuda_graph_modules='full_iteration'
            )
        assert cfg.cuda_graph_impl == 'full_iteration'
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_full_iteration_inference_scope_string_in_config_migrated(self):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl='local', cuda_graph_modules='full_iteration_inference'
            )
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_full_iteration_inference_scope_string_noops_without_local_impl(self):
        with pytest.warns(DeprecationWarning, match="has no effect"):
            cfg = _base_cuda_graph_config(cuda_graph_modules='full_iteration_inference')
        assert cfg.cuda_graph_impl == 'none'
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_deprecated_full_iteration_scope_rejects_conflicting_new_scope(self):
        with pytest.raises(
            AssertionError,
            match="cuda_graph_modules='full_iteration' cannot be combined with "
            "inference_cuda_graph_scope='block'",
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules='full_iteration',
                inference_cuda_graph_scope='block',
            )

    def test_deprecated_full_iteration_inference_scope_rejects_conflicting_new_scope(self):
        with pytest.raises(
            AssertionError,
            match="cuda_graph_modules='full_iteration_inference' cannot be combined with "
            "inference_cuda_graph_scope='layer'",
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules='full_iteration_inference',
                inference_cuda_graph_scope='layer',
            )

    def test_chunk_wise_cli_accepts_yarn_and_fixed_length_mrope_rotary(self, monkeypatch):
        """The chunk capture materializes yarn and fixed-length mRoPE tables, so the CLI accepts
        them; only mRoPE with the packing scheduler (no [3, batch, seq] position ids) fails
        before the model is built."""
        # There is no --mrope-section CLI flag; mrope_section is set on args like the
        # Qwen-VL trainers do.
        for position_embedding_type in ('rope', 'yarn', 'mrope'):
            args, _, _ = _validated_cuda_graph_cli_args(
                monkeypatch,
                [
                    '--cuda-graph-impl',
                    'chunk_wise',
                    '--position-embedding-type',
                    position_embedding_type,
                ],
                mrope_section=[8, 4, 4],
            )
            assert args.cuda_graph_impl == 'chunk_wise'
        with pytest.raises(ValueError, match="mrope together with packed \\(THD\\) sequences"):
            _validated_cuda_graph_cli_args(
                monkeypatch,
                [
                    '--cuda-graph-impl',
                    'chunk_wise',
                    '--position-embedding-type',
                    'mrope',
                    '--sequence-packing-scheduler',
                    'dp_balanced',
                    '--calculate-per-token-loss',
                    '--pad-packed-seq-alignment',
                    'max',
                    '--max-seqlen-per-dp-cp-rank',
                    '128',
                ],
                mrope_section=[8, 4, 4],
            )

    def test_chunk_wise_cli_requires_one_sample_per_packed_microbatch(self, monkeypatch):
        """The graph helper asserts this after the model is built and the checkpoint is
        loaded; the CLI fails first."""
        packed_chunk_wise = [
            '--cuda-graph-impl',
            'chunk_wise',
            '--cuda-graph-dynamic-microbatches',
            '--sequence-packing-scheduler',
            'dp_balanced',
            '--calculate-per-token-loss',
            '--pad-packed-seq-alignment',
            'max',
            '--max-seqlen-per-dp-cp-rank',
            '128',
            '--thd-max-packed-sequences',
            '4',
        ]
        args, _, _ = _validated_cuda_graph_cli_args(monkeypatch, packed_chunk_wise)
        assert args.cuda_graph_impl == 'chunk_wise'
        with pytest.raises(ValueError, match="requires --micro-batch-size 1"):
            _validated_cuda_graph_cli_args(monkeypatch, packed_chunk_wise, micro_batch_size=2)
        # Fixed-length chunk graphs take any micro batch size.
        args, _, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'chunk_wise'], micro_batch_size=2
        )
        assert args.micro_batch_size == 2

    def test_chunk_wise_cli_requires_one_sequence_per_item_for_pipeline_slots(self, monkeypatch):
        """With pipeline parallelism every in-flight microbatch owns a graph slot, and their
        number is derived from the global batch size: an --sft item, a pack of conversations
        that the scheduler takes apart, could need more slots than the capture made."""
        packed_chunk_wise = [
            '--cuda-graph-impl',
            'chunk_wise',
            '--cuda-graph-dynamic-microbatches',
            '--sequence-packing-scheduler',
            'dp_balanced',
            '--calculate-per-token-loss',
            '--pad-packed-seq-alignment',
            'max',
            '--max-seqlen-per-dp-cp-rank',
            '128',
            '--thd-max-packed-sequences',
            '4',
            '--pipeline-p2p-fixed-shape',
        ]
        two_stages = dict(pipeline_model_parallel_size=2, world_size=2)
        real_sft = dict(sft=True, data_path=['conversations.jsonl'])
        for refused in (real_sft, dict(dataloader_inter_document_masking=True)):
            with pytest.raises(ValueError, match="needs one sequence per dataset item"):
                _validated_cuda_graph_cli_args(
                    monkeypatch, packed_chunk_wise, **refused, **two_stages
                )
        # Mock --sft data (asked for, or built because no data source is declared) has one
        # sequence per item; one pipeline stage has one slot.
        for accepted in (
            dict(sft=True, mock_data=True, **two_stages),
            dict(sft=True, **two_stages),
            real_sft,
        ):
            args, _, _ = _validated_cuda_graph_cli_args(monkeypatch, packed_chunk_wise, **accepted)
            assert args.cuda_graph_impl == 'chunk_wise' and args.sft

    @pytest.mark.parametrize("fsdp", ["use_megatron_fsdp", "use_torch_fsdp2"])
    def test_chunk_wise_cli_rejects_fsdp(self, monkeypatch, fsdp):
        """FSDP gathers the parameters of a layer in its forward hooks. A block replay runs
        none of the hooks inside the block (the capture already fails: Megatron-FSDP sees the
        blocks called outside the model forward), so the combination is refused up front."""
        with pytest.raises(ValueError, match="does not support Megatron-FSDP or Torch FSDP2"):
            _validated_cuda_graph_cli_args(
                monkeypatch, ['--cuda-graph-impl', 'chunk_wise'], **{fsdp: True}
            )

    @pytest.mark.parametrize("data_path", ["sft", "dataloader_inter_document_masking"])
    def test_chunk_wise_cli_rejects_packed_data_paths_without_the_packing_scheduler(
        self, monkeypatch, data_path
    ):
        """Those paths hand the block packed sequence parameters although its graphs are
        captured for fixed-length batches; the replay would raise after the whole capture."""
        with pytest.raises(ValueError, match="only with --sequence-packing-scheduler"):
            _validated_cuda_graph_cli_args(
                monkeypatch, ['--cuda-graph-impl', 'chunk_wise'], **{data_path: True}
            )
        args, _, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'transformer_engine'], **{data_path: True}
        )
        assert args.cuda_graph_impl == 'transformer_engine'

    @pytest.mark.parametrize(
        ("cuda_graph_impl", "expected"), [("chunk_wise", True), ("local", False)]
    )
    def test_te_rng_tracker_follows_te_captured_graphs_in_cli_and_run_config(
        self, monkeypatch, cuda_graph_impl, expected
    ):
        """Transformer Engine captures the chunk-wise graphs whatever transformer_impl builds
        the layers, so both configuration paths turn its RNG tracker on for them."""
        from megatron.training.config.common_config import RNGConfig

        args, _, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', cuda_graph_impl, '--transformer-impl', 'local']
        )
        assert args.te_rng_tracker is expected

        transformer = _base_cuda_graph_config(transformer_impl="local")
        transformer.cuda_graph_impl = cuda_graph_impl
        rng = RNGConfig()
        rng.finalize_model_config(transformer)
        assert rng.te_rng_tracker is expected
        assert transformer.use_te_rng_tracker is expected

    def test_enable_cuda_graph_flag_migrates_to_local_impl(self, monkeypatch):
        args, _, print_messages = _validated_cuda_graph_cli_args(
            monkeypatch, ['--enable-cuda-graph']
        )
        assert args.cuda_graph_impl == 'local'
        assert any("--enable-cuda-graph is deprecated" in msg for msg in print_messages)

    def test_full_iteration_inference_scope_cli_migrates_to_block_scope(self, monkeypatch):
        args, warning_messages, _ = _validated_cuda_graph_cli_args(
            monkeypatch,
            ['--cuda-graph-impl', 'local', '--cuda-graph-modules', 'full_iteration_inference'],
        )
        assert args.cuda_graph_impl == 'local'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert args.cuda_graph_modules == []
        assert any(
            "--cuda-graph-modules 'full_iteration_inference' is deprecated" in msg
            for msg in warning_messages
        )

    def test_full_iteration_inference_scope_cli_noops_without_local_impl(self, monkeypatch):
        args, warning_messages, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-scope', 'full_iteration_inference']
        )
        assert args.cuda_graph_impl == 'none'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert args.cuda_graph_modules == []
        assert any("has no effect when --cuda-graph-impl=none" in msg for msg in warning_messages)

    def test_full_iteration_inference_scope_cli_rejects_conflicting_new_scope(self, monkeypatch):
        with pytest.raises(
            AssertionError,
            match="cuda_graph_modules='full_iteration_inference' cannot be combined with "
            "inference_cuda_graph_scope='layer'",
        ):
            _validated_cuda_graph_cli_args(
                monkeypatch,
                [
                    '--cuda-graph-impl',
                    'local',
                    '--cuda-graph-modules',
                    'full_iteration_inference',
                    '--inference-cuda-graph-scope',
                    'layer',
                ],
            )

    def test_new_scope_cli_accepts_block(self, monkeypatch):
        args, _, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'local', '--inference-cuda-graph-scope', 'block']
        )
        assert args.cuda_graph_impl == 'local'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.block

    def test_new_scope_cli_accepts_layer(self, monkeypatch):
        args, _, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'local', '--inference-cuda-graph-scope', 'layer']
        )
        assert args.cuda_graph_impl == 'local'
        assert args.inference_cuda_graph_scope == InferenceCudaGraphScope.layer

    def test_removed_module_scoped_scope_name_is_not_accepted(self, monkeypatch):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                'test_cuda_graphs.py',
                '--cuda-graph-impl',
                'local',
                '--inference-cuda-graph-scope',
                'module_scoped',
            ],
        )
        with pytest.raises(SystemExit):
            parse_args()

    def test_removed_old_inference_bool_flag_is_not_accepted(self, monkeypatch):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        monkeypatch.setattr(
            sys, "argv", ['test_cuda_graphs.py', '--inference-use-full-iteration-cuda-graph']
        )
        with pytest.raises(SystemExit):
            parse_args()

    # --- Backward compat: cuda_graph_scope → cuda_graph_modules rename ---

    def test_deprecated_cuda_graph_scope_kwarg_migrates_to_modules(self):
        with pytest.warns(DeprecationWarning, match="cuda_graph_scope is deprecated"):
            cfg = _base_cuda_graph_config(cuda_graph_scope=['attn'])
        assert cfg.cuda_graph_modules == [CudaGraphModule.attn]
        assert cfg.cuda_graph_scope is None

    def test_new_cuda_graph_modules_does_not_populate_deprecated_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_modules=['attn', 'mlp'])
        assert cfg.cuda_graph_modules == [CudaGraphModule.attn, CudaGraphModule.mlp]
        assert cfg.cuda_graph_scope is None

    def test_new_full_iteration_impl_does_not_populate_deprecated_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_impl='full_iteration', cuda_graph_modules=[])
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_cli_migrates_to_modules(self, monkeypatch):
        args, warning_messages, _ = _validated_cuda_graph_cli_args(
            monkeypatch, ['--cuda-graph-impl', 'local', '--cuda-graph-scope', 'attn']
        )
        assert args.cuda_graph_modules == [CudaGraphModule.attn]
        assert any('--cuda-graph-scope is deprecated' in msg for msg in warning_messages)

    def test_cuda_graph_scope_is_standalone_class_for_pickle_compat(self):
        from megatron.core.transformer.enums import CudaGraphScope

        # CudaGraphScope is preserved as a standalone class (not an alias) so that
        # pre-refactor checkpoints can be deserialized without value-collision errors.
        assert CudaGraphScope is not CudaGraphModule
        assert CudaGraphScope.attn.value == 2  # original ordinals preserved
        assert CudaGraphScope.mamba.value == 7

    def test_cuda_graph_scope_and_inference_scope_in_safe_globals(self):
        from megatron.core.safe_globals import SAFE_GLOBALS
        from megatron.core.transformer.enums import CudaGraphScope

        assert CudaGraphScope in SAFE_GLOBALS
        assert InferenceCudaGraphScope in SAFE_GLOBALS

    def test_deprecated_cuda_graph_scope_enum_instance_migrates_to_modules(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning, match="cuda_graph_scope is deprecated"):
            cfg = _base_cuda_graph_config(cuda_graph_scope=[CudaGraphScope.attn])
        assert cfg.cuda_graph_modules == [CudaGraphModule.attn]
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_full_iteration_enum_migrates_to_impl(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning):
            cfg = _base_cuda_graph_config(cuda_graph_scope=[CudaGraphScope.full_iteration])
        assert cfg.cuda_graph_impl == "full_iteration"
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_full_iteration_inference_enum_migrates_to_scope(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning):
            cfg = _base_cuda_graph_config(
                cuda_graph_impl="local", cuda_graph_scope=[CudaGraphScope.full_iteration_inference]
            )
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None

    def test_deprecated_cuda_graph_scope_full_iteration_inference_noops_without_local_impl(self):
        from megatron.core.transformer.enums import CudaGraphScope

        with pytest.warns(DeprecationWarning, match="has no effect"):
            cfg = _base_cuda_graph_config(
                cuda_graph_scope=[CudaGraphScope.full_iteration_inference]
            )
        assert cfg.cuda_graph_impl == "none"
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        assert cfg.cuda_graph_modules == []
        assert cfg.cuda_graph_scope is None


class TestCudaGraphReplay:
    def test_gtp_forward_ensures_captured_params_ready_before_replay(self, monkeypatch):
        calls = []
        first = object()
        second = object()
        runner = object.__new__(_CudaGraphRunner)
        runner._gtp_fwd_params_to_ensure_ready = (first, second)
        runner.grad_enabled = False
        runner.fwd_graph_outputs = (object(),)
        runner.get_mismatch_errors = lambda args, kwargs: []
        runner.get_tensors = lambda args, kwargs, check_types: []
        runner.to_list = lambda value: list(value) if isinstance(value, tuple) else [value]
        runner._make_pipeline_output_viewless = lambda output: output
        monkeypatch.setattr(
            cuda_graphs_module,
            "ensure_params_ready",
            lambda params: calls.append(("ready", tuple(params))),
        )
        monkeypatch.setattr(
            _CudagraphReplayNode, "apply", lambda *args: calls.append("replay") or (object(),)
        )

        runner.replay_graph_capture(False, (), {})

        assert calls == [("ready", (first, second)), "replay"]


class TestParallelTransformerBlockCudagraphs:
    def setup_method(self, method):
        # initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)

        # initialize transformer model
        num_layers = 8
        hidden_size = 64
        self.transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
        )
        self.parallel_transformer_block = TransformerBlock(
            self.transformer_config, get_gpt_layer_with_transformer_engine_spec()
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        CudaGraphManager.global_mempool = None

    @pytest.mark.flaky_in_dev  # Issue #5474
    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_gpu_cudagraph(self):
        parallel_transformer_block = self.parallel_transformer_block
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        sequence_length = 32
        micro_batch_size = 2
        transformer_config: TransformerConfig = parallel_transformer_block.config
        num_layers = transformer_config.num_layers
        hidden_size = transformer_config.hidden_size
        hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )

        for _ in range(num_layers):
            assert hasattr(parallel_transformer_block.layers[0], "cudagraph_manager")
            assert (
                len(parallel_transformer_block.layers[0].cudagraph_manager.cudagraph_runners) == 1
            )
            del (
                parallel_transformer_block.layers[_]
                .cudagraph_manager.cudagraph_runners[0]
                .fwd_graph
            )

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_dense_mlp_scope_constructs(self):
        config = TransformerConfig(
            num_layers=8,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
            cuda_graph_modules=[CudaGraphModule.mlp],
        )

        block = TransformerBlock(config, get_gpt_layer_with_transformer_engine_spec())

        assert block.layers
        assert all(not layer.is_moe_layer for layer in block.layers)
        assert all(isinstance(layer.cudagraph_manager, CudaGraphManager) for layer in block.layers)


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
class TestPackedSeqCudagraphs:
    """Training CUDA graphs over thd input with padding between sequences.

    The padded cu_seqlens describe a slot layout that differs from the actual lengths,
    and pad_between_seqs is set explicitly so TE does spend a GPU sync inferring it.
    cp_size == 2 additionally captures TE's ring-P2P context-parallel attention inside the graphs.
    """

    SEQ_LENGTHS = [7, 5]
    SLOT_STARTS = [0, 8, 16]  # slot layout aligned to 2 * cp_size for every cp_size tested
    BIN_SIZE = 32
    NVTE_ENV_VARS = (
        "NVTE_FLASH_ATTN",
        "NVTE_FUSED_ATTN",
        "NVTE_UNFUSED_ATTN",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    )

    def setup_method(self, method):
        self.original_nvte_env = {name: os.environ.get(name) for name in self.NVTE_ENV_VARS}
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

    def teardown_method(self, method):
        try:
            Utils.destroy_model_parallel()
            _CudagraphGlobalRecord.cudagraph_created = False
            _CudagraphGlobalRecord.cudagraph_record = []
            CudaGraphManager.global_mempool = None
        finally:
            for name, value in self.original_nvte_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def _build_packed_seq_params(self, device):
        # Actual boundaries: each sequence's real tokens inside its slot; the trailing bin
        # padding [SLOT_STARTS[-1], BIN_SIZE) forms a ghost slot of pad tokens.
        boundaries = [0]
        for length in self.SEQ_LENGTHS:
            boundaries.append(boundaries[-1] + length)
        boundaries.append(boundaries[-1] + self.BIN_SIZE - self.SLOT_STARTS[-1])
        cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
        cu_seqlens_padded = torch.tensor(
            self.SLOT_STARTS + [self.BIN_SIZE], dtype=torch.int32, device=device
        )
        return PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
            max_seqlen_q=self.BIN_SIZE,
            max_seqlen_kv=self.BIN_SIZE,
            pad_between_seqs=True,
        )

    @pytest.mark.parametrize("cp_size", [1, 2])
    def test_thd_capture_with_pad_between_seqs(self, cp_size, monkeypatch):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(context_parallel_size=cp_size)
        model_parallel_cuda_manual_seed(123)
        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_UNFUSED_ATTN"] = "0"

        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            context_parallel_size=cp_size,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            attention_backend=AttnBackend.fused,
            deterministic_mode=True,
            cuda_graph_impl="local",
            cuda_graph_warmup_steps=1,
            use_cpu_initialization=True,
        )
        block = TransformerBlock(config, get_gpt_layer_with_transformer_engine_spec()).cuda()
        block.train()
        # GRPO's forward-only logprob schedules request graph creation before the
        # first training forward. This must preserve saved-tensor lifetime tracking.
        create_cudagraphs()
        # CUDA-graphed backward assumes DDP-style grad accumulation buffers.
        for param in block.parameters():
            param.main_grad = torch.zeros_like(param)

        packed_seq_params = self._build_packed_seq_params(torch.device('cuda'))
        # Each CP rank holds its 1/cp_size share of the bin's tokens.
        hidden_states = torch.randn(
            (self.BIN_SIZE // cp_size, 1, config.hidden_size),
            dtype=torch.bfloat16,
            device='cuda',
            requires_grad=True,
        )

        eager_out = block(
            hidden_states=hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )
        hidden_states_metadata = hidden_states.cg_buffer_metadata
        assert hidden_states_metadata.is_cudagraph_input
        assert hidden_states_metadata.is_saved_for_backward

        # The second layer's TE input layernorm saves the first layer's output for backward.
        # This naturally exercises a CUDA graph output whose pool buffer must stay alive until
        # backward capture.
        first_runner = block.layers[0].cudagraph_manager.cudagraph_runners[0]
        first_runner_record = next(
            record
            for record in _CudagraphGlobalRecord.cudagraph_record
            if record[0] is first_runner and record[1] == "fwd"
        )
        recorded_outputs = first_runner_record[4]
        output_metadata = first_runner.get_arg_metas(recorded_outputs)[0].cg_buffer_metadata
        output_metadata_state = (
            f"input={output_metadata.is_cudagraph_input}, "
            f"output={output_metadata.is_cudagraph_output}, "
            f"saved={output_metadata.is_saved_for_backward}"
        )
        assert output_metadata.is_cudagraph_input, output_metadata_state
        assert output_metadata.is_cudagraph_output, output_metadata_state
        assert output_metadata.is_saved_for_backward, output_metadata_state

        # The q/kv aliases for each offsets tensor must share one metadata object while recording
        # every graph-input use for replay-buffer sharing.
        actual_cu_seqlens_metadata = packed_seq_params.cu_seqlens_q.cg_buffer_metadata
        padded_cu_seqlens_metadata = packed_seq_params.cu_seqlens_q_padded.cg_buffer_metadata
        assert packed_seq_params.cu_seqlens_kv.cg_buffer_metadata is actual_cu_seqlens_metadata
        assert (
            packed_seq_params.cu_seqlens_kv_padded.cg_buffer_metadata is padded_cu_seqlens_metadata
        )
        assert actual_cu_seqlens_metadata.is_cudagraph_input
        assert padded_cu_seqlens_metadata.is_cudagraph_input
        eager_out.sum().backward()

        # Graph warmup must see the recorded offsets, not zero-filled placeholders:
        # TE's THD context-parallel backward indexes with them and asserts out of bounds on zeros.
        # Resolved by TransfomerEngine#3205.
        warmup_offsets = []
        attention_forward = TEDotProductAttention.forward

        def recording_forward(module, *args, **kwargs):
            if is_graph_warmup():
                bound = inspect.signature(attention_forward).bind(module, *args, **kwargs)
                params = bound.arguments["packed_seq_params"]
                warmup_offsets.append(
                    (params.cu_seqlens_q.clone(), params.cu_seqlens_q_padded.clone())
                )
            return attention_forward(module, *args, **kwargs)

        monkeypatch.setattr(TEDotProductAttention, "forward", recording_forward)

        # Reproduce the stale AccumulateGrad stream failure without inheriting an override
        # enabled by an earlier test. Keep the eager graph alive through backward capture.
        set_override = getattr(torch.autograd.graph, "set_override_stale_capture_stream", None)
        if set_override is None:
            create_cudagraphs()
        else:
            override_calls = []

            def tracked_override(enabled):
                override_calls.append(enabled)
                return set_override(enabled)

            set_override(False)
            try:
                with monkeypatch.context() as capture_patch:
                    capture_patch.setattr(
                        torch.autograd.graph, "set_override_stale_capture_stream", tracked_override
                    )
                    create_cudagraphs()
                assert override_calls == [True]
            finally:
                # Preserve MCore's enabled policy even if this regression test fails.
                set_override(True)

        assert warmup_offsets
        for cu_seqlens, cu_seqlens_padded in warmup_offsets:
            assert torch.equal(cu_seqlens, packed_seq_params.cu_seqlens_q)
            assert torch.equal(cu_seqlens_padded, packed_seq_params.cu_seqlens_q_padded)

        runners = []
        for layer in block.layers:
            layer_runners = layer.cudagraph_manager.cudagraph_runners
            assert len(layer_runners) == 1
            assert layer_runners[0].fwd_graph is not None
            runners.extend(layer_runners)

        # There are four cu_seqlens arguments per layer: q/kv pairs for the real and padded
        # offsets. Each pair and every later layer should alias one of two shared buffers. Within
        # each buffer group, only its first graph-input occurrence performs the replay copy.
        cu_seqlens_buffers = [
            tensor
            for runner in runners
            for tensor in runner.fwd_graph_input_surface[: runner.num_dgrads]
            if tensor.dtype == torch.int32 and tensor.shape == packed_seq_params.cu_seqlens_q.shape
        ]
        assert len(cu_seqlens_buffers) == 4 * len(runners)
        buffers_by_ptr = {}
        for tensor in cu_seqlens_buffers:
            buffers_by_ptr.setdefault(tensor.data_ptr(), []).append(tensor)
        assert len(buffers_by_ptr) == 2
        for shared_buffers in buffers_by_ptr.values():
            assert sum(not tensor.can_skip_replay_copy for tensor in shared_buffers) == 1

        graphed_out = block(
            hidden_states=hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )
        assert torch.equal(graphed_out, eager_out), (
            "CUDA graph replay output is not bitwise equal to eager output: "
            f"max_abs_diff={(graphed_out.float() - eager_out.float()).abs().max().item()}"
        )
        graphed_out.sum().backward()

        # Destroy captured graphs deterministically before parallel-state teardown.
        for layer in block.layers:
            for runner in layer.cudagraph_manager.cudagraph_runners:
                if hasattr(runner, "fwd_graph"):
                    del runner.fwd_graph
                if hasattr(runner, "bwd_graph"):
                    del runner.bwd_graph
        torch.cuda.synchronize()


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
@pytest.mark.parametrize(
    "total_num_layers, pp, vpp, account_for_embedding_in_pipeline_split, account_for_loss_in_pipeline_split, num_layers_in_first_pipeline_stage, num_layers_in_last_pipeline_stage, pp_layout, first_layer_numbers_golden, last_layer_numbers_golden",
    [
        (4, 1, None, False, False, None, None, None, [1], [4]),
        (8, 2, None, False, False, None, None, None, [1, 5], [4, 8]),
        (8, 2, 2, False, False, None, None, None, [1, 3, 5, 7], [2, 4, 6, 8]),
        (14, 4, None, True, True, None, None, None, [1, 4, 8, 12], [3, 7, 11, 14]),
        (
            14,
            4,
            2,
            True,
            True,
            None,
            None,
            None,
            [1, 2, 4, 6, 8, 10, 12, 14],
            [1, 3, 5, 7, 9, 11, 13, 14],
        ),
        (12, 4, None, False, False, 2, 2, None, [1, 3, 7, 11], [2, 6, 10, 12]),
        (
            12,
            4,
            2,
            False,
            False,
            2,
            2,
            None,
            [1, 2, 4, 6, 7, 8, 10, 12],
            [1, 3, 5, 6, 7, 9, 11, 12],
        ),
        (
            14,
            4,
            2,
            False,
            False,
            None,
            None,
            [
                ["embedding", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "decoder"],
                ["decoder", "loss"],
            ],
            [1, 2, 4, 6, 8, 10, 12, 14],
            [1, 3, 5, 7, 9, 11, 13, 14],
        ),
    ],
)
def test_cuda_graph_determine_first_last_layer_logic(
    total_num_layers,
    pp,
    vpp,
    account_for_embedding_in_pipeline_split,
    account_for_loss_in_pipeline_split,
    num_layers_in_first_pipeline_stage,
    num_layers_in_last_pipeline_stage,
    pp_layout,
    first_layer_numbers_golden,
    last_layer_numbers_golden,
):
    # Initialize RNG tracker
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    # Initialize parallel state
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=pp, virtual_pipeline_model_parallel_size=vpp
    )

    # initialize model
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    hidden_size = 128
    transformer_config = TransformerConfig(
        num_layers=total_num_layers,
        hidden_size=hidden_size,
        num_attention_heads=1,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        virtual_pipeline_model_parallel_size=vpp,
        pipeline_model_parallel_size=pp,
        deallocate_pipeline_outputs=True,
        cuda_graph_impl="local",
        use_te_rng_tracker=True,
        account_for_embedding_in_pipeline_split=account_for_embedding_in_pipeline_split,
        account_for_loss_in_pipeline_split=account_for_loss_in_pipeline_split,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
        pipeline_model_parallel_layout=pp_layout,
    )
    model = []
    for i in range(vpp or 1):
        this_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=1024,
            position_embedding_type="rope",
            vp_stage=i,
        ).cuda()
        model.append(this_model)

    # create runner by running a fake forward pass
    sequence_length, micro_batch_size = 32, 1
    hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size)).cuda()
    attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
    for m in model:
        _ = m(
            input_ids=None,
            position_ids=None,
            attention_mask=attention_mask,
            decoder_input=hidden_states,
        )

    # Check if cuda graph is correctly setting is first/last layer
    for m in model:
        for l in m.decoder.layers:
            assert hasattr(l, "cudagraph_manager")
            assert (
                len(l.cudagraph_manager.cudagraph_runners) == 1
            ), "Cuda graph runner should be created"
            runner = l.cudagraph_manager.cudagraph_runners[0]
            assert runner.is_first_layer is not None and runner.is_last_layer is not None
            assert runner.is_first_layer == (l.layer_number in first_layer_numbers_golden)
            assert runner.is_last_layer == (l.layer_number in last_layer_numbers_golden)

            del l.cudagraph_manager.cudagraph_runners[0].fwd_graph

    # Destroy all captured graphs deterministically
    for m in model:
        for l in m.decoder.layers:
            for runner in getattr(l.cudagraph_manager, "cudagraph_runners", []):
                # Safely delete both graphs if present
                if hasattr(runner, "fwd_graph"):
                    del runner.fwd_graph
                if hasattr(runner, "bwd_graph"):
                    del runner.bwd_graph

    # Ensure all pending work is complete and graph destruction runs now
    torch.cuda.synchronize()

    # Teardown
    Utils.destroy_model_parallel()
    _CudagraphGlobalRecord.cudagraph_created = False
    _CudagraphGlobalRecord.cudagraph_record = []
    CudaGraphManager.global_mempool = None
    CudaGraphManager.fwd_mempools = None
    CudaGraphManager.bwd_mempools = None


class TestLLaVACudaGraph:
    """Test CUDA graphs with LLaVA model focusing on is_last_layer logic for encoder/decoder transitions."""

    def setup_method(self, method):
        # Initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
        )
        model_parallel_cuda_manual_seed(123)

        from copy import deepcopy

        from megatron.core.models.multimodal.llava_model import LLaVAModel
        from megatron.core.models.vision.vit_layer_specs import (
            get_vit_layer_with_transformer_engine_spec,
        )

        # Create language transformer config with CUDA graphs enabled
        self.language_hidden_size = 64
        self.language_num_attention_heads = 4
        language_config = TransformerConfig(
            num_layers=2,
            hidden_size=self.language_hidden_size,
            num_attention_heads=self.language_num_attention_heads,
            use_cpu_initialization=True,
            cuda_graph_impl="local",  # Enable CUDA graphs
        )

        # Create vision transformer config
        vision_config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=2,
            use_cpu_initialization=True,
            cuda_graph_impl="local",  # Enable CUDA graphs for vision model too
        )

        # Create vision projection config
        vision_projection_config = TransformerConfig(
            num_layers=1,
            hidden_size=self.language_hidden_size,
            ffn_hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
        )

        # Get layer specs
        language_layer_submodules = get_gpt_layer_with_transformer_engine_submodules()
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
        vision_projection_spec = deepcopy(get_submodules(language_layer_submodules.mlp))
        assert isinstance(vision_projection_spec, MLPSubmodules)

        # Set vision model type
        vision_config.vision_model_type = "clip"
        language_config.language_model_type = "dummy"

        # Create LLaVA model with both encoder and decoder
        self.llava_model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=ModuleSpec(
                module=TransformerLayer, submodules=language_layer_submodules
            ),
            language_vocab_size=8192,
            language_max_sequence_length=4096,
            vision_transformer_config=vision_config,
            vision_transformer_layer_spec=vision_layer_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_projection_config,
            vision_projection_layer_spec=vision_projection_spec,
            img_h=336,
            img_w=336,
            patch_dim=14,
            pre_process=True,
            post_process=True,
            add_encoder=True,
            add_decoder=True,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_llava_cudagraph_is_last_layer_logic(self):
        """Test that is_last_layer logic correctly resets prev_bwd_hidden_state_inputgrad for LLaVA models."""

        # Move model to CUDA
        self.llava_model.cuda()
        # Cudagraph backward capture assumes the model has DDP so create main_grads for params
        for param in self.llava_model.parameters():
            param.main_grad = torch.zeros_like(param)

        set_te_cuda_graph_current_microbatch(self.llava_model.vision_model, 1)
        set_te_cuda_graph_current_microbatch(self.llava_model.language_model, 1)

        # Create test inputs
        batch_size = 2
        seq_length = 1024
        num_images = 1

        images = torch.ones((num_images, 3, 336, 336), dtype=torch.float32).cuda()

        # Create text input with image tokens
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long).cuda()
        # Insert image token (using default image token index)
        input_ids[0, 5] = self.llava_model.image_token_index

        position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1).cuda()
        attention_mask = None

        # Create labels and loss mask for training
        labels = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long).cuda()
        loss_mask = torch.ones((batch_size, seq_length), dtype=torch.float32).cuda()

        # Create num_image_tiles
        num_image_tiles = torch.ones(num_images, dtype=torch.int).cuda()

        # First forward pass - this should record the CUDA graphs
        output1, loss_mask1 = self.llava_model(
            images=images,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            num_image_tiles=num_image_tiles,
        )

        # Verify that CUDA graph managers were created
        if hasattr(self.llava_model.vision_model, 'decoder') and hasattr(
            self.llava_model.vision_model.decoder, 'layers'
        ):
            for layer in self.llava_model.vision_model.decoder.layers:
                if hasattr(layer, 'cudagraph_manager'):
                    assert (
                        layer.cudagraph_manager is not None
                    ), "Vision model layers should have CUDA graph managers"

        if hasattr(self.llava_model.language_model, 'decoder') and hasattr(
            self.llava_model.language_model.decoder, 'layers'
        ):
            for layer in self.llava_model.language_model.decoder.layers:
                if hasattr(layer, 'cudagraph_manager'):
                    assert (
                        layer.cudagraph_manager is not None
                    ), "Language model layers should have CUDA graph managers"

                    # Verify that CUDA graphs were created successfully
                    for runner in layer.cudagraph_manager.cudagraph_runners:
                        assert hasattr(runner, 'fwd_graph')
                        assert hasattr(runner, 'bwd_graph')

        # Perform backward pass to trigger backward graph recording
        if isinstance(output1, tuple):
            loss = output1[0].sum()
        else:
            loss = output1.sum()
        loss.backward()

        # Import the CUDA graph creation function
        from megatron.core.transformer.cuda_graphs import create_cudagraphs

        # Create the CUDA graphs - this is where the is_last_layer logic is tested
        create_cudagraphs()

        # Verify that CUDA graphs were created successfully
        assert _CudagraphGlobalRecord.cudagraph_created, "CUDA graphs should be created"

        if hasattr(self.llava_model.vision_model, 'decoder') and hasattr(
            self.llava_model.vision_model.decoder, 'layers'
        ):
            for layer in self.llava_model.vision_model.decoder.layers:
                del layer.cudagraph_manager.cudagraph_runners[0].fwd_graph
                del layer.cudagraph_manager.cudagraph_runners[0].bwd_graph

        if hasattr(self.llava_model.language_model, 'decoder') and hasattr(
            self.llava_model.language_model.decoder, 'layers'
        ):
            for layer in self.llava_model.language_model.decoder.layers:
                del layer.cudagraph_manager.cudagraph_runners[0].fwd_graph
                del layer.cudagraph_manager.cudagraph_runners[0].bwd_graph


class TestParallelHybridBlockCudagraphs:
    def setup_method(self, method):
        # initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)

        # Ensure that this test is capturing to a fresh memory pool.
        CudaGraphManager.global_mempool = None

        def get_pg_collection():
            return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])

        def get_mamba_block(hybrid_layer_pattern):
            transformer_config = TransformerConfig(
                hidden_size=256,  # The Mamba layer places several constraints on this
                # Need to specify num_attention_heads and num_layers or TransformerConfig
                # will generate errors.
                num_layers=len(hybrid_layer_pattern),
                num_attention_heads=4,
                use_cpu_initialization=True,
                cuda_graph_impl="local",
            )
            layer_config_list = validate_segment_layers(hybrid_layer_pattern, transformer_config)
            modules = hybrid_stack_spec.submodules
            return HybridStack(
                transformer_config,
                modules,
                layer_config_list=layer_config_list,
                pp_layer_offset=0,
                pg_collection=get_pg_collection(),
            )

        self.mamba_block = get_mamba_block(hybrid_layer_pattern="M-M*-")
        self.transformer_config = self.mamba_block.config

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_gpu_cudagraph(self):
        parallel_mamba_block = self.mamba_block
        parallel_mamba_block.cuda()

        # [sequence length, batch size, hidden size]
        sequence_length = 32
        micro_batch_size = 2
        transformer_config: TransformerConfig = parallel_mamba_block.config
        num_layers = transformer_config.num_layers
        hidden_size = transformer_config.hidden_size
        hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size))
        hidden_states = hidden_states.cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states = parallel_mamba_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )

        for _ in range(num_layers):
            assert hasattr(parallel_mamba_block.layers[0], "cudagraph_manager")
            assert len(parallel_mamba_block.layers[0].cudagraph_manager.cudagraph_runners) == 1

            del parallel_mamba_block.layers[_].cudagraph_manager.cudagraph_runners[0].fwd_graph

    def test_mhc_hybrid_layers_are_te_cudagraph_capturable(self):
        """Regression: a mHC-enabled HybridStack must expose graph-capturable layers.

        When ``enable_hyper_connections=True``, ``HybridStack`` wraps every layer in
        ``HyperConnectionHybridLayer``. That wrapper must subclass
        ``TECudaGraphMixin`` and be recognized by ``_te_cuda_graph_layer_is_graphable`` so TE
        cuda-graph discovery finds the wrapped layers. Before the fix the wrapper
        subclassed plain ``MegatronModule``, so discovery rejected every layer (0
        graphable) and CUDA graph capture was silently skipped for the whole hybrid
        model -- making the mHC hybrid run fully eager (several times slower than the
        graphed GPT mHC path). This test fails on the pre-fix code via both assertions.
        """
        # The wrapper must be graph-capturable by construction.
        assert issubclass(HyperConnectionHybridLayer, GraphableMegatronModule)

        layer_type_list = validate_segment_layers("M-M*-")  # mamba / mlp / attention mix
        config = TransformerConfig(
            hidden_size=256,
            num_layers=len(layer_type_list),
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
            enable_hyper_connections=True,
            num_residual_streams=4,
            cuda_graph_modules=[CudaGraphModule.attn, CudaGraphModule.mamba, CudaGraphModule.mlp],
        )
        block = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_type_list=layer_type_list,
            pp_layer_offset=0,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=["tp", "pp", "cp"]
            ),
        )

        # Every layer is wrapped, and the wrappers are discoverable as graphable.
        assert all(isinstance(layer, HyperConnectionHybridLayer) for layer in block.layers)
        graphable = [
            layer for layer in block.layers if _te_cuda_graph_layer_is_graphable(layer, config)
        ]
        assert len(graphable) > 0, (
            "mHC HybridStack produced 0 graphable layers -- TE cuda-graph capture would "
            "be silently skipped for the entire model (the pre-fix bug)."
        )


class TestHybridTECudaGraphDiscovery:
    @staticmethod
    def _bare_hybrid_wrapper(*, offload_in_graph=None):
        wrapper = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
        torch.nn.Module.__init__(wrapper)
        # Intentionally minimal: individual CPU mocks provide only the state they exercise.
        wrapper.config = SimpleNamespace(
            cuda_graph_modules=[CudaGraphModule.attn], fine_grained_activation_offloading=True
        )
        object.__setattr__(wrapper, '_offload_module_in_cuda_graph_cached', None)
        if offload_in_graph is not None:
            object.__setattr__(
                wrapper, '_compute_offload_module_in_cuda_graph', lambda: offload_in_graph
            )
        return wrapper

    @staticmethod
    def _bare_transformer_inner(*, has_attention, offload_core_attn, is_moe=False):
        from megatron.core.transformer.identity_op import IdentityOp

        inner = TransformerLayer.__new__(TransformerLayer)
        torch.nn.Module.__init__(inner)
        inner.self_attention = torch.nn.Linear(2, 2) if has_attention else IdentityOp()
        inner.cross_attention = IdentityOp()
        inner.mlp = IdentityOp()
        inner.offload_attn_norm = False
        inner.offload_qkv_linear = False
        inner.offload_core_attn = offload_core_attn
        inner.offload_attn_proj = False
        inner.offload_mlp_norm = False
        inner.is_moe_layer = is_moe
        inner.offload_module_in_cuda_graph = offload_core_attn
        inner.config = SimpleNamespace(
            cuda_graph_modules=[CudaGraphModule.attn], fine_grained_activation_offloading=True
        )
        return inner

    @staticmethod
    def _recording_offload_interface(calls):
        class RecordingOffloadInterface:
            @staticmethod
            def backward_record(hidden_states):
                calls.append('backward_record')
                return hidden_states + 1

            @staticmethod
            def forward_record():
                calls.append('forward_record')

        return RecordingOffloadInterface

    @staticmethod
    def _rotary_sample_helper(
        monkeypatch,
        *,
        layer_kind,
        position_embedding_type,
        has_attention=True,
        multi_latent_attention=False,
        is_thd=False,
        context_parallel_size=1,
        cuda_graph_modules=None,
        num_layers=1,
    ):
        """Build a CPU-only TE sample-input helper for rotary contract tests."""
        from megatron.core.transformer.identity_op import IdentityOp

        if cuda_graph_modules is None:
            cuda_graph_modules = [CudaGraphModule.attn]

        layers = []
        for _ in range(num_layers):
            inner = TransformerLayer.__new__(TransformerLayer)
            torch.nn.Module.__init__(inner)
            inner.self_attention = torch.nn.Linear(2, 2) if has_attention else IdentityOp()
            inner.cross_attention = IdentityOp()

            if layer_kind == 'hybrid':
                layer = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
                torch.nn.Module.__init__(layer)
                layer.inner_layer = inner
            else:
                assert layer_kind == 'transformer'
                layer = inner

            def get_layer_static_inputs(_seq_length, _micro_batch_size):
                static_inputs = {'hidden_states': torch.ones(8, 1, 8, requires_grad=True)}
                if is_thd:
                    static_inputs['cu_seqlens_q'] = torch.tensor([0, 8], dtype=torch.int32)
                return static_inputs

            object.__setattr__(layer, 'get_layer_static_inputs', get_layer_static_inputs)
            layers.append(layer)

        rotary_pos_emb = torch.ones(8, 1, 1, 2)
        rotary_calls = []

        class RotaryEmbedding:
            @staticmethod
            def get_rotary_seq_len(*_args):
                return 8

            @staticmethod
            def __call__(_seq_length, offset=0, packed_seq=False, cp_group=None):
                # Mirror RotaryEmbedding.forward so unexpected keywords still fail.
                del offset, cp_group
                rotary_calls.append({'seq_length': _seq_length, 'packed_seq': packed_seq})
                return rotary_pos_emb

        class YarnRotaryEmbedding(RotaryEmbedding):
            @staticmethod
            def __call__(_seq_length, offset=0, packed_seq=False, cp_group=None):
                # Mirror YarnRotaryEmbedding.forward: returns (emb, mscale).
                del offset, cp_group
                rotary_calls.append({'seq_length': _seq_length, 'packed_seq': packed_seq})
                return rotary_pos_emb, 1.0

        class MultimodalRotaryEmbedding:
            # No get_rotary_seq_len, like the real module.
            @staticmethod
            def __call__(
                position_ids, mrope_section, cp_group=None, return_raw_freqs=False, packed_seq=False
            ):
                # Mirror MultimodalRotaryEmbedding.forward: position_ids [3, bs, seq] ->
                # [seq, bs, 1, dim] materialized embeddings.
                del cp_group
                assert position_ids.dim() == 3 and position_ids.shape[0] == 3
                rotary_calls.append(
                    {
                        'position_ids_shape': tuple(position_ids.shape),
                        'position_ids_dtype': position_ids.dtype,
                        'mrope_section': mrope_section,
                        'return_raw_freqs': return_raw_freqs,
                        'packed_seq': packed_seq,
                    }
                )
                return torch.ones(position_ids.shape[2], position_ids.shape[1], 1, 2)

        if position_embedding_type == 'yarn':
            rotary_module = YarnRotaryEmbedding()
        elif position_embedding_type == 'mrope':
            rotary_module = MultimodalRotaryEmbedding()
        else:
            rotary_module = RotaryEmbedding()

        chunk = SimpleNamespace(
            decoder=SimpleNamespace(layers=layers),
            position_embedding_type=position_embedding_type,
            rotary_pos_emb=rotary_module,
            mrope_section=[1, 0, 0],
            _fused_mrope_available=False,
        )
        helper = object.__new__(TECudaGraphHelper)
        helper.config = SimpleNamespace(
            multi_latent_attention=multi_latent_attention,
            cuda_graph_modules=cuda_graph_modules,
            context_parallel_size=context_parallel_size,
            sequence_parallel=False,
            tensor_model_parallel_size=1,
            apply_rope_fusion=False,
            rotary_interleaved=False,
            fused_single_qkv_rope=False,
            sequence_packing_scheduler="dp_balanced" if is_thd else None,
        )
        helper._rotary_calls = rotary_calls
        helper.seq_length = 8
        helper.micro_batch_size = 1
        helper.num_model_chunks = 1
        helper.num_microbatches = 1
        helper.flattened_callables = layers
        helper.num_layers_per_chunk = [len(layers)]
        helper.callables_per_chunk = [layers]
        helper.chunks_with_decoder = [chunk]
        helper.tp_group = None
        helper.dp_cp_group = None
        helper._needs_full_local_padding_mask = lambda *_args: False
        helper._uses_mhc_direct_write_arena = lambda: False
        monkeypatch.setattr(cuda_graphs_module, 'is_te_min_version', lambda _version: True)
        return helper, rotary_pos_emb

    @pytest.mark.parametrize(
        ('has_attention', 'position_embedding_type', 'multi_latent_attention', 'expects_rotary'),
        [
            (True, 'rope', False, True),
            (False, 'rope', False, False),
            (True, 'learned_absolute', False, False),
            (True, 'rope', True, False),
        ],
    )
    def test_hybrid_wrapper_sample_inputs_include_rotary_embeddings(
        self,
        monkeypatch,
        has_attention,
        position_embedding_type,
        multi_latent_attention,
        expects_rotary,
    ):
        """The TE sample signature must match the wrapped attention replay signature."""
        helper, rotary_pos_emb = self._rotary_sample_helper(
            monkeypatch,
            layer_kind='hybrid',
            position_embedding_type=position_embedding_type,
            has_attention=has_attention,
            multi_latent_attention=multi_latent_attention,
        )

        _sample_args, sample_kwargs = helper._get_sample_arguments([1, -1])

        if expects_rotary:
            assert sample_kwargs[0]['rotary_pos_emb'] is rotary_pos_emb
        else:
            assert 'rotary_pos_emb' not in sample_kwargs[0]

    @pytest.mark.parametrize(
        (
            'layer_kind',
            'position_embedding_type',
            'has_attention',
            'multi_latent_attention',
            'is_thd',
            'context_parallel_size',
            'cuda_graph_modules',
            'expected_fragment',
        ),
        [
            ('transformer', 'yarn', True, False, False, 1, [CudaGraphModule.attn], 'Yarn'),
            ('hybrid', 'yarn', True, False, False, 1, [CudaGraphModule.attn], 'Yarn'),
            ('transformer', 'mrope', True, False, False, 1, [CudaGraphModule.attn], 'mRoPE'),
            ('transformer', 'rope', True, False, True, 2, [CudaGraphModule.attn], 'THD'),
            ('hybrid', 'rope', True, False, True, 2, [CudaGraphModule.attn], 'THD'),
            ('transformer', 'yarn', True, False, False, 1, [], 'Yarn'),
            ('hybrid', 'rope', True, False, True, 1, [CudaGraphModule.attn], None),
            ('transformer', 'rope', True, False, False, 2, [CudaGraphModule.attn], None),
            ('transformer', 'yarn', True, True, True, 2, [CudaGraphModule.attn], None),
            ('hybrid', 'rope', True, True, True, 2, [CudaGraphModule.attn], None),
            ('hybrid', 'yarn', False, False, True, 2, [CudaGraphModule.attn], None),
            ('transformer', 'yarn', True, False, True, 2, [CudaGraphModule.mlp], None),
        ],
    )
    def test_rotary_sample_contract_warning_matrix(
        self,
        monkeypatch,
        layer_kind,
        position_embedding_type,
        has_attention,
        multi_latent_attention,
        is_thd,
        context_parallel_size,
        cuda_graph_modules,
        expected_fragment,
    ):
        records = []
        monkeypatch.setattr(
            cuda_graphs_module,
            'log_on_each_pipeline_stage',
            lambda **kwargs: records.append((kwargs['level'], kwargs['msg'])),
        )
        helper, _rotary_pos_emb = self._rotary_sample_helper(
            monkeypatch,
            layer_kind=layer_kind,
            position_embedding_type=position_embedding_type,
            has_attention=has_attention,
            multi_latent_attention=multi_latent_attention,
            is_thd=is_thd,
            context_parallel_size=context_parallel_size,
            cuda_graph_modules=cuda_graph_modules,
        )

        helper._get_sample_arguments([1, -1])

        if expected_fragment is None:
            assert records == []
        else:
            assert len(records) == 1
            assert records[0][0] == logging.WARNING
            assert expected_fragment in records[0][1]
            assert f"position_embedding_type={position_embedding_type!r}" in records[0][1]
            assert f'input_format={"THD" if is_thd else "SBHD"}' in records[0][1]
            assert f'context_parallel_size={context_parallel_size}' in records[0][1]
            assert 'numerical results are not reliable' in records[0][1]
            if position_embedding_type == 'mrope':
                assert 'affecting GPTModel.' in records[0][1]
                assert 'GPTModel and HybridModel' not in records[0][1]

    @staticmethod
    def _make_chunk_callable(helper):
        """Make the helper's single callable the chunk decoder itself (a whole-block
        callable)."""
        chunk = helper.chunks_with_decoder[0]
        decoder = helper.flattened_callables[0]
        decoder.layers = [decoder]
        chunk.decoder = decoder

    @pytest.mark.parametrize(
        ('position_embedding_type', 'is_thd'),
        [('rope', False), ('rope', True), ('yarn', False), ('yarn', True), ('mrope', False)],
    )
    def test_chunk_callable_materializes_rotary_table(
        self, monkeypatch, position_embedding_type, is_thd
    ):
        """A chunk callable is the whole training forward and TE flattens only the captured
        keyword tensors, so its sample must carry a runtime-shaped rotary table for rope, yarn
        and (fixed-length) mRoPE; none of them is a per-layer sample-contract warning."""
        records = []
        monkeypatch.setattr(
            cuda_graphs_module,
            'log_on_each_pipeline_stage',
            lambda **kwargs: records.append((kwargs['level'], kwargs['msg'])),
        )
        helper, rotary_pos_emb = self._rotary_sample_helper(
            monkeypatch,
            layer_kind='transformer',
            position_embedding_type=position_embedding_type,
            cuda_graph_modules=[],
            is_thd=is_thd,
        )
        self._make_chunk_callable(helper)

        _sample_args, sample_kwargs = helper._get_sample_arguments([1, -1])

        assert records == []
        table = sample_kwargs[0]['rotary_pos_emb']
        assert isinstance(table, torch.Tensor)
        # [seq, batch, 1, dim]: seq = 8 tokens, batch = 1 (hidden_states [8, 1, 8]).
        assert table.shape == (8, 1, 1, 2)
        assert table.dtype == torch.float32
        assert len(helper._rotary_calls) == 1
        call = helper._rotary_calls[0]
        # THD block callables capture the full packed table, not the per-CP-rank slice.
        assert call['packed_seq'] is is_thd
        if position_embedding_type == 'mrope':
            # Built from a [3, batch, seq] int64 position_ids sample with the model's
            # mrope_section, mirroring GPTModel._preprocess.
            assert call['position_ids_shape'] == (3, 1, 8)
            assert call['position_ids_dtype'] == torch.long
            assert call['mrope_section'] == [1, 0, 0]
            assert call['return_raw_freqs'] is False
        else:
            assert table is rotary_pos_emb
            assert call['seq_length'] == 8

    @pytest.mark.parametrize('position_embedding_type', ['rope', 'yarn', 'mrope'])
    def test_chunk_callable_rotary_table_per_in_flight_slot(
        self, monkeypatch, position_embedding_type
    ):
        """Two microbatches in flight (forward, forward, backward, backward) are two graph
        slots. The rope / yarn tables are constants, so the slots share one sample tensor. An
        mRoPE table follows each microbatch's position_ids and the rotation's backward reads it
        again, so every slot gets its own static buffer: the second microbatch's forward replay
        would overwrite a shared one before the first microbatch's backward ran."""
        helper, rotary_pos_emb = self._rotary_sample_helper(
            monkeypatch,
            layer_kind='transformer',
            position_embedding_type=position_embedding_type,
            cuda_graph_modules=[],
        )
        self._make_chunk_callable(helper)
        helper.num_microbatches = 2

        _sample_args, sample_kwargs = helper._get_sample_arguments([1, 1, -1, -1])

        first, second = (kwargs['rotary_pos_emb'] for kwargs in sample_kwargs)
        if position_embedding_type == 'mrope':
            assert first is not second
            assert len(helper._rotary_calls) == 2
        else:
            assert first is second is rotary_pos_emb
            assert len(helper._rotary_calls) == 1

    def test_chunk_callable_rejects_mrope_with_packed_sequences(self, monkeypatch):
        """The packing scheduler provides [1, tokens] position ids, so no runtime mRoPE table
        exists for a THD chunk callable to copy into the sample: refuse instead of replaying
        attention without RoPE."""
        records = []
        monkeypatch.setattr(
            cuda_graphs_module,
            'log_on_each_pipeline_stage',
            lambda **kwargs: records.append((kwargs['level'], kwargs['msg'])),
        )
        helper, _rotary_pos_emb = self._rotary_sample_helper(
            monkeypatch,
            layer_kind='transformer',
            position_embedding_type='mrope',
            cuda_graph_modules=[],
            is_thd=True,
        )
        self._make_chunk_callable(helper)

        with pytest.raises(ValueError, match="'mrope' together with packed \\(THD\\) sequences"):
            helper._get_sample_arguments([1, -1])
        assert records == []
        assert helper._rotary_calls == []

    def test_rotary_sample_contract_warning_is_deduplicated_for_yarn_thd_cp2(self, monkeypatch):
        records = []
        monkeypatch.setattr(
            cuda_graphs_module,
            'log_on_each_pipeline_stage',
            lambda **kwargs: records.append((kwargs['level'], kwargs['msg'])),
        )
        helper, _rotary_pos_emb = self._rotary_sample_helper(
            monkeypatch,
            layer_kind='hybrid',
            position_embedding_type='yarn',
            is_thd=True,
            context_parallel_size=2,
            num_layers=2,
        )

        helper._get_sample_arguments([1, -1])

        assert len(records) == 1
        assert records[0][0] == logging.WARNING
        assert "position_embedding_type='yarn'" in records[0][1]
        assert 'input_format=THD' in records[0][1]
        assert 'context_parallel_size=2' in records[0][1]

    def test_hybrid_mtp_layers_are_flattened_and_adjacent_layers_are_grouped(self, monkeypatch):
        from megatron.core.transformer import cuda_graphs

        class FakeGraphLayer(torch.nn.Module):
            def __init__(self, group_with_next=False, graphable=True):
                super().__init__()
                self.group_with_next = group_with_next
                self.graphable = graphable
                self.group_tail = None

            def _can_group_te_cuda_graph_with(self, next_layer):
                return self.group_with_next and next_layer.graphable

            def _set_te_cuda_graph_group_tail(self, next_layer):
                self.group_tail = next_layer

        head = FakeGraphLayer(group_with_next=True)
        tail = FakeGraphLayer()
        eager = FakeGraphLayer(graphable=False)
        mtp_stack = HybridStack.__new__(HybridStack)
        torch.nn.Module.__init__(mtp_stack)
        mtp_stack.layers = torch.nn.ModuleList([head, tail, eager])

        monkeypatch.setattr(
            cuda_graphs, '_te_cuda_graph_layer_is_graphable', lambda layer, config: layer.graphable
        )
        callables = cuda_graphs._get_mtp_te_callables(mtp_stack, object())

        assert callables == [head]
        assert head.group_tail is tail

        gpt_mtp_layer = FakeGraphLayer()
        assert cuda_graphs._get_mtp_te_callables(gpt_mtp_layer, object()) == [gpt_mtp_layer]

        class Holder:
            pass

        mtp_layer = Holder()
        mtp_layer.mtp_model_layer = mtp_stack
        chunk = Holder()
        chunk.mtp = Holder()
        chunk.mtp.layers = [mtp_layer]
        assert cuda_graphs._is_mtp_te_callable(head, chunk)
        assert cuda_graphs._is_mtp_te_callable(tail, chunk)
        assert not cuda_graphs._is_mtp_te_callable(eager, Holder())

    def test_capture_group_tail_does_not_change_module_registration(self):
        from megatron.core.transformer.identity_op import IdentityOp

        class Config:
            context_parallel_size = 1
            recompute_granularity = None
            recompute_modules = []
            fp8 = False
            first_last_layers_bf16 = False

        inner = TransformerLayer.__new__(TransformerLayer)
        torch.nn.Module.__init__(inner)
        inner.self_attention = torch.nn.Linear(2, 2)
        inner.cross_attention = IdentityOp()
        inner.mlp = IdentityOp()

        head = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
        torch.nn.Module.__init__(head)
        head.config = Config()
        head.inner_layer = inner

        tail = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
        torch.nn.Module.__init__(tail)
        tail.config = Config()
        tail.capture_weight = torch.nn.Parameter(torch.ones(1))
        object.__setattr__(tail, '_inner_is_partial_moe_capture', lambda: True)
        object.__setattr__(tail, '_get_submodules_under_cudagraphs', lambda: [tail])

        state_dict_keys = tuple(head.state_dict())
        head._set_te_cuda_graph_group_tail(tail)

        assert head._get_te_cuda_graph_group_tail() is tail
        assert '_te_cuda_graph_group_tail' not in head._modules
        assert tuple(head.state_dict()) == state_dict_keys
        assert any(param is tail.capture_weight for param in head.parameters())

        head.te_cuda_graphs = [object()]
        head.train()
        assert head._get_active_te_cuda_graph_group_tail() is tail
        head.eval()
        assert head._get_active_te_cuda_graph_group_tail() is None

    def test_capture_group_does_not_cross_first_last_bf16_boundary(self):
        from megatron.core.transformer.identity_op import IdentityOp

        class Config:
            context_parallel_size = 1
            recompute_granularity = None
            recompute_modules = []
            fp8 = True
            first_last_layers_bf16 = True
            num_layers_at_start_in_bf16 = 1
            num_layers_at_end_in_bf16 = 1
            num_layers = 4

        inner = TransformerLayer.__new__(TransformerLayer)
        torch.nn.Module.__init__(inner)
        inner.self_attention = torch.nn.Linear(2, 2)
        inner.cross_attention = IdentityOp()
        inner.mlp = IdentityOp()
        inner.is_mtp_layer = False

        head = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
        torch.nn.Module.__init__(head)
        head.config = Config()
        head.inner_layer = inner

        tail = HyperConnectionHybridLayer.__new__(HyperConnectionHybridLayer)
        torch.nn.Module.__init__(tail)
        tail.config = Config()
        object.__setattr__(tail, '_inner_is_partial_moe_capture', lambda: True)

        head.layer_number, tail.layer_number = 1, 2
        assert not head._can_group_te_cuda_graph_with(tail)

        head.layer_number, tail.layer_number = 2, 3
        assert head._can_group_te_cuda_graph_with(tail)

        head.layer_number, tail.layer_number = 3, 4
        assert not head._can_group_te_cuda_graph_with(tail)

        inner.is_mtp_layer = True
        head.layer_number, tail.layer_number = 1, 2
        assert head._can_group_te_cuda_graph_with(tail)

    @pytest.mark.parametrize('offload_in_graph', [False, True])
    def test_hybrid_offload_graph_replay_args_include_te_stream_and_event(
        self, monkeypatch, offload_in_graph
    ):
        wrapper = self._bare_hybrid_wrapper(offload_in_graph=offload_in_graph)
        graph_stream = object()
        graph_event = object()
        off_interface = SimpleNamespace(
            cuda_graph_stream=lambda: graph_stream, cuda_graph_event=lambda: graph_event
        )
        object.__setattr__(wrapper, 'get_offload_interface', lambda: off_interface)

        cudagraph_args, cudagraph_kwargs = wrapper._get_te_cuda_graph_replay_args(
            torch.ones(2, 1, 4)
        )

        assert len(cudagraph_args) == 1
        if offload_in_graph:
            assert cudagraph_kwargs['cuda_graph_stream'] is graph_stream
            assert cudagraph_kwargs['cuda_graph_event'] is graph_event
        else:
            assert 'cuda_graph_stream' not in cudagraph_kwargs
            assert 'cuda_graph_event' not in cudagraph_kwargs

    def test_hybrid_capture_records_offload_boundary_exactly_once(self):
        calls = []
        wrapper = self._bare_hybrid_wrapper(offload_in_graph=True)
        off_interface = self._recording_offload_interface(calls)
        object.__setattr__(wrapper, 'get_offload_interface', lambda: off_interface)
        object.__setattr__(wrapper, '_get_te_cuda_graph_group_tail', lambda: None)
        object.__setattr__(wrapper, '_inner_is_partial_moe_capture', lambda: False)

        def forward(hidden_states, **_kwargs):
            calls.append('body')
            assert torch.equal(hidden_states, torch.full_like(hidden_states, 2))
            return hidden_states * 2, None

        object.__setattr__(wrapper, 'forward', forward)
        output = wrapper._te_cuda_graph_capture(torch.ones(2, 1, 4))

        assert calls == ['backward_record', 'body', 'forward_record']
        assert torch.equal(output[0], torch.full((2, 1, 4), 4.0))

    def test_hybrid_capture_requires_offload_interface_for_enabled_boundary(self):
        wrapper = self._bare_hybrid_wrapper(offload_in_graph=True)
        object.__setattr__(wrapper, 'get_offload_interface', lambda: None)

        with pytest.raises(AssertionError, match='offload[ _]interface'):
            wrapper._te_cuda_graph_capture(torch.ones(2, 1, 4))

    def test_hybrid_capture_impl_rejects_raw_packed_sequence_kwargs(self):
        wrapper = self._bare_hybrid_wrapper(offload_in_graph=False)

        with pytest.raises(AssertionError):
            wrapper._te_cuda_graph_capture_impl(
                torch.ones(2, 1, 4), cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32)
            )

    def test_transformer_capture_impl_rejects_raw_packed_sequence_kwargs(self):
        layer = TransformerLayer.__new__(TransformerLayer)
        torch.nn.Module.__init__(layer)

        with pytest.raises(AssertionError):
            layer._te_cuda_graph_capture_impl(
                torch.ones(2, 1, 4), cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32)
            )

    def test_grouped_hybrid_capture_has_one_outer_offload_boundary(self):
        calls = []
        head = self._bare_hybrid_wrapper(offload_in_graph=True)
        tail = self._bare_hybrid_wrapper(offload_in_graph=True)
        off_interface = self._recording_offload_interface(calls)
        object.__setattr__(head, 'get_offload_interface', lambda: off_interface)
        object.__setattr__(tail, 'get_offload_interface', lambda: off_interface)
        object.__setattr__(head, '_get_te_cuda_graph_group_tail', lambda: tail)

        def head_forward(hidden_states, **_kwargs):
            calls.append('attention_body')
            return hidden_states + 1, None

        def tail_capture_impl(hidden_states, **_kwargs):
            calls.append('moe_prefix_body')
            return (hidden_states + 1,)

        object.__setattr__(head, 'forward', head_forward)
        object.__setattr__(tail, '_te_cuda_graph_capture_impl', tail_capture_impl)

        output = head._te_cuda_graph_capture(torch.ones(2, 1, 4))

        assert calls == ['backward_record', 'attention_body', 'moe_prefix_body', 'forward_record']
        assert torch.equal(output[0], torch.full((2, 1, 4), 4.0))

    def test_moe_only_tail_does_not_claim_attention_offload(self):
        # Reproduce the stale inner-layer flag: cuda_graph_modules contains ``attn``
        # globally even though this split Hybrid layer has no attention body.
        inner = self._bare_transformer_inner(
            has_attention=False, offload_core_attn=True, is_moe=True
        )

        tail = self._bare_hybrid_wrapper()
        tail.inner_layer = inner

        assert not tail._compute_offload_module_in_cuda_graph()
        assert not tail.offload_module_in_cuda_graph

    def test_hybrid_explicit_attn_scope_claims_inner_offload(self):
        attention_inner = self._bare_transformer_inner(has_attention=True, offload_core_attn=True)

        attention_wrapper = self._bare_hybrid_wrapper()
        attention_wrapper.inner_layer = attention_inner
        assert attention_wrapper.offload_module_in_cuda_graph

    def test_hybrid_offload_property_caches_and_grouping_invalidates(self):
        calls = []
        head = self._bare_hybrid_wrapper()
        tail = self._bare_hybrid_wrapper()
        object.__setattr__(
            head,
            '_compute_inner_offload_module_in_cuda_graph',
            lambda: calls.append('head') or False,
        )
        object.__setattr__(
            tail,
            '_compute_inner_offload_module_in_cuda_graph',
            lambda: calls.append('tail') or True,
        )
        object.__setattr__(head, '_can_group_te_cuda_graph_with', lambda _tail: True)

        assert not head.offload_module_in_cuda_graph
        assert not head.offload_module_in_cuda_graph
        assert calls == ['head']

        head._set_te_cuda_graph_group_tail(tail)
        assert head._offload_module_in_cuda_graph_cached is None
        assert head.offload_module_in_cuda_graph
        assert head.offload_module_in_cuda_graph
        assert calls == ['head', 'head', 'tail']
        assert head._offload_module_in_cuda_graph_cached is True

    @pytest.mark.parametrize('tail_raises', [False, True])
    @pytest.mark.parametrize('grouped', [False, True])
    def test_hybrid_partial_replay_runs_eager_tail_and_propagates_errors(
        self, monkeypatch, tail_raises, grouped
    ):
        calls = []
        head = self._bare_hybrid_wrapper()

        class Tail:
            @staticmethod
            def _resume_partial_moe_cuda_graph(_outputs, padding_mask=None):
                calls.append('eager_tail')
                if tail_raises:
                    raise RuntimeError('eager tail failed')
                return torch.full((2, 1, 4), 3.0), None

        object.__setattr__(
            head, '_get_te_cuda_graph_group_tail', lambda: Tail() if grouped else None
        )
        object.__setattr__(head, '_inner_is_partial_moe_capture', lambda: not grouped)
        object.__setattr__(
            head, '_resume_partial_moe_cuda_graph', Tail._resume_partial_moe_cuda_graph
        )

        def graph_replay(_self, *_args, **_kwargs):
            calls.append('graph_replay')
            return (torch.ones(2, 1, 4),)

        monkeypatch.setattr(HyperConnectionHybridLayer, '_te_cuda_graph_replay_graph', graph_replay)

        if tail_raises:
            with pytest.raises(RuntimeError, match='eager tail failed'):
                head._te_cuda_graph_replay(torch.ones(2, 1, 4))
        else:
            output = head._te_cuda_graph_replay(torch.ones(2, 1, 4))
            assert torch.equal(output[0], torch.full((2, 1, 4), 3.0))

        assert calls == ['graph_replay', 'eager_tail']


# Global storage for comparing unique buffer counts across different num_microbatches,
# keyed by (pp_size, vpp_size)
_unique_buffer_counts = {}


class TestTECudaGraphHelper:
    @pytest.mark.launch_on_gb200
    @pytest.mark.parametrize(
        "pp_size,cp_size,chunks",
        [
            (1, 1, 1),
            (2, 1, 1),
            (2, 2, 1),
            pytest.param(
                2,
                2,
                2,
                marks=pytest.mark.skipif(
                    'torch.cuda.is_available() and "GB200" in torch.cuda.get_device_name().upper()',
                    reason="GB200 CUDA OOM followed by NCCL timeout",
                ),
            ),
        ],
    )
    def test_scheduler_thd_replay_preserves_pipeline_inputs(
        self, monkeypatch, pp_size, cp_size, chunks
    ):
        """Changing packed microbatch counts must preserve outputs and every gradient."""
        from megatron.core.extensions.te_cuda_graphs import set_te_cuda_graph_current_microbatch
        from megatron.core.pipeline_parallel.schedules import (
            get_pp_rank_microbatches,
            get_schedule_table,
        )
        from megatron.core.transformer.cuda_graphs import (
            convert_te_cuda_graph_schedule_table_to_order,
        )

        Utils.initialize_model_parallel(
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            virtual_pipeline_model_parallel_size=chunks if chunks > 1 else None,
        )
        # Match training initialization: DDP keeps AccumulateGrad nodes alive,
        # so construct and warm up on the non-default training stream.
        training_stream = torch.cuda.Stream()
        training_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(training_stream):
            monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
            monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
            monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
            monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
            model_parallel_cuda_manual_seed(321)
            init_num_microbatches_calculator(
                rank=0,
                global_batch_size=4,
                micro_batch_size=1,
                data_parallel_size=1,
                decrease_batch_size_if_needed=False,
            )
            config = TransformerConfig(
                num_layers=pp_size * chunks,
                hidden_size=64,
                num_attention_heads=4,
                use_cpu_initialization=True,
                cuda_graph_impl="transformer_engine",
                cuda_graph_thd_max_packed_sequences=4,
                cuda_graph_modules=[CudaGraphModule.attn],
                sequence_packing_scheduler="dp_balanced",
                max_seqlen_per_dp_cp_rank=32 // cp_size,
                pad_packed_seq_alignment="max",
                cuda_graph_warmup_steps=1,
                use_te_rng_tracker=True,
                bf16=True,
                params_dtype=torch.bfloat16,
                pipeline_dtype=torch.bfloat16,
                pipeline_model_parallel_size=pp_size,
                context_parallel_size=cp_size,
                virtual_pipeline_model_parallel_size=chunks if chunks > 1 else None,
                microbatch_group_size_per_vp_stage=2,
                attention_backend=AttnBackend.fused,
                deterministic_mode=True,
                attention_dropout=0.0,
                hidden_dropout=0.0,
            )
            models = []
            for chunk in range(chunks):
                model = GPTModel(
                    config=config,
                    transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                    vocab_size=128,
                    max_sequence_length=32,
                    parallel_output=True,
                    position_embedding_type="rope",
                    vp_stage=chunk if chunks > 1 else None,
                ).cuda()
                models.append(
                    DistributedDataParallel(
                        config,
                        DistributedDataParallelConfig(),
                        model,
                        disable_bucketing=True,
                        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                    )
                )
            helper = TECudaGraphHelper(models, config, 32, 1, optimizers=[])
            rotary = models[0].module.rotary_pos_emb(32, packed_seq=True)
            physical = torch.tensor([0, 8, 16, 24, 32], dtype=torch.int32, device="cuda")
            packed = []
            for valid_end in (32, 24):
                valid = physical.clone()
                valid[-1] = valid_end
                packed.append(
                    PackedSeqParams(
                        qkv_format="thd",
                        cu_seqlens_q=valid,
                        cu_seqlens_kv=valid.clone(),
                        cu_seqlens_q_padded=physical,
                        cu_seqlens_kv_padded=physical.clone(),
                        max_seqlen_q=32,
                        max_seqlen_kv=32,
                        pad_between_seqs=True,
                    )
                )
            inputs, expected, expected_params = {}, {}, {}
            for chunk, callables in enumerate(helper.callables_per_chunk):
                layer = callables[0]
                for mb in range(4):
                    hidden = torch.randn(
                        32 // cp_size,
                        1,
                        64,
                        dtype=torch.bfloat16,
                        device="cuda",
                        requires_grad=True,
                    )
                    inputs[chunk, mb] = hidden.detach().clone()
                    output, _ = layer(
                        hidden,
                        attention_mask=None,
                        rotary_pos_emb=rotary,
                        packed_seq_params=packed[mb % 2],
                    )
                    value = output.detach().clone()
                    valid_tokens = (24 if mb % 2 else 32) // cp_size
                    output[:valid_tokens].float().sum().backward()
                    expected[chunk, mb] = value, hidden.grad.detach().clone()
                    if mb in (1, 3):
                        expected_params[chunk, mb + 1] = {
                            name: param.main_grad.detach().clone()
                            for name, param in layer.named_parameters()
                            if param.requires_grad
                        }
                models[chunk].zero_grad_buffer()
            del hidden, output
            try:
                helper.create_cudagraphs()
                assert helper.graphs_created()
                assert helper.num_microbatches == (1 if pp_size == 1 else 4)
                for count in (4, 2, 4):
                    _, _, warmup, _ = get_pp_rank_microbatches(
                        count, chunks, 2, p2p_communicator=helper.p2p_communicator
                    )
                    order = convert_te_cuda_graph_schedule_table_to_order(
                        warmup, chunks, get_schedule_table(count, chunks, 2)
                    )
                    forwards, backwards, pending = [0] * chunks, [0] * chunks, {}
                    for entry in order:
                        chunk = abs(entry) - 1
                        layer = helper.callables_per_chunk[chunk][0]
                        if entry > 0:
                            mb = forwards[chunk]
                            forwards[chunk] += 1
                            set_te_cuda_graph_current_microbatch(models[chunk], mb)
                            hidden = inputs[chunk, mb].clone().requires_grad_(True)
                            output, _ = layer(
                                hidden,
                                attention_mask=None,
                                rotary_pos_emb=rotary,
                                packed_seq_params=packed[mb % 2],
                            )
                            valid_tokens = (24 if mb % 2 else 32) // cp_size
                            torch.testing.assert_close(
                                output[:valid_tokens],
                                expected[chunk, mb][0][:valid_tokens],
                                msg=f"output count={count} chunk={chunk} mb={mb}",
                            )
                            pending[chunk, mb] = hidden, output
                        else:
                            mb = backwards[chunk]
                            backwards[chunk] += 1
                            hidden, output = pending.pop((chunk, mb))
                            valid_tokens = (24 if mb % 2 else 32) // cp_size
                            output[:valid_tokens].float().sum().backward()
                            torch.testing.assert_close(
                                hidden.grad,
                                expected[chunk, mb][1],
                                msg=f"input grad count={count} chunk={chunk} mb={mb}",
                            )
                    assert not pending
                    for chunk, callables in enumerate(helper.callables_per_chunk):
                        actual = {
                            name: param.main_grad
                            for name, param in callables[0].named_parameters()
                            if param.requires_grad
                        }
                        assert actual.keys() == expected_params[chunk, count].keys()
                        for name, grad in actual.items():
                            torch.testing.assert_close(
                                grad,
                                expected_params[chunk, count][name],
                                msg=f"parameter {name} count={count} chunk={chunk}",
                            )
                        models[chunk].zero_grad_buffer()
            finally:
                if helper.graphs_created():
                    helper.delete_cuda_graphs()
        torch.cuda.current_stream().wait_stream(training_stream)

    def setup_method(self, method):
        # Initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        # Note: _unique_buffer_counts is intentionally NOT cleared here so we can
        # compare values across parametrized test runs

    @pytest.mark.parametrize(
        ("local_layout", "cuda_graph_modules", "expected"),
        [
            ("dense", [CudaGraphModule.attn, CudaGraphModule.moe], False),
            ("dense", [], False),
            ("direct-moe", [CudaGraphModule.attn, CudaGraphModule.moe], True),
            ("nested-moe", [], True),
            ("direct-moe", [CudaGraphModule.moe_router], False),
        ],
    )
    def test_paged_stash_te_capture_context_requires_rank_local_whole_moe(
        self, monkeypatch, local_layout, cuda_graph_modules, expected
    ):
        layer = torch.nn.Module()
        layer.is_moe_layer = local_layout == "direct-moe"
        if local_layout == "nested-moe":
            inner_layer = torch.nn.Module()
            inner_layer.is_moe_layer = True
            layer.inner_layer = inner_layer

        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)

        helper = object.__new__(TECudaGraphHelper)
        helper.config = _te_whole_moe_paged_stash_config(cuda_graph_modules=cuda_graph_modules)
        helper.model = []
        helper.pg_collection = SimpleNamespace(pp=object())
        helper.pp_group = object()
        helper.flattened_callables = [layer]
        helper._captures_dsa_metric_writes = False
        helper.callables_per_chunk = []
        helper.num_microbatches = 1
        helper._start_capturing = lambda: 0.0
        helper._finish_capturing = lambda _start_time: None
        helper._get_cuda_graph_input_data = lambda: ([()], {'_order': [1, -1]})
        helper._validate_mhc_static_hidden_inputs = lambda _sample_args: None
        helper._uses_mhc_direct_write_arena = lambda: False

        capture_enabled = []

        def record_capture_context(enabled, order=None, config=None):
            capture_enabled.append(enabled)
            return nullcontext()

        monkeypatch.setattr(
            paged_stash_module, "paged_stash_te_graph_capture", record_capture_context
        )
        # create_cudagraphs() imports make_graphed_callables from TE at call time,
        # so patch the TE module attribute rather than a cuda_graphs global.
        monkeypatch.setattr(
            "transformer_engine.pytorch.graph.make_graphed_callables", lambda *args, **kwargs: ()
        )

        helper.create_cudagraphs()

        assert capture_enabled == [expected]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mhc_static_input_aliasing_requires_disjoint_liveness_windows(self):
        config = _base_cuda_graph_config(
            enable_hyper_connections=True,
            recompute_granularity="selective",
            recompute_modules=["mhc"],
            mhc_recompute_layer_num=2,
            overlap_moe_expert_parallel_comm=True,
            expert_model_parallel_size=2,
            num_moe_experts=4,
            moe_token_dispatcher_type="alltoall",
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=[CudaGraphModule.attn],
            # The aliasing check only runs for the direct-write arena, which is
            # opt-in: without the switch this shape captures the whole attention
            # range and has no arena slot to alias.
            mhc_recompute_attn_cuda_graph_split=True,
            bf16=True,
        )
        helper = object.__new__(TECudaGraphHelper)
        helper.config = config

        shared = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        unique = torch.randn_like(shared)
        # Samples 0 and 2 alias one static buffer (as MCore consumed-sample
        # reuse and TE _reuse_graph_input_output_buffers legally do); sample 1
        # owns its own bytes.
        sample_args = [(shared,), (unique,), (shared.detach().requires_grad_(True),)]

        # Disjoint windows: sample 0 fully retires before sample 2's forward.
        helper._mhc_sample_order_intervals = {0: [0, 3], 1: [1, 4], 2: [5, 6]}
        helper._validate_mhc_static_hidden_inputs(sample_args)

        # Overlapping windows on a shared address must fail at capture time:
        # sample 2's forward starts before sample 0's backward retired, which
        # is exactly the aliasing that corrupts recompute direct-write replay.
        helper._mhc_sample_order_intervals = {0: [0, 5], 1: [1, 4], 2: [3, 6]}
        with pytest.raises(RuntimeError, match="windows overlap"):
            helper._validate_mhc_static_hidden_inputs(sample_args)

        # An entry whose backward never retires in the order is live forever,
        # so any aliasing against it fails.
        helper._mhc_sample_order_intervals = {0: [0, None], 1: [1, 4], 2: [3, 6]}
        with pytest.raises(RuntimeError, match="windows overlap"):
            helper._validate_mhc_static_hidden_inputs(sample_args)

        # A sample with no recorded window at all is rejected outright.
        helper._mhc_sample_order_intervals = {1: [1, 4]}
        with pytest.raises(RuntimeError, match="no recorded"):
            helper._validate_mhc_static_hidden_inputs(sample_args)

    @pytest.mark.parametrize("num_microbatches", [16, 64, 256])
    @pytest.mark.parametrize("pp_size", [1, 2, 4])
    @pytest.mark.parametrize("vpp_size", [None, 2])
    def test_get_cuda_graph_input_data(self, num_microbatches, pp_size, vpp_size):
        """Test _get_cuda_graph_input_data function in TECudaGraphHelper."""

        if vpp_size and pp_size == 1:
            pytest.skip("vpp_size must be None when pp_size is 1")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vpp_size,
        )

        # Set up test configuration
        seq_length = 128
        micro_batch_size = 2
        num_layers = 8
        vocab_size = 1024
        hidden_size = 64
        num_attention_heads = 4

        # Initialize num_microbatches calculator
        init_num_microbatches_calculator(
            rank=0,
            global_batch_size=micro_batch_size * num_microbatches,
            micro_batch_size=micro_batch_size,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )

        # Create transformer config directly
        transformer_config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
            use_te_rng_tracker=True,
            bf16=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp_size,
            virtual_pipeline_model_parallel_size=vpp_size,
            pipeline_dtype=torch.bfloat16,
            context_parallel_size=1,
        )

        # Create model
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        model = []
        for i in range(vpp_size or 1):
            this_model = GPTModel(
                config=transformer_config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=vocab_size,
                max_sequence_length=seq_length,
                parallel_output=True,
                position_embedding_type="rope",
                vp_stage=i if vpp_size else None,
            ).cuda()
            model.append(this_model)

        # Initialize TECudaGraphHelper
        cuda_graph_helper = TECudaGraphHelper(
            model=model,
            config=transformer_config,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            optimizers=[],
        )

        # Call _get_cuda_graph_input_data (which internally calls _get_sample_arguments)
        sample_args, make_graphed_callables_kwargs = cuda_graph_helper._get_cuda_graph_input_data()

        # Extract sample_kwargs from the kwargs dict
        # For TE >= 1.10.0, sample_kwargs should always be present
        assert (
            'sample_kwargs' in make_graphed_callables_kwargs
        ), "sample_kwargs should be present in make_graphed_callables_kwargs for TE >= 1.10.0"
        if is_te_min_version("2.19.0"):
            assert make_graphed_callables_kwargs['clone_param_grads_on_return'] is False
        else:
            assert 'clone_param_grads_on_return' not in make_graphed_callables_kwargs
        sample_kwargs = make_graphed_callables_kwargs['sample_kwargs']

        # Basic checks
        num_graphable_layers = len(cuda_graph_helper.flattened_callables)
        if pp_size > 1:
            expected_length = num_graphable_layers * num_microbatches
        else:
            expected_length = num_graphable_layers
        assert len(sample_args) == expected_length, (
            f"sample_args length mismatch: expected {expected_length}, " f"got {len(sample_args)}"
        )
        assert len(sample_kwargs) == expected_length, (
            f"sample_kwargs length mismatch: expected {expected_length}, "
            f"got {len(sample_kwargs)}"
        )

        # Check that all elements are not None
        for i, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            assert args_item is not None, f"sample_args[{i}] is None"
            assert kwargs_item is not None, f"sample_kwargs[{i}] is None"
            assert isinstance(args_item, tuple), f"sample_args[{i}] should be a tuple"
            assert isinstance(kwargs_item, dict), f"sample_kwargs[{i}] should be a dict"
            assert len(args_item) > 0, f"sample_args[{i}] should not be empty"
            # Check that hidden_states is present
            assert "hidden_states" in kwargs_item or (
                len(args_item) > 0 and torch.is_tensor(args_item[0])
            ), f"sample_args[{i}] or sample_kwargs[{i}] should contain hidden_states"

        # Check tensor properties
        for i, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            # Get hidden_states from args or kwargs
            if len(args_item) > 0 and torch.is_tensor(args_item[0]):
                hidden_states = args_item[0]
            elif "hidden_states" in kwargs_item:
                hidden_states = kwargs_item["hidden_states"]
            else:
                continue

            assert torch.is_tensor(hidden_states), f"hidden_states at index {i} should be a tensor"
            # Check shape matches expected (accounting for TP/CP)
            expected_seq_len = seq_length // transformer_config.context_parallel_size
            if transformer_config.sequence_parallel:
                expected_seq_len = expected_seq_len // transformer_config.tensor_model_parallel_size
            assert hidden_states.shape[0] == expected_seq_len, (
                f"hidden_states seq_len mismatch at index {i}: "
                f"expected {expected_seq_len}, got {hidden_states.shape[0]}"
            )
            assert hidden_states.shape[1] == micro_batch_size, (
                f"hidden_states batch_size mismatch at index {i}: "
                f"expected {micro_batch_size}, got {hidden_states.shape[1]}"
            )
            assert hidden_states.shape[2] == transformer_config.hidden_size, (
                f"hidden_states hidden_size mismatch at index {i}: "
                f"expected {transformer_config.hidden_size}, got {hidden_states.shape[2]}"
            )

        # Memory optimization check: verify that buffers with same signature are reused
        # Create a mapping of sample_keys to indices
        sample_keys_to_indices = {}
        for idx, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            # Create sample_keys similar to the function
            args_keys = tuple((t.shape, t.dtype, t.layout) for t in args_item if torch.is_tensor(t))
            kwargs_keys = tuple(
                (k, v.shape, v.dtype, v.layout)
                for k, v in sorted(kwargs_item.items())
                if torch.is_tensor(v)
            )
            sample_keys = args_keys + kwargs_keys

            if sample_keys not in sample_keys_to_indices:
                sample_keys_to_indices[sample_keys] = []
            sample_keys_to_indices[sample_keys].append(idx)

        # Check that buffers with same signature share references (memory optimization)
        # The optimization reuses buffers when:
        # 1. They have the same signature (shape, dtype, layout)
        # 2. The backward pass of the original buffer has completed
        # 3. A new forward pass with matching signature needs a buffer
        # Count how many times each tensor is reused
        unique_tensors = set()
        tensor_reuse_count = {}
        for idx, (args_item, kwargs_item) in enumerate(zip(sample_args, sample_kwargs)):
            # Get the first tensor from args (hidden_states)
            if len(args_item) > 0 and torch.is_tensor(args_item[0]):
                tensor_ptr = args_item[0].data_ptr()
                unique_tensors.add(tensor_ptr)
                tensor_reuse_count[tensor_ptr] = tensor_reuse_count.get(tensor_ptr, 0) + 1

        # With memory optimization, we should see some buffers reused
        # (i.e., some tensors should appear multiple times)
        max_reuse = max(tensor_reuse_count.values()) if tensor_reuse_count else 0
        total_entries = len(sample_args)
        unique_buffer_count = len(unique_tensors)

        # Verify that memory optimization is working:
        # - The number of unique buffers should be <= total entries
        # - With the 1F1B schedule and multiple microbatches, we should see some buffer reuse
        # - The number of unique buffers should be bounded as num_microbatches grows.
        assert unique_buffer_count <= total_entries, (
            f"Memory optimization check: unique_buffer_count ({unique_buffer_count}) "
            f"should be <= total_entries ({total_entries})"
        )
        global _unique_buffer_counts
        # Use (pp_size, vpp_size) as key to track unique buffer counts per configuration
        config_key = (pp_size, vpp_size)
        if config_key not in _unique_buffer_counts:
            _unique_buffer_counts[config_key] = unique_buffer_count
        else:
            assert unique_buffer_count == _unique_buffer_counts[config_key], (
                f"Unique buffer count mismatch: expected {_unique_buffer_counts[config_key]}, "
                f"got {unique_buffer_count}"
            )

        # Verify that buffers with the same signature can potentially be reused
        # (the actual reuse depends on the schedule, but the mechanism should work)
        if expected_length > 1:
            # Check that we have multiple entries with the same signature
            has_duplicate_signatures = any(
                len(indices) > 1 for indices in sample_keys_to_indices.values()
            )
            assert has_duplicate_signatures, (
                "Memory optimization: expected duplicate signatures for buffer reuse, "
                "but all signatures are unique"
            )

            # We tested with a large number of microbatches, so we should see some buffer reuse.
            if pp_size > 1:
                assert max_reuse > 1, "Expected some buffer reuse"

        # Verify that make_graphed_callables_kwargs contains expected keys
        assert (
            '_order' in make_graphed_callables_kwargs
        ), "make_graphed_callables_kwargs should contain '_order'"
        assert (
            'num_warmup_iters' in make_graphed_callables_kwargs
        ), "make_graphed_callables_kwargs should contain 'num_warmup_iters'"
        assert (
            'allow_unused_input' in make_graphed_callables_kwargs
        ), "make_graphed_callables_kwargs should contain 'allow_unused_input'"

        # Verify the order in kwargs matches expectations
        order = make_graphed_callables_kwargs['_order']
        num_model_chunks = cuda_graph_helper.num_model_chunks
        forward_count = sum(1 for chunk_id in order if chunk_id > 0)
        if pp_size > 1:
            # Verify that all forward passes in order have corresponding entries in sample_args
            assert forward_count == num_microbatches * num_model_chunks, (
                f"Forward count mismatch: expected {num_microbatches * num_model_chunks}, "
                f"got {forward_count}"
            )
            expected_order_length = num_microbatches * num_model_chunks * 2
        else:
            assert num_model_chunks == 1, "Expected only one model chunk for pp_size == 1"
            assert forward_count == 1, "Expected only one forward pass for pp_size == 1"
            expected_order_length = 2
        assert (
            len(order) == expected_order_length
        ), f"Order length mismatch: expected {expected_order_length}, got {len(order)}"


class TestRequiredNumMicrobatchSlots:
    """Pure-Python tests for ``_get_required_num_microbatch_slots_from_order``.

    The method derives the smallest cuda-graph slot count that guarantees no
    in-flight microbatch's static buffer is reused before its backward
    completes. ``order`` is a 1F1B / interleaved-1F1B schedule transcript
    where ``+chunk_id`` denotes a forward and ``-chunk_id`` a backward.
    Non-integer entries (e.g. ``0.5`` for wgrad sub-steps) are skipped.
    """

    @staticmethod
    def _slots(order, num_chunks):
        return TECudaGraphHelper._get_required_num_microbatch_slots_from_order(order, num_chunks)

    def test_single_chunk_single_microbatch(self):
        # F0 then B0: one slot is enough.
        assert self._slots([1, -1], 1) == 1

    def test_single_chunk_pp_pipeline_4_microbatches_pp2(self):
        # PP=2 1F1B with 4 microbatches: warmup F-F, then F-B-F-B-..., then cooldown B-B.
        # Max in-flight = 2.
        order = [1, 1, -1, 1, -1, 1, -1, -1]
        assert self._slots(order, 1) == 2

    def test_two_chunks_independent(self):
        # Two model chunks (VPP=2), each running a tiny PP=2-style 1F1B in turn.
        # Per chunk max in-flight = 2 -> 2 slots.
        order = [1, 1, -1, -1, 2, 2, -2, -2]
        assert self._slots(order, 2) == 2

    def test_two_chunks_interleaved(self):
        # Worst case: forwards stack up across chunks before any backward.
        # F0 F0 F1 F1 B1 B1 B0 B0 -> per-chunk max in-flight = 2.
        order = [1, 1, 2, 2, -2, -2, -1, -1]
        assert self._slots(order, 2) == 2

    def test_skips_non_integer_entries(self):
        # Float c_ids (e.g. 0.5 for wgrad sub-steps) must be ignored.
        order = [1, 0.5, -0.5, -1]
        assert self._slots(order, 1) == 1

    def test_minimum_slot_is_one(self):
        # Empty / no-op order still returns at least 1 (we always need a slot).
        assert self._slots([], 1) == 1

    def test_unbalanced_order_asserts(self):
        # Forward without matching backward -> outstanding != 0 at end -> assert.
        with pytest.raises(AssertionError):
            self._slots([1], 1)

    def test_negative_outstanding_asserts(self):
        # Backward before any forward for a chunk -> outstanding goes negative.
        with pytest.raises(AssertionError):
            self._slots([-1], 1)


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

    return HAVE_DEEP_EP


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


def test_dsa_topk_layers_the_mtp_reuses():
    """The decoder layers whose DSA top-k indices an MTP layer of the same model chunk reuses:
    the sources of the MTP's sharing layers that the MTP does not compute itself."""
    from megatron.core.transformer.cuda_graphs import _dsa_topk_layers_the_mtp_reuses

    def attention(layer_number, source_layer=None, index_share=True):
        module = torch.nn.Identity()
        module.index_share = index_share
        module.layer_number = layer_number
        module.skip_topk = source_layer is not None
        module.source_layer = source_layer if source_layer is not None else layer_number
        return module

    def chunk(*mtp_layers):
        model_chunk = torch.nn.Module()
        model_chunk.mtp = torch.nn.Sequential(*mtp_layers, torch.nn.Linear(2, 2))
        return model_chunk

    # Layer 4 reuses decoder layer 3.
    assert _dsa_topk_layers_the_mtp_reuses(chunk(attention(4, source_layer=3))) == (3,)
    # Layer 4 selects its own keys and layer 5 reuses those: nothing crosses the boundary.
    assert _dsa_topk_layers_the_mtp_reuses(chunk(attention(4), attention(5, source_layer=4))) == ()
    # Two MTP layers reusing two decoder layers.
    reused = _dsa_topk_layers_the_mtp_reuses(
        chunk(attention(5, source_layer=3), attention(6, source_layer=2))
    )
    assert reused == (2, 3)
    # No sharing (dsa_indexer_topk_freq == 1), no DSA layer, no MTP.
    assert _dsa_topk_layers_the_mtp_reuses(chunk(attention(4, index_share=False))) == ()
    assert _dsa_topk_layers_the_mtp_reuses(chunk()) == ()
    assert _dsa_topk_layers_the_mtp_reuses(torch.nn.Module()) == ()


class TestChunkWiseCudaGraphs:
    """Chunk-wise TE CUDA graphs (cuda_graph_impl='chunk_wise'): config validation and callable
    selection."""

    _PACKED = dict(
        sequence_packing_scheduler="dp_balanced",
        cuda_graph_dynamic_microbatches=True,
        pad_packed_seq_alignment="max",
        max_seqlen_per_dp_cp_rank=128,
        thd_max_packed_sequences=8,
    )
    _NO_DROPOUT = dict(hidden_dropout=0.0, attention_dropout=0.0)
    _FULL_RECOMPUTE = dict(
        recompute_granularity='full', recompute_method='uniform', recompute_num_layers=1
    )
    _MHC_RECOMPUTE = dict(
        enable_hyper_connections=True,
        recompute_granularity='selective',
        recompute_modules=['mhc'],
        mhc_recompute_layer_num=2,
    )
    _DROP_PADDING_MOE = dict(
        num_moe_experts=4,
        moe_token_dispatcher_type="alltoall",
        moe_expert_capacity_factor=1.0,
        moe_pad_expert_input_to_capacity=True,
    )

    def test_chunk_config_accepts_full_recompute(self):
        cfg = _base_cuda_graph_config(
            cuda_graph_impl='chunk_wise',
            cuda_graph_modules=[],
            recompute_granularity='full',
            recompute_method='uniform',
            recompute_num_layers=1,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        assert cfg.cuda_graph_impl == 'chunk_wise'

    def test_chunk_config_accepts_fine_grained_activation_offloading(self):
        cfg = _base_cuda_graph_config(
            cuda_graph_impl='chunk_wise',
            cuda_graph_modules=[],
            fine_grained_activation_offloading=True,
            offload_modules=['core_attn', 'attn_proj'],
        )
        assert cfg.cuda_graph_impl == 'chunk_wise'
        assert cfg.fine_grained_activation_offloading

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({'cuda_graph_modules': [CudaGraphModule.attn]}, "cuda_graph_modules must be empty"),
            (
                {'overlap_moe_expert_parallel_comm': True},
                "incompatible with overlap_moe_expert_parallel_comm and delay_wgrad_compute",
            ),
            (
                {'delay_wgrad_compute': True},
                "incompatible with overlap_moe_expert_parallel_comm and delay_wgrad_compute",
            ),
            # The third delayed-weight-gradient option: caught here, not by the capture.
            (
                {
                    'overlap_dispatch_backward_with_experts_wgrad': True,
                    'num_moe_experts': 4,
                    'moe_expert_capacity_factor': 1.0,
                    'moe_pad_expert_input_to_capacity': True,
                },
                "incompatible with overlap_dispatch_backward_with_experts_wgrad",
            ),
            # A host-side tracker that only the Python of the MoE forward / backward fills.
            (
                {
                    'log_moe_overload_factor': True,
                    'num_moe_experts': 4,
                    'moe_expert_capacity_factor': 1.0,
                    'moe_pad_expert_input_to_capacity': True,
                },
                "does not support log_moe_overload_factor",
            ),
            # A context the schedule opens around the forward step, outside the recorded block.
            ({'enable_autocast': True}, "does not support enable_autocast"),
            # Routes the router records into, and replays from, Python state of every forward.
            (
                {
                    'moe_enable_routing_replay': True,
                    'num_moe_experts': 4,
                    'moe_expert_capacity_factor': 1.0,
                    'moe_pad_expert_input_to_capacity': True,
                },
                "does not support moe_enable_routing_replay",
            ),
            # A router whose iteration count the host decides from the values of each batch.
            (
                {
                    'num_moe_experts': 4,
                    'moe_router_load_balancing_type': 'sinkhorn',
                    'moe_router_topk': 1,
                },
                "does not support sinkhorn routing",
            ),
            # Weight shards gathered on side streams that only the local and the
            # full-iteration graph runners join.
            (
                {'tensor_model_parallel_size': 2, 'tensor_parallel_num_weight_shards': 4},
                "does not support GTP weight rematerialization",
            ),
            (
                {**_PACKED, 'cuda_graph_dynamic_microbatches': False},
                "chunk CUDA graphs require cuda_graph_dynamic_microbatches",
            ),
            # Each source of randomness a captured full recompute cannot reproduce.
            (
                {**_FULL_RECOMPUTE, 'hidden_dropout': 0.1, 'attention_dropout': 0.0},
                "full activation recompute requires hidden_dropout=0",
            ),
            (
                {**_FULL_RECOMPUTE, 'hidden_dropout': 0.0, 'attention_dropout': 0.1},
                "full activation recompute requires hidden_dropout=0",
            ),
            (
                {**_FULL_RECOMPUTE, **_NO_DROPOUT, 'moe_input_jitter_eps': 1e-2},
                "full activation recompute requires hidden_dropout=0",
            ),
            # mHC recompute is captured with the block, like under full-iteration graphs.
            (
                {**_MHC_RECOMPUTE, 'hidden_dropout': 0.1, 'attention_dropout': 0.0},
                "mHC recompute with cuda_graph_impl='full_iteration' or chunk-wise",
            ),
            (
                {**_MHC_RECOMPUTE, 'hidden_dropout': 0.0, 'attention_dropout': 0.1},
                "mHC recompute with cuda_graph_impl='full_iteration' or chunk-wise",
            ),
            # The attention-only split is a per-layer capture.
            (
                {**_MHC_RECOMPUTE, **_NO_DROPOUT, 'mhc_recompute_attn_cuda_graph_split': True},
                "mhc_recompute_attn_cuda_graph_split requires cuda_graph_impl='transformer_engine'",
            ),
        ],
        ids=[
            "cuda_graph_modules",
            "overlap_moe_expert_parallel_comm",
            "delay_wgrad_compute",
            "overlap_dispatch_backward_with_experts_wgrad",
            "log_moe_overload_factor",
            "enable_autocast",
            "moe_enable_routing_replay",
            "sinkhorn_routing",
            "gtp_weight_shards",
            "packed_without_dynamic_microbatches",
            "full_recompute_hidden_dropout",
            "full_recompute_attention_dropout",
            "full_recompute_moe_input_jitter",
            "mhc_recompute_hidden_dropout",
            "mhc_recompute_attention_dropout",
            "mhc_recompute_attention_split",
        ],
    )
    def test_chunk_config_rejections(self, overrides, match):
        overrides = {'cuda_graph_modules': [], **overrides}
        with pytest.raises(ValueError, match=match):
            _base_cuda_graph_config(cuda_graph_impl='chunk_wise', **overrides)

    @pytest.mark.parametrize(
        "overrides, match",
        [
            (
                {'recompute_modules': ['core_attn'], 'attention_dropout': 0.1},
                "attention dropout is not supported with graphed attention recomputation",
            ),
            (
                {'recompute_modules': ['mlp'], 'hidden_dropout': 0.1},
                "hidden dropout is not supported with graphed MLP/MoE recomputation",
            ),
            (
                {'recompute_modules': ['moe'], **_DROP_PADDING_MOE, 'hidden_dropout': 0.1},
                "hidden dropout is not supported with graphed MLP/MoE recomputation",
            ),
            (
                {'recompute_modules': ['moe'], **_DROP_PADDING_MOE, 'moe_input_jitter_eps': 1e-2},
                "moe_input_jitter_eps is not supported with graphed moe recomputation",
            ),
        ],
        ids=["core_attn_dropout", "mlp_dropout", "moe_dropout", "moe_input_jitter"],
    )
    def test_chunk_config_rejects_randomness_in_a_captured_selective_recompute(
        self, overrides, match
    ):
        """A block graph records the recompute with the forward: a dropout mask or a router
        jitter drawn again in the recomputed forward would differ from the first one. These
        refusals are the per-layer graphs' (a whole layer in scope); a chunk captures it too."""
        settings = {**self._NO_DROPOUT, 'recompute_granularity': 'selective', **overrides}
        with pytest.raises(AssertionError, match=match):
            _base_cuda_graph_config(cuda_graph_impl='chunk_wise', cuda_graph_modules=[], **settings)

    def test_chunk_config_takes_fixed_length_slots_from_the_step(self):
        """Fixed-length batches: one slot per microbatch of the step, captured in the order
        the schedule runs them. Dynamic slots (the topology floor of the schedule, captured in
        the order of that larger count) are for packed sequences."""
        with pytest.warns(UserWarning, match="cuda_graph_dynamic_microbatches only applies"):
            fixed_length = _base_cuda_graph_config(
                cuda_graph_impl='chunk_wise',
                cuda_graph_modules=[],
                cuda_graph_dynamic_microbatches=True,
            )
        assert fixed_length.cuda_graph_dynamic_microbatches is False
        packed = _base_cuda_graph_config(
            cuda_graph_impl='chunk_wise', cuda_graph_modules=[], **self._PACKED
        )
        assert packed.cuda_graph_dynamic_microbatches is True

    def test_chunk_config_accepts_mhc_recompute_without_dropout(self):
        config = _base_cuda_graph_config(
            cuda_graph_impl='chunk_wise',
            cuda_graph_modules=[],
            **self._MHC_RECOMPUTE,
            **self._NO_DROPOUT,
        )
        assert config.cuda_graph_impl == 'chunk_wise'

    @pytest.mark.parametrize("cuda_graph_impl", ["chunk_wise", "transformer_engine"])
    def test_attention_split_flag_without_mhc_recompute_only_warns(self, cuda_graph_impl):
        """Without mHC recompute the split flag has no effect under any implementation."""
        with pytest.warns(UserWarning, match="mhc_recompute_attn_cuda_graph_split has no effect"):
            config = _base_cuda_graph_config(
                cuda_graph_impl=cuda_graph_impl,
                cuda_graph_modules=[],
                mhc_recompute_attn_cuda_graph_split=True,
            )
        assert config.mhc_recompute_attn_cuda_graph_split

    def test_chunk_config_rejects_dynamic_context_parallel(self, monkeypatch):
        """The captured block records the context-parallel communicator of the capture sample.
        Dynamic CP reaches graph validation only as packed DSA, where a per-layer graph that
        keeps attention eager is accepted with a warning."""
        from megatron.core.transformer.transformer_config import MLATransformerConfig

        monkeypatch.setattr(
            transformer_config_module, "_validate_dsa_kernel_backend_dependencies", lambda _: None
        )
        monkeypatch.setattr(transformer_config_module, "is_te_min_version", lambda _version: True)
        dynamic_cp_dsa = dict(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            experimental_attention_variant="dsa",
            dsa_kernel_backend="cudnn",
            sequence_packing_scheduler="default_dynamic_cp",
            dynamic_context_parallel=True,
            max_seqlen_per_dp_cp_rank=128,
            pad_packed_seq_alignment="max",
            thd_max_packed_sequences=4,
            context_parallel_size=2,
            cp_comm_type="allgather",
            calculate_per_token_loss=True,
        )
        with pytest.warns(UserWarning, match="packed DSA with context parallelism"):
            MLATransformerConfig(
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=[CudaGraphModule.mlp],
                **dynamic_cp_dsa,
            )
        with pytest.raises(ValueError, match="does not support dynamic_context_parallel"):
            MLATransformerConfig(
                cuda_graph_impl="chunk_wise",
                cuda_graph_modules=[],
                cuda_graph_dynamic_microbatches=True,
                **dynamic_cp_dsa,
            )

    @pytest.mark.parametrize(
        ("overrides", "warns"),
        [
            ({**_FULL_RECOMPUTE, 'moe_router_force_load_balancing': True}, True),
            (
                {
                    'recompute_granularity': 'selective',
                    'recompute_modules': ['moe'],
                    'moe_router_force_load_balancing': True,
                },
                True,
            ),
            (
                {
                    'recompute_granularity': 'selective',
                    'recompute_modules': ['core_attn'],
                    'moe_router_force_load_balancing': True,
                },
                False,
            ),
            ({'moe_router_force_load_balancing': True}, False),
            (_FULL_RECOMPUTE, False),
            ({**_FULL_RECOMPUTE, 'moe_router_force_biased': 0.5}, True),
        ],
        ids=[
            "full_recompute",
            "selective_moe_recompute",
            "selective_core_attn_recompute",
            "no_recompute",
            "full_recompute_without_force_load_balancing",
            "full_recompute_with_forced_bias",
        ],
    )
    def test_chunk_config_warns_when_recompute_redraws_forced_router_logits(self, overrides, warns):
        """moe_router_force_load_balancing and moe_router_force_biased draw random router
        logits; a recompute of the router (full recompute, or selective recompute of 'moe')
        draws them again."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _base_cuda_graph_config(
                cuda_graph_impl='chunk_wise',
                cuda_graph_modules=[],
                **self._NO_DROPOUT,
                **self._DROP_PADDING_MOE,
                **overrides,
            )
        redraw_warnings = [
            warning
            for warning in caught
            if "random router logits are drawn again in the recompute" in str(warning.message)
        ]
        assert len(redraw_warnings) == int(warns)

    def test_chunk_config_computes_the_lm_head_wgrad_in_a_captured_post_process(self):
        """With packed sequences the LM head is part of the captured post-process, so its
        weight gradient cannot be deferred to the schedule: the option is turned off with a
        warning. Fixed-length chunk graphs keep the LM head eager and the option untouched."""
        deferred = dict(
            cuda_graph_impl='chunk_wise',
            cuda_graph_modules=[],
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
            gradient_accumulation_fusion=True,
            defer_embedding_wgrad_compute=True,
        )
        with pytest.warns(UserWarning, match="defer_embedding_wgrad_compute is disabled"):
            packed = _base_cuda_graph_config(
                **deferred, **self._PACKED, pipeline_p2p_fixed_shape=True
            )
        assert packed.defer_embedding_wgrad_compute is False

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fixed_length = _base_cuda_graph_config(**deferred)
        assert fixed_length.defer_embedding_wgrad_compute is True
        assert not [w for w in caught if "defer_embedding_wgrad_compute" in str(w.message)]

    def test_chunk_config_records_the_mtp_backward_in_a_captured_post_process(self):
        """With packed sequences the MTP loss is part of the captured post-process: its
        backward is recorded in the backward graph, so the early backward (which the eager
        warm-up steps would run and a replay would not) is turned off with a warning.
        Fixed-length chunk graphs keep the MTP eager and the option untouched."""
        early = dict(
            cuda_graph_impl='chunk_wise',
            cuda_graph_modules=[],
            mtp_num_layers=1,
            mtp_loss_early_backward=True,
        )
        with pytest.warns(UserWarning, match="mtp_loss_early_backward is disabled"):
            packed = _base_cuda_graph_config(**early, **self._PACKED)
        assert packed.mtp_loss_early_backward is False

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fixed_length = _base_cuda_graph_config(**early)
        assert fixed_length.mtp_loss_early_backward is True
        assert not [w for w in caught if "mtp_loss_early_backward" in str(w.message)]

    @pytest.mark.parametrize("variant", ["gdn", "kda"])
    def test_chunk_config_rejects_gated_delta_net_with_packed_sequences(self, variant):
        """FLA builds the GDN / KDA chunk indices on the host from the cu_seqlens values and
        caches them per tensor, so a captured block graph would replay the capture sample's pack
        layout for every microbatch. Fixed-length chunk graphs with the same layers, and the
        per-layer TE graphs of the same packed configuration, are unaffected."""
        gated_delta_net = dict(
            experimental_attention_variant=variant,
            linear_attention_freq=1,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
        )
        packed = self._PACKED
        match = "does not support packed-sequence .* GDN / KDA"
        with pytest.raises(ValueError, match=match):
            _base_cuda_graph_config(
                cuda_graph_impl='chunk_wise', cuda_graph_modules=[], **gated_delta_net, **packed
            )
        # Also with context parallelism, where the message has to win over the generic
        # THD CP layout-conversion rejection of attention-capturing graphs.
        with pytest.raises(ValueError, match=match):
            _base_cuda_graph_config(
                cuda_graph_impl='chunk_wise',
                cuda_graph_modules=[],
                context_parallel_size=2,
                cp_partition_mode="contiguous",
                **gated_delta_net,
                **packed,
            )
        fixed_length = _base_cuda_graph_config(
            cuda_graph_impl='chunk_wise', cuda_graph_modules=[], **gated_delta_net
        )
        assert fixed_length.cuda_graph_impl == 'chunk_wise'
        per_layer = _base_cuda_graph_config(
            cuda_graph_impl='transformer_engine', cuda_graph_modules=[], **gated_delta_net, **packed
        )
        assert per_layer.cuda_graph_impl == 'transformer_engine'

    @pytest.mark.parametrize("pattern", ["G-*-", "*-K-", "*-/G-"])
    def test_chunk_config_rejects_hybrid_gated_delta_net_layers_with_packed_sequences(
        self, pattern
    ):
        """A hybrid stack selects GDN / KDA layers per pattern symbol and leaves
        experimental_attention_variant unset: the same rejection must key on the pattern (and
        the captured block re-checks its layers, _check_packed_layers_are_capturable)."""
        packed = self._PACKED
        with pytest.raises(ValueError, match="does not support packed-sequence .* GDN / KDA"):
            _base_cuda_graph_config(
                cuda_graph_impl='chunk_wise',
                cuda_graph_modules=[],
                is_hybrid_model=True,
                hybrid_layer_pattern=pattern,
                **packed,
            )
        fixed_length = _base_cuda_graph_config(
            cuda_graph_impl='chunk_wise',
            cuda_graph_modules=[],
            is_hybrid_model=True,
            hybrid_layer_pattern=pattern,
        )
        assert fixed_length.cuda_graph_impl == 'chunk_wise'
        attention_only = _base_cuda_graph_config(
            cuda_graph_impl='chunk_wise',
            cuda_graph_modules=[],
            is_hybrid_model=True,
            hybrid_layer_pattern="*-/*-",
            **packed,
        )
        assert attention_only.cuda_graph_impl == 'chunk_wise'

    @pytest.mark.parametrize("layer_family", ["gated_delta_net", "gated_delta_product"])
    def test_captured_block_rejects_gated_delta_net_layers_with_packed_sequences(
        self, layer_family
    ):
        """The block-level check sees the layers themselves, whatever selected them: a GDN /
        KDA layer of a hybrid pattern, or the gated delta product mixer a layer spec puts
        into a Mamba-symbol layer (no config field names it)."""
        from megatron.core.ssm.gated_delta_net.common import _GDNBase
        from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
        from megatron.core.transformer.chunk_cuda_graph import ChunkCudaGraphBlockMixin

        layer_class = _GDNBase if layer_family == "gated_delta_net" else GatedDeltaProductMixer

        class Block(ChunkCudaGraphBlockMixin, torch.nn.Module):
            config = SimpleNamespace(cuda_graph_impl="none")

        block = Block()
        block.attention = torch.nn.Identity()
        block._check_packed_layers_are_capturable()
        block.linear_attention = layer_class.__new__(layer_class)
        torch.nn.Module.__init__(block.linear_attention)
        with pytest.raises(ValueError, match="does not support packed-sequence .* GDN / KDA"):
            block._check_packed_layers_are_capturable()

    _STATIC_HYBRIDEP = dict(
        num_moe_experts=4,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_expert_rank_capacity_factor=1.2,
        # A HybridEP rank capacity needs the op fuser (or grouped tensors).
        use_transformer_engine_op_fuser=True,
    )

    @pytest.mark.parametrize(
        "moe",
        [
            # Dropless all-to-all: token counts per expert vary with the routing.
            dict(num_moe_experts=4, moe_token_dispatcher_type="alltoall"),
            # HybridEP without a rank capacity.
            dict(
                num_moe_experts=4,
                moe_token_dispatcher_type="flex",
                moe_flex_dispatcher_backend="hybridep",
            ),
        ],
    )
    def test_chunk_config_requires_static_moe_token_shapes(self, moe):
        """The captured block includes the MoE dispatch. The requirement is checked when the
        config is built, not only when the graphs are captured after the warm-up steps."""
        with pytest.raises(AssertionError, match="need static token shapes"):
            _base_cuda_graph_config(cuda_graph_impl='chunk_wise', cuda_graph_modules=[], **moe)
        # The same MoE without graphs is fine.
        _base_cuda_graph_config(cuda_graph_impl='none', **moe)

    @pytest.mark.parametrize(
        "moe",
        [
            _STATIC_HYBRIDEP,
            dict(
                num_moe_experts=4,
                moe_token_dispatcher_type="alltoall",
                moe_expert_capacity_factor=1.0,
                moe_pad_expert_input_to_capacity=True,
            ),
        ],
    )
    def test_chunk_config_accepts_static_moe_token_shapes(self, moe):
        config = _base_cuda_graph_config(cuda_graph_impl='chunk_wise', cuda_graph_modules=[], **moe)
        assert config.cuda_graph_impl == 'chunk_wise'

    def test_chunk_config_turns_off_hybridep_runtime_padding(self):
        """Every dispatcher rank holds the same static token count under chunk graphs, and
        HybridEP's runtime padding (an all-reduce plus a host read per MoE layer) cannot be
        captured: packed chunk graphs do not need the flag, and a set flag is turned off."""
        packed = self._PACKED
        config = _base_cuda_graph_config(
            cuda_graph_impl='chunk_wise', cuda_graph_modules=[], **self._STATIC_HYBRIDEP, **packed
        )
        assert config.moe_hybridep_pad_variable_tokens is False
        with pytest.warns(UserWarning, match="moe_hybridep_pad_variable_tokens"):
            config = _base_cuda_graph_config(
                cuda_graph_impl='chunk_wise',
                cuda_graph_modules=[],
                moe_hybridep_pad_variable_tokens=True,
                **self._STATIC_HYBRIDEP,
                **packed,
            )
        assert config.moe_hybridep_pad_variable_tokens is False
        assert config.moe_hybridep_pad_uneven_dispatch_inputs is False
        # Eager packed HybridEP still needs the runtime padding.
        eager_packed = dict(sequence_packing_scheduler="dp_balanced", max_seqlen_per_dp_cp_rank=128)
        with pytest.raises(ValueError, match="moe_hybridep_pad_variable_tokens=True"):
            _base_cuda_graph_config(cuda_graph_impl='none', **self._STATIC_HYBRIDEP, **eager_packed)

    def test_chunk_config_has_no_inference_graph_scope(self):
        config = _base_cuda_graph_config(cuda_graph_impl='chunk_wise', cuda_graph_modules=[])
        assert config.inference_cuda_graph_scope == InferenceCudaGraphScope.none
        with pytest.raises(AssertionError, match="Invalid inference CUDA graph scope"):
            _base_cuda_graph_config(
                cuda_graph_impl='chunk_wise',
                cuda_graph_modules=[],
                inference_cuda_graph_scope=InferenceCudaGraphScope.layer,
            )

    @pytest.mark.parametrize(
        ("cuda_graph_impl", "cuda_graph_modules", "expected"),
        [
            ("chunk_wise", [], True),
            ("transformer_engine", [CudaGraphModule.attn], True),
            ("transformer_engine", [CudaGraphModule.mlp], False),
            ("full_iteration", [], False),
            ("none", [], False),
        ],
    )
    def test_graph_dynamic_dsa_pack_routing_follows_te_graphs_that_capture_attention(
        self, cuda_graph_impl, cuda_graph_modules, expected
    ):
        """The balanced DSA indexer routes varying packs through fixed-capacity graph inputs
        under per-layer TE graphs that capture attention and under chunk-wise graphs."""
        config = SimpleNamespace(
            dsa_cp_balance_indexer=True,
            cuda_graph_impl=cuda_graph_impl,
            cuda_graph_modules=cuda_graph_modules,
            sequence_packing_scheduler="dp_balanced",
        )
        derived = TransformerConfig.dsa_cp_balance_indexer_graph_dynamic_packs.fget(config)
        assert derived is expected

    def test_chunk_uses_outer_block_callable(self):
        config = SimpleNamespace(cuda_graph_impl='chunk_wise')
        for block_type in (TransformerBlock, HybridStack):
            block = object.__new__(block_type)
            object.__setattr__(block, 'config', config)
            assert isinstance(block, GraphableMegatronModule)
            assert _chunk_cuda_graph_block_is_graphable(block, config)
            assert not block._should_call_local_cudagraph()
            # Under per-layer TE graphs the layers are the callables, never their block.
            per_layer = SimpleNamespace(cuda_graph_impl='transformer_engine', cuda_graph_modules=[])
            object.__setattr__(block, 'config', per_layer)
            assert not _te_cuda_graph_layer_is_graphable(block, per_layer)
            assert not _chunk_cuda_graph_block_is_graphable(block, per_layer)
            assert not block._should_call_te_cudagraph()
        layer = object.__new__(TransformerLayer)
        object.__setattr__(layer, 'config', config)
        assert not _chunk_cuda_graph_block_is_graphable(layer, config)
        assert not _te_cuda_graph_layer_is_graphable(layer, config)

    def test_chunk_config_warns_when_packed_pp_keeps_the_p2p_shape_exchange(self):
        """Packing pads every microbatch to max_seqlen_per_dp_cp_rank, so the pipeline P2P shape
        is static; without pipeline_p2p_fixed_shape every P2P still exchanges shapes (a host
        synchronization)."""
        packed_pp = dict(
            cuda_graph_impl='chunk_wise',
            cuda_graph_modules=[],
            cuda_graph_dynamic_microbatches=True,
            sequence_packing_scheduler="dp_balanced",
            pad_packed_seq_alignment="max",
            max_seqlen_per_dp_cp_rank=128,
            thd_max_packed_sequences=8,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
        )
        with pytest.warns(UserWarning, match="pipeline_p2p_fixed_shape=True"):
            _base_cuda_graph_config(**packed_pp)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _base_cuda_graph_config(**packed_pp, pipeline_p2p_fixed_shape=True)
            _base_cuda_graph_config(**{**packed_pp, "pipeline_model_parallel_size": 1})
        assert not [w for w in caught if "pipeline_p2p_fixed_shape" in str(w.message)]

    def test_te_capture_runs_on_the_shared_capture_stream_for_pool_sharing(self, monkeypatch):
        """The chunk graphs and the optimizer-step graph share one memory pool, whose free blocks
        are keyed by the allocating stream: TE's capture (torch.cuda.graph without a stream) must
        therefore run on get_shared_capture_stream(). The context redirects torch's default
        capture stream for the capture only (other graphs of the process keep theirs, also when
        the capture raises); the full-iteration path keeps passing the stream explicitly."""
        from megatron.core import full_cuda_graph

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        monkeypatch.setattr(full_cuda_graph, "_shared_capture_stream", None)
        previous_stream = torch.cuda.Stream()
        monkeypatch.setattr(torch.cuda.graphs.graph, "default_capture_stream", previous_stream)
        with pytest.raises(RuntimeError, match="capture failed"):
            with full_cuda_graph.capture_te_graphs_on_shared_stream():
                shared_stream = full_cuda_graph.get_shared_capture_stream()
                assert shared_stream is not previous_stream
                assert torch.cuda.graphs.graph.default_capture_stream is shared_stream
                # What TE's make_graphed_callables builds: a graph context without a stream.
                assert torch.cuda.graph(torch.cuda.CUDAGraph()).capture_stream is shared_stream
                raise RuntimeError("capture failed")
        assert torch.cuda.graphs.graph.default_capture_stream is previous_stream

    @staticmethod
    def _capture_input_helper(monkeypatch, callables, **config_overrides):
        """A PP=1 helper whose _get_cuda_graph_input_data() runs without a model: the tests read
        the keyword arguments it builds for Transformer Engine's make_graphed_callables."""

        def fake_make_graphed_callables(modules, sample_args, pool=None, **kwargs):
            raise AssertionError("only the keyword arguments are under test")

        config = SimpleNamespace(
            cuda_graph_impl="chunk_wise",
            cuda_graph_modules=[],
            cuda_graph_warmup_steps=1,
            cuda_graph_use_single_mempool=True,
            cuda_graph_retain_backward_graph=False,
            microbatch_group_size_per_vp_stage=1,
            overlap_moe_expert_parallel_comm=False,
            delay_wgrad_compute=False,
            fine_grained_activation_offloading=False,
            fp8=None,
            fp4=None,
        )
        for name, value in config_overrides.items():
            setattr(config, name, value)
        pp_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
        helper = object.__new__(TECudaGraphHelper)
        helper.config = config
        helper.pp_group = pp_group
        helper.p2p_communicator = SimpleNamespace(
            pp_group=pp_group, virtual_pipeline_model_parallel_size=None
        )
        helper.pg_collection = None
        helper.tp_group = helper.dp_cp_group = None
        helper.num_model_chunks = 1
        helper.flattened_callables = list(callables)
        helper.num_layers_per_chunk = [len(callables)]
        helper._get_sample_arguments = lambda order, chunk_id_list=None: ([()] * len(callables), [])
        # The helper imports make_graphed_callables from Transformer Engine at call time.
        monkeypatch.setattr(
            "transformer_engine.pytorch.graph.make_graphed_callables", fake_make_graphed_callables
        )
        monkeypatch.setattr(cuda_graphs_module, "get_num_microbatches", lambda: 1)
        monkeypatch.setattr(cuda_graphs_module, "log_on_each_pipeline_stage", lambda **kwargs: None)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda *args: 0)
        return helper

    @staticmethod
    def _create_cudagraphs_helper(capture_kwargs, **config_overrides):
        """A helper whose create_cudagraphs() reaches make_graphed_callables with stand-in
        sample inputs (the caller replaces make_graphed_callables)."""
        helper = object.__new__(TECudaGraphHelper)
        helper.config = _base_cuda_graph_config(
            cuda_graph_impl='chunk_wise', cuda_graph_modules=[], **config_overrides
        )
        helper.model = []
        helper.flattened_callables = [torch.nn.Identity()]
        helper.callables_per_chunk = []
        helper.num_microbatches = 1
        helper._captures_dsa_metric_writes = False
        helper._start_capturing = lambda: 0.0
        helper._finish_capturing = lambda _start_time: None
        helper._get_cuda_graph_input_data = lambda: ([()], {'_order': [1, -1], **capture_kwargs})
        helper._validate_mhc_static_hidden_inputs = lambda _sample_args: None
        helper._uses_mhc_direct_write_arena = lambda: False
        return helper

    @pytest.mark.parametrize(
        ("config_overrides", "te_takes_pool", "shares_pool"),
        [
            (dict(), True, True),
            (dict(cuda_graph_warmup_steps=0), True, True),
            (dict(cuda_graph_use_single_mempool=False), True, False),
            (dict(), False, False),
            (dict(cuda_graph_impl="transformer_engine"), True, False),
        ],
        ids=[
            "shared",
            "no_eager_warmup_step",
            "single_mempool_off",
            "te_without_pool_argument",
            "per_layer_graphs",
        ],
    )
    def test_chunk_capture_shares_the_graph_pool_with_the_optimizer_graph(
        self, monkeypatch, config_overrides, te_takes_pool, shares_pool
    ):
        """The chunk graphs share one pool with the optimizer-step graph, which reuses the
        blocks they leave idle, whenever cuda_graph_use_single_mempool is on and Transformer
        Engine takes a pool: also without an eager warm-up step, because the optimizer's first
        step (which allocates its state) is never the captured one. A capture round never
        reuses the handle of an earlier round, whose graphs may be gone."""
        from megatron.core import full_cuda_graph

        handles = []

        def new_pool_handle():
            handles.append(object())
            return handles[-1]

        earlier_round = object()
        monkeypatch.setattr(torch.cuda, "graph_pool_handle", new_pool_handle)
        monkeypatch.setattr(full_cuda_graph, "_shared_graph_pool", earlier_round)
        helper = self._capture_input_helper(monkeypatch, [object()], **config_overrides)
        if not te_takes_pool:
            monkeypatch.setattr(
                "transformer_engine.pytorch.graph.make_graphed_callables",
                lambda modules, sample_args, **kwargs: None,
            )

        _, kwargs = helper._get_cuda_graph_input_data()

        if not shares_pool:
            assert 'pool' not in kwargs
            assert handles == []
            assert full_cuda_graph._shared_graph_pool is earlier_round
            return
        assert kwargs['pool'] is not earlier_round
        assert kwargs['pool'] is handles[0]
        # The optimizer-step graph asks for the same handle.
        assert full_cuda_graph.get_graph_pool(True) is kwargs['pool']
        # The next round, also when these graphs were never deleted, gets its own.
        _, next_round_kwargs = helper._get_cuda_graph_input_data()
        assert next_round_kwargs['pool'] is handles[1]
        assert full_cuda_graph.get_graph_pool(True) is handles[1]

    @pytest.mark.parametrize("shares_pool", [True, False])
    @pytest.mark.parametrize("capture_fails", [False, True])
    def test_chunk_capture_runs_on_the_shared_stream_only_for_a_shared_pool(
        self, monkeypatch, shares_pool, capture_fails
    ):
        """create_cudagraphs points torch's default capture stream at the shared capture stream
        exactly while make_graphed_callables captures into the shared pool, and puts the
        previous stream back, also when the capture raises."""
        from megatron.core import full_cuda_graph

        shared_stream, previous_stream, pool = object(), object(), object()
        monkeypatch.setattr(full_cuda_graph, "_shared_capture_stream", shared_stream)
        monkeypatch.setattr(torch.cuda.graphs.graph, "default_capture_stream", previous_stream)
        helper = self._create_cudagraphs_helper({'pool': pool} if shares_pool else {})
        seen = []

        def fake_make_graphed_callables(modules, sample_args, **kwargs):
            seen.append((torch.cuda.graphs.graph.default_capture_stream, kwargs.get('pool')))
            if capture_fails:
                raise RuntimeError("capture failed")
            return ()

        # The helper imports make_graphed_callables from Transformer Engine at call time.
        monkeypatch.setattr(
            "transformer_engine.pytorch.graph.make_graphed_callables", fake_make_graphed_callables
        )

        if capture_fails:
            with pytest.raises(RuntimeError, match="capture failed"):
                helper.create_cudagraphs()
        else:
            helper.create_cudagraphs()
            assert helper.graphs_created()

        if shares_pool:
            assert seen == [(shared_stream, pool)]
        else:
            assert seen == [(previous_stream, None)]
        assert torch.cuda.graphs.graph.default_capture_stream is previous_stream

    @pytest.mark.parametrize("offloading", [True, False])
    @pytest.mark.parametrize("capture_fails", [False, True])
    def test_chunk_capture_tells_the_offload_manager_whether_the_graphs_exist(
        self, monkeypatch, offloading, capture_fails
    ):
        """After a successful capture the graphs carry the offloads and the manager stops
        serving eager forwards; a capture that raised must leave the eager policy in charge."""
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )

        calls = []
        monkeypatch.setattr(
            off_interface,
            "enter_block_capture",
            staticmethod(lambda keep_last_group=False: calls.append(("enter", keep_last_group))),
        )
        monkeypatch.setattr(
            off_interface,
            "exit_block_capture",
            staticmethod(lambda captured=True: calls.append(("exit", captured))),
        )
        offload = (
            dict(fine_grained_activation_offloading=True, offload_modules=['core_attn'])
            if offloading
            else {}
        )
        helper = self._create_cudagraphs_helper({}, **offload)

        def fake_make_graphed_callables(modules, sample_args, **kwargs):
            if capture_fails:
                raise RuntimeError("capture failed")
            return ()

        # The helper imports make_graphed_callables from Transformer Engine at call time.
        monkeypatch.setattr(
            "transformer_engine.pytorch.graph.make_graphed_callables", fake_make_graphed_callables
        )

        if capture_fails:
            with pytest.raises(RuntimeError, match="capture failed"):
                helper.create_cudagraphs()
        else:
            helper.create_cudagraphs()

        if offloading:
            keep_last_group = helper.config.fine_grained_offloading_graph_keep_last_group
            assert calls == [("enter", keep_last_group), ("exit", not capture_fails)]
        else:
            assert calls == []

    @pytest.mark.parametrize("failure", ["sample_inputs", "capture"])
    def test_capture_that_raises_ends_the_capture_state(self, monkeypatch, failure):
        """A capture that is refused while its sample inputs are built (a block that cannot be
        captured), or that raises in Transformer Engine, is over: later eager forwards must not
        take the capture path."""
        helper = self._create_cudagraphs_helper({})

        def start_capturing():
            cuda_graphs_module._set_capture_start()
            return 0.0

        def refuse_sample_inputs():
            raise ValueError("cannot be captured")

        def fail_capture(modules, sample_args, **kwargs):
            raise RuntimeError("capture failed")

        helper._start_capturing = start_capturing
        if failure == "sample_inputs":
            helper._get_cuda_graph_input_data = refuse_sample_inputs
        # create_cudagraphs() imports make_graphed_callables from Transformer Engine at call time.
        monkeypatch.setattr("transformer_engine.pytorch.graph.make_graphed_callables", fail_capture)

        try:
            with pytest.raises((ValueError, RuntimeError), match="captured|capture failed"):
                helper.create_cudagraphs()
            assert not cuda_graphs_module.is_graph_capturing()
        finally:
            cuda_graphs_module._set_capture_end()

    @pytest.mark.parametrize("cuda_graph_impl", ["chunk_wise", "transformer_engine"])
    def test_delete_cuda_graphs_releases_what_the_chunk_graphs_held(
        self, monkeypatch, cuda_graph_impl
    ):
        """Deleting the chunk graphs forgets the shared pool handle (a later round must not
        capture into the pool of destroyed graphs) and hands the offloads back to the eager
        forwards. Per-layer graphs own neither."""
        from megatron.core import full_cuda_graph
        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )

        released = []
        monkeypatch.setattr(
            off_interface, "release_block_capture", staticmethod(lambda: released.append(True))
        )
        pool = object()
        monkeypatch.setattr(full_cuda_graph, "_shared_graph_pool", pool)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        monkeypatch.setattr(cuda_graphs_module, "log_on_each_pipeline_stage", lambda **kwargs: None)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda *args: 0)
        helper = object.__new__(TECudaGraphHelper)
        helper.config = SimpleNamespace(
            cuda_graph_impl=cuda_graph_impl, fine_grained_activation_offloading=True
        )
        helper.callables_per_chunk = []
        helper.chunks_with_decoder = []
        helper.tp_group = helper.dp_cp_group = None
        helper._graphs_created = True

        helper.delete_cuda_graphs()

        assert not helper.graphs_created()
        if cuda_graph_impl == "chunk_wise":
            assert full_cuda_graph._shared_graph_pool is None
            assert released == [True]
        else:
            assert full_cuda_graph._shared_graph_pool is pool
            assert released == []

    def test_only_the_gpt_post_process_stays_eager_for_the_config_logger(self):
        """GPTModel dumps the inputs and logits of every forward from its post-process;
        HybridModel dumps nothing per forward, so its post-process is still captured."""
        from megatron.core.models.hybrid.hybrid_model import HybridPostProcessBlock

        config = SimpleNamespace(cuda_graph_impl='chunk_wise', config_logger_dir="/config/logger")
        hybrid_block = HybridPostProcessBlock.__new__(HybridPostProcessBlock)
        torch.nn.Module.__init__(hybrid_block)
        hybrid_block.config = config
        hybrid_block.output_layer = torch.nn.Linear(2, 2)

        assert hybrid_block.eager_only_reason() is None
        assert "config_logger_dir" in self._postprocess_block(config).eager_only_reason()

    @staticmethod
    def _postprocess_block(config, *, mtp_process=False):
        """A post-process chunk callable with only the state its capture predicates read."""
        from megatron.core.models.gpt.gpt_model import GPTPostProcessBlock

        block = GPTPostProcessBlock.__new__(GPTPostProcessBlock)
        torch.nn.Module.__init__(block)
        block.config = config
        block.output_layer = torch.nn.Linear(2, 2)
        object.__setattr__(block, '_owner', SimpleNamespace(mtp_process=mtp_process))
        return block

    @pytest.mark.parametrize(
        ("scenario", "eager_reason"),
        [
            ("capturable", None),
            ("lm_head_forward_hook", "forward hooks are registered on the output layer"),
            ("config_logger", "config_logger_dir dumps the inputs and logits of every forward"),
            ("no_decoder_layer", "its pipeline stage owns no decoder layer"),
        ],
        ids=["capturable", "lm_head_forward_hook", "config_logger", "no_decoder_layer"],
    )
    def test_chunk_discovery_keeps_a_post_process_python_must_observe_eager(
        self, monkeypatch, scenario, eager_reason
    ):
        """A replay runs no Python: forward hooks on the LM head (logits savers, distillation
        captures) and the config logger's per-forward dump need the eager post-process, and a
        stage without decoder layers has no decoder graph for it to follow. The post-process is
        then not a callable and the reason is logged."""
        config = SimpleNamespace(
            cuda_graph_impl='chunk_wise',
            config_logger_dir="/config/logger" if scenario == "config_logger" else "",
        )
        decoder = TransformerBlock.__new__(TransformerBlock)
        torch.nn.Module.__init__(decoder)
        decoder.config = config
        decoder.layers = torch.nn.ModuleList(
            [] if scenario == "no_decoder_layer" else [torch.nn.Identity(), torch.nn.Identity()]
        )
        postprocess_block = self._postprocess_block(config)
        if scenario == "lm_head_forward_hook":
            postprocess_block.output_layer.register_forward_hook(lambda *_args: None)
        model_chunk = SimpleNamespace(decoder=decoder, postprocess_block=postprocess_block)
        # The post-process lives on the last pipeline stage, so the reason is logged there
        # (log_on_each_pipeline_stage), not on global rank 0 only.
        records = []
        monkeypatch.setattr(
            cuda_graphs_module,
            "log_on_each_pipeline_stage",
            lambda **kwargs: (
                records.append((kwargs["level"], kwargs["msg"]))
                if kwargs["level"] >= logging.WARNING
                else None
            ),
        )
        monkeypatch.setattr(
            cuda_graphs_module,
            "log_single_rank",
            lambda *args, **kwargs: pytest.fail("the eager reason must reach every stage"),
        )
        monkeypatch.setattr(torch.distributed, "get_rank", lambda *args: 0)
        helper = object.__new__(TECudaGraphHelper)
        helper.model = [model_chunk]
        helper.config = config
        helper.tp_group = helper.dp_cp_group = None

        helper._discover_layers()

        if eager_reason is None:
            assert helper.callables_per_chunk == [[decoder, postprocess_block]]
            assert helper.flattened_callables == [decoder, postprocess_block]
            assert helper.callables_per_chunk_is_mtp == [[False, False]]
            assert records == []
            return
        if scenario == "no_decoder_layer":
            assert helper.callables_per_chunk == [[]]
            assert helper.flattened_callables == []
            assert helper.chunks_with_decoder == [None]
        else:
            assert helper.callables_per_chunk == [[decoder]]
            assert helper.flattened_callables == [decoder]
            assert helper.chunks_with_decoder == [model_chunk]
        assert len(records) == 1
        level, message = records[0]
        assert level == logging.WARNING
        assert "runs eagerly because " + eager_reason in message

    @pytest.mark.parametrize(
        "scenario", ["freeze_base_model_for_mtp", "forward_only", "frozen_parameters_only"]
    )
    def test_chunk_discovery_keeps_a_decoder_that_runs_without_autograd_eager(
        self, monkeypatch, scenario
    ):
        """``freeze_base_model_for_mtp`` (the model wraps the decoder call) and
        ``freeze_all_layers`` (the training loop runs the step without autograd and builds the
        helper with ``forward_only``) run the decoder under torch.no_grad() and keep none of
        its activations. A block graph would record them, with a backward nothing replays, so
        the block and the post-process that follows it stay eager. Frozen parameters alone do
        not decide: a pipeline stage that holds only frozen layers still passes gradients to
        whatever trains upstream of it, and is captured."""
        config = SimpleNamespace(
            cuda_graph_impl='chunk_wise',
            config_logger_dir="",
            freeze_base_model_for_mtp=scenario == "freeze_base_model_for_mtp",
        )
        decoder = TransformerBlock.__new__(TransformerBlock)
        torch.nn.Module.__init__(decoder)
        decoder.config = config
        decoder.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
        model_chunk = torch.nn.Module()
        model_chunk.decoder = decoder
        model_chunk.postprocess_block = postprocess_block = self._postprocess_block(config)
        if scenario == "frozen_parameters_only":
            model_chunk.requires_grad_(False)
        messages = []
        monkeypatch.setattr(
            cuda_graphs_module,
            "log_on_each_pipeline_stage",
            lambda **kwargs: (
                messages.append(kwargs["msg"]) if kwargs["level"] >= logging.WARNING else None
            ),
        )
        monkeypatch.setattr(torch.distributed, "get_rank", lambda *args: 0)
        helper = object.__new__(TECudaGraphHelper)
        helper.model = [model_chunk]
        helper.config = config
        helper.forward_only = scenario == "forward_only"
        helper.tp_group = helper.dp_cp_group = None

        helper._discover_layers()

        if scenario == "frozen_parameters_only":
            assert helper.flattened_callables == [decoder, postprocess_block]
            assert messages == []
            return
        assert helper.flattened_callables == []
        assert len(messages) == 2
        assert "the decoder block runs eagerly because the training forward runs it without" in (
            messages[0]
        )
        assert "runs eagerly because the decoder block of its stage is not captured" in messages[1]

    @pytest.mark.parametrize(
        ("postprocess_config", "mtp_process", "postprocess_quantized"),
        [
            (dict(mtp_num_layers=1, keep_mtp_in_bf16=False), True, True),
            (dict(mtp_num_layers=1, keep_mtp_in_bf16=True), True, False),
            (dict(mtp_num_layers=1, keep_mtp_in_bf16=False), False, False),
            (dict(mtp_num_layers=None, keep_mtp_in_bf16=False), False, False),
            (dict(mtp_num_layers=1, keep_mtp_in_bf16=True, fp8_output_proj=True), True, True),
        ],
        ids=[
            "quantized_mtp",
            "mtp_kept_in_bf16",
            "mtp_on_another_stage",
            "no_mtp",
            "mxfp8_output_projection",
        ],
    )
    def test_chunk_capture_autocast_follows_the_quantized_modules_of_each_callable(
        self, monkeypatch, postprocess_config, mtp_process, postprocess_quantized
    ):
        """The capture opens Transformer Engine's FP8 autocast per callable. The decoder block
        always holds quantized modules; the post-process block only with MTP layers that are
        not kept in BF16 or with the MXFP8 output projection, exactly as its eager forward."""
        from megatron.core.transformer.chunk_cuda_graph import ChunkCudaGraphBlockMixin

        class DecoderBlock(ChunkCudaGraphBlockMixin, torch.nn.Module):
            config = SimpleNamespace(cuda_graph_impl="none")

        fp8_config = dict(fp8="hybrid", fp8_recipe="mxfp8", fp8_output_proj=False)
        fp8_config.update(postprocess_config)
        postprocess_block = self._postprocess_block(
            SimpleNamespace(**fp8_config), mtp_process=mtp_process
        )
        helper = self._capture_input_helper(
            monkeypatch, [DecoderBlock(), postprocess_block], **fp8_config
        )
        monkeypatch.setattr("megatron.core.fp8_utils.get_fp8_recipe", lambda config: "recipe")

        _, kwargs = helper._get_cuda_graph_input_data()

        assert kwargs['fp8_enabled'] == (True, postprocess_quantized)
        assert kwargs['fp8_recipe'] == "recipe"

    @pytest.mark.parametrize(
        ("capacity_field", "tail_padding_policy", "real_documents_per_pack"),
        [
            ("cuda_graph_thd_max_packed_sequences", "extend_last", 3),
            ("cuda_graph_thd_max_packed_sequences", "append_dummy_seq", 3),
            ("thd_max_packed_sequences", "extend_last", 4),
            ("thd_max_packed_sequences", "append_dummy_seq", 3),
        ],
    )
    def test_thd_capture_slot_bound_covers_what_the_packing_scheduler_produces(
        self, monkeypatch, capacity_field, tail_padding_policy, real_documents_per_pack
    ):
        """The packed-microbatch bound of the capture (one of the three terms the slot count is
        the maximum of) against the packs the scheduler really makes. The dp_balanced
        scheduler caps the real documents per pack below the static THD capacity:
        cuda_graph_thd_max_packed_sequences includes the dummy padding document whenever
        padding is enabled, thd_max_packed_sequences only under the append_dummy_seq tail
        policy. The bound has to be derived from the cap the scheduler really applies."""
        from megatron.core.datasets import data_schedule

        global_batch_size, seq_length, capacity = 48, 8, 4
        config = SimpleNamespace(
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=8 * seq_length,
            pad_packed_seq_alignment="max",
            thd_tail_padding_policy=tail_padding_policy,
            thd_max_packed_sequences=None,
            cuda_graph_thd_max_packed_sequences=None,
            virtual_pipeline_model_parallel_size=None,
            microbatch_group_size_per_vp_stage=None,
        )
        setattr(config, capacity_field, capacity)

        def group():
            return SimpleNamespace(size=lambda: 1)

        # The scheduler exactly as wrap_data_iterator configures it for this config.
        schedulers = []

        class RecordingScheduler(data_schedule.DpBalancedScheduler):
            def run(self, *args, **kwargs):
                schedulers.append(self)
                return None, 0, 0, 0

        monkeypatch.setitem(
            data_schedule.scheduler_map,
            data_schedule.PackingSchedulerEnum.DP_BALANCED,
            RecordingScheduler,
        )
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        data_schedule.wrap_data_iterator(
            None,
            config,
            1,
            SimpleNamespace(
                dp_cp_gtp_remat=None,
                dp_cp=group(),
                dp_gtp_remat=None,
                dp=group(),
                tp=group(),
                pp=group(),
                dynamic_cp=None,
            ),
        )
        (scheduler,) = schedulers
        assert scheduler.max_num_seqs == real_documents_per_pack
        # A step of short documents: only the document cap limits a pack.
        packs = scheduler.get_groups_and_subsamples(
            [(sample_id, 1) for sample_id in range(global_batch_size)]
        )
        assert len(packs) == math.ceil(global_batch_size / real_documents_per_pack)

        helper = object.__new__(TECudaGraphHelper)
        helper.config = config
        helper.dp_group = group()
        helper.dp_cp_group = group()
        helper.micro_batch_size = 1
        helper.seq_length = seq_length
        helper.thd_sequence_length_upper_bound = None

        bound, mode = helper._get_thd_varlen_max_num_microbatches(global_batch_size, 1)

        assert mode == "thd_varlen_upper_bound"
        # Short documents are the worst case for the pack count, so the bound is exact here.
        assert bound == len(packs)

    @staticmethod
    def _fixed_length_gpt(cuda_graph_impl, **config_overrides):
        """A small fixed-length GPT; under chunk_wise its decoder block is the graph callable
        and the LM head and loss stay eager."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            create_attention_mask_in_dataloader=False,
            use_te_rng_tracker=True,
            cuda_graph_impl=cuda_graph_impl,
            cuda_graph_modules=[],
            **config_overrides,
        )
        model_parallel_cuda_manual_seed(123)
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=32,
            position_embedding_type="rope",
        ).cuda()
        model.train()
        # The capture resets gradient buffers through the DDP wrapper; the test owns the grads.
        model.zero_grad_buffer = lambda: None
        return model

    @staticmethod
    @contextmanager
    def _real_capture(monkeypatch, micro_batch_size):
        """Process state for a real chunk capture on the training stream; yields the
        ``(pool, default capture stream)`` every make_graphed_callables call saw, the capture
        stream that was the default before, and a list of helpers to delete on exit."""
        from megatron.core import full_cuda_graph
        from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper

        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        init_num_microbatches_calculator(
            rank=0,
            rampup_batch_size=None,
            global_batch_size=micro_batch_size,
            micro_batch_size=micro_batch_size,
            data_parallel_size=1,
        )
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        previous_stream = torch.cuda.Stream()
        monkeypatch.setattr(torch.cuda.graphs.graph, "default_capture_stream", previous_stream)
        monkeypatch.setattr(full_cuda_graph, "_shared_graph_pool", None)
        monkeypatch.setattr(full_cuda_graph, "_shared_capture_stream", None)
        for name, value in (("curr_iteration", 0), ("cuda_graph", None), ("result", None)):
            monkeypatch.setattr(OptimizerCudaGraphWrapper, name, value)

        # The helper imports make_graphed_callables from Transformer Engine at call time.
        import transformer_engine.pytorch.graph as te_graph

        te_make_graphed_callables = te_graph.make_graphed_callables
        captures, helpers = [], []

        @functools.wraps(te_make_graphed_callables)
        def recording_make_graphed_callables(*args, **kwargs):
            captures.append((kwargs.get('pool'), torch.cuda.graphs.graph.default_capture_stream))
            return te_make_graphed_callables(*args, **kwargs)

        monkeypatch.setattr(te_graph, "make_graphed_callables", recording_make_graphed_callables)
        # Model construction and the eager steps run on a non-default stream, like training.
        training_stream = torch.cuda.Stream()
        training_stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(training_stream):
                yield captures, previous_stream, helpers
        finally:
            torch.cuda.synchronize()
            for helper in helpers:
                if helper.graphs_created():
                    helper.delete_cuda_graphs()
            torch.cuda.current_stream().wait_stream(training_stream)
            destroy_num_microbatches_calculator()
            Utils.destroy_model_parallel()

    @staticmethod
    def _forward_backward(model, step, micro_batch_size=2, seq_length=32):
        """One training forward and backward on the batch of ``step``."""
        generator = torch.Generator(device="cuda").manual_seed(step)
        tokens = torch.randint(
            0, 128, (micro_batch_size, seq_length + 1), device="cuda", generator=generator
        )
        position_ids = torch.arange(seq_length, device="cuda").repeat(micro_batch_size, 1)
        for param in model.parameters():
            # Leaf-grad buffers at fixed addresses: an optimizer-step graph reads them.
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            else:
                param.grad.zero_()
        loss = model(tokens[:, :-1], position_ids, None, labels=tokens[:, 1:])
        loss.float().mean().backward()

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("warmup_steps", [0, 1])
    def test_chunk_and_optimizer_graphs_share_a_pool_and_match_eager_training(
        self, monkeypatch, warmup_steps
    ):
        """The chunk graphs and the optimizer-step graph are captured into the shared pool on
        the shared capture stream, so the optimizer step reuses the blocks the chunk graphs
        leave idle. Only its transient tensors may live there: every replay of a chunk graph
        overwrites those blocks. The optimizer state is allocated by the first optimizer step,
        which runs eagerly also without a warm-up step (the chunk graphs are then captured
        before any optimizer step). Several optimizer steps match an eager twin."""
        from megatron.core import full_cuda_graph
        from megatron.core.optimizer.optimizer_cuda_graph import OptimizerCudaGraphWrapper

        num_steps, micro_batch_size, seq_length = 5, 2, 32
        with self._real_capture(monkeypatch, micro_batch_size) as (captures, previous, helpers):
            model = self._fixed_length_gpt('chunk_wise', cuda_graph_warmup_steps=warmup_steps)
            eager = self._fixed_length_gpt('none')
            eager.load_state_dict(model.state_dict())
            initial = {name: param.detach().clone() for name, param in eager.named_parameters()}

            eager_optimizer = torch.optim.Adam(eager.parameters(), lr=1e-2, capturable=True)
            for step in range(num_steps):
                self._forward_backward(eager, step)
                eager_optimizer.step()

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, capturable=True)
            optimizer_step = OptimizerCudaGraphWrapper(
                optimizer.step,
                cuda_graph_warmup_steps=warmup_steps,
                use_single_mempool=model.config.cuda_graph_use_single_mempool,
            )
            helper = TECudaGraphHelper(
                model=[model],
                config=model.config,
                seq_length=seq_length,
                micro_batch_size=micro_batch_size,
            )
            helpers.append(helper)
            for step in range(num_steps):
                if step == warmup_steps:
                    helper.create_cudagraphs()
                    assert len(model.decoder.te_cuda_graphs) == 1
                self._forward_backward(model, step)
                optimizer_step()
            torch.cuda.synchronize()
            assert OptimizerCudaGraphWrapper.cuda_graph is not None

            ((pool, capture_stream),) = captures
            from transformer_engine.pytorch.graph import make_graphed_callables

            # The recording stand-in wraps Transformer Engine's function, signature included.
            if 'pool' in inspect.signature(make_graphed_callables).parameters:
                assert pool is not None
                # Still the handle the optimizer-step graph was captured into afterwards.
                assert pool is full_cuda_graph._shared_graph_pool
                assert capture_stream is full_cuda_graph.get_shared_capture_stream()
            assert torch.cuda.graphs.graph.default_capture_stream is previous

            # Relative L2 error of the parameters against the size of the eager update.
            update_sq, error_sq = 0.0, 0.0
            for (name, param), reference in zip(model.named_parameters(), eager.parameters()):
                update_sq += (reference.float() - initial[name].float()).square().sum().item()
                error_sq += (param.float() - reference.float()).square().sum().item()
            assert update_sq > 0
            assert error_sq <= 1e-2 * update_sq, (error_sq, update_sq)

            helper.delete_cuda_graphs()
            assert full_cuda_graph._shared_graph_pool is None


class TestPartialCudaGraph:
    """Test that CUDA graph outputs match non-CUDA graph outputs for various scopes."""

    def setup_method(self, method):
        self.seq_length = 512
        self.micro_batch_size = 2
        self.tp_size = 2
        self.cp_size = 2
        self.cuda_graph_helper = None
        # Store original environment variable values
        self.original_env = {
            'CUDA_DEVICE_MAX_CONNECTIONS': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS'),
            'NVTE_ALLOW_NONDETERMINISTIC_ALGO': os.environ.get('NVTE_ALLOW_NONDETERMINISTIC_ALGO'),
        }
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'

    def teardown_method(self, method):
        # Restore original environment variable values
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None
        gc.collect()

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        layer_spec_fn=get_gpt_decoder_block_spec,
        **config_kwargs,
    ):
        args = get_args()
        cfg_container = get_run_config()
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = layer_spec_fn(
            config,
            use_transformer_engine=True,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
        )
        if args.mtp_num_layers:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec, use_transformer_engine=True
            )
        else:
            mtp_block_spec = None
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=cfg_container.tokenizer.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            mtp_block_spec=mtp_block_spec,
        )

    def create_test_args(
        self, cuda_graph_impl, cuda_graph_modules, cuda_graph_warmup_steps, ep_size, **kwargs
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_cuda_graphs.py']
        args = parse_args()
        args.num_layers = 4
        args.mtp_num_layers = 1
        args.hidden_size = 512
        args.num_attention_heads = 8
        args.max_position_embeddings = 512
        args.global_batch_size = self.micro_batch_size * 8 // self.tp_size // self.cp_size
        args.micro_batch_size = self.micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = self.seq_length
        args.tensor_model_parallel_size = self.tp_size
        args.sequence_parallel = True if self.tp_size > 1 else False
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = self.cp_size
        args.train_iters = 10
        args.lr = 3e-5
        args.bf16 = True
        args.add_bias_linear = False
        args.swiglu = True
        args.use_distributed_optimizer = True
        args.position_embedding_type = "rope"
        args.rotary_percent = 1.0
        args.hidden_dropout = 0.0
        args.attention_dropout = 0.0

        # MoE settings
        args.num_experts = 4
        args.expert_model_parallel_size = ep_size
        args.expert_tensor_parallel_size = 1 if ep_size > 1 else self.tp_size
        args.moe_shared_expert_intermediate_size = 1024
        args.moe_layer_freq = [0, 0, 1, 1]
        args.moe_permute_fusion = True
        args.moe_router_fusion = True
        args.moe_router_topk = 2
        args.moe_router_dtype = "fp32"

        # CUDA graph settings
        args.cuda_graph_impl = cuda_graph_impl
        args.cuda_graph_modules = cuda_graph_modules
        args.cuda_graph_warmup_steps = cuda_graph_warmup_steps

        # fp8 settings
        if fp8_available:
            args.fp8 = "e4m3"
            args.fp8_recipe = "tensorwise"
            args.first_last_layers_bf16 = True
            args.num_layers_at_start_in_bf16 = 1
            args.num_layers_at_end_in_bf16 = 1

        for key, value in kwargs.items():
            assert hasattr(args, key) or hasattr(TransformerConfig, key), f"Unknown argument: {key}"
            setattr(args, key, value)

        validate_args(args)
        cfg = pretrain_cfg_container_from_args(args)
        cfg.tokenizer.vocab_size = 1024
        set_global_variables(args, cfg, build_tokenizer=False)
        return args

    def get_batch(self, seq_length, micro_batch_size, cp_size):
        data = list(range(seq_length // cp_size))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length // cp_size, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length // cp_size).repeat((micro_batch_size, 1)).cuda()
        return input_ids, labels, position_ids, attention_mask, loss_mask

    def _run_test_helper(
        self,
        ep_size,
        cuda_graph_impl,
        cuda_graph_modules,
        cuda_graph_warmup_steps,
        padding_mask=None,
        mlp_chunks_for_training=1,
        **kwargs,
    ):
        """Test fp8_param with gpt_model."""
        args = self.create_test_args(
            cuda_graph_impl, cuda_graph_modules, cuda_graph_warmup_steps, ep_size, **kwargs
        )

        set_args(args)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        input_ids, labels, position_ids, attention_mask, loss_mask = self.get_batch(
            self.seq_length, self.micro_batch_size, self.cp_size
        )
        if padding_mask is not None:
            loss_mask = loss_mask * (~padding_mask).to(loss_mask.dtype)

        gpt_model, optimizer, _ = setup_model_and_optimizer(
            ModelType.encoder_or_decoder, self.model_provider
        )
        assert len(gpt_model) == 1  # Assume only one model in the model provider.
        if mlp_chunks_for_training > 1:
            # Not a training argument; the layers read it from the shared TransformerConfig.
            gpt_model[0].config.mlp_chunks_for_training = mlp_chunks_for_training

        if cuda_graph_impl == "transformer_engine":
            self.cuda_graph_helper = TECudaGraphHelper(
                model=gpt_model,
                config=gpt_model[0].config,
                seq_length=self.seq_length,
                micro_batch_size=self.micro_batch_size,
                optimizers=[optimizer],
            )

        loss_list = []

        for i in range(100):
            gpt_model[0].zero_grad_buffer()
            optimizer.zero_grad()

            # Capture CUDA graphs after warmup if helper is provided
            if self.cuda_graph_helper is not None and i == cuda_graph_warmup_steps:
                self.cuda_graph_helper.create_cudagraphs()
                if padding_mask is not None:
                    # Eager warmup routed with a mask, so every graphed MoE router must have
                    # reserved a padding_mask graph input of the batch-first mask shape as the
                    # layer sees it: GPTModel scatters the mask along the sequence under
                    # sequence parallelism before the decoder block.
                    expected_shape = list(padding_mask.shape)
                    if args.sequence_parallel:
                        expected_shape[1] //= args.tensor_model_parallel_size
                    for layer in get_attr_wrapped_model(gpt_model[0], "decoder").layers:
                        if layer._moe_router_in_cuda_graph():
                            assert layer._te_cuda_graph_padding_mask_shape == tuple(
                                expected_shape
                            ), layer.layer_number

            gpt_model[0].set_is_first_microbatch()
            output = gpt_model[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                padding_mask=padding_mask,
            )

            # Check output shapes
            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length // self.cp_size

            # Verify gradients
            loss = output.mean()
            loss.backward()

            for param in gpt_model[0].parameters():
                assert param.main_grad is not None

            update_successful, _, _ = optimizer.step()
            assert update_successful

            loss_list.append(loss.item())

        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            # Keep the layer handles for post-run assertions: the helper is
            # nulled below, but the layer objects (and attributes the replay
            # tail set on them) outlive graph teardown.
            self.last_flattened_callables = self.cuda_graph_helper.flattened_callables
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None

        return torch.tensor(loss_list)

    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("2.10.0")),
        reason="Partial CUDA graph UT support requires TransformerEngine version >= 2.10.0",
    )
    @pytest.mark.parametrize("ep_size", [1, 4])
    @pytest.mark.parametrize("moe_dropless_dispatcher", [False, True])
    @pytest.mark.parametrize(
        "moe_dispatcher_type",
        [
            "alltoall",
            "deepep",
            "hybridep",
            "ncclep",
            pytest.param("ncclep_fp8", marks=pytest.mark.launch_on_gb200),
        ],
    )
    def test_moe_partial_cudagraph(self, ep_size, moe_dropless_dispatcher, moe_dispatcher_type):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            context_parallel_size=self.cp_size,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=1 if ep_size > 1 else self.tp_size,
            expert_model_parallel_size=ep_size,
        )

        extra_kwargs = {}
        if moe_dispatcher_type == "deepep":
            if not is_deep_ep_available():
                pytest.skip("Deep EP is not available")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "deepep"
        elif moe_dispatcher_type == "hybridep":
            if not is_hybrid_ep_available():
                pytest.skip("Hybrid EP is not available")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "hybridep"
        elif moe_dispatcher_type == "ncclep":
            if not is_nccl_ep_available():
                pytest.skip("NCCL EP is not available")
            if ep_size < 2:
                pytest.skip("NCCL EP requires expert_model_parallel_size >= 2 (ep_bootstrap)")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "ncclep"
        elif moe_dispatcher_type == "ncclep_fp8":
            if not is_nccl_ep_available():
                pytest.skip("NCCL EP is not available")
            if ep_size < 2:
                pytest.skip("NCCL EP requires expert_model_parallel_size >= 2 (ep_bootstrap)")
            if not is_nccl_ep_fp8_dispatch_available():
                pytest.skip(
                    "NCCL EP MXFP8 wire needs EpBuffer quant-recipe support and MXFP8 hardware"
                )
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "ncclep"
            # MXFP8 wire dtypes require the op-fuser grouped GEMM (it consumes the carrier).
            extra_kwargs["moe_grouped_gemm"] = True
            extra_kwargs["use_transformer_engine_op_fuser"] = True
            extra_kwargs["moe_dispatch_fwd_dtype"] = "mxfp8"
            extra_kwargs["moe_combine_bwd_dtype"] = "mxfp8"
        else:
            extra_kwargs["moe_token_dispatcher_type"] = moe_dispatcher_type
        if not moe_dropless_dispatcher:
            if moe_dispatcher_type in ("deepep", "ncclep", "ncclep_fp8"):
                pytest.skip(f"{moe_dispatcher_type} doesn't support drop&pad MoE")
            extra_kwargs["moe_expert_capacity_factor"] = 1.0
            extra_kwargs["moe_pad_expert_input_to_capacity"] = True

        loss_list_ref = self._run_test_helper(ep_size, "none", None, 0, **extra_kwargs)
        for cuda_graph_modules in [
            None,
            [CudaGraphModule.attn],
            [CudaGraphModule.moe],
            [CudaGraphModule.mlp, CudaGraphModule.moe_router],
            [
                CudaGraphModule.attn,
                CudaGraphModule.mlp,
                CudaGraphModule.moe_router,
                CudaGraphModule.moe_preprocess,
            ],
        ]:
            if (
                moe_dropless_dispatcher
                or moe_dispatcher_type in ("hybridep", "ncclep", "ncclep_fp8")
            ) and (cuda_graph_modules is None or CudaGraphModule.moe in cuda_graph_modules):
                # Dropless MoE or a dynamic-shape flex backend (Hybrid EP / NCCL EP) can't be
                # captured at the "moe" scope (the dispatch does a device-to-host sync). Skip;
                # the surrounding compute submodules are still graphed.
                continue
            cuda_graph_warmup_steps = 3
            loss_list = self._run_test_helper(
                ep_size,
                "transformer_engine",
                cuda_graph_modules,
                cuda_graph_warmup_steps,
                **extra_kwargs,
            )
            assert torch.equal(loss_list, loss_list_ref)

        if moe_dispatcher_type == "hybridep":
            reset_hybrid_ep_buffer()
        if moe_dispatcher_type in ("ncclep", "ncclep_fp8"):
            from megatron.core.transformer.moe.fused_a2a import nccl_ep_finalize

            nccl_ep_finalize()
        Utils.destroy_model_parallel()

    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("2.10.0")),
        reason="Partial CUDA graph UT support requires TransformerEngine version >= 2.10.0",
    )
    @pytest.mark.parametrize(
        "moe_dispatcher_type,mlp_chunks",
        [("alltoall", 1), ("alltoall", 2), ("hybridep", 1), ("ncclep", 1)],
        ids=["alltoall", "alltoall-chunked", "hybridep", "ncclep"],
    )
    def test_moe_partial_cudagraph_padding_mask(self, moe_dispatcher_type, mlp_chunks):
        """Graphed routing must see the MoE padding mask exactly like eager routing.

        Eager warmup records the mask shape, TE capture reserves a padding_mask graph input for
        every graphed MoE router (whether or not attention is graphed too), and replay feeds the
        real mask. The loss trajectory must match the eager run bit for bit. With MLP chunking the
        layer splits the mask per chunk, so the reserved graph input must still be the whole-layer
        mask (checked at capture); that case graphs the whole layer, which is where the chunk loop
        runs inside the graph. Whole-layer graphs with MLP chunking are not bit-exact against
        eager even without a mask (pre-existing), so the chunked case only checks capture/replay.
        """
        ep_size = 4
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            context_parallel_size=self.cp_size,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            expert_model_parallel_size=ep_size,
        )

        extra_kwargs = {}
        if moe_dispatcher_type == "hybridep":
            if not is_hybrid_ep_available():
                pytest.skip("Hybrid EP is not available")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "hybridep"
        elif moe_dispatcher_type == "ncclep":
            if not is_nccl_ep_available():
                pytest.skip("NCCL EP is not available")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "ncclep"
        else:
            extra_kwargs["moe_token_dispatcher_type"] = moe_dispatcher_type
        if mlp_chunks > 1:
            # The chunked case graphs the whole layer, which a dropless dispatcher cannot do (its
            # dispatch syncs token counts to the host); use drop-and-pad for both runs.
            extra_kwargs["moe_expert_capacity_factor"] = 1.0
            extra_kwargs["moe_pad_expert_input_to_capacity"] = True

        # Batch-first mask matching input_ids: pad the tail of the second sample.
        seq_per_cp = self.seq_length // self.cp_size
        padding_mask = torch.zeros((self.micro_batch_size, seq_per_cp), dtype=torch.bool).cuda()
        padding_mask[1, seq_per_cp // 2 :] = True

        loss_list_ref = None
        if mlp_chunks == 1:
            loss_list_ref = self._run_test_helper(
                ep_size, "none", None, 0, padding_mask=padding_mask, **extra_kwargs
            )
        if mlp_chunks > 1:
            # Whole-layer graph: eager warmup chunks the MLP, capture must reserve the unchunked
            # [b, s] mask (a chunk-shaped input fails at capture with a mask/token-count mismatch).
            cuda_graph_modules_list = [None]
        else:
            cuda_graph_modules_list = [
                # Attention eager: replay used to reset kwargs and drop the mask.
                [CudaGraphModule.mlp, CudaGraphModule.moe_router],
                # Attention graphed: the mask reached TE but was not a captured input.
                [
                    CudaGraphModule.attn,
                    CudaGraphModule.mlp,
                    CudaGraphModule.moe_router,
                    CudaGraphModule.moe_preprocess,
                ],
            ]
        for cuda_graph_modules in cuda_graph_modules_list:
            loss_list = self._run_test_helper(
                ep_size,
                "transformer_engine",
                cuda_graph_modules,
                3,
                padding_mask=padding_mask,
                mlp_chunks_for_training=mlp_chunks,
                **extra_kwargs,
            )
            if loss_list_ref is not None:
                assert torch.equal(loss_list, loss_list_ref), cuda_graph_modules

        if moe_dispatcher_type == "hybridep":
            reset_hybrid_ep_buffer()
        if moe_dispatcher_type == "ncclep":
            from megatron.core.transformer.moe.fused_a2a import nccl_ep_finalize

            nccl_ep_finalize()
        Utils.destroy_model_parallel()

    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("2.10.0")),
        reason="Partial CUDA graph UT support requires TransformerEngine version >= 2.10.0",
    )
    @pytest.mark.parametrize("ep_size", [1, 4])
    def test_mhc_moe_partial_cudagraph(self, ep_size):
        """Test that mHC (Hyper Connection) layers produce identical loss curves
        with and without TE partial CUDA graph capture.

        This validates the fix where HyperConnectionTransformerLayer overrides
        _te_cuda_graph_replay_impl (not _te_cuda_graph_replay) so that the parent's
        delay_offload_until_cuda_graph lifecycle and overlap_moe_expert_parallel_comm
        handling are preserved.
        """
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            context_parallel_size=self.cp_size,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=1 if ep_size > 1 else self.tp_size,
            expert_model_parallel_size=ep_size,
        )

        extra_kwargs = {
            "enable_hyper_connections": True,
            "num_residual_streams": 4,
            "mtp_num_layers": None,  # mHC is incompatible with MTP
        }

        loss_list_ref = self._run_test_helper(ep_size, "none", None, 0, **extra_kwargs)
        for cuda_graph_modules in [
            [CudaGraphModule.attn],
            [CudaGraphModule.mlp, CudaGraphModule.moe_router],
            [
                CudaGraphModule.attn,
                CudaGraphModule.mlp,
                CudaGraphModule.moe_router,
                CudaGraphModule.moe_preprocess,
            ],
        ]:
            cuda_graph_warmup_steps = 3
            loss_list = self._run_test_helper(
                ep_size,
                "transformer_engine",
                cuda_graph_modules,
                cuda_graph_warmup_steps,
                **extra_kwargs,
            )
            assert torch.equal(loss_list, loss_list_ref), (
                f"mHC loss mismatch with cuda_graph_modules={cuda_graph_modules}, ep_size={ep_size}. "
                f"Max diff: {torch.max(torch.abs(loss_list - loss_list_ref))}"
            )

        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("2.10.0")),
        reason="Partial CUDA graph UT support requires TransformerEngine version >= 2.10.0",
    )
    def test_mhc_recompute_whole_attention_cudagraph(self):
        """mHC selective recompute under whole-attention capture matches eager.

        With mhc_recompute_attn_cuda_graph_split off (the default), an attn-scope
        graph captures the whole attention range and the replay's non-split tail
        runs the MLP-side mHC group eagerly: mlp_hyper_connection registers its
        checkpoints against the manager __call__ stashed on the layer, the block
        discards at group end, and the unified hook replays them in backward. A
        graphed run must therefore reproduce the eager loss curve bit for bit --
        this is the executing coverage for that tail, with a live manager rather
        than a mocked boundary.
        """
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            context_parallel_size=self.cp_size,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=self.tp_size,
            expert_model_parallel_size=1,
        )

        extra_kwargs = {
            "enable_hyper_connections": True,
            "num_residual_streams": 4,
            "mtp_num_layers": None,  # mHC is incompatible with MTP
            "recompute_granularity": "selective",
            "recompute_modules": ["mhc"],
            "mhc_recompute_layer_num": 2,
        }

        loss_list_ref = self._run_test_helper(1, "none", None, 0, **extra_kwargs)
        loss_list = self._run_test_helper(
            1, "transformer_engine", [CudaGraphModule.attn], 3, **extra_kwargs
        )
        assert torch.equal(loss_list, loss_list_ref), (
            "mHC recompute under whole-attention capture diverged from eager. "
            f"Max diff: {torch.max(torch.abs(loss_list - loss_list_ref))}"
        )
        # Loss parity alone is blind to an inert manager (an empty recompute
        # group discards and replays nothing, bit-identically), so pin the
        # layer-side threading directly: the graphed replay tail must have
        # created the pre-MLP checkpoint against a live manager.
        layers = self.last_flattened_callables
        assert any(
            getattr(layer, "pre_mlp_norm_checkpoint", None) is not None for layer in layers
        ), (
            "no layer created pre_mlp_norm_checkpoint during graphed replay: the "
            "whole-attention tail is not threading the recompute manager"
        )

        Utils.destroy_model_parallel()


class _SimpleModule(MegatronModule):
    """Minimal MegatronModule for testing CudaGraphManager with function_name."""

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def my_op(self, x):
        return self.linear(x)


class _CheckpointDependencyModule(MegatronModule):
    """A visible projection consuming the output of a checkpointed child module."""

    def __init__(self, config):
        super().__init__(config)
        self.checkpointed = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.projection = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.my_op(x)

    def my_op(self, x):
        checkpoint = CheckpointWithoutOutput()
        hidden = checkpoint.checkpoint(self.checkpointed, x)
        output = self.projection(hidden)
        checkpoint.discard_output_and_register_recompute(output)
        return output


class _RepeatedParameterModule(MegatronModule):
    """Invoke one graphed operation twice so two runners share its parameter."""

    def __init__(self, config):
        super().__init__(config)
        self.weight = torch.nn.Parameter(torch.randn(config.hidden_size))

    def forward(self, x):
        return self.project(x) + self.project(0.5 * x)

    def project(self, x):
        return x + self.weight


class _SimpleNonModule:
    """non-nn.Module base_module for testing the function_name= form of `CudaGraphManager`."""

    def __init__(self, config):
        self.weight = torch.randn(config.hidden_size, config.hidden_size, device="cuda")

    def my_op(self, x):
        return x @ self.weight


def _make_simple_module(config):
    return _SimpleModule(config).cuda().eval()


def _make_simple_non_module(config):
    return _SimpleNonModule(config)


class TestCheckpointParameterDiscovery:
    """Local-CG discovery and DDP readiness for a checkpointed nested module."""

    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        if _CudagraphGlobalRecord.cudagraph_created:
            delete_cuda_graphs()
        else:
            _CudagraphGlobalRecord.cudagraph_record = []
            _CudagraphGlobalRecord.cudagraph_inference_record = []
            CudaGraphManager.global_mempool = None
        torch.cuda.set_stream(torch.cuda.default_stream())
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_nested_checkpoint_parameter_gradients_and_ddp_readiness(self):
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() < 2:
            pytest.skip("test requires at least two data-parallel ranks")

        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
            cuda_graph_warmup_steps=0,
            recompute_granularity="selective",
            recompute_modules=["layernorm"],
        )
        torch.manual_seed(123)
        reference = _CheckpointDependencyModule(config).cuda()
        module = _CheckpointDependencyModule(config).cuda()
        module.load_state_dict(reference.state_dict())

        manager = CudaGraphManager(
            config, base_module=module, function_name="my_op", need_backward=True
        )
        ddp_model = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=1_000_000),
            module,
        )

        # Distinct inputs make the expected DDP result the average of different local gradients.
        torch.manual_seed(1000 + ddp_model.dp_group.rank())
        test_input = torch.randn(4, config.hidden_size, device="cuda", requires_grad=True)
        reference_output = reference.my_op(test_input.detach().clone().requires_grad_(True))
        reference_output_value = reference_output.detach().clone()
        reference_output.sum().backward()
        reference_wgrads = {
            name: param.grad.detach().clone() for name, param in reference.named_parameters()
        }
        for grad in reference_wgrads.values():
            torch.distributed.all_reduce(grad, group=ddp_model.dp_group)
            grad.div_(ddp_model.dp_group.size())
        del reference_output

        ddp_model.zero_grad_buffer()
        record_input = test_input.detach().clone().requires_grad_(True)
        ddp_model(record_input).sum().backward()
        ddp_model.finish_grad_sync()
        ddp_model.zero_grad_buffer()
        create_cudagraphs()

        runner = manager.cudagraph_runners[0]
        assert any(param is module.checkpointed.weight for param in runner.params_to_backprop)

        # The first replay uses DDP's recorded ready counts; the second verifies per-step reset.
        for _ in range(2):
            ddp_model.zero_grad_buffer()
            replay_output = ddp_model(test_input.detach().clone().requires_grad_(True))
            replay_output.sum().backward()
            ddp_model.finish_grad_sync()
            torch.cuda.synchronize()

            torch.testing.assert_close(replay_output, reference_output_value)
            for name, param in module.named_parameters():
                assert param.grad is None
                torch.testing.assert_close(param.main_grad, reference_wgrads[name])


class TestRepeatedParameterCapture:
    """Local-CG capture preserves every use of a parameter shared by multiple runners."""

    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        if _CudagraphGlobalRecord.cudagraph_created:
            delete_cuda_graphs()
        else:
            _CudagraphGlobalRecord.cudagraph_record = []
            _CudagraphGlobalRecord.cudagraph_inference_record = []
            CudaGraphManager.global_mempool = None
        torch.cuda.set_stream(torch.cuda.default_stream())
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("1.5.0")),
        reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
    )
    def test_two_runners_accumulate_shared_parameter_and_preserve_record_grad(self):
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() < 2:
            pytest.skip("test requires at least two data-parallel ranks")

        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
            cuda_graph_warmup_steps=0,
        )
        torch.manual_seed(123)
        reference = _RepeatedParameterModule(config).cuda()
        module = _RepeatedParameterModule(config).cuda()
        module.load_state_dict(reference.state_dict())

        manager = CudaGraphManager(
            config, base_module=module, function_name="project", need_backward=True
        )
        ddp_model = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=1_000_000),
            module,
        )

        torch.manual_seed(1000 + ddp_model.dp_group.rank())
        test_input = torch.randn(4, config.hidden_size, device="cuda", requires_grad=True)
        reference_output = reference(test_input.detach().clone().requires_grad_(True))
        reference_output_value = reference_output.detach().clone()
        reference_output.sum().backward()
        reference_local_wgrad = reference.weight.grad.detach().clone()
        reference_wgrad = reference_local_wgrad.clone()
        torch.distributed.all_reduce(reference_wgrad, group=ddp_model.dp_group)
        reference_wgrad.div_(ddp_model.dp_group.size())

        ddp_model.zero_grad_buffer()
        record_output = ddp_model(test_input.detach().clone().requires_grad_(True))
        record_output.sum().backward()
        ddp_model.finish_grad_sync()
        record_wgrad = module.weight.main_grad.detach().clone()
        torch.testing.assert_close(record_wgrad, reference_wgrad)

        create_cudagraphs()

        assert len(manager.cudagraph_runners) == 2
        torch.testing.assert_close(module.weight.main_grad, record_wgrad)

        ddp_model.zero_grad_buffer()
        with ddp_model.no_sync():
            replay_output = ddp_model(test_input.detach().clone().requires_grad_(True))
            replay_output.sum().backward()
        torch.cuda.synchronize()

        torch.testing.assert_close(replay_output, reference_output_value)
        torch.testing.assert_close(module.weight.main_grad, reference_local_wgrad)

        for _ in range(2):
            ddp_model.zero_grad_buffer()
            replay_output = ddp_model(test_input.detach().clone().requires_grad_(True))
            replay_output.sum().backward()
            ddp_model.finish_grad_sync()
            torch.cuda.synchronize()

            torch.testing.assert_close(replay_output, reference_output_value)
            torch.testing.assert_close(module.weight.main_grad, reference_wgrad)


class TestInlineCaptureManager:
    """Tests for CudaGraphManager with inline_capture, function_name, eager, and cache_key."""

    def _make_config(self):
        return TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=1,
            use_cpu_initialization=True,
            cuda_graph_impl="local",
            inference_rng_tracker=True,
        )

    def setup_method(self, method):
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    def teardown_method(self, method):
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "make_module",
        [
            pytest.param(_make_simple_module, id="nn_module"),
            pytest.param(_make_simple_non_module, id="plain_class"),
        ],
    )
    @torch.inference_mode()
    def test_inline_capture_matches_eager(self, make_module):
        """Inline-captured graph output must match eager execution."""
        config = self._make_config()
        module = make_module(config)

        # Get eager reference before wrapping
        x = torch.randn(4, config.hidden_size, device="cuda")
        eager_out = module.my_op(x).clone()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        # First call captures, second replays
        graph_out_1 = module.my_op(x)
        graph_out_2 = module.my_op(x)
        assert torch.equal(eager_out, graph_out_1)
        assert torch.equal(eager_out, graph_out_2)
        assert len(mgr.cudagraph_runners) == 1
        assert mgr.cudagraph_runners[0].fwd_graph_recorded

    @torch.inference_mode()
    def test_eager_bypass(self):
        """eager=True must bypass graph capture entirely."""
        config = self._make_config()
        module = _SimpleModule(config).cuda().eval()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        x = torch.randn(4, config.hidden_size, device="cuda")
        _ = module.my_op(x, eager=True)
        _ = module.my_op(x, eager=True)
        assert len(mgr.cudagraph_runners) == 0, "eager=True should not create runners"

    @torch.inference_mode()
    def test_cache_key_routing(self):
        """Different cache_keys must create separate runners."""
        config = self._make_config()
        module = _SimpleModule(config).cuda().eval()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        x = torch.randn(4, config.hidden_size, device="cuda")
        module.my_op(x, cache_key="key_a")
        module.my_op(x, cache_key="key_b")

        assert len(mgr.cudagraph_runners) == 2
        assert mgr.custom_cudagraphs_lookup_table["key_a"] is not None
        assert mgr.custom_cudagraphs_lookup_table["key_b"] is not None
        assert (
            mgr.custom_cudagraphs_lookup_table["key_a"]
            is not mgr.custom_cudagraphs_lookup_table["key_b"]
        )

        # Same key reuses the runner
        module.my_op(x, cache_key="key_a")
        assert len(mgr.cudagraph_runners) == 2

    @torch.inference_mode()
    def test_num_warmup_steps_override(self):
        """num_warmup_steps on the manager must override the config value on runners."""
        config = self._make_config()
        config.cuda_graph_warmup_steps = 3
        module = _SimpleModule(config).cuda().eval()

        mgr = CudaGraphManager(
            config,
            base_module=module,
            function_name="my_op",
            inline_capture=True,
            num_warmup_steps=0,
            need_backward=False,
        )

        x = torch.randn(4, config.hidden_size, device="cuda")
        module.my_op(x, cache_key="test")

        runner = mgr.cudagraph_runners[0]
        assert (
            runner.num_warmup_steps == 0
        ), f"Expected 0 warmup steps (manager override), got {runner.num_warmup_steps}"


class TestSkipFp8WeightUpdateTensor:
    """Regression test for the TE 2.15 ``set_skip_fp8_weight_update_tensor`` removal."""

    @staticmethod
    def _read_skip_tensor():
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        getter = getattr(FP8GlobalStateManager, "get_skip_fp8_weight_update_tensor", None)
        if getter is not None:
            return getter()
        return FP8GlobalStateManager.quantization_state.skip_fp8_weight_update_tensor

    @staticmethod
    def _reset_skip_tensor():
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        if "skip_fp8_weight_update_tensor" in vars(FP8GlobalStateManager):
            FP8GlobalStateManager.skip_fp8_weight_update_tensor = None
        qstate = getattr(FP8GlobalStateManager, "quantization_state", None)
        if qstate is not None and hasattr(qstate, "skip_fp8_weight_update_tensor"):
            qstate.skip_fp8_weight_update_tensor = None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sets_value_in_place(self):
        """Helper writes the right value and reuses the same storage across calls."""
        from megatron.core.transformer.cuda_graphs import _set_skip_fp8_weight_update_tensor

        self._reset_skip_tensor()
        try:
            _set_skip_fp8_weight_update_tensor(True)
            t = self._read_skip_tensor()
            assert t.shape == (1,) and t.dtype == torch.float32 and t.is_cuda
            assert t.item() == 1.0

            # data_ptr must stay stable so captured cudagraphs read the same address.
            ptr = t.data_ptr()
            _set_skip_fp8_weight_update_tensor(False)
            assert self._read_skip_tensor().data_ptr() == ptr
            assert self._read_skip_tensor().item() == 0.0
        finally:
            self._reset_skip_tensor()


if __name__ == "__main__":

    test = TestParallelTransformerBlockCudagraphs()
    test.setup_method(method=None)
    test.test_gpu_cudagraph()
    test.teardown_method(method=None)

    llava_test = TestLLaVACudaGraph()
    llava_test.setup_method(method=None)
    llava_test.test_llava_cudagraph_is_last_layer_logic()
    llava_test.teardown_method(method=None)

    test = TestPartialCudaGraph()
    test.setup_method(method=None)
    test.test_moe_partial_cudagraph(4, True, "alltoall")
    test.teardown_method(method=None)

    test = TestPartialCudaGraph()
    test.setup_method(method=None)
    test.test_mhc_moe_partial_cudagraph(4)
    test.teardown_method(method=None)
