# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.pipeline_parallel.schedules import set_current_microbatch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.chunk_cuda_graphs import (
    ChunkCudaGraphRuntimeSlots,
    build_chunk_cuda_graph_slot_plan,
    build_chunk_cuda_graph_slot_plan_from_schedule,
    get_cuda_graph_schedule_stage_order_from_counts,
    get_required_num_microbatch_slots_per_chunk,
)
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    TECudaGraphHelper,
    _CudagraphGlobalRecord,
    _restore_module_grad_state,
    _restore_moe_metrics_tracker,
    _snapshot_module_grad_state,
    _snapshot_moe_metrics_tracker,
    reset_chunk_cuda_graph_runtime_slots,
    set_current_cuda_graph_slot,
)
from megatron.core.transformer.enums import CudaGraphModule, CudaGraphScope, InferenceCudaGraphScope
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.fused_a2a import reset_hybrid_ep_buffer
from megatron.core.transformer.moe.moe_logging import MetricEntry, MoEMetricsTracker
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.utils import is_fa_min_version, is_te_min_version
from megatron.training import arguments as training_arguments
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.test_utilities import Utils

fp8_available, _ = check_fp8_support()


def _base_cuda_graph_config(**kwargs) -> TransformerConfig:
    return TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4, **kwargs)


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
    def test_local_impl_defaults_to_layer_scope(self):
        cfg = _base_cuda_graph_config(cuda_graph_impl='local')
        assert cfg.inference_cuda_graph_scope == InferenceCudaGraphScope.layer

    def test_full_iteration_impl_requires_empty_scope(self):
        with pytest.raises(
            AssertionError,
            match='cuda_graph_modules must be empty when cuda_graph_impl="full_iteration"',
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='full_iteration', cuda_graph_modules=[CudaGraphModule.attn]
            )

    def test_chunk_granularity_requires_per_module_full_chunk_scope(self):
        cfg = _base_cuda_graph_config(
            cuda_graph_impl='local',
            cuda_graph_modules=[],
            cuda_graph_granularity='chunk',
        )
        assert cfg.cuda_graph_granularity == 'chunk'

        with pytest.raises(
            AssertionError, match="chunk CUDA graph granularity is only supported"
        ):
            _base_cuda_graph_config(
                cuda_graph_impl='full_iteration',
                cuda_graph_modules=[],
                cuda_graph_granularity='chunk',
            )

        with pytest.raises(AssertionError, match="requires empty cuda_graph_modules"):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules=[CudaGraphModule.attn],
                cuda_graph_granularity='chunk',
            )

    def test_local_chunk_granularity_allows_dynamic_microbatch_slots(self):
        cfg = _base_cuda_graph_config(
            cuda_graph_impl='local',
            cuda_graph_modules=[],
            cuda_graph_granularity='chunk',
            cuda_graph_dynamic_microbatches=True,
            cuda_graph_num_microbatch_slots=8,
        )
        assert cfg.cuda_graph_dynamic_microbatches is True
        assert cfg.cuda_graph_num_microbatch_slots == 8

        with pytest.raises(AssertionError, match="requires cuda_graph_granularity=chunk"):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules=[],
                cuda_graph_granularity='layer',
                cuda_graph_dynamic_microbatches=True,
            )

        with pytest.raises(AssertionError, match="requires cuda_graph_dynamic_microbatches"):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                cuda_graph_modules=[],
                cuda_graph_granularity='chunk',
                cuda_graph_num_microbatch_slots=8,
            )

    def test_thd_cuda_graph_requires_padding_alignment_flag(self):
        with pytest.raises(AssertionError, match="THD CUDA graph requires"):
            _base_cuda_graph_config(
                cuda_graph_impl='local',
                sequence_packing_scheduler='dp_balanced',
                max_seqlen_per_dp_cp_rank=128,
                moe_token_dispatcher_type='alltoall',
            )

        cfg = _base_cuda_graph_config(
            cuda_graph_impl='local',
            sequence_packing_scheduler='dp_balanced',
            max_seqlen_per_dp_cp_rank=128,
            pad_packed_seq_alignment=0,
            moe_token_dispatcher_type='alltoall',
        )
        assert cfg.pad_packed_seq_alignment == 0

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


class TestCudaGraphMoEMetrics:
    def test_restore_moe_metrics_removes_capture_only_entries(self):
        tracker = MoEMetricsTracker()
        tracker.metrics["load_balancing_loss"] = MetricEntry(
            values=torch.ones(4), needs_dp_avg=False
        )

        cached_metrics = _snapshot_moe_metrics_tracker(tracker)

        tracker.metrics["load_balancing_loss"].values.add_(3)
        tracker.metrics["load_balancing_loss"].needs_dp_avg = True
        tracker.metrics["z_loss"] = MetricEntry(values=torch.full((4,), 9.0))

        _restore_moe_metrics_tracker(tracker, cached_metrics)

        assert list(tracker.metrics.keys()) == ["load_balancing_loss"]
        assert torch.equal(tracker.metrics["load_balancing_loss"].values, torch.ones(4))
        assert tracker.metrics["load_balancing_loss"].needs_dp_avg is False


class TestCudaGraphGradState:
    def test_restore_module_grad_state_restores_existing_grad_buffers_and_flags(self):
        module = torch.nn.Linear(3, 2, bias=False)
        param = module.weight
        param.main_grad = torch.full_like(param, 1.0)
        param.grad = torch.full_like(param, 2.0)
        param.grad_added_to_main_grad = False

        grad_state = _snapshot_module_grad_state(module)

        param.main_grad.add_(10.0)
        param.grad.add_(20.0)
        param.grad_added_to_main_grad = True

        _restore_module_grad_state(grad_state)

        assert torch.equal(param.main_grad, torch.full_like(param, 1.0))
        assert torch.equal(param.grad, torch.full_like(param, 2.0))
        assert param.grad_added_to_main_grad is False

    def test_restore_module_grad_state_removes_capture_only_grad_attrs(self):
        module = torch.nn.Linear(3, 2, bias=False)
        param = module.weight
        assert not hasattr(param, "main_grad")
        assert param.grad is None
        assert not hasattr(param, "grad_added_to_main_grad")

        grad_state = _snapshot_module_grad_state(module)

        param.main_grad = torch.ones_like(param)
        param.grad = torch.ones_like(param)
        param.grad_added_to_main_grad = True

        _restore_module_grad_state(grad_state)

        assert not hasattr(param, "main_grad")
        assert param.grad is None
        assert not hasattr(param, "grad_added_to_main_grad")


class TestLocalChunkCudaGraphOutput:
    def test_output_alias_is_safe_for_pipeline_deallocate(self):
        graph_static_output = torch.ones(2, 3, requires_grad=True)
        returned_output = TransformerBlock._make_local_chunk_output_pipeline_safe(
            graph_static_output
        )

        assert returned_output is not graph_static_output
        assert returned_output._base is None
        assert returned_output.data_ptr() == graph_static_output.data_ptr()
        assert returned_output.requires_grad == graph_static_output.requires_grad

        graph_static_output_ptr = graph_static_output.data_ptr()
        returned_output.data = torch.empty((1,), dtype=returned_output.dtype)

        assert graph_static_output.data_ptr() == graph_static_output_ptr
        assert graph_static_output.shape == (2, 3)

    def test_tuple_outputs_preserve_non_tensors(self):
        graph_static_output = torch.ones(2, 3, requires_grad=True)
        aux = object()

        returned_output, returned_aux = TransformerBlock._make_local_chunk_output_pipeline_safe(
            (graph_static_output, aux)
        )

        assert returned_output is not graph_static_output
        assert returned_output._base is None
        assert returned_aux is aux


class TestLocalChunkCudaGraphDryRun:
    @staticmethod
    def _manager_for_events(monkeypatch, pp_rank, vpp_size=None):
        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda: pp_rank)
        monkeypatch.setattr(
            parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: vpp_size
        )
        manager = CudaGraphManager.__new__(CudaGraphManager)
        manager.config = SimpleNamespace(
            microbatch_group_size_per_vp_stage=1,
            overlap_moe_expert_parallel_comm=False,
            pipeline_model_parallel_size=2,
        )
        return manager

    def test_rank0_dry_run_events_match_1f1b_order(self, monkeypatch):
        manager = self._manager_for_events(monkeypatch, pp_rank=0)

        assert manager._get_local_chunk_dry_run_events(4) == [
            ("forward", 0),
            ("forward", 1),
            ("backward", 0),
            ("forward", 2),
            ("backward", 1),
            ("forward", 3),
            ("backward", 2),
            ("backward", 3),
        ]

    def test_rank1_dry_run_events_start_in_steady_state(self, monkeypatch):
        manager = self._manager_for_events(monkeypatch, pp_rank=1)

        assert manager._get_local_chunk_dry_run_events(3) == [
            ("forward", 0),
            ("backward", 0),
            ("forward", 1),
            ("backward", 1),
            ("forward", 2),
            ("backward", 2),
        ]

    def test_vpp_dry_run_events_follow_interleaved_chunk_order(self, monkeypatch):
        manager = self._manager_for_events(monkeypatch, pp_rank=0, vpp_size=2)

        assert manager._get_local_chunk_dry_run_schedule_events(3, num_model_chunks=2) == [
            ("forward", 0, 0),
            ("forward", 1, 0),
            ("forward", 0, 1),
            ("forward", 1, 1),
            ("backward", 1, 0),
            ("forward", 0, 2),
            ("backward", 0, 0),
            ("forward", 1, 2),
            ("backward", 1, 1),
            ("backward", 0, 1),
            ("backward", 1, 2),
            ("backward", 0, 2),
        ]

    def test_vpp_dry_run_events_use_default_pp_group_size(self, monkeypatch):
        manager = self._manager_for_events(monkeypatch, pp_rank=0, vpp_size=2)
        manager.config.microbatch_group_size_per_vp_stage = None

        assert manager._get_local_chunk_dry_run_schedule_events(4, num_model_chunks=2) == [
            ("forward", 0, 0),
            ("forward", 0, 1),
            ("forward", 1, 0),
            ("forward", 1, 1),
            ("forward", 0, 2),
            ("backward", 1, 0),
            ("forward", 0, 3),
            ("backward", 1, 1),
            ("forward", 1, 2),
            ("backward", 0, 0),
            ("forward", 1, 3),
            ("backward", 0, 1),
            ("backward", 1, 2),
            ("backward", 1, 3),
            ("backward", 0, 2),
            ("backward", 0, 3),
        ]

    def test_slot_dry_run_events_reuse_slot_ids_beyond_requested_slots(self, monkeypatch):
        manager = self._manager_for_events(monkeypatch, pp_rank=0)
        manager.config.microbatch_group_size_per_vp_stage = 1

        events = manager._get_local_chunk_dry_run_slot_schedule_events(
            requested_slots=2,
            num_model_chunks=1,
            probe_num_microbatches=4,
        )

        assert events == [
            ("forward", 0, 0, 0),
            ("forward", 0, 1, 1),
            ("backward", 0, 0, 0),
            ("forward", 0, 2, 0),
            ("backward", 0, 1, 1),
            ("forward", 0, 3, 1),
            ("backward", 0, 2, 0),
            ("backward", 0, 3, 1),
        ]

    def test_slot_dry_run_rejects_too_few_requested_slots(self, monkeypatch):
        manager = self._manager_for_events(monkeypatch, pp_rank=0)

        with pytest.raises(AssertionError, match="topology-required"):
            manager._get_local_chunk_dry_run_slot_schedule_events(
                requested_slots=1,
                num_model_chunks=1,
                probe_num_microbatches=4,
            )

class TestChunkCudaGraphRuntimeSlots:
    def test_runtime_slots_reuse_released_forward_slots(self):
        runtime_slots = ChunkCudaGraphRuntimeSlots(2)

        assert runtime_slots.forward(0) == 0
        assert runtime_slots.forward(1) == 1
        assert runtime_slots.backward(0) == 0
        assert runtime_slots.forward(2) == 0
        assert runtime_slots.backward(1) == 1
        assert runtime_slots.forward(3) == 1
        assert runtime_slots.backward(2) == 0
        assert runtime_slots.backward(3) == 1
        assert list(runtime_slots.available_slots) == [0, 1]

    def test_runtime_slots_support_more_total_microbatches_than_slots(self):
        runtime_slots = ChunkCudaGraphRuntimeSlots(2)

        seen_slots = set()
        for microbatch_id in range(32):
            slot = runtime_slots.forward(microbatch_id)
            seen_slots.add(slot)
            assert slot in (0, 1)
            assert runtime_slots.backward(microbatch_id) == slot

        assert seen_slots == {0, 1}
        assert list(runtime_slots.available_slots) == [0, 1]

    def test_runtime_slots_reject_slot_exhaustion(self):
        runtime_slots = ChunkCudaGraphRuntimeSlots(1)
        runtime_slots.forward(0)

        with pytest.raises(AssertionError, match="No free chunk CUDA graph runtime slot"):
            runtime_slots.forward(1)

    def test_reset_chunk_cuda_graph_runtime_slots_sets_decoder_metadata(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    cuda_graph_impl="local",
                    cuda_graph_granularity="chunk",
                    cuda_graph_dynamic_microbatches=True,
                    cuda_graph_num_microbatch_slots=2,
                )
                self.decoder = SimpleNamespace(layers=[])

        model = DummyModel()

        reset_chunk_cuda_graph_runtime_slots(
            model,
            num_microbatches=4,
            num_model_chunks=1,
            num_warmup_microbatches=1,
        )
        set_current_microbatch(model, 0)
        assert model.decoder.cuda_graph_forward_slot == 0
        set_current_microbatch(model, 1)
        assert model.decoder.cuda_graph_forward_slot == 1
        set_current_cuda_graph_slot(model, 0, forward=False)
        assert model.decoder.cuda_graph_current_slot == 0
        assert model.decoder.cuda_graph_current_op == "backward"
        set_current_microbatch(model, 2)
        assert model.decoder.cuda_graph_forward_slot == 0


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
        language_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        vision_layer_spec = get_vit_layer_with_transformer_engine_spec()
        assert isinstance(language_layer_spec.submodules, TransformerLayerSubmodules)
        vision_projection_spec = deepcopy(language_layer_spec.submodules.mlp.submodules)

        # Set vision model type
        vision_config.vision_model_type = "clip"
        language_config.language_model_type = "dummy"

        # Create LLaVA model with both encoder and decoder
        self.llava_model = LLaVAModel(
            language_transformer_config=language_config,
            language_transformer_layer_spec=language_layer_spec,
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

        set_current_microbatch(self.llava_model.vision_model, 1)
        set_current_microbatch(self.llava_model.language_model, 1)

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
            layer_type_list = validate_segment_layers(hybrid_layer_pattern)
            transformer_config = TransformerConfig(
                hidden_size=256,  # The Mamba layer places several constraints on this
                # Need to specify num_attention_heads and num_layers or TransformerConfig
                # will generate errors.
                num_layers=len(layer_type_list),
                num_attention_heads=4,
                use_cpu_initialization=True,
                cuda_graph_impl="local",
            )
            modules = hybrid_stack_spec.submodules
            return HybridStack(
                transformer_config,
                modules,
                layer_type_list=layer_type_list,
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


# Global storage for comparing unique buffer counts across different num_microbatches,
# keyed by (pp_size, vpp_size)
_unique_buffer_counts = {}


class TestTECudaGraphHelper:
    def setup_method(self, method):
        # Initialize parallel state
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        # Note: _unique_buffer_counts is intentionally NOT cleared here so we can
        # compare values across parametrized test runs

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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_cuda_graph_input_data_chunk_granularity(self):
        """Chunk granularity captures one TransformerBlock callable per model chunk."""

        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        init_num_microbatches_calculator(
            rank=0,
            global_batch_size=4,
            micro_batch_size=2,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )

        seq_length = 128
        micro_batch_size = 2
        transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
            cuda_graph_granularity="chunk",
            cuda_graph_modules=[],
            use_te_rng_tracker=True,
            bf16=True,
            pipeline_dtype=torch.bfloat16,
        )

        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        model = [
            GPTModel(
                config=transformer_config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=1024,
                max_sequence_length=seq_length,
                parallel_output=True,
                position_embedding_type="rope",
            ).cuda()
        ]

        cuda_graph_helper = TECudaGraphHelper(
            model=model,
            config=transformer_config,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            optimizers=[],
        )

        assert cuda_graph_helper.num_layers_per_chunk == [1]
        assert len(cuda_graph_helper.flattened_callables) == 1
        assert isinstance(cuda_graph_helper.flattened_callables[0], TransformerBlock)

        sample_args, make_graphed_callables_kwargs = cuda_graph_helper._get_cuda_graph_input_data()
        sample_kwargs = make_graphed_callables_kwargs['sample_kwargs']

        assert len(sample_args) == 1
        assert len(sample_kwargs) == 1
        assert sample_args[0][0].shape == (seq_length, micro_batch_size, 64)
        assert sample_kwargs[0]["attention_mask"].shape == (
            micro_batch_size,
            1,
            seq_length,
            seq_length,
        )
        assert "rotary_pos_emb" in sample_kwargs[0]
        assert cuda_graph_helper.cuda_graph_chunk_slot_plan.num_slots_per_chunk == (1,)
        assert cuda_graph_helper.cuda_graph_chunk_slot_plan.forward_slot_by_virtual_microbatch == (
            0,
        )
        assert cuda_graph_helper.cuda_graph_chunk_slot_plan.backward_slot_by_virtual_microbatch == (
            0,
        )
        assert model[0].decoder.cuda_graph_chunk_slot_plan is (
            cuda_graph_helper.cuda_graph_chunk_slot_plan
        )
        set_current_microbatch(model[0], 0)
        assert model[0].decoder.cuda_graph_current_slot == 0
        assert model[0].decoder.cuda_graph_current_op == "forward"
        assert model[0].decoder.cuda_graph_forward_slot == 0
        assert model[0].decoder.cuda_graph_backward_slot == 0
        assert model[0].decoder.layers[0].cuda_graph_forward_slot == 0
        assert model[0].decoder.layers[0].cuda_graph_backward_slot == 0
        set_current_cuda_graph_slot(model[0], 0, forward=False)
        assert model[0].decoder.cuda_graph_current_slot == 0
        assert model[0].decoder.cuda_graph_current_op == "backward"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_cuda_graph_input_data_dynamic_microbatch_chunk_pp2(self):
        """Dynamic microbatch slots work with chunk granularity under PP 1F1B."""

        if Utils.world_size < 2:
            pytest.skip("Requires torchrun with at least 2 ranks for PP2.")

        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=2)
        runtime_num_microbatches = 8
        micro_batch_size = 1
        init_num_microbatches_calculator(
            rank=torch.distributed.get_rank(),
            global_batch_size=runtime_num_microbatches * micro_batch_size,
            micro_batch_size=micro_batch_size,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )

        seq_length = 64
        transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
            cuda_graph_granularity="chunk",
            cuda_graph_modules=[],
            cuda_graph_dynamic_microbatches=True,
            use_te_rng_tracker=True,
            bf16=True,
            pipeline_dtype=torch.bfloat16,
            pipeline_model_parallel_size=2,
        )

        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        model = [
            GPTModel(
                config=transformer_config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=1024,
                max_sequence_length=seq_length,
                parallel_output=True,
                position_embedding_type="rope",
            ).cuda()
        ]

        cuda_graph_helper = TECudaGraphHelper(
            model=model,
            config=transformer_config,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            optimizers=[],
        )

        sample_args, make_graphed_callables_kwargs = cuda_graph_helper._get_cuda_graph_input_data()

        assert cuda_graph_helper.num_layers_per_chunk == [1]
        assert len(cuda_graph_helper.flattened_callables) == 1
        assert cuda_graph_helper.num_microbatches == runtime_num_microbatches
        assert len(sample_args) == cuda_graph_helper.num_microbatches
        assert len(make_graphed_callables_kwargs['sample_kwargs']) == (
            cuda_graph_helper.num_microbatches
        )
        expected_slots = 2 if parallel_state.get_pipeline_model_parallel_rank() == 0 else 1
        expected_microbatch_1_slot = (
            1 if parallel_state.get_pipeline_model_parallel_rank() == 0 else 0
        )
        assert cuda_graph_helper.cuda_graph_chunk_slot_plan.num_slots_per_chunk == (
            expected_slots,
        )
        assert len(cuda_graph_helper.cuda_graph_chunk_slot_plan.forward_slot_by_virtual_microbatch) == (
            cuda_graph_helper.num_microbatches
        )
        assert model[0].decoder.cuda_graph_chunk_slot_plan is (
            cuda_graph_helper.cuda_graph_chunk_slot_plan
        )
        set_current_microbatch(model[0], 1)
        assert model[0].decoder.cuda_graph_current_slot == expected_microbatch_1_slot
        assert model[0].decoder.cuda_graph_current_op == "forward"
        assert model[0].decoder.cuda_graph_forward_slot == expected_microbatch_1_slot
        assert model[0].decoder.cuda_graph_backward_slot == expected_microbatch_1_slot
        set_current_cuda_graph_slot(model[0], 1, forward=False)
        assert model[0].decoder.cuda_graph_current_slot == expected_microbatch_1_slot
        assert model[0].decoder.cuda_graph_current_op == "backward"

        transformer_config.cuda_graph_num_microbatch_slots = runtime_num_microbatches + 2
        cuda_graph_helper = TECudaGraphHelper(
            model=model,
            config=transformer_config,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            optimizers=[],
        )
        sample_args, make_graphed_callables_kwargs = cuda_graph_helper._get_cuda_graph_input_data()

        assert cuda_graph_helper.num_microbatches == runtime_num_microbatches + 2
        assert len(sample_args) == cuda_graph_helper.num_microbatches
        assert len(make_graphed_callables_kwargs['sample_kwargs']) == (
            cuda_graph_helper.num_microbatches
        )
        assert len(cuda_graph_helper.cuda_graph_chunk_slot_plan.forward_slot_by_virtual_microbatch) == (
            cuda_graph_helper.num_microbatches
        )

    @pytest.mark.skipif(
        not (torch.cuda.is_available() and HAVE_TE and is_te_min_version("2.10.0")),
        reason="Chunk CUDA graph smoke requires CUDA and TransformerEngine >= 2.10.0",
    )
    def test_create_cudagraphs_thd_chunk_granularity_smoke(self):
        """TE can capture a whole TransformerBlock chunk with THD static inputs."""

        Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        init_num_microbatches_calculator(
            rank=0,
            global_batch_size=1,
            micro_batch_size=1,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )

        seq_length = 16
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=256,
            use_cpu_initialization=True,
            cuda_graph_impl="transformer_engine",
            cuda_graph_granularity="chunk",
            cuda_graph_modules=[],
            use_te_rng_tracker=True,
            bf16=True,
            pipeline_dtype=torch.bfloat16,
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=seq_length,
            pad_packed_seq_alignment=0,
            thd_max_num_seqs=4,
            moe_token_dispatcher_type="alltoall",
        )

        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        model = (
            GPTModel(
                config=transformer_config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=128,
                max_sequence_length=seq_length,
                parallel_output=True,
                position_embedding_type="rope",
            )
            .cuda()
            .bfloat16()
        )
        model.zero_grad_buffer = lambda *args, **kwargs: None

        cuda_graph_helper = TECudaGraphHelper(
            model=[model],
            config=transformer_config,
            seq_length=seq_length,
            micro_batch_size=1,
            optimizers=[],
        )
        cuda_graph_helper.create_cudagraphs()

        assert len(model.decoder.cuda_graphs) == 1
        cuda_graph_helper.delete_cuda_graphs()


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


class TestCudaGraphScheduleStageOrder:
    """Pure-Python coverage for warmup/steady/cooldown schedule boundaries."""

    @staticmethod
    def _stages(num_warmup_microbatches, num_scheduled_microbatches):
        return list(
            get_cuda_graph_schedule_stage_order_from_counts(
                num_warmup_microbatches, num_scheduled_microbatches
            )
        )

    def test_pp2_1f1b_stage_boundaries(self):
        assert self._stages(num_warmup_microbatches=1, num_scheduled_microbatches=4) == [
            "warmup",
            "steady",
            "steady",
            "steady",
            "steady",
            "steady",
            "steady",
            "cooldown",
        ]

    def test_single_microbatch_has_no_steady_region(self):
        assert self._stages(num_warmup_microbatches=1, num_scheduled_microbatches=1) == [
            "warmup",
            "cooldown",
        ]

    def test_single_microbatch_without_warmup_is_steady(self):
        assert self._stages(num_warmup_microbatches=0, num_scheduled_microbatches=1) == [
            "steady",
            "steady",
        ]

    def test_non_integer_entries_keep_position_stage(self):
        order = [1, 1, -1, -1.5, 1, -1, -1]
        assert TECudaGraphHelper._get_schedule_stage_order_from_order(order) == [
            "warmup",
            "warmup",
            "steady",
            "steady",
            "steady",
            "cooldown",
            "cooldown",
        ]


class TestChunkCudaGraphSlotPlan:
    """Pure-Python coverage for Megatron-owned chunk graph slot planning."""

    def test_pp2_rank0_reuses_two_slots(self):
        order = [1, 1, -1, 1, -1, 1, -1, -1]
        plan = build_chunk_cuda_graph_slot_plan(order, num_model_chunks=1)

        assert plan.num_slots_per_chunk == (2,)
        assert plan.slot_ids == (0, 1, 0, 0, 1, 1, 0, 1)
        assert plan.op_types == (
            "forward",
            "forward",
            "backward",
            "forward",
            "backward",
            "forward",
            "backward",
            "backward",
        )

    def test_pp2_rank1_only_needs_one_slot(self):
        order = [1, -1, 1, -1, 1, -1, 1, -1]
        plan = build_chunk_cuda_graph_slot_plan(order, num_model_chunks=1)

        assert plan.num_slots_per_chunk == (1,)
        assert plan.slot_ids == (0, 0, 0, 0, 0, 0, 0, 0)

    def test_slots_are_per_chunk(self):
        order = [1, 2, -2, 1, -1, -1]

        assert get_required_num_microbatch_slots_per_chunk(order, num_model_chunks=2) == (2, 1)

        plan = build_chunk_cuda_graph_slot_plan(order, num_model_chunks=2)
        assert plan.num_slots_per_chunk == (2, 1)
        assert plan.chunk_ids == (0, 1, 1, 0, 0, 0)
        assert plan.slot_ids == (0, 0, 0, 1, 0, 1)

    def test_non_integer_entries_are_auxiliary(self):
        order = [1, -1.5, -1]
        plan = build_chunk_cuda_graph_slot_plan(order, num_model_chunks=1)

        assert plan.num_slots_per_chunk == (1,)
        assert plan.op_types == ("forward", "aux", "backward")
        assert plan.slot_ids == (0, None, 0)
        assert plan.chunk_ids == (0, None, 0)

    def test_invalid_order_asserts(self):
        with pytest.raises(AssertionError, match="negative outstanding"):
            build_chunk_cuda_graph_slot_plan([-1], num_model_chunks=1)

    def test_schedule_table_preserves_virtual_microbatch_slots(self):
        schedule_table = [(0, 0), (1, 0), (2, 0), (3, 0)]
        plan = build_chunk_cuda_graph_slot_plan_from_schedule(
            num_warmup_microbatches=1,
            num_model_chunks=1,
            schedule_table=schedule_table,
        )

        assert plan.order == (1, 1, -1, 1, -1, 1, -1, -1)
        assert plan.virtual_microbatch_ids == (0, 1, 0, 2, 1, 3, 2, 3)
        assert plan.microbatch_ids == (0, 1, 0, 2, 1, 3, 2, 3)
        assert plan.slot_ids == (0, 1, 0, 0, 1, 1, 0, 1)
        assert plan.forward_slot_by_virtual_microbatch == (0, 1, 0, 1)
        assert plan.backward_slot_by_virtual_microbatch == (0, 1, 0, 1)
        assert plan.forward_slot_by_chunk_microbatch == ((0, 1, 0, 1),)
        assert plan.backward_slot_by_chunk_microbatch == ((0, 1, 0, 1),)
        assert plan.get_forward_slot(0, 2) == 0
        assert plan.get_backward_slot(0, 3) == 1

    def test_schedule_table_preserves_chunk_local_microbatch_slots(self):
        schedule_table = [(0, 0), (1, 0), (0, 1), (1, 1)]
        plan = build_chunk_cuda_graph_slot_plan_from_schedule(
            num_warmup_microbatches=3,
            num_model_chunks=2,
            schedule_table=schedule_table,
        )

        assert plan.forward_slot_by_chunk_microbatch == ((0, 1), (0, 1))
        assert plan.backward_slot_by_chunk_microbatch == ((0, 1), (0, 1))
        assert plan.get_forward_slot(1, 0) == 0
        assert plan.get_backward_slot(1, 1) == 1

    def test_required_slots_stabilize_for_larger_microbatch_count(self):
        def make_schedule_table(num_microbatches):
            return [
                (microbatch_id, chunk_id)
                for microbatch_id in range(num_microbatches)
                for chunk_id in range(2)
            ]

        short_plan = build_chunk_cuda_graph_slot_plan_from_schedule(
            num_warmup_microbatches=3,
            num_model_chunks=2,
            schedule_table=make_schedule_table(4),
        )
        long_plan = build_chunk_cuda_graph_slot_plan_from_schedule(
            num_warmup_microbatches=3,
            num_model_chunks=2,
            schedule_table=make_schedule_table(12),
        )

        assert short_plan.num_slots_per_chunk == (3, 2)
        assert long_plan.num_slots_per_chunk == short_plan.num_slots_per_chunk
        assert all(num_slots <= 4 for num_slots in long_plan.num_slots_per_chunk)


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

    return HAVE_DEEP_EP


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


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
            vocab_size=args.vocab_size,
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
        args.vocab_size = 1024
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
        set_global_variables(args, False)
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
        self, ep_size, cuda_graph_impl, cuda_graph_modules, cuda_graph_warmup_steps, **kwargs
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

        gpt_model, optimizer, _ = setup_model_and_optimizer(
            self.model_provider, ModelType.encoder_or_decoder
        )
        assert len(gpt_model) == 1  # Assume only one model in the model provider.

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

            gpt_model[0].set_is_first_microbatch()
            output = gpt_model[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
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
    @pytest.mark.parametrize("moe_dispatcher_type", ["alltoall", "deepep", "hybridep"])
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
        else:
            extra_kwargs["moe_token_dispatcher_type"] = moe_dispatcher_type
        if not moe_dropless_dispatcher:
            if moe_dispatcher_type == "deepep":
                pytest.skip("Deep EP doesn't support drop&pad MoE")
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
            if (moe_dropless_dispatcher or moe_dispatcher_type == "hybridep") and (
                cuda_graph_modules is None or CudaGraphModule.moe in cuda_graph_modules
            ):
                # Dropless MoE or Hybrid EP doesn't work with "moe" scope cudagraph. Skip.
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


class _SimpleModule(MegatronModule):
    """Minimal MegatronModule for testing CudaGraphManager with function_name."""

    def __init__(self, config):
        super().__init__(config)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def my_op(self, x):
        return self.linear(x)


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
