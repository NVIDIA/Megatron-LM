# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Megatron-FSDP CUDA graph argument and config wiring."""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from megatron.core.distributed.fsdp.mcore_fsdp_adapter import _validate_cuda_graph_config
from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import (
    _get_allocator_namespace,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.training.arguments import (
    _get_cuda_graph_recompute_overlap,
    _validate_megatron_fsdp_cuda_graph_buffers,
)
from megatron.training.training import (
    _capture_cudagraphs_with_forward_pre_hook_restore,
    _restore_forward_pre_hook_after_cuda_graph_capture,
    get_megatron_ddp_config,
    wrap_model_chunks_with_ddp,
)

_RECOMPUTE_MODULES = ["shared_experts", "moe_act", "mlp", "layernorm", "moe"]


@pytest.mark.parametrize(
    ("cuda_graph_modules", "expected_overlap"),
    [
        pytest.param([], _RECOMPUTE_MODULES, id="whole-layer"),
        pytest.param([CudaGraphModule.mlp], ["mlp", "layernorm"], id="mlp"),
        pytest.param(
            [CudaGraphModule.moe], ["shared_experts", "moe_act", "layernorm", "moe"], id="moe"
        ),
        pytest.param([CudaGraphModule.moe_router], ["shared_experts", "moe"], id="moe-router"),
        pytest.param(
            [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
            ["shared_experts", "moe"],
            id="moe-router-and-preprocess",
        ),
        pytest.param([CudaGraphModule.attn], [], id="attention"),
        pytest.param(
            [CudaGraphModule.mlp, CudaGraphModule.moe], _RECOMPUTE_MODULES, id="mlp-and-moe"
        ),
    ],
)
def test_recompute_overlap_uses_normalized_cuda_graph_enums(cuda_graph_modules, expected_overlap):
    """Each normalized capture scope reports only recompute modules it captures."""
    assert (
        _get_cuda_graph_recompute_overlap(cuda_graph_modules, _RECOMPUTE_MODULES)
        == expected_overlap
    )


@pytest.mark.parametrize(
    ("cuda_graph_modules", "expected_overlap"),
    [
        pytest.param([], ["gdn"], id="whole-layer"),
        pytest.param([CudaGraphModule.attn], ["gdn"], id="attention"),
        pytest.param([CudaGraphModule.moe_router], [], id="router-only"),
    ],
)
def test_gdn_recompute_overlap_follows_attention_scope(cuda_graph_modules, expected_overlap):
    """Whole-GDN recompute overlaps only scopes that capture GDN attention."""
    assert (
        _get_cuda_graph_recompute_overlap(
            cuda_graph_modules, ["gdn"], captures_gdn_attention=True
        )
        == expected_overlap
    )
    assert _get_cuda_graph_recompute_overlap(cuda_graph_modules, ["gdn"]) == []


def test_router_overlap_excludes_eager_shared_experts():
    """Shared experts outside the router graph must not produce a false warning."""
    assert _get_cuda_graph_recompute_overlap(
        [CudaGraphModule.moe_router], _RECOMPUTE_MODULES, router_captures_shared_experts=False
    ) == ["moe"]


@pytest.mark.parametrize(
    ("cuda_graph_impl", "fsdp_double_buffer", "nccl_ub", "expected_error"),
    [
        pytest.param("transformer_engine", False, False, None, id="te-planned-allocator"),
        pytest.param(
            "transformer_engine", True, False, "does not support.*--fsdp-double-buffer", id="te-db"
        ),
        pytest.param(
            "transformer_engine",
            False,
            True,
            "does not yet support --use-nccl-ub",
            id="te-nccl-ub",
        ),
        pytest.param(
            "local",
            False,
            False,
            "does not yet support --cuda-graph-impl=local",
            id="local",
        ),
        pytest.param("full_iteration", False, False, None, id="full-iteration"),
    ],
)
def test_megatron_fsdp_cuda_graph_buffer_validation(
    cuda_graph_impl, fsdp_double_buffer, nccl_ub, expected_error
):
    """Validate the supported per-layer graph and allocator combinations."""
    args = SimpleNamespace(
        cuda_graph_impl=cuda_graph_impl,
        use_megatron_fsdp=True,
        overlap_param_gather=True,
        overlap_grad_reduce=True,
        data_parallel_sharding_strategy="optim_grads_params",
        fsdp_double_buffer=fsdp_double_buffer,
        nccl_ub=nccl_ub,
    )

    if expected_error is None:
        _validate_megatron_fsdp_cuda_graph_buffers(args)
    else:
        with pytest.raises(ValueError, match=expected_error):
            _validate_megatron_fsdp_cuda_graph_buffers(args)


@pytest.mark.parametrize(
    ("strategy", "expected_error"),
    [
        pytest.param("no_shard", None, id="persistent-no-shard"),
        pytest.param("optim", None, id="persistent-optim"),
        pytest.param("optim_grads", "does not yet support.*optim_grads", id="sharded-grad"),
        pytest.param("optim_grads_params", None, id="fully-sharded"),
    ],
)
def test_te_planned_sharding_strategy_matrix(strategy, expected_error):
    """Only persistent or pre-replay claimed main_grad strategies are accepted."""
    args = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        use_megatron_fsdp=True,
        overlap_param_gather=True,
        overlap_grad_reduce=True,
        data_parallel_sharding_strategy=strategy,
        fsdp_double_buffer=False,
        nccl_ub=False,
    )

    if expected_error is None:
        _validate_megatron_fsdp_cuda_graph_buffers(args)
    else:
        with pytest.raises(ValueError, match=expected_error):
            _validate_megatron_fsdp_cuda_graph_buffers(args)


@pytest.mark.parametrize(
    ("overlap_param_gather", "overlap_grad_reduce", "expected_error"),
    [
        pytest.param(False, True, "--overlap-param-gather", id="param-gather"),
        pytest.param(True, False, "--overlap-grad-reduce", id="grad-reduce"),
        pytest.param(True, True, None, id="both-enabled"),
    ],
)
def test_te_planned_fully_sharded_requires_communication_overlap(
    overlap_param_gather, overlap_grad_reduce, expected_error
):
    """Planned lifetimes require per-unit AG and RS scheduling."""
    args = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        use_megatron_fsdp=True,
        overlap_param_gather=overlap_param_gather,
        overlap_grad_reduce=overlap_grad_reduce,
        data_parallel_sharding_strategy="optim_grads_params",
        fsdp_double_buffer=False,
        nccl_ub=False,
    )

    if expected_error is None:
        _validate_megatron_fsdp_cuda_graph_buffers(args)
    else:
        with pytest.raises(ValueError, match=expected_error):
            _validate_megatron_fsdp_cuda_graph_buffers(args)



@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("cuda_graph_dynamic_microbatches", True, id="dynamic-graph-slots"),
        pytest.param("variable_seq_lengths", True, id="variable-sequence-lengths"),
        pytest.param("sequence_packing_scheduler", "dp_balanced", id="sequence-packing"),
        pytest.param("rl_use_sequence_packing", True, id="rl-sequence-packing"),
    ],
)
def test_te_planned_rejects_dynamic_microbatch_topology_inputs(field, value):
    """Dynamic planned schedules require the deferred retrace lifecycle."""
    args = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        use_megatron_fsdp=True,
        overlap_param_gather=True,
        overlap_grad_reduce=True,
        data_parallel_sharding_strategy="optim_grads_params",
        fsdp_double_buffer=False,
        nccl_ub=False,
        cuda_graph_dynamic_microbatches=False,
        variable_seq_lengths=False,
        sequence_packing_scheduler=None,
        rl_use_sequence_packing=False,
    )
    setattr(args, field, value)

    with pytest.raises(ValueError, match="does not support dynamic microbatch/topology"):
        _validate_megatron_fsdp_cuda_graph_buffers(args)

@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
def test_per_layer_restrictions_do_not_leak_into_other_ddp_backends(cuda_graph_impl):
    """Megatron-FSDP-specific restrictions must not leak into other DDP backends."""
    args = SimpleNamespace(
        cuda_graph_impl=cuda_graph_impl,
        use_megatron_fsdp=False,
        fsdp_double_buffer=False,
        nccl_ub=True,
    )

    _validate_megatron_fsdp_cuda_graph_buffers(args)


def test_full_iteration_does_not_use_per_layer_fsdp_allocator_restrictions():
    """Full-iteration capture owns a different allocation lifecycle."""
    args = SimpleNamespace(
        cuda_graph_impl="full_iteration",
        use_megatron_fsdp=True,
        data_parallel_sharding_strategy="optim_grads",
        fsdp_double_buffer=True,
        nccl_ub=True,
    )

    _validate_megatron_fsdp_cuda_graph_buffers(args)


@pytest.mark.parametrize(
    (
        "cuda_graph_impl",
        "graph_mode",
        "planned",
        "nccl_ub",
        "strategy",
        "double_buffer",
        "expected_error",
    ),
    [
        pytest.param("none", False, False, True, "optim_grads", False, None, id="none"),
        pytest.param(
            "none",
            False,
            True,
            False,
            "optim_grads_params",
            False,
            "supported only with.*transformer_engine",
            id="planned-without-te",
        ),
        pytest.param(
            "full_iteration",
            False,
            False,
            False,
            "optim_grads",
            False,
            "requires megatron_fsdp_cuda_graph_mode",
            id="full-iteration-without-graph-mode",
        ),
        pytest.param(
            "full_iteration",
            True,
            False,
            True,
            "optim_grads",
            True,
            None,
            id="full-iteration",
        ),
        pytest.param(
            "local",
            True,
            False,
            False,
            "optim_grads_params",
            False,
            "does not yet support",
            id="local",
        ),
        pytest.param(
            "transformer_engine",
            False,
            True,
            False,
            "optim_grads_params",
            False,
            "requires megatron_fsdp_cuda_graph_mode",
            id="te-without-graph-mode",
        ),
        pytest.param(
            "transformer_engine",
            True,
            False,
            False,
            "optim_grads_params",
            False,
            "requires megatron_fsdp_use_planned_allocator",
            id="te-without-planned",
        ),
        pytest.param(
            "transformer_engine",
            True,
            True,
            True,
            "optim_grads_params",
            False,
            "does not yet support NCCL user buffers",
            id="te-nccl-ub",
        ),
        pytest.param(
            "transformer_engine",
            True,
            True,
            False,
            "optim_grads",
            False,
            "does not yet support optim_grads",
            id="te-optim-grads",
        ),
        pytest.param(
            "transformer_engine",
            True,
            True,
            False,
            "optim_grads_params",
            True,
            "does not support fsdp_double_buffer=True",
            id="te-double-buffer",
        ),
        pytest.param(
            "transformer_engine",
            True,
            True,
            False,
            "no_shard",
            False,
            None,
            id="te-no-shard-supported",
        ),
        pytest.param(
            "transformer_engine",
            True,
            True,
            False,
            "optim",
            False,
            None,
            id="te-optim-supported",
        ),
        pytest.param(
            "transformer_engine",
            True,
            True,
            False,
            "optim_grads_params",
            False,
            None,
            id="te-supported",
        ),
    ],
)
def test_programmatic_fsdp_wrapper_revalidates_cuda_graph_config(
    cuda_graph_impl,
    graph_mode,
    planned,
    nccl_ub,
    strategy,
    double_buffer,
    expected_error,
):
    """Programmatic callers cannot bypass CLI argument validation."""
    config = SimpleNamespace(cuda_graph_impl=cuda_graph_impl)
    ddp_config = SimpleNamespace(
        megatron_fsdp_cuda_graph_mode=graph_mode,
        megatron_fsdp_use_planned_allocator=planned,
        overlap_param_gather=True,
        overlap_grad_reduce=True,
        nccl_ub=nccl_ub,
        data_parallel_sharding_strategy=strategy,
        fsdp_double_buffer=double_buffer,
    )

    if expected_error is None:
        _validate_cuda_graph_config(config, ddp_config)
    else:
        with pytest.raises(ValueError, match=expected_error):
            _validate_cuda_graph_config(config, ddp_config)


@pytest.mark.parametrize(
    ("disabled_overlap", "expected_error"),
    [
        pytest.param(
            "overlap_param_gather", "requires overlap_param_gather=True", id="param-gather"
        ),
        pytest.param(
            "overlap_grad_reduce", "requires overlap_grad_reduce=True", id="grad-reduce"
        ),
    ],
)
def test_programmatic_te_planned_fully_sharded_requires_overlap(
    disabled_overlap, expected_error
):
    """Programmatic setup cannot rely on later constructor-side overlap mutation."""
    config = SimpleNamespace(cuda_graph_impl="transformer_engine")
    ddp_config = SimpleNamespace(
        megatron_fsdp_cuda_graph_mode=True,
        megatron_fsdp_use_planned_allocator=True,
        overlap_param_gather=True,
        overlap_grad_reduce=True,
        nccl_ub=False,
        data_parallel_sharding_strategy="optim_grads_params",
        fsdp_double_buffer=False,
    )
    setattr(ddp_config, disabled_overlap, False)

    with pytest.raises(ValueError, match=expected_error):
        _validate_cuda_graph_config(config, ddp_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("cuda_graph_dynamic_microbatches", True, id="dynamic-graph-slots"),
        pytest.param("variable_seq_lengths", True, id="variable-sequence-lengths"),
        pytest.param("sequence_packing_scheduler", "dp_balanced", id="sequence-packing"),
        pytest.param("rl_use_sequence_packing", True, id="rl-sequence-packing"),
    ],
)
def test_programmatic_te_planned_rejects_dynamic_topology(field, value):
    """Programmatic callers cannot bypass the deferred dynamic-topology boundary."""
    config = SimpleNamespace(
        cuda_graph_impl="transformer_engine",
        cuda_graph_dynamic_microbatches=False,
        variable_seq_lengths=False,
        sequence_packing_scheduler=None,
        rl_use_sequence_packing=False,
    )
    ddp_config = SimpleNamespace(
        megatron_fsdp_cuda_graph_mode=True,
        megatron_fsdp_use_planned_allocator=True,
        overlap_param_gather=True,
        overlap_grad_reduce=True,
        nccl_ub=False,
        data_parallel_sharding_strategy="optim_grads_params",
        fsdp_double_buffer=False,
    )
    setattr(config, field, value)

    with pytest.raises(ValueError, match="do not support dynamic microbatch/topology"):
        _validate_cuda_graph_config(config, ddp_config)


def test_feature_contract_is_checked_before_disabling_forward_hooks(monkeypatch):
    """An unsupported TE build must leave caller-owned DDP hooks untouched."""
    events = []

    def fail_contract():
        events.append("validate-contract")
        raise RuntimeError("unsupported capture-time-hooks protocol")

    helper = SimpleNamespace(
        validate_capture_feature_contract=fail_contract,
        create_cudagraphs=lambda: events.append("capture"),
        cuda_graph_set_manual_hooks=lambda: events.append("manual-hooks"),
    )
    monkeypatch.setattr(
        "megatron.training.training.disable_forward_pre_hook",
        lambda model, param_sync: events.append("disable-hooks"),
    )
    monkeypatch.setattr(
        "megatron.training.training.enable_forward_pre_hook",
        lambda model: events.append("enable-hooks"),
    )

    with pytest.raises(RuntimeError, match="capture-time-hooks protocol"):
        _capture_cudagraphs_with_forward_pre_hook_restore(["model"], helper, True)

    assert events == ["validate-contract"]


def test_capture_failure_restores_forward_pre_hooks_without_installing_manual_hooks(monkeypatch):
    """A failed capture must not leak caller-disabled DDP hooks."""
    events = []

    def fail_capture():
        events.append("capture")
        raise RuntimeError("capture failed")

    helper = SimpleNamespace(
        create_cudagraphs=fail_capture,
        cuda_graph_set_manual_hooks=lambda: events.append("manual-hooks"),
    )
    monkeypatch.setattr(
        "megatron.training.training.disable_forward_pre_hook",
        lambda model, param_sync: events.append(("disable", model, param_sync)),
    )
    monkeypatch.setattr(
        "megatron.training.training.enable_forward_pre_hook",
        lambda model: events.append(("enable", model)),
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        _capture_cudagraphs_with_forward_pre_hook_restore(["model"], helper, True)

    assert events == [("disable", ["model"], False), "capture", ("enable", ["model"])]


def test_capture_failure_remains_primary_when_hook_restore_also_fails(monkeypatch):
    """Hook cleanup failures are reported without replacing the capture failure."""
    events = []

    def fail_capture():
        raise RuntimeError("capture failed")

    def fail_restore(model):
        events.append(("enable", model))
        raise RuntimeError("restore failed")

    helper = SimpleNamespace(
        create_cudagraphs=fail_capture,
        cuda_graph_set_manual_hooks=lambda: events.append("manual-hooks"),
    )
    monkeypatch.setattr(
        "megatron.training.training.disable_forward_pre_hook",
        lambda model, param_sync: events.append(("disable", model)),
    )
    monkeypatch.setattr(
        "megatron.training.training.enable_forward_pre_hook",
        fail_restore,
    )

    with pytest.raises(RuntimeError, match="capture failed") as exc_info:
        _capture_cudagraphs_with_forward_pre_hook_restore(["model"], helper, True)

    assert events == [("disable", ["model"]), ("enable", ["model"])]
    assert any("restore failed" in note for note in exc_info.value.__notes__)
    assert "manual-hooks" not in events
def test_successful_capture_restores_then_installs_manual_hooks(monkeypatch):


    """Manual hook installation happens only after capture and hook restoration succeed."""
    events = []
    helper = SimpleNamespace(
        create_cudagraphs=lambda: events.append("capture"),
        cuda_graph_set_manual_hooks=lambda: events.append("manual-hooks"),
    )
    monkeypatch.setattr(
        "megatron.training.training.disable_forward_pre_hook",
        lambda model, param_sync: events.append("disable"),
    )
    monkeypatch.setattr(
        "megatron.training.training.enable_forward_pre_hook", lambda model: events.append("enable")
    )

    _capture_cudagraphs_with_forward_pre_hook_restore(["model"], helper, True)

    assert events == ["disable", "capture", "enable", "manual-hooks"]


def test_partial_disable_failure_restores_already_disabled_chunks(monkeypatch):
    """A later chunk's disable failure must restore every earlier chunk."""
    events = []

    def disable_one(model_chunks, param_sync):
        assert len(model_chunks) == 1
        model_chunk = model_chunks[0]
        events.append(("disable", model_chunk, param_sync))
        if model_chunk == "second":
            raise RuntimeError("second disable failed")

    helper = SimpleNamespace(
        create_cudagraphs=lambda: events.append("capture"),
        cuda_graph_set_manual_hooks=lambda: events.append("manual-hooks"),
    )
    monkeypatch.setattr("megatron.training.training.disable_forward_pre_hook", disable_one)
    monkeypatch.setattr(
        "megatron.training.training.enable_forward_pre_hook",
        lambda model_chunks: events.append(("enable", model_chunks[0])),
    )

    with pytest.raises(RuntimeError, match="second disable failed"):
        _capture_cudagraphs_with_forward_pre_hook_restore(["first", "second"], helper, True)

    assert events == [
        ("disable", "first", False),
        ("disable", "second", False),
        ("enable", "second"),
        ("enable", "first"),
    ]


def test_restore_forward_pre_hook_rebuilds_partial_handle_map(monkeypatch):
    """Surviving handles are removed before the complete DDP hook set is rebuilt."""
    events = []

    class Handle:
        def remove(self):
            events.append("remove-survivor")

    model_chunk = SimpleNamespace(
        remove_forward_pre_hook_handles={"survivor": Handle()}
    )

    def enable_one(model_chunks):
        assert model_chunks == [model_chunk]
        assert model_chunk.remove_forward_pre_hook_handles == {}
        events.append("enable-complete-set")

    monkeypatch.setattr(
        "megatron.training.training.enable_forward_pre_hook",
        enable_one,
    )

    _restore_forward_pre_hook_after_cuda_graph_capture(model_chunk)

    assert events == ["remove-survivor", "enable-complete-set"]


def _make_ddp_args(cuda_graph_impl):
    """Build the minimal argument namespace consumed by get_megatron_ddp_config."""
    return SimpleNamespace(
        use_torch_fsdp2=False,
        accumulate_allreduce_grads_in_fp32=False,
        check_for_nan_in_loss_and_grad=False,
        check_for_large_grads=False,
        ddp_num_buckets=None,
        ddp_bucket_size=None,
        ddp_pad_buckets_for_high_nccl_busbw=False,
        ddp_reduce_scatter_with_fp32_accumulation=False,
        ddp_param_name_patterns_for_fp32_local_accumulation=(),
        ddp_average_in_collective=False,
        megatron_fsdp_main_params_dtype=None,
        megatron_fsdp_main_grads_dtype=None,
        megatron_fsdp_grad_comm_dtype=None,
        use_precision_aware_optimizer=False,
        use_megatron_fsdp=True,
        cuda_graph_impl=cuda_graph_impl,
    )


@pytest.mark.parametrize(
    ("cuda_graph_impl", "expected_graph_mode", "expected_planned_allocator"),
    [
        pytest.param("none", False, False, id="disabled"),
        pytest.param("local", True, False, id="local"),
        pytest.param("transformer_engine", True, True, id="transformer-engine"),
        pytest.param("full_iteration", True, False, id="full-iteration"),
    ],
)
def test_planned_allocator_is_enabled_only_for_te_per_layer_capture(
    cuda_graph_impl, expected_graph_mode, expected_planned_allocator
):
    """General graph-safe mode remains independent from TE allocation planning."""
    config = get_megatron_ddp_config(_make_ddp_args(cuda_graph_impl))

    assert config.megatron_fsdp_cuda_graph_mode is expected_graph_mode
    assert config.megatron_fsdp_use_planned_allocator is expected_planned_allocator


class _RecordingDP:
    """Minimal DDP stand-in that records the original module."""

    def __init__(self, *, module, **kwargs):
        self.module = module
        self.allocator_namespace = _get_allocator_namespace(module)


def test_model_chunks_receive_distinct_namespace_labels():
    """VPP chunks receive distinct human-readable labels before unique PGB IDs are appended."""
    chunks = [nn.Linear(2, 2), nn.Linear(2, 2)]
    ddp_config = SimpleNamespace(bucket_size=None, use_distributed_optimizer=False)

    wrapped = wrap_model_chunks_with_ddp(chunks, object(), ddp_config, DP=_RecordingDP)

    assert [chunk.module._megatron_fsdp_buffer_namespace for chunk in wrapped] == [
        "model_chunk_0",
        "model_chunk_1",
    ]


def test_separate_single_chunk_wraps_get_unique_allocator_namespaces():
    """Independent model groups cannot collide even when both use model_chunk_0."""
    ddp_config = SimpleNamespace(bucket_size=None, use_distributed_optimizer=False)

    first = wrap_model_chunks_with_ddp([nn.Linear(2, 2)], object(), ddp_config, DP=_RecordingDP)[0]
    second = wrap_model_chunks_with_ddp([nn.Linear(2, 2)], object(), ddp_config, DP=_RecordingDP)[0]

    assert first.module._megatron_fsdp_buffer_namespace == "model_chunk_0"
    assert second.module._megatron_fsdp_buffer_namespace == "model_chunk_0"
    assert first.allocator_namespace != second.allocator_namespace
    assert first.allocator_namespace.startswith("model_chunk_0_param_and_grad_buffer_")
    assert second.allocator_namespace.startswith("model_chunk_0_param_and_grad_buffer_")


def test_wrap_preserves_caller_provided_namespace_label():
    """Upcycling and other callers may provide a more descriptive chunk label."""
    chunk = nn.Linear(2, 2)
    chunk._megatron_fsdp_buffer_namespace = "upcycling_source"
    ddp_config = SimpleNamespace(bucket_size=None, use_distributed_optimizer=False)

    wrapped = wrap_model_chunks_with_ddp([chunk], object(), ddp_config, DP=_RecordingDP)[0]

    assert wrapped.module._megatron_fsdp_buffer_namespace == "upcycling_source"
    assert wrapped.allocator_namespace.startswith("upcycling_source_param_and_grad_buffer_")
