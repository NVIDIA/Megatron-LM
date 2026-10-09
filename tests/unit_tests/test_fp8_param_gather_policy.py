# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FP8 gather policy, compact storage support, and allocation invariants."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core import fp4_utils, fp8_utils

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.parametrize("layout", [False, True])
@pytest.mark.parametrize("fp8_param_gather", [False, True])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("owner", [False, True])
@pytest.mark.parametrize("tp_sharded", [False, True])
@pytest.mark.parametrize(
    "kind", ["mxfp8", "grouped_mxfp8", "blockwise", "bf16", "fp32", "float8", "nvfp4"]
)
def test_reuse_policy(monkeypatch, layout, fp8_param_gather, reuse, owner, tp_sharded, kind):
    """Only recipe, ownership, and the existing MXFP8 opt-in select staging."""
    param = SimpleNamespace(
        kind=kind, is_managed_by_layer_wise_optimizer=owner, tensor_model_parallel=tp_sharded
    )
    config = SimpleNamespace(
        fp8_param_gather=fp8_param_gather,
        reuse_grad_buf_for_mxfp8_param_ag=reuse,
        use_layer_wise_param_layout=layout,
    )
    monkeypatch.setattr(fp8_utils, "is_mxfp8tensor", lambda p: p.kind == "mxfp8")
    monkeypatch.setattr(fp8_utils, "is_grouped_mxfp8tensor", lambda p: p.kind == "grouped_mxfp8")
    monkeypatch.setattr(fp8_utils, "is_blockwise_float8tensor", lambda p: p.kind == "blockwise")
    expected = fp8_param_gather and (
        (kind in ("mxfp8", "grouped_mxfp8") and reuse) or (kind == "blockwise" and owner)
    )
    assert fp8_utils.uses_grad_buffer_for_fp8_param_gather(param, config) is expected
    # Grouped MXFP8 follows the global reuse opt-in but is not a plain whole-param
    # destination: its existing grouped storage path handles member offsets separately.
    assert fp8_utils.is_layerwise_fp8_param(param) is (kind in ("mxfp8", "blockwise"))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("owner", [False, True])
def test_high_precision_never_reuses_grad_buffer(dtype, owner):
    param = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
    param.is_managed_by_layer_wise_optimizer = owner
    config = SimpleNamespace(fp8_param_gather=True, reuse_grad_buf_for_mxfp8_param_ag=True)
    assert not fp8_utils.uses_grad_buffer_for_fp8_param_gather(param, config)
    assert not fp8_utils.is_layerwise_fp8_param(param)


@pytest.mark.parametrize("kind", ["float8", "nvfp4", "grouped_mxfp8", "grouped_bf16"])
def test_compact_copy_back_rejects_unsupported_storage_before_mutation(monkeypatch, kind):
    quantizer = Mock()
    supported = SimpleNamespace(
        kind="mxfp8", data=SimpleNamespace(_get_quantizer=lambda: quantizer)
    )
    unsupported = SimpleNamespace(kind=kind)
    monkeypatch.setattr(fp8_utils, "is_mxfp8tensor", lambda p: p.kind == "mxfp8")
    monkeypatch.setattr(fp8_utils, "is_blockwise_float8tensor", lambda p: False)
    monkeypatch.setattr(fp8_utils, "is_grouped_tensor", lambda p: p.kind.startswith("grouped"))
    monkeypatch.setattr(fp8_utils, "is_float8tensor", lambda p: p.kind != "grouped_bf16")
    monkeypatch.setattr(fp4_utils, "is_nvfp4tensor", lambda p: p.kind == "nvfp4")
    copy = Mock()
    monkeypatch.setattr(fp8_utils, "copy_tensors_to_quantized_params", copy)
    with pytest.raises(TypeError, match="LayerWise|GroupedTensor|NVFP4|MXFP8|Float8Blockwise"):
        fp8_utils.copy_back_gathered_bf16_into_fp8_params(
            [supported, unsupported], [torch.zeros(1), torch.zeros(1)]
        )
    quantizer.set_usage.assert_not_called()
    copy.assert_not_called()


def require_fp8_recipe(recipe):
    """Check installed TE/device support without a Hopper-only blockwise exclusion.

    Once generic FP8 support and the recipe class exist, run the real recipe. An
    unsupported kernel on a particular TE build must fail with its actual diagnostic,
    rather than quietly counting every Blackwell blockwise case as skipped coverage.
    """
    from transformer_engine.pytorch.fp8 import check_fp8_support

    supported, reason = check_fp8_support()
    if not supported:
        pytest.skip(f"Transformer Engine reports FP8 unavailable: {reason}")
    if recipe == "mxfp8" and torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell or newer; run this case on GB200")
    if recipe == "blockwise":
        try:
            from transformer_engine.common.recipe import Float8BlockScaling
            from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
                Float8BlockwiseQTensor,
            )
        except ImportError as exc:
            pytest.fail(f"Blockwise coverage requires a TE build with Float8BlockScaling: {exc}")
        assert Float8BlockScaling is not None and Float8BlockwiseQTensor is not None


def assert_param_storage_policy(ddp, args):
    """Check actual allocations and optimizer routing, independently of forward numerics."""
    counts = {"muon_fp8": 0, "adam_blockwise": 0, "high_precision": 0}
    compact_layerwise = (
        args.use_layer_wise_distributed_optimizer and not args.use_layer_wise_param_layout
    )
    for buffer in ddp.buffers + ddp.expert_parallel_buffers:
        for bucket in buffer.buckets:
            owners = {
                getattr(p, "is_managed_by_layer_wise_optimizer", False) for p in bucket.params
            }
            assert len(owners) == 1, "A bucket must not mix optimizer ownership"
            owner = owners.pop()
            # The tag selects Muon staging, not whole-parameter ownership: compact
            # LayerWise also owns the scalar Adam fallback, as on main.
            assert buffer.ddp_config.use_distributed_optimizer is (not compact_layerwise)
            reused = [
                p
                for p in bucket.params
                if fp8_utils.uses_grad_buffer_for_fp8_param_gather(p, ddp.ddp_config)
            ]
            assert bucket.reuse_grad_buffer_for_param_ag == bool(reused)
            grad_ptr = bucket.grad_data.untyped_storage().data_ptr()
            if reused and bucket.param_data is not None:
                assert bucket.param_data.untyped_storage().data_ptr() == grad_ptr
            for param in bucket.params:
                if fp8_utils.is_float8tensor(
                    param
                ) or fp8_utils.is_grouped_tensor_with_quantized_storage(param):
                    if owner:
                        counts["muon_fp8"] += 1
                    if (
                        fp8_utils.is_blockwise_float8tensor(param)
                        and not owner
                        and buffer.ddp_config.use_distributed_optimizer
                    ):
                        counts["adam_blockwise"] += 1
                        assert not bucket.reuse_grad_buffer_for_param_ag
                        assert buffer.param_dtype == torch.uint8
                        assert (
                            bucket.param_data is not None and bucket.param_data.dtype == torch.uint8
                        )
                        assert bucket.param_data.untyped_storage().data_ptr() != grad_ptr
                    payloads = [
                        getattr(holder, attr, None)
                        for holder in (param, param.data)
                        for attr in (
                            "_data",
                            "_rowwise_data",
                            "_columnwise_data",
                            "rowwise_data",
                            "columnwise_data",
                        )
                    ]
                    payloads = [value for value in payloads if torch.is_tensor(value)]
                    assert payloads, "Quantized storage was not exercised"
                    assert all(value.untyped_storage().data_ptr() != grad_ptr for value in payloads)
                else:
                    counts["high_precision"] += 1
                    assert not fp8_utils.uses_grad_buffer_for_fp8_param_gather(
                        param, ddp.ddp_config
                    )
                    assert param.data.untyped_storage().data_ptr() != grad_ptr
                    if buffer.ddp_config.use_distributed_optimizer:
                        assert bucket.param_data is not None
                        assert (
                            param.data.untyped_storage().data_ptr()
                            == bucket.param_data.untyped_storage().data_ptr()
                        )
                    elif not reused:
                        assert bucket.param_data is None
    assert counts[
        "high_precision"
    ], "Expected embeddings/norms with persistent high-precision storage"
    return counts


@pytest.mark.parametrize("completion", ["sync", "wait", "force"])
@pytest.mark.parametrize("dp_size", [1, 2, 3])
@pytest.mark.parametrize("has_fp8", [False, True])
@pytest.mark.parametrize("mixed_dtypes", [False, True])
def test_compact_transport_lifecycle(monkeypatch, completion, dp_size, has_fp8, mixed_dtypes):
    """Reuse the transport plan across updates, including mixed dtypes and empty ranks."""
    from megatron.core.distributed import param_and_grad_buffer as buffers

    dtypes = [torch.bfloat16, torch.float32] if mixed_dtypes else [torch.bfloat16]
    # The third rank deliberately owns no parameters in this bucket.
    owner_count = min(dp_size, 2)
    high_precision_by_rank = [
        [torch.nn.Parameter(torch.tensor([2.0 + rank, 4.0], dtype=dtype)) for dtype in dtypes]
        for rank in range(owner_count)
    ]
    high_precision = [param for params in high_precision_by_rank for param in params]
    quantized = [
        torch.nn.Parameter(torch.full((2,), -99.0, dtype=torch.bfloat16))
        for _ in range(owner_count)
    ]
    for param in quantized:
        param.test_fp8 = True
    # The source must be the master rather than the deliberately stale forward weight.
    quantized[0].main_param = torch.tensor([1.006, 2.02], dtype=torch.float32)
    expected_remote = torch.tensor([11.0625, 13.5], dtype=torch.bfloat16)
    owner_lists = [
        high_precision_by_rank[rank] + ([quantized[rank]] if has_fp8 else [])
        for rank in range(owner_count)
    ] + [[] for _ in range(dp_size - owner_count)]
    params_list = [param for params in owner_lists for param in params]
    param_index_map = {}
    offset = 0
    for param in params_list:
        param_index_map[param] = (offset, offset + param.numel(), 0)
        offset += param.numel()
    grad_data = torch.full((offset,), 17.0, dtype=torch.float32)
    config = SimpleNamespace(
        overlap_param_gather=completion != "sync", use_distributed_optimizer=False
    )
    bucket = buffers._ParamAndGradBucket(
        params=params_list,
        param_data=None,
        grad_data=grad_data,
        offset=0,
        numel_unpadded=offset,
        gradient_scaling_factor=1.0,
        bucket_id=0,
        param_index_map=param_index_map,
        params_with_extra_main_grads=[],
        ddp_config=config,
        reuse_grad_buffer_for_param_ag=has_fp8,
    )
    group = buffers._ParamAndGradBucketGroup.__new__(buffers._ParamAndGradBucketGroup)
    group.ddp_config = config
    group.buckets = [bucket]
    group.param_sync_via_bucket_group = True
    group.param_gather_handle = None
    group.param_gather_dispatched = False
    group.next_param_gather_bucket_group = None
    group.intra_distributed_optimizer_instance_size = dp_size
    group.intra_distributed_optimizer_instance_rank = 0
    group.intra_distributed_optimizer_instance_group = object()
    reuse_policy = Mock(side_effect=lambda p, config: getattr(p, "test_fp8", False))
    monkeypatch.setattr(buffers, "uses_grad_buffer_for_fp8_param_gather", reuse_policy)
    monkeypatch.setattr(
        buffers, "_param_uses_quantized_storage", lambda p: getattr(p, "test_fp8", False)
    )
    monkeypatch.setattr(buffers, "post_all_gather_processing", Mock())
    bucket.set_layerwise_params_list(owner_lists)
    gather_plan = bucket.layerwise_gather_plan
    reuse_policy.reset_mock()
    copy_sources = []

    def copy_back(params, values):
        for param, value in zip(params, values):
            copy_sources.append(value.clone())
            param.data.copy_(value)

    monkeypatch.setattr(buffers, "copy_back_gathered_bf16_into_fp8_params", copy_back)
    work = Mock()

    def gather(outputs, source, **kwargs):
        for rank, output in enumerate(outputs[1:], start=1):
            if rank >= owner_count:
                assert output.numel() == 0
            elif source.untyped_storage().data_ptr() == grad_data.untyped_storage().data_ptr():
                output.copy_(expected_remote)
            else:
                remote = next(p for p in high_precision_by_rank[rank] if p.dtype == source.dtype)
                output.copy_(remote.detach())
        return work if kwargs["async_op"] else None

    collective = Mock(side_effect=gather)
    monkeypatch.setattr(torch.distributed, "all_gather", collective)
    before_high_precision = [p.detach().clone() for p in high_precision]
    for step in range(2):
        if step:
            # Parameter values change; owner lists, dtype groups and receive sizes do not.
            quantized[0].main_param.add_(0.5)
            grad_data.fill_(17.0)
            collective.reset_mock()
            copy_sources.clear()
        group.start_param_sync()
        assert bucket.layerwise_gather_plan is gather_plan
        reuse_policy.assert_not_called()
        if completion != "sync":
            assert (
                group.param_gather_handle is not None
            ), "DP1 still needs local copy-back completion"
            assert len(bucket.layerwise_gather_list) == len(dtypes) + has_fp8
            for params_by_rank, received, reuse in bucket.layerwise_gather_list:
                assert all(
                    getattr(p, "test_fp8", False) == reuse
                    for params in params_by_rank
                    for p in params
                )
                assert (
                    received[0].untyped_storage().data_ptr()
                    == grad_data.untyped_storage().data_ptr()
                ) == reuse
                if dp_size > owner_count:
                    assert received[-1].numel() == 0
            if completion == "force":
                group.start_param_sync(force_sync=True)
            else:
                group.finish_param_sync(skip_next_bucket_dispatch=True)
        assert collective.call_count == (len(dtypes) + has_fp8 if dp_size > 1 else 0)
        assert bucket.layerwise_gather_list is None
        assert group.param_gather_handle is None
        for param, before in zip(high_precision, before_high_precision):
            assert torch.equal(param, before)
        if has_fp8:
            assert torch.equal(quantized[0], quantized[0].main_param.to(torch.bfloat16))
            if dp_size > 1:
                assert torch.equal(quantized[1], expected_remote)
            assert len(copy_sources) == owner_count
            assert torch.count_nonzero(grad_data) == 0
        else:
            assert torch.all(grad_data == 17), "High-precision AG must not touch gradient storage"


@pytest.mark.parametrize("layout", [False, True])
def test_existing_mxfp8_cli_flag_is_preserved(monkeypatch, layout):
    import sys

    from megatron.training.arguments import parse_args

    argv = ["test", "--reuse-grad-buf-for-mxfp8-param-ag"]
    if not layout:
        argv.append("--no-use-layer-wise-param-layout")
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args()
    assert args.reuse_grad_buf_for_mxfp8_param_ag
    assert args.use_layer_wise_param_layout is layout
    assert not hasattr(args, "layer_wise_param_layout")


@pytest.mark.parametrize(
    "flag",
    [
        "--layer-wise-param-layout",
        "--reuse-grad-buf-for-param-gather",
        "--reuse-grad-buf-for-blockwise-fp8-param-gather",
    ],
)
def test_no_new_gather_flags(monkeypatch, flag):
    import sys

    from megatron.training.arguments import parse_args

    monkeypatch.setattr(sys, "argv", ["test", flag])
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("native_sibling", [False, True])
def test_nested_optimizer_staging_and_deferred_gather(native_sibling):
    """Shared-chunk sync waits for reused and native Adam leaves to update."""
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    events = []
    group = SimpleNamespace(
        buckets=[
            SimpleNamespace(params_list=[SimpleNamespace(is_managed_by_layer_wise_optimizer=False)])
        ]
    )
    model = SimpleNamespace(
        bucket_groups=[group],
        expert_parallel_bucket_groups=[],
        finish_pending_param_sync=lambda: events.append("finish"),
        zero_grad_buffer=lambda: events.append("zero"),
        _start_bucket_group_param_sync=lambda *args, **kwargs: events.append("gather"),
    )
    config = SimpleNamespace(overlap_param_gather_with_optimizer_step=False, timers=None)
    leaves = []
    for index, reuse in enumerate((True, not native_sibling)):
        leaf = DistributedOptimizer.__new__(DistributedOptimizer)
        leaf.is_stub_optimizer = False
        leaf.reuse_grad_buffer_for_param_ag = reuse
        leaf.ddp_config = SimpleNamespace(overlap_param_gather=False)
        leaf.model_chunks = [model]
        leaf._copy_main_params_to_param_buffer = lambda index=index: events.append(f"stage{index}")
        leaves.append(leaf)

    def step(index):
        assert all(leaf._defer_param_sync for leaf in leaves)
        events.append(f"step{index}")
        return True

    for index, leaf in enumerate(leaves):
        leaf.step_with_ready_grads = lambda index=index: step(index)
    inner = ChainedOptimizer.__new__(ChainedOptimizer)
    inner.chained_optimizers = [leaves[0]]
    inner.config = config
    outer = ChainedOptimizer.__new__(ChainedOptimizer)
    outer.chained_optimizers = [inner, leaves[1]]
    outer.config = config
    outer.prepare_model_params_for_param_sync()
    assert events == ["finish", "zero", "stage0"] + ([] if native_sibling else ["stage1"])
    events.clear()
    assert outer.step_with_ready_grads()
    assert events == ["step0", "step1", "gather"]
    assert not any(leaf._defer_param_sync for leaf in leaves)
    # A raw configuration opt-in alone must not activate staging/deferral.
    for leaf in leaves:
        leaf.reuse_grad_buffer_for_param_ag = False
        leaf.ddp_config.reuse_grad_buf_for_mxfp8_param_ag = True
    assert not outer._should_defer_mxfp8_param_sync()


@pytest.mark.parametrize("layout", [False, True])
@pytest.mark.parametrize("num_buckets", [None, 2])
def test_layerwise_ddp_preserves_resolved_bucket_config(layout, num_buckets):
    """Compact all-reduce and padded sharding keep the resolved bucket size."""
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer import TransformerConfig
    from megatron.training.training import resolve_ddp_bucket_size, wrap_model_chunks_with_ddp
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel()
    try:
        module = torch.nn.Linear(16, 16, bias=True, device="cuda", dtype=torch.bfloat16)
        config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            use_layer_wise_param_layout=layout,
            overlap_grad_reduce=True,
            num_buckets=num_buckets,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        # Training resolves the user-specified count before DDP creates compact buffers.
        resolved_bucket_size = resolve_ddp_bucket_size(
            config, pg_collection.dp_cp, True, sum(param.numel() for param in module.parameters())
        )
        config.bucket_size = resolved_bucket_size
        ddp = wrap_model_chunks_with_ddp(
            [module],
            TransformerConfig(num_layers=1, num_attention_heads=1),
            config,
            use_layer_wise_distributed_optimizer=True,
            use_layer_wise_param_layout=layout,
            pg_collection=pg_collection,
        )[0]
        assert config.use_distributed_optimizer, "Muon must not mutate the shared config"
        assert config.num_buckets == num_buckets
        assert config.bucket_size == resolved_bucket_size
        assert module.weight.is_managed_by_layer_wise_optimizer
        assert not module.bias.is_managed_by_layer_wise_optimizer
        assert (ddp.full_param_layout is not None) is layout
        assert len(ddp.buffers) == 2
        for buffer in ddp.buffers:
            assert buffer.ddp_config.use_distributed_optimizer is layout
            assert buffer.ddp_config.bucket_size == resolved_bucket_size
            if layout:
                assert buffer.param_data is not None
            else:
                assert ddp.ddp_config is not config
                assert buffer.ddp_config.num_buckets is None
                assert buffer.numel == buffer.numel_unpadded
                assert buffer.param_data is None
    finally:
        Utils.destroy_model_parallel()
