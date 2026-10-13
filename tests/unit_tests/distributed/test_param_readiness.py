# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parameter readiness: ``ensure_params_ready`` and DDP's ``_BucketParamReadyCallback``.

The GPU test in ``generalized_tensor_parallel/`` runs one configuration: healthy DDP, pre-hooks
enabled, ``align_param_gather=False``. These cover the branches it never exercises --
``align_param_gather=True``, pre-hooks removed mid-sequence, a collected DDP or bucket group,
CUDA-graph capture deferral, and out-of-order bucket dedup -- which would be contrived to set up
on real GPUs.
"""

from types import SimpleNamespace

import torch

import megatron.core.distributed.distributed_data_parallel as ddp_module
from megatron.core.distributed.distributed_data_parallel import (
    DistributedDataParallel,
    _BucketParamReadyCallback,
)
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV1
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import PARAM_READY_CALLBACK_ATTR, ensure_params_ready


class _FakeBucketGroup:
    def __init__(self):
        self.finished = []
        self.param_gather_dispatched = False
        self.param_gather_handle = None

    def finish_param_sync(self, skip_next_bucket_dispatch=False):
        self.finished.append(skip_next_bucket_dispatch)
        self.param_gather_dispatched = True
        self.param_gather_handle = None


def _fake_ddp(align_param_gather=False, hooks_enabled=True):
    ddp = object.__new__(DistributedDataParallel)
    ddp.ddp_config = SimpleNamespace(align_param_gather=align_param_gather)
    ddp.overlap_param_gather_with_optimizer_step = False
    ddp.remove_forward_pre_hook_handles = {object(): object()} if hooks_enabled else {}
    return ddp


class TestParamReadinessProtocol:
    def test_no_marker_is_a_noop(self):
        """A parameter no backend owns must be readable without any callback."""
        param = torch.nn.Parameter(torch.zeros(2))
        assert not hasattr(param, PARAM_READY_CALLBACK_ATTR)
        ensure_params_ready([param])  # must not raise

    def test_callbacks_are_deduplicated_per_bucket(self):
        """Backends share one callback per bucket, so it must fire once per call, in order."""
        calls = []
        shared = lambda: calls.append("shared")  # noqa: E731
        other = lambda: calls.append("other")  # noqa: E731

        first, second, third = (torch.nn.Parameter(torch.zeros(1)) for _ in range(3))
        setattr(first, PARAM_READY_CALLBACK_ATTR, shared)
        setattr(second, PARAM_READY_CALLBACK_ATTR, shared)
        setattr(third, PARAM_READY_CALLBACK_ATTR, other)
        plain = torch.nn.Parameter(torch.zeros(1))

        ensure_params_ready([first, second, plain, first, third])

        assert calls == ["shared", "other"]


class TestBucketParamReadiness:
    """Each test must hold a strong reference to the fake DDP.

    The callback stores DDP weakly, so letting ``_fake_ddp()`` stay a temporary would collect it
    immediately and make every callback early-return -- turning these into vacuous passes.
    """

    def test_cuda_graph_capture_defers_publication_until_replay(self, monkeypatch):
        ddp = _fake_ddp()
        bucket_group = _FakeBucketGroup()
        ready_callback = _BucketParamReadyCallback(ddp, bucket_group)
        param = torch.nn.Parameter(torch.zeros(1))
        setattr(param, PARAM_READY_CALLBACK_ATTR, ready_callback)
        capturing = [True]
        monkeypatch.setattr(ddp_module, "is_graph_capturing", lambda: capturing[0])

        ensure_params_ready([param])
        assert bucket_group.finished == []
        assert not bucket_group.param_gather_dispatched

        capturing[0] = False
        ensure_params_ready([param])
        assert bucket_group.finished == [False]
        assert bucket_group.param_gather_dispatched

    def test_publishes_once_then_is_idempotent(self):
        ddp = _fake_ddp()
        bucket_group = _FakeBucketGroup()
        ready_callback = _BucketParamReadyCallback(ddp, bucket_group)

        ready_callback()
        assert bucket_group.finished == [False]

        # Already published this iteration: no second collective.
        ready_callback()
        assert bucket_group.finished == [False]

    def test_in_flight_gather_is_drained(self):
        ddp = _fake_ddp()
        bucket_group = _FakeBucketGroup()
        bucket_group.param_gather_dispatched = True
        bucket_group.param_gather_handle = object()

        _BucketParamReadyCallback(ddp, bucket_group)()
        assert bucket_group.finished == [False]

    def test_align_param_gather_skips_next_bucket_dispatch(self):
        ddp = _fake_ddp(align_param_gather=True)
        bucket_group = _FakeBucketGroup()
        _BucketParamReadyCallback(ddp, bucket_group)()
        assert bucket_group.finished == [True]

    def test_disabled_forward_hooks_do_not_start_a_gather(self):
        """With the pre-hooks removed the caller drives param sync; starting one here could
        gather into a buffer it is actively using."""
        ddp = _fake_ddp(hooks_enabled=False)
        bucket_group = _FakeBucketGroup()
        _BucketParamReadyCallback(ddp, bucket_group)()
        assert bucket_group.finished == []

    def test_in_flight_gather_is_drained_even_with_hooks_disabled(self):
        """``disable_forward_pre_hook(param_sync=False)`` leaves hooks removed without a force
        sync, so an outstanding all-gather still has to be drained before the buffer is read --
        but without chaining the next bucket's dispatch, which is the caller's schedule now."""
        ddp = _fake_ddp(hooks_enabled=False)
        bucket_group = _FakeBucketGroup()
        bucket_group.param_gather_dispatched = True
        bucket_group.param_gather_handle = object()

        _BucketParamReadyCallback(ddp, bucket_group)()
        assert bucket_group.finished == [True]  # drained, next dispatch skipped

    def test_dead_bucket_group_is_a_noop(self):
        """The bucket group is held weakly so the callback, which lives on parameters that
        outlive DDP, cannot retain the bucket-group graph."""
        ddp = _fake_ddp()
        ready_callback = _BucketParamReadyCallback(ddp, _FakeBucketGroup())
        assert ready_callback._bucket_group() is None
        ready_callback()  # must not raise

    def test_dead_ddp_is_a_noop(self):
        """The callback holds DDP weakly, so a collected DDP must not resurrect work."""
        bucket_group = _FakeBucketGroup()
        ready_callback = _BucketParamReadyCallback(_fake_ddp(), bucket_group)
        # No strong reference above, so the referent is already gone.
        assert ready_callback._ddp() is None
        ready_callback()
        assert bucket_group.finished == []


class TestDSAFineGrainedReadiness:
    def test_fp32_projection_gathers_at_called_dsattention_boundary(self):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=1,
            dsa_kernel_backend="none",
            dsa_indexer_weights_proj_use_quantization=False,
            dsa_indexer_weights_proj_output_dtype="fp32",
        )

        recurse_types = FullyShardedDataParallelV1._fine_grained_recurse_module_types(
            config, DistributedDataParallelConfig()
        )

        assert DSAttention in recurse_types

    def test_bf16_projection_keeps_leaf_level_gather(self):
        config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=1)

        recurse_types = FullyShardedDataParallelV1._fine_grained_recurse_module_types(
            config, DistributedDataParallelConfig()
        )

        assert DSAttention not in recurse_types
