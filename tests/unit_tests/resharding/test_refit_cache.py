# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the refit/reshard caches.

Covers:
- ``_PlanCacheKey`` separation across configurations that route to different
  global ranks (the rank-offset bug — two non-collocated configs with identical
  parallel sizes used to silently share a plan).
- ``_PlanCacheKey`` separation across execution batch limits.
- ``get_refit_tensor_dict`` / ``invalidate_refit_tensor_cache`` (module-level
  named_refit_tensors cache + invalidation when ``_harmonize_buffer_dtypes``
  replaces a buffer).
"""

import pytest
import torch
import torch.nn as nn

import megatron.core.resharding.refit as refit
from megatron.core.resharding.copy_services.base import CopyService
from megatron.core.resharding.refit import _get_parallel_config, _ParallelConfig, _PlanCacheKey
from megatron.core.resharding.utils import (
    ReshardPlan,
    get_refit_tensor_dict,
    invalidate_refit_tensor_cache,
)


def _config(tp=1, pp=1, ep=1, dp=1, expert_tp=1, gtp_remat=1, expert_gtp_remat=1):
    return _ParallelConfig(
        tp_size=tp,
        pp_size=pp,
        ep_size=ep,
        dp_size=dp,
        expert_tp_size=expert_tp,
        gtp_remat_size=gtp_remat,
        expert_gtp_remat_size=expert_gtp_remat,
    )


class TestPlanCacheKey:
    """Plan cache must distinguish configs that route to different global ranks."""

    def test_equality_with_same_inputs(self):
        k1 = _PlanCacheKey(rank=0, src_config=_config(), dst_config=_config(), num_experts=None)
        k2 = _PlanCacheKey(rank=0, src_config=_config(), dst_config=_config(), num_experts=None)
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_different_src_rank_offset_distinguishes(self):
        """Same sizes + rank, different src_rank_offset → different cache key."""
        k1 = _PlanCacheKey(
            rank=0,
            src_config=_config(tp=2, dp=2),
            dst_config=_config(tp=2, dp=2),
            num_experts=None,
            src_rank_offset=0,
            dst_rank_offset=4,
        )
        k2 = _PlanCacheKey(
            rank=0,
            src_config=_config(tp=2, dp=2),
            dst_config=_config(tp=2, dp=2),
            num_experts=None,
            src_rank_offset=8,
            dst_rank_offset=12,
        )
        assert k1 != k2
        assert hash(k1) != hash(k2)

    def test_different_dst_rank_offset_distinguishes(self):
        k1 = _PlanCacheKey(
            rank=0,
            src_config=_config(tp=2, dp=2),
            dst_config=_config(tp=2, dp=2),
            num_experts=None,
            src_rank_offset=0,
            dst_rank_offset=4,
        )
        k2 = _PlanCacheKey(
            rank=0,
            src_config=_config(tp=2, dp=2),
            dst_config=_config(tp=2, dp=2),
            num_experts=None,
            src_rank_offset=0,
            dst_rank_offset=8,
        )
        assert k1 != k2

    def test_default_offsets_match_collocated(self):
        """Collocated callers (no offsets specified) reuse the same plan."""
        k1 = _PlanCacheKey(
            rank=3, src_config=_config(tp=2, dp=4), dst_config=_config(tp=2, dp=4), num_experts=None
        )
        k2 = _PlanCacheKey(
            rank=3,
            src_config=_config(tp=2, dp=4),
            dst_config=_config(tp=2, dp=4),
            num_experts=None,
            src_rank_offset=0,
            dst_rank_offset=0,
        )
        assert k1 == k2

    def test_num_experts_distinguishes(self):
        k1 = _PlanCacheKey(rank=0, src_config=None, dst_config=None, num_experts=8)
        k2 = _PlanCacheKey(rank=0, src_config=None, dst_config=None, num_experts=16)
        assert k1 != k2

    def test_execution_batch_bytes_distinguishes(self):
        k1 = _PlanCacheKey(
            rank=0, src_config=None, dst_config=None, num_experts=None, execution_batch_bytes=128
        )
        k2 = _PlanCacheKey(
            rank=0, src_config=None, dst_config=None, num_experts=None, execution_batch_bytes=256
        )
        assert k1 != k2

    def test_gtp_remat_sizes_distinguish(self):
        base = _config(tp=2, dp=2)
        plain = _PlanCacheKey(rank=0, src_config=base, dst_config=base, num_experts=None)

        for config in (_config(tp=2, dp=2, gtp_remat=4), _config(tp=2, dp=2, expert_gtp_remat=2)):
            assert plain != _PlanCacheKey(
                rank=0, src_config=config, dst_config=config, num_experts=None
            )


def test_parallel_config_includes_gtp_remat_sizes():
    class Group:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

    class Core:
        pg_collection = type(
            "PG",
            (),
            {
                "tp": Group(2),
                "pp": Group(3),
                "ep": Group(4),
                "dp": Group(5),
                "expt_tp": Group(6),
                "gtp_remat": Group(7),
                "expt_gtp_remat": Group(8),
            },
        )()

    assert _get_parallel_config(Core()) == _config(
        tp=2, pp=3, ep=4, dp=5, expert_tp=6, gtp_remat=7, expert_gtp_remat=8
    )


class TestPlanCacheKeyNonCollocated:
    """Non-collocated ranks set src_config or dst_config to None.

    Cache key must distinguish the three rank classes (source-only, dest-only,
    idle) so they don't share plans across roles.
    """

    def test_source_only_vs_dest_only_distinguish(self):
        """Source-only (dst_config=None) and dest-only (src_config=None) on the
        same global rank must produce different plans."""
        config = _config(tp=2, dp=2)
        src_only = _PlanCacheKey(rank=0, src_config=config, dst_config=None, num_experts=None)
        dst_only = _PlanCacheKey(rank=0, src_config=None, dst_config=config, num_experts=None)
        assert src_only != dst_only

    def test_idle_rank_distinguishes_from_active(self):
        """Idle rank (both configs None) is distinct from a rank with either model."""
        idle = _PlanCacheKey(rank=5, src_config=None, dst_config=None, num_experts=None)
        with_src = _PlanCacheKey(rank=5, src_config=_config(), dst_config=None, num_experts=None)
        with_dst = _PlanCacheKey(rank=5, src_config=None, dst_config=_config(), num_experts=None)
        assert idle != with_src
        assert idle != with_dst
        assert with_src != with_dst

    def test_non_collocated_offset_combinations(self):
        """src_rank_offset and dst_rank_offset together distinguish non-collocated
        layouts that share parallel sizes."""
        config = _config(tp=2, dp=2)
        # Two non-collocated layouts: world=[src 0-3, dst 4-7] vs [src 0-3, dst 8-11].
        layout_a = _PlanCacheKey(
            rank=0,
            src_config=config,
            dst_config=config,
            num_experts=None,
            src_rank_offset=0,
            dst_rank_offset=4,
        )
        layout_b = _PlanCacheKey(
            rank=0,
            src_config=config,
            dst_config=config,
            num_experts=None,
            src_rank_offset=0,
            dst_rank_offset=8,
        )
        assert layout_a != layout_b


def test_prepare_swap_threads_execution_batch_bytes(monkeypatch):
    """The public preparation API must include the configured limit in planning."""
    plan = ReshardPlan(send_ops=[], recv_ops=[])
    forwarded = {}

    monkeypatch.setattr(refit, "_unwrap_model_cores", lambda *_args: (None, None, None))

    def fake_build(*args, **kwargs):
        forwarded["args"] = args
        forwarded["kwargs"] = kwargs
        return plan

    monkeypatch.setattr(refit, "_build_or_get_plan", fake_build)

    refit.prepare_swap_model_weights(None, None, execution_batch_bytes=123)

    assert forwarded["kwargs"] == {"execution_batch_bytes": 123}


def test_service_cache_distinguishes_process_groups(monkeypatch):
    """A backend service must never reuse a communicator from another group."""

    class StubService:
        def __init__(self, group=None, *, max_group_bytes=None):
            self.group = group
            self.max_group_bytes = max_group_bytes

        def close(self):
            pass

    monkeypatch.setattr(refit, "NCCLM2NCopyService", StubService)
    monkeypatch.setattr(refit, "_service_cache", {})
    first_group = object()
    second_group = object()

    first = refit.get_or_create_service("nccl_m2n", group=first_group)
    first_again = refit.get_or_create_service("nccl_m2n", group=first_group)
    second = refit.get_or_create_service("nccl_m2n", group=second_group)

    assert first_again is first
    assert second is not first
    assert second.group is second_group


def test_service_cache_distinguishes_m2n_execution_limits(monkeypatch):
    """The first cached M2N service must not silently fix later calls to its limit."""

    class StubService:
        def __init__(self, group=None, *, max_group_bytes=None):
            self.group = group
            self.max_group_bytes = max_group_bytes

        def close(self):
            pass

    monkeypatch.setattr(refit, "NCCLM2NCopyService", StubService)
    monkeypatch.setattr(refit, "_service_cache", {})

    default = refit.get_or_create_service("nccl_m2n")
    limited = refit.get_or_create_service("nccl_m2n", execution_batch_bytes=128)
    limited_again = refit.get_or_create_service("nccl_m2n", execution_batch_bytes=128)
    other_limit = refit.get_or_create_service("nccl_m2n", execution_batch_bytes=256)

    assert default.max_group_bytes is None
    assert limited.max_group_bytes == 128
    assert limited_again is limited
    assert other_limit is not limited


def test_non_m2n_service_cache_ignores_execution_limit(monkeypatch):
    """Generic batching is plan state and must not duplicate non-M2N services."""

    class StubService:
        def __init__(self, group=None):
            self.group = group

        def close(self):
            pass

    monkeypatch.setattr(refit, "NCCLCopyService", StubService)
    monkeypatch.setattr(refit, "_service_cache", {})

    first = refit.get_or_create_service("nccl", execution_batch_bytes=128)
    second = refit.get_or_create_service("nccl", execution_batch_bytes=256)

    assert second is first


def test_swap_threads_execution_limit_to_named_service(monkeypatch):
    """The public API must use the configured limit when it creates an M2N service."""

    class StubService:
        supports_idle_ranks = True

    forwarded = {}

    def fake_get_or_create_service(backend, group=None, execution_batch_bytes=None):
        forwarded.update(backend=backend, group=group, execution_batch_bytes=execution_batch_bytes)
        return StubService()

    monkeypatch.setattr(refit, "get_or_create_service", fake_get_or_create_service)
    monkeypatch.setattr(refit, "reshard_model_weights", lambda *_args, **_kwargs: None)

    group = object()
    refit.swap_model_weights(
        None,
        None,
        refit_method="nccl_m2n",
        group=group,
        transform=refit.ReshardTransform(),
        execution_batch_bytes=123,
    )

    assert forwarded == {"backend": "nccl_m2n", "group": group, "execution_batch_bytes": 123}


def test_swap_rejects_multiple_pools_for_service_without_idle_ranks():
    class NoIdleRanksService(CopyService):
        supports_idle_ranks = False

        def __init__(self):
            pass

        def submit_send(self, src_tensor, dest_rank, task_id=None):
            pass

        def submit_recv(self, dest_tensor, src_rank, task_id=None):
            pass

        def run(self):
            pass

    service = NoIdleRanksService()

    with pytest.raises(ValueError, match="does not support num_dst_pools > 1"):
        refit.swap_model_weights(None, None, refit_method=service, num_dst_pools=2)


class TestNeedsMxfp8Conversion:
    """_needs_mxfp8_conversion gracefully handles non-target ranks (model=None)."""

    def test_none_returns_false(self):
        """Source-only and idle ranks pass target_model=None to _setup_mxfp8_..."""
        from megatron.core.resharding.refit import _needs_mxfp8_conversion

        assert _needs_mxfp8_conversion(None) is False

    def test_mxfp8_model_returns_true(self):
        from megatron.core.resharding.refit import _needs_mxfp8_conversion

        class _Cfg:
            transformer_impl = "inference_optimized"
            fp8 = "hybrid"
            fp8_recipe = "mxfp8"

        class _Model:
            config = _Cfg()

        assert _needs_mxfp8_conversion(_Model()) is True

    def test_non_inference_optimized_returns_false(self):
        from megatron.core.resharding.refit import _needs_mxfp8_conversion

        class _Cfg:
            transformer_impl = "transformer_engine"
            fp8 = "hybrid"
            fp8_recipe = "mxfp8"

        class _Model:
            config = _Cfg()

        assert _needs_mxfp8_conversion(_Model()) is False

    def test_non_mxfp8_recipe_returns_false(self):
        from megatron.core.resharding.refit import _needs_mxfp8_conversion

        class _Cfg:
            transformer_impl = "inference_optimized"
            fp8 = "hybrid"
            fp8_recipe = "delayed"

        class _Model:
            config = _Cfg()

        assert _needs_mxfp8_conversion(_Model()) is False

    def test_inactive_mxfp8_recipe_returns_false(self):
        from megatron.core.resharding.refit import _needs_mxfp8_conversion

        class _Cfg:
            transformer_impl = "inference_optimized"
            fp8 = None
            fp8_recipe = "mxfp8"

        class _Model:
            config = _Cfg()

        assert _needs_mxfp8_conversion(_Model()) is False

    def test_list_wrapped_model(self):
        """The function unwraps a single-element list/tuple."""
        from megatron.core.resharding.refit import _needs_mxfp8_conversion

        class _Cfg:
            transformer_impl = "inference_optimized"
            fp8 = "hybrid"
            fp8_recipe = "mxfp8"

        class _Model:
            config = _Cfg()

        assert _needs_mxfp8_conversion([_Model()]) is True


class TestSetupMxfp8TransformOnPlan:
    """_setup_mxfp8_transform_on_plan is a no-op on non-target ranks and idempotent."""

    def test_target_none_leaves_transform_unset(self):
        """Source-only / idle ranks should leave plan.transform at None."""
        from megatron.core.resharding.refit import _setup_mxfp8_transform_on_plan
        from megatron.core.resharding.utils import ReshardPlan

        plan = ReshardPlan(send_ops=[], recv_ops=[])
        _setup_mxfp8_transform_on_plan(plan, None)
        assert plan.transform is None

    def test_non_mxfp8_target_leaves_transform_unset(self):
        from megatron.core.resharding.refit import _setup_mxfp8_transform_on_plan
        from megatron.core.resharding.utils import ReshardPlan

        class _Cfg:
            transformer_impl = "transformer_engine"
            fp8_recipe = None

        class _Model:
            config = _Cfg()

        plan = ReshardPlan(send_ops=[], recv_ops=[])
        _setup_mxfp8_transform_on_plan(plan, _Model())
        assert plan.transform is None

    def test_already_populated_skips_rebuild(self):
        """Idempotent: if plan.transform is already set, do not re-quantize."""
        from megatron.core.resharding.refit import _setup_mxfp8_transform_on_plan
        from megatron.core.resharding.transforms import ReshardTransform
        from megatron.core.resharding.utils import ReshardPlan

        sentinel = ReshardTransform()
        plan = ReshardPlan(send_ops=[], recv_ops=[], transform=sentinel)

        # Even with an MXFP8 model, the existing transform should not be replaced.
        class _Cfg:
            transformer_impl = "inference_optimized"
            fp8_recipe = "mxfp8"

        class _Model:
            config = _Cfg()

        _setup_mxfp8_transform_on_plan(plan, _Model())
        assert plan.transform is sentinel

    def test_flashinfer_uses_canonical_triton_buffers(self, monkeypatch):
        """FlashInfer refit derives Major-K weights from canonical Triton storage."""
        from megatron.core.resharding import refit
        from megatron.core.resharding.utils import ReshardPlan

        class _Config:
            transformer_impl = "inference_optimized"
            fp8 = "hybrid"
            fp8_recipe = "mxfp8"
            inference_grouped_gemm_backend = "flashinfer"

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = _Config()
                self.decoder = nn.Linear(4, 4, bias=False)

        captured = {}

        def _quantize(_decoder, *, backend):
            captured["backend"] = backend
            return {}

        monkeypatch.setattr(refit, "_should_quantize_param", lambda _param: True)
        monkeypatch.setattr(refit, "quantize_params_to_mxfp8", _quantize)

        plan = ReshardPlan(send_ops=[], recv_ops=[])
        refit._setup_mxfp8_transform_on_plan(plan, _Model())
        assert captured["backend"] == plan.transform.backend == "triton"


class TestRefitTensorCache:
    """get_refit_tensor_dict caches the param/buffer dict on the module."""

    def test_returns_same_dict_on_repeat(self):
        model = nn.Linear(4, 4, bias=False)
        d1 = get_refit_tensor_dict(model)
        d2 = get_refit_tensor_dict(model)
        assert d1 is d2

    def test_contains_parameters(self):
        model = nn.Linear(4, 4)
        d = get_refit_tensor_dict(model)
        assert "weight" in d and "bias" in d

    def test_contains_persistent_buffers(self):
        model = nn.Module()
        model.register_buffer("running_mean", torch.zeros(4))
        d = get_refit_tensor_dict(model)
        assert "running_mean" in d

    def test_excludes_non_persistent_buffers(self):
        model = nn.Module()
        model.register_buffer("tmp", torch.zeros(4), persistent=False)
        d = get_refit_tensor_dict(model)
        assert "tmp" not in d

    def test_invalidate_drops_cache(self):
        model = nn.Linear(4, 4, bias=False)
        d1 = get_refit_tensor_dict(model)
        invalidate_refit_tensor_cache(model)
        d2 = get_refit_tensor_dict(model)
        assert d1 is not d2

    def test_invalidate_picks_up_replaced_buffer(self):
        """Mirrors _harmonize_buffer_dtypes: replace _buffers entry, invalidate, re-read."""
        model = nn.Module()
        model.register_buffer("buf", torch.zeros(4, dtype=torch.bfloat16))
        d1 = get_refit_tensor_dict(model)
        old_buf = d1["buf"]

        model._buffers["buf"] = old_buf.to(torch.float32)
        invalidate_refit_tensor_cache(model)

        d2 = get_refit_tensor_dict(model)
        assert d2["buf"].dtype == torch.float32
        assert d2["buf"] is not old_buf

    def test_invalidate_when_no_cache_is_safe(self):
        """Calling invalidate before any get_refit_tensor_dict call should not raise."""
        model = nn.Linear(4, 4, bias=False)
        invalidate_refit_tensor_cache(model)  # no-op

    def test_cache_is_per_module(self):
        m1 = nn.Linear(4, 4, bias=False)
        m2 = nn.Linear(4, 4, bias=False)
        d1 = get_refit_tensor_dict(m1)
        d2 = get_refit_tensor_dict(m2)
        assert d1 is not d2
