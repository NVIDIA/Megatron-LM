# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the refit/reshard caches.

Covers:
- ``_PlanCacheKey`` separation across configurations that route to different
  global ranks (the rank-offset bug — two non-collocated configs with identical
  parallel sizes used to silently share a plan).
- ``get_refit_tensor_dict`` / ``invalidate_refit_tensor_cache`` (module-level
  named_refit_tensors cache + invalidation when ``_harmonize_buffer_dtypes``
  replaces a buffer).
"""

import torch
import torch.nn as nn

from megatron.core.resharding.refit import _PlanCacheKey
from megatron.core.resharding.utils import get_refit_tensor_dict, invalidate_refit_tensor_cache


class TestPlanCacheKey:
    """Plan cache must distinguish configs that route to different global ranks."""

    def test_equality_with_same_inputs(self):
        k1 = _PlanCacheKey(
            rank=0, src_config=(1, 1, 1, 1, 1), dst_config=(1, 1, 1, 1, 1), num_experts=None
        )
        k2 = _PlanCacheKey(
            rank=0, src_config=(1, 1, 1, 1, 1), dst_config=(1, 1, 1, 1, 1), num_experts=None
        )
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_different_src_rank_offset_distinguishes(self):
        """Same sizes + rank, different src_rank_offset → different cache key."""
        k1 = _PlanCacheKey(
            rank=0,
            src_config=(2, 1, 1, 2, 1),
            dst_config=(2, 1, 1, 2, 1),
            num_experts=None,
            src_rank_offset=0,
            dst_rank_offset=4,
        )
        k2 = _PlanCacheKey(
            rank=0,
            src_config=(2, 1, 1, 2, 1),
            dst_config=(2, 1, 1, 2, 1),
            num_experts=None,
            src_rank_offset=8,
            dst_rank_offset=12,
        )
        assert k1 != k2
        assert hash(k1) != hash(k2)

    def test_different_dst_rank_offset_distinguishes(self):
        k1 = _PlanCacheKey(
            rank=0,
            src_config=(2, 1, 1, 2, 1),
            dst_config=(2, 1, 1, 2, 1),
            num_experts=None,
            src_rank_offset=0,
            dst_rank_offset=4,
        )
        k2 = _PlanCacheKey(
            rank=0,
            src_config=(2, 1, 1, 2, 1),
            dst_config=(2, 1, 1, 2, 1),
            num_experts=None,
            src_rank_offset=0,
            dst_rank_offset=8,
        )
        assert k1 != k2

    def test_default_offsets_match_collocated(self):
        """Collocated callers (no offsets specified) reuse the same plan."""
        k1 = _PlanCacheKey(
            rank=3, src_config=(2, 1, 1, 4, 1), dst_config=(2, 1, 1, 4, 1), num_experts=None
        )
        k2 = _PlanCacheKey(
            rank=3,
            src_config=(2, 1, 1, 4, 1),
            dst_config=(2, 1, 1, 4, 1),
            num_experts=None,
            src_rank_offset=0,
            dst_rank_offset=0,
        )
        assert k1 == k2

    def test_num_experts_distinguishes(self):
        k1 = _PlanCacheKey(rank=0, src_config=None, dst_config=None, num_experts=8)
        k2 = _PlanCacheKey(rank=0, src_config=None, dst_config=None, num_experts=16)
        assert k1 != k2


class TestPlanCacheKeyNonCollocated:
    """Non-collocated ranks set src_config or dst_config to None.

    Cache key must distinguish the three rank classes (source-only, dest-only,
    idle) so they don't share plans across roles.
    """

    def test_source_only_vs_dest_only_distinguish(self):
        """Source-only (dst_config=None) and dest-only (src_config=None) on the
        same global rank must produce different plans."""
        sizes = (2, 1, 1, 2, 1)
        src_only = _PlanCacheKey(rank=0, src_config=sizes, dst_config=None, num_experts=None)
        dst_only = _PlanCacheKey(rank=0, src_config=None, dst_config=sizes, num_experts=None)
        assert src_only != dst_only

    def test_idle_rank_distinguishes_from_active(self):
        """Idle rank (both configs None) is distinct from a rank with either model."""
        idle = _PlanCacheKey(rank=5, src_config=None, dst_config=None, num_experts=None)
        with_src = _PlanCacheKey(
            rank=5, src_config=(1, 1, 1, 1, 1), dst_config=None, num_experts=None
        )
        with_dst = _PlanCacheKey(
            rank=5, src_config=None, dst_config=(1, 1, 1, 1, 1), num_experts=None
        )
        assert idle != with_src
        assert idle != with_dst
        assert with_src != with_dst

    def test_non_collocated_offset_combinations(self):
        """src_rank_offset and dst_rank_offset together distinguish non-collocated
        layouts that share parallel sizes."""
        sizes = (2, 1, 1, 2, 1)
        # Two non-collocated layouts: world=[src 0-3, dst 4-7] vs [src 0-3, dst 8-11].
        layout_a = _PlanCacheKey(
            rank=0,
            src_config=sizes,
            dst_config=sizes,
            num_experts=None,
            src_rank_offset=0,
            dst_rank_offset=4,
        )
        layout_b = _PlanCacheKey(
            rank=0,
            src_config=sizes,
            dst_config=sizes,
            num_experts=None,
            src_rank_offset=0,
            dst_rank_offset=8,
        )
        assert layout_a != layout_b


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
            fp8_recipe = "mxfp8"

        class _Model:
            config = _Cfg()

        assert _needs_mxfp8_conversion(_Model()) is True

    def test_non_inference_optimized_returns_false(self):
        from megatron.core.resharding.refit import _needs_mxfp8_conversion

        class _Cfg:
            transformer_impl = "transformer_engine"
            fp8_recipe = "mxfp8"

        class _Model:
            config = _Cfg()

        assert _needs_mxfp8_conversion(_Model()) is False

    def test_non_mxfp8_recipe_returns_false(self):
        from megatron.core.resharding.refit import _needs_mxfp8_conversion

        class _Cfg:
            transformer_impl = "inference_optimized"
            fp8_recipe = "delayed"

        class _Model:
            config = _Cfg()

        assert _needs_mxfp8_conversion(_Model()) is False

    def test_list_wrapped_model(self):
        """The function unwraps a single-element list/tuple."""
        from megatron.core.resharding.refit import _needs_mxfp8_conversion

        class _Cfg:
            transformer_impl = "inference_optimized"
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
