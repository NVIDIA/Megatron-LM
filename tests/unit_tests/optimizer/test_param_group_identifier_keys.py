# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for ``param_group_identifier_keys`` and the param-group save/load matching.

The identifier tuple is the fingerprint used by
:meth:`MegatronOptimizer._filter_and_reorder_param_groups` and
:meth:`DistributedOptimizer.load_state_dict` to match saved param_groups onto
the current optimizer's param_groups during checkpoint resume. It MUST cover
every per-group field that influences scheduler or optimizer behavior;
otherwise two groups distinguishable only by, say, ``max_lr`` collide in the
matching dict and the second one's config silently overwrites the first's
on load — producing wrong LRs after restart and (on a converged-enough model)
loss explosion at the next optimizer step.

These tests pin the identifier composition and verify the matching tolerates
keys that aren't present on every group (e.g. ``start_wd``/``end_wd``/
``optimizer`` from :class:`ParamGroupOverride` are only set on groups that
explicitly override them).
"""

from megatron.core.optimizer.optimizer import (
    MegatronOptimizer,
    param_group_identifier_keys,
)
from megatron.core.optimizer_param_scheduler import ParamGroupOverride


def _make_pg(**kw) -> dict:
    """Build a minimal param_group dict for matcher tests. ``params`` is required."""
    pg = {"params": []}
    pg.update(kw)
    return pg


def test_identifier_keys_cover_all_param_group_override_fields():
    """REGRESSION: every key in ``ParamGroupOverride`` must be in the identifier.

    If someone adds a new field to ``ParamGroupOverride`` (per-group user-facing
    config), it must also appear in ``param_group_identifier_keys`` — otherwise
    two groups distinguishable only by that field will collide on load. This
    test fails loudly when the two get out of sync.
    """
    missing = set(ParamGroupOverride.__optional_keys__) - set(param_group_identifier_keys)
    assert not missing, (
        f"ParamGroupOverride fields not in param_group_identifier_keys: {missing}. "
        f"Either add to identifier_keys, or argue why this field should NOT participate "
        f"in save/load matching."
    )


def test_identifier_keys_include_structural_flags():
    """``lr_mult`` / ``is_expert_parallel`` / ``is_decoupled_lr`` are set on every
    param_group at construction time and must remain in the identifier so saves
    from one process layout match loads in another.
    """
    for key in ("lr_mult", "is_expert_parallel", "is_decoupled_lr"):
        assert key in param_group_identifier_keys, (
            f"{key!r} missing from param_group_identifier_keys; this would let groups "
            f"collide across e.g. EP-on/EP-off boundaries on resume."
        )


def test_filter_reorder_distinguishes_groups_by_max_lr():
    """REGRESSION: two groups that differ ONLY by max_lr/min_lr must be matched
    correctly across save/load. Pre-fix, both would have produced the same legacy
    4-tuple ``(wd_mult, lr_mult, is_expert_parallel, is_decoupled_lr)`` of
    ``(1.0, 1.0, False, False)`` and collided in the matching dict — the second
    saved group's config (max_lr) silently clobbered the first's at load time,
    producing wrong LRs at the next optimizer step.
    """
    # Two current groups with same wd/structural flags but different max_lr —
    # this is exactly the recipe pattern (trunk-WD vs. projector-WD).
    current = [
        _make_pg(
            wd_mult=1.0,
            lr_mult=1.0,
            is_expert_parallel=False,
            is_decoupled_lr=False,
            max_lr=2e-5,
            min_lr=2e-6,
        ),
        _make_pg(
            wd_mult=1.0,
            lr_mult=1.0,
            is_expert_parallel=False,
            is_decoupled_lr=False,
            max_lr=5e-4,
            min_lr=5e-5,
        ),
    ]
    # Saved groups (deliberately reordered to exercise the reorder logic).
    saved = [
        _make_pg(
            wd_mult=1.0,
            lr_mult=1.0,
            is_expert_parallel=False,
            is_decoupled_lr=False,
            max_lr=5e-4,
            min_lr=5e-5,
            # Some recognizable extra field to confirm the right saved group was matched.
            _tag="from_projector",
        ),
        _make_pg(
            wd_mult=1.0,
            lr_mult=1.0,
            is_expert_parallel=False,
            is_decoupled_lr=False,
            max_lr=2e-5,
            min_lr=2e-6,
            _tag="from_trunk",
        ),
    ]

    reordered = MegatronOptimizer._filter_and_reorder_param_groups(current, saved)

    assert len(reordered) == 2
    # current[0] has max_lr=2e-5 → must match the saved group with max_lr=2e-5.
    assert reordered[0]["max_lr"] == 2e-5
    assert reordered[0]["_tag"] == "from_trunk"
    # current[1] has max_lr=5e-4 → must match the saved group with max_lr=5e-4.
    assert reordered[1]["max_lr"] == 5e-4
    assert reordered[1]["_tag"] == "from_projector"


def test_filter_reorder_tolerates_missing_optional_keys():
    """Some identifier keys (``start_wd`` / ``end_wd`` / ``optimizer``) come from
    ``ParamGroupOverride`` and are only present on groups that explicitly
    override them. Default groups don't carry these keys at all, so the matcher
    must use a sentinel for missing keys (rather than KeyError-ing). Two groups
    missing the same set of keys must remain matchable.
    """
    # Both groups have only the always-present keys; they should match by tuple
    # of values + the same sentinel placeholder for missing keys.
    current = [
        _make_pg(
            wd_mult=1.0,
            lr_mult=1.0,
            is_expert_parallel=False,
            is_decoupled_lr=False,
            max_lr=1e-3,
            min_lr=1e-4,
            # NOT setting start_wd, end_wd, optimizer — these are absent.
        ),
    ]
    saved = [
        _make_pg(
            wd_mult=1.0,
            lr_mult=1.0,
            is_expert_parallel=False,
            is_decoupled_lr=False,
            max_lr=1e-3,
            min_lr=1e-4,
            _tag="saved_match",
        ),
    ]
    reordered = MegatronOptimizer._filter_and_reorder_param_groups(current, saved)
    assert len(reordered) == 1
    assert reordered[0]["_tag"] == "saved_match"


def test_filter_reorder_distinguishes_by_optional_override_key():
    """When a group sets a key that another doesn't (e.g. ``start_wd``), the
    identifier tuple must reflect that — the two groups must be distinguishable
    rather than collapsed into one match.
    """
    common = dict(
        wd_mult=1.0,
        lr_mult=1.0,
        is_expert_parallel=False,
        is_decoupled_lr=False,
        max_lr=1e-3,
        min_lr=1e-4,
    )
    current = [
        _make_pg(**common, start_wd=0.05),  # explicit per-group start_wd
        _make_pg(**common),                  # default start_wd (absent → sentinel)
    ]
    saved = [
        _make_pg(**common, _tag="default_wd"),
        _make_pg(**common, start_wd=0.05, _tag="explicit_wd"),
    ]
    reordered = MegatronOptimizer._filter_and_reorder_param_groups(current, saved)
    assert reordered[0]["_tag"] == "explicit_wd"
    assert reordered[0].get("start_wd") == 0.05
    assert reordered[1]["_tag"] == "default_wd"
    assert "start_wd" not in reordered[1]


def test_filter_reorder_handles_nemo_pre_prefix():
    """NeMo renames ``lr_mult``/``wd_mult`` to ``pre_lr_mult``/``pre_wd_mult``.
    The matcher's per-key fallback must look up ``pre_<key>`` if ``<key>`` is
    missing — this preserves NeMo-saved checkpoint compatibility.
    """
    common = dict(
        is_expert_parallel=False,
        is_decoupled_lr=False,
        max_lr=1e-3,
        min_lr=1e-4,
    )
    # Current uses standard names, saved uses NeMo's pre_-prefixed names.
    current = [_make_pg(**common, wd_mult=1.0, lr_mult=1.0)]
    saved = [_make_pg(**common, pre_wd_mult=1.0, pre_lr_mult=1.0, _tag="from_nemo")]
    reordered = MegatronOptimizer._filter_and_reorder_param_groups(current, saved)
    assert reordered[0]["_tag"] == "from_nemo"
