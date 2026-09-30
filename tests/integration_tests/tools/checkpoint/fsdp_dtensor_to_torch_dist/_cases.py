# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parametrization builders for the reverse-converter end-to-end suite.

Turns the :mod:`registry` families and their known-limitation flags into
``pytest.param`` lists with the right skip/xfail marks — so the test modules stay
one-liners and a new family / limitation is expressed purely as registry data.
Underscore-prefixed so pytest never mistakes it for a test module.
"""

import importlib

import pytest

from tests.integration_tests.tools.checkpoint.fsdp_dtensor_to_torch_dist import registry


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _requires_marks(fam):
    """Skip a family whose optional dependency (e.g. flash-linear-attention) is absent."""
    return [
        pytest.mark.skip(reason=f"requires '{dep}' (not importable)")
        for dep in fam.requires
        if not _importable(dep)
    ]


def resume_params():
    """One param per family (id = family name)."""
    return [
        pytest.param(fam, marks=_requires_marks(fam), id=fam.name)
        for fam in registry.all_families()
    ]


def bitexact_params():
    """One param per family; FP8 (not bit-exact by design) is a non-strict xfail."""
    out = []
    for fam in registry.all_families():
        marks = _requires_marks(fam)
        if fam.bitexact_xfail:
            marks.append(pytest.mark.xfail(reason=fam.bitexact_xfail, strict=False))
        out.append(pytest.param(fam, marks=marks, id=fam.name))
    return out


def reshard_params():
    """(family, ReshardCase) per configured layout. EP>1-optimizer is a strict xfail."""
    out = []
    for fam in registry.all_families():
        for rc in fam.reshard_cases:
            marks = _requires_marks(fam)
            if rc.xfail:
                marks.append(pytest.mark.xfail(reason=rc.xfail, strict=True))
            suffix = "optim" if rc.with_optimizer else "weights"
            out.append(pytest.param(fam, rc, marks=marks, id=f"{fam.name}-{rc.layout}-{suffix}"))
    return out


def source_shard_params():
    """(family, SourceShardCase) per configured layout. Unproducible layouts (PP2) skip."""
    out = []
    for fam in registry.all_families():
        for sc in fam.source_shard_cases:
            marks = _requires_marks(fam)
            if sc.unsupported:
                marks.append(pytest.mark.skip(reason=sc.unsupported))
            out.append(pytest.param(fam, sc, marks=marks, id=f"{fam.name}-{sc.layout}"))
    return out
