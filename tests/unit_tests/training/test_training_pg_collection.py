# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""pretrain skips the coarse seed for a multi-module carrier; a plain collection or None seeds stock."""

from types import SimpleNamespace
from unittest import mock

import pytest

from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.training import training as training_mod


class _Stop(Exception):
    """Bail out of pretrain right after the seed groups are resolved."""


def _make_collection(**groups):
    """A real ProcessGroupCollection with the requested seed group attributes set."""
    pgc = ProcessGroupCollection()
    for name, value in groups.items():
        setattr(pgc, name, value)
    return pgc


def _drive_pretrain(pg_collection):
    """Run pretrain up to initialize_megatron and capture its resolved seed kwargs."""
    captured = {}

    def _fake_initialize(**kwargs):
        captured["seed"] = kwargs
        raise _Stop()

    with (
        mock.patch.object(training_mod.ft_integration, "setup"),
        mock.patch.object(training_mod, "initialize_megatron", side_effect=_fake_initialize),
    ):
        with pytest.raises(_Stop):
            training_mod.pretrain(
                cfg_container=SimpleNamespace(),
                train_valid_test_dataset_provider=None,
                model_type=None,
                forward_step_func=None,
                pg_collection=pg_collection,
            )
    return captured


def test_pretrain_stock_none_seeds_defaults():
    captured = _drive_pretrain(None)
    # No carrier -> stock seeds normally (not skipped) with mpu-default groups.
    assert captured["seed"]["skip_random_seed"] is False
    for key in (
        "seed_pp_group",
        "seed_dp_group",
        "seed_tp_group",
        "seed_ep_group",
        "seed_etp_group",
    ):
        assert captured["seed"][key] is None


def test_pretrain_plain_collection_seeds_that_collection():
    pgc = _make_collection(pp="pp", dp="dp", tp="tp", ep="ep", expt_tp="etp")
    captured = _drive_pretrain(pgc)
    assert captured["seed"]["skip_random_seed"] is False
    assert captured["seed"]["seed_pp_group"] == "pp"
    assert captured["seed"]["seed_dp_group"] == "dp"
    assert captured["seed"]["seed_tp_group"] == "tp"
    assert captured["seed"]["seed_ep_group"] == "ep"
    assert captured["seed"]["seed_etp_group"] == "etp"


def test_pretrain_multi_module_defers_seed_to_builder():
    language = _make_collection(pp="lpp", dp="ldp", tp="ltp", ep="lep", expt_tp="letp")
    encoder = _make_collection(pp="epp", dp="edp", tp="etp", ep="eep", expt_tp="eetp")
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": encoder, "language": language}, language_model_module_name="language"
    )
    captured = _drive_pretrain(carrier)
    # A multi-module carrier skips the coarse seed entirely; the builder seeds each module (no pick).
    assert captured["seed"]["skip_random_seed"] is True
    for key in (
        "seed_pp_group",
        "seed_dp_group",
        "seed_tp_group",
        "seed_ep_group",
        "seed_etp_group",
    ):
        assert captured["seed"][key] is None


def test_pretrain_multi_module_without_language_seeds_none():
    encoder = _make_collection(pp="epp", dp="edp", tp="etp", ep="eep", expt_tp="eetp")
    carrier = MultiModuleProcessGroupCollection(
        module_pgs={"encoder": encoder}, language_model_module_name=None
    )
    captured = _drive_pretrain(carrier)
    # Encoder-only carrier also skips the coarse seed; the builder seeds each module.
    assert captured["seed"]["skip_random_seed"] is True
    for key in (
        "seed_pp_group",
        "seed_dp_group",
        "seed_tp_group",
        "seed_ep_group",
        "seed_etp_group",
    ):
        assert captured["seed"][key] is None


def test_training_source_has_no_language_else_first_heuristic():
    """The next(iter(module_pgs...)) / language-else-first pick must be gone."""
    import inspect

    source = inspect.getsource(training_mod)
    assert "module_pgs.values()" not in source
    assert "next(iter(schedule_pg_collection" not in source
