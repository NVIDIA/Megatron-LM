# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""NVLink topology in the runtime-scheduler cost models: it follows from the hardware.

One table states the GPUs per NVLink domain of each hardware. Every cost model takes its
domain size from there; nothing a user writes sets one. The tests with the packaged
predictors pin what a domain boundary does to the price of a CP group.
"""

import importlib
import inspect
import json
import re
from pathlib import Path

import pytest

from megatron.core.datasets.dcp_runtime_scheduler.models import (
    make_model,
    model_class,
    qwen3_h100_p2p,
    usp_calib_v1,
)
from megatron.core.datasets.dcp_runtime_scheduler.models._benchmark import attention_predictor
from megatron.core.datasets.dcp_runtime_scheduler.models.base import (
    NVLINK_DOMAIN_SIZES,
    DcpCostModel,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.lightweight import (
    LightweightRuntimePlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import RuntimePlanGeometry

_NON_ATTENTION = {"forward_ms_per_layer": 0.5, "backward_ms_per_layer": 1.0}
_H100_PACKAGED = Path(qwen3_h100_p2p.__file__).with_suffix(".json")
_USP_PACKAGED = Path(usp_calib_v1.__file__).with_suffix(".json")


def _gqa(hardware="h100", world=8, **options):
    """Ordinary GQA with the packaged predictor, bound as the training entry binds it."""
    allowed = tuple(cp for cp in (1, 2, 4, 8, 16, 32, 64) if cp <= world)
    geometry = RuntimePlanGeometry(world, 4096, 1, allowed, 2 * world, 64, True)
    model = make_model(
        "dcp_benchmark",
        geometry,
        hardware=hardware,
        non_attention=dict(_NON_ATTENTION),
        query_heads=32,
        kv_heads=4,
        head_dim=128,
        layer_count=2,
        **options,
    )
    return geometry, model


def _qwen35(**options):
    geometry = RuntimePlanGeometry(32, 16384, 32, (1, 2, 4, 8, 16, 32), 64, 64, True)
    model = make_model(
        "qwen35_full", geometry, hardware="gb200", non_attention=dict(_NON_ATTENTION), **options
    )
    return geometry, model


def _h100_calibration(**options):
    geometry = RuntimePlanGeometry(32, 4096, 8, (1, 2, 4, 8), 64, 64, True)
    return geometry, make_model("qwen3_h100_p2p", geometry, **options)


def _original_profile(**options):
    geometry = RuntimePlanGeometry(32, 4096, 1, (8, 16, 32), 64, 64, True)
    return geometry, make_model("usp_calib_v1", geometry, allow_provisional=True, **options)


# Model name -> (builder, hardware the runtime file names for it).
_BUILDERS = {
    "dcp_benchmark": (_gqa, "h100"),
    "qwen35_full": (_qwen35, "gb200"),
    "qwen3_h100_p2p": (_h100_calibration, "h100"),
    "usp_calib_v1": (_original_profile, None),
}


def _price(model, pack, cp, start):
    ranks = list(range(start, start + cp))
    return (
        model.forward(pack, cp, ranks=ranks),
        model.backward(pack, cp, ranks=ranks),
        model.program_costs(pack, cp, ranks=ranks),
    )


def _prices(model, pack, cp, starts):
    """The distinct prices of one pack over ``starts``."""
    return {repr(_price(model, pack, cp, start)) for start in starts}


# --------------------------------------------------------------------------------------
# One table, and every cost model reads it
# --------------------------------------------------------------------------------------
def test_domain_table_equals_the_table_of_the_packaged_predictor():
    topology = importlib.import_module(attention_predictor.__name__ + "._legacy.attention.topology")
    assert dict(NVLINK_DOMAIN_SIZES) == {"h100": 8, "gb200": 72}
    assert dict(NVLINK_DOMAIN_SIZES) == topology.CALIBRATION_DOMAIN_SIZES
    with pytest.raises(TypeError):
        NVLINK_DOMAIN_SIZES["h100"] = 16


@pytest.mark.parametrize("name", sorted(name for name in _BUILDERS if _BUILDERS[name][1]))
def test_hardware_bound_cost_model_exposes_the_domain_of_its_hardware(name):
    build, hardware = _BUILDERS[name]
    _, model = build()
    assert isinstance(model, DcpCostModel)
    assert model.hardware == hardware
    assert model.nvlink_domain_size == NVLINK_DOMAIN_SIZES[hardware]
    assert model.metadata()["nvlink_domain_size"] == NVLINK_DOMAIN_SIZES[hardware]


def test_ordinary_gqa_takes_the_domain_of_either_hardware():
    assert _gqa("h100")[1].nvlink_domain_size == 8
    assert _gqa("gb200")[1].nvlink_domain_size == 72


def test_h100_calibration_node_is_the_domain_of_the_hardware_its_file_was_measured_on():
    domain = json.loads(_H100_PACKAGED.read_text())["domain"]
    assert domain["ranks_per_node"] == NVLINK_DOMAIN_SIZES[domain["hardware"]]
    assert domain["nodes"] * domain["ranks_per_node"] == domain["world_size"]
    # Every measured CP group fits one node.
    assert max(domain["cp_sizes"]) <= domain["ranks_per_node"]
    assert _h100_calibration()[1].nvlink_domain_size == domain["ranks_per_node"]


def test_original_dynamic_usp_profile_reports_the_nodes_of_its_packaged_table():
    table = json.loads(_USP_PACKAGED.read_text())
    stated = re.match(r"(\d+) GPU per node, NVLink in-node", table["hardware"])
    _, model = _original_profile()
    assert isinstance(model, DcpCostModel)
    # The runtime file names no hardware for this profile.
    assert not hasattr(model, "hardware")
    assert model.nvlink_domain_size == int(stated.group(1)) == 8
    assert model.metadata()["nvlink_domain_size"] == 8
    # The profile reads no rank positions: the number selects nothing.
    assert len(_prices(model, (8192, 4096), 8, range(25))) == 1


# --------------------------------------------------------------------------------------
# What a domain boundary does to the price of a CP group
# --------------------------------------------------------------------------------------
def test_h100_group_inside_one_node_has_one_price_on_every_node():
    geometry, model = _gqa("h100", 32)
    for cp in (2, 4, 8):
        # Every start that keeps the group inside one node, size-aligned or not.
        inside = [start for start in range(32 - cp + 1) if start // 8 == (start + cp - 1) // 8]
        assert len(_prices(model, geometry.full_pack(cp, 2), cp, inside)) == 1


def test_h100_group_larger_than_a_node_has_one_price_at_every_node_boundary():
    # CP 16 covers two whole nodes: the split of the group the parameters were fitted on.
    geometry, model = _gqa("h100", 32)
    assert len(_prices(model, geometry.full_pack(16, 2), 16, (0, 8, 16))) == 1


@pytest.mark.parametrize("cp,start", [(2, 7), (4, 6), (8, 4), (16, 4), (16, 12)])
def test_h100_group_split_by_a_node_boundary_elsewhere_is_rejected(cp, start):
    geometry, model = _gqa("h100", 32)
    pack = geometry.full_pack(cp, 1)
    ranks = list(range(start, start + cp))
    # The group leaves a node somewhere else than the same group at rank 0 does.
    assert [rank // 8 - start // 8 for rank in ranks] != [rank // 8 for rank in range(cp)]
    for call in (model.forward, model.backward, model.program_costs):
        with pytest.raises(ValueError, match="changed domain partition"):
            call(pack, cp, ranks=ranks)
    # For the planner the pack does not exist at that start, and it forms no such group.
    costs = LightweightRuntimePlanner(geometry, model).costs
    assert costs.pack(pack, cp, start) is None
    assert costs.pack(pack, cp, 0) is not None
    assert not geometry.supports_group(start, cp)


def test_gb200_price_does_not_depend_on_the_start():
    # At most 64 ranks: all of them are in the one 72-GPU domain.
    geometry, model = _gqa("gb200", 64)
    for cp, starts in (
        (2, range(63)),
        (4, range(61)),
        (8, range(0, 57, 3)),
        (16, (0, 1, 8, 16, 47, 48)),
        (32, (0, 1, 32)),
    ):
        assert len(_prices(model, geometry.full_pack(cp, 2), cp, starts)) == 1


def test_full_qwen35_price_does_not_depend_on_the_start():
    geometry, model = _qwen35()
    for cp in (1, 2, 4, 8, 16):
        assert len(_prices(model, geometry.full_pack(cp, 2), cp, range(32 - cp + 1))) == 1


def test_h100_calibration_prices_a_group_inside_one_node_and_rejects_any_other():
    _, model = _h100_calibration()
    for cp in (1, 2, 4, 8):
        pack = (4096,) * cp
        assert len(_prices(model, pack, cp, range(0, 32, cp))) == 1
        for start in range(32 - cp + 1):
            if start % cp:
                with pytest.raises(ValueError, match="aligned group inside one H100 node"):
                    model.forward(pack, cp, ranks=list(range(start, start + cp)))


# --------------------------------------------------------------------------------------
# Nothing a user writes sets a domain size
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(_BUILDERS))
def test_no_cost_model_constructor_takes_a_domain_size(name):
    parameters = inspect.signature(model_class(name).__init__).parameters
    assert not [parameter for parameter in parameters if "domain" in parameter]
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    build, _ = _BUILDERS[name]
    for option in ("legacy_domain_size", "nvlink_domain_size", "domain_size"):
        with pytest.raises(TypeError, match=option):
            build(**{option: 8})


def test_benchmark_adapter_hands_no_domain_size_to_the_predictor():
    _, model = _gqa("h100")
    assert "legacy_domain_size" not in model.predictor_options
    assert "legacy_domain_size" not in model.metadata()["predictor_options"]
    # The predictor's own override stays unset, so it uses its table entry for the hardware.
    assert model.predictors["gqa"]._predictor(2).legacy_domain_size is None


def test_h100_refit_file_cannot_state_another_node_size(tmp_path):
    data = json.loads(_H100_PACKAGED.read_text())
    data["domain"]["ranks_per_node"] = 4
    path = tmp_path / "refit.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="outside the 32-H100 classic P2P recipe"):
        _h100_calibration(calibration=str(path))


def test_replacement_table_of_the_original_profile_sets_no_domain_size(tmp_path):
    table = json.loads(_USP_PACKAGED.read_text())
    table["nvlink_domain_size"] = 16
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    _, packaged = _original_profile()
    _, replaced = _original_profile(calibration=str(path))
    # A replacement table has no field for where it was measured; such a key is not read.
    assert replaced.nvlink_domain_size is None
    assert replaced.metadata()["nvlink_domain_size"] is None
    for cp in (8, 16, 32):
        assert _price(replaced, (8192, 4096), cp, 0) == _price(packaged, (8192, 4096), cp, 0)
