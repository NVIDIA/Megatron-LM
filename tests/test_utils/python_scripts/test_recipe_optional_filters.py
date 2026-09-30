# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from tests.test_utils.python_scripts import recipe_parser


@pytest.mark.parametrize(
    ("filter_function", "field", "value"),
    [
        (recipe_parser.filter_by_tag, "tag", "latest"),
        (recipe_parser.filter_by_environment, "environment", "dev"),
        (recipe_parser.filter_by_platform, "platforms", "dgx_h100"),
    ],
)
def test_filters_skip_missing_optional_fields(filter_function, field, value):
    missing = recipe_parser.dotdict(spec={"test_case": "missing"})
    matching = recipe_parser.dotdict(spec={"test_case": "matching", field: value})
    other = recipe_parser.dotdict(spec={"test_case": "other", field: "other"})
    workloads = [missing, matching, other]

    assert filter_function(workloads, value) == [matching]
    assert workloads == [missing, matching, other]


def test_tag_filter_selects_unit_tests_from_complete_recipe_set():
    workloads = recipe_parser.load_workloads(
        container_tag="fixture", environment="dev", tag="latest"
    )
    test_workloads = [workload for workload in workloads if workload.type != "build"]

    assert test_workloads
    assert all(workload.spec["tag"] == "latest" for workload in test_workloads)
    assert all(workload.spec["model"] == "unit-tests" for workload in test_workloads)
