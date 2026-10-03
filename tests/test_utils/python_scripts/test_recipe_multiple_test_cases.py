# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pathlib import Path

import pytest

from tests.test_utils.python_scripts import recipe_parser


@pytest.mark.parametrize(
    ("recipe", "expected_cases"),
    [
        ("h100/gpt-perf-dp8.yaml", {"gpt_583m_perf", "gpt_583m_perf_async_sched"}),
        (
            "gb200/gpt-perf-dp4.yaml",
            {"gpt_583m_perf_gb200_4gpu", "gpt_583m_perf_async_sched_gb200_4gpu"},
        ),
    ],
)
def test_performance_recipes_include_every_test_case(recipe, expected_cases):
    recipes_dir = Path(recipe_parser.__file__).parent.parent / "recipes"
    workloads = recipe_parser.load_and_flatten(str(recipes_dir / recipe))

    assert {workload.spec["test_case"] for workload in workloads} == expected_cases
    assert len(workloads) == len(expected_cases)


def test_test_cases_expand_with_parameters_and_preserve_cadence():
    manifest = recipe_parser.dotdict(
        products=[
            {
                "test_case": ["first", "second"],
                "products": [
                    {
                        "environment": ["dev"],
                        "platforms": ["dgx_h100", "dgx_gb200"],
                        "scope": ["mr-github", "nightly"],
                    }
                ],
            }
        ]
    )

    products = recipe_parser.flatten_products(manifest).products

    assert len(products) == 8
    assert {
        (product["test_case"], product["platforms"], product["scope"]) for product in products
    } == {
        (test_case, platform, scope)
        for test_case in ("first", "second")
        for platform in ("dgx_h100", "dgx_gb200")
        for scope in ("L1", "L2")
    }
    for product in products:
        assert product["cadence"] == (
            ["nightly"] if product["scope"] == "L2" else recipe_parser.DEFAULT_CADENCE
        )
