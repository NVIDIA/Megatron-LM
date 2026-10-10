# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ('selection', 'expected'),
    [
        ('first,first', ['first']),
        ('first,second,first', ['first', 'second']),
        ('missing,first,first', ['first']),
    ],
)
def test_repeated_test_case_selection_does_not_duplicate_workloads(selection, expected):
    module_spec = importlib.util.spec_from_file_location(
        'recipe_parser', Path(__file__).with_name('recipe_parser.py')
    )
    parser = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(parser)
    workloads = [parser.dotdict(spec={'test_case': name}) for name in ['first', 'second']]

    result = parser.filter_by_test_cases(workloads, selection)

    assert [workload.spec['test_case'] for workload in result] == expected
