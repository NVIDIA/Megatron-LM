# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import importlib.util
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize('override', [None, 'custom/image'])
def test_each_build_keeps_its_own_image(tmp_path, monkeypatch, override):
    module_spec = importlib.util.spec_from_file_location(
        'recipe_parser', Path(__file__).with_name('recipe_parser.py')
    )
    parser = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(parser)
    scripts = tmp_path / 'python_scripts'
    scripts.mkdir()
    recipes = tmp_path / 'recipes'
    recipes.mkdir()
    monkeypatch.setattr(parser, 'BASE_PATH', scripts)
    for environment in ['dev', 'lts']:
        build = {
            'type': 'build',
            'spec': {'name': f'build-{environment}', 'source': {'image': f'image/{environment}'}},
        }
        (recipes / f'_build-{environment}.yaml').write_text(yaml.safe_dump(build))
    recipe = {
        'type': 'basic',
        'spec': {'build': 'build-{environment}', 'name': '{test_case}', 'platforms': 'cpu'},
        'products': [{'test_case': ['test'], 'products': [{'environment': ['dev', 'lts']}]}],
    }
    (recipes / 'test.yaml').write_text(yaml.safe_dump(recipe))

    workloads = parser.load_workloads(container_tag='test-tag', container_image=override)
    images = {
        workload.spec['name']: workload.spec['source']['image']
        for workload in workloads
        if workload['type'] == 'build'
    }
    assert images == {
        f'build-{environment}': f'{override or f"image/{environment}"}:test-tag'
        for environment in ['dev', 'lts']
    }
