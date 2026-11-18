# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]


def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.name', 'CI Test')
    git(tmp_path, 'config', 'user.email', 'ci@example.com')
    for name in ('megatron/core/source.py', 'megatron/bridge/old.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original\n')
    git(tmp_path, 'add', '.')
    git(tmp_path, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'base')
    return tmp_path, git(tmp_path, 'rev-parse', 'HEAD')


def classify(directory, base, head, merge_base=False):
    output = directory / 'output'
    result = subprocess.run(
        ['bash', str(ROOT / '.github/scripts/bridge_only.sh')],
        cwd=directory,
        env={
            **os.environ,
            'BASE_SHA': base,
            'HEAD_SHA': head,
            'USE_MERGE_BASE': str(merge_base).lower(),
            'GITHUB_OUTPUT': str(output),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'Bridge-only changes:' in result.stdout
    return output.read_text().strip() == 'bridge_only=true'


@pytest.mark.parametrize(
    'paths,expected',
    [
        (['examples/bridge/convert.py'], True),
        (['megatron/bridge/model.py'], True),
        (['examples/bridge/a.py', 'megatron/bridge/deep/b.py'], True),
        (['megatron/bridge/line\nbreak.py'], True),
        (['megatron/bridge/a.py', 'megatron/core/a.py'], False),
        (['examples/bridge/a.py', 'pyproject.toml'], False),
        (['megatron/bridge/a.py', 'tests/unit_tests/models/bridge/test_a.py'], False),
        (['examples/bridge_extra/a.py'], False),
        (['megatron/bridge_extra/a.py'], False),
        (['examples/bridge/a.py', 'outside\nmegatron/bridge/a.py'], False),
    ],
)
def test_classifies_complete_diff(repository, paths, expected):
    directory, base = repository
    for name in paths:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('new\n')
    git(directory, 'add', '.')
    git(directory, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'change')
    assert classify(directory, base, git(directory, 'rev-parse', 'HEAD')) == expected


def test_deleting_bridge_file_skips(repository):
    directory, base = repository
    git(directory, 'rm', 'megatron/bridge/old.py')
    git(directory, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'delete')
    assert classify(directory, base, git(directory, 'rev-parse', 'HEAD'))


def test_moving_core_into_bridge_keeps_ci(repository):
    directory, base = repository
    git(directory, 'mv', 'megatron/core/source.py', 'megatron/bridge/source.py')
    git(directory, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'move')
    assert not classify(directory, base, git(directory, 'rev-parse', 'HEAD'))


@pytest.mark.parametrize('base_ref', ['same', '', '0' * 40, '--invalid'])
def test_empty_or_unavailable_diff_keeps_ci(repository, base_ref):
    directory, head = repository
    assert not classify(directory, head if base_ref == 'same' else base_ref, head)


def test_pr_diff_includes_earlier_non_bridge_commit(repository):
    directory, base = repository
    (directory / 'megatron/core/source.py').write_text('changed\n')
    git(directory, 'add', '.')
    git(directory, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'core change')
    (directory / 'megatron/bridge/old.py').write_text('bridge change\n')
    git(directory, 'add', '.')
    git(directory, '-c', 'commit.gpgsign=false', 'commit', '-qm', 'bridge change')
    assert not classify(directory, base, git(directory, 'rev-parse', 'HEAD'), merge_base=True)


def test_required_summary_accepts_bridge_skip_without_test_results(tmp_path):
    workflow = yaml.safe_load((ROOT / '.github/workflows/cicd-main.yml').read_text())
    step = next(s for s in workflow['jobs']['Nemo_CICD_Test']['steps'] if s.get('id') == 'result')
    result = subprocess.run(
        ['bash', '-e', '-u', '-o', 'pipefail'],
        input=step['run'],
        env={**os.environ, 'BRIDGE_ONLY': 'true'},
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'Bridge-only changes' in result.stdout
