# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Keep fresh numerical evidence independent of selective CI execution."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
pytest_plugins = ['pytester']


@pytest.mark.parametrize('mode', ['full', 'baseline', 'enforce'])
@pytest.mark.parametrize('valid_cache', [False, True])
@pytest.mark.parametrize(
    'bucket',
    [
        'tests/unit_tests/determinism/kernels/**/*.py',
        'tests/unit_tests/determinism/correctness/**/*.py',
        'tests/unit_tests/resharding/test_nccl_m2n_copy_service.py',
    ],
)
def test_action_runs_evidence_exhaustively_and_preserves_other_testmon_modes(
    tmp_path, mode, valid_cache, bucket
):
    bash_version = subprocess.check_output(
        ['bash', '-c', 'printf "%s" "${BASH_VERSINFO[0]}"'], text=True
    )
    if int(bash_version) < 4:
        pytest.skip('The CI action requires modern Bash errexit semantics')
    action = yaml.safe_load((ROOT / '.github/actions/action.yml').read_text())
    step = next(step for step in action['runs']['steps'] if step.get('id') == 'unit-testmon')
    assert step['env']['TEST_CASE'] == '${{ inputs.test_case }}'
    output = tmp_path / 'outputs'
    summary = tmp_path / 'summary'
    # Replace only the cache-validation command. Execute the actual action shell.
    script = ('python() { return ' + ('0' if valid_cache else '1') + '; }\n') + step['run']
    result = subprocess.run(
        ['bash', '-e', '-u', '-o', 'pipefail'],
        input=script,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env={
            **os.environ,
            'REQUESTED_MODE': mode,
            'TEST_CASE': bucket,
            'IDENTITY_OUTCOME': 'success' if valid_cache else 'failure',
            'RESTORE_OUTCOME': 'success' if valid_cache else 'failure',
            'MATCHED_KEY': 'synthetic-cache' if valid_cache else '',
            'RUNNER_TEMP': str(tmp_path),
            'GITHUB_OUTPUT': str(output),
            'GITHUB_STEP_SUMMARY': str(summary),
        },
    )
    if mode == 'baseline' and not valid_cache:
        assert result.returncode != 0  # Preserve the trusted-producer precondition.
        assert not output.exists()
        return
    assert result.returncode == 0, result.stderr
    values = dict(line.split('=', 1) for line in output.read_text().splitlines())
    expected = mode if valid_cache else 'full'
    if '/determinism/' in bucket:
        expected = 'full'
    assert values == {'mode': expected}
    assert f'- Effective mode: {expected}' in summary.read_text()


def test_collection_only_cannot_supply_complete_replay_evidence(pytester, monkeypatch):
    """The collection phase in selective CI must remain U even when pytest succeeds."""
    from tools.determinism.coverage import UNVERIFIED, aggregate

    monkeypatch.setenv('PYTHONPATH', str(ROOT))
    monkeypatch.setenv('RANK', '0')
    pytester.makeini('[pytest]\n')
    pytester.makeconftest("""
from tools.determinism import pytest_plugin
pytest_plugin._context = lambda root: {'revision': 'a' * 40, 'dirty': False, 'world_size': 1}
""")
    pytester.makepyfile("""
import pytest
from tools.determinism.coverage import observe_replay

@pytest.mark.determinism_model(model_id='synthetic')
def test_replay():
    with observe_replay({}, {'replays': 2}) as observation:
        observation['compared_outputs'] = 1
""")
    output = pytester.path / 'evidence'
    result = pytester.runpytest_subprocess(
        '-p',
        'tools.determinism.pytest_plugin',
        '--collect-only',
        '--determinism-evidence-dir',
        str(output),
        '--determinism-evidence-scope=model',
    )
    assert result.ret == 0
    shards = [json.loads(path.read_text()) for path in output.glob('rank-*.json')]
    assert len(shards) == 1
    report = aggregate(shards)
    assert report['counts']['total'] == report['counts'][UNVERIFIED] == 1
    assert all(
        not case['test_complete'] and not case['observations']
        for case in shards[0]['cases'].values()
    )
