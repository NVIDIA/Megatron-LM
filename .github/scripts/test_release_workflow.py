# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise release guards and uploaded-distribution checks without publishing."""

import hashlib
import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / '.github/workflows/release.yaml').read_text())
SPEC = importlib.util.spec_from_file_location(
    'verify_release', Path(__file__).with_name('verify_pypi_release.py')
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def evaluate(expression: str, context: dict, failed=False, cancelled=False):
    """Evaluate workflow guards against supplied event and job states."""
    expression = expression.removeprefix('${{').removesuffix('}}').strip()
    expression = expression.replace('!cancelled()', str(not cancelled)).replace(
        '!failure()', str(not failed)
    )
    expression = re.sub(
        r'\b(?:github|inputs|needs)\.[\w.-]+', lambda match: repr(context[match[0]]), expression
    )
    expression = expression.replace('&&', ' and ').replace('||', ' or ')
    expression = re.sub(r'!(?!=)', 'not ', expression)
    return eval(' '.join(expression.split()), {'__builtins__': {}}, {})


def release_context(event='workflow_dispatch', dry_run=False, target='pypi') -> dict:
    """Build event context with successful upstream jobs."""
    return {
        'github.repository': 'NVIDIA/Megatron-LM',
        'github.event_name': event,
        'inputs.dry-run': dry_run,
        'inputs.publish-target': target,
        'needs.bump.result': 'success',
        'needs.build-test-wheels.result': 'success',
        'needs.verify-published-wheels.result': 'success',
    }


@pytest.mark.parametrize('event', ['push', 'merge_group', 'workflow_dispatch'])
@pytest.mark.parametrize('dry_run', [True, False])
@pytest.mark.parametrize('target', ['pypi', 'testpypi'])
def test_only_explicit_uploads_can_publish(event, dry_run, target):
    """Pushes, merge groups, dry runs, failures and cancellation cannot publish."""
    context = release_context(event, dry_run, target)
    guard = WORKFLOW['jobs']['publish-wheels']['if']
    assert bool(evaluate(guard, context)) == (event == 'workflow_dispatch' and not dry_run)
    assert not evaluate(guard, context, failed=True)
    assert not evaluate(guard, context, cancelled=True)


@pytest.mark.parametrize('job', ['bump', 'build-test-wheels'])
@pytest.mark.parametrize('state', ['failure', 'cancelled', 'skipped'])
def test_missing_success_blocks_publication(job, state):
    """Publishing requires successful completion of all required producers."""
    context = release_context()
    context[f'needs.{job}.result'] = state
    assert not evaluate(WORKFLOW['jobs']['publish-wheels']['if'], context)


@pytest.mark.parametrize('state', ['failure', 'cancelled', 'skipped'])
def test_real_release_requires_verified_upload(state):
    """An absent verification cannot finalize a real release; dry runs may continue."""
    context = release_context()
    context['needs.verify-published-wheels.result'] = state
    guard = WORKFLOW['jobs']['finalize']['if']
    assert not evaluate(guard, context)
    context['inputs.dry-run'] = True
    assert evaluate(guard, context)
    assert not evaluate(guard, context, failed=True)


def test_testpypi_has_no_release_side_effects():
    """TestPyPI uploads keep both shared release phases in validation mode."""
    context = release_context(target='testpypi')
    for name in ('bump', 'finalize'):
        for key in ('validate-only', 'dry-run'):
            assert evaluate(WORKFLOW['jobs'][name]['with'][key], context)
    context['inputs.publish-docs'] = True
    assert not evaluate(WORKFLOW['jobs']['finalize']['with']['publish-docs'], context)


def test_builder_has_no_upload_credentials():
    """Builds cannot inherit publishing secrets or mint OIDC credentials."""
    caller = WORKFLOW['jobs']['build-test-wheels']
    assert 'secrets' not in caller
    assert caller['permissions'] == {'contents': 'read'}
    builder = yaml.safe_load((ROOT / '.github/workflows/_build_test_wheel.yml').read_text())
    assert 'secrets' not in builder[True]['workflow_call']
    assert builder['permissions'] == {'contents': 'read'}
    publisher = WORKFLOW['jobs']['publish-wheels']
    assert publisher['permissions'] == {'id-token': 'write'}
    assert not any(
        step.get('uses', '').startswith('actions/checkout@') for step in publisher['steps']
    )


@pytest.mark.parametrize(
    'ref,production,branch,valid',
    [
        ('a' * 40, True, 'refs/heads/main', True),
        ('main', True, 'refs/heads/main', False),
        ('a' * 40, True, 'refs/heads/pull-request/123', False),
        ('a' * 40, False, 'refs/heads/pull-request/123', True),
    ],
)
def test_release_input_validation(tmp_path, ref, production, branch, valid):
    """Execute the release SHA and workflow branch guard."""
    script = tmp_path / 'validate.sh'
    script.write_text(WORKFLOW['jobs']['validate-inputs']['steps'][0]['run'])
    env = dict(
        os.environ, RELEASE_REF=ref, IS_PRODUCTION=str(production).lower(), GITHUB_REF=branch
    )
    result = subprocess.run(
        ['bash', '-euo', 'pipefail', str(script)], env=env, check=False, capture_output=True
    )
    assert (result.returncode == 0) == valid


@pytest.mark.parametrize('mutation', ['none', 'missing', 'changed', 'yanked', 'extra'])
def test_distribution_inventory_and_hashes(tmp_path, mutation):
    """Reject missing, extra, changed, or yanked uploads against local hashes."""
    names = ['megatron_core-1.0-py3-none-any.whl', 'megatron_core-1.0.tar.gz']
    for name in names:
        (tmp_path / name).write_bytes(b'fixture')
    version, files = VERIFY.release_files(tmp_path, 'megatron-core')
    assert version == '1.0'
    assert files == {name: hashlib.sha256(b'fixture').hexdigest() for name in names}
    metadata = {
        'urls': [
            {'filename': name, 'digests': {'sha256': digest}} for name, digest in files.items()
        ]
    }
    if mutation == 'missing':
        metadata['urls'].pop()
    elif mutation == 'changed':
        metadata['urls'][0]['digests']['sha256'] = 'wrong'
    elif mutation == 'yanked':
        metadata['urls'][0]['yanked'] = True
    elif mutation == 'extra':
        metadata['urls'].append({'filename': 'unexpected'})
    if mutation == 'none':
        VERIFY.check_release(metadata, files)
    else:
        with pytest.raises(ValueError):
            VERIFY.check_release(metadata, files)
