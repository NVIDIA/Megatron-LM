#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import os
import re
import subprocess
import textwrap
from pathlib import Path


source = Path('.github/workflows/cicd-main.yml').read_text().split('\njobs:\n', 1)[1]
jobs = dict(re.findall(r'^  ([\w-]+):\n(.*?)(?=^  [\w-]+:|\Z)', source, re.M | re.S))
dependencies = {}
for name, block in jobs.items():
    match = re.search(r'^    needs: (\[[^\n]+\])$', block, re.M)
    if match:
        dependencies[name] = match[1].strip('[]').replace(' ', '').split(',')
    else:
        match = re.search(r'^    needs:\n((?:      - .+\n)+)', block, re.M)
        dependencies[name] = re.findall(r'      - (.+)', match[1]) if match else []


def ancestors(name, visiting=()):
    assert name not in visiting, f'Dependency cycle: {visiting + (name,)}'
    result = set(dependencies[name])
    for dependency in dependencies[name]:
        result.update(ancestors(dependency, visiting + (name,)))
    return result


for name in jobs:
    ancestors(name)
assert 'cicd-wait-in-queue' not in ancestors('cicd-container-build-cpu')
assert 'linting' in ancestors('cicd-container-build-cpu')
assert 'cicd-container-build-cpu' in ancestors('cicd-wait-in-queue')
assert 'cicd-wait-in-queue' in ancestors('cicd-container-build-external')


def enabled(name, *, statuses=None, values=None, success=False, cancelled=False):
    # Evaluate the actual boolean predicates with string-valued GitHub contexts.
    # success=False models an intentionally skipped ancestor on the other build path.
    context = {
        'needs.is-not-external-contributor.outputs.is_maintainer': 'true',
        'vars.ENABLE_GB200_TESTING': 'true',
        'github.repository': 'NVIDIA/Megatron-LM',
    }
    context.update(values or {})
    expression = re.search(r'^    if: \|\n((?:      .+\n)+)', jobs[name], re.M)[1]
    expression = re.sub(
        r'needs\.([\w-]+)\.result',
        lambda match: repr((statuses or {}).get(match[1], 'success')),
        expression,
    )
    expression = re.sub(
        r'(?:needs\.[\w-]+\.outputs|vars|github)\.[\w-]+',
        lambda match: repr(context.get(match[0], 'false')),
        expression,
    )
    expression = expression.replace('success()', repr(success)).replace('cancelled()', repr(cancelled))
    expression = expression.replace('&&', ' and ').replace('||', ' or ')
    expression = re.sub(r'!(?!=)', 'not ', expression)
    return eval(' '.join(expression.split()), {'__builtins__': {}})


cpu = 'cicd-container-build-cpu'
external = 'cicd-container-build-external'
gate = 'cicd-wait-in-queue'
external_values = {'needs.is-not-external-contributor.outputs.is_maintainer': 'false'}
assert enabled(cpu, success=True)
assert not enabled(cpu, success=True, values=external_values)
assert not enabled(cpu, statuses={'linting': 'failure'})
assert enabled(gate)
assert enabled(gate, statuses={cpu: 'skipped'}, values=external_values)
assert enabled(external, statuses={cpu: 'skipped'}, values=external_values)
assert not enabled(external)
for result in ('failure', 'cancelled', 'skipped'):
    assert not enabled(gate, statuses={cpu: result})
    assert not enabled(gate, statuses={'linting': result})
    assert not enabled(external, statuses={gate: result}, values=external_values)
for flag in ('docs_only', 'is_deployment_workflow'):
    values = {f'needs.pre-flight.outputs.{flag}': 'true'}
    assert not enabled(cpu, success=True, values=values)
    assert not enabled(gate, values=values)
for flag in ('is_ci_workload', 'is_merge_group', 'force_run_all'):
    values = {f'needs.pre-flight.outputs.{flag}': 'true'}
    assert enabled(cpu, values=values)
    assert enabled(external, values={**external_values, **values}, statuses={gate: 'skipped'})
    if flag != 'force_run_all':
        assert not enabled(gate, values=values)

# Regular test jobs remain eligible despite the unused build branch being skipped,
# but each direct dependency must succeed before they can run.
consumers = (
    'cicd-parse-downstream-testing', 'cicd-parse-unit-tests',
    'cicd-unit-tests-latest', 'cicd-parse-unit-tests-gb200',
    'cicd-unit-tests-latest-gb200',
)
for name in consumers:
    assert enabled(name), name
    for dependency in dependencies[name]:
        for result in ('failure', 'cancelled', 'skipped'):
            assert not enabled(name, statuses={dependency: result}), (name, dependency, result)
for name in (cpu, external, gate, 'cicd-container-build', *consumers, 'Coverage'):
    assert not enabled(name, success=True, cancelled=True), name
assert enabled('Coverage')
assert not enabled('Coverage', statuses={'Nemo_CICD_Test': 'failure'})
assert not enabled('Coverage', values={'needs.pre-flight.outputs.docs_only': 'true'})
assert not enabled('Coverage', values={'needs.pre-flight.outputs.is_deployment_workflow': 'true'})

# Execute the real aggregate step for every pair of terminal build results.
aggregate = re.search(r'        run: \|\n((?:          .+\n)+)', jobs['cicd-container-build'])[1]
for cpu_result in ('success', 'failure', 'cancelled', 'skipped'):
    for external_result in ('success', 'failure', 'cancelled', 'skipped'):
        result = subprocess.run(
            ['bash', '-e', '-c', textwrap.dedent(aggregate)],
            env={**os.environ, 'CPU_RESULT': cpu_result, 'EXTERNAL_RESULT': external_result},
        )
        expected = (cpu_result, external_result) in {('success', 'skipped'), ('skipped', 'success')}
        assert (result.returncode == 0) == expected, (cpu_result, external_result)

print('CI queue dependencies, admission conditions, and build result checks passed')
PY
