# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Check the rank placement and NVLink fabric of a GLM-5.2 allocation; print rank 0's node.

Reads the `srun -l` output of the README's placement probe: every task of the training layout
prints `node=<SLURMD_NODENAME> localid=<SLURM_LOCALID> imex=<IMEX channels>` followed by
`nvidia-smi -q -i <SLURM_LOCALID>`. Fails unless:

* tasks 0..N-1 sit four to a node in block order (local id = rank % 4), on one driver version;
* every GPU's fabric registration finished (State Completed, Status Success, a CliqueId and a
  ClusterUUID) and every fabric health row reads its healthy value or N/A (a row not listed in
  HEALTHY must not report a problem such as True, Degraded or Error);
* every node lists an IMEX channel;
* each pipeline stage's ranks (64 consecutive ranks) share one (ClusterUUID, CliqueId), i.e.
  one NVLink domain.

Standard library only. If a check fails, prints PLACEMENT ERROR lines to stderr and nothing to
stdout; otherwise prints a PLACEMENT OK line, which names the driver version and the GPUs' power
limits, to stderr and rank 0's node to stdout.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict

# Healthy value of each Fabric > Health row of `nvidia-smi -q` (R580 drivers). A row that the
# driver or fabric does not report reads N/A or Not Supported.
HEALTHY = {
    'Summary': 'Healthy',
    'Bandwidth': 'Full',
    'Route Recovery in progress': 'False',
    'Route Unhealthy': 'False',
    'Access Timeout Recovery': 'False',
    'Incorrect Configuration': 'None',
    'Partition Assigned': 'True',
}
UNREPORTED = ('n/a', 'not supported')
# Values that a health row missing from HEALTHY must not contain.
PROBLEM = re.compile(
    r'^true$|degraded|unhealthy|incorrect|fail|lost|limited|in progress|error', re.I
)
# GPU attribute sections that parse() reads. `GPU Power Readings` holds the GPU's own power
# limit; `Module Power Readings` holds a separate module limit.
SECTIONS = {'Fabric': 'fabric', 'GPU Power Readings': 'power'}


def parse(path):
    """Return {rank: {'node', 'localid', 'imex', 'driver', 'power_limit', 'fabric': {...},
    'health': {...}}}."""
    lines = defaultdict(list)
    with open(path) as stream:
        for line in stream:
            match = re.match(r'^\s*(\d+): (.*)$', line.rstrip('\n'))
            if match:
                lines[int(match.group(1))].append(match.group(2))
    tasks = {}
    for rank, text in lines.items():
        task = {'fabric': {}, 'health': {}, 'driver': None, 'node': None, 'power_limit': None}
        section = None
        for line in text:
            header = re.match(r'^node=(\S+) localid=(\d+) imex=(.*)$', line)
            if header:
                task.update(
                    node=header.group(1), localid=int(header.group(2)), imex=header.group(3).split()
                )
                continue
            # nvidia-smi -q nests by four spaces: GPU attributes at 4, Fabric and power fields at
            # 8, Fabric > Health fields at 12.
            indent = len(line) - len(line.lstrip(' '))
            key, sep, value = line.strip().partition(':')
            key, value = key.strip(), value.strip()
            if key == 'Driver Version' and sep:
                task['driver'] = value
            if indent <= 4:
                section = SECTIONS.get(key) if indent == 4 and not sep else None
            elif section == 'power':
                if indent == 8 and key == 'Current Power Limit' and sep:
                    task['power_limit'] = value
            elif section is not None and indent == 8:
                section = 'health' if key == 'Health' and not sep else 'fabric'
                if sep:
                    task['fabric'][key] = value
            elif section == 'health' and indent == 12 and sep:
                task['health'][key] = value
        tasks[rank] = task
    return tasks


def check(tasks, ranks, per_node, per_stage):
    errors = []
    if sorted(tasks) != list(range(ranks)):
        return [f'expected tasks 0..{ranks - 1}, got {len(tasks)} tasks'], None
    for rank, task in sorted(tasks.items()):
        where = f'rank {rank} ({task["node"]})'
        if task['node'] is None:
            errors.append(f'rank {rank}: no node= line')
            continue
        if task['localid'] != rank % per_node:
            errors.append(f'{where}: local id {task["localid"]}, expected {rank % per_node}')
        if task['node'] != tasks[rank - rank % per_node]['node']:
            errors.append(f'{where}: not on the node of rank {rank - rank % per_node}')
        if not task['imex']:
            errors.append(f'{where}: no IMEX channel')
        fabric = task['fabric']
        if fabric.get('State') != 'Completed' or fabric.get('Status') != 'Success':
            errors.append(
                f'{where}: fabric State {fabric.get("State")} Status {fabric.get("Status")}'
            )
        if fabric.get('CliqueId', 'N/A') == 'N/A' or fabric.get('ClusterUUID', 'N/A') == 'N/A':
            errors.append(f'{where}: no fabric CliqueId/ClusterUUID')
        for key, value in task['health'].items():
            if key in HEALTHY:
                problem = value.lower() not in (HEALTHY[key].lower(), *UNREPORTED)
            else:
                problem = PROBLEM.search(value) is not None
            if problem:
                errors.append(f'{where}: fabric health {key} = {value}')
    nodes = [tasks[rank]['node'] for rank in range(0, ranks, per_node)]
    if len(set(nodes)) != len(nodes):
        errors.append('a node hosts ranks of two blocks of four')
    drivers = {task['driver'] for task in tasks.values()}
    if len(drivers) != 1:
        errors.append(f'driver versions differ: {sorted(map(str, drivers))}')
    for start in range(0, ranks, per_stage):
        domains = {
            (tasks[rank]['fabric'].get('ClusterUUID'), tasks[rank]['fabric'].get('CliqueId'))
            for rank in range(start, start + per_stage)
        }
        if len(domains) != 1:
            errors.append(
                f'ranks {start}-{start + per_stage - 1} span NVLink domains '
                f'{sorted(domains, key=str)}'
            )
    return errors, nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('probe', help='srun -l output of the placement probe')
    parser.add_argument('--ranks', type=int, default=256)
    parser.add_argument('--ranks-per-node', type=int, default=4)
    parser.add_argument('--ranks-per-stage', type=int, default=64)
    options = parser.parse_args()
    tasks = parse(options.probe)
    errors, nodes = check(tasks, options.ranks, options.ranks_per_node, options.ranks_per_stage)
    if errors:
        for error in errors:
            print(f'PLACEMENT ERROR {error}', file=sys.stderr)
        sys.exit(1)
    # Reported, not checked: compare them with the nodes of the measured runs (README).
    limits = Counter(task['power_limit'] or 'unknown' for task in tasks.values())
    print(
        f'PLACEMENT OK: {options.ranks} ranks on {len(nodes)} nodes; '
        f'each {options.ranks_per_stage}-rank stage in one NVLink domain; '
        f'driver {tasks[0]["driver"]}; GPU power limit '
        + ', '.join(
            f'{limit} on {count} GPU{"s" if count != 1 else ""}'
            for limit, count in sorted(limits.items())
        ),
        file=sys.stderr,
    )
    print(nodes[0])


if __name__ == '__main__':
    main()
