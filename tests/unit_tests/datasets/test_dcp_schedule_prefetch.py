# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The runtime DCP scheduler from start-up to the plan of a step, on CPU: a rank is a thread."""

import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from megatron.core.datasets import data_schedule, dcp_runtime_worker
from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.data_schedule import validate_dynamic_cp_runtime, wrap_data_iterator
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import RuntimeScheduleError

_USP, _BENCHMARK = {'type': 'dynamic_usp'}, {'type': 'dcp_benchmark', 'hardware': 'h100'}
_BENCHMARK['non_attention'] = {'forward_ms_per_layer': 0.5, 'backward_ms_per_layer': 1.0}
# Ordinary-GQA training settings inside the domain of the packaged attention predictor.
_GQA = dict(num_attention_heads=32, num_query_groups=4, kv_channels=128, num_layers=48, bf16=True)
_GQA.update(attention_backend='fused', cp_comm_type='p2p', thd_tail_padding_policy='extend_last')
_H100 = 'hardware=h100 (NVLink domain: 8 consecutive ranks), CP group sizes='
_NO_REMAT = dict(dynamic_cp={8: None}, dp_gtp_remat=None, dp_cp_gtp_remat=None)


def _config(path, payload=_USP, vpp=None, **training):
    path.write_text(json.dumps(payload))
    values = dict(sequence_packing_scheduler='default_dynamic_cp', pad_packed_seq_alignment='max')
    values.update(max_seqlen_per_dp_cp_rank=1024, min_dynamic_context_parallel_size=8)
    values.update(virtual_pipeline_model_parallel_size=vpp, microbatch_group_size_per_vp_stage=None)
    if payload['type'] == 'dcp_benchmark':
        values.update(_GQA, max_seqlen_per_dp_cp_rank=4096, min_dynamic_context_parallel_size=1)
    return SimpleNamespace(dynamic_cp_runtime_config=str(path), **dict(values, **training))


class _Ranks:
    """Stages x 8 DPxCP ranks as threads; a collective blocks until its whole group issued it. One
    stage: the real CPU worker; else the planner here, ``fault`` after ``plans`` of the last."""

    def __init__(self, monkeypatch, stages=2):
        self.local, self.rounds, self.barriers, self.stages = threading.local(), {}, {}, stages
        self.calls, self.started, self.local.rank, self.process = {}, [], 0, prefetch._CpuPlanner
        tensor, zeros = torch.tensor, torch.zeros  # tensors stay on the CPU
        monkeypatch.setattr(torch, 'tensor', lambda *a, **k: tensor(*a, **{**k, 'device': 'cpu'}))
        monkeypatch.setattr(torch, 'zeros', lambda *a, **k: zeros(*a, **{**k, 'device': 'cpu'}))
        monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
        for name in ('all_reduce', 'broadcast_object_list'):
            monkeypatch.setattr(torch.distributed, name, getattr(self, name))
        monkeypatch.setattr(torch.distributed, 'get_global_rank', lambda group, i: group.members[i])
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda: self.local.rank)
        monkeypatch.setattr(prefetch, 'get_batch_and_global_seqlens', self.gather)
        monkeypatch.setattr(prefetch, '_CpuPlanner', self.start_worker)
        monkeypatch.setattr(data_schedule, 'broadcast_scalars', lambda values, *_: values)
        names = 'reroute_samples_to_dcp_ranks build_packed_microbatches create_data_iterator'
        for name in (*names.split(), 'get_global_real_seqlen_stats'):  # the token work of run()
            monkeypatch.setattr(data_schedule, name, lambda *a, **k: (0.0, 0.0))

    def group(self, ids, size=None):  # ``size`` counts ranks that are not simulated
        me = lambda: ids.index(self.local.rank)  # noqa: E731
        return SimpleNamespace(members=ids, size=lambda: size or len(ids), rank=me)

    def groups(self, rank):
        one, pp = self.group([rank]), self.group(range(rank % 8, 8 * self.stages, 8))
        dp_cp = self.group(range(rank - rank % 8, rank - rank % 8 + 8))
        return SimpleNamespace(dp=one, tp=one, pp=pp, dp_cp=dp_cp, **_NO_REMAT)

    def exchange(self, group, call, value):
        key = tuple(group.members)
        self.calls.setdefault(self.local.rank, []).append((call, len(key)))
        given = self.rounds.setdefault(key, {})
        take = lambda: given.update(now=[given.pop(rank) for rank in key])  # noqa: E731
        given[self.local.rank] = value
        self.barriers.setdefault(key, threading.Barrier(len(key), take)).wait(30)
        return given['now']

    def all_reduce(self, tensor, op=torch.distributed.ReduceOp.SUM, group=None):
        rows = torch.stack(self.exchange(group, (op.name, tensor.numel()), tensor.clone()))
        tensor.copy_(rows.max(0).values if op.name == 'MAX' else rows.sum(0))

    def broadcast_object_list(self, payload, src=0, group=None, device=None):
        sent = self.exchange(group, 'broadcast', list(payload))[group.members.index(src)]
        payload[:] = json.loads(json.dumps(sent))

    def gather(self, iterator, count, dp_group, include_original_seqlens=False):
        """Stands for get_batch_and_global_seqlens; a loader item is the list of its lengths."""
        lengths = [length for _ in range(count) for length in next(iterator)]
        padded = [-(-length // 16) * 16 for length in lengths]
        keys = data_schedule.DpBalancedScheduler.get_required_sample_keys(None)
        batch = [dict.fromkeys(keys, 0) for _ in lengths]
        return batch, list(enumerate(padded)), [], [], padded, lengths

    def start_worker(self, spec):
        self.started.append(self.local.rank)
        if self.stages == 1:
            return self.process(spec)
        faulty, made = self.fault is not None and self.local.rank >= 8, []
        if faulty and not self.plans:
            self.fault()
        planner = dcp_runtime_worker._initialize_planner(spec)

        def solve(*request):
            made.append(request)
            if faulty and len(made) > self.plans:
                return self.fault()
            return dcp_runtime_worker._solve(planner, spec, *request)

        return SimpleNamespace(solve=solve, stop=int, close=int)


class _Rerun(SimpleNamespace):  # a RerunDataIterator: a replayed step reads ``replay``
    def __next__(self):
        return next(self.replay if self.replaying else self.iterable)


def _train(sim, tmp_path, vpp=2, asynchronous=True, ahead=True, who='', what='', value=None):
    """Every rank validates and calls wrap_data_iterator for five steps (the third replays the
    first, the last two read one loader item) with a fault of ``_FAULTS``: (data, counts, error)."""
    sim.fault, sim.plans = value if callable(value) else None, 2 * (what != 'no worker')

    def body(rank):
        hit = who == 'all' or rank >= 8 * (sim.stages - 1) and (who == 'stage' or rank % 8 == 5)
        own = lambda kind, default: value if hit and what == kind else default  # noqa: E731
        sim.local.rank, rng, counts, error = rank, random.Random(0), [], None
        items = [[rng.randint(100, 6000) for _ in range(6)] for _ in range(6)]
        items[-2][0] += own('tokens', 0)
        data = _Rerun(iterable=iter(items), items=items, replay=iter(items), replaying=False)
        file = dict(_USP, trace_dir=str(tmp_path), **{'async': asynchronous}, **own('file', {}))
        config, groups = _config(tmp_path / f'{rank}.json', file, own('vpp', vpp)), sim.groups(rank)
        try:
            validate_dynamic_cp_runtime(config, groups)
            for replayed, n in zip((False, False, True, False, False), (2, 2, 2, 1, 1)):
                data.replaying, passed = replayed, [data, None] if vpp else data
                counts.append(wrap_data_iterator(passed, config, n, groups, prefetch_next=ahead)[1])
        except Exception as caught:  # pylint: disable=broad-exception-caught
            error = caught
        getattr(data, '_dcp_schedule_prefetcher', SimpleNamespace(close=int)).close()
        return data, counts, error

    with ThreadPoolExecutor(8 * sim.stages) as pool:
        return list(pool.map(body, range(8 * sim.stages)))


@pytest.mark.parametrize(
    'payload,training,world,stages,outcome',
    [
        (_BENCHMARK, {}, 64, 1, _H100 + '[1, 2, 4, 8, 16, 32, 64], at most 64 sequences per CP'),
        (dict(_BENCHMARK, max_cp_size=4), {}, 12, 1, _H100 + '[1, 2, 4], at most 64 sequences'),
        # The dummy tail sequence of the padding takes one of the 64 entries of a pack.
        (_USP, dict(thd_max_packed_sequences=64), 32, 2, 'sizes=[8, 16, 32], at most 63 sequences'),
        (dict(_USP, solver='x'), {}, 32, 1, "unknown key(s) ['solver'] for type 'dynamic_usp'"),
        (_USP, dict(sequence_packing_scheduler='dp_balanced'), 32, 1, 'requires dynamic context'),
        (_USP, dict(min_dynamic_context_parallel_size=1), 32, 1, 'in (8, 16, 32); got DPxCP=32'),
        (_BENCHMARK, {}, 8, 2, 'only with type "dynamic_usp" without "hardware"'),
        (_BENCHMARK, {}, 12, 1, 'set "max_cp_size": 4 in the runtime configuration'),
        (dict(_BENCHMARK, max_cp_size=64), {}, 128, 1, 'up to 64; this run has DPxCP=128'),
        (_BENCHMARK, dict(num_attention_heads=7), 8, 1, 'this model has (7, 4, 128)'),
        # An odd per-rank buffer passes the domain check, but a ring has no layout for it.
        (_BENCHMARK, dict(max_seqlen_per_dp_cp_rank=4097), 8, 1, 'set "max_cp_size": 1 in the'),
    ],
)
def test_startup_line_or_refusal(tmp_path, monkeypatch, payload, training, world, stages, outcome):
    sim, config = _Ranks(monkeypatch), _config(tmp_path / 'runtime.json', payload, **training)
    groups = SimpleNamespace(dp_cp=sim.group([0], world), pp=sim.group([0], stages))
    try:
        seen = prefetch.describe_runtime_spec(validate_dynamic_cp_runtime(config, groups))
    except ValueError as refusal:
        seen = f'Refused: {refusal}'
    assert outcome in seen


@pytest.mark.parametrize('async_on', [False, True])
@pytest.mark.parametrize('stages,vpp', [(1, None), (2, 2), (2, None), (4, None)])
def test_stages_plan_alone_and_agree(tmp_path, monkeypatch, stages, vpp, async_on):
    sim = _Ranks(monkeypatch, stages)
    outcomes = _train(sim, tmp_path, vpp, async_on)
    counts, held = outcomes[0][1], set()
    # Under virtual stages every count of microbatches is a multiple of the stages; else any is.
    assert any(count % stages for count in counts) is (stages > 1 and not vpp)
    schedule = None if stages == 1 else 'interleaved' if vpp else 'non_interleaved'
    for rank, (data, own_counts, error) in enumerate(outcomes):
        assert error is None and own_counts == counts and len(counts) == 5
        trace = map(json.loads, (tmp_path / f'rank{rank:05d}.jsonl').read_text().splitlines())
        *_, last = steps = [step for step in trace if step['event'] == 'consume']
        # "async" on: a batch read ahead is used, but not by a replay or a step of another count.
        assert [step['lookahead'] for step in steps] == [False, async_on, False, False, async_on]
        assert [step['global_lengths'] for step in steps[3:]] == data.items[4:]
        assert last['statistics'].get('pipeline', {}).get('schedule') == schedule
        assert last['incumbent_error'] is last['statistics']['incumbent_unpriced'] is None
        assert stages > 1 or last['worker_pid'] != os.getpid()  # one stage: the real worker process
        assert stages > 1 or last['statistics']['selected_incumbent']  # and kept the default layout
        held.add((json.dumps(last['rows']), last.get('pipeline_agreement_key'), *sim.calls[rank]))
    # Every rank holds the same plan and key; one CPU worker per stage, started by its leader.
    assert len(held) == 1 and sorted(sim.started) == list(range(0, 8 * stages, 8))


_ANOTHER = 'another pipeline stage'
# (who, what, value, error of the ranks with the fault, phrase of the others' error or None for the
# same). Who: the last of two stages or its rank 5 ('stage rank'); of one stage, rank 5 or all.
# What: after two plans the CPU worker of the stage is the value, or it does not start; a sample of
# the fifth item grows by it; the file has it; same file, but the rank has no virtual stages.
_FAULTS = [
    ('stage', 'worker', lambda: 1 / 0, (RuntimeError, 'CPU DCP planner failed'), _ANOTHER),
    ('stage', 'silent', lambda: threading.Event().wait(2), (RuntimeError, 'within 1 s'), _ANOTHER),
    ('stage', 'answer', dict(rows=[[[0]] * 8]).copy, (RuntimeError, 'stages disagree'), None),
    ('stage', 'no worker', lambda: 1 / 0, (RuntimeError, 'ZeroDivisionError'), _ANOTHER),
    ('stage', 'file', {'max_cp_size': 8}, (ValueError, 'stages read different settings'), None),
    ('rank', 'file', {'max_cp_size': 8}, (ValueError, '1 of 8 DPxCP ranks read a runtime'), None),
    ('stage rank', 'vpp', None, (ValueError, '1 of 8 DPxCP ranks of this pipeline'), _ANOTHER),
    ('rank', 'tokens', 16, (RuntimeError, 'CP replicas disagree'), None),
    ('all', 'tokens', 8192, (RuntimeScheduleError, 'requires CP >= 9'), None),
    ('all', 'tokens', -8192, (RuntimeError, 'positive integer logical lengths'), None),
]


@pytest.mark.parametrize('lookahead', [False, True])
@pytest.mark.parametrize('fault', _FAULTS, ids=lambda fault: ' '.join(fault[:2]))
def test_a_fault_stops_every_rank_and_nobody_waits(tmp_path, monkeypatch, fault, lookahead):
    # Two stages for the fault of a stage; else one, whose CPU worker is the real process.
    (who, what, _, own, others), sim = fault, _Ranks(monkeypatch, 1 + ('stage' in fault[0]))
    assert prefetch.PLANNER_REPLY_TIMEOUT_S == 300.0  # below the 10 minutes of the process groups
    monkeypatch.setattr(prefetch, 'PLANNER_REPLY_TIMEOUT_S', 1.0 if what == 'silent' else 300.0)
    finished = {'no worker': 0, 'file': 0, 'vpp': 0, 'tokens': 3}.get(what, 2)
    outcomes = _train(sim, tmp_path, 2 if 'stage' in who else None, True, lookahead, *fault[:3])
    for rank, (_, counts, error) in enumerate(outcomes):
        phrase = own[1] if rank >= 8 or others is None else others
        assert len(counts) == finished and type(error) is own[0] and phrase in str(error)
    assert len(set(map(tuple, sim.calls.values()))) == 1  # the same collectives before the error
