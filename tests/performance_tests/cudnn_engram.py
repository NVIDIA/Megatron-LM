# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Engram/DDP training comparison with reduced table and full channel geometry."""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
import transformer_engine

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.fusions.test_cudnn_engram import close, make_module
from tests.unit_tests.test_utilities import Utils


def main():
    """Check three-way numerical parity before paired compiled-module timing."""
    import cudnn

    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--tokens', type=int, default=4096)
    p.add_argument('--microbatches', type=int, default=32)
    p.add_argument('--pairs', type=int, default=12)
    p.add_argument('--baseline', choices=['native', 'compiled'], default='compiled')
    p.add_argument('--profile', action='store_true')
    a = p.parse_args()
    if not __debug__:
        p.error('correctness checks require Python without -O')
    if Path(a.output).exists():
        p.error('output already exists; choose a fresh result path')
    if a.tokens < 64 or a.tokens > 8192 or a.tokens % 64:
        p.error('tokens must be 64..8192, divisible by 64')
    if a.microbatches < 1 or a.pairs < 1:
        p.error('microbatches and pairs must be positive')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        p.error('this benchmark validates world size one only')
    torch.cuda.set_device(0)
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(4176)
    torch.manual_seed(4176)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    result = {
        'status': 'running',
        'scope': 'actual Engram + MCore DDP forward/backward FP32 main_grad, world1; table reduced to 1024 rows, full H5120/4 streams/projection input6144; no full model or EP claim',
        'geometry': {
            'tokens': a.tokens,
            'hidden': 5120,
            'streams': 4,
            'embedding_width': 6144,
            'table_rows': 1024,
            'microbatches': a.microbatches,
        },
        'device': torch.cuda.get_device_name(),
        'capability': torch.cuda.get_device_capability(),
        'sms': torch.cuda.get_device_properties(0).multi_processor_count,
        'torch': torch.__version__,
        'te': transformer_engine.__version__,
        'cudnn': cudnn.__file__,
        'baseline': a.baseline,
        'checks': {},
        'timing': [],
    }

    def save():
        Path(a.output).write_text(json.dumps(result, indent=2) + '\n')

    def verify(name, x, y, limit=0.03):
        close(x, y, limit)
        result['checks'][name] = {
            'relative_l2': float(
                (x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-20)
            ),
            'limit': limit,
        }

    save()
    try:
        native = make_module(groups, 'native', source_width=True)
        candidate = make_module(groups, 'cudnn', source_width=True)
        with torch.no_grad():
            native.q_weight.uniform_(0.7, 1.3)
            native.k_weight.uniform_(0.7, 1.3)
        candidate.load_state_dict(native.state_dict())
        modules = {'native': native, 'cudnn': candidate}
        if a.baseline == 'compiled':
            native.forward = torch.compile(native.forward, fullgraph=False, dynamic=False)
        if a.baseline == 'compiled':
            candidate.forward = torch.compile(candidate.forward, fullgraph=False, dynamic=False)
        result['candidate_compiled'] = a.baseline == 'compiled'
        arms = {
            k: DistributedDataParallel(
                m.config,
                DistributedDataParallelConfig(
                    overlap_grad_reduce=False,
                    grad_reduce_in_fp32=True,
                    use_distributed_optimizer=False,
                ),
                m,
                pg_collection=groups,
            )
            for k, m in modules.items()
        }
        xs = [
            torch.randn(a.tokens, 1, 20480, device='cuda', dtype=torch.bfloat16) * v
            for v in (1.0, 0.03)
        ]
        ids = [torch.randint(0, 1024, (1, a.tokens, 1, 24), device='cuda') for _ in range(2)]
        masks = [
            torch.ones(1, a.tokens, device='cuda', dtype=torch.bool),
            torch.rand(1, a.tokens, device='cuda') > 0.13,
        ]
        dys = [torch.randn_like(x) / 128 for x in xs]

        def execute(arm, mbs, capture=False):
            ddp = arms[arm]
            ddp.zero_grad_buffer()
            saved = []
            for mb in range(mbs):
                j = mb % 2
                x = xs[j].detach().requires_grad_()
                y = ddp(x, ids[j], masks[j])
                y.backward(dys[j])
                if capture:
                    saved.append((y.detach().clone(), x.grad.detach().clone()))
            ddp.finish_grad_sync()
            if capture:
                return saved, {n: v.main_grad.clone() for n, v in modules[arm].named_parameters()}

        result['stage'] = 'correctness'
        save()
        if a.baseline == 'compiled':
            compiled_forward = native.forward
            native.forward = compiled_forward._torchdynamo_orig_callable
            eager_reference = execute('native', 2, True)
            native.forward = compiled_forward
        else:
            eager_reference = None
        reference = execute('native', 2, True)
        actual = execute('cudnn', 2, True)
        if eager_reference is not None:
            for provider, measured in [('compiled', reference), ('cudnn', actual)]:
                for mb, (ref, got) in enumerate(zip(eager_reference[0], measured[0])):
                    for label, x, y in zip(('out', 'dx'), got, ref):
                        verify(
                            f'{provider}_vs_eager.mb{mb}.{label}',
                            x,
                            y,
                            0.01 if label == 'out' else 0.03,
                        )
                for n, v in eager_reference[1].items():
                    verify(f'{provider}_vs_eager.{n}', measured[1][n], v)
            del eager_reference
        result['compiler_stats'] = {k: dict(v) for k, v in torch._dynamo.utils.counters.items()}
        result['parameter_dtypes'] = {n: str(v.dtype) for n, v in native.named_parameters()}

        for mb, (ref, got) in enumerate(zip(reference[0], actual[0])):
            for label, x, y in zip(('out', 'dx'), got, ref):
                verify(f'mb{mb}.{label}', x, y, 0.01 if label == 'out' else 0.03)
        for n, v in reference[1].items():
            verify(n, actual[1][n], v)
        del reference, actual
        gc.collect()
        torch.cuda.empty_cache()
        result['stage'] = 'timing'
        save()
        for arm in arms:
            execute(arm, a.microbatches)
        for pair in range(a.pairs):
            sample = {}
            for arm in (list(arms) if pair % 2 == 0 else list(reversed(arms))):
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                t = time.perf_counter()
                start.record()
                execute(arm, a.microbatches)
                end.record()
                end.synchronize()
                sample[arm] = {
                    'wall_ms': (time.perf_counter() - t) * 1000,
                    'gpu_ms': start.elapsed_time(end),
                }
            result['timing'].append(sample)
            save()
            print('pair', pair, sample, flush=True)
        if a.profile:
            from torch.profiler import ProfilerActivity, profile

            for arm in arms:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    execute(arm, a.microbatches)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(a.output + '.' + arm + '.trace.json')
        result['median_wall_ms'] = {
            arm: statistics.median(s[arm]['wall_ms'] for s in result['timing']) for arm in arms
        }
        result['latency_reduction'] = (
            1 - result['median_wall_ms']['cudnn'] / result['median_wall_ms']['native']
        )
        result['paired_wins'] = sum(
            s['cudnn']['wall_ms'] < s['native']['wall_ms'] for s in result['timing']
        )
        result['route'] = {
            'candidate_forward_calls': candidate._cudnn_gate.forward_calls,
            'candidate_plan_builds': candidate._cudnn_gate.plan_builds,
        }
        result['status'] = 'passed'
        save()
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ('median_wall_ms', 'latency_reduction', 'paired_wins', 'route')
                },
                indent=2,
            ),
            flush=True,
        )
    except BaseException as e:
        result['status'] = 'failed'
        result['error'] = repr(e)
        save()
        raise
    finally:
        Utils.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
