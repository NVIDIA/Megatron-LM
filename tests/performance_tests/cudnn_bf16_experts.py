# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare real TEGroupedMLP/DDP BF16 experts for DSv4.1 on SM100.

Run with torchrun from the repository root. Numerical checks precede paired
AB/BA measurements. This covers local experts after dispatch, without routing,
EP communication, shared experts or optimizer work. It is not full-model timing.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import transformer_engine

from tests.unit_tests.fusions.test_cudnn_bf16_module import (
    _assert_error,
    _canonical,
    _interleave,
    _make_arm,
    _run,
)
from tests.unit_tests.test_utilities import Utils


def main() -> None:
    """Compare cuDNN with a selected native TE route at the expert-module boundary."""
    import cudnn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--microbatches', type=int, default=32)
    parser.add_argument('--pairs', type=int, default=12)
    parser.add_argument('--baseline', choices=['legacy', 'auto'], default='auto')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--counts-file')
    args = parser.parse_args()
    if args.pairs < 1 or args.microbatches < 1:
        parser.error("pairs and microbatches must be positive")
    if Path(args.output).exists():
        parser.error("output already exists; use a fresh file")
    os.environ['NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM'] = (
        '0' if args.baseline == 'legacy' else '1'
    )
    torch.cuda.set_device(0)
    Utils.initialize_model_parallel()
    torch.manual_seed(4122)
    experts, hidden, intermediate, rows = (48, 5120, 2304, 24576)
    count_fixture = json.loads(Path(args.counts_file).read_text()) if args.counts_file else None
    if count_fixture:
        assert len(count_fixture['counts']) == experts and all(
            (type(v) is int and v >= 0 for v in count_fixture['counts'])
        )
        rows = sum(count_fixture['counts'])
    adapter_file = __import__("megatron.core.fusions.cudnn_bf16_experts", fromlist=[""]).__file__
    assert adapter_file is not None
    result = dict(
        status='running',
        scope='actual TEGroupedMLP + DDP fwd/bwd/main_grad accumulation, world1, no router/EP/optimizer',
        device=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        sms=torch.cuda.get_device_properties(0).multi_processor_count,
        geometry=dict(
            local_experts=experts,
            hidden=hidden,
            intermediate=intermediate,
            rows=rows,
            microbatches=args.microbatches,
        ),
        recipe=(
            'DSv4.1 expert geometry dba1be0a40aa45a94ad051997016db3960a90277; '
            'Megatron-Bridge BF16 EP8 recipe ece66187628007544c33fcff280089b61d5c758a; '
            'post-dispatch balanced and synthetic uneven routing strata, not checkpoint traces'
        ),
        counts_fixture=count_fixture,
        baseline=args.baseline,
        torch=torch.__version__,
        te=transformer_engine.__version__,
        cudnn=cudnn.__file__,
        adapter_sha256=hashlib.sha256(Path(adapter_file).read_bytes()).hexdigest(),
        checks={},
        timing=[],
    )

    def save() -> None:
        """Persist partial evidence so failed runs remain inspectable."""
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')

    def check(name: str, x: torch.Tensor, y: torch.Tensor, limit: float) -> None:
        """Check a numerical boundary before any performance sampling."""
        _assert_error(x, y, limit)
        result['checks'][name] = {
            'relative_l2': float(
                (x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-20)
            ),
            'limit': limit,
        }

    save()
    try:
        arms = {
            b: _make_arm(b, experts, hidden, intermediate) for b in ['transformer_engine', 'cudnn']
        }
        native, cudnn_experts = (arms['transformer_engine'][0], arms['cudnn'][0])
        with torch.no_grad():
            for (n, s), (_, d) in zip(native.named_parameters(), cudnn_experts.named_parameters()):
                d.copy_(_interleave(s) if n.startswith('linear_fc1') else s)
        counts0 = (
            torch.tensor(count_fixture['counts'], device='cuda', dtype=torch.int64)
            if count_fixture
            else torch.full((experts,), rows // experts, device='cuda', dtype=torch.int64)
        )
        counts1 = counts0.clone()
        counts1[1] += counts1[0]
        counts1[0] = 0
        counts = [counts0, counts1]
        xs = [
            torch.randn(rows, hidden, device='cuda', dtype=torch.bfloat16) * (1 if j == 0 else 8)
            for j in range(2)
        ]
        ps = [torch.rand(rows, device='cuda') * 0.25 for _ in range(2)]
        dys = [torch.randn_like(x) / 128 for x in xs]
        result['stage'] = 'correctness'
        save()
        ref = _run(native, arms['transformer_engine'][1], xs, counts, ps, dys, False)
        out = _run(cudnn_experts, arms['cudnn'][1], xs, counts, ps, dys, False)
        for j, (actual, expected) in enumerate(zip(out[0], ref[0])):
            for k, (x, y) in enumerate(zip(actual, expected)):
                check(f"mb{j}.{['y', 'dx', 'dp'][k]}", x, y, 0.01 if k == 0 else 0.03)
        for n, g in ref[1].items():
            check(n, _canonical(out[1][n]) if n.startswith('linear_fc1') else out[1][n], g, 0.03)
        del ref, out
        gc.collect()
        torch.cuda.empty_cache()
        print('correctness passed', len(result['checks']), flush=True)

        def step(backend: str) -> None:
            """Execute one real module training step with accumulated main_grad."""
            m, ddp = arms[backend]
            ddp.zero_grad_buffer()
            m.set_is_first_microbatch()
            for mb in range(args.microbatches):
                j = mb % 2
                x = xs[j].detach().requires_grad_()
                prob = ps[j].detach().requires_grad_()
                with nullcontext() if mb == args.microbatches - 1 else ddp.no_sync():
                    y, bias = ddp(x, counts[j], prob)
                    assert bias is None
                    y.backward(dys[j])
            ddp.finish_grad_sync()

        for backend in arms:
            step(backend)
        result['stage'] = 'timing'
        save()
        for pair in range(args.pairs):
            sample = {}
            order = list(arms) if pair % 2 == 0 else list(reversed(arms))
            for backend in order:
                torch.cuda.synchronize()
                t = time.perf_counter()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                step(backend)
                end.record()
                end.synchronize()
                sample[backend] = {
                    'wall_ms': (time.perf_counter() - t) * 1000,
                    'gpu_ms': start.elapsed_time(end),
                }
            result['timing'].append(sample)
            save()
            print('pair', pair, sample, flush=True)
        if args.profile:
            from torch.profiler import ProfilerActivity, profile

            result['profile'] = {}
            for backend in arms:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    step(backend)
                    torch.cuda.synchronize()
                result['profile'][backend] = [
                    dict(
                        name=e.key,
                        device_us=e.device_time_total,
                        cpu_us=e.cpu_time_total,
                        calls=e.count,
                    )
                    for e in prof.key_averages()
                    if e.device_time_total > 0
                ]
                prof.export_chrome_trace(args.output + '.' + backend + '.trace.json')
        owner = cudnn_experts._cudnn_bf16_ops[0]
        result['route'] = {
            key: getattr(owner, key)
            for key in ['forward_calls', 'backward_calls', 'max_inflight', 'created_contexts']
        }
        result['median_wall_ms'] = {
            b: statistics.median((r[b]['wall_ms'] for r in result['timing'])) for b in arms
        }
        result['latency_reduction'] = (
            1 - result['median_wall_ms']['cudnn'] / result['median_wall_ms']['transformer_engine']
        )
        result['status'] = 'passed'
        save()
        print(
            json.dumps({k: result[k] for k in ['median_wall_ms', 'latency_reduction', 'route']}),
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


if __name__ == '__main__':
    main()
