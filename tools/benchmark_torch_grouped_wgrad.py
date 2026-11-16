# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Isolated BF16 TE GroupedLinear and dW comparison using captured MoE shapes."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm

from megatron.core import parallel_state
from megatron.core.extensions import transformer_engine as impl
from megatron.core.extensions.torch_grouped_wgrad import _grouped_mm
from megatron.core.extensions.transformer_engine import TEGroupedLinear
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal


def error(actual, ref):
    a, b = actual.detach().float(), ref.detach().float()
    return {
        'relative_l2': float((a - b).norm() / b.norm().clamp_min(1e-30)),
        'max_abs': float((a - b).abs().max()),
        'finite': bool(torch.isfinite(a).all()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shapes', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--layers', type=int, default=20)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--rounds-per-block', type=int, default=3)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    torch.set_num_threads(2)
    dist.init_process_group('nccl')
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    model_parallel_cuda_manual_seed(1234)
    stats = {'calls': 0}
    original = impl.torch_grouped_linear

    def counted(*args, **kwargs):
        stats['calls'] += 1
        return original(*args, **kwargs)

    impl.torch_grouped_linear = counted
    grouped_mm = _grouped_mm()
    if grouped_mm is None:
        raise RuntimeError("This benchmark requires PyTorch grouped matmul.")
    torch.manual_seed(20261011)
    rows = [json.loads(s) for s in Path(args.shapes).read_text().splitlines()]
    cases = []
    for seq in (8192, 32768):
        chosen = [r for r in rows if r['seq'] == seq and r['step'] == 0][: args.layers * 2]
        assert len(chosen) == args.layers * 2
        for idx, r in enumerate(chosen):
            cases.append(
                {
                    **r,
                    'routing_case': idx // 2,
                    'fc': 'fc1' if (r['I'], r['O']) == (2048, 1024) else 'fc2',
                }
            )
    manifest = {
        'torch': torch.__version__,
        'device': str(torch.cuda.get_device_properties(0)),
        'helper_sha256': hashlib.sha256(
            (Path(impl.__file__).parent / 'torch_grouped_wgrad.py').read_bytes()
        ).hexdigest(),
        'upstream_base': '736353b7e01408bef826ff55f871378f5bfc2f0c',
        'te_impl': impl.__file__,
        'te_impl_sha256': hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest(),
        'shape_file': args.shapes,
        'shape_sha256': hashlib.sha256(Path(args.shapes).read_bytes()).hexdigest(),
        'case_count': len(cases),
        'args': vars(args),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'env': {
            k: os.getenv(k)
            for k in [
                'CUDA_DEVICE_MAX_CONNECTIONS',
                'NVTE_USE_CUTLASS_GROUPED_GEMM',
                'OMP_NUM_THREADS',
            ]
        },
        'scope': 'single GPU, single GroupedLinear (64 experts); captured rank0/MB0 shapes, generated BF16 inputs/weights; no PP/CP/EP communication or optimizer',
        'timing': 'eager calls, CUDA event intervals include launch gaps; wall includes final synchronize. Per case ABBA, 6 samples/backend/metric, 5 calls/sample. Warmup excluded.',
        'metrics': {
            'dw': 'Both allocate new BF16 dW storage. X/dY and CUDA cumulative offsets already available; no concatenation/padding/copy-back.',
            'module_fb': 'GroupedLinear forward + dgrad + wgrad with autograd. Includes candidate async offsets setup and common grad clearing. No activation function or recompute.',
        },
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for ci, c in enumerate(cases):
        counts = c['counts']
        g = len(counts)
        i = c['I']
        o = c['O']
        m = sum(counts)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=2048,
            num_attention_heads=16,
            num_moe_experts=g,
            moe_grouped_gemm=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            gradient_accumulation_fusion=False,
            use_cpu_initialization=False,
        )
        layer = TEGroupedLinear(
            g,
            i,
            o,
            parallel_mode='column',
            config=config,
            init_method=init_method_normal(0.02),
            bias=False,
            skip_bias_add=False,
            is_expert=True,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        x = torch.randn(m, i, device='cuda', dtype=torch.bfloat16).requires_grad_(True)
        dy = torch.randn(m, o, device='cuda', dtype=torch.bfloat16)
        cpu_counts = torch.tensor(counts, dtype=torch.int64)
        offsets = torch.tensor(counts, device='cuda', dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        xparts = list(x.detach().split(counts))
        yparts = list(dy.split(counts))
        params = list(layer.parameters())

        def module_fb():
            layer.zero_grad(set_to_none=True)
            x.grad = None
            y = layer(x, cpu_counts)
            if isinstance(y, tuple):
                y = y[0]
            y.backward(dy)
            return y

        def te_dw():
            d = torch.empty(g, o, i, device='cuda', dtype=torch.bfloat16)
            general_grouped_gemm(
                xparts,
                yparts,
                list(d.unbind(0)),
                [None] * g,
                torch.bfloat16,
                layout='NT',
                m_splits=counts,
                grad=True,
                use_bias=False,
                accumulate=False,
            )
            return d

        def torch_dw():
            return grouped_mm(dy.t(), x.detach(), offs=offsets)

        layer.config.moe_use_torch_grouped_wgrad = False
        before = stats['calls']
        y0 = module_fb().detach().clone()
        dx0 = x.grad.detach().clone()
        dw0 = torch.stack([w.grad.detach() for w in params])
        assert stats['calls'] == before
        layer.config.moe_use_torch_grouped_wgrad = True
        y1 = module_fb().detach()
        assert stats['calls'] == before + 1
        dw1 = torch.stack([w.grad.detach() for w in params])
        checks = {'forward': error(y1, y0), 'dgrad': error(x.grad, dx0), 'wgrad': error(dw1, dw0)}
        assert all(v['finite'] and v['relative_l2'] < 0.005 for v in checks.values()), checks
        del y0, y1, dx0, dw0, dw1
        result = {'case': c, 'sum_tokens': m, 'correctness': checks, 'timings': {}}
        for metric in ('dw', 'module_fb'):
            values = {b: [] for b in ('te', 'torch')}
            for backend in ('te', 'torch'):
                layer.config.moe_use_torch_grouped_wgrad = backend == 'torch'
                fn = (
                    module_fb if metric == 'module_fb' else (te_dw if backend == 'te' else torch_dw)
                )
                for _ in range(5):
                    fn()
            torch.cuda.synchronize()
            order = (
                ('te', 'torch', 'torch', 'te') if ci % 2 == 0 else ('torch', 'te', 'te', 'torch')
            )
            for block, backend in enumerate(order):
                layer.config.moe_use_torch_grouped_wgrad = backend == 'torch'
                fn = (
                    module_fb if metric == 'module_fb' else (te_dw if backend == 'te' else torch_dw)
                )
                before = stats['calls']
                for sample in range(args.rounds_per_block):
                    torch.cuda.synchronize()
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    begin = time.perf_counter()
                    a.record()
                    for _ in range(args.repeats):
                        fn()
                    b.record()
                    b.synchronize()
                    values[backend].append(
                        {
                            'block': block,
                            'sample': sample,
                            'event_ms': a.elapsed_time(b) / args.repeats,
                            'wall_ms': (time.perf_counter() - begin) * 1000 / args.repeats,
                        }
                    )
                if metric == 'module_fb':
                    expected = args.rounds_per_block * args.repeats if backend == 'torch' else 0
                    assert stats['calls'] - before == expected
            result['timings'][metric] = {
                backend: {
                    'event_ms': statistics.median(t['event_ms'] for t in rr),
                    'wall_ms': statistics.median(t['wall_ms'] for t in rr),
                    'samples': rr,
                }
                for backend, rr in values.items()
            }
        with (out / 'cases.jsonl').open('a') as f:
            f.write(json.dumps(result) + '\n')
        print(
            json.dumps(
                {
                    'case': ci,
                    'seq': c['seq'],
                    'fc': c['fc'],
                    'tokens': m,
                    'dw': {b: result['timings']['dw'][b]['event_ms'] for b in ['te', 'torch']},
                    'module_fb': {
                        b: result['timings']['module_fb'][b]['event_ms'] for b in ['te', 'torch']
                    },
                }
            ),
            flush=True,
        )
        del layer, params, x, dy, xparts, yparts, offsets
        gc.collect()
    (out / 'COMPLETE').write_text('PASS\n')
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
