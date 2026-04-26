"""Standalone tests for Muon CPU offloading in LayerWiseDistributedOptimizer.

Run with:
    torchrun --nproc-per-node=4 tests/test_muon_cpu_offload.py

Avoids the pytest conftest circular-import issue by running as a plain script.
"""

import os
import sys
import traceback
from datetime import timedelta

import torch
import torch.distributed

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig


def init_distributed():
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='nccl', world_size=world_size, rank=rank,
            timeout=timedelta(minutes=2),
        )
    return rank, world_size


def create_model(seed, tp, pp):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config = TransformerConfig(
        num_layers=6,
        hidden_size=16,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec()
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=128,
        max_sequence_length=4,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    )
    model.cuda(torch.cuda.current_device())
    return model


def create_optimizer_with_cpu_offload(model, cpu_offload=True):
    config = OptimizerConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=False,
        use_layer_wise_distributed_optimizer=True,
        optimizer='muon',
        lr=0.0,
        optimizer_cpu_offload=cpu_offload,
    )
    optimizer = get_megatron_optimizer(config, [model])

    if isinstance(optimizer, LayerWiseDistributedOptimizer):
        for opt in optimizer.chained_optimizers:
            if getattr(opt, 'init_state_fn', None) is None:
                continue
            if not hasattr(opt, 'optimizer'):
                opt.init_state_fn(opt)
            else:
                opt.init_state_fn(opt.optimizer)
        if cpu_offload:
            optimizer.offload_optimizer_states()
    return optimizer


def _iter_fp16_opts(optimizer):
    for opt in optimizer.chained_optimizers:
        if getattr(opt, 'is_stub_optimizer', False):
            continue
        if isinstance(opt, Float16OptimizerWithFloat16Params):
            yield opt


def test_states_on_cpu(rank):
    """After init with cpu_offload=True, fp32 master weights and state are on CPU."""
    parallel_state.initialize_model_parallel(2, 2)
    try:
        model = create_model(seed=2, tp=2, pp=2)
        optimizer = create_optimizer_with_cpu_offload(model, cpu_offload=True)

        assert isinstance(optimizer, LayerWiseDistributedOptimizer)
        assert optimizer._cpu_offload

        for opt in _iter_fp16_opts(optimizer):
            for group in opt.fp32_from_float16_groups:
                for param in group:
                    assert not param.data.is_cuda, (
                        f"[rank {rank}] fp32 master weight should be on CPU, "
                        f"got {param.data.device}"
                    )
            for state_vals in opt.optimizer.state.values():
                for key, val in state_vals.items():
                    if isinstance(val, torch.Tensor):
                        assert not val.is_cuda, (
                            f"[rank {rank}] optimizer state '{key}' should be on CPU, "
                            f"got {val.device}"
                        )

        print(f"  [rank {rank}] PASSED: test_states_on_cpu")
    finally:
        parallel_state.destroy_model_parallel()


def test_roundtrip_correctness(rank):
    """Offload -> reload preserves fp32 master weight values exactly."""
    parallel_state.initialize_model_parallel(2, 2)
    try:
        model = create_model(seed=2, tp=2, pp=2)
        optimizer = create_optimizer_with_cpu_offload(model, cpu_offload=True)

        assert isinstance(optimizer, LayerWiseDistributedOptimizer)

        snapshots = {}
        for opt in _iter_fp16_opts(optimizer):
            for gidx, group in enumerate(opt.fp32_from_float16_groups):
                for pidx, param in enumerate(group):
                    snapshots[(id(opt), gidx, pidx)] = param.data.clone()

        optimizer.reload_optimizer_states()

        for opt in _iter_fp16_opts(optimizer):
            for gidx, group in enumerate(opt.fp32_from_float16_groups):
                for pidx, param in enumerate(group):
                    assert param.data.is_cuda, (
                        f"[rank {rank}] After reload, param should be on GPU"
                    )
                    expected = snapshots[(id(opt), gidx, pidx)].to(param.data.device)
                    assert torch.equal(param.data, expected), (
                        f"[rank {rank}] Master weight mismatch after roundtrip"
                    )

        optimizer.offload_optimizer_states()

        for opt in _iter_fp16_opts(optimizer):
            for gidx, group in enumerate(opt.fp32_from_float16_groups):
                for pidx, param in enumerate(group):
                    assert not param.data.is_cuda, (
                        f"[rank {rank}] After offload, param should be on CPU"
                    )

        print(f"  [rank {rank}] PASSED: test_roundtrip_correctness")
    finally:
        parallel_state.destroy_model_parallel()


def test_step_runs(rank):
    """A full optimizer.step() succeeds with CPU offloading."""
    parallel_state.initialize_model_parallel(2, 2)
    try:
        model = create_model(seed=2, tp=2, pp=2)
        optimizer = create_optimizer_with_cpu_offload(model, cpu_offload=True)

        assert isinstance(optimizer, LayerWiseDistributedOptimizer)

        for param in model.parameters():
            if param.requires_grad:
                g = torch.randn_like(param.data)
                param.grad = g
                param.main_grad = g

        update_successful, grad_norm, num_zeros = optimizer.step()
        assert isinstance(update_successful, bool), (
            f"[rank {rank}] update_successful should be bool, got {type(update_successful)}"
        )

        for opt in _iter_fp16_opts(optimizer):
            for group in opt.fp32_from_float16_groups:
                for param in group:
                    assert not param.data.is_cuda, (
                        f"[rank {rank}] After step, fp32 master weights should be on CPU"
                    )

        print(f"  [rank {rank}] PASSED: test_step_runs")
    finally:
        parallel_state.destroy_model_parallel()


def test_numerical_equivalence(rank, n_steps=5):
    """Offloaded and non-offloaded optimizers produce identical fp32 master weights."""
    parallel_state.initialize_model_parallel(2, 2)
    try:
        model_off = create_model(seed=42, tp=2, pp=2)
        model_ref = create_model(seed=42, tp=2, pp=2)

        opt_off = create_optimizer_with_cpu_offload(model_off, cpu_offload=True)
        opt_ref = create_optimizer_with_cpu_offload(model_ref, cpu_offload=False)

        assert isinstance(opt_off, LayerWiseDistributedOptimizer)
        assert isinstance(opt_ref, LayerWiseDistributedOptimizer)

        for step_i in range(n_steps):
            torch.manual_seed(1000 + step_i + rank)

            for p_off, p_ref in zip(model_off.parameters(), model_ref.parameters()):
                if not p_off.requires_grad:
                    continue
                g = torch.randn_like(p_off.data)
                p_off.grad = g.clone()
                p_off.main_grad = p_off.grad
                p_ref.grad = g.clone()
                p_ref.main_grad = p_ref.grad

            opt_off.step()
            opt_ref.step()

        opt_off.reload_optimizer_states()

        for opt_o, opt_r in zip(
            _iter_fp16_opts(opt_off), _iter_fp16_opts(opt_ref)
        ):
            for grp_o, grp_r in zip(
                opt_o.fp32_from_float16_groups, opt_r.fp32_from_float16_groups
            ):
                for pidx, (p_o, p_r) in enumerate(zip(grp_o, grp_r)):
                    p_o_gpu = p_o.data.to('cuda') if not p_o.data.is_cuda else p_o.data
                    assert torch.equal(p_o_gpu, p_r.data), (
                        f"[rank {rank}] fp32 master weight mismatch at param {pidx} "
                        f"after {n_steps} steps, "
                        f"max diff = {(p_o_gpu - p_r.data).abs().max().item()}"
                    )

            for (key_o, state_o), (key_r, state_r) in zip(
                opt_o.optimizer.state.items(), opt_r.optimizer.state.items()
            ):
                common_keys = set(state_o.keys()) & set(state_r.keys())
                for skey in common_keys:
                    v_o, v_r = state_o[skey], state_r[skey]
                    if not isinstance(v_o, torch.Tensor):
                        continue
                    v_o_gpu = v_o.to('cuda') if not v_o.is_cuda else v_o
                    assert torch.equal(v_o_gpu, v_r), (
                        f"[rank {rank}] optimizer state '{skey}' mismatch "
                        f"after {n_steps} steps, "
                        f"max diff = {(v_o_gpu - v_r).abs().max().item()}"
                    )

        opt_off.offload_optimizer_states()

        print(f"  [rank {rank}] PASSED: test_numerical_equivalence ({n_steps} steps)")
    finally:
        parallel_state.destroy_model_parallel()


def main():
    rank, world_size = init_distributed()

    tests = [
        ("test_states_on_cpu", test_states_on_cpu),
        ("test_roundtrip_correctness", test_roundtrip_correctness),
        ("test_step_runs", test_step_runs),
        ("test_numerical_equivalence", test_numerical_equivalence),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        torch.distributed.barrier()
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print(f"{'='*60}")
        try:
            fn(rank)
            passed += 1
        except Exception:
            failed += 1
            if rank == 0:
                traceback.print_exc()
            print(f"  [rank {rank}] FAILED: {name}")

    torch.distributed.barrier()
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
        print(f"{'='*60}")

    torch.distributed.destroy_process_group()
    sys.exit(1 if failed > 0 else 0)


if __name__ == '__main__':
    main()
