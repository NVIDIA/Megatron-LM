# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Looped decoder acceptance with rank-distinct data and an unrolled reference.

Run with torchrun --nproc-per-node=1 or 2. NCCL requires distinct GPUs.
Gloo is a diagnostic backend; GPU communication coalescing may need an
external backend adapter. It does not qualify two-GPU NCCL execution. See docs/user-guide/features/looped_decoder.md.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig


def make_model(args, repeats, num_layers=4):
    """Build real dense BF16 decoder layers using the local backend."""
    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=4,
        bf16=True,
        params_dtype=torch.bfloat16,
        persist_layer_norm=False,
        transformer_impl='local',
        hidden_dropout=0.0,
        attention_dropout=0.0,
        decoder_num_repeats=repeats,
        decoder_repeat_start=1,
        decoder_repeat_end=3,
    )
    return (
        GPTModel(
            config,
            get_gpt_layer_local_spec(),
            vocab_size=args.vocab_size,
            max_sequence_length=args.seq_length,
            position_embedding_type='learned_absolute',
        )
        .cuda()
        .bfloat16()
    )


def make_reference(args, model, repeats):
    """Expand distinct physical modules, tying only reference leaf Parameters."""
    order = [0] + [1, 2] * repeats + [3]
    reference = make_model(args, 1, num_layers=len(order))
    for name, child in model.named_children():
        if name != 'decoder':
            reference.get_submodule(name).load_state_dict(child.state_dict())
    reference.decoder.final_layernorm.load_state_dict(model.decoder.final_layernorm.state_dict())
    for logical, physical in enumerate(order):
        layer = reference.decoder.layers[logical]
        layer.load_state_dict(model.decoder.layers[physical].state_dict())
        layer.layer_number = physical + 1
        for name, _ in list(layer.named_parameters()):
            parent, _, leaf = name.rpartition('.')
            param = reference.decoder.layers[order.index(physical)].get_parameter(name)
            layer.get_submodule(parent).register_parameter(leaf, param)
    mapping = {}
    for name, _ in model.named_parameters():
        parts = name.split('.')
        if name.startswith('decoder.layers.'):
            parts[2] = str(order.index(int(parts[2])))
        mapping[name] = '.'.join(parts)
    return reference, mapping


def wrap(model, distributed_optimizer):
    """Use Megatron DDP, FP32 gradient buffers and FP32 Adam master weights."""
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True, use_distributed_optimizer=distributed_optimizer
    )
    ddp = DistributedDataParallel(model.config, ddp_config, model)
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            lr=1e-3,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=distributed_optimizer,
            clip_grad=0.0,
            weight_decay=0.0,
        ),
        [ddp],
    )
    return ddp, optimizer


def run_case(args, repeats, distributed_optimizer):
    """Check outputs, gradients, updates, state size and distributed checkpoint resume."""
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    rank, world = dist.get_rank(), dist.get_world_size()
    model = make_model(args, repeats)
    reference, names = make_reference(args, model, repeats)
    masters = {
        name: torch.nn.Parameter(p.detach().float().clone()) for name, p in model.named_parameters()
    }
    ref_opt = torch.optim.AdamW(masters.values(), lr=1e-3, weight_decay=0.0)
    ddp, optimizer = wrap(model, distributed_optimizer)
    positions = torch.arange(args.seq_length, device='cuda').view(1, -1)
    mask = torch.ones(1, 1, args.seq_length, args.seq_length, device='cuda', dtype=torch.bool).triu(
        1
    )

    def inputs(sample):
        ids = (positions * 7 + sample * 13) % args.vocab_size
        labels = (ids + 1) % args.vocab_size
        return ids, labels

    expected_grads = {name: torch.zeros_like(p) for name, p in masters.items()}
    expected_losses = []
    for sample in range(world):
        reference.zero_grad(set_to_none=True)
        ids, labels = inputs(sample)
        losses = reference(ids, positions, mask, labels=labels)
        expected_losses.append(losses.detach())
        (losses.mean() / world).backward()
        for name, ref_name in names.items():
            expected_grads[name].add_(reference.get_parameter(ref_name).grad.float())
    optimizer.zero_grad()
    ddp.zero_grad_buffer()
    ids, labels = inputs(rank)
    actual = ddp(ids, positions, mask, labels=labels)
    torch.testing.assert_close(actual, expected_losses[rank], rtol=0, atol=0)
    actual.mean().backward()
    ddp.finish_grad_sync()
    base_opt = optimizer.chained_optimizers[0]
    max_grad_error = 0.0
    checked_numel = 0
    for name, param in model.named_parameters():
        if distributed_optimizer:
            if param not in base_opt.model_param_gbuf_map:
                continue
            owned = base_opt._get_model_param_range_map(param)['param']
            start, end = owned.start, owned.end
        else:
            start, end = 0, param.numel()
        actual_grad = param.main_grad.flatten()[start:end]
        expected_grad = expected_grads[name].flatten()[start:end]
        max_grad_error = max(max_grad_error, (actual_grad - expected_grad).abs().max().item())
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        checked_numel += end - start
    assert optimizer.step()[0]
    for name, master in masters.items():
        master.grad = expected_grads[name]
    ref_opt.step()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, masters[name].bfloat16(), rtol=0, atol=0)
    state_numel = sum(
        value.numel()
        for state in base_opt.optimizer.state.values()
        for key, value in state.items()
        if key in ('exp_avg', 'exp_avg_sq')
    )
    state_size = torch.tensor([state_numel], dtype=torch.int64, device='cuda')
    if distributed_optimizer:
        dist.all_reduce(state_size)
    param_numel = sum(p.numel() for p in model.parameters())
    assert state_size.item() == 2 * param_numel
    # Native Adam tracks steps per parameter; TE/Apex FusedAdam tracks them
    # per nonempty parameter group.
    steps = [state['step'] for state in base_opt.optimizer.state.values() if 'step' in state]
    if not steps:
        steps = [group['step'] for group in base_opt.optimizer.param_groups if group['params']]
    assert steps and all(int(step) == 1 for step in steps)

    metadata = dict(distrib_optim_sharding_type='dp_reshardable', checkpoint_version=3.1)
    model_sd = model.sharded_state_dict()
    checkpoint = {
        'model': model_sd,
        'optimizer': optimizer.sharded_state_dict(model_sd, metadata=metadata),
    }
    path = args.output_dir / f'checkpoint-r{repeats}-dist{int(distributed_optimizer)}'
    if rank == 0:
        path.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    dist_checkpointing.save(checkpoint, path)
    bootstrap_path = path.parent / f'{path.name}-optimizer-common.pt'
    if distributed_optimizer:
        if rank == 0:
            torch.save(optimizer.state_dict(), bootstrap_path)
        dist.barrier()
    restored = make_model(args, repeats)
    restored_ddp, restored_opt = wrap(restored, distributed_optimizer)
    # Native distributed Adam needs disk common metadata before tensor templates.
    if distributed_optimizer:
        restored_opt.load_state_dict(torch.load(bootstrap_path, weights_only=False))
    else:
        restored_ddp.zero_grad_buffer()
        assert restored_opt.step()[0]
    for state in restored_opt.optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                if key != 'step':
                    value.fill_(float('nan'))
                elif not distributed_optimizer:
                    value.zero_()
    for group in restored_opt.optimizer.param_groups:
        for param in group['params']:
            param.data.fill_(float('nan'))
        if 'step' in group and not distributed_optimizer:
            group['step'] = 0
    restore_sd = restored.sharded_state_dict()
    template = {
        'model': restore_sd,
        'optimizer': restored_opt.sharded_state_dict(
            restore_sd, is_loading=True, metadata=metadata
        ),
    }
    loaded = dist_checkpointing.load(template, path)
    restored.load_state_dict(loaded['model'])
    restored_opt.load_state_dict(loaded['optimizer'])
    for group, restored_group in zip(
        optimizer.optimizer.param_groups, restored_opt.optimizer.param_groups
    ):
        for param, restored_param in zip(group['params'], restored_group['params']):
            torch.testing.assert_close(param, restored_param, rtol=0, atol=0)
            for key, value in optimizer.optimizer.state[param].items():
                restored_value = restored_opt.optimizer.state[restored_param][key]
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, restored_value, rtol=0, atol=0)
                else:
                    assert value == restored_value
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, restored.get_parameter(name), rtol=0, atol=0)
    for wrapped, opt in ((ddp, optimizer), (restored_ddp, restored_opt)):
        opt.zero_grad()
        wrapped.zero_grad_buffer()
        ids, labels = inputs(rank + world)
        wrapped(ids, positions, mask, labels=labels).mean().backward()
        wrapped.finish_grad_sync()
        assert opt.step()[0]
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, restored.get_parameter(name), rtol=0, atol=0)
    result = dict(
        repeats=repeats,
        distributed_optimizer=distributed_optimizer,
        rank=rank,
        dp_size=world,
        physical_layers=4,
        execution_order=list(model.decoder.layer_execution_order),
        parameters=param_numel,
        optimizer_moment_numel=state_size.item(),
        local_checked_gradient_numel=checked_numel,
        loss=actual.mean().item(),
        output_max_error=0.0,
        gradient_max_error=max_grad_error,
        update_max_error=0.0,
        resumed_next_step_max_error=0.0,
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
    )
    print('LOOPED_RESULT ' + json.dumps(result), flush=True)
    return result


def main():
    """Run the DP1/DP2 matrix in the available GPU environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('nccl', 'gloo'), default='nccl')
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--seq-length', type=int, default=256)
    parser.add_argument('--vocab-size', type=int, default=4096)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if (
        args.backend == 'nccl'
        and int(os.environ.get('LOCAL_WORLD_SIZE', '1')) > torch.cuda.device_count()
    ):
        raise ValueError('NCCL acceptance requires one distinct GPU per local rank')
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    dist.init_process_group(args.backend)
    try:
        if dist.get_world_size() not in (1, 2):
            raise ValueError('Run this acceptance test with DP1 or DP2')
        parallel_state.initialize_model_parallel(1, 1)
        results = [run_case(args, r, opt) for opt in (False, True) for r in (1, 2, 3)]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f'results-rank{dist.get_rank()}.json').write_text(
            json.dumps(dict(backend=args.backend, torch=torch.__version__, cases=results), indent=2)
        )
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
