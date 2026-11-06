# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open

from megatron.lite.model.deepseek_v41.lite.checkpoint import load_hf_weights
from megatron.lite.model.deepseek_v41.lite.export import save_hf_weights, to_release
from megatron.lite.primitive.ckpt.hf_weights import _dequantize_block_scaled_tensor
from megatron.lite.primitive.modules.engram import ShardedFrozenTable
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.runtime.contracts import ParallelConfig


def _bits(tensor):
    return tensor.detach().contiguous().cpu().view(torch.uint8)


def _files(path):
    mapping = json.loads((Path(path) / 'model.safetensors.index.json').read_text())['weight_map']
    return {
        key: safe_open(str(Path(path) / file), framework='pt').get_tensor(key)
        for key, file in mapping.items()
    }


def _put(module, name, shape, dtype=torch.bfloat16, buffer=False):
    parts = name.split('.')
    for part in parts[:-1]:
        if part not in module._modules:
            module.add_module(part, torch.nn.Module())
        module = module._modules[part]
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    value = (torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 0.02).to(dtype)
    if buffer:
        module.register_buffer(parts[-1], value)
    else:
        module.register_parameter(parts[-1], torch.nn.Parameter(value))


def _model(ps):
    model = torch.nn.Module()
    model.local_layer_range = (ps.pp_rank, ps.pp_rank + 1) if ps.pp_size > 1 else (0, 2)
    model.layout = SimpleNamespace(has_embed=ps.pp_is_first, has_head=ps.pp_is_last)
    model.engrams = torch.nn.ModuleDict()
    for layer in range(*model.local_layer_range):
        prefix = f'layers.{layer}.'
        for expert in range(2 // ps.ep_size):
            for fc, shape in [(1, (128, 64)), (2, (64, 64))]:
                name = f'ffn.experts.linear_fc{fc}.weight{expert}'
                _put(model, prefix + name, shape)
                # Local indices must contain distinct global expert values.
                param = dict(model.named_parameters())[prefix + name]
                param.data.add_(ps.ep_rank * 0.01)
        for name in ['attention.linear_q_down_proj.weight', 'ffn.shared_experts.linear_fc2.weight']:
            _put(model, prefix + name, (64, 64))
        _put(model, prefix + 'ffn.router.weight', (2, 64))
        _put(model, prefix + 'ffn.router.expert_bias', (2,), torch.float32, True)
        for name in ['mapping_proj.weight', 'bias', 'alpha_pre', 'alpha_post', 'alpha_res']:
            _put(
                model,
                prefix + 'attn_mixes.' + name,
                (3,) if name == 'bias' else ((24, 64) if name.endswith('weight') else (1,)),
                torch.float32,
            )
        group = ps.dp_cp_group or SimpleNamespace(size=lambda: 1, rank=lambda: 0)
        table = ShardedFrozenTable(5, 32, group, initialize=False)
        start = table.embedding.lookup.boundaries[table.group.rank()]
        table.embedding.weight.copy_(
            torch.arange(start * 32, (start + table.embedding.weight.shape[0]) * 32)
            .reshape(-1, 32)
            .remainder(16)
            .to(torch.float8_e4m3fn)
        )
        table.embedding.scale.view(torch.uint8).fill_(120)
        module = torch.nn.Module()
        module.embed = table
        model.engrams[str(layer)] = module
    if ps.pp_is_first:
        _put(model, 'embed.weight', (64, 64))
    if ps.pp_is_last:
        _put(model, 'head.weight', (64, 64))
        _put(model, 'norm.weight', (64,))
    return model


def _owner_worker(rank, root, ep, pp):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{root}/store', rank=rank, world_size=2)
    ps = init_parallel(ParallelConfig(ep=ep, pp=pp))
    cfg = SimpleNamespace(n_routed_experts=2, vocab_size=64)
    model = _model(ps)
    synced = []
    model.start_param_sync = lambda **kw: synced.append(kw["force_sync"])
    # Deterministic global values, independent of local expert numbering.
    if ep == 2 and rank == 1:
        for n, p in model.named_parameters():
            if '.experts.' in n:
                global_name = n[:-1] + '1'
                seed = int(hashlib.sha256(global_name.encode()).hexdigest()[:8], 16)
                p.data.copy_(
                    (
                        torch.randn(p.shape, generator=torch.Generator().manual_seed(seed)) * 0.02
                    ).bfloat16()
                )
    for target in ('bf16', 'release'):
        reference = _model(ParallelState())
        options = dict(target=target, buffer_max_size_bytes=16384)
        output, refpath, again = (
            f'{root}/{s}-{target}' for s in ('distributed', 'reference', 'again')
        )
        save_hf_weights(reference, refpath, cfg, ParallelState(), **options)
        save_hf_weights(model, output, cfg, ps, **options)
        actual, expected = _files(output), _files(refpath)
        assert actual.keys() == expected.keys()
        assert all(
            a.shape == expected[k].shape and torch.equal(_bits(a), _bits(expected[k]))
            for k, a in actual.items()
        )
        load_hf_weights(model, output, cfg, ps)
        save_hf_weights(model, again, cfg, ps, **options)
        reloaded = _files(again)
        assert all(torch.equal(_bits(v), _bits(reloaded[k])) for k, v in actual.items())
    assert synced == [True] * 4  # direct saves also finish pending gathers
    dist.destroy_process_group()


@pytest.mark.parametrize('ep,pp', [(2, 1), (1, 2)])
def test_owner_export_matches_single_rank_and_is_idempotent(tmp_path, ep, pp):
    mp.spawn(_owner_worker, args=(str(tmp_path), ep, pp), nprocs=2, join=True)


@pytest.mark.parametrize('key', ['layers.0.attn.wq_a.weight', 'layers.0.ffn.experts.0.w1.weight'])
def test_native_encoding_is_idempotent_and_within_rounding_bound(key):
    value = (torch.randn(64, 64, generator=torch.Generator().manual_seed(19)) * 0.02).bfloat16()
    value[0, :2] = torch.tensor([-0.0, 0.0])
    first = to_release(key, value)
    codes, scale = first[0][1], first[1][1]
    decoded = _dequantize_block_scaled_tensor(codes, scale, value.shape).bfloat16()
    again = to_release(key, decoded)
    assert all(torch.equal(_bits(x[1]), _bits(y[1])) for x, y in zip(first, again))
    expanded = (
        scale.float()
        .repeat_interleave(1 if codes.dtype == torch.int8 else 32, 0)
        .repeat_interleave(32, 1)
    )
    # FP4's widest gap is 2*scale; E4M3's relative rounding error is at most 1/16.
    bound = expanded if codes.dtype == torch.int8 else value.float().abs() / 16 + expanded / 512
    assert torch.all((decoded.float() - value.float()).abs() <= bound)


def test_runtime_export_finishes_overlapped_parameter_gather():
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime

    synced = []
    chunk = SimpleNamespace(start_param_sync=lambda **kw: synced.append(kw['force_sync']))

    def export(*args, **kwargs):
        assert synced == [True]
        yield 'head.weight', torch.ones(1)

    protocol = SimpleNamespace(export_hf_weights=export)
    handle = SimpleNamespace(_model=chunk, _parallel_state=None, _extras={'protocol': protocol})
    assert len(list(object.__new__(MegatronLiteRuntime).export_weights(handle))) == 1


def test_row_fragments_are_bounded_and_preserve_order():
    from megatron.lite.primitive.ckpt.row_export import iter_row_shards

    value = torch.arange(160, dtype=torch.uint8).reshape(5, 32)
    parts = [p for _, p in iter_row_shards([("rows", value, 0, 5)], max_bytes=64)]
    assert [p.start for p in parts] == [0, 2, 4] and all(p.tensor.numel() <= 64 for p in parts)
    assert torch.equal(torch.cat([p.tensor for p in parts]), value)
