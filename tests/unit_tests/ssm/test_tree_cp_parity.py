# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compare tree execution with independent causal paths, including CP backward.

Run on at least two GPUs through torch.distributed.run. The reference recomputes
each root-to-segment path, but supervises only that segment, so every physical
node contributes to the objective exactly once. Tolerances are fixed before the
GPU gate: BF16 relative L2 error <= 2.5%, plus an elementwise absolute guard.
"""

from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist
from flash_attn import flash_attn_func

from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import TreePackedSeqParams, TreeQueryRun
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from tests.unit_tests.test_utilities import Utils

# An internal branch, its descendant, and a sibling crossing a CP zigzag boundary.
_LENGTHS = (48, 32, 16, 32)
_STARTS = (0, 48, 80, 96)
_PARENTS = (-1, 0, 1, 0)
_DEPTHS = (0, 48, 80, 48)
_PATHS = (tuple(range(48)), tuple(range(80)), tuple(range(96)), (*range(48), *range(96, 128)))


def _cp_indices(rank: int, size: int) -> torch.Tensor:
    indices = torch.arange(128, device='cuda')
    if size == 1:
        return indices
    chunks = indices.chunk(2 * size)
    return torch.cat((chunks[rank], chunks[2 * size - rank - 1]))


def _metadata(rank: int, size: int) -> TreePackedSeqParams:
    indices = _cp_indices(rank, size)
    runs = []
    positions = torch.empty(128, device='cuda', dtype=torch.long)
    for segment, (start, length, depth) in enumerate(zip(_STARTS, _LENGTHS, _DEPTHS)):
        positions[start : start + length] = torch.arange(depth, depth + length, device='cuda')
        local = ((indices >= start) & (indices < start + length)).nonzero().flatten()
        if local.numel() == 0:
            continue
        global_indices = indices[local]
        splits = (global_indices[1:] != global_indices[:-1] + 1).nonzero().flatten()
        boundaries = [0, *(splits + 1).tolist(), local.numel()]
        for left, right in zip(boundaries, boundaries[1:]):
            runs.append(
                TreeQueryRun(
                    segment,
                    int(global_indices[left]),
                    int(global_indices[right - 1]) + 1,
                    local[left:right],
                )
            )
    cu = torch.tensor([0, 128], device='cuda', dtype=torch.int32)
    return TreePackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=128,
        max_seqlen_kv=128,
        tree_segment_starts=_STARTS,
        tree_segment_lengths=_LENGTHS,
        tree_segment_parents=_PARENTS,
        tree_segment_depths=_DEPTHS,
        tree_local_position_ids=positions[indices],
        tree_cp_gather_inverse=torch.argsort(
            torch.cat([_cp_indices(r, size) for r in range(size)])
        ),
        tree_query_runs=tuple(runs),
        tree_cp_local_token_count=indices.numel(),
    )


def _assert_parity(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all(), name
    assert torch.isfinite(expected).all(), name
    error = (actual - expected).norm()
    scale = expected.norm().clamp_min(1e-8)
    print(
        f'rank={dist.get_rank()} {name}: relative_l2={float(error / scale):.6g} '
        f'max_abs={float((actual - expected).abs().max()):.6g} '
        f'reference_norm={float(scale):.6g}',
        flush=True,
    )
    assert error <= 0.025 * scale + 1e-6, f'{name}: relative L2 error {error / scale}'
    torch.testing.assert_close(
        actual, expected, rtol=0.05, atol=0.005, msg=lambda message: f'{name}: {message}'
    )


@pytest.fixture(params=[1, 2], ids=['cp1', 'cp2'])
def groups(request) -> Iterator[ProcessGroupCollection]:
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, context_parallel_size=request.param
    )
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    yield ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
    Utils.destroy_model_parallel()


def _config(cp_size: int) -> TransformerConfig:
    return TransformerConfig(
        hidden_size=256,
        num_layers=1,
        num_attention_heads=4,
        context_parallel_size=cp_size,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        mamba_state_dim=16,
        mamba_head_dim=64,
        mamba_num_groups=8,
        use_mamba_mem_eff_path=True,
    )


def test_tree_attention_forward_backward(groups: ProcessGroupCollection) -> None:
    rank, size = groups.cp.rank(), groups.cp.size()
    model = (
        SelfAttention(
            _config(size),
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            pg_collection=groups,
        )
        .cuda()
        .train()
    )
    torch.manual_seed(37)
    values = [
        torch.randn(128, heads, 64, device='cuda', dtype=torch.bfloat16) * 0.2
        for heads in (4, 2, 2)
    ]
    weights = torch.randn_like(values[0]) * 0.1
    indices = _cp_indices(rank, size)
    actual_inputs = [value[indices].detach().requires_grad_() for value in values]
    reference_inputs = [value.detach().requires_grad_() for value in values]
    actual = model._tree_attention_forward(*actual_inputs, _metadata(rank, size))
    reference_parts = []
    for path, length in zip(_PATHS, _LENGTHS):
        path_indices = torch.tensor(path, device='cuda')
        q, k, v = [value[path_indices].unsqueeze(0) for value in reference_inputs]
        reference_parts.append(flash_attn_func(q, k, v, causal=True)[0, -length:])
    reference = torch.cat(reference_parts)
    _assert_parity(actual, reference[indices], 'attention output')
    (actual.float() * weights[indices].float()).sum().backward()
    (reference.float() * weights.float()).sum().backward()
    for name, local, full in zip(('q', 'k', 'v'), actual_inputs, reference_inputs):
        _assert_parity(local.grad, full.grad[indices], f'attention {name} gradient')


@pytest.mark.parametrize('fused_reference', [True, False], ids=['fused', 'unfused'])
def test_tree_mamba_forward_backward(groups: ProcessGroupCollection, fused_reference: bool) -> None:
    rank, size = groups.cp.rank(), groups.cp.size()
    spec = hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules
    actual_model = (
        MambaMixer(_config(size), spec, 256, layer_number=1, pg_collection=groups).cuda().train()
    )
    # TP=1 gives each rank its own reference CP group without global-state changes.
    reference_groups = ProcessGroupCollection(tp=groups.tp, cp=groups.tp)
    reference_config = _config(1)
    reference_config.use_mamba_mem_eff_path = fused_reference
    reference_model = (
        MambaMixer(reference_config, spec, 256, layer_number=1, pg_collection=reference_groups)
        .cuda()
        .train()
    )
    reference_model.load_state_dict(actual_model.state_dict())
    torch.manual_seed(71)
    inputs = torch.randn(128, 1, 256, device='cuda', dtype=torch.bfloat16) * 0.2
    weights = torch.randn_like(inputs) * 0.1
    indices = _cp_indices(rank, size)
    actual_input = inputs[indices].detach().requires_grad_()
    reference_input = inputs.detach().requires_grad_()
    actual, bias = actual_model(actual_input, packed_seq_params=_metadata(rank, size))
    if bias is not None:
        actual = actual + bias
    parts = []
    for path, length in zip(_PATHS, _LENGTHS):
        output, bias = reference_model(reference_input[torch.tensor(path, device='cuda')])
        if bias is not None:
            output = output + bias
        parts.append(output[-length:])
    reference = torch.cat(parts)
    _assert_parity(actual, reference[indices], 'Mamba output')
    (actual.float() * weights[indices].float()).sum().backward()
    (reference.float() * weights.float()).sum().backward()
    _assert_parity(actual_input.grad, reference_input.grad[indices], 'Mamba input gradient')
    reference_params = dict(reference_model.named_parameters())
    failures = []
    for name, parameter in actual_model.named_parameters():
        expected = reference_params[name].grad
        assert parameter.grad is not None and expected is not None, name
        gradient = parameter.grad.float()
        dist.all_reduce(gradient, group=groups.cp)
        try:
            _assert_parity(gradient, expected, f'Mamba {name} gradient')
        except AssertionError as error:
            failures.append(str(error))
    assert not failures, '\n'.join(failures)


def test_tree_attention_rejects_softcapping(groups: ProcessGroupCollection) -> None:
    model = SelfAttention(
        _config(groups.cp.size()),
        get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
        layer_number=1,
        pg_collection=groups,
    ).cuda()
    model.config.attn_logit_softcapping = 30.0
    qkv = torch.zeros(128 // groups.cp.size(), 4, 64, device='cuda', dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match='softcapping'):
        model._tree_attention_forward(qkv, qkv, qkv, _metadata(groups.cp.rank(), groups.cp.size()))


def test_tree_edge_projection_sequence_parallel_backward() -> None:
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=1)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        config = _config(1)
        config.tensor_model_parallel_size = 2
        config.sequence_parallel = True
        model = LanguageModule(config, pg_collection=groups)
        output_layer = ColumnParallelLinear(
            256, 16, config=config, init_method=config.init_method, bias=False, tp_group=groups.tp
        ).cuda()
        params = _metadata(0, 1)
        sources = torch.tensor([5, 63, 64, 101], device='cuda')
        params.tree_edge_local_indices = sources
        params.tree_edge_output_indices = torch.arange(4, device='cuda')
        torch.manual_seed(19)
        full = torch.randn(128, 1, 256, device='cuda', dtype=torch.bfloat16)
        local = full.chunk(2)[groups.tp.rank()].detach().requires_grad_()
        selected, restore = model._select_tree_edge_hidden_states(local, params, output_layer)
        assert restore and not output_layer.sequence_parallel
        torch.testing.assert_close(selected, full[sources], rtol=0, atol=0)
        selected.float().sum().backward()
        expected = torch.zeros_like(full)
        expected[sources] = 1
        torch.testing.assert_close(local.grad, expected.chunk(2)[groups.tp.rank()], rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
