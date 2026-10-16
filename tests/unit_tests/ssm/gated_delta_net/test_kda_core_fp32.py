# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in FP32 KDA core: independent HF recurrence and real public-module wiring.

Oracle: huggingface/transformers@93c8b7b485963a10800c91f55304db6be211c2bd,
src/transformers/models/glm5_next/modeling_glm5_next.py, SHA256
2092bbb4efa2a8087b74f4a4da37635c503fe1df9ae73f1e6e8342af8b4b8e8b:
l2norm:416-424 and recurrent_kimi_delta_attention:439-478.

The reference below uses only ordinary Torch, never FLA/MCore math. Parity
compares the FP32 core before restoring BF16 and all five input VJPs, with a
BF16-representable cotangent. The separate module cases execute real FLA and
verify BF16 restoration before normal/recomputed output normalization. FP32
storage does not imply IEEE GEMMs; ambient matmul settings are not changed.
These are TP1/CP1 independent-replica tests, not parallel-training qualification
or closure of the composite mHC scalar-gradient failure.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import kda as kda_module
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils

_HEADS = 64
_DIM = 128
_SEED = 20260914


@pytest.fixture(scope='module', autouse=True)
def _local_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


def _config(**overrides):
    values = dict(
        hidden_size=4096,
        num_layers=1,
        num_attention_heads=_HEADS,
        num_query_groups=_HEADS,
        kv_channels=64,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=_DIM,
        linear_value_head_dim=_DIM,
        linear_num_key_heads=_HEADS,
        linear_num_value_heads=_HEADS,
        kda_two_stage_gates=True,
        kda_disable_fp8=True,
        kda_conv_in_fp32=True,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
        normalization='RMSNorm',
        layernorm_epsilon=1e-5,
        layernorm_zero_centered_gamma=False,
        activation_func=F.silu,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        linear_cp_mode='headwise',
        experimental_attention_variant='kda',
        is_hybrid_model=True,
        transformer_impl='transformer_engine',
        deterministic_mode=False,
        gradient_accumulation_fusion=False,
        fp8=None,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def test_core_precision_default_and_explicit_optin():
    assert _config().kda_core_in_fp32 is False
    assert _config(kda_core_in_fp32=True).kda_core_in_fp32 is True


@pytest.mark.parametrize(
    'overrides,message',
    [
        ({'kda_two_stage_gates': False}, 'requires kda_two_stage_gates=True'),
        ({'context_parallel_size': 2}, 'CP=1 only'),
        ({'kda_core_in_fp32': 'true'}, 'must be a bool'),
    ],
)
def test_core_precision_rejects_unsupported_config(overrides, message):
    with pytest.raises(ValueError, match=message):
        _config(**dict({'kda_core_in_fp32': True}, **overrides))


def test_runtime_dynamic_cp_rejected_before_tensor_or_collective_work():
    """A guard-only control: real forward/resolver, no model or substitute kernel."""

    class RuntimeGroup:
        calls = 0

        def size(self):
            self.calls += 1
            return 2

    group = RuntimeGroup()
    control = SimpleNamespace(
        core_in_fp32=True,
        config=SimpleNamespace(context_parallel_size=1, dynamic_context_parallel=True),
        pg_collection=SimpleNamespace(cp=group),
    )
    # No linear_cp_mode, projections or correctly shaped input exist. Accessing
    # any later path would fail rather than satisfy the requested ValueError.
    with pytest.raises(ValueError, match='runtime CP=1 only'):
        kda_module.KimiDeltaAttention.forward(control, torch.empty(0, device='cpu'), None)
    assert group.calls == 1


@pytest.mark.parametrize('use_qk_l2norm', [False, True])
def test_core_preparation_dtype_norm_and_preactivated_gates(use_qk_l2norm):
    q = torch.tensor([[[[0.5, -0.75, 1.25, 0.125]]]], dtype=torch.bfloat16)
    k, v = q.flip(-1), q * 2
    g = torch.tensor([[[[-0.25, -1.0, -2.5, -4.0]]]], dtype=torch.float32)
    beta = torch.tensor([[[0.625]]], dtype=torch.float32)
    actual = kda_module._prepare_fp32_kda_core_inputs(q, k, v, g, beta, use_qk_l2norm=use_qk_l2norm)
    assert all(t.dtype == torch.float32 for t in actual.values())
    for name, operand in (('q', q), ('k', k)):
        expected = operand.float()
        if use_qk_l2norm:
            expected = expected / torch.sqrt((expected * expected).sum(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)
    for name, expected in (('v', v.float()), ('g', g), ('beta', beta)):
        torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)


def _hf_recurrent_core(query, key, value, g, beta):
    """HF source order, FP32 result before its final activation-dtype restoration."""
    query, key, value, g, beta = [x.to(torch.float32) for x in (query, key, value, g, beta)]
    query = query / torch.sqrt((query * query).sum(-1, keepdim=True) + 1e-6)
    key = key / torch.sqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    query = query * (query.shape[-1] ** -0.5)
    batch, sequence, heads, width = key.shape
    state = torch.zeros(
        batch, heads, width, value.shape[-1], device=value.device, dtype=torch.float32
    )
    outputs = []
    for token in range(sequence):
        q_i, k_i, v_i = query[:, token], key[:, token], value[:, token]
        g_i, beta_i = g[:, token][..., None].exp(), beta[:, token][..., None]
        state = state * g_i
        memory = (state * k_i[..., None]).sum(-2)
        delta = (v_i - memory) * beta_i
        state = state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_i.unsqueeze(-1)).sum(-2))
    return torch.stack(outputs, dim=1)


def _similarity(actual, expected, name):
    assert actual.shape == expected.shape, name
    a, b = actual.detach().double().flatten(), expected.detach().double().flatten()
    assert a.numel() and torch.isfinite(a).all() and torch.isfinite(b).all(), name
    assert torch.count_nonzero(a) and torch.count_nonzero(b), name
    cosine = F.cosine_similarity(a[None], b[None]).item()
    similarity = (2 * (a * b).sum() / ((a * a + b * b).sum())).item()
    assert cosine > 0.999, f'{name}: cosine={cosine}'
    assert similarity > 0.999, f'{name}: tensor_similarity={similarity}'
    return dict(name=name, cosine=cosine, tensor_similarity=similarity)


def _observe_actual_chunk_node(output, counts):
    """Observe the real autograd Function and backward, not only a wrapper name."""
    pending, visited, nodes = [output.grad_fn], set(), []
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        assert len(visited) <= 512, 'unexpectedly large core autograd graph'
        cls = getattr(node, '_forward_cls', None)
        if (
            cls is not None
            and cls.__module__ == 'fla.ops.kda.chunk'
            and cls.__name__ == 'ChunkKDAFunction'
        ):
            nodes.append(node)
        pending.extend(child for child, _ in node.next_functions)
    assert len(nodes) == 1, 'expected actual FLA ChunkKDAFunction, not a fallback'

    def backward_observed(*_):
        counts['backward'] += 1
        return None

    counts['forward'] += 1
    nodes[0].register_hook(backward_observed)


_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
_requires_fla = pytest.mark.skipif(not kda_module.HAVE_FLA_KDA, reason='requires real FLA KDA')


@pytest.mark.internal
@_requires_cuda
@_requires_fla
@pytest.mark.parametrize('sequence', [17, 65])
def test_fp32_core_matches_independent_hf_output_and_five_vjps(sequence):
    """Production heads/dimensions, including a cross-chunk incomplete tail."""
    torch.manual_seed(_SEED)
    torch.cuda.manual_seed(_SEED)
    shape = (1, sequence, _HEADS, _DIM)
    q, k, v = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    g = -5 * torch.sigmoid(torch.randn(shape, device='cuda', dtype=torch.float32))
    beta = torch.randn(shape[:-1], device='cuda', dtype=torch.bfloat16).sigmoid().float()
    actual_inputs = [t.detach().clone().requires_grad_() for t in (q, k, v, g, beta)]
    native_inputs = [t.detach().clone().requires_grad_() for t in (q, k, v, g, beta)]
    prepared = kda_module._prepare_fp32_kda_core_inputs(*actual_inputs, use_qk_l2norm=True)
    actual, _ = kda_module.chunk_kda(
        **prepared,
        scale=_DIM**-0.5,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        safe_gate=True,
        lower_bound=-5.0,
        state_v_first=True,
        cu_seqlens=None,
        cp_context=None,
    )
    assert actual.dtype == torch.float32
    backend = dict(forward=0, backward=0)
    _observe_actual_chunk_node(actual, backend)
    expected = _hf_recurrent_core(*native_inputs)
    comparisons = [_similarity(actual, expected, 'output')]
    gradient = torch.randn(shape, device='cuda', dtype=torch.bfloat16).float()
    actual.backward(gradient)
    expected.backward(gradient)
    for name, left, right in zip(('q', 'k', 'v', 'g', 'beta'), actual_inputs, native_inputs):
        assert left.grad is not None and right.grad is not None, name
        comparisons.append(_similarity(left.grad, right.grad, name + ' VJP'))
    assert backend == dict(forward=1, backward=1)
    print(
        'KDA_CORE_FP32_PARITY '
        + json.dumps(
            dict(
                sequence=sequence,
                heads=_HEADS,
                head_dim=_DIM,
                seed=_SEED,
                comparisons=comparisons,
                actual_backend='fla.ops.kda.chunk.ChunkKDAFunction',
                actual_calls=backend,
                fp32_storage_does_not_imply_ieee=True,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            ),
            sort_keys=True,
        )
    )


@pytest.mark.internal
@_requires_cuda
@_requires_fla
@pytest.mark.parametrize('enabled,recompute_norm', [(False, False), (True, False), (True, True)])
def test_public_kda_core_precision_and_norm_checkpoint_contract(
    enabled, recompute_norm, monkeypatch
):
    Utils.initialize_distributed()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    try:
        model_parallel_cuda_manual_seed(_SEED)
        torch.manual_seed(_SEED)
        torch.cuda.manual_seed(_SEED)
        config = _config(
            kda_core_in_fp32=enabled,
            recompute_granularity='selective' if recompute_norm else None,
            recompute_modules=['gdn_norm_out'] if recompute_norm else [],
        )
        spec = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention
        real = build_module(
            spec,
            config=config,
            layer_number=1,
            conv_init=0.02,
            pg_collection=ProcessGroupCollection(
                tp=parallel_state.get_tensor_model_parallel_group(),
                cp=parallel_state.get_context_parallel_group(),
            ),
        ).cuda()
        real.train()
        fp32_parameters = {'A_log', 'dt_bias', 'conv1d.weight'}
        observed_fp32 = set()
        for name, parameter in real.named_parameters():
            if name in fp32_parameters:
                assert parameter.dtype == torch.float32 and parameter.keep_in_fp32 is True, name
                observed_fp32.add(name)
            else:
                assert parameter.dtype == torch.bfloat16, name
        assert observed_fp32 == fp32_parameters
        assert real.core_in_fp32 is enabled
        assert real.recompute_norm_out is recompute_norm
        records, backend = dict(norm_calls=0), dict(forward=0, backward=0)
        original_pre = real.pre_gated_delta_rule
        original_core = real.gated_delta_rule
        original_norm = real._gated_norm_and_layout_restore

        def observe_pre(*args, **kwargs):
            values = original_pre(*args, **kwargs)
            records['pre'] = tuple(
                t.detach() for t in (values[0], values[1], values[2], values[5], values[4])
            )
            return values

        def observe_core(**kwargs):
            records['core_input_dtypes'] = {
                name: str(kwargs[name].dtype) for name in ('q', 'k', 'v', 'g', 'beta')
            }
            records['kernel_l2norm'] = kwargs['use_qk_l2norm_in_kernel']
            assert all(kwargs[name].dtype == torch.float32 for name in ('g', 'beta'))
            assert all(
                kwargs[name].dtype == (torch.float32 if enabled else torch.bfloat16)
                for name in ('q', 'k', 'v')
            )
            assert kwargs['use_qk_l2norm_in_kernel'] is (not enabled)
            assert kwargs['use_gate_in_kernel'] is False
            assert kwargs['state_v_first'] is True
            if enabled:
                assert kwargs['scale'] == _DIM**-0.5
                assert kwargs['use_beta_sigmoid_in_kernel'] is False
                pre_q, pre_k, pre_v, pre_g, pre_beta = records['pre']
                for name, value in (('q', pre_q), ('k', pre_k)):
                    value = value.float()
                    expected = value / torch.sqrt((value * value).sum(-1, keepdim=True) + 1e-6)
                    torch.testing.assert_close(kwargs[name], expected, rtol=0, atol=0)
                for name, value in (('v', pre_v), ('g', pre_g), ('beta', pre_beta)):
                    torch.testing.assert_close(kwargs[name], value.float(), rtol=0, atol=0)
            else:
                assert 'scale' not in kwargs and 'use_beta_sigmoid_in_kernel' not in kwargs
                for name, value in zip(('q', 'k', 'v', 'g', 'beta'), records['pre']):
                    assert kwargs[name].data_ptr() == value.data_ptr()
            out = original_core(**kwargs)
            assert out[0].dtype == (torch.float32 if enabled else torch.bfloat16)
            records['raw_core'] = out[0].detach()
            _observe_actual_chunk_node(out[0], backend)
            return out

        def observe_norm(core, *args, **kwargs):
            records['norm_calls'] += 1
            records['norm_input_dtype'] = str(core.dtype)
            assert core.dtype == records['pre'][0].dtype == torch.bfloat16
            torch.testing.assert_close(core, records['raw_core'].to(torch.bfloat16), rtol=0, atol=0)
            return original_norm(core, *args, **kwargs)

        monkeypatch.setattr(real, 'pre_gated_delta_rule', observe_pre)
        monkeypatch.setattr(real, 'gated_delta_rule', observe_core)
        monkeypatch.setattr(real, '_gated_norm_and_layout_restore', observe_norm)
        hidden = torch.randn(17, 1, 4096, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        output, bias = real(hidden, attention_mask=None)
        assert bias is None and output.dtype == hidden.dtype
        assert torch.isfinite(output).all()
        output.backward(torch.randn_like(output))
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
        for name, parameter in real.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert backend == dict(forward=1, backward=1)
        assert records['norm_calls'] == (2 if recompute_norm else 1)
        print(
            'KDA_CORE_FP32_MODULE '
            + json.dumps(
                dict(
                    kda_core_in_fp32=config.kda_core_in_fp32,
                    recompute_norm=recompute_norm,
                    norm_calls=records['norm_calls'],
                    actual_backend='fla.ops.kda.chunk.ChunkKDAFunction',
                    actual_calls=backend,
                    core_output_dtype=str(records['raw_core'].dtype),
                    norm_input_dtype=records['norm_input_dtype'],
                    core_input_dtypes=records['core_input_dtypes'],
                    use_qk_l2norm_in_kernel=records['kernel_l2norm'],
                    seed=_SEED,
                ),
                sort_keys=True,
            )
        )
    finally:
        Utils.destroy_model_parallel()
