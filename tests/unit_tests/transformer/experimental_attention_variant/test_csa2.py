# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 ownership, loss-gradient and packed-layout contracts on small reference inputs."""

import copy
from functools import partial

import pytest
import torch
from torch import nn

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.csa import (
    CompressedSparseAttentionSubmodules,
    CompressorSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa2 import (
    CompressedSparseAttention2,
    CSA2Compressor,
    CSA2Indexer,
    CSA2IndexerSubmodules,
    CSA2State,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig


@pytest.fixture(autouse=True)
def loss_scale(monkeypatch):
    """Keep the auxiliary-loss scale local to each reference test."""
    monkeypatch.setattr(DSAIndexerLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))


class Group:
    def size(self):
        return 1

    def rank(self):
        return 0


class Linear(nn.Module):
    def __init__(self, input_size, output_size, *, config, init_method, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size, dtype=config.params_dtype))
        init_method(self.weight)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None

    def backward_dw(self):
        pass


class Norm(nn.Module):
    def __init__(self, *, config, hidden_size, eps, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class CpuRoPE(nn.Module):
    def forward(self, n, packed_seq=False):
        frequencies = torch.outer(
            torch.arange(n).float(), 1 / (10000 ** (torch.arange(0, 4, 2).float() / 4))
        )
        return torch.cat((frequencies, frequencies), -1)[:, None, None, :]


@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("objective", ["attention", "indexer", "joint"])
def test_joint_output_rope_preserves_storage_padding_and_gradients(monkeypatch, rotate, objective):
    """Exercise the real joint autograd boundary with an analytic backend oracle."""
    from types import SimpleNamespace

    from megatron.core.transformer.experimental_attention_variant.csa_utils import csa2_indexer

    torch.manual_seed(610)
    padding = torch.tensor([False, False, False, False, True])
    signs = torch.tensor([1.0, 1.0, -1.0, -1.0])
    rope = (
        SimpleNamespace(apply_=lambda x: x.mul_(signs), undo=lambda x, out: out.copy_(x * signs))
        if rotate
        else None
    )
    seen = {}

    def attention(q, kv, _indices, _scale, *, attn_sink, **_kwargs):
        out = (q + kv.mean(0) + attn_sink[None, :, None]).tanh()
        seen['raw'] = out.clone().masked_fill(padding[:, None, None], 0)
        # Real sparse kernels may leave fully padded rows undefined.
        out[padding] = torch.nan
        seen['owned'] = out
        return out, torch.zeros_like(q[..., 0]), None

    def backward(q, kv, out, grad, _lse, sink, *_args, **_kwargs):
        torch.testing.assert_close(out, seen['raw'])
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad[padding]) == 0
        dpre = grad * (1 - out.square())
        return dpre, dpre.sum((0, 1)).expand_as(kv) / kv.shape[0], dpre.sum((0, 2))

    def indexer_loss(iq, ik, weights, *_args, **_kwargs):
        loss = (iq.sum() + 2 * ik.sum() + 3 * weights.sum()) / 4
        return loss, (
            torch.ones_like(iq) / 4,
            torch.ones_like(ik) / 2,
            torch.ones_like(weights) * 0.75,
        )

    monkeypatch.setattr(csa2_indexer.fused_sparse_attention, '_csa_fwd_flash_mla', attention)
    monkeypatch.setattr(
        csa2_indexer.fused_sparse_attention, '_csa_sparse_attention_backward', backward
    )
    monkeypatch.setattr(csa2_indexer, '_precompute_indexer_loss', indexer_loss)
    shapes = ((5, 2, 4), (3, 4), (2, 1, 4), (2,), (5, 1, 2), (2, 2), (5, 1))
    leaves = [torch.randn(shape, requires_grad=True) for shape in shapes]
    q, local, global_kv, sink, iq, ik, weights = leaves
    prepared = csa2_indexer.CSA2IndexerInputs(
        iq,
        ik,
        weights,
        1,
        torch.zeros(5, dtype=torch.int32),
        torch.full((5,), 2),
        ~padding,
        2,
        2,
        (5,),
        thd_layout=SimpleNamespace(cp_size=1),
    )
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        output, loss = csa2_indexer.fused_csa2_indexer_sparse_attn(
            prepared,
            q,
            local,
            global_kv,
            sink,
            torch.zeros(5, 1, dtype=torch.int32),
            torch.zeros(5, 1, dtype=torch.int32),
            1.0,
            0.25,
            True,
            attention_indices=torch.zeros(5, 2, dtype=torch.int32),
            attention_topk_length=torch.tensor([2, 2, 2, 2, 0]),
            q_padding_mask=padding,
            out_rope=rope,
        )
    assert output.data_ptr() == seen['owned'].data_ptr()
    assert any(t.data_ptr() == output.data_ptr() for t in saved)
    assert torch.count_nonzero(output[padding]) == 0
    snapshot = output.detach().clone()
    upstream = torch.randn_like(output).masked_fill(padding[:, None], torch.nan)
    upstream_snapshot = upstream.clone()

    reference = [t.detach().clone().requires_grad_() for t in leaves]
    rq, rl, rg, rs, riq, rik, rw = reference
    expected = (rq + torch.cat((rl, rg.squeeze(1))).mean(0) + rs[None, :, None]).tanh()
    expected = expected.masked_fill(padding[:, None, None], 0)
    if rotate:
        expected = expected * signs
    expected_loss = (riq.sum() + 2 * rik.sum() + 3 * rw.sum()) / 4
    torch.testing.assert_close(output, expected.flatten(1))
    actual_targets, expected_targets, probes = [], [], []
    if objective != 'indexer':
        actual_targets.append(output)
        expected_targets.append(expected.flatten(1))
        probes.append(upstream)
    if objective != 'attention':
        actual_targets.append(loss)
        expected_targets.append(expected_loss)
        probes.append(torch.tensor(2.0))
    actual = torch.autograd.grad(actual_targets, leaves, probes, allow_unused=True)
    expected = torch.autograd.grad(expected_targets, reference, probes, allow_unused=True)
    for observed, wanted in zip(actual, expected):
        if wanted is None:
            assert observed is None
        else:
            torch.testing.assert_close(observed, wanted)
    torch.testing.assert_close(output, snapshot)
    torch.testing.assert_close(upstream, upstream_snapshot, equal_nan=True)


@pytest.mark.parametrize('cp_size', [1, 2])
def test_packed_output_rope_uses_query_positions_without_post_attention_copy(monkeypatch, cp_size):
    """The optimized CSA2 return aliases its sanitized, rotated attention output."""
    from megatron.core.ops.attention.csa.output_rope import OutputRopeParams
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.experimental_attention_variant import csa2

    class CPGroup(Group):
        def size(self):
            return cp_size

        def rank(self):
            return cp_size - 1

    _, layers = _setup()
    layer = layers[0]
    layer.use_fused_kernels = True
    layer.eval()
    group = CPGroup()
    monkeypatch.setattr(csa2, 'get_pg_size', lambda group: group.size())
    layer.pg_collection.cp = group
    rows = 16 // cp_size
    logical = torch.tensor([0, 5, 12], dtype=torch.int32)
    physical = torch.tensor([0, 8, 16], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=8,
        max_seqlen_kv=8,
        cp_group=group,
    )
    params = OutputRopeParams(torch.ones(8, 4), torch.zeros(8, 4), 4, 4, physical)
    seen = {}

    def attention(query, *args, **kwargs):
        seen['rope'] = kwargs.get('out_rope', args[-1])
        seen['output'] = query.flatten(1)
        return seen['output'], None

    monkeypatch.setattr(
        layer, '_forward_thd_cp' if cp_size > 1 else '_forward_attention', attention
    )
    q = torch.randn(rows, 2, 8, requires_grad=True)
    kv = torch.randn(rows, 1, 8, requires_grad=True)
    boundary = (
        dict(boundary_hidden=torch.empty(3, 1, 16), boundary_kv=torch.empty(3, 1, 8))
        if cp_size > 1
        else {}
    )
    output = layer(
        q,
        kv,
        kv,
        None,
        x=torch.randn(rows, 1, 16),
        qr=torch.randn(rows, 1, 8),
        packed_seq_params=packed,
        csa2_state=CSA2State(),
        out_rope=params,
        **boundary,
    )
    assert output.data_ptr() == seen['output'].data_ptr()
    expected_positions = ([0, 1, 2, 3, 4, 0, 0, 0] if cp_size == 1 else []) + [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        0,
    ]
    assert seen['rope'].position_ids.tolist() == expected_positions
    assert (seen['rope'].cp_rank, seen['rope'].cp_size) == (0, 1)
    valid = torch.tensor(([True] * 5 + [False] * 3 if cp_size == 1 else []) + [True] * 7 + [False])
    assert torch.count_nonzero(output[~valid]) == 0


@pytest.mark.parametrize('rows', [0, 4])
def test_output_rope_native_and_empty_fallback_rotate_once(monkeypatch, rows):
    """Native THD fallback remains differentiable and empty outputs never launch RoPE."""
    from megatron.core.ops.attention.csa import output_rope

    _, layers = _setup()
    layer = layers[0]
    layer.compress_ratio = 0
    layer.use_fused_kernels = rows == 0
    q = torch.randn(rows, 2, 8, requires_grad=True)
    local = torch.randn(rows, 8, requires_grad=True)
    window = torch.arange(rows).view(rows, 1).expand(rows, 4).int()
    params = output_rope.OutputRopeParams(
        torch.ones(max(rows, 1), 4),
        torch.zeros(max(rows, 1), 4),
        4,
        4,
        cu_seqlens_q=torch.tensor([0, rows], dtype=torch.int32),
    )
    calls = []

    def rotate(x, *_args, **_kwargs):
        calls.append(x)
        return x * torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])

    monkeypatch.setattr(output_rope, 'fused_mla_rope_out_of_place', rotate)
    output, loss = layer._forward_attention(
        q, local, window, None, None, CSA2State(), False, params
    )
    assert loss is None and output.shape == (rows, 16)
    assert len(calls) == (1 if rows else 0)
    output.sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def _setup():
    cfg = MLATransformerConfig(
        num_layers=3,
        hidden_size=16,
        num_attention_heads=2,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        multi_latent_attention=True,
        experimental_attention_variant='dsv4_hybrid',
        dsv4_version='v4.1',
        qk_layernorm=True,
        normalization='RMSNorm',
        q_lora_rank=8,
        v_head_dim=8,
        qk_head_dim=4,
        qk_pos_emb_head_dim=4,
        kv_lora_rank=4,
        csa_compress_ratios=[2, 2, 2],
        csa2_kv_source_layers=[0],
        csa2_index_source_layers=[0, 1],
        csa_window_size=4,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=8,
        dsa_indexer_topk=2,
        dsa_indexer_rotate_activation=False,
        dsa_kernel_backend='none',
        add_bias_linear=False,
        attention_dropout=0,
        hidden_dropout=0,
        dsa_indexer_loss_coeff=0.01,
    )
    g = Group()
    groups = ProcessGroupCollection(tp=g, cp=g, pp=g, dp=g, dp_cp=g, ep=g, expt_dp=g)
    rope = CpuRoPE()
    mods = CompressedSparseAttentionSubmodules(
        compressor=partial(
            CSA2Compressor,
            submodules=CompressorSubmodules(linear_wkv=Linear, linear_wgate=Linear, norm=Norm),
        ),
        indexer=partial(
            CSA2Indexer,
            submodules=CSA2IndexerSubmodules(
                linear_wq_b=Linear, linear_wk=Linear, k_norm=Norm, linear_weights_proj=Linear
            ),
        ),
    )
    layers = [
        CompressedSparseAttention2(
            cfg, mods, i + 1, AttnMaskType.causal, 'self', groups, rotary_pos_emb=rope
        )
        for i in range(3)
    ]
    return cfg, layers


def _muon_attention(monkeypatch, layer_number):
    """Construct real V4.1 attention modules with CPU projection implementations."""
    from megatron.core.transformer.experimental_attention_variant import (
        deepseek_v4_hybrid_attention as attention,
    )
    from megatron.core.transformer.experimental_attention_variant.csa2_module_spec import (
        get_csa2_module_spec_for_backend,
    )
    from megatron.core.transformer.spec_utils import build_module

    class Backend:
        def linear(self):
            return Linear

        column_parallel_linear = linear
        row_parallel_linear = linear

        def layer_norm(self, **kwargs):
            return Norm

    cfg, _ = _setup()
    cfg.num_layers = 4
    cfg.csa_compress_ratios = [2, 2, 2, 0]
    cfg.num_attention_heads = 4
    cfg.dsa_indexer_n_heads = 3
    cfg.output_projection_groups = 2
    cfg.output_projection_lora_rank = 4
    monkeypatch.setattr(attention, 'YarnRotaryEmbedding', lambda *a, **kw: CpuRoPE())
    monkeypatch.setattr(attention, 'RotaryEmbedding', lambda *a, **kw: CpuRoPE())
    spec = get_csa2_module_spec_for_backend(Backend())
    spec.submodules.linear_o_group_proj = None
    g = Group()
    groups = ProcessGroupCollection(tp=g, cp=g, pp=g, dp=g, dp_cp=g, ep=g, expt_dp=g)
    return build_module(spec, config=cfg, layer_number=layer_number, pg_collection=groups)


@pytest.mark.parametrize('layer_number', [1, 2, 3, 4], ids=['full', 'reindex', 'reuse', 'swa'])
@pytest.mark.parametrize('per_head', [False, True])
@pytest.mark.parametrize('coefficients,steps', [('quintic', 5), ('deepseekv4', 10)])
def test_v41_muon_head_updates_and_resume(monkeypatch, layer_number, per_head, coefficients, steps):
    """Full Q heads (including RoPE) match separate optimizers and survive master copies."""
    pytest.importorskip('emerging_optimizers')
    from emerging_optimizers.orthogonalized_optimizers import Muon

    from megatron.core.optimizer.emerging_optimizers import (
        TensorParallelMuon,
        _get_qkv_split_shapes,
        get_supported_coefficient_types,
    )
    from megatron.core.tensor_parallel.layers import copy_tensor_model_parallel_attributes

    if coefficients not in get_supported_coefficient_types():
        pytest.skip(f'{coefficients} is unavailable in this emerging-optimizers release')
    torch.manual_seed(6326)
    model = _muon_attention(monkeypatch, layer_number)
    expected = {'linear_q_up_proj.weight': (4, 8), 'linear_kv_proj.weight': (1, 8)}
    if layer_number in (1, 2):
        expected['core_attention.indexer.linear_wq_b.weight'] = (3, 8)
    if layer_number == 1:
        expected['core_attention.indexer.linear_wk.weight'] = (1, 8)
    annotated = {
        name: param
        for name, param in model.named_parameters()
        if getattr(param, 'qkv_layout', None) is not None
    }
    assert annotated.keys() == expected.keys()

    for name, param in annotated.items():
        heads, head_dim = expected[name]
        layout = param.qkv_layout
        assert (layout.num_groups, layout.per_head_split_shapes) == (heads, (head_dim,))
        param.is_qkv = True
        param.qkv_split_shapes, param.qkv_split_shapes_global = _get_qkv_split_shapes(
            layout, split_qkv_per_head=per_head
        )
        # Same attribute-copy helper used by the BF16/FP8 master-weight wrapper.
        master = nn.Parameter(param.detach().clone().float())
        copy_tensor_model_parallel_attributes(master, param)
        kwargs = dict(
            lr=0.01,
            momentum=0.95,
            nesterov=True,
            weight_decay=0.1,
            fp32_matmul_prec='highest',
            scale_mode='spectral',
            extra_scale_factor=0.18,
            coefficient_type=coefficients,
            num_ns_steps=steps,
        )

        def make_optimizer(weight):
            return TensorParallelMuon(
                [weight],
                split_qkv=True,
                split_qkv_per_head=per_head,
                is_qkv_fn=lambda p: p.is_qkv,
                **kwargs,
            )

        optimizer = make_optimizer(master)
        split_size = head_dim if per_head else param.shape[0]
        references = [nn.Parameter(p.clone()) for p in master.detach().split(split_size)]
        reference = Muon(references, **kwargs)
        for step in range(3):
            grad = torch.randn_like(master)
            master.grad = grad.clone()
            for head, head_grad in zip(references, grad.split(split_size)):
                head.grad = head_grad.clone()
            optimizer.step()
            reference.step()
            torch.testing.assert_close(master, torch.cat(references), atol=2e-6, rtol=2e-5)
            if step == 0:
                state = copy.deepcopy(optimizer.state_dict())
                optimizer = make_optimizer(master)
                optimizer.load_state_dict(state)


def _forward(layers, state):
    outputs, inputs = [], []
    for layer in layers:
        x = torch.randn(8, 2, 16, requires_grad=True)
        qr = torch.randn(8, 2, 8, requires_grad=True)
        q = torch.randn(8, 2, 2, 8, requires_grad=True)
        kv = torch.randn(8, 2, 1, 8, requires_grad=True)
        inputs.append((x, qr, q, kv))
        outputs.append(layer(q, kv, kv, None, x=x, qr=qr, csa2_state=state))
    return outputs, inputs


def test_shared_owners_receive_main_and_auxiliary_gradients(monkeypatch):
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "save_loss_to_tracker", lambda **kwargs: None)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    _, layers = _setup()
    first, first_inputs = _forward(layers, CSA2State())
    second, _ = _forward(layers, CSA2State())
    assert layers[2].indexer is None and layers[1].compressor is None
    sum(output.square().mean() for output in first + second).backward()
    assert layers[0].compressor.linear_wkv.weight.grad.abs().sum() > 0
    assert layers[0].indexer.linear_wk.weight.grad.abs().sum() > 0
    assert layers[1].indexer.linear_wq_b.weight.grad.abs().sum() > 0
    assert torch.isfinite(first_inputs[0][0].grad).all()


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "batch,dim,padded", [(1, 128, False), (1, 512, True), (2, 128, True), (2, 512, False)]
)
def test_cudnn_r2_pool_matches_fp32_reference_and_graph(monkeypatch, batch, dim, padded):
    """Exercise fused dispatch, padding gradients and warmed CUDA Graph replay."""
    import copy

    from tests.unit_tests.determinism.kernels.harness import deterministic_algorithms

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cuDNN compressor requires SM100 or newer")
    pytest.importorskip("cudnn.csa.compressor")
    cfg, _ = _setup()
    cfg.dsa_kernel_backend = "cudnn"
    cfg.params_dtype = torch.bfloat16
    cfg.v_head_dim = dim
    compressor = CSA2Compressor(
        cfg,
        CompressorSubmodules(linear_wkv=Linear, linear_wgate=Linear, norm=Norm),
        2,
        ProcessGroupCollection(tp=Group()),
    )
    assert compressor._fused_r2_pool is not None
    copy.deepcopy(compressor)
    monkeypatch.setattr(
        compressor,
        "_pool_r2_native",
        lambda *args: pytest.fail("the cuDNN pooling test must use the fused path"),
    )
    torch.manual_seed(790)
    kv = torch.randn(66, batch, dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(kv)
    valid = torch.arange(33, device="cuda") % 5 != 2 if padded else None
    if valid is not None:
        invalid = ~valid.repeat_interleave(2)[:, None, None]
        kv = kv.masked_fill(invalid, torch.nan)
        gate = gate.masked_fill(invalid, torch.nan)
    kv.requires_grad_()
    gate.requires_grad_()
    clean_kv = kv if valid is None else kv.masked_fill(invalid, 0)
    clean_gate = gate if valid is None else gate.masked_fill(invalid, 0)
    expected = (
        (
            clean_kv.float().reshape(33, 2, batch, dim)
            * clean_gate.float().reshape(33, 2, batch, dim).softmax(1)
        )
        .sum(1)
        .bfloat16()
    )
    if valid is not None:
        expected = expected.masked_fill(~valid[:, None, None], 0)
    probe = torch.randn_like(expected)
    if valid is not None:
        probe = probe.masked_fill(~valid[:, None, None], torch.nan)
    reference_grads = torch.autograd.grad(expected, (kv, gate), probe)

    def run(keys=kv, gates=gate):
        output = compressor._pool_r2(keys, gates, keys.dtype, valid)
        return output, *torch.autograd.grad(output, (keys, gates), probe)

    with deterministic_algorithms(False):
        actual = run()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            static_kv = kv.detach().clone().requires_grad_()
            static_gate = gate.detach().clone().requires_grad_()
            # Autograd leaves must warm up on the stream used for capture.
            run(static_kv, static_gate)
            run(static_kv, static_gate)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run(static_kv, static_gate)
        graph.replay()
        torch.cuda.synchronize()
    for observed, reference in zip(actual, (expected, *reference_grads)):
        assert torch.isfinite(observed).all()
        assert observed.dtype == torch.bfloat16
        cosine = torch.nn.functional.cosine_similarity(
            observed.float().flatten(), reference.float().flatten(), dim=0
        )
        assert cosine >= 0.999999
    for observed, reference in zip(captured, actual):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)


def test_r2_pool_keeps_native_fallback(monkeypatch):
    """Unsupported inputs and strict determinism must not launch the frontend."""
    from megatron.core.ops.attention.csa.cudnn_pooling import _pool_r2

    def forbidden(*args, **kwargs):
        pytest.fail("native fallback must not launch a cuDNN kernel")

    kv = torch.randn(8, 2, 8)
    assert _pool_r2(kv, kv, kv.dtype, forward_fn=forbidden, backward_fn=forbidden) is None
    _, layers = _setup()
    compressor = layers[0].compressor
    compressor._fused_r2_pool = lambda *args: None
    expected = (kv.reshape(4, 2, 2, 8) * kv.reshape(4, 2, 2, 8).softmax(1)).sum(1)
    torch.testing.assert_close(compressor._pool_r2(kv, kv, kv.dtype), expected)


def test_auxiliary_objective_does_not_train_main_compressor(monkeypatch):
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "save_loss_to_tracker", lambda **kwargs: None)
    _, layers = _setup()
    state = CSA2State(defer_indexer_loss=True)
    _forward(layers[:1], state)
    state.indexer_loss.backward()
    assert layers[0].compressor.linear_wkv.weight.grad is None
    assert layers[0].indexer.linear_wk.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("ratio", [1, 2, 4, 128])
def test_compression_capacity_preserves_main_alignment(ratio):
    import math

    from megatron.core.transformer.experimental_attention_variant.csa_utils.cp_utils import (
        get_cp_compressor_capacity,
    )

    for rows in (1, 7, 128, 257):
        halo = 8 if ratio == 4 else ratio
        alignment = 32 // math.gcd(32, ratio)
        expected = max(1, (rows + halo) // ratio)
        expected = (expected + alignment - 1) // alignment * alignment
        assert get_cp_compressor_capacity(rows, ratio) == expected


def test_candidate_mask_does_not_change_tied_causal_prefix():
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
        candidate_blocks_from_scores,
    )

    short = torch.zeros(1, 3, 6)
    visible = torch.tensor([[1, 3, 6]])
    short.masked_fill_(torch.arange(6)[None, None, :] >= visible[..., None], -torch.inf)
    long = torch.nn.functional.pad(short, (0, 5), value=-torch.inf)
    a = candidate_blocks_from_scores(short, visible, 2, 2)
    b = candidate_blocks_from_scores(long, visible, 2, 2)
    assert torch.equal(a.indices, b.indices)
    assert torch.equal(a.lengths, b.lengths)


def test_window_teacher_includes_sink_and_ignores_invalid_keys():
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_teacher_lse import (
        fused_csa_window_lse,
    )

    query = torch.ones(3, 2, 4)
    keys = torch.ones(3, 4)
    sink = torch.tensor([0.0, 32.0])
    indices = torch.tensor([[0, -1], [0, 1], [-1, -1]], dtype=torch.int32)
    result = fused_csa_window_lse(query, keys, sink, indices, 0.5)
    expected = torch.stack(
        (
            torch.logaddexp(sink, torch.full_like(sink, 2.0)),
            torch.logaddexp(sink, torch.full_like(sink, 2.0 + torch.log(torch.tensor(2.0)).item())),
            sink,
        )
    )
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_window_teacher_gpu_parity_and_bit_exact_replay():
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_teacher_lse import (
        fused_csa_window_lse,
    )
    from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

    query = torch.randn(32, 64, 512, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.linspace(0, 32, 64, device="cuda")
    rows = torch.arange(32, device="cuda")[:, None]
    ids = rows - torch.arange(8, device="cuda")[None, :]
    ids = ids.clamp_min(-1).int()
    scale = 512**-0.5
    expected = fused_csa_window_lse(query.cpu(), keys.cpu(), sink.cpu(), ids.cpu(), scale)
    actual = fused_csa_window_lse(query, keys, sink, ids, scale)
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
    assert_replays_bit_exact(
        lambda q, k, s, indices: fused_csa_window_lse(q, k, s, indices, scale),
        (query, keys, sink, ids),
        backward=False,
        replays=3,
        contention=True,
        what="CSA2 window/sink teacher",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_candidate_score_and_compaction_gpu_parity_and_replay():
    pytest.importorskip("cudnn")
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
        _select_blocks,
        candidate_blocks_from_scores,
    )
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_indexer import (
        _score_chunk,
    )
    from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

    q = torch.randn(16, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(16, 1, device="cuda", dtype=torch.int32)
    starts = torch.zeros(16, device="cuda", dtype=torch.int32)
    visible = torch.arange(16, device="cuda", dtype=torch.int32) * 2

    def score(query, keys, w):
        return _score_chunk(
            query,
            keys,
            w,
            ids,
            starts,
            starts,
            visible,
            width=32,
            stride=32,
            mode=0,
            block_size=1,
            selection=True,
        )

    scores = score(q, k, weights)
    scaled_weights = (weights.float() / 64).bfloat16().float()
    expected = (
        torch.einsum("qhd,kd->qhk", q.float(), k.float()).relu() * scaled_weights[..., None]
    ).sum(1)
    expected.masked_fill_(torch.arange(32, device="cuda")[None, :] >= visible[:, None], -torch.inf)
    torch.testing.assert_close(scores, expected, rtol=2e-4, atol=2e-4)
    blocks = _select_blocks(scores, visible, 2, 8)
    reference = candidate_blocks_from_scores(expected, visible, 2, 8)
    assert torch.equal(blocks.indices, reference.indices)
    assert torch.equal(blocks.lengths, reference.lengths)
    assert_replays_bit_exact(
        score,
        (q, k, weights),
        backward=False,
        replays=3,
        contention=True,
        what="CSA2 BF16 candidate scoring",
    )
    assert_replays_bit_exact(
        lambda s: (
            _select_blocks(s, visible, 2, 8).indices,
            _select_blocks(s, visible, 2, 8).lengths,
        ),
        (scores,),
        backward=False,
        replays=3,
        what="CSA2 candidate compaction",
    )


def test_cp_layout_uses_main_packed_metadata():
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
        build_csa2_thd_layout,
    )

    class CPGroup:
        def size(self):
            return 2

        def rank(self):
            return 1

    logical = torch.tensor([0, 5, 12], dtype=torch.int32)
    padded = torch.tensor([0, 8, 16], dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded,
        max_seqlen_q=8,
        max_seqlen_kv=8,
    )
    layout = build_csa2_thd_layout(packed, 8, cp_group=CPGroup())
    assert layout.global_start == 8
    assert layout.valid_tokens.tolist() == [True] * 7 + [False]
    assert layout.position_ids.tolist() == list(range(7)) + [0]
    assert layout.for_compression(2).valid_groups.sum() == 5


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("output_rope", [False, True])
@pytest.mark.parametrize("indexer_loss", ["candidate", "sparse", "sparse_cudnn"])
def test_fused_shared_attention_and_indexer_gradients_match_native(
    monkeypatch, ratio, output_rope, indexer_loss
):
    """Exercise real Full/Reindex/Reuse kernels with shared producer gradients.

    ``sparse`` supervises each layer's Top-K with the Triton objective and
    ``sparse_cudnn`` with MCORE_CSA2_CUDNN_INDEXER_LOSS's cuDNN-FE kernels, both
    with the production H32 indexer.
    """
    import os
    import shutil
    import subprocess
    from dataclasses import replace

    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.ops.attention.csa.output_rope import OutputRopeParams
    from megatron.core.transformer.experimental_attention_variant.csa_utils import csa2_indexer

    pytest.importorskip("cudnn")
    flash_mla_cuda = pytest.importorskip("flash_mla.cuda")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    cudnn_calls = []
    if indexer_loss == "sparse_cudnn":
        if torch.cuda.get_device_capability()[0] != 10:
            pytest.skip("the cuDNN sparse indexer loss requires an SM10x GPU")
        cudnn_loss = csa2_indexer._precompute_indexer_loss_cudnn

        def counted_cudnn_loss(*args):
            cudnn_calls.append(tuple(args[0].shape))
            return cudnn_loss(*args)

        monkeypatch.setattr(csa2_indexer, "_CUDNN_INDEXER_LOSS", True)
        monkeypatch.setattr(csa2_indexer, "_precompute_indexer_loss_cudnn", counted_cudnn_loss)
    if torch.cuda.get_device_capability()[0] == 9:
        # CI omits SM90 kernels. Check the binary before launching: a failed
        # FlashMLA call can leave CUDA errors behind even after pytest.skip.
        cuobjdump = shutil.which("cuobjdump")
        if cuobjdump is None:
            pytest.skip("cuobjdump is required to check FlashMLA's SM90 kernel availability")
        kernels = subprocess.check_output(
            [cuobjdump, "--list-elf", flash_mla_cuda.__file__], text=True, timeout=30
        )
        if ".sm_90" not in kernels:
            pytest.skip("FlashMLA was built without SM90 kernels")
    monkeypatch.setattr(DSAIndexerLossLoggingHelper, "save_loss_to_tracker", lambda **kwargs: None)
    cfg, _ = _setup()
    cfg = replace(
        cfg,
        params_dtype=torch.bfloat16,
        bf16=True,
        num_attention_heads=64,
        v_head_dim=512,
        q_lora_rank=128,
        qk_pos_emb_head_dim=64,
        csa_compress_ratios=[ratio] * 3,
        dsa_indexer_n_heads=64 if indexer_loss == "candidate" else 32,
        dsa_indexer_use_sparse_loss=indexer_loss != "candidate",
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=32,
        csa_window_size=8,
        csa2_candidate_source_layer=0,
        csa2_candidate_topk_blocks=4,
        csa2_candidate_block_size=8,
    )
    g = Group()
    groups = ProcessGroupCollection(tp=g, cp=g)
    modules = CompressedSparseAttentionSubmodules(
        compressor=partial(
            CSA2Compressor,
            submodules=CompressorSubmodules(linear_wkv=Linear, linear_wgate=Linear, norm=Norm),
        ),
        indexer=partial(
            CSA2Indexer,
            submodules=CSA2IndexerSubmodules(
                linear_wq_b=Linear, linear_wk=Linear, k_norm=Norm, linear_weights_proj=Linear
            ),
        ),
    )
    rope = RotaryEmbedding(64, rotary_percent=1.0, cp_group=g)
    native = (
        nn.ModuleList(
            [
                CompressedSparseAttention2(
                    cfg, modules, i + 1, AttnMaskType.causal, 'self', groups, rotary_pos_emb=rope
                )
                for i in range(3)
            ]
        )
        .cuda()
        .bfloat16()
    )
    fused = (
        nn.ModuleList(
            [
                CompressedSparseAttention2(
                    replace(cfg, dsa_kernel_backend='cudnn'),
                    modules,
                    i + 1,
                    AttnMaskType.causal,
                    'self',
                    groups,
                    rotary_pos_emb=rope,
                )
                for i in range(3)
            ]
        )
        .cuda()
        .bfloat16()
    )
    # Attention sinks are FP32 in the real module, including BF16 training.
    for layers in (native, fused):
        for layer in layers:
            layer.attn_sink.data = layer.attn_sink.data.float()
    fused.load_state_dict(native.state_dict())
    shapes = ((32, 2, 64, 512), (32, 2, 1, 512), (32, 2, 16), (32, 2, 128))
    inputs = [
        tuple(torch.randn(shape, device='cuda', dtype=torch.bfloat16) for shape in shapes)
        for _ in range(3)
    ]
    out_rope = None
    if output_rope:
        cos, sin = rope.get_cached_cos_sin(32, dtype=torch.bfloat16, mscale=1.0)
        out_rope = OutputRopeParams(cos, sin, 448, 64, sbhd_batch_size=2)
    results = []
    for layers in (native, fused):
        state, outputs, leaves = CSA2State(), [], []
        for layer, values in zip(layers, inputs):
            q, kv, x, qr = [v.detach().clone().requires_grad_() for v in values]
            leaves.extend((q, kv, x, qr))
            outputs.append(layer(q, kv, kv, None, x=x, qr=qr, csa2_state=state, out_rope=out_rope))
        sum(y.float().square().mean() for y in outputs).backward()
        grads = [p.grad for p in layers.parameters()] + [x.grad for x in leaves]
        results.append((outputs, grads))
    if indexer_loss == "sparse_cudnn":
        # The Full owner and the Reindex layer each ran the cuDNN objective; Reuse has none.
        assert len(cudnn_calls) == 2
    for actual, expected in zip(results[1][0], results[0][0]):
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        assert cosine > 0.999
    for actual, expected in zip(results[1][1], results[0][1]):
        if expected is None:
            assert actual is None
            continue
        assert actual is not None and torch.isfinite(actual).all()
        norm = expected.float().norm()
        if norm < 1e-7:
            assert actual.float().norm() < 1e-6
            continue
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        assert cosine > 0.99
        assert (actual.float() - expected.float()).norm() / norm < 0.03


def _sparse_indexer_loss_inputs(width, per_token, generator):
    """MODE 2 inputs: two packed sequences, padding rows and ids outside their key spans."""
    span, rows_per_sequence, padding_rows, heads, teacher_heads = 640, 128, 8, 32, 64
    rows, keys, device = 2 * rows_per_sequence + padding_rows, 2 * span, 'cuda'

    def randn(*shape, scale=1.0):
        values = torch.randn(shape, device=device, generator=generator) * scale
        return values.to(torch.bfloat16)

    sequence = torch.arange(rows, device=device) // rows_per_sequence
    position = torch.arange(rows, device=device) % rows_per_sequence
    real = sequence < 2
    starts = torch.where(real, sequence * span, 0).int()
    visible = torch.where(real, ((position + 1) * 5).clamp(max=span), 0).int()
    # Random Top-K among each row's visible keys: ascending physical ids, -1 tail.
    draw = torch.rand(rows, span, device=device, generator=generator)
    draw = draw.masked_fill(torch.arange(span, device=device) >= visible[:, None], 2.0)
    local = draw.topk(width, dim=-1, largest=False).indices
    sentinel = torch.iinfo(torch.int32).max
    ids = torch.where(draw.gather(1, local) < 2.0, local + starts[:, None], sentinel)
    ids = ids.sort(-1).values
    ids = ids.masked_fill(ids == sentinel, -1).int()
    # Ids just past a row's span, and padding rows with in-range ids, are not supervised.
    for row in (3, rows_per_sequence + 12):
        ids[row, -1] = starts[row] + visible[row]
    ids[~real] = 7
    q, k, weights = randn(rows, heads, 128), randn(keys, 128), randn(rows, heads)
    teacher_q, teacher_k = randn(rows, teacher_heads, 512, scale=0.5), randn(keys, 512)
    scale = 512**-0.5
    supervised = (ids >= starts[:, None]) & (ids < (starts + visible)[:, None])
    logits = torch.einsum(
        'rhd,rwd->rhw', teacher_q.float(), teacher_k[ids.clamp_min(0).long()].float()
    )
    logits = (logits * scale).masked_fill(~supervised[:, None, :], float('-inf'))
    # FlashMLA LSE over window and Top-K keys, plus the sink: some mass is outside Top-K.
    window = torch.randn(rows, teacher_heads, device=device, generator=generator) + 1.0
    lse = torch.logaddexp(torch.logsumexp(logits, -1), window)
    divisor = torch.ones((), device=device) if per_token else real.sum().float()
    options = (width, 2, 1, scale, 0.25)
    args = (q, k, weights, teacher_q, teacher_k, lse, ids, visible, starts, visible, divisor)
    return (*args, options), ~real


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [72, 512])
@pytest.mark.parametrize("per_token", [False, True])
def test_cudnn_sparse_indexer_loss_matches_triton_and_replays(width, per_token):
    """MCORE_CSA2_CUDNN_INDEXER_LOSS keeps Triton's MODE 2 objective and unit-loss gradients.

    The cuDNN backward rounds the score-gradient matrix to BF16 before the dQ/dK
    MMAs, where Triton keeps it in FP32, hence the 1% relative-norm tolerance.
    """
    pytest.importorskip("cudnn")
    from megatron.core.transformer.experimental_attention_variant.csa_utils import csa2_indexer

    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the cuDNN sparse indexer loss requires an SM10x GPU")
    generator = torch.Generator(device='cuda').manual_seed(width + per_token)
    args, padding = _sparse_indexer_loss_inputs(width, per_token, generator)
    q, k, weights, teacher_q, teacher_k, lse, ids, *_, options = args
    assert csa2_indexer._cudnn_indexer_loss_unsupported(*args[:7], options) is None
    wide = q.repeat_interleave(2, dim=1)
    assert csa2_indexer._cudnn_indexer_loss_unsupported(wide, *args[1:7], options) is not None

    def relative_errors(actual, expected):
        return [
            ((a.float() - e.float()).norm() / e.float().norm()).item()
            for a, e in zip(actual, expected)
        ]

    expected_loss, expected = csa2_indexer._precompute_indexer_loss(*args, full_lse=True)
    loss, gradients = csa2_indexer._precompute_indexer_loss_cudnn(*args)
    torch.testing.assert_close(loss, expected_loss, rtol=2e-4, atol=1e-6)
    for actual, wanted in zip(gradients, expected):
        assert actual.shape == wanted.shape and actual.dtype == wanted.dtype
        assert torch.isfinite(actual).all()
    assert max(relative_errors(gradients, expected)) < 1e-2
    # Padding rows supervise no key, so their Q/W gradients stay exactly zero.
    assert gradients[0][padding].count_nonzero() == 0
    assert gradients[2][padding].count_nonzero() == 0

    # Capture after warmup, then replay on new values in the same storage.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        csa2_indexer._precompute_indexer_loss_cudnn(*args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_loss, captured = csa2_indexer._precompute_indexer_loss_cudnn(*args)
    for tensor in (q, k, weights, teacher_q, teacher_k):
        tensor.copy_(torch.randn(tensor.shape, device='cuda', generator=generator))
    lse.add_(torch.rand(lse.shape, device='cuda', generator=generator))
    graph.replay()
    replay_loss, replayed = captured_loss.clone(), [g.clone() for g in captured]
    eager_loss, eager = csa2_indexer._precompute_indexer_loss_cudnn(*args)
    torch.testing.assert_close(replay_loss, eager_loss, rtol=1e-6, atol=0)
    # dK accumulates with unordered FP32 bulk adds, so allow last-bit BF16 differences.
    assert max(relative_errors(replayed, eager)) < 1e-3
    assert max(relative_errors(eager, gradients)) > 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "ratio,heads,cp", [(r, h, c) for r in (1, 2) for h in (32, 64) for c in (1, 2)]
)
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_packed_selection_matches_scoring_precision(monkeypatch, ratio, heads, cp, precision):
    """Check compact/source/Reindex selection across empty, padded and CP-cut segments."""
    from dataclasses import replace
    from types import SimpleNamespace

    from megatron.core.ops.attention.csa.indexer_quantization import quantize_indexer_mxfp8_logical
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
        candidate_blocks_from_scores,
        fused_candidate_blocks,
    )
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_indexer import (
        fused_candidate_topk,
        fused_compact_topk,
        prepare_csa2_indexer_inputs,
    )
    from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
        build_csa2_thd_layout,
    )
    from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("Compact selection requires SM100")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(487)
    logical = torch.tensor([0, 65, 65, 322], dtype=torch.int32, device="cuda")
    physical = torch.tensor([0, 66, 66, 326], dtype=torch.int32, device="cuda")
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=260,
        max_seqlen_kv=260,
    )
    group = SimpleNamespace(size=lambda: cp, rank=lambda: cp - 1)
    layout = build_csa2_thd_layout(params, 326 // cp, cp_group=group)
    compressed = layout.for_compression(ratio)
    q = torch.randn(326 // cp, heads, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(compressed.capacity, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(q.shape[0], heads, dtype=torch.bfloat16, device="cuda")
    inputs = prepare_csa2_indexer_inputs(q, k, w, ratio, None, layout, compressed)

    def dequantize(x):
        if precision == "bf16":
            return x.float()
        data, scales = quantize_indexer_mxfp8_logical(x)
        scale = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
        return (data.float().reshape(-1, 128) * scale).reshape(x.shape)

    weights = (w.float() * (128**-0.5 * heads**-0.5)).bfloat16().float()
    scores = (
        torch.einsum('qhd,kd->qhk', dequantize(q), dequantize(k)).relu() * weights[..., None]
    ).sum(1)
    keys = torch.arange(k.shape[0], device=q.device)[None, :]
    valid = (keys >= inputs.starts[:, None]) & (keys < (inputs.starts + inputs.visible)[:, None])
    scores.masked_fill_(~valid | ~inputs.valid[:, None], -torch.inf)

    def topk(s):
        ids = s.argsort(dim=-1, descending=True, stable=True)[:, :7]
        sentinel = torch.iinfo(torch.int32).max
        ids = ids.masked_fill(~s.gather(1, ids).isfinite(), sentinel).sort(-1).values
        return ids.masked_fill(ids == sentinel, -1).int()

    expected = topk(scores)
    torch.testing.assert_close(fused_compact_topk(inputs, 7, precision), expected)
    selected, blocks = fused_candidate_blocks(
        inputs, topk_blocks=3, block_size=8, precision=precision, token_topk=7
    )
    torch.testing.assert_close(selected, expected)
    local_keys = inputs.starts[:, None] + torch.arange(inputs.max_keys, device=q.device)
    local_scores = scores.gather(1, local_keys.clamp_max(k.shape[0] - 1))
    local_scores.masked_fill_(local_keys >= (inputs.starts + inputs.visible)[:, None], -torch.inf)
    expected_blocks = candidate_blocks_from_scores(local_scores, inputs.visible, 3, 8)
    torch.testing.assert_close(blocks.indices, expected_blocks.indices)
    torch.testing.assert_close(blocks.lengths, expected_blocks.lengths)
    mask = blocks.to_mask(k.shape[0], thd_layout=layout, compressed_layout=compressed)
    torch.testing.assert_close(
        fused_candidate_topk(replace(inputs, candidates=blocks), 7, precision),
        topk(scores.masked_fill(~mask, -torch.inf)),
    )

    def select(query, keys, weights):
        prepared = prepare_csa2_indexer_inputs(
            query, keys, weights, ratio, None, layout, compressed
        )
        source_topk, candidates = fused_candidate_blocks(
            prepared, topk_blocks=3, block_size=8, precision=precision, token_topk=7
        )
        return (
            fused_compact_topk(prepared, 7, precision),
            source_topk,
            candidates.indices,
            fused_candidate_topk(replace(prepared, candidates=candidates), 7, precision),
        )

    assert_replays_bit_exact(
        select, (q, k, w), backward=False, replays=3, what=f"CSA2 {precision} packed selection"
    )
    # Compact cuDNN selection has an allocation-free output contract during
    # capture. The caller must supply slot-local outputs for both precisions.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_compact_topk(inputs, 7, precision)
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def _cp_compressor_tokens(rank, device="cpu"):
    """Include empty segments, padding and groups spanning a CP boundary."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
        CSA2THDLayout,
        get_thd_token_metadata,
    )

    logical = torch.tensor([0, 3, 3, 14, 25], dtype=torch.int32, device=device)
    padded = torch.tensor([0, 4, 4, 16, 32], dtype=torch.int32, device=device)
    rows = torch.arange(rank * 8, (rank + 1) * 8, device=device)
    seq, positions, valid = get_thd_token_metadata(logical, padded, rows)
    return CSA2THDLayout(8, 16, logical, padded, seq, positions, valid, rank, 4)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("sanitized", [False, True])
def test_cp_compressor_projects_before_compaction_and_reuses_hidden(ratio, rank, sanitized):
    """Compare all gradients and saved hidden storage to the compact-hidden oracle."""
    import copy

    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_context_parallel import (
        build_csa2_cp_compression_layout,
    )

    cfg, _ = _setup()
    compressor = CSA2Compressor(
        cfg,
        CompressorSubmodules(linear_wkv=Linear, linear_wgate=Linear, norm=Norm),
        ratio,
        ProcessGroupCollection(tp=Group()),
    )
    reference = copy.deepcopy(compressor)
    tokens = _cp_compressor_tokens(rank)
    layout = build_csa2_cp_compression_layout(tokens, ratio, 4)
    x = torch.randn(8, 1, cfg.hidden_size)
    boundary = torch.randn(4, 1, cfg.hidden_size)
    x[~tokens.valid_tokens] = 0 if sanitized else torch.nan
    x.requires_grad_()
    boundary.requires_grad_()
    xr, br = (t.detach().clone().requires_grad_() for t in (x, boundary))
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        output, actual_layout = compressor(
            x, thd_layout=tokens, boundary_hidden=boundary, input_is_sanitized=sanitized
        )
    joined = torch.cat((br, xr))
    inputs = joined[layout.local.source_indices].reshape(-1, 1, cfg.hidden_size)
    inputs = inputs.masked_fill(
        ~layout.local.valid_groups.repeat_interleave(ratio)[:, None, None], 0
    )
    latent, gate = reference._project(inputs)
    if ratio == 2:
        latent = reference._pool_r2(latent, gate, xr.dtype, layout.local.valid_groups)
    expected = reference._normalize_thd(latent, layout.local)
    probe = torch.randn_like(output)
    output.backward(probe)
    expected.backward(probe)
    torch.testing.assert_close(output, expected)
    for actual, wanted in [(x.grad, xr.grad), (boundary.grad, br.grad)]:
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, wanted)
    for actual, wanted in zip(compressor.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, wanted.grad)
    assert torch.equal(actual_layout.gather_indices, layout.gather_indices)
    hidden_saved = [t for t in saved if t.ndim >= 2 and t.shape[-1] == cfg.hidden_size]
    assert all(t.numel() <= x.numel() for t in hidden_saved)
    if sanitized:
        # Linear's [rows, hidden] view must retain the existing hidden storage,
        # never a compact hidden allocation with aligned CP capacity.
        assert any(
            t.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for t in hidden_saved
        )
        assert all(
            t.shape[0] in (ratio, x.shape[0]) or t is compressor.linear_wkv.weight
            for t in hidden_saved
        )


@pytest.mark.parametrize(
    "strategy,fine_grained,overwrite,delayed,expected",
    [
        (None, False, False, False, False),
        (None, False, True, False, True),
        (None, False, False, True, True),
        ("no_shard", True, False, False, False),
        ("optim", False, False, False, True),
        ("optim_grads", False, False, False, True),
        ("optim_grads_params", True, False, False, True),
        ("optim_grads_params", False, False, False, False),
    ],
)
def test_cp_compressor_single_projection_contract(
    strategy, fine_grained, overwrite, delayed, expected
):
    """Catch overwrite requirements even before FSDP's late first-forward hook."""
    from types import SimpleNamespace

    cfg, _ = _setup()
    compressor = CSA2Compressor(
        cfg,
        CompressorSubmodules(linear_wkv=Linear, linear_wgate=Linear, norm=Norm),
        2,
        ProcessGroupCollection(tp=Group()),
    )
    cfg.delay_wgrad_compute = delayed
    weight = compressor.linear_wgate.weight
    weight.overwrite_main_grad = overwrite
    if strategy is not None:
        weight._megatron_fsdp_model = SimpleNamespace(
            data_parallel_sharding_strategy=strategy,
            enable_fine_grained_param_gather_hook=fine_grained,
        )
    calls = []
    hook = compressor.linear_wkv.register_forward_pre_hook(
        lambda _module, args: calls.append(args[0])
    )
    x = torch.randn(8, 1, cfg.hidden_size, requires_grad=True)
    output, _ = compressor(x, thd_layout=_cp_compressor_tokens(1), boundary_hidden=x[:4])
    output.sum().backward()
    hook.remove()
    assert compressor._cp_requires_single_projection() is expected
    assert len(calls) == (1 if expected else 2)
    assert torch.isfinite(compressor.linear_wkv.weight.grad).all()


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("late_overwrite", [False, True])
def test_cp_compressor_accumulates_both_gemms_on_each_microbatch(ratio, late_overwrite):
    """Model TE's first-microbatch overwrite and verify split GEMMs accumulate."""
    import copy
    from types import SimpleNamespace

    class FusedWgrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, overwrite):
            ctx.save_for_backward(x, weight)
            ctx.overwrite = overwrite
            return torch.nn.functional.linear(x, weight)

        @staticmethod
        def backward(ctx, grad):
            x, weight = ctx.saved_tensors
            dw = grad.flatten(0, -2).t() @ x.flatten(0, -2)
            if ctx.overwrite:
                weight.main_grad.copy_(dw)
            else:
                weight.main_grad.add_(dw)
            return grad @ weight, None, None

    class FusedLinear(Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.disable_parameter_transpose_cache = False
            self.is_first_microbatch = True
            self.weight.main_grad = torch.zeros_like(self.weight)

        def forward(self, x):
            first = None if self.disable_parameter_transpose_cache else self.is_first_microbatch
            self.is_first_microbatch = False
            if late_overwrite:
                self.weight.overwrite_main_grad = True
            return FusedWgrad.apply(x, self.weight, first is True or late_overwrite), None

    cfg, _ = _setup()
    reference = CSA2Compressor(
        cfg,
        CompressorSubmodules(linear_wkv=Linear, linear_wgate=Linear, norm=Norm),
        ratio,
        ProcessGroupCollection(tp=Group()),
    )
    compressor = CSA2Compressor(
        copy.deepcopy(cfg),
        CompressorSubmodules(linear_wkv=FusedLinear, linear_wgate=FusedLinear, norm=Norm),
        ratio,
        ProcessGroupCollection(tp=Group()),
    )
    compressor.load_state_dict(reference.state_dict())
    if late_overwrite:
        for linear in (compressor.linear_wkv, compressor.linear_wgate):
            if linear is not None:
                linear.weight._megatron_fsdp_model = SimpleNamespace(
                    data_parallel_sharding_strategy="optim_grads",
                    enable_fine_grained_param_gather_hook=False,
                )
    for _ in range(2):
        if late_overwrite:
            # FSDP overwrites its per-microbatch buffer before reducing it.
            reference.zero_grad(set_to_none=True)
        x = torch.randn(8, 1, cfg.hidden_size, requires_grad=True)
        boundary = torch.randn(4, 1, cfg.hidden_size, requires_grad=True)
        xr, br = (t.detach().clone().requires_grad_() for t in (x, boundary))
        tokens = _cp_compressor_tokens(1)
        actual, _ = compressor(x, thd_layout=tokens, boundary_hidden=boundary)
        expected, _ = reference(xr, thd_layout=tokens, boundary_hidden=br)
        probe = torch.randn_like(actual)
        actual.backward(probe)
        expected.backward(probe)
        for linear, wanted in (
            (compressor.linear_wkv, reference.linear_wkv),
            (compressor.linear_wgate, reference.linear_wgate),
        ):
            if linear is not None:
                torch.testing.assert_close(linear.weight.main_grad, wanted.weight.grad)
                assert linear.disable_parameter_transpose_cache is False
        torch.testing.assert_close(x.grad, xr.grad)
        torch.testing.assert_close(boundary.grad, br.grad)


@pytest.mark.parametrize("local_rows,halo_rows", [(0, 0), (0, 2), (2, 0), (2, 2)])
@pytest.mark.parametrize("empty_compact", [False, True])
def test_cp_projection_empty_storage_and_backward(local_rows, halo_rows, empty_compact):
    """Never index an absent source or gradient tensor, including all-padding maps."""
    from megatron.core.ops.attention.csa.compressor_projection import (
        compact_cp_projection,
        cp_projection_maps,
    )

    total = local_rows + halo_rows
    count = 0 if empty_compact else max(total, 1) + 1
    source = torch.arange(count).reshape(-1, 1).clamp_max(max(total - 1, 0))
    valid = torch.arange(count) < total
    mapping, inverse = cp_projection_maps(source, valid, halo_rows, halo_rows, local_rows)
    local = torch.randn(local_rows, 1, 4, requires_grad=True)
    halo = torch.randn(halo_rows, 1, 4, requires_grad=True)
    output = compact_cp_projection(local, halo, mapping, inverse)
    expected = torch.zeros(count, 1, 4)
    if total and count:
        expected[:total] = torch.cat((halo, local))
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    for value in (local, halo):
        expected_grad = torch.zeros_like(value) if empty_compact else torch.ones_like(value)
        torch.testing.assert_close(value.grad, expected_grad)


@pytest.mark.parametrize("ratio", [1, 2])
def test_cp_compressor_rejects_empty_shard_and_short_halo(ratio):
    """Keep the existing nonempty-CP contract and fail before projecting a short halo."""
    from dataclasses import replace

    cfg, _ = _setup()
    compressor = CSA2Compressor(
        cfg,
        CompressorSubmodules(linear_wkv=Linear, linear_wgate=Linear, norm=Norm),
        ratio,
        ProcessGroupCollection(tp=Group()),
    )
    tokens = _cp_compressor_tokens(1)
    x = torch.randn(8, 1, cfg.hidden_size)
    with pytest.raises(ValueError, match="compression halo"):
        compressor(x, thd_layout=tokens, boundary_hidden=x[: ratio - 1])
    empty = replace(
        tokens,
        total_tokens=0,
        sequence_ids=tokens.sequence_ids[:0],
        position_ids=tokens.position_ids[:0],
        valid_tokens=tokens.valid_tokens[:0],
    )
    with pytest.raises(ValueError, match="nonempty contiguous CP"):
        compressor(x[:0], thd_layout=empty, boundary_hidden=x[:4])


def test_padding_row_list_capacity_and_degenerate_inputs(monkeypatch):
    """The static padding-row list behind MCORE_CSA2_SPARSE_PAD_ZERO, on the host.

    Unused entries rewrite the first valid row unchanged, an all-padding input rewrites row 0,
    empty inputs are untouched, and more padding rows than MCORE_CSA2_PAD_ROW_CAPACITY trip
    the (device-side under CUDA) assert.
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils import thd_utils

    def bits(tensor):
        return tensor.view(torch.int32).tolist()

    monkeypatch.setattr(thd_utils, "PAD_ROW_CAPACITY", 4)
    invalid = torch.tensor([True, False, False, True, True, False])
    rows, keep = thd_utils.padding_rows(invalid)
    assert rows.tolist() == [0, 3, 4, 1] and keep.tolist() == [False, False, False, True]
    values = torch.tensor([[float("nan")], [1.0], [-2.0], [float("inf")], [-0.0], [5.0]])
    original = values.clone()
    saved = thd_utils.save_rows(values, rows)
    assert thd_utils.zero_rows_(values, rows, keep) is values
    assert bits(values) == bits(original.masked_fill(invalid[:, None], 0))
    thd_utils.restore_rows_(values, rows, saved)
    assert bits(values) == bits(original)

    rows, keep = thd_utils.padding_rows(torch.ones(2, dtype=torch.bool))
    assert rows.tolist() == [0, 1, 0, 0] and not keep.any()
    rows, keep = thd_utils.padding_rows(torch.zeros(3, dtype=torch.bool))
    assert rows.tolist() == [0] * 4 and keep.all()

    empty = torch.empty(0, 3)
    rows, keep = thd_utils.padding_rows(torch.zeros(0, dtype=torch.bool))
    assert thd_utils.zero_rows_(empty, rows, keep) is empty
    assert thd_utils.save_rows(empty, rows) is None
    assert thd_utils.restore_rows_(empty, rows, None) is empty

    with pytest.raises(RuntimeError, match="MCORE_CSA2_PAD_ROW_CAPACITY"):
        thd_utils.padding_rows(torch.ones(5, dtype=torch.bool))


@pytest.mark.parametrize(
    "value,expected", [(None, 512), ("64", 64), (" 8 ", 8), ("0", None), ("-3", None), ("x", None)]
)
def test_pad_row_capacity_parsing(monkeypatch, value, expected):
    """MCORE_CSA2_PAD_ROW_CAPACITY takes a positive integer and names itself in errors."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import thd_utils

    if value is None:
        monkeypatch.delenv("MCORE_CSA2_PAD_ROW_CAPACITY", raising=False)
    else:
        monkeypatch.setenv("MCORE_CSA2_PAD_ROW_CAPACITY", value)
    if expected is None:
        with pytest.raises(ValueError, match="MCORE_CSA2_PAD_ROW_CAPACITY"):
            thd_utils._pad_row_capacity()
    else:
        assert thd_utils._pad_row_capacity() == expected


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("cp", [1, 2])
def test_compact_topk_metadata_gives_padding_rows_empty_key_spans(cp, ratio):
    """Padding segments (MCORE_CSA2_TOPK_PAD_SEGMENTS) keep every real row's Q segment start,
    K span and causal offset, and give padding rows empty K spans, without new bounds."""
    from types import SimpleNamespace

    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_indexer import (
        prepare_csa2_indexer_inputs,
    )
    from megatron.core.transformer.experimental_attention_variant.csa_utils.thd_utils import (
        build_csa2_thd_layout,
    )

    logical = torch.tensor([0, 65, 65, 322], dtype=torch.int32)
    physical = torch.tensor([0, 66, 66, 326], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=logical,
        cu_seqlens_kv=logical,
        cu_seqlens_q_padded=physical,
        cu_seqlens_kv_padded=physical,
        max_seqlen_q=260,
        max_seqlen_kv=260,
    )
    rows = 326 // cp
    group = SimpleNamespace(size=lambda: cp, rank=lambda: cp - 1)
    layout = build_csa2_thd_layout(params, rows, cp_group=group)
    compressed = layout.for_compression(ratio)
    inputs = prepare_csa2_indexer_inputs(
        torch.zeros(rows, 2, 8),
        torch.zeros(compressed.capacity, 8),
        torch.zeros(rows, 2),
        ratio,
        None,
        layout,
        compressed,
    )
    cu_q, cu_k, max_q, max_k = inputs.packed_metadata
    offsets = inputs.packed_causal_offsets
    new_q, new_k, new_max_q, new_max_k, new_offsets = inputs.compact_topk_metadata
    assert (new_max_q, new_max_k) == (max_q, max_k)
    assert new_q.numel() == new_k.numel() == new_offsets.numel() + 1
    assert bool((new_q.diff() >= 0).all()) and bool((new_k.diff() >= 0).all())
    assert new_q[0] == cu_q[0] and new_q[-1] == cu_q[-1]
    row_ids = torch.arange(rows, dtype=cu_q.dtype)
    old = torch.searchsorted(cu_q[1:], row_ids, right=True)
    new = torch.searchsorted(new_q[1:], row_ids, right=True)
    real = inputs.valid
    assert bool(real.any()) and bool((~real).any())
    assert torch.equal(new_q[new][real], cu_q[old][real])
    assert torch.equal(new_k[new][real], cu_k[old][real])
    assert torch.equal(new_k[new + 1][real], cu_k[old + 1][real])
    assert torch.equal(new_offsets[new][real], offsets[old][real])
    assert torch.equal(new_k[new + 1][~real], new_k[new][~real])


@pytest.mark.parametrize(
    "value,expected", [(None, 152), ("32", 32), (" 128 ", 128), ("0", None), ("1.5", None)]
)
def test_candidate_chunk_budget_parsing(monkeypatch, value, expected):
    """MCORE_CSA2_CANDIDATE_CHUNK_MIB takes a positive number of MiB and names itself in errors."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import csa2_candidates

    if value is None:
        monkeypatch.delenv("MCORE_CSA2_CANDIDATE_CHUNK_MIB", raising=False)
    else:
        monkeypatch.setenv("MCORE_CSA2_CANDIDATE_CHUNK_MIB", value)
    if expected is None:
        with pytest.raises(ValueError, match="MCORE_CSA2_CANDIDATE_CHUNK_MIB"):
            csa2_candidates._score_chunk_max_bytes()
    else:
        assert csa2_candidates._score_chunk_max_bytes() == expected << 20
