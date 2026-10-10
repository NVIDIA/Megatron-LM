# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Real-CUDA tests for GDN using the existing GDP chunkwise CP backend.

Run with torch.distributed.run; supports two or four ranks. These tests do not
replace the kernels or collectives under test with mocks.
"""

import copy
import os
from contextlib import contextmanager
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

pytest.importorskip("fla.ops.gated_delta_product")
pytest.importorskip("megatron.core.ssm.context_parallel.gdp")

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from megatron.core.ssm.context_parallel.gdp import FLAGatedDeltaProductCPBackend
from megatron.core.ssm.context_parallel.gdp_common import gdp_chunkwise_context_parallel


@pytest.fixture(scope="session", autouse=True)
def distributed():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    owns_group = not dist.is_initialized()
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    yield
    if owns_group and dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="session")
def singleton_group(distributed):
    groups = [dist.new_group([r]) for r in range(dist.get_world_size())]
    return groups[dist.get_rank()]


@contextmanager
def _model_groups(tp=1):
    from megatron.core import config as global_config
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    global_config.ENABLE_EXPERIMENTAL = True
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=1,
        context_parallel_size=dist.get_world_size() // tp,
    )
    model_parallel_cuda_manual_seed(123)
    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        parallel_state.destroy_model_parallel()


def _config(cp, *, tp=1, sp=False, mode="chunkwise", layout="contiguous", **extra):
    from megatron.core.transformer import TransformerConfig

    kw = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        activation_func=F.silu,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        sequence_parallel=sp,
        experimental_attention_variant="gdn",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
        gradient_accumulation_fusion=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        linear_cp_mode=mode,
        linear_cp_layout=layout,
    )
    kw.update(extra)
    return TransformerConfig(**kw)


def _gdn(config, groups, **kwargs):
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_experimental_attention_variant_module_spec,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    spec = get_experimental_attention_variant_module_spec(config)
    return (
        spec.module(
            config,
            submodules=spec.submodules,
            layer_number=1,
            pg_collection=groups,
            bias=False,
            conv_bias=True,
            **kwargs,
        )
        .cuda()
        .bfloat16()
    )


def test_configuration():
    assert _config(4, linear_num_key_heads=1).linear_num_key_heads == 1
    with pytest.raises(ValueError, match="contiguous"):
        _config(2, layout="zigzag")
    with pytest.raises(AssertionError, match="head-partition"):
        _config(4, mode="headwise", layout="zigzag", linear_num_key_heads=1)
    with pytest.raises(ValueError, match="GDN2"):
        _config(2, experimental_attention_variant="gdn2")


@pytest.mark.parametrize(
    "packed,recompute,last_rank_only,key_heads",
    [
        (False, False, False, 4),
        (True, False, False, 4),
        (False, True, False, 4),
        (False, False, True, 4),
        (True, True, True, 4),
        (False, False, False, 1),
        (False, "full", False, 4),
    ],
)
def test_layer_forward_backward(
    singleton_group, monkeypatch, packed, recompute, last_rank_only, key_heads
):
    """Real module, strict checkpoint load, all parameter gradients, and path assertion."""
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net import common
    from megatron.core.ssm.gated_delta_net import gdn as module

    with _model_groups() as pg:
        cp, rank = pg.cp.size(), pg.cp.rank()
        extra = (
            dict(recompute_granularity="selective", recompute_modules=["gdn_norm_out"])
            if recompute is True
            else {}
        )
        extra["linear_num_key_heads"] = key_heads
        cfg = _config(cp, **extra)
        ref = _gdn(_config(1, **extra), ProcessGroupCollection(tp=pg.tp, cp=singleton_group))
        model = _gdn(cfg, pg)
        model.load_state_dict(copy.deepcopy(ref.state_dict()), strict=True)
        assert {k: v.shape for k, v in model.named_parameters()} == {
            k: v.shape for k, v in ref.named_parameters()
        }
        torch.manual_seed(456)
        length = 128 * cp
        x = torch.randn(length, 1, 128, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x) / x.numel()
        if last_rank_only:
            dy[: (cp - 1) * 128] = 0
        params = None
        if packed:
            cu = torch.tensor([0, 40, length - 40, length], dtype=torch.int32, device="cuda")
            params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                total_tokens=length,
                max_seqlen_q=length - 80,
                max_seqlen_kv=length - 80,
            )
        xr = x.clone().requires_grad_()
        yr, _ = ref(xr, attention_mask=None, packed_seq_params=params)
        yr.backward(dy)
        shapes = []
        original = module.gdp_chunkwise_context_parallel

        def spy(**kwargs):
            shapes.append(tuple(kwargs["q"].shape))
            return original(**kwargs)

        monkeypatch.setattr(module, "gdp_chunkwise_context_parallel", spy)

        def forbidden(*args, **kwargs):
            raise AssertionError("Chunkwise GDN used headwise feature all-to-all")

        monkeypatch.setattr(module, "a2a_cp_to_hp", forbidden)
        monkeypatch.setattr(common, "a2a_hp_to_cp", forbidden)
        sl = slice(rank * 128, (rank + 1) * 128)
        xx = x[sl].clone().requires_grad_()
        if recompute == "full":
            from megatron.core import tensor_parallel

            yy = tensor_parallel.checkpoint(
                lambda inputs: model(inputs, attention_mask=None, packed_seq_params=params)[0],
                False,
                xx,
            )
        else:
            yy, _ = model(xx, attention_mask=None, packed_seq_params=params)
        yy.backward(dy[sl])
        assert shapes and all(s == (1, 128, 8, 32) for s in shapes)
        _assert_close(yy, yr[sl], "layer.output")
        _assert_close(xx.grad, xr.grad[sl], "layer.dx")
        for (name, p), (rn, rp) in zip(model.named_parameters(), ref.named_parameters()):
            assert name == rn and p.grad is not None and rp.grad is not None
            grad = p.grad.clone()
            dist.all_reduce(grad, group=pg.cp)
            _assert_close(grad, rp.grad, "layer." + name)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_causal_conv_halo(width):
    from causal_conv1d import causal_conv1d_fn

    from megatron.core.ssm.causal_conv1d import causal_conv1d_cp

    cp, rank = dist.get_world_size(), dist.get_rank()
    local = 16
    x = torch.zeros(1, cp * local, 8, device="cuda", dtype=torch.bfloat16)
    x[:, local - 3 : local] = torch.arange(1, 4, device="cuda").view(1, 3, 1)
    x[:, -3:] = torch.arange(4, 7, device="cuda").view(1, 3, 1)
    weight = torch.full((8, width), 0.25, device="cuda", dtype=torch.bfloat16)
    xr, wr = x.clone().requires_grad_(), weight.clone().requires_grad_()
    if width == 1:
        yr = F.silu(F.conv1d(xr.transpose(1, 2).float(), wr.float().unsqueeze(1), groups=8))
        yr = yr.transpose(1, 2).to(x.dtype)
    else:
        yr = causal_conv1d_fn(xr.transpose(1, 2), wr, activation="silu").transpose(1, 2)
    dy = torch.zeros_like(yr)
    dy[:, local : local + 3] = 1
    yr.backward(dy)
    sl = slice(rank * local, (rank + 1) * local)
    xx, ww = x[:, sl].clone().requires_grad_(), weight.clone().requires_grad_()
    yy = causal_conv1d_cp(xx, ww, None, "silu", dist.group.WORLD)
    yy.backward(dy[:, sl])
    _assert_close(yy, yr[:, sl], "conv.output")
    _assert_close(xx.grad, xr.grad[:, sl], "conv.dx")
    dist.all_reduce(ww.grad)
    _assert_close(ww.grad, wr.grad, "conv.dw")
    if width > 2:
        with pytest.raises(ValueError, match="at least"):
            causal_conv1d_cp(xx[:, :1], ww, None, "silu", dist.group.WORLD)


@pytest.mark.parametrize("sp", [False, True])
def test_tp2_cp2(singleton_group, sp):
    from megatron.core.process_groups_config import ProcessGroupCollection

    if dist.get_world_size() != 4:
        pytest.skip("TP2 CP2 requires four ranks")
    with _model_groups(tp=2) as pg:
        ref = _gdn(_config(1, tp=2, sp=sp), ProcessGroupCollection(tp=pg.tp, cp=singleton_group))
        model = _gdn(_config(2, tp=2, sp=sp), pg)
        model.load_state_dict(copy.deepcopy(ref.state_dict()), strict=True)
        torch.manual_seed(456)
        x = torch.randn(256, 1, 128, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x) / x.numel()
        ref_idx = torch.arange(256, device="cuda")
        if sp:
            ref_idx = ref_idx.chunk(2)[pg.tp.rank()]
        xr = x[ref_idx].clone().requires_grad_()
        yr, _ = ref(xr, attention_mask=None)
        yr.backward(dy[ref_idx])
        # Reconstruct the full-sequence TP-reference only for comparison.
        if sp:
            ys, dxs = [torch.empty_like(yr) for _ in range(2)], [
                torch.empty_like(xr) for _ in range(2)
            ]
            dist.all_gather(ys, yr.detach(), group=pg.tp)
            dist.all_gather(dxs, xr.grad, group=pg.tp)
            yr, dxr = torch.cat(ys), torch.cat(dxs)
        else:
            dxr = xr.grad
        idx = torch.arange(pg.cp.rank() * 128, (pg.cp.rank() + 1) * 128, device="cuda")
        if sp:
            idx = idx.chunk(2)[pg.tp.rank()]
        xx = x[idx].clone().requires_grad_()
        yy, _ = model(xx, attention_mask=None)
        yy.backward(dy[idx])
        _assert_close(yy, yr[idx], "tp.output")
        _assert_close(xx.grad, dxr[idx], "tp.dx")
        for (name, p), (_, rp) in zip(model.named_parameters(), ref.named_parameters()):
            grad = p.grad.clone()
            dist.all_reduce(grad, group=pg.cp)
            ref_grad = rp.grad.clone()
            if getattr(p, "sequence_parallel", False):
                # These replicated parameters are TP-reduced by gradient finalization.
                dist.all_reduce(grad, group=pg.tp)
                dist.all_reduce(ref_grad, group=pg.tp)
            _assert_close(grad, ref_grad, "tp." + name)


@pytest.mark.parametrize("packed", [False, True])
def test_gpt_mixed_stack(singleton_group, packed):
    """Actual GPT GDN-attention-GDN specs, residual layout and all gradients."""
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_transformer_layer_with_experimental_attention_variant_spec,
    )
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.spec_utils import build_module

    with _model_groups() as pg:
        cp, rank = pg.cp.size(), pg.cp.rank()
        cfg = _config(cp, num_layers=3, linear_attention_freq=[1, 0, 1])
        ref_cfg = _config(1, num_layers=3, linear_attention_freq=[1, 0, 1])

        def build(cfg, group):
            from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

            torch.manual_seed(123)
            model_parallel_cuda_manual_seed(123)
            specs = get_transformer_layer_with_experimental_attention_variant_spec(cfg)
            return (
                torch.nn.ModuleList(
                    [
                        build_module(spec, config=cfg, layer_number=i + 1, pg_collection=group)
                        for i, spec in enumerate(specs)
                    ]
                )
                .cuda()
                .bfloat16()
            )

        reference_pg = copy.copy(pg)
        reference_pg.cp = singleton_group
        reference = build(ref_cfg, reference_pg)
        model = build(cfg, pg)
        model.load_state_dict(copy.deepcopy(reference.state_dict()), strict=True)
        for parameter in reference.parameters():
            rank0_parameter = parameter.detach().clone()
            dist.broadcast(rank0_parameter, src=0)
            torch.testing.assert_close(parameter, rank0_parameter, atol=0, rtol=0)
        torch.manual_seed(456)
        length = cp * 128
        x = torch.randn(length, 1, 128, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x) / x.numel()
        params = None
        if packed:
            cu = torch.tensor([0, length // 4, length], device="cuda", dtype=torch.int32)
            params = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=length * 3 // 4,
                max_seqlen_kv=length * 3 // 4,
                total_tokens=length,
            )
            idx = torch.cat(
                [
                    torch.arange(start, stop, device="cuda")
                    .reshape(2 * cp, -1)[[rank, 2 * cp - rank - 1]]
                    .flatten()
                    for start, stop in zip(cu.tolist()[:-1], cu.tolist()[1:])
                ]
            )
        else:
            idx = (
                torch.arange(length, device="cuda")
                .reshape(2 * cp, -1)[[rank, 2 * cp - rank - 1]]
                .flatten()
            )

        def forward(layers, inputs, stages):
            for layer in layers:
                inputs, _ = layer(inputs, attention_mask=None, packed_seq_params=params)
                stages.append(inputs.detach())
            return inputs

        ref_stages, cp_stages = [], []
        ref_attn, cp_attn = {}, {}
        for i, (r, m) in enumerate(zip(reference, model)):
            r.self_attention.register_forward_hook(
                lambda module, inputs, output, i=i: ref_attn.__setitem__(i, output[0].detach())
            )
            m.self_attention.register_forward_hook(
                lambda module, inputs, output, i=i: cp_attn.__setitem__(i, output[0].detach())
            )
        xr = x.clone().requires_grad_()
        yr = forward(reference, xr, ref_stages)
        yr.backward(dy)
        xx = x[idx].clone().requires_grad_()
        yy = forward(model, xx, cp_stages)
        yy.backward(dy[idx])
        for i, (a, b) in enumerate(zip(cp_stages, ref_stages)):
            print(
                f"mixed stage {i}: max={(a.float()-b[idx].float()).abs().max().item()}; "
                f"attn max={(cp_attn[i].float()-ref_attn[i][idx].float()).abs().max().item()}",
                flush=True,
            )
        # CP softmax and BF16 residual additions already differ from CP=1 by
        # occasional ULPs. Measure the existing headwise path as a control;
        # keep the original tolerance for the additional chunkwise error.
        control_cfg = _config(
            cp, mode="headwise", layout="zigzag", num_layers=3, linear_attention_freq=[1, 0, 1]
        )
        control = build(control_cfg, pg)
        control.load_state_dict(copy.deepcopy(reference.state_dict()), strict=True)
        yc = forward(control, x[idx].clone().requires_grad_(), [])
        a, b, c = yy.detach().float(), yr[idx].detach().float(), yc.detach().float()
        tolerance = 5e-3 + 5e-3 * b.abs()
        error, control_error = (a - b).abs(), (c - b).abs()
        relative = (a - b).norm() / b.norm()
        print(
            f"mixed control: headwise max={control_error.max().item()}, "
            f"chunkwise max={error.max().item()}, relative_l2={relative.item()}, "
            f"headwise strict mismatches={(control_error > tolerance).sum().item()}",
            flush=True,
        )
        ok = torch.tensor(
            int(
                bool((error <= control_error + tolerance).all())
                and relative.item() < 0.01
                and ((c - b).norm() / b.norm()).item() < 0.01
            ),
            device=a.device,
        )
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        assert ok.item(), "Mixed-stack error exceeds the measured headwise CP error plus tolerance"
        _assert_close(xx.grad, xr.grad[idx], "mixed.dx")
        for (name, p), (_, rp) in zip(model.named_parameters(), reference.named_parameters()):
            grad = p.grad.clone()
            dist.all_reduce(grad, group=pg.cp)
            _assert_close(grad, rp.grad, "mixed." + name)


def test_packed_sample_isolation(singleton_group):
    from megatron.core.packed_seq_params import PackedSeqParams

    with _model_groups() as pg:
        cp, rank = pg.cp.size(), pg.cp.rank()
        length = cp * 128
        model = _gdn(_config(cp), pg)
        cu = torch.tensor([0, 40, length - 40, length], device="cuda", dtype=torch.int32)
        params = PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, total_tokens=length
        )
        torch.manual_seed(1234)
        x = torch.randn(length, 1, 128, device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(rank * 128, (rank + 1) * 128, device="cuda")
        local = x[indices].clone().requires_grad_()
        y, _ = model(local, attention_mask=None, packed_seq_params=params)
        mask = ((indices >= 40) & (indices < length - 40)).view(-1, 1, 1)
        y.backward(mask.expand_as(y).to(y.dtype) / y.numel())
        assert torch.count_nonzero(local.grad[~mask[:, 0, 0]]) == 0
        x[:40] += 5
        y2, _ = model(x[indices], attention_mask=None, packed_seq_params=params)
        torch.testing.assert_close(y2[mask[:, 0, 0]], y[mask[:, 0, 0]], atol=0, rtol=0)


def test_headwise_regression(singleton_group):
    from megatron.core.process_groups_config import ProcessGroupCollection

    with _model_groups() as pg:
        cp, rank = pg.cp.size(), pg.cp.rank()
        ref = _gdn(
            _config(1, mode="headwise", layout="zigzag"),
            ProcessGroupCollection(tp=pg.tp, cp=singleton_group),
        )
        model = _gdn(_config(cp, mode="headwise", layout="zigzag"), pg)
        model.load_state_dict(copy.deepcopy(ref.state_dict()), strict=True)
        torch.manual_seed(456)
        x = torch.randn(cp * 128, 1, 128, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x) / x.numel()
        xr = x.clone().requires_grad_()
        yr, _ = ref(xr, attention_mask=None)
        yr.backward(dy)
        idx = (
            torch.arange(len(x), device="cuda")
            .reshape(cp * 2, -1)[[rank, cp * 2 - rank - 1]]
            .flatten()
        )
        xx = x[idx].clone().requires_grad_()
        yy, _ = model(xx, attention_mask=None)
        yy.backward(dy[idx])
        _assert_close(yy, yr[idx], "headwise.output")
        _assert_close(xx.grad, xr.grad[idx], "headwise.dx")


def test_unsupported_fusion_is_rejected():
    with _model_groups() as pg:
        with pytest.raises(ValueError, match="pre-GDR fusion"):
            _gdn(_config(pg.cp.size(), gdn_pre_gated_delta_rule_fusion=True), pg)
        with pytest.raises(ValueError, match="equal key and value"):
            _gdn(_config(pg.cp.size(), linear_value_head_dim=64), pg)


def test_hybrid_packed_metadata(singleton_group, monkeypatch):
    """HybridStack builds packed metadata once and actually delivers it to GDN."""
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.ssm.gated_delta_net import gdn as gdn_module
    from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    with _model_groups() as pg:
        cp, rank = pg.cp.size(), pg.cp.rank()

        def build(config, groups):
            torch.manual_seed(123)
            model_parallel_cuda_manual_seed(123)
            return (
                hybrid_stack_spec.module(
                    config=config,
                    submodules=hybrid_stack_spec.submodules,
                    pg_collection=groups,
                    layer_config_list=[GDNLayerConfig.from_config(config) for _ in range(2)],
                    post_layer_norm=False,
                )
                .cuda()
                .bfloat16()
            )

        ref_pg = copy.copy(pg)
        ref_pg.cp = singleton_group
        reference = build(_config(1, num_layers=2, linear_attention_freq=[1, 1]), ref_pg)
        model = build(_config(cp, num_layers=2, linear_attention_freq=[1, 1]), pg)
        model.load_state_dict(copy.deepcopy(reference.state_dict()), strict=True)
        length = cp * 128
        cu = torch.tensor([0, 40, length - 40, length], device="cuda", dtype=torch.int32)
        params = PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, total_tokens=length
        )
        torch.manual_seed(456)
        x = torch.randn(length, 1, 128, device="cuda", dtype=torch.bfloat16)
        dy = torch.randn_like(x) / x.numel()
        xr = x.clone().requires_grad_()
        yr = reference(xr, attention_mask=None, packed_seq_params=params)
        yr.backward(dy)

        def no_per_layer_metadata(*args, **kwargs):
            raise AssertionError("HybridStack failed to pass its cached metadata to GDN")

        monkeypatch.setattr(gdn_module, "build_packed_sequence_cp_metadata", no_per_layer_metadata)
        sl = slice(rank * 128, (rank + 1) * 128)
        xx = x[sl].clone().requires_grad_()
        yy = model(xx, attention_mask=None, packed_seq_params=params)
        yy.backward(dy[sl])
        _assert_close(yy, yr[sl], "hybrid.output")
        _assert_close(xx.grad, xr.grad[sl], "hybrid.dx")
        for (name, p), (_, rp) in zip(model.named_parameters(), reference.named_parameters()):
            grad = p.grad.clone()
            dist.all_reduce(grad, group=pg.cp)
            _assert_close(grad, rp.grad, "hybrid." + name)


def _assert_close(actual, expected, label):
    a, b = actual.detach().float(), expected.detach().float()
    maximum = float((a - b).abs().max())
    relative = float((a - b).norm() / b.norm().clamp_min(1e-12))
    print(
        f"rank={dist.get_rank()} {label}: max_abs={maximum:.6g} relative_l2={relative:.6g}",
        flush=True,
    )
    elementwise_ok = torch.isclose(a, b, atol=5e-3, rtol=5e-3).all()
    relative_ok = relative < 0.01 if b.norm() > 1e-12 else maximum <= 1e-12
    passed = torch.tensor(int(bool(elementwise_ok) and relative_ok), device=a.device)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if not passed.item():
        # Make every rank fail together, rather than leaving peers in a later collective.
        torch.testing.assert_close(a, b, atol=5e-3, rtol=5e-3)
        assert relative_ok, f"{label}: relative L2 error {relative} exceeds 1%"
        raise AssertionError(f"{label}: comparison failed on another rank")


def _recurrent_reference(q, k, v, g, beta):
    """Independent FP32 token recurrence, with no chunked kernel or communication."""
    ratio = v.shape[2] // q.shape[2]
    q, k = q.float().repeat_interleave(ratio, 2), k.float().repeat_interleave(ratio, 2)
    state = torch.zeros(q.shape[0], v.shape[2], q.shape[-1], v.shape[-1], device=q.device)
    output = []
    for t in range(q.shape[1]):
        state = state * g[:, t].float().exp()[..., None, None]
        delta = v[:, t].float() - torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], delta * beta[:, t, :, None])
        output.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state) * q.shape[-1] ** -0.5)
    return torch.stack(output, dim=1)


@pytest.mark.parametrize(
    "local_length,h,hv,kdim,vdim",
    [
        (63, 2, 2, 32, 32),
        (64, 2, 4, 32, 32),
        (65, 2, 2, 64, 64),
        (128, 2, 4, 32, 32),
        (257, 2, 2, 32, 32),
        (64, 2, 2, 128, 128),
    ],
)
def test_backend_equivalence(local_length, h, hv, kdim, vdim):
    """GDP single-update CP equals GDN on a full sequence, including all gradients."""
    world, rank = dist.get_world_size(), dist.get_rank()
    length = local_length * world
    torch.manual_seed(123)
    device = torch.device("cuda", torch.cuda.current_device())
    q = F.normalize(torch.randn(1, length, h, kdim, device=device), dim=-1).bfloat16()
    k = F.normalize(torch.randn_like(q).float(), dim=-1).bfloat16()
    v = torch.randn(1, length, hv, vdim, device=device, dtype=torch.bfloat16)
    g = -torch.rand(1, length, hv, device=device) * 0.1
    beta = torch.rand_like(g)
    operands = [x.detach().requires_grad_() for x in (q, k, v, g, beta)]
    # Same global loss normalization on every rank. Keep a FP32 oracle to
    # distinguish BF16 re-chunking error from a changed delta-rule equation.
    dy = torch.randn_like(v) / v.numel()
    oracle_inputs = [x.detach().float().requires_grad_() for x in operands]
    oracle = _recurrent_reference(*oracle_inputs)
    oracle.backward(dy.float())
    expected, _ = chunk_gated_delta_rule(*operands, use_qk_l2norm_in_kernel=False)
    expected.backward(dy)
    sl = slice(rank * local_length, (rank + 1) * local_length)
    local = [x.detach()[:, sl].contiguous().requires_grad_() for x in operands]
    # GDN's preparation expands grouped Q/K before calling the core kernel.
    # GDP's output kernel expects that same equal-head contract.
    kernel_inputs = [
        local[0].repeat_interleave(hv // h, 2),
        local[1].repeat_interleave(hv // h, 2),
        *local[2:],
    ]
    actual = gdp_chunkwise_context_parallel(
        *kernel_inputs,
        cu_seqlens=None,
        num_householder=1,
        scale=kdim**-0.5,
        cp_group=dist.group.WORLD,
        backend=FLAGatedDeltaProductCPBackend(),
    )
    actual.backward(dy[:, sl].contiguous())
    _assert_close(actual, expected[:, sl], "backend.output")
    _assert_close(actual, oracle[:, sl], "backend.output.fp32")
    for name, tensor, ref, fp32 in zip(
        ("q", "k", "v", "g", "beta"), local, operands, oracle_inputs
    ):
        _assert_close(tensor.grad, ref.grad[:, sl], f"backend.d{name}")
        target = fp32.grad[:, sl]
        cp_error = (tensor.grad.float() - target).norm() / target.norm().clamp_min(1e-12)
        baseline_error = (ref.grad[:, sl].float() - target).norm() / target.norm().clamp_min(1e-12)
        print(
            f"FP32 gradient {name}: cp={cp_error.item():.6g}, baseline={baseline_error.item():.6g}",
            flush=True,
        )
        assert baseline_error < 0.01
        assert cp_error < 0.01
