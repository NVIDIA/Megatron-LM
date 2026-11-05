# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real native runtime-CP transitions for DSv4; run on eight Blackwell GPUs."""

import hashlib
import json
from copy import copy
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from megatron.core.dynamic_cp_group import LogicalCPGroup
from megatron.core.extensions.transformer_engine import HAVE_TE, TELinear
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    HAVE_HADAMARD,
    _build_attention,
    _make_config,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_packed_cp import (
    _CP1,
    _compare_results,
    _packed_real_rows,
)

pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and HAVE_TE and HAVE_HADAMARD),
        reason="requires CUDA, Transformer Engine and the real Hadamard kernel",
    ),
]

_SINGLETONS = (1,) * 8
_LOCAL_TOKENS = 512


@pytest.fixture
def native_runtime_parent():
    """Create no runtime subgroup PGs; all logical groups share the WORLD parent."""
    Utils.initialize_distributed()
    if dist.get_world_size() != 8:
        pytest.skip("mixed CP3/CP5 layouts require exactly eight GPUs")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("the fused CSA test kernels require Blackwell")
    pytest.importorskip("flash_mla")
    pytest.importorskip("cudnn.deepseek_sparse_attention")
    transport = pytest.importorskip("transformer_engine.pytorch.attention.native_cp_transport")
    Utils.initialize_model_parallel(context_parallel_size=1)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    parent = pg.dp_cp
    try:
        dist.barrier(group=parent, device_ids=[torch.cuda.current_device()])
        reason = transport.get_native_cp_transport_unavailable_reason(parent)
        if reason:
            pytest.skip(reason)
        # Exercise the production constructor's fixed halo-sized arena rather
        # than masking staging-size bugs with an oversized test allocation.
        yield parent, transport, pg
    finally:
        torch.cuda.synchronize()
        dist.barrier(device_ids=[torch.cuda.current_device()])
        if transport.get_native_cp_transport(parent) is not None:
            transport.destroy_native_cp_transport(parent)
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        torch.compiler.reset()
        Utils.destroy_model_parallel()


def _fingerprint(*tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(
            tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        )
    return digest.hexdigest()


def _authenticate(parent, group, *tensors):
    record = (group.ranks, _fingerprint(*tensors))
    gathered = [None] * parent.size()
    dist.all_gather_object(gathered, record, group=parent)
    assert all(gathered[rank] == record for rank in group.ranks), gathered


def _logical_group(parent, transport, layout):
    ranks = dist.get_process_group_ranks(parent)
    start = 0
    for size in layout:
        if start <= parent.rank() < start + size:
            group = LogicalCPGroup.from_parent_interval(ranks, start, size, dist.get_rank())
            if size > 1:
                transport.set_native_cp_parent_group(group, parent)
            return group
        start += size
    raise AssertionError(f"layout {layout} does not cover the parent")


def _packed_case(group, *, static_slots):
    """Reproduce ceil(2*WORLD), ceil(2*CP), extend-last physical storage."""
    cp = group.size()
    total = cp * _LOCAL_TOKENS

    def stored(length):
        padded = (length + 15) // 16 * 16
        return (padded + 2 * cp - 1) // (2 * cp) * (2 * cp)

    middle = min(1012, total - 384)
    remaining = total - stored(129) - stored(middle)
    logical = [129, 0, middle, remaining // 16 * 16 - 7]
    physical = [stored(length) if length else 0 for length in logical]
    assert sum(physical) <= total
    physical[-1] += total - sum(physical)
    logical_cu, physical_cu = [0], [0]
    for real, padded in zip(logical, physical):
        logical_cu.append(logical_cu[-1] + real)
        physical_cu.append(physical_cu[-1] + padded)
    if static_slots:
        logical_cu.extend([logical_cu[-1]] * (65 - len(logical_cu)))
        physical_cu.extend([physical_cu[-1]] * (65 - len(physical_cu)))
    real = torch.tensor(logical_cu, dtype=torch.int32, device="cuda")
    padded = torch.tensor(physical_cu, dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=real,
        cu_seqlens_kv=real,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded,
        max_seqlen_q=max(physical),
        max_seqlen_kv=max(physical),
        local_cp_size=cp,
        cp_group=group if cp > 1 else None,
        cp_singleton_group=group if cp == 1 else None,
        cp_partition_mode="contiguous",
    )
    generator = torch.Generator(device="cuda").manual_seed(191 + min(group.ranks))
    whole = torch.randn(total, 1, 256, dtype=torch.bfloat16, device="cuda", generator=generator)
    gradient = torch.randn(whole.shape, dtype=whole.dtype, device="cuda", generator=generator)
    rows = slice(group.rank() * _LOCAL_TOKENS, (group.rank() + 1) * _LOCAL_TOKENS)
    return packed, whole, gradient, rows


def _build_models(pg, *, ratio, coeff, recompute, static_slots):
    # The TE projections use CPU initialization: seed both CPU and CUDA explicitly.
    torch.manual_seed(198)
    model_parallel_cuda_manual_seed(198)
    config = _make_config(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=64,
        v_head_dim=512,
        qk_pos_emb_head_dim=64,
        q_lora_rank=128,
        csa_compress_ratios=[ratio],
        csa_window_size=128,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
        dsa_indexer_loss_coeff=coeff,
        dsa_indexer_use_sparse_loss=True,
        dsa_kernel_backend="cudnn",
        context_parallel_size=1,
        dynamic_context_parallel=True,
        use_native_cp_transport=True,
        sequence_packing_scheduler="default_dynamic_cp",
        max_seqlen_per_dp_cp_rank=_LOCAL_TOKENS,
        cp_partition_mode="contiguous",
        attention_cp_layout="contiguous",
        linear_cp_layout="contiguous",
        thd_max_packed_sequences=64 if static_slots else None,
        qk_layernorm=True,
        apply_rope_fusion=True,
        gradient_accumulation_fusion=True,
        recompute_granularity="selective" if recompute else None,
        recompute_modules=["mla_up_proj"] if recompute else [],
    )
    model = _build_attention(config, 1, pg).cuda()
    ref_pg = copy(pg)
    ref_pg.cp = _CP1()
    reference = _build_attention(
        replace(config, dynamic_context_parallel=False, use_native_cp_transport=False), 1, ref_pg
    ).cuda()
    reference.load_state_dict(model.state_dict())
    weights = _fingerprint(*(parameter for parameter in model.parameters()))
    all_weights = [None] * dist.get_world_size()
    dist.all_gather_object(all_weights, weights)
    assert len(set(all_weights)) == 1, all_weights
    return model, reference


def _clear_grads(model):
    model.zero_grad(set_to_none=True)
    for module in model.modules():
        if isinstance(module, TELinear) and module.fuse_wgrad_accumulation:
            if getattr(module.weight, "main_grad", None) is None:
                module.weight.main_grad = torch.zeros_like(module.weight, dtype=torch.float32)
            else:
                module.weight.main_grad.zero_()


def _forward(model, hidden, packed, group):
    _clear_grads(model)
    hidden = hidden.detach().clone().requires_grad_()
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
    output, _ = model(hidden, attention_mask=None, packed_seq_params=packed)
    assert torch.isfinite(output).all()
    valid = _packed_real_rows(packed, hidden.shape[0], group)
    output = output.masked_fill(~valid[:, None, None], 0)
    values = DSAIndexerLossLoggingHelper.tracker.get("values")
    loss = values.detach().float().sum().clone() if values is not None else output.new_zeros(())
    return hidden, output, valid, loss


def _backward(model, forward, gradient, group):
    hidden, output, valid, loss = forward
    _clear_grads(model)
    output.backward(gradient)
    assert torch.count_nonzero(hidden.grad[~valid]) == 0, "Padding must not affect valid tokens"
    fused = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, TELinear) and module.fuse_wgrad_accumulation
    }
    result = {"output": output.detach(), "input_grad": hidden.grad.detach()}
    for name, parameter in model.named_parameters():
        value = parameter.main_grad if id(parameter) in fused else parameter.grad
        assert value is not None, f"Missing gradient for {name}"
        value = value.detach().to(dtype=torch.float32, copy=True)
        result[f"param:{name}"] = reduce_from_tensor_model_parallel_region(value, group)
    return result, reduce_from_tensor_model_parallel_region(loss, group)


@pytest.mark.parametrize(
    "ratio,coeff,recompute,static_slots",
    [
        (0, 0.0, False, False),
        (4, 0.0, False, False),
        (4, 0.2, False, True),
        (4, 0.2, True, True),
        (128, 0.0, False, False),
        (128, 0.0, True, True),
    ],
)
@pytest.mark.parametrize("interleaved", [False, True])
def test_native_runtime_cp_layer_matches_cp1(
    native_runtime_parent, ratio, coeff, recompute, static_slots, interleaved
):
    """Same layer switches groups; overlapping forwards must retain their own CP context."""
    parent, transport, pg = native_runtime_parent
    model, reference = _build_models(
        pg, ratio=ratio, coeff=coeff, recompute=recompute, static_slots=static_slots
    )
    failures = []
    layouts = (_SINGLETONS, (3, 5), (5, 3), _SINGLETONS)
    pending = []
    for index, layout in enumerate(layouts):
        group = _logical_group(parent, transport, layout)
        packed, whole, gradient, rows = _packed_case(group, static_slots=static_slots)
        _authenticate(
            parent, group, whole, gradient, packed.cu_seqlens_q, packed.cu_seqlens_q_padded
        )
        actual = _forward(model, whole[rows], packed, group)
        expected = _forward(
            reference,
            whole,
            # The reference uses its explicit build-time CP1 group; do not
            # fabricate runtime CP1 metadata without a singleton descriptor.
            replace(packed, local_cp_size=None, cp_group=None, cp_singleton_group=None),
            _CP1(),
        )
        pending.append((index, group, gradient, rows, actual, expected))
        if interleaved and index == 1:
            continue
        for step, cp_group, upstream, selection, candidate, control in reversed(pending):
            result, loss = _backward(model, candidate, upstream[selection], cp_group)
            expected_result, expected_loss = _backward(reference, control, upstream, _CP1())
            label = (
                f"ratio={ratio}:step={step}:group={cp_group.ranks}:"
                f"recompute={recompute}:static65={static_slots}:interleaved={interleaved}"
            )
            _compare_results(result, expected_result, label, failures, selection)
            if coeff:
                print(
                    "DSV4_DYNAMIC_CP_LOSS "
                    + json.dumps(
                        dict(comparison=label, actual=loss.item(), expected=expected_loss.item())
                    ),
                    flush=True,
                )
                try:
                    torch.testing.assert_close(loss, expected_loss, rtol=1e-3, atol=1e-4)
                except AssertionError as error:
                    failures.append(f"{label}/indexer_loss: {error}")
        pending.clear()
    assert not failures, "\n\n".join(failures)


def test_runtime_group_restored_when_attention_raises(native_runtime_parent, monkeypatch):
    """Runtime binding must not corrupt the build-time group on exceptional exits."""
    parent, transport, pg = native_runtime_parent
    model, _ = _build_models(pg, ratio=4, coeff=0.2, recompute=False, static_slots=False)
    before = [
        (module, module.pg_collection.cp)
        for module in model.modules()
        if hasattr(getattr(module, "pg_collection", None), "cp")
    ]
    group = _logical_group(parent, transport, (3, 5))
    packed, whole, _, rows = _packed_case(group, static_slots=False)

    def fail(*args, **kwargs):
        raise RuntimeError("intentional runtime CP attention failure")

    monkeypatch.setattr(model.core_attention, "forward", fail)
    with pytest.raises(RuntimeError, match="intentional runtime CP attention failure"):
        model(whole[rows], attention_mask=None, packed_seq_params=packed)
    assert all(module.pg_collection.cp is original for module, original in before)
