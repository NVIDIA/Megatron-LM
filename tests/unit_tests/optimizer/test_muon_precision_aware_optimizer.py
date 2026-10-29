# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon with the precision-aware optimizer.

The precision-aware flag applies to the scalar Adam next to Muon; with
``muon_store_param_remainders`` Muon also keeps BF16 params with int16 remainders instead of
FP32 main params. Sections, in order:

- The remainder encoding: split / merge round-trips every FP32 bit pattern, and the fused
  update kernel matches the FP32 ``add_`` ops it replaces bit for bit.
- Config guards for ``muon_store_param_remainders`` and the precision-aware optimizer
  under Muon.
- End to end through ``get_megatron_optimizer``: layer-sharded Muon, with or without
  remainders, keeps the same FP32 main params as without the precision-aware optimizer, step
  after step, while the scalar Adam params go through a precision-aware DistributedOptimizer.

Launch: torchrun --nproc-per-node=<n> -m pytest
    tests/unit_tests/optimizer/test_muon_precision_aware_optimizer.py
"""

import pytest
import torch
import torch.nn as nn

from megatron.core.optimizer.param_remainders import (
    apply_update_with_remainder,
    merge_main_param,
    split_main_param,
)

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _fp32_samples(device, n=1 << 16):
    """Random FP32 values over many magnitudes plus every low-half pattern at ties and
    signed zeros, infinities and subnormals."""
    torch.manual_seed(0)
    values = torch.randn(n, device=device) * torch.logspace(-30, 30, n, device=device)
    ties = (torch.randint(0, 1 << 15, (1024,), device=device, dtype=torch.int32) << 16) | 0x8000
    specials = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), 1e-40, -1e-40, 3.4e38, -3.4e38], device=device
    )
    return torch.cat([values, ties.view(torch.float32), -ties.view(torch.float32), specials])


def _split(main):
    param = torch.empty_like(main, dtype=torch.bfloat16)
    remainder = torch.empty_like(main, dtype=torch.int16)
    split_main_param(main, param, remainder)
    return param, remainder


@pytest.mark.parametrize("device", _DEVICES)
def test_split_merge_round_trips_every_bit(device):
    main = _fp32_samples(device)
    param, remainder = _split(main)
    assert torch.equal(merge_main_param(param, remainder).view(torch.int32), main.view(torch.int32))


@pytest.mark.parametrize("device", _DEVICES)
def test_bf16_half_matches_round_to_nearest_off_ties(device):
    """The BF16 half is the FP32 value rounded half away from zero: the same as PyTorch's
    round-to-nearest-even cast except on exact ties, where it may be one ulp further out."""
    main = _fp32_samples(device)
    param, _ = _split(main)
    finite = torch.isfinite(main) & torch.isfinite(main.bfloat16().float())
    tie = (main.view(torch.int32) & 0xFFFF) == 0x8000
    off_tie = finite & ~tie
    assert torch.equal(param[off_tie].float(), main.bfloat16()[off_tie].float())
    finite_tie = finite & tie
    assert bool((param[finite_tie].float().abs() > main[finite_tie].abs()).all())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("weight_decay_alpha", [None, -1e-3], ids=["no_wd", "wd"])
def test_fused_update_matches_fp32_add_ops(device, weight_decay_alpha):
    torch.manual_seed(1)
    main = torch.randn(3, 4099, device=device)  # not a multiple of the kernel block
    update = torch.randn(3, 4099, device=device)
    update_alpha = -3e-4

    reference = main.clone()
    if weight_decay_alpha is not None:
        reference.add_(reference, alpha=weight_decay_alpha)
    reference.add_(update, alpha=update_alpha)

    param, remainder = _split(main)
    apply_update_with_remainder(
        param, remainder, update, update_alpha=update_alpha, weight_decay_alpha=weight_decay_alpha
    )
    merged = merge_main_param(param, remainder)
    assert torch.equal(
        merged.view(torch.int32), reference.view(torch.int32)
    ), f"max |diff| {(merged - reference).abs().max().item():.3e}"


# ---------------------------------------------------------------------------
# Config guards
# ---------------------------------------------------------------------------

_MUON_CONFIG = dict(
    optimizer='muon',
    bf16=True,
    use_layer_wise_distributed_optimizer=True,
    muon_tp_mode='layer_sharded',
    muon_split_qkv=False,
)


def _config(**kwargs):
    pytest.importorskip("transformer_engine", reason="precision-aware optimizer needs TE")
    from megatron.core.optimizer import OptimizerConfig

    return OptimizerConfig(**{**_MUON_CONFIG, **kwargs})


def test_precision_aware_optimizer_accepted_with_muon():
    config = _config(use_precision_aware_optimizer=True, exp_avg_dtype=torch.bfloat16)
    plain = config.without_precision_aware_optimizer()
    assert not plain.use_precision_aware_optimizer
    assert not plain.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    assert plain.exp_avg_dtype == torch.float32
    assert config.use_precision_aware_optimizer, "the copy must not touch the original"


def test_precision_aware_optimizer_rejected_with_lion_scalar_optimizer():
    with pytest.raises(AssertionError, match="scalar optimizer is adam"):
        _config(use_precision_aware_optimizer=True, muon_scalar_optimizer='lion')


def test_optimizer_cuda_graph_keeps_full_fp32_adam_master_params():
    """TE FusedAdam cannot store param remainders in capturable mode."""
    assert _config(use_precision_aware_optimizer=True).store_param_remainders
    config = _config(use_precision_aware_optimizer=True, optimizer_cuda_graph=True)
    assert not config.store_param_remainders


def test_param_remainders_need_precision_aware_optimizer():
    with pytest.raises(ValueError, match="use_precision_aware_optimizer"):
        _config(muon_store_param_remainders=True)


def test_param_remainders_need_layer_sharded_muon():
    with pytest.raises(ValueError, match="layer_sharded"):
        _config(
            use_precision_aware_optimizer=True,
            muon_store_param_remainders=True,
            muon_tp_mode='duplicated',
        )


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class _Model(nn.Module):
    """Five matrices for Muon (two of one shape, for batched NS) plus biases for Adam. ``fc4``
    is kept in FP32 (as e.g. routers are), so Muon also steps an FP32 param."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 48)
        self.fc3 = nn.Linear(48, 48)
        self.fc4 = nn.Linear(48, 24)
        self.fc5 = nn.Linear(24, 10)


@pytest.fixture
def _model_parallel():
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _build(precision_aware, remainders, ns_batch_size, copy_from=None):
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer import TransformerConfig
    from megatron.training.training import wrap_model_chunks_with_ddp

    model = _Model().bfloat16().cuda()
    model.fc4.float()
    model = wrap_model_chunks_with_ddp(
        [model],
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(),
        use_layer_wise_distributed_optimizer=True,
    )[0]
    if copy_from is None:
        model.broadcast_params()
    else:
        model.module.load_state_dict(copy_from.module.state_dict())

    config = OptimizerConfig(
        **_MUON_CONFIG,
        lr=1e-2,
        weight_decay=0.1,
        clip_grad=1.0,
        use_distributed_optimizer=False,
        muon_ns_batch_size=ns_batch_size,
        use_precision_aware_optimizer=precision_aware,
        muon_store_param_remainders=remainders,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
    pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()
    optimizer = get_megatron_optimizer(
        config, [model], pg_collection=pg_collection, use_gloo_process_groups=False
    )
    return model, optimizer


def _muon_wrappers(optimizer):
    """The Float16 wrappers inside the LayerWise optimizer (first child of the chain)."""
    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

    layer_wise = optimizer.chained_optimizers[0]
    assert isinstance(layer_wise, LayerWiseDistributedOptimizer)
    return layer_wise.chained_optimizers


def _main_params(optimizer):
    """id(model param) -> FP32 main value of every Muon param this rank updates."""
    from megatron.core.optimizer.param_remainders import Float16OptimizerWithParamRemainders

    mains = {}
    for wrapper in _muon_wrappers(optimizer):
        if isinstance(wrapper, Float16OptimizerWithParamRemainders):
            for params, remainders in zip(wrapper.float16_groups, wrapper.remainder_groups):
                for param, remainder in zip(params, remainders):
                    mains[id(param)] = merge_main_param(param, remainder)
        else:
            for params, main_group in zip(wrapper.float16_groups, wrapper.fp32_from_float16_groups):
                for param, main_param in zip(params, main_group):
                    mains[id(param)] = main_param.detach().clone()
        for group in wrapper.fp32_from_fp32_groups:
            for param in group:
                mains[id(param)] = param.detach().clone()
    return mains


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("remainders", [False, True], ids=["fp32_main", "remainders"])
@pytest.mark.parametrize("ns_batch_size", [1, 2], ids=["ns_batch1", "ns_batch2"])
def test_muon_keeps_fp32_main_params_bitwise(_model_parallel, ns_batch_size, remainders):
    pytest.importorskip("emerging_optimizers", reason="LayerShardedMuon needs emerging-optimizers")
    pytest.importorskip("transformer_engine", reason="precision-aware optimizer needs TE")
    from megatron.core.optimizer.param_remainders import Float16OptimizerWithParamRemainders

    reference_model, reference = _build(False, False, ns_batch_size)
    model, optimizer = _build(True, remainders, ns_batch_size, copy_from=reference_model)
    assert all(
        isinstance(w, Float16OptimizerWithParamRemainders) == remainders
        for w in _muon_wrappers(optimizer)
    )
    # The scalar Adam is precision-aware, Muon's side is not unless it stores remainders.
    assert optimizer.chained_optimizers[1].config.use_precision_aware_optimizer
    assert _muon_wrappers(optimizer)[0].config.use_precision_aware_optimizer == remainders

    names = [name for name, _ in model.module.named_parameters()]
    params = dict(model.module.named_parameters())
    reference_params = dict(reference_model.module.named_parameters())
    torch.manual_seed(2)
    for step in range(3):
        for name in names:
            grad = torch.randn_like(params[name], dtype=torch.float32)
            torch.distributed.broadcast(grad, src=0)
            params[name].main_grad = grad.clone()
            reference_params[name].main_grad = grad.clone()

        _, grad_norm, _ = optimizer.step()
        _, reference_grad_norm, _ = reference.step()
        assert float(grad_norm) == pytest.approx(float(reference_grad_norm), rel=1e-6)

        mains, reference_mains = _main_params(optimizer), _main_params(reference)
        by_name = {id(params[n]): n for n in names}
        reference_by_name = {n: id(reference_params[n]) for n in names}
        assert mains, "this rank owns no Muon params"
        for param_id, main in mains.items():
            name = by_name[param_id]
            reference_main = reference_mains[reference_by_name[name]]
            assert torch.equal(
                main.view(torch.int32), reference_main.view(torch.int32)
            ), f"step {step} {name}: max |diff| {(main - reference_main).abs().max().item():.3e}"

        # Every rank sees every updated BF16 param after the param sync; Muon's equal the
        # reference off ties, the precision-aware Adam biases stay close.
        for name in names:
            torch.testing.assert_close(
                params[name].float(), reference_params[name].float(), atol=1e-2, rtol=1e-2
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_remainder_checkpoint_carries_fp32_main_params(_model_parallel):
    pytest.importorskip("emerging_optimizers", reason="LayerShardedMuon needs emerging-optimizers")
    pytest.importorskip("transformer_engine", reason="precision-aware optimizer needs TE")
    model, optimizer = _build(True, True, ns_batch_size=1)
    for param in model.module.parameters():
        param.main_grad = torch.randn_like(param, dtype=torch.float32)
    optimizer.step()

    for wrapper in _muon_wrappers(optimizer):
        state_dict = wrapper.state_dict()
        saved = [[t.clone() for t in group] for group in state_dict['fp32_from_fp16_params']]
        for params, remainders, saved_group in zip(
            wrapper.float16_groups, wrapper.remainder_groups, saved
        ):
            for param, remainder, main in zip(params, remainders, saved_group):
                assert main.dtype == torch.float32
                assert torch.equal(merge_main_param(param, remainder), main)
                param.data.zero_()
                remainder.zero_()
        wrapper.load_state_dict(state_dict)
        for params, remainders, saved_group in zip(
            wrapper.float16_groups, wrapper.remainder_groups, saved
        ):
            for param, remainder, main in zip(params, remainders, saved_group):
                assert torch.equal(merge_main_param(param, remainder), main)
