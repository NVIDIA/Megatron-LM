# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer, rounding
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _bf16_neighbours(x):
    """The BF16 values just below and above each FP32 element (by magnitude), via truncation."""
    bits = x.view(torch.int32) & -(1 << 16)
    return bits.view(torch.float32), (bits + (1 << 16)).view(torch.float32)


def _round(x, offset=0, seed=1):
    out = torch.empty_like(x, dtype=torch.bfloat16)
    rounding.stochastic_round_to_bf16_(out, x.clone(), offset, seed)
    return out.float()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestStochasticRounding:

    @pytest.mark.parametrize("value", [0.025 + 1e-6, -0.0371, 1.0 + 2**-10, 3.0e-5])
    def test_unbiased_and_only_neighbours(self, value):
        x = torch.full((1 << 22,), value, dtype=torch.float32, device="cuda")
        y = _round(x)
        lo, hi = (v.item() for v in _bf16_neighbours(x[:1]))
        assert set(y.unique().tolist()) <= {lo, hi}
        expected_far = (abs(value) - abs(lo)) / (abs(hi) - abs(lo))
        assert abs((y == hi).double().mean().item() - expected_far) < 2e-3
        bias_in_ulp = (y.double().mean().item() - x[0].item()) / abs(hi - lo)
        assert abs(bias_in_ulp) < 2e-3

    def test_random_values_round_to_neighbours(self):
        x = torch.randn(1 << 20, device="cuda") * 0.03
        lo, hi = _bf16_neighbours(x)
        y = _round(x)
        assert bool(((y == lo) | (y == hi)).all())

    def test_special_values(self):
        tiny = torch.finfo(torch.float32).tiny
        x = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.0, -2.5, tiny, 1e-45],
            device="cuda",
        )
        for seed in range(64):
            y = _round(x, seed=seed)
            assert torch.isnan(y[0])
            assert y[1] == float("inf") and y[2] == -float("inf")
            assert y[3] == 0.0 and not torch.signbit(y[3]) and torch.signbit(y[4])
            assert y[5] == 1.0 and y[6] == -2.5
            assert torch.isfinite(y[7:]).all() and (y[7:] >= 0).all()

    def test_deterministic_seeded_and_chunk_invariant(self, monkeypatch):
        x = torch.randn(3_000_001, device="cuda")
        a, b = _round(x, seed=7), _round(x, seed=7)
        assert torch.equal(a, b)
        assert not torch.equal(a, _round(x, seed=8))
        # Small chunks and a CPU-resident source must give the same result as one shot.
        monkeypatch.setattr(rounding, "_CHUNK", 1 << 16)
        dst = torch.empty_like(x, dtype=torch.bfloat16)
        rounding.round(dst, x.cpu(), "stochastic", seed=7)
        assert torch.equal(dst.float(), a)

    def test_nearest_is_plain_copy(self):
        x = torch.randn(4097, device="cuda")
        dst = torch.empty_like(x, dtype=torch.bfloat16)
        rounding.round(dst, x, "nearest")
        assert torch.equal(dst, x.bfloat16())


@pytest.mark.parametrize(
    "src_dtype,dst_dtype", [(torch.float32, torch.float16), (torch.float64, torch.bfloat16)]
)
def test_stochastic_rejects_unsupported_casts(src_dtype, dst_dtype):
    src = torch.tensor([1.03127, -0.0113], dtype=src_dtype)
    dst = torch.zeros_like(src, dtype=dst_dtype)
    with pytest.raises(ValueError, match="Stochastic rounding only supports FP32 -> BF16 casts"):
        rounding.round(dst, src, "stochastic")
    assert torch.equal(dst, torch.zeros_like(dst))
    rounding.round(dst, src, "nearest")
    assert torch.equal(dst, src.to(dst_dtype))


def test_stochastic_allows_exact_fp32_copy():
    src = torch.tensor([-0.0, 1.03127, -0.0113], dtype=torch.float32)
    dst = torch.zeros_like(src)
    rounding.round(dst, src, "stochastic")
    assert torch.equal(dst.view(torch.int32), src.view(torch.int32))


def test_fmix32_scalar_and_tensor_match_in_place():
    words = [0, 1, 0x80000000, 0xFFFFFFFF]
    expected = [0, 0x514E28B7, 0x6D3C65A0, 0x81F16F39]
    tensor = torch.tensor(words, dtype=torch.int64)
    assert rounding._fmix32_(tensor) is tensor
    assert tensor.tolist() == expected
    assert [rounding._fmix32_(word) for word in words] == expected


class TestMainParamsRoundingConfig:

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="must be 'nearest' or 'stochastic'"):
            OptimizerConfig(main_params_rounding="up", use_distributed_optimizer=True).validate()

    def test_requires_distributed_optimizer(self):
        with pytest.raises(ValueError, match="requires the distributed optimizer"):
            OptimizerConfig(main_params_rounding="stochastic").validate()

    @pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
    def test_rejects_non_bf16_training(self, fp16):
        with pytest.raises(ValueError, match="requires BF16 training"):
            OptimizerConfig(
                main_params_rounding="stochastic", use_distributed_optimizer=True, fp16=fp16
            ).validate()

    def test_rejects_precision_aware_without_offload(self):
        with pytest.raises(ValueError, match="precision-aware"):
            OptimizerConfig(
                main_params_rounding="stochastic",
                bf16=True,
                use_distributed_optimizer=True,
                use_precision_aware_optimizer=True,
            ).validate()

    def test_accepts_supported_modes(self):
        OptimizerConfig(
            main_params_rounding="stochastic", bf16=True, use_distributed_optimizer=True
        ).validate()
        OptimizerConfig(
            main_params_rounding="stochastic",
            bf16=True,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=True,
            optimizer_cpu_offload=True,
        ).validate()


def _train(offload_fraction, rounding, steps):
    """Train a small BF16 model with DDP and the distributed optimizer."""
    torch.manual_seed(0)
    module = torch.nn.Sequential(
        torch.nn.Linear(1024, 2048, bias=False),
        torch.nn.LayerNorm(2048),
        torch.nn.Linear(2048, 512, bias=False),
    )
    module = module.cuda().bfloat16()
    with torch.no_grad():
        module[1].weight.fill_(1.0)  # |w| ~ 1 like a norm weight: half a BF16 ULP is 2**-8
    ddp = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=True),
        module,
    )
    kwargs = dict(
        optimizer="adam",
        lr=2e-6,
        bf16=True,
        use_distributed_optimizer=True,
        main_params_rounding=rounding,
        main_params_rounding_seed=1234,
    )
    if offload_fraction is not None:
        kwargs.update(
            optimizer_cpu_offload=True,
            optimizer_offload_fraction=offload_fraction,
            use_precision_aware_optimizer=True,
            overlap_cpu_optimizer_d2h_h2d=True,
        )
    optimizer = get_megatron_optimizer(OptimizerConfig(**kwargs), [ddp])
    params = list(ddp.module.parameters())
    initial = [p.detach().clone() for p in params]
    torch.manual_seed(1)
    for _ in range(steps):
        x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        ddp(x).float().pow(2).mean().backward()
        ddp.finish_grad_sync()
        assert optimizer.step()[0]
    torch.cuda.synchronize()
    return optimizer, params, initial


def _model_and_main_param_shards(optimizer):
    """Pair each locally owned BF16 model slice with its FP32 master shard."""
    inner = (
        optimizer.chained_optimizers[0] if hasattr(optimizer, "chained_optimizers") else optimizer
    )
    assert inner.model_param_group_index_map
    for param in inner.model_param_group_index_map:
        span = inner._get_model_param_range_map(param)["param"]
        model_shard = param.detach().view(-1)[span.start : span.end]
        main_shard = inner._get_main_param_and_optimizer_states(param)["param"].detach()
        assert main_shard.dtype == torch.float32
        assert main_shard.shape == model_shard.shape
        yield model_shard, main_shard.to(model_shard.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestMainParamsRoundingOptimizer:

    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("offload_fraction", [None, 0.0, 0.5, 1.0])
    def test_nearest_matches_cast(self, offload_fraction):
        optimizer, _, _ = _train(offload_fraction, "nearest", steps=3)
        for model_shard, main_shard in _model_and_main_param_shards(optimizer):
            assert torch.equal(model_shard, main_shard.bfloat16())

    @pytest.mark.parametrize("offload_fraction", [None, 0.0, 0.5, 1.0])
    def test_stochastic_rounds_to_neighbours_and_moves_weights(self, offload_fraction):
        optimizer, params, initial = _train(offload_fraction, "stochastic", steps=3)
        for model_shard, main_shard in _model_and_main_param_shards(optimizer):
            lo, hi = _bf16_neighbours(main_shard)
            value = model_shard.float()
            assert bool(((value == lo) | (value == hi)).all())
        _, nearest_params, _ = _train(offload_fraction, "nearest", steps=3)
        moved = sum(int((p != i).sum()) for p, i in zip(params, initial))
        moved_nearest = sum(int((p != i).sum()) for p, i in zip(nearest_params, initial))
        assert moved > moved_nearest
