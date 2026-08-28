# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts for the vLLM deployment block-FP8 adapter."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from megatron.lite.model.deepseek_v4.lite.resync import export_resync_weights
import megatron.lite.model.deepseek_v4.vllm.primitive.block_fp8 as deployment_fp8
from megatron.lite.model.deepseek_v4.vllm.primitive.block_fp8 import (
    BLOCK_SHAPE,
    DeploymentBlockFP8Adapter,
    DeploymentFusedBlockFP8Adapter,
    DeploymentGroupedBlockFP8Adapter,
    bind_source_scale_to_visible_weight,
    fp8_gemm_nt,
    pack_block_fp8_activation,
    pack_block_fp8_weight,
    pack_grouped_block_fp8_weight,
)


@pytest.fixture
def fake_vllm(monkeypatch):
    calls: list[tuple] = []

    deep_gemm = ModuleType("vllm.utils.deep_gemm")

    def per_block_cast_to_fp8(x, block_size, use_ue8m0):
        calls.append(("weight_quant", x, block_size, use_ue8m0))
        qweight = x.float().clamp(-1, 1).to(torch.float8_e4m3fn)
        scales = torch.ones(
            x.shape[0] // 128,
            x.shape[1] // 128,
            dtype=torch.float32,
            device=x.device,
        )
        return qweight, scales

    deep_gemm.per_block_cast_to_fp8 = per_block_cast_to_fp8
    class FakeScaleFormat:
        @classmethod
        def from_oracle(cls):
            return SimpleNamespace(name="UE8M0")

    deep_gemm.DeepGemmQuantScaleFMT = FakeScaleFormat

    def gemm_op(
        activation,
        weight,
        output,
        *,
        is_deep_gemm_e8m0_used,
    ):
        qinput, input_scale = activation
        qweight, weight_scale = weight
        calls.append(
            (
                "gemm",
                qinput,
                input_scale,
                qweight,
                weight_scale,
                output,
                is_deep_gemm_e8m0_used,
            )
        )
        output.fill_(3)

    deep_gemm.fp8_gemm_nt = gemm_op

    fp8_utils = ModuleType(
        "vllm.model_executor.layers.quantization.utils.fp8_utils"
    )

    def post_process(*, wq, ws, quant_block_shape, use_e8m0):
        calls.append(
            ("weight_postprocess", wq, ws, quant_block_shape, use_e8m0)
        )
        return wq, ws.to(torch.int32)

    def activation_quant(x, group_size, use_ue8m0):
        calls.append(("activation_quant", x, group_size, use_ue8m0))
        qactivation = x.float().clamp(-1, 1).to(torch.float8_e4m3fn)
        packed_k = (x.shape[1] // group_size + 3) // 4
        scales = torch.ones(
            x.shape[0], packed_k, dtype=torch.int32, device=x.device
        )
        return qactivation, scales

    fp8_utils.deepgemm_post_process_fp8_weight_block = post_process
    fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm = activation_quant

    monkeypatch.setitem(sys.modules, "vllm.utils.deep_gemm", deep_gemm)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.quantization.utils.fp8_utils",
        fp8_utils,
    )

    return calls


def _weight() -> nn.Parameter:
    return nn.Parameter(torch.randn(128, 256, dtype=torch.bfloat16))


@pytest.mark.gpus(1)
@pytest.mark.parametrize("export_storage_dtype", [torch.bfloat16, torch.float32])
def test_post_update_actor_pack_matches_export_reload_bitwise(
    export_storage_dtype: torch.dtype,
) -> None:
    """Close the actor-forward versus rollout-reload FP8 lifecycle boundary."""
    if not torch.cuda.is_available():
        pytest.skip("requires one CUDA GPU")

    # Post-optimizer values no longer carry checkpoint source scales.  Include
    # values close to block extrema so this compares the real quantizers, not
    # merely their nominal scale formula.
    values = torch.linspace(
        -1.00390625,
        1.00390625,
        128 * 256,
        device="cuda",
        dtype=torch.float32,
    ).reshape(128, 256)
    visible = nn.Parameter(values.to(torch.bfloat16))
    actor = pack_block_fp8_weight(visible)

    config = SimpleNamespace(
        expert_dtype="fp8",
        quantization_config={"weight_block_size": [128, 128]},
    )
    name = "layers.0.ffn.experts.0.w1.weight"
    exported = dict(
        export_resync_weights(
            [(name, visible.detach().to(export_storage_dtype))],
            config,
            resync_config={"expert_dtype": "fp8"},
        )
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )

    rollout_qweight, rollout_scales = deepgemm_post_process_fp8_weight_block(
        wq=exported[name],
        ws=exported["layers.0.ffn.experts.0.w1.scale"],
        quant_block_shape=BLOCK_SHAPE,
        use_e8m0=True,
    )

    assert torch.equal(actor.qweight, rollout_qweight)
    assert torch.equal(actor.scales, rollout_scales)


def test_weight_path_calls_vllm_and_packs_official_layout(fake_vllm) -> None:
    master = _weight()
    packed = pack_block_fp8_weight(master)

    assert [call[0] for call in fake_vllm] == [
        "weight_quant",
        "weight_postprocess",
    ]
    quant_call, post_call = fake_vllm
    assert quant_call[1] is not master
    assert quant_call[1].data_ptr() == master.data_ptr()
    assert quant_call[2:] == ([128, 128], False)
    assert post_call[3:] == (BLOCK_SHAPE, True)
    assert packed.qweight.dtype == torch.float8_e4m3fn
    assert packed.qweight.shape == master.shape
    assert packed.scales.dtype == torch.int32
    assert packed.cache_key == (
        id(master), master._version, master.device, master.dtype, tuple(master.shape)
    )


def test_fixed_scale_requantization_reconstructs_fp8_bitwise() -> None:
    qweight = torch.randn(128, 256).clamp(-4, 4).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.125, 0.25]], dtype=torch.float32)
    expanded = scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    master = (qweight.float() * expanded).to(torch.bfloat16)

    reconstructed = deployment_fp8.requantize_block_fp8_weight(master, scales)

    assert torch.equal(reconstructed.qweight, qweight)
    assert reconstructed.scales is scales


def test_module_owned_source_scale_applies_to_plain_visible_weight() -> None:
    owner = nn.Module()
    qweight = torch.randn(128, 256).clamp(-4, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.125, 0.25]], dtype=torch.float32)
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    visible_weight = nn.Parameter((qweight.float() * expanded).to(torch.bfloat16))
    owner._fp8_source_scales_by_parameter = {"weight": scale}

    bound = bind_source_scale_to_visible_weight(owner, "weight", visible_weight)
    reconstructed = deployment_fp8.quantize_block_fp8_weight(bound)

    assert bound is visible_weight
    assert torch.equal(reconstructed.qweight, qweight)
    assert reconstructed.scales is scale


def test_cleared_module_source_scale_requantizes_same_visible_weight(fake_vllm) -> None:
    owner = nn.Module()
    visible_weight = _weight()
    scale = torch.tensor([[0.125, 0.25]], dtype=torch.float32)
    owner._fp8_source_scales_by_parameter = {"weight": scale}

    bind_source_scale_to_visible_weight(owner, "weight", visible_weight)
    owner._fp8_source_scales_by_parameter.clear()
    rebound = bind_source_scale_to_visible_weight(owner, "weight", visible_weight)
    packed = pack_block_fp8_weight(rebound)

    assert rebound is visible_weight
    assert not hasattr(visible_weight, "_fp8_source_scales")
    assert not hasattr(visible_weight, "_fp8_source_scale_version")
    assert [call[0] for call in fake_vllm] == [
        "weight_quant",
        "weight_postprocess",
    ]


def test_native_fp8_source_scale_uses_official_float32_requantization(fake_vllm) -> None:
    owner = nn.Module()
    qweight = torch.randn(128, 256).clamp(-4, 4).to(torch.float8_e4m3fn)
    scale = torch.tensor([[2.0**-12, 2.0**-11]], dtype=torch.float32)
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    visible_weight = nn.Parameter((qweight.float() * expanded).to(torch.bfloat16))
    owner._fp8_source_scales_by_parameter = {"weight": scale}
    bind_source_scale_to_visible_weight(owner, "weight", visible_weight)
    packed = pack_block_fp8_weight(visible_weight)

    assert [call[0] for call in fake_vllm] == ["weight_postprocess"]
    post_call = fake_vllm[0]
    assert post_call[2].dtype == torch.float32
    assert post_call[2] is scale
    assert torch.equal(post_call[1], qweight)
    assert torch.equal(packed.qweight, qweight)


def test_grouped_weight_scales_are_transformed_jointly(fake_vllm) -> None:
    masters = [_weight(), _weight()]
    packed = pack_grouped_block_fp8_weight(masters)

    assert [call[0] for call in fake_vllm] == [
        "weight_quant",
        "weight_quant",
        "weight_postprocess",
    ]
    post_call = fake_vllm[-1]
    assert post_call[1].shape == (2, 128, 256)
    assert post_call[2].shape == (2, 1, 2)
    assert packed.qweight.shape == (2, 128, 256)
    assert packed.scales.shape == (2, 1, 2)
    assert tuple(key[0] for key in packed.cache_key) == tuple(id(master) for master in masters)


def test_fused_release_parameters_preserve_bytes_scales_and_cache(fake_vllm) -> None:
    qweights = [
        torch.randn(128, 256).clamp(-4, 4).to(torch.float8_e4m3fn)
        for _ in range(2)
    ]
    scales = [
        torch.tensor([[0.125, 0.25]], dtype=torch.float32),
        torch.tensor([[0.5, 1.0]], dtype=torch.float32),
    ]
    masters = []
    for qweight, scale in zip(qweights, scales, strict=True):
        expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        master = nn.Parameter((qweight.float() * expanded).to(torch.bfloat16))
        master._fp8_source_scales = scale
        master._fp8_source_scale_version = master._version
        masters.append(master)

    adapter = DeploymentFusedBlockFP8Adapter(cache_weight=True)
    packed = adapter.pack_weight(masters)

    assert [call[0] for call in fake_vllm] == ["weight_postprocess"]
    post = fake_vllm[0]
    assert torch.equal(post[1], torch.cat(qweights))
    assert torch.equal(post[2], torch.cat(scales))
    assert adapter.pack_weight(masters) is packed
    assert tuple(key[0] for key in packed.cache_key) == tuple(id(master) for master in masters)

    with torch.no_grad():
        masters[1].add_(1)
    assert adapter.pack_weight(masters) is not packed


def test_activation_and_gemm_call_use_packed_vllm_contract(fake_vllm) -> None:
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    packed_weight = pack_block_fp8_weight(_weight())
    packed_activation = pack_block_fp8_activation(x)

    assert packed_activation.qactivation.shape == x.shape
    assert packed_activation.scales.shape == (4, 1)
    activation_call = fake_vllm[-1]
    assert activation_call[0] == "activation_quant"
    assert activation_call[1:] == (x, 128, True)

    output = fp8_gemm_nt(x, packed_weight)
    assert output.dtype == torch.bfloat16
    assert output.shape == (4, 128)
    assert torch.equal(output, torch.full_like(output, 3))
    gemm_call = fake_vllm[-1]
    assert gemm_call[0] == "gemm"
    assert gemm_call[-1] is True
    assert gemm_call[1].dtype == torch.float8_e4m3fn
    assert gemm_call[3] is packed_weight.qweight
    assert gemm_call[4] is packed_weight.scales


def test_opt_in_cache_uses_parameter_version_and_invalidates(fake_vllm) -> None:
    master = _weight()
    adapter = DeploymentBlockFP8Adapter(cache_weight=True)

    first = adapter.pack_weight(master)
    second = adapter.pack_weight(master)
    assert second is first
    assert [call[0] for call in fake_vllm].count("weight_quant") == 1

    with torch.no_grad():
        master.add_(1)
    third = adapter.pack_weight(master)
    assert third is not first
    assert third.cache_key[1] > first.cache_key[1]
    assert [call[0] for call in fake_vllm].count("weight_quant") == 2

    adapter.clear_cache()
    assert adapter._cached_weight is None


def test_grouped_cache_packs_only_first_microbatch_until_step_invalidation(
    fake_vllm,
) -> None:
    masters = [_weight(), _weight()]
    adapter = DeploymentGroupedBlockFP8Adapter(cache_weight=True)

    first = adapter.pack_weight("experts", masters)
    for _microbatch in range(1, 4):
        assert adapter.pack_weight("experts", masters) is first
    assert [call[0] for call in fake_vllm].count("weight_quant") == len(masters)

    adapter.clear_cache()
    next_step = adapter.pack_weight("experts", masters)
    assert next_step is not first
    assert [call[0] for call in fake_vllm].count("weight_quant") == 2 * len(
        masters
    )


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (torch.randn(128, 128), TypeError, "must be a 2-D BF16 tensor"),
        (torch.randn(2, 128, 128, dtype=torch.bfloat16), TypeError, "must be a 2-D BF16 tensor"),
        (torch.randn(128, 129, dtype=torch.bfloat16), ValueError, "must be divisible"),
    ],
)
def test_invalid_master_weight_fails_before_vllm_import(
    value, error, message
) -> None:
    with pytest.raises(error, match=message):
        pack_block_fp8_weight(value)


def test_invalid_activation_and_gemm_contracts_fail_closed(fake_vllm) -> None:
    with pytest.raises(ValueError, match="contiguous 2-D BF16"):
        pack_block_fp8_activation(torch.randn(2, 128))
    with pytest.raises(ValueError, match="contiguous 2-D BF16"):
        pack_block_fp8_activation(torch.randn(2, 2, 128, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="contiguous 2-D BF16"):
        pack_block_fp8_activation(torch.randn(2, 129, dtype=torch.bfloat16))
    noncontiguous = torch.randn(2, 256, dtype=torch.bfloat16)[:, ::2]
    with pytest.raises(ValueError, match="contiguous 2-D BF16"):
        pack_block_fp8_activation(noncontiguous)
