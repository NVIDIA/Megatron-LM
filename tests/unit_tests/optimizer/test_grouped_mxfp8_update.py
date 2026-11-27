# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Whole grouped MXFP8 publication into fixed, unswizzled storage."""

import pytest
import torch

from megatron.core.fp8_utils import (
    copy_tensor_to_quantized_param,
    get_grouped_quantized_members,
    prepare_grouped_mxfp8_weight_,
    update_grouped_mxfp8_weight_,
)
from tests.unit_tests.test_utilities import Utils

te = pytest.importorskip("transformer_engine.pytorch")
tex = pytest.importorskip("transformer_engine_torch")
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

pytestmark = [pytest.mark.internal, pytest.mark.launch_on_gb200]

_PARENT_FIELDS = ("rowwise_data", "columnwise_data", "scale_inv", "columnwise_scale_inv")
_MEMBER_FIELDS = (
    "_rowwise_data",
    "_columnwise_data",
    "_rowwise_scale_inv",
    "_columnwise_scale_inv",
)


@pytest.fixture(autouse=True)
def _distributed():
    available, reason = te.is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _weight(shape, *, columnwise=True, with_2d_quantization=False):
    quantizer = MXFP8Quantizer(
        DType.kFloat8E4M3,
        rowwise=True,
        columnwise=columnwise,
        with_2d_quantization=with_2d_quantization,
    )
    return torch.nn.Parameter(
        GroupedTensor.make_grouped_tensor_with_shapes(
            num_tensors=shape[0],
            shapes=[shape[1:]] * shape[0],
            quantizer=quantizer,
            dtype=torch.bfloat16,
            device="cuda",
        )
    )


def _storage_identity(weight):
    members = get_grouped_quantized_members(weight)
    return (
        id(weight.quantizer),
        tuple(getattr(weight, field).data_ptr() for field in _PARENT_FIELDS),
        tuple(id(member) for member in members),
        tuple(
            tuple(getattr(member, field).data_ptr() for field in _MEMBER_FIELDS)
            for member in members
        ),
    )


def _assert_reference(weight, source):
    quantizer = MXFP8Quantizer(
        DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        with_2d_quantization=weight.quantizer.with_2d_quantization,
    )
    quantizer.optimize_for_gemm = False
    assert not weight._with_gemm_swizzled_scales
    for member, values in zip(get_grouped_quantized_members(weight), source, strict=True):
        reference = quantizer(values.contiguous())
        assert not member._with_gemm_swizzled_scales
        for field in _MEMBER_FIELDS:
            torch.testing.assert_close(
                getattr(member, field), getattr(reference, field), rtol=0, atol=0, msg=field
            )
        torch.testing.assert_close(
            member.dequantize(dtype=torch.float32),
            reference.dequantize(dtype=torch.float32),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "shape,with_2d_quantization", [((3, 128, 256), False), ((2, 256, 128), True)]
)
@torch.no_grad()
def test_grouped_mxfp8_update_reuses_storage_and_dispatches_once(
    shape, with_2d_quantization, monkeypatch
):
    generator = torch.Generator(device="cuda").manual_seed(1863)
    source = torch.randn(shape, dtype=torch.bfloat16, device="cuda", generator=generator)
    weight = _weight(shape, with_2d_quantization=with_2d_quantization)
    # Preparation must preserve the ordinary scale layout, every byte, and all
    # existing aliases, not just the values represented by later updates.
    tex.group_quantize(source.view(-1, shape[-1]), weight.quantizer, shape[0], None, output=weight)
    identity = _storage_identity(weight)
    initial_storage = tuple(getattr(weight, field).clone() for field in _PARENT_FIELDS)
    quantizer_settings = vars(weight.quantizer).copy()

    def reject_swizzle(*args, **kwargs):
        pytest.fail("Preparation and publication must retain unswizzled weight storage")

    monkeypatch.setattr(tex, "grouped_swizzle_for_gemm", reject_swizzle)
    prepare_grouped_mxfp8_weight_(weight, param_name="experts.weight")
    prepare_grouped_mxfp8_weight_(weight, param_name="experts.weight")
    assert vars(weight.quantizer) == quantizer_settings
    assert _storage_identity(weight) == identity
    for field, initial in zip(_PARENT_FIELDS, initial_storage, strict=True):
        torch.testing.assert_close(getattr(weight, field), initial, rtol=0, atol=0, msg=field)
    _assert_reference(weight, source)

    dispatches = []
    group_quantize = tex.group_quantize

    def traced_group_quantize(*args, **kwargs):
        dispatches.append(kwargs.get("output"))
        return group_quantize(*args, **kwargs)

    def reject_member_cast(*args, **kwargs):
        pytest.fail("Whole grouped MXFP8 publication must not launch per-member casts")

    monkeypatch.setattr(tex, "group_quantize", traced_group_quantize)
    monkeypatch.setattr(MXFP8Quantizer, "update_quantized", reject_member_cast)
    # Output buffers determine both write directions even if the shared compute
    # quantizer is configured for an evaluation forward.
    weight.quantizer.set_usage(rowwise=True, columnwise=False)
    for multiplier in (0.125, -4.0):
        source.mul_(multiplier)
        copy_tensor_to_quantized_param(weight, source)
        assert len(dispatches) == 1 and dispatches.pop() is weight
        assert not weight.quantizer.columnwise_usage
        assert _storage_identity(weight) == identity
        _assert_reference(weight, source)


@torch.no_grad()
def test_grouped_mxfp8_update_cuda_graph_replay():
    shape = (3, 128, 256)
    source = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    weight = _weight(shape)
    prepare_grouped_mxfp8_weight_(weight)
    identity = _storage_identity(weight)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        update_grouped_mxfp8_weight_(weight, source)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        update_grouped_mxfp8_weight_(weight, source)
        torch.cuda.synchronize()
    kernels = [
        event.name
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert len(kernels) == 1 and "group_quantize_mxfp8_kernel" in kernels[0], kernels
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update_grouped_mxfp8_weight_(weight, source)
    for multiplier in (0.5, -3.0):
        source.mul_(multiplier)
        graph.replay()
        _assert_reference(weight, source)
        assert _storage_identity(weight) == identity

    # Checkpoint-style dequantization reads canonical scales directly; it must
    # not add an unswizzle kernel to recover the ordinary scale layout.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        for member in get_grouped_quantized_members(weight):
            member.dequantize(dtype=torch.float32)
        torch.cuda.synchronize()
    kernels = [
        event.name
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert kernels and not any("swizzle" in name.lower() for name in kernels), kernels


@pytest.mark.parametrize("shape", [(2, 64, 128), (2, 128, 64), (65, 128, 128)])
def test_grouped_mxfp8_prepare_rejects_unsupported_shape(shape):
    with pytest.raises(ValueError, match="experts.weight.*uniform.*128"):
        prepare_grouped_mxfp8_weight_(_weight(shape), param_name="experts.weight")


@torch.no_grad()
def test_grouped_mxfp8_update_requires_preparation_and_full_bf16_source():
    shape = (2, 128, 128)
    weight = _weight(shape)
    source = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="not prepared"):
        update_grouped_mxfp8_weight_(weight, source)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="not prepared"):
        with torch.cuda.graph(graph):
            update_grouped_mxfp8_weight_(weight, source)
    with pytest.raises(RuntimeError, match="columnwise_data.*allocated"):
        prepare_grouped_mxfp8_weight_(_weight(shape, columnwise=False))
    prepare_grouped_mxfp8_weight_(weight)
    for invalid_source in (source.float(), source[:, :, :64], source.transpose(-1, -2)):
        with pytest.raises(ValueError, match="contiguous BF16.*complete parameter"):
            update_grouped_mxfp8_weight_(weight, invalid_source)
