# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real dense precision contracts; fresh native-TE/workspace-zero processes.

Tensorwise FP8 and NVFP4 admission are negative; real MXFP8 requires Blackwell.
"""

import os

import pytest
import torch
from transformer_engine.pytorch.module import linear as te_linear
from transformer_engine.pytorch.tensor import QuantizedTensorStorage

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers import batch_invariant_kernels as bik
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal
from tests.unit_tests.test_utilities import Utils


def _config(recipe, **overrides):
    options = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_backend=AttnBackend.flash,
        flash_attention_version=3,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        batch_invariant_mode=True,
        batch_invariant_backend="te_native",
        fp8="e4m3",
        fp8_recipe=recipe,
    )
    options.update(overrides)
    return TransformerConfig(**options)


@pytest.mark.parametrize("recipe", ["mxfp8", "delayed", "blockwise"])
@torch.inference_mode()
def test_dense_fp8_target_batch_invariance(monkeypatch, recipe):
    assert Utils.world_size == 1
    assert os.environ.get("CUBLASLT_WORKSPACE_SIZE") == "0"
    assert not bik.is_batch_invariant_mode_enabled()
    bik.enable_batch_invariant_mode(backend="te_native", collective="ordered")
    Utils.initialize_model_parallel(1, 1)
    try:
        assert (
            recipe != "mxfp8" or torch.cuda.get_device_capability()[0] >= 10
        ), "MXFP8 needs Blackwell"
        torch.manual_seed(321)
        model_parallel_cuda_manual_seed(321, inference_rng_tracker=True, force_reset_rng=True)
        config = _config(recipe)
        # Use the production MCore TE adapter, FP8 context, quantization, and GEMM.
        layer = (
            TEColumnParallelLinear(
                input_size=128,
                output_size=256,
                config=config,
                init_method=init_method_normal(config.init_method_std),
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )
            .cuda()
            .eval()
        )
        target = (torch.sin(torch.arange(1024, device="cuda") / 29) * 0.05).reshape(8, 128)
        target = target.bfloat16()
        current, calls, outputs = {}, [], []
        original = te_linear.general_gemm

        def quantized_gemm(*args, **kwargs):
            # Witness native GEMM quantized operands; BF16 source weights alone cannot prove FP8.
            operands = [arg for arg in args[:2] if isinstance(arg, QuantizedTensorStorage)]
            assert len(operands) == 2, "dense path did not consume two quantized operands"
            activation = next(
                arg for arg in operands if arg.dequantize().shape[0] == current["rows"]
            )
            decoded = activation.dequantize()
            begin = current["begin"]
            snapshot = decoded[begin : begin + 8].detach().clone()
            result = original(*args, **kwargs)
            calls.append(
                dict(
                    physical=decoded.shape[0],
                    quantized_type=type(activation).__name__,
                    target_quantized=snapshot,
                    begin=begin,
                )
            )
            return result

        monkeypatch.setattr(te_linear, "general_gemm", quantized_gemm)
        for rows, begin, magnitude in ((64, 0, 1.0), (128, 120, 73.0), (128, 0, 73.0)):
            current.update(rows=rows, begin=begin)
            inputs = target[:1].repeat(rows, 1) * magnitude
            inputs[begin : begin + 8] = target
            assert torch.equal(inputs[begin : begin + 8], target)
            before = len(calls)
            with get_fp8_context(config):
                result, bias = layer(inputs)
            assert bias is None and len(calls) == before + 1
            assert result.shape == (rows, 256)
            outputs.append(result[begin : begin + 8].detach().clone())
        assert {call["physical"] for call in calls} == {64, 128}
        assert outputs[0].abs().max() > 0
        for call, output in zip(calls[1:], outputs[1:]):
            quantized_equal = torch.equal(call["target_quantized"], calls[0]["target_quantized"])
            output_equal = torch.equal(output, outputs[0])
            error = (output.float() - outputs[0].float()).abs().max().item()
            print(
                f"BI_FP8_WITNESS {(recipe, call['quantized_type'], call['physical'])} "
                f"quantized/output_equal {quantized_equal} {output_equal} max_target_error {error}"
            )
        assert all(torch.equal(output, outputs[0]) for output in outputs[1:]), (
            "accepted BI FP8 configuration changed target output",
            recipe,
        )
    finally:
        bik.disable_batch_invariant_mode()
        Utils.destroy_model_parallel()
