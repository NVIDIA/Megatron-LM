# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Hybrid stacks with fused TP communication (``inference_fuse_tp_communication``).

A HybridStack builds attention-only ("*") and MLP-only ("-") TransformerLayers, with an
IdentityOp in place of the other sublayer. Each layer hands its residual add only to the output
projection it has (linear_proj or linear_fc2), and the stack links the layers as TransformerBlock
does: the fused reduce-scatter + add + rms-norm + all-gather of each output projection applies
the input norm of the next layer's linear_qkv or linear_fc1, which then reads its input from
symmetric memory.
"""

from unittest import mock

import pytest
import torch
import torch.distributed as dist

import megatron.core.tensor_parallel.inference_layers as inference_layers
from megatron.core import parallel_state
from megatron.core.inference.communication.torch_symm_triton.utils import is_device_nvls_capable
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_inference_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_S, _B, _H = 64, 2, 128

# Alternating attention-only and MLP-only layers, and the same kinds back to back, so that
# every output projection (linear_proj, linear_fc2) feeds every input linear (linear_qkv,
# linear_fc1) once.
_PATTERNS = ["*-*-", "**--"]

# (tp, disable_nvls) for each path of the row-parallel layer: no reduce-scatter at TP=1, the
# NCCL reduce-scatter when the Triton NVLS kernels are disabled, and the NVLS reduce-scatter,
# fused with the next layer's input norm.
_PATHS = {"tp1": (1, False), "tp2_nccl": (2, True), "tp2_nvls": (2, False)}

# The residual is N(0, 1) and the sub-layers add a fraction of its norm, so a dropped residual
# is far outside this tolerance; it covers bf16 rounding at |x| ~ 4.
_ATOL = 2e-2
_RTOL = 2e-2


def _build(pattern: str, tp: int, *, fuse: bool, disable_nvls: bool = False) -> HybridStack:
    """Smallest inference-optimized hybrid stack of the pattern's layers, without a final
    norm, on the GPU."""
    config = TransformerConfig(
        num_layers=len(pattern),
        hidden_size=_H,
        ffn_hidden_size=256,
        num_attention_heads=4,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.bfloat16,
        bf16=True,
        init_method_std=0.05,
        inference_fuse_tp_communication=fuse,
        inference_disable_triton_nvls_kernels=disable_nvls,
    )
    model_parallel_cuda_manual_seed(123)
    stack = HybridStack(
        config,
        hybrid_inference_stack_spec.submodules,
        layer_config_list=validate_segment_layers(pattern, config),
        post_layer_norm=False,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    return stack.cuda().eval()


def _input_linear(layer):
    """The linear that applies the layer's input norm."""
    if hasattr(layer.self_attention, "linear_qkv"):
        return layer.self_attention.linear_qkv
    return layer.mlp.linear_fc1


def _output_linear(layer):
    """The projection that gives the layer's output."""
    if hasattr(layer.mlp, "linear_fc2"):
        return layer.mlp.linear_fc2
    return layer.self_attention.linear_proj


def _all_ranks_agree(value: bool) -> bool:
    flag = torch.tensor([float(value)], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return flag.item() == 1.0


def _forward(stack: HybridStack, x: torch.Tensor) -> torch.Tensor:
    """Run the stack on a copy of x; the fused kernel updates the residual in place."""
    with torch.no_grad(), InferenceMode.active():
        out = stack(hidden_states=x.clone(), attention_mask=None)
    torch.cuda.synchronize()
    dist.barrier()
    return out.float()


def _require_nvls():
    buf = SymmetricMemoryManager.get_buffer(
        "tp", process_group=parallel_state.get_tensor_model_parallel_group()
    )  # collective: create on every rank before any skip
    capable = is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device()))
    hdl = buf.symm_mem_hdl
    if not _all_ranks_agree(capable and hdl is not None and bool(hdl.multicast_ptr)):
        pytest.skip("NVLS multicast unavailable")


@pytest.mark.skipif(
    not inference_layers.HAVE_TE, reason="inference_optimized requires Transformer Engine"
)
class TestHybridFusedTPCommunication:

    def _init(self, tp: int):
        if Utils.world_size % tp != 0:
            pytest.skip(f"requires a world size divisible by TP={tp}")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        SymmetricMemoryManager.destroy()

    def teardown_method(self, method):
        torch.cuda.synchronize()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp", [1, 2])
    @pytest.mark.parametrize("pattern", _PATTERNS)
    def test_stack_wiring(self, pattern, tp):
        """Each layer's output projection fuses into the next layer's input linear, and the
        last one only reduce-scatters and adds the residual."""
        self._init(tp)
        layers = _build(pattern, tp, fuse=True).layers
        for i, layer in enumerate(layers):
            prev_output = _output_linear(layers[i - 1]) if i > 0 else None
            handoff = getattr(prev_output, "fused_tp_handoff", None)
            assert _input_linear(layer).fused_tp_handoff is handoff, f"layer {i} input"

            norm = _output_linear(layer).next_layer_norm_weights
            if i + 1 < len(layers):
                next_input = _input_linear(layers[i + 1])
                assert norm is not None, f"layer {i} output fuses into nothing"
                assert norm.data_ptr() == next_input.layer_norm_weight.data_ptr(), f"layer {i}"
            else:
                assert norm is None

    @pytest.mark.parametrize("path", list(_PATHS))
    @pytest.mark.parametrize("pattern", _PATTERNS)
    def test_fused_matches_unfused(self, pattern, path):
        """Flag-on output equals flag-off output on every path. With NVLS, the fused kernel
        runs in every output projection but the last one, and only the first layer applies its
        own input norm."""
        tp, disable_nvls = _PATHS[path]
        self._init(tp)
        if path == "tp2_nvls":
            _require_nvls()

        unfused = _build(pattern, tp, fuse=False, disable_nvls=disable_nvls)
        fused = _build(pattern, tp, fuse=True, disable_nvls=disable_nvls)
        gen = torch.Generator(device="cpu").manual_seed(4321)
        fused_params = dict(fused.named_parameters())
        with torch.no_grad():
            for name, param in unfused.named_parameters():
                if name.endswith("layer_norm_weight"):
                    # Gains away from 1, so a norm applied with the wrong weights shows.
                    param.copy_(1.0 + 0.1 * torch.randn(param.shape, generator=gen))
                # Copy in place: the fused wiring holds the norm weights by reference.
                fused_params.pop(name).copy_(param)
        assert not fused_params, f"parameters only in the fused stack: {sorted(fused_params)}"

        gen = torch.Generator(device="cpu").manual_seed(1234)
        x = torch.randn(_S, _B, _H, generator=gen).to(torch.bfloat16)
        x = x.chunk(tp, dim=0)[parallel_state.get_tensor_model_parallel_rank()].contiguous().cuda()

        expected = _forward(unfused, x)
        fn_fused = inference_layers.fused_multimem_rs_add_norm_ag
        fn_norm = inference_layers._te_rms_norm_kernel
        with (
            mock.patch.object(
                inference_layers, "fused_multimem_rs_add_norm_ag", wraps=fn_fused
            ) as fused_spy,
            mock.patch.object(inference_layers, "_te_rms_norm_kernel", wraps=fn_norm) as norm_spy,
        ):
            actual = _forward(fused, x)

        max_diff = (actual - expected).abs().max().reshape(1)
        dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
        ok = _all_ranks_agree(torch.allclose(actual, expected, atol=_ATOL, rtol=_RTOL))
        assert ok, f"fused vs unfused: max |diff| = {max_diff.item():.4f}"

        nvls = path == "tp2_nvls"
        assert fused_spy.call_count == (len(pattern) - 1 if nvls else 0)
        applied = [call.kwargs["weight"] for call in norm_spy.call_args_list]
        for i, layer in enumerate(fused.layers):
            applies_norm = any(w is _input_linear(layer).layer_norm_weight for w in applied)
            assert applies_norm == (i == 0 or not nvls), f"layer {i} applies its own norm"
