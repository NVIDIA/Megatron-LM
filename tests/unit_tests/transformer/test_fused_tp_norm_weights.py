# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fused TP communication reads the next layer's norm weights as they are at forward time.

With ``inference_fuse_tp_communication``, TransformerBlock gives each
``InferenceRowParallelLinear`` the norm weights of the column-parallel layer that takes its
output, for the fused NVLS reduce-scatter + add + rms-norm + all-gather kernel. Moving the model
to the GPU after CPU initialization, casting it (as Float16Module does) and DDP's parameter
buffer all rebind the parameters to new storage after that wiring is set up.
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
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_inference_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

requires_te = pytest.mark.skipif(
    not inference_layers.HAVE_TE, reason="inference_optimized requires Transformer Engine"
)

_S, _B, _H = 64, 2, 128
_FFN = 512
_HEADS = 4
_NUM_LAYERS = 2

# Covers bf16 rounding at |x| ~ 4. Norm gains in [0.5, 1.5] applied as 1.0 (their initial
# value) or misread from stale storage change the output far beyond it.
_ATOL = 2e-2
_RTOL = 2e-2


def _rebind_storage(block: TransformerBlock) -> TransformerBlock:
    """Point every parameter at new storage with the same values, as DDP does."""
    for param in block.parameters():
        param.data = param.detach().clone()
    return block


# How each parameter's storage changes after construction: (config overrides, transform).
_REBINDS = {
    # Built on the CPU and moved to the GPU, as get_model does with --use-cpu-initialization.
    "cuda": (dict(use_cpu_initialization=True, params_dtype=torch.bfloat16), lambda b: b.cuda()),
    # Built in FP32 and cast to BF16, as Float16Module does.
    "bfloat16": (dict(params_dtype=torch.float32), lambda b: b.bfloat16()),
    # Built in BF16 and moved into another buffer, as DDP does.
    "storage": (dict(params_dtype=torch.bfloat16), _rebind_storage),
}


def _build(tp: int, rebind: str, *, fuse: bool) -> TransformerBlock:
    """Smallest inference-optimized GPT block, without a final norm, on the GPU."""
    overrides, transform = _REBINDS[rebind]
    config = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_H,
        ffn_hidden_size=_FFN,
        num_attention_heads=_HEADS,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        init_method_std=0.05,
        inference_fuse_tp_communication=fuse,
        **overrides,
    )
    model_parallel_cuda_manual_seed(123)
    block = TransformerBlock(config, get_gpt_layer_with_inference_spec(), post_layer_norm=False)
    return transform(block).eval()


def _assert_same_storage(held: torch.Tensor, param: torch.Tensor, what: str):
    """The fused kernel would read ``param``'s current storage through ``held``."""
    held_at = (held.device, held.dtype, held.data_ptr())
    param_at = (param.device, param.dtype, param.data_ptr())
    assert held_at == param_at, f"{what} holds {held_at}, the parameter is at {param_at}"


def _all_ranks_agree(value: bool) -> bool:
    flag = torch.tensor([float(value)], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return flag.item() == 1.0


def _require_nvls():
    buf = SymmetricMemoryManager.get_buffer(  # collective: create on every rank before any skip
        "tp", process_group=parallel_state.get_tensor_model_parallel_group()
    )
    hdl = buf.symm_mem_hdl
    capable = is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device()))
    if not _all_ranks_agree(capable and hdl is not None and bool(hdl.multicast_ptr)):
        pytest.skip("NVLS multicast unavailable")


def _set_same_weights(unfused: TransformerBlock, fused: TransformerBlock):
    """Copy, in place, the same weights into both blocks, with norm gains away from 1."""
    gen = torch.Generator(device="cpu").manual_seed(4321)
    fused_params = dict(fused.named_parameters())
    with torch.no_grad():
        for name, param in unfused.named_parameters():
            if name.endswith("layer_norm_weight"):
                param.copy_(0.5 + torch.rand(param.shape, generator=gen))
            fused_params.pop(name).copy_(param)
    assert not fused_params, f"parameters only in the fused block: {sorted(fused_params)}"


def _input_shard(tp: int) -> torch.Tensor:
    """This rank's sequence-parallel shard of a fixed N(0, 1) [S, B, H] input."""
    gen = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(_S, _B, _H, generator=gen).to(torch.bfloat16)
    rank = parallel_state.get_tensor_model_parallel_rank()
    return x.chunk(tp, dim=0)[rank].contiguous().cuda()


def _forward(block: TransformerBlock, x: torch.Tensor) -> torch.Tensor:
    # The fused kernel updates the residual, here the block's input, in place.
    with torch.no_grad(), InferenceMode.active():
        out = block(hidden_states=x.clone(), attention_mask=None).float()
    torch.cuda.synchronize()
    dist.barrier()
    return out


@requires_te
class TestFusedTPNormWeights:

    def teardown_method(self, method):
        torch.cuda.synchronize()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("rebind", list(_REBINDS))
    def test_row_parallel_layers_hold_the_norm_parameters(self, rebind):
        """After the parameters move, each layer still points at the next norm's storage."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        block = _build(1, rebind, fuse=True)
        layers = block.layers
        for i, layer in enumerate(layers):
            _assert_same_storage(
                layer.self_attention.linear_proj.next_layer_norm_weights,
                layer.mlp.linear_fc1.layer_norm_weight,
                f"layer {i} linear_proj",
            )
            fc2_weights = layer.mlp.linear_fc2.next_layer_norm_weights
            if i + 1 < len(layers):
                _assert_same_storage(
                    fc2_weights,
                    layers[i + 1].self_attention.linear_qkv.layer_norm_weight,
                    f"layer {i} linear_fc2",
                )
            elif fc2_weights is not None:
                # The last layer has no next norm, but the kernel still reads these.
                param = layer.mlp.linear_fc1.layer_norm_weight
                assert (fc2_weights.device, fc2_weights.dtype, fc2_weights.shape) == (
                    param.device,
                    param.dtype,
                    param.shape,
                ), f"last linear_fc2 holds {fc2_weights.dtype} on {fc2_weights.device}"

        # The references add no parameters or state to the row-parallel layers.
        unfused = _build(1, rebind, fuse=False)
        assert list(block.state_dict()) == list(unfused.state_dict())
        assert [n for n, _ in block.named_parameters()] == [
            n for n, _ in unfused.named_parameters()
        ]

    @pytest.mark.parametrize("rebind", list(_REBINDS))
    def test_fused_matches_unfused(self, rebind):
        """The fused kernel normalizes with the current norm weights."""
        tp = 2
        if Utils.world_size % tp != 0:
            pytest.skip(f"requires a world size divisible by TP={tp}")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        SymmetricMemoryManager.destroy()
        _require_nvls()

        unfused = _build(tp, rebind, fuse=False)
        fused = _build(tp, rebind, fuse=True)
        _set_same_weights(unfused, fused)
        x = _input_shard(tp)
        expected = _forward(unfused, x)
        fused_kernel = inference_layers.fused_multimem_rs_add_norm_ag
        with mock.patch.object(
            inference_layers, "fused_multimem_rs_add_norm_ag", wraps=fused_kernel
        ) as spy:
            actual = _forward(fused, x)
        # Every linear_proj, and every linear_fc2 with a next layer, fused.
        assert spy.call_count >= 2 * _NUM_LAYERS - 1

        max_diff = (actual - expected).abs().max().reshape(1)
        dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
        ok = _all_ranks_agree(torch.allclose(actual, expected, atol=_ATOL, rtol=_RTOL))
        assert ok, f"fused vs unfused: max |diff| = {max_diff.item():.4f}"
