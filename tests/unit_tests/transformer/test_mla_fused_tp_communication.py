# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MLA layers with fused TP communication (``inference_fuse_tp_communication``).

The fused reduce-scatter + add + rms-norm + all-gather of a row-parallel layer applies the
input norm of the column-parallel layer that follows it. In an MLA layer this works from
attention's linear_proj into the MLP's linear_fc1, as in a GPT layer. The MLA attention input
cannot take it: a separate input_layernorm normalizes the sequence-parallel shard that the
down projections consume, so the previous layer's linear_fc2 leaves that norm to the layer.
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
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

_S, _B, _H = 64, 2, 128
_NUM_LAYERS = 2

# The residual is N(0, 1) and the sub-layers add a fraction of its norm; this covers bf16
# rounding at |x| ~ 4.
_ATOL = 2e-2
_RTOL = 2e-2


def _build(tp: int, *, fuse: bool) -> TransformerBlock:
    """Smallest inference-optimized block of MLA layers, without a final norm, on the GPU."""
    config = MLATransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_H,
        ffn_hidden_size=256,
        num_attention_heads=4,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=64,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        qk_layernorm=True,
        rope_type="rope",
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
    )
    model_parallel_cuda_manual_seed(123)
    spec = get_gpt_layer_with_inference_spec(qk_layernorm=True, multi_latent_attention=True)
    block = TransformerBlock(config, spec, post_layer_norm=False)
    block.eval()
    return block


def _all_ranks_agree(value: bool) -> bool:
    flag = torch.tensor([float(value)], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return flag.item() == 1.0


def _forward(block: TransformerBlock, x: torch.Tensor) -> torch.Tensor:
    """Run the block on a copy of x; the fused kernel updates the residual in place."""
    with torch.no_grad(), InferenceMode.active():
        out = block(hidden_states=x.clone(), attention_mask=None)
    torch.cuda.synchronize()
    dist.barrier()
    return out.float()


@pytest.mark.skipif(
    not inference_layers.HAVE_TE, reason="inference_optimized requires Transformer Engine"
)
class TestMLAFusedTPCommunication:

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
    def test_block_fuses_only_attention_output(self, tp):
        """The block builds, and each linear_proj is wired to its own layer's linear_fc1."""
        self._init(tp)
        block = _build(tp, fuse=True)
        for layer in block.layers:
            assert layer.get_qkv_layer_norm_weights() is None
            fused_norm = layer.self_attention.linear_proj.next_layer_norm_weights
            assert fused_norm.data_ptr() == layer.mlp.linear_fc1.layer_norm_weight.data_ptr()

    def test_fused_matches_unfused(self):
        """With NVLS, flag-on output equals flag-off output, and each linear_fc1 takes its
        normalized input from the fused kernel."""
        tp = 2
        self._init(tp)
        buf = SymmetricMemoryManager.get_buffer(
            "tp", process_group=parallel_state.get_tensor_model_parallel_group()
        )  # collective: create on every rank before any skip
        capable = is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device()))
        hdl = buf.symm_mem_hdl
        if not _all_ranks_agree(capable and hdl is not None and bool(hdl.multicast_ptr)):
            pytest.skip("NVLS multicast unavailable")

        unfused, fused = _build(tp, fuse=False), _build(tp, fuse=True)
        gen = torch.Generator(device="cpu").manual_seed(4321)
        fused_params = dict(fused.named_parameters())
        with torch.no_grad():
            for name, param in unfused.named_parameters():
                if name.endswith(("layer_norm_weight", "layernorm.weight")):
                    # Gains away from 1, so a norm applied with the wrong weights shows.
                    param.copy_(1.0 + 0.1 * torch.randn(param.shape, generator=gen))
                # Copy in place: the fused wiring holds the norm weights by reference.
                fused_params.pop(name).copy_(param)
        assert not fused_params, f"parameters only in the fused block: {sorted(fused_params)}"
        for block in (unfused, fused):
            for layer in block.layers:
                # The up projections all-gather back to back into the shared symmetric
                # buffer without barrier_before (see multimem_all_gather); take them
                # through NCCL so that this test covers fused TP communication only.
                layer.self_attention.linear_q_up_proj.triton_nvls_kernels_allowed = False
                layer.self_attention.linear_kv_up_proj.triton_nvls_kernels_allowed = False

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
        # The fused kernel runs in every linear_proj and linear_fc2. Only linear_proj's
        # output is consumed: linear_fc1 skips its norm, while MLA's input_layernorm
        # normalizes the residual stream that linear_fc2 returns.
        assert fused_spy.call_count == 2 * _NUM_LAYERS
        fc1_norms = [layer.mlp.linear_fc1.layer_norm_weight for layer in fused.layers]
        for call in norm_spy.call_args_list:
            assert not any(call.kwargs["weight"] is w for w in fc1_norms)
