# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MLA layers with fused TP communication (``inference_fuse_tp_communication``).

The fused reduce-scatter + add + rms-norm + all-gather of a row-parallel layer applies the
input norm of the column-parallel layer that follows it. In an MLA layer this works from
attention's linear_proj into the MLP's linear_fc1, as in a GPT layer. The MLA attention input
cannot take it: a separate input_layernorm normalizes the sequence-parallel shard that the
down projections consume. So the previous layer's linear_fc2 only reduce-scatters and adds the
residual, as the last layer's does, and fuses only into a next GPT layer's linear_qkv.

The blocks below are stacks of MLA ("M") and GPT ("G") layers that share one config.
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
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

_S, _B, _H = 64, 2, 128
_Q_LORA_RANK = 64

# (layer pattern, q_lora_rank, qk_layernorm). The DeepSeek-V3 style MLA layer applies its
# query and KV latent norms in its up projections. Without q_lora_rank and qk_layernorm, MLA's
# linear_q_proj all-gathers the normalized hidden state itself.
_STACKS = {
    "mla": ("MM", _Q_LORA_RANK, True),
    "mla_no_q_lora": ("MM", None, False),
    "gpt": ("GG", None, True),
    "mixed": ("GMG", _Q_LORA_RANK, True),
}

# (tp, disable_nvls) for each path of the row-parallel layer: no reduce-scatter at TP=1, the
# NCCL reduce-scatter when the Triton NVLS kernels are disabled, and the NVLS reduce-scatter,
# fused with the next layer's norm where the wiring allows.
_PATHS = {"tp1": (1, False), "tp2_nccl": (2, True), "tp2_nvls": (2, False)}

# The residual is N(0, 1) and the sub-layers add a fraction of its norm, so a dropped residual
# is far outside this tolerance; it covers bf16 rounding at |x| ~ 4.
_ATOL = 2e-2
_RTOL = 2e-2


def _build(stack: str, tp: int, *, fuse: bool, disable_nvls: bool = False) -> TransformerBlock:
    """Smallest inference-optimized block of the stack's layers, without a final norm, on the
    GPU."""
    pattern, q_lora_rank, qk_layernorm = _STACKS[stack]
    config = MLATransformerConfig(
        num_layers=len(pattern),
        hidden_size=_H,
        ffn_hidden_size=256,
        num_attention_heads=4,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=64,
        qk_head_dim=64,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        qk_layernorm=qk_layernorm,
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
        inference_disable_triton_nvls_kernels=disable_nvls,
    )
    model_parallel_cuda_manual_seed(123)
    specs = {
        kind: get_gpt_layer_with_inference_spec(
            qk_layernorm=qk_layernorm, multi_latent_attention=kind == "M"
        )
        for kind in set(pattern)
    }
    submodules = TransformerBlockSubmodules(layer_specs=[specs[kind] for kind in pattern])
    block = TransformerBlock(config, submodules, post_layer_norm=False)
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
    @pytest.mark.parametrize("stack", list(_STACKS))
    def test_block_wiring(self, stack, tp):
        """The block builds. Each linear_proj fuses into its own layer's linear_fc1, and each
        linear_fc2 fuses only into a next GPT layer's linear_qkv."""
        self._init(tp)
        pattern = _STACKS[stack][0]
        layers = _build(stack, tp, fuse=True).layers
        for i, layer in enumerate(layers):
            proj, fc1 = layer.self_attention.linear_proj, layer.mlp.linear_fc1
            assert proj.next_layer_norm_weights.data_ptr() == fc1.layer_norm_weight.data_ptr()
            assert fc1.fused_tp_handoff is proj.fused_tp_handoff

            if pattern[i] == "G":
                qkv = layer.self_attention.linear_qkv
                norm = layer.get_qkv_layer_norm_weights()
                assert norm.data_ptr() == qkv.layer_norm_weight.data_ptr()
                prev_fc2 = layers[i - 1].mlp.linear_fc2 if i > 0 else None
                assert qkv.fused_tp_handoff is getattr(prev_fc2, "fused_tp_handoff", None)
            else:
                assert layer.get_qkv_layer_norm_weights() is None

            fc2_norm = layer.mlp.linear_fc2.next_layer_norm_weights
            if pattern[i + 1 : i + 2] == "G":
                next_qkv = layers[i + 1].self_attention.linear_qkv
                assert fc2_norm.data_ptr() == next_qkv.layer_norm_weight.data_ptr()
            else:
                # Last layer, or an MLA layer next: reduce-scatter and residual add only.
                assert fc2_norm is None

    @pytest.mark.parametrize("path", list(_PATHS))
    @pytest.mark.parametrize("stack", list(_STACKS))
    def test_fused_matches_unfused(self, stack, path):
        """Flag-on output equals flag-off output on every path. With NVLS, the fused kernel
        runs where the next layer's input norm allows, and only there its consumer skips its
        own norm."""
        tp, disable_nvls = _PATHS[path]
        self._init(tp)
        if path == "tp2_nvls":
            _require_nvls()
        pattern, q_lora_rank, _ = _STACKS[stack]

        unfused = _build(stack, tp, fuse=False, disable_nvls=disable_nvls)
        fused = _build(stack, tp, fuse=True, disable_nvls=disable_nvls)
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
            for kind, layer in zip(pattern, block.layers):
                if kind == "M":
                    # Without latent caching, the query and KV up projections all-gather back
                    # to back into the shared symmetric buffer without barrier_before (see
                    # multimem_all_gather), a race of its own that a separate fix closes.
                    # Until then take them through NCCL so that this test covers fused TP
                    # communication only.
                    attention = layer.self_attention
                    q_proj = (
                        attention.linear_q_proj
                        if q_lora_rank is None
                        else attention.linear_q_up_proj
                    )
                    q_proj.triton_nvls_kernels_allowed = False
                    attention.linear_kv_up_proj.triton_nvls_kernels_allowed = False

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

        # With NVLS, every linear_proj fuses into its linear_fc1, and a linear_fc2 fuses only
        # into a next GPT layer's linear_qkv. Every other consumer applies its own norm.
        nvls = path == "tp2_nvls"
        fc2_fusions = pattern[1:].count("G")
        assert fused_spy.call_count == ((len(pattern) + fc2_fusions) if nvls else 0)
        applied = [call.kwargs["weight"] for call in norm_spy.call_args_list]
        for i, (kind, layer) in enumerate(zip(pattern, fused.layers)):
            # (consumer, whether a fused kernel feeds it with NVLS)
            consumers = [(layer.mlp.linear_fc1, True)]
            if kind == "G":
                consumers.append((layer.self_attention.linear_qkv, i > 0))
            for consumer, fusable in consumers:
                applies_norm = any(w is consumer.layer_norm_weight for w in applied)
                where = "linear_fc1" if consumer is layer.mlp.linear_fc1 else "linear_qkv"
                assert applies_norm != (
                    nvls and fusable
                ), f"layer {i} ({kind}) {where} applies its own norm: {applies_norm}"
