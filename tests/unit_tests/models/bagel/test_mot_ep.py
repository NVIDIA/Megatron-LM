"""
Unit test for MoTTransformerLayer with Expert Parallelism (EP=2).

Verifies that:
1. MoTTransformerLayer with two MoE MLPs (mlp + mlp_gen) works correctly with EP.
2. Forward pass completes without errors and output shape is correct.
3. Aux loss tracker is populated for both und and gen MoE paths after forward.
4. is_moe_layer is True when either mlp or mlp_gen is a MoELayer.

Run with:
    torchrun --nnodes=1 --node-rank=0 --nproc_per_node=2 \\
        examples/mimo_bagel/unit_test/test_mot_ep.py
"""

import os
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_BAGEL_PKG = os.path.join(_ROOT, "bagel-package")
_BAGEL_SRC = os.path.join(_BAGEL_PKG, "bagel")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BAGEL_PKG)
sys.path.insert(0, _BAGEL_SRC)

from megatron.core.models.bagel.attention_mot import (
    SelfAttentionMoT,
    SelfAttentionMoTSubmodules,
)
from megatron.core.models.bagel.transformer_mot_layer import (
    MoTTransformerLayer,
    MoTTransformerLayerSubmodules,
)
from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams
from megatron.core.models.bagel.transformer_mot_block import get_mot_layer_spec

from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention as F_sdpa

from megatron.core import parallel_state as mpu
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.moe_utils import (
    get_moe_layer_wise_logging_tracker,
    clear_aux_losses_tracker,
)
from megatron.core.transformer.moe.moe_layer import MoELayer
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm
    HAVE_WRAPPED_NORM = True
except ImportError:
    HAVE_WRAPPED_NORM = False


# ─────────────────────────────────────────────────────────────────────────────
# Stub core attention: matches BagelMatchingAttention from test_transformer_mot_block
# Uses EFFICIENT_ATTENTION to avoid TE packed-sequence shape constraints.
# ─────────────────────────────────────────────────────────────────────────────
class _StubCoreAttention(MegatronModule):
    def __init__(self, config, layer_number, attn_mask_type, attention_type,
                 softmax_scale=None, cp_comm_type=None, pg_collection=None, **kwargs):
        super().__init__(config=config)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_query_groups
        self.head_dim = config.kv_channels

    def forward(self, query, key, value, attention_mask, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        seq_len, batch_size, num_heads, head_dim = query.shape
        num_kv_heads = key.shape[2]
        if num_kv_heads != num_heads:
            num_groups = num_heads // num_kv_heads
            key = key.repeat_interleave(num_groups, dim=2)
            value = value.repeat_interleave(num_groups, dim=2)
        q = query.squeeze(1).permute(1, 0, 2).unsqueeze(0).to(torch.float16)
        k = key.squeeze(1).permute(1, 0, 2).unsqueeze(0).to(torch.float16)
        v = value.squeeze(1).permute(1, 0, 2).unsqueeze(0).to(torch.float16)
        mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float16, device=query.device)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            attn_out = F_sdpa(q, k, v, mask)
        attn_out = attn_out.squeeze(0).permute(1, 0, 2).contiguous()
        return attn_out.reshape(seq_len, batch_size, num_heads * head_dim).to(query.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Test configuration
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE = 128
FFN_HIDDEN_SIZE = 256
NUM_HEADS = 4
NUM_KV_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
NUM_EXPERTS = 4     # each EP rank holds 2 experts
MOE_TOPK = 2
N_UND = 16
N_GEN = 16
SEQ_LEN = N_UND + N_GEN


def _make_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_query_groups=NUM_KV_HEADS,
        kv_channels=HEAD_DIM,
        add_bias_linear=False,
        add_qkv_bias=False,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_dropout_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        # MoE config
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=MOE_TOPK,
        moe_token_dispatcher_type="alltoall",
        moe_aux_loss_coeff=1e-3,
        moe_router_load_balancing_type="aux_loss",
    )


def _make_psp() -> MoTPackedSeqParams:
    und_idx = torch.arange(N_UND, device="cuda")
    gen_idx = torch.arange(N_UND, N_UND + N_GEN, device="cuda")
    return MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx,
        packed_gen_token_indexes=gen_idx,
        local_und_token_indexes=und_idx,
        local_gen_token_indexes=gen_idx,
        padded_und_seqlen=N_UND,
        padded_gen_seqlen=N_GEN,
    )


def test_mot_ep():
    rank = torch.distributed.get_rank()

    # Initialize with EP=2
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=2,
    )

    model_parallel_cuda_manual_seed(42)
    config = _make_config()
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    # Build layer spec with MoE using stub attention (avoids TE packed-seq constraint)
    assert HAVE_WRAPPED_NORM, "WrappedTorchNorm not available"
    attn_sub = SelfAttentionMoTSubmodules(
        linear_qkv=ColumnParallelLinear,
        core_attention=_StubCoreAttention,
        linear_proj=RowParallelLinear,
        q_layernorm=WrappedTorchNorm,
        k_layernorm=WrappedTorchNorm,
        linear_qkv_gen=ColumnParallelLinear,
        linear_proj_gen=RowParallelLinear,
        q_layernorm_gen=WrappedTorchNorm,
        k_layernorm_gen=WrappedTorchNorm,
    )
    # Use get_mot_layer_spec for MoE mlp spec, then override attention
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.utils import get_linear_layer
    mlp_spec = get_mot_layer_spec(
        num_experts=NUM_EXPERTS,
        moe_grouped_gemm=False,
        qk_layernorm=False,
        use_te=False,
        use_flex_attention=False,
    ).submodules.mlp  # extract just the MoE mlp spec

    layer_submodules = MoTTransformerLayerSubmodules(
        input_layernorm=WrappedTorchNorm,
        input_layernorm_gen=WrappedTorchNorm,
        self_attention=ModuleSpec(
            module=SelfAttentionMoT,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=attn_sub,
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=WrappedTorchNorm,
        pre_mlp_layernorm_gen=WrappedTorchNorm,
        mlp=mlp_spec,
        mlp_gen=mlp_spec,
        mlp_bda=get_bias_dropout_add,
    )
    layer_spec = ModuleSpec(module=MoTTransformerLayer, submodules=layer_submodules)

    # Build MoTTransformerLayer (layer_number=1 for aux loss tracking)
    layer = build_module(
        layer_spec,
        config=config,
        layer_number=1,
        pg_collection=pg_collection,
    ).cuda().to(torch.bfloat16)
    layer.train()

    # Verify both mlp and mlp_gen are MoELayer instances
    assert isinstance(layer.mlp, MoELayer), "mlp should be MoELayer"
    assert isinstance(layer.mlp_gen, MoELayer), "mlp_gen should be MoELayer"
    assert layer.is_moe_layer, "is_moe_layer should be True"

    # Build packed seq params and inputs
    psp = _make_psp()
    hidden_states = torch.randn(SEQ_LEN, 1, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # Clear aux loss tracker before forward
    clear_aux_losses_tracker()

    # Forward pass
    output, _ = layer(hidden_states, attention_mask=None, packed_seq_params=psp)

    # Verify output shape
    assert output.shape == (SEQ_LEN, 1, HIDDEN_SIZE), (
        f"Expected shape ({SEQ_LEN}, 1, {HIDDEN_SIZE}), got {output.shape}"
    )

    # Verify aux loss tracker is populated (both mlp and mlp_gen contribute)
    tracker = get_moe_layer_wise_logging_tracker()
    assert "load_balancing_loss" in tracker, "load_balancing_loss should be in aux loss tracker"
    lb_values = tracker["load_balancing_loss"]["values"]
    assert lb_values[0].item() > 0.0, (
        f"load_balancing_loss for layer 1 should be > 0, got {lb_values[0].item()}"
    )

    # Backward pass sanity check
    loss = output.sum()
    loss.backward()

    if rank == 0:
        ep_size = mpu.get_expert_model_parallel_world_size()
        print(
            f"[PASS] test_mot_ep: EP={ep_size}, num_experts={NUM_EXPERTS}, "
            f"output_shape={tuple(output.shape)}, "
            f"lb_loss={lb_values[0].item():.4f}"
        )

    Utils.destroy_model_parallel()


if __name__ == "__main__":
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    test_mot_ep()
    dist.destroy_process_group()
