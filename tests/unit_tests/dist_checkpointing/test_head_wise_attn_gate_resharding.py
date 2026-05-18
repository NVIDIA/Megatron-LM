# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end dist-ckpt regression test for head_wise_attn_gate.

Parametrized over (src_tp, dst_tp) in {(1,2), (2,1)}, num_query_groups in
{NUM_HEADS, NUM_HEADS//2} (MHA + GQA), and add_qkv_bias in {True, False}.
"""

import pytest
import torch

from megatron.core.dist_checkpointing import load, save
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


NUM_HEADS = 4
HIDDEN_SIZE = 32
SEQ_LEN = 4
BATCH_SIZE = 2
INPUT_SEED = 7


def _make_config(num_query_groups: int, add_qkv_bias: bool) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_query_groups=num_query_groups,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_qkv_bias=add_qkv_bias,
        head_wise_attn_gate=True,
    )


def _build_attention(seed: int, num_query_groups: int, add_qkv_bias: bool) -> SelfAttention:
    model_parallel_cuda_manual_seed(seed)
    submodules = get_gpt_layer_local_spec().submodules.self_attention.submodules
    config = _make_config(num_query_groups=num_query_groups, add_qkv_bias=add_qkv_bias)
    attn = SelfAttention(config, submodules, layer_number=1).cuda()
    attn.eval()
    return attn


def _fixed_input():
    torch.manual_seed(INPUT_SEED)
    hidden_states = torch.randn(
        SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float32, device="cuda"
    )
    attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN, dtype=bool, device="cuda")
    return hidden_states, attention_mask


class TestHeadWiseAttnGateTPResharding:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("src_tp,dst_tp", [(1, 2), (2, 1)])
    @pytest.mark.parametrize("num_query_groups", [NUM_HEADS, NUM_HEADS // 2])
    @pytest.mark.parametrize("add_qkv_bias", [False, True])
    def test_logits_equality_across_tp_reshard(
        self, tmp_path_dist_ckpt, src_tp, dst_tp, num_query_groups, add_qkv_bias
    ):
        if Utils.world_size < max(src_tp, dst_tp):
            pytest.skip(f"need world_size >= {max(src_tp, dst_tp)}")

        Utils.initialize_model_parallel(src_tp, 1)
        attn_src = _build_attention(
            seed=123, num_query_groups=num_query_groups, add_qkv_bias=add_qkv_bias
        )

        hidden_states, attention_mask = _fixed_input()
        with torch.no_grad():
            output_src, _ = attn_src(hidden_states, attention_mask)
        output_src_cpu = output_src.detach().to(torch.float32).cpu()

        ckpt_tag = (
            f"head_wise_gate_tp{src_tp}_to_tp{dst_tp}"
            f"_g{num_query_groups}_bias{int(add_qkv_bias)}"
        )
        with TempNamedDir(tmp_path_dist_ckpt / ckpt_tag) as ckpt_dir:
            save(attn_src.sharded_state_dict(prefix=""), ckpt_dir)
            del attn_src
            Utils.destroy_model_parallel()

            Utils.initialize_model_parallel(dst_tp, 1)
            attn_dst = _build_attention(
                seed=999, num_query_groups=num_query_groups, add_qkv_bias=add_qkv_bias
            )
            state_dict, missing_keys, unexpected_keys = load(
                attn_dst.sharded_state_dict(prefix=""),
                ckpt_dir,
                strict=StrictHandling.RETURN_ALL,
            )
            assert all('_extra_state' in k for k in missing_keys), missing_keys
            assert all('_extra_state' in k for k in unexpected_keys), unexpected_keys
            attn_dst.load_state_dict(state_dict)

            hidden_states, attention_mask = _fixed_input()
            with torch.no_grad():
                output_dst, _ = attn_dst(hidden_states, attention_mask)
            output_dst_cpu = output_dst.detach().to(torch.float32).cpu()

        if torch.distributed.get_rank() == 0:
            torch.testing.assert_close(
                output_dst_cpu, output_src_cpu, atol=1e-4, rtol=1e-4
            )
