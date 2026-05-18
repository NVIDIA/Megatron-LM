# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""End-to-end dist-ckpt regression test for head_wise_attn_gate.

The forward path of SelfAttention peels the trailing
num_attention_heads_per_partition rows off each rank's local linear_qkv tensor
and treats them as per-head gate scalars. Default axis-0 sharding metadata
({"weight": 0}) would re-balance the global (QKV_dim + H, hidden_size) tensor
uniformly under a TP change, so the trailing rows on non-last new ranks land
inside the QKV block -- silently turning V/K rows into gate scalars without a
shape error. SelfAttention.sharded_state_dict installs a ShardedTensorFactory
that registers QKV and gate as independent sub-tensors so each is resharded
independently. These tests verify the fix by running save TP=N -> load TP=M
and checking that the SelfAttention forward output is preserved up to
numerical noise.

Coverage matrix (this file):
  - (src_tp, dst_tp) in (1->2) and (2->1).
  - num_query_groups in (2, 4) at NUM_HEADS=4, exercising both GQA
    (num_query_groups < num_attention_heads) and MHA
    (num_query_groups == num_attention_heads).
  - add_qkv_bias in (True, False), since the bias gate-row surgery in
    SelfAttention.__init__ only fires when bias is present.

Known unverified paths (acknowledged as follow-ups):

  - context_parallel_size > 1: head_wise gate slicing happens BEFORE any
    sequence-parallel / CP split (the slice is on the output-feature dim,
    not the sequence dim), so it should compose, but no test pins it.
  - fp8: linear_qkv shares a single tensor-level FP8 scale across the QKV
    and gate sub-blocks. The init-time `head_wise_attn_gate_init_weight_scale`
    default of 0.1 keeps the magnitudes within ~1 decade of QKV, but no
    fp8 end-to-end run exists. set_save_original_input(self.linear_qkv)
    interaction (selective recompute under fp8) is also un-tested.
  - backward gradient flow to the gate rows: forward output equality
    implies correct param values were loaded, but autograd through the
    gate is not asserted explicitly. Easy to add but not done here.
  - packed_seq_params.qkv_format == 'thd': core_attn_out is reshaped to
    [t, 1, h] just before the gate is applied; the
    `core_attn_out.view(*gate_states.shape[:3], -1)` chain composes with
    that reshape, but no test runs the THD path.
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


# Sized so that QKV_dim / gate boundary does NOT line up with uniform axis-0
# splits at TP=2: with the default MHA case (num_query_groups=4), QKV_dim=96
# and H=4, total=100; a naive reshard at TP=2 puts the boundary at row 50,
# deep inside the QKV block.
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
        # Disable stochastic dropout so save/load forward pass is deterministic
        # across the two TP configs.
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_qkv_bias=add_qkv_bias,
        head_wise_attn_gate=True,
    )


def _build_attention(seed: int, num_query_groups: int, add_qkv_bias: bool) -> SelfAttention:
    """Build a SelfAttention layer with the local (non-TE) spec under the
    currently-initialized model-parallel state."""
    model_parallel_cuda_manual_seed(seed)
    submodules = get_gpt_layer_local_spec().submodules.self_attention.submodules
    config = _make_config(num_query_groups=num_query_groups, add_qkv_bias=add_qkv_bias)
    attn = SelfAttention(config, submodules, layer_number=1).cuda()
    attn.eval()
    return attn


def _fixed_input():
    """Deterministic input shared across all ranks so different TP setups see
    the same activations."""
    torch.manual_seed(INPUT_SEED)
    hidden_states = torch.randn(
        SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float32, device="cuda"
    )
    attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN, dtype=bool, device="cuda")
    return hidden_states, attention_mask


class TestHeadWiseAttnGateTPResharding:
    """Regression test for apply_head_wise_attn_gate_sharded_factory."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("src_tp,dst_tp", [(1, 2), (2, 1)])
    @pytest.mark.parametrize(
        "num_query_groups",
        [
            # MHA: gate rows aligned 1:1 with q heads, both = NUM_HEADS.
            NUM_HEADS,
            # GQA: 2 q heads per group; gate rows still 1:1 with q heads,
            # but the QKV layout per group is heavier (q1 q2 k v) so the
            # uniform-reshard boundary lands in different sub-block
            # positions than MHA.
            NUM_HEADS // 2,
        ],
    )
    @pytest.mark.parametrize("add_qkv_bias", [False, True])
    def test_logits_equality_across_tp_reshard(
        self,
        tmp_path_dist_ckpt,
        src_tp,
        dst_tp,
        num_query_groups,
        add_qkv_bias,
    ):
        """Save at src_tp, load at dst_tp, verify SelfAttention's forward
        output matches the source run up to numerical noise. Parametrized
        over reshard direction, num_query_groups (MHA / GQA), and bias
        presence, since:

          - GQA changes the per-group QKV layout (q-block size) but not the
            gate-row count, so the resharding factory must split at the
            correct global axis even when QKV/gate ratios differ from MHA.
          - add_qkv_bias=True exercises the linear_qkv.bias factory path
            (1D tensor, separate from weight) AND the init-time bias-row
            surgery (head_wise_attn_gate_init_bias=2.0), so a buggy
            resharding of the bias would break sigmoid(gate) at the load
            side.

        SelfAttention.forward's output is replicated across the TP group
        (linear_proj is RowParallelLinear and all-reduces), so rank 0's
        output on both sides is directly comparable.
        """
        if Utils.world_size < max(src_tp, dst_tp):
            pytest.skip(
                f"need world_size >= {max(src_tp, dst_tp)} "
                f"(got {Utils.world_size})"
            )

        # ---- Phase 1: build at src_tp, forward, save ----
        Utils.initialize_model_parallel(src_tp, 1)
        attn_src = _build_attention(
            seed=123, num_query_groups=num_query_groups, add_qkv_bias=add_qkv_bias
        )

        hidden_states, attention_mask = _fixed_input()
        with torch.no_grad():
            output_src, _ = attn_src(hidden_states, attention_mask)
        # Stash to CPU so it survives parallel-state teardown.
        output_src_cpu = output_src.detach().to(torch.float32).cpu()

        ckpt_tag = (
            f"head_wise_gate_tp{src_tp}_to_tp{dst_tp}"
            f"_g{num_query_groups}_bias{int(add_qkv_bias)}"
        )
        with TempNamedDir(tmp_path_dist_ckpt / ckpt_tag) as ckpt_dir:
            save(attn_src.sharded_state_dict(prefix=""), ckpt_dir)
            del attn_src
            Utils.destroy_model_parallel()

            # ---- Phase 2: re-init at dst_tp, load ckpt, forward ----
            Utils.initialize_model_parallel(dst_tp, 1)
            # Different init seed so the post-load weights only match if the
            # dist-ckpt load actually populates them.
            attn_dst = _build_attention(
                seed=999, num_query_groups=num_query_groups, add_qkv_bias=add_qkv_bias
            )
            state_dict, missing_keys, unexpected_keys = load(
                attn_dst.sharded_state_dict(prefix=""),
                ckpt_dir,
                strict=StrictHandling.RETURN_ALL,
            )
            # Any mismatches should only be TE-style _extra_state entries; the
            # local backend doesn't have any, but assert defensively.
            assert all('_extra_state' in k for k in missing_keys), missing_keys
            assert all('_extra_state' in k for k in unexpected_keys), unexpected_keys
            attn_dst.load_state_dict(state_dict)

            hidden_states, attention_mask = _fixed_input()
            with torch.no_grad():
                output_dst, _ = attn_dst(hidden_states, attention_mask)
            output_dst_cpu = output_dst.detach().to(torch.float32).cpu()

        # Output is identical on every TP rank post-AllReduce; compare on
        # rank 0 to avoid duplicate work.
        if torch.distributed.get_rank() == 0:
            torch.testing.assert_close(
                output_dst_cpu, output_src_cpu, atol=1e-4, rtol=1e-4
            )
