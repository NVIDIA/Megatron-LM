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
independently. This test verifies the fix by running save TP=N -> load TP=M
and checking the SelfAttention forward output is preserved up to numerical
noise.
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
# splits at TP=2: head_dim=8, QKV_dim=3*4*8=96, H=4, total=100. Naive uniform
# reshard at TP=2 puts the boundary at row 50, deep inside the QKV block.
NUM_HEADS = 4
HIDDEN_SIZE = 32
SEQ_LEN = 4
BATCH_SIZE = 2
INPUT_SEED = 7


def _make_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        # Disable stochastic dropout so save/load forward pass is deterministic
        # across the two TP configs.
        attention_dropout=0.0,
        hidden_dropout=0.0,
        head_wise_attn_gate=True,
    )


def _build_attention(seed: int) -> SelfAttention:
    """Build a SelfAttention layer with the local (non-TE) spec under the
    currently-initialized model-parallel state."""
    model_parallel_cuda_manual_seed(seed)
    submodules = get_gpt_layer_local_spec().submodules.self_attention.submodules
    attn = SelfAttention(_make_config(), submodules, layer_number=1).cuda()
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
    def test_logits_equality_across_tp_reshard(
        self, tmp_path_dist_ckpt, src_tp, dst_tp
    ):
        """Save at src_tp, load at dst_tp, verify SelfAttention's forward
        output matches the source run up to numerical noise.

        The output of SelfAttention.forward is replicated across the TP group
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
        attn_src = _build_attention(seed=123)

        hidden_states, attention_mask = _fixed_input()
        with torch.no_grad():
            output_src, _ = attn_src(hidden_states, attention_mask)
        # Stash to CPU so it survives parallel-state teardown.
        output_src_cpu = output_src.detach().to(torch.float32).cpu()

        with TempNamedDir(
            tmp_path_dist_ckpt / f"head_wise_gate_tp{src_tp}_to_tp{dst_tp}"
        ) as ckpt_dir:
            save(attn_src.sharded_state_dict(prefix=""), ckpt_dir)
            del attn_src
            Utils.destroy_model_parallel()

            # ---- Phase 2: re-init at dst_tp, load ckpt, forward ----
            Utils.initialize_model_parallel(dst_tp, 1)
            # Different init seed so the post-load weights only match if the
            # dist-ckpt load actually populates them.
            attn_dst = _build_attention(seed=999)
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
