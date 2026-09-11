# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""End-to-end parity for layer-sharded Muon through the real training-loop path.

Runs ``setup_model_and_optimizer`` on a TP(2) x GTP_remat(2) grid and checks that the
loss trajectory of ``--muon-tp-mode layer_sharded`` (the two-stage exchange over a
genuine GTP_remat x TP domain) is bitwise identical to ``duplicated`` mode.

Launch with 4 ranks:
  torchrun --nproc-per-node=4 -m pytest tests/unit_tests/optimizer/test_layer_sharded_e2e_parity.py
"""

import pytest
import torch

pytest.importorskip("emerging_optimizers", reason="LayerShardedMuon requires emerging-optimizers")

from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon
from tests.unit_tests.test_fp8_param import TestFP8Param as _Harness
from tests.unit_tests.test_utilities import Utils

# Plain bf16 run: the harness defaults to FP8, which is orthogonal here. ``fp8=None`` turns
# it off; the harness's positional ``recipe`` must then not be "mxfp8", because validate_args
# ties GTP + mxfp8 to --fp8-param-gather.
_RECIPE = "tensorwise"
_COMMON = dict(
    fp8=None,
    num_steps=8,
    num_layers=2,
    padded_vocab_size=512,
    hidden_size=256,
    num_attention_heads=8,
    ffn_hidden_size=512,
    global_batch_size=2,
    optimizer="muon",
    muon_scalar_optimizer="adam",
    muon_momentum=0.9,
    muon_scale_mode="spectral",
    muon_num_ns_steps=5,
    muon_coefficient_type="quintic",
    muon_split_qkv=False,
    lr=1e-3,
    clip_grad=0.0,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    # 4 shards per weight over TP 2 => GTP_remat 2: a genuinely 2-D (GTP_remat x TP) domain.
    tensor_parallel_num_weight_shards=4,
    untie_embeddings_and_output_weights=True,
    overlap_param_gather=False,
    overlap_grad_reduce=False,
)

_MOE = dict(
    num_experts=2,
    moe_grouped_gemm=True,
    moe_single_grouped_weight=False,
    moe_ffn_hidden_size=512,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=2,
    expert_tensor_parallel_num_weight_shards=4,
    moe_token_dispatcher_type="alltoall",
    moe_router_topk=1,
    moe_router_pre_softmax=True,
    moe_router_load_balancing_type="none",
    moe_aux_loss_coeff=0.0,
    add_bias_linear=False,
)


class _LayerShardedHarness(_Harness):
    """Records, right after construction, which optimizer instances and exchange paths exist."""

    def _on_model_built(self, model_chunks, optimizer, args):
        found = []

        def visit(opt):
            if isinstance(opt, LayerShardedMuon):
                found.append(opt)
                return
            children = getattr(opt, 'chained_optimizers', None)
            if children:
                # ChainedOptimizer (incl. LayerWiseDistributedOptimizer): its ``optimizer``
                # property asserts a single child, so recurse into the children instead.
                for child in children:
                    visit(child)
                return
            # Float16OptimizerWithFloat16Params / FP32Optimizer store the base optimizer as
            # a plain attribute; vars() avoids triggering any property.
            inner = vars(opt).get('optimizer')
            if inner is not None and inner is not opt:
                visit(inner)

        visit(optimizer)
        wired = [pgs for o in found for pgs in o._group_process_groups.values()]
        self.layer_sharded_instances = len(found)
        self.wired_groups = len(wired)
        # A domain is 2-D only when both axis groups are non-trivial.
        self.two_d_groups = sum(
            1
            for pgs in wired
            if pgs[0] is not None and pgs[0].size() > 1 and pgs[1] is not None and pgs[1].size() > 1
        )


def _assert_bitwise_across_ranks(loss_ref, loss_test, label):
    local = torch.stack((loss_ref, loss_test)).cuda()
    gathered = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, local)
    for rank, pair in enumerate(gathered):
        finite = bool(torch.isfinite(pair).all())
        assert finite, f"{label}: non-finite loss on rank {rank}: {pair.tolist()}"
        max_diff = (pair[0] - pair[1]).abs().max().item()
        assert torch.equal(pair[0], pair[1]), (
            f"{label}: rank {rank} diverges, max |diff|={max_diff:.3e}\n"
            f"  duplicated   : {pair[0].tolist()}\n  layer_sharded: {pair[1].tolist()}"
        )


def _run_pair(extra, label):
    if Utils.world_size != 4:
        pytest.skip("Requires exactly 4 torchrun ranks for TP2 x GTP2")
    harness = _LayerShardedHarness()
    harness.setup_method(None)
    harness.seq_length = 64
    harness.micro_batch_size = 1
    try:
        common = {**_COMMON, **extra}
        loss_dup = harness._run_test_helper(
            tp_size=2, recipe=_RECIPE, fp8_param_gather=False, muon_tp_mode="duplicated", **common
        )
        assert harness.layer_sharded_instances == 0
        loss_lsh = harness._run_test_helper(
            tp_size=2,
            recipe=_RECIPE,
            fp8_param_gather=False,
            muon_tp_mode="layer_sharded",
            **common,
        )
        assert harness.layer_sharded_instances >= 1, "no LayerShardedMuon was constructed"
        assert harness.wired_groups > 0, "the layer-wise wiring assigned no NS homes"
        assert harness.two_d_groups == harness.wired_groups, (
            f"only {harness.two_d_groups} of {harness.wired_groups} wired groups span a 2-D "
            "(GTP_remat x TP) domain; the test grid is not what it claims"
        )
        _assert_bitwise_across_ranks(loss_dup, loss_lsh, label)
        if torch.distributed.get_rank() == 0:
            print(
                f"[{label}] {len(loss_dup)} steps bitwise identical; LayerShardedMuon "
                f"instances={harness.layer_sharded_instances}, two-stage groups="
                f"{harness.wired_groups}, losses={loss_lsh.tolist()}"
            )
    finally:
        harness.teardown_method(None)
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import update_gtp_config

        update_gtp_config(pad_for_alignment=16, calculate_per_token_loss=False)


@pytest.mark.launch_on_gb200
def test_dense_layer_sharded_matches_duplicated_end_to_end():
    """Dense GPT, TP2 x GTP2: layer_sharded (two-stage) through the training loop ==
    duplicated, bitwise."""
    _run_pair({}, "dense")


@pytest.mark.launch_on_gb200
def test_moe_layer_sharded_matches_duplicated_end_to_end():
    """GPT-MoE (EP1, ETP2 x EGTP2): layer_sharded (two-stage) == duplicated, bitwise."""
    _run_pair(_MOE, "moe")
