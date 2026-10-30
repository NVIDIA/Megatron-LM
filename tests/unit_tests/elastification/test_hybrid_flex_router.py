# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-backed tests for FlextronRouter.

Covers construction, forward-pass shape/structure for each axis, and
DP-aware Gumbel determinism (same seed + iteration => identical output).
The weight-sync tests also run on CPU with gloo and need 4 ranks.

Run with:

    torchrun --nproc_per_node=1 -m pytest tests/unit_tests/elastification/test_hybrid_flex_router.py
"""

from argparse import Namespace

import pytest
import torch

import megatron.elastification.router.hybrid_flex_router as _router_module
import megatron.training as _megatron_training
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_pg_rank
from megatron.elastification.router.hybrid_flex_router import FlextronRouter
from tests.unit_tests.test_utilities import Utils, forbid_global_process_groups


def _router_config(
    hidden_size=256, ffn_hidden_size=128, num_heads=8, mamba_num_heads=8, num_moe_experts=8
):
    """Build a TransformerConfig with every attr FlextronRouter reads."""
    config = TransformerConfig(
        hidden_size=hidden_size,
        num_layers=2,
        num_attention_heads=num_heads,
        ffn_hidden_size=ffn_hidden_size,
        num_moe_experts=num_moe_experts,
        use_cpu_initialization=True,
    )
    flex_fields = dict(
        flextron=True,
        soft_mask=True,
        add_skipping=False,
        flex_hetero_ffn=False,
        flex_hetero_mamba=False,
        flex_hetero_moe_expert=False,
        hybrid_layer_pattern="ME",
        normalize_router_logits=False,
        router_inter_dim=32,
        router_std=0.1,
        router_gbs=2,
        router_beta=1.0,
        loss_alpha=1.0,
        tau_init=1.0,
        tau_decay=0.9999,
        hard_sample_th=0.996,
        # Enable the scaler with a constant 1.0 so `scale` is defined inside
        # the axis forwards (they use it unconditionally) but its value is a
        # no-op. The get_args stub in setup_method supplies train_iters so
        # add_scaler_schedule can construct the linspace.
        linear_scaler_start=1.0,
        linear_scaler_end=1.0,
        budget_list=[1.0, 0.5],
        budget_probs=[1.0, 1.0],
        budget_type="param",
        original_model_sample_prob=0.0,
        curr_iteration=0,
        mamba_num_heads=mamba_num_heads,
        emb_int_list=[hidden_size, hidden_size // 2],
        mlp_int_list=[ffn_hidden_size, ffn_hidden_size // 2],
        mamba_int_list=[mamba_num_heads, mamba_num_heads // 2],
        moe_expert_int_list=[num_moe_experts, num_moe_experts // 2],
        override_selected_budget=None,
    )
    for k, v in flex_fields.items():
        setattr(config, k, v)
    return config


def _router(config):
    """Build a router on the groups of a model that lives on the global grid."""
    return FlextronRouter(config, pg_collection=ProcessGroupCollection.use_mpu_process_groups())


@pytest.mark.internal
class TestFlextronRouter:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        # FlextronRouter calls _sync_router_weights in __init__, which does an
        # NCCL broadcast on CPU params before we get a chance to call .cuda().
        # NCCL has no CPU backend, so we stub the sync out — it's a no-op at
        # world_size=1 anyway. Restored in teardown.
        self._orig_sync = FlextronRouter._sync_router_weights
        FlextronRouter._sync_router_weights = lambda self: None
        # forward() pulls `args.curr_iteration` via megatron.training.get_args()
        # which fails outside a full Megatron initialize. Install a minimal
        # stub that returns the attrs the router needs.
        self._orig_get_args = _megatron_training.get_args
        # train_iters needs to be >= max curr_iteration used in any test: the
        # scaler is a linspace of length train_iters and axis forwards index
        # it with curr_iteration. Since start=end=1.0 the values are all 1.0
        # regardless of length, so overshooting is free.
        _megatron_training.get_args = lambda: Namespace(
            train_iters=1000, train_samples=1000, global_batch_size=1
        )
        self._orig_get_train_state = _megatron_training.get_train_state
        _megatron_training.get_train_state = lambda: Namespace(iteration=0)
        # The router's __init__ also reads global microbatch state via
        # get_current_global_batch_size / get_micro_batch_size. These return
        # None outside a full Megatron initialize — stub them at the module
        # level where the router imported them.
        self._orig_gbs = _router_module.get_current_global_batch_size
        self._orig_mbs = _router_module.get_micro_batch_size
        _router_module.get_current_global_batch_size = lambda: 1
        _router_module.get_micro_batch_size = lambda: 1

    def teardown_method(self, method):
        FlextronRouter._sync_router_weights = self._orig_sync
        _megatron_training.get_args = self._orig_get_args
        _megatron_training.get_train_state = self._orig_get_train_state
        _router_module.get_current_global_batch_size = self._orig_gbs
        _router_module.get_micro_batch_size = self._orig_mbs
        Utils.destroy_model_parallel()

    def test_construction(self):
        config = _router_config()
        router = _router(config).cuda()
        # Each gate is a Sequential of two linear layers + activation.
        assert hasattr(router, "gate_mlp")
        assert hasattr(router, "gate_emb")
        assert hasattr(router, "gate_mamba")
        assert hasattr(router, "gate_moe_expert")
        # Attention head elasticity is not supported.
        assert not hasattr(router, "gate_head")
        # Skipping was disabled in the config.
        assert not hasattr(router, "gate_skip_layer")

    def test_router_params_marked_for_pp_sync(self):
        config = _router_config()
        router = _router(config).cuda()
        for p in router.parameters():
            # _mark_router_params_for_pp_sync adds this attribute to every
            # trainable parameter so the PP gradient sync picks them up.
            assert getattr(p, "flextron_router_pp_sync", False) is True

    def test_forward_returns_five_axis_outputs(self):
        config = _router_config()
        router = _router(config).cuda()
        out = router(1.0)
        assert len(out) == 5
        # Order (per hybrid_flex_router.forward):
        # (mlp, skipping, emb, mamba, moe_expert)
        mlp, skipping, emb, mamba, moe_expert = out
        # Skipping is None when add_skipping=False.
        assert skipping is None
        # Each axis output is a (logits, choice) tuple.
        for axis in (mlp, emb, mamba, moe_expert):
            assert isinstance(axis, tuple) and len(axis) == 2

    def test_emb_output_shape_matches_choice_count(self):
        config = _router_config()
        router = _router(config).cuda()
        _, _, emb, _, _ = router(1.0)
        logits, choice = emb
        # Logits have one entry per emb_int_list choice.
        assert logits.numel() == len(config.emb_int_list)
        assert choice in config.emb_int_list

    @pytest.mark.flaky_in_dev
    def test_gumbel_determinism(self):
        """Two routers at the same config + iteration + fwd_pass_count should
        produce identical Gumbel-softmax samples.

        Quarantined as flaky_in_dev (#5155): ``FlextronRouter.forward`` draws
        ``hard_sample`` from Python's un-seeded global ``random``, which is not
        covered by the Gumbel ``torch.manual_seed``, so the two routers
        intermittently disagree on the hard/soft branch and the bit-exact
        ``assert_close`` fails.
        """
        config = _router_config()
        config.curr_iteration = 0

        router_a = _router(config).cuda()
        router_b = _router(config).cuda()
        # Copy weights so both routers are in the same parameter state; the
        # determinism check is about the Gumbel RNG, not init noise.
        router_b.load_state_dict(router_a.state_dict())

        out_a = router_a(1.0)
        out_b = router_b(1.0)
        for axis_a, axis_b in zip(out_a, out_b):
            if axis_a is None:
                assert axis_b is None
                continue
            logits_a, choice_a = axis_a
            logits_b, choice_b = axis_b
            torch.testing.assert_close(logits_a, logits_b, atol=0, rtol=0)
            assert choice_a == choice_b

    def test_fwd_pass_count_increments(self):
        config = _router_config()
        router = _router(config).cuda()
        assert router.fwd_pass_count == 0
        router(1.0)
        assert router.fwd_pass_count == 1
        router(1.0)
        assert router.fwd_pass_count == 2

    def test_different_iterations_give_different_samples(self):
        """Bumping curr_iteration changes the Gumbel seed; logits should differ."""
        config = _router_config()
        router = _router(config).cuda()

        # Iteration 0 via the default setup-method stub.
        out_iter_0 = router(1.0)

        # Swap the stub to return iteration 100, reset fwd_pass_count so
        # that is the only thing that varies. train_iters must stay >=
        # curr_iteration (matches setup-method stub length).
        _megatron_training.get_args = lambda: Namespace(
            train_iters=1000, train_samples=1000, global_batch_size=1
        )
        _megatron_training.get_train_state = lambda: Namespace(iteration=100)
        router.fwd_pass_count = 0
        out_iter_100 = router(1.0)

        # Emb-axis logits should differ between iterations.
        _, _, emb_0, _, _ = out_iter_0
        _, _, emb_100, _, _ = out_iter_100
        assert not torch.allclose(emb_0[0], emb_100[0])


def _stub_router_training_state(monkeypatch):
    """Stub the training state that FlextronRouter.__init__ reads outside a Megatron run."""
    monkeypatch.setattr(_router_module, "get_current_global_batch_size", lambda: 1)
    monkeypatch.setattr(_router_module, "get_micro_batch_size", lambda: 1)
    monkeypatch.setattr(
        _megatron_training,
        "get_args",
        lambda: Namespace(train_iters=10, train_samples=10, global_batch_size=1),
    )


def _router_with_rank_weights(monkeypatch, pg_collection):
    """Build a router without its weight sync, then fill every weight with this global rank.

    The router uses the local tensor-parallel linear layers, which also build without
    Transformer Engine and on CPU.
    """
    monkeypatch.setattr(_router_module, "HAVE_TE", False)
    monkeypatch.setattr(_router_module, "ColumnParallelLinear", ColumnParallelLinear, raising=False)
    monkeypatch.setattr(_router_module, "RowParallelLinear", RowParallelLinear, raising=False)
    with monkeypatch.context() as patch:
        patch.setattr(FlextronRouter, "_sync_router_weights", lambda self: None)
        router = FlextronRouter(_router_config(), pg_collection=pg_collection)
    if torch.cuda.is_available():
        router.cuda()
    with torch.no_grad():
        for param in router.parameters():
            param.fill_(torch.distributed.get_rank())
    return router


def _source_rank(router):
    """Return the global rank whose weights the router holds, checking that it is a single one."""
    values = torch.cat([param.detach().flatten() for param in router.parameters()]).unique()
    assert values.numel() == 1, f"router weights come from several ranks: {values.tolist()}"
    return int(values.item())


@pytest.mark.internal
@pytest.mark.parametrize("pipeline_model_parallel_size", [1, 2])
def test_router_weights_are_synced_per_tensor_parallel_shard(
    monkeypatch, pipeline_model_parallel_size
):
    """Each TP shard comes from the lowest of its PP x DP-CP replicas, not from global rank 0."""
    tp_size = 2
    model_parallel_size = tp_size * pipeline_model_parallel_size
    if Utils.world_size < 4 or Utils.world_size % model_parallel_size:
        pytest.skip(f"needs a world size of at least 4 that {model_parallel_size} divides")
    _stub_router_training_state(monkeypatch)
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        router = _router_with_rank_weights(monkeypatch, pg_collection)

        router._sync_router_weights()

        holders = [None] * Utils.world_size
        torch.distributed.all_gather_object(
            holders, (get_pg_rank(pg_collection.tp), _source_rank(router))
        )
        for tp_rank in range(tp_size):
            replicas = [
                rank for rank, (held_tp_rank, _) in enumerate(holders) if held_tp_rank == tp_rank
            ]
            # Every replica of the shard holds the weights of the lowest replica.
            assert {holders[rank][1] for rank in replicas} == {min(replicas)}
        # The TP ranks keep different shards.
        assert len({source for _, source in holders}) == tp_size
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.internal
def test_router_weights_sync_stays_inside_the_model_groups(monkeypatch):
    """Two models that share a job keep their own router weights.

    Each pair of ranks (2k, 2k + 1) is one model with two data-parallel replicas, unlike the
    global grid, whose data-parallel group spans the whole job.
    """
    if Utils.world_size < 4 or Utils.world_size % 2:
        pytest.skip("needs an even world size of at least 4")
    _stub_router_training_state(monkeypatch)
    Utils.initialize_model_parallel()
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp, _ = torch.distributed.new_subgroups_by_enumeration(
            [[rank, rank + 1] for rank in range(0, Utils.world_size, 2)]
        )
        router = _router_with_rank_weights(monkeypatch, pg_collection)

        with forbid_global_process_groups():
            router._sync_router_weights()

        rank = torch.distributed.get_rank()
        assert _source_rank(router) == rank - rank % 2
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("missing", ["pp", "dp_cp"])
def test_router_rejects_a_collection_without_its_sync_groups(missing):
    groups = dict(tp=None, pp=None, dp_cp=None, dp_gtp_remat=None)
    del groups[missing]

    with pytest.raises(ValueError, match=f"FlextronRouter requires pg_collection to set {missing}"):
        FlextronRouter(_router_config(), pg_collection=ProcessGroupCollection(**groups))
