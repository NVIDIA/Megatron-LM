# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-backed tests for FlextronRouter.

Covers construction, forward-pass shape/structure for each axis, and
DP-aware Gumbel determinism (same seed + iteration => identical output).

Run with:

    torchrun --nproc_per_node=1 -m pytest tests/unit_tests/elastification/test_hybrid_flex_router.py
"""

from argparse import Namespace

import pytest
import torch

import megatron.elastification.router.hybrid_flex_router as _router_module
import megatron.training as _megatron_training
from megatron.core.transformer import TransformerConfig
from megatron.elastification.router.hybrid_flex_router import FlextronRouter
from tests.unit_tests.test_utilities import Utils


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
        flex_hetero_head=False,
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
        head_int_list=[num_heads, num_heads // 2],
        mamba_int_list=[mamba_num_heads, mamba_num_heads // 2],
        moe_expert_int_list=[num_moe_experts, num_moe_experts // 2],
        override_selected_budget=None,
    )
    for k, v in flex_fields.items():
        setattr(config, k, v)
    return config


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
            curr_iteration=0, train_iters=1000, train_samples=1000, global_batch_size=1
        )
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
        _router_module.get_current_global_batch_size = self._orig_gbs
        _router_module.get_micro_batch_size = self._orig_mbs
        Utils.destroy_model_parallel()

    def test_construction(self):
        config = _router_config()
        router = FlextronRouter(config).cuda()
        # Each gate is a Sequential of two linear layers + activation.
        assert hasattr(router, "gate_mlp")
        assert hasattr(router, "gate_emb")
        assert hasattr(router, "gate_mamba")
        assert hasattr(router, "gate_head")
        assert hasattr(router, "gate_moe_expert")
        # Skipping was disabled in the config.
        assert not hasattr(router, "gate_skip_layer")

    def test_router_params_marked_for_pp_sync(self):
        config = _router_config()
        router = FlextronRouter(config).cuda()
        for p in router.parameters():
            # _mark_router_params_for_pp_sync adds this attribute to every
            # trainable parameter so the PP gradient sync picks them up.
            assert getattr(p, "pipeline_parallel", False) is True

    def test_forward_returns_six_axis_outputs(self):
        config = _router_config()
        router = FlextronRouter(config).cuda()
        out = router(1.0)
        assert len(out) == 6
        # Order (per hybrid_flex_router.forward):
        # (mlp, skipping, emb, mamba, head, moe_expert)
        mlp, skipping, emb, mamba, head, moe_expert = out
        # Skipping is None when add_skipping=False.
        assert skipping is None
        # Each axis output is a (logits, choice) tuple.
        for axis in (mlp, emb, mamba, head, moe_expert):
            assert isinstance(axis, tuple) and len(axis) == 2

    def test_emb_output_shape_matches_choice_count(self):
        config = _router_config()
        router = FlextronRouter(config).cuda()
        _, _, emb, _, _, _ = router(1.0)
        logits, choice = emb
        # Logits have one entry per emb_int_list choice.
        assert logits.numel() == len(config.emb_int_list)
        assert choice in config.emb_int_list

    def test_gumbel_determinism(self):
        """Two routers at the same config + iteration + fwd_pass_count should
        produce identical Gumbel-softmax samples."""
        config = _router_config()
        config.curr_iteration = 0

        router_a = FlextronRouter(config).cuda()
        router_b = FlextronRouter(config).cuda()
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
        router = FlextronRouter(config).cuda()
        assert router.fwd_pass_count == 0
        router(1.0)
        assert router.fwd_pass_count == 1
        router(1.0)
        assert router.fwd_pass_count == 2

    def test_different_iterations_give_different_samples(self):
        """Bumping curr_iteration changes the Gumbel seed; logits should differ."""
        config = _router_config()
        router = FlextronRouter(config).cuda()

        # Iteration 0 via the default setup-method stub.
        out_iter_0 = router(1.0)

        # Swap the stub to return iteration 100, reset fwd_pass_count so
        # that is the only thing that varies. train_iters must stay >=
        # curr_iteration (matches setup-method stub length).
        _megatron_training.get_args = lambda: Namespace(
            curr_iteration=100, train_iters=1000, train_samples=1000, global_batch_size=1
        )
        router.fwd_pass_count = 0
        out_iter_100 = router(1.0)

        # Emb-axis logits should differ between iterations.
        _, _, emb_0, _, _, _ = out_iter_0
        _, _, emb_100, _, _, _ = out_iter_100
        assert not torch.allclose(emb_0[0], emb_100[0])
