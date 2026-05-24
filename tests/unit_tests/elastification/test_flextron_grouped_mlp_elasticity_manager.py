# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-backed tests for FlextronGroupedMLPElasticityManager.

Covers the multi-hook MLP masking pipeline: setup-mask init, the
input/fc1-post/output hook trio that applies emb + intermediate masking,
and detach. The fc1_post_hook calls into expert-tensor-parallel state,
so we initialize MPU at world_size=1 (mask split is the whole mask).

Run with:
    torchrun --nproc_per_node=1 -m pytest tests/unit_tests/elastification/test_flextron_grouped_mlp_elasticity_manager.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.elastification.flextron_elasticity_hooks import (
    FlextronGroupedMLPElasticityManager,
    add_flextron_grouped_mlp_elasticity,
)
from tests.unit_tests.test_utilities import Utils


def _config(hidden_size=64, ffn_hidden_size=128, soft_mask=True):
    return SimpleNamespace(
        flextron=True,
        soft_mask=soft_mask,
        flex_hetero_ffn=False,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        emb_int_list=[hidden_size, hidden_size // 2],
        mlp_int_list=[ffn_hidden_size, ffn_hidden_size // 2],
        hybrid_layer_pattern="E",
        layernorm_epsilon=1e-5,
    )


class _StubGroupedMLP(nn.Module):
    """Minimal module exposing the surface attach_hooks needs:
    - register_forward_*_hook (inherited from nn.Module)
    - a ``linear_fc1`` child (for fc1_post_hook)

    Forward chain mimics a real GroupedMLP: hidden -> fc1 -> ffn-sized
    "intermediate" -> projected back to hidden-sized output. Both stages
    return ``(tensor, None)`` so the hooks see the (out, bias) tuple shape
    they expect."""

    def __init__(self, hidden_size, ffn_hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        # Stash the captured intermediate so tests can inspect what fc1_post_hook
        # produced before the output projection runs.
        self._captured_intermediate = None

        class _FC1(nn.Module):
            def forward(_self, x):
                inter = x.new_zeros(*x.shape[:-1], ffn_hidden_size)
                inter[..., : x.shape[-1]] = x  # plant the input into the lower channels
                return inter, None

        self.linear_fc1 = _FC1()

    def forward(self, hidden_states):
        intermediate, _ = self.linear_fc1(hidden_states)
        # fc1_post_hook may have masked the intermediate before we get here.
        self._captured_intermediate = intermediate.detach().clone()
        # Project back to hidden dim: take the lower hidden_size channels.
        out = intermediate[..., : self.hidden_size].contiguous()
        return (out, None)


@pytest.mark.internal
class TestFlextronGroupedMLPElasticityManager:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _make_module(self, cfg):
        return _StubGroupedMLP(cfg.hidden_size, cfg.ffn_hidden_size).cuda().to(torch.bfloat16)

    def test_attach_registers_expected_hook_count(self):
        cfg = _config()
        mod = self._make_module(cfg)
        mgr = FlextronGroupedMLPElasticityManager(cfg)
        mgr.attach_hooks(mod)
        # setup + input_mask + fc1_post + output_mask + cleanup = 5
        assert len(mgr.hook_handles) == 5
        mgr.detach_hooks()

    def test_init_emb_masks_match_choice_list(self):
        cfg = _config(hidden_size=64)
        mgr = FlextronGroupedMLPElasticityManager(cfg)
        mgr._init_embedding_masks()
        # One mask per emb_int_list entry, each shape == [hidden_size].
        assert mgr.emb_masks.shape == (len(cfg.emb_int_list), cfg.hidden_size)
        # Mask 0 (full): all-ones over the full hidden dim.
        torch.testing.assert_close(
            mgr.emb_masks[0], torch.ones(cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
        )
        # Mask 1 (half): ones on lower half, zeros on upper half.
        expected = torch.zeros(cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
        expected[: cfg.hidden_size // 2] = 1.0
        torch.testing.assert_close(mgr.emb_masks[1], expected)

    def test_init_mlp_masks_dedupe_and_sort(self):
        """``_init_mlp_masks`` dedupes via set() and sorts descending. Verify
        the lookup maps each unique value to the right index."""
        cfg = _config(ffn_hidden_size=128)
        cfg.mlp_int_list = [128, 128, 64]  # duplicate to exercise dedupe
        mgr = FlextronGroupedMLPElasticityManager(cfg)
        mgr._init_mlp_masks()
        # Two unique values, sorted descending: [128, 64].
        assert mgr.mlp_intermediate_masks.shape[0] == 2
        assert mgr.mlp_intermediate_masks_lookup == {128: 0, 64: 1}

    def test_no_router_emb_is_passthrough(self):
        """With current_router_emb None, no hook should mutate output."""
        cfg = _config()
        mod = self._make_module(cfg)
        x = torch.randn(2, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
        baseline_out, baseline_bias = mod(x)

        mgr = FlextronGroupedMLPElasticityManager(cfg)
        mgr.attach_hooks(mod)
        # current_router_emb is None — the input/fc1/output hooks all early-out.
        out, bias = mod(x)
        torch.testing.assert_close(out, baseline_out)
        mgr.detach_hooks()

    def test_soft_mask_zeros_upper_intermediate_at_half_budget(self):
        """fc1_post_hook applies the mlp_intermediate_mask. Soft-mask weighted
        sum on a one-hot at the half-budget choice should leave the upper
        ffn channels zeroed."""
        cfg = _config(hidden_size=64, ffn_hidden_size=128, soft_mask=True)
        mod = self._make_module(cfg)

        mgr = FlextronGroupedMLPElasticityManager(cfg)
        mgr.attach_hooks(mod)
        # One-hot router_emb on full-emb, one-hot router_mlp on half-ffn (index 1).
        emb_logits = torch.tensor([1.0, 0.0], dtype=torch.bfloat16, device="cuda")
        mlp_logits = torch.tensor([0.0, 1.0], dtype=torch.bfloat16, device="cuda")
        mgr.set_elasticity_params(
            router_emb=(emb_logits, cfg.hidden_size),
            router_mlp=(mlp_logits, cfg.ffn_hidden_size // 2),
        )

        x = torch.ones(2, cfg.hidden_size, dtype=torch.bfloat16, device="cuda")
        mod(x)
        intermediate = mod._captured_intermediate
        # mlp_int_list sorted-desc dedupe = [128, 64]; one-hot on index 1 -> 64.
        # Lower 64 channels active, upper 64 zeroed by the intermediate mask.
        assert (intermediate[..., 64:] == 0).all()
        assert not (intermediate[..., :64] == 0).all()
        mgr.detach_hooks()

    def test_set_elasticity_params_only_updates_provided_axes(self):
        """Calling set_elasticity_params with only one kwarg must not clear
        the other (regression-guard for the ``if x is not None`` pattern)."""
        cfg = _config()
        mgr = FlextronGroupedMLPElasticityManager(cfg)
        sentinel_emb = (torch.tensor([1.0, 0.0]), cfg.hidden_size)
        sentinel_mlp = (torch.tensor([0.0, 1.0]), cfg.ffn_hidden_size // 2)
        mgr.set_elasticity_params(router_emb=sentinel_emb, router_mlp=sentinel_mlp)

        # Update only emb; mlp must still be the prior value.
        new_emb = (torch.tensor([0.0, 1.0]), cfg.hidden_size // 2)
        mgr.set_elasticity_params(router_emb=new_emb)

        assert mgr.current_router_emb is new_emb
        assert mgr.current_router_mlp is sentinel_mlp

    def test_detach_clears_hook_handles(self):
        cfg = _config()
        mod = self._make_module(cfg)
        mgr = FlextronGroupedMLPElasticityManager(cfg)
        mgr.attach_hooks(mod)
        assert len(mgr.hook_handles) == 5
        mgr.detach_hooks()
        assert mgr.hook_handles == []


@pytest.mark.internal
class TestAddFlextronGroupedMLPElasticity:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_factory_returns_manager_with_layer_idx(self):
        cfg = _config()
        mod = _StubGroupedMLP(cfg.hidden_size, cfg.ffn_hidden_size).cuda().to(torch.bfloat16)
        mgr = add_flextron_grouped_mlp_elasticity(mod, cfg, layer_idx=0)
        assert isinstance(mgr, FlextronGroupedMLPElasticityManager)
        assert mgr.layer_idx == 0
        assert len(mgr.hook_handles) == 5
        mgr.detach_hooks()
