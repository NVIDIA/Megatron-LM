# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-backed tests for FlextronMambaElasticityManager.

Builds a real MambaMixer (mirroring ``tests/unit_tests/ssm/test_mamba_mixer.py``)
and verifies that the elasticity hooks attach, behave as no-ops without
elasticity params, and produce different activations once params are set.

Run with:

    torchrun --nproc_per_node=1 -m pytest tests/unit_tests/elastification/test_flextron_mamba_elasticity_manager.py
"""

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.elastification.flextron_elasticity_hooks import (
    FlextronMambaElasticityManager,
    add_flextron_mamba_elasticity,
)
from tests.unit_tests.test_utilities import Utils


def _flextron_fields(hidden_size, num_heads):
    """Return dict of flextron attrs to copy onto a TransformerConfig."""
    return dict(
        flextron=True,
        soft_mask=True,
        flex_hetero_mamba=False,
        flex_hetero_ffn=False,
        flex_hetero_head=False,
        flex_hetero_moe_expert=False,
        hybrid_layer_pattern="M",
        emb_int_list=[hidden_size, hidden_size // 2],
        mamba_int_list=[num_heads, num_heads // 2],
    )


@pytest.mark.internal
class TestFlextronMambaElasticityManager:

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_mixer_and_config(self, hidden_size=256, num_heads=8):
        """Construct a bf16 MambaMixer on CUDA + a flextron-enabled config."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=hidden_size,
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
            use_mamba_mem_eff_path=True,
        )
        # Inject the flextron fields directly (bypassing inject_flextron_config
        # to avoid pulling in the whole args-parser stack).
        for k, v in _flextron_fields(hidden_size, num_heads).items():
            setattr(config, k, v)

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        mixer = MambaMixer(
            config,
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            config.hidden_size,
            layer_number=1,
            pg_collection=pg_collection,
        )
        mixer.cuda()
        return mixer, config

    def test_attach_produces_hook_handles(self):
        mixer, config = self._build_mixer_and_config()
        mgr = FlextronMambaElasticityManager(config)
        mgr.attach_hooks(mixer)
        # There are several hooks: setup + input_mask + in_proj(pre,post) +
        # conv1d + norm(pre,post) + output + cleanup. Exact count can change;
        # we just verify > 1.
        assert len(mgr.hook_handles) > 1
        mgr.detach_hooks()

    def test_current_router_none_preserves_output(self):
        """Without elasticity params set, a forward with hooks attached must
        produce (approximately) the same output as a forward without hooks."""
        mixer, config = self._build_mixer_and_config()

        seq_len, micro_batch = 16, 2
        x = torch.ones((seq_len, micro_batch, config.hidden_size), device="cuda")

        # Baseline (no elasticity).
        baseline_out, _ = mixer(x)

        mgr = FlextronMambaElasticityManager(config)
        mgr.attach_hooks(mixer)
        # current_router_emb/mamba both None -> all hooks except setup/cleanup
        # should no-op.
        hooked_out, _ = mixer(x)
        torch.testing.assert_close(hooked_out, baseline_out, atol=1e-2, rtol=1e-2)
        mgr.detach_hooks()

    def test_full_budget_one_hot_approx_matches_baseline(self):
        """One-hot on index 0 (full emb + full mamba heads) should approximately
        reproduce the baseline output (up to bf16 / eps drift)."""
        mixer, config = self._build_mixer_and_config()

        seq_len, micro_batch = 16, 2
        x = torch.ones((seq_len, micro_batch, config.hidden_size), device="cuda")
        baseline_out, _ = mixer(x)

        mgr = FlextronMambaElasticityManager(config)
        mgr.attach_hooks(mixer)
        emb_logits = torch.tensor([1.0, 0.0], dtype=torch.bfloat16, device="cuda")
        mamba_logits = torch.tensor([1.0, 0.0], dtype=torch.bfloat16, device="cuda")
        mgr.set_elasticity_params(
            router_emb=(emb_logits, config.hidden_size),
            router_mamba=(mamba_logits, config.mamba_int_list[0]),
        )
        full_out, _ = mixer(x)
        # Full-budget one-hot should not materially change the output.
        torch.testing.assert_close(full_out, baseline_out, atol=5e-2, rtol=5e-2)
        mgr.detach_hooks()

    def test_small_budget_one_hot_changes_output(self):
        """One-hot on a smaller choice should change the output norm."""
        mixer, config = self._build_mixer_and_config()

        seq_len, micro_batch = 16, 2
        x = torch.randn(
            (seq_len, micro_batch, config.hidden_size), device="cuda"
        )
        baseline_out, _ = mixer(x)

        mgr = FlextronMambaElasticityManager(config)
        mgr.attach_hooks(mixer)
        emb_logits = torch.tensor([0.0, 1.0], dtype=torch.bfloat16, device="cuda")
        mamba_logits = torch.tensor([0.0, 1.0], dtype=torch.bfloat16, device="cuda")
        mgr.set_elasticity_params(
            router_emb=(emb_logits, config.emb_int_list[1]),
            router_mamba=(mamba_logits, config.mamba_int_list[1]),
        )
        small_out, _ = mixer(x)
        # The small-budget output should measurably differ from the baseline.
        assert not torch.allclose(small_out, baseline_out, atol=1e-2)
        mgr.detach_hooks()

    def test_detach_restores_baseline(self):
        mixer, config = self._build_mixer_and_config()
        seq_len, micro_batch = 16, 2
        x = torch.ones((seq_len, micro_batch, config.hidden_size), device="cuda")
        baseline_out, _ = mixer(x)

        mgr = FlextronMambaElasticityManager(config)
        mgr.attach_hooks(mixer)
        emb_logits = torch.tensor([0.0, 1.0], dtype=torch.bfloat16, device="cuda")
        mamba_logits = torch.tensor([0.0, 1.0], dtype=torch.bfloat16, device="cuda")
        mgr.set_elasticity_params(
            router_emb=(emb_logits, config.emb_int_list[1]),
            router_mamba=(mamba_logits, config.mamba_int_list[1]),
        )
        _ = mixer(x)
        mgr.detach_hooks()

        detached_out, _ = mixer(x)
        torch.testing.assert_close(detached_out, baseline_out, atol=1e-2, rtol=1e-2)


@pytest.mark.internal
class TestAddFlextronMambaElasticity:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_factory_returns_manager(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
            use_mamba_mem_eff_path=True,
        )
        for k, v in _flextron_fields(256, 8).items():
            setattr(config, k, v)

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        mixer = MambaMixer(
            config,
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            config.hidden_size,
            layer_number=1,
            pg_collection=pg_collection,
        ).cuda()

        mgr = add_flextron_mamba_elasticity(mixer, config, layer_idx=0)
        assert isinstance(mgr, FlextronMambaElasticityManager)
        assert mgr.layer_idx == 0
        mgr.detach_hooks()
