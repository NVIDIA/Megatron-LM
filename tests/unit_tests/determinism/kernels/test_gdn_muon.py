# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay the scalar optimizer route for GDN's fused input projection."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.optimizer import Adam, OptimizerConfig, _get_param_groups
from megatron.core.optimizer.emerging_optimizers import _EMERGING_OPTIMIZERS
from megatron.core.optimizer.layer_wise_optimizer import tag_params_for_buffer_routing
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.common import HAVE_FLA
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_FLA), reason="needs a GPU and FLA"
)


class TestGDNMuonDeterminism:
    def setup_method(self, method):
        Utils.initialize_model_parallel()
        seeded()
        model_parallel_cuda_manual_seed(1234)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("optimizer_name", ["muon", "adaptive_muon"])
    def test_scalar_projection_update_replays(self, optimizer_name):
        if optimizer_name not in _EMERGING_OPTIMIZERS:
            pytest.skip("needs emerging-optimizers")
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            activation_func=F.silu,
            normalization="RMSNorm",
            use_cpu_initialization=True,
            transformer_impl="transformer_engine",
            experimental_attention_variant="gdn",
            linear_attention_freq=[1],
            gdn_kernel_backend="torch",
            deterministic_mode=True,
        )
        spec = get_experimental_attention_variant_module_spec(config=config)
        gdn = spec.module(
            config,
            submodules=spec.submodules,
            layer_number=1,
            pg_collection=ProcessGroupCollection(
                tp=parallel_state.get_tensor_model_parallel_group(),
                cp=parallel_state.get_context_parallel_group(),
            ),
        ).cuda()
        groups = _get_param_groups(
            [gdn],
            OptimizerConfig(optimizer=optimizer_name, lr=1e-3),
            _EMERGING_OPTIMIZERS[optimizer_name].default_param_overrides,
        )
        scalar_groups = [group for group in groups if group.get("optimizer") == "adam"]
        scalar_params = [p for group in scalar_groups for p in group["params"]]
        scalar_ids = {id(p) for p in scalar_params}
        assert id(gdn.in_proj.weight) in scalar_ids
        assert id(gdn.out_proj.weight) not in scalar_ids
        # DDP buffer ownership must agree with the optimizer's actual parameter groups.
        tag_params_for_buffer_routing([gdn])
        assert not gdn.in_proj.weight.is_managed_by_layer_wise_optimizer
        assert gdn.out_proj.weight.is_managed_by_layer_wise_optimizer
        initial = [p.detach().clone() for p in scalar_params]
        initial_projection = gdn.in_proj.weight.detach().clone()
        gradients = [[torch.randn_like(p) for p in scalar_params] for _ in range(2)]

        def run_steps():
            with torch.no_grad():
                for param, value in zip(scalar_params, initial):
                    param.copy_(value)
            optimizer = Adam(scalar_groups, lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
            for grads in gradients:
                for param, grad in zip(scalar_params, grads):
                    param.grad = grad.clone()
                optimizer.step()
            results = [p.detach().clone() for p in scalar_params]
            for param in scalar_params:
                state = optimizer.state[param]
                assert "exp_avg" in state and "exp_avg_sq" in state
                results.extend(
                    state[key].clone() for key in sorted(state) if torch.is_tensor(state[key])
                )
            assert not torch.equal(gdn.in_proj.weight, initial_projection)
            return results

        assert_replays_bit_exact(
            run_steps,
            (),
            replays=4,
            backward=False,
            contention=True,
            what=f"GDN {optimizer_name} scalar projection update",
        )
