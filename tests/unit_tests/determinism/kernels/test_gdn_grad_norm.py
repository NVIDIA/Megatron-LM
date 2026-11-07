# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay GDN's separate gradient clipping through the real optimizer step."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
)
from megatron.core.optimizer import Adam, OptimizerConfig
from megatron.core.optimizer.optimizer import (
    GDN_GRAD_NORM_GROUP,
    MTP_GRAD_NORM_GROUP,
    FP32Optimizer,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.common import HAVE_FLA
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_FLA), reason="needs a GPU and FLA"
)


class TestGDNGradNormDeterminism:
    def setup_method(self, method):
        Utils.initialize_model_parallel()
        seeded()
        model_parallel_cuda_manual_seed(1234)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("separate", [False, True])
    @pytest.mark.parametrize("with_mtp", [False, True])
    def test_clipped_optimizer_step_replays(self, separate, with_mtp):
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
            gated_delta_net_separate_grad_norm=separate,
        )
        spec = get_experimental_attention_variant_module_spec(config=config)
        gdn = spec.module(
            config,
            submodules=spec.submodules,
            layer_number=1,
            bias=True,
            pg_collection=ProcessGroupCollection(
                tp=parallel_state.get_tensor_model_parallel_group(),
                cp=parallel_state.get_context_parallel_group(),
            ),
        ).cuda()
        projection_params = list(gdn.in_proj.parameters())
        for param in projection_params:
            expected = GDN_GRAD_NORM_GROUP if separate else None
            assert getattr(param, "grad_norm_group", None) == expected
        assert getattr(gdn.out_proj.weight, "grad_norm_group", None) is None
        params = [gdn.out_proj.weight, *projection_params]
        groups = [None] + [GDN_GRAD_NORM_GROUP if separate else None] * len(projection_params)
        if with_mtp:
            mtp = torch.nn.Parameter(torch.randn(4097, device="cuda"))
            mtp.grad_norm_group = MTP_GRAD_NORM_GROUP
            params.append(mtp)
            groups.append(MTP_GRAD_NORM_GROUP)
        initial = [p.detach().clone() for p in params]
        # The large projection gradient should not shrink the main gradient when enabled.
        gradients = [
            [torch.randn_like(p) * (1e-4 if i == 0 else 100.0) for i, p in enumerate(params)]
            for _ in range(2)
        ]
        opt_config = OptimizerConfig(lr=1e-3, clip_grad=1.0)

        def run_steps():
            with torch.no_grad():
                for param, value in zip(params, initial):
                    param.copy_(value)
            inner = Adam(params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
            optimizer = FP32Optimizer(inner, opt_config, init_state_fn=None)
            results = []
            for grads in gradients:
                norms = {
                    group: torch.cat(
                        [g.double().flatten() for g, key in zip(grads, groups) if key == group]
                    )
                    .norm()
                    .item()
                    for group in set(groups)
                }
                for param, grad in zip(params, grads):
                    param.main_grad = grad.clone()
                success, main_norm, _ = optimizer.step()
                assert success
                assert main_norm == pytest.approx(norms[None], rel=1e-5)
                assert set(optimizer.grad_norms_by_group) == set(groups) - {None}
                for group, norm in optimizer.grad_norms_by_group.items():
                    assert norm == pytest.approx(norms[group], rel=1e-5)
                for param, grad, group in zip(params, grads, groups):
                    scale = min(1.0, opt_config.clip_grad / (norms[group] + 1e-6))
                    torch.testing.assert_close(param.grad, grad * scale, rtol=1e-5, atol=1e-8)
                    results.append(param.grad.clone())
                results.append(
                    torch.tensor(
                        [main_norm, *optimizer.grad_norms_by_group.values()], device="cuda"
                    )
                )
            results.extend(p.detach().clone() for p in params)
            for param in params:
                state = inner.state[param]
                assert "exp_avg" in state and "exp_avg_sq" in state
                results.extend(
                    state[key].clone() for key in sorted(state) if torch.is_tensor(state[key])
                )
            return results

        assert_replays_bit_exact(
            run_steps,
            (),
            replays=4,
            backward=False,
            contention=True,
            what=f"GDN separate grad norm={separate}, MTP={with_mtp}",
        )
