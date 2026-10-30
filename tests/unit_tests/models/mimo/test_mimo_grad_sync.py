# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-role gradient finalization, token normalization and legacy participation helpers."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.mimo.training.grad_sync import (
    _token_normalization_scale,
    _vision_participation_count,
    configure_grad_sync,
    mark_modality_participation,
    reset_modality_participation,
)
from examples.mimo.training.runtime import wrap_active_modules_with_ddp
from examples.mimo.training.topology import ModuleGridSpec, create_topology
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer import MegatronModule, TransformerConfig
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    create_hypercomm_grid,
    destroy_all_grids,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(("num_tokens", "expected"), [(0.0, 1.0), (4.0, 0.25)])
def test_token_normalization_scale(num_tokens, expected):
    scale = _token_normalization_scale(torch.tensor(num_tokens))

    assert scale.ndim == 0
    torch.testing.assert_close(scale, torch.tensor(expected))


def test_configure_grad_sync_installs_production_overlap_hooks(run_config):
    def no_sync():
        return None

    def start_grad_sync(*_unused):
        return None

    config = SimpleNamespace(no_sync_func=None, grad_sync_func=None)
    model = SimpleNamespace(
        config=config,
        no_sync=no_sync,
        start_grad_sync=start_grad_sync,
        language_model=None,
        modality_submodules={},
    )
    language_grid = SimpleNamespace(get_rank_enum=lambda _name: [[0]])
    topology = SimpleNamespace(grids={MIMO_LANGUAGE_MODULE_KEY: language_grid}, module_pgs={})

    configure_grad_sync(
        SimpleNamespace(overlap_grad_reduce=True, align_grad_reduce=True), model, topology
    )

    assert config.no_sync_func is no_sync
    assert config.grad_sync_func is start_grad_sync


def test_packed_disagg_encoder_ddp_uses_shared_token_denominator():
    """Independent scalar objective catches an extra encoder-DP or participation average."""
    Utils.initialize_distributed()
    topology = None
    try:
        if dist.get_world_size() != 8:
            pytest.skip("Requires 4 encoder and 4 decoder ranks")
        rank = dist.get_rank()
        encoder = rank < 4
        name = "images" if encoder else MIMO_LANGUAGE_MODULE_KEY
        width = 512
        topology = create_topology(
            [
                ModuleGridSpec(name="images", num_ranks=4),
                ModuleGridSpec(name=MIMO_LANGUAGE_MODULE_KEY, num_ranks=4, cp=2, rank_offset=4),
            ],
            packed=True,
        )
        language_config = TransformerConfig(
            num_layers=1,
            hidden_size=width,
            num_attention_heads=1,
            bf16=True,
            params_dtype=torch.bfloat16,
            context_parallel_size=2,
            calculate_per_token_loss=True,
        )
        # This deliberately starts with the vision default False. Production wrapping
        # must override it before constructing the encoder's independent DDP buffers.
        encoder_config = TransformerConfig(
            num_layers=1,
            hidden_size=width,
            num_attention_heads=1,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        class ScalarModule(MegatronModule):
            def __init__(self):
                super().__init__(encoder_config if encoder else language_config)
                self.pg_collection = topology.module_pgs[name]
                self.weight = torch.nn.Parameter(torch.full((width,), 0.5, device="cuda"))

            def forward(self, coefficient):
                return (self.weight * coefficient).sum()

        module = ScalarModule()
        model = SimpleNamespace(
            config=language_config,
            mimo_config=SimpleNamespace(module_to_grid_map=topology.grids),
            language_model=None if encoder else module,
            language_model_input_projections=torch.nn.ModuleDict(),
            modality_submodules=torch.nn.ModuleDict({"images": module} if encoder else {}),
        )
        args = SimpleNamespace(mimo_packed=True, overlap_grad_reduce=False)
        wrap_active_modules_with_ddp(
            args,
            model,
            topology,
            DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
                average_in_collective=False,
            ),
        )
        configure_grad_sync(args, model, topology)
        opt_config = OptimizerConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            lr=0.125,
            weight_decay=0,
            clip_grad=2.0,
            adam_eps=0.125,
            adam_beta1=0.9,
            adam_beta2=0.95,
        )
        optimizer = get_mimo_optimizer(model, opt_config)
        ddp = model.modality_submodules["images"] if encoder else model.language_model
        optimizer.zero_grad()
        ddp.zero_grad_buffer()
        # Only E0/E2 contribute; E1/E3 never execute forward/backward. D0 has no
        # supervised tokens. The independent reference is sum(coefficients)/16.
        coefficient = (1, 0, 3, 0, 0, 2, 4, 6)[rank]
        if not encoder or coefficient:
            ddp(torch.tensor(float(coefficient), device="cuda")).backward()
        local_tokens = torch.tensor((0, 0, 0, 0, 0, 4, 4, 8)[rank], device="cuda")
        model.config.finalize_model_grads_func([model], local_tokens)
        assert all(buffer.gradient_scaling_factor == 1.0 for buffer in ddp.buffers)
        gradient = 4 / 16 if encoder else 12 / 16
        inner = optimizer.chained_optimizers[0]
        param = module.weight
        assert param in inner.model_param_group_index_map
        shard = inner._get_model_param_range_map(param)["param"]
        local_grad = param.main_grad.flatten()[shard.start : shard.end]
        torch.testing.assert_close(
            local_grad, torch.full_like(local_grad, gradient), rtol=0, atol=0
        )

        success, norm, _ = optimizer.step()
        expected_norm = (width * ((4 / 16) ** 2 + (12 / 16) ** 2)) ** 0.5
        assert success
        assert norm == pytest.approx(expected_norm, rel=1e-6)
        clipped = gradient * min(1.0, opt_config.clip_grad / (expected_norm + 1e-6))
        expected_weight = 0.5 - opt_config.lr * clipped / (abs(clipped) + opt_config.adam_eps)
        state = inner._get_main_param_and_optimizer_states(param)
        for key, expected in (
            ("exp_avg", (1 - opt_config.adam_beta1) * clipped),
            ("exp_avg_sq", (1 - opt_config.adam_beta2) * clipped**2),
            ("param", expected_weight),
        ):
            torch.testing.assert_close(
                state[key], torch.full_like(state[key], expected), rtol=2e-6, atol=1e-8
            )
        torch.testing.assert_close(param, torch.full_like(param, expected_weight), rtol=0, atol=0)
    finally:
        if topology is not None:
            topology.destroy()
        Utils.destroy_model_parallel()


class TestVisionParticipation:
    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        destroy_all_grids()

    def test_scalar_fp32_scale_applies_to_bf16_gradients(self):
        grad = torch.ones(4, dtype=torch.bfloat16, device="cuda")
        scale = _token_normalization_scale(torch.tensor(2.0, dtype=torch.float32, device="cuda"))

        grad.mul_(scale)

        assert scale.ndim == 0
        torch.testing.assert_close(grad, torch.full_like(grad, 0.5))

    def test_vision_participation_correction(self):
        """Partial participation: text-only ranks upscale present ranks.

        With only some DP ranks holding image input, the participation count is
        < dp_size and the correction factor dp_size/participation is applied.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=self.world_size)
        vision_dp = grid.get_pg("dp")
        dp_size = dist.get_world_size(vision_dp)

        submodule = SimpleNamespace()
        fake_model = SimpleNamespace(modality_submodules={"images": submodule})

        rank = dist.get_rank(vision_dp)
        has_image = rank < dp_size // 2
        batch = (
            {"modality_inputs": {"images": {"hidden_states": torch.ones(1, device="cuda")}}}
            if has_image
            else {"modality_inputs": {}}
        )
        reset_modality_participation(fake_model)
        mark_modality_participation(fake_model, batch)

        count = _vision_participation_count(submodule, vision_dp)
        assert count.ndim == 0
        torch.testing.assert_close(
            count, torch.tensor(float(dp_size // 2), dtype=count.dtype, device=count.device)
        )
        factor = dp_size / count
        torch.testing.assert_close(
            factor, torch.tensor(2.0, dtype=factor.dtype, device=factor.device)
        )

        reset_modality_participation(fake_model)
        assert getattr(submodule, "_mimo_rank_processed_input") is False
