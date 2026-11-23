# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optimizer and gradient-buffer contracts for DSv4's grouped projection."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core import dist_checkpointing
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.extensions.te_batched_linear import TEBatchedLinear
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    TensorParallelAdaptiveMuon,
    TensorParallelMuon,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def _initialize(weight):
    with torch.no_grad():
        weight.copy_(
            torch.linspace(-0.27, 0.23, weight.numel(), device=weight.device).view_as(weight)
        )


@pytest.mark.parametrize(
    "mfsdp,mxfp8,missing_plane,requires_grad,rejected",
    [
        (True, True, "data", True, True),
        (True, True, "scale", True, True),
        (True, True, None, True, False),  # Replicated compute weights keep both planes.
        (False, True, None, True, False),  # Ordinary DDP.
        (True, False, None, True, False),  # BF16 primary weights.
        (True, True, "data", False, False),  # No backward needs columnwise weights.
    ],
)
def test_native_projection_phase_sharded_mxfp8_boundary(
    monkeypatch, mfsdp, mxfp8, missing_plane, requires_grad, rejected
):
    """Reject the unsupported storage phase before making a view or entering TE."""
    if TEBatchedLinear is None:
        pytest.skip("Transformer Engine BatchedLinear is unavailable")
    from megatron.core.extensions import te_batched_linear as adapter

    class QuantizedWeight:
        _columnwise_data = None if missing_plane == "data" else object()
        _columnwise_scale_inv = None if missing_plane == "scale" else object()

    monkeypatch.setattr(adapter, "MXFP8TensorStorage", QuantizedWeight)
    weight = QuantizedWeight() if mxfp8 else SimpleNamespace()
    if mfsdp:
        weight._mfsdp_parameter_group = object()
    calls = []
    operation = object()

    def grouped_operation():
        calls.append("view")
        return operation

    def te_forward(actual_operation, ctx, input_):
        assert actual_operation is operation
        calls.append("TE")
        return input_

    monkeypatch.setattr(adapter.BatchedLinear, "op_forward", te_forward)
    projection = SimpleNamespace(weight=weight, _grouped_operation=grouped_operation)
    ctx = SimpleNamespace(requires_grad=requires_grad)
    input_ = object()
    if rejected:
        with pytest.raises(NotImplementedError, match="MFSDPv2 phase-sharded MXFP8"):
            TEBatchedLinear.op_forward(projection, ctx, input_)
        assert not calls
    else:
        assert TEBatchedLinear.op_forward(projection, ctx, input_) is input_
        assert calls == ["view", "TE"]


class _Projection(MegatronModule):
    """Small real training model isolating the grouped projection's optimizer owner."""

    def __init__(self, config, backend) -> None:
        super().__init__(config)
        self.recipe = None
        if backend == "fallback":
            self.weight = torch.nn.Parameter(
                torch.empty(128, 64, device="cuda", dtype=torch.bfloat16)
            )
            _initialize(self.weight)
            self.projection = None
            return
        if TEBatchedLinear is None:
            pytest.skip(
                "Transformer Engine BatchedLinear is unavailable; fallback is tested separately"
            )
        import transformer_engine.common.recipe as recipes
        import transformer_engine.pytorch as te

        if backend == "primary_mxfp8":
            available, reason = te.is_mxfp8_available(return_reason=True)
            if not available:
                pytest.skip(reason)
            self.recipe = recipes.MXFP8BlockScaling(fp8_format=recipes.Format.E4M3)
        init_context = (
            te.quantized_model_init(
                enabled=True, recipe=self.recipe, preserve_high_precision_init_val=True
            )
            if self.recipe is not None
            else nullcontext()
        )
        with init_context:
            self.projection = TEBatchedLinear(
                2,
                64,
                64,
                input_batch_dim=-2,
                bias=False,
                dtype=torch.bfloat16,
                device="cuda",
                accumulate_into_main_grad=True,
                save_original_input=True,
                init_method=_initialize,
            )

    def projection_weight(self):
        return self.weight if self.projection is None else self.projection.weight

    def forward(self, inputs):
        if self.projection is None:
            return torch.einsum("...gd,grd->...gr", inputs, self.weight.view(2, 64, 64))
        if self.recipe is None:
            return self.projection(inputs)
        import transformer_engine.pytorch as te

        with te.autocast(enabled=True, recipe=self.recipe):
            return self.projection(inputs)


@pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason="Requires emerging-optimizers")
@pytest.mark.parametrize("backend", ["fallback", "bf16", "primary_mxfp8"])
@pytest.mark.parametrize("optimizer_name", ["muon", "adaptive_muon"])
def test_matrix_optimizer_step_and_resume(backend, optimizer_name, tmp_path_dist_ckpt):
    """Real DDP/MCore optimizer steps retain matrix routing, initialization and resume."""
    Utils.initialize_model_parallel()
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=2,
            params_dtype=torch.bfloat16,
            bf16=True,
            gradient_accumulation_fusion=backend != "fallback",
        )
        optimizer_config = OptimizerConfig(
            optimizer=optimizer_name,
            lr=1e-3,
            weight_decay=0.0,
            bf16=True,
            use_distributed_optimizer=False,
            clip_grad=0.0,
            muon_split_qkv=False,
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
            adaptive_muon_moment2_method="adamuon",
        )

        def build():
            module = _Projection(config, backend)
            weight = module.projection_weight()
            assert weight.shape == (128, 64)
            expected_initial = torch.empty(weight.shape, device=weight.device, dtype=torch.bfloat16)
            _initialize(expected_initial)
            if backend == "primary_mxfp8":
                torch.testing.assert_close(
                    weight.get_high_precision_init_val(), expected_initial.cpu(), rtol=0, atol=0
                )
                assert not torch.equal(weight.dequantize(), expected_initial)
            model = DistributedDataParallel(
                config,
                DistributedDataParallelConfig(
                    use_distributed_optimizer=False,
                    overlap_grad_reduce=False,
                    grad_reduce_in_fp32=True,
                ),
                module,
                pg_collection=groups,
            )
            optimizer = get_megatron_optimizer(
                optimizer_config, [model], pg_collection=groups, use_gloo_process_groups=False
            )
            expected_type = (
                TensorParallelMuon if optimizer_name == "muon" else TensorParallelAdaptiveMuon
            )
            owners = [child.optimizer for child in optimizer.chained_optimizers]
            assert len(owners) == 1 and isinstance(owners[0], expected_type)
            torch.testing.assert_close(weight.main_param, expected_initial.float(), rtol=0, atol=0)
            if backend == "primary_mxfp8":
                assert weight.get_high_precision_init_val() is None
            return model, optimizer, weight, expected_type

        model, optimizer, weight, optimizer_type = build()
        reference = torch.nn.Parameter(weight.main_param.detach().clone())
        reference_optimizer = optimizer_type(
            [reference],
            lr=1e-3,
            nesterov=optimizer_config.muon_nesterov,
            weight_decay=0.0,
            pg_collection=groups,
            split_qkv=False,
            num_ns_steps=5,
            scale_mode="spectral",
            tp_mode="duplicated",
        )
        initial_identity = id(weight)

        def step(model, optimizer, weight, iteration, reference_optimizer=None):
            optimizer.zero_grad()
            model.zero_grad_buffer()
            assert not weight.grad_added_to_main_grad
            assert torch.count_nonzero(weight.main_grad) == 0
            for microbatch in range(2):
                torch.manual_seed(1234 + 2 * iteration + microbatch)
                inputs = torch.randn(
                    32, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
                )
                model(inputs).float().square().mean().backward()
                if backend != "fallback":
                    assert weight.grad_added_to_main_grad
            model.finish_grad_sync()
            assert torch.isfinite(weight.main_grad).all() and torch.count_nonzero(weight.main_grad)
            if reference_optimizer is not None:
                reference.grad = weight.main_grad.detach().float().clone()
                reference_optimizer.step()
            optimizer.step()
            if reference_optimizer is not None:
                torch.testing.assert_close(weight.main_param, reference, rtol=0, atol=0)

        step(model, optimizer, weight, 0, reference_optimizer)
        assert id(model.module.projection_weight()) == initial_identity
        resumed, resumed_optimizer, resumed_weight, _ = build()
        metadata = {"dp_cp_group": groups.dp_cp}
        with TempNamedDir(tmp_path_dist_ckpt / "grouped-projection") as checkpoint:
            # Training checkpoints unwrap DDP before constructing the model state.
            model_state = model.module.sharded_state_dict(metadata=metadata)
            dist_checkpointing.save(
                {"model": model_state, "optimizer": optimizer.sharded_state_dict(model_state)},
                checkpoint,
            )
            resume_state = resumed.module.sharded_state_dict(metadata=metadata)
            restored = dist_checkpointing.load(
                {
                    "model": resume_state,
                    "optimizer": resumed_optimizer.sharded_state_dict(
                        resume_state, is_loading=True
                    ),
                },
                checkpoint,
            )
        resumed.load_state_dict(restored["model"])
        resumed_optimizer.load_state_dict(restored["optimizer"])
        torch.testing.assert_close(
            optimizer.state_dict(), resumed_optimizer.state_dict(), rtol=0, atol=0
        )
        step(model, optimizer, weight, 1, reference_optimizer)
        step(resumed, resumed_optimizer, resumed_weight, 1)
        torch.testing.assert_close(weight.main_param, resumed_weight.main_param, rtol=0, atol=0)
        torch.testing.assert_close(
            optimizer.state_dict(), resumed_optimizer.state_dict(), rtol=0, atol=0
        )
    finally:
        Utils.destroy_model_parallel()
