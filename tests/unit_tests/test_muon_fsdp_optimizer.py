# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Tests for Muon + Megatron-FSDP integration.

Organized by implementation phase:
  Phase 1: FSDPMuonChainedOptimizer (protocol adapter, no_shard dispatch)
  Phase 2: FSDPZeROTensorParallelMuon.orthogonalize() (core allgather -> NS -> reshard)
  Phase 3: Factory integration + end-to-end
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from packaging.version import Version

from megatron.core.optimizer.emerging_optimizers import (
    FSDPMuonChainedOptimizer,
    FSDPZeROTensorParallelMuon,
    TensorParallelMuon,
    _get_mfsdp_models,
)
from tests.unit_tests.test_utilities import Utils

# Skip all tests in this file for LTS versions
pytestmark = pytest.mark.skipif(
    Version(os.getenv("NVIDIA_PYTORCH_VERSION", "24.01")) <= Version("25.05"),
    reason="Skip muon FSDP optimizer for LTS test",
)

WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


# Helpers


def make_sharded_grad(full_rows, cols, dp_size, dp_rank, seed=42, device="cuda"):
    """Create full grad and return `(full_grad, padded_grad, local_shard, shard_rows)`.

    The full gradient is split into `dp_size` equal-sized shards along dim-0.
    When `full_rows` is not divisible by `dp_size`, the last shard is
    zero-padded so all shards have the same size (mimicking FSDP bucket padding).
    Both the unpadded (`full_grad`) and padded (`padded_grad`) views are
    returned so reference computations can choose either.
    """
    torch.manual_seed(seed)
    full_grad = torch.randn(full_rows, cols, device=device, dtype=torch.float32)

    shard_rows = (full_rows + dp_size - 1) // dp_size  # ceil division
    # Pad full_grad so it can be evenly split
    padded_rows = shard_rows * dp_size
    if padded_rows > full_rows:
        pad = torch.zeros(padded_rows - full_rows, cols, device=device, dtype=torch.float32)
        padded_grad = torch.cat([full_grad, pad], dim=0)
    else:
        padded_grad = full_grad

    local_shard = padded_grad[dp_rank * shard_rows : (dp_rank + 1) * shard_rows].clone()
    return full_grad, padded_grad, local_shard, shard_rows


def make_reference_orthogonalize(full_grad, param_like=None, **muon_kwargs):
    """Run `TensorParallelMuon.orthogonalize` on full grad as reference.

    Returns the orthogonalized full gradient (plain tensor, no sharding).
    """
    if param_like is None:
        param_like = torch.randn_like(full_grad)
    opt = TensorParallelMuon(
        params=[torch.nn.Parameter(param_like)],
        pg_collection=None,
        tp_mode="duplicated",
        **muon_kwargs,
    )
    return opt.orthogonalize(param_like, full_grad)


def make_dp_device_mesh(dp_size):
    """Create 1D device mesh for DP-only DTensor tests."""
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))


# Phase 1: FSDPMuonChainedOptimizer (single-rank, mock-based)


class TestPhase1ChainedOptimizer:
    """Phase 1 tests for FSDPMuonChainedOptimizer protocol adapter."""

    @staticmethod
    def _make_mfsdp_mock(model_auto_sync=False):
        """Create a mock MegatronFSDP model."""
        mock = MagicMock()
        mock.model_auto_sync = model_auto_sync
        return mock

    @staticmethod
    def _make_inner_mock():
        """Create a mock inner optimizer."""
        mock = MagicMock()
        mock.step.return_value = (True, 1.0, 0)
        return mock

    @pytest.mark.parametrize("model_auto_sync", [True, False])
    def test_phase1_chained_optimizer_step_protocol(self, model_auto_sync):
        """Verify step() call order: finish_grad_sync -> inner.step -> install.

        When `model_auto_sync=True`, finish_grad_sync is NOT called (FSDP handles it).
        """
        inner = self._make_inner_mock()
        mfsdp = self._make_mfsdp_mock(model_auto_sync=model_auto_sync)
        wrapper = FSDPMuonChainedOptimizer(inner, [mfsdp])

        wrapper.step()

        if model_auto_sync:
            mfsdp.finish_grad_sync.assert_not_called()
        else:
            mfsdp.finish_grad_sync.assert_called_once()

        inner.step.assert_called_once()
        mfsdp.install_optimized_model_weights.assert_called_once()

        if not model_auto_sync:
            all_calls = []
            all_calls.append(("finish_grad_sync", mfsdp.finish_grad_sync.call_count))
            all_calls.append(("step", inner.step.call_count))
            all_calls.append(("install", mfsdp.install_optimized_model_weights.call_count))
            assert all(c[1] == 1 for c in all_calls)

    def test_phase1_chained_optimizer_getattr_delegation(self):
        """Verify attribute delegation to inner optimizer."""
        inner = self._make_inner_mock()
        inner.param_groups = [{"lr": 0.01}]
        inner.state_dict.return_value = {"state": {}, "param_groups": []}
        mfsdp = self._make_mfsdp_mock()

        wrapper = FSDPMuonChainedOptimizer(inner, [mfsdp])

        assert wrapper.param_groups == [{"lr": 0.01}]

        sd = wrapper.state_dict()
        inner.state_dict.assert_called_once()
        assert sd == {"state": {}, "param_groups": []}

        wrapper.load_state_dict({"state": {}, "param_groups": []})
        inner.load_state_dict.assert_called_once()

    @pytest.mark.parametrize("set_to_none", [True, False])
    def test_phase1_chained_optimizer_zero_grad(self, set_to_none):
        """Verify zero_grad delegates to inner with correct args."""
        inner = self._make_inner_mock()
        mfsdp = self._make_mfsdp_mock()
        wrapper = FSDPMuonChainedOptimizer(inner, [mfsdp])

        wrapper.zero_grad(set_to_none=set_to_none)
        inner.zero_grad.assert_called_once_with(set_to_none)

    def test_phase1_get_mfsdp_models_error(self):
        """`_get_mfsdp_models` raises RuntimeError for non-FSDP modules."""
        with pytest.raises(RuntimeError, match="Could not find any MegatronFSDP"):
            _get_mfsdp_models([nn.Module()])

    def test_phase1_get_mfsdp_models_success(self):
        """`_get_mfsdp_models` extracts inner module from FSDP-wrapped chunks."""
        inner_module = MagicMock()
        chunk = MagicMock()
        chunk.module = inner_module
        chunk.finish_grad_sync = MagicMock()

        result = _get_mfsdp_models([chunk])
        assert result == [inner_module]

    def test_phase1_multiple_mfsdp_models(self):
        """`step()` calls finish_grad_sync and install on ALL mfsdp models."""
        inner = self._make_inner_mock()
        mfsdps = [self._make_mfsdp_mock(model_auto_sync=False) for _ in range(3)]
        wrapper = FSDPMuonChainedOptimizer(inner, mfsdps)

        wrapper.step()

        for mfsdp in mfsdps:
            mfsdp.finish_grad_sync.assert_called_once()
            mfsdp.install_optimized_model_weights.assert_called_once()


# Phase 2: FSDPZeROTensorParallelMuon.orthogonalize() (multi-rank)


def _skip_if_single_rank():
    return pytest.mark.skipif(WORLD_SIZE <= 1, reason="Multi-rank test requires WORLD_SIZE > 1")


def _skip_if_no_dtensor():
    return pytest.mark.skipif(
        Version(torch.__version__.split("+")[0]) < Version("2.4.0"),
        reason="DTensor tests require PyTorch >= 2.4.0",
    )


@_skip_if_single_rank()
class TestPhase2Orthogonalize:
    """Phase 2 tests for FSDPZeROTensorParallelMuon.orthogonalize().

    These are the most critical tests verifying mathematical correctness
    of the allgather -> Newton-Schulz -> reshard cycle.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Initialize distributed for multi-rank tests."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def _make_fsdp_muon(self, dp_group=None, **kwargs):
        """Create an FSDPZeROTensorParallelMuon with a dummy param."""
        defaults = dict(
            lr=0.01,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=None,
            tp_mode="duplicated",
        )
        defaults.update(kwargs)
        dummy = torch.nn.Parameter(torch.randn(4, 4, device="cuda"))
        return FSDPZeROTensorParallelMuon(params=[dummy], dp_group=dp_group, **defaults)

    def test_phase2_orthogonalize_fallback_dp_group_none(self):
        """`dp_group=None` delegates to parent `TensorParallelMuon.orthogonalize`."""
        M, C = 64, 32
        torch.manual_seed(42)
        grad = torch.randn(M, C, device="cuda")
        p = torch.randn(M, C, device="cuda")

        fsdp_muon = self._make_fsdp_muon(dp_group=None)
        result_fsdp = fsdp_muon.orthogonalize(p, grad.clone())

        ref_result = make_reference_orthogonalize(grad.clone(), param_like=p)

        torch.testing.assert_close(result_fsdp, ref_result, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("shape", [(64, 32), (48, 24), (100, 50)])
    def test_phase2_orthogonalize_correctness_plain_tensors(self, shape):
        """Core invariant: allgather -> NS -> reshard == NS(padded_G)[shard_slice].

        `FSDPZeROTensorParallelMuon` runs Newton-Schulz on the *padded* full
        matrix because (a) production FSDP feeds it the padded global DTensor
        shape and (b) `get_muon_scale_factor` is dimension-dependent, so the
        scale only matches if both reference and code see the same padded
        dimensions. We therefore reference NS on `padded_grad`, not on the
        unpadded `full_grad`.
        """
        M, C = shape
        dp_group = torch.distributed.group.WORLD
        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()

        _, padded_grad, local_shard, shard_rows = make_sharded_grad(M, C, dp_size, dp_rank, seed=42)

        ref_full = make_reference_orthogonalize(padded_grad)
        start = dp_rank * shard_rows
        end = start + shard_rows
        ref_shard = ref_full[start:end]

        p = torch.randn(shard_rows, C, device="cuda")
        fsdp_muon = self._make_fsdp_muon(dp_group=dp_group)
        result = fsdp_muon.orthogonalize(p, local_shard)

        torch.testing.assert_close(result, ref_shard, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("M", [30, 31, 33])
    def test_phase2_orthogonalize_padding_correctness(self, M):
        """Padding correctness when `M % dp_size != 0`.

        Per the design described in
        `test_phase2_orthogonalize_correctness_plain_tensors`, the
        reference runs NS on the *padded* matrix so that scale-factor and NS
        cycle dimensions agree. We therefore expect `result` to match the
        padded reference for the entire shard (real and padding rows alike).
        """
        C = 16
        dp_group = torch.distributed.group.WORLD
        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()

        _, padded_grad, local_shard, shard_rows = make_sharded_grad(
            M, C, dp_size, dp_rank, seed=123
        )

        ref_full = make_reference_orthogonalize(padded_grad)
        start = dp_rank * shard_rows
        end = start + shard_rows
        ref_shard = ref_full[start:end]

        p = torch.randn(shard_rows, C, device="cuda")
        fsdp_muon = self._make_fsdp_muon(dp_group=dp_group)
        result = fsdp_muon.orthogonalize(p, local_shard)

        torch.testing.assert_close(result, ref_shard, atol=1e-5, rtol=1e-4)

    @_skip_if_no_dtensor()
    @pytest.mark.parametrize("shape", [(64, 32), (48, 24)])
    def test_phase2_orthogonalize_dtensor_output_invariants(self, shape):
        """DTensor input produces DTensor output with correct metadata.

        Input: Shard(0) DTensor on a 1D DP device mesh.
        Output: must be DTensor with same device_mesh, placements, global shape.
        Also verifies mathematical correctness.
        """
        from torch.distributed.tensor import DTensor, Shard

        M, C = shape
        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = make_dp_device_mesh(dp_size)
        dp_group = device_mesh.get_group("dp")

        _, padded_grad, local_shard, shard_rows = make_sharded_grad(M, C, dp_size, dp_rank, seed=42)

        # The global shape is declared as `(shard_rows * dp_size, C)` to match
        # the padded total; FSDP bucket padding makes all shards equal.
        padded_rows = shard_rows * dp_size
        grad_dtensor = DTensor.from_local(
            local_shard,
            device_mesh=device_mesh,
            placements=[Shard(0)],
            shape=torch.Size([padded_rows, C]),
            stride=local_shard.stride(),
            run_check=False,
        )

        # Param is also a DTensor: `FSDPZeROTensorParallelMuon` reads
        # `p.shape[0]` for the DTensor path.
        p_local = torch.randn(shard_rows, C, device="cuda")
        p_dtensor = DTensor.from_local(
            p_local,
            device_mesh=device_mesh,
            placements=[Shard(0)],
            shape=torch.Size([padded_rows, C]),
            stride=p_local.stride(),
            run_check=False,
        )

        fsdp_muon = self._make_fsdp_muon(dp_group=dp_group)
        result = fsdp_muon.orthogonalize(p_dtensor, grad_dtensor)

        assert isinstance(result, DTensor), "Output should be a DTensor"
        assert result.device_mesh == device_mesh, "Device mesh should match"
        assert result.placements == grad_dtensor.placements, "Placements should match"
        assert result.shape == grad_dtensor.shape, "Global shape should match"
        assert result.to_local().shape == local_shard.shape, "Local shape should match shard"

        ref_full = make_reference_orthogonalize(padded_grad)
        start = dp_rank * shard_rows
        end = start + shard_rows
        ref_shard = ref_full[start:end]

        torch.testing.assert_close(result.to_local(), ref_shard, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("shape", [(64, 32), (48, 24)])
    def test_phase2_full_step_cycle_plain_tensors(self, shape):
        """Full FSDPZeROTensorParallelMuon.step() cycles with plain tensors.

        Runs 3 complete optimization steps through the
        `OrthogonalizedOptimizer.step()` path, testing the full chain:
        momentum -> Nesterov -> orthogonalize -> `p.add_()`.
        """
        M, C = shape
        dp_group = torch.distributed.group.WORLD
        dp_size = torch.distributed.get_world_size()

        shard_rows = (M + dp_size - 1) // dp_size

        torch.manual_seed(42)
        p = torch.nn.Parameter(torch.randn(shard_rows, C, device="cuda"))

        optimizer = FSDPZeROTensorParallelMuon(
            params=[p],
            dp_group=dp_group,
            lr=0.01,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=None,
            tp_mode="duplicated",
        )

        weights_history = [p.data.clone()]
        for step_i in range(3):
            torch.manual_seed(100 + step_i)
            p.grad = torch.randn_like(p)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            weights_history.append(p.data.clone())

        for i in range(len(weights_history) - 1):
            assert not torch.equal(
                weights_history[i], weights_history[i + 1]
            ), f"Weight should change at step {i}"

    @_skip_if_no_dtensor()
    @pytest.mark.parametrize("shape", [(64, 32), (48, 24)])
    def test_phase2_full_step_cycle_dtensors(self, shape):
        """Full step() cycles with Shard(0) DTensor params and grads.

        Critical integration test for the DTensor return-type fix:
        `p.add_(orthogonalized_dtensor)` must work without dispatch errors.
        """
        from torch.distributed.tensor import DTensor, Shard

        M, C = shape
        dp_size = torch.distributed.get_world_size()
        device_mesh = make_dp_device_mesh(dp_size)
        dp_group = device_mesh.get_group("dp")

        shard_rows = (M + dp_size - 1) // dp_size
        padded_rows = shard_rows * dp_size

        torch.manual_seed(42)
        p_local = torch.randn(shard_rows, C, device="cuda")
        p_dtensor = torch.nn.Parameter(
            DTensor.from_local(
                p_local,
                device_mesh=device_mesh,
                placements=[Shard(0)],
                shape=torch.Size([padded_rows, C]),
                stride=p_local.stride(),
                run_check=False,
            )
        )
        initial_local = p_dtensor.data.to_local().clone()

        optimizer = FSDPZeROTensorParallelMuon(
            params=[p_dtensor],
            dp_group=dp_group,
            lr=0.01,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=None,
            tp_mode="duplicated",
        )

        for step_i in range(3):
            torch.manual_seed(200 + step_i)
            grad_local = torch.randn(shard_rows, C, device="cuda")
            p_dtensor.grad = DTensor.from_local(
                grad_local,
                device_mesh=device_mesh,
                placements=[Shard(0)],
                shape=torch.Size([padded_rows, C]),
                stride=grad_local.stride(),
                run_check=False,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        final_local = p_dtensor.data.to_local()
        assert not torch.equal(
            final_local, initial_local
        ), "DTensor param should be updated after 3 steps"
        assert isinstance(p_dtensor.data, DTensor), "Param should remain a DTensor"


# Phase 3: Factory integration + end-to-end


@_skip_if_single_rank()
class TestPhase3FactoryIntegration:
    """Phase 3 tests for `_build_megatron_fsdp_emerging_optimizer` factory."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Initialize distributed for multi-rank tests."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"])
    def test_phase3_factory_dispatches_correct_muon_cls(self, strategy):
        """Factory dispatches `TensorParallelMuon` for "no_shard",
        `FSDPZeROTensorParallelMuon` otherwise.

        Uses mocks to avoid needing a real FSDP-wrapped model for the dispatch logic.
        """
        from megatron.core.optimizer import OptimizerConfig, _build_megatron_fsdp_emerging_optimizer
        from megatron.core.process_groups_config import ProcessGroupCollection

        model_chunk = MagicMock()
        model_chunk.config.num_attention_heads = 8
        model_chunk.config.num_query_groups = 8
        model_chunk.config.kv_channels = 16

        # 2D linear param (Muon-managed).
        linear_weight = torch.nn.Parameter(torch.randn(32, 16, device="cuda"))
        linear_weight.requires_grad = True
        linear_weight.is_embedding_or_output_parameter = False

        # 1D bias (routed to Adam by the default param overrides).
        bias_param = torch.nn.Parameter(torch.randn(32, device="cuda"))
        bias_param.requires_grad = True
        bias_param.is_embedding_or_output_parameter = False

        model_chunk.named_parameters.return_value = [
            ("layer.linear.weight", linear_weight),
            ("layer.linear.bias", bias_param),
        ]
        model_chunk.parameters.return_value = iter([linear_weight, bias_param])

        model_chunk.ddp_config.use_megatron_fsdp = True
        model_chunk.ddp_config.data_parallel_sharding_strategy = strategy

        config = OptimizerConfig(
            optimizer="muon",
            lr=0.01,
            weight_decay=0.01,
            bf16=False,
            fp16=False,
            use_distributed_optimizer=False,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
            muon_split_qkv=False,
            muon_extra_scale_factor=1.0,
        )

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        # `_get_param_groups` would call into Megatron internals
        # (`get_global_unique_param_name` etc.) that don't work on MagicMock
        # chunks, so we stub it out with a minimal one-group fake.
        def fake_get_param_groups(model_chunks, config, overrides):
            return [
                {
                    "params": [linear_weight],
                    "is_expert_parallel": False,
                    "wd_mult": 1.0,
                    "lr_mult": 1.0,
                }
            ]

        # Mock get_megatron_optimizer to avoid full FSDP setup for Adam. Pin
        # the mocked sub-optimizer's `config` to the real config so the
        # ChainedOptimizer assertion that all sub-optimizers share a config
        # passes (defaults would auto-spawn a fresh MagicMock that is unequal).
        with patch("megatron.core.optimizer._get_param_groups", side_effect=fake_get_param_groups):
            with patch("megatron.core.optimizer.get_megatron_optimizer") as mock_get_opt:
                mock_adam = MagicMock()
                mock_adam_sub = MagicMock()
                mock_adam_sub.config = config
                mock_adam.chained_optimizers = [mock_adam_sub]
                mock_get_opt.return_value = mock_adam

                with patch("megatron.core.optimizer._get_mfsdp_models") as mock_mfsdp:
                    mock_mfsdp.return_value = [MagicMock()]

                    result = _build_megatron_fsdp_emerging_optimizer(
                        config=config,
                        model_chunks=[model_chunk],
                        config_overrides={},
                        pg_collection=pg_collection,
                        eopt_name="muon",
                        use_layer_wise=False,
                    )

        assert isinstance(result, FSDPMuonChainedOptimizer)

        from megatron.core.optimizer.optimizer import ChainedOptimizer, FP32Optimizer

        inner = object.__getattribute__(result, "inner")
        assert isinstance(inner, ChainedOptimizer)

        muon_wrapper = inner.chained_optimizers[0]
        assert isinstance(muon_wrapper, FP32Optimizer)

        base_opt = muon_wrapper.optimizer
        if strategy == "no_shard":
            assert isinstance(base_opt, TensorParallelMuon)
            assert not isinstance(base_opt, FSDPZeROTensorParallelMuon)
        else:
            assert isinstance(base_opt, FSDPZeROTensorParallelMuon)
            assert base_opt.dp_group is not None

    def test_phase3_factory_expert_dp_group(self):
        """Verify expert Muon uses `expt_dp` group, non-expert uses `dp_cp`."""
        from megatron.core.optimizer import OptimizerConfig, _build_megatron_fsdp_emerging_optimizer
        from megatron.core.process_groups_config import ProcessGroupCollection

        model_chunk = MagicMock()
        model_chunk.config.num_attention_heads = 8
        model_chunk.config.num_query_groups = 8
        model_chunk.config.kv_channels = 16

        linear_weight = torch.nn.Parameter(torch.randn(32, 16, device="cuda"))
        linear_weight.requires_grad = True
        linear_weight.is_embedding_or_output_parameter = False

        expert_weight = torch.nn.Parameter(torch.randn(32, 16, device="cuda"))
        expert_weight.requires_grad = True
        expert_weight.is_embedding_or_output_parameter = False

        model_chunk.named_parameters.return_value = [
            ("layer.linear.weight", linear_weight),
            ("layer.experts.0.linear.weight", expert_weight),
        ]
        model_chunk.parameters.return_value = iter([linear_weight, expert_weight])

        model_chunk.ddp_config.use_megatron_fsdp = True
        model_chunk.ddp_config.data_parallel_sharding_strategy = "optim"

        config = OptimizerConfig(
            optimizer="muon",
            lr=0.01,
            weight_decay=0.01,
            bf16=False,
            fp16=False,
            use_distributed_optimizer=False,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
            muon_split_qkv=False,
            muon_extra_scale_factor=1.0,
        )

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        def fake_get_param_groups(model_chunks, config, overrides):
            groups = []
            for chunk in model_chunks:
                for name, param in chunk.named_parameters():
                    if not param.requires_grad:
                        continue
                    if len(param.shape) != 2:
                        continue
                    is_expert = "experts" in name and "shared" not in name
                    groups.append(
                        {
                            "params": [param],
                            "is_expert_parallel": is_expert,
                            "wd_mult": 1.0,
                            "lr_mult": 1.0,
                        }
                    )
            return groups

        with patch("megatron.core.optimizer._get_param_groups", side_effect=fake_get_param_groups):
            with patch("megatron.core.optimizer.get_megatron_optimizer") as mock_get_opt:
                mock_adam = MagicMock()
                mock_adam_sub = MagicMock()
                mock_adam_sub.config = config
                mock_adam.chained_optimizers = [mock_adam_sub]
                mock_get_opt.return_value = mock_adam

                with patch("megatron.core.optimizer._get_mfsdp_models") as mock_mfsdp:
                    mock_mfsdp.return_value = [MagicMock()]

                    result = _build_megatron_fsdp_emerging_optimizer(
                        config=config,
                        model_chunks=[model_chunk],
                        config_overrides={},
                        pg_collection=pg_collection,
                        eopt_name="muon",
                        use_layer_wise=False,
                    )

        from megatron.core.optimizer.optimizer import ChainedOptimizer, FP32Optimizer

        inner = object.__getattribute__(result, "inner")
        assert isinstance(inner, ChainedOptimizer)

        # Expect at least 3 optimizers: non-expert Muon, expert Muon, Adam.
        assert len(inner.chained_optimizers) >= 3, (
            f"Expected >= 3 chained optimizers (muon + expert_muon + adam), "
            f"got {len(inner.chained_optimizers)}"
        )

        non_expert_muon = inner.chained_optimizers[0]
        assert isinstance(non_expert_muon, FP32Optimizer)
        assert isinstance(non_expert_muon.optimizer, FSDPZeROTensorParallelMuon)
        assert non_expert_muon.optimizer.dp_group == pg_collection.dp_cp

        expert_muon = inner.chained_optimizers[1]
        assert isinstance(expert_muon, FP32Optimizer)
        assert isinstance(expert_muon.optimizer, FSDPZeROTensorParallelMuon)
        assert expert_muon.optimizer.dp_group == pg_collection.expt_dp
