# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for Muon + Megatron-FSDP integration."""

import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from packaging.version import Version

from megatron.core.optimizer.emerging_optimizers import (
    FSDPMuonChainedOptimizer,
    FSDPZeROTensorParallelMuon,
    HAVE_EMERGING_OPTIMIZERS,
    TensorParallelMuon,
    _get_mfsdp_models,
)
from tests.unit_tests.test_utilities import Utils


pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv("NVIDIA_PYTORCH_VERSION", "99.99")) <= Version("25.05"),
        reason="Skip Muon FSDP optimizer tests on LTS containers",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS,
        reason="Muon tests require the emerging-optimizers package",
    ),
]

WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))


def _skip_if_single_rank():
    return pytest.mark.skipif(WORLD_SIZE <= 1, reason="Multi-rank test requires WORLD_SIZE > 1")


def _skip_if_no_dtensor():
    return pytest.mark.skipif(
        Version(torch.__version__.split("+")[0]) < Version("2.4.0"),
        reason="DTensor tests require PyTorch >= 2.4.0",
    )


def _make_fsdp_muon(params, dp_group=None, **kwargs):
    defaults = dict(
        lr=0.05,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        num_ns_steps=2,
        pg_collection=None,
        tp_mode="duplicated",
        split_qkv=False,
    )
    defaults.update(kwargs)
    return FSDPZeROTensorParallelMuon(params=params, dp_group=dp_group, **defaults)


def _reference_update(full_grad, **kwargs):
    from emerging_optimizers import utils

    defaults = dict(
        lr=0.05,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,
        num_ns_steps=2,
        pg_collection=None,
        tp_mode="duplicated",
        split_qkv=False,
    )
    defaults.update(kwargs)
    param = nn.Parameter(torch.zeros_like(full_grad))
    optimizer = TensorParallelMuon(params=[param], **defaults)
    with utils.fp32_matmul_precision(optimizer.fp32_matmul_prec):
        return optimizer.orthogonalize(param, full_grad)


def _local_rows(plan, rank):
    return plan.get(rank, 0)


def _local_slice(full_tensor, plan, rank):
    start = sum(_local_rows(plan, r) for r in range(rank))
    rows = _local_rows(plan, rank)
    return full_tensor[start : start + rows].clone()


def _make_dtensor(local_tensor, global_shape, device_mesh):
    from torch.distributed.tensor import DTensor, Shard

    from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
        update_uneven_dtensor_chunk_metadata,
    )

    global_stride = torch.empty(
        global_shape, dtype=local_tensor.dtype, device=local_tensor.device
    ).stride()
    dtensor = DTensor.from_local(
        local_tensor=local_tensor,
        device_mesh=device_mesh,
        placements=[Shard(0)],
        shape=torch.Size(global_shape),
        stride=global_stride,
        run_check=False,
    )
    update_uneven_dtensor_chunk_metadata(dtensor)
    return dtensor


class TestFSDPMuonChainedOptimizer:
    @staticmethod
    def _make_mfsdp_mock(model_auto_sync=False):
        mock = MagicMock()
        mock.model_auto_sync = model_auto_sync
        return mock

    @staticmethod
    def _make_inner_mock():
        mock = MagicMock()
        mock.step.return_value = (True, 1.0, 0)
        return mock

    @pytest.mark.parametrize("model_auto_sync", [True, False])
    def test_step_protocol(self, model_auto_sync):
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

    def test_getattr_delegation(self):
        inner = self._make_inner_mock()
        inner.param_groups = [{"lr": 0.01}]
        inner.state_dict.return_value = {"state": {}, "param_groups": []}
        wrapper = FSDPMuonChainedOptimizer(inner, [self._make_mfsdp_mock()])

        assert wrapper.param_groups == [{"lr": 0.01}]
        assert wrapper.state_dict() == {"state": {}, "param_groups": []}

        wrapper.load_state_dict({"state": {}, "param_groups": []})
        inner.load_state_dict.assert_called_once()

    @pytest.mark.parametrize("set_to_none", [True, False])
    def test_zero_grad_delegates(self, set_to_none):
        inner = self._make_inner_mock()
        wrapper = FSDPMuonChainedOptimizer(inner, [self._make_mfsdp_mock()])

        wrapper.zero_grad(set_to_none=set_to_none)

        inner.zero_grad.assert_called_once_with(set_to_none)

    def test_get_mfsdp_models_error(self):
        with pytest.raises(RuntimeError, match="Could not find any MegatronFSDP"):
            _get_mfsdp_models([nn.Module()])

    def test_get_mfsdp_models_success(self):
        inner_module = MagicMock()
        chunk = MagicMock()
        chunk.module = inner_module
        chunk.finish_grad_sync = MagicMock()

        assert _get_mfsdp_models([chunk]) == [inner_module]


@_skip_if_single_rank()
@_skip_if_no_dtensor()
class TestFSDPZeROTensorParallelMuon:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def test_step_gathers_only_split_boundary_params(self):
        from torch.distributed.device_mesh import init_device_mesh

        from megatron.core.optimizer import emerging_optimizers as eopt_mod

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        # Param 0 is fully local on rank 0. Param 1 crosses the rank 0/1
        # boundary. Param 2 is fully local on rank 1. Other ranks hold empty
        # local shards and still participate in the boundary gather for param 1.
        plans = [
            {0: 4},
            {0: 2, 1: 4},
            {1: 5},
        ]
        full_params = [
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) / 100
            for rows in (4, 6, 5)
        ]
        full_grads = [
            torch.arange(rows * cols, device="cuda", dtype=torch.float32).view(rows, cols) / 50
            + 0.1
            for rows in (4, 6, 5)
        ]

        params = []
        initial_locals = []
        for full_param, plan in zip(full_params, plans):
            local_param = _local_slice(full_param, plan, dp_rank).contiguous()
            param = nn.Parameter(_make_dtensor(local_param, full_param.shape, device_mesh))
            params.append(param)
            initial_locals.append(local_param.clone())

        for param, full_grad, plan in zip(params, full_grads, plans):
            local_grad = _local_slice(full_grad, plan, dp_rank).contiguous()
            param.grad = _make_dtensor(local_grad, full_grad.shape, device_mesh)

        optimizer = _make_fsdp_muon(params, dp_group=dp_group)
        assert optimizer._get_boundary_gather_param_indices(optimizer.param_groups[0]) == {1}

        real_gather = eopt_mod.gather_uneven_dtensor_to_full_tensor
        gather_shapes = []

        def counting_gather(value):
            gather_shapes.append(tuple(value.shape))
            return real_gather(value)

        with patch.object(eopt_mod, "gather_uneven_dtensor_to_full_tensor", counting_gather):
            optimizer.step()

        assert gather_shapes == [(6, cols)]

        lr = optimizer.param_groups[0]["lr"]
        for idx, (param, full_param, full_grad, plan) in enumerate(
            zip(params, full_params, full_grads, plans)
        ):
            expected_update = _reference_update(full_grad)
            expected_local = _local_slice(full_param - lr * expected_update, plan, dp_rank)
            local_value = param.data.to_local()

            if local_value.numel() == 0:
                assert local_value.shape[0] == 0
                continue

            torch.testing.assert_close(
                local_value,
                expected_local,
                atol=1e-5,
                rtol=1e-4,
                msg=f"param {idx} local update mismatch on rank {dp_rank}",
            )
            assert not torch.equal(local_value, initial_locals[idx])

    def test_boundary_detector_includes_middle_split_params(self):
        from torch.distributed.device_mesh import init_device_mesh

        dp_size = torch.distributed.get_world_size()
        dp_rank = torch.distributed.get_rank()
        device_mesh = init_device_mesh("cuda", (dp_size,), mesh_dim_names=("dp",))
        dp_group = device_mesh.get_group("dp")
        cols = 4

        # Optimizer groups can exclude Adam-managed tensors that are present in
        # the underlying M-FSDP flat buffer. After that filtering, a split Muon
        # tensor is not guaranteed to be first or last in the Muon param group.
        plans = [
            {0: 2, 1: 2},
            {0: 1, 1: 3},
            {0: 3, 1: 1},
        ]

        params = []
        for plan in plans:
            full_param = torch.zeros(sum(plan.values()), cols, device="cuda")
            local_param = _local_slice(full_param, plan, dp_rank).contiguous()
            params.append(nn.Parameter(_make_dtensor(local_param, full_param.shape, device_mesh)))

        optimizer = _make_fsdp_muon(params, dp_group=dp_group)

        assert optimizer._get_boundary_gather_param_indices(optimizer.param_groups[0]) == {
            0,
            1,
            2,
        }


@_skip_if_single_rank()
class TestFSDPFactoryIntegration:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"])
    def test_factory_dispatches_correct_muon_cls(self, strategy):
        from megatron.core.optimizer import OptimizerConfig, _build_megatron_fsdp_emerging_optimizer
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.optimizer.optimizer import ChainedOptimizer, FP32Optimizer

        model_chunk = MagicMock()
        model_chunk.config.num_attention_heads = 8
        model_chunk.config.num_query_groups = 8
        model_chunk.config.kv_channels = 16

        linear_weight = nn.Parameter(torch.randn(32, 16, device="cuda"))
        linear_weight.is_embedding_or_output_parameter = False
        bias_param = nn.Parameter(torch.randn(32, device="cuda"))
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

        def fake_get_param_groups(model_chunks, config, overrides):
            return [
                {
                    "params": [linear_weight],
                    "is_expert_parallel": False,
                    "wd_mult": 1.0,
                    "lr_mult": 1.0,
                }
            ]

        with patch("megatron.core.optimizer._get_param_groups", side_effect=fake_get_param_groups):
            with patch("megatron.core.optimizer.get_megatron_optimizer") as mock_get_opt:
                mock_adam = MagicMock()
                mock_adam_sub = MagicMock()
                mock_adam_sub.config = config
                mock_adam.chained_optimizers = [mock_adam_sub]
                mock_get_opt.return_value = mock_adam
                with patch("megatron.core.optimizer._get_mfsdp_models", return_value=[MagicMock()]):
                    result = _build_megatron_fsdp_emerging_optimizer(
                        config=config,
                        model_chunks=[model_chunk],
                        config_overrides={},
                        pg_collection=pg_collection,
                        eopt_name="muon",
                        use_layer_wise=False,
                    )

        assert isinstance(result, FSDPMuonChainedOptimizer)
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
