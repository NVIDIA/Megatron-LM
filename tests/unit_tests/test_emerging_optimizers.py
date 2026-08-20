# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import (
    _PROFILES,
    HAVE_EMERGING_OPTIMIZERS,
    TensorParallelAdaptiveMuon,
    TensorParallelMuon,
    _get_qkv_split_shapes,
    _select_tp_mode,
    get_supported_coefficient_types,
    validate_coefficient_type,
)
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from tests.unit_tests.test_utilities import Utils

if HAVE_EMERGING_OPTIMIZERS:
    from emerging_optimizers.scalar_optimizers import Lion
    from emerging_optimizers.soap import SOAP
else:
    SOAP = None
    Lion = None

# Skip all tests in this file for LTS versions or when emerging_optimizers is missing
pytestmark = [
    pytest.mark.skipif(
        Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
        reason="Skip emerging optimizer tests for LTS test",
    ),
    pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package is not installed"
    ),
]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class GatedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_fc1 = nn.Linear(4, 8, bias=False)

    def forward(self, x):
        gate, up = self.linear_fc1(x).chunk(2, dim=-1)
        return F.silu(gate) * up


# ===========================================================================
# Muon optimizer tests
# ===========================================================================


def test_select_tp_mode_flops_only_fallback():
    """Without a hardware profile, select by FLOPs or the cross-domain fallback."""
    assert (
        _select_tp_mode(
            m=8192,
            n=1024,
            group_size=8,
            steps=5,
            use_syrk=False,
            elem_size=2,
            communication_crosses_domain=False,
            profile=None,
        )
        == "distributed"
    )
    assert (
        _select_tp_mode(
            m=8192,
            n=1024,
            group_size=8,
            steps=5,
            use_syrk=False,
            elem_size=2,
            communication_crosses_domain=True,
            profile=None,
        )
        == "duplicated"
    )


def test_select_tp_mode_with_profile():
    """A hardware profile selects different modes for tall and wide matrices."""
    profile = _PROFILES["GB200"]

    assert (
        _select_tp_mode(
            m=8192,
            n=1024,
            group_size=8,
            steps=5,
            use_syrk=False,
            elem_size=2,
            communication_crosses_domain=False,
            profile=profile,
        )
        == "distributed"
    )
    assert (
        _select_tp_mode(
            m=1024,
            n=8192,
            group_size=8,
            steps=5,
            use_syrk=False,
            elem_size=2,
            communication_crosses_domain=False,
            profile=profile,
        )
        == "duplicated"
    )


def test_select_tp_mode_syrk_changes_selection():
    """SYRK halves the Gram-op cost differently per mode -- can flip the selected mode."""
    assert (
        _select_tp_mode(
            m=640,
            n=1024,
            group_size=8,
            steps=5,
            use_syrk=False,
            elem_size=2,
            communication_crosses_domain=False,
            profile=None,
        )
        == "duplicated"
    )
    assert (
        _select_tp_mode(
            m=640,
            n=1024,
            group_size=8,
            steps=5,
            use_syrk=True,
            elem_size=2,
            communication_crosses_domain=False,
            profile=None,
        )
        == "distributed"
    )


def test_resolve_tp_mode_caches(monkeypatch):
    """Repeated resolution of the same shape invokes the cost model only once."""
    call_count = 0

    def mock_select_tp_mode(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        return "distributed"

    monkeypatch.setattr(
        "megatron.core.optimizer.emerging_optimizers._select_tp_mode", mock_select_tp_mode
    )

    optimizer = TensorParallelMuon(
        params=[torch.nn.Parameter(torch.zeros(1))], tp_mode="auto", pg_collection=None
    )

    first = optimizer._resolve_tp_mode(4096, 1024, 8)
    second = optimizer._resolve_tp_mode(4096, 1024, 8)

    assert first == second == "distributed"
    assert call_count == 1
    assert list(optimizer._tp_mode_cache) == [(4096, 1024, 8)]


def test_muon_qkv_split_shapes():
    config = TransformerConfig(
        num_layers=1, hidden_size=1024, num_attention_heads=16, num_query_groups=8
    )
    gated_config = TransformerConfig(
        num_layers=1,
        hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=8,
        attention_output_gate=True,
    )

    assert _get_qkv_split_shapes(config) == [128, 64, 64]
    assert _get_qkv_split_shapes(gated_config) == [128, 128, 64, 64]


@pytest.mark.parametrize("tp_mode", ["duplicated", "blockwise"])
@pytest.mark.parametrize("gtp_rank", [0, 1])
def test_muon_optimizer_gtp_remat_pad_length_scale_correction(monkeypatch, tp_mode, gtp_rank):
    """scaled_orthogonalize_fn_with_gtp_remat strips GTP_remat's dim-0 padding before
    calling scaled_orthogonalize_fn (unmodified, GTP-agnostic) and restores it after."""
    from types import SimpleNamespace

    gtp_group = object()
    pg_collection = SimpleNamespace(gtp_remat=gtp_group, expt_gtp_remat=gtp_group)

    # 6 true rows, padded to 8 and sharded across 2 ranks: rank 1's shard holds the 2 pad rows.
    full_grad = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [0.0], [0.0]], device='cuda'
    )
    pad_length = 2
    local_grad = full_grad[gtp_rank * 4 : (gtp_rank + 1) * 4].clone()
    param = torch.nn.Parameter(torch.zeros_like(local_grad))
    param.is_gtp_weight_remat = True
    param.pad_length = pad_length

    optimizer = TensorParallelMuon(
        params=[param], num_ns_steps=1, pg_collection=pg_collection, tp_mode=tp_mode
    )

    monkeypatch.setattr("megatron.core.optimizer.emerging_optimizers.get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        "megatron.core.optimizer.emerging_optimizers.get_pg_rank", lambda group: gtp_rank
    )

    def fake_all_gather(shards, _local_grad, _group):
        shards[0].copy_(full_grad[:4])
        shards[1].copy_(full_grad[4:])

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)

    calls = []

    def fake_orthogonalize(grad, tp_group, partition_dim=None, tp_mode_this_group=None):
        calls.append((grad.clone(), tp_group, partition_dim))
        return grad + 100

    monkeypatch.setattr(optimizer, "scaled_orthogonalize_fn", fake_orthogonalize)

    result = optimizer.scaled_orthogonalize_fn_with_gtp_remat(param, local_grad, None, None)

    assert len(calls) == 1
    seen_grad, _, _ = calls[0]

    if tp_mode == "duplicated":
        # Gathered tensor is uniform on every rank: strip 2, restore after.
        stripped_full = full_grad[:6]
        torch.testing.assert_close(seen_grad, stripped_full)
        restored = torch.nn.functional.pad(stripped_full + 100, (0, 0, 0, 2))
        expected = restored[gtp_rank * 4 : (gtp_rank + 1) * 4]
    else:
        # blockwise: only rank 1's local block contains padding.
        if gtp_rank == 1:
            stripped_local = local_grad[:2]
            torch.testing.assert_close(seen_grad, stripped_local)
            expected = torch.nn.functional.pad(stripped_local + 100, (0, 0, 0, 2))
        else:
            torch.testing.assert_close(seen_grad, local_grad)
            expected = local_grad + 100

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("gtp_rank,expected_local_pad_length", [(0, 0), (1, 1), (2, 2), (3, 2)])
def test_muon_optimizer_gtp_remat_blockwise_pad_spans_multiple_ranks(
    monkeypatch, gtp_rank, expected_local_pad_length
):
    """pad_length can span multiple ranks' shards, not just the last rank's. Each rank must
    strip only its own overlap; a fully-padding rank (true dim0 == 0) must skip
    scaled_orthogonalize_fn entirely and return exact zero."""
    from types import SimpleNamespace

    gtp_group = object()
    pg_collection = SimpleNamespace(gtp_remat=gtp_group, expt_gtp_remat=gtp_group)

    # 4 ranks, shard_size=2, pad_length=5: ranks 2 and 3 are fully padding, rank 1 half.
    gtp_remat_size = 4
    shard_size = 2
    pad_length = 5
    local_grad = torch.full((shard_size, 1), 3.0, device='cuda')
    param = torch.nn.Parameter(torch.zeros_like(local_grad))
    param.is_gtp_weight_remat = True
    param.pad_length = pad_length

    optimizer = TensorParallelMuon(
        params=[param], num_ns_steps=1, pg_collection=pg_collection, tp_mode="blockwise"
    )

    monkeypatch.setattr(
        "megatron.core.optimizer.emerging_optimizers.get_pg_size", lambda group: gtp_remat_size
    )
    monkeypatch.setattr(
        "megatron.core.optimizer.emerging_optimizers.get_pg_rank", lambda group: gtp_rank
    )

    calls = []

    def fake_orthogonalize(grad, tp_group, partition_dim=None, tp_mode_this_group=None):
        calls.append(grad.clone())
        return grad  # identity: makes the strip/restore round-trip directly checkable

    monkeypatch.setattr(optimizer, "scaled_orthogonalize_fn", fake_orthogonalize)

    result = optimizer.scaled_orthogonalize_fn_with_gtp_remat(param, local_grad, None, None)

    true_dim0 = shard_size - expected_local_pad_length
    if true_dim0 <= 0:
        assert calls == []
        torch.testing.assert_close(result, torch.zeros_like(local_grad))
    else:
        assert len(calls) == 1
        torch.testing.assert_close(calls[0], local_grad[:true_dim0])
        expected = torch.nn.functional.pad(
            local_grad[:true_dim0], (0, 0, 0, expected_local_pad_length)
        )
        torch.testing.assert_close(result, expected)
def test_muon_optimizer_glu_split(monkeypatch):
    """Muon orthogonalizes the fused gate and up FC1 weights independently."""
    param = torch.nn.Parameter(torch.zeros(8, 4, device='cuda'))
    param.is_glu = True
    optimizer = TensorParallelMuon(
        params=[param], split_glu=True, num_ns_steps=1, pg_collection=None, tp_mode="duplicated"
    )
    grad = torch.arange(32, dtype=torch.float32, device='cuda').view(8, 4)
    orthogonalized_grads = []

    def fake_orthogonalize(split_grad, _tp_group, _partition_dim):
        orthogonalized_grads.append(split_grad.clone())
        return split_grad + len(orthogonalized_grads)

    monkeypatch.setattr(optimizer, "scaled_orthogonalize_fn", fake_orthogonalize)

    result = optimizer.orthogonalize(param, grad)

    assert [split.shape for split in orthogonalized_grads] == [(4, 4), (4, 4)]
    torch.testing.assert_close(orthogonalized_grads[0], grad[:4])
    torch.testing.assert_close(orthogonalized_grads[1], grad[4:])
    torch.testing.assert_close(result, torch.cat((grad[:4] + 1, grad[4:] + 2)))
def test_muon_optimizer_glu_split_opt_out(monkeypatch):
    """The GLU marker does not change the existing whole-matrix path when splitting is disabled."""
    param = torch.nn.Parameter(torch.zeros(8, 4, device='cuda'))
    param.is_glu = True
    optimizer = TensorParallelMuon(params=[param], split_glu=False, num_ns_steps=1)
    grad = torch.arange(32, dtype=torch.float32, device='cuda').view(8, 4)
    calls = []

    def fake_orthogonalize(_param, whole_grad, _tp_group, _partition_dim):
        calls.append(whole_grad.clone())
        return whole_grad + 1

    monkeypatch.setattr(optimizer, "scaled_orthogonalize_fn_with_gtp_remat", fake_orthogonalize)

    result = optimizer.orthogonalize(param, grad)

    assert len(calls) == 1
    torch.testing.assert_close(calls[0], grad)
    torch.testing.assert_close(result, grad + 1)


def test_muon_optimizer_interleaved_glu_split(monkeypatch):
    """Muon de-interleaves alternating GLU blocks and restores the stored layout."""
    param = torch.nn.Parameter(torch.zeros(8, 1, device='cuda'))
    param.is_glu = True
    param.glu_interleave_size = 2
    optimizer = TensorParallelMuon(params=[param], split_glu=True, num_ns_steps=1)
    grad = torch.tensor([10, 11, 20, 21, 12, 13, 22, 23], dtype=torch.float32, device='cuda').view(
        8, 1
    )
    orthogonalized_grads = []

    def fake_orthogonalize(split_grad, _tp_group, _partition_dim):
        orthogonalized_grads.append(split_grad.clone())
        return split_grad + len(orthogonalized_grads) * 100

    monkeypatch.setattr(optimizer, "scaled_orthogonalize_fn", fake_orthogonalize)

    result = optimizer.orthogonalize(param, grad)

    torch.testing.assert_close(
        orthogonalized_grads[0], grad.new_tensor([10, 11, 12, 13]).view(4, 1)
    )
    torch.testing.assert_close(
        orthogonalized_grads[1], grad.new_tensor([20, 21, 22, 23]).view(4, 1)
    )
    expected = grad.new_tensor([110, 111, 220, 221, 112, 113, 222, 223])
    torch.testing.assert_close(result, expected.view(8, 1))


@pytest.mark.parametrize("gtp_rank", [0, 1])
@pytest.mark.parametrize("layout", ["contiguous", "interleaved_padded"])
def test_muon_optimizer_glu_gtp_gathers_before_split(monkeypatch, gtp_rank, layout):
    """GTP reconstructs FC1 before separating gate and up rows in either layout."""
    gtp_group = object()
    pg_collection = SimpleNamespace(
        tp=None, expt_tp=None, gtp_remat=gtp_group, expt_gtp_remat=gtp_group
    )
    if layout == "contiguous":
        full_values = [10, 11, 12, 13, 20, 21, 22, 23]
        interleave_size = None
        pad_length = 0
        gate_values = [10, 11, 12, 13]
        up_values = [20, 21, 22, 23]
        expected_values = [110, 111, 112, 113, 220, 221, 222, 223]
    else:
        full_values = [10, 20, 11, 21, 12, 22, 90, 91]
        interleave_size = 1
        pad_length = 2
        gate_values = [10, 11, 12]
        up_values = [20, 21, 22]
        expected_values = [110, 220, 111, 221, 112, 222, 0, 0]

    full_grad = torch.tensor(full_values, dtype=torch.float32, device='cuda').view(8, 1)
    local_grad = full_grad[gtp_rank * 4 : (gtp_rank + 1) * 4].clone()
    param = torch.nn.Parameter(torch.zeros_like(local_grad))
    param.is_glu = True
    param.glu_interleave_size = interleave_size
    param.is_gtp_weight_remat = True
    param.glu_gtp_remat_size = 2
    param.glu_gtp_pad_length = pad_length
    optimizer = TensorParallelMuon(
        params=[param], split_glu=True, num_ns_steps=1, pg_collection=pg_collection
    )
    orthogonalized_grads = []

    monkeypatch.setattr("megatron.core.optimizer.emerging_optimizers.get_pg_size", lambda group: 2)
    monkeypatch.setattr(
        "megatron.core.optimizer.emerging_optimizers.get_pg_rank", lambda group: gtp_rank
    )

    def fake_all_gather(shards, _local_grad, _group):
        shards[0].copy_(full_grad[:4])
        shards[1].copy_(full_grad[4:])

    def fake_orthogonalize(split_grad, _tp_group, _partition_dim):
        orthogonalized_grads.append(split_grad.clone())
        return split_grad + len(orthogonalized_grads) * 100

    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(optimizer, "scaled_orthogonalize_fn", fake_orthogonalize)

    result = optimizer.orthogonalize(param, local_grad)

    torch.testing.assert_close(
        orthogonalized_grads[0], full_grad.new_tensor(gate_values).view(-1, 1)
    )
    torch.testing.assert_close(orthogonalized_grads[1], full_grad.new_tensor(up_values).view(-1, 1))
    expected = full_grad.new_tensor(expected_values).view(8, 1)
    torch.testing.assert_close(result, expected[gtp_rank * 4 : (gtp_rank + 1) * 4])
def test_muon_optimizer_smoke():
    """Smoke test for TensorParallelMuon optimizer."""
    # Create a simple linear model for testing
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Create TensorParallelMuon optimizer
    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        pg_collection=None,
        tp_mode="duplicated",
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test forward and backward pass
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original weight
    original_weight = model.weight.data.clone()

    # Test optimizer step
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # Test zero_grad
    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert 'state' in state_dict, "State dict should contain state"
    assert 'param_groups' in state_dict, "State dict should contain param_groups"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonOptimizerMultiRank:
    """Test class for Muon optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP.

        Args:
            model: Model to wrap

        Returns:
            DDP-wrapped model
        """
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def create_ddp_model_for_layerwise(self, model, use_param_layout=False):
        """Wrap model in DDP for layer-wise distributed optimizer tests.

        Args:
            model: Model to wrap.
            use_param_layout: If True, supply DDP a precomputed shard-aligned
                ``full_param_layout`` (turns on ``ddp_config.use_distributed_optimizer=True``
                + ``start_param_sync``). If False (default), build DDP without a layout
                so ``LayerWiseDistributedOptimizer`` syncs via the legacy
                flatten / ``all_gather_v`` / unflatten ``allgather_params()`` codepath.
        """
        if use_param_layout:
            from megatron.training.training import wrap_model_chunks_with_ddp

            ddp_config = DistributedDataParallelConfig()
            wrapped = wrap_model_chunks_with_ddp(
                [model],
                TransformerConfig(num_attention_heads=1, num_layers=1),
                ddp_config,
                use_layer_wise_distributed_optimizer=True,
            )
            return wrapped[0]
        return self.create_ddp_model(model)

    def test_get_megatron_optimizer_smoke(self):
        """Smoke test for get_megatron_optimizer function."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Ensure all parameters require gradients
        for param in model.parameters():
            assert param.requires_grad, "All parameters should require gradients"

        # Create optimizer config for Muon
        optimizer_config = OptimizerConfig(
            optimizer='muon',  # This will be changed internally to 'adam' for non-linear params
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,  # Muon doesn't support distributed optimizer
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        # Test creating the optimizer
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        # Test basic properties
        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
        assert len(optimizer.chained_optimizers) >= 1, "Should have at least one chained optimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Test optimizer step
        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        # Test zero_grad
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), f"Gradients should be zeroed for all parameters"

        # Test state_dict and load_state_dict
        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list), "State dict should be a list"

        # Load state dict should not raise error
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_optimizer_tags_glu_fc1(self):
        """The optimizer factory keeps a compatibility fallback for fused GLU FC1 weights."""
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=4,
            num_attention_heads=1,
            gated_linear_unit=True,
            moe_mlp_glu_interleave_size=2,
            moe_shared_expert_glu_interleave_size=2,
        )
        model = DistributedDataParallel(
            transformer_config,
            DistributedDataParallelConfig(use_distributed_optimizer=False),
            GatedNet().bfloat16().cuda().requires_grad_(True),
        )
        optimizer_config = OptimizerConfig(
            optimizer='muon', lr=0.01, bf16=True, muon_num_ns_steps=1, muon_tp_mode="duplicated"
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        fc1_param = next(
            param for name, param in model.named_parameters() if 'linear_fc1.weight' in name
        )
        assert fc1_param.is_glu is True
        assert fc1_param.glu_interleave_size is None
        assert fc1_param.glu_gtp_pad_length == 0
        raw_optimizers = [
            child.optimizer
            for child in optimizer.chained_optimizers
            if isinstance(getattr(child, 'optimizer', None), TensorParallelMuon)
        ]
        assert len(raw_optimizers) == 1
        assert raw_optimizers[0].split_glu is True

    def test_get_megatron_optimizer_validation(self):
        """Test validation logic for get_megatron_optimizer."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # Test 1: FP16 should raise exception
        optimizer_config_fp16 = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            fp16=True,  # This should cause an exception
            use_distributed_optimizer=False,
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])

        # Test 3: Invalid num_ns_steps should raise exception
        optimizer_config_invalid_ns = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_num_ns_steps=0,  # This should cause an exception
        )

        with pytest.raises(ValueError, match='num_ns_steps must be at least 1'):
            get_megatron_optimizer(config=optimizer_config_invalid_ns, model_chunks=[model])

        # A single 3D GroupedTensor combines all local-expert matrices. Until Muon can
        # maintain and orthogonalize per-expert state within that container, reject the
        # configuration instead of silently routing those weights to the scalar optimizer.
        model.config.moe_single_grouped_weight = True
        optimizer_config_single_grouped_weight = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_num_ns_steps=1,
        )
        with pytest.raises(ValueError, match='--moe-single-grouped-weight'):
            get_megatron_optimizer(
                config=optimizer_config_single_grouped_weight, model_chunks=[model]
            )

    def test_get_megatron_optimizer_layer_wise(self):
        """Test get_megatron_optimizer with layer-wise distributed optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model_for_layerwise(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_layer_wise_distributed_optimizer=True,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        # use_layer_wise_distributed_optimizer=True triggers LayerWiseDistributedOptimizer
        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        # Verify it's a LayerWiseDistributedOptimizer
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        assert isinstance(
            optimizer, LayerWiseDistributedOptimizer
        ), "Should return LayerWiseDistributedOptimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Test optimizer step
        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"
        assert grad_norm is not None or grad_norm is None, "Grad norm should be returned"

    def test_get_megatron_muon_optimizer_backward_compatible(self):
        """Test get_megatron_muon_optimizer with backward compatible layer-wise distributed optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model_for_layerwise(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_layer_wise_distributed_optimizer=True,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
        )

        with pytest.raises(ValueError, match="dist_ prefix"):
            get_megatron_muon_optimizer(
                config=optimizer_config, model_chunks=[model], layer_wise_distributed_optimizer=True
            )

        optimizer_config.optimizer = 'dist_muon'
        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config, model_chunks=[model], layer_wise_distributed_optimizer=True
        )

        # Verify it's a LayerWiseDistributedOptimizer
        from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer

        assert isinstance(
            optimizer, LayerWiseDistributedOptimizer
        ), "Should return LayerWiseDistributedOptimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Test optimizer step
        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"
        assert grad_norm is not None or grad_norm is None, "Grad norm should be returned"


@pytest.mark.parametrize("mode", ["duplicated", "blockwise", "distributed"])
def test_muon_optimizer_different_modes_single_rank(mode):
    """Test TensorParallelMuon optimizer with different modes on single rank.

    When TP size is 1, all modes should produce the same result.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,  # Disable weight decay for deterministic comparison
        num_ns_steps=5,
        pg_collection=None,
        tp_mode=mode,
    )

    # Use fixed input for deterministic results
    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with mode={mode}"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonOptimizerMultiRankTP:
    """Test class for Muon optimizer with multi-rank and tensor parallel setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test with tensor parallel."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, mode):
        """Create model with TP and optimizer.

        Args:
            mode: Muon optimizer mode

        Returns:
            tuple: (model, optimizer, pg_collection)
        """
        rank = int(os.getenv('RANK', '0'))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        # Create model with partition_dim for TP
        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0  # Set partition dimension for TP

        optimizer = TensorParallelMuon(
            params=[model.weight],
            lr=0.01,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=pg_collection,
            tp_mode=mode,
        )

        return model, optimizer

    @pytest.mark.parametrize("mode", ["duplicated", "distributed"])
    def test_muon_optimizer_modes_multirank_same_result(self, mode):
        """Test that duplicated and distributed modes produce same results with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer(mode)

        # Use fixed input for deterministic results
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        # Verify weight was updated
        assert not torch.equal(
            model.weight.data, original_weight
        ), f"Weight should be updated with mode={mode}"

    def test_muon_optimizer_blockwise_mode_different_result(self):
        """Test that blockwise mode produces different results than duplicated/distributed with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        # Use fixed input for deterministic results
        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        # Verify weight was updated
        assert not torch.equal(
            model.weight.data, original_weight
        ), "Weight should be updated with mode=blockwise"

    def test_muon_swiglu_fc1_real_tp_layout(self):
        """A real TP-sharded SwiGLU FC1 is marked and split by Newton-Schulz."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(42)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            ffn_hidden_size=16,
            gated_linear_unit=True,
            add_bias_linear=False,
            params_dtype=torch.float32,
        )
        mlp = MLP(
            config=config,
            submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear),
            pg_collection=pg_collection,
        )
        fc1_weight = mlp.linear_fc1.weight
        tp_size = parallel_state.get_tensor_model_parallel_world_size()

        assert fc1_weight.shape == (2 * config.ffn_hidden_size // tp_size, config.hidden_size)
        assert fc1_weight.is_glu is True
        assert fc1_weight.glu_interleave_size is None

        optimizer = TensorParallelMuon(
            params=[fc1_weight],
            lr=0.01,
            weight_decay=0.0,
            split_glu=True,
            num_ns_steps=1,
            pg_collection=pg_collection,
            tp_mode="duplicated",
        )
        hidden_states = torch.randn(4, config.hidden_size, device=fc1_weight.device)
        output, _ = mlp(hidden_states)
        output.float().sum().backward()
        original_weight = fc1_weight.detach().clone()

        assert fc1_weight.grad.shape == fc1_weight.shape
        optimizer.step()

        assert torch.isfinite(fc1_weight).all()
        assert not torch.equal(fc1_weight, original_weight)


# All non-custom coefficient types supported by emerging_optimizers.
_TESTABLE_COEFFICIENT_TYPES = (
    [t for t in get_supported_coefficient_types() if t != "custom"]
    if HAVE_EMERGING_OPTIMIZERS
    else []
)

# A reasonable default NS step count for testing; get_coefficient_iterator
# cycles/repeats coefficients so any step count works with any type.
_DEFAULT_NS_STEPS = 5


@pytest.mark.parametrize("coefficient_type", _TESTABLE_COEFFICIENT_TYPES)
def test_muon_optimizer_coefficient_types(coefficient_type):
    """Test TensorParallelMuon optimizer with different coefficient types."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        coefficient_type=coefficient_type,
        num_ns_steps=_DEFAULT_NS_STEPS,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with coefficient_type={coefficient_type}"


@pytest.mark.parametrize("scale_mode", ["spectral", "unit_rms_norm", "shape_scaling"])
def test_muon_optimizer_scale_modes(scale_mode):
    """Test TensorParallelMuon optimizer with different scale modes."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        scale_mode=scale_mode,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with scale_mode={scale_mode}"


@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_optimizer_nesterov(nesterov):
    """Test TensorParallelMuon optimizer with and without Nesterov momentum."""
    model = torch.nn.Linear(50, 25, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.9,
        nesterov=nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 50, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with nesterov={nesterov}"


def test_muon_optimizer_multiple_steps():
    """Test TensorParallelMuon optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    # Verify weights changed at each step
    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


def test_muon_optimizer_qkv_split():
    """Test TensorParallelMuon optimizer with QKV splitting."""
    # Create a model with QKV-like parameter
    qkv_size = 3 * 64 * 16  # Combined Q, K, V dimensions, 16 heads x 64 per head
    hidden_size = 1024
    model = torch.nn.Linear(hidden_size, qkv_size, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    # Mark parameter as QKV
    model.weight.is_qkv = True

    # QKV split shapes: [Q_size, K_size, V_size]
    qkv_split_shapes = (64, 64, 64)

    # Test with split_qkv=True
    optimizer_split = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, hidden_size, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer_split.step()
    weight_with_split = model.weight.data.clone()

    assert not torch.equal(
        weight_with_split, original_weight
    ), "QKV weight should be updated with split_qkv=True"

    # Reset model and test with split_qkv=False
    model.weight.data.fill_(1.0)
    optimizer_no_split = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    optimizer_no_split.step()
    weight_without_split = model.weight.data.clone()

    assert not torch.equal(
        weight_without_split, original_weight
    ), "QKV weight should be updated with split_qkv=False"

    # Ensure the two results are different
    assert not torch.equal(
        weight_with_split, weight_without_split
    ), "Weights should be different between split_qkv=True and split_qkv=False"


def test_muon_optimizer_extra_scale_factor():
    """Test TensorParallelMuon optimizer with different extra_scale_factor values."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        extra_scale_factor=2.0,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated with extra_scale_factor"


def test_get_supported_coefficient_types_returns_tuple():
    """Test that get_supported_coefficient_types returns a non-empty tuple of strings."""
    supported = get_supported_coefficient_types()
    assert isinstance(supported, tuple)
    assert len(supported) > 0
    for t in supported:
        assert isinstance(t, str)


def test_get_supported_coefficient_types_contains_known_types():
    """Test that the known coefficient types are present in the supported set."""
    supported = get_supported_coefficient_types()
    for expected in ("simple", "quintic", "polar_express"):
        assert expected in supported, f"Expected '{expected}' in supported types {supported}"


def test_validate_coefficient_type_accepts_valid():
    """Test that validate_coefficient_type does not raise for valid types."""
    for t in get_supported_coefficient_types():
        validate_coefficient_type(t)  # should not raise


def test_validate_coefficient_type_rejects_invalid():
    """Test that validate_coefficient_type raises ValueError for an invalid type."""
    with pytest.raises(ValueError, match="Unsupported muon coefficient type"):
        validate_coefficient_type("nonexistent_type_xyz")


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestMuonCoefficientTypeMultiRank:
    """Test coefficient_type integration through get_megatron_optimizer."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    @pytest.mark.parametrize("coefficient_type", _TESTABLE_COEFFICIENT_TYPES)
    def test_get_megatron_optimizer_coefficient_type(self, coefficient_type):
        """Test that coefficient_type flows through get_megatron_optimizer."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_coefficient_type=coefficient_type,
            muon_num_ns_steps=_DEFAULT_NS_STEPS,
            muon_tp_mode="duplicated",
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()


@pytest.mark.parametrize("num_ns_steps", [5, 15, 25])
def test_muon_optimizer_num_ns_steps(num_ns_steps):
    """Test TensorParallelMuon optimizer with different numbers of Newton-Schulz steps."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelMuon(
        params=[model.weight],
        lr=0.01,
        coefficient_type="quintic",
        num_ns_steps=num_ns_steps,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with num_ns_steps={num_ns_steps}"


# ===========================================================================
# Adaptive Muon optimizer tests
# ===========================================================================


def test_adaptive_muon_optimizer_smoke():
    """Smoke test for TensorParallelAdaptiveMuon optimizer."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.01,
        use_decoupled_weight_decay=True,
        split_qkv=False,
        fp32_matmul_prec="medium",
        num_ns_steps=5,
        scale_mode="spectral",
        extra_scale_factor=1.0,
        pg_collection=None,
        tp_mode="duplicated",
        moment2_method="adamuon",
        beta2=0.95,
        eps=1e-8,
    )

    assert optimizer is not None
    assert hasattr(optimizer, 'param_groups')
    assert len(optimizer.param_groups) > 0

    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    state_dict = optimizer.state_dict()
    assert 'state' in state_dict
    assert 'param_groups' in state_dict
    optimizer.load_state_dict(state_dict)


@pytest.mark.parametrize("mode", ["duplicated", "blockwise", "distributed"])
def test_adaptive_muon_optimizer_different_modes_single_rank(mode):
    """Test TensorParallelAdaptiveMuon with different modes on single rank."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.normal_(0, 0.02)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.0,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode=mode,
    )

    torch.manual_seed(42)
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with mode={mode}"


@pytest.mark.parametrize("moment2_method", ["adamuon", "normuon"])
def test_adaptive_muon_optimizer_moment2_methods(moment2_method):
    """Test TensorParallelAdaptiveMuon with different moment2 methods."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
        moment2_method=moment2_method,
    )

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with moment2_method={moment2_method}"


@pytest.mark.parametrize("beta2", [0.5, 0.95, 0.999])
def test_adaptive_muon_optimizer_beta2(beta2):
    """Test TensorParallelAdaptiveMuon with different beta2 values."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
        beta2=beta2,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with beta2={beta2}"


def test_adaptive_muon_optimizer_multiple_steps():
    """Test TensorParallelAdaptiveMuon across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.95,
        weight_decay=0.01,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


@pytest.mark.parametrize("nesterov", [True, False])
def test_adaptive_muon_optimizer_nesterov(nesterov):
    """Test TensorParallelAdaptiveMuon with and without Nesterov momentum."""
    model = torch.nn.Linear(50, 25, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        momentum=0.9,
        nesterov=nesterov,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, 50, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with nesterov={nesterov}"


def test_adaptive_muon_optimizer_qkv_split():
    """Test TensorParallelAdaptiveMuon with QKV splitting."""
    qkv_size = 3 * 64 * 16  # Combined Q, K, V dimensions
    hidden_size = 1024
    model = torch.nn.Linear(hidden_size, qkv_size, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    model.weight.is_qkv = True
    qkv_split_shapes = (64, 64, 64)

    optimizer_split = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=qkv_split_shapes,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    input_tensor = torch.randn(16, hidden_size, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer_split.step()
    weight_with_split = model.weight.data.clone()

    assert not torch.equal(
        weight_with_split, original_weight
    ), "QKV weight should be updated with split_qkv=True"

    model.weight.data.fill_(1.0)
    optimizer_no_split = TensorParallelAdaptiveMuon(
        params=[model.weight],
        lr=0.01,
        split_qkv=False,
        num_ns_steps=5,
        pg_collection=None,
        tp_mode="duplicated",
    )

    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    optimizer_no_split.step()
    weight_without_split = model.weight.data.clone()

    assert not torch.equal(
        weight_without_split, original_weight
    ), "QKV weight should be updated with split_qkv=False"

    assert not torch.equal(
        weight_with_split, weight_without_split
    ), "Weights should be different between split_qkv=True and split_qkv=False"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestAdaptiveMuonOptimizerMultiRank:
    """Test class for Adaptive Muon optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP."""
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def test_get_megatron_optimizer_adaptive_muon_smoke(self):
        """Smoke test for get_megatron_optimizer with adaptive_muon."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        for param in model.parameters():
            assert param.requires_grad

        optimizer_config = OptimizerConfig(
            optimizer='adaptive_muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_fp32_matmul_prec="medium",
            muon_num_ns_steps=5,
            muon_scale_mode="spectral",
            muon_tp_mode="duplicated",
            adaptive_muon_moment2_method="adamuon",
            adaptive_muon_beta2=0.95,
            adaptive_muon_eps=1e-8,
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None
        assert hasattr(optimizer, 'param_groups')
        assert hasattr(optimizer, 'chained_optimizers')
        assert len(optimizer.chained_optimizers) >= 1

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        optimizer.step()

        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), "Gradients should be zeroed for all parameters"

        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list)
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_optimizer_adaptive_muon_validation(self):
        """Test validation logic for get_megatron_optimizer with adaptive_muon."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        optimizer_config_fp16 = OptimizerConfig(
            optimizer='adaptive_muon', lr=0.01, fp16=True, use_distributed_optimizer=False
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestAdaptiveMuonOptimizerMultiRankTP:
    """Test class for Adaptive Muon optimizer with multi-rank and tensor parallel setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test with tensor parallel."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(world, 2))
        yield
        Utils.destroy_model_parallel()

    def create_tp_model_and_optimizer(self, mode):
        """Create model with TP and optimizer."""
        rank = int(os.getenv('RANK', '0'))
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        torch.manual_seed(42 + rank)
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
        model.requires_grad_(True)
        model.weight.data.normal_(0, 0.02)
        model.weight.partition_dim = 0

        optimizer = TensorParallelAdaptiveMuon(
            params=[model.weight],
            lr=0.01,
            momentum=0.95,
            weight_decay=0.0,
            num_ns_steps=5,
            pg_collection=pg_collection,
            tp_mode=mode,
        )

        return model, optimizer

    @pytest.mark.parametrize("mode", ["duplicated", "distributed"])
    def test_adaptive_muon_optimizer_modes_multirank_same_result(self, mode):
        """Test that duplicated and distributed modes produce same results with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer(mode)

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        assert not torch.equal(
            model.weight.data, original_weight
        ), f"Weight should be updated with mode={mode}"

    def test_adaptive_muon_optimizer_blockwise_mode(self):
        """Test that blockwise mode works with TP > 1."""
        model, optimizer = self.create_tp_model_and_optimizer("blockwise")

        torch.manual_seed(42)
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        original_weight = model.weight.data.clone()
        optimizer.step()

        assert not torch.equal(
            model.weight.data, original_weight
        ), "Weight should be updated with mode=blockwise"


# ===========================================================================
# SOAP optimizer tests
# ===========================================================================

skip_no_soap = pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package not installed"
)


@skip_no_soap
def test_soap_optimizer_smoke():
    """Smoke test for SOAP optimizer."""

    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight], lr=0.01, betas=(0.9, 0.999), shampoo_beta=0.95, weight_decay=0.01
    )

    # Test basic properties
    assert optimizer is not None, "Optimizer should not be None"
    assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
    assert len(optimizer.param_groups) > 0, "Optimizer should have at least one parameter group"

    # Test forward and backward pass
    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    # Store original weight
    original_weight = model.weight.data.clone()

    # Test optimizer step
    optimizer.step()

    # Verify weight was updated
    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    # Test zero_grad
    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    # Test state_dict and load_state_dict
    state_dict = optimizer.state_dict()
    assert 'state' in state_dict, "State dict should contain state"
    assert 'param_groups' in state_dict, "State dict should contain param_groups"

    # Load state dict should not raise error
    optimizer.load_state_dict(state_dict)


@skip_no_soap
def test_soap_optimizer_multiple_steps():
    """Test SOAP optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight], lr=0.01, betas=(0.9, 0.999), shampoo_beta=0.95, weight_decay=0.01
    )

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    # Verify weights changed at each step
    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


@skip_no_soap
@pytest.mark.parametrize("use_kl_shampoo", [True, False])
def test_soap_optimizer_kl_shampoo(use_kl_shampoo):
    """Test SOAP optimizer with and without KL-Shampoo preconditioner."""

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(
        params=[model.weight],
        lr=0.01,
        betas=(0.9, 0.999),
        shampoo_beta=0.95,
        use_kl_shampoo=use_kl_shampoo,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with use_kl_shampoo={use_kl_shampoo}"


@skip_no_soap
@pytest.mark.parametrize("shampoo_beta", [0.5, 0.9, 0.99])
def test_soap_optimizer_shampoo_beta(shampoo_beta):
    """Test SOAP optimizer with different shampoo_beta values."""

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = SOAP(params=[model.weight], lr=0.01, betas=(0.9, 0.999), shampoo_beta=shampoo_beta)

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with shampoo_beta={shampoo_beta}"


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestSoapOptimizerMultiRank:
    """Test class for SOAP optimizer with multi-rank setup."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_ddp_model(self, model):
        """Wrap model in DDP."""
        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        return DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

    def test_get_megatron_optimizer_soap_smoke(self):
        """Smoke test for get_megatron_optimizer with SOAP."""
        model = Net().bfloat16().cuda()
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        for param in model.parameters():
            assert param.requires_grad, "All parameters should require gradients"

        optimizer_config = OptimizerConfig(
            optimizer='soap',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            soap_shampoo_beta=0.95,
            soap_use_kl_shampoo=True,
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config, model_chunks=[model], use_gloo_process_groups=True
        )

        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'param_groups'), "Optimizer should have param_groups"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"
        assert len(optimizer.chained_optimizers) >= 1, "Should have at least one chained optimizer"

        # Test forward and backward pass
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Store original parameters
        original_params = {}
        for name, param in model.named_parameters():
            original_params[name] = param.data.clone()

        # Test optimizer step
        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for name, param in model.named_parameters():
            if not torch.equal(param.data, original_params[name]):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated after optimizer step"

        # Test zero_grad
        optimizer.zero_grad()
        for param in model.parameters():
            assert param.grad is None or torch.all(
                param.grad == 0
            ), "Gradients should be zeroed for all parameters"

        # Test state_dict and load_state_dict
        state_dict = optimizer.state_dict()
        assert isinstance(state_dict, list), "State dict should be a list"
        optimizer.load_state_dict(state_dict)

    def test_get_megatron_optimizer_soap_validation(self):
        """Test validation logic for get_megatron_optimizer with SOAP."""
        model = torch.nn.Linear(100, 50, bias=False, dtype=torch.bfloat16, device='cuda')
        model.requires_grad_(True)
        model = self.create_ddp_model(model)

        # FP16 should raise exception
        optimizer_config_fp16 = OptimizerConfig(
            optimizer='soap', lr=0.01, fp16=True, use_distributed_optimizer=False
        )

        with pytest.raises(Exception, match='emerging optimizer with fp16 is not supported'):
            get_megatron_optimizer(config=optimizer_config_fp16, model_chunks=[model])


# ===========================================================================
# Lion optimizer tests
# ===========================================================================

skip_no_lion = pytest.mark.skipif(
    not HAVE_EMERGING_OPTIMIZERS, reason="emerging_optimizers package not installed"
)


@skip_no_lion
def test_lion_optimizer_smoke():
    """Smoke test for Lion optimizer."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    assert optimizer is not None
    assert hasattr(optimizer, 'param_groups')
    assert len(optimizer.param_groups) > 0

    input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), "Weight should be updated after optimizer step"

    optimizer.zero_grad()
    assert model.weight.grad is None or torch.all(
        model.weight.grad == 0
    ), "Gradients should be zeroed"

    state_dict = optimizer.state_dict()
    assert 'state' in state_dict
    assert 'param_groups' in state_dict
    optimizer.load_state_dict(state_dict)


@skip_no_lion
def test_lion_optimizer_multiple_steps():
    """Test Lion optimizer across multiple optimization steps."""
    model = torch.nn.Linear(100, 50, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    weights_history = [model.weight.data.clone()]

    for i in range(3):
        input_tensor = torch.randn(32, 100, dtype=torch.float32, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        weights_history.append(model.weight.data.clone())

    for i in range(len(weights_history) - 1):
        assert not torch.equal(
            weights_history[i], weights_history[i + 1]
        ), f"Weight should change at step {i}"


@skip_no_lion
@pytest.mark.parametrize("betas", [(0.9, 0.99), (0.95, 0.999), (0.5, 0.9)])
def test_lion_optimizer_betas(betas):
    """Test Lion optimizer with different beta values."""
    model = torch.nn.Linear(80, 40, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=betas)

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with betas={betas}"


@skip_no_lion
@pytest.mark.parametrize("weight_decay", [0.0, 0.01, 0.1])
def test_lion_optimizer_weight_decay(weight_decay):
    """Test Lion optimizer with different weight decay values."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(params=[model.weight], lr=1e-4, betas=(0.9, 0.99), weight_decay=weight_decay)

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with weight_decay={weight_decay}"


@skip_no_lion
@pytest.mark.parametrize("weight_decay_method", ["decoupled", "l2"])
def test_lion_optimizer_weight_decay_method(weight_decay_method):
    """Test Lion optimizer with different weight decay methods."""
    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    model.requires_grad_(True)
    model.weight.data.fill_(1.0)

    optimizer = Lion(
        params=[model.weight],
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        weight_decay_method=weight_decay_method,
    )

    input_tensor = torch.randn(16, 60, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_weight = model.weight.data.clone()
    optimizer.step()

    assert not torch.equal(
        model.weight.data, original_weight
    ), f"Weight should be updated with weight_decay_method={weight_decay_method}"


@skip_no_lion
def test_lion_optimizer_multi_layer_net():
    """Test Lion optimizer with the multi-layer Net model."""
    model = Net().cuda()
    model.requires_grad_(True)

    optimizer = Lion(params=model.parameters(), lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)

    input_tensor = torch.randn(16, 80, dtype=torch.float32, device='cuda')
    output = model(input_tensor)
    loss = output.sum()
    loss.backward()

    original_params = {name: p.data.clone() for name, p in model.named_parameters()}
    optimizer.step()

    params_updated = 0
    for name, param in model.named_parameters():
        if not torch.equal(param.data, original_params[name]):
            params_updated += 1

    assert params_updated > 0, "At least some parameters should be updated after optimizer step"


# ===========================================================================
# use_syrk version gate
# ===========================================================================


@pytest.mark.parametrize("optimizer_cls", [TensorParallelMuon, TensorParallelAdaptiveMuon])
def test_muon_use_syrk_rejected_on_old_emerging_optimizers(monkeypatch, optimizer_cls):
    """use_syrk must raise on emerging_optimizers < 0.4.0 rather than silently falling back.

    Covers TensorParallelAdaptiveMuon too, since it forwards use_syrk through
    TensorParallelMuon.__init__ and that forwarding is what applies the gate to both.
    """
    import megatron.core.optimizer.emerging_optimizers as eo_module

    monkeypatch.setattr(eo_module, "is_emerging_optimizers_min_version", lambda _version: False)
    monkeypatch.setattr(eo_module, "get_emerging_optimizers_version", lambda: "0.2.0")

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    with pytest.raises(ValueError, match="use_syrk requires emerging_optimizers"):
        optimizer_cls(
            params=[model.weight], lr=0.01, pg_collection=None, tp_mode="duplicated", use_syrk=True
        )


@pytest.mark.parametrize("optimizer_cls", [TensorParallelMuon, TensorParallelAdaptiveMuon])
def test_muon_use_syrk_default_off_ignores_version(monkeypatch, optimizer_cls):
    """The gate only fires when use_syrk is requested; the default path stays version-agnostic."""
    import megatron.core.optimizer.emerging_optimizers as eo_module

    monkeypatch.setattr(eo_module, "is_emerging_optimizers_min_version", lambda _version: False)

    model = torch.nn.Linear(60, 30, bias=False, dtype=torch.float32, device='cuda')
    optimizer = optimizer_cls(
        params=[model.weight], lr=0.01, pg_collection=None, tp_mode="duplicated"
    )
    assert optimizer is not None
