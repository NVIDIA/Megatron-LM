# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import logging
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    TensorParallelAdaptiveMuon,
    TensorParallelMuon,
    _get_qkv_split_shapes,
    _localize_qkv_split_shapes,
    _qkv_split_groups_are_complete,
    get_supported_coefficient_types,
    validate_coefficient_type,
)
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.attention import QKVLayout
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
from tests.unit_tests.test_utilities import Utils

if HAVE_EMERGING_OPTIMIZERS:
    from emerging_optimizers import utils as emerging_optimizer_utils
    from emerging_optimizers.scalar_optimizers import Lion

    from megatron.core.optimizer.emerging_optimizers import SOAP
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


# ===========================================================================
# Muon optimizer tests
# ===========================================================================


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
    assert _get_qkv_split_shapes(config, split_qkv_per_head=True) == [64] * 32
    assert _get_qkv_split_shapes(gated_config, split_qkv_per_head=True) == [64] * 48

    mla_layout = QKVLayout.from_splits(4, (128, 64))
    assert _get_qkv_split_shapes(mla_layout) == [128, 64]
    assert _get_qkv_split_shapes(mla_layout, split_qkv_per_head=True) == [128, 64] * 4


def test_muon_local_qkv_head_split_shapes_can_differ_by_tp_rank():
    """Rank-local per-head layouts report complete and fragmented heads."""
    global_split_shapes = [64] * 20

    rank_0_shapes, rank_0_complete = _localize_qkv_split_shapes(
        global_split_shapes, local_start=0, local_rows=160
    )
    rank_1_shapes, rank_1_complete = _localize_qkv_split_shapes(
        global_split_shapes, local_start=160, local_rows=160
    )
    aligned_shapes, aligned_complete = _localize_qkv_split_shapes(
        global_split_shapes, local_start=0, local_rows=640
    )

    assert rank_0_shapes == [64, 64, 32]
    assert rank_1_shapes == [32, 64, 64]
    assert not rank_0_complete
    assert not rank_1_complete
    assert aligned_shapes == [64] * 10
    assert aligned_complete


def test_muon_qkv_query_group_layout_localization():
    """Projection splitting detects query groups fragmented by TP row ranges."""
    split_shapes = [256, 64, 64]

    assert _qkv_split_groups_are_complete(split_shapes, local_start=0, local_rows=384)
    assert _qkv_split_groups_are_complete(split_shapes, local_start=384, local_rows=768)
    assert not _qkv_split_groups_are_complete(split_shapes, local_start=0, local_rows=192)
    assert not _qkv_split_groups_are_complete(split_shapes, local_start=192, local_rows=192)


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

    def test_optimizer_factory_tags_qkv_when_tp_exceeds_query_groups(self):
        """The public optimizer factory tags query groups fragmented across TP ranks."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_size = pg_collection.tp.size()
        assert tp_size == 2
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            num_query_groups=1,
            kv_channels=4,
            tensor_model_parallel_size=tp_size,
            use_cpu_initialization=False,
            add_bias_linear=False,
        )
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=32,
            max_sequence_length=8,
            pre_process=False,
            post_process=False,
            pg_collection=pg_collection,
        )
        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            use_distributed_optimizer=False,
            muon_split_qkv=True,
            muon_tp_mode="blockwise",
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )

        qkv_weight = model.decoder.layers[0].self_attention.linear_qkv.weight
        assert optimizer is not None
        assert qkv_weight.shape[0] == 8
        assert qkv_weight.is_qkv
        assert qkv_weight.qkv_split_shapes == [8, 4, 4]
        assert qkv_weight.qkv_split_shapes_global == [8, 4, 4]
        assert not qkv_weight.qkv_split_groups_are_complete

    @pytest.mark.parametrize("split_per_head", [False, True])
    def test_optimizer_factory_uses_heterogeneous_layer_qkv_layout(self, split_per_head):
        """Each heterogeneous attention layer supplies its own logical QKV layout."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_size = pg_collection.tp.size()
        assert tp_size == 2
        model_parallel_cuda_manual_seed(123)
        block_configs = {
            "block_configs": [
                {
                    "attention": {
                        "no_op": False,
                        "replace_with_linear": False,
                        "num_query_groups": 2,
                    },
                    "mlp": {"no_op": False, "replace_with_linear": False, "ffn_hidden_size": 16},
                },
                {
                    "attention": {
                        "no_op": False,
                        "replace_with_linear": False,
                        "num_query_groups": 1,
                    },
                    "mlp": {"no_op": False, "replace_with_linear": False, "ffn_hidden_size": 16},
                },
            ]
        }
        transformer_config = HeterogeneousTransformerConfig(
            num_layers=2,
            hidden_size=8,
            num_attention_heads=2,
            kv_channels=4,
            tensor_model_parallel_size=tp_size,
            use_cpu_initialization=False,
            add_bias_linear=False,
            heterogeneous_layers_config_encoded_json=json.dumps(block_configs),
        )
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_heterogeneous_layer_spec(transformer_config),
            vocab_size=32,
            max_sequence_length=8,
            pre_process=False,
            post_process=False,
            pg_collection=pg_collection,
        )
        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            use_distributed_optimizer=False,
            muon_split_qkv=True,
            muon_split_qkv_per_head=split_per_head,
            muon_tp_mode="blockwise",
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )

        first_qkv = model.decoder.layers[0].self_attention.linear_qkv.weight
        second_qkv = model.decoder.layers[1].self_attention.linear_qkv.weight
        assert optimizer is not None
        assert transformer_config.num_query_groups == 2
        assert first_qkv.qkv_layout.num_groups == 2
        assert second_qkv.qkv_layout.num_groups == 1
        assert first_qkv.shape[0] == 12
        assert second_qkv.shape[0] == 8
        assert first_qkv.is_qkv
        assert second_qkv.is_qkv
        if split_per_head:
            assert first_qkv.qkv_split_shapes_global == [4] * 6
            assert second_qkv.qkv_split_shapes_global == [4] * 4
            assert first_qkv.qkv_split_heads_are_complete
            assert second_qkv.qkv_split_heads_are_complete
        else:
            assert first_qkv.qkv_split_shapes_global == [4, 4, 4] * 2
            assert second_qkv.qkv_split_shapes_global == [8, 4, 4]
            assert first_qkv.qkv_split_groups_are_complete
            assert not second_qkv.qkv_split_groups_are_complete

    @pytest.mark.parametrize("split_per_head", [False, True])
    @pytest.mark.parametrize("q_lora_rank", [None, 4])
    def test_optimizer_factory_uses_mla_up_projection_layouts(self, split_per_head, q_lora_rank):
        """MLA up-projections use module-owned layouts with TP-aware Muon splitting."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_size = pg_collection.tp.size()
        assert tp_size == 2
        model_parallel_cuda_manual_seed(123)
        transformer_config = MLATransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=4,
            qk_head_dim=4,
            qk_pos_emb_head_dim=2,
            v_head_dim=3,
            tensor_model_parallel_size=tp_size,
            use_cpu_initialization=False,
            add_bias_linear=False,
            multi_latent_attention=True,
            rope_type="rope",
            rotary_base=10000,
            original_max_position_embeddings=8,
        )
        model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                multi_latent_attention=True
            ),
            vocab_size=32,
            max_sequence_length=8,
            pre_process=False,
            post_process=False,
            pg_collection=pg_collection,
        )
        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            use_distributed_optimizer=False,
            muon_split_qkv=True,
            muon_split_qkv_per_head=split_per_head,
            muon_tp_mode="blockwise",
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )

        attention = model.decoder.layers[0].self_attention
        q_up_weight = (
            attention.linear_q_proj.weight
            if q_lora_rank is None
            else attention.linear_q_up_proj.weight
        )
        kv_up_weight = attention.linear_kv_up_proj.weight
        assert optimizer is not None
        assert q_up_weight.is_qkv
        assert kv_up_weight.is_qkv
        assert q_up_weight.qkv_split_shapes_global == [4, 2] * 2
        assert kv_up_weight.qkv_split_shapes_global == [4, 3] * 2
        if split_per_head:
            assert q_up_weight.qkv_split_heads_are_complete
            assert kv_up_weight.qkv_split_heads_are_complete
        else:
            assert q_up_weight.qkv_split_groups_are_complete
            assert kv_up_weight.qkv_split_groups_are_complete
        if q_lora_rank is not None:
            assert not getattr(attention.linear_q_down_proj.weight, 'is_qkv', False)
        assert not getattr(attention.linear_kv_down_proj.weight, 'is_qkv', False)

    def test_optimizer_factory_skips_mismatched_qkv_layout(self):
        """A QKV layout mismatch falls back to whole-matrix Muon orthogonalization."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_size = pg_collection.tp.size()
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            num_query_groups=1,
            kv_channels=4,
            tensor_model_parallel_size=tp_size,
        )
        model = torch.nn.Module()
        model.config = transformer_config
        model.linear_qkv = torch.nn.Linear(8, 7, bias=False, dtype=torch.float32, device='cuda')
        model.linear_qkv.weight.tensor_model_parallel = True
        model.linear_qkv.weight.partition_dim = 0
        optimizer_config = OptimizerConfig(
            optimizer='muon', lr=0.01, use_distributed_optimizer=False, muon_split_qkv=True
        )

        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )

        qkv_weight = model.linear_qkv.weight
        assert optimizer is not None
        assert not qkv_weight.is_qkv
        assert qkv_weight.qkv_split_shapes is None
        assert qkv_weight.qkv_split_shapes_global is None

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

    def test_muon_optimizer_per_head_split_gathers_fragmented_heads(self):
        """Per-head splitting reconstructs heads that cross TP rank boundaries."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_group = pg_collection.tp
        tp_rank = tp_group.rank()
        local_grad = torch.arange(12, dtype=torch.float32, device='cuda').view(3, 4)
        local_grad = local_grad + tp_rank * local_grad.numel()
        param = torch.nn.Parameter(torch.zeros_like(local_grad))
        param.partition_dim = 0
        param.is_qkv = True
        param.qkv_split_shapes = [2, 2]
        param.qkv_split_shapes_global = [2, 2, 2]
        param.qkv_split_heads_are_complete = False

        optimizer = TensorParallelMuon(
            params=[param],
            split_qkv=True,
            split_qkv_per_head=True,
            is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
            qkv_split_shapes=[2, 2, 2],
            pg_collection=pg_collection,
            tp_mode="blockwise",
        )

        def center_rows(x, tp_group=None, partition_dim=None):
            del tp_group, partition_dim
            return x - x.mean(dim=-2, keepdim=True)

        optimizer.scaled_orthogonalize_fn = center_rows
        actual = optimizer.orthogonalize(param, local_grad)

        shards = [torch.empty_like(local_grad) for _ in range(tp_group.size())]
        torch.distributed.all_gather(shards, local_grad, tp_group)
        global_grad = torch.cat(shards, dim=0)
        expected_global = torch.cat(
            [center_rows(head) for head in torch.split(global_grad, [2, 2, 2], dim=0)], dim=0
        )
        expected = expected_global[tp_rank * 3 : (tp_rank + 1) * 3]
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize("split_shapes", ([4, 2, 2], [4, 4, 2, 2], [3, 5]))
    def test_muon_optimizer_projection_split_gathers_fragmented_query_group(self, split_shapes):
        """Projection splitting reconstructs a query group split over TP ranks."""
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        tp_group = pg_collection.tp
        tp_rank = tp_group.rank()
        global_rows = sum(split_shapes)
        assert global_rows % tp_group.size() == 0
        local_rows = global_rows // tp_group.size()
        local_grad = torch.arange(local_rows * 4, dtype=torch.float32, device='cuda').view(
            local_rows, 4
        )
        local_grad = local_grad + tp_rank * local_grad.numel()
        param = torch.nn.Parameter(torch.zeros_like(local_grad))
        param.partition_dim = 0
        param.is_qkv = True
        param.qkv_split_shapes = split_shapes
        param.qkv_split_shapes_global = split_shapes
        param.qkv_split_groups_are_complete = False

        optimizer = TensorParallelMuon(
            params=[param],
            split_qkv=True,
            is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
            qkv_split_shapes=split_shapes,
            pg_collection=pg_collection,
            tp_mode="blockwise",
        )

        def center_rows(x, tp_group=None, partition_dim=None):
            del tp_group, partition_dim
            return x - x.mean(dim=-2, keepdim=True)

        optimizer.scaled_orthogonalize_fn = center_rows
        actual = optimizer.orthogonalize(param, local_grad)

        shards = [torch.empty_like(local_grad) for _ in range(tp_group.size())]
        torch.distributed.all_gather(shards, local_grad, tp_group)
        global_grad = torch.cat(shards, dim=0)
        expected_global = torch.cat(
            [
                center_rows(projection)
                for projection in torch.split(global_grad, split_shapes, dim=0)
            ],
            dim=0,
        )
        expected = expected_global[tp_rank * local_rows : (tp_rank + 1) * local_rows]
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("layout", "expects_fallback"),
    (("local_projection", False), ("per_head", True), ("fragmented", True)),
)
def test_muon_qkv_distributed_mode_routing_warns_once(monkeypatch, layout, expects_fallback):
    """Only complete local projection splits retain distributed NS."""
    grad = torch.arange(16, dtype=torch.float32, device='cuda').view(4, 4)
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.partition_dim = 0
    param.is_qkv = True
    param.qkv_split_shapes = [2, 1, 1]
    param.qkv_split_shapes_global = [2, 1, 1]
    param.qkv_split_groups_are_complete = layout != "fragmented"
    param.qkv_split_heads_are_complete = True

    optimizer = TensorParallelMuon(
        params=[param],
        split_qkv=True,
        split_qkv_per_head=layout == "per_head",
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[2, 1, 1],
        pg_collection=None,
        tp_mode="distributed",
    )
    orthogonalize_args = []
    log_records = []

    def passthrough(x, tp_group=None, partition_dim=None):
        orthogonalize_args.append((tp_group, partition_dim))
        return x

    def record_log(_logger, level, message):
        log_records.append((level, message))

    optimizer.scaled_orthogonalize_fn = passthrough
    monkeypatch.setattr("megatron.core.optimizer.emerging_optimizers.log_single_rank", record_log)

    torch.testing.assert_close(optimizer.orthogonalize(param, grad), grad)
    torch.testing.assert_close(optimizer.orthogonalize(param, grad), grad)

    warning_messages = [message for level, message in log_records if level == logging.WARNING]
    if expects_fallback:
        assert len(warning_messages) == 1
        assert "falling back to non-TP Newton-Schulz" in warning_messages[0]
        assert all(
            tp_group is None and partition_dim is None
            for tp_group, partition_dim in orthogonalize_args
        )
    else:
        assert warning_messages == []
        assert all(partition_dim == 0 for _, partition_dim in orthogonalize_args)


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


def test_muon_optimizer_qkv_split_per_head_is_opt_in():
    """Per-head splitting is guarded and differs from projection splitting."""
    grad = torch.arange(48, dtype=torch.float32, device='cuda').view(16, 3)
    projection_param = torch.nn.Parameter(torch.zeros_like(grad))
    projection_param.is_qkv = True
    projection_param.qkv_split_shapes = [4, 2, 2]
    head_param = torch.nn.Parameter(torch.zeros_like(grad))
    head_param.is_qkv = True
    head_param.qkv_split_shapes = [2] * 8
    orthogonalize_call_shapes = []

    def center_rows(x, tp_group=None, partition_dim=None):
        del tp_group, partition_dim
        orthogonalize_call_shapes.append(tuple(x.shape))
        return x - x.mean(dim=-2, keepdim=True)

    projection_optimizer = TensorParallelMuon(
        params=[projection_param],
        split_qkv=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[4, 2, 2],
        pg_collection=None,
    )
    projection_optimizer.scaled_orthogonalize_fn = center_rows
    projection_out = projection_optimizer.orthogonalize(projection_param, grad)
    orthogonalize_call_shapes.clear()

    head_optimizer = TensorParallelMuon(
        params=[head_param],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[2] * 8,
        pg_collection=None,
    )
    head_optimizer.scaled_orthogonalize_fn = center_rows
    head_out = head_optimizer.orthogonalize(head_param, grad)
    assert orthogonalize_call_shapes == [(8, 2, 3)]

    expected_head_out = torch.cat(
        [center_rows(head) for head in torch.split(grad, [2] * 8, dim=0)], dim=0
    )
    torch.testing.assert_close(head_out, expected_head_out)
    assert not torch.equal(projection_out, head_out)


def test_muon_optimizer_qkv_split_per_head_requires_split_qkv():
    """The per-head switch cannot enable QKV splitting by itself."""
    param = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.float32, device='cuda'))
    with pytest.raises(ValueError, match="split_qkv_per_head requires split_qkv=True"):
        TensorParallelMuon(
            params=[param], split_qkv=False, split_qkv_per_head=True, pg_collection=None
        )


def test_muon_optimizer_uniform_per_head_splits_use_batched_ns():
    """Uniform per-head splits use Emerging-Optimizers' batched Newton-Schulz path."""
    grad = torch.arange(16, dtype=torch.float32, device='cuda').view(4, 4)
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.is_qkv = True
    param.qkv_split_shapes = [2, 2]
    optimizer = TensorParallelMuon(
        params=[param],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[2, 2],
        pg_collection=None,
    )
    call_shapes = []

    def center_rows(x, tp_group=None, partition_dim=None):
        del tp_group, partition_dim
        call_shapes.append(tuple(x.shape))
        return x - x.mean(dim=-2, keepdim=True)

    optimizer.scaled_orthogonalize_fn = center_rows
    actual = optimizer.orthogonalize(param, grad)
    assert call_shapes == [(2, 2, 4)]
    expected = torch.cat(
        [head - head.mean(dim=-2, keepdim=True) for head in torch.split(grad, [2, 2])]
    )
    torch.testing.assert_close(actual, expected)


def test_muon_optimizer_batched_per_head_ns_matches_individual_heads():
    """The pinned Emerging-Optimizers 3D Newton-Schulz path matches 2D head calls."""
    torch.manual_seed(42)
    grad = torch.randn(8, 16, dtype=torch.float32, device='cuda')
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.is_qkv = True
    param.qkv_split_shapes = [2] * 4
    optimizer = TensorParallelMuon(
        params=[param],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[2] * 4,
        fp32_matmul_prec="highest",
        num_ns_steps=2,
        pg_collection=None,
    )

    with emerging_optimizer_utils.fp32_matmul_precision(optimizer.fp32_matmul_prec):
        actual = optimizer.orthogonalize(param, grad)
        expected = torch.cat(
            [
                optimizer.scaled_orthogonalize_fn(head, tp_group=None, partition_dim=None)
                for head in torch.split(grad, [2] * 4)
            ]
        )
    torch.testing.assert_close(actual, expected)


def test_muon_optimizer_nonuniform_per_head_splits_use_unbatched_ns():
    """Nonuniform per-head splits keep using individual Newton-Schulz calls."""
    grad = torch.arange(12, dtype=torch.float32, device='cuda').view(3, 4)
    param = torch.nn.Parameter(torch.zeros_like(grad))
    param.is_qkv = True
    param.qkv_split_shapes = [2, 1]
    optimizer = TensorParallelMuon(
        params=[param],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: getattr(p, 'is_qkv', False),
        qkv_split_shapes=[2, 1],
        pg_collection=None,
    )
    call_shapes = []

    def center_rows(x, tp_group=None, partition_dim=None):
        del tp_group, partition_dim
        call_shapes.append(tuple(x.shape))
        return x - x.mean(dim=-2, keepdim=True)

    optimizer.scaled_orthogonalize_fn = center_rows
    actual = optimizer.orthogonalize(param, grad)
    assert call_shapes == [(2, 4), (1, 4)]
    expected = torch.cat([center_rows(head) for head in torch.split(grad, [2, 1])])
    torch.testing.assert_close(actual, expected)


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
