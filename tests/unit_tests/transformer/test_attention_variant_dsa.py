# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerLossAutoScaler,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
    compute_dsa_indexer_loss,
    rotate_activation,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False
    _hadamard_transform = None


def mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Mock implementation of hadamard_transform for testing without the library installed.

    This is a simple identity-like transformation that preserves shape and applies scaling.
    """
    return x * scale


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    """Automatically patch hadamard_transform in dsa module if not installed."""
    if not HAVE_HADAMARD:
        with patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
            mock_hadamard_transform,
        ):
            yield
    else:
        yield


class TestRotateActivation:
    """Test rotate_activation function."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def test_rotate_activation_shape(self):
        """Test that rotate_activation preserves shape."""
        batch_size = 2
        seq_len = 16
        hidden_size = 128

        x = torch.randn(seq_len, batch_size, hidden_size, dtype=torch.bfloat16).cuda()
        output = rotate_activation(x)

        assert output.shape == x.shape
        assert output.dtype == torch.bfloat16

    def test_rotate_activation_dtype_check(self):
        """Test that rotate_activation only accepts bfloat16."""
        x = torch.randn(16, 2, 128, dtype=torch.float32).cuda()

        with pytest.raises(AssertionError, match="only support bf16"):
            rotate_activation(x)


@pytest.mark.parametrize("seqlen_and_topk", [[16, 32], [64, 32]])
class TestComputeDSAIndexerLoss:
    """Test compute_dsa_indexer_loss function."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_loss_shape(self, seqlen_and_topk):
        """Test that indexer loss returns a scalar."""
        batch_size = 2
        seqlen = seqlen_and_topk[0]
        num_heads = 4
        head_dim = 128
        index_topk = seqlen_and_topk[1]

        # Create dummy index scores
        index_scores = torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32).cuda()

        # Apply causal mask to index_scores before computing topk
        causal_mask = torch.triu(
            torch.full(
                (seqlen, seqlen), float('-inf'), dtype=torch.float32, device=index_scores.device
            ),
            diagonal=1,
        )
        # [batch_size, seqlen, seqlen] + [seqlen, seqlen] -> [batch_size, seqlen, seqlen]
        masked_index_scores = index_scores + causal_mask

        # Get topk indices from masked index_scores
        topk_k = min(index_topk, seqlen)
        topk_indices = masked_index_scores.topk(topk_k, dim=-1)[1]

        query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        key = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        softmax_scale = head_dim**-0.5

        loss = compute_dsa_indexer_loss(
            index_scores=index_scores,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=softmax_scale,
            loss_coeff=1.0,
            sparse_loss=False,
            pg_collection=self.pg_collection,
        )

        assert loss.shape == torch.Size([])
        assert loss.dtype == torch.float32
        assert loss >= 0  # KL divergence should be non-negative

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_loss_sparse(self, seqlen_and_topk):
        """Test sparse indexer loss computation."""
        batch_size = 2
        seqlen = seqlen_and_topk[0]
        num_heads = 4
        head_dim = 128
        index_topk = seqlen_and_topk[1]

        # Create dummy index scores
        index_scores = torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32).cuda()

        # Apply causal mask to index_scores before computing topk
        causal_mask = torch.triu(
            torch.full(
                (seqlen, seqlen), float('-inf'), dtype=torch.float32, device=index_scores.device
            ),
            diagonal=1,
        )
        # [batch_size, seqlen, seqlen] + [seqlen, seqlen] -> [batch_size, seqlen, seqlen]
        masked_index_scores = index_scores + causal_mask

        # Get topk indices from masked index_scores
        topk_k = min(index_topk, seqlen)
        topk_indices = masked_index_scores.topk(topk_k, dim=-1)[1]

        query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        key = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        softmax_scale = head_dim**-0.5

        loss_sparse = compute_dsa_indexer_loss(
            index_scores=index_scores,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=softmax_scale,
            loss_coeff=1.0,
            sparse_loss=True,
            pg_collection=self.pg_collection,
        )

        loss_dense = compute_dsa_indexer_loss(
            index_scores=index_scores,
            topk_indices=topk_indices,
            query=query,
            key=key,
            softmax_scale=softmax_scale,
            loss_coeff=1.0,
            sparse_loss=False,
            pg_collection=self.pg_collection,
        )

        # Sparse loss should be different from dense loss
        if seqlen > index_topk:
            assert loss_sparse != loss_dense
        else:
            assert loss_sparse == loss_dense
        assert loss_sparse >= 0
        assert loss_dense >= 0


class TestDSAIndexerLossAutoScaler:
    """Test DSAIndexerLossAutoScaler autograd function."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_pass(self):
        """Test that forward pass preserves output."""
        output = torch.randn(16, 2, 128).cuda()
        output.requires_grad_(True)
        indexer_loss = torch.tensor(0.5).cuda()
        indexer_loss.requires_grad_(True)

        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        assert torch.allclose(result, output, atol=0, rtol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_pass(self):
        """Test that backward pass triggers indexer loss backward and scales gradient correctly."""
        output = torch.randn(16, 2, 128).cuda()
        output.requires_grad_(True)

        # Create indexer_loss with computation graph
        # This simulates compute_dsa_indexer_loss which computes KL divergence
        dummy_input = torch.randn(10).cuda()
        dummy_input.requires_grad_(True)
        indexer_loss = dummy_input.mean()

        # Set loss scale
        scale = torch.tensor(2.0).cuda()
        DSAIndexerLossAutoScaler.set_loss_scale(scale)

        # Apply the autograd function
        result = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

        # Trigger backward
        main_loss = result.sum()
        main_loss.backward()

        # Check that gradients flow back to output
        assert output.grad is not None, "Gradient should flow back to parameters"

        # Check that indexer_loss backward was triggered
        assert dummy_input.grad is not None, "Indexer loss backward should be triggered"

        # Verify the gradient is scaled correctly
        expected_grad_per_element = scale.item() / len(dummy_input)
        assert torch.allclose(
            dummy_input.grad,
            torch.full_like(dummy_input, expected_grad_per_element),
            rtol=0,
            atol=0,
        ), f"Gradient should be scaled by loss scale, expected {expected_grad_per_element}, got {dummy_input.grad[0].item()}"


@pytest.mark.parametrize("seqlen", [16, 64])
class TestDSAIndexer:
    """Test DSA Indexer module basic functionality with TP=1."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        # Create MLA config with sparse attention parameters
        self.index_topk = 32
        self.config = MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            # Sparse attention specific configs
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=self.index_topk,
        )

        # Create indexer submodules spec
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )

        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'cp']
        )
        self.indexer = DSAIndexer(self.config, indexer_submodules, self.pg_collection)

        yield
        Utils.destroy_model_parallel()

    def test_dsa_indexer_constructor(self, seqlen):
        """Test indexer initialization."""
        assert isinstance(self.indexer, DSAIndexer)
        assert self.indexer.hidden_size == 256
        assert self.indexer.index_n_heads == 8
        assert self.indexer.index_head_dim == 64
        assert self.indexer.index_topk == 32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_forward(self, seqlen):
        """Test indexer forward pass."""
        batch_size = 2

        self.indexer.cuda()

        # Create input tensors
        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Forward pass
        topk_indices = self.indexer(x, qr)

        # Check output shape
        assert topk_indices.shape == (batch_size, seqlen, min(self.config.dsa_indexer_topk, seqlen))
        assert topk_indices.dtype == torch.long
        assert torch.all((topk_indices >= 0) & (topk_indices < seqlen))
        # Make sure no duplicate indices are selected
        assert torch.all(
            torch.sort(topk_indices, dim=-1).values[:, :, 1:]
            != torch.sort(topk_indices, dim=-1).values[:, :, :-1]
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_forward_with_scores(self, seqlen):
        """Test indexer forward pass with scores."""
        batch_size = 2

        self.indexer.cuda()

        # Create input tensors
        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Forward pass with scores
        index_scores, topk_indices = self.indexer.forward_with_scores(x, qr)

        # Check output shapes
        assert index_scores.shape == (batch_size, seqlen, seqlen)
        assert topk_indices.shape == (batch_size, seqlen, min(self.config.dsa_indexer_topk, seqlen))
        assert index_scores.dtype == torch.float32
        assert topk_indices.dtype == torch.long
        assert torch.all((topk_indices >= 0) & (topk_indices < seqlen))
        # Make sure no duplicate indices are selected
        assert torch.all(
            torch.sort(topk_indices, dim=-1).values[:, :, 1:]
            != torch.sort(topk_indices, dim=-1).values[:, :, :-1]
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_with_mask(self, seqlen):
        """Test indexer with attention mask."""
        batch_size = 2

        self.indexer.cuda()

        # Create input tensors
        x = torch.randn(seqlen, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seqlen, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()
        mask = torch.triu(
            torch.full((batch_size, seqlen, seqlen), float('-inf'), dtype=torch.float32).cuda(),
            diagonal=1,
        )

        # Forward pass with mask
        index_scores, topk_indices = self.indexer.forward_with_scores(x, qr, mask=mask)

        # Check that masked positions are not selected
        # For causal mask, topk_indices[b, i, :] should all be <= i (except for the case that
        # i < index_topk).
        for b in range(batch_size):
            for i in range(seqlen):
                assert torch.all(topk_indices[b, i] <= max(self.index_topk, i))


class TestDSAttention:
    """Test DSAttention module basic functionality with TP=1."""

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        # Create MLA config with sparse attention parameters
        self.config = MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            # Sparse attention specific configs
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=False,
        )

        # Create sparse attention submodules spec
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )
        indexer_spec = ModuleSpec(module=DSAIndexer, submodules=indexer_submodules)
        sparse_attention_submodules = DSAttentionSubmodules(indexer=indexer_spec)

        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp', 'cp']
        )

        self.sparse_attention = DSAttention(
            config=self.config,
            submodules=sparse_attention_submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=self.pg_collection,
        )

        yield
        Utils.destroy_model_parallel()

    def test_dsa_constructor(self):
        """Test sparse attention initialization."""
        assert isinstance(self.sparse_attention, DSAttention)
        assert hasattr(self.sparse_attention, 'indexer')
        assert isinstance(self.sparse_attention.indexer, DSAIndexer)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_forward(self):
        """Test sparse attention forward pass."""
        seq_len = 16
        batch_size = 2
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads

        self.sparse_attention.cuda()

        # Create input tensors [seq_len, batch, num_heads, head_dim]
        query = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        key = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        value = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        # Original hidden states and low-rank query
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Create causal attention mask
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        # Forward pass
        output = self.sparse_attention(
            query=query,
            key=key,
            value=value,
            x=x,
            qr=qr,
            attention_mask=attention_mask,
            attn_mask_type=AttnMaskType.causal,
        )

        # Check output shape
        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert output.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_backward(self):
        """Test sparse attention backward pass with indexer loss."""
        seq_len = 16
        batch_size = 2
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads

        self.sparse_attention.train()
        self.sparse_attention.cuda()

        # Create input tensors
        query = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        key = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        value = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )

        # Original hidden states and low-rank query
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Create causal attention mask
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        # Forward pass
        output = self.sparse_attention(
            query=query,
            key=key,
            value=value,
            x=x,
            qr=qr,
            attention_mask=attention_mask,
            attn_mask_type=AttnMaskType.causal,
        )

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for inputs
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None

        # Check that indexer parameters have gradients
        for name, param in self.sparse_attention.indexer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Indexer parameter {name} has no gradient"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_topk_selection(self):
        """Test that sparse attention correctly selects top-k indices."""
        seq_len = 16
        batch_size = 2
        num_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_heads

        self.sparse_attention.eval()
        self.sparse_attention.cuda()

        # Create input tensors
        query = torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        key = torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        value = torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()

        # Original hidden states and low-rank query
        x = torch.randn(seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16).cuda()
        qr = torch.randn(seq_len, batch_size, self.config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Create causal attention mask
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        with torch.no_grad():
            # Get topk indices from indexer
            _, topk_indices = self.sparse_attention.indexer.forward_with_scores(x, qr)

            # Forward pass
            output = self.sparse_attention(
                query=query,
                key=key,
                value=value,
                x=x,
                qr=qr,
                attention_mask=attention_mask,
                attn_mask_type=AttnMaskType.causal,
            )

        # Check that topk_indices are valid
        assert torch.all(topk_indices >= 0)
        assert torch.all(topk_indices < seq_len)
        assert topk_indices.shape[2] == min(self.config.dsa_indexer_topk, seq_len)


# ======================================================================================
# Tensor Parallel Consistency Tests
# ======================================================================================


@pytest.mark.parametrize("tensor_model_parallel_size", [2, 4, 8])
@pytest.mark.parametrize("sequence_parallel", [False, True])
class TestIndexerTensorParallel:
    """Test DSA Indexer with different TP sizes and SP settings, compare with TP=1 baseline."""

    def _create_config(self, sequence_parallel=False):
        """Helper to create MLA config."""
        # Get TP size from parallel_state
        tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

        return MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=tensor_model_parallel_size,
            sequence_parallel=sequence_parallel,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            # Sparse attention specific configs
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
        )

    def _create_indexer(self, config, pg_collection):
        """Helper to create indexer."""
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )

        return DSAIndexer(config, indexer_submodules, pg_collection)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_weight_consistency(self, tensor_model_parallel_size, sequence_parallel):
        """Test that indexer weights are identical across ALL GPUs."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config = self._create_config(sequence_parallel=sequence_parallel)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        indexer = self._create_indexer(config, pg_collection).cuda()

        # Check that all weights are identical across ALL ranks (not just TP group)
        world_size = torch.distributed.get_world_size()
        world_rank = torch.distributed.get_rank()

        if world_size > 1:
            for name, param in indexer.named_parameters():
                # Gather weights from ALL ranks in WORLD group
                param_list = [torch.zeros_like(param.data) for _ in range(world_size)]
                torch.distributed.all_gather(param_list, param.data)

                # All weights should be identical across all GPUs
                for i in range(1, world_size):
                    assert torch.allclose(
                        param_list[0], param_list[i], rtol=0, atol=0
                    ), f"Parameter {name} differs between rank 0 and rank {i} (world)"

        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_forward_consistency(self, tensor_model_parallel_size, sequence_parallel):
        """Test that indexer gives consistent results across different TP sizes and SP settings."""
        # First run with TP=1 to get baseline
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config_tp1 = self._create_config(sequence_parallel=False)  # TP=1 doesn't use SP
        pg_collection_tp1 = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        indexer_tp1 = self._create_indexer(config_tp1, pg_collection_tp1).cuda()

        seq_len = 64
        batch_size = 2

        # Create one common input (all ranks create same input with same seed)
        x_input = torch.randn(
            seq_len, batch_size, config_tp1.hidden_size, dtype=torch.bfloat16
        ).cuda()
        qr_input = torch.randn(
            seq_len, batch_size, config_tp1.q_lora_rank, dtype=torch.bfloat16
        ).cuda()

        # Forward pass with gradients enabled
        index_scores_tp1, topk_indices_tp1 = indexer_tp1.forward_with_scores(x_input, qr_input)

        # Backward pass
        loss_tp1 = index_scores_tp1.sum()
        loss_tp1.backward()

        # Save gradients from TP=1
        indexer_tp1_grads = {
            name: param.grad.clone().cpu()
            for name, param in indexer_tp1.named_parameters()
            if param.grad is not None
        }

        Utils.destroy_model_parallel()

        # Now run with target TP size
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config_tpn = self._create_config(sequence_parallel=sequence_parallel)
        pg_collection_tpn = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        indexer_tpn = self._create_indexer(config_tpn, pg_collection_tpn).cuda()

        # Prepare input: split along seqlen if SP is enabled
        if sequence_parallel:
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            seq_per_rank = seq_len // tensor_model_parallel_size
            start_idx = tp_rank * seq_per_rank
            end_idx = (tp_rank + 1) * seq_per_rank
            x_tpn = x_input[start_idx:end_idx]
            qr_tpn = qr_input[start_idx:end_idx]
        else:
            # No SP: all TP ranks see full input
            x_tpn = x_input
            qr_tpn = qr_input

        # Forward pass with gradients enabled
        index_scores_tpn, topk_indices_tpn = indexer_tpn.forward_with_scores(x_tpn, qr_tpn)

        # Backward pass
        loss_tpn = index_scores_tpn.sum()
        loss_tpn.backward()

        # Compare forward outputs
        assert index_scores_tpn.shape == index_scores_tp1.shape
        assert topk_indices_tpn.shape == topk_indices_tp1.shape

        # Check that index scores are close (allow for floating point accumulation errors)
        assert torch.allclose(
            index_scores_tpn, index_scores_tp1, rtol=0, atol=0
        ), f"Index scores mismatch between TP=1 and TP={tensor_model_parallel_size}, SP={sequence_parallel}"

        # Check that topk indices are exactly the same
        assert torch.equal(
            topk_indices_tpn, topk_indices_tp1
        ), f"Top-k indices mismatch between TP=1 and TP={tensor_model_parallel_size}, SP={sequence_parallel}"

        # Compare gradients - indexer grads should be identical (duplicated weights)
        for name, param in indexer_tpn.named_parameters():
            if param.grad is not None and name in indexer_tp1_grads:
                assert torch.allclose(
                    param.grad.cpu(), indexer_tp1_grads[name], rtol=0, atol=0
                ), f"Indexer gradient {name} mismatch between TP=1 and TP={tensor_model_parallel_size}, SP={sequence_parallel}"

        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_gradient_sync(self, tensor_model_parallel_size, sequence_parallel):
        """Test that gradients are properly synchronized within TP group."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config = self._create_config(sequence_parallel=sequence_parallel)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        indexer = self._create_indexer(config, pg_collection).cuda()

        seq_len = 64
        batch_size = 2

        # Create one common input (all ranks create same input with same seed)
        x_input = torch.randn(seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16).cuda()
        qr_input = torch.randn(seq_len, batch_size, config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Prepare input: split along seqlen if SP is enabled
        if sequence_parallel:
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            seq_per_rank = seq_len // tp_size
            start_idx = tp_rank * seq_per_rank
            end_idx = (tp_rank + 1) * seq_per_rank
            x = x_input[start_idx:end_idx]
            qr = qr_input[start_idx:end_idx]
        else:
            # No SP: all TP ranks see full input
            x = x_input
            qr = qr_input

        # Forward and backward
        index_scores, topk_indices = indexer.forward_with_scores(x, qr)
        loss = index_scores.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in indexer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

        # After TP sync, check that gradients are identical within TP group
        # Note: We only check TP group because DDP sync happens separately
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            for name, param in indexer.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Gather gradients from all ranks in TP group only
                    grad_list = [torch.zeros_like(param.grad) for _ in range(tp_size)]
                    torch.distributed.all_gather(grad_list, param.grad, group=pg_collection.tp)

                    # All gradients should be identical within TP group after sync
                    for i in range(1, tp_size):
                        assert torch.allclose(
                            grad_list[0], grad_list[i], rtol=0, atol=0
                        ), f"Gradient for {name} differs between TP rank 0 and rank {i} after TP sync"

        Utils.destroy_model_parallel()


@pytest.mark.parametrize("tensor_model_parallel_size", [2, 4])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("use_sparse_indexer_loss", [False, True])
class TestDSAttentionTensorParallel:
    """Test DSAttention with different TP sizes, SP settings, and sparse indexer loss."""

    def _create_config(self, sequence_parallel=False, use_sparse_indexer_loss=False):
        """Helper to create MLA config."""
        # Get TP size from parallel_state
        tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

        return MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            tensor_model_parallel_size=tensor_model_parallel_size,
            sequence_parallel=sequence_parallel,
            # MLA specific configs
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            # Sparse attention specific configs
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            dsa_indexer_loss_coeff=1.0,
            dsa_indexer_use_sparse_loss=use_sparse_indexer_loss,
        )

    def _create_sparse_attention(self, config, pg_collection):
        """Helper to create sparse attention."""
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.transformer.spec_utils import ModuleSpec

        indexer_submodules = DSAIndexerSubmodules(
            linear_wq_b=ModuleSpec(module=TELinear),
            linear_wk=ModuleSpec(module=TELinear),
            k_norm=ModuleSpec(module=TENorm),
            linear_weights_proj=ModuleSpec(module=TELinear),
        )
        indexer_spec = ModuleSpec(module=DSAIndexer, submodules=indexer_submodules)
        sparse_attention_submodules = DSAttentionSubmodules(indexer=indexer_spec)

        return DSAttention(
            config=config,
            submodules=sparse_attention_submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=pg_collection,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_weight_consistency(
        self, tensor_model_parallel_size, sequence_parallel, use_sparse_indexer_loss
    ):
        """Test that sparse attention indexer weights are identical across ALL GPUs."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config = self._create_config(
            sequence_parallel=sequence_parallel, use_sparse_indexer_loss=use_sparse_indexer_loss
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        sparse_attention = self._create_sparse_attention(config, pg_collection).cuda()

        # Check that all indexer weights are identical across ALL ranks
        world_size = torch.distributed.get_world_size()
        world_rank = torch.distributed.get_rank()

        if world_size > 1:
            for name, param in sparse_attention.indexer.named_parameters():
                # Gather weights from ALL ranks in WORLD group
                param_list = [torch.zeros_like(param.data) for _ in range(world_size)]
                torch.distributed.all_gather(param_list, param.data)

                # All weights should be identical across all GPUs
                for i in range(1, world_size):
                    torch.testing.assert_close(param_list[0], param_list[i], rtol=0, atol=0)

        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_forward_consistency(
        self, tensor_model_parallel_size, sequence_parallel, use_sparse_indexer_loss
    ):
        """Test that sparse attention gives consistent results across different TP, SP, and sparse loss settings."""
        # First run with TP=1 to get baseline
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config_tp1 = self._create_config(
            sequence_parallel=False, use_sparse_indexer_loss=use_sparse_indexer_loss
        )  # TP=1 doesn't use SP
        pg_collection_tp1 = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        sparse_attention_tp1 = self._create_sparse_attention(config_tp1, pg_collection_tp1).cuda()

        seq_len = 64
        batch_size = 2
        num_heads = config_tp1.num_attention_heads
        head_dim = config_tp1.hidden_size // num_heads

        # Create one common input (all ranks create same input with same seed)
        query_input = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        key_input = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        value_input = (
            torch.randn(seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16)
            .cuda()
            .requires_grad_(True)
        )
        x_input = torch.randn(
            seq_len, batch_size, config_tp1.hidden_size, dtype=torch.bfloat16
        ).cuda()
        qr_input = torch.randn(
            seq_len, batch_size, config_tp1.q_lora_rank, dtype=torch.bfloat16
        ).cuda()
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        # Forward pass with gradients enabled
        sparse_attention_tp1.train()
        output_tp1 = sparse_attention_tp1(
            query=query_input,
            key=key_input,
            value=value_input,
            x=x_input,
            qr=qr_input,
            attention_mask=attention_mask,
            attn_mask_type=AttnMaskType.causal,
        )

        # Backward pass
        loss_tp1 = output_tp1.sum()
        loss_tp1.backward()

        # Save gradients from TP=1
        indexer_tp1_grads = {
            name: param.grad.clone()
            for name, param in sparse_attention_tp1.indexer.named_parameters()
            if param.grad is not None
        }
        query_tp1_grad = query_input.grad.clone().cpu()
        key_tp1_grad = key_input.grad.clone().cpu()
        value_tp1_grad = value_input.grad.clone().cpu()

        Utils.destroy_model_parallel()

        # Now run with target TP size
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config_tpn = self._create_config(
            sequence_parallel=sequence_parallel, use_sparse_indexer_loss=use_sparse_indexer_loss
        )
        pg_collection_tpn = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        sparse_attention_tpn = self._create_sparse_attention(config_tpn, pg_collection_tpn).cuda()

        # Create one common input (all ranks create same input with same seed)
        query_input = torch.randn(
            seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        key_input = torch.randn(
            seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        value_input = torch.randn(
            seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        x_input = torch.randn(
            seq_len, batch_size, config_tp1.hidden_size, dtype=torch.bfloat16
        ).cuda()
        qr_input = torch.randn(
            seq_len, batch_size, config_tp1.q_lora_rank, dtype=torch.bfloat16
        ).cuda()
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        # Prepare input: split along seqlen if SP is enabled
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        if sequence_parallel:
            seq_per_rank = seq_len // tensor_model_parallel_size
            start_idx = tp_rank * seq_per_rank
            end_idx = (tp_rank + 1) * seq_per_rank
            x_tpn = x_input[start_idx:end_idx]
            qr_tpn = qr_input[start_idx:end_idx]
        else:
            x_tpn = x_input
            qr_tpn = qr_input

        query_input = query_input.detach()
        key_input = key_input.detach()
        value_input = value_input.detach()
        head_per_rank = num_heads // tensor_model_parallel_size
        start_head = tp_rank * head_per_rank
        end_head = (tp_rank + 1) * head_per_rank
        query_tpn = query_input[:, :, start_head:end_head, :].clone().requires_grad_(True)
        key_tpn = key_input[:, :, start_head:end_head, :].clone().requires_grad_(True)
        value_tpn = value_input[:, :, start_head:end_head, :].clone().requires_grad_(True)
        attention_mask_tpn = attention_mask

        # Forward pass with gradients enabled
        sparse_attention_tpn.train()
        output_tpn = sparse_attention_tpn(
            query=query_tpn,
            key=key_tpn,
            value=value_tpn,
            x=x_tpn,
            qr=qr_tpn,
            attention_mask=attention_mask_tpn,
            attn_mask_type=AttnMaskType.causal,
        )

        # Backward pass
        loss_tpn = output_tpn.sum()
        loss_tpn.backward()

        from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

        output_tpn_gathered = gather_from_tensor_model_parallel_region(
            output_tpn, group=pg_collection_tpn.tp
        )
        assert output_tpn_gathered.shape == output_tp1.shape
        assert torch.allclose(
            output_tpn_gathered.detach(), output_tp1.detach(), rtol=0, atol=0
        ), f"Sparse attention outputs mismatch between TP=1 and TP={tensor_model_parallel_size}, SP={sequence_parallel}, sparse_loss={use_sparse_indexer_loss}"

        # 1. Check indexer gradients.
        for name, param in sparse_attention_tpn.indexer.named_parameters():
            if param.grad is not None and name in indexer_tp1_grads:
                torch.testing.assert_close(
                    param.grad, indexer_tp1_grads[name], rtol=1e-5, atol=1e-5
                )

        # 2. Query/Key/Value gradients need to be gathered along num_heads dim (dim 2) if SP is enabled
        # Flatten last two dims: [seq_len, batch, num_heads, head_dim] -> [seq_len, batch, num_heads * head_dim]
        sq, b, nh, hd = query_tpn.grad.shape
        query_grad_flat = query_tpn.grad.reshape(sq, b, nh * hd)
        key_grad_flat = key_tpn.grad.reshape(sq, b, nh * hd)
        value_grad_flat = value_tpn.grad.reshape(sq, b, nh * hd)

        # Gather along last dim
        query_grad_gathered_flat = gather_from_tensor_model_parallel_region(
            query_grad_flat, group=pg_collection_tpn.tp
        )
        key_grad_gathered_flat = gather_from_tensor_model_parallel_region(
            key_grad_flat, group=pg_collection_tpn.tp
        )
        value_grad_gathered_flat = gather_from_tensor_model_parallel_region(
            value_grad_flat, group=pg_collection_tpn.tp
        )

        # Reshape back: [seq_len, batch, num_heads * head_dim] -> [seq_len, batch, num_heads, head_dim]
        query_tpn_grad_gathered = query_grad_gathered_flat.reshape(sq, b, num_heads, hd)
        key_tpn_grad_gathered = key_grad_gathered_flat.reshape(sq, b, num_heads, hd)
        value_tpn_grad_gathered = value_grad_gathered_flat.reshape(sq, b, num_heads, hd)

        assert torch.allclose(
            query_tpn_grad_gathered.cpu(), query_tp1_grad, rtol=0, atol=0
        ), f"Query gradient mismatch between TP=1 and TP={tensor_model_parallel_size}"
        assert torch.allclose(
            key_tpn_grad_gathered.cpu(), key_tp1_grad, rtol=0, atol=0
        ), f"Key gradient mismatch between TP=1 and TP={tensor_model_parallel_size}"
        assert torch.allclose(
            value_tpn_grad_gathered.cpu(), value_tp1_grad, rtol=0, atol=0
        ), f"Value gradient mismatch between TP=1 and TP={tensor_model_parallel_size}"

        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_gradient_sync(
        self, tensor_model_parallel_size, sequence_parallel, use_sparse_indexer_loss
    ):
        """Test that indexer gradients are properly synchronized within TP group."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config = self._create_config(
            sequence_parallel=sequence_parallel, use_sparse_indexer_loss=use_sparse_indexer_loss
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        sparse_attention = self._create_sparse_attention(config, pg_collection).cuda()
        sparse_attention.train()

        seq_len = 64
        batch_size = 2
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        # Create one common input (all ranks create same input with same seed)
        query_input = torch.randn(
            seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        key_input = torch.randn(
            seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        value_input = torch.randn(
            seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        x_input = torch.randn(seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16).cuda()
        qr_input = torch.randn(seq_len, batch_size, config.q_lora_rank, dtype=torch.bfloat16).cuda()

        # Prepare input: split along seqlen if SP is enabled
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        if sequence_parallel:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            seq_per_rank = seq_len // tp_size
            start_idx = tp_rank * seq_per_rank
            end_idx = (tp_rank + 1) * seq_per_rank
            x = x_input[start_idx:end_idx]
            qr = qr_input[start_idx:end_idx]
        else:
            x = x_input
            qr = qr_input

        # query, key, value should be split along num_heads dim
        head_per_rank = num_heads // tensor_model_parallel_size
        start_head = tp_rank * head_per_rank
        end_head = (tp_rank + 1) * head_per_rank
        query = query_input[:, :, start_head:end_head, :]
        key = key_input[:, :, start_head:end_head, :]
        value = value_input[:, :, start_head:end_head, :]

        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool).cuda()
        attention_mask = torch.tril(attention_mask)

        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

        # Forward and backward
        output = sparse_attention(
            query=query,
            key=key,
            value=value,
            x=x,
            qr=qr,
            attention_mask=attention_mask,
            attn_mask_type=AttnMaskType.causal,
        )

        loss = output.sum()
        loss.backward()

        # Check that gradients exist before sync
        assert query.grad is not None
        assert key.grad is not None
        assert value.grad is not None

        # Check that indexer parameters have gradients
        for name, param in sparse_attention.indexer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Indexer parameter {name} has no gradient"

        # Check that indexer gradients are identical within TP group
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            for name, param in sparse_attention.indexer.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Gather gradients from all ranks in TP group only
                    grad_list = [torch.zeros_like(param.grad) for _ in range(tp_size)]
                    torch.distributed.all_gather(grad_list, param.grad, group=pg_collection.tp)

                    # All gradients should be identical within TP group after sync
                    for i in range(1, tp_size):
                        assert torch.allclose(
                            grad_list[0], grad_list[i], rtol=0, atol=0
                        ), f"Indexer gradient for {name} differs between TP rank 0 and rank {i} after TP sync"

        Utils.destroy_model_parallel()
