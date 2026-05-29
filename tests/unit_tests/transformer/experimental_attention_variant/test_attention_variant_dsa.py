# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
    get_experimental_attention_variant_module_spec,
)
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
    FusedDSAIndexerLoss,
    _compute_index_scores,
    compute_dsa_indexer_loss,
    fused_qk_topk_naive,
    fused_qk_topk_naive_thd,
    rotate_activation,
)
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
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

    @pytest.fixture(scope='class', autouse=True)
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

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        request.cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp']
        )
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_loss_per_token_scale(self, seqlen_and_topk):
        batch_size = 2
        seqlen = seqlen_and_topk[0]
        num_heads = 4
        head_dim = 128
        index_topk = seqlen_and_topk[1]

        index_scores = torch.randn(batch_size, seqlen, seqlen, dtype=torch.float32).cuda()
        causal_mask = torch.triu(
            torch.full(
                (seqlen, seqlen), float('-inf'), dtype=torch.float32, device=index_scores.device
            ),
            diagonal=1,
        )
        masked_index_scores = index_scores + causal_mask
        topk_k = min(index_topk, seqlen)
        topk_indices = masked_index_scores.topk(topk_k, dim=-1)[1]

        query = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        key = torch.randn(seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16).cuda()
        softmax_scale = head_dim**-0.5

        for sparse_loss in [False, True]:
            loss_mean = compute_dsa_indexer_loss(
                index_scores=index_scores.clone(),
                topk_indices=topk_indices,
                query=query,
                key=key,
                softmax_scale=softmax_scale,
                loss_coeff=1.0,
                sparse_loss=sparse_loss,
                pg_collection=self.pg_collection,
            )
            loss_sum = compute_dsa_indexer_loss(
                index_scores=index_scores.clone(),
                topk_indices=topk_indices,
                query=query,
                key=key,
                softmax_scale=softmax_scale,
                loss_coeff=1.0,
                sparse_loss=sparse_loss,
                pg_collection=self.pg_collection,
                calculate_per_token_loss=True,
            )

            assert torch.allclose(loss_sum, loss_mean * (batch_size * seqlen), rtol=1e-3, atol=1e-3)


class TestDSAIndexerLossAutoScaler:
    """Test DSAIndexerLossAutoScaler autograd function."""

    @pytest.fixture(scope='class', autouse=True)
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

        DSAIndexerLossAutoScaler.main_loss_backward_scale = None


class TestFusedDSAIndexerLossGradient:
    """Test that FusedDSAIndexerLoss manual backward matches autograd backward."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        request.cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp']
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("calculate_per_token_loss", [False, True])
    def test_fused_indexer_loss_gradient_matches_autograd(self, calculate_per_token_loss):
        """
        Test that the manually written backward in FusedDSAIndexerLoss produces
        the same gradients as PyTorch autograd on the unfused implementation.
        """
        batch_size = 2
        num_heads = 4
        head_dim = 64
        index_n_heads = 8
        index_head_dim = 64
        softmax_scale = head_dim**-0.5
        loss_coeff = 1.0

        for seqlen, index_topk in [[16, 8], [32, 16], [64, 32]]:
            for sparse_loss in [False, True]:
                tag = (
                    f"[seqlen={seqlen}, topk={index_topk}, sparse={sparse_loss}, "
                    f"per_token={calculate_per_token_loss}]"
                )
                torch.manual_seed(42)

                q_ref = (
                    torch.randn(
                        seqlen, batch_size, index_n_heads, index_head_dim, dtype=torch.float32
                    )
                    .cuda()
                    .requires_grad_(True)
                )
                weights_ref = (
                    torch.randn(seqlen, batch_size, index_n_heads, dtype=torch.float32)
                    .cuda()
                    .requires_grad_(True)
                )
                k_ref = (
                    torch.randn(seqlen, batch_size, index_head_dim, dtype=torch.float32)
                    .cuda()
                    .requires_grad_(True)
                )
                query = torch.randn(
                    seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                ).cuda()
                key = torch.randn(
                    seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                ).cuda()
                mask = torch.triu(
                    torch.full((seqlen, seqlen), float('-inf'), dtype=torch.float32).cuda(),
                    diagonal=1,
                )

                # Method 1: Autograd (reference)
                index_scores_ref = _compute_index_scores(q_ref, weights_ref, k_ref)
                index_scores_masked = index_scores_ref + mask.unsqueeze(0)
                topk_k = min(index_topk, seqlen)
                topk_indices = index_scores_masked.topk(topk_k, dim=-1)[1]

                loss_ref = compute_dsa_indexer_loss(
                    index_scores=index_scores_masked,
                    topk_indices=topk_indices,
                    query=query,
                    key=key,
                    softmax_scale=softmax_scale,
                    loss_coeff=loss_coeff,
                    sparse_loss=sparse_loss,
                    pg_collection=self.pg_collection,
                    calculate_per_token_loss=calculate_per_token_loss,
                )
                loss_ref.backward()

                grad_q_ref = q_ref.grad.clone()
                grad_weights_ref = weights_ref.grad.clone()
                grad_k_ref = k_ref.grad.clone()

                # Method 2: FusedDSAIndexerLoss (manual backward)
                q_fused = q_ref.detach().clone().requires_grad_(True)
                weights_fused = weights_ref.detach().clone().requires_grad_(True)
                k_fused = k_ref.detach().clone().requires_grad_(True)

                topk_indices_fused, loss_fused = FusedDSAIndexerLoss.apply(
                    q_fused,
                    weights_fused,
                    k_fused,
                    query.detach(),
                    key.detach(),
                    softmax_scale,
                    index_topk,
                    loss_coeff,
                    mask,
                    sparse_loss,
                    self.pg_collection,
                    calculate_per_token_loss,
                )
                loss_fused.backward()

                grad_q_fused = q_fused.grad
                grad_weights_fused = weights_fused.grad
                grad_k_fused = k_fused.grad

                # Compare
                assert torch.allclose(
                    loss_fused, loss_ref, rtol=1e-5, atol=1e-5
                ), f"{tag} Loss mismatch: fused={loss_fused.item()}, ref={loss_ref.item()}"

                assert torch.equal(
                    topk_indices_fused, topk_indices
                ), f"{tag} Top-k indices mismatch between fused and reference"

                assert torch.allclose(
                    grad_q_fused, grad_q_ref, rtol=1e-5, atol=1e-5
                ), f"{tag} grad_q mismatch: max diff = {(grad_q_fused - grad_q_ref).abs().max().item()}"

                assert torch.allclose(
                    grad_weights_fused, grad_weights_ref, rtol=1e-5, atol=1e-5
                ), f"{tag} grad_weights mismatch: max diff = {(grad_weights_fused - grad_weights_ref).abs().max().item()}"

                assert torch.allclose(
                    grad_k_fused, grad_k_ref, rtol=1e-5, atol=1e-5
                ), f"{tag} grad_k mismatch: max diff = {(grad_k_fused - grad_k_ref).abs().max().item()}"


class TestFusedDSAIndexerLossGradientTP:
    """Test FusedDSAIndexerLoss gradient consistency across different TP sizes."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fused_indexer_loss_gradient_tp_consistency(self):
        """
        Test that FusedDSAIndexerLoss produces consistent gradients across TP ranks
        and matches TP=1 baseline.

        Tests all combinations of sparse_loss=[False, True] and TP=[2, 4] in a
        single test to minimise process-group init/destroy overhead.
        """
        seqlen = 64
        index_topk = 32
        batch_size = 2
        num_heads = 8
        head_dim = 64
        index_n_heads = 8
        index_head_dim = 64
        softmax_scale = head_dim**-0.5
        loss_coeff = 1.0

        # =============================================
        # Compute TP=1 baselines for both sparse_loss values
        # =============================================
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        pg_collection_tp1 = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])

        # Create inputs (shared across all variants)
        q_input = torch.randn(
            seqlen, batch_size, index_n_heads, index_head_dim, dtype=torch.float32
        ).cuda()
        weights_input = torch.randn(seqlen, batch_size, index_n_heads, dtype=torch.float32).cuda()
        k_input = torch.randn(seqlen, batch_size, index_head_dim, dtype=torch.float32).cuda()
        query_input = torch.randn(
            seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        key_input = torch.randn(
            seqlen, batch_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        mask = torch.triu(
            torch.full((seqlen, seqlen), float('-inf'), dtype=torch.float32).cuda(), diagonal=1
        )

        # {sparse_loss: (topk_indices, loss, grad_q, grad_weights, grad_k)}
        baselines = {}
        for sparse_loss in [False, True]:
            q_tp1 = q_input.clone().requires_grad_(True)
            weights_tp1 = weights_input.clone().requires_grad_(True)
            k_tp1 = k_input.clone().requires_grad_(True)

            topk_indices_tp1, loss_tp1 = FusedDSAIndexerLoss.apply(
                q_tp1,
                weights_tp1,
                k_tp1,
                query_input.detach(),
                key_input.detach(),
                softmax_scale,
                index_topk,
                loss_coeff,
                mask,
                sparse_loss,
                pg_collection_tp1,
                False,
            )
            loss_tp1.backward()

            baselines[sparse_loss] = (
                topk_indices_tp1.clone(),
                loss_tp1.detach().clone(),
                q_tp1.grad.clone(),
                weights_tp1.grad.clone(),
                k_tp1.grad.clone(),
            )

        Utils.destroy_model_parallel()

        # =============================================
        # Test each TP size against baselines
        # =============================================
        for tensor_model_parallel_size in [2, 4]:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )
            torch.manual_seed(42)
            model_parallel_cuda_manual_seed(42)

            pg_collection_tpn = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp'])
            tp_rank = parallel_state.get_tensor_model_parallel_rank()

            # query and key split along heads for TP
            head_per_rank = num_heads // tensor_model_parallel_size
            start_head = tp_rank * head_per_rank
            end_head = (tp_rank + 1) * head_per_rank
            query_tpn = query_input[:, :, start_head:end_head, :].clone()
            key_tpn = key_input[:, :, start_head:end_head, :].clone()

            for sparse_loss in [False, True]:
                topk_indices_tp1, loss_tp1_value, grad_q_tp1, grad_weights_tp1, grad_k_tp1 = (
                    baselines[sparse_loss]
                )
                tag = f"[TP={tensor_model_parallel_size}, sparse_loss={sparse_loss}]"

                q_tpn = q_input.clone().requires_grad_(True)
                weights_tpn = weights_input.clone().requires_grad_(True)
                k_tpn = k_input.clone().requires_grad_(True)

                topk_indices_tpn, loss_tpn = FusedDSAIndexerLoss.apply(
                    q_tpn,
                    weights_tpn,
                    k_tpn,
                    query_tpn.detach(),
                    key_tpn.detach(),
                    softmax_scale,
                    index_topk,
                    loss_coeff,
                    mask,
                    sparse_loss,
                    pg_collection_tpn,
                    False,
                )
                loss_tpn.backward()

                # Loss should be the same
                assert torch.allclose(
                    loss_tpn, loss_tp1_value, rtol=1e-5, atol=1e-5
                ), f"{tag} Loss mismatch: got {loss_tpn.item()}, TP=1 got {loss_tp1_value.item()}"

                # Top-k indices should be the same
                assert torch.equal(
                    topk_indices_tpn, topk_indices_tp1
                ), f"{tag} Top-k indices mismatch between TP=1 and TP=N"

                # Gradients should match (indexer params are duplicated across TP)
                assert torch.allclose(
                    q_tpn.grad, grad_q_tp1, rtol=1e-5, atol=1e-5
                ), f"{tag} grad_q mismatch: max diff = {(q_tpn.grad - grad_q_tp1).abs().max().item()}"

                assert torch.allclose(
                    weights_tpn.grad, grad_weights_tp1, rtol=1e-5, atol=1e-5
                ), f"{tag} grad_weights mismatch: max diff = {(weights_tpn.grad - grad_weights_tp1).abs().max().item()}"

                assert torch.allclose(
                    k_tpn.grad, grad_k_tp1, rtol=1e-5, atol=1e-5
                ), f"{tag} grad_k mismatch: max diff = {(k_tpn.grad - grad_k_tp1).abs().max().item()}"

                # Check gradients are identical across all TP ranks
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                if tp_size > 1:
                    for grad_tensor, name in [
                        (q_tpn.grad, "grad_q"),
                        (weights_tpn.grad, "grad_weights"),
                        (k_tpn.grad, "grad_k"),
                    ]:
                        grad_list = [torch.zeros_like(grad_tensor) for _ in range(tp_size)]
                        torch.distributed.all_gather(
                            grad_list, grad_tensor, group=pg_collection_tpn.tp
                        )

                        for i in range(1, tp_size):
                            assert torch.allclose(
                                grad_list[0], grad_list[i], rtol=0, atol=0
                            ), f"{tag} {name} differs between TP rank 0 and rank {i}"

            Utils.destroy_model_parallel()


@pytest.mark.parametrize("seqlen", [16, 64])
class TestDSAIndexer:
    """Test DSA Indexer module basic functionality with TP=1."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        # Create MLA config with sparse attention parameters
        cls = request.cls
        cls.index_topk = 32
        cls.config = MLATransformerConfig(
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
            dsa_indexer_topk=cls.index_topk,
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

        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        cls.indexer = DSAIndexer(cls.config, indexer_submodules, cls.pg_collection)

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

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        # Create MLA config with sparse attention parameters
        cls.config = MLATransformerConfig(
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

        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        cls.sparse_attention = DSAttention(
            config=cls.config,
            submodules=sparse_attention_submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=cls.pg_collection,
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


class TestIndexerTensorParallel:
    """Test DSA Indexer with different TP sizes and SP settings, compare with TP=1 baseline."""

    TP_SIZES = [2, 4, 8]
    SP_VALUES = [False, True]

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
    def test_dsa_indexer_weight_consistency(self):
        """Test that indexer weights are identical across ALL GPUs."""
        for tensor_model_parallel_size in self.TP_SIZES:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )
            world_size = torch.distributed.get_world_size()

            for sequence_parallel in self.SP_VALUES:
                torch.manual_seed(123)
                model_parallel_cuda_manual_seed(123)

                config = self._create_config(sequence_parallel=sequence_parallel)
                pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                    required_pgs=['tp', 'cp']
                )
                indexer = self._create_indexer(config, pg_collection).cuda()
                tag = f"[TP={tensor_model_parallel_size}, SP={sequence_parallel}]"

                # Check that all weights are identical across ALL ranks
                if world_size > 1:
                    for name, param in indexer.named_parameters():
                        param_list = [torch.zeros_like(param.data) for _ in range(world_size)]
                        torch.distributed.all_gather(param_list, param.data)

                        for i in range(1, world_size):
                            assert torch.allclose(
                                param_list[0], param_list[i], rtol=0, atol=0
                            ), f"{tag} Parameter {name} differs between rank 0 and rank {i} (world)"

            Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_forward_consistency(self):
        """Test that indexer gives consistent results across different TP sizes and SP settings."""
        seq_len = 64
        batch_size = 2

        # TP=1 baseline (once for all TP sizes; TP=1 doesn't use SP)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        config_tp1 = self._create_config(sequence_parallel=False)
        pg_collection_tp1 = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        indexer_tp1 = self._create_indexer(config_tp1, pg_collection_tp1).cuda()

        x_input = torch.randn(
            seq_len, batch_size, config_tp1.hidden_size, dtype=torch.bfloat16
        ).cuda()
        qr_input = torch.randn(
            seq_len, batch_size, config_tp1.q_lora_rank, dtype=torch.bfloat16
        ).cuda()

        index_scores_tp1, topk_indices_tp1 = indexer_tp1.forward_with_scores(x_input, qr_input)
        loss_tp1 = index_scores_tp1.sum()
        loss_tp1.backward()

        indexer_tp1_grads = {
            name: param.grad.clone().cpu()
            for name, param in indexer_tp1.named_parameters()
            if param.grad is not None
        }

        Utils.destroy_model_parallel()

        # Test each TP size with both SP values
        for tensor_model_parallel_size in self.TP_SIZES:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )

            for sequence_parallel in self.SP_VALUES:
                torch.manual_seed(123)
                model_parallel_cuda_manual_seed(123)

                config_tpn = self._create_config(sequence_parallel=sequence_parallel)
                pg_collection_tpn = ProcessGroupCollection.use_mpu_process_groups(
                    required_pgs=['tp', 'cp']
                )
                indexer_tpn = self._create_indexer(config_tpn, pg_collection_tpn).cuda()
                tag = f"[TP={tensor_model_parallel_size}, SP={sequence_parallel}]"

                if sequence_parallel:
                    tp_rank = parallel_state.get_tensor_model_parallel_rank()
                    seq_per_rank = seq_len // tensor_model_parallel_size
                    start_idx = tp_rank * seq_per_rank
                    end_idx = (tp_rank + 1) * seq_per_rank
                    x_tpn = x_input[start_idx:end_idx]
                    qr_tpn = qr_input[start_idx:end_idx]
                else:
                    x_tpn = x_input
                    qr_tpn = qr_input

                index_scores_tpn, topk_indices_tpn = indexer_tpn.forward_with_scores(x_tpn, qr_tpn)
                loss_tpn = index_scores_tpn.sum()
                loss_tpn.backward()

                assert index_scores_tpn.shape == index_scores_tp1.shape
                assert topk_indices_tpn.shape == topk_indices_tp1.shape
                assert torch.allclose(
                    index_scores_tpn, index_scores_tp1, rtol=0, atol=0
                ), f"{tag} Index scores mismatch vs TP=1"
                assert torch.equal(
                    topk_indices_tpn, topk_indices_tp1
                ), f"{tag} Top-k indices mismatch vs TP=1"

                for name, param in indexer_tpn.named_parameters():
                    if param.grad is not None and name in indexer_tp1_grads:
                        assert torch.allclose(
                            param.grad.cpu(), indexer_tp1_grads[name], rtol=0, atol=0
                        ), f"{tag} Indexer gradient {name} mismatch vs TP=1"

            Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_indexer_gradient_sync(self):
        """Test that gradients are properly synchronized within TP group."""
        seq_len = 64
        batch_size = 2

        for tensor_model_parallel_size in self.TP_SIZES:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )

            for sequence_parallel in self.SP_VALUES:
                torch.manual_seed(123)
                model_parallel_cuda_manual_seed(123)

                config = self._create_config(sequence_parallel=sequence_parallel)
                pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                    required_pgs=['tp', 'cp']
                )
                indexer = self._create_indexer(config, pg_collection).cuda()
                tag = f"[TP={tensor_model_parallel_size}, SP={sequence_parallel}]"

                x_input = torch.randn(
                    seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16
                ).cuda()
                qr_input = torch.randn(
                    seq_len, batch_size, config.q_lora_rank, dtype=torch.bfloat16
                ).cuda()

                if sequence_parallel:
                    tp_rank = parallel_state.get_tensor_model_parallel_rank()
                    tp_size = parallel_state.get_tensor_model_parallel_world_size()
                    seq_per_rank = seq_len // tp_size
                    start_idx = tp_rank * seq_per_rank
                    end_idx = (tp_rank + 1) * seq_per_rank
                    x = x_input[start_idx:end_idx]
                    qr = qr_input[start_idx:end_idx]
                else:
                    x = x_input
                    qr = qr_input

                index_scores, topk_indices = indexer.forward_with_scores(x, qr)
                loss = index_scores.sum()
                loss.backward()

                for name, param in indexer.named_parameters():
                    if param.requires_grad:
                        assert param.grad is not None, f"{tag} Parameter {name} has no gradient"

                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                if tp_size > 1:
                    for name, param in indexer.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_list = [torch.zeros_like(param.grad) for _ in range(tp_size)]
                            torch.distributed.all_gather(
                                grad_list, param.grad, group=pg_collection.tp
                            )

                            for i in range(1, tp_size):
                                assert torch.allclose(
                                    grad_list[0], grad_list[i], rtol=0, atol=0
                                ), f"{tag} Gradient for {name} differs between TP rank 0 and rank {i}"

            Utils.destroy_model_parallel()


class TestDSAttentionTensorParallel:
    """Test DSAttention with different TP sizes, SP settings, and sparse indexer loss."""

    TP_SIZES = [2, 4]
    SP_VALUES = [False, True]
    SPARSE_VALUES = [False, True]

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
    def test_dsa_weight_consistency(self):
        """Test that sparse attention indexer weights are identical across ALL GPUs."""
        for tensor_model_parallel_size in self.TP_SIZES:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )
            world_size = torch.distributed.get_world_size()

            for sequence_parallel in self.SP_VALUES:
                for use_sparse_indexer_loss in self.SPARSE_VALUES:
                    torch.manual_seed(123)
                    model_parallel_cuda_manual_seed(123)

                    config = self._create_config(
                        sequence_parallel=sequence_parallel,
                        use_sparse_indexer_loss=use_sparse_indexer_loss,
                    )
                    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                        required_pgs=['tp', 'cp']
                    )
                    sparse_attention = self._create_sparse_attention(config, pg_collection).cuda()
                    tag = f"[TP={tensor_model_parallel_size}, SP={sequence_parallel}, sparse={use_sparse_indexer_loss}]"

                    if world_size > 1:
                        for name, param in sparse_attention.indexer.named_parameters():
                            param_list = [torch.zeros_like(param.data) for _ in range(world_size)]
                            torch.distributed.all_gather(param_list, param.data)

                            for i in range(1, world_size):
                                torch.testing.assert_close(
                                    param_list[0], param_list[i], rtol=0, atol=0
                                )

            Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_forward_consistency(self):
        """Test that sparse attention gives consistent results across different TP, SP, and sparse loss settings."""
        from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

        seq_len = 64
        batch_size = 2

        # TP=1 baselines: one per use_sparse_indexer_loss value (SP is always False for TP=1)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

        baselines = {}  # {sparse_loss: (output, indexer_grads, q_grad, k_grad, v_grad)}
        for use_sparse_indexer_loss in self.SPARSE_VALUES:
            torch.manual_seed(123)
            model_parallel_cuda_manual_seed(123)

            config_tp1 = self._create_config(
                sequence_parallel=False, use_sparse_indexer_loss=use_sparse_indexer_loss
            )
            pg_collection_tp1 = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=['tp', 'cp']
            )
            sparse_attention_tp1 = self._create_sparse_attention(
                config_tp1, pg_collection_tp1
            ).cuda()

            num_heads = config_tp1.num_attention_heads
            head_dim = config_tp1.hidden_size // num_heads

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

            loss_tp1 = output_tp1.sum()
            loss_tp1.backward()

            baselines[use_sparse_indexer_loss] = (
                output_tp1.detach().clone(),
                {
                    name: param.grad.clone()
                    for name, param in sparse_attention_tp1.indexer.named_parameters()
                    if param.grad is not None
                },
                query_input.grad.clone().cpu(),
                key_input.grad.clone().cpu(),
                value_input.grad.clone().cpu(),
                num_heads,
                head_dim,
            )

        Utils.destroy_model_parallel()

        # Test each TP size with all (SP, sparse) combos
        for tensor_model_parallel_size in self.TP_SIZES:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )

            for sequence_parallel in self.SP_VALUES:
                for use_sparse_indexer_loss in self.SPARSE_VALUES:
                    torch.manual_seed(123)
                    model_parallel_cuda_manual_seed(123)

                    (
                        output_tp1,
                        indexer_tp1_grads,
                        query_tp1_grad,
                        key_tp1_grad,
                        value_tp1_grad,
                        num_heads,
                        head_dim,
                    ) = baselines[use_sparse_indexer_loss]

                    config_tpn = self._create_config(
                        sequence_parallel=sequence_parallel,
                        use_sparse_indexer_loss=use_sparse_indexer_loss,
                    )
                    pg_collection_tpn = ProcessGroupCollection.use_mpu_process_groups(
                        required_pgs=['tp', 'cp']
                    )
                    sparse_attention_tpn = self._create_sparse_attention(
                        config_tpn, pg_collection_tpn
                    ).cuda()
                    tag = f"[TP={tensor_model_parallel_size}, SP={sequence_parallel}, sparse={use_sparse_indexer_loss}]"

                    query_input_tpn = torch.randn(
                        seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                    ).cuda()
                    key_input_tpn = torch.randn(
                        seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                    ).cuda()
                    value_input_tpn = torch.randn(
                        seq_len, batch_size, num_heads, head_dim, dtype=torch.bfloat16
                    ).cuda()
                    x_input_tpn = torch.randn(
                        seq_len, batch_size, config_tpn.hidden_size, dtype=torch.bfloat16
                    ).cuda()
                    qr_input_tpn = torch.randn(
                        seq_len, batch_size, config_tpn.q_lora_rank, dtype=torch.bfloat16
                    ).cuda()
                    attention_mask_tpn = torch.ones(
                        batch_size, 1, seq_len, seq_len, dtype=torch.bool
                    ).cuda()
                    attention_mask_tpn = torch.tril(attention_mask_tpn)

                    tp_rank = parallel_state.get_tensor_model_parallel_rank()
                    if sequence_parallel:
                        seq_per_rank = seq_len // tensor_model_parallel_size
                        start_idx = tp_rank * seq_per_rank
                        end_idx = (tp_rank + 1) * seq_per_rank
                        x_tpn = x_input_tpn[start_idx:end_idx]
                        qr_tpn = qr_input_tpn[start_idx:end_idx]
                    else:
                        x_tpn = x_input_tpn
                        qr_tpn = qr_input_tpn

                    head_per_rank = num_heads // tensor_model_parallel_size
                    start_head = tp_rank * head_per_rank
                    end_head = (tp_rank + 1) * head_per_rank
                    query_tpn = (
                        query_input_tpn[:, :, start_head:end_head, :].clone().requires_grad_(True)
                    )
                    key_tpn = (
                        key_input_tpn[:, :, start_head:end_head, :].clone().requires_grad_(True)
                    )
                    value_tpn = (
                        value_input_tpn[:, :, start_head:end_head, :].clone().requires_grad_(True)
                    )

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

                    loss_tpn = output_tpn.sum()
                    loss_tpn.backward()

                    output_tpn_gathered = gather_from_tensor_model_parallel_region(
                        output_tpn, group=pg_collection_tpn.tp
                    )
                    assert output_tpn_gathered.shape == output_tp1.shape
                    assert torch.allclose(
                        output_tpn_gathered.detach(), output_tp1, rtol=0, atol=0
                    ), f"{tag} Sparse attention outputs mismatch vs TP=1"

                    for name, param in sparse_attention_tpn.indexer.named_parameters():
                        if param.grad is not None and name in indexer_tp1_grads:
                            torch.testing.assert_close(
                                param.grad, indexer_tp1_grads[name], rtol=1e-5, atol=1e-5
                            )

                    sq, b, nh, hd = query_tpn.grad.shape
                    query_grad_gathered = gather_from_tensor_model_parallel_region(
                        query_tpn.grad.reshape(sq, b, nh * hd), group=pg_collection_tpn.tp
                    ).reshape(sq, b, num_heads, hd)
                    key_grad_gathered = gather_from_tensor_model_parallel_region(
                        key_tpn.grad.reshape(sq, b, nh * hd), group=pg_collection_tpn.tp
                    ).reshape(sq, b, num_heads, hd)
                    value_grad_gathered = gather_from_tensor_model_parallel_region(
                        value_tpn.grad.reshape(sq, b, nh * hd), group=pg_collection_tpn.tp
                    ).reshape(sq, b, num_heads, hd)

                    assert torch.allclose(
                        query_grad_gathered.cpu(), query_tp1_grad, rtol=0, atol=0
                    ), f"{tag} Query gradient mismatch vs TP=1"
                    assert torch.allclose(
                        key_grad_gathered.cpu(), key_tp1_grad, rtol=0, atol=0
                    ), f"{tag} Key gradient mismatch vs TP=1"
                    assert torch.allclose(
                        value_grad_gathered.cpu(), value_tp1_grad, rtol=0, atol=0
                    ), f"{tag} Value gradient mismatch vs TP=1"

            Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dsa_gradient_sync(self):
        """Test that indexer gradients are properly synchronized within TP group."""
        seq_len = 64
        batch_size = 2

        for tensor_model_parallel_size in self.TP_SIZES:
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=1,
            )

            for sequence_parallel in self.SP_VALUES:
                for use_sparse_indexer_loss in self.SPARSE_VALUES:
                    torch.manual_seed(123)
                    model_parallel_cuda_manual_seed(123)

                    config = self._create_config(
                        sequence_parallel=sequence_parallel,
                        use_sparse_indexer_loss=use_sparse_indexer_loss,
                    )
                    pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                        required_pgs=['tp', 'cp']
                    )
                    sparse_attention = self._create_sparse_attention(config, pg_collection).cuda()
                    sparse_attention.train()
                    tag = f"[TP={tensor_model_parallel_size}, SP={sequence_parallel}, sparse={use_sparse_indexer_loss}]"

                    num_heads = config.num_attention_heads
                    head_dim = config.hidden_size // num_heads

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
                        seq_len, batch_size, config.hidden_size, dtype=torch.bfloat16
                    ).cuda()
                    qr_input = torch.randn(
                        seq_len, batch_size, config.q_lora_rank, dtype=torch.bfloat16
                    ).cuda()

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

                    head_per_rank = num_heads // tensor_model_parallel_size
                    start_head = tp_rank * head_per_rank
                    end_head = (tp_rank + 1) * head_per_rank
                    query = query_input[:, :, start_head:end_head, :]
                    key = key_input[:, :, start_head:end_head, :]
                    value = value_input[:, :, start_head:end_head, :]

                    attention_mask = torch.ones(
                        batch_size, 1, seq_len, seq_len, dtype=torch.bool
                    ).cuda()
                    attention_mask = torch.tril(attention_mask)

                    query.requires_grad_(True)
                    key.requires_grad_(True)
                    value.requires_grad_(True)

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

                    assert query.grad is not None
                    assert key.grad is not None
                    assert value.grad is not None

                    for name, param in sparse_attention.indexer.named_parameters():
                        if param.requires_grad:
                            assert (
                                param.grad is not None
                            ), f"{tag} Indexer parameter {name} has no gradient"

                    tp_size = parallel_state.get_tensor_model_parallel_world_size()
                    if tp_size > 1:
                        for name, param in sparse_attention.indexer.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                grad_list = [torch.zeros_like(param.grad) for _ in range(tp_size)]
                                torch.distributed.all_gather(
                                    grad_list, param.grad, group=pg_collection.tp
                                )

                                for i in range(1, tp_size):
                                    assert torch.allclose(
                                        grad_list[0], grad_list[i], rtol=0, atol=0
                                    ), f"{tag} Gradient for {name} differs between TP rank 0 and rank {i}"

            Utils.destroy_model_parallel()


@pytest.mark.internal
class TestDSAModuleSpecDispatch:
    """Tests for get_dsa_module_spec_for_backend and get_experimental_attention_variant_module_spec."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def _make_dsa_config(self, **kwargs):
        return MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            **kwargs,
        )

    def test_get_experimental_attention_variant_module_spec_dsa(self):
        """get_experimental_attention_variant_module_spec dispatches to DSA for variant='dsa'."""
        config = self._make_dsa_config(experimental_attention_variant="dsa")
        spec = get_experimental_attention_variant_module_spec(config)
        assert spec.module == MLASelfAttention
        assert spec.submodules.core_attention.module == DSAttention

    def test_get_dsa_module_spec_for_backend(self):
        """get_dsa_module_spec_for_backend returns the correct full spec structure."""
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

        config = self._make_dsa_config()
        backend = TESpecProvider()
        spec = get_dsa_module_spec_for_backend(config, backend=backend)
        assert spec.module == MLASelfAttention
        assert spec.submodules.core_attention.module == DSAttention
        assert spec.submodules.core_attention.submodules.indexer.module == DSAIndexer
        assert spec.params["attn_mask_type"] == AttnMaskType.causal

    def test_get_dsa_module_spec_requires_mla(self):
        """get_dsa_module_spec_for_backend rejects configs without MLA."""
        from megatron.core.transformer import TransformerConfig as _TransformerConfig

        config = _TransformerConfig(num_layers=2, hidden_size=256, num_attention_heads=4)
        with pytest.raises(AssertionError, match="only MLA supports sparse attention"):
            get_dsa_module_spec_for_backend(config, backend=None)

    def test_get_dsa_module_spec_rejects_qk_l2_norm(self):
        """get_dsa_module_spec_for_backend rejects configs with qk_l2_norm=True."""
        config = self._make_dsa_config(qk_l2_norm=True)
        with pytest.raises(AssertionError, match="qk_l2_norm is not supported"):
            get_dsa_module_spec_for_backend(config, backend=None)


# ===========================================================================
# THD: FusedDSAIndexerLoss
# ===========================================================================


class TestFusedDSAIndexerLossThd:
    """``FusedDSAIndexerLoss`` THD branch — per-segment loop that delegates
    each segment to the SBHD naive helpers with ``b=1`` and aggregates
    via row-weighted-mean.

    For a single-segment THD batch (``cu_seqlens_q = [0, sq]``) the THD
    invocation must produce numerically equivalent loss + gradients as
    the SBHD invocation with ``b=1`` on the same data — the only
    difference is the (B=1) per-segment slicing/concat glue.
    """

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        request.cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=['tp']
        )
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize('sparse_loss', [False, True], ids=['dense_loss', 'sparse_loss'])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_single_segment_matches_sbhd_b1(self, sparse_loss):
        """B=1 THD invocation should match the equivalent SBHD-b=1 call
        (loss + gradients) for both sparse and dense KL loss variants.
        """
        torch.manual_seed(0)
        sq = 32
        n_compressed = sq // 4  # ratio=4 → compressed K len per segment
        ratio = 4
        num_heads = 4
        head_dim = 64
        idx_nh, idx_hd = 4, 32
        topk = 4
        softmax_scale = head_dim**-0.5
        loss_coeff = 0.5

        # ---- Common inputs (SBHD-shape with b=1) -----------------------
        def _rand(*shape, dtype=torch.float32):
            return torch.randn(*shape, dtype=dtype, device='cuda')

        q_sbhd = _rand(sq, 1, idx_nh, idx_hd).requires_grad_(True)
        w_sbhd = _rand(sq, 1, idx_nh).requires_grad_(True)
        k_sbhd = _rand(n_compressed, 1, idx_hd).requires_grad_(True)
        query_sbhd = _rand(sq, 1, num_heads, head_dim, dtype=torch.bfloat16)
        key_sbhd = _rand(n_compressed, 1, num_heads, head_dim, dtype=torch.bfloat16)

        # SBHD per-batch causal mask: (1, sq, n_compressed).
        cols = torch.arange(n_compressed, device='cuda').unsqueeze(0).expand(sq, -1)
        positions = torch.arange(1, sq + 1, device='cuda').unsqueeze(1)
        mask_sbhd = torch.where(cols >= positions // ratio, float('-inf'), 0.0).unsqueeze(0)

        # ---- SBHD reference --------------------------------------------
        topk_indices_sbhd, loss_sbhd = FusedDSAIndexerLoss.apply(
            q_sbhd,
            w_sbhd,
            k_sbhd,
            query_sbhd,
            key_sbhd,
            softmax_scale,
            topk,
            loss_coeff,
            mask_sbhd,
            sparse_loss,
            self.pg_collection,
            False,  # calculate_per_token_loss
        )
        loss_sbhd.backward()
        grad_q_sbhd = q_sbhd.grad.clone()
        grad_w_sbhd = w_sbhd.grad.clone()
        grad_k_sbhd = k_sbhd.grad.clone()

        # ---- THD equivalent (B=1, total_q=sq) --------------------------
        q_thd = q_sbhd.detach().squeeze(1).clone().requires_grad_(True)
        w_thd = w_sbhd.detach().squeeze(1).clone().requires_grad_(True)
        k_thd = k_sbhd.detach().squeeze(1).clone().requires_grad_(True)
        query_thd = query_sbhd.squeeze(1)
        key_thd = key_sbhd.squeeze(1)

        cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device='cuda')
        cu_seqlens_comp = torch.tensor([0, n_compressed], dtype=torch.int32, device='cuda')

        topk_indices_thd, loss_thd = FusedDSAIndexerLoss.apply(
            q_thd,
            w_thd,
            k_thd,
            query_thd,
            key_thd,
            softmax_scale,
            topk,
            loss_coeff,
            None,  # mask: built per-segment internally for THD
            sparse_loss,
            self.pg_collection,
            False,  # calculate_per_token_loss
            cu_seqlens_q,
            cu_seqlens_comp,
            ratio,
        )
        loss_thd.backward()
        grad_q_thd = q_thd.grad
        grad_w_thd = w_thd.grad
        grad_k_thd = k_thd.grad

        tag = f"[sparse={sparse_loss}]"

        # Loss + grads must match the SBHD-b=1 reference (same math,
        # same data; only the slicing-and-concat glue differs).
        assert torch.allclose(loss_thd, loss_sbhd, rtol=1e-5, atol=1e-5), (
            f"{tag} loss mismatch: thd={loss_thd.item()}, " f"sbhd={loss_sbhd.item()}"
        )
        # topk_indices_thd is (total_q, topk); SBHD is (1, sq, topk).
        assert torch.equal(
            topk_indices_thd, topk_indices_sbhd.squeeze(0).int()
        ), f"{tag} topk mismatch"
        assert torch.allclose(
            grad_q_thd, grad_q_sbhd.squeeze(1), rtol=1e-5, atol=1e-5
        ), f"{tag} grad_q mismatch"
        assert torch.allclose(
            grad_w_thd, grad_w_sbhd.squeeze(1), rtol=1e-5, atol=1e-5
        ), f"{tag} grad_w mismatch"
        assert torch.allclose(
            grad_k_thd, grad_k_sbhd.squeeze(1), rtol=1e-5, atol=1e-5
        ), f"{tag} grad_k mismatch"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_missing_kwarg_raises(self):
        """THD mode requires both ``cu_seqlens_compressed_idx`` and
        ``ratio``; supplying ``cu_seqlens_q`` alone raises ``ValueError``.
        """
        sq, n_compressed = 8, 2
        idx_nh, idx_hd = 4, 32
        num_heads, head_dim = 4, 64
        q = torch.zeros(sq, idx_nh, idx_hd, dtype=torch.float32, device='cuda')
        w = torch.zeros(sq, idx_nh, dtype=torch.float32, device='cuda')
        k = torch.zeros(n_compressed, idx_hd, dtype=torch.float32, device='cuda')
        query = torch.zeros(sq, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        key = torch.zeros(n_compressed, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        cu_q = torch.tensor([0, sq], dtype=torch.int32, device='cuda')
        with pytest.raises(ValueError, match="THD mode requires"):
            FusedDSAIndexerLoss.apply(
                q,
                w,
                k,
                query,
                key,
                head_dim**-0.5,
                2,
                1.0,
                None,
                False,
                self.pg_collection,
                False,  # calculate_per_token_loss
                cu_q,  # cu_seqlens_q supplied
                None,  # cu_seqlens_compressed_idx MISSING
                None,  # ratio MISSING
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_thd_multiseg_zero_compressed_segment_row_mean_normalization(self):
        """THD aggregated loss should be row-mean over ``total_q`` even when
        some segments have ``seqlen_compressed == 0``.

        Build a two-segment THD batch where segment 0 has no compressed keys
        (sq=2, ratio=4 -> 0 compressed) and segment 1 has compressed keys
        (sq=8 -> 2 compressed). Compare THD loss to an SBHD per-segment
        reference aggregated as ``sum(loss_b * sq_b) / total_q``.
        """
        torch.manual_seed(7)
        seg_q_lens = [2, 8]
        seg_comp_lens = [0, 2]
        total_q = sum(seg_q_lens)
        total_comp = sum(seg_comp_lens)
        ratio = 4
        idx_nh, idx_hd = 4, 32
        num_heads, head_dim = 4, 64
        topk = 2
        softmax_scale = head_dim**-0.5
        loss_coeff = 0.5
        dev = 'cuda'

        # THD inputs
        q_thd = torch.randn(total_q, idx_nh, idx_hd, dtype=torch.float32, device=dev)
        w_thd = torch.randn(total_q, idx_nh, dtype=torch.float32, device=dev)
        k_thd = torch.randn(total_comp, idx_hd, dtype=torch.float32, device=dev)
        query_thd = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16, device=dev)
        key_thd = torch.randn(total_comp, num_heads, head_dim, dtype=torch.bfloat16, device=dev)

        cu_seqlens_q = torch.tensor([0, 2, 10], dtype=torch.int32, device=dev)
        cu_seqlens_comp = torch.tensor([0, 0, 2], dtype=torch.int32, device=dev)

        _, loss_thd = FusedDSAIndexerLoss.apply(
            q_thd,
            w_thd,
            k_thd,
            query_thd,
            key_thd,
            softmax_scale,
            topk,
            loss_coeff,
            None,
            False,
            self.pg_collection,
            False,  # calculate_per_token_loss
            cu_seqlens_q,
            cu_seqlens_comp,
            ratio,
        )

        # SBHD per-segment reference: segment 0 contributes zero because it has
        # no compressed keys; segment 1 contributes normally.
        weighted_losses = []
        for b, (sq_b, sk_b) in enumerate(zip(seg_q_lens, seg_comp_lens)):
            if sk_b == 0:
                continue
            q_start = int(cu_seqlens_q[b].item())
            q_end = int(cu_seqlens_q[b + 1].item())
            k_start = int(cu_seqlens_comp[b].item())
            k_end = int(cu_seqlens_comp[b + 1].item())

            q_b = q_thd[q_start:q_end].unsqueeze(1)
            w_b = w_thd[q_start:q_end].unsqueeze(1)
            k_b = k_thd[k_start:k_end].unsqueeze(1)
            query_b = query_thd[q_start:q_end].unsqueeze(1)
            key_b = key_thd[k_start:k_end].unsqueeze(1)

            cols = torch.arange(sk_b, device=dev).unsqueeze(0).expand(sq_b, -1)
            positions = torch.arange(1, sq_b + 1, device=dev).unsqueeze(1)
            mask_b = torch.where(cols >= positions // ratio, float('-inf'), 0.0).unsqueeze(0)

            _, loss_b = FusedDSAIndexerLoss.apply(
                q_b,
                w_b,
                k_b,
                query_b,
                key_b,
                softmax_scale,
                topk,
                loss_coeff,
                mask_b,
                False,
                self.pg_collection,
                False,  # calculate_per_token_loss
            )
            weighted_losses.append(loss_b * sq_b)

        expected_loss = torch.stack(weighted_losses).sum() / float(total_q)
        assert torch.allclose(loss_thd, expected_loss, rtol=1e-5, atol=1e-5), (
            f"THD loss should be row-mean over total_q={total_q}: "
            f"thd={loss_thd.item()}, expected={expected_loss.item()}"
        )


# ===========================================================================
# THD: fused_qk_topk_naive_thd (force_unfused_dsa + indexer + inference path)
# ===========================================================================


class TestFusedQkTopkNaiveThd:
    """``fused_qk_topk_naive_thd`` — per-segment naive PyTorch QK + top-K
    used by the THD ``force_unfused_dsa + indexer + inference`` path
    (i.e., the THD branch of :meth:`CSAIndexer.forward`).

    Coverage:
      * B=1 single-segment THD matches SBHD-b=1 ``fused_qk_topk_naive``
        ranking (same scores → same top-K positions among valid rows).
      * Output shape + dtype contract.
      * ``-1`` sentinel marking on invalid tail positions (rows whose
        causal-valid count is < topk).
      * Multi-segment dispatch isolates per-segment KV scopes (segment
        ``b``'s top-K can only reference KV positions in
        ``[0, seqlen_kv[b])``).
    """

    def _build_causal_mask(self, sq, sk, ratio, device):
        """SBHD-shape ``(1, sq, sk)`` causal mask (mirrors
        ``_build_causal_mask_seg`` for the reference path)."""
        cols = torch.arange(sk, device=device).unsqueeze(0).expand(sq, -1)
        positions = torch.arange(1, sq + 1, device=device).unsqueeze(1)
        return torch.where(cols >= positions // ratio, float('-inf'), 0.0).unsqueeze(0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_segment_matches_sbhd(self):
        """B=1 single-segment THD top-K should match the SBHD-b=1
        reference among valid rows (rows whose causal-valid count is
        smaller than ``topk`` get ``-1`` sentinels in THD where SBHD
        returns garbage tail; we compare only the valid prefix).
        """
        torch.manual_seed(0)
        sq, n_compressed = 32, 8
        idx_nh, idx_hd = 4, 32
        topk = 4
        ratio = 4
        dev = 'cuda'

        q_thd = torch.randn(sq, idx_nh, idx_hd, dtype=torch.float32, device=dev)
        k_thd = torch.randn(n_compressed, idx_hd, dtype=torch.float32, device=dev)
        w_thd = torch.randn(sq, idx_nh, dtype=torch.float32, device=dev)

        cu_q = torch.tensor([0, sq], dtype=torch.int32, device=dev)
        cu_kv = torch.tensor([0, n_compressed], dtype=torch.int32, device=dev)

        _, topk_thd = fused_qk_topk_naive_thd(q_thd, k_thd, w_thd, topk, cu_q, cu_kv, ratio)

        # SBHD reference: same data with b=1 + caller-supplied mask.
        q_sbhd = q_thd.unsqueeze(1)
        k_sbhd = k_thd.unsqueeze(1)
        w_sbhd = w_thd.unsqueeze(1)
        mask_sbhd = self._build_causal_mask(sq, n_compressed, ratio, dev)
        _, topk_sbhd = fused_qk_topk_naive(q_sbhd, k_sbhd, w_sbhd, topk, mask_sbhd)
        topk_sbhd = topk_sbhd.squeeze(0)  # (sq, topk)

        # Per-row: compare only the leading ``n_valid`` slots; THD marks
        # the rest as -1, SBHD's tail is undefined (masked -inf
        # positions, ties may break differently).
        for row in range(sq):
            n_valid = min((row + 1) // ratio, n_compressed, topk)
            assert torch.equal(
                topk_thd[row, :n_valid].cpu(), topk_sbhd[row, :n_valid].cpu()
            ), f"row {row}: top-K mismatch among valid slots"
            assert (
                topk_thd[row, n_valid:] == -1
            ).all(), f"row {row}: THD must mark invalid tail as -1"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_shape_and_dtype(self):
        """Returns ``(None, (total_q, topk) int64)``."""
        torch.manual_seed(0)
        sq_a, sq_b = 8, 4
        kv_a, kv_b = 2, 1
        idx_nh, idx_hd = 2, 16
        topk = 3
        ratio = 4
        dev = 'cuda'

        q = torch.randn(sq_a + sq_b, idx_nh, idx_hd, dtype=torch.float32, device=dev)
        k = torch.randn(kv_a + kv_b, idx_hd, dtype=torch.float32, device=dev)
        w = torch.randn(sq_a + sq_b, idx_nh, dtype=torch.float32, device=dev)
        cu_q = torch.tensor([0, sq_a, sq_a + sq_b], dtype=torch.int32, device=dev)
        cu_kv = torch.tensor([0, kv_a, kv_a + kv_b], dtype=torch.int32, device=dev)

        scores, topk_idxs = fused_qk_topk_naive_thd(q, k, w, topk, cu_q, cu_kv, ratio)
        assert scores is None
        assert topk_idxs.shape == (sq_a + sq_b, topk)
        assert topk_idxs.dtype == torch.int64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_per_segment_kv_scope(self):
        """Segment ``b``'s top-K LOCAL ids must live in
        ``[0, seqlen_kv[b])`` (per-segment scope) — they are NOT
        flat-global ids into the concatenated K tensor.
        """
        torch.manual_seed(0)
        sq_a, sq_b = 16, 12
        kv_a, kv_b = 4, 3
        idx_nh, idx_hd = 2, 16
        topk = 2
        ratio = 4
        dev = 'cuda'

        q = torch.randn(sq_a + sq_b, idx_nh, idx_hd, dtype=torch.float32, device=dev)
        k = torch.randn(kv_a + kv_b, idx_hd, dtype=torch.float32, device=dev)
        w = torch.randn(sq_a + sq_b, idx_nh, dtype=torch.float32, device=dev)
        cu_q = torch.tensor([0, sq_a, sq_a + sq_b], dtype=torch.int32, device=dev)
        cu_kv = torch.tensor([0, kv_a, kv_a + kv_b], dtype=torch.int32, device=dev)

        _, topk_idxs = fused_qk_topk_naive_thd(q, k, w, topk, cu_q, cu_kv, ratio)

        # Segment 0 rows: valid ids must be in [0, kv_a).
        seg0 = topk_idxs[:sq_a]
        seg0_valid = seg0[seg0 >= 0]
        assert (seg0_valid < kv_a).all(), (
            f"segment 0 has out-of-range ids: max = {seg0_valid.max().item()}, "
            f"expected < {kv_a}"
        )
        # Segment 1 rows: valid ids must be in [0, kv_b).
        seg1 = topk_idxs[sq_a:]
        seg1_valid = seg1[seg1 >= 0]
        assert (seg1_valid < kv_b).all(), (
            f"segment 1 has out-of-range ids: max = {seg1_valid.max().item()}, "
            f"expected < {kv_b}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_invalid_tail_marked_minus_one(self):
        """Early rows where ``(pos+1)//ratio < topk`` should have ``-1``
        sentinels in the tail of their top-K row.
        """
        torch.manual_seed(0)
        sq, n_compressed = 8, 4
        idx_nh, idx_hd = 2, 16
        topk = 4
        ratio = 4
        dev = 'cuda'

        q = torch.randn(sq, idx_nh, idx_hd, dtype=torch.float32, device=dev)
        k = torch.randn(n_compressed, idx_hd, dtype=torch.float32, device=dev)
        w = torch.randn(sq, idx_nh, dtype=torch.float32, device=dev)
        cu_q = torch.tensor([0, sq], dtype=torch.int32, device=dev)
        cu_kv = torch.tensor([0, n_compressed], dtype=torch.int32, device=dev)

        _, topk_idxs = fused_qk_topk_naive_thd(q, k, w, topk, cu_q, cu_kv, ratio)

        # Causal-valid count per row: min((pos+1)//ratio, n_compressed, topk).
        for row in range(sq):
            n_valid = min((row + 1) // ratio, n_compressed, topk)
            row_idxs = topk_idxs[row]
            assert (
                row_idxs[:n_valid] >= 0
            ).all() or n_valid == 0, f"row {row}: leading {n_valid} should be valid ids"
            assert (row_idxs[n_valid:] == -1).all(), f"row {row}: tail beyond {n_valid} must be -1"
