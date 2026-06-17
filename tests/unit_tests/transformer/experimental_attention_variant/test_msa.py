# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_experimental_attention_variant_module_spec,
    get_msa_module_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.msa import (
    MSAIndexerLossAutoscaler,
    MSAIndexerLossLoggingHelper,
    MSASelfAttention,
    MSASelfAttentionSubmodules,
    _block_max_pool,
    _build_block_mask,
    _compute_msa_kl_loss,
    _select_topk_blocks,
)
from tests.unit_tests.test_utilities import Utils

# ======================================================================================
# Pure-function tests (no model-parallel init required)
# ======================================================================================


class TestBuildBlockMask:
    """Test _build_block_mask utility function."""

    def test_basic_shape(self):
        batch = 2
        seqlen = 16
        num_kv_groups = 2
        k = 4
        block_size = 4
        num_blocks = seqlen // block_size
        device = torch.device("cpu")

        block_indices = torch.randint(0, num_blocks, (batch, seqlen, num_kv_groups, k))
        mask = _build_block_mask(
            block_indices,
            num_blocks,
            block_size,
            seqlen,
            num_kv_groups,
            batch,
            device,
            torch.float32,
        )

        assert mask.shape == (batch, num_kv_groups, seqlen, seqlen)
        assert mask.dtype == torch.float32

    def test_causal_property(self):
        batch = 1
        seqlen = 8
        num_kv_groups = 1
        k = 2
        block_size = 4
        num_blocks = seqlen // block_size
        device = torch.device("cpu")

        # All queries select block 0 only
        block_indices = torch.zeros(batch, seqlen, num_kv_groups, k, dtype=torch.long)
        mask = _build_block_mask(
            block_indices,
            num_blocks,
            block_size,
            seqlen,
            num_kv_groups,
            batch,
            device,
            torch.float32,
        )

        # For qi=0, no blocks should be selected (start=0, min(start+block_size,seqlen)=4,
        # but start >= qi, so no tokens match for qi=0)
        for qi in range(seqlen):
            for kj in range(seqlen):
                if kj < qi and kj < block_size:
                    assert mask[0, 0, qi, kj] == 0.0, f"Expected 0.0 at ({qi},{kj})"
                elif kj > qi:
                    assert mask[0, 0, qi, kj] == float("-inf"), (
                        f"Expected -inf at ({qi},{kj})"
                    )

    def test_invalid_blocks_ignored(self):
        batch = 1
        seqlen = 8
        num_kv_groups = 1
        k = 2
        block_size = 4
        num_blocks = seqlen // block_size
        device = torch.device("cpu")

        # Block index -1 should be ignored
        block_indices = torch.full(
            (batch, seqlen, num_kv_groups, k), -1, dtype=torch.long
        )
        mask = _build_block_mask(
            block_indices,
            num_blocks,
            block_size,
            seqlen,
            num_kv_groups,
            batch,
            device,
            torch.float32,
        )

        assert torch.all(mask == float("-inf")), (
            "All entries should be -inf when all blocks are invalid"
        )


class TestBlockMaxPool:
    """Test _block_max_pool utility function."""

    def test_basic_shape(self):
        sq = 8
        b_size = 2
        n_kv = 2
        sk = 8
        block_size = 4

        scores = torch.randn(sq, b_size, n_kv, sk)
        num_blocks = (sk + block_size - 1) // block_size
        block_scores = _block_max_pool(scores, block_size, num_blocks, sq)

        assert block_scores.shape == (sq, b_size, n_kv, num_blocks)

    def test_causal_block_masking(self):
        sq = 8
        b_size = 1
        n_kv = 1
        sk = 8
        block_size = 4
        num_blocks = 2

        scores = torch.full((sq, b_size, n_kv, sk), -100.0)
        scores[4, 0, 0, :] = 50.0  # High score at token 4 in block 1

        block_scores = _block_max_pool(scores, block_size, num_blocks, sq)

        # For qi=0, block 1 should be -inf (future block)
        assert block_scores[0, 0, 0, 1] == float("-inf"), (
            "Future block should be masked out at qi=0"
        )

        # For qi=5, block 0 should be accessible (causal allows past tokens)
        assert block_scores[5, 0, 0, 0] > float("-inf"), (
            "Past block should be accessible at qi=5"
        )

    def test_short_sequence_padding(self):
        sq = 6
        b_size = 1
        n_kv = 1
        sk = 6
        block_size = 4
        num_blocks = 2

        scores = torch.randn(sq, b_size, n_kv, sk)
        block_scores = _block_max_pool(scores, block_size, num_blocks, sq)

        assert block_scores.shape == (sq, b_size, n_kv, num_blocks)
        assert not torch.isnan(block_scores).any(), "No NaNs expected"


class TestSelectTopkBlocks:
    """Test _select_topk_blocks utility function."""

    def test_basic_shape(self):
        sq = 8
        b_size = 2
        n_kv = 2
        k = 3
        block_size = 4
        num_blocks = 2

        block_scores = torch.randn(sq, b_size, n_kv, num_blocks)
        block_indices = _select_topk_blocks(block_scores, k, sq, block_size, num_blocks)

        assert block_indices.shape == (b_size, sq, n_kv, k)
        assert block_indices.dtype == torch.long

    def test_local_block_included(self):
        sq = 8
        b_size = 1
        n_kv = 1
        k = 1
        block_size = 4
        num_blocks = 2

        # Make block 0 very negative, block 1 very positive
        block_scores = torch.full((sq, b_size, n_kv, num_blocks), float("-inf"))
        block_scores[:, :, :, 1] = 100.0

        block_indices = _select_topk_blocks(block_scores, k, sq, block_size, num_blocks)

        # For qi=0, local block is block 0; for qi=5, local block is block 1
        # qi=0's local block is 0, so at k=1 it should be included even if block_scores are -inf
        assert block_indices[0, 0, 0, 0] == 0, "Local block should be included for qi=0"
        # qi=5's local block is block 1, which has high score
        assert block_indices[0, 5, 0, 0] == 1, "Top-1 block should be block 1 for qi=5"

    def test_no_valid_blocks(self):
        sq = 8
        b_size = 1
        n_kv = 1
        k = 3
        block_size = 4
        num_blocks = 2

        block_scores = torch.full((sq, b_size, n_kv, num_blocks), float("-inf"))
        block_indices = _select_topk_blocks(block_scores, k, sq, block_size, num_blocks)

        assert block_indices.shape == (b_size, sq, n_kv, k)
        # At qi=0, local block is block 0 which should still be included
        assert block_indices[0, 0, 0, 0] == 0


# ======================================================================================
# Tests requiring model-parallel init (TP=1)
# ======================================================================================


@pytest.fixture(scope="module", autouse=True)
def msa_module_init():
    """Initialize model parallel for the MSA test module."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1
    )
    yield
    Utils.destroy_model_parallel()


class TestMSAIndexerLossLoggingHelper:
    """Test MSAIndexerLossLoggingHelper static methods."""

    def teardown_method(self):
        MSAIndexerLossLoggingHelper.tracker = {}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_save_and_clean_loss(self):
        """Test saving loss to tracker and cleaning up."""
        loss = torch.tensor(1.0, device="cuda")
        MSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=loss, layer_number=1, num_layers=4
        )
        assert "values" in MSAIndexerLossLoggingHelper.tracker
        assert MSAIndexerLossLoggingHelper.tracker["values"].shape == (4,)

        MSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        assert "values" not in MSAIndexerLossLoggingHelper.tracker

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_none_layer_number_ignored(self):
        """Test that None layer_number is safely ignored."""
        MSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss=torch.tensor(1.0, device="cuda"),
            layer_number=None,
            num_layers=4,
        )
        assert "values" not in MSAIndexerLossLoggingHelper.tracker


class TestMSAIndexerLossAutoscaler:
    """Test MSAIndexerLossAutoscaler autograd function."""

    def teardown_method(self):
        MSAIndexerLossAutoscaler.main_loss_backward_scale = None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_pass(self):
        """Test that forward pass preserves output."""
        output = torch.randn(16, 2, 128).cuda()
        output.requires_grad_(True)
        indexer_loss = torch.tensor(0.5).cuda()
        indexer_loss.requires_grad_(True)

        result = MSAIndexerLossAutoscaler.apply(output, indexer_loss)

        assert torch.allclose(result, output, atol=0, rtol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_pass(self):
        """Test that backward pass triggers indexer loss backward and scales gradient correctly."""
        output = torch.randn(16, 2, 128).cuda()
        output.requires_grad_(True)

        dummy_input = torch.randn(10).cuda()
        dummy_input.requires_grad_(True)
        indexer_loss = dummy_input.mean()

        scale = torch.tensor(2.0).cuda()
        MSAIndexerLossAutoscaler.set_loss_scale(scale)

        result = MSAIndexerLossAutoscaler.apply(output, indexer_loss)

        main_loss = result.sum()
        main_loss.backward()

        assert output.grad is not None, "Gradient should flow back to parameters"
        assert dummy_input.grad is not None, "Indexer loss backward should be triggered"

        expected_grad_per_element = scale.item() / len(dummy_input)
        assert torch.allclose(
            dummy_input.grad,
            torch.full_like(dummy_input, expected_grad_per_element),
            rtol=0,
            atol=0,
        ), (
            f"Gradient should be scaled by loss scale, expected {expected_grad_per_element}"
        )


def _create_msa_config(
    num_layers=2, hidden_size=256, num_attention_heads=16, num_query_groups=4
):
    """Helper to create a TransformerConfig with MSA parameters."""
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        # MSA specific configs
        msa_block_size=4,
        msa_num_selected_blocks=2,
        msa_indexer_head_dim=32,
        msa_loss_coeff=1.0,
        msa_warmup=False,
        # RoPE
        rotary_percent=1.0,
        rotary_base=10000,
    )


def _create_msa_submodules():
    """Helper to create MSA submodules spec."""
    from megatron.core.extensions.transformer_engine import TELinear, TENorm
    from megatron.core.transformer.spec_utils import ModuleSpec

    return MSASelfAttentionSubmodules(
        linear_qkv=ModuleSpec(module=TELinear),
        linear_idx_q=ModuleSpec(module=TELinear),
        linear_idx_k=ModuleSpec(module=TELinear),
        linear_proj=ModuleSpec(module=TELinear),
        q_layernorm=ModuleSpec(module=TENorm),
        k_layernorm=ModuleSpec(module=TENorm),
    )


class TestMSAComputeKL:
    """Test _compute_msa_kl_loss function."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_kl_loss_shape(self):
        """Test that KL loss returns a scalar."""
        sq = 8
        b_size = 2
        num_heads = 8
        num_kv_groups = 4
        head_dim = 64
        block_size = 4
        num_blocks = 2

        query = torch.randn(
            sq, b_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        key = torch.randn(
            sq, b_size, num_kv_groups, head_dim, dtype=torch.bfloat16
        ).cuda()
        idx_scores = torch.randn(
            sq, b_size, num_kv_groups, sq, dtype=torch.bfloat16
        ).cuda()
        block_indices = torch.randint(
            0, num_blocks, (b_size, sq, num_kv_groups, 2)
        ).cuda()

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "cp"]
        )

        loss = _compute_msa_kl_loss(
            query=query,
            key=key,
            idx_scores=idx_scores,
            block_indices=block_indices,
            block_size=block_size,
            num_blocks=num_blocks,
            softmax_scale=head_dim**-0.5,
            loss_coeff=1.0,
            num_kv_groups=num_kv_groups,
            num_query_heads=num_heads,
            pg_collection=pg_collection,
        )

        assert loss.shape == torch.Size([])
        assert loss.dtype == torch.float32
        assert loss >= 0 or torch.isclose(loss, torch.tensor(0.0))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_kl_loss_zero_when_identical(self):
        """Test that KL loss is near zero when index and main distributions match."""
        sq = 4
        b_size = 1
        num_heads = 4
        num_kv_groups = 2
        head_dim = 32
        block_size = 4
        num_blocks = 1

        torch.manual_seed(42)

        query = torch.randn(
            sq, b_size, num_heads, head_dim, dtype=torch.bfloat16
        ).cuda()
        key = torch.randn(
            sq, b_size, num_kv_groups, head_dim, dtype=torch.bfloat16
        ).cuda()

        # Set idx_scores to match scaled query-key scores
        q_per_group = num_heads // num_kv_groups
        idx_scores_list = []
        for r in range(num_kv_groups):
            start_h = r * q_per_group
            end_h = start_h + q_per_group
            q_group = query[:, :, start_h:end_h, :]
            k_group = key[:, :, r : r + 1, :]
            q_2d = q_group.reshape(sq, b_size * q_per_group, head_dim)
            k_2d = k_group.reshape(sq, b_size, head_dim).transpose(0, 1)
            k_2d = k_2d.unsqueeze(1).expand(-1, q_per_group, -1, -1)
            k_2d = k_2d.reshape(b_size * q_per_group, sq, head_dim)
            scores = torch.bmm(q_2d.float(), k_2d.float().transpose(1, 2))
            scores = scores.reshape(sq, b_size, q_per_group, sq) * (head_dim**-0.5)
            avg_score = scores.mean(dim=2, keepdim=True)
            idx_scores_list.append(avg_score)

        idx_scores = torch.cat(idx_scores_list, dim=2)

        block_indices = torch.zeros(
            b_size, sq, num_kv_groups, 1, dtype=torch.long
        ).cuda()

        pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "cp"]
        )

        loss = _compute_msa_kl_loss(
            query=query,
            key=key,
            idx_scores=idx_scores,
            block_indices=block_indices,
            block_size=block_size,
            num_blocks=num_blocks,
            softmax_scale=head_dim**-0.5,
            loss_coeff=1.0,
            num_kv_groups=num_kv_groups,
            num_query_heads=num_heads,
            pg_collection=pg_collection,
        )

        # KL divergence should be small when distributions are similar
        assert loss < 0.5, (
            f"KL loss should be small for similar distributions, got {loss.item()}"
        )


class TestMSASelfAttention:
    """Test MSASelfAttention module basic functionality with TP=1."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_method(self, request):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)

        cls = request.cls
        cls.config = _create_msa_config()
        cls.submodules = _create_msa_submodules()
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "cp"]
        )

        cls.msa_attention = MSASelfAttention(
            config=cls.config,
            submodules=cls.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=cls.pg_collection,
        )

        yield
        Utils.destroy_model_parallel()

    def test_msa_constructor(self):
        """Test MSA initialization."""
        assert isinstance(self.msa_attention, MSASelfAttention)
        assert self.msa_attention.hidden_size == 256
        assert self.msa_attention.msa_block_size == 4
        assert self.msa_attention.msa_k == 2
        assert self.msa_attention.msa_idx_dim == 32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_msa_forward(self):
        """Test MSA forward pass."""
        seq_len = 16
        batch_size = 2

        self.msa_attention.cuda()

        hidden_states = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()

        attention_mask = torch.ones(
            batch_size, 1, seq_len, seq_len, dtype=torch.bool
        ).cuda()
        attention_mask = torch.tril(attention_mask)

        output, bias = self.msa_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        assert output.shape == (seq_len, batch_size, self.config.hidden_size)
        assert output.dtype == torch.bfloat16
        # bias can be None or a tensor
        if bias is not None:
            assert bias.shape == output.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_msa_backward(self):
        """Test MSA backward pass with indexer loss."""
        seq_len = 16
        batch_size = 2

        self.msa_attention.train()
        self.msa_attention.cuda()

        hidden_states = torch.randn(
            seq_len, batch_size, self.config.hidden_size, dtype=torch.bfloat16
        ).cuda()
        hidden_states.requires_grad_(True)

        attention_mask = torch.ones(
            batch_size, 1, seq_len, seq_len, dtype=torch.bool
        ).cuda()
        attention_mask = torch.tril(attention_mask)

        output, bias = self.msa_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None

        # Check that MSA parameters have gradients
        for name, param in self.msa_attention.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_msa_warmup_mode(self):
        """Test MSA with warmup mode."""
        config_warmup = _create_msa_config()
        config_warmup.msa_warmup = True
        submodules = _create_msa_submodules()

        msa_warmup = MSASelfAttention(
            config=config_warmup,
            submodules=submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=self.pg_collection,
        ).cuda()

        seq_len = 16
        batch_size = 2

        hidden_states = torch.randn(
            seq_len, batch_size, config_warmup.hidden_size, dtype=torch.bfloat16
        ).cuda()

        attention_mask = torch.ones(
            batch_size, 1, seq_len, seq_len, dtype=torch.bool
        ).cuda()
        attention_mask = torch.tril(attention_mask)

        output, bias = msa_warmup(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        assert output.shape == (seq_len, batch_size, config_warmup.hidden_size)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_msa_forward_no_loss(self):
        """Test MSA forward pass with loss_coeff=0 (inference path)."""
        config_no_loss = _create_msa_config()
        config_no_loss.msa_loss_coeff = 0.0
        submodules = _create_msa_submodules()

        msa_no_loss = MSASelfAttention(
            config=config_no_loss,
            submodules=submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            pg_collection=self.pg_collection,
        ).cuda()
        msa_no_loss.eval()

        seq_len = 16
        batch_size = 2

        hidden_states = torch.randn(
            seq_len, batch_size, config_no_loss.hidden_size, dtype=torch.bfloat16
        ).cuda()

        attention_mask = torch.ones(
            batch_size, 1, seq_len, seq_len, dtype=torch.bool
        ).cuda()
        attention_mask = torch.tril(attention_mask)

        with torch.no_grad():
            output, bias = msa_no_loss(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        assert output.shape == (seq_len, batch_size, config_no_loss.hidden_size)


# ======================================================================================
# Module spec tests
# ======================================================================================


class TestMSAModuleSpec:
    """Test MSA module spec construction."""

    def test_get_msa_module_spec(self):
        """Test that get_msa_module_spec returns a valid spec."""
        spec = get_msa_module_spec()
        assert spec is not None
        assert spec.module == MSASelfAttention
        assert hasattr(spec, "submodules")

    def test_get_experimental_attention_variant_msa_spec(self):
        """Test that MSA spec is returned for 'msa' variant."""
        gpt_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        spec = get_experimental_attention_variant_module_spec(
            experimental_attention_variant="msa",
            gpt_layer_spec=gpt_layer_spec,
        )
        assert spec is not None
        assert spec.module == MSASelfAttention

    def test_get_msa_module_spec_from_config(self):
        """Test MSA spec construction with config parameters."""
        spec = get_msa_module_spec()
        assert spec.submodules.linear_qkv is not None
        assert spec.submodules.linear_idx_q is not None
        assert spec.submodules.linear_idx_k is not None
        assert spec.submodules.linear_proj is not None
        assert spec.submodules.q_layernorm is not None
        assert spec.submodules.k_layernorm is not None
