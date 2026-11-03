# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn as nn

import megatron.core.parallel_state as ps
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.model_initialization import initialize_muon_ht_parameters
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.models.dist_utils import (
    prepare_existing_model_chunks_for_distributed_training,
)
from tests.unit_tests.test_utilities import Utils


def _model_process_groups() -> ProcessGroupCollection:
    """Build the explicit groups consumed by the model-initialization lifecycle."""
    return ProcessGroupCollection(
        tp=ps.get_tensor_model_parallel_group(),
        pp=ps.get_pipeline_model_parallel_group(),
        dp=ps.get_data_parallel_group(with_gtp_remat=False),
        cp=ps.get_context_parallel_group(),
        gtp_remat=ps.get_gtp_weight_remat_group(check_initialized=False),
        expt_tp=ps.get_expert_tensor_parallel_group(check_initialized=False),
        expt_gtp_remat=ps.get_expert_gtp_weight_remat_group(check_initialized=False),
    )


class Test:

    transformer_config = TransformerConfig(
        num_layers=1, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
    )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "enabled,perform_initialization", [(True, True), (False, True), (True, False)]
    )
    def test_default_hyperball_initialization_uses_fan_in_only_when_enabled(
        self, enabled, perform_initialization
    ):
        """The model lifecycle uses Gaussian draws, not projection to sqrt(d_out)."""
        Utils.initialize_model_parallel(1, 1)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            muon_ht_init=enabled,
            perform_initialization=perform_initialization,
        )
        model = nn.Linear(12, 8, bias=True)
        before = model.weight.detach().clone()
        bias_before = model.bias.detach().clone()
        torch.manual_seed(123)
        expected = torch.empty_like(model.weight).normal_(std=12**-0.5)
        torch.manual_seed(123)
        result = prepare_existing_model_chunks_for_distributed_training(
            [model],
            config,
            _model_process_groups(),
            wrap_with_ddp=False,
            mixed_precision_wrapper=None,
        )
        torch.testing.assert_close(
            result[0].weight,
            expected if enabled and perform_initialization else before,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(result[0].bias, bias_before, rtol=0, atol=0)

    def test_fan_in_initialization_skips_excluded_and_shared_parameters(self):
        """Only managed matrices are sampled, and tied parameters consume one RNG draw."""
        model = nn.Module()
        model.weight = nn.Parameter(torch.empty(8, 16))
        model.tied = model.weight
        model.excluded = nn.Parameter(torch.ones(5, 4))
        model.excluded.use_muon = False
        model.embedding = nn.Parameter(torch.ones(5, 4))
        model.embedding.is_embedding_or_output_parameter = True
        model.frozen = nn.Parameter(torch.ones(5, 4), requires_grad=False)
        groups = ProcessGroupCollection(tp=None, expt_tp=None, gtp_remat=None, expt_gtp_remat=None)
        torch.manual_seed(111)
        expected = torch.empty(8, 16).normal_(std=16**-0.5)
        next_draw = torch.randn(3)
        torch.manual_seed(111)
        initialize_muon_ht_parameters([model, model], None, 1e-15, groups)
        torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.randn(3), next_draw, rtol=0, atol=0)
        for param in (model.excluded, model.embedding, model.frozen):
            torch.testing.assert_close(param, torch.ones_like(param), rtol=0, atol=0)

    @pytest.mark.parametrize("fsdp_flag", ["use_torch_fsdp2", "use_megatron_fsdp"])
    def test_hyperball_meta_fsdp_init_is_rejected_instead_of_silently_skipped(self, fsdp_flag):
        """A deferred FSDP materializer cannot bypass the model initialization contract."""
        config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            muon_ht_init=True,
            init_model_with_meta_device=True,
        )
        groups = ProcessGroupCollection(tp=None, expt_tp=None, gtp_remat=None, expt_gtp_remat=None)
        with pytest.raises(ValueError, match="materialized parameters before FSDP"):
            prepare_existing_model_chunks_for_distributed_training(
                [nn.Linear(12, 8, device="meta")],
                config,
                groups,
                wrap_with_ddp=False,
                **{fsdp_flag: True},
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_muon_ht_init_runs_before_distributed_wrapping(self):
        """The model initialization lifecycle should enforce a configured Hyperball radius."""
        Utils.initialize_model_parallel(1, 1)
        config = TransformerConfig(
            num_layers=1, hidden_size=12, num_attention_heads=4, muon_ht_radius=2.5
        )
        model = nn.Linear(12, 12, bias=True, device="cuda")

        result = prepare_existing_model_chunks_for_distributed_training(
            [model],
            config,
            _model_process_groups(),
            wrap_with_ddp=False,
            mixed_precision_wrapper=None,
        )

        torch.testing.assert_close(result[0].weight.norm(), torch.tensor(2.5, device="cuda"))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_muon_ht_init_respects_disabled_parameter_initialization(self):
        """Checkpoint-oriented model construction should not modify uninitialized parameters."""
        Utils.initialize_model_parallel(1, 1)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            muon_ht_radius=2.5,
            perform_initialization=False,
        )
        model = nn.Linear(12, 12, bias=False, device="cuda")
        weight_before = model.weight.detach().clone()

        result = prepare_existing_model_chunks_for_distributed_training(
            [model],
            config,
            _model_process_groups(),
            wrap_with_ddp=False,
            mixed_precision_wrapper=None,
        )

        torch.testing.assert_close(result[0].weight, weight_before, rtol=0.0, atol=0.0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_embedding_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = VocabParallelEmbedding(
            num_embeddings=16,
            embedding_dim=4,
            init_method=self.transformer_config.init_method,
            config=self.transformer_config,
        ).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = VocabParallelEmbedding(
            num_embeddings=16,
            embedding_dim=4,
            init_method=self.transformer_config.init_method,
            config=self.transformer_config,
        ).weight

        rank = ps.get_tensor_model_parallel_rank()
        assert tp4.shape[0] * 4 == tp1.shape[0]
        assert torch.equal(tp1[rank * 4 : (rank + 1) * 4], tp4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_row_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = RowParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            input_is_parallel=False,
            config=self.transformer_config,
            skip_bias_add=False,
        ).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = RowParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            input_is_parallel=False,
            config=self.transformer_config,
            skip_bias_add=False,
        ).weight

        rank = ps.get_tensor_model_parallel_rank()
        assert tp4.shape[1] * 4 == tp1.shape[1]
        assert torch.equal(tp1[:, rank * 4 : (rank + 1) * 4], tp4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_col_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = ColumnParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            config=self.transformer_config,
            skip_bias_add=False,
        ).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = ColumnParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            config=self.transformer_config,
            skip_bias_add=False,
        ).weight

        rank = ps.get_tensor_model_parallel_rank()
        assert tp4.shape[0] * 4 == tp1.shape[0]
        assert torch.equal(tp1[rank * 4 : (rank + 1) * 4], tp4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.timeout(100)
    def test_te_col_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = TEColumnParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            config=self.transformer_config,
            skip_bias_add=False,
            gather_output=False,
            is_expert=False,
        ).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = TEColumnParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            config=self.transformer_config,
            skip_bias_add=False,
            gather_output=False,
            is_expert=False,
        ).weight

        if torch.distributed.get_rank() == 0:
            assert tp4.shape[0] * 4 == tp1.shape[0]
            assert torch.allclose(tp1[:4], tp4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.timeout(100)
    def test_te_row_init(self):

        Utils.initialize_model_parallel(1, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)

        tp1 = TERowParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            input_is_parallel=True,
            config=self.transformer_config,
            skip_bias_add=False,
            is_expert=False,
        ).weight
        Utils.destroy_model_parallel()

        Utils.initialize_model_parallel(4, 1)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(41)  # intentionally different.
        tp4 = TERowParallelLinear(
            input_size=16,
            output_size=16,
            init_method=self.transformer_config.init_method,
            bias=True,
            input_is_parallel=True,
            config=self.transformer_config,
            skip_bias_add=False,
            is_expert=False,
        ).weight

        if torch.distributed.get_rank() == 0:
            assert tp4.shape[1] * 4 == tp1.shape[1]
            assert torch.allclose(tp1[:, :4], tp4)
