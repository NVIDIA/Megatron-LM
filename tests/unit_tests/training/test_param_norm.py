# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.utils import common_utils
from tests.unit_tests.test_utilities import Utils


def _build_tiny_moe_gpt(
    tensor_parallel_size: int,
    expert_parallel_size: int,
    expert_tensor_parallel_size: int,
    tensor_parallel_num_weight_shards: int | None = None,
    expert_tensor_parallel_num_weight_shards: int | None = None,
    bf16: bool = False,
    add_bias_linear: bool = False,
) -> GPTModel:
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=4,
        ffn_hidden_size=16,
        num_moe_experts=2,
        moe_ffn_hidden_size=16,
        # Shared experts do not support linear biases.
        moe_shared_expert_intermediate_size=None if add_bias_linear else 16,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        tensor_model_parallel_size=tensor_parallel_size,
        expert_model_parallel_size=expert_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        tensor_parallel_num_weight_shards=tensor_parallel_num_weight_shards,
        expert_tensor_parallel_num_weight_shards=expert_tensor_parallel_num_weight_shards,
        sequence_parallel=tensor_parallel_size > 1,
        use_cpu_initialization=True,
        add_bias_linear=add_bias_linear,
        normalization="RMSNorm",
        moe_grouped_gemm=True,
        bf16=bf16,
        params_dtype=torch.bfloat16 if bf16 else torch.float32,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=config.num_moe_experts, moe_grouped_gemm=True
        ),
        vocab_size=16,
        max_sequence_length=8,
        position_embedding_type="rope",
    
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
    if not add_bias_linear:
        assert any(".shared_experts." in name for name, _ in model.named_parameters())
    return model.cuda()


def _fill_parameters_with_ones(model: GPTModel) -> None:
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(1.0)


@pytest.mark.parametrize(
    ("tensor_parallel_size", "expert_parallel_size", "expert_tensor_parallel_size"),
    ((2, 2, 1), (2, 1, 2), (4, 1, 2), (2, 1, 4)),
    ids=("expert-parallel", "expert-tensor-parallel", "tp-larger-than-etp", "etp-larger-than-tp"),
)
def test_moe_param_norm_counts_each_logical_parameter_once(
    monkeypatch,
    tensor_parallel_size: int,
    expert_parallel_size: int,
    expert_tensor_parallel_size: int,
):
    """Parameter norm should be invariant to expert and expert-tensor parallelism."""
    if Utils.world_size < 4 or Utils.world_size % 4 != 0:
        pytest.skip("test requires a world size divisible by four")

    monkeypatch.setattr(
        common_utils, "get_args", lambda: SimpleNamespace(use_megatron_fsdp=False, bf16=False)
    )

    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
        reference_model = _build_tiny_moe_gpt(
            tensor_parallel_size=1, expert_parallel_size=1, expert_tensor_parallel_size=1
        )
        _fill_parameters_with_ones(reference_model)
        expected_numel = sum(param.numel() for param in reference_model.parameters())
        expected_norm = math.sqrt(expected_numel)
        reference_norm = common_utils.calc_params_l2_norm(reference_model)

        assert reference_norm == pytest.approx(expected_norm)
        del reference_model

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            expert_model_parallel_size=expert_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
        )
        distributed_model = _build_tiny_moe_gpt(
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=expert_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
        )
        _fill_parameters_with_ones(distributed_model)

        actual_norm = common_utils.calc_params_l2_norm(distributed_model)

        assert actual_norm == pytest.approx(expected_norm)
    finally:
        Utils.destroy_model_parallel()


def test_moe_param_norm_uses_expert_gtp_topology_when_it_differs_from_dense_gtp(monkeypatch):
    """Expert parameters must use EGTP even when EP, TP, and ETP alone do not distinguish them."""
    from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
        GTP_CONFIG,
        GTPShardedParam,
        reset_gtp_state,
        update_gtp_config,
    )
    from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

    if not HAVE_GTP:
        pytest.skip("GTP requires TransformerEngine >= 2.19")
    if Utils.world_size < 2 or Utils.world_size % 2 != 0:
        pytest.skip("test requires an even world size")

    monkeypatch.setattr(
        common_utils, "get_args", lambda: SimpleNamespace(use_megatron_fsdp=False, bf16=False)
    )
    # Keep the all-ones assertion focused on topology rather than physical GTP padding.
    original_pad_for_alignment = GTP_CONFIG.pad_for_alignment
    update_gtp_config(pad_for_alignment=0)

    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
        reference_model = _build_tiny_moe_gpt(
            tensor_parallel_size=1, expert_parallel_size=1, expert_tensor_parallel_size=1
        )
        _fill_parameters_with_ones(reference_model)
        expected_numel = sum(param.numel() for param in reference_model.parameters())
        expected_norm = math.sqrt(expected_numel)
        del reference_model

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            gtp_remat_size=1,
            expert_gtp_remat_size=2,
        )
        model = _build_tiny_moe_gpt(
            tensor_parallel_size=1,
            expert_parallel_size=1,
            expert_tensor_parallel_size=1,
            tensor_parallel_num_weight_shards=1,
            expert_tensor_parallel_num_weight_shards=2,
        )
        _fill_parameters_with_ones(model)

        expert_params = [param for name, param in model.named_parameters() if ".experts." in name]
        assert any(isinstance(param, GTPShardedParam) for param in expert_params)

        actual_norm = common_utils.calc_params_l2_norm(model)

        assert actual_norm == pytest.approx(expected_norm)
    finally:
        update_gtp_config(pad_for_alignment=original_pad_for_alignment)
        reset_gtp_state()
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_distributed_optimizer", (False, True), ids=("optimizer", "distopt"))
@pytest.mark.parametrize(
    (
        "tensor_parallel_size",
        "expert_parallel_size",
        "expert_tensor_parallel_size",
        "gtp_weight_remat_size",
        "expert_gtp_weight_remat_size",
    ),
    ((2, 2, 1, 1, 1), (2, 1, 2, 1, 1), (4, 1, 2, 1, 1), (2, 1, 4, 1, 1), (1, 1, 1, 1, 2)),
    ids=(
        "expert-parallel",
        "expert-tensor-parallel",
        "tp-larger-than-etp",
        "etp-larger-than-tp",
        "expert-gtp-differs-from-dense-gtp",
    ),
)
def test_moe_gradient_stats_and_clipping_count_each_logical_gradient_once(
    tensor_parallel_size: int,
    expert_parallel_size: int,
    expert_tensor_parallel_size: int,
    gtp_weight_remat_size: int,
    expert_gtp_weight_remat_size: int,
    use_distributed_optimizer: bool,
):
    """Gradient norm, clipping, and zero count should include each logical gradient once."""
    if expert_gtp_weight_remat_size > 1:
        from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

        if not HAVE_GTP:
            pytest.skip("GTP requires TransformerEngine >= 2.19")
    if Utils.world_size < 4 or Utils.world_size % 4 != 0:
        pytest.skip("test requires a world size divisible by four")

    original_pad_for_alignment = None
    if expert_gtp_weight_remat_size > 1:
        from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
            GTP_CONFIG,
            update_gtp_config,
        )

        # Keep the all-ones assertion focused on topology rather than physical GTP padding.
        original_pad_for_alignment = GTP_CONFIG.pad_for_alignment
        update_gtp_config(pad_for_alignment=0)

    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
        reference_model = _build_tiny_moe_gpt(
            tensor_parallel_size=1, expert_parallel_size=1, expert_tensor_parallel_size=1, bf16=True
        )
        expected_numel = sum(param.numel() for param in reference_model.parameters())
        expected_norm = math.sqrt(expected_numel)
        del reference_model

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            expert_model_parallel_size=expert_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            gtp_remat_size=gtp_weight_remat_size,
            expert_gtp_remat_size=expert_gtp_weight_remat_size,
        )
        model = _build_tiny_moe_gpt(
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=expert_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            tensor_parallel_num_weight_shards=(tensor_parallel_size * gtp_weight_remat_size),
            expert_tensor_parallel_num_weight_shards=(
                expert_tensor_parallel_size * expert_gtp_weight_remat_size
            ),
            bf16=True,
        )
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, use_distributed_optimizer=use_distributed_optimizer
        )
        model = DistributedDataParallel(model.config, ddp_config, model)

        max_norm = expected_norm / 2.0
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=0.0,
                bf16=True,
                clip_grad=max_norm,
                log_num_zeros_in_grad=True,
                use_distributed_optimizer=use_distributed_optimizer,
            ),
            [model],
        )

        for param in model.parameters():
            assert hasattr(param, "main_grad")
            param.main_grad.zero_()

        found_inf = optimizer.prepare_grads()
        assert not found_inf
        assert optimizer.count_zeros() == expected_numel

        for param in model.parameters():
            param.main_grad.fill_(1.0)

        update_successful, actual_norm, actual_num_zeros = optimizer.step()

        assert update_successful
        assert actual_num_zeros == 0
        actual_norm_value = (
            actual_norm.item() if isinstance(actual_norm, torch.Tensor) else actual_norm
        )
        assert actual_norm_value == pytest.approx(expected_norm)

        expected_clip_coefficient = max_norm / (expected_norm + 1.0e-6)
        grads_checked = 0
        for param in optimizer.get_parameters():
            if param.grad is None:
                continue
            torch.testing.assert_close(
                param.grad,
                torch.full_like(param.grad, expected_clip_coefficient),
                rtol=1.0e-5,
                atol=1.0e-6,
            )
            grads_checked += 1
        assert grads_checked > 0
    finally:
        if expert_gtp_weight_remat_size > 1:
            from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
                reset_gtp_state,
                update_gtp_config,
            )

            update_gtp_config(pad_for_alignment=original_pad_for_alignment)
            reset_gtp_state()
        Utils.destroy_model_parallel()


def test_layer_wise_muon_grad_norm_uses_expert_tp_group_for_row_parallel_bias():
    """LayerWise Muon must deduplicate replicated expert FC2 bias grads over ETP.

    With TP=2, EP=2, and ETP=1, every rank is ETP rank zero.  The two EP ranks own
    distinct row-parallel expert biases, so both gradients must contribute to the global
    norm. Falling back to the regular TP rank drops the expert on TP rank one and
    undercounts the squared norm by a factor of two.
    """
    from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
    from megatron.core.process_groups_config import ProcessGroupCollection

    if Utils.world_size < 4 or Utils.world_size % 4 != 0:
        pytest.skip("test requires a world size divisible by four")

    tensor_parallel_size = 2
    expert_parallel_size = 2
    expert_tensor_parallel_size = 1

    try:
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            expert_model_parallel_size=expert_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
        )
        model = _build_tiny_moe_gpt(
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=expert_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            bf16=True,
            add_bias_linear=True,
        )

        expert_fc2_biases = [
            param
            for name, param in model.named_parameters()
            if ".experts." in name and ".linear_fc2.bias" in name
        ]
        assert len(expert_fc2_biases) == model.config.num_moe_experts // expert_parallel_size
        for parameter in expert_fc2_biases:
            assert parameter.ndim == 1
            assert parameter.allreduce is False
            assert parameter.tensor_model_parallel is False

        model = DistributedDataParallel(
            model.config, DistributedDataParallelConfig(use_distributed_optimizer=False), model
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="muon",
                lr=0.0,
                weight_decay=0.0,
                bf16=True,
                use_distributed_optimizer=False,
                use_layer_wise_distributed_optimizer=True,
                muon_tp_mode="duplicated",
            ),
            [model],
            use_gloo_process_groups=False,
            pg_collection=pg_collection,
        )

        assert isinstance(optimizer, LayerWiseDistributedOptimizer)
        assert pg_collection.tp.size() == tensor_parallel_size
        assert pg_collection.expt_tp.size() == expert_tensor_parallel_size

        for parameter in model.parameters():
            parameter.main_grad.zero_()
        for parameter in expert_fc2_biases:
            parameter.main_grad.fill_(1.0)
        assert optimizer.prepare_grads() is False

        actual_norm = optimizer.get_grad_norm()
        actual_norm_value = (
            actual_norm.item() if isinstance(actual_norm, torch.Tensor) else actual_norm
        )
        expected_norm = math.sqrt(model.config.num_moe_experts * model.config.hidden_size)

        assert actual_norm_value == pytest.approx(expected_norm)
    finally:
        Utils.destroy_model_parallel()
