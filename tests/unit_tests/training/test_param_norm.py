# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from types import SimpleNamespace

import pytest
import torch

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
    bf16: bool = False,
) -> GPTModel:
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=4,
        ffn_hidden_size=16,
        num_moe_experts=2,
        moe_ffn_hidden_size=16,
        moe_shared_expert_intermediate_size=16,
        moe_router_topk=1,
        moe_router_pre_softmax=True,
        tensor_model_parallel_size=tensor_parallel_size,
        expert_model_parallel_size=expert_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        sequence_parallel=tensor_parallel_size > 1,
        use_cpu_initialization=True,
        add_bias_linear=False,
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
    )
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


@pytest.mark.parametrize("use_distributed_optimizer", (False, True), ids=("optimizer", "distopt"))
@pytest.mark.parametrize(
    ("tensor_parallel_size", "expert_parallel_size", "expert_tensor_parallel_size"),
    ((2, 2, 1), (2, 1, 2), (4, 1, 2), (2, 1, 4)),
    ids=("expert-parallel", "expert-tensor-parallel", "tp-larger-than-etp", "etp-larger-than-tp"),
)
def test_moe_grad_norm_and_clipping_count_each_logical_gradient_once(
    tensor_parallel_size: int,
    expert_parallel_size: int,
    expert_tensor_parallel_size: int,
    use_distributed_optimizer: bool,
):
    """Gradient clipping should use each logical parameter's gradient exactly once."""
    if Utils.world_size < 4 or Utils.world_size % 4 != 0:
        pytest.skip("test requires a world size divisible by four")

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
        )
        model = _build_tiny_moe_gpt(
            tensor_parallel_size=tensor_parallel_size,
            expert_parallel_size=expert_parallel_size,
            expert_tensor_parallel_size=expert_tensor_parallel_size,
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
                use_distributed_optimizer=use_distributed_optimizer,
            ),
            [model],
        )

        for param in model.parameters():
            assert hasattr(param, "main_grad")
            param.main_grad.fill_(1.0)

        update_successful, actual_norm, _ = optimizer.step()

        assert update_successful
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
        Utils.destroy_model_parallel()
