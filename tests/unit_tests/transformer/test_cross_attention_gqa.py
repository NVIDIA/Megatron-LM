# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import CrossAttention, CrossAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def model_parallel_size(request):
    tp_size = request.param
    if Utils.world_size % tp_size != 0:
        pytest.skip(f"Tensor parallel size {tp_size} requires a divisible world size")
    Utils.initialize_model_parallel(tp_size, 1)
    model_parallel_cuda_manual_seed(123)
    original_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield tp_size
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_allow_tf32
        Utils.destroy_model_parallel()


def _dense_linear(inputs, weight, bias):
    # Native MCore linears add bias after the matrix product, including its
    # BF16 rounding, instead of using a fused addmm.
    return torch.matmul(inputs, weight.t()) + bias


def _dense_cross_attention(query_input, context_input, parameters, mask, num_query_groups):
    """Independent dense reference: each KV group stores its K head followed by V."""
    num_heads, head_dim = 8, 8
    query_length, batch_size, hidden_size = query_input.shape
    context_length = context_input.shape[0]
    query = _dense_linear(query_input, parameters["linear_q.weight"], parameters["linear_q.bias"])
    query = query.reshape(query_length, batch_size, num_heads, head_dim).permute(1, 2, 0, 3)
    key_value = _dense_linear(
        context_input, parameters["linear_kv.weight"], parameters["linear_kv.bias"]
    )
    key_value = key_value.reshape(context_length, batch_size, num_query_groups, 2, head_dim)
    key = key_value[:, :, :, 0, :].repeat_interleave(num_heads // num_query_groups, dim=2)
    value = key_value[:, :, :, 1, :].repeat_interleave(num_heads // num_query_groups, dim=2)
    key = key.permute(1, 2, 0, 3)
    value = value.permute(1, 2, 0, 3)
    # Accumulate the scaled score product in FP32, then round to the input dtype,
    # as the native attention's baddbmm does before its FP32 softmax.
    scores = (torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(head_dim)).to(
        query.dtype
    )
    probabilities = torch.softmax(scores.float().masked_fill(mask, -10000.0), dim=-1).to(
        query.dtype
    )
    context = torch.matmul(probabilities, value).permute(2, 0, 1, 3).contiguous()
    context = context.reshape(query_length, batch_size, hidden_size)
    return _dense_linear(context, parameters["linear_proj.weight"], parameters["linear_proj.bias"])


def _parameter_partition(tensor, name, tp_size, tp_rank):
    if name == "linear_proj.bias":
        return tensor
    partition_dim = 1 if name == "linear_proj.weight" else 0
    return tensor.chunk(tp_size, dim=partition_dim)[tp_rank]


def _check_dense_equivalence(
    tp_size, num_query_groups, dtype, recompute=False, sequence_parallel=False
):
    num_heads, head_dim, hidden_size = 8, 8, 64
    query_length, context_length, batch_size = 8, 12, 2
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    tp_rank = torch.distributed.get_rank(pg_collection.tp)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_query_groups=num_query_groups,
        kv_channels=head_dim,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=sequence_parallel,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        use_cpu_initialization=True,
        perform_initialization=False,
        gradient_accumulation_fusion=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=True,
        masked_softmax_fusion=False,
        attention_softmax_in_fp32=True,
        recompute_granularity="selective" if recompute else None,
    )
    attention = CrossAttention(
        config=config,
        submodules=CrossAttentionSubmodules(
            linear_q=ColumnParallelLinear,
            linear_kv=ColumnParallelLinear,
            core_attention=DotProductAttention,
            linear_proj=RowParallelLinear,
        ),
        layer_number=1,
        attn_mask_type=AttnMaskType.padding,
        pg_collection=pg_collection,
    ).cuda()
    attention.train()

    # Generate the same dense tensors on every rank, independently of Megatron's
    # rank-specific initialization. Random entries distinguish all heads, while
    # strictly positive biases catch missing or incorrectly gathered KV biases.
    generator = torch.Generator().manual_seed(317)

    def random_tensor(shape, bias=False):
        if bias:
            tensor = torch.rand(shape, generator=generator) * 0.16 + 0.04
        else:
            tensor = torch.randn(shape, generator=generator) * 0.1
        return tensor.cuda().to(dtype)

    parameter_shapes = {
        "linear_q.weight": (hidden_size, hidden_size),
        "linear_q.bias": (hidden_size,),
        "linear_kv.weight": (2 * num_query_groups * head_dim, hidden_size),
        "linear_kv.bias": (2 * num_query_groups * head_dim,),
        "linear_proj.weight": (hidden_size, hidden_size),
        "linear_proj.bias": (hidden_size,),
    }
    reference_parameters = {
        name: random_tensor(shape, bias=name.endswith("bias")).requires_grad_()
        for name, shape in parameter_shapes.items()
    }
    actual_parameters = dict(attention.named_parameters())
    assert actual_parameters.keys() == reference_parameters.keys()
    with torch.no_grad():
        for name, parameter in actual_parameters.items():
            parameter.copy_(
                _parameter_partition(reference_parameters[name], name, tp_size, tp_rank)
            )

    # Use unequal source and target lengths; both are divisible by every tested TP size.
    query = (random_tensor((query_length, batch_size, hidden_size)) * 5).requires_grad_()
    encoder = (random_tensor((context_length, batch_size, hidden_size)) * 5).requires_grad_()
    output_gradient = random_tensor(query.shape)
    mask = torch.zeros(batch_size, 1, query_length, context_length, dtype=torch.bool, device="cuda")
    mask[0, :, :, -3:] = True
    mask[1, :, :, -5:] = True

    def sequence_partition(tensor):
        return tensor.chunk(tp_size, dim=0)[tp_rank] if sequence_parallel else tensor

    actual_query = sequence_partition(query).detach().clone().requires_grad_()
    actual_encoder = sequence_partition(encoder).detach().clone().requires_grad_()
    core_calls = []
    hook = attention.core_attention.register_forward_hook(lambda *_: core_calls.append(None))
    try:
        output, output_bias = attention(actual_query, mask, key_value_states=actual_encoder)
        actual_output = output + output_bias
        expected_output = _dense_cross_attention(
            query, encoder, reference_parameters, mask, num_query_groups
        )
        actual_output.backward(sequence_partition(output_gradient))
        expected_output.backward(output_gradient)
    finally:
        hook.remove()

    if sequence_parallel:
        # RowParallelLinear's bias is replicated. As in finalize_model_grads,
        # combine its sequence-local contributions before comparing the full gradient.
        torch.distributed.all_reduce(attention.linear_proj.bias.grad, group=pg_collection.tp)

    assert len(core_calls) == (2 if recompute else 1)
    tolerance = (
        {"rtol": 3e-2, "atol": 1e-3} if dtype == torch.bfloat16 else {"rtol": 2e-5, "atol": 2e-6}
    )
    # Row-parallel BF16 projection rounds each partial GEMM before reducing,
    # unlike the full dense projection. Keep a separate output tolerance; the
    # input and parameter gradients below retain their tighter absolute bound.
    output_tolerance = dict(rtol=3e-2, atol=5e-3) if dtype == torch.bfloat16 else tolerance
    torch.testing.assert_close(
        actual_output, sequence_partition(expected_output), **output_tolerance
    )
    torch.testing.assert_close(actual_query.grad, sequence_partition(query.grad), **tolerance)
    torch.testing.assert_close(actual_encoder.grad, sequence_partition(encoder.grad), **tolerance)
    for name, parameter in actual_parameters.items():
        assert parameter.grad is not None, name
        assert reference_parameters[name].grad is not None, name
        expected_gradient = _parameter_partition(
            reference_parameters[name].grad, name, tp_size, tp_rank
        )
        gradient_tolerance = tolerance
        if dtype == torch.bfloat16 and num_query_groups < tp_size and name == "linear_kv.weight":
            # Shared KV gradients undergo an additional BF16 reduce-scatter before
            # the weight GEMM. Allow rounding near individual zero crossings while
            # bounding the whole gradient's error to detect missing contributions.
            gradient_tolerance = dict(rtol=3e-2, atol=3e-3)
            error = parameter.grad.float() - expected_gradient.float()
            assert error.norm() <= 0.02 * expected_gradient.float().norm(), name
        torch.testing.assert_close(
            parameter.grad,
            expected_gradient,
            msg=lambda message, name=name: f"{name}: {message}",
            **gradient_tolerance,
        )


@pytest.mark.parametrize("model_parallel_size", [1, 2, 4], indirect=True)
@pytest.mark.parametrize("num_query_groups", [8, 4, 2, 1])
def test_cross_attention_dense_equivalence(model_parallel_size, num_query_groups):
    _check_dense_equivalence(model_parallel_size, num_query_groups, torch.float32)


@pytest.mark.parametrize(
    "model_parallel_size,num_query_groups,dtype,recompute,sequence_parallel",
    [
        pytest.param(1, 1, torch.bfloat16, False, False, id="tp1-mqa-bf16"),
        pytest.param(2, 8, torch.bfloat16, False, False, id="tp2-mha-bf16"),
        pytest.param(4, 8, torch.bfloat16, False, False, id="tp4-mha-bf16"),
        pytest.param(2, 1, torch.bfloat16, False, False, id="tp2-mqa-bf16"),
        pytest.param(4, 2, torch.bfloat16, False, False, id="tp4-gqa-bf16"),
        pytest.param(2, 1, torch.float32, True, False, id="tp2-mqa-recompute"),
        pytest.param(4, 2, torch.float32, True, False, id="tp4-gqa-recompute"),
        pytest.param(2, 1, torch.float32, False, True, id="tp2-mqa-sp"),
        pytest.param(4, 2, torch.float32, False, True, id="tp4-gqa-sp"),
        pytest.param(2, 1, torch.bfloat16, True, True, id="tp2-mqa-bf16-recompute-sp"),
        pytest.param(4, 2, torch.bfloat16, True, True, id="tp4-gqa-bf16-recompute-sp"),
        pytest.param(4, 4, torch.float32, True, True, id="tp4-gqa-recompute-sp"),
    ],
    indirect=["model_parallel_size"],
)
def test_cross_attention_training_modes(
    model_parallel_size, num_query_groups, dtype, recompute, sequence_parallel
):
    _check_dense_equivalence(
        model_parallel_size, num_query_groups, dtype, recompute, sequence_parallel
    )
