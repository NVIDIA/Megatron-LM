# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parameter/state association for distributed optimizer groups with mixed dtypes."""

import io
from copy import deepcopy

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _check_parameter_state_round_trip(model, optimizer):
    """Check shard association and checkpoint restore with controlled gradients only."""
    # Initialize Adam state through a real step without changing parameter values.
    for param in model.parameters():
        param.main_grad.zero_()
    assert optimizer.step()[0]

    for param in optimizer.model_param_gbuf_map:
        param_range = optimizer._get_model_param_range_map(param)["param"]
        state = optimizer._get_main_param_and_optimizer_states(param)
        expected = param.detach().view(-1)[param_range.start : param_range.end].float()
        torch.testing.assert_close(state["param"], expected, rtol=0, atol=0)

    def step():
        for index, param in enumerate(model.parameters()):
            param.main_grad.fill_(index + 1.0)
        assert optimizer.step()[0]

    def snapshot():
        return {
            param: {
                key: value.detach().clone()
                for key, value in optimizer._get_main_param_and_optimizer_states(param).items()
            }
            for param in optimizer.model_param_gbuf_map
        }

    step()
    first_step = snapshot()
    metadata = deepcopy(optimizer.state_dict())
    checkpoint = optimizer.get_parameter_state_dp_zero()
    if checkpoint is not None:
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
    step()
    second_step = snapshot()

    optimizer.load_state_dict(metadata)
    optimizer.load_parameter_state_from_dp_zero(checkpoint)
    for param, state in snapshot().items():
        for key, value in state.items():
            torch.testing.assert_close(value, first_step[param][key], rtol=0, atol=0)
    step()
    for param, state in snapshot().items():
        for key, value in state.items():
            torch.testing.assert_close(value, second_step[param][key], rtol=0, atol=0)


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("precision_aware", [False, True])
@pytest.mark.parametrize(
    "dtypes",
    [
        pytest.param((torch.bfloat16, torch.float32), id="bf16-first"),
        pytest.param((torch.float32, torch.bfloat16), id="fp32-first"),
        pytest.param((torch.bfloat16, torch.bfloat16), id="bf16-only"),
        pytest.param((torch.float32, torch.float32), id="fp32-only"),
    ],
)
def test_parameter_state_round_trip(dtypes, precision_aware):
    """Checkpoint state must stay with its model parameter after shard reordering."""
    Utils.initialize_model_parallel()
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model = torch.nn.Module()
        model.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.full((256, 128 * (index + 1)), index + 1.0, dtype=dtype, device="cuda")
                )
                for index, dtype in enumerate((*dtypes, dtypes[0]))
            ]
        )
        # The vector uses a separate weight-decay group, exercising group indices too.
        model.biases = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.full((1024,), 4.0, device="cuda"))]
        )
        ddp = DistributedDataParallel(
            TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=1, bf16=True),
            DistributedDataParallelConfig(use_distributed_optimizer=True),
            model,
            pg_collection=pg_collection,
        )
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=0.01,
                weight_decay=0.0,
                bf16=True,
                use_distributed_optimizer=True,
                use_precision_aware_optimizer=precision_aware,
                store_param_remainders=False,
                clip_grad=0.0,
            ),
            [ddp],
            use_gloo_process_groups=True,
        )
        if hasattr(optimizer, "chained_optimizers"):
            assert len(optimizer.chained_optimizers) == 1
            optimizer = optimizer.chained_optimizers[0]
        assert isinstance(optimizer, DistributedOptimizer)
        _check_parameter_state_round_trip(model, optimizer)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("precision_aware", [False, True])
def test_mhc_bf16_parameter_state_round_trip(precision_aware):
    """Exercise real mHC FP32 markers and BF16 conversion without a training loop."""
    Utils.initialize_model_parallel()
    try:
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            enable_mhc_connections=True,
            mhc_num_residual_streams=2,
            mhc_sinkhorn_iterations=3,
            use_fused_mhc=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            gradient_accumulation_fusion=False,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=8,
            hybrid_layer_pattern="*-",
            parallel_output=False,
        ).cuda()
        mhc_params = {
            param for layer in model.decoder.layers for param in layer.hyper_connection.parameters()
        }
        # HyperConnectionModule uses mark_keep_in_fp32; exercise the production markers.
        assert mhc_params and all(getattr(param, "keep_in_fp32", False) for param in mhc_params)
        mixed_precision_model = Float16Module(config, model)
        assert all(param.dtype == torch.float32 for param in mhc_params)
        assert model.embedding.word_embeddings.weight.dtype == torch.bfloat16
        assert all(
            param.dtype == torch.bfloat16
            for layer in model.decoder.layers
            for param in layer.inner_layer.parameters()
        )
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(use_distributed_optimizer=True, grad_reduce_in_fp32=True),
            mixed_precision_model,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        ddp.broadcast_params()
        # Assert the triggering buffer order instead of relying implicitly on traversal order.
        assert [buffer.param_dtype for buffer in ddp.buffers] == [torch.bfloat16, torch.float32]
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=0.001,
                weight_decay=0.0,
                bf16=True,
                use_distributed_optimizer=True,
                use_precision_aware_optimizer=precision_aware,
                store_param_remainders=False,
                clip_grad=0.0,
            ),
            [ddp],
            use_gloo_process_groups=True,
        )
        if hasattr(optimizer, "chained_optimizers"):
            assert len(optimizer.chained_optimizers) == 1
            optimizer = optimizer.chained_optimizers[0]
        assert isinstance(optimizer, DistributedOptimizer)
        has_mixed_mhc_group = any(
            any(param in mhc_params for param in group["params"])
            and any(param.dtype == torch.bfloat16 for param in group["params"])
            for group in optimizer.opt_group_ranges
        )
        # With more DP ranks, some ranks may not own any of the small mHC parameters.
        mixed_group_count = torch.tensor(int(has_mixed_mhc_group), device="cuda")
        torch.distributed.all_reduce(mixed_group_count, group=ddp.dp_cp_group)
        assert mixed_group_count.item() > 0, "No rank owns mHC and BF16 shards in the same group"

        _check_parameter_state_round_trip(model, optimizer)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("quantized_first", [True, False])
def test_split_state_dict_reads_the_checkpoint_buffer(quantized_first):
    """A checkpoint saved without FP8 parameter storage holds the matrix and the bias in one
    BF16 buffer; loading with FP8 storage splits it into the current FP8 and BF16 buffers.
    The source is read at its checkpoint index, also when the FP8 buffer now comes first."""
    from types import SimpleNamespace

    from megatron.core.optimizer.distrib_optimizer import _get_dtype_key

    # Registration order: matrix (index 0) then bias (index 1), 64 elements each.
    matrix = torch.zeros(8, 8, dtype=torch.bfloat16)
    bias = torch.zeros(64, dtype=torch.bfloat16)
    bf16_key, uint8_key = _get_dtype_key(torch.bfloat16), _get_dtype_key(torch.uint8)
    fp8_buffer = SimpleNamespace(params=[matrix], param_indices=[0])
    bf16_buffer = SimpleNamespace(params=[bias], param_indices=[1])
    buffers = [fp8_buffer, bf16_buffer] if quantized_first else [bf16_buffer, fp8_buffer]
    optimizer = SimpleNamespace(
        buffers=buffers,
        gbuf_ranges=[{uint8_key if b is fp8_buffer else bf16_key: None} for b in buffers],
        optimizer_state_keys=('exp_avg', 'exp_avg_sq'),
        _is_distopt_quantized_param=lambda param: param is matrix,
    )
    keys = ('param',) + optimizer.optimizer_state_keys
    # The saved buffer lays params out in reverse registration order: bias, then matrix.
    saved = {key: torch.arange(128, dtype=torch.float32) + 1000 * i for i, key in enumerate(keys)}
    state_dict = {'buckets_coalesced': True, 0: {bf16_key: dict(saved)}}

    DistributedOptimizer.split_state_dict_if_needed(optimizer, state_dict)

    fp8_idx = 0 if quantized_first else 1
    bf16_idx = 1 - fp8_idx
    assert set(state_dict) == {'buckets_coalesced', fp8_idx, bf16_idx}
    for key in keys:
        torch.testing.assert_close(state_dict[fp8_idx][uint8_key][key], saved[key][64:])
        torch.testing.assert_close(state_dict[bf16_idx][bf16_key][key], saved[key][:64])
    assert state_dict[fp8_idx][uint8_key]['numel_unpadded'] == 64
    assert state_dict[bf16_idx][bf16_key]['numel_unpadded'] == 64
