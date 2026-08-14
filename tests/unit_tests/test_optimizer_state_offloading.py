# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for OptimizerStateOffloader."""

import os
import sys
from functools import partial
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.fp8_utils import dequantize_fp8_tensor, is_float8tensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_te_min_version, unwrap_model
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.checkpointing import (
    _replace_fp8_skeleton_data_with_cpu_placeholders,
    _should_load_fp8_skeleton_on_cpu,
    load_checkpoint,
    save_checkpoint,
)
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import get_model, setup_model_and_optimizer
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
    initialize_gpt_model,
)
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch.fp8 import check_fp8_support
    from transformer_engine.pytorch.optimizers import FusedAdam  # noqa: F401

    FP8_AVAILABLE, REASON_FOR_NO_FP8 = check_fp8_support()
    TE_FUSED_ADAM_AVAILABLE = True
except ImportError:
    FP8_AVAILABLE = False
    REASON_FOR_NO_FP8 = "Transformer Engine FP8 support is not available"
    TE_FUSED_ADAM_AVAILABLE = False

_SEED = 1234
RUN_FP8_OFFLOAD_INTEGRATION = os.getenv("MEGATRON_RUN_FP8_OFFLOAD_INTEGRATION") == "1"


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class FakeFloat8Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class MockGroupedLinear(nn.Module):
    def __init__(self, num_gemms=2, single_grouped_weight=True, single_grouped_bias=True):
        super().__init__()
        self.num_gemms = num_gemms
        self.single_grouped_weight = single_grouped_weight
        self.single_grouped_bias = single_grouped_bias
        self.use_bias = True

    def _split_grouped_checkpoint_tensor(self, tensor, _key):
        return list(tensor.unbind(dim=0))


class MockGroupedModel(nn.Module):
    def __init__(self, grouped_module):
        super().__init__()
        self.experts = grouped_module


def create_model_and_optimizer(
    hidden_size=256, offload_optimizer_states=True, model_dtype=torch.bfloat16, **optimizer_kwargs
):
    """Helper to create model and optimizer for tests."""
    model = SimpleModel(hidden_size=hidden_size).to(dtype=model_dtype, device="cuda")
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )

    default_config = dict(
        optimizer='adam',
        bf16=model_dtype == torch.bfloat16,
        fp16=model_dtype == torch.float16,
        lr=0.001,
        use_distributed_optimizer=True,
        offload_optimizer_states=offload_optimizer_states,
    )
    default_config.update(optimizer_kwargs)

    optimizer_config = OptimizerConfig(**default_config)
    optim = get_megatron_optimizer(optimizer_config, [model])
    return model, optim


def get_single_distributed_optimizer(optim):
    """Return the distributed optimizer under the test wrapper."""
    return optim.chained_optimizers[0]


def run_forward_backward_step(model, optim, hidden_size=256, input_dtype=torch.bfloat16):
    """Run a single forward-backward-step cycle."""
    input_tensor = torch.randn(8, hidden_size, dtype=input_dtype, device='cuda')
    output = model(input_tensor)
    output.sum().backward()
    optim.step()
    optim.zero_grad()


def get_optimizer_memory_state(dist_optim):
    """Return deterministic optimizer residency bytes."""
    offloader = dist_optim._state_offloader
    if offloader is not None:
        return offloader._collect_memory_state()

    state_gpu_bytes = {"exp_avg": 0, "exp_avg_sq": 0, "master_param": 0}
    for param_state in dist_optim.optimizer.state.values():
        for key in state_gpu_bytes:
            tensor = param_state.get(key, None)
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                state_gpu_bytes[key] += tensor.untyped_storage().size()

    mcore_master_gpu_bytes = 0
    for group in dist_optim.shard_fp32_from_float16_groups:
        for tensor in group:
            mcore_master_gpu_bytes += tensor.untyped_storage().size()

    return {
        "state_gpu_exp_avg": state_gpu_bytes["exp_avg"],
        "state_gpu_exp_avg_sq": state_gpu_bytes["exp_avg_sq"],
        "state_gpu_master": state_gpu_bytes["master_param"],
        "state_cpu_exp_avg": 0,
        "state_cpu_exp_avg_sq": 0,
        "state_cpu_master": 0,
        "mcore_master_gpu": mcore_master_gpu_bytes,
        "mcore_master_cpu": 0,
    }


def get_model_param_snapshot(model):
    return {name: get_tensor_snapshot(param) for name, param in model.named_parameters()}


def get_tensor_snapshot(tensor):
    if is_float8tensor(tensor):
        return dequantize_fp8_tensor(tensor).detach().float().clone()
    return tensor.detach().float().clone()


def get_optimizer_state_snapshot(dist_optim):
    snapshot = []
    for group in dist_optim.optimizer.param_groups:
        group_state = {"step": group.get("step", None), "params": []}
        if isinstance(group_state["step"], torch.Tensor):
            group_state["step"] = group_state["step"].detach().clone()
        for param in group["params"]:
            param_state = dist_optim.optimizer.state.get(param, {})
            group_state["params"].append(
                {
                    key: value.detach().float().clone()
                    for key, value in param_state.items()
                    if key in ("exp_avg", "exp_avg_sq", "master_param")
                    and isinstance(value, torch.Tensor)
                }
            )
        snapshot.append(group_state)
    snapshot.append(
        {
            "mcore_master": [
                tensor.detach().float().clone()
                for group in dist_optim.shard_fp32_from_float16_groups
                for tensor in group
            ]
        }
    )
    return snapshot


def fp8_model_provider(pre_process=True, post_process=True, config=None, **_):
    model_parallel_cuda_manual_seed(_SEED)
    args = get_args()
    if config is None:
        config = core_transformer_config_from_args(args)
    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=args.vocal_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
    )


def destroy_fp8_test_state():
    Utils.destroy_model_parallel()
    destroy_global_vars()
    destroy_num_microbatches_calculator()


def create_fp8_test_args(
    mode,
    train_iters,
    load_main_params_from_ckpt=False,
    load=None,
    no_load_optim=None,
    fp8_param_gather=True,
):
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    sys.argv = ["test_optimizer_state_offloading.py"]
    args = parse_args()
    args.num_layers = 1
    args.vocal_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 64
    args.micro_batch_size = 1
    args.create_attention_mask_in_dataloader = True
    args.seq_length = 64
    args.tensor_model_parallel_size = 1
    args.sequence_parallel = False
    args.pipeline_model_parallel_size = 1
    args.context_parallel_size = 1
    args.train_iters = train_iters
    args.lr = 3e-5
    args.optimizer = "adam"
    args.neptune_project = ""
    args.wandb_project = ""
    args.tensorboard_dir = None
    args.bf16 = True
    args.add_bias_linear = False
    args.swiglu = True
    args.use_distributed_optimizer = True
    args.fp8 = "e4m3"
    args.fp8_recipe = "blockwise"
    args.fp8_param_gather = fp8_param_gather
    args.no_load_optim = load_main_params_from_ckpt if no_load_optim is None else no_load_optim
    args.load_main_params_from_ckpt = load_main_params_from_ckpt
    args.load = load
    args.use_precision_aware_optimizer = True
    args.main_grads_dtype = "fp32"
    args.main_params_dtype = "fp32"
    args.exp_avg_dtype = "bf16"
    args.exp_avg_sq_dtype = "bf16"
    args.offload_optimizer_states = mode != "baseline"
    if mode == "chunked_state_offload":
        args.offload_optimizer_states_chunk_numel = 1

    validate_args(args)
    set_global_variables(args, False)
    return args


def get_fp8_batch(seq_length, micro_batch_size):
    data = torch.arange(seq_length, dtype=torch.int64, device="cuda")
    input_ids = data.repeat((micro_batch_size, 1))
    labels = (data + 1).repeat((micro_batch_size, 1))
    position_ids = data.repeat((micro_batch_size, 1))
    attention_mask = torch.ones(
        (micro_batch_size, 1, seq_length, seq_length), dtype=bool, device="cuda"
    )
    loss_mask = torch.ones((micro_batch_size, seq_length), device="cuda")
    return input_ids, labels, position_ids, attention_mask, loss_mask


def get_high_precision_model_state_dict(model):
    return {
        name: get_tensor_snapshot(param).to(device=param.device)
        for name, param in model.named_parameters()
    }


def run_fp8_training_variant(mode, train_iters=1, load_main_params_from_ckpt=False):
    destroy_fp8_test_state()
    args = create_fp8_test_args(mode, train_iters, load_main_params_from_ckpt)
    set_args(args)
    torch.manual_seed(_SEED)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)

    gpt_model, optim, _ = setup_model_and_optimizer(
        fp8_model_provider, ModelType.encoder_or_decoder
    )
    assert len(gpt_model) == 1
    model = gpt_model[0]
    dist_optim = get_single_distributed_optimizer(optim)
    offloader = dist_optim._state_offloader

    num_fp8_params = sum(1 for _, param in model.named_parameters() if is_float8tensor(param))
    assert num_fp8_params > 0

    if load_main_params_from_ckpt:
        optim.reload_model_params(state_dict={"model": get_high_precision_model_state_dict(model)})
        torch.cuda.synchronize()

    input_ids, labels, position_ids, attention_mask, loss_mask = get_fp8_batch(
        args.seq_length, args.micro_batch_size
    )
    losses = []
    grad_norms = []

    for _ in range(train_iters):
        model.zero_grad_buffer()
        optim.zero_grad()
        model.set_is_first_microbatch()
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )
        loss = output.float().mean()
        loss.backward()

        if mode == "state_offload" and offloader._offloaded:
            dist_optim.reload_offloaded_states()

        update_successful, grad_norm, _ = optim.step()
        assert update_successful
        losses.append(loss.detach().float().clone())
        if isinstance(grad_norm, torch.Tensor):
            grad_norms.append(grad_norm.detach().float().clone())
        else:
            grad_norms.append(torch.tensor(grad_norm, dtype=torch.float32, device="cuda"))

        if mode == "state_offload":
            dist_optim.offload_states()
            dist_optim.release_offloaded_gpu_states()

    torch.cuda.synchronize()
    residency_after_step = get_optimizer_memory_state(dist_optim)
    moment_dtypes = {
        state[key].dtype
        for state in dist_optim.optimizer.state.values()
        for key in ("exp_avg", "exp_avg_sq")
        if key in state
    }
    offloaded_moment_dtypes = (
        {
            state[key].dtype
            for state in offloader._opt_state_cpu_buffers.values()
            for key in ("exp_avg", "exp_avg_sq")
            if key in state
        }
        if offloader is not None
        else set()
    )

    if offloader is not None and offloader._offloaded:
        dist_optim.get_parameter_state_dp_reshardable()

    torch.cuda.synchronize()
    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "model": get_model_param_snapshot(model),
        "optimizer": get_optimizer_state_snapshot(dist_optim),
        "residency_after_step": residency_after_step,
        "moment_dtypes": moment_dtypes,
        "offloaded_moment_dtypes": offloaded_moment_dtypes,
    }


def assert_optimizer_snapshots_close(actual, expected):
    assert len(actual) == len(expected)
    for actual_group, expected_group in zip(actual[:-1], expected[:-1]):
        if isinstance(expected_group["step"], torch.Tensor):
            torch.testing.assert_close(actual_group["step"], expected_group["step"])
        else:
            assert actual_group["step"] == expected_group["step"]
        assert len(actual_group["params"]) == len(expected_group["params"])
        for actual_param, expected_param in zip(actual_group["params"], expected_group["params"]):
            assert actual_param.keys() == expected_param.keys()
            for key in expected_param:
                torch.testing.assert_close(actual_param[key], expected_param[key])

    actual_mcore = actual[-1]["mcore_master"]
    expected_mcore = expected[-1]["mcore_master"]
    assert len(actual_mcore) == len(expected_mcore)
    for actual_tensor, expected_tensor in zip(actual_mcore, expected_mcore):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def assert_residency_offloaded(state):
    assert state["state_gpu_exp_avg"] == 0
    assert state["state_gpu_exp_avg_sq"] == 0
    assert state["mcore_master_gpu"] == 0
    assert state["state_cpu_exp_avg"] > 0
    assert state["state_cpu_exp_avg_sq"] > 0
    assert state["mcore_master_cpu"] > 0


def test_load_fp8_skeleton_on_cpu_gate():
    args = SimpleNamespace(
        offload_optimizer_states=True,
        fp8_param_gather=True,
        load_main_params_from_ckpt=True,
        no_load_optim=True,
        finetune=False,
    )
    assert _should_load_fp8_skeleton_on_cpu(args)

    args.no_load_optim = False
    assert not _should_load_fp8_skeleton_on_cpu(args)

    args.finetune = True
    assert _should_load_fp8_skeleton_on_cpu(args)


def test_replace_fp8_skeleton_data_with_cpu_placeholders(monkeypatch):
    fp8_data = FakeFloat8Tensor((2, 3), torch.bfloat16)
    sharded_state_dict = {
        "fp8": SimpleNamespace(data=fp8_data, dtype=fp8_data.dtype),
        "factory": SimpleNamespace(data=FakeFloat8Tensor((4,), torch.float32)),
        "plain": SimpleNamespace(data=torch.ones(1, device="cpu"), dtype=torch.float32),
    }

    monkeypatch.setattr(
        "megatron.training.checkpointing.is_float8tensor",
        lambda data: isinstance(data, FakeFloat8Tensor),
    )
    monkeypatch.setattr(
        "megatron.training.checkpointing.dequantize_fp8_tensor",
        lambda data: torch.full(tuple(data.shape), 3.0, dtype=data.dtype),
    )

    _replace_fp8_skeleton_data_with_cpu_placeholders(sharded_state_dict)

    fp8_placeholder = sharded_state_dict["fp8"].data
    assert isinstance(fp8_placeholder, torch.Tensor)
    assert fp8_placeholder.device.type == "cpu"
    assert fp8_placeholder.shape == fp8_data.shape
    assert fp8_placeholder.dtype == fp8_data.dtype
    assert sharded_state_dict["fp8"].dtype == fp8_data.dtype
    # The placeholder must hold the dequantized current weights, not
    # uninitialized memory: under non-strict loading, keys the checkpoint does
    # not overwrite keep these values, and torch.empty garbage here previously
    # caused loss spikes after finetune loads.
    torch.testing.assert_close(fp8_placeholder, torch.full((2, 3), 3.0, dtype=torch.bfloat16))
    factory_placeholder = sharded_state_dict["factory"].data
    assert factory_placeholder.device.type == "cpu"
    assert factory_placeholder.dtype == torch.float32
    torch.testing.assert_close(factory_placeholder, torch.full((4,), 3.0, dtype=torch.float32))
    torch.testing.assert_close(sharded_state_dict["plain"].data, torch.ones(1))


def test_chained_optimizer_split_state_dict_single_model_chunk():
    class MockOptimizer:
        def __init__(self, model_chunk):
            self.config = None
            self.model_chunks = [model_chunk]
            self.reload_state_dict = None

        def reload_model_params(self, state_dict=None):
            self.reload_state_dict = state_dict

    model_chunk = object()
    optimizer_1 = MockOptimizer(model_chunk)
    optimizer_2 = MockOptimizer(model_chunk)
    chained_optimizer = ChainedOptimizer([optimizer_1, optimizer_2])

    state_dict = {"weight": torch.ones(1)}
    chained_optimizer.reload_model_params(state_dict)

    assert optimizer_1.reload_state_dict is state_dict
    assert optimizer_2.reload_state_dict is state_dict


def test_normalize_grouped_state_dict_stacks_indexed_tensors():
    model = MockGroupedModel(
        MockGroupedLinear(single_grouped_weight=True, single_grouped_bias=True)
    )
    state_dict = {
        "decoder.experts.weight0": torch.ones(2, 3),
        "decoder.experts.weight1": torch.full((2, 3), 2.0),
        "decoder.experts.bias0": torch.ones(2),
        "decoder.experts.bias1": torch.full((2,), 2.0),
    }

    DistributedOptimizer._normalize_state_dict_for_grouped_params(state_dict, model)

    assert set(state_dict) == {"decoder.experts.weight", "decoder.experts.bias"}
    torch.testing.assert_close(
        state_dict["decoder.experts.weight"],
        torch.stack([torch.ones(2, 3), torch.full((2, 3), 2.0)], dim=0),
    )
    torch.testing.assert_close(
        state_dict["decoder.experts.bias"],
        torch.stack([torch.ones(2), torch.full((2,), 2.0)], dim=0),
    )


def test_normalize_grouped_state_dict_splits_grouped_tensors():
    model = MockGroupedModel(
        MockGroupedLinear(single_grouped_weight=False, single_grouped_bias=False)
    )
    state_dict = {
        "decoder.experts.weight": torch.stack([torch.ones(2, 3), torch.full((2, 3), 2.0)], dim=0),
        "decoder.experts.bias": torch.stack([torch.ones(2), torch.full((2,), 2.0)], dim=0),
    }

    DistributedOptimizer._normalize_state_dict_for_grouped_params(state_dict, model)

    assert set(state_dict) == {
        "decoder.experts.weight0",
        "decoder.experts.weight1",
        "decoder.experts.bias0",
        "decoder.experts.bias1",
    }
    torch.testing.assert_close(state_dict["decoder.experts.weight0"], torch.ones(2, 3))
    torch.testing.assert_close(state_dict["decoder.experts.weight1"], torch.full((2, 3), 2.0))
    torch.testing.assert_close(state_dict["decoder.experts.bias0"], torch.ones(2))
    torch.testing.assert_close(state_dict["decoder.experts.bias1"], torch.full((2,), 2.0))


def _load_checkpoint_no_arg_checks(*args, **kwargs):
    with mock.patch('megatron.training.checkpointing.check_checkpoint_args'):
        with mock.patch('megatron.training.checkpointing.update_num_microbatches'):
            return load_checkpoint(*args, **kwargs)


def _check_equal_dp_zero_state(state_a, state_b):
    if parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0:
        diffs = diff(state_a, state_b)
        is_equal = not any(map(bool, diffs))
    else:
        diffs = None
        is_equal = True

    all_equal = torch.tensor(int(is_equal), device='cuda')
    torch.distributed.all_reduce(all_equal, op=torch.distributed.ReduceOp.MIN)
    if not bool(all_equal.item()):
        raise RuntimeError(f'[{Utils.rank}] {diffs}')


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_chunked_offload_optimizer_save_load(tmp_path_dist_ckpt):
    """Offloaded optimizer states must survive an fs_model_space save/load round trip."""
    Utils.initialize_model_parallel()

    def assert_masters_offloaded(dist_optim):
        for group in dist_optim.shard_fp32_from_float16_groups:
            for tensor in group:
                assert tensor.untyped_storage().size() == 0

    def setup_offloaded_model_and_optimizer(seed, with_initialized_states):
        builder_args = parse_args(ignore_unknown_args=True)
        with mock.patch('megatron.training.training.get_args', new=lambda: builder_args):
            init_basic_mock_args(builder_args, tp=1, pp=1, bf16=True)
            model = get_model(
                partial(
                    initialize_gpt_model,
                    seed=seed,
                    tensor_model_parallel_size=1,
                    pipeline_model_parallel_size=1,
                    pipeline_dtype=torch.bfloat16,
                    bf16=True,
                )
            )
        config = OptimizerConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=True,
            offload_optimizer_states=True,
            offload_optimizer_states_chunk_numel=1,
        )
        optimizer = get_megatron_optimizer(config, model)

        if with_initialized_states:
            # Mimic mid-training state on the save side: Adam states exist and
            # everything is offloaded. The load side keeps the real startup state
            # (lazy Adam states not initialized yet, only masters offloaded).
            torch.manual_seed(seed + 1)
            model_parallel_cuda_manual_seed(seed + 1)
            for group in optimizer.optimizer.param_groups:
                for param in group['params']:
                    if len(optimizer.optimizer.state[param]) == 0:
                        optimizer.optimizer.state[param]['exp_avg'] = torch.rand_like(param.data)
                        optimizer.optimizer.state[param]['exp_avg_sq'] = torch.rand_like(param.data)

            dist_optim = optimizer.chained_optimizers[0]
            dist_optim._state_offloader.mark_optimizer_states_initialized()
            dist_optim.offload_states()
            dist_optim.release_offloaded_gpu_states()
            torch.cuda.synchronize()
        return unwrap_model(model), optimizer

    with TempNamedDir(
        tmp_path_dist_ckpt / 'test_chunked_offload_optimizer_save_load', sync=True
    ) as ckpt_dir:
        mock_args = parse_args(ignore_unknown_args=True)
        with mock.patch('megatron.training.checkpointing.get_args', new=lambda: mock_args):
            init_basic_mock_args(mock_args, tp=1, pp=1)
            init_checkpointing_mock_args(mock_args, ckpt_dir)
            mock_args.offload_optimizer_states = True

            model_a, optimizer_a = setup_offloaded_model_and_optimizer(
                seed=2, with_initialized_states=True
            )
            dist_optim_a = optimizer_a.chained_optimizers[0]
            assert_masters_offloaded(dist_optim_a)

            from megatron.training.training import preprocess_common_state_dict

            save_checkpoint(
                10,
                model_a,
                optimizer_a,
                None,
                0,
                preprocess_common_state_dict_fn=preprocess_common_state_dict,
            )
            state_a = dist_optim_a.get_parameter_state_dp_zero(use_gloo_comm=False)

            # Both save consumers above must be served from the offloader CPU copies:
            # nothing gets re-materialized on GPU by the save itself.
            assert_masters_offloaded(dist_optim_a)
            for param_state in dist_optim_a.optimizer.state.values():
                for key in ('exp_avg', 'exp_avg_sq'):
                    moment = param_state.get(key)
                    if isinstance(moment, torch.Tensor):
                        assert moment.untyped_storage().size() == 0

            model_b, optimizer_b = setup_offloaded_model_and_optimizer(
                seed=3, with_initialized_states=False
            )
            dist_optim_b = optimizer_b.chained_optimizers[0]
            assert_masters_offloaded(dist_optim_b)
            _load_checkpoint_no_arg_checks(model_b, optimizer_b, None)

            state_b = dist_optim_b.get_parameter_state_dp_zero(use_gloo_comm=False)
            _check_equal_dp_zero_state(state_a, state_b)

            # The loaded values must also survive a fresh offload/reload cycle, i.e. they
            # must land in the offloader CPU buffers, not only in the live GPU tensors.
            dist_optim_b.offload_states()
            dist_optim_b.release_offloaded_gpu_states()
            torch.cuda.synchronize()
            assert_masters_offloaded(dist_optim_b)
            state_b_after_offload = dist_optim_b.get_parameter_state_dp_zero(use_gloo_comm=False)
            _check_equal_dp_zero_state(state_a, state_b_after_offload)

    Utils.destroy_model_parallel()


def run_training_variant(mode, inputs, hidden_size=64):
    torch.manual_seed(1234)
    offload_optimizer_states = mode != "baseline"
    optimizer_kwargs = {"lr": 0.01}
    if mode == "chunked_state_offload":
        optimizer_kwargs["offload_optimizer_states_chunk_numel"] = 1
    model, optim = create_model_and_optimizer(
        hidden_size=hidden_size,
        offload_optimizer_states=offload_optimizer_states,
        **optimizer_kwargs,
    )
    dist_optim = get_single_distributed_optimizer(optim)
    losses = []

    for input_tensor in inputs:
        output = model(input_tensor.clone())
        loss = output.float().sum()
        loss.backward()

        if mode == "state_offload" and dist_optim._state_offloader._offloaded:
            dist_optim.reload_offloaded_states()

        optim.step()
        optim.zero_grad()
        losses.append(loss.detach().float().clone())

        if mode == "state_offload":
            dist_optim.offload_states()
            dist_optim.release_offloaded_gpu_states()

    torch.cuda.synchronize()
    residency_after_step = get_optimizer_memory_state(dist_optim)

    if dist_optim._state_offloader is not None and dist_optim._state_offloader._offloaded:
        dist_optim.get_parameter_state_dp_reshardable()

    torch.cuda.synchronize()
    return {
        "losses": losses,
        "model": get_model_param_snapshot(model),
        "optimizer": get_optimizer_state_snapshot(dist_optim),
        "residency_after_step": residency_after_step,
    }


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_baseline_full_offload_and_chunked_offload_e2e_equivalence():
    """Compare baseline, full offload, and chunked offload training semantics."""
    Utils.initialize_model_parallel()
    try:
        hidden_size = 64
        torch.manual_seed(5678)
        inputs = [
            torch.randn(8, hidden_size, dtype=torch.bfloat16, device='cuda') for _ in range(3)
        ]

        baseline = run_training_variant("baseline", inputs, hidden_size)
        state_offload = run_training_variant("state_offload", inputs, hidden_size)
        chunked = run_training_variant("chunked_state_offload", inputs, hidden_size)

        for variant in (state_offload, chunked):
            for actual_loss, expected_loss in zip(variant["losses"], baseline["losses"]):
                torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-4)
            assert variant["model"].keys() == baseline["model"].keys()
            for name in baseline["model"]:
                torch.testing.assert_close(
                    variant["model"][name],
                    baseline["model"][name],
                    rtol=1e-4,
                    atol=1e-3,
                    msg=f"model parameter {name} mismatch",
                )
            assert_optimizer_snapshots_close(variant["optimizer"], baseline["optimizer"])

        baseline_state = baseline["residency_after_step"]
        chunked_state = chunked["residency_after_step"]
        full_offload_state = state_offload["residency_after_step"]
        assert baseline_state["state_gpu_exp_avg"] > 0
        assert baseline_state["state_gpu_exp_avg_sq"] > 0
        assert baseline_state["mcore_master_gpu"] > 0
        assert_residency_offloaded(full_offload_state)
        assert_residency_offloaded(chunked_state)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
@pytest.mark.skipif(not FP8_AVAILABLE, reason=REASON_FOR_NO_FP8)
@pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
@pytest.mark.timeout(180)
def test_fp8_param_gather_chunked_offload_load_main_params_smoke():
    try:
        chunked = run_fp8_training_variant("chunked_state_offload", load_main_params_from_ckpt=True)

        assert len(chunked["losses"]) == 1
        assert len(chunked["grad_norms"]) == 1
        assert_residency_offloaded(chunked["residency_after_step"])
    finally:
        destroy_fp8_test_state()


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
@pytest.mark.skipif(not FP8_AVAILABLE, reason=REASON_FOR_NO_FP8)
@pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
@pytest.mark.skipif(
    not RUN_FP8_OFFLOAD_INTEGRATION,
    reason="Set MEGATRON_RUN_FP8_OFFLOAD_INTEGRATION=1 to run this FP8 integration test",
)
@pytest.mark.timeout(300)
def test_fp8_blockwise_full_offload_and_chunked_offload_integration():
    """Compare FP8 blockwise baseline, full offload, and chunked offload semantics."""
    try:
        baseline = run_fp8_training_variant("baseline")
        state_offload = run_fp8_training_variant("state_offload")
        chunked = run_fp8_training_variant("chunked_state_offload")

        for variant in (state_offload, chunked):
            for actual_loss, expected_loss in zip(variant["losses"], baseline["losses"]):
                torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-4, atol=1e-4)
            for actual_grad_norm, expected_grad_norm in zip(
                variant["grad_norms"], baseline["grad_norms"]
            ):
                torch.testing.assert_close(
                    actual_grad_norm, expected_grad_norm, rtol=1e-4, atol=1e-4
                )
            assert variant["model"].keys() == baseline["model"].keys()
            for name in baseline["model"]:
                torch.testing.assert_close(
                    variant["model"][name],
                    baseline["model"][name],
                    rtol=1e-3,
                    atol=1e-2,
                    msg=f"FP8 model parameter {name} mismatch",
                )
            assert_optimizer_snapshots_close(variant["optimizer"], baseline["optimizer"])

        baseline_state = baseline["residency_after_step"]
        full_offload_state = state_offload["residency_after_step"]
        chunked_state = chunked["residency_after_step"]
        assert baseline_state["state_gpu_exp_avg"] > 0
        assert baseline_state["state_gpu_exp_avg_sq"] > 0
        assert baseline_state["mcore_master_gpu"] > 0
        assert_residency_offloaded(full_offload_state)
        assert_residency_offloaded(chunked_state)
    finally:
        destroy_fp8_test_state()


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_offload_optimizer_states_checkpoint_without_optimizer_forces_main_params_from_ckpt():
    try:
        args = create_fp8_test_args(
            "state_offload",
            train_iters=1,
            load="/tmp/checkpoint",
            no_load_optim=True,
            fp8_param_gather=False,
        )
        assert args.load_main_params_from_ckpt
    finally:
        destroy_fp8_test_state()


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_offload_optimizer_states_checkpoint_with_optimizer_keeps_main_params_reload_off():
    try:
        args = create_fp8_test_args(
            "state_offload",
            train_iters=1,
            load="/tmp/checkpoint",
            no_load_optim=False,
            fp8_param_gather=False,
        )
        assert not args.load_main_params_from_ckpt
    finally:
        destroy_fp8_test_state()


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_chunked_offload_initializes_mcore_master_weights_on_cpu():
    """Check chunked offload starts with mcore master weights already offloaded."""
    Utils.initialize_model_parallel()
    try:
        model, optim = create_model_and_optimizer(
            hidden_size=64, offload_optimizer_states_chunk_numel=1
        )
        dist_optim = get_single_distributed_optimizer(optim)
        offloader = dist_optim._state_offloader

        state = offloader._collect_memory_state()
        assert offloader._offloaded is True
        assert offloader._offloaded_mcore_master_weights is True
        assert offloader._optimizer_states_initialized is False
        assert state["mcore_master_gpu"] == 0
        assert state["mcore_master_cpu"] > 0
        assert state["state_cpu_exp_avg"] == 0
        assert state["state_cpu_exp_avg_sq"] == 0

        original_cpu_buffers = [
            [tensor.clone() for tensor in group]
            for group in offloader._shard_fp32_from_float16_cpu_buffers
        ]
        for group in dist_optim.shard_fp32_from_float16_groups:
            for tensor in group:
                assert tensor.untyped_storage().size() == 0

        offloader.reload()
        offloader.sync_before_step()

        state = offloader._collect_memory_state()
        assert offloader._offloaded is False
        assert state["mcore_master_gpu"] == state["mcore_master_cpu"]
        for group_idx, group in enumerate(dist_optim.shard_fp32_from_float16_groups):
            for param_idx, tensor in enumerate(group):
                assert tensor.untyped_storage().size() > 0
                torch.testing.assert_close(tensor.cpu(), original_cpu_buffers[group_idx][param_idx])
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_chunked_offload_releases_and_restores_mcore_master_weights():
    """Check chunked residency and full offload idempotence after master release."""
    Utils.initialize_model_parallel()
    try:
        model, optim = create_model_and_optimizer(
            hidden_size=64, offload_optimizer_states_chunk_numel=1
        )
        dist_optim = get_single_distributed_optimizer(optim)
        offloader = dist_optim._state_offloader

        run_forward_backward_step(model, optim, hidden_size=64)

        state = offloader._collect_memory_state()
        assert offloader._offloaded is True
        assert offloader._offloaded_mcore_master_weights is True
        assert state["state_gpu_exp_avg"] == 0
        assert state["state_gpu_exp_avg_sq"] == 0
        assert state["mcore_master_gpu"] == 0
        assert state["state_cpu_exp_avg"] > 0
        assert state["state_cpu_exp_avg_sq"] > 0
        assert state["mcore_master_cpu"] > 0
        for group in dist_optim.shard_fp32_from_float16_groups:
            for tensor in group:
                assert tensor.untyped_storage().size() == 0

        offloader.offload()
        offloader.release_gpu_memory()
        torch.cuda.synchronize()

        state = offloader._collect_memory_state()
        assert state["state_gpu_exp_avg"] == 0
        assert state["state_gpu_exp_avg_sq"] == 0
        assert state["mcore_master_gpu"] == 0

        dist_optim.get_parameter_state_dp_reshardable()

        state = offloader._collect_memory_state()
        assert offloader._offloaded is False
        assert state["state_gpu_exp_avg"] > 0
        assert state["state_gpu_exp_avg_sq"] > 0
        assert state["mcore_master_gpu"] > 0
        for group in dist_optim.shard_fp32_from_float16_groups:
            for tensor in group:
                assert tensor.is_cuda
                assert tensor.untyped_storage().size() > 0
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_chunked_reload_offloaded_states_defers_mcore_master_weights():
    """Check grad-finalize reload hook does not full-load chunked master weights."""
    Utils.initialize_model_parallel()
    try:
        model, optim = create_model_and_optimizer(
            hidden_size=64, offload_optimizer_states_chunk_numel=1
        )
        dist_optim = get_single_distributed_optimizer(optim)
        offloader = dist_optim._state_offloader

        run_forward_backward_step(model, optim, hidden_size=64)

        state = offloader._collect_memory_state()
        assert state["mcore_master_gpu"] == 0
        assert state["mcore_master_cpu"] > 0
        assert offloader._offloaded_mcore_master_weights is True

        dist_optim.reload_offloaded_states()
        torch.cuda.synchronize()

        state = offloader._collect_memory_state()
        assert state["mcore_master_gpu"] == 0
        assert state["mcore_master_cpu"] > 0
        assert offloader._offloaded_mcore_master_weights is True
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_incremental_first_step_state_init_offloads_chunks():
    """Test that first-step lazy optimizer states can be initialized and offloaded in chunks."""
    Utils.initialize_model_parallel()
    model, optim = create_model_and_optimizer(offload_optimizer_states_chunk_numel=1)
    dist_optim = get_single_distributed_optimizer(optim)
    offloader = dist_optim._state_offloader

    assert offloader._optimizer_states_initialized is False

    run_forward_backward_step(model, optim)

    assert offloader._optimizer_states_initialized is True
    assert offloader._offloaded is True
    assert offloader._opt_state_cpu_buffers

    for state in offloader.adam_optimizer.state.values():
        for state_name in offloader.OPTIMIZER_STATE_KEYS:
            if state_name in state:
                assert state[state_name].untyped_storage().size() == 0

    dist_optim.get_parameter_state_dp_reshardable()

    assert offloader._offloaded is False

    for state in offloader.adam_optimizer.state.values():
        for state_name in offloader.OPTIMIZER_STATE_KEYS:
            if state_name in state:
                assert state[state_name].is_cuda
                assert state[state_name].untyped_storage().size() > 0

    Utils.destroy_model_parallel()


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_state_dict_syncs_pending_reload():
    """Test that checkpoint state reads synchronize an already-enqueued H2D reload."""
    Utils.initialize_model_parallel()
    model, optim = create_model_and_optimizer()
    dist_optim = get_single_distributed_optimizer(optim)
    offloader = dist_optim._state_offloader

    run_forward_backward_step(model, optim)

    offloader.offload()
    assert offloader._d2h_inflight is True
    offloader.release_gpu_memory()
    assert offloader._d2h_inflight is False

    offloader.reload()
    assert offloader._has_h2d_pending_work()

    dist_optim.get_parameter_state_dp_reshardable()

    assert not offloader._has_h2d_pending_work()
    Utils.destroy_model_parallel()


@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_fp16_overflow_skip_syncs_pending_reload():
    """Check skipped fp16 steps do not leave offload H2D reloads in flight."""
    Utils.initialize_model_parallel()
    try:
        hidden_size = 64
        model, optim = create_model_and_optimizer(
            hidden_size=hidden_size, model_dtype=torch.float16, initial_loss_scale=128.0
        )
        dist_optim = get_single_distributed_optimizer(optim)
        offloader = dist_optim._state_offloader

        run_forward_backward_step(model, optim, hidden_size=hidden_size, input_dtype=torch.float16)

        dist_optim.offload_states()
        dist_optim.release_offloaded_gpu_states()

        input_tensor = torch.randn(8, hidden_size, dtype=torch.float16, device='cuda')
        output = model(input_tensor)
        output.sum().backward()

        injected_inf = False
        for param in model.parameters():
            main_grad = getattr(param, "main_grad", None)
            if isinstance(main_grad, torch.Tensor):
                main_grad.fill_(float("inf"))
                injected_inf = True
        assert injected_inf

        dist_optim.reload_offloaded_states()
        assert offloader._has_h2d_pending_work()

        update_successful, _, _ = optim.step()

        assert update_successful is False
        assert not offloader._has_h2d_pending_work()

        dist_optim.offload_states()
        dist_optim.release_offloaded_gpu_states()
    finally:
        Utils.destroy_model_parallel()


# =============================================================================
# Test 1: Basic OptimizerStateOffloader Initialization
# =============================================================================
@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_offloader_initialization():
    """Test that OptimizerStateOffloader initializes correctly."""
    Utils.initialize_model_parallel()
    model, optim = create_model_and_optimizer()
    dist_optim = optim.chained_optimizers[0]

    # Offloader is created in __init__ when offload_optimizer_states=True
    assert dist_optim._state_offloader is not None
    offloader = dist_optim._state_offloader

    # Verify offloader properties
    assert offloader.adam_optimizer is not None
    assert offloader._d2h_stream is not None
    assert offloader._h2d_stream is not None
    assert offloader._offloaded is False

    # Before first step, optimizer states are not initialized yet
    assert offloader._optimizer_states_initialized is False

    # Run one step to initialize optimizer states
    run_forward_backward_step(model, optim)

    # After first step, optimizer states should be marked as initialized
    assert offloader._optimizer_states_initialized is True
    Utils.destroy_model_parallel()


# =============================================================================
# Test 2: Early Master Weight Offloading Before First Step
# =============================================================================
@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_early_master_weight_offloading():
    """Test that master weights can be offloaded before the first optimizer step."""
    Utils.initialize_model_parallel()
    model, optim = create_model_and_optimizer()
    dist_optim = optim.chained_optimizers[0]

    # Offloader is created in __init__
    assert dist_optim._state_offloader is not None
    offloader = dist_optim._state_offloader

    # Before first step, optimizer states are not initialized
    assert offloader._optimizer_states_initialized is False

    # Capture original master weights before offload
    original_master_weights = []
    for group in dist_optim.shard_fp32_from_float16_groups:
        group_weights = [tensor.clone() for tensor in group]
        original_master_weights.append(group_weights)

    # Offload before first step - should only offload master weights
    offloader.offload()
    offloader.release_gpu_memory()
    torch.cuda.synchronize()

    # Verify master weights were offloaded (storage resized to 0)
    for group in dist_optim.shard_fp32_from_float16_groups:
        for tensor in group:
            assert tensor.untyped_storage().size() == 0, "Master weight should be offloaded"

    # Reload master weights
    offloader.reload()
    offloader.sync_before_step()

    # Verify master weights match after reload
    for group_idx, group in enumerate(dist_optim.shard_fp32_from_float16_groups):
        for param_idx, tensor in enumerate(group):
            original = original_master_weights[group_idx][param_idx]
            torch.testing.assert_close(
                tensor,
                original,
                msg=f"Master weight [{group_idx}][{param_idx}] mismatch after offload/reload",
            )

    # Now run a step and verify optimizer states can be offloaded after
    run_forward_backward_step(model, optim)
    assert offloader._optimizer_states_initialized is True

    Utils.destroy_model_parallel()


# =============================================================================
# Test 3: Offload and Reload Correctness
# =============================================================================
@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
@pytest.mark.parametrize("offload_optimizer_states", [True, False])
@pytest.mark.parametrize("offload_master_weights", [True, False])
def test_offload_reload_correctness(offload_optimizer_states, offload_master_weights):
    """Test that offload/reload preserves optimizer state values."""
    if not offload_optimizer_states and not offload_master_weights:
        pytest.skip("At least one offload type required")

    Utils.initialize_model_parallel()
    model, optim = create_model_and_optimizer()
    dist_optim = optim.chained_optimizers[0]

    # Run steps to build up optimizer state
    for _ in range(3):
        run_forward_backward_step(model, optim)

    offloader = dist_optim._state_offloader

    # Capture original states before offload
    original_states = {}
    for param, state in offloader.adam_optimizer.state.items():
        original_states[param] = {
            k: v.clone() for k, v in state.items() if isinstance(v, torch.Tensor)
        }

    # Offload
    offloader.offload(
        offload_optimizer_states=offload_optimizer_states,
        offload_master_weights=offload_master_weights,
    )

    # Release GPU memory
    offloader.release_gpu_memory()
    torch.cuda.synchronize()

    # Reload
    offloader.reload()
    offloader.sync_before_step()

    # Verify states match after reload
    for param, state in offloader.adam_optimizer.state.items():
        if param in original_states:
            for key, original_tensor in original_states[param].items():
                if key in state and isinstance(state[key], torch.Tensor):
                    reloaded_tensor = state[key]
                    assert reloaded_tensor.device.type == 'cuda', f"State {key} should be on GPU"
                    torch.testing.assert_close(
                        reloaded_tensor,
                        original_tensor,
                        msg=f"State {key} mismatch after offload/reload",
                    )
    Utils.destroy_model_parallel()


# =============================================================================
# Test 4: GPU Memory Release Verification
# =============================================================================
@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_gpu_memory_release():
    """Test that GPU memory is actually freed after release_gpu_memory()."""
    Utils.initialize_model_parallel()
    # Use larger model for measurable memory impact
    model, optim = create_model_and_optimizer(hidden_size=1024)
    dist_optim = optim.chained_optimizers[0]

    # Initialize optimizer states
    run_forward_backward_step(model, optim, hidden_size=1024)

    offloader = dist_optim._state_offloader

    # Measure memory before offload
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    memory_before = torch.cuda.memory_allocated()

    # Offload and release
    offloader.offload()
    offloader.release_gpu_memory()

    # Wait for async operations
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    memory_after = torch.cuda.memory_allocated()

    # Memory should decrease
    memory_freed = memory_before - memory_after
    assert memory_freed > 0, f"Expected memory to be freed, but got {memory_freed} bytes difference"
    Utils.destroy_model_parallel()


# =============================================================================
# Test 5: Multiple Offload/Reload Cycles
# =============================================================================
@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_multiple_offload_reload_cycles():
    """Test that multiple offload/reload cycles work correctly."""
    Utils.initialize_model_parallel()
    model, optim = create_model_and_optimizer()
    dist_optim = optim.chained_optimizers[0]

    # Initialize
    run_forward_backward_step(model, optim)

    offloader = dist_optim._state_offloader

    # Run multiple cycles
    for cycle in range(5):
        # Offload
        offloader.offload()
        offloader.release_gpu_memory()

        # Reload
        offloader.reload()
        offloader.sync_before_step()

        # Run optimizer step
        run_forward_backward_step(model, optim)

    # Verify model can still produce valid outputs
    input_tensor = torch.randn(8, 256, dtype=torch.bfloat16, device='cuda')
    output = model(input_tensor)
    assert not output.isnan().any(), "Model output contains NaN after multiple cycles"
    Utils.destroy_model_parallel()


# =============================================================================
# Test 6: Training Correctness with Offloading
# =============================================================================
@pytest.mark.skipif(not TE_FUSED_ADAM_AVAILABLE, reason="Requires TE FusedAdam")
def test_training_correctness_with_offloading():
    """Test that training with offloading produces same results as without."""
    Utils.initialize_model_parallel()
    torch.manual_seed(42)

    # Model 1: with offloading
    model1, optim1 = create_model_and_optimizer(offload_optimizer_states=True, lr=0.01)

    # Model 2: without offloading (reference)
    torch.manual_seed(42)
    model2, optim2 = create_model_and_optimizer(offload_optimizer_states=False, lr=0.01)

    # Train both models
    n_steps = 10
    torch.manual_seed(123)
    dist_optim1 = optim1.chained_optimizers[0]

    # Offloader is created in __init__ when offload_optimizer_states=True
    assert dist_optim1._state_offloader is not None
    offloader = dist_optim1._state_offloader

    for step in range(n_steps):
        input_tensor = torch.randn(8, 256, dtype=torch.bfloat16, device='cuda')

        # Model 1 with offloading
        # Offload states (master weights can be offloaded from the start,
        # optimizer states will be skipped until after first step)
        offloader.offload()
        offloader.release_gpu_memory()

        output1 = model1(input_tensor)
        loss1 = output1.sum()
        loss1.backward()

        offloader.reload()
        offloader.sync_before_step()
        optim1.step()
        optim1.zero_grad()

        # Model 2 without offloading
        output2 = model2(input_tensor)
        loss2 = output2.sum()
        loss2.backward()
        optim2.step()
        optim2.zero_grad()

    # Compare final model weights
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        torch.testing.assert_close(
            p1.data,
            p2.data,
            atol=1e-5,
            rtol=1e-4,
            msg=f"Parameter {n1} mismatch between offloaded and non-offloaded training",
        )
    Utils.destroy_model_parallel()
