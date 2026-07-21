"""Value-reconciliation and lifecycle tests for Megatron-LM issue #5790."""

import inspect
import os

import pytest
import torch
from torch import nn
from torch.optim import SGD, Adam

from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
    MixedPrecisionPolicy,
    fully_shard_model,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from tests.unit_tests.test_utilities import Utils

DP_SHARD = "dp_shard"
TP = "tp"
NO_SHARD = "no_shard"
OPTIM = "optim"
OPTIM_GRADS = "optim_grads"
OPTIM_GRADS_PARAMS = "optim_grads_params"


class TinyTiedEmbeddingLM(nn.Module):
    def __init__(
        self,
        vocab_size=16,
        hidden_size=8,
        *,
        device="cuda",
        dtype=torch.float32,
        tensor_parallel=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
        self.lm_head.weight = self.embedding.weight

        tied_weight = self.embedding.weight
        tied_weight.shared_embedding = True
        tied_weight.is_embedding_or_output_parameter = True
        if tensor_parallel:
            set_tensor_model_parallel_attributes(
                tensor=tied_weight, is_parallel=True, dim=0, stride=1
            )

    def forward(self, tokens):
        return self.lm_head(self.embedding(tokens))


class TinyTiedEmbeddingWithProjectionLM(nn.Module):
    def __init__(self, vocab_size=16, hidden_size=8, *, device="cuda", dtype=torch.float32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
        self.lm_head.weight = self.embedding.weight

        tied_weight = self.embedding.weight
        tied_weight.shared_embedding = True
        tied_weight.is_embedding_or_output_parameter = True

    def forward(self, tokens):
        return self.lm_head(self.proj(self.embedding(tokens)))


class TinyDistinctLinearPair(nn.Module):
    def __init__(self, hidden_size=8, *, device="cuda"):
        super().__init__()
        self.left = nn.Linear(hidden_size, hidden_size, bias=False, device=device)
        self.right = nn.Linear(hidden_size, hidden_size, bias=False, device=device)

    def forward(self, inputs):
        return self.right(self.left(inputs))


def test_static_targets_megatron_fsdp_v1_parameter_lifecycle():
    assert MegatronFSDP.__name__ == "MegatronFSDP"
    assert callable(fully_shard_optimizer)
    assert hasattr(MegatronFSDP, "_replace_param_with_distributed_if_needed")
    assert hasattr(MegatronFSDP, "_reestablish_shared_weights")
    assert hasattr(MegatronFSDP, "reregister_parameters")
    assert not hasattr(MegatronFSDP, "reapply_parameter_state")

    source = inspect.getsource(MegatronFSDP._replace_param_with_distributed_if_needed)
    assert "_reestablish_shared_weights" in source


def _require_torchrun_world_size(expected_world_size):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip(
            "Use torchrun so LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT are set."
        )
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != expected_world_size:
        pytest.skip(f"Requires WORLD_SIZE={expected_world_size}, got {world_size}.")
    if not torch.cuda.is_available():
        pytest.skip("Megatron-FSDP v1 tests require CUDA.")


def _initialize_mcore_parallel(tensor_model_parallel_size):
    Utils.set_world_size(
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ.get("LOCAL_RANK", os.environ["RANK"])),
    )
    Utils.initialize_model_parallel(tensor_model_parallel_size=tensor_model_parallel_size)


def _destroy_device_mesh(device_mesh):
    del device_mesh
    try:
        from torch.distributed.device_mesh import _mesh_resources

        _mesh_resources.child_to_root_mapping.clear()
        _mesh_resources.root_to_flatten_mapping.clear()
        _mesh_resources.mesh_stack.clear()
        _mesh_resources.mesh_dim_group_options.clear()
        _mesh_resources.flatten_name_to_root_dims.clear()
    except Exception:
        pass


def _init_device_mesh(dp_size, tp_size):
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=(DP_SHARD, TP))


def _wrap_tied_model(
    *,
    dp_size,
    tp_size,
    tensor_parallel=False,
    initial_weight=None,
    dtype=torch.float32,
    zero_dp_strategy=OPTIM_GRADS_PARAMS,
    mixed_precision_policy=MixedPrecisionPolicy(),
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    model = TinyTiedEmbeddingLM(device=device, dtype=dtype, tensor_parallel=tensor_parallel)
    if initial_weight is not None:
        with torch.no_grad():
            model.embedding.weight.fill_(initial_weight)
    device_mesh = _init_device_mesh(dp_size=dp_size, tp_size=tp_size)
    mfsdp_model = fully_shard_model(
        module=model,
        device_mesh=device_mesh,
        dp_shard_dim=DP_SHARD,
        tp_dim=TP,
        fsdp_unit_modules=[TinyTiedEmbeddingLM],
        zero_dp_strategy=zero_dp_strategy,
        mixed_precision_policy=mixed_precision_policy,
        sync_model_each_microbatch=True,
    )
    return mfsdp_model, device_mesh, device


def _wrap_projection_model(
    *,
    dp_size,
    tp_size,
    dtype=torch.float32,
    mixed_precision_policy=MixedPrecisionPolicy(
        main_params_dtype=torch.float32, main_grads_dtype=torch.float32
    ),
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    model = TinyTiedEmbeddingWithProjectionLM(device=device, dtype=dtype)
    device_mesh = _init_device_mesh(dp_size=dp_size, tp_size=tp_size)
    mfsdp_model = fully_shard_model(
        module=model,
        device_mesh=device_mesh,
        dp_shard_dim=DP_SHARD,
        tp_dim=TP,
        fsdp_unit_modules=[TinyTiedEmbeddingWithProjectionLM],
        zero_dp_strategy=OPTIM_GRADS_PARAMS,
        mixed_precision_policy=mixed_precision_policy,
        sync_model_each_microbatch=True,
    )
    return mfsdp_model, device_mesh, device


def _wrap_distinct_linear_pair(*, dp_size, tp_size):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    model = TinyDistinctLinearPair(device=device)
    device_mesh = _init_device_mesh(dp_size=dp_size, tp_size=tp_size)
    mfsdp_model = fully_shard_model(
        module=model,
        device_mesh=device_mesh,
        dp_shard_dim=DP_SHARD,
        tp_dim=TP,
        fsdp_unit_modules=[TinyDistinctLinearPair],
        zero_dp_strategy=OPTIM_GRADS_PARAMS,
        sync_model_each_microbatch=True,
    )
    return mfsdp_model, device_mesh, device


def _new_replacement_state(mfsdp_model, device):
    return {
        name: torch.randn(
            mfsdp_model._parameter_specs[canonical_name]["shape"],
            device=device,
            dtype=torch.float32,
        )
        for name, canonical_name in mfsdp_model._parameter_fqn_to_canonical.items()
    }


def _assign_new_tied_weight_to_current_module(mfsdp_model, device):
    module = mfsdp_model.module
    module.load_state_dict(_new_replacement_state(mfsdp_model, device), assign=True)
    module.to(device=device, dtype=torch.float32)
    module.lm_head.weight = module.embedding.weight
    return module.embedding.weight


def _assign_new_tied_weight(mfsdp_model, device):
    mfsdp_model._replace_param_with_raw_if_needed()
    return _assign_new_tied_weight_to_current_module(mfsdp_model, device)


def _post_accumulate_hooks(param):
    return getattr(param, "_post_accumulate_grad_hooks", None)


def _hook_count(param):
    hooks = _post_accumulate_hooks(param)
    return 0 if hooks is None else len(hooks)


def _local_tensor(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _full_tensor(tensor):
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


def _projection_patterns(device, dtype):
    embedding = torch.arange(16 * 8, device=device, dtype=torch.float32).reshape(16, 8)
    projection = 1000 + torch.arange(8 * 8, device=device, dtype=torch.float32).reshape(8, 8)
    return embedding.to(dtype=dtype), projection.to(dtype=dtype)


def _load_projection_replacement(mfsdp_model, embedding, projection):
    module = mfsdp_model.module
    result = module.load_state_dict(
        {
            "embedding.weight": embedding,
            "proj.weight": projection,
            "lm_head.weight": embedding.clone(),
        },
        assign=True,
    )
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    module.lm_head.weight = module.embedding.weight


def _parameter_buffer(mfsdp_model, parameter_name, buffer_name):
    raw_param = mfsdp_model.raw_param[parameter_name]
    pg_buffer = mfsdp_model.param_and_grad_buffer
    group = pg_buffer.parameter_groups[pg_buffer.param_to_param_group[raw_param]]
    buffer = getattr(group, buffer_name)
    if buffer is None:
        return None, None, None
    item_id = buffer.param_idx[raw_param]
    item = buffer.get_item(item_id, only_shard=buffer.is_data_distributed)
    return buffer, item_id, item


def _expected_local_buffer_item(buffer, item_id, full_value):
    if buffer.is_data_distributed:
        start, end = buffer._get_item_slice_in_shard(item_id)
    else:
        start, end = 0, full_value.numel()
    return full_value.flatten()[start:end].to(dtype=buffer.data.dtype)


def _buffer_data_snapshot(mfsdp_model):
    return {
        (group_id, buffer_name): buffer.data.detach().clone()
        for group_id, group in enumerate(mfsdp_model.param_and_grad_buffer.parameter_groups)
        for buffer_name in ("model_weight_buffer", "main_weight_buffer", "main_grad_buffer")
        if (buffer := getattr(group, buffer_name)) is not None
    }


def _assert_buffer_data_snapshot(mfsdp_model, expected):
    actual = _buffer_data_snapshot(mfsdp_model)
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def _materialize_raw_parameters(mfsdp_model):
    for group in mfsdp_model.param_and_grad_buffer.parameter_groups:
        buffer = group.model_weight_buffer
        assert buffer is not None
        bucket = buffer.fetch_bucket(set_param_data=True)
        if buffer.is_data_distributed:
            torch.distributed.all_gather_into_tensor(
                bucket.data, buffer.get_shard_from_local_buffer(), group=buffer.data_parallel_group
            )


def _tensor_observation(tensor):
    if tensor is None:
        return None
    observation = {
        "identity": id(tensor),
        "type": type(tensor).__name__,
        "global_shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "placements": tuple(str(placement) for placement in getattr(tensor, "placements", ())),
    }
    try:
        local = _local_tensor(tensor).detach()
        storage_nbytes = local.untyped_storage().nbytes()
        required_nbytes = (local.storage_offset() + local.numel()) * local.element_size()
        if storage_nbytes < required_nbytes:
            observation["unavailable"] = (
                f"storage is not materialized: {storage_nbytes} < {required_nbytes} bytes"
            )
            return observation
        observation.update(
            {
                "local_shape": tuple(local.shape),
                "stride": tuple(local.stride()),
                "checksum": float(local.double().sum().item()),
                "abs_checksum": float(local.double().abs().sum().item()),
            }
        )
    except RuntimeError as error:
        observation["unavailable"] = str(error)
    return observation


def _buffer_weight_item(mfsdp_model, buffer_name):
    return _parameter_buffer(mfsdp_model, "embedding.weight", buffer_name)[2]


def _record_value_stage(mfsdp_model, stage):
    distributed_param = dict(mfsdp_model.param_and_grad_buffer.optimizer_named_parameters)[
        "embedding.weight"
    ]
    snapshot = {
        "stage": stage,
        "rank": torch.distributed.get_rank(),
        "is_param_fsdp_distributed": mfsdp_model.is_param_fsdp_distributed,
        "module": _tensor_observation(mfsdp_model.module.embedding.weight),
        "raw": _tensor_observation(mfsdp_model.raw_param["embedding.weight"]),
        "optimizer_named_distributed": _tensor_observation(distributed_param),
        "model_weight": _tensor_observation(
            _buffer_weight_item(mfsdp_model, "model_weight_buffer")
        ),
        "main_weight": _tensor_observation(_buffer_weight_item(mfsdp_model, "main_weight_buffer")),
    }
    print(f"ISSUE_5790_VALUE_SNAPSHOT={snapshot}", flush=True)
    return snapshot


def _assert_constant(tensor, value):
    local = _local_tensor(tensor).detach()
    torch.testing.assert_close(local, torch.full_like(local, value), rtol=0, atol=0)


def _assert_authoritative_weight_buffers(mfsdp_model, value):
    model_weight = _buffer_weight_item(mfsdp_model, "model_weight_buffer")
    assert model_weight is not None
    _assert_constant(model_weight, value)

    main_weight = _buffer_weight_item(mfsdp_model, "main_weight_buffer")
    if main_weight is None:
        print("ISSUE_5790_MAIN_WEIGHT=N/A", flush=True)
    else:
        _assert_constant(main_weight, value)


def _preinitialize_reference_adam(optimizer, parameters):
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad()


def test_dp2_tp1_assign_values_survive_reregister_round_trip_forward_and_step():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(
            dp_size=2, tp_size=1, initial_weight=1.0
        )
        module = mfsdp_model.module
        owned_raw = mfsdp_model.raw_param["embedding.weight"]
        owned_distributed = dict(mfsdp_model.param_and_grad_buffer.optimizer_named_parameters)[
            "embedding.weight"
        ]
        ownership_before = _ownership_snapshot(mfsdp_model)
        _record_value_stage(mfsdp_model, "fully_shard_after_before_assign")
        _assert_authoritative_weight_buffers(mfsdp_model, 1.0)

        replacement_state = {
            name: torch.full(
                mfsdp_model._parameter_specs[canonical_name]["shape"],
                7.0,
                device=device,
                dtype=torch.float32,
            )
            for name, canonical_name in mfsdp_model._parameter_fqn_to_canonical.items()
        }
        assert set(replacement_state) == {"embedding.weight", "lm_head.weight"}
        load_result = module.load_state_dict(replacement_state, assign=True)
        assert load_result.missing_keys == []
        assert load_result.unexpected_keys == []
        module.lm_head.weight = module.embedding.weight
        _record_value_stage(mfsdp_model, "assign_after_before_reregister")
        _assert_constant(module.embedding.weight, 7.0)
        _assert_authoritative_weight_buffers(mfsdp_model, 1.0)

        mfsdp_model.reregister_parameters()
        _record_value_stage(mfsdp_model, "reregister_after")
        assert mfsdp_model.raw_param["embedding.weight"] is owned_raw
        assert mfsdp_model.module.embedding.weight is owned_distributed
        assert _ownership_snapshot(mfsdp_model) == ownership_before
        _assert_constant(mfsdp_model.module.embedding.weight, 7.0)
        _assert_authoritative_weight_buffers(mfsdp_model, 7.0)

        mfsdp_model._replace_param_with_raw_if_needed()
        _record_value_stage(mfsdp_model, "distributed_to_raw_after")
        _assert_authoritative_weight_buffers(mfsdp_model, 7.0)

        mfsdp_model._replace_param_with_distributed_if_needed()
        _record_value_stage(mfsdp_model, "raw_to_distributed_after")
        _assert_constant(mfsdp_model.module.embedding.weight, 7.0)
        _assert_authoritative_weight_buffers(mfsdp_model, 7.0)

        reference = TinyTiedEmbeddingLM(device=device)
        with torch.no_grad():
            reference.embedding.weight.fill_(7.0)
        reference_optimizer = Adam(reference.parameters(), lr=1.0e-2)
        _preinitialize_reference_adam(reference_optimizer, reference.parameters())

        optimizer = fully_shard_optimizer(Adam(mfsdp_model.parameters(), lr=1.0e-2))
        _record_value_stage(mfsdp_model, "fully_shard_optimizer_after")
        _assert_authoritative_weight_buffers(mfsdp_model, 7.0)

        tokens = torch.tensor([[0, 1, 2, 3]], device=device)
        expected_logits = torch.full((1, 4, 16), 392.0, device=device)
        logits = mfsdp_model(tokens)
        reference_logits = reference(tokens)
        _record_value_stage(mfsdp_model, "first_forward_after")
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)

        loss = logits.float().square().mean()
        reference_loss = reference_logits.float().square().mean()
        torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
        loss.backward()
        reference_loss.backward()
        _record_value_stage(mfsdp_model, "backward_after")

        optimizer.step()
        reference_optimizer.step()
        _record_value_stage(mfsdp_model, "first_optimizer_step_after")

        distributed_weight = dict(mfsdp_model.param_and_grad_buffer.optimizer_named_parameters)[
            "embedding.weight"
        ]
        updated_weight = (
            distributed_weight.full_tensor()
            if hasattr(distributed_weight, "full_tensor")
            else distributed_weight
        )
        torch.testing.assert_close(
            updated_weight, reference.embedding.weight, rtol=1.0e-6, atol=1.0e-6
        )
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def _assert_projection_buffer_shards(
    mfsdp_model, embedding, projection, main_embedding=None, main_projection=None
):
    model_values = {"embedding.weight": embedding, "proj.weight": projection}
    main_values = {
        "embedding.weight": embedding if main_embedding is None else main_embedding,
        "proj.weight": projection if main_projection is None else main_projection,
    }
    for parameter_name in ("embedding.weight", "proj.weight"):
        for buffer_name in ("model_weight_buffer", "main_weight_buffer"):
            buffer, item_id, actual = _parameter_buffer(mfsdp_model, parameter_name, buffer_name)
            assert buffer is not None
            expected_full = (
                main_values[parameter_name]
                if buffer_name == "main_weight_buffer"
                else model_values[parameter_name]
            )
            expected = _expected_local_buffer_item(buffer, item_id, expected_full)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dp2_tp1_nonuniform_multi_parameter_reconciliation_oracle():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_projection_model(dp_size=2, tp_size=1)
        assert mfsdp_model.is_param_fsdp_distributed is True
        owned_raw = dict(mfsdp_model.raw_param)
        owned_distributed = dict(mfsdp_model.param_and_grad_buffer.optimizer_named_parameters)
        ownership_before = _ownership_snapshot(mfsdp_model)
        embedding, projection = _projection_patterns(device, torch.float32)

        _load_projection_replacement(mfsdp_model, embedding, projection)
        assert mfsdp_model.is_param_fsdp_distributed is True
        assert mfsdp_model.module.embedding.weight is mfsdp_model.module.lm_head.weight
        assert mfsdp_model.module.proj.weight is not mfsdp_model.module.embedding.weight
        mfsdp_model.reregister_parameters()

        assert _ownership_snapshot(mfsdp_model) == ownership_before
        assert mfsdp_model.module.embedding.weight is owned_distributed["embedding.weight"]
        assert mfsdp_model.module.lm_head.weight is owned_distributed["embedding.weight"]
        assert mfsdp_model.module.proj.weight is owned_distributed["proj.weight"]
        _assert_projection_buffer_shards(mfsdp_model, embedding, projection)

        embedding_buffer, embedding_item_id, _ = _parameter_buffer(
            mfsdp_model, "embedding.weight", "model_weight_buffer"
        )
        projection_buffer, projection_item_id, _ = _parameter_buffer(
            mfsdp_model, "proj.weight", "model_weight_buffer"
        )
        if embedding_buffer is projection_buffer:
            assert embedding_item_id != projection_item_id

        for name, expected in (("embedding.weight", embedding), ("proj.weight", projection)):
            actual = _full_tensor(owned_distributed[name]).detach().reshape(expected.shape)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        mfsdp_model._replace_param_with_raw_if_needed()
        _materialize_raw_parameters(mfsdp_model)
        for name, expected in (("embedding.weight", embedding), ("proj.weight", projection)):
            torch.testing.assert_close(owned_raw[name], expected, rtol=0, atol=0)
        mfsdp_model._replace_param_with_distributed_if_needed()
        mfsdp_model._replace_param_with_raw_if_needed()
        _materialize_raw_parameters(mfsdp_model)
        for name, expected in (("embedding.weight", embedding), ("proj.weight", projection)):
            torch.testing.assert_close(owned_raw[name], expected, rtol=0, atol=0)
        mfsdp_model._replace_param_with_distributed_if_needed()

        reference = TinyTiedEmbeddingWithProjectionLM(device=device)
        with torch.no_grad():
            reference.embedding.weight.copy_(embedding)
            reference.proj.weight.copy_(projection)
        learning_rate = 0.25
        optimizer = fully_shard_optimizer(SGD(mfsdp_model.parameters(), lr=learning_rate))
        reference_optimizer = SGD(reference.parameters(), lr=learning_rate)
        tokens = torch.tensor([[0, 3, 7, 15]], device=device)

        logits = mfsdp_model(tokens)
        reference_logits = reference(tokens)
        torch.testing.assert_close(logits, reference_logits, rtol=1.0e-6, atol=1.0e-4)
        loss = logits.float().square().mean() * 1.0e-12
        reference_loss = reference_logits.float().square().mean() * 1.0e-12
        torch.testing.assert_close(loss, reference_loss, rtol=1.0e-6, atol=1.0e-9)
        loss.backward()
        reference_loss.backward()

        reference_parameters = dict(reference.named_parameters())
        distributed_parameters = dict(mfsdp_model.param_and_grad_buffer.optimizer_named_parameters)
        for name in ("embedding.weight", "proj.weight"):
            assert distributed_parameters[name].grad is not None
            actual_grad = _full_tensor(distributed_parameters[name].grad).detach()
            expected_grad = reference_parameters[name].grad
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2.0e-5, atol=2.0e-6)

        optimizer.step()
        reference_optimizer.step()
        for name in ("embedding.weight", "proj.weight"):
            actual = _full_tensor(distributed_parameters[name]).detach()
            torch.testing.assert_close(actual, reference_parameters[name], rtol=2.0e-5, atol=2.0e-5)
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_bf16_model_fp32_main_weight_reconciliation_oracle():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        policy = MixedPrecisionPolicy(
            main_params_dtype=torch.float32, main_grads_dtype=torch.float32
        )
        mfsdp_model, device_mesh, device = _wrap_projection_model(
            dp_size=2, tp_size=1, dtype=torch.bfloat16, mixed_precision_policy=policy
        )
        assert mfsdp_model.is_param_fsdp_distributed is True
        owned_raw = dict(mfsdp_model.raw_param)
        embedding, projection = _projection_patterns(device, torch.bfloat16)
        _load_projection_replacement(mfsdp_model, embedding, projection)
        mfsdp_model.reregister_parameters()
        _assert_projection_buffer_shards(mfsdp_model, embedding, projection)

        distributed_parameters = dict(mfsdp_model.param_and_grad_buffer.optimizer_named_parameters)
        for name, expected in (("embedding.weight", embedding), ("proj.weight", projection)):
            assert distributed_parameters[name].dtype == torch.float32
            actual_main = _full_tensor(distributed_parameters[name]).detach()
            torch.testing.assert_close(actual_main, expected.float(), rtol=0, atol=0)

        mfsdp_model._replace_param_with_raw_if_needed()
        _materialize_raw_parameters(mfsdp_model)
        for name, expected in (("embedding.weight", embedding), ("proj.weight", projection)):
            assert owned_raw[name].dtype == torch.bfloat16
            torch.testing.assert_close(owned_raw[name], expected, rtol=0, atol=0)
        mfsdp_model._replace_param_with_distributed_if_needed()
        mfsdp_model._replace_param_with_raw_if_needed()
        _materialize_raw_parameters(mfsdp_model)
        for name, expected in (("embedding.weight", embedding), ("proj.weight", projection)):
            torch.testing.assert_close(owned_raw[name], expected, rtol=0, atol=0)
        mfsdp_model._replace_param_with_distributed_if_needed()

        reference = TinyTiedEmbeddingWithProjectionLM(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            reference.embedding.weight.copy_(embedding)
            reference.proj.weight.copy_(projection)
        reference_parameters = dict(reference.named_parameters())
        main_parameters = {
            name: nn.Parameter(parameter.detach().float().clone())
            for name, parameter in (("embedding.weight", embedding), ("proj.weight", projection))
        }
        learning_rate = 0.25
        optimizer = fully_shard_optimizer(SGD(mfsdp_model.parameters(), lr=learning_rate))
        main_optimizer = SGD(main_parameters.values(), lr=learning_rate)
        tokens = torch.tensor([[0, 3, 7, 15]], device=device)

        logits = mfsdp_model(tokens)
        reference_logits = reference(tokens)
        torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)
        loss = logits.float().square().mean() * 1.0e-12
        reference_loss = reference_logits.float().square().mean() * 1.0e-12
        torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
        loss.backward()
        reference_loss.backward()

        for name in ("embedding.weight", "proj.weight"):
            assert distributed_parameters[name].grad is not None
            assert distributed_parameters[name].grad.dtype == torch.float32
            actual_grad = _full_tensor(distributed_parameters[name].grad).detach()
            expected_grad = reference_parameters[name].grad.float()
            torch.testing.assert_close(actual_grad, expected_grad, rtol=2.0e-2, atol=2.0e-3)
            main_parameters[name].grad = expected_grad.clone()

        optimizer.step()
        main_optimizer.step()
        for name in ("embedding.weight", "proj.weight"):
            actual_main = _full_tensor(distributed_parameters[name]).detach()
            torch.testing.assert_close(actual_main, main_parameters[name], rtol=2.0e-5, atol=2.0e-5)

        with torch.no_grad():
            reference.embedding.weight.copy_(main_parameters["embedding.weight"].to(torch.bfloat16))
            reference.proj.weight.copy_(main_parameters["proj.weight"].to(torch.bfloat16))
        updated_logits = mfsdp_model(tokens)
        updated_reference_logits = reference(tokens)
        torch.testing.assert_close(updated_logits, updated_reference_logits, rtol=0, atol=0)
        _assert_projection_buffer_shards(
            mfsdp_model,
            main_parameters["embedding.weight"].detach().to(torch.bfloat16),
            main_parameters["proj.weight"].detach().to(torch.bfloat16),
            main_embedding=main_parameters["embedding.weight"].detach(),
            main_projection=main_parameters["proj.weight"].detach(),
        )
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def _module_parameter_ids(mfsdp_model):
    return {
        name: id(param)
        for name, param in mfsdp_model.module.named_parameters(remove_duplicate=False)
    }


def _ownership_snapshot(mfsdp_model):
    pg_buffer = mfsdp_model.param_and_grad_buffer
    return {
        "raw_param": tuple(
            sorted((name, id(param)) for name, param in mfsdp_model.raw_param.items())
        ),
        "model_param_to_name": tuple(
            sorted((id(param), name) for param, name in mfsdp_model.param_to_name.items())
        ),
        "buffer_param_to_name": tuple(
            sorted((id(param), name) for param, name in pg_buffer.param_to_name.items())
        ),
        "buffer_params": tuple(id(param) for param in pg_buffer.params),
        "optimizer_named": tuple(
            (name, id(param)) for name, param in pg_buffer.optimizer_named_parameters
        ),
    }


def _install_external_tied_parameter(mfsdp_model, parameter):
    mfsdp_model.module.embedding.weight = parameter
    mfsdp_model.module.lm_head.weight = parameter


def test_dp2_tp1_reregister_rejects_extra_fqn():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        mfsdp_model.module.register_parameter(
            "extra_weight", nn.Parameter(torch.ones(1, device=device))
        )

        with pytest.raises(RuntimeError, match="FQN set mismatch.*extra"):
            mfsdp_model.reregister_parameters()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_reregister_rejects_missing_fqn():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, _ = _wrap_tied_model(dp_size=2, tp_size=1)
        mfsdp_model.module.embedding.register_parameter("weight", None)

        with pytest.raises(RuntimeError, match="FQN set mismatch.*missing"):
            mfsdp_model.reregister_parameters()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_reregister_rejects_original_tied_parameters_becoming_untied():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        mfsdp_model.module.load_state_dict(_new_replacement_state(mfsdp_model, device), assign=True)
        assert mfsdp_model.module.embedding.weight is not mfsdp_model.module.lm_head.weight

        with pytest.raises(RuntimeError, match="alias topology"):
            mfsdp_model.reregister_parameters()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_reregister_rejects_original_distinct_parameters_becoming_tied():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, _ = _wrap_distinct_linear_pair(dp_size=2, tp_size=1)
        assert mfsdp_model.module.left.weight is not mfsdp_model.module.right.weight
        mfsdp_model.module.right.weight = mfsdp_model.module.left.weight

        with pytest.raises(RuntimeError, match="alias topology"):
            mfsdp_model.reregister_parameters()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("mismatch", ["dtype", "device", "dtensor"])
def test_dp2_tp1_reregister_rejects_parameter_metadata_mismatch(mismatch):
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        shape = mfsdp_model._parameter_specs["embedding.weight"]["shape"]
        if mismatch == "dtype":
            replacement = nn.Parameter(torch.ones(shape, device=device, dtype=torch.float64))
            error_match = "dtype mismatch"
        elif mismatch == "device":
            replacement = nn.Parameter(torch.ones(shape, device="cpu", dtype=torch.float32))
            error_match = "device mismatch"
        else:
            from torch.distributed.tensor import Replicate, distribute_tensor

            replacement = nn.Parameter(
                distribute_tensor(
                    torch.ones(shape, device=device),
                    device_mesh=device_mesh,
                    placements=(Replicate(), Replicate()),
                )
            )
            error_match = "DTensor mesh mismatch"
        _install_external_tied_parameter(mfsdp_model, replacement)

        with pytest.raises(RuntimeError, match=error_match):
            mfsdp_model.reregister_parameters()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("sharding_strategy", [NO_SHARD, OPTIM, OPTIM_GRADS])
def test_dp2_tp1_unvalidated_sharding_strategies_fail_before_mutation(sharding_strategy):
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(
            dp_size=2, tp_size=1, zero_dp_strategy=sharding_strategy
        )
        shape = mfsdp_model._parameter_specs["embedding.weight"]["shape"]
        replacement = nn.Parameter(torch.full(shape, 9.0, device=device))
        _install_external_tied_parameter(mfsdp_model, replacement)
        buffers_before = _buffer_data_snapshot(mfsdp_model)
        module_ids_before = _module_parameter_ids(mfsdp_model)

        with pytest.raises(RuntimeError, match="only supports.*optim_grads_params"):
            mfsdp_model.reregister_parameters()

        _assert_buffer_data_snapshot(mfsdp_model, buffers_before)
        assert _module_parameter_ids(mfsdp_model) == module_ids_before
        assert mfsdp_model._fsdp_parameter_reconciliation_failed_reason is None
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_delayed_wgrad_storage_fails_before_mutation():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        mfsdp_model.raw_param["embedding.weight"].skip_backward_post_hook = True
        shape = mfsdp_model._parameter_specs["embedding.weight"]["shape"]
        replacement = nn.Parameter(torch.full(shape, 13.0, device=device))
        _install_external_tied_parameter(mfsdp_model, replacement)
        buffers_before = _buffer_data_snapshot(mfsdp_model)
        module_ids_before = _module_parameter_ids(mfsdp_model)

        with pytest.raises(RuntimeError, match="delayed-wgrad"):
            mfsdp_model.reregister_parameters()

        _assert_buffer_data_snapshot(mfsdp_model, buffers_before)
        assert _module_parameter_ids(mfsdp_model) == module_ids_before
        assert mfsdp_model._fsdp_parameter_reconciliation_failed_reason is None
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_asymmetric_validation_failure_reaches_consensus_before_copy():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        shape = mfsdp_model._parameter_specs["embedding.weight"]["shape"]
        replacement = nn.Parameter(torch.full(shape, 15.0, device=device))
        _install_external_tied_parameter(mfsdp_model, replacement)
        if torch.distributed.get_rank() == 0:
            mfsdp_model.module.register_parameter(
                "rank_zero_extra", nn.Parameter(torch.ones(1, device=device))
            )
        buffers_before = _buffer_data_snapshot(mfsdp_model)
        module_ids_before = _module_parameter_ids(mfsdp_model)

        with pytest.raises(RuntimeError, match="group rank 0:.*FQN set mismatch"):
            mfsdp_model.reregister_parameters()

        _assert_buffer_data_snapshot(mfsdp_model, buffers_before)
        assert _module_parameter_ids(mfsdp_model) == module_ids_before
        assert mfsdp_model._fsdp_parameter_reconciliation_failed_reason is None
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_asymmetric_valid_plans_reach_consensus_before_copy():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        if torch.distributed.get_rank() == 0:
            shape = mfsdp_model._parameter_specs["embedding.weight"]["shape"]
            replacement = nn.Parameter(torch.full(shape, 16.0, device=device))
            _install_external_tied_parameter(mfsdp_model, replacement)
        buffers_before = _buffer_data_snapshot(mfsdp_model)
        module_ids_before = _module_parameter_ids(mfsdp_model)

        with pytest.raises(RuntimeError, match="validation plans differ before mutation"):
            mfsdp_model.reregister_parameters()

        _assert_buffer_data_snapshot(mfsdp_model, buffers_before)
        assert _module_parameter_ids(mfsdp_model) == module_ids_before
        assert mfsdp_model._fsdp_parameter_reconciliation_failed_reason is None
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_validation_failure_is_transactional_and_later_forward_works():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        owned_module_parameters = dict(mfsdp_model.module.named_parameters(remove_duplicate=False))
        ownership_before = _ownership_snapshot(mfsdp_model)
        distributed_flag_before = mfsdp_model.is_param_fsdp_distributed
        shape = mfsdp_model._parameter_specs["embedding.weight"]["shape"]
        invalid = nn.Parameter(torch.ones(shape, device=device, dtype=torch.float64))
        _install_external_tied_parameter(mfsdp_model, invalid)
        module_ids_before_call = _module_parameter_ids(mfsdp_model)

        with pytest.raises(RuntimeError, match="dtype mismatch"):
            mfsdp_model.reregister_parameters()

        assert _module_parameter_ids(mfsdp_model) == module_ids_before_call
        assert mfsdp_model.is_param_fsdp_distributed is distributed_flag_before
        assert _ownership_snapshot(mfsdp_model) == ownership_before

        mfsdp_model.module.embedding._parameters["weight"] = owned_module_parameters[
            "embedding.weight"
        ]
        mfsdp_model.module.lm_head._parameters["weight"] = owned_module_parameters["lm_head.weight"]
        tokens = torch.tensor([[0, 1, 2, 3]], device=device)
        assert torch.isfinite(mfsdp_model(tokens)).all()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_partial_copy_failure_terminally_closes_wrapper(monkeypatch):
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        policy = MixedPrecisionPolicy(
            main_params_dtype=torch.float32, main_grads_dtype=torch.float32
        )
        mfsdp_model, device_mesh, device = _wrap_tied_model(
            dp_size=2, tp_size=1, dtype=torch.bfloat16, mixed_precision_policy=policy
        )
        raw_param = mfsdp_model.raw_param["embedding.weight"]
        distributed_param = dict(mfsdp_model.param_and_grad_buffer.optimizer_named_parameters)[
            "embedding.weight"
        ]
        model_buffer, model_item_id, model_item = _parameter_buffer(
            mfsdp_model, "embedding.weight", "model_weight_buffer"
        )
        main_buffer, _, main_item = _parameter_buffer(
            mfsdp_model, "embedding.weight", "main_weight_buffer"
        )
        _, _, main_grad_item = _parameter_buffer(
            mfsdp_model, "embedding.weight", "main_grad_buffer"
        )
        assert model_buffer is not None
        assert main_buffer is not None
        main_grad_item.fill_(17.0)
        model_before = model_item.detach().clone()
        main_before = main_item.detach().clone()
        main_grad_before = main_grad_item.detach().clone()
        _materialize_raw_parameters(mfsdp_model)
        raw_before = _local_tensor(raw_param).detach().clone()
        distributed_before = _full_tensor(distributed_param).detach().clone()

        shape = mfsdp_model._parameter_specs["embedding.weight"]["shape"]
        replacement_value = (
            torch.arange(torch.Size(shape).numel(), device=device, dtype=torch.float32)
            .reshape(shape)
            .to(torch.bfloat16)
        )
        replacement = nn.Parameter(replacement_value)
        _install_external_tied_parameter(mfsdp_model, replacement)
        module_ids_before_call = _module_parameter_ids(mfsdp_model)
        distributed_flag_before = mfsdp_model.is_param_fsdp_distributed
        ownership_before = _ownership_snapshot(mfsdp_model)

        def fail_main_weight_copy(_item_id, _item_data):
            raise RuntimeError("injected main-weight copy failure")

        monkeypatch.setattr(main_buffer, "set_item", fail_main_weight_copy)
        with pytest.raises(RuntimeError, match="copy phase"):
            mfsdp_model.reregister_parameters()

        expected_model = _expected_local_buffer_item(model_buffer, model_item_id, replacement_value)
        torch.testing.assert_close(model_item, expected_model, rtol=0, atol=0)
        assert not torch.equal(model_item, model_before)
        torch.testing.assert_close(main_item, main_before, rtol=0, atol=0)
        torch.testing.assert_close(main_grad_item, main_grad_before, rtol=0, atol=0)
        torch.testing.assert_close(_local_tensor(raw_param), raw_before, rtol=0, atol=0)
        torch.testing.assert_close(
            _full_tensor(distributed_param), distributed_before, rtol=0, atol=0
        )
        assert _module_parameter_ids(mfsdp_model) == module_ids_before_call
        assert mfsdp_model.is_param_fsdp_distributed is distributed_flag_before
        assert _ownership_snapshot(mfsdp_model) == ownership_before
        assert mfsdp_model._fsdp_parameter_reconciliation_failed_reason is not None

        with pytest.raises(RuntimeError, match="terminally closed"):
            mfsdp_model.reregister_parameters()
        tokens = torch.tensor([[0, 1, 2, 3]], device=device)
        with pytest.raises(RuntimeError, match="terminally closed"):
            mfsdp_model(tokens)
        owned_optimizer = SGD(
            [param for _, param in mfsdp_model.param_and_grad_buffer.optimizer_named_parameters],
            lr=0.1,
        )
        with pytest.raises(RuntimeError, match="terminally closed"):
            fully_shard_optimizer(owned_optimizer)
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_repeated_no_change_reregister_is_true_noop():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(
            dp_size=2, tp_size=1, initial_weight=3.0
        )
        raw_param = mfsdp_model.raw_param["embedding.weight"]
        dist_param = dict(mfsdp_model.param_and_grad_buffer.optimizer_named_parameters)[
            "embedding.weight"
        ]
        module_ids_before = _module_parameter_ids(mfsdp_model)
        ownership_before = _ownership_snapshot(mfsdp_model)
        hooks_before = tuple(
            sorted((name, id(handle)) for name, handle in mfsdp_model.grad_acc_hooks.items())
        )
        raw_hook_count_before = _hook_count(raw_param)
        values_before = _local_tensor(dist_param).detach().clone()

        assert mfsdp_model.reregister_parameters() is mfsdp_model
        assert mfsdp_model.reregister_parameters() is mfsdp_model

        assert _module_parameter_ids(mfsdp_model) == module_ids_before
        assert _ownership_snapshot(mfsdp_model) == ownership_before
        assert (
            tuple(sorted((name, id(handle)) for name, handle in mfsdp_model.grad_acc_hooks.items()))
            == hooks_before
        )
        assert _hook_count(raw_param) == raw_hook_count_before
        torch.testing.assert_close(_local_tensor(dist_param), values_before, rtol=0, atol=0)

        hook_executions = []
        user_handle = raw_param.register_post_accumulate_grad_hook(
            lambda _param: hook_executions.append(1)
        )
        tokens = torch.tensor([[0, 1, 2, 3]], device=device)
        mfsdp_model(tokens).float().sum().backward()
        assert len(hook_executions) == 1
        user_handle.remove()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_reregister_rejects_after_successful_optimizer_initialization():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, _ = _wrap_tied_model(dp_size=2, tp_size=1)
        fully_shard_optimizer(Adam(mfsdp_model.parameters(), lr=1.0e-2))
        assert mfsdp_model._fsdp_optimizer_initialization_attempted is True

        with pytest.raises(RuntimeError, match="optimizer"):
            mfsdp_model.reregister_parameters()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_partial_optimizer_initialization_closes_lifecycle(monkeypatch):
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, _ = _wrap_tied_model(dp_size=2, tp_size=1)
        optimizer = Adam(mfsdp_model.parameters(), lr=1.0e-2)

        def fail_optimizer_zero_grad():
            assert optimizer.state
            raise RuntimeError("injected optimizer initialization failure")

        monkeypatch.setattr(optimizer, "zero_grad", fail_optimizer_zero_grad)
        with pytest.raises(RuntimeError, match="injected optimizer initialization failure"):
            fully_shard_optimizer(optimizer)

        assert optimizer.state
        assert mfsdp_model._fsdp_optimizer_initialization_attempted is True
        with pytest.raises(RuntimeError, match="optimizer"):
            mfsdp_model.reregister_parameters()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_failed_first_forward_attempt_closes_lifecycle(monkeypatch):
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)

        def fail_forward(_tokens):
            raise RuntimeError("injected forward failure")

        monkeypatch.setattr(mfsdp_model.module, "forward", fail_forward)
        tokens = torch.tensor([[0, 1, 2, 3]], device=device)
        with pytest.raises(RuntimeError, match="injected forward failure"):
            mfsdp_model(tokens)

        assert mfsdp_model._fsdp_first_forward_attempted is True
        with pytest.raises(RuntimeError, match="forward attempt"):
            mfsdp_model.reregister_parameters()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_reregister_rejects_after_first_successful_forward():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        tokens = torch.tensor([[0, 1, 2, 3]], device=device)
        mfsdp_model(tokens)
        assert mfsdp_model._fsdp_first_forward_attempted is True

        with pytest.raises(RuntimeError, match="forward attempt"):
            mfsdp_model.reregister_parameters()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_tp2_dp2_tied_replacement_gradient_local_global_shape():
    _require_torchrun_world_size(4)
    _initialize_mcore_parallel(tensor_model_parallel_size=2)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(
            dp_size=2, tp_size=2, tensor_parallel=True
        )
        _assign_new_tied_weight(mfsdp_model, device)
        mfsdp_model.reregister_parameters()
        mfsdp_model._replace_param_with_distributed_if_needed()
        optimizer = fully_shard_optimizer(Adam(mfsdp_model.parameters(), lr=1.0e-2))
        tokens = torch.tensor([[0, 1, 2, 3]], device=device)

        loss = mfsdp_model(tokens).float().square().mean()
        loss.backward()
        optimizer.step()

        dist_param = dict(mfsdp_model.param_and_grad_buffer.optimizer_named_parameters)[
            "embedding.weight"
        ]
        grad = dist_param.grad
        assert grad is not None
        assert tuple(grad.shape) == tuple(dist_param.shape)

        local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
        local_param = dist_param.to_local() if hasattr(dist_param, "to_local") else dist_param
        assert tuple(local_grad.shape) == tuple(local_param.shape)
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()
