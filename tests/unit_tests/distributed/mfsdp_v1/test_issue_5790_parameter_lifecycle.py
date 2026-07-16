"""Baseline repro tests for Megatron-LM issue #5790.

These tests intentionally exercise post-wrap Parameter replacement on the
Megatron-FSDP v1 path. They should fail on the unmodified baseline until
Megatron-FSDP reestablishes owner metadata, distributed parameter mappings,
shared-weight markers, and gradient hooks on rebuilt Parameters.
"""

import inspect
import os
import pytest
import torch
from torch import nn
from torch.optim import Adam

from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
    fully_shard_model,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.megatron_fsdp import MegatronFSDP
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from tests.unit_tests.test_utilities import Utils


DP_SHARD = "dp_shard"
TP = "tp"
OPTIM_GRADS_PARAMS = "optim_grads_params"


class TinyTiedEmbeddingLM(nn.Module):
    def __init__(self, vocab_size=16, hidden_size=8, *, device="cuda", tensor_parallel=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, device=device)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device)
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
    def __init__(self, vocab_size=16, hidden_size=8, *, device="cuda"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, device=device)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, device=device)
        self.lm_head.weight = self.embedding.weight

        tied_weight = self.embedding.weight
        tied_weight.shared_embedding = True
        tied_weight.is_embedding_or_output_parameter = True

    def forward(self, tokens):
        return self.lm_head(self.proj(self.embedding(tokens)))


def test_static_targets_megatron_fsdp_v1_parameter_lifecycle():
    assert MegatronFSDP.__name__ == "MegatronFSDP"
    assert callable(fully_shard_optimizer)
    assert hasattr(MegatronFSDP, "_replace_param_with_distributed_if_needed")
    assert hasattr(MegatronFSDP, "_reestablish_shared_weights")
    assert hasattr(MegatronFSDP, "reregister_parameters")
    assert hasattr(MegatronFSDP, "reapply_parameter_state")

    source = inspect.getsource(MegatronFSDP._replace_param_with_distributed_if_needed)
    assert "_reestablish_shared_weights" in source


def _require_torchrun_world_size(expected_world_size):
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Use torchrun so LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT are set.")
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

    return init_device_mesh(
        "cuda",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=(DP_SHARD, TP),
    )


def _wrap_tied_model(*, dp_size, tp_size, tensor_parallel=False):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    model = TinyTiedEmbeddingLM(device=device, tensor_parallel=tensor_parallel)
    device_mesh = _init_device_mesh(dp_size=dp_size, tp_size=tp_size)
    mfsdp_model = fully_shard_model(
        module=model,
        device_mesh=device_mesh,
        dp_shard_dim=DP_SHARD,
        tp_dim=TP,
        fsdp_unit_modules=[TinyTiedEmbeddingLM],
        zero_dp_strategy=OPTIM_GRADS_PARAMS,
        sync_model_each_microbatch=True,
    )
    return mfsdp_model, device_mesh, device


def _wrap_tied_projection_model(*, dp_size, tp_size):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    model = TinyTiedEmbeddingWithProjectionLM(device=device)
    device_mesh = _init_device_mesh(dp_size=dp_size, tp_size=tp_size)
    mfsdp_model = fully_shard_model(
        module=model,
        device_mesh=device_mesh,
        dp_shard_dim=DP_SHARD,
        tp_dim=TP,
        fsdp_unit_modules=[TinyTiedEmbeddingWithProjectionLM],
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


def _assign_new_tied_weight_while_distributed(mfsdp_model, device):
    assert mfsdp_model.is_param_fsdp_distributed
    return _assign_new_tied_weight_to_current_module(mfsdp_model, device)


def _assign_tied_weight_only_while_distributed(mfsdp_model, device):
    assert mfsdp_model.is_param_fsdp_distributed
    module = mfsdp_model.module
    replacement = nn.Parameter(
        torch.randn(
            mfsdp_model._parameter_specs["embedding.weight"]["shape"],
            device=device,
            dtype=torch.float32,
        )
    )
    module.embedding.weight = replacement
    module.lm_head.weight = replacement
    return replacement


def _post_accumulate_hooks(param):
    return getattr(param, "_post_accumulate_grad_hooks", None)


def _hook_count(param):
    hooks = _post_accumulate_hooks(param)
    return 0 if hooks is None else len(hooks)


@pytest.mark.parametrize(
    "attr_name",
    [
        "_megatron_fsdp_model",
        "_is_shared",
        "orig_param",
        "megatron_fsdp_dist_index",
    ],
)
def test_dp2_tp1_tied_replacement_preserves_fsdp_parameter_metadata(attr_name):
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        replacement = _assign_new_tied_weight(mfsdp_model, device)
        mfsdp_model.reregister_parameters()

        mfsdp_model._replace_param_with_raw_if_needed()
        assert id(replacement) == id(mfsdp_model.module.lm_head.weight)
        assert hasattr(replacement, attr_name), f"replacement Parameter lost {attr_name}"
        if attr_name == "_megatron_fsdp_model":
            assert replacement._megatron_fsdp_model is mfsdp_model
        if attr_name == "_is_shared":
            assert replacement._is_shared is True
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_tied_replacement_updates_distributed_parameter_mapping():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        replacement = _assign_new_tied_weight(mfsdp_model, device)
        mfsdp_model.reregister_parameters()

        assert mfsdp_model.raw_param["embedding.weight"] is replacement
        assert replacement in mfsdp_model.param_and_grad_buffer.param_to_name
        assert mfsdp_model.param_and_grad_buffer.param_to_name[replacement] == "embedding.weight"
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_public_reregister_handles_distributed_state_replacement():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        old_raw = mfsdp_model.raw_param["embedding.weight"]
        replacement = _assign_new_tied_weight_while_distributed(mfsdp_model, device)

        mfsdp_model.reregister_parameters()

        assert mfsdp_model.raw_param["embedding.weight"] is replacement
        assert mfsdp_model.raw_param["embedding.weight"] is not old_raw
        mfsdp_model._replace_param_with_raw_if_needed()
        assert id(replacement) == id(mfsdp_model.module.lm_head.weight)
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_public_reregister_preserves_unchanged_raw_parameters_while_distributed():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_projection_model(dp_size=2, tp_size=1)
        old_projection_raw = mfsdp_model.raw_param["proj.weight"]
        projection_dist_param = dict(mfsdp_model.param_and_grad_buffer.optimizer_named_parameters)[
            "proj.weight"
        ]

        replacement = _assign_tied_weight_only_while_distributed(mfsdp_model, device)
        mfsdp_model.reregister_parameters()

        assert mfsdp_model.raw_param["embedding.weight"] is replacement
        assert mfsdp_model.raw_param["proj.weight"] is old_projection_raw
        assert mfsdp_model.raw_param["proj.weight"] is not projection_dist_param
        mfsdp_model._replace_param_with_raw_if_needed()
        assert mfsdp_model.module.proj.weight is old_projection_raw
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_tied_replacement_has_live_gradient_hook_on_new_parameter():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        replacement = _assign_new_tied_weight(mfsdp_model, device)
        user_handle = replacement.register_post_accumulate_grad_hook(lambda p: None)

        mfsdp_model.reregister_parameters()
        hooks_after_first = _post_accumulate_hooks(replacement)
        assert hooks_after_first is not None
        assert user_handle.id in hooks_after_first

        hook_count_after_first = _hook_count(replacement)
        mfsdp_model.reregister_parameters()
        assert _hook_count(replacement) == hook_count_after_first
        assert user_handle.id in _post_accumulate_hooks(replacement)
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_tied_replacement_runs_two_forward_backward_optimizer_steps():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        replacement = _assign_new_tied_weight(mfsdp_model, device)
        mfsdp_model.reregister_parameters()
        mfsdp_model._replace_param_with_distributed_if_needed()
        optimizer = fully_shard_optimizer(Adam(mfsdp_model.parameters(), lr=1.0e-2))
        tokens = torch.tensor([[0, 1, 2, 3]], device=device)

        for _ in range(2):
            loss = mfsdp_model(tokens).float().square().mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        mfsdp_model._replace_param_with_raw_if_needed()
        assert id(replacement) == id(mfsdp_model.module.lm_head.weight)
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


def test_dp2_tp1_strict_reregister_fails_when_alias_topology_changes():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        mfsdp_model._replace_param_with_raw_if_needed()
        module = mfsdp_model.module
        module.load_state_dict(
            {
                name: torch.randn(
                    mfsdp_model._parameter_specs[canonical_name]["shape"],
                    device=device,
                    dtype=torch.float32,
                )
                for name, canonical_name in mfsdp_model._parameter_fqn_to_canonical.items()
            },
            assign=True,
        )

        assert module.embedding.weight is not module.lm_head.weight
        with pytest.raises(RuntimeError, match="alias topology"):
            mfsdp_model.reregister_parameters(strict=True)
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_reregister_rejects_parameter_replacement_after_optimizer_creation():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        fully_shard_optimizer(Adam(mfsdp_model.parameters(), lr=1.0e-2))
        _assign_new_tied_weight(mfsdp_model, device)

        with pytest.raises(RuntimeError, match="optimizer"):
            mfsdp_model.reregister_parameters()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()


def test_dp2_tp1_reregister_rejects_distributed_state_replacement_after_optimizer_creation():
    _require_torchrun_world_size(2)
    _initialize_mcore_parallel(tensor_model_parallel_size=1)
    device_mesh = None
    try:
        mfsdp_model, device_mesh, device = _wrap_tied_model(dp_size=2, tp_size=1)
        fully_shard_optimizer(Adam(mfsdp_model.parameters(), lr=1.0e-2))
        _assign_new_tied_weight_while_distributed(mfsdp_model, device)

        with pytest.raises(RuntimeError, match="optimizer"):
            mfsdp_model.reregister_parameters()
    finally:
        if device_mesh is not None:
            _destroy_device_mesh(device_mesh)
        Utils.destroy_model_parallel()
