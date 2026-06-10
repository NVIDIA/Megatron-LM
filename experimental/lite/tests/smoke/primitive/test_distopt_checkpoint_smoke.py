from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Replicate, Shard

from megatron.core.dist_checkpointing import load_plain_tensors
from megatron.core.dist_checkpointing.dict_utils import diff
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from megatron.lite.primitive.ckpt import attach_model_sharded_state_dict
from megatron.lite.primitive.optimizers.megatron_wrap import build_mc_stack
from megatron.lite.primitive.parallel import ParallelState, init_parallel
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import OptimizerConfig as LiteOptimizerConfig
from megatron.lite.runtime.contracts.config import ParallelConfig
from megatron.lite.runtime.contracts.handle import ModelHandle
from tests.unit_tests.test_utilities import Utils

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu, pytest.mark.distributed]


class TinyDense(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TinyTopologyAwareState(nn.Module):
    dense_shape = (8, 8)
    expert_shape = (8, 4)

    def __init__(self, ps: ParallelState):
        super().__init__()
        self.dense_weight = nn.Parameter(
            _local_shard(_global_tensor(self.dense_shape, 1.0), 0, ps.tp_rank, ps.tp_size)
            .cuda()
            .bfloat16()
        )
        self.experts_weight = nn.Parameter(
            _local_shard(_global_tensor(self.expert_shape, 101.0), 0, ps.etp_rank, ps.etp_size)
            .cuda()
            .bfloat16()
        )

    def forward(self, x):
        return x


@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_distopt():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for distopt smoke tests.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 8:
        pytest.skip("Megatron Lite smoke tests are capped at single-node 8 GPUs.")

    Utils.set_world_size(
        int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))
    )
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _global_tensor(shape: tuple[int, ...], offset: float) -> torch.Tensor:
    return torch.arange(offset, offset + int(torch.tensor(shape).prod().item())).reshape(shape)


def _local_shard(tensor: torch.Tensor, dim: int, rank: int, size: int) -> torch.Tensor:
    if size <= 1:
        return tensor.clone()
    chunks = torch.chunk(tensor, size, dim=dim)
    return chunks[rank].contiguous().clone()


def _topology_placements(name: str) -> list:
    if _is_expert_param(name):
        return [Replicate(), Replicate(), Replicate(), Shard(0)]
    return [Replicate(), Replicate(), Replicate(), Shard(0)]


def _is_expert_param(name: str) -> bool:
    return "experts" in name


def _build_sharded_model_and_distopt(parallel: ParallelConfig):
    ps = init_parallel(parallel)
    model = TinyTopologyAwareState(ps)
    model_cfg = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_experts=2,
        moe_intermediate_size=16,
        add_bias_linear=False,
    )
    engine_cfg = SimpleNamespace(
        model_name="tiny_topology_state",
        parallel=parallel,
        optimizer=LiteOptimizerConfig(optimizer="adam", lr=1.0e-3, weight_decay=0.0),
        deterministic=False,
    )
    wrapped_chunks, optimizer = build_mc_stack(
        [model], model_cfg=model_cfg, engine_cfg=engine_cfg, ps=ps, is_expert=_is_expert_param
    )
    _seed_optimizer_state(optimizer)
    attach_model_sharded_state_dict(
        wrapped_chunks, ps, get_placements=_topology_placements, is_expert=_is_expert_param
    )
    return wrapped_chunks, optimizer, ps


def _seed_optimizer_state(optimizer) -> None:
    for inner_optimizer in _inner_optimizers(optimizer):
        for group in inner_optimizer.param_groups:
            for param in group["params"]:
                state = inner_optimizer.state[param]
                base = param.detach().float().abs()
                state["exp_avg"] = (base + 0.125).to(dtype=param.dtype)
                state["exp_avg_sq"] = (base + 0.25).to(dtype=param.dtype)
    reload_model_params = getattr(optimizer, "reload_model_params", None)
    if callable(reload_model_params):
        reload_model_params()


def _inner_optimizers(optimizer):
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        for chained_optimizer in chained:
            yield from _inner_optimizers(chained_optimizer)
        return
    try:
        yield optimizer.optimizer
    except AttributeError:
        yield optimizer


def _distopt_handle(wrapped_chunks, optimizer, ps: ParallelState, parallel: ParallelConfig):
    return ModelHandle(
        model=wrapped_chunks,
        optimizer=optimizer,
        parallel_state=ps,
        config=SimpleNamespace(parallel=parallel),
        _extras={
            "model_chunks": wrapped_chunks,
            "protocol": SimpleNamespace(
                PLACEMENT_FN=_topology_placements, EXPERT_CLASSIFIER=_is_expert_param
            ),
        },
    )


def _build_model_and_distopt():
    torch.manual_seed(2468)
    model = TinyDense().bfloat16().cuda()
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=True)
    wrapped = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(optimizer="adam", lr=1.0e-3, bf16=True, use_distributed_optimizer=True),
        [wrapped],
    )
    attach_model_sharded_state_dict([wrapped], _single_node_parallel_state())
    return wrapped, optimizer


def _single_node_parallel_state() -> ParallelState:
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    return ParallelState(dp_size=world, dp_rank=rank, dp_cp_size=world, dp_cp_rank=rank)


def _shared_tmp_path(tmp_path) -> str:
    payload = [str(tmp_path) if torch.distributed.get_rank() == 0 else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def _train_step(model, optimizer, x: torch.Tensor):
    output = model(x)
    loss = output.float().square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if hasattr(model, "zero_grad_buffer"):
        model.zero_grad_buffer()
    return loss.detach()


def _local_named_params(model) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().float().clone() for name, param in model.named_parameters()}


def _assert_model_close(lhs, rhs):
    lhs_params = _local_named_params(lhs)
    rhs_params = _local_named_params(rhs)
    assert lhs_params.keys() == rhs_params.keys()
    for name in lhs_params:
        torch.testing.assert_close(lhs_params[name], rhs_params[name], atol=0.0, rtol=0.0)


def test_distopt_checkpoint_load_matches_uninterrupted_training_single_node(tmp_path):
    model_for_ckpt, optimizer_for_ckpt = _build_model_and_distopt()
    direct_model, direct_optimizer = _build_model_and_distopt()
    loaded_model, loaded_optimizer = _build_model_and_distopt()
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    torch.manual_seed(1357)
    x0 = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    x1 = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)

    _train_step(model_for_ckpt, optimizer_for_ckpt, x0)
    _train_step(direct_model, direct_optimizer, x0)
    checkpoint_dir = _shared_tmp_path(tmp_path)

    runtime.save_checkpoint(
        ModelHandle(
            model=model_for_ckpt,
            optimizer=optimizer_for_ckpt,
            _extras={"model_chunks": [model_for_ckpt]},
        ),
        checkpoint_dir,
        step=1,
    )
    assert (
        runtime.load_checkpoint(
            ModelHandle(
                model=loaded_model,
                optimizer=loaded_optimizer,
                _extras={"model_chunks": [loaded_model]},
            ),
            checkpoint_dir,
        )
        == 1
    )

    _train_step(direct_model, direct_optimizer, x1)
    _train_step(loaded_model, loaded_optimizer, x1)
    _assert_model_close(direct_model, loaded_model)


def test_distopt_checkpoint_reshards_from_pp_ep_to_tp_pp_ep_etp(tmp_path):
    if torch.distributed.get_world_size() < 8:
        pytest.skip("TP2/PP2/EP2/ETP2 distopt reshard smoke requires 8 GPUs.")

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    checkpoint_root = _shared_tmp_path(tmp_path)
    source_dir = os.path.join(checkpoint_root, "source")
    reserialized_dir = os.path.join(checkpoint_root, "reserialized")
    source_parallel = ParallelConfig(tp=1, ep=2, etp=1, pp=2, cp=1)
    target_parallel = ParallelConfig(tp=2, ep=2, etp=2, pp=2, cp=1)

    source_chunks, source_optimizer, source_ps = _build_sharded_model_and_distopt(source_parallel)
    runtime.save_checkpoint(
        _distopt_handle(source_chunks, source_optimizer, source_ps, source_parallel),
        source_dir,
        step=3,
        save_rng=False,
    )

    target_chunks, target_optimizer, target_ps = _build_sharded_model_and_distopt(target_parallel)
    assert (
        runtime.load_checkpoint(
            _distopt_handle(target_chunks, target_optimizer, target_ps, target_parallel),
            source_dir,
            load_rng=False,
        )
        == 3
    )
    runtime.save_checkpoint(
        _distopt_handle(target_chunks, target_optimizer, target_ps, target_parallel),
        reserialized_dir,
        step=3,
        save_rng=False,
    )

    plain_source = load_plain_tensors(os.path.join(source_dir, "step_3"))
    plain_reserialized = load_plain_tensors(os.path.join(reserialized_dir, "step_3"))
    diffs = diff(plain_source, plain_reserialized)
    assert not any(map(bool, diffs)), diffs
