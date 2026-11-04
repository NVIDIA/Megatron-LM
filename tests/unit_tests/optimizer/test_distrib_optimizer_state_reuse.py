# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Ordinary Adam checkpoint loading must not rebuild compatible FP32 moments."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.validation import StrictHandling
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import USING_TE_OPTIMIZER, DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not USING_TE_OPTIMIZER,
    reason="requires CUDA and Transformer Engine FusedAdam",
)


class _CheckpointModel(torch.nn.Module):
    """Replicated mixed-dtype parameters with nonempty shards at any test world size."""

    def __init__(self):
        super().__init__()
        # Each rank owns at least 1M elements in each dtype buffer, so a duplicate
        # state copy is much larger than the allocation measurement's tolerance.
        shape = (Utils.world_size * 1024, 1024)
        self.weight = torch.nn.Parameter(
            torch.full(shape, 0.25, device="cuda", dtype=torch.bfloat16)
        )
        self.scale = torch.nn.Parameter(torch.full(shape, 0.5, device="cuda"))
        self.bias = torch.nn.Parameter(torch.full((128,), 0.125, device="cuda"))
        self.config = TransformerConfig(
            num_layers=1, hidden_size=128, num_attention_heads=1, bf16=True
        )

    def sharded_state_dict(self):
        """Describe parameters replicated over the data-parallel group."""
        return {
            name: ShardedTensor.from_rank_offsets(name, param, replica_id=(0, 0, Utils.rank))
            for name, param in self.state_dict(keep_vars=True).items()
        }


@pytest.fixture
def model_parallel():
    """Use the entire torchrun world for DP, with TP and PP fixed to one."""
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _build_optimizer():
    model = _CheckpointModel()
    ddp = DistributedDataParallel(
        model.config,
        DistributedDataParallelConfig(use_distributed_optimizer=True),
        model,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=False,
            clip_grad=0.0,
        ),
        [ddp],
        use_gloo_process_groups=True,
    )
    if hasattr(optimizer, "chained_optimizers"):
        assert len(optimizer.chained_optimizers) == 1
        optimizer = optimizer.chained_optimizers[0]
    assert isinstance(optimizer, DistributedOptimizer)
    assert not optimizer.optimizer.state
    return model, optimizer


def _step(model, optimizer, iteration):
    optimizer.zero_grad()
    for index, param in enumerate(model.parameters()):
        param.main_grad.fill_(0.01 * (index + 1) + 0.001 * iteration)
    assert optimizer.step()[0]


def _moment_pointers(optimizer):
    pointers = {
        param: {name: value.data_ptr() for name, value in state.items()}
        for param, state in optimizer.optimizer.state.items()
    }
    assert pointers and all(pointers.values())
    return pointers


def _moments_on_cpu(optimizer):
    return {
        param: {name: value.detach().cpu().clone() for name, value in state.items()}
        for param, state in optimizer.optimizer.state.items()
    }


def _assert_optimizers_equal(left, right):
    for left_group, right_group in zip(
        left.optimizer.param_groups, right.optimizer.param_groups, strict=True
    ):
        assert {k: v for k, v in left_group.items() if k != "params"} == {
            k: v for k, v in right_group.items() if k != "params"
        }
        for left_param, right_param in zip(
            left_group["params"], right_group["params"], strict=True
        ):
            torch.testing.assert_close(left_param, right_param, rtol=0, atol=0)
            left_state = left.optimizer.state[left_param]
            right_state = right.optimizer.state[right_param]
            assert left_state.keys() == right_state.keys()
            for name, value in left_state.items():
                torch.testing.assert_close(value, right_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("initialized", [False, True])
def test_metadata_restore_does_not_duplicate_moments(model_parallel, initialized):
    """Both initial target construction and a later restore need only one state copy."""
    model, optimizer = _build_optimizer()
    if initialized:
        _step(model, optimizer, 0)
        pointers = _moment_pointers(optimizer)
        values = _moments_on_cpu(optimizer)

    metadata = deepcopy(optimizer.state_dict())
    for index, group in enumerate(metadata["optimizer"]["param_groups"]):
        group["lr"] = 0.003 * (index + 1)
        group["betas"] = (0.7, 0.8)
        group["step"] = 7
    metadata["optimizer"]["param_groups"].reverse()

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    optimizer.load_state_dict(metadata)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    peak_growth = torch.cuda.max_memory_allocated() - before
    live_growth = after - before
    assert peak_growth <= live_growth + 1024**2, (peak_growth, live_growth)
    restored_pointers = _moment_pointers(optimizer)
    if initialized:
        assert restored_pointers == pointers
        for param, state in optimizer.optimizer.state.items():
            for name, tensor in state.items():
                torch.testing.assert_close(tensor.cpu(), values[param][name], rtol=0, atol=0)
    else:
        # Loading targets are uninitialized until checkpoint tensor IO fills them.
        for state in optimizer.optimizer.state.values():
            state["exp_avg"].fill_(0.125)
            state["exp_avg_sq"].fill_(0.25)

    saved_groups = {group["wd_mult"]: group for group in metadata["optimizer"]["param_groups"]}
    for group in optimizer.optimizer.param_groups:
        saved = saved_groups[group["wd_mult"]]
        assert group["lr"] == saved["lr"]
        assert group["betas"] == saved["betas"]
        assert group["step"] == saved["step"]

    # Build an independent reference with TE's public loader; it must produce
    # exactly the same subsequent update as the storage-preserving path.
    reference_groups = []
    for group in optimizer.optimizer.param_groups:
        reference_groups.append(
            {
                **{k: deepcopy(v) for k, v in group.items() if k != "params"},
                "params": [torch.nn.Parameter(p.detach().clone()) for p in group["params"]],
            }
        )
    reference = type(optimizer.optimizer)(
        reference_groups, adam_w_mode=optimizer.config.decoupled_weight_decay
    )
    reference.load_state_dict(deepcopy(optimizer.optimizer.state_dict()))
    for iteration in (1, 2):
        _step(model, optimizer, iteration)
        for group, reference_group in zip(
            optimizer.optimizer.param_groups, reference.param_groups, strict=True
        ):
            for param, reference_param in zip(
                group["params"], reference_group["params"], strict=True
            ):
                reference_param.grad = param.grad.detach().clone()
        reference.step()
        for group, reference_group in zip(
            optimizer.optimizer.param_groups, reference.param_groups, strict=True
        ):
            for param, reference_param in zip(
                group["params"], reference_group["params"], strict=True
            ):
                torch.testing.assert_close(param, reference_param, rtol=0, atol=0)
                for name, tensor in optimizer.optimizer.state[param].items():
                    torch.testing.assert_close(
                        tensor, reference.state[reference_param][name], rtol=0, atol=0
                    )


@pytest.mark.parametrize("sharding_type", ["dp_reshardable", "fully_reshardable"])
def test_checkpoint_round_trip_preserves_next_update(
    model_parallel, tmp_path_dist_ckpt, sharding_type
):
    """Real checkpoint targets, tensor reads and final restoration retain exact updates."""
    source_model, source = _build_optimizer()
    _step(source_model, source, 0)
    metadata = {
        "distrib_optim_sharding_type": sharding_type,
        "checkpoint_version": 3.1,
        "distrib_optim_fully_reshardable_mem_efficient": False,
    }
    with TempNamedDir(tmp_path_dist_ckpt / f"ordinary-adam-{sharding_type}") as checkpoint:
        save(
            source.sharded_state_dict(source_model.sharded_state_dict(), metadata=metadata),
            checkpoint,
        )
        restored_model, restored = _build_optimizer()
        target = restored.sharded_state_dict(
            restored_model.sharded_state_dict(), metadata=metadata, is_loading=True
        )
        pointers = _moment_pointers(restored)
        loaded, missing, unexpected = load(target, checkpoint, strict=StrictHandling.RETURN_ALL)
        assert not missing and not unexpected
        restored.load_state_dict(loaded)
        assert _moment_pointers(restored) == pointers
        _assert_optimizers_equal(source, restored)
        for iteration in (1, 2):
            _step(source_model, source, iteration)
            _step(restored_model, restored, iteration)
            _assert_optimizers_equal(source, restored)
            for source_param, restored_param in zip(
                source_model.parameters(), restored_model.parameters(), strict=True
            ):
                torch.testing.assert_close(source_param, restored_param, rtol=0, atol=0)


@pytest.mark.parametrize("exclusion", ["precision-aware", "pre-hook", "post-hook", "subclass"])
def test_unqualified_optimizers_keep_public_loader(model_parallel, exclusion, monkeypatch):
    """Custom loading behavior and precision-aware configurations retain TE's loader."""
    model, optimizer = _build_optimizer()
    _step(model, optimizer, 0)
    if exclusion == "precision-aware":
        monkeypatch.setattr(optimizer.config, "use_precision_aware_optimizer", True)
    elif exclusion == "pre-hook":
        optimizer.optimizer.register_load_state_dict_pre_hook(lambda opt, state: state)
    elif exclusion == "post-hook":
        optimizer.optimizer.register_load_state_dict_post_hook(lambda opt: None)
    else:

        class CustomAdam(type(optimizer.optimizer)):
            """An optimizer subclass owns its loading semantics."""

        inner = CustomAdam(
            optimizer.optimizer.param_groups, adam_w_mode=optimizer.config.decoupled_weight_decay
        )
        inner.load_state_dict(optimizer.optimizer.state_dict())
        optimizer.optimizer = inner
    metadata = deepcopy(optimizer.state_dict())
    with patch.object(
        optimizer.optimizer, "load_state_dict", wraps=optimizer.optimizer.load_state_dict
    ) as loader:
        optimizer.load_state_dict(metadata)
    loader.assert_called_once()


@pytest.mark.parametrize(
    "exclusion",
    [
        "master-weights",
        "capturable",
        "moment-dtype",
        "missing-hooks",
        "parameter-subclass",
        "state-subclass",
        "state-shape",
        "state-device",
    ],
)
def test_incompatible_representations_use_public_loader(model_parallel, exclusion, monkeypatch):
    """Do not bypass conversion when the actual optimizer or tensor storage differs."""
    model, optimizer = _build_optimizer()
    _step(model, optimizer, 0)
    inner = optimizer.optimizer
    payload = inner.state_dict()
    assert optimizer._can_use_torch_optimizer_loader(payload)
    group = next(group for group in inner.param_groups if group["params"])
    state = next(iter(payload["state"].values()))

    class TensorSubclass(torch.Tensor):
        """Stand in for a tensor with storage semantics not handled by this fast path."""

    if exclusion == "master-weights":
        monkeypatch.setattr(inner, "master_weights", True)
    elif exclusion == "capturable":
        monkeypatch.setattr(inner, "capturable", True)
    elif exclusion == "moment-dtype":
        monkeypatch.setattr(inner, "exp_avg_dtype", torch.bfloat16)
    elif exclusion == "missing-hooks":
        monkeypatch.delattr(inner, "_optimizer_load_state_dict_pre_hooks")
    elif exclusion == "parameter-subclass":
        group["params"][0] = group["params"][0].as_subclass(TensorSubclass)
    elif exclusion == "state-subclass":
        state["exp_avg"] = state["exp_avg"].as_subclass(TensorSubclass)
    elif exclusion == "state-shape":
        state["exp_avg"] = state["exp_avg"][:1]
    else:
        state["exp_avg"] = state["exp_avg"].cpu()
    # Altered attributes deliberately need not form a runnable optimizer: only
    # prove that unqualified representations are delegated to their public loader.
    with patch.object(inner, "load_state_dict") as loader:
        optimizer._load_inner_optimizer_state_dict(payload)
    loader.assert_called_once_with(payload)
