# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""A GPT model trains, checkpoints and serves on a collection that is not the global grid.

initialize_model_parallel() builds and registers the global grid with TP=2 in the default rank
order, so the parallel_state accessors return valid groups. The model runs on a second
collection that build_process_group_collection() lays out in another rank order and that is not
registered: its tensor- and data-parallel groups differ from the global ones on every rank.

Model build, DDP and optimizer construction, the training steps, the distributed checkpoint save
and load, and the inference build run with the parallel_state accessors, the stage predicates
and use_mpu_process_groups() raising, and with every ProcessGroupFallbackWarning and
ParallelStateDeprecationWarning raised as an error. A component that does not use the
collection it is given fails here; the failure names the phase, and the chained error names the
accessor or the fallback. The tests then check the groups that the components keep: those of
each role span the ranks of the collection's group, and no attribute of a module, model wrapper,
DDP buffer, bucket group, optimizer or inference wrapper is WORLD or a group object of the
global grid, nor holds one in a list, tuple, dict or process-group collection. Groups that other
objects keep are not searched. They also check that the results follow the collection: the
training losses match those of the same run on the global grid, the replicas of the
collection's data-parallel group stay equal, the checkpoint is laid out by the collection's
ranks and loads on the global grid, and the inference model reproduces the trained model's
logits.

A collective that passes no group runs over WORLD and reads no accessor. The comparison with the
global grid cannot see it either, because WORLD spans the same ranks in both layouts. Two other
checks can: a replica's loss and gradients must not change with another replica's batch, which a
forward or backward collective over WORLD breaks, and the tensor-parallel shards of the built
model must differ, which a broadcast over WORLD breaks.
"""

import contextlib
import math
import warnings
from types import SimpleNamespace

import pytest
import torch

from megatron.core import dist_checkpointing, parallel_state, process_groups_config
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer import optimizer as optimizer_module
from megatron.core.parallel_state import ParallelStateDeprecationWarning
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.process_groups_config import (
    ProcessGroupCollection,
    ProcessGroupFallbackWarning,
    warn_global_process_group_fallback,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_model_config, unwrap_model
from megatron.inference import utils as inference_utils
from megatron.training.initialize import _set_random_seed
from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars, forbid_global_process_groups

_TP = 2
# Data-parallel ranks are adjacent and tensor-parallel ranks strided, the other way round from
# the default order tp-cp-ep-dp-pp of the global grid.
_ORDER = "dp-tp-cp-ep-pp"
_VOCAB_SIZE = 128
_SEQ_LENGTH = 16
_MICRO_BATCH_SIZE = 2
_NUM_MICROBATCHES = 2
_TRAIN_STEPS = 2
_SEED = 1234
# The version save_checkpoint() writes; the optimizer loads its keys by it.
_CHECKPOINT_VERSION = 3.1

pytestmark = pytest.mark.skipif(
    Utils.world_size < 4 or Utils.world_size % _TP != 0,
    reason="needs an even number of ranks, at least 4",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="builds and trains the model with CUDA and NCCL"
)


def _ranks(group):
    """Global ranks of a process group, or None."""
    return None if group is None else torch.distributed.get_process_group_ranks(group)


def _assert_on_every_rank(problems):
    """Fail on every rank if any rank found a problem, listing the problems by rank.

    Every rank reaches the same collectives, so a rank that finds a problem does not leave the
    others waiting in a later collective. A problem that two checks find is listed once.
    """
    every_rank = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(every_rank, problems)
    report = [
        f"rank {rank}: {problem}"
        for rank, rank_problems in enumerate(every_rank)
        for problem in dict.fromkeys(rank_problems)
    ]
    assert not report, "\n".join(report)


@contextlib.contextmanager
def _on_the_collection_only(phase):
    """Run `phase` with the global grid forbidden and every process-group fallback an error.

    Every parallel_state accessor that tools/check_process_group_usage.py counts, the stage
    predicates and use_mpu_process_groups() raise, and a ProcessGroupFallbackWarning or
    ParallelStateDeprecationWarning raises as an error. The other deprecated parallel_state
    functions, such as the virtual-pipeline getters and get_all_ranks(), raise only when called
    from outside Megatron Core, whose calls of them do not warn. Each warning shows once per call
    site or owner in a process, so the registries that record them are emptied first: a fallback
    that warned earlier in the session raises here too. A failure is re-raised with `phase` in
    its message.
    """
    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        patch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())
        patch.setattr(optimizer_module, "_warned_world_grad_stats_owners", set())
        patch.setattr(parallel_state, "_warned_accessor_call_sites", set())
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        warnings.simplefilter("error", ParallelStateDeprecationWarning)
        try:
            with forbid_global_process_groups():
                yield
        except Exception as error:
            raise AssertionError(
                f"{phase} failed with the global grid forbidden: {type(error).__name__}: {error}"
            ) from error


def _dp_cp_group(pg_collection):
    """The group of the replica ids in a checkpoint, as save_checkpoint() picks it."""
    return pg_collection.dp_cp_gtp_remat or pg_collection.dp_cp


def _checkpoint_metadata(dp_cp_group):
    """The sharded state dict metadata that save_checkpoint() builds for this model.

    A torch_dist checkpoint of the distributed optimizer in the dp_reshardable layout, with the
    replica ids computed over `dp_cp_group`.
    """
    return {
        "distrib_optim_sharding_type": "dp_reshardable",
        "singleton_local_shards": False,
        "chained_optim_avoid_prefix": True,
        "dp_cp_group": dp_cp_group,
    }


def _model_config():
    """A two-layer GPT on the Megatron Core layer spec, in bf16 and without dropout."""
    transformer = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        tensor_model_parallel_size=_TP,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    return GPTModelConfig(
        transformer=transformer,
        vocab_size=_VOCAB_SIZE,
        seq_length=_SEQ_LENGTH,
        transformer_layer_spec=get_gpt_layer_local_spec(),
    )


def _build_model(pg_collection, seed, *, wrap_with_ddp=True):
    """Build the model as the training loop does: Float16Module, then DDP over `pg_collection`.

    The random number generators are seeded as initialize_megatron() seeds them, the CPU one
    included: the learned position embeddings are drawn on the CPU. With DDP, the seed differs
    between data-parallel ranks, so the replicas match only once DDP broadcasts the weights over
    the data-parallel group of `pg_collection`.
    """
    # The conftest sets the NVTE attention variables before each test, and the model's default
    # attention backend refuses them.
    clear_nvte_env_vars()
    _set_random_seed(seed, data_parallel_random_init=wrap_with_ddp, pg_collection=pg_collection)
    ddp_config = (
        DistributedDataParallelConfig(use_distributed_optimizer=True, overlap_grad_reduce=True)
        if wrap_with_ddp
        else None
    )
    return GPTModelBuilder(_model_config()).build_distributed_models(
        pg_collection=pg_collection,
        ddp_config=ddp_config,
        wrap_with_ddp=wrap_with_ddp,
        data_parallel_random_init=wrap_with_ddp,
    )


def _build_model_and_optimizer(pg_collection, seed):
    model = _build_model(pg_collection, seed)
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=1e-3,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )
    optimizer = get_megatron_optimizer(optimizer_config, model, pg_collection=pg_collection)
    return model, optimizer


def _batches(pg_collection, step):
    """Token and label microbatches; they differ between the collection's data-parallel ranks."""
    generator = torch.Generator().manual_seed(_SEED + 100 * step + pg_collection.dp.rank())
    for _ in range(_NUM_MICROBATCHES):
        tokens = torch.randint(
            _VOCAB_SIZE, (_MICRO_BATCH_SIZE, _SEQ_LENGTH + 1), generator=generator
        ).cuda()
        yield tokens[:, :-1], tokens[:, 1:]


def _forward_step(data_iterator, model):
    tokens, labels = next(data_iterator)
    position_ids = torch.arange(_SEQ_LENGTH, device="cuda").expand_as(tokens)
    losses = model(tokens, position_ids, None, labels=labels)

    def loss_func(output_tensor):
        loss = output_tensor.float().mean()
        # The schedule scales the returned loss in place, so report a copy.
        return loss, {"lm loss": loss.detach().clone()}

    return losses, loss_func


def _train_step(model, optimizer, pg_collection, step):
    """One iteration as train_step() runs it; returns the mean loss of its microbatches."""
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()
    forward_backward_func = get_forward_backward_func(pp_size=pg_collection.pp.size(), vp_size=None)
    losses_reduced = forward_backward_func(
        forward_step_func=_forward_step,
        data_iterator=_batches(pg_collection, step),
        model=model,
        num_microbatches=_NUM_MICROBATCHES,
        seq_length=_SEQ_LENGTH,
        micro_batch_size=_MICRO_BATCH_SIZE,
        forward_only=False,
        pg_collection=pg_collection,
    )
    update_successful, _, _ = optimizer.step()
    assert update_successful, f"the optimizer skipped step {step}"
    return torch.stack([losses["lm loss"] for losses in losses_reduced]).mean().item()


def _named_parameters(model_chunk):
    return {
        name: param.detach().clone() for name, param in unwrap_model(model_chunk).named_parameters()
    }


def _gather(group, tensor):
    """`tensor` of every rank of `group`, ordered by rank in the group; every rank calls it."""
    gathered = [torch.empty_like(tensor) for _ in range(group.size())]
    torch.distributed.all_gather(gathered, tensor.detach().contiguous(), group=group)
    return gathered


def _differences_across(group, tensors, description):
    """Name each tensor that is not the same on every rank of `group`; every rank calls it."""
    problems = []
    for name, tensor in tensors.items():
        gathered = _gather(group, tensor)
        if not all(torch.equal(gathered[0], other) for other in gathered[1:]):
            largest = max((other - gathered[0]).abs().max().item() for other in gathered[1:])
            problems.append(
                f"{name} differs across {description} {_ranks(group)} (by up to {largest})"
            )
    return problems


def _equal_across(group, tensors, description):
    """Name each tensor that is the same on every rank of `group`; every rank calls it."""
    problems = []
    for name, tensor in tensors.items():
        gathered = _gather(group, tensor)
        if all(torch.equal(gathered[0], other) for other in gathered[1:]):
            problems.append(f"{name} is the same on every rank of {description} {_ranks(group)}")
    return problems


def _tensor_parallel_shards(model_chunk, tp_group):
    """Every parameter's shards over `tp_group`, ordered by tensor-parallel rank."""
    return {
        name: _gather(tp_group, param)
        for name, param in unwrap_model(model_chunk).named_parameters()
    }


def _optimizer_shards(optimizer):
    """This rank's main parameters and Adam moments, in the order the optimizer keeps them."""
    shards = []
    for index, distributed_optimizer in enumerate(optimizer.chained_optimizers):
        inner = distributed_optimizer.optimizer
        for group_index, param_group in enumerate(inner.param_groups):
            for param_index, param in enumerate(param_group["params"]):
                label = f"optimizer {index}, group {group_index}, parameter {param_index}"
                shards.append((f"{label}, main parameter", param.detach()))
                state = inner.state[param]
                for key in distributed_optimizer.optimizer_state_keys:
                    shards.append((f"{label}, {key}", state.get(key)))
    return shards


def _foreign_group_problems(label, owner, foreign):
    """Describe each attribute of `owner` that is a `foreign` group.

    The lists, tuples, dicts and process-group collections among the attributes are searched
    too, but not other objects.
    """
    return [
        problem
        for attribute, value in vars(owner).items()
        for problem in _foreign_value_problems(f"{label}.{attribute}", value, foreign)
    ]


def _foreign_value_problems(label, value, foreign):
    if id(value) in foreign:
        return [f"{label} is {foreign[id(value)]}"]
    if isinstance(value, ProcessGroupCollection):
        return _foreign_group_problems(label, value, foreign)
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    else:
        return []
    return [
        problem
        for key, item in items
        for problem in _foreign_value_problems(f"{label}[{key!r}]", item, foreign)
    ]


def _group_problems(owner, groups, pg_collection, foreign):
    """Describe each group of `owner` that is not the collection's.

    The (attribute, field) pairs of `groups` name the attributes that must span the ranks of the
    collection's groups, and no attribute of `owner` may be one of the `foreign` groups.
    """
    problems = _foreign_group_problems(type(owner).__name__, owner, foreign)
    for attribute, field in groups.items():
        actual = _ranks(getattr(owner, attribute, None))
        expected = _ranks(getattr(pg_collection, field))
        if actual != expected:
            problems.append(
                f"{type(owner).__name__}.{attribute} spans ranks {actual}, not those of the "
                f"collection's {field} group {expected}"
            )
    return problems


# The process groups that modules keep as attributes, by collection field.
_MODULE_GROUPS = {"tp_group": "tp", "cp_group": "cp", "pp_group": "pp"}


def _named_modules(model_chunk):
    """The modules of `model_chunk` by name, and its wrappers, such as Float16Module, by type."""
    model = unwrap_model(model_chunk)
    wrappers = []
    while model_chunk is not model:
        wrappers.append((type(model_chunk).__name__, model_chunk))
        model_chunk = model_chunk.module
    return wrappers + [
        (name or type(model).__name__, module) for name, module in model.named_modules()
    ]


def _module_problems(model_chunk, pg_collection, foreign):
    """Describe each module or wrapper that keeps a group that is not the collection's.

    No module may keep one of the `foreign` groups, and the groups of _MODULE_GROUPS and of a
    module's collection must span the ranks of the collection's groups.
    """
    problems = []
    expected = {
        field: _ranks(group)
        for field, group in vars(pg_collection).items()
        if group is None or isinstance(group, torch.distributed.ProcessGroup)
    }
    for name, module in _named_modules(model_chunk):
        problems += _foreign_group_problems(name, module, foreign)
        held = {
            f"{name}.{attribute}": (field, vars(module)[attribute])
            for attribute, field in _MODULE_GROUPS.items()
            if attribute in vars(module)
        }
        module_collection = vars(module).get("pg_collection")
        if isinstance(module_collection, ProcessGroupCollection):
            held.update(
                (f"{name}.pg_collection.{field}", (field, group))
                for field, group in vars(module_collection).items()
                if field in expected
            )
        for label, (field, group) in held.items():
            if _ranks(group) != expected[field]:
                problems.append(
                    f"{label} spans ranks {_ranks(group)}, not those of the collection's {field} "
                    f"group {expected[field]}"
                )
    return problems


def _model_layout_problems(model_state, tp_rank, dp_cp_rank):
    """Describe each model tensor not placed by these tensor- and data-parallel ranks."""
    problems = []
    for sharded in nested_values(model_state):
        if not isinstance(sharded, ShardedTensor):
            continue
        # The axes after the prepended (layer) axes are split over the tensor-parallel group.
        split = [
            axis
            for axis in range(sharded.prepend_axis_num, len(sharded.global_shape))
            if sharded.axis_fragmentations[axis] > 1
        ]
        chunks = {
            axis: sharded.global_offset[axis]
            // sharded.local_shape[axis - sharded.prepend_axis_num]
            for axis in split
        }
        if any(chunk != tp_rank for chunk in chunks.values()):
            problems.append(
                f"{sharded.key} holds chunks {chunks}, not those of tensor-parallel rank {tp_rank}"
            )
        # The tensor-parallel rank tells apart the replicas of a tensor that is not split.
        expected_replica_id = (0, 0 if split else tp_rank, dp_cp_rank)
        if sharded.replica_id != expected_replica_id:
            problems.append(
                f"{sharded.key} has replica id {sharded.replica_id}, not {expected_replica_id}"
            )
    return problems


def _optimizer_layout_problems(optimizer_state, mp_rank, dp_rank):
    """Describe each dp_reshardable optimizer entry not placed by these ranks.

    The entries are keyed by the model-parallel rank and replicated over the data-parallel group
    that the distributed optimizer shards over.
    """
    problems = []
    index = f"dp_group_idx_{mp_rank}."
    for sharded in nested_values(optimizer_state):
        key = getattr(sharded, "key", "")
        if key.startswith("optimizer.distributed.") and index not in key:
            problems.append(f"{key} is not keyed by {index}")
        if isinstance(sharded, ShardedObject) and sharded.replica_id != (0, 0, dp_rank):
            problems.append(f"{key} has replica id {sharded.replica_id}, not (0, 0, {dp_rank})")
    return problems


def _foreign_groups(collection, global_grid):
    """The groups that a component on `collection` must not keep, by id, with a description.

    These are WORLD and the groups of the global grid that the collection does not share. In this
    layout several of them span the same ranks as the collection's groups (the pipeline-,
    context-, expert-parallel and GTP groups, the embedding groups, tp_dp_cp and intra_dist_opt),
    so only the group object tells them apart.
    """
    foreign = {id(torch.distributed.group.WORLD): "the default (WORLD) group"}
    for field, group in vars(global_grid).items():
        if isinstance(group, torch.distributed.ProcessGroup):
            foreign.setdefault(id(group), f"the global grid's {field} group")
    for name, group in vars(parallel_state).items():
        if isinstance(group, torch.distributed.ProcessGroup):
            foreign.setdefault(id(group), f"the global grid's group parallel_state.{name}")
    for group in vars(collection).values():
        foreign.pop(id(group), None)
    return foreign


@pytest.fixture(scope="module")
def grids():
    """The global grid in the default rank order, and an unregistered collection in another one."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=_TP)
    collection, _ = parallel_state.build_process_group_collection(
        tensor_model_parallel_size=_TP, order=_ORDER
    )
    global_grid = parallel_state.get_registered_process_groups()
    yield SimpleNamespace(
        collection=collection,
        global_grid=global_grid,
        foreign=_foreign_groups(collection, global_grid),
    )
    # Also destroys the groups of the collection, which are on the list of created groups.
    Utils.destroy_model_parallel()


def _train(pg_collection, description):
    """Build a model and its optimizer on `pg_collection` and train them for a few steps."""
    with _on_the_collection_only(f"building the model, DDP and the optimizer on {description}"):
        model, optimizer = _build_model_and_optimizer(pg_collection, _SEED)
    initial = _named_parameters(model[0])
    # What train() sets up before its first iteration.
    config = get_model_config(model[0])
    config.no_sync_func = model[0].no_sync
    config.finalize_model_grads_func = finalize_model_grads
    config.grad_scale_func = optimizer.scale_loss
    with _on_the_collection_only(f"running the training steps on {description}"):
        losses = [
            _train_step(model, optimizer, pg_collection, step) for step in range(_TRAIN_STEPS)
        ]
    return SimpleNamespace(model=model, optimizer=optimizer, initial=initial, losses=losses)


@pytest.fixture(scope="module")
def trained(grids):
    """A model and its optimizer, built on the collection and trained for a few steps."""
    return _train(grids.collection, "the collection")


@pytest.fixture(scope="module")
def reference(grids):
    """The same model and training, on the groups of the global grid."""
    return _train(grids.global_grid, "the global grid's collection")


@pytest.fixture(scope="module")
def checkpoint(trained, grids, tmp_path_dist_ckpt):
    """A torch_dist checkpoint of the trained model and optimizer, saved on the collection."""
    metadata = _checkpoint_metadata(_dp_cp_group(grids.collection))
    with TempNamedDir(
        tmp_path_dist_ckpt / "test_process_group_collection_end_to_end", sync=True
    ) as directory:
        with _on_the_collection_only("saving the distributed checkpoint"):
            model_state = unwrap_model(trained.model[0]).sharded_state_dict(metadata=metadata)
            optimizer_state = trained.optimizer.sharded_state_dict(model_state, metadata=metadata)
            dist_checkpointing.save(
                {
                    "checkpoint_version": _CHECKPOINT_VERSION,
                    "model": model_state,
                    "optimizer": optimizer_state,
                },
                str(directory),
            )
        yield str(directory)


def test_the_collection_differs_from_the_global_grid(grids):
    collection, global_grid = grids.collection, grids.global_grid
    rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    dp_size = world_size // _TP
    problems = []
    if parallel_state.get_registered_process_groups() is collection:
        problems.append("the collection is registered, so the accessors return its groups")
    # dp-tp: a rank's data-parallel peers are adjacent and its tensor-parallel peers strided.
    expected = {
        "tp": [rank % dp_size + dp_size * tp_rank for tp_rank in range(_TP)],
        "dp": [rank // dp_size * dp_size + dp_rank for dp_rank in range(dp_size)],
    }
    for field, ranks in expected.items():
        if _ranks(getattr(collection, field)) != ranks:
            problems.append(
                f"the collection's {field} group is {_ranks(getattr(collection, field))}"
            )
    for field in ("tp", "mp", "dp", "dp_cp", "intra_dp_cp"):
        if _ranks(getattr(collection, field)) == _ranks(getattr(global_grid, field)):
            problems.append(
                f"the collection's {field} group spans the ranks of the global one, "
                f"{_ranks(getattr(global_grid, field))}"
            )
    _assert_on_every_rank(problems)


def test_the_block_rejects_global_reads_and_fallbacks(monkeypatch):
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())
    monkeypatch.setattr(optimizer_module, "_warned_world_grad_stats_owners", set())
    with pytest.raises(AssertionError, match=r"probe failed .*get_tensor_model_parallel_group"):
        with _on_the_collection_only("probe"):
            parallel_state.get_tensor_model_parallel_group()
    # A fallback that already warned in this process raises in the block all the same.
    owner = "test_process_group_collection_end_to_end probe"
    with pytest.warns(ProcessGroupFallbackWarning):
        warn_global_process_group_fallback(owner, deprecated_in="0.21", removed_in="0.23")
    with pytest.raises(AssertionError, match=r"probe failed .*ProcessGroupFallbackWarning"):
        with _on_the_collection_only("probe"):
            warn_global_process_group_fallback(owner, deprecated_in="0.21", removed_in="0.23")
    # So does the fallback of an optimizer without a grad-stats group, which has its own registry.
    with pytest.warns(ProcessGroupFallbackWarning):
        optimizer_module._warn_grad_stats_reduced_over_world(owner)
    with pytest.raises(AssertionError, match=r"probe failed .*ProcessGroupFallbackWarning"):
        with _on_the_collection_only("probe"):
            optimizer_module._warn_grad_stats_reduced_over_world(owner)
    # Deprecated functions that are not accessors warn, and the warning raises.
    with pytest.raises(AssertionError, match=r"probe failed .*ParallelStateDeprecationWarning"):
        with _on_the_collection_only("probe"):
            parallel_state.get_virtual_pipeline_model_parallel_world_size()


@requires_cuda
def test_model_and_optimizer_are_built_on_the_collection(trained, grids):
    collection, foreign = grids.collection, grids.foreign
    (ddp_chunk,) = trained.model
    model_params = dict(unwrap_model(ddp_chunk).named_parameters())
    # DDP broadcast the weights over the collection's data-parallel group, so the replicas are
    # equal while the tensor-parallel ranks keep the shards they drew from their own seeds. A
    # broadcast over WORLD would make the shards equal too.
    problems = _differences_across(
        collection.dp_cp, trained.initial, "the built model's replicas in the data-parallel group"
    )
    problems += _equal_across(
        collection.tp,
        {
            name: param
            for name, param in trained.initial.items()
            if getattr(model_params[name], "tensor_model_parallel", False) and param.dim() > 1
        },
        "the built model's tensor-parallel group",
    )
    if unwrap_model(ddp_chunk).pg_collection is not collection:
        problems.append("the model does not keep the collection it was built on")
    problems += _module_problems(ddp_chunk, collection, foreign)
    problems += _group_problems(
        ddp_chunk,
        {
            "dp_group": "dp",
            "dp_cp_group": "dp_cp",
            "intra_dp_cp_group": "intra_dp_cp",
            "expt_dp_group": "expt_dp",
            "intra_expt_dp_group": "intra_expt_dp",
            "tp_group": "tp",
            "pp_group": "pp",
            "ep_group": "ep",
        },
        collection,
        foreign,
    )
    for buffer in ddp_chunk.buffers:
        problems += _group_problems(
            buffer, {"data_parallel_group": "intra_dp_cp"}, collection, foreign
        )
    for bucket_group in ddp_chunk.bucket_groups:
        problems += _group_problems(
            bucket_group,
            {"intra_distributed_optimizer_instance_group": "intra_dp_cp"},
            collection,
            foreign,
        )
    (distributed_optimizer,) = trained.optimizer.chained_optimizers
    if not getattr(distributed_optimizer, "duplicate_filter_groups_from_collection", False):
        problems.append("the optimizer filters replicated gradients by the global grid")
    problems += _group_problems(
        distributed_optimizer,
        {
            "tp_group": "tp",
            "data_parallel_group": "intra_dp_cp",
            "data_parallel_group_gloo": "intra_dp_cp_gloo",
        },
        collection,
        foreign,
    )
    if distributed_optimizer.data_parallel_group_idx != collection.mp.rank():
        problems.append(
            f"the optimizer's checkpoint index is {distributed_optimizer.data_parallel_group_idx}, "
            f"not the collection's model-parallel rank {collection.mp.rank()}"
        )
    grad_stats_group = distributed_optimizer.get_grad_stats_parallel_group()
    if id(grad_stats_group) in foreign:
        problems.append(
            f"the optimizer reduces gradient statistics over {foreign[id(grad_stats_group)]}"
        )
    elif _ranks(grad_stats_group) != _ranks(collection.intra_dist_opt):
        problems.append(
            f"the optimizer reduces gradient statistics over {_ranks(grad_stats_group)}"
        )
    _assert_on_every_rank(problems)


@requires_cuda
def test_training_steps_run_on_the_collection(trained, grids):
    collection = grids.collection
    (ddp_chunk,) = trained.model
    params = _named_parameters(ddp_chunk)
    model_params = dict(unwrap_model(ddp_chunk).named_parameters())
    replicated = {
        name: param
        for name, param in params.items()
        if not getattr(model_params[name], "tensor_model_parallel", False)
    }
    # The data-parallel replicas trained on different batches, so they stay equal only if the
    # distributed optimizer gathers the updated weights over the collection's data-parallel
    # group. test_training_matches_the_global_grid checks that the gradients are not reduced
    # over a group of the global grid with other ranks.
    problems = _differences_across(
        collection.dp_cp, params, "the trained model's replicas in the data-parallel group"
    )
    problems += _differences_across(
        collection.tp, replicated, "the trained model's replicas in the tensor-parallel group"
    )
    every_rank = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(every_rank, trained.losses)
    tp_peer_losses = [every_rank[peer] for peer in _ranks(collection.tp)]
    if any(losses != trained.losses for losses in tp_peer_losses):
        problems.append(f"the tensor-parallel peers report the losses {tp_peer_losses}")
    if not all(math.isfinite(loss) for loss in trained.losses):
        problems.append(f"the losses are {trained.losses}")
    # The vocabulary-parallel cross entropy normalizes over the collection's tensor-parallel
    # group: at initialization the loss is close to log(vocabulary size).
    if abs(trained.losses[0] - math.log(_VOCAB_SIZE)) > 0.2:
        problems.append(
            f"the first loss is {trained.losses[0]}, not about log({_VOCAB_SIZE}) = "
            f"{math.log(_VOCAB_SIZE):.3f}"
        )
    unchanged = [
        name
        for name, param in params.items()
        if getattr(model_params[name], "tensor_model_parallel", False)
        and torch.equal(param, trained.initial[name])
    ]
    if unchanged:
        problems.append(f"the training steps did not change {unchanged}")
    _assert_on_every_rank(problems)


@requires_cuda
def test_training_matches_the_global_grid(trained, reference, grids):
    """Each data-parallel rank reports the losses of the same run on the global grid.

    Both runs start from the weights of data-parallel rank 0, which DDP broadcasts, and train on
    the same batches per data-parallel rank. Data-parallel rank 0 draws the same weights in both
    layouts, because _build_model() seeds by the data- and tensor-parallel ranks in the collection
    it builds on, the CPU generator included.

    A collective of the run on the collection over a group of the global grid with other ranks
    than the collection's group changes its losses. A collective over a global group with the
    same ranks does not; the checks by object in
    test_model_and_optimizer_are_built_on_the_collection see those. Nor does a collective over
    WORLD: WORLD spans the same ranks in both layouts, and the run on the global grid is the same
    run with its ranks renumbered, so the error is the same in both.
    test_a_replica_does_not_depend_on_other_replicas_batches checks for WORLD instead.
    """
    every_rank = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(
        every_rank,
        (grids.collection.dp.rank(), trained.losses, grids.global_grid.dp.rank(), reference.losses),
    )
    on_the_collection = {dp_rank: losses for dp_rank, losses, _, _ in every_rank}
    on_the_global_grid = {dp_rank: losses for _, _, dp_rank, losses in every_rank}
    for dp_rank, losses in on_the_collection.items():
        expected = on_the_global_grid[dp_rank]
        # The first step runs no data-parallel reduction before its loss.
        assert math.isclose(losses[0], expected[0], rel_tol=1e-5) and all(
            math.isclose(loss, expected_loss, abs_tol=1e-3)
            for loss, expected_loss in zip(losses[1:], expected[1:])
        ), (
            f"data-parallel rank {dp_rank} reports the losses {losses} on the collection and "
            f"{expected} on the global grid"
        )


def _loss_and_local_gradients(ddp_chunk, data_iterator):
    """The loss of one microbatch on this rank and its gradients, which DDP does not reduce."""
    ddp_chunk.zero_grad_buffer()
    with ddp_chunk.no_sync():
        output, loss_func = _forward_step(data_iterator, ddp_chunk)
        loss, _ = loss_func(output)
        loss.backward()
    gradients = {
        name: param.main_grad.detach().clone()
        for name, param in unwrap_model(ddp_chunk).named_parameters()
    }
    ddp_chunk.zero_grad_buffer()
    return loss.item(), gradients


@requires_cuda
def test_a_replica_does_not_depend_on_other_replicas_batches(trained, grids):
    """A replica's loss and gradients do not change with the other replicas' batches.

    Until DDP reduces the gradients, a replica of the collection's data-parallel group computes
    from its own batch only. A forward or backward collective over WORLD, or over another group
    that reaches into other replicas, mixes their batches in. Neither the forbidden accessors nor
    test_training_matches_the_global_grid can see a collective over WORLD.
    """
    collection = grids.collection
    (ddp_chunk,) = trained.model
    dp_rank = collection.dp.rank()
    with _on_the_collection_only("running a forward and backward pass without the reduction"):
        loss, gradients = _loss_and_local_gradients(ddp_chunk, _batches(collection, 1000))
        # Every replica but the first takes another batch.
        other_step = 1000 if dp_rank == 0 else 2000
        other_loss, other_gradients = _loss_and_local_gradients(
            ddp_chunk, _batches(collection, other_step)
        )
    problems = []
    if dp_rank == 0:
        if not math.isclose(other_loss, loss, rel_tol=1e-5):
            problems.append(
                f"the loss of data-parallel rank 0 moves from {loss} to {other_loss} when the "
                "other replicas take other batches"
            )
        for name, gradient in gradients.items():
            change = torch.linalg.vector_norm(other_gradients[name].float() - gradient.float())
            if change > 1e-3 * torch.linalg.vector_norm(gradient.float()):
                problems.append(
                    f"the gradient of {name} on data-parallel rank 0 changes by {change.item()} "
                    "when the other replicas take other batches"
                )
    elif other_loss == loss:
        problems.append(f"the other batch leaves the loss at {loss}, so the check is not live")
    _assert_on_every_rank(problems)


@requires_cuda
def test_checkpoint_is_laid_out_by_the_collection(trained, grids):
    collection = grids.collection
    dp_cp_group = _dp_cp_group(collection)
    with _on_the_collection_only("building the sharded state dicts"):
        metadata = _checkpoint_metadata(dp_cp_group)
        model_state = unwrap_model(trained.model[0]).sharded_state_dict(metadata=metadata)
        optimizer_state = trained.optimizer.sharded_state_dict(model_state, metadata=metadata)
    problems = _model_layout_problems(model_state, collection.tp.rank(), dp_cp_group.rank())
    problems += _optimizer_layout_problems(
        optimizer_state, collection.mp.rank(), collection.intra_dp_cp.rank()
    )
    _assert_on_every_rank(problems)


@requires_cuda
def test_checkpoint_loads_into_a_new_model_and_optimizer(trained, checkpoint, grids):
    collection = grids.collection
    with _on_the_collection_only("building a second model and optimizer"):
        model, optimizer = _build_model_and_optimizer(collection, _SEED + 1)
    saved = _named_parameters(trained.model[0])
    differ_before_loading = any(
        not torch.equal(param, saved[name]) for name, param in _named_parameters(model[0]).items()
    )
    with _on_the_collection_only("loading the distributed checkpoint"):
        metadata = {
            **_checkpoint_metadata(_dp_cp_group(collection)),
            "checkpoint_version": _CHECKPOINT_VERSION,
        }
        gpt_model = unwrap_model(model[0])
        model_state = gpt_model.sharded_state_dict(metadata=metadata)
        optimizer_state = optimizer.sharded_state_dict(
            model_state, is_loading=True, metadata=metadata
        )
        loaded = dist_checkpointing.load(
            {"model": model_state, "optimizer": optimizer_state}, checkpoint
        )
        gpt_model.load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])
    problems = [] if differ_before_loading else ["the second model starts as the trained one"]
    problems += [
        f"the reloaded {name} differs from the saved one"
        for name, param in _named_parameters(model[0]).items()
        if not torch.equal(param, saved[name])
    ]
    saved_shards = _optimizer_shards(trained.optimizer)
    loaded_shards = _optimizer_shards(optimizer)
    if [label for label, _ in loaded_shards] != [label for label, _ in saved_shards]:
        problems.append("the reloaded optimizer holds other shards than the saved one")
    else:
        problems += [
            f"the reloaded {label} differs from the saved one"
            for (label, loaded_shard), (_, saved_shard) in zip(loaded_shards, saved_shards)
            if loaded_shard is None or not torch.equal(loaded_shard, saved_shard)
        ]
    _assert_on_every_rank(problems)


@requires_cuda
def test_checkpoint_loads_on_the_global_grid(trained, checkpoint, grids):
    """The checkpoint holds the model itself: the global grid, laid out otherwise, loads it."""
    global_grid = grids.global_grid
    with _on_the_collection_only("loading the checkpoint on the global grid's collection"):
        model = _build_model(global_grid, _SEED + 2, wrap_with_ddp=False)
        metadata = {
            **_checkpoint_metadata(_dp_cp_group(global_grid)),
            "checkpoint_version": _CHECKPOINT_VERSION,
        }
        gpt_model = unwrap_model(model[0])
        loaded = dist_checkpointing.load(
            {"model": gpt_model.sharded_state_dict(metadata=metadata)}, checkpoint
        )
        gpt_model.load_state_dict(loaded["model"])
    saved = _tensor_parallel_shards(trained.model[0], grids.collection.tp)
    reloaded = _tensor_parallel_shards(model[0], global_grid.tp)
    problems = [
        f"{name} by tensor-parallel rank differs between the collection and the global grid"
        for name, shards in saved.items()
        if not all(torch.equal(a, b) for a, b in zip(shards, reloaded[name]))
    ]
    _assert_on_every_rank(problems)


@requires_cuda
def test_inference_model_is_built_and_loaded_on_the_collection(
    trained, checkpoint, grids, monkeypatch, run_config
):
    collection = grids.collection
    built_on, loaded_with = [], []

    def get_model_builder(args, tokenizer_config, pg_collection):
        built_on.append(pg_collection)
        clear_nvte_env_vars()
        return GPTModelBuilder(_model_config())

    def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, **kwargs):
        """Load the model weights from the checkpoint with the groups the caller passes."""
        loaded_with.append(kwargs)
        gpt_model = unwrap_model(ddp_model[0])
        metadata = {
            **_checkpoint_metadata(kwargs["dp_cp_group"]),
            "checkpoint_version": _CHECKPOINT_VERSION,
        }
        loaded = dist_checkpointing.load(
            {"model": gpt_model.sharded_state_dict(metadata=metadata)}, checkpoint
        )
        gpt_model.load_state_dict(loaded["model"], strict=kwargs["strict"])

    # The arguments get_model_for_inference reads, with the builder and the checkpoint loader
    # of the training script replaced by the ones of this model. It also reads the tokenizer
    # config of `run_config`, which this builder ignores: the model config sets the vocabulary.
    args = SimpleNamespace(
        load=checkpoint,
        inference_ckpt_non_strict=False,
        transformer_impl="local",
        fp8_recipe=None,
        modelopt_enabled=False,
    )
    monkeypatch.setattr(inference_utils, "get_args", lambda: args)
    monkeypatch.setattr(inference_utils, "get_model_builder", get_model_builder)
    monkeypatch.setattr(inference_utils, "load_checkpoint", load_checkpoint)
    tokens = torch.randint(
        _VOCAB_SIZE,
        (_MICRO_BATCH_SIZE, _SEQ_LENGTH),
        generator=torch.Generator().manual_seed(_SEED),
    ).cuda()

    with _on_the_collection_only("building the model with get_model_for_inference"):
        model = inference_utils.get_model_for_inference(pg_collection=collection)
    with (
        _on_the_collection_only("running a forward step of the inference wrapper"),
        InferenceMode.active(),
        torch.no_grad(),
    ):
        context = StaticInferenceContext(_MICRO_BATCH_SIZE, _SEQ_LENGTH)
        context.config.materialize_only_last_token_logits = False
        wrapper = GPTInferenceWrapper(model, context)
        wrapper.prep_model_for_inference()
        inference_input = wrapper.prep_inference_input(prompts_tokens=tokens)
        logits = wrapper.run_one_forward_step(
            wrapper.get_batch_for_context_window(inference_input, 0, _SEQ_LENGTH)
        )
    with _on_the_collection_only("running the trained model forward"), torch.no_grad():
        expected = trained.model[0](
            tokens,
            inference_input["position_ids"],
            inference_input["attention_mask"],
            runtime_gather_output=True,
        )

    problems = []
    if len(built_on) != 1 or built_on[0] is not collection:
        problems.append("get_model_for_inference built the model on another collection")
    checkpoint_groups = {
        "tp_group": "tp",
        "pp_group": "pp",
        "dp_cp_group": "dp_cp",
        "dp_group": "dp",
        "expt_dp_group": "expt_dp",
    }
    for argument, field in checkpoint_groups.items():
        if _ranks(loaded_with[0][argument]) != _ranks(getattr(collection, field)):
            problems.append(
                f"get_model_for_inference loads with {argument} {_ranks(loaded_with[0][argument])}"
            )
    problems += _module_problems(model, collection, grids.foreign)
    problems += _group_problems(
        wrapper, {"tp_group": "tp", "pp_group": "pp"}, collection, grids.foreign
    )
    # The context was given no collection, so it resolves the one the model was built on.
    if wrapper.inference_context.config.pg_collection is not collection:
        problems.append("the inference context does not run on the collection of the model")
    try:
        torch.testing.assert_close(logits, expected, rtol=2e-2, atol=2e-2)
    except AssertionError as mismatch:
        problems.append(f"the inference logits differ from the trained model's: {mismatch}")
    _assert_on_every_rank(problems)
