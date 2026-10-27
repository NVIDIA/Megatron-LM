# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import psutil
import torch


def chunk_bias(bias, parallel_mode, tp_size=1, ep_size=1):
    assert parallel_mode in ["row", "column"]
    if bias.dim() == 2:
        num_experts, hidden_size = bias.shape
        if parallel_mode == 'column':
            bias = bias.reshape(ep_size, num_experts // ep_size, tp_size, hidden_size // tp_size)
            bias = bias.permute(0, 2, 1, 3) # (ep_size, tp_size, local_eps, hidden_size)
        else:
            bias = bias.reshape(ep_size, num_experts // ep_size, hidden_size) # (ep_size, local_eps, hidden_size)
        return bias
    else:
        hidden_size = bias.shape
        if parallel_mode == "column":
            bias = bias.reshape(tp_size, hidden_size[0] // tp_size) # (tp_size, hidden_size)
        return bias


def chunk_weight(weight, parallel_mode, tp_size=1, ep_size=1):
    assert parallel_mode in ["row", "column"]
    if weight.dim() == 3:
        num_experts, out_features, in_features = weight.shape
        if parallel_mode == "column":
            weight = weight.reshape(ep_size, num_experts // ep_size, tp_size, out_features // tp_size, in_features)
            weight = weight.permute(0, 2, 1, 3, 4)
        else:
            weight = weight.reshape(ep_size, num_experts // ep_size, out_features, tp_size, in_features // tp_size)
            weight = weight.permute(0, 3, 1, 2, 4)
        return weight # (ep_size, tp_size, local_eps, output_features, in_features)
    else:
        out_features, in_features = weight.shape
        if parallel_mode == "column":
            weight = weight.reshape(tp_size, out_features // tp_size, in_features)
        else:
            weight = weight.reshape(out_features, tp_size, in_features // tp_size).permute(1, 0, 2)
        return weight # (tp_size, output_features, in_features)


def print_memory_usage(key, rank, num_ranks):
    '''Print memory usage.'''
    process = psutil.Process()
    mem_info = process.memory_info()
    print("> memory usage: '%s', rank %d / %d, mem %.1f/%.1f gb." % (
        key,
        rank,
        num_ranks,
        mem_info.rss / 1024**3,
        100 * mem_info.rss / process.memory_percent() / 1024**3,
    ))

class _ConverterFakeProcessGroup:
    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def set_rank(self, rank):
        self._rank = rank

    def set_size(self, size):
        self._size = size


def initialize_checkpoint_converter_distributed():
    """Initialize a single-process gloo group unless torch.distributed is already initialized.

    Model construction reads ranks and sizes from the fake process groups through
    ``get_pg_rank`` and ``get_pg_size``, which ignore a group until torch.distributed is
    initialized. The in-process store needs no port, so the loader and saver processes of a
    conversion, and other jobs on the same host, cannot clash on one.
    """
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='gloo', store=torch.distributed.HashStore(), rank=0, world_size=1
        )


def build_checkpoint_converter_process_groups(
    tensor_parallel_size, pipeline_parallel_size, expert_parallel_size
):
    """Build the process-group collection that a checkpoint converter passes to its models.

    Checkpoint conversion runs in one process and does not call
    ``initialize_model_parallel``. It builds the model of every rank of a parallel
    layout in turn, so each group is a lightweight stand-in that reports the
    requested size and the rank of the model being built. The groups cannot
    communicate. Models read them through ``get_pg_rank`` and ``get_pg_size``, so
    torch.distributed must be initialized (see
    ``initialize_checkpoint_converter_distributed``).

    All models of a converter share these groups. Before building or loading a
    model, the converter sets that model's ranks on them, for example with
    ``set_checkpoint_converter_pipeline_rank``.

    GTP rematerialization is not active during conversion. Its data-parallel
    groups therefore alias the corresponding non-GTP groups, matching
    ``initialize_model_parallel`` when the GTP axes have size one.

    Context parallelism is not used during conversion either, but the model
    still needs a size-one context-parallel group: ``LanguageModule`` requires
    tensor-, context- and pipeline-parallel groups once ``torch.distributed`` is
    initialized.

    Conversion creates no other groups. Every other field, such as the embedding,
    GTP and distributed-optimizer groups, is set to None and the runtime
    context-parallel maps are empty. In a converter process, this is what
    ``ProcessGroupCollection.use_mpu_process_groups`` returns once
    ``install_checkpoint_converter_process_groups`` mirrors the groups into
    ``parallel_state``.

    Args:
        tensor_parallel_size: Tensor-parallel size of the checkpoint layout.
        pipeline_parallel_size: Pipeline-parallel size of the checkpoint layout.
        expert_parallel_size: Expert-parallel size of the checkpoint layout.

    Returns:
        ProcessGroupCollection: The fake groups, with every rank set to 0.
    """
    from dataclasses import fields

    from megatron.core.process_groups_config import ProcessGroupCollection

    dp_group = _ConverterFakeProcessGroup(size=1)
    expert_dp_group = _ConverterFakeProcessGroup(size=1)
    groups = dict(
        tp=_ConverterFakeProcessGroup(size=tensor_parallel_size),
        cp=_ConverterFakeProcessGroup(size=1),
        pp=_ConverterFakeProcessGroup(size=pipeline_parallel_size),
        mp=_ConverterFakeProcessGroup(size=tensor_parallel_size * pipeline_parallel_size),
        ep=_ConverterFakeProcessGroup(size=expert_parallel_size),
        expt_tp=_ConverterFakeProcessGroup(size=tensor_parallel_size),
        tp_ep=_ConverterFakeProcessGroup(size=tensor_parallel_size * expert_parallel_size),
        tp_ep_pp=_ConverterFakeProcessGroup(
            size=tensor_parallel_size * expert_parallel_size * pipeline_parallel_size
        ),
        dp=dp_group,
        dp_gtp_remat=dp_group,
        dp_cp=dp_group,
        dp_cp_gtp_remat=dp_group,
        intra_dp_cp=dp_group,
        expt_dp=expert_dp_group,
        expt_dp_gtp_remat=expert_dp_group,
        dynamic_cp={},
        dynamic_tp_cp={},
    )
    for field in fields(ProcessGroupCollection):
        groups.setdefault(field.name, None)
    return ProcessGroupCollection(**groups)


def install_checkpoint_converter_process_groups(parallel_state, pg_collection):
    """Mirror a converter's process groups into the global groups of ``parallel_state``.

    The converter passes ``pg_collection`` to the models it builds, but code that
    takes no collection still reads the global groups during conversion. For
    example, ``megatron.training.checkpointing`` names legacy checkpoint files
    after the global tensor- and pipeline-parallel ranks. Mirroring the same group
    objects keeps those reads consistent with the collection, including the ranks
    the converter sets on its groups. The global pipeline rank also needs
    ``set_checkpoint_converter_pipeline_rank``.

    Args:
        parallel_state: The ``megatron.core.parallel_state`` module.
        pg_collection: A collection built by ``build_checkpoint_converter_process_groups``.
    """
    parallel_state._TENSOR_MODEL_PARALLEL_GROUP = pg_collection.tp
    parallel_state._CONTEXT_PARALLEL_GROUP = pg_collection.cp
    parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = pg_collection.pp
    parallel_state._EXPERT_MODEL_PARALLEL_GROUP = pg_collection.ep
    parallel_state._DATA_PARALLEL_GROUP = pg_collection.dp
    parallel_state._DATA_PARALLEL_GROUP_WITH_CP = pg_collection.dp_cp
    parallel_state._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = pg_collection.intra_dp_cp
    parallel_state._DATA_PARALLEL_GROUP_WITH_GTP_REMAT = pg_collection.dp_gtp_remat
    parallel_state._DATA_PARALLEL_GROUP_WITH_CP_WITH_GTP_REMAT = pg_collection.dp_cp_gtp_remat
    # The collection has no intra-partial GTP fields; with GTP inactive they equal the
    # intra-partial data-parallel group and the expert data-parallel group.
    parallel_state._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_WITH_GTP_REMAT = (
        pg_collection.intra_dp_cp
    )
    parallel_state._EXPERT_DATA_PARALLEL_GROUP = pg_collection.expt_dp
    parallel_state._EXPERT_DATA_PARALLEL_GROUP_WITH_GTP_REMAT = pg_collection.expt_dp_gtp_remat
    parallel_state._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_WITH_GTP_REMAT = (
        pg_collection.expt_dp_gtp_remat
    )
    parallel_state._EXPERT_TENSOR_PARALLEL_GROUP = pg_collection.expt_tp
    parallel_state._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = pg_collection.tp_ep
    parallel_state._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = pg_collection.tp_ep_pp
    parallel_state._MODEL_PARALLEL_GROUP = pg_collection.mp


def set_checkpoint_converter_pipeline_rank(parallel_state, pg_collection, pipeline_rank):
    """Select the pipeline stage whose model the converter builds, loads or saves next.

    Models take their stage from the pipeline group of ``pg_collection``. Code that
    reads the global pipeline rank instead, such as legacy checkpoint names in
    ``megatron.training.checkpointing``, needs the ``parallel_state`` override too:
    ``get_pipeline_model_parallel_rank`` asks torch.distributed for the rank of its
    group, which a fake group cannot answer.

    Args:
        parallel_state: The ``megatron.core.parallel_state`` module.
        pg_collection: A collection built by ``build_checkpoint_converter_process_groups``.
        pipeline_rank: The pipeline-parallel rank to select.
    """
    pg_collection.pp.set_rank(pipeline_rank)
    parallel_state.set_pipeline_model_parallel_rank(pipeline_rank)


def initialize_checkpoint_converter_fake_process_groups(
    parallel_state, tensor_parallel_size, pipeline_parallel_size, expert_parallel_size
):
    """Build a converter's process groups and mirror them into ``parallel_state``.

    Converters that pass the returned collection to their models need the mirror
    only for code that still reads the global groups. See
    ``build_checkpoint_converter_process_groups`` and
    ``install_checkpoint_converter_process_groups``.

    Returns:
        ProcessGroupCollection: The fake groups now mirrored in ``parallel_state``.
    """
    pg_collection = build_checkpoint_converter_process_groups(
        tensor_parallel_size, pipeline_parallel_size, expert_parallel_size
    )
    install_checkpoint_converter_process_groups(parallel_state, pg_collection)
    return pg_collection
