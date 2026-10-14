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


def initialize_checkpoint_converter_fake_process_groups(
    parallel_state, tensor_parallel_size, pipeline_parallel_size, expert_parallel_size
):
    """Initialize the process-group globals used by checkpoint converters.

    Checkpoint conversion runs in one process and does not call
    ``initialize_model_parallel``. Model construction still resolves the full
    ``ProcessGroupCollection``, so populate its parallel-state dependencies with
    lightweight groups that report the requested target sizes.

    GTP rematerialization is not active during conversion. Its data-parallel
    groups therefore alias the corresponding non-GTP groups, matching
    ``initialize_model_parallel`` when the GTP axes have size one.
    """
    fake_tp_group = _ConverterFakeProcessGroup(size=tensor_parallel_size)
    fake_pp_group = _ConverterFakeProcessGroup(size=pipeline_parallel_size)
    fake_ep_group = _ConverterFakeProcessGroup(size=expert_parallel_size)
    fake_dp_group = _ConverterFakeProcessGroup(size=1)
    fake_expert_dp_group = _ConverterFakeProcessGroup(size=1)
    fake_expert_tp_group = _ConverterFakeProcessGroup(size=tensor_parallel_size)
    fake_expert_tp_ep_group = _ConverterFakeProcessGroup(
        size=tensor_parallel_size * expert_parallel_size
    )
    fake_expert_tp_ep_pp_group = _ConverterFakeProcessGroup(
        size=tensor_parallel_size * expert_parallel_size * pipeline_parallel_size
    )
    fake_model_parallel_group = _ConverterFakeProcessGroup(
        size=tensor_parallel_size * pipeline_parallel_size
    )

    parallel_state._TENSOR_MODEL_PARALLEL_GROUP = fake_tp_group
    parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = fake_pp_group
    parallel_state._EXPERT_MODEL_PARALLEL_GROUP = fake_ep_group
    parallel_state._DATA_PARALLEL_GROUP = fake_dp_group
    parallel_state._DATA_PARALLEL_GROUP_WITH_CP = fake_dp_group
    parallel_state._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = fake_dp_group
    parallel_state._DATA_PARALLEL_GROUP_WITH_GTP_REMAT = fake_dp_group
    parallel_state._DATA_PARALLEL_GROUP_WITH_CP_WITH_GTP_REMAT = fake_dp_group
    parallel_state._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_WITH_GTP_REMAT = fake_dp_group
    parallel_state._EXPERT_DATA_PARALLEL_GROUP = fake_expert_dp_group
    parallel_state._EXPERT_DATA_PARALLEL_GROUP_WITH_GTP_REMAT = fake_expert_dp_group
    parallel_state._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_WITH_GTP_REMAT = (
        fake_expert_dp_group
    )
    parallel_state._EXPERT_TENSOR_PARALLEL_GROUP = fake_expert_tp_group
    parallel_state._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = fake_expert_tp_ep_group
    parallel_state._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = fake_expert_tp_ep_pp_group
    parallel_state._MODEL_PARALLEL_GROUP = fake_model_parallel_group
