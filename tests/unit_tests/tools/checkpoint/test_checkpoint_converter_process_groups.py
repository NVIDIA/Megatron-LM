# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from tools.checkpoint.utils import initialize_checkpoint_converter_fake_process_groups

_FAKE_GROUP_GLOBALS = (
    '_TENSOR_MODEL_PARALLEL_GROUP',
    '_PIPELINE_MODEL_PARALLEL_GROUP',
    '_EXPERT_MODEL_PARALLEL_GROUP',
    '_DATA_PARALLEL_GROUP',
    '_DATA_PARALLEL_GROUP_WITH_CP',
    '_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP',
    '_DATA_PARALLEL_GROUP_WITH_GTP_REMAT',
    '_DATA_PARALLEL_GROUP_WITH_CP_WITH_GTP_REMAT',
    '_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_WITH_GTP_REMAT',
    '_EXPERT_DATA_PARALLEL_GROUP',
    '_EXPERT_DATA_PARALLEL_GROUP_WITH_GTP_REMAT',
    '_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_WITH_GTP_REMAT',
    '_EXPERT_TENSOR_PARALLEL_GROUP',
    '_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP',
    '_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP',
    '_MODEL_PARALLEL_GROUP',
)


def test_fake_groups_support_default_process_group_collection():
    original_groups = {name: getattr(parallel_state, name) for name in _FAKE_GROUP_GLOBALS}
    try:
        initialize_checkpoint_converter_fake_process_groups(
            parallel_state, tensor_parallel_size=4, pipeline_parallel_size=2, expert_parallel_size=1
        )

        groups = ProcessGroupCollection.use_mpu_process_groups()

        assert groups.tp.size() == 4
        assert groups.pp.size() == 2
        assert groups.ep.size() == 1
        assert groups.dp.size() == 1
        assert groups.dp_cp.size() == 1
        assert groups.dp_cp_gtp_remat.size() == 1
        assert groups.intra_dp_cp.size() == 1
        assert groups.expt_dp.size() == 1
        assert groups.expt_dp_gtp_remat.size() == 1
    finally:
        for name, group in original_groups.items():
            setattr(parallel_state, name, group)
