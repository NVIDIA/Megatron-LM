# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, get_mtp_metric_slots
from tests.unit_tests.test_utilities import Utils


def test_pipeline_report_uses_matching_hybrid_mtp_buffers():
    Utils.initialize_distributed()
    tracker = MoEMetricsTracker()
    metric_name = "seq_load_balancing_loss"
    config = SimpleNamespace(mtp_num_layers=1, mtp_hybrid_override_pattern="*E")
    num_layers = 88 + get_mtp_metric_slots(config)

    assert num_layers == 90
    tracker.record(
        metric_name, torch.tensor(1.0, device="cuda"), layer_number=1, num_layers=num_layers
    )
    if Utils.rank > 0:
        tracker.record(
            metric_name, torch.tensor(2.0, device="cuda"), layer_number=90, num_layers=num_layers
        )

    world_group = torch.distributed.new_group()
    try:
        pg_collection = type(
            "ProcessGroups", (), {"pp": world_group, "dp_cp_gtp_remat": world_group}
        )()
        tracker.report(
            loss_scale=1.0,
            iteration=1,
            force_initialize=True,
            track_names=[metric_name],
            num_layers=num_layers,
            num_moe_layers=2,
            pg_collection=pg_collection,
        )
    finally:
        torch.distributed.destroy_process_group(world_group)

    assert tracker.metrics[metric_name].values.shape == (90,)
