# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.distributed as dist

from .copy_services.base import CopyService
from .utils import ReshardPlan

logger = logging.getLogger(__name__)


def execute_reshard_plan(
    plan: ReshardPlan,
    src_module: torch.nn.Module,
    dst_module: torch.nn.Module,
    service: CopyService,
) -> None:
    """
    Execute a reshard plan (from centralized controller).
    A communication service must be provided to abstract transport.
    Expected service API: submit_send(tensor, dest_rank), submit_recv(tensor, src_rank), run().
    """

    src_params = {name: p for name, p in src_module.named_parameters(recurse=True)}
    dst_params = {name: p for name, p in dst_module.named_parameters(recurse=True)}
    submit_send_with_id = getattr(service, "submit_send_with_id", None)
    submit_recv_with_id = getattr(service, "submit_recv_with_id", None)

    # Submit sends
    for op in plan.send_ops:
        src_param = src_params.get(op.param_name)
        if src_param is not None:
            src_view = src_param.data[op.my_slice].contiguous()
            if submit_send_with_id is not None and op.task_id is not None:
                submit_send_with_id(op.task_id, src_view, op.peer_rank)
            else:
                service.submit_send(src_view, op.peer_rank)

    # Submit recvs
    recv_writebacks: List[Tuple[torch.Tensor, torch.nn.Parameter, tuple[slice, ...]]] = []
    for op in plan.recv_ops:
        dst_param = dst_params.get(op.param_name)
        if dst_param is not None:
            dst_slice_view = dst_param.data[op.my_slice]
            recv_buffer = torch.empty_like(dst_slice_view.contiguous())
            if submit_recv_with_id is not None and op.task_id is not None:
                submit_recv_with_id(op.task_id, recv_buffer, op.peer_rank)
            else:
                service.submit_recv(recv_buffer, op.peer_rank)
            recv_writebacks.append((recv_buffer, dst_param, op.my_slice))

    # Execute
    logger.info(f"Executing {len(plan.send_ops)} sends + {len(plan.recv_ops)} recvs")
    service.run()
    dist.barrier()

    # Write back received buffers into their destination parameter slices
    for recv_buffer, dst_param, dst_slice in recv_writebacks:
        with torch.no_grad():
            dst_param.data[dst_slice].copy_(recv_buffer)

    logger.info("Reshard complete")
