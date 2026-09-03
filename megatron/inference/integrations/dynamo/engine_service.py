# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed Megatron engine service for Dynamo."""

from __future__ import annotations

import asyncio

import torch
import torch.distributed as dist

from megatron.core.inference.engine_endpoint import InferenceEngineEndpoint
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
from megatron.core.utils import get_pg_size
from megatron.inference.integrations.dynamo.args import add_engine_service_args
from megatron.inference.integrations.dynamo.dynamic_engine import DynamoDynamicInferenceEngine
from megatron.inference.integrations.dynamo.telemetry import EngineEventReporter
from megatron.inference.utils import add_inference_args, get_dynamic_inference_engine
from megatron.post_training.arguments import add_modelopt_args
from megatron.training import get_args
from megatron.training.arguments import parse_and_validate_args
from megatron.training.initialize import initialize_megatron


def _extra_args(parser):
    parser = add_inference_args(add_modelopt_args(parser))
    parser.add_argument("--dynamo-parent-event-address", required=True)
    return add_engine_service_args(parser)


async def _serve() -> None:
    args = get_args()
    args.return_log_probs = True
    disaggregated = args.role in ("prefill", "decode")
    engine_class = DynamoDynamicInferenceEngine if disaggregated else DynamicInferenceEngine
    engine = get_dynamic_inference_engine(engine_class=engine_class)

    replica_group = (
        engine.pg_collection.expt_dp
        if args.expert_model_parallel_size > 1
        else engine.pg_collection.dp
    )
    replica_count = get_pg_size(replica_group)
    if replica_count != 1:
        raw_dp_size = get_pg_size(engine.pg_collection.dp)
        raise ValueError(
            "Megatron Dynamo service requires one complete model replica; "
            f"got logical DP={replica_count}, regular DP={raw_dp_size}, "
            f"EP={args.expert_model_parallel_size}"
        )

    if disaggregated:
        engine.setup_kv_transfer(role=args.role)

    reporter = EngineEventReporter(engine, args.dynamo_parent_event_address)
    reporter.start()

    try:
        address = await engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=args.coordinator_port, hostname=args.coordinator_host
        )
        if dist.get_rank() == 0:
            endpoint = InferenceEngineEndpoint.from_engine(
                address, engine, logical_data_parallel_size=replica_count
            )
            reporter.observe("ready", endpoint.to_dict())
        await engine.engine_loop_task
    finally:
        reporter.stop()


def main() -> None:
    parse_and_validate_args(
        extra_args_provider=_extra_args, args_defaults={"no_load_rng": True, "no_load_optim": True}
    )
    initialize_megatron()
    try:
        with torch.inference_mode():
            asyncio.run(_serve())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
