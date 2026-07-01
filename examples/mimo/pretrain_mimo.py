# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Heterogeneous Nemotron6-MoE VLM training through the stock pretrain loop."""

from __future__ import annotations

import argparse

from examples.mimo.model_providers import resolve_provider
from examples.mimo.model_providers.nemotron_moe_vlm import add_model_provider_args
from examples.mimo.training.args import (
    add_hetero_grid_args,
    build_module_grid_specs,
    validate_hetero_grid_args,
)
from examples.mimo.training.builder import MimoBuildConfig
from examples.mimo.training.data import add_mock_data_args, build_train_valid_test_data_loaders
from examples.mimo.training.distributed import initialize_distributed, shutdown_distributed
from examples.mimo.training.step import mimo_forward_step
from examples.mimo.training.topology import create_topology
from megatron.core.enums import ModelType
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_global_variables
from megatron.training.training import pretrain
from megatron.training.vocab_utils import calculate_padded_vocab_size


def extra_args_provider(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register model-provider, heterogeneous-grid, and mock-data arguments."""
    parser = add_model_provider_args(parser)
    parser = add_hetero_grid_args(parser)
    parser = add_mock_data_args(parser)
    return parser


def _parse_and_validate() -> argparse.Namespace:
    """Parse stock plus MIMO arguments and validate the disjoint module grids."""
    args = parse_args(extra_args_provider)
    validate_hetero_grid_args(args, args.world_size)
    physical_world_size = args.world_size
    # Stock validate_args expects a single-module world; the language grid stands in for it.
    args.world_size = (
        args.llm_dp
        * args.tensor_model_parallel_size
        * args.pipeline_model_parallel_size
        * args.context_parallel_size
    )
    try:
        validate_args(args, {"dataloader_type": "external"})
    finally:
        args.world_size = physical_world_size
    if not args.use_distributed_optimizer:
        raise ValueError("heterogeneous MIMO training requires --use-distributed-optimizer")

    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = calculate_padded_vocab_size(
            args.vocab_size, args.make_vocab_size_divisible_by, args.llm_tp, logging_enabled=False
        )
    return args


def main() -> None:
    """Build the heterogeneous topology and run stock pretraining."""
    args = _parse_and_validate()
    set_global_variables(args, build_tokenizer=False)
    provider = resolve_provider(args)

    topology = None
    try:
        initialize_distributed()
        # The grid/rank-layout args model a single encoder region; the builder itself is
        # generic over any number of encoder grids in the topology.
        encoder_name = provider.encoder_module_names[0] if provider.encoder_module_names else None
        specs = build_module_grid_specs(args, args.world_size, encoder_name)
        topology = create_topology(specs)

        communicator = provider.build_communicator(args, topology)

        loaders = build_train_valid_test_data_loaders(args, topology)
        iterators = tuple(iter(loader) if loader is not None else None for loader in loaders)

        model_cfg = MimoBuildConfig(_topology=topology)
        cfg = pretrain_cfg_container_from_args(args, model_cfg)

        def train_valid_test_data_provider(_train_val_test_num_samples):
            return iterators

        train_valid_test_data_provider.is_distributed = True
        pretrain(
            cfg,
            train_valid_test_data_provider,
            ModelType.encoder_or_decoder,
            mimo_forward_step,
            model_provider=None,
            skip_model_parallel_init=True,
            p2p_communicator=communicator,
            pg_collection=topology.schedule_pg_collection,
        )
    finally:
        try:
            if topology is not None:
                topology.destroy()
        finally:
            shutdown_distributed()


if __name__ == "__main__":
    main()
