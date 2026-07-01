# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Role-aware data iterator selection for heterogeneous MIMO training."""

from __future__ import annotations

import argparse
from typing import Optional

import torch

from examples.mimo.training.topology import HeteroTopology
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage


def select_data_iterator(args: argparse.Namespace, topology: HeteroTopology) -> Optional[object]:
    """Create the per-role data iterator this rank needs, or None if it consumes no data."""
    if args.dataset_provider != "mock":
        # Energon/mistral provider backends are deferred to a later PR.
        raise ValueError(f"unsupported dataset provider: {args.dataset_provider}")
    return select_mock_data_iterator(args, topology)


def select_mock_data_iterator(
    args: argparse.Namespace, topology: HeteroTopology
) -> Optional["MockVLMIterator"]:
    """Pick the mock iterator for this rank's role: encoder PP-first or language PP-edge stages."""
    encoder_name = _encoder_name(topology)
    llm_grid = topology.grids[MIMO_LANGUAGE_MODULE_KEY]
    llm_pgc = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
    llm_mbs = args.micro_batch_size

    llm_needs_data = llm_grid.is_current_rank_in_grid() and (
        is_pp_first_stage(llm_pgc.pp) or is_pp_last_stage(llm_pgc.pp)
    )

    if encoder_name is None:
        if llm_needs_data:
            return MockVLMIterator(
                args, llm_mbs, encoder_name, get_mock_data_seed(args, llm_pgc, 100_000)
            )
        return None

    encoder_grid = topology.grids[encoder_name]
    encoder_pgc = topology.module_pgs[encoder_name]
    if (args.micro_batch_size * args.llm_dp) % args.encoder_dp != 0:
        raise ValueError("micro_batch_size * llm_dp must be divisible by encoder_dp")
    encoder_mbs = args.micro_batch_size * args.llm_dp // args.encoder_dp

    encoder_needs_data = encoder_grid.is_current_rank_in_grid() and is_pp_first_stage(
        encoder_pgc.pp
    )

    # A rank in the language grid always feeds the language batch (PP-edge stages); a pure-encoder
    # rank feeds the encoder batch. Colocated ranks (in both) follow the language schedule.
    if llm_needs_data:
        return MockVLMIterator(
            args, llm_mbs, encoder_name, get_mock_data_seed(args, llm_pgc, 100_000)
        )
    if encoder_needs_data:
        return MockVLMIterator(
            args, encoder_mbs, encoder_name, get_mock_data_seed(args, encoder_pgc, 0)
        )
    return None


def get_mock_data_seed(args: argparse.Namespace, pg_collection, module_seed_offset: int) -> int:
    """Seed mock data per DP lane so PP/TP stages in a lane see coherent batches."""
    dp_lane = pg_collection.dp.rank() if pg_collection.dp is not None else 0
    return args.seed + module_seed_offset + dp_lane


def _encoder_name(topology: HeteroTopology) -> Optional[str]:
    """Return the single modality (encoder) grid name, or None for a language-only run."""
    modality = [name for name in topology.grids if name != MIMO_LANGUAGE_MODULE_KEY]
    return modality[0] if modality else None


class MockVLMIterator:
    """Infinite iterator yielding synthetic VLM-like next-token microbatches.

    Minimal self-contained mock so data selection is testable without a provider backend.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        micro_batch_size: int,
        encoder_name: Optional[str],
        seed: int,
    ) -> None:
        self.args = args
        self.micro_batch_size = micro_batch_size
        self.encoder_name = encoder_name
        self.image_seq_length = args.image_seq_length or args.seq_length // 2
        self.vision_encoder_key = getattr(args, "vision_encoder_key", "clip_encoder")
        self.dtype = torch.float32 if args.fp32 else torch.bfloat16
        self.generator = torch.Generator(device="cuda")
        self.generator.manual_seed(seed)
        if self.image_seq_length >= args.seq_length:
            raise ValueError("--image-seq-length must be smaller than --seq-length")

    def __iter__(self):
        return self

    def __next__(self):
        args = self.args
        image_tokens = torch.full(
            (self.micro_batch_size, self.image_seq_length),
            args.image_token_id,
            dtype=torch.long,
            device="cuda",
        )
        text_tokens = torch.randint(
            1,
            args.vocab_size,
            (self.micro_batch_size, args.seq_length - self.image_seq_length),
            device="cuda",
            generator=self.generator,
        )
        input_ids = torch.cat([image_tokens, text_tokens], dim=1)
        labels = torch.full_like(input_ids, -100)
        labels[:, :-1] = input_ids[:, 1:]
        labels[(labels == args.image_token_id)] = -100
        loss_mask = (labels != -100).to(dtype=torch.float32)
        encoder_hidden_states = torch.randn(
            self.image_seq_length,
            self.micro_batch_size,
            args.hidden_size,
            device="cuda",
            dtype=self.dtype,
            generator=self.generator,
        )
        modality_inputs = {}
        if self.encoder_name is not None:
            modality_inputs[self.encoder_name] = {
                self.vision_encoder_key: {
                    "hidden_states": encoder_hidden_states,
                    "attention_mask": None,
                }
            }
        return {
            "input_ids": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(args.seq_length, device="cuda")
            .unsqueeze(0)
            .expand(self.micro_batch_size, -1)
            .clone(),
            "modality_inputs": modality_inputs,
        }
