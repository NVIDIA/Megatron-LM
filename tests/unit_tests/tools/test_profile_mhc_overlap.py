# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse

import pytest

from tools import profile_mhc_overlap


@pytest.mark.parametrize(
    "controlled_argument", sorted(profile_mhc_overlap.CONTROLLED_TRAINING_ARGS)
)
@pytest.mark.parametrize("use_equals_syntax", (False, True))
def test_rejects_controlled_trailing_training_arguments(controlled_argument, use_equals_syntax):
    argument = f"{controlled_argument}=override" if use_equals_syntax else controlled_argument

    with pytest.raises(ValueError, match="profiling wrapper controls"):
        profile_mhc_overlap.parse_args(["--dry-run", "--", argument])


def test_allows_model_shape_and_batch_overrides():
    args = profile_mhc_overlap.parse_args(
        [
            "--dry-run",
            "--",
            "--seq-length",
            "2048",
            "--max-position-embeddings",
            "2048",
            "--global-batch-size=32",
        ]
    )

    assert args.training_args == [
        "--seq-length",
        "2048",
        "--max-position-embeddings",
        "2048",
        "--global-batch-size=32",
    ]


def test_generated_command_uses_registered_megatron_arguments():
    from megatron.training.arguments import add_megatron_arguments

    args = profile_mhc_overlap.parse_args(["--preset", "h100-nccl", "--dry-run"])
    preset = profile_mhc_overlap.PRESETS[args.preset]
    command = profile_mhc_overlap._torchrun_command(
        args, preset, profile_mhc_overlap.RunCase("all", high_priority_comm=True), profile_ranks=[0]
    )
    assert command.count("--high-priority-a2a-comm-stream") == 1
    script_index = next(
        index for index, argument in enumerate(command) if argument.endswith("pretrain_gpt.py")
    )
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_megatron_arguments(parser)

    parsed = parser.parse_args(command[script_index + 1 :])

    assert parsed.mhc_high_priority_stream_mode == "all"
    assert parsed.mhc_recompute_layer_num == 2
    assert parsed.enable_hyper_connections
    assert parsed.high_priority_a2a_comm_stream
    assert parsed.nvtx_ranges
    assert parsed.profile_ranks == [0]
