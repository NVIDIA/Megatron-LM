# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Full-validation iteration counts must broadcast as a vector for one or more sets."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core import parallel_state
from megatron.training import training
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("multiple", [False, True])
def test_full_validation_broadcast_preserves_dataset_iteration_counts(multiple: bool) -> None:
    """Every TP rank must call the evaluator with the source's per-dataset counts."""
    Utils.initialize_model_parallel(2, 1)
    try:
        expected = [2, 3] if multiple else [3]
        source = parallel_state.get_tensor_model_parallel_rank() == 0
        counts: list[int | None] = list(expected) if source else [None] * len(expected)
        args = SimpleNamespace(
            multiple_validation_sets=multiple,
            full_validation=True,
            eval_iters=counts if multiple else counts[0],
            validation_set_names=None,
        )
        iterator = [iter(()), iter(())] if multiple else iter(())
        with (
            patch.object(training, "get_args", return_value=args),
            patch.object(training, "get_wandb_writer", return_value=None),
            patch.object(training, "print_rank_last"),
            patch.object(training, "evaluate", return_value=({}, None, False)) as evaluate,
        ):
            training.evaluate_and_print_results(
                "test", None, iterator, [], 0, None, None, write_to_tensorboard=False
            )
        assert [call.kwargs["eval_iters"] for call in evaluate.call_args_list] == expected
        assert args.eval_iters == (expected if multiple else expected[0])
    finally:
        Utils.destroy_model_parallel()
