# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise evaluate's global token reduction without a GPU or distributed backend."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.training import training as training_mod


@pytest.mark.parametrize("sft", [False, True])
@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_evaluate_weights_valid_tokens_across_batches_ranks_and_iterations(sft, scheduled, empty):
    args = SimpleNamespace(
        eval_global_batch_size=4,
        eval_micro_batch_size=1,
        data_parallel_size=2,
        gtp_weight_remat_size=1,
        cuda_graph_impl="none",
        moe_expert_rank_capacity_factor=None,
        seq_length=8,
        decoder_seq_length=None,
        eval_iters=2,
        empty_unused_memory_level=0,
        consumed_valid_samples=0,
        exit_duration_in_mins=None,
        sft=sft,
    )
    config = SimpleNamespace(sequence_packing_scheduler="default_dynamic_cp" if scheduled else None)
    groups = SimpleNamespace(pp=object(), dp_cp=object(), dp_cp_gtp_remat=object())
    # Local token counts differ between microbatches and evaluation iterations.
    local = [[(2, 1), (36, 9)], [(8, 2), (0, 0)]]
    if scheduled:
        local = [local[0] + [(0, 0)], local[1][:1]]
    counts = [len(step) for step in local]
    losses = [
        [
            {"lm loss": torch.tensor([0.0, 0.0] if empty else pair, dtype=torch.float32)}
            for pair in step
        ]
        for step in local
    ]
    remote = iter([torch.tensor([0.0, 0.0] if empty else pair) for pair in [(42, 6), (12, 2)]])
    tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs["device"] = "cpu"
        return tensor(*args, **kwargs)

    def reduce(value, group=None):
        assert group is groups.dp_cp_gtp_remat
        value.add_(next(remote))

    forward_backward = mock.Mock(side_effect=losses)
    model = mock.Mock()
    with (
        mock.patch.object(training_mod, "get_args", return_value=args),
        mock.patch.object(training_mod, "get_timers", return_value=mock.MagicMock()),
        mock.patch.object(training_mod, "get_rerun_state_machine", return_value=mock.Mock()),
        mock.patch.object(training_mod, "get_forward_backward_func", return_value=forward_backward),
        mock.patch.object(training_mod, "get_attr_wrapped_model", return_value=groups),
        mock.patch.object(training_mod, "is_pp_last_stage", return_value=True),
        mock.patch.object(training_mod, "has_nvidia_modelopt", False),
        mock.patch.object(training_mod, "ft_integration"),
        mock.patch.object(training_mod, "_otel_managed_span", return_value=mock.MagicMock()),
        mock.patch.object(
            training_mod,
            "wrap_data_iterator",
            side_effect=[(iter([]), count, 0, 0) for count in counts],
        ),
        mock.patch.object(torch, "tensor", side_effect=cpu_tensor),
        mock.patch.object(torch.distributed, "all_reduce", side_effect=reduce) as all_reduce,
    ):
        result, _, exited = training_mod.evaluate(
            lambda *a, **kw: None, iter([]), [model], None, config
        )

    # (2 + 36 + 8 + 42 + 12) / (1 + 9 + 2 + 6 + 2) == 5, not mean of means.
    torch.testing.assert_close(result["lm loss"], tensor(0.0 if empty else 5.0))
    assert all_reduce.call_count == 2
    assert [call.kwargs["num_microbatches"] for call in forward_backward.call_args_list] == counts
    assert args.consumed_valid_samples == 8
    assert not exited
    model.eval.assert_called_once()
    model.train.assert_called_once()
