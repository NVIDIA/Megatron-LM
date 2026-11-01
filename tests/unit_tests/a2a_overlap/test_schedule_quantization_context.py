# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core import fp8_utils
from megatron.core.enums import Fp8Recipe
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.common.model_chunk_schedule_plan import TransformerLayerSchedulePlan
from megatron.core.pipeline_parallel import combined_1f1b, schedules
from megatron.core.pipeline_parallel.utils import AbstractSchedulePlan
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.amax_reduction_group_utils import (
    forbid_global_amax_group,
    global_amax_group,
    record_amax_groups,
    with_copied_amax_groups,
)
from tests.unit_tests.test_utilities import Utils


def _schedule_plan(layer):
    plan = TransformerLayerSchedulePlan.__new__(TransformerLayerSchedulePlan)
    plan.layer = layer
    return plan


def test_schedule_uses_layer_quantization_context():
    expected_context = nullcontext()
    layer = SimpleNamespace(get_inner_quantization_context=Mock(return_value=expected_context))

    context = _schedule_plan(layer).get_low_precision_context()

    assert context is expected_context
    layer.get_inner_quantization_context.assert_called_once_with()


def test_transformer_layer_uses_fp4_context():
    config = SimpleNamespace(fp8=None, fp8_recipe=Fp8Recipe.delayed, fp4="e2m1")
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.layer_number = 3
    layer.pg_collection = ProcessGroupCollection()
    expected_context = nullcontext()

    with patch(
        "megatron.core.fp4_utils.get_fp4_context", return_value=expected_context
    ) as get_fp4_context:
        context = layer.get_inner_quantization_context()

    assert context is expected_context
    get_fp4_context.assert_called_once_with(config, 2, pg_collection=layer.pg_collection)


def test_transformer_layer_uses_fp8_context():
    config = SimpleNamespace(fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise, fp4=None)
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.layer_number = 3
    layer.pg_collection = ProcessGroupCollection()
    expected_context = nullcontext()

    with patch(
        "megatron.core.fp8_utils.get_fp8_context", return_value=expected_context
    ) as get_fp8_context:
        context = layer.get_inner_quantization_context()

    assert context is expected_context
    get_fp8_context.assert_called_once_with(config, 2, pg_collection=layer.pg_collection)


def test_mtp_layer_uses_global_fp8_context():
    config = SimpleNamespace(fp8="e4m3", fp8_recipe=Fp8Recipe.tensorwise, fp4=None)
    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.layer_number = 1
    layer.pg_collection = ProcessGroupCollection()
    expected_context = nullcontext()

    with patch(
        "megatron.core.transformer.multi_token_prediction.get_fp8_context",
        return_value=expected_context,
    ) as get_fp8_context:
        context = layer.get_inner_quantization_context()

    assert context is expected_context
    get_fp8_context.assert_called_once_with(config, pg_collection=layer.pg_collection)


def test_mtp_layer_uses_fp4_context():
    config = SimpleNamespace(fp8=None, fp8_recipe=Fp8Recipe.delayed, fp4="e2m1")
    layer = MultiTokenPredictionLayer.__new__(MultiTokenPredictionLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.pg_collection = ProcessGroupCollection()
    expected_context = nullcontext()

    with patch(
        "megatron.core.transformer.multi_token_prediction.get_fp4_context",
        return_value=expected_context,
    ) as get_fp4_context:
        context = layer.get_inner_quantization_context()

    assert context is expected_context
    # MTP layer numbers are depth-local, so no decoder layer index is passed.
    get_fp4_context.assert_called_once_with(config, pg_collection=layer.pg_collection)


class _NoOpSchedulePlan(AbstractSchedulePlan):
    """Stands in for a model chunk's schedule plan; running it computes nothing."""

    event = None

    @staticmethod
    def run(f_schedule_plan, b_schedule_plan, b_grad=None, **kwargs):
        """Return a placeholder forward output."""
        return torch.zeros(1)


class _ModelChunk(torch.nn.Module):
    """Stands in for a GPT model chunk that owns a process-group collection."""

    def __init__(self, pg_collection):
        super().__init__()
        self.pg_collection = pg_collection

    def set_input_tensor(self, input_tensor):
        """Accept the pipeline input, as a model chunk does."""

    def build_schedule_plan(self):
        """Return a schedule plan that computes nothing."""
        return _NoOpSchedulePlan()


class _Wrapper(torch.nn.Module):
    """Stands in for a DDP wrapper, which holds no collection of its own."""

    def __init__(self, module):
        super().__init__()
        self.module = module


def _forward_step(data_iterator, model, return_schedule_plan=False):
    """Build the chunk's schedule plan, as a GPT forward_step_func does for combined 1F1B."""
    return model.build_schedule_plan(), Mock()


def _run_combined_step(config, f_model=None, b_model=None):
    """Run one combined 1F1B step whose schedule plans and loss computation do nothing."""
    b_output_tensor = torch.zeros(1, requires_grad=True)
    b_output_tensor.schedule_plan = _NoOpSchedulePlan()
    b_output_tensor.loss_func = None
    with patch.object(
        schedules, "forward_step_calc_loss", return_value=(torch.zeros(1), torch.ones(1))
    ):
        combined_1f1b.combined_forward_backward_step(
            forward_step_func=_forward_step,
            data_iterator=None,
            f_model=f_model,
            num_microbatches=1,
            input_tensor=None,
            forward_data_store=[],
            b_model=b_model,
            b_input_tensor=torch.zeros(1, requires_grad=True),
            b_output_tensor=b_output_tensor,
            b_output_tensor_grad=torch.ones(1),
            config=config,
            cp_group_size=1,
            is_last_stage=True,
        )


def _fp8_schedule_config(fp8_recipe):
    return SimpleNamespace(enable_autocast=False, timers=None, fp8="e4m3", fp8_recipe=fp8_recipe)


@pytest.mark.parametrize("wrapped", [False, True], ids=["bare", "wrapped"])
def test_combined_1f1b_delayed_fp8_context_uses_forward_chunk_collection(wrapped):
    """In an interleaved step the forward chunk's modules run inside the outer context."""
    f_collection, b_collection = ProcessGroupCollection(), ProcessGroupCollection()
    f_model, b_model = _ModelChunk(f_collection), _ModelChunk(b_collection)
    if wrapped:
        f_model, b_model = _Wrapper(f_model), _Wrapper(b_model)
    config = _fp8_schedule_config(Fp8Recipe.delayed)

    with patch.object(
        combined_1f1b, "get_fp8_context", return_value=nullcontext()
    ) as get_fp8_context:
        _run_combined_step(config, f_model=f_model, b_model=b_model)

    get_fp8_context.assert_called_once()
    assert get_fp8_context.call_args.args == (config,)
    # Empty collections compare equal, so check identity.
    assert get_fp8_context.call_args.kwargs.get("pg_collection") is f_collection


def test_combined_1f1b_backward_only_delayed_fp8_context_uses_backward_chunk_collection():
    b_collection = ProcessGroupCollection()
    config = _fp8_schedule_config(Fp8Recipe.delayed)

    with patch.object(
        combined_1f1b, "get_fp8_context", return_value=nullcontext()
    ) as get_fp8_context:
        _run_combined_step(config, b_model=_Wrapper(_ModelChunk(b_collection)))

    get_fp8_context.assert_called_once()
    assert get_fp8_context.call_args.args == (config,)
    assert get_fp8_context.call_args.kwargs.get("pg_collection") is b_collection


def test_combined_1f1b_opens_no_outer_fp8_context_for_per_layer_recipes():
    """The layers open their own contexts for recipes other than delayed scaling."""
    model = _ModelChunk(ProcessGroupCollection())

    with patch.object(combined_1f1b, "get_fp8_context") as get_fp8_context:
        _run_combined_step(_fp8_schedule_config(Fp8Recipe.tensorwise), f_model=model, b_model=model)

    get_fp8_context.assert_not_called()


@pytest.mark.skipif(not fp8_utils.HAVE_TE, reason="requires Transformer Engine")
class TestCombined1F1BDelayedScalingAmaxGroup:
    """The outer delayed-scaling context of combined 1F1B reduces amaxes over the model's group."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("backward_only", [False, True], ids=["fwd-bwd", "bwd-only"])
    @pytest.mark.parametrize("tp_only_amax_red", [False, True])
    def test_reduces_over_the_model_collection_group(self, tp_only_amax_red, backward_only):
        Utils.initialize_model_parallel(tensor_model_parallel_size=min(2, Utils.world_size))
        # Same ranks as the global groups but distinct communicators, so reading the global
        # group fails the identity check below even where the ranks agree.
        pg_collection = with_copied_amax_groups()
        expected = pg_collection.tp_cp if tp_only_amax_red else pg_collection.tp_dp_cp
        assert expected is not global_amax_group(tp_only_amax_red)
        model = _Wrapper(_ModelChunk(pg_collection))
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            fp8="e4m3",
            fp8_recipe=Fp8Recipe.delayed,
            tp_only_amax_red=tp_only_amax_red,
        )

        with forbid_global_amax_group(), record_amax_groups() as groups:
            _run_combined_step(config, f_model=None if backward_only else model, b_model=model)

        assert len(groups) == 1 and groups[0] is expected

    def test_custom_grid_differs_from_global_grid(self):
        """A model on its own grid reduces over its own TP x DP x CP ranks."""
        world_size = Utils.world_size
        if world_size < 4 or world_size % 2:
            pytest.skip("needs an even number of ranks, at least 4")
        # Global grid: TP=1 and PP=2, so the global amax group is the DP group of a stage.
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        # Custom grid: PP is the fastest dimension, so its TP groups cross the global stages.
        grid = HyperCommGrid([2, world_size // 2, 1, 1], ["pp", "tp", "cp", "dp"])
        pg_collection = ProcessGroupCollection(
            tp_cp=grid.create_pg(["tp", "cp"]), tp_dp_cp=grid.create_pg(["tp", "cp", "dp"])
        )
        custom_ranks = list(range(torch.distributed.get_rank() % 2, world_size, 2))
        global_ranks = torch.distributed.get_process_group_ranks(global_amax_group(False))
        assert custom_ranks != global_ranks
        model = _Wrapper(_ModelChunk(pg_collection))
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            fp8="e4m3",
            fp8_recipe=Fp8Recipe.delayed,
        )

        with record_amax_groups() as groups:
            _run_combined_step(config, f_model=model, b_model=model)

        assert len(groups) == 1
        assert torch.distributed.get_process_group_ranks(groups[0]) == custom_ranks
