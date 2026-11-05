# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tensor-parallel helpers warn when they fall back to the global tensor-parallel group."""

import warnings

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.tensor_parallel import cross_entropy
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    _initialize_affine_weight_cpu,
)
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from megatron.core.tensor_parallel.random import checkpoint, model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from tests.unit_tests.test_utilities import Utils

HELPER = "get_tensor_model_parallel_group_if_none"
VERSIONS = "deprecated since Megatron Core 0.21 and will be removed in 0.23"

# The accessors behind the fallbacks of get_tensor_model_parallel_group_if_none and
# vocab_parallel_cross_entropy.
_GLOBAL_TP_ACCESSORS = (
    "get_tensor_model_parallel_group",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
    "get_expert_tensor_parallel_group",
)


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _fallback_warnings(record):
    return [w for w in record if issubclass(w.category, ProcessGroupFallbackWarning)]


def _forbid_global_tp_group(monkeypatch):
    """Make every read of the global tensor-parallel groups raise."""

    def forbid(*args, **kwargs):
        raise AssertionError("read the global tensor-parallel group")

    for name in _GLOBAL_TP_ACCESSORS:
        monkeypatch.setattr(parallel_state, name, forbid)
    monkeypatch.setattr(cross_entropy, "get_tensor_model_parallel_group", forbid)


def _distinct_tp_group():
    """A communicator with this rank's global TP ranks that is not the global TP group."""
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    world_size = torch.distributed.get_world_size()
    ranks = [list(range(start, start + tp_size)) for start in range(0, world_size, tp_size)]
    group, _ = torch.distributed.new_subgroups_by_enumeration(ranks)
    assert group is not parallel_state.get_tensor_model_parallel_group()
    assert torch.distributed.get_process_group_ranks(
        group
    ) == torch.distributed.get_process_group_ranks(parallel_state.get_tensor_model_parallel_group())
    return group


def _config(**kwargs):
    return TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        tensor_model_parallel_size=2,
        **kwargs,
    )


@pytest.mark.skipif(Utils.world_size < 2, reason="needs at least 2 ranks for TP=2")
class TestTensorModelParallelGroupIfNone:
    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_fallback_warns_once_per_calling_function(self):
        def dense_caller():
            return get_tensor_model_parallel_group_if_none(None)

        def expert_caller():
            return get_tensor_model_parallel_group_if_none(None, is_expert=True)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(3):
                assert dense_caller() is parallel_state.get_tensor_model_parallel_group()
                assert expert_caller() is parallel_state.get_expert_tensor_parallel_group()

        fallbacks = _fallback_warnings(record)
        assert len(fallbacks) == 2
        for warning, caller in zip(fallbacks, (dense_caller, expert_caller)):
            message = str(warning.message)
            assert f"{HELPER} (from {caller.__qualname__}) was called without `tp_group`" in message
            assert VERSIONS in message
            # The warning points at the code outside Megatron Core that reached the fallback.
            assert warning.filename == __file__

    def test_fallback_names_the_function_that_omitted_the_group(self):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            ColumnParallelLinear(
                8, 8, config=_config(), init_method=torch.nn.init.zeros_, bias=False
            )
            copy_to_tensor_model_parallel_region(torch.ones(2, device="cuda"))

        messages = [str(w.message) for w in _fallback_warnings(record)]
        assert len(messages) == 2
        assert f"{HELPER} (from ColumnParallelLinear.__init__) was called" in messages[0]
        assert f"{HELPER} (from copy_to_tensor_model_parallel_region) was called" in messages[1]
        assert all(w.filename == __file__ for w in _fallback_warnings(record))

    def test_explicit_group_never_warns_or_reads_the_global_group(self, monkeypatch):
        group = _distinct_tp_group()
        _forbid_global_tp_group(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            assert get_tensor_model_parallel_group_if_none(group) is group
            assert get_tensor_model_parallel_group_if_none(group, is_expert=True) is group
            output = copy_to_tensor_model_parallel_region(torch.ones(2, device="cuda"), group=group)
        assert torch.equal(output, torch.ones(2, device="cuda"))

    def test_strict_mode_raises_on_every_fallback(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            for _ in range(2):
                with pytest.raises(ProcessGroupFallbackWarning, match=f"{HELPER} \\(from "):
                    get_tensor_model_parallel_group_if_none(None)

    def test_no_warning_without_model_parallel_state(self):
        # Nothing is read from parallel_state, so nothing is deprecated: None passes through.
        Utils.destroy_model_parallel()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            assert get_tensor_model_parallel_group_if_none(None) is None


@pytest.mark.skipif(Utils.world_size < 2, reason="needs at least 2 ranks for TP=2")
class TestVocabParallelCrossEntropyFallback:
    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _inputs():
        generator = torch.Generator(device="cuda").manual_seed(1234)
        logits = torch.randn(5, 3, 8, device="cuda", generator=generator)
        target = torch.randint(0, 16, (5, 3), device="cuda", generator=generator)
        return logits, target

    def test_omitted_tp_group_warns_once(self):
        logits, target = self._inputs()
        expected = vocab_parallel_cross_entropy(
            logits.clone(), target, tp_group=parallel_state.get_tensor_model_parallel_group()
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            losses = [vocab_parallel_cross_entropy(logits.clone(), target) for _ in range(2)]

        fallbacks = _fallback_warnings(record)
        assert len(fallbacks) == 1
        message = str(fallbacks[0].message)
        assert "vocab_parallel_cross_entropy was called without `tp_group`" in message
        assert VERSIONS in message
        for loss in losses:
            assert torch.equal(loss, expected)

    def test_explicit_tp_group_never_warns_or_reads_the_global_group(self, monkeypatch):
        logits, target = self._inputs()
        expected = vocab_parallel_cross_entropy(
            logits.clone(), target, tp_group=parallel_state.get_tensor_model_parallel_group()
        )
        group = _distinct_tp_group()
        _forbid_global_tp_group(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            loss = vocab_parallel_cross_entropy(logits.clone(), target, tp_group=group)
        assert torch.equal(loss, expected)

    def test_strict_mode_raises_on_every_fallback(self):
        logits, target = self._inputs()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            for _ in range(2):
                with pytest.raises(
                    ProcessGroupFallbackWarning,
                    match="vocab_parallel_cross_entropy was called without `tp_group`",
                ):
                    vocab_parallel_cross_entropy(logits.clone(), target)


def test_initialize_affine_weight_cpu_takes_the_tp_rank_from_its_caller():
    master_weight = torch.arange(64, dtype=torch.float32).view(8, 8)

    def init_method(weight):
        weight.copy_(master_weight)

    with pytest.raises(TypeError, match="rank"):
        _initialize_affine_weight_cpu(torch.empty(4, 8), 8, 8, 4, 0, init_method)

    for rank in range(2):
        weight = torch.empty(4, 8)
        _initialize_affine_weight_cpu(
            weight,
            8,
            8,
            4,
            0,
            init_method,
            rank=rank,
            world_size=2,
            skip_set_tensor_parallel_attributes=True,
        )
        assert torch.equal(weight, master_weight[4 * rank : 4 * (rank + 1)])


@pytest.mark.skipif(Utils.world_size < 2, reason="needs at least 2 ranks for TP=2")
class TestTensorParallelCallersPassTheirGroup:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_row_parallel_linear_forward_uses_its_group(self, monkeypatch):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        group = _distinct_tp_group()
        layer = RowParallelLinear(
            8,
            4,
            config=_config(),
            init_method=torch.nn.init.normal_,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            tp_group=group,
        ).cuda()
        inputs = torch.randn(3, 2, 4, device="cuda", requires_grad=True)

        _forbid_global_tp_group(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            output, _ = layer(inputs)
            output.sum().backward()
        assert inputs.grad is not None

    def test_checkpoint_splits_the_saved_input_over_the_given_group(self, monkeypatch):
        # The global TP group has one rank; tp_group spans the world, so a global read would
        # keep the whole input instead of 1 / world_size of it.
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        group = torch.distributed.new_group(ranks=list(range(Utils.world_size)))

        def forward(first, second):
            return first + second

        first = torch.ones((4, 4), device="cuda", requires_grad=True)
        second = torch.full((4, 4), 2.0, device="cuda")

        _forbid_global_tp_group(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            output = checkpoint(forward, True, first, second, tp_group=group)
            assert first.data.shape == (16 // Utils.world_size,)
            output.sum().backward()
        assert torch.equal(output, torch.full((4, 4), 3.0, device="cuda"))
        assert torch.equal(first.grad, torch.ones((4, 4), device="cuda"))

    @pytest.mark.parametrize(
        "layer_spec",
        [get_gpt_layer_local_spec, get_gpt_layer_with_transformer_engine_spec],
        ids=["local", "transformer_engine"],
    )
    def test_full_recompute_with_distributed_activations_uses_the_block_group(
        self, monkeypatch, layer_spec
    ):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        config = _config(
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
            distribute_saved_activations=True,
        )
        block = TransformerBlock(
            config, layer_spec(), pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        ).cuda()
        hidden_states = torch.randn(32, 2, config.hidden_size, device="cuda", requires_grad=True)
        attention_mask = torch.ones((1, 1, 32, 32), dtype=bool, device="cuda")

        _forbid_global_tp_group(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            output = block(hidden_states=hidden_states, attention_mask=attention_mask)
            output.sum().backward()
        assert hidden_states.grad is not None
