# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inference reaches the global process groups only through warned compatibility fallbacks.

`InferenceConfig.resolve_pg_collection` is the one process-group boundary of the inference
components, and the pipeline helpers in `megatron.core.inference.communication_utils` take the
caller's pipeline group. Each fallback to the global grid emits one `ProcessGroupFallbackWarning`.
Callers that pass their groups never warn, and every read of the global grid raises while their
code runs.
"""

import contextlib
import sys
import warnings
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference import communication_utils
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext, StaticInferenceContext
from megatron.core.inference.engines import StaticInferenceEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.shards import build_inference_pg_collection
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils

# The groups that inference reads from `InferenceConfig.pg_collection`.
INFERENCE_GROUPS = ("tp", "pp", "dp", "mp", "ep")

# The accessors that tools/check_process_group_usage.py counts.
_ACCESSOR_SUFFIXES = ("_group", "_groups", "_gloo", "_rank", "_ranks", "_world_size", "_src_rank")
_NOT_ACCESSORS = {
    "get_nccl_options",
    "get_all_ranks",
    "get_global_memory_buffer",
    "get_virtual_pipeline_model_parallel_rank",
    "get_virtual_pipeline_model_parallel_world_size",
}
# Pipeline-stage predicates also read the global grid; the checker does not count them.
_STAGE_PREDICATES = ("is_pipeline_first_stage", "is_pipeline_last_stage")


class GlobalProcessGroupRead(Exception):
    """A read of the global parallel grid where the test forbids one.

    Not an AssertionError, so code that treats AssertionError as "not initialized" cannot hide it.
    """


@contextlib.contextmanager
def forbid_global_process_groups():
    """Make every read of the global parallel grid raise inside the block.

    Patches the parallel_state accessors and stage predicates, the copies that Megatron modules
    imported by name, and `ProcessGroupCollection.use_mpu_process_groups`.
    """
    # Keyed by id() because module attributes need not be hashable.
    names = {
        id(value): name
        for name, value in vars(parallel_state).items()
        if getattr(value, "__module__", None) == parallel_state.__name__
        and (
            name in _STAGE_PREDICATES
            or (
                name.startswith("get_")
                and name.endswith(_ACCESSOR_SUFFIXES)
                and name not in _NOT_ACCESSORS
            )
        )
    }

    def forbidden(name):
        def read(*args, **kwargs):
            raise GlobalProcessGroupRead(f"read of the global parallel grid: {name}")

        return read

    with pytest.MonkeyPatch.context() as patch:
        for module_name, module in list(sys.modules.items()):
            if module is None or module_name.split(".")[0] != "megatron":
                continue
            for attribute, value in list(vars(module).items()):
                if id(value) in names:
                    patch.setattr(module, attribute, forbidden(names[id(value)]))
        patch.setattr(
            ProcessGroupCollection,
            "use_mpu_process_groups",
            classmethod(forbidden("ProcessGroupCollection.use_mpu_process_groups")),
        )
        yield


@contextlib.contextmanager
def no_fallback_warnings():
    """Turn every `ProcessGroupFallbackWarning` inside the block into an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        yield


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _fallback_warnings(record):
    return [w for w in record if issubclass(w.category, ProcessGroupFallbackWarning)]


def _check_single_fallback_warning(record, owner, argument):
    """Check that `owner` warned once about `argument`, on the 0.21 to 0.23 schedule."""
    matches = [
        w
        for w in _fallback_warnings(record)
        if f"{owner} was called without `{argument}`" in str(w.message)
    ]
    assert len(matches) == 1, [str(w.message) for w in record]
    message = str(matches[0].message)
    assert "deprecated since Megatron Core 0.21 and will be removed in 0.23" in message
    # The warning points at the code that omitted the argument, not at Megatron Core.
    assert matches[0].filename == __file__


def _require_world_size_multiple_of(size):
    if Utils.world_size < size or Utils.world_size % size != 0:
        pytest.skip(f"needs a world size that is a multiple of {size}")


def _exchange_global_ranks(stage, num_stages, pp_group):
    """Run the pipeline helpers on this rank's global rank.

    Broadcasts from the last stage, then sends each stage's rank to the next stage. Returns the
    broadcast value and the value received from the previous stage (None on the first stage).
    """
    rank = torch.full([2], float(torch.distributed.get_rank()), device="cuda")
    is_last_stage = stage == num_stages - 1
    broadcast = communication_utils.broadcast_from_last_pipeline_stage(
        [2], torch.float32, tensor=rank if is_last_stage else None, pp_group=pp_group
    )
    received = None
    if stage > 0:
        received = torch.zeros(2, device="cuda")
        communication_utils.recv_from_prev_pipeline_rank_(received, pp_group=pp_group)
    if not is_last_stage:
        communication_utils.send_to_next_pipeline_rank(rank, pp_group=pp_group)
    return int(broadcast[0].item()), None if received is None else int(received[0].item())


class TestPipelineHelpers:
    """The pipeline helpers use a given group and warn once each when they fall back."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_stage_predicates_warn_once_without_pp_group(self):
        _require_world_size_multiple_of(2)
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        stage = parallel_state.get_pipeline_model_parallel_group().rank()

        with pytest.warns(ProcessGroupFallbackWarning) as record:
            for _ in range(2):
                assert communication_utils.is_pipeline_first_stage(None) == (stage == 0)
                assert communication_utils.is_pipeline_last_stage(None) == (stage == 1)

        _check_single_fallback_warning(record, "is_pipeline_first_stage", "pp_group")
        _check_single_fallback_warning(record, "is_pipeline_last_stage", "pp_group")
        assert len(_fallback_warnings(record)) == 2

    def test_broadcast_and_p2p_warn_once_without_pp_group(self):
        _require_world_size_multiple_of(2)
        Utils.initialize_model_parallel(pipeline_model_parallel_size=Utils.world_size)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        stage_ranks = torch.distributed.get_process_group_ranks(pp_group)
        stage, num_stages = pp_group.rank(), pp_group.size()

        with pytest.warns(ProcessGroupFallbackWarning) as record:
            for _ in range(2):
                broadcast, received = _exchange_global_ranks(stage, num_stages, pp_group=None)
                assert broadcast == stage_ranks[-1]
                assert received == (stage_ranks[stage - 1] if stage > 0 else None)

        owners = ["broadcast_from_last_pipeline_stage"]
        if stage > 0:
            owners.append("recv_from_prev_pipeline_rank_")
        if stage < num_stages - 1:
            owners.append("send_to_next_pipeline_rank")
        for owner in owners:
            _check_single_fallback_warning(record, owner, "pp_group")
        assert len(_fallback_warnings(record)) == len(owners)

    def test_helpers_use_given_pp_group(self):
        _require_world_size_multiple_of(2)
        # The global grid has a single pipeline stage; the given group has one stage per rank.
        Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
        grid = HyperCommGrid([Utils.world_size], ["pp"])
        pp_group = grid.create_pg("pp")
        stage_ranks = torch.distributed.get_process_group_ranks(pp_group)
        stage, num_stages = pp_group.rank(), pp_group.size()

        with forbid_global_process_groups(), no_fallback_warnings():
            assert communication_utils.is_pipeline_first_stage(pp_group) == (stage == 0)
            assert communication_utils.is_pipeline_last_stage(pp_group) == (stage == num_stages - 1)
            broadcast, received = _exchange_global_ranks(stage, num_stages, pp_group)

        assert broadcast == stage_ranks[-1]
        assert received == (stage_ranks[stage - 1] if stage > 0 else None)
        grid.destroy()


class TestInferenceConfigBoundary:
    """`InferenceConfig.resolve_pg_collection` takes the configured, then the model's groups."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_fallback_warns_once_and_builds_the_groups_inference_reads(self):
        _require_world_size_multiple_of(2)
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        config = InferenceConfig()

        with pytest.warns(ProcessGroupFallbackWarning) as record:
            pg_collection = config.resolve_pg_collection()
            assert config.resolve_pg_collection() is pg_collection

        _check_single_fallback_warning(record, "InferenceConfig", "pg_collection")
        assert config.pg_collection is pg_collection
        assert set(vars(pg_collection)) == set(INFERENCE_GROUPS)
        assert pg_collection.tp is parallel_state.get_tensor_model_parallel_group()
        assert pg_collection.pp is parallel_state.get_pipeline_model_parallel_group()
        assert pg_collection.dp is parallel_state.get_data_parallel_group(with_gtp_remat=False)
        assert pg_collection.mp is parallel_state.get_model_parallel_group()
        assert pg_collection.ep is parallel_state.get_expert_model_parallel_group()

    def test_uses_collection_of_model(self):
        # Every group that inference reads is set; None means the axis is off.
        model = torch.nn.Module()
        model.pg_collection = ProcessGroupCollection(**dict.fromkeys(INFERENCE_GROUPS))
        config = InferenceConfig()

        with forbid_global_process_groups(), no_fallback_warnings():
            assert config.resolve_pg_collection(model) is model.pg_collection
        assert config.pg_collection is model.pg_collection

    def test_partial_model_collection_takes_the_fallback(self):
        _require_world_size_multiple_of(2)
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model = torch.nn.Module()
        model.pg_collection = ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "cp", "pp", "embd"]
        )
        config = InferenceConfig()

        with pytest.warns(ProcessGroupFallbackWarning) as record:
            pg_collection = config.resolve_pg_collection(model)

        _check_single_fallback_warning(record, "InferenceConfig", "pg_collection")
        assert pg_collection is not model.pg_collection
        assert pg_collection.dp is parallel_state.get_data_parallel_group(with_gtp_remat=False)

    def test_configured_collection_comes_first(self):
        configured = ProcessGroupCollection(**dict.fromkeys(INFERENCE_GROUPS))
        model = torch.nn.Module()
        model.pg_collection = ProcessGroupCollection(**dict.fromkeys(INFERENCE_GROUPS))
        config = InferenceConfig(pg_collection=configured)

        with forbid_global_process_groups(), no_fallback_warnings():
            assert config.resolve_pg_collection(model) is configured


class TestComponentsShareOneCollection:
    """Contexts, wrappers, controllers and engines use the collection of one inference config."""

    def teardown_method(self, method):
        InferenceMode.unset_active()
        delete_cuda_graphs()
        Utils.destroy_model_parallel()

    def test_dynamic_context_resolves_for_wrapper_and_controller(self):
        _require_world_size_multiple_of(2)
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        model_config = TransformerConfig(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=8,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
        )
        inference_config = InferenceConfig(
            max_sequence_length=64,
            buffer_size_gb=0.01,
            block_size_tokens=16,
            max_tokens=64,
            max_requests=8,
            unified_memory_level=0,
        )

        # Apart from the config's warned fallback, the context reads none of the global grid.
        with contextlib.ExitStack() as stack:
            for name in (
                "get_expert_model_parallel_world_size",
                "get_pipeline_model_parallel_rank",
            ):
                stack.enter_context(
                    mock.patch.object(
                        parallel_state, name, side_effect=GlobalProcessGroupRead(name)
                    )
                )
            with pytest.warns(ProcessGroupFallbackWarning) as record:
                context = DynamicInferenceContext(model_config, inference_config)

        _check_single_fallback_warning(record, "InferenceConfig", "pg_collection")
        pg_collection = inference_config.pg_collection
        assert pg_collection.tp is parallel_state.get_tensor_model_parallel_group()
        assert context.expert_model_parallel_group is pg_collection.ep
        assert context.pipeline_parallel_group is None

        model = SimpleNamespace(config=model_config, vocab_size=16)
        with forbid_global_process_groups(), no_fallback_warnings():
            wrapper = GPTInferenceWrapper(model, context)
            controller = TextGenerationController(wrapper, SimpleNamespace(eod=0))

        assert wrapper.tp_group is pg_collection.tp
        assert wrapper.pp_group is pg_collection.pp
        assert controller.pp_group is pg_collection.pp
        assert controller.dp_group is pg_collection.dp

    def test_static_context_takes_collection_of_model(self):
        _require_world_size_multiple_of(2)
        # The global grid has TP=1; the model runs on its own TP=2 groups.
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        pg_collection = build_inference_pg_collection(
            Utils.world_size, tp_size=2, pp_size=1, cp_size=1, ep_size=1, expt_tp_size=1
        )
        model_config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)
        model = SimpleNamespace(config=model_config, vocab_size=16, pg_collection=pg_collection)

        with forbid_global_process_groups(), no_fallback_warnings():
            context = StaticInferenceContext(max_batch_size=1, max_sequence_length=8)
            wrapper = GPTInferenceWrapper(model, context)
            controller = TextGenerationController(wrapper, SimpleNamespace(eod=0))

        assert context.config.pg_collection is pg_collection
        assert wrapper.tp_group is pg_collection.tp
        assert wrapper.tp_size == 2
        assert controller.pp_group is pg_collection.pp
        assert controller.dp_group is pg_collection.dp

    @pytest.mark.skipif(
        not is_fa_min_version("2.7.3"), reason="need latest flash attn for dynamic batching"
    )
    def test_static_engine_shares_collection_of_model(self):
        _require_world_size_multiple_of(2)
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        pg_collection = build_inference_pg_collection(
            Utils.world_size, tp_size=2, pp_size=1, cp_size=1, ep_size=1, expt_tp_size=1
        )
        model_parallel_cuda_manual_seed(123)
        vocab_size = 100
        model = GPTModel(
            config=TransformerConfig(
                num_layers=2,
                hidden_size=32,
                num_attention_heads=4,
                use_cpu_initialization=True,
                tensor_model_parallel_size=2,
                params_dtype=torch.bfloat16,
            ),
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=vocab_size,
            max_sequence_length=64,
            pg_collection=pg_collection,
        ).cuda()
        model.to(torch.bfloat16)
        tokenizer = mock.Mock(vocab_size=vocab_size, eod=vocab_size - 1)

        with warnings.catch_warnings(record=True) as caught, forbid_global_process_groups():
            warnings.simplefilter("always")
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            context = StaticInferenceContext(4, 64)
            controller = TextGenerationController(
                inference_wrapped_model=GPTInferenceWrapper(model, context), tokenizer=tokenizer
            )
            engine = StaticInferenceEngine(controller, max_batch_size=4, buffer_size_gb=0.1)

        # The engine catches construction errors and falls back to the legacy engine.
        fallbacks = [str(w.message) for w in caught if "legacy static engine" in str(w.message)]
        assert not fallbacks, fallbacks
        assert not engine.legacy
        assert context.config.pg_collection is pg_collection
        assert controller.inference_wrapped_model.tp_group is pg_collection.tp
        assert controller.pp_group is pg_collection.pp
        assert engine.dynamic_engine.context.config.pg_collection is pg_collection
        assert engine.dynamic_engine.pg_collection is pg_collection
