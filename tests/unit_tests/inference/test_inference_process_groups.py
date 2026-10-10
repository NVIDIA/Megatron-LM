# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inference components use the process groups they are given.

An unset field of a `ProcessGroupCollection` reads as None, and torch.distributed takes a None
group as the world group, so the inference boundary rejects such a field instead of using the
world size or rank. Every read of the global parallel grid raises while the code under test runs.
"""

import contextlib
import sys
from types import SimpleNamespace

import pytest
import torch

import megatron.core.inference.text_generation_server.run_mcore_engine as run_mcore_engine_module
from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.shards import build_inference_pg_collection
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

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
    """A read of the global parallel grid inside `forbid_global_process_groups`.

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


@pytest.fixture(scope="module")
def tp2_collection():
    """A collection for a TP=2, DP=world/2 grid, built without the global grid.

    Its TP size differs from the world size, and its DP rank differs from the global rank on
    every rank but 0, so a world-group read changes the result.
    """
    Utils.initialize_distributed()
    if Utils.world_size < 4 or Utils.world_size % 2 != 0:
        pytest.skip("needs an even world size of at least 4")
    return build_inference_pg_collection(
        Utils.world_size, tp_size=2, pp_size=1, cp_size=1, ep_size=1, expt_tp_size=1
    )


def _partial_collection(collection, *names, **explicit):
    """A collection that sets only the named fields of `collection`, plus `explicit` values."""
    fields = {name: vars(collection)[name] for name in names}
    fields.update(explicit)
    return ProcessGroupCollection(**fields)


def _wrapper(pg_collection):
    """A GPT inference wrapper around a stub model, on a context that carries `pg_collection`."""
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)
    model = SimpleNamespace(config=config, vocab_size=16)
    context = StaticInferenceContext(max_batch_size=1, max_sequence_length=8)
    context.config.pg_collection = pg_collection
    return GPTInferenceWrapper(model, context)


class TestInferenceWrapperProcessGroups:
    """`AbstractModelInferenceWrapper` takes TP and PP from its collection and requires both."""

    def test_uses_collection_groups(self, tp2_collection):
        with forbid_global_process_groups():
            wrapper = _wrapper(tp2_collection)

        assert wrapper.tp_group is tp2_collection.tp
        assert wrapper.pp_group is tp2_collection.pp
        assert wrapper.tp_size == 2

    @pytest.mark.parametrize("unset", ["tp", "pp"])
    @pytest.mark.parametrize("explicit_none", [False, True], ids=["absent", "none"])
    def test_rejects_unset_group(self, tp2_collection, unset, explicit_none):
        kept = [name for name in ("tp", "pp") if name != unset]
        explicit = {unset: None} if explicit_none else {}
        pg_collection = _partial_collection(tp2_collection, *kept, **explicit)

        with forbid_global_process_groups():
            with pytest.raises(ValueError, match=f"needs the {unset} process group"):
                _wrapper(pg_collection)


class TestTextGenerationControllerProcessGroups:
    """`TextGenerationController` reuses the wrapper's PP group and seeds from its DP group."""

    def test_seeds_from_collection_dp_rank(self, tp2_collection):
        with forbid_global_process_groups():
            wrapper = _wrapper(tp2_collection)
            controller = TextGenerationController(wrapper, SimpleNamespace(eod=0))

        assert controller.pp_group is wrapper.pp_group
        assert controller.dp_group is tp2_collection.dp
        seed = wrapper.model.config.inference_sampling_seed
        assert controller.sampling_rng.initial_seed() == seed + tp2_collection.dp.rank()

    def test_seed_offset_rejects_unset_dp(self, tp2_collection):
        wrapper = _wrapper(_partial_collection(tp2_collection, "tp", "pp"))
        assert wrapper.inference_context.config.offset_sampling_seed_by_dp_rank

        with forbid_global_process_groups():
            with pytest.raises(ValueError, match="needs a data-parallel group"):
                TextGenerationController(wrapper, SimpleNamespace(eod=0))

    @pytest.mark.parametrize("seed_offset_off_by", ["config", "deterministic_mode"])
    def test_unset_dp_is_allowed_without_seed_offset(self, tp2_collection, seed_offset_off_by):
        wrapper = _wrapper(_partial_collection(tp2_collection, "tp", "pp"))
        if seed_offset_off_by == "config":
            wrapper.inference_context.config.offset_sampling_seed_by_dp_rank = False
        else:
            wrapper.model.config.deterministic_mode = True

        with forbid_global_process_groups():
            controller = TextGenerationController(wrapper, SimpleNamespace(eod=0))

        assert controller.dp_group is None
        seed = wrapper.model.config.inference_sampling_seed
        assert controller.sampling_rng.initial_seed() == seed


class _StubGroup:
    """A process-group stand-in that reports a fixed rank and size."""

    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _StubTokenizer:
    def detokenize(self, tokens):
        return "".join(chr(ord("a") + token) for token in tokens)


class TestRunMcoreEngine:
    """`run_mcore_engine` post-processes on the first stage of the engine's own pipeline."""

    @pytest.mark.parametrize("pp_rank, builds_response", [(0, True), (1, False)])
    def test_post_processes_on_first_stage_of_engine_pipeline(
        self, monkeypatch, pp_rank, builds_response
    ):
        # The stub group is consulted only when torch.distributed is initialized.
        Utils.initialize_distributed()
        prompt_tokens = [1, 2]
        monkeypatch.setattr(
            run_mcore_engine_module,
            "broadcast_float_list",
            lambda size, float_list, data_parallel: torch.tensor(float_list),
        )
        monkeypatch.setattr(
            run_mcore_engine_module,
            "tokenize_prompts",
            lambda **kwargs: (torch.tensor([prompt_tokens]), torch.tensor([len(prompt_tokens)])),
        )
        result = SimpleNamespace(
            generated_text="c",
            prompt_tokens=prompt_tokens,
            generated_tokens=torch.tensor([2]),
            prompt_log_probs=[-0.5],
            generated_log_probs=[-0.25],
            segments=["a", "b", "c"],
        )
        engine = SimpleNamespace(
            controller=SimpleNamespace(
                pp_group=_StubGroup(rank=pp_rank, size=2), tokenizer=_StubTokenizer()
            ),
            get_new_request_id=lambda: 0,
            generate=lambda inference_requests: [result],
        )

        with forbid_global_process_groups():
            response = run_mcore_engine_module.run_mcore_engine(
                engine, prompts=["ab"], tokens_to_generate=1
            )

        if builds_response:
            assert response["text"] == ["abc"]
            assert response["tokens"] == [[1, 2, 2]]
        else:
            assert response is None
