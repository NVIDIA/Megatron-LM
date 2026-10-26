# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The inference helpers use their caller's process groups, or warn before they use the global ones.

A caller that passes the groups reads none of the global grid. A caller that omits them gets one
`ProcessGroupFallbackWarning` per helper and the same groups as before, which the helper reads
only through `resolve_process_groups`: every other read of the global grid raises meanwhile.
"""

import warnings
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.inference import communication_utils
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.shards import build_inference_pg_collection
from megatron.core.inference.text_generation_server import tokenization
from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    tokenization as dynamic_tokenization,
)
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.utils import get_pg_size
from tests.unit_tests.test_utilities import (
    Utils,
    build_test_pg_collection,
    destroy_test_pg_collection,
    forbid_global_process_groups,
)

# The sizes that build_inference_pg_collection takes, by collection field.
_SIZE_FIELDS = ("tp", "cp", "pp", "ep", "expt_tp")


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _reads_only_through_resolver():
    """Allow reads of the global grid only from process_groups_config.py, i.e. the resolver."""
    return forbid_global_process_groups(forbid_shim=False)


def _fallback_messages(record):
    return [str(w.message) for w in record if issubclass(w.category, ProcessGroupFallbackWarning)]


def _check_fallback_warnings(record, *expected):
    """Check that the fallback warnings are exactly one per (owner, argument) in `expected`."""
    fallbacks = [w for w in record if issubclass(w.category, ProcessGroupFallbackWarning)]
    assert len(fallbacks) == len(expected), _fallback_messages(record)
    for owner, argument in expected:
        matches = [
            w for w in fallbacks if f"{owner} was called without `{argument}`" in str(w.message)
        ]
        assert len(matches) == 1, _fallback_messages(record)
        assert "deprecated since Megatron Core 0.21 and will be removed in 0.23" in str(
            matches[0].message
        )
        # The warning points at the code that omitted the argument, not at Megatron Core.
        assert matches[0].filename == __file__


def _require_even_world_size():
    if Utils.world_size < 2 or Utils.world_size % 2 != 0:
        pytest.skip("needs an even world size")


def _pairs_grid():
    """A grid of rank pairs: `tp` groups {0, 1}, {2, 3}, ... and `dp` groups across the pairs."""
    return HyperCommGrid([2, Utils.world_size // 2], ["tp", "dp"])


def _first_rank_of_pair(rank):
    return rank - rank % 2


class TestPipelineHelpers:
    """Without `pp_group`, the pipeline helpers read the global group through the resolver."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_fallback_reads_global_group_only_through_resolver(self):
        _require_even_world_size()
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        stage_ranks = torch.distributed.get_process_group_ranks(pp_group)
        stage = pp_group.rank()
        rank = torch.distributed.get_rank()
        value = torch.full([2], float(rank), device="cuda")

        with pytest.warns(ProcessGroupFallbackWarning) as record, _reads_only_through_resolver():
            assert communication_utils.is_pipeline_first_stage(None) == (stage == 0)
            assert communication_utils.is_pipeline_last_stage(None) == (stage == 1)
            broadcast = communication_utils.broadcast_from_last_pipeline_stage(
                [2], torch.float32, tensor=value if stage == 1 else None
            )
            received = torch.zeros(2, device="cuda")
            if stage == 0:
                communication_utils.send_to_next_pipeline_rank(value)
            else:
                communication_utils.recv_from_prev_pipeline_rank_(received)

        assert int(broadcast[0].item()) == stage_ranks[-1]
        if stage == 1:
            assert int(received[0].item()) == stage_ranks[0]
        p2p_owner = "send_to_next_pipeline_rank" if stage == 0 else "recv_from_prev_pipeline_rank_"
        _check_fallback_warnings(
            record,
            ("is_pipeline_first_stage", "pp_group"),
            ("is_pipeline_last_stage", "pp_group"),
            ("broadcast_from_last_pipeline_stage", "pp_group"),
            (p2p_owner, "pp_group"),
        )

    def test_fallback_without_global_group_raises(self):
        # Without a global grid, the fallback group would be None, which torch.distributed
        # treats as the whole world.
        Utils.destroy_model_parallel()
        with pytest.warns(ProcessGroupFallbackWarning):
            with pytest.raises(ValueError, match="no `pp` group to fall back to"):
                communication_utils.is_pipeline_last_stage(None)


class TestBroadcastHelpers:
    """With `data_parallel`, the broadcast helpers run within the caller's model-parallel group."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_given_mp_group_reads_no_global_group(self):
        _require_even_world_size()
        # The global model-parallel groups are single ranks; the given ones are rank pairs.
        Utils.initialize_model_parallel()
        grid = _pairs_grid()
        mp_group = grid.create_pg("tp")
        rank = torch.distributed.get_rank()

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            ints = communication_utils.broadcast_int_list(
                2, int_list=[rank, 2 * rank], data_parallel=True, mp_group=mp_group
            )
            floats = communication_utils.broadcast_float_list(
                1, float_list=[rank + 0.5], data_parallel=True, mp_group=mp_group
            )
            source = torch.full([3], float(rank), device="cuda")
            tensor = communication_utils.broadcast_tensor(
                [3],
                torch.float32,
                tensor=source if rank == _first_rank_of_pair(rank) else None,
                data_parallel=True,
                mp_group=mp_group,
            )

        src = _first_rank_of_pair(rank)
        assert ints.tolist() == [src, 2 * src]
        assert floats.tolist() == [src + 0.5]
        assert tensor.tolist() == [float(src)] * 3
        grid.destroy()

    def test_fallback_uses_global_model_parallel_group(self):
        _require_even_world_size()
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        src = parallel_state.get_model_parallel_src_rank()
        rank = torch.distributed.get_rank()

        with pytest.warns(ProcessGroupFallbackWarning) as record, _reads_only_through_resolver():
            ints = communication_utils.broadcast_int_list(
                2, int_list=[rank, 2 * rank], data_parallel=True
            )
            source = torch.full([3], float(rank), device="cuda")
            tensor = communication_utils.broadcast_tensor(
                [3], torch.float32, tensor=source if rank == src else None, data_parallel=True
            )

        assert ints.tolist() == [src, 2 * src]
        assert tensor.tolist() == [float(src)] * 3
        _check_fallback_warnings(
            record, ("broadcast_list", "mp_group"), ("broadcast_tensor", "mp_group")
        )


def _tokenizer():
    return SimpleNamespace(eod=0, tokenize=lambda text: [ord(c) - ord("a") + 1 for c in text])


def _prompts(rank):
    # Lengths that differ between ranks, so a rank holds the tokens of the rank it received from.
    return ["a" * (rank + 1), "b"]


@pytest.mark.parametrize(
    "module", [tokenization, dynamic_tokenization], ids=["static_server", "dynamic_server"]
)
class TestTokenizePrompts:
    """With `data_parallel`, `tokenize_prompts` broadcasts within the caller's groups."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _check_tokens(self, module, tokens, lengths):
        src = _first_rank_of_pair(torch.distributed.get_rank())
        expected_tokens, expected_lengths = module._tokenize_prompts_and_batch(
            _tokenizer(), _prompts(src), 1, False
        )
        assert torch.equal(tokens, expected_tokens)
        assert torch.equal(lengths, expected_lengths)

    def test_given_collection_reads_no_global_group(self, module):
        _require_even_world_size()
        # The global grid has TP=1: every rank starts a model-parallel group of its own.
        Utils.initialize_model_parallel()
        grid = _pairs_grid()
        pg_collection = ProcessGroupCollection(dp=grid.create_pg("dp"), mp=grid.create_pg("tp"))
        rank = torch.distributed.get_rank()

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            tokens, lengths = module.tokenize_prompts(
                _tokenizer(),
                prompts=_prompts(rank),
                tokens_to_generate=1,
                add_BOS=False,
                data_parallel=True,
                pg_collection=pg_collection,
            )

        self._check_tokens(module, tokens, lengths)
        grid.destroy()

    def test_fallback_uses_global_groups(self, module):
        _require_even_world_size()
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        rank = torch.distributed.get_rank()

        with pytest.warns(ProcessGroupFallbackWarning) as record, _reads_only_through_resolver():
            tokens, lengths = module.tokenize_prompts(
                _tokenizer(),
                prompts=_prompts(rank),
                tokens_to_generate=1,
                add_BOS=False,
                data_parallel=True,
            )

        self._check_tokens(module, tokens, lengths)
        # The broadcasts use the groups that tokenize_prompts resolved, and do not warn again.
        _check_fallback_warnings(record, ("tokenize_prompts", "pg_collection"))


class TestBuildInferencePgCollection:
    """Sizes left as None come from the training collection, else from the global grid."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_sizes_from_training_collection(self):
        _require_even_world_size()
        # The global grid has TP=1; the training collection has TP=2.
        Utils.initialize_model_parallel()
        training_pg_collection = build_test_pg_collection(tp=2)

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            pg_collection = build_inference_pg_collection(
                Utils.world_size, pp_size=1, training_pg_collection=training_pg_collection
            )

        sizes = {name: get_pg_size(vars(pg_collection)[name]) for name in _SIZE_FIELDS}
        assert sizes == {"tp": 2, "cp": 1, "pp": 1, "ep": 1, "expt_tp": 2}
        destroy_test_pg_collection(training_pg_collection)

    def test_fallback_uses_global_sizes(self):
        _require_even_world_size()
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        expected = {
            "tp": parallel_state.get_tensor_model_parallel_world_size(),
            "cp": parallel_state.get_context_parallel_world_size(),
            "pp": parallel_state.get_pipeline_model_parallel_world_size(),
            "ep": parallel_state.get_expert_model_parallel_world_size(),
            "expt_tp": parallel_state.get_expert_tensor_parallel_world_size(),
        }

        with pytest.warns(ProcessGroupFallbackWarning) as record, _reads_only_through_resolver():
            pg_collection = build_inference_pg_collection(Utils.world_size)

        assert {name: get_pg_size(vars(pg_collection)[name]) for name in _SIZE_FIELDS} == expected
        _check_fallback_warnings(
            record, ("build_inference_pg_collection", "training_pg_collection")
        )


@pytest.fixture
def unit_rounders(monkeypatch):
    """Pad to the next multiple of the TP size only, so that the TP size shows in the result."""
    monkeypatch.setattr(DynamicInferenceContext, "TOKEN_ROUNDER", 1)
    monkeypatch.setattr(DynamicInferenceContext, "REQUEST_ROUNDER", 1)


@pytest.mark.usefixtures("unit_rounders")
class TestRoundUpWithoutTensorParallelSize:
    """Without `tp_size`, the rounders pad to the global TP size and warn."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_fallback_pads_to_global_tp_size(self):
        _require_even_world_size()
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)

        with pytest.warns(ProcessGroupFallbackWarning) as record, _reads_only_through_resolver():
            assert DynamicInferenceContext.round_up_tokens(3) == 4
            assert DynamicInferenceContext.round_up_requests(5) == 6

        _check_fallback_warnings(
            record,
            ("DynamicInferenceContext.round_up_tokens", "tp_size"),
            ("DynamicInferenceContext.round_up_requests", "tp_size"),
        )

    def test_fallback_without_global_grid_pads_to_one(self):
        Utils.destroy_model_parallel()
        with pytest.warns(ProcessGroupFallbackWarning) as record:
            assert DynamicInferenceContext.round_up_tokens(3) == 3

        _check_fallback_warnings(record, ("DynamicInferenceContext.round_up_tokens", "tp_size"))

    def test_given_tp_size_reads_no_global_state(self, monkeypatch):
        def no_global_state():
            raise AssertionError("parallel_state.is_initialized() reads global state")

        monkeypatch.setattr(parallel_state, "is_initialized", no_global_state)
        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            assert DynamicInferenceContext.round_up_tokens(3, tp_size=2) == 4
            assert DynamicInferenceContext.round_up_requests(5, tp_size=2) == 6
