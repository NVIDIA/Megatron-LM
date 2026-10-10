# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The schedules run on the caller's process groups, or warn once and use the global ones."""

import warnings
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig, parallel_state, process_groups_config
from megatron.core.enums import ModelType
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.utils import is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from tests.unit_tests.test_utilities import Utils

_NO_PIPELINING = "forward_backward_no_pipelining"
_WITHOUT_INTERLEAVING = "forward_backward_pipelining_without_interleaving"
_WITH_INTERLEAVING = "forward_backward_pipelining_with_interleaving"

# Tensor, pipeline and virtual-pipeline sizes of the global grid that each schedule runs on.
_GRIDS = {
    _NO_PIPELINING: (2, 1, None),
    _WITHOUT_INTERLEAVING: (1, 4, None),
    _WITH_INTERLEAVING: (1, 4, 2),
}
_SEQ_LENGTH, _MICRO_BATCH_SIZE, _HIDDEN_SIZE = 8, 2, 4
_NUM_MICROBATCHES = 8

requires_4_ranks = pytest.mark.skipif(
    Utils.world_size < 4 or Utils.world_size % 4 != 0, reason="needs a multiple of 4 ranks"
)


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


@contextmanager
def _forbid_global_process_groups():
    """Make every read of the global process groups in parallel_state raise inside the block."""

    def forbid(*args, **kwargs):
        raise AssertionError("read the global process groups in parallel_state")

    with pytest.MonkeyPatch.context() as patch:
        for name, value in list(vars(parallel_state).items()):
            if callable(value) and name.startswith(("get_", "is_pipeline_")):
                patch.setattr(parallel_state, name, forbid)
        patch.setattr(ProcessGroupCollection, "use_mpu_process_groups", forbid)
        yield


def _assert_one_fallback_warning(record, owner, argument):
    """Check that ``record`` holds exactly one fallback warning, from ``owner``, raised here."""
    fallbacks = [w for w in record if issubclass(w.category, ProcessGroupFallbackWarning)]
    assert len(fallbacks) == 1, [str(w.message) for w in fallbacks]
    (warning,) = fallbacks
    message = str(warning.message)
    assert f"{owner} was called without `{argument}`" in message
    assert "deprecated since Megatron Core 0.21 and will be removed in 0.23" in message
    # The warning points at the code that omitted the argument, not at Megatron Core.
    assert warning.filename == __file__


class _FakeGroup:
    """Stand-in for a process group exposing only rank() and size()."""

    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _StageModel(torch.nn.Module):
    """A pipeline stage that receives the activation of the previous stage."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = ModelType.encoder_or_decoder
        self.input_tensor = None

    def set_input_tensor(self, input_tensor):
        (self.input_tensor,) = input_tensor


def _forward_step_func(data_iterator, model):
    """The first stage starts from the microbatch index and every later stage adds one."""
    if model.input_tensor is None:
        shape = (_SEQ_LENGTH, _MICRO_BATCH_SIZE, _HIDDEN_SIZE)
        output = torch.full(shape, float(next(data_iterator)), device="cuda")
    else:
        output = model.input_tensor + 1

    def loss_func(output_tensor):
        loss = output_tensor.mean()
        return loss, {"value": loss.item()}

    return output, loss_func


def _config(owner):
    tensor, pipeline, virtual = _GRIDS[owner]
    config = ModelParallelConfig(
        tensor_model_parallel_size=tensor,
        pipeline_model_parallel_size=pipeline,
        virtual_pipeline_model_parallel_size=virtual,
        pipeline_dtype=torch.float,
    )
    config.hidden_size = _HIDDEN_SIZE
    return config


def _run_schedule(owner, config, **groups):
    """Run the forward passes of ``owner`` on one model chunk per virtual stage."""
    num_chunks = _GRIDS[owner][2] or 1
    return getattr(schedule, owner)(
        forward_step_func=_forward_step_func,
        data_iterator=[iter(range(_NUM_MICROBATCHES)) for _ in range(num_chunks)],
        model=[_StageModel(config) for _ in range(num_chunks)],
        num_microbatches=_NUM_MICROBATCHES,
        seq_length=_SEQ_LENGTH,
        micro_batch_size=_MICRO_BATCH_SIZE,
        forward_only=True,
        **groups,
    )


@requires_4_ranks
@pytest.mark.parametrize("owner", [_NO_PIPELINING, _WITHOUT_INTERLEAVING, _WITH_INTERLEAVING])
def test_schedule_without_groups_warns_once_and_matches_explicit_groups(owner):
    tensor, pipeline, virtual = _GRIDS[owner]
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tensor,
        pipeline_model_parallel_size=pipeline,
        virtual_pipeline_model_parallel_size=virtual,
    )
    try:
        config = _config(owner)
        # The layout of the global grid (order tp-cp-ep-dp-pp) on distinct communicators.
        grid = HyperCommGrid(
            [tensor, 1, Utils.world_size // (tensor * pipeline), pipeline], ["tp", "cp", "dp", "pp"]
        )
        pp_group = grid.create_pg("pp")
        explicit_groups = {
            "pg_collection": ProcessGroupCollection(
                tp=grid.create_pg("tp"), cp=grid.create_pg("cp"), pp=pp_group
            )
        }
        if owner != _NO_PIPELINING:
            explicit_groups["p2p_communicator"] = P2PCommunicator(pp_group=pp_group, config=config)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            default_results = [_run_schedule(owner, config) for _ in range(2)]
        _assert_one_fallback_warning(record, owner, "pg_collection")

        with _forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            explicit_results = _run_schedule(owner, config, **explicit_groups)

        num_stages = pipeline * (virtual or 1)
        expected = []
        if is_pp_last_stage(pp_group):
            expected = [{"value": float(i + num_stages - 1)} for i in range(_NUM_MICROBATCHES)]
        assert default_results == [expected, expected]
        assert explicit_results == expected
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("missing", ["tp", "cp"])
@pytest.mark.parametrize("owner", [_NO_PIPELINING, _WITHOUT_INTERLEAVING, _WITH_INTERLEAVING])
def test_schedule_rejects_a_collection_without_tp_or_cp(owner, missing):
    groups = {"tp": _FakeGroup(), "cp": _FakeGroup()}
    del groups[missing]
    config = ModelParallelConfig()
    with (
        _forbid_global_process_groups(),
        pytest.raises(ValueError, match=f"{owner} requires pg_collection to set {missing}"),
    ):
        _run_schedule(
            owner,
            config,
            pg_collection=ProcessGroupCollection(**groups),
            p2p_communicator=SimpleNamespace(config=config),
        )


@pytest.mark.parametrize("given", ["pg_collection", "p2p_communicator"])
@pytest.mark.parametrize("owner", [_WITHOUT_INTERLEAVING, _WITH_INTERLEAVING])
def test_pipelined_schedule_takes_both_groups_or_neither(owner, given):
    groups = {
        "pg_collection": ProcessGroupCollection(tp=_FakeGroup(), cp=_FakeGroup()),
        "p2p_communicator": SimpleNamespace(),
    }
    with (
        _forbid_global_process_groups(),
        pytest.raises(
            ValueError,
            match=f"{owner}: provide both p2p_communicator and pg_collection, or neither",
        ),
    ):
        _run_schedule(owner, ModelParallelConfig(), **{given: groups[given]})


def test_get_pp_rank_microbatches_requires_a_communicator():
    with pytest.raises(TypeError, match="p2p_communicator"):
        schedule.get_pp_rank_microbatches(8, 1, 1)
    p2p_communicator = SimpleNamespace(
        pp_group=_FakeGroup(rank=1, size=4), virtual_pipeline_model_parallel_size=None
    )
    with _forbid_global_process_groups():
        microbatches = schedule.get_pp_rank_microbatches(8, 1, 1, p2p_communicator=p2p_communicator)
    # Total, all in warmup, warmup (stages after this one), remaining.
    assert microbatches == (8, False, 2, 6)


@requires_4_ranks
def test_get_forward_backward_func_without_sizes_warns_once():
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=4,
        virtual_pipeline_model_parallel_size=2,
    )
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            for _ in range(2):
                selected = schedule.get_forward_backward_func()
                assert selected is schedule.forward_backward_pipelining_with_interleaving
        _assert_one_fallback_warning(record, "get_forward_backward_func", "pp_size/vp_size")

        with _forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            for pp_size, vp_size, expected in (
                (4, 2, schedule.forward_backward_pipelining_with_interleaving),
                (4, None, schedule.forward_backward_pipelining_without_interleaving),
                (1, None, schedule.forward_backward_no_pipelining),
            ):
                selected = schedule.get_forward_backward_func(pp_size=pp_size, vp_size=vp_size)
                assert selected is expected
    finally:
        Utils.destroy_model_parallel()


@requires_4_ranks
def test_default_groups_keep_the_data_parallel_ranks_of_the_global_grid(monkeypatch):
    """Under GTP-remat, the default collection's readers reduce over the same ranks as before."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, gtp_remat_size=2
    )
    try:
        received = []

        def finalize_model_grads(model, num_tokens, pg_collection, force_all_reduce):
            received.append(pg_collection)

        config = ModelParallelConfig()
        config.calculate_per_token_loss = False
        config.finalize_model_grads_func = finalize_model_grads
        monkeypatch.setattr(schedule, "backward_step", lambda *args: None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ProcessGroupFallbackWarning)
            schedule.forward_backward_no_pipelining(
                forward_step_func=_forward_step_func,
                data_iterator=iter(range(1)),
                model=[_StageModel(config)],
                num_microbatches=1,
                seq_length=_SEQ_LENGTH,
                micro_batch_size=_MICRO_BATCH_SIZE,
                forward_only=False,
            )
        (pg_collection,) = received
        groups = vars(pg_collection)

        data_parallel = parallel_state.get_data_parallel_group(with_context_parallel=True)
        replicate = parallel_state.get_data_parallel_group(
            with_context_parallel=True, with_gtp_remat=False
        )
        assert data_parallel.size() == 2 * replicate.size()
        # finalize_model_grads and the hybrid-CP schedule use dp_cp_gtp_remat when it is set.
        assert (groups["dp_cp_gtp_remat"] or groups["dp_cp"]) is data_parallel

        # The fields map to parallel_state as in use_mpu_process_groups(): dp_cp is the
        # replicate group, and no field outside the ones the readers use is set.
        assert groups["dp_cp"] is replicate
        expected = {
            "tp": parallel_state.get_tensor_model_parallel_group(),
            "cp": parallel_state.get_context_parallel_group(),
            "pp": parallel_state.get_pipeline_model_parallel_group(),
            "embd": parallel_state.get_embedding_group(check_initialized=False),
            "pos_embd": parallel_state.get_position_embedding_group(check_initialized=False),
            "dp_cp": replicate,
            "dp_cp_gtp_remat": data_parallel,
            "tp_dp_cp": parallel_state.get_tensor_and_data_parallel_group(
                with_context_parallel=True
            ),
            "gtp_remat": parallel_state.get_gtp_weight_remat_group(),
            "expt_gtp_remat": parallel_state.get_expert_gtp_weight_remat_group(
                check_initialized=False
            ),
        }
        assert set(groups) == set(expected)
        for name, group in expected.items():
            assert groups[name] is group, name
    finally:
        Utils.destroy_model_parallel()
