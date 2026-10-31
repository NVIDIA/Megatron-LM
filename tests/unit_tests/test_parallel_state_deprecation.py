# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The parallel_state accessors of the global grid warn that they are deprecated.

Every group, rank and size accessor, pipeline-stage predicate, virtual-pipeline getter and
setter, rank and size override, and memory-buffer alias emits a ParallelStateDeprecationWarning
once per call site. The warning names its replacement and the releases that deprecate and remove
the function. Bootstrap, teardown and the registry do not warn, and neither do Megatron Core's
own compatibility fallbacks, which report a single ProcessGroupFallbackWarning instead; the
fallbacks of training read the grid in a global_process_group_fallback() block to the same
effect. Strict process-group fallbacks leave the accessor warning a warning.

The CPU tests run without torch.distributed. The tests on the standard grid need GPUs; they use
a TP2 x PP2 grid on 4 or more ranks, and smaller grids otherwise.
"""

import contextlib
import inspect
import os
import re
import threading
import warnings
from unittest.mock import Mock

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.parallel_state import ParallelStateDeprecationWarning
from megatron.core.process_groups_config import (
    ProcessGroupCollection,
    ProcessGroupFallbackWarning,
    enable_strict_process_group_fallbacks,
    global_process_group_fallback,
    resolve_process_groups,
)
from megatron.core.utils import get_tensor_model_parallel_group_if_none
from megatron.training import checkpointing
from megatron.training.utils.common_utils import (
    average_losses_across_data_parallel_group,
    is_first_or_last_pipeline_stage,
)
from tasks import finetune_utils
from tests.unit_tests.test_utilities import Utils

VERSIONS = "is deprecated since Megatron Core 0.21 and will be removed in 0.23."
PP_UTILS = "megatron.core.pipeline_parallel.utils"
# What each deprecated function's warning names as its replacement, by tier.
GROUPS = {
    "get_model_parallel_group": "pg_collection.mp",
    "get_tensor_model_parallel_group": "pg_collection.tp",
    "get_gtp_weight_remat_group": "pg_collection.gtp_remat",
    "get_pipeline_model_parallel_group": "pg_collection.pp",
    "get_data_parallel_group": "dp_gtp_remat, dp (with_gtp_remat=False)",
    "get_data_parallel_group_gloo": "pg_collection.intra_dp_cp_gloo",
    "get_context_parallel_group": "pg_collection.cp",
    "get_context_parallel_global_ranks": "get_process_group_ranks(pg_collection.cp)",
    "get_hierarchical_context_parallel_groups": "pg_collection.hcp",
    "get_dynamic_data_context_parallel_group_map": "pg_collection.dynamic_cp",
    "get_dynamic_tensor_data_context_parallel_group_map": "pg_collection.dynamic_tp_cp",
    "get_dynamic_data_context_parallel_groups": "pg_collection.dynamic_cp[group_size]",
    "get_hybrid_data_context_parallel_groups": "pg_collection.dynamic_cp[group_size]",
    "get_dynamic_tensor_data_context_parallel_group": "pg_collection.dynamic_tp_cp[group_size]",
    "get_embedding_group": "pg_collection.embd",
    "get_position_embedding_group": "pg_collection.pos_embd",
    "get_amax_reduction_group": (
        "amax_reduction_group(pg_collection, tp_only_amax_red) for with_context_parallel=True; "
        "otherwise pg_collection.tp for tp_only_amax_red=True, or a TP x DP group"
    ),
    "get_tensor_and_data_parallel_group": "pg_collection.tp_dp_cp",
    "get_tensor_and_context_parallel_group": "pg_collection.tp_cp",
    "get_expert_gtp_weight_remat_group": "pg_collection.expt_gtp_remat",
    "get_expert_model_parallel_group": "pg_collection.ep",
    "get_expert_tensor_parallel_group": "pg_collection.expt_tp",
    "get_expert_tensor_and_model_parallel_group": "pg_collection.tp_ep",
    "get_expert_tensor_model_pipeline_parallel_group": "pg_collection.tp_ep_pp_with_egtp_remat",
    "get_expert_data_parallel_group": "expt_dp_gtp_remat, expt_dp (with_gtp_remat=False)",
    "get_expert_data_parallel_group_gloo": "pg_collection.intra_expt_dp_gloo",
    "get_intra_distributed_optimizer_instance_group": "pg_collection.intra_dist_opt",
    "get_inter_distributed_optimizer_instance_group": "pg_collection.inter_dist_opt",
}
RANKS_AND_SIZES = {
    "get_gtp_weight_remat_world_size": "pg_collection.gtp_remat.size()",
    "get_gtp_weight_remat_rank": "pg_collection.gtp_remat.rank()",
    "get_gtp_weight_remat_global_ranks": "get_process_group_ranks(pg_collection.gtp_remat)",
    "get_tensor_model_parallel_world_size": "pg_collection.tp.size()",
    "get_tensor_model_parallel_rank": "pg_collection.tp.rank()",
    "get_pipeline_model_parallel_world_size": "pg_collection.pp.size()",
    "get_pipeline_model_parallel_rank": "pg_collection.pp.rank()",
    "get_tensor_model_parallel_src_rank": "get_global_rank(pg_collection.tp, 0)",
    "get_model_parallel_src_rank": "get_global_rank(pg_collection.mp, 0)",
    "get_data_parallel_src_rank": "get_global_rank(pg_collection.dp, 0)",
    "get_pipeline_model_parallel_first_rank": f"{PP_UTILS}.get_pp_first_rank(pg_collection.pp)",
    "get_pipeline_model_parallel_last_rank": f"{PP_UTILS}.get_pp_last_rank(pg_collection.pp)",
    # The replacements return None at the ends of the pipeline, where these accessors wrap
    # around, so the warning says so and names where the wrapped peer comes from.
    "get_pipeline_model_parallel_next_rank": (
        f"{PP_UTILS}.get_pp_next_rank(pg_collection.pp), which does not wrap around: on the last "
        "stage it returns None instead of the first stage's rank. Code that relies on the "
        "wrap-around takes the next rank from "
        "torch.distributed.get_process_group_ranks(pg_collection.pp)"
    ),
    "get_pipeline_model_parallel_prev_rank": (
        f"{PP_UTILS}.get_pp_prev_rank(pg_collection.pp), which does not wrap around: on the first "
        "stage it returns None instead of the last stage's rank. Code that relies on the "
        "wrap-around takes the previous rank from "
        "torch.distributed.get_process_group_ranks(pg_collection.pp)"
    ),
    "get_data_parallel_world_size": "the size() of the data-parallel field of pg_collection",
    "get_data_parallel_rank": "the rank() of the data-parallel field of pg_collection",
    "get_context_parallel_world_size": "pg_collection.cp.size()",
    "get_context_parallel_rank": "pg_collection.cp.rank()",
    "get_tensor_and_context_parallel_world_size": "pg_collection.tp_cp.size()",
    "get_tensor_and_context_parallel_rank": "pg_collection.tp_cp.rank()",
    "get_expert_gtp_weight_remat_world_size": "pg_collection.expt_gtp_remat.size()",
    "get_expert_gtp_weight_remat_rank": "pg_collection.expt_gtp_remat.rank()",
    "get_expert_gtp_weight_remat_global_ranks": (
        "get_process_group_ranks(pg_collection.expt_gtp_remat)"
    ),
    "get_expert_model_parallel_src_rank": "get_global_rank(pg_collection.ep, 0)",
    "get_expert_model_parallel_world_size": "pg_collection.ep.size()",
    "get_expert_model_parallel_rank": "pg_collection.ep.rank()",
    "get_expert_tensor_parallel_world_size": "pg_collection.expt_tp.size()",
    "get_expert_tensor_parallel_rank": "pg_collection.expt_tp.rank()",
    "get_expert_tensor_and_model_parallel_world_size": "pg_collection.tp_ep.size()",
    "get_expert_tensor_and_model_parallel_rank": "pg_collection.tp_ep.rank()",
    "get_expert_data_parallel_world_size": "the size() of the expert data-parallel field",
    "get_expert_data_parallel_rank": "the rank() of the expert data-parallel field",
    "get_all_ranks": "the rank() of each group of pg_collection",
}
STAGE_PREDICATES = {
    "is_pipeline_first_stage": f"{PP_UTILS}.is_pp_first_stage(pg_collection.pp)",
    "is_pipeline_last_stage": f"{PP_UTILS}.is_pp_last_stage(pg_collection.pp)",
    "is_rank_in_embedding_group": "membership in pg_collection.embd",
    "is_rank_in_position_embedding_group": "membership in pg_collection.pos_embd",
}
VIRTUAL_PIPELINE = {
    "get_virtual_pipeline_model_parallel_rank": "the vp_stage of the model chunk",
    "set_virtual_pipeline_model_parallel_rank": "an explicit vp_stage argument",
    "get_virtual_pipeline_model_parallel_world_size": "virtual_pipeline_model_parallel_size",
    "set_virtual_pipeline_model_parallel_world_size": "virtual_pipeline_model_parallel_size",
}
OVERRIDES = dict.fromkeys(
    (
        "set_tensor_model_parallel_world_size",
        "set_tensor_model_parallel_rank",
        "set_pipeline_model_parallel_world_size",
        "set_pipeline_model_parallel_rank",
        "set_data_parallel_rank",
        "set_expert_model_parallel_world_size",
        "set_expert_model_parallel_rank",
        "set_expert_tensor_parallel_world_size",
        "set_expert_tensor_parallel_rank",
    ),
    "a ProcessGroupCollection whose groups have that rank and size",
)
MEMORY_BUFFER = {
    "get_global_memory_buffer": "megatron.core.utils.get_global_memory_buffer()",
    "destroy_global_memory_buffer": "megatron.core.utils.destroy_global_memory_buffer()",
    "_set_global_memory_buffer": "megatron.core.utils.get_global_memory_buffer(), which creates",
}
DEPRECATED = {
    **GROUPS,
    **RANKS_AND_SIZES,
    **STAGE_PREDICATES,
    **VIRTUAL_PIPELINE,
    **OVERRIDES,
    **MEMORY_BUFFER,
}
# Bootstrap, teardown, the registry, and helpers that build groups from their arguments.
NOT_DEPRECATED = {
    "initialize_model_parallel",
    "destroy_model_parallel",
    "is_initialized",
    "model_parallel_is_initialized",
    "register_process_groups",
    "get_registered_process_groups",
    "build_process_group_collection",
    "create_all_gather_groups",
    "register_model_parallel_teardown",
    "create_group",
    "get_nccl_options",
    "update_pg_timeout",
    "overwrite_nccl_comm_cfgs",
    "generate_masked_orthogonal_rank_groups",
    "create_hierarchical_groups",
    "get_valid_dynamic_context_parallel_group_sizes",
    "create_dynamic_dp_cp_groups",
    "default_embedding_ranks",
    "default_position_embedding_ranks",
}
# The code object that the deprecation wrapper runs, shared by every deprecated function.
WRAPPER_CODE = parallel_state._deprecated_accessor("")(lambda: None).__code__
# Arguments of the functions whose parameters have no default, as callers pass them.
ARGUMENTS = {name: (1,) for name in OVERRIDES.keys() | {"set_virtual_pipeline_model_parallel_rank"}}
ARGUMENTS["set_virtual_pipeline_model_parallel_world_size"] = (2,)

requires_grid = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the standard grid is built with NCCL on GPUs"
)


@pytest.fixture(autouse=True)
def fresh_warning_records(monkeypatch):
    """Each test observes the first warning of every call site and fallback."""
    monkeypatch.setattr(parallel_state, "_warned_accessor_call_sites", set())
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


@contextlib.contextmanager
def _recorded_warnings():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        yield record


def _deprecated_calls(record):
    """Return (function, file name, line number) for each accessor warning in ``record``."""
    calls = []
    for warning in record:
        if issubclass(warning.category, ParallelStateDeprecationWarning):
            name = re.match(r"megatron\.core\.parallel_state\.(\w+)\(\)", str(warning.message))
            calls.append((name.group(1), warning.filename, warning.lineno))
    return calls


def _fallbacks(record):
    """Return the (owner, argument) of each ProcessGroupFallbackWarning in ``record``."""
    return [
        re.match(r"(.+) was called without `(\w+)`", str(warning.message)).groups()
        for warning in record
        if issubclass(warning.category, ProcessGroupFallbackWarning)
    ]


def _fallback_files(record):
    """Return the file that each ProcessGroupFallbackWarning in ``record`` points at."""
    return [
        warning.filename
        for warning in record
        if issubclass(warning.category, ProcessGroupFallbackWarning)
    ]


def test_every_parallel_state_function_is_classified():
    """A new parallel_state function is either deprecated or listed as bootstrap or helper."""
    deprecated = {
        name
        for name, value in vars(parallel_state).items()
        if inspect.isfunction(value) and value.__code__ is WRAPPER_CODE
    }
    public = {
        name
        for name, value in vars(parallel_state).items()
        if inspect.isfunction(value)
        and value.__module__ == parallel_state.__name__
        and not name.startswith("_")
    }
    assert deprecated == DEPRECATED.keys()
    assert public - deprecated == NOT_DEPRECATED


@pytest.mark.parametrize("name", sorted(DEPRECATED))
def test_each_deprecated_function_names_its_replacement(name):
    """The warning comes before the function runs, so an error filter stops it unchanged."""
    function = vars(parallel_state)[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParallelStateDeprecationWarning)
        with pytest.raises(ParallelStateDeprecationWarning) as raised:
            function(*ARGUMENTS.get(name, ()))

    message = str(raised.value)
    assert message.startswith(f"megatron.core.parallel_state.{name}() {VERSIONS}")
    assert DEPRECATED[name] in message.partition(" Replace it with ")[2]
    if "pg_collection" in message:
        assert "pg_collection is the ProcessGroupCollection of the model" in message


def test_each_call_site_warns_once(monkeypatch):
    monkeypatch.setattr(parallel_state, "_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", 2)
    monkeypatch.setattr(parallel_state, "_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK", 1)
    get_size = parallel_state.get_virtual_pipeline_model_parallel_world_size
    get_rank = parallel_state.get_virtual_pipeline_model_parallel_rank
    with _recorded_warnings() as record:
        for _ in range(3):
            get_size()
        get_size()
        both = (get_size(), get_rank())

    # The accessors still return what they read.
    assert both == (2, 1)
    calls = _deprecated_calls(record)
    size_lines = [line for name, _, line in calls if name == get_size.__name__]
    rank_lines = [line for name, _, line in calls if name == get_rank.__name__]
    # Three call sites of one accessor, and a second accessor on the same line as the last.
    assert len(size_lines) == len(set(size_lines)) == 3
    assert rank_lines == size_lines[-1:]
    # The warning points at the caller, not at parallel_state.
    assert {filename for _, filename, _ in calls} == {__file__}


def test_an_error_filter_raises_at_every_call():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParallelStateDeprecationWarning)
        for _ in range(2):
            with pytest.raises(ParallelStateDeprecationWarning):
                parallel_state.get_virtual_pipeline_model_parallel_rank()


def test_an_error_filter_raises_in_a_thread_while_another_raises_at_the_same_site(monkeypatch):
    """A call site is recorded only once its warning has been shown, so a call that comes while
    another thread's warning is raising at the same site raises as well."""
    worker_is_warning = threading.Event()
    main_call_done = threading.Event()
    warn = warnings.warn

    def warn_slowly_in_the_worker(*args, **kwargs):
        if threading.current_thread() is worker:
            worker_is_warning.set()
            main_call_done.wait(timeout=10)
        warn(*args, **kwargs)

    def call_accessor():
        parallel_state.get_virtual_pipeline_model_parallel_rank()

    def run_worker():
        with pytest.raises(ParallelStateDeprecationWarning) as raised:
            call_accessor()
        raised_in_worker.append(raised.value)

    raised_in_worker = []
    worker = threading.Thread(target=run_worker)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParallelStateDeprecationWarning)
        monkeypatch.setattr(warnings, "warn", warn_slowly_in_the_worker)
        worker.start()
        try:
            assert worker_is_warning.wait(timeout=10)
            with pytest.raises(ParallelStateDeprecationWarning):
                call_accessor()
        finally:
            main_call_done.set()
            worker.join()

    assert len(raised_in_worker) == 1


def test_a_warning_hook_that_calls_an_accessor_warns_once_for_it(monkeypatch):
    """A hook that shows warnings may call an accessor itself, for example to tag the message
    with a rank. Under an "always" filter its call warns once instead of recursing."""
    shown = []

    def showwarning(message, category, filename, lineno, file=None, line=None):
        shown.append((category, str(message).partition("()")[0], filename))
        parallel_state.get_virtual_pipeline_model_parallel_rank()

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        monkeypatch.setattr(warnings, "showwarning", showwarning)
        parallel_state.get_virtual_pipeline_model_parallel_world_size()

    assert shown == [
        (
            ParallelStateDeprecationWarning,
            "megatron.core.parallel_state.get_virtual_pipeline_model_parallel_world_size",
            __file__,
        ),
        (
            ParallelStateDeprecationWarning,
            "megatron.core.parallel_state.get_virtual_pipeline_model_parallel_rank",
            __file__,
        ),
    ]


def test_a_warning_hook_that_reads_the_grid_in_a_fallback_block_warns_once(monkeypatch):
    """A hook outside Megatron Core reads the grid in a global_process_group_fallback() block.
    Under an "always" filter its fallback warns once instead of recursing."""
    shown = []

    def showwarning(message, category, filename, lineno, file=None, line=None):
        shown.append((category, str(message).partition(" ")[0]))
        with global_process_group_fallback("showwarning", deprecated_in="0.21", removed_in="0.23"):
            parallel_state.get_virtual_pipeline_model_parallel_rank()

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        monkeypatch.setattr(warnings, "showwarning", showwarning)
        parallel_state.get_virtual_pipeline_model_parallel_world_size()

    assert shown == [
        (
            ParallelStateDeprecationWarning,
            "megatron.core.parallel_state.get_virtual_pipeline_model_parallel_world_size()",
        ),
        (ProcessGroupFallbackWarning, "showwarning"),
    ]


def test_a_compiled_region_calls_an_accessor_without_a_graph_break(monkeypatch):
    """torch.compile traces through a deprecated accessor, as it did before the deprecation,
    and the traced call does not warn."""
    monkeypatch.setattr(parallel_state, "_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE", 2)

    def scale(x):
        return x * parallel_state.get_tensor_model_parallel_world_size()

    compiled = torch.compile(scale, backend="eager", fullgraph=True)
    with _recorded_warnings() as record:
        scaled = compiled(torch.ones(3))

    torch.testing.assert_close(scaled, torch.full((3,), 2.0))
    assert _deprecated_calls(record) == []


def test_the_warning_is_a_future_warning_apart_from_fallback_warnings():
    assert issubclass(ParallelStateDeprecationWarning, FutureWarning)
    assert not issubclass(ParallelStateDeprecationWarning, ProcessGroupFallbackWarning)
    assert not issubclass(ProcessGroupFallbackWarning, ParallelStateDeprecationWarning)


def test_the_test_suite_ignores_the_warning(pytestconfig):
    """Tests use the global grid on purpose, so pyproject.toml filters the warning for them."""
    entry = "ignore::megatron.core.parallel_state.ParallelStateDeprecationWarning"
    assert entry in pytestconfig.getini("filterwarnings")
    assert any(
        action == "ignore" and category is ParallelStateDeprecationWarning
        for action, _, category, _, _ in warnings.filters
    )


def test_calls_from_megatron_core_do_not_warn():
    """A caller whose source file is in megatron/core is exempt; core warns for its fallbacks."""
    core_caller = compile(
        "parallel_state.get_virtual_pipeline_model_parallel_rank()", parallel_state.__file__, "exec"
    )
    with _recorded_warnings() as record:
        exec(core_caller, {"parallel_state": parallel_state})
        ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "pp", "cp", "ep"])

    assert _deprecated_calls(record) == []


def test_a_core_fallback_on_no_grid_warns_once_without_accessor_warnings():
    if parallel_state.model_parallel_is_initialized():
        pytest.skip("a grid is initialized")
    with _recorded_warnings() as record:
        pg_collection = resolve_process_groups(
            None, owner="Owner", required=("tp", "pp"), deprecated_in="0.21", removed_in="0.23"
        )

    assert (pg_collection.tp, pg_collection.pp) == (None, None)
    assert _fallbacks(record) == [("Owner", "pg_collection")]
    assert _deprecated_calls(record) == []


def test_bootstrap_teardown_and_registry_do_not_warn():
    """Registering a collection and tearing it down reads no deprecated function."""
    if torch.distributed.is_initialized():
        pytest.skip("tearing down would destroy the groups of the distributed run")
    extras = parallel_state.StandardGridExtras(
        **{name: None for name in parallel_state.StandardGridExtras.__dataclass_fields__}
    )
    pg_collection = ProcessGroupCollection()
    with _recorded_warnings() as record:
        parallel_state.register_process_groups(pg_collection, extras=extras)
        registered = parallel_state.get_registered_process_groups()
        initialized = (
            parallel_state.is_initialized(),
            parallel_state.model_parallel_is_initialized(),
        )
        parallel_state.destroy_model_parallel()

    assert registered is pg_collection
    assert initialized == (False, False)
    assert parallel_state.get_registered_process_groups() is None
    assert _deprecated_calls(record) == []


def test_a_fallback_block_reports_its_reads_with_one_fallback_warning():
    with _recorded_warnings() as record:
        with global_process_group_fallback(
            "Owner", "pp_group", deprecated_in="0.21", removed_in="0.23"
        ):
            parallel_state.get_virtual_pipeline_model_parallel_world_size()
            parallel_state.get_pipeline_model_parallel_group(check_initialized=False)
        parallel_state.get_virtual_pipeline_model_parallel_rank()

    assert _fallbacks(record) == [("Owner", "pp_group")]
    # The fallback warning points at the code that falls back, not at Megatron Core.
    assert _fallback_files(record) == [__file__]
    assert [name for name, _, _ in _deprecated_calls(record)] == [
        "get_virtual_pipeline_model_parallel_rank"
    ]


def test_a_fallback_block_ends_when_its_body_raises():
    with _recorded_warnings() as record:
        with pytest.raises(RuntimeError):
            with global_process_group_fallback("Owner", deprecated_in="0.21", removed_in="0.23"):
                raise RuntimeError("the fallback failed")
        parallel_state.get_virtual_pipeline_model_parallel_rank()

    assert [name for name, _, _ in _deprecated_calls(record)] == [
        "get_virtual_pipeline_model_parallel_rank"
    ]


def test_strict_fallbacks_raise_but_leave_the_accessor_warning_a_warning():
    """Strict mode turns ProcessGroupFallbackWarning into an error, and nothing else."""
    body_ran = False
    with _recorded_warnings() as record:
        enable_strict_process_group_fallbacks()
        with pytest.raises(ProcessGroupFallbackWarning):
            resolve_process_groups(None, owner="StrictOwner", required=("tp",))
        with pytest.raises(ProcessGroupFallbackWarning):
            with global_process_group_fallback("StrictOwner", "tp_group"):
                body_ran = True
        parallel_state.get_virtual_pipeline_model_parallel_rank()

    assert not body_ran
    assert [name for name, _, _ in _deprecated_calls(record)] == [
        "get_virtual_pipeline_model_parallel_rank"
    ]


def test_training_fallbacks_read_the_overrides_without_accessor_warnings(monkeypatch):
    """Training's fallbacks keep the accessors' semantics, including the rank and size
    overrides, and report the fallback once instead of each accessor they call."""
    for name, value in (
        ("_MPU_TENSOR_MODEL_PARALLEL_RANK", 1),
        ("_MPU_PIPELINE_MODEL_PARALLEL_RANK", 2),
        ("_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE", 4),
        ("_MPU_EXPERT_MODEL_PARALLEL_RANK", 3),
        ("_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE", 4),
    ):
        monkeypatch.setattr(parallel_state, name, value)

    with _recorded_warnings() as record:
        name = checkpointing.get_checkpoint_name("ckpt", 5)
        on_first_or_last_stage = is_first_or_last_pipeline_stage(None)

    assert name == os.path.join("ckpt", "iter_0000005", "mp_rank_01_002_003", "model_optim_rng.pt")
    assert not on_first_or_last_stage
    assert _fallbacks(record) == [
        ("get_checkpoint_name", "pg_collection"),
        ("is_first_or_last_pipeline_stage", "pp_group"),
    ]
    assert _deprecated_calls(record) == []


@pytest.mark.parametrize("fields", [("dp_gtp_remat", "dp"), ("dp",)])
def test_finetuning_samples_over_the_registered_grid(monkeypatch, fields):
    """The finetuning data loader splits the data over the data-parallel group of the grid that
    initialize_megatron() registered, and reads no deprecated function. As in pretraining, a
    collection without dp_gtp_remat, which predates that field, splits it over dp."""
    sampled, *others = fields
    pg_collection = ProcessGroupCollection(
        **{sampled: Mock(**{"size.return_value": 4, "rank.return_value": 3})},
        **{field: Mock(**{"size.return_value": 2, "rank.return_value": 1}) for field in others},
    )
    monkeypatch.setattr(parallel_state, "_REGISTERED_PROCESS_GROUPS", pg_collection)

    with _recorded_warnings() as record:
        data_loader = finetune_utils.build_data_loader(range(8), 2, 0, drop_last=False)

    assert (data_loader.sampler.num_replicas, data_loader.sampler.rank) == (4, 3)
    assert _deprecated_calls(record) == []
    assert _fallbacks(record) == []


@requires_grid
class TestStandardGrid:
    """On the standard grid, bootstrap and fallbacks report no accessor warning."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _initialize():
        tensor_parallel_size = 2 if Utils.world_size % 2 == 0 else 1
        pipeline_parallel_size = 2 if Utils.world_size % 4 == 0 else 1
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
        )

    def test_bootstrap_teardown_and_registry_do_not_warn(self):
        with _recorded_warnings() as record:
            self._initialize()
            initialized = (
                parallel_state.is_initialized(),
                parallel_state.model_parallel_is_initialized(),
            )
            tensor_parallel_size = parallel_state.get_registered_process_groups().tp.size()
            Utils.destroy_model_parallel()

            pg_collection, extras = parallel_state.build_process_group_collection(
                tensor_model_parallel_size=tensor_parallel_size
            )
            parallel_state.register_process_groups(pg_collection, extras=extras)
            registered = parallel_state.get_registered_process_groups()
            torch.distributed.barrier()
            parallel_state.destroy_model_parallel()

        assert initialized == (True, True)
        assert registered is pg_collection
        assert _deprecated_calls(record) == []

    def test_core_fallbacks_warn_once_without_accessor_warnings(self):
        self._initialize()
        with _recorded_warnings() as record:
            pg_collection = resolve_process_groups(
                None,
                owner="Owner",
                required=("tp", "pp", "dp_cp"),
                deprecated_in="0.21",
                removed_in="0.23",
            )
            tp_group = get_tensor_model_parallel_group_if_none(None)
            shim = ProcessGroupCollection.use_mpu_process_groups()

        assert [owner for owner, _ in _fallbacks(record)] == [
            "Owner",
            "get_tensor_model_parallel_group_if_none "
            "(from TestStandardGrid.test_core_fallbacks_warn_once_without_accessor_warnings)",
        ]
        assert _deprecated_calls(record) == []
        global_tp_group = parallel_state.get_tensor_model_parallel_group()
        assert pg_collection.tp is tp_group is shim.tp is global_tp_group

    def test_training_fallbacks_match_the_model_groups(self):
        """Without groups, training helpers read the same grid, with one warning each."""
        self._initialize()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        loss = torch.full((1,), float(torch.distributed.get_rank()), device="cuda")
        with _recorded_warnings() as record:
            name = checkpointing.get_checkpoint_name("ckpt", 1)
            on_first_or_last_stage = is_first_or_last_pipeline_stage(None)
            averaged = average_losses_across_data_parallel_group([loss])

        assert _fallbacks(record) == [
            ("get_checkpoint_name", "pg_collection"),
            ("is_first_or_last_pipeline_stage", "pp_group"),
            ("average_losses_across_data_parallel_group", "group"),
        ]
        assert _deprecated_calls(record) == []
        assert name == checkpointing.get_checkpoint_name("ckpt", 1, pg_collection=pg_collection)
        assert on_first_or_last_stage == is_first_or_last_pipeline_stage(
            None, pp_group=pg_collection.pp
        )
        torch.testing.assert_close(
            averaged,
            average_losses_across_data_parallel_group([loss], group=pg_collection.dp_gtp_remat),
        )
