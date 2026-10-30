# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MoE layers take their process groups from the caller; the global-grid fallbacks warn."""

import importlib.util
import pathlib
import re
import sys
import warnings

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker
from megatron.core.transformer.moe.moe_utils import (
    get_default_pg_collection,
    get_updated_expert_bias,
)
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_CHECKER_SPEC = importlib.util.spec_from_file_location(
    "check_process_group_usage",
    pathlib.Path(__file__).resolve().parents[4] / "tools" / "check_process_group_usage.py",
)
_checker = importlib.util.module_from_spec(_CHECKER_SPEC)
_CHECKER_SPEC.loader.exec_module(_checker)

# Fields that MoELayer validates: the ones it, its router and its token dispatchers read.
LAYER_FIELDS = ("ep", "tp", "cp", "expt_tp", "tp_ep", "tp_cp", "tp_dp_cp")
# Fields of the default collection: the layer fields and the experts' data-parallel groups.
DEFAULT_FIELDS = LAYER_FIELDS + ("expt_dp", "expt_dp_gtp_remat")

_FALLBACK_MESSAGE = re.compile(r"^(\S+) was called without `([^`]+)`")

requires_te = pytest.mark.skipif(not HAVE_TE, reason="grouped experts need Transformer Engine")


class CustomMoELayer(MoELayer):
    """A subclass: the fallback warning names the class that the caller built."""


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


@pytest.fixture(autouse=True)
def destroy_model_parallel():
    yield
    Utils.destroy_model_parallel()


def _initialize_or_skip(tp=1, ep=1, cp=1, etp=None, gtp=1, egtp=1):
    etp = tp if etp is None else etp
    for size in (tp * cp * gtp, etp * ep * egtp):
        if Utils.world_size % size:
            pytest.skip(f"needs a world size divisible by {size}")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
        gtp_remat_size=gtp,
        expert_gtp_remat_size=egtp,
    )


def _global_moe_groups():
    """Return the groups that the global accessors give for the fields of the default collection."""
    return {
        "ep": parallel_state.get_expert_model_parallel_group(),
        "tp": parallel_state.get_tensor_model_parallel_group(),
        "cp": parallel_state.get_context_parallel_group(),
        "expt_tp": parallel_state.get_expert_tensor_parallel_group(),
        "expt_dp": parallel_state.get_expert_data_parallel_group(with_gtp_remat=False),
        "expt_dp_gtp_remat": parallel_state.get_expert_data_parallel_group(check_initialized=False),
        "tp_ep": parallel_state.get_expert_tensor_and_model_parallel_group(),
        "tp_cp": parallel_state.get_tensor_and_context_parallel_group(),
        "tp_dp_cp": parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
    }


def _fallback_warnings(record):
    """Return (owner, argument) for each process-group fallback warning in ``record``."""
    found = []
    for warning in record:
        if not issubclass(warning.category, ProcessGroupFallbackWarning):
            continue
        message = str(warning.message)
        assert "since Megatron Core 0.21 and will be removed in 0.23" in message, message
        # The warning points at the code that omitted the groups, not at Megatron Core.
        assert warning.filename == __file__
        found.append(_FALLBACK_MESSAGE.match(message).groups())
    return found


def _forbid_global_process_groups(monkeypatch):
    """Make the accessors the checker counts, and the collection shim, raise when called."""

    def forbidden(*args, **kwargs):
        raise AssertionError("read a process group from megatron.core.parallel_state")

    accessors = {
        id(value)
        for name, value in vars(parallel_state).items()
        if callable(value) and _checker._is_deprecated_accessor(name)
    }
    # Patch the accessors in parallel_state and wherever a module imported them by name.
    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith("megatron.core"):
            continue
        for attribute, value in list(vars(module).items()):
            if id(value) in accessors:
                monkeypatch.setattr(module, attribute, forbidden)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", classmethod(forbidden))


def _record_all_reduce_groups(monkeypatch):
    """Return a list that collects the group of every all-reduce."""
    groups = []
    all_reduce = torch.distributed.all_reduce

    def recording_all_reduce(tensor, *args, group=None, **kwargs):
        groups.append(group)
        return all_reduce(tensor, *args, group=group, **kwargs)

    monkeypatch.setattr(torch.distributed, "all_reduce", recording_all_reduce)
    return groups


def _assert_same_groups(actual, expected):
    assert len(actual) == len(expected)
    for actual_group, expected_group in zip(actual, expected):
        assert actual_group is expected_group


def _moe_config(tp, ep, grouped_gemm):
    return TransformerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_ffn_hidden_size=64,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=grouped_gemm,
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        sequence_parallel=tp > 1,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
    )


def _moe_submodules(grouped_gemm):
    if grouped_gemm:
        layer = get_gpt_layer_with_transformer_engine_submodules(
            num_experts=4, moe_grouped_gemm=True
        )
    else:
        layer = get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False)
    return get_submodules(layer.mlp)


def _report_metrics(pg_collection):
    tracker = MoEMetricsTracker()
    tracker.record("z_loss", torch.ones((), device="cuda"), layer_number=1, num_layers=1)
    tracker.report(
        loss_scale=1.0, iteration=1, num_layers=1, num_moe_layers=1, pg_collection=pg_collection
    )


@pytest.mark.parametrize(
    "layout",
    [
        dict(),
        dict(tp=2, ep=2),
        dict(tp=2, cp=2, ep=4, etp=1),
        dict(ep=2, egtp=2),
        dict(gtp=2, ep=2),
    ],
    ids=["tp1-ep1", "tp2-ep2", "tp2-cp2-ep4-etp1", "ep2-egtp2", "gtp2-ep2"],
)
def test_default_collection_warns_and_matches_the_global_accessors(layout):
    _initialize_or_skip(**layout)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        pg_collection = get_default_pg_collection()
        get_default_pg_collection()
    assert _fallback_warnings(record) == [("get_default_pg_collection", "pg_collection")]

    expected = _global_moe_groups()
    assert set(vars(pg_collection)) == set(DEFAULT_FIELDS) == set(expected)
    for name, group in expected.items():
        assert vars(pg_collection)[name] is group, name
        assert torch.distributed.get_process_group_ranks(
            vars(pg_collection)[name]
        ) == torch.distributed.get_process_group_ranks(group)
    if layout.get("egtp", 1) > 1:
        # The replicate expert DP group and the EGTP-inclusive one differ on this grid.
        assert expected["expt_dp_gtp_remat"].size() == layout["egtp"] * expected["expt_dp"].size()


@pytest.mark.parametrize("grouped_gemm", [False, pytest.param(True, marks=requires_te)])
@pytest.mark.parametrize("tp,ep", [(1, 1), (2, 2), (1, 4)])
def test_moe_layer_without_collection_warns_once_per_owner(tp, ep, grouped_gemm):
    _initialize_or_skip(tp=tp, ep=ep)
    model_parallel_cuda_manual_seed(123)
    config = _moe_config(tp, ep, grouped_gemm)
    submodules = _moe_submodules(grouped_gemm)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        layers = [
            MoELayer(config, submodules),
            MoELayer(config, submodules),
            CustomMoELayer(config, submodules),
        ]
    assert sorted(_fallback_warnings(record)) == [
        ("CustomMoELayer", "pg_collection"),
        ("MoELayer", "pg_collection"),
    ]

    expected = _global_moe_groups()
    for layer in layers:
        router, dispatcher, experts = layer.router, layer.token_dispatcher, layer.experts
        held = {
            "ep": [layer.ep_group, dispatcher.ep_group, experts.ep_group],
            "tp": [layer.tp_group, layer.attn_tp_group, router.tp_group],
            "cp": [router.cp_group],
            "expt_tp": [router.expt_tp_group, dispatcher.tp_group, experts.tp_group],
            "tp_ep": [dispatcher.tp_ep_group],
            "tp_cp": [router.tp_cp_group],
            "tp_dp_cp": [router.tp_dp_cp_group],
        }
        if grouped_gemm:
            # The grouped linear layers keep the collection that MoELayer passed down.
            pg_collection = experts.linear_fc1._pg_collection
            assert set(vars(pg_collection)) == set(DEFAULT_FIELDS)
            held["expt_dp"] = [pg_collection.expt_dp]
            held["expt_dp_gtp_remat"] = [pg_collection.expt_dp_gtp_remat]
        else:
            held["expt_dp"] = [experts.dp_group]
        for name, groups in held.items():
            for group in groups:
                assert group is expected[name], name


def test_default_collection_without_global_grid_raises():
    Utils.initialize_distributed()
    parallel_state.destroy_model_parallel()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="parallel_state has not created the ep, tp, cp"):
            get_default_pg_collection()
        with pytest.raises(RuntimeError, match="^MoELayer was called without pg_collection"):
            MoELayer(_moe_config(1, 1, False), _moe_submodules(False))
    assert _fallback_warnings(record) == [
        ("get_default_pg_collection", "pg_collection"),
        ("MoELayer", "pg_collection"),
    ]


@requires_te
def test_explicit_collection_reads_no_global_groups(monkeypatch):
    _initialize_or_skip(tp=2, ep=2)
    model_parallel_cuda_manual_seed(123)
    # Grouped experts receive the collection. SequentialMLP builds its expert MLPs without it,
    # and their linear layers resolve the GTP_remat group from parallel_state.
    config = _moe_config(2, 2, grouped_gemm=True)
    submodules = _moe_submodules(grouped_gemm=True)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    _forbid_global_process_groups(monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        layer = MoELayer(config, submodules, pg_collection=pg_collection)
        hidden_states = torch.randn(
            16, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        output, _ = layer(hidden_states)
        output.sum().backward()
        _report_metrics(pg_collection)
        tokens_per_expert = torch.ones(1, config.num_moe_experts, device="cuda")
        get_updated_expert_bias(
            tokens_per_expert,
            torch.zeros_like(tokens_per_expert),
            0.01,
            tp_dp_cp_group=pg_collection.tp_dp_cp,
        )
    assert hidden_states.grad is not None


@pytest.mark.parametrize("field", LAYER_FIELDS)
def test_collection_without_a_layer_field_raises(field):
    _initialize_or_skip()
    model_parallel_cuda_manual_seed(123)
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    delattr(pg_collection, field)
    with pytest.raises(ValueError, match=rf"MoELayer requires pg_collection to set {field}\b"):
        MoELayer(_moe_config(1, 1, False), _moe_submodules(False), pg_collection=pg_collection)


def test_collection_without_gtp_inclusive_groups_is_accepted():
    """Collections built for inference shards set no GTP-inclusive data-parallel groups."""
    _initialize_or_skip()
    model_parallel_cuda_manual_seed(123)
    full = vars(ProcessGroupCollection.use_mpu_process_groups())
    names = (
        "tp",
        "cp",
        "pp",
        "ep",
        "embd",
        "pos_embd",
        "dp",
        "tp_cp",
        "mp",
        "expt_tp",
        "expt_dp",
        "tp_ep",
        "tp_ep_pp",
        "dp_cp",
        "tp_dp_cp",
    )
    pg_collection = ProcessGroupCollection(**{name: full[name] for name in names})
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        MoELayer(_moe_config(1, 1, False), _moe_submodules(False), pg_collection=pg_collection)


def test_moe_metrics_fallbacks_warn_and_keep_their_groups(monkeypatch):
    _initialize_or_skip(tp=2)
    global_pp = parallel_state.get_pipeline_model_parallel_group()
    global_dp = parallel_state.get_data_parallel_group()
    full = ProcessGroupCollection.use_mpu_process_groups()
    without_dp = ProcessGroupCollection.use_mpu_process_groups()
    delattr(without_dp, "dp_cp_gtp_remat")
    groups = _record_all_reduce_groups(monkeypatch)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _report_metrics(None)
        _report_metrics(None)
    assert _fallback_warnings(record) == [("MoEMetricsTracker.report", "pg_collection")]
    _assert_same_groups(groups, [global_pp, global_dp] * 2)

    groups.clear()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _report_metrics(without_dp)
    assert _fallback_warnings(record) == [
        ("MoEMetricsTracker.report", "pg_collection.dp_cp_gtp_remat")
    ]
    _assert_same_groups(groups, [without_dp.pp, global_dp])

    groups.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        _report_metrics(full)
    _assert_same_groups(groups, [full.pp, full.dp_cp_gtp_remat])


def test_expert_bias_fallback_warns_and_keeps_its_group(monkeypatch):
    _initialize_or_skip(tp=2)
    global_tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
    tokens_per_expert = torch.arange(4.0, device="cuda").unsqueeze(0) * (Utils.rank + 1)
    expert_bias = torch.zeros_like(tokens_per_expert)
    groups = _record_all_reduce_groups(monkeypatch)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fallback = get_updated_expert_bias(tokens_per_expert.clone(), expert_bias, 0.1)
    assert _fallback_warnings(record) == [("get_updated_expert_bias", "tp_dp_cp_group")]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        explicit = get_updated_expert_bias(
            tokens_per_expert.clone(), expert_bias, 0.1, tp_dp_cp_group=global_tp_dp_cp
        )
    _assert_same_groups(groups, [global_tp_dp_cp, global_tp_dp_cp])
    assert torch.equal(fallback, explicit)
