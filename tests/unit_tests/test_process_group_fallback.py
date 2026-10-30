# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deprecated global process-group fallbacks keep working and warn once per owner."""

import logging
import warnings

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.multimodal.context_parallel import (
    gather_from_context_parallel_ranks,
    split_to_context_parallel_ranks,
)
from megatron.core.process_groups_config import (
    ProcessGroupCollection,
    ProcessGroupFallbackWarning,
    enable_strict_process_group_fallbacks,
    resolve_process_groups,
    warn_global_process_group_fallback,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.rank_log_setup import suppress_duplicate_logs_off_rank0
from tests.unit_tests.test_utilities import Utils

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

requires_te_hierarchical_cp = pytest.mark.skipif(
    not HAVE_TE or not is_te_min_version("1.12.0"),
    reason="hierarchical (a2a+p2p) context parallelism needs Transformer Engine >= 1.12",
)


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _fallback_warning(record, owner, argument="pg_collection"):
    """Return the single fallback warning for ``owner`` and check it names this file."""
    matches = [
        w
        for w in record
        if issubclass(w.category, FutureWarning)
        and f"{owner} was called without `{argument}`" in str(w.message)
    ]
    assert len(matches) == 1
    # The warning points at the code that omitted the argument, not at Megatron Core or torch.
    assert matches[0].filename == __file__
    return matches[0]


def test_fallback_warning_is_emitted_once_per_owner():
    with pytest.warns(FutureWarning) as record:
        warn_global_process_group_fallback("Owner", "cp_group")
    message = str(_fallback_warning(record, "Owner", "cp_group").message)
    assert "deprecated since Megatron Core 0.21 and will be removed in 0.23" in message
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_global_process_group_fallback("Owner", "cp_group")
    with pytest.warns(FutureWarning, match="OtherOwner was called without `pg_collection`"):
        warn_global_process_group_fallback("OtherOwner")


def test_fallback_warning_is_a_future_warning():
    # Filters and tests written for FutureWarning keep matching the fallback warning.
    with pytest.warns(FutureWarning) as record:
        warn_global_process_group_fallback("Owner")
    assert _fallback_warning(record, "Owner").category is ProcessGroupFallbackWarning


def test_fallback_warning_carries_the_versions_of_the_call():
    with pytest.warns(ProcessGroupFallbackWarning) as record:
        warn_global_process_group_fallback(
            "Owner", "tp_group", deprecated_in="0.22", removed_in="0.24"
        )
    message = str(_fallback_warning(record, "Owner", "tp_group").message)
    assert "deprecated since Megatron Core 0.22 and will be removed in 0.24" in message


def test_each_omitted_argument_of_an_owner_warns_once():
    with pytest.warns(ProcessGroupFallbackWarning) as record:
        for _ in range(2):
            warn_global_process_group_fallback("Owner", "tp_group")
            warn_global_process_group_fallback("Owner", "cp_group")
    _fallback_warning(record, "Owner", "tp_group")
    _fallback_warning(record, "Owner", "cp_group")


def test_strict_mode_turns_every_fallback_into_an_error():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        warnings.warn("an unrelated FutureWarning", FutureWarning)
        # A fallback that raised is not recorded as warned, so the next one raises too.
        for _ in range(2):
            with pytest.raises(ProcessGroupFallbackWarning, match="Owner was called without"):
                warn_global_process_group_fallback("Owner")
    assert [str(w.message) for w in record] == ["an unrelated FutureWarning"]


@pytest.fixture
def restore_pytree_logger_level():
    """The rank-0 log suppression quiets torch's pytree logger; later tests expect the default."""
    pytree_logger = logging.getLogger("torch.utils._pytree")
    original_level = pytree_logger.level
    yield
    pytree_logger.setLevel(original_level)


def test_enable_strict_mode_raises_on_every_fallback_and_nothing_else():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        enable_strict_process_group_fallbacks()
        warnings.warn("an unrelated FutureWarning", FutureWarning)
        for _ in range(2):
            with pytest.raises(ProcessGroupFallbackWarning, match="Owner was called without"):
                warn_global_process_group_fallback("Owner")
    assert [str(w.message) for w in record] == ["an unrelated FutureWarning"]


def test_enable_strict_mode_raises_for_fallbacks_that_already_warned():
    # A fallback reached before strict mode, for example by another library, warned once.
    with pytest.warns(ProcessGroupFallbackWarning):
        warn_global_process_group_fallback("Owner", "tp_group")
    with warnings.catch_warnings():
        enable_strict_process_group_fallbacks()
        with pytest.raises(
            ProcessGroupFallbackWarning, match="Owner was called without `tp_group`"
        ):
            warn_global_process_group_fallback("Owner", "tp_group")


def test_enable_strict_mode_precedes_earlier_filters_once():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore")
        enable_strict_process_group_fallbacks()
        enable_strict_process_group_fallbacks()
        strict_filters = [
            f for f in warnings.filters if f[0] == "error" and f[2] is ProcessGroupFallbackWarning
        ]
        assert strict_filters == [warnings.filters[0]]
        with pytest.raises(ProcessGroupFallbackWarning, match="Owner was called without"):
            warn_global_process_group_fallback("Owner")


@pytest.mark.parametrize("rank", ["0", "1"], ids=["rank0", "rank1"])
@pytest.mark.parametrize("strict_first", [False, True], ids=["after", "before"])
def test_enable_strict_mode_raises_despite_rank0_log_suppression(
    rank, strict_first, monkeypatch, restore_pytree_logger_level
):
    """Rank 0 raises under strict mode, so the ranks whose logs are quieted must raise too.

    A rank that only warned would go on and wait for rank 0 at its next collective.
    """
    monkeypatch.setenv("RANK", rank)
    with warnings.catch_warnings():
        if strict_first:
            enable_strict_process_group_fallbacks()
        suppress_duplicate_logs_off_rank0()
        if not strict_first:
            enable_strict_process_group_fallbacks()
        with pytest.raises(ProcessGroupFallbackWarning, match="Owner was called without"):
            warn_global_process_group_fallback("Owner")


@pytest.mark.skipif(Utils.world_size < 2, reason="needs ranks other than rank 0")
def test_strict_mode_raises_on_every_rank_after_rank0_log_suppression(
    monkeypatch, restore_pytree_logger_level
):
    """The order of a training entry point: quiet duplicate logs on import, then go strict."""
    Utils.initialize_distributed()
    # The launcher's rank decides which ranks the suppression quiets.
    monkeypatch.setenv("RANK", str(torch.distributed.get_rank()))
    with warnings.catch_warnings():
        suppress_duplicate_logs_off_rank0()
        enable_strict_process_group_fallbacks()
        try:
            warn_global_process_group_fallback("Owner")
        except ProcessGroupFallbackWarning:
            raised = True
        else:
            raised = False
    raised_on_ranks = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(raised_on_ranks, raised)
    assert raised_on_ranks == [True] * len(raised_on_ranks)


def _forbid_global_collection(monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("resolve_process_groups read the global grid")

    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", forbid)
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", forbid)


def test_resolve_returns_an_explicit_collection_without_warning(monkeypatch):
    _forbid_global_collection(monkeypatch)
    # An explicit None marks an axis that is off, so it satisfies the requirement.
    pg_collection = ProcessGroupCollection(tp=object(), embd=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = resolve_process_groups(pg_collection, owner="Owner", required=("tp", "embd"))
    assert resolved is pg_collection


def test_resolve_requires_every_field_of_an_explicit_collection(monkeypatch):
    _forbid_global_collection(monkeypatch)
    with pytest.raises(ValueError, match="Owner requires pg_collection to set pp, embd"):
        resolve_process_groups(
            ProcessGroupCollection(tp=object()), owner="Owner", required=("tp", "pp", "embd")
        )


def test_resolve_in_strict_mode_raises_before_reading_an_initialized_grid(monkeypatch):
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)

    def forbid(*args, **kwargs):
        raise AssertionError("resolve_process_groups read the global grid")

    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", forbid)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        with pytest.raises(
            ProcessGroupFallbackWarning, match="Owner was called without `tp_group`"
        ):
            resolve_process_groups(None, owner="Owner", required=("tp",), argument="tp_group")


@pytest.fixture
def uninitialized_grid(monkeypatch):
    """parallel_state without a grid, as before initialize_model_parallel."""
    for name in (
        "_TENSOR_MODEL_PARALLEL_GROUP",
        "_PIPELINE_MODEL_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP",
        "_CONTEXT_PARALLEL_GROUP",
    ):
        monkeypatch.setattr(parallel_state, name, None)
    assert not parallel_state.model_parallel_is_initialized()


UNINITIALIZED = "megatron.core.parallel_state, which is not initialized"
WORLD = "torch.distributed treats a None group as the default (WORLD) group"


def test_resolve_without_collection_or_global_grid_warns_and_returns_none_groups(
    uninitialized_grid,
):
    with pytest.warns(ProcessGroupFallbackWarning) as record:
        resolved = resolve_process_groups(
            None,
            owner="Owner",
            required=("tp", "cp"),
            argument="tp_group",
            deprecated_in="0.22",
            removed_in="0.24",
        )
    # The groups the global accessors return without a grid, which torch.distributed reads as
    # WORLD; the warning says so.
    assert vars(resolved) == {"tp": None, "cp": None}
    message = str(_fallback_warning(record, "Owner", "tp_group").message)
    assert f"{UNINITIALIZED}, so the tp, cp groups it returns are None. {WORLD}" in message
    assert "deprecated since Megatron Core 0.22 and will be removed in 0.24" in message


def test_resolve_without_global_grid_names_only_the_groups_that_are_none(
    uninitialized_grid, monkeypatch
):
    # parallel_state has a tensor-parallel group but no pipeline- or data-parallel group.
    tp_group = object()
    monkeypatch.setattr(parallel_state, "_TENSOR_MODEL_PARALLEL_GROUP", tp_group)
    with pytest.warns(ProcessGroupFallbackWarning) as record:
        resolved = resolve_process_groups(None, owner="Owner", required=("tp", "cp"))
        resolved_tp = resolve_process_groups(None, owner="TPOnly", required=("tp",))
    assert vars(resolved) == {"tp": tp_group, "cp": None}
    message = str(_fallback_warning(record, "Owner").message)
    assert f"{UNINITIALIZED}, so the cp groups it returns are None. {WORLD}" in message
    # No group is None, so the warning is the one for the global grid.
    assert vars(resolved_tp) == {"tp": tp_group}
    assert UNINITIALIZED not in str(_fallback_warning(record, "TPOnly").message)


def test_resolve_without_global_grid_does_not_name_the_gtp_remat_groups(
    uninitialized_grid, monkeypatch
):
    # A None GTP weight-remat group means that the axis is off, not WORLD.
    monkeypatch.setattr(parallel_state, "_GTP_WEIGHT_REMAT_GROUP", None)
    monkeypatch.setattr(parallel_state, "_EXPERT_GTP_WEIGHT_REMAT_GROUP", None)
    required = ("tp", "gtp_remat", "expt_gtp_remat")
    with pytest.warns(ProcessGroupFallbackWarning) as record:
        resolved = resolve_process_groups(None, owner="Owner", required=required)
        resolved_gtp = resolve_process_groups(None, owner="GTPOnly", required=required[1:])
    assert vars(resolved) == {"tp": None, "gtp_remat": None, "expt_gtp_remat": None}
    message = str(_fallback_warning(record, "Owner").message)
    assert f"{UNINITIALIZED}, so the tp groups it returns are None. {WORLD}" in message
    # No other group is None, so the warning is the one for the global grid.
    assert vars(resolved_gtp) == {"gtp_remat": None, "expt_gtp_remat": None}
    assert UNINITIALIZED not in str(_fallback_warning(record, "GTPOnly").message)


def test_resolve_without_global_grid_raises_where_the_accessor_needs_a_grid(uninitialized_grid):
    # get_data_parallel_group asserts without a grid, so no None dp group reaches torch.distributed.
    with pytest.raises(AssertionError, match="data parallel group is not initialized"):
        resolve_process_groups(None, owner="Owner", required=("tp", "dp"))


def test_fallbacks_with_and_without_global_grid_warn_once_each(uninitialized_grid, monkeypatch):
    with pytest.warns(ProcessGroupFallbackWarning) as record:
        for _ in range(2):
            resolve_process_groups(None, owner="Owner", required=("tp",))
    assert UNINITIALIZED in str(_fallback_warning(record, "Owner").message)

    # Once the grid is initialized, the same owner warns again, about the global groups.
    global_groups = ProcessGroupCollection(tp=object())
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        ProcessGroupCollection, "use_mpu_process_groups", lambda required_pgs: global_groups
    )
    with pytest.warns(ProcessGroupFallbackWarning) as record:
        for _ in range(2):
            assert resolve_process_groups(None, owner="Owner", required=("tp",)) is global_groups
    assert UNINITIALIZED not in str(_fallback_warning(record, "Owner").message)


def test_enable_strict_mode_raises_on_fallback_without_global_grid(uninitialized_grid):
    with warnings.catch_warnings():
        enable_strict_process_group_fallbacks()
        with pytest.raises(ProcessGroupFallbackWarning, match=r"groups it returns are None"):
            resolve_process_groups(None, owner="Owner", required=("tp",), argument="tp_group")


@pytest.mark.filterwarnings(
    "error::megatron.core.process_groups_config.ProcessGroupFallbackWarning"
)
def test_pytest_error_filter_raises_on_fallback_without_global_grid(uninitialized_grid):
    with pytest.raises(ProcessGroupFallbackWarning, match=r"groups it returns are None"):
        resolve_process_groups(None, owner="Owner", required=("tp",), argument="tp_group")


def test_mtp_placement_without_mtp_layers_needs_no_pp_group():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert mtp_on_this_rank(mtp_num_layers=None, ignore_virtual=True) is False


class TestGlobalProcessGroupFallbacks:
    """Each fallback resolves the same groups as the global grid, which has size > 1 here."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_rotary_embedding_without_cp_group_uses_global_group(self):
        Utils.initialize_model_parallel(context_parallel_size=2)
        with pytest.warns(FutureWarning) as record:
            rope = RotaryEmbedding(kv_channels=8, rotary_percent=1.0, use_cpu_initialization=True)
        _fallback_warning(record, "RotaryEmbedding", "cp_group")
        assert rope.cp_group is parallel_state.get_context_parallel_group()

    def test_dot_product_attention_without_collection_uses_global_groups(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        config = TransformerConfig(
            num_layers=1, hidden_size=16, num_attention_heads=4, tensor_model_parallel_size=2
        )
        with pytest.warns(FutureWarning) as record:
            attention = DotProductAttention(
                config, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self"
            )
        _fallback_warning(record, "DotProductAttention")
        assert attention.tp_group is parallel_state.get_tensor_model_parallel_group()

    @pytest.mark.skipif(not HAVE_TE, reason="the GPT attention spec needs Transformer Engine")
    @pytest.mark.parametrize("test_mode", [False, True])
    def test_self_attention_without_collection_uses_global_groups(self, monkeypatch, test_mode):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
        global_dp_group = parallel_state.get_data_parallel_group()
        if not test_mode:
            # The data-parallel groups are resolved only in test_mode, so building must not
            # need an initialized DP group.
            def forbid_dp_group(*args, **kwargs):
                raise AssertionError("SelfAttention resolved the DP group outside test_mode")

            monkeypatch.setattr(parallel_state, "get_data_parallel_group", forbid_dp_group)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            use_cpu_initialization=True,
            test_mode=test_mode,
        )
        with pytest.warns(FutureWarning) as record:
            attention = SelfAttention(
                config,
                get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
            )
        # The fallback runs in Attention.__init__, reached through SelfAttention's super().
        _fallback_warning(record, "SelfAttention")
        groups = vars(attention.pg_collection)
        assert groups["tp"] is parallel_state.get_tensor_model_parallel_group()
        assert groups["cp"] is parallel_state.get_context_parallel_group()
        assert "hcp" in groups
        if test_mode:
            assert groups["dp"] is global_dp_group
        else:
            assert "dp" not in groups

    @requires_te_hierarchical_cp
    def test_self_attention_without_collection_supports_hierarchical_cp(self):
        Utils.initialize_model_parallel(
            context_parallel_size=2, hierarchical_context_parallel_sizes=[2, 1]
        )
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            context_parallel_size=2,
            hierarchical_context_parallel_sizes=[2, 1],
            cp_comm_type="a2a+p2p",
            use_cpu_initialization=True,
        )
        with pytest.warns(FutureWarning) as record:
            attention = SelfAttention(
                config,
                get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                cp_comm_type="a2a+p2p",
            )
        _fallback_warning(record, "SelfAttention")
        assert (
            attention.pg_collection.hcp is parallel_state.get_hierarchical_context_parallel_groups()
        )

    @requires_te_hierarchical_cp
    @pytest.mark.parametrize("context_parallel_size", [1, 2])
    def test_te_attention_requires_hcp_only_for_hierarchical_cp(self, context_parallel_size):
        Utils.initialize_model_parallel(context_parallel_size=context_parallel_size)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            context_parallel_size=context_parallel_size,
        )
        # A collection built for another grid carries no hcp unless the caller sets it.
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])

        def build():
            return TEDotProductAttention(
                config,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                cp_comm_type="a2a+p2p",
                pg_collection=pg_collection,
            )

        if context_parallel_size == 1:
            build()
        else:
            with pytest.raises(ValueError, match="requires pg_collection.hcp"):
                build()

    @pytest.mark.skipif(
        Utils.world_size % 4 != 0, reason="four pipeline stages need a multiple of four ranks"
    )
    def test_resolve_without_collection_warns_and_uses_global_groups(self):
        # With four pipeline stages, the middle stages are not in the embedding group.
        Utils.initialize_model_parallel(pipeline_model_parallel_size=4)
        with pytest.warns(ProcessGroupFallbackWarning) as record:
            pg_collection = resolve_process_groups(
                None,
                owner="Owner",
                required=("pp", "embd"),
                deprecated_in="0.22",
                removed_in="0.24",
            )
        message = str(_fallback_warning(record, "Owner").message)
        assert "since Megatron Core 0.22 and will be removed in 0.24" in message
        # Only the required fields are built, and a non-member's embd is an explicit None.
        assert set(vars(pg_collection)) == {"pp", "embd"}
        assert pg_collection.pp is parallel_state.get_pipeline_model_parallel_group()
        assert pg_collection.embd is parallel_state.get_embedding_group(check_initialized=False)
        middle_stage = parallel_state.get_pipeline_model_parallel_rank() in (1, 2)
        assert (pg_collection.embd is None) == middle_stage

    def test_mtp_placement_without_pp_group_uses_global_group(self):
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        with pytest.warns(FutureWarning) as record:
            on_this_rank = mtp_on_this_rank(mtp_num_layers=1, ignore_virtual=True)
        _fallback_warning(record, "mtp_on_this_rank", "pp_group")
        # Without a layout, MTP sits on the last of the two pipeline stages only.
        assert on_this_rank == (parallel_state.get_pipeline_model_parallel_rank() == 1)

    def test_context_parallel_helpers_without_cp_group_use_global_group(self):
        Utils.initialize_model_parallel(context_parallel_size=2)
        cp_rank = parallel_state.get_context_parallel_rank()

        global_t = torch.arange(12, dtype=torch.float32, device="cuda").reshape(4, 3)
        with pytest.warns(FutureWarning) as record:
            local_t, global_pad = split_to_context_parallel_ranks(global_t)
        _fallback_warning(record, "split_to_context_parallel_ranks", "cp_group")
        assert torch.equal(local_t, global_t[cp_rank * 2 : (cp_rank + 1) * 2])
        assert global_pad == 0

        global_s = torch.arange(8, dtype=torch.float32, device="cuda").reshape(1, 4, 2)
        local_s = global_s[:, cp_rank * 2 : (cp_rank + 1) * 2].contiguous()
        with pytest.warns(FutureWarning) as record:
            gathered = gather_from_context_parallel_ranks(local_s, 0)
        # The fallback runs inside an autograd Function, reached through torch's apply().
        _fallback_warning(record, "GatherFromContextParallelRanks", "cp_group")
        assert torch.equal(gathered, global_s)
