# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deprecated global process-group fallbacks keep working and warn once per owner."""

import warnings

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.multimodal.context_parallel import (
    gather_from_context_parallel_ranks,
    gather_from_context_parallel_ranks_dynamic_res,
    split_to_context_parallel_ranks,
    split_to_context_parallel_ranks_dynamic_res,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import (
    ProcessGroupCollection,
    ProcessGroupFallbackWarning,
    resolve_process_groups,
    warn_global_process_group_fallback,
)
from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    mtp_on_this_rank,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
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


def test_warning_hook_that_reaches_the_same_fallback_does_not_recurse():
    shown = []

    def showwarning(message, category, *args, **kwargs):
        shown.append(str(message))
        # For example, a logging hook whose code builds a component that falls back too.
        warn_global_process_group_fallback("Owner")

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = showwarning
        warn_global_process_group_fallback("Owner")
        warn_global_process_group_fallback("Owner")
    assert len(shown) == 1
    assert "Owner was called without `pg_collection`" in shown[0]


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


def _forbid_global_collection(monkeypatch):
    def forbid(*args, **kwargs):
        raise AssertionError("resolve_process_groups read the global grid")

    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", forbid)


def test_resolve_returns_an_explicit_collection_without_warning(monkeypatch):
    _forbid_global_collection(monkeypatch)
    # An explicit None marks an axis that is off, so it satisfies the requirement.
    pg_collection = ProcessGroupCollection(tp=object(), embd=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = resolve_process_groups(pg_collection, owner="Owner", required=("tp", "embd"))
    assert resolved is pg_collection


def test_resolve_rejects_a_single_name_before_falling_back(monkeypatch):
    _forbid_global_collection(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(TypeError, match=r"\('tp',\), not the string 'tp'"):
            resolve_process_groups(None, owner="Owner", required="tp")


def test_resolve_requires_every_field_of_an_explicit_collection(monkeypatch):
    _forbid_global_collection(monkeypatch)
    with pytest.raises(ValueError, match="Owner requires pg_collection to set pp, embd"):
        resolve_process_groups(
            ProcessGroupCollection(tp=object()), owner="Owner", required=("tp", "pp", "embd")
        )


def test_resolve_in_strict_mode_raises_before_reading_the_global_grid(monkeypatch):
    _forbid_global_collection(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ProcessGroupFallbackWarning)
        with pytest.raises(
            ProcessGroupFallbackWarning, match="Owner was called without `tp_group`"
        ):
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
    @pytest.mark.parametrize(
        "test_mode, qk_layernorm", [(False, False), (False, True), (True, False), (True, True)]
    )
    def test_self_attention_without_collection_uses_global_groups(
        self, monkeypatch, test_mode, qk_layernorm
    ):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
        global_dp_group = parallel_state.get_data_parallel_group()
        # Only run_realtime_tests reads the DP groups, and only with test_mode and qk_layernorm.
        needs_dp = test_mode and qk_layernorm
        if not needs_dp:
            # Building must not need an initialized DP group.
            def forbid_dp_group(*args, **kwargs):
                raise AssertionError("SelfAttention resolved the DP group without needing it")

            monkeypatch.setattr(parallel_state, "get_data_parallel_group", forbid_dp_group)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            use_cpu_initialization=True,
            test_mode=test_mode,
            qk_layernorm=qk_layernorm,
        )
        with pytest.warns(FutureWarning) as record:
            attention = SelfAttention(
                config,
                get_gpt_layer_with_transformer_engine_submodules(
                    qk_layernorm=qk_layernorm
                ).self_attention.submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
            )
        # The fallback runs in Attention.__init__, reached through SelfAttention's super().
        _fallback_warning(record, "SelfAttention")
        groups = vars(attention.pg_collection)
        assert groups["tp"] is parallel_state.get_tensor_model_parallel_group()
        assert groups["cp"] is parallel_state.get_context_parallel_group()
        assert "hcp" in groups
        if needs_dp:
            # Without GTP remat the replicate group and the GTP-inclusive group are one group.
            assert groups["dp"] is global_dp_group
            assert groups["dp_gtp_remat"] is global_dp_group
        else:
            assert "dp" not in groups
            assert "dp_gtp_remat" not in groups

    @pytest.mark.skipif(not HAVE_TE, reason="the GPT attention spec needs Transformer Engine")
    @pytest.mark.parametrize(
        "test_mode, qk_layernorm", [(False, False), (False, True), (True, False)]
    )
    def test_self_attention_without_collection_builds_before_initialization(
        self, test_mode, qk_layernorm
    ):
        """Before initialize_model_parallel() the fallback finds no group and TE defers its own."""
        Utils.destroy_model_parallel()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            test_mode=test_mode,
            qk_layernorm=qk_layernorm,
        )
        submodules = get_gpt_layer_with_transformer_engine_submodules(
            qk_layernorm=qk_layernorm
        ).self_attention.submodules
        with pytest.warns(FutureWarning) as record:
            attention = SelfAttention(
                config, submodules, layer_number=1, attn_mask_type=AttnMaskType.causal
            )
        _fallback_warning(record, "SelfAttention")
        # Core attention is handed no collection, so it takes its own fallback and warns as well.
        _fallback_warning(record, "TEDotProductAttention")
        assert attention.tp_group is None
        assert attention.core_attention._tp_group is None
        # The linears are handed no collection either. With the fallback's all-None collection
        # a per-module FP8/FP4 recipe would raise for its missing amax group at forward time,
        # instead of using the global group once model parallelism is initialized.
        assert attention.linear_qkv._pg_collection is None
        assert attention.linear_proj._pg_collection is None

        # An explicit collection without a TP group is still rejected.
        with pytest.raises(ValueError, match="non-None tp process group"):
            SelfAttention(
                config,
                submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                pg_collection=ProcessGroupCollection(tp=None, cp=None),
            )

    @pytest.mark.skipif(not HAVE_TE, reason="the GPT attention spec needs Transformer Engine")
    def test_self_attention_realtime_tests_need_an_initialized_dp_group(self):
        """With test_mode and qk_layernorm the fallback reads the DP group when it is built."""
        Utils.destroy_model_parallel()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            test_mode=True,
            qk_layernorm=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with pytest.raises(AssertionError, match="is not initialized"):
                SelfAttention(
                    config,
                    get_gpt_layer_with_transformer_engine_submodules(
                        qk_layernorm=True
                    ).self_attention.submodules,
                    layer_number=1,
                    attn_mask_type=AttnMaskType.causal,
                )

    @pytest.mark.skipif(not HAVE_TE, reason="the GPT attention spec needs Transformer Engine")
    @pytest.mark.skipif(not HAVE_GTP, reason="GTP requires Transformer Engine >= 2.19")
    @pytest.mark.skipif(Utils.world_size % 2 != 0, reason="GTP remat of 2 needs an even world size")
    @pytest.mark.parametrize("collection", ["omitted", "full", "dp_only"])
    def test_realtime_tests_compare_qk_layernorm_across_gtp_remat_peers(
        self, monkeypatch, collection
    ):
        """run_realtime_tests covers GTP-remat peers, as the global DP accessors did by default."""
        Utils.initialize_model_parallel(gtp_remat_size=2)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            test_mode=True,
            qk_layernorm=True,
        )
        kwargs = {}
        if collection == "full":
            kwargs["pg_collection"] = ProcessGroupCollection.use_mpu_process_groups()
        elif collection == "dp_only":
            # A collection for a custom grid may carry only its own dp group.
            kwargs["pg_collection"] = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=['tp', 'cp', 'dp']
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            attention = SelfAttention(
                config,
                get_gpt_layer_with_transformer_engine_submodules(
                    qk_layernorm=True
                ).self_attention.submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                **kwargs,
            ).cuda()

        # Identical parameters pass on every rank.
        attention.run_realtime_tests()

        # Each replicate-DP group stays consistent; the groups, one per GTP-remat peer, differ.
        replicate_dp_group = parallel_state.get_data_parallel_group(with_gtp_remat=False)
        replicate_dp_ranks = torch.distributed.get_process_group_ranks(replicate_dp_group)
        with torch.no_grad():
            attention.q_layernorm.weight.fill_(1.0 + min(replicate_dp_ranks))

        if collection == "dp_only":

            def forbid_dp_group(*args, **kwargs):
                raise AssertionError("run_realtime_tests read the global DP group")

            monkeypatch.setattr(parallel_state, "get_data_parallel_group", forbid_dp_group)
            attention.run_realtime_tests()
        else:
            with pytest.raises(AssertionError, match="Discrepancy between q_w in DP ranks"):
                attention.run_realtime_tests()

    @pytest.mark.skipif(not HAVE_TE, reason="the GPT attention spec needs Transformer Engine")
    def test_self_attention_test_mode_requires_a_dp_group(self):
        Utils.initialize_model_parallel()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            test_mode=True,
            qk_layernorm=True,
        )
        with pytest.raises(ValueError, match="requires a dp process group"):
            SelfAttention(
                config,
                get_gpt_layer_with_transformer_engine_submodules(
                    qk_layernorm=True
                ).self_attention.submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                    required_pgs=['tp', 'cp']
                ),
            )

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

    @pytest.mark.skipif(not HAVE_TE, reason="TEDotProductAttention needs Transformer Engine")
    def test_te_attention_without_collection_uses_global_groups(self):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            tensor_model_parallel_size=2,
            context_parallel_size=2,
        )
        with pytest.warns(FutureWarning) as record:
            attention = TEDotProductAttention(
                config, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self"
            )
        _fallback_warning(record, "TEDotProductAttention")
        assert attention._tp_group is parallel_state.get_tensor_model_parallel_group()
        assert attention.cp_group is parallel_state.get_context_parallel_group()

    @pytest.mark.skipif(not HAVE_TE, reason="context parallelism needs TEDotProductAttention")
    def test_mtp_block_without_collection_uses_global_groups(self):
        Utils.initialize_model_parallel(context_parallel_size=2)
        config = TransformerConfig(
            mtp_num_layers=1,
            num_layers=2,
            hidden_size=64,
            num_attention_heads=8,
            use_cpu_initialization=True,
            context_parallel_size=2,
        )
        spec = get_gpt_mtp_block_spec(
            config=config,
            spec=get_gpt_layer_with_transformer_engine_spec(),
            use_transformer_engine=True,
        )
        with pytest.warns(FutureWarning) as record:
            mtp = MultiTokenPredictionBlock(config=config, spec=spec)
        _fallback_warning(record, "MultiTokenPredictionBlock")
        assert mtp.cp_group is parallel_state.get_context_parallel_group()
        assert mtp.tp_group is parallel_state.get_tensor_model_parallel_group()

    @pytest.mark.skipif(not HAVE_TE, reason="the GPT attention spec needs Transformer Engine")
    def test_mtp_block_without_collection_resolves_dp_groups_for_realtime_tests(self):
        Utils.initialize_model_parallel()
        config = TransformerConfig(
            mtp_num_layers=1,
            num_layers=2,
            hidden_size=64,
            num_attention_heads=8,
            use_cpu_initialization=True,
            test_mode=True,
            qk_layernorm=True,
        )
        spec = get_gpt_mtp_block_spec(
            config=config,
            spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
            use_transformer_engine=True,
        )
        with pytest.warns(FutureWarning) as record:
            mtp = MultiTokenPredictionBlock(config=config, spec=spec)
        _fallback_warning(record, "MultiTokenPredictionBlock")
        attentions = [m for m in mtp.modules() if isinstance(m, SelfAttention)]
        assert attentions
        for attention in attentions:
            # Without GTP remat the two groups are one object, so check the fields themselves.
            groups = vars(attention.pg_collection)
            assert groups["dp_gtp_remat"] is parallel_state.get_data_parallel_group()
            assert groups["dp"] is parallel_state.get_data_parallel_group(with_gtp_remat=False)
            assert attention._realtime_test_dp_group() is groups["dp_gtp_remat"]

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

    def test_dynamic_res_context_parallel_helpers_without_cp_group_use_global_group(self):
        Utils.initialize_model_parallel(context_parallel_size=2)
        cp_rank = parallel_state.get_context_parallel_rank()

        # Two images of one patch grid each, so each of the two CP ranks owns one image.
        patch_dim = 16
        per_img_seq = patch_dim * patch_dim
        hidden = 3 * patch_dim * patch_dim
        global_t = torch.cat(
            [torch.full((per_img_seq, hidden), float(i), device="cuda") for i in range(2)]
        ).unsqueeze(0)
        global_imgs_sizes = torch.tensor(
            [[patch_dim, patch_dim]] * 2, dtype=torch.int32, device="cuda"
        )
        cu_seqlens = torch.tensor(
            [0, per_img_seq, 2 * per_img_seq], dtype=torch.int32, device="cuda"
        )
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=per_img_seq,
            max_seqlen_kv=per_img_seq,
        )

        with pytest.warns(FutureWarning) as record:
            local_t, local_imgs_sizes, *_ = split_to_context_parallel_ranks_dynamic_res(
                global_t, global_imgs_sizes, packed_seq_params, patch_dim=patch_dim
            )
        _fallback_warning(record, "split_to_context_parallel_ranks_dynamic_res", "cp_group")
        assert local_imgs_sizes.shape == (1, 2)
        assert torch.all(local_t == float(cp_rank))

        with pytest.warns(FutureWarning) as record:
            gathered = gather_from_context_parallel_ranks_dynamic_res(
                local_t.permute(1, 0, 2).contiguous()
            )
        _fallback_warning(record, "gather_from_context_parallel_ranks_dynamic_res", "cp_group")
        torch.testing.assert_close(gathered, global_t.permute(1, 0, 2).contiguous(), rtol=0, atol=0)
