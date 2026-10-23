# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit DCP caps are checked during initialization, not per microbatch."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from megatron.core.extensions import transformer_engine as te_extension
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net import common, gdn, kda
from megatron.core.transformer import attention, multi_latent_attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


class _Group:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


@pytest.fixture
def make_attention(monkeypatch):
    """Keep MCore's real constructor/forward and replace only the TE/CUDA boundary."""
    if not te_extension.HAVE_TE:
        pytest.skip("requires Transformer Engine")

    calls = []

    def te_init(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.cp_group = kwargs.get("cp_group")
        self.cp_global_ranks = kwargs.get("cp_global_ranks")
        self.cp_comm_type = kwargs.get("cp_comm_type", "p2p")

    def set_group(self, group, ranks, stream, comm_type):
        calls.append(("set_group", group, comm_type))
        self.cp_group = group
        self.cp_global_ranks = ranks
        self.cp_comm_type = comm_type

    def attention_forward(self, query, *args, **kwargs):
        calls.append(("forward", self.cp_group, self.cp_comm_type))
        return query

    monkeypatch.setattr(te_extension.te.pytorch.DotProductAttention, "__init__", te_init)
    monkeypatch.setattr(
        te_extension.te.pytorch.DotProductAttention, "set_context_parallel_group", set_group
    )
    monkeypatch.setattr(te_extension.TEDotProductAttention, "_forward", attention_forward)
    monkeypatch.setattr(te_extension.TEDotProductAttention, "cp_stream", None)
    monkeypatch.setattr(te_extension, "is_te_min_version", lambda *a, **k: True)
    monkeypatch.setattr(te_extension, "is_layer_window_attention", lambda *a: False)
    monkeypatch.setattr(
        te_extension,
        "get_hierarchical_context_parallel_groups",
        lambda **kw: [_Group(2), _Group(4)],
    )
    monkeypatch.setattr(
        te_extension, "get_cuda_rng_tracker", lambda: SimpleNamespace(is_initialized=lambda: False)
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda: object())
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: list(range(group.size()))
    )
    monkeypatch.setenv("NVTE_APPLY_QK_LAYER_SCALING", "0")

    def make(comm="a2a", static_cp=8, cap=4, dynamic=True):
        config = SimpleNamespace(
            batch_invariant_mode=False,
            apply_query_key_layer_scaling=False,
            num_query_groups=8,
            num_attention_heads=32,
            context_parallel_size=static_cp,
            dynamic_context_parallel=dynamic,
            max_cp_size=cap,
            deterministic_mode=False,
            window_size=None,
            window_attn_skip_freq=None,
            kv_channels=4,
            softmax_type="vanilla",
            attn_logit_softcapping=None,
            qk_clip=False,
            log_max_attention_logit=False,
            attention_dropout=0.0,
            sequence_parallel=False,
            tensor_model_parallel_size=2,
        )
        module = te_extension.TEDotProductAttention(
            config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            cp_comm_type=comm,
            pg_collection=SimpleNamespace(tp=_Group(2), cp=_Group(static_cp), hcp=[]),
        )
        return module, calls

    return make


def _run_attention(module, heads, runtime_cp=4, layout="thd"):
    shapes = {
        "thd": lambda h: (8, h, 4),
        "sbhd": lambda h: (8, 1, h, 4),
        "bshd": lambda h: (1, 8, h, 4),
    }
    tensors = [torch.empty(shapes[layout](h), device="meta") for h in heads]
    packed = PackedSeqParams(local_cp_size=runtime_cp, cp_group=_Group(runtime_cp))
    result = module.forward(*tensors, None, AttnMaskType.causal, packed_seq_params=packed)
    assert result is tensors[0]


@pytest.mark.parametrize("layout", ["thd", "sbhd", "bshd"])
@pytest.mark.parametrize("heads", [(16, 4, 4), (16, 16, 16)])
def test_attention_binds_runtime_group_and_restores_static_group(make_attention, layout, heads):
    module, calls = make_attention()
    static_group = module.cp_group
    _run_attention(module, heads, layout=layout)
    assert calls[0][1].size() == 4
    assert calls[1][0] == "forward"
    assert module.cp_group is static_group


@pytest.mark.parametrize("heads", [(2, 4, 4), (16, 2, 4), (16, 4, 2), (16, 1, 1), (16, 0, 0)])
def test_a2a_cap_checks_each_local_head_count_at_initialization(heads):
    config = SimpleNamespace(dynamic_context_parallel=True, max_cp_size=4, context_parallel_size=8)
    with pytest.raises(ValueError, match="TP-local Q/K/V.*max_cp_size"):
        TransformerConfig._validate_a2a_max_cp_size(config, "a2a", heads, 2)


def test_replicated_kv_head_is_valid_with_cap_one(make_attention):
    module, calls = make_attention(cap=1)
    _run_attention(module, (16, 1, 1), runtime_cp=1)
    assert calls[1] == ("forward", None, "a2a")


@pytest.mark.parametrize("comm", ["a2a", "p2p", "all_gather"])
def test_attention_does_not_add_runtime_expansion_cap_check(make_attention, comm):
    module, calls = make_attention(comm=comm)
    _run_attention(module, (16, 8, 8), runtime_cp=8)
    assert calls[1][0] == "forward"


@pytest.mark.parametrize("comm", ["p2p", "all_gather"])
def test_non_a2a_layer_does_not_require_head_divisibility(make_attention, comm):
    module, calls = make_attention(comm=comm)
    module.config.cp_comm_type = ["a2a", comm]
    _run_attention(module, (1, 1, 1))
    assert calls[1][0] == "forward"


@pytest.mark.parametrize("comm", ["a2a", "p2p", "all_gather"])
def test_cp1_preserves_requested_communication_with_explicit_cap(make_attention, comm):
    module, calls = make_attention(comm=comm, static_cp=1)
    assert module.cp_comm_type == comm
    assert module.cp_group is None
    assert te_extension.TEDotProductAttention.cp_stream is None
    _run_attention(module, (16, 4, 4))
    assert all(call[2] == comm for call in calls)
    assert module.cp_group is None


@pytest.mark.parametrize("dynamic, cap", [(True, None), (False, 4)])
def test_attention_default_or_static_path_is_unchanged(make_attention, dynamic, cap):
    module, calls = make_attention(dynamic=dynamic, cap=cap)
    _run_attention(module, (1, 1, 1), runtime_cp=8)
    assert calls[1][0] == "forward"


@pytest.mark.parametrize("dynamic, cap", [(True, None), (False, 4), (True, 4)])
def test_hierarchical_attention_construction_is_unchanged(make_attention, dynamic, cap):
    module, _ = make_attention(comm="a2a+p2p", dynamic=dynamic, cap=cap)
    assert module.cp_comm_type == "a2a+p2p"
    assert [group.size() for group in module.cp_group] == [2, 4]


@pytest.fixture
def check_attention_constructor(monkeypatch):
    """Exercise real owner constructors after their parent resolves head metadata."""

    class ReachedParameterSetup(Exception):
        pass

    def stop_before_parameters(*args, **kwargs):
        raise ReachedParameterSetup

    def initialize_metadata(self, config, pg_collection, **kwargs):
        torch.nn.Module.__init__(self)
        self.config = config
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.world_size = pg_collection.tp.size()
        self.num_query_groups_per_partition = max(1, config.num_query_groups // self.world_size)
        self.num_attention_heads_per_partition = config.num_attention_heads // min(
            self.world_size, config.num_query_groups
        )
        if isinstance(self, multi_latent_attention.MultiLatentAttention):
            self.num_attention_heads_per_partition = config.mla_local_heads
        self.query_projection_size = config.num_attention_heads * config.kv_channels
        self.kv_projection_size = config.num_query_groups * config.kv_channels

    monkeypatch.setattr(attention.Attention, "__init__", initialize_metadata)
    monkeypatch.setattr(
        multi_latent_attention.MultiLatentAttention, "__init__", initialize_metadata
    )
    monkeypatch.setattr(
        multi_latent_attention.MLASelfAttention, "_resolve_qk_norm_config", stop_before_parameters
    )

    def check(layer_type=attention.SelfAttention, actual_tp=2, **overrides):
        config = SimpleNamespace(
            dynamic_context_parallel=True,
            max_cp_size=2,
            context_parallel_size=8,
            tensor_model_parallel_size=2,
            num_attention_heads=32,
            num_query_groups=4,
            hidden_size=128,
            kv_channels=4,
            init_method=lambda *args: None,
            gated_attention_proj_granularity="elementwise",
            attention_output_gate=False,
            head_wise_attn_gate=False,
            add_bias_linear=False,
            add_qkv_bias=False,
            q_lora_rank=4,
            mla_local_heads=4,
        )
        comm = overrides.pop("comm", "a2a")
        config.__dict__.update(overrides)
        config._validate_a2a_max_cp_size = MethodType(
            TransformerConfig._validate_a2a_max_cp_size, config
        )
        with pytest.raises(ReachedParameterSetup):
            layer_type(
                config,
                submodules=SimpleNamespace(
                    linear_qkv=stop_before_parameters, linear_q=stop_before_parameters
                ),
                layer_number=1,
                cp_comm_type=comm,
                pg_collection=SimpleNamespace(tp=_Group(actual_tp), cp=_Group(8)),
            )

    return check


@pytest.mark.parametrize("actual_tp,cap", [(2, 2), (8, 1)])
def test_standard_constructor_handles_local_kv_and_replication(
    check_attention_constructor, actual_tp, cap
):
    check_attention_constructor(actual_tp=actual_tp, max_cp_size=cap)


@pytest.mark.parametrize("actual_tp,cap", [(2, 4), (8, 2)])
def test_standard_constructor_uses_actual_tp_to_reject_cap(
    check_attention_constructor, actual_tp, cap
):
    with pytest.raises(ValueError, match=f"TP={actual_tp}.*local_k_heads="):
        check_attention_constructor(actual_tp=actual_tp, max_cp_size=cap)


@pytest.mark.parametrize("comm", ["p2p", "all_gather", "a2a+p2p", None])
def test_standard_constructor_uses_resolved_layer_communication(
    monkeypatch, check_attention_constructor, comm
):
    monkeypatch.setattr(
        attention, "get_pg_size", lambda *args: pytest.fail("unexpected cap metadata lookup")
    )
    check_attention_constructor(comm=comm, max_cp_size=8, cp_comm_type=["a2a", comm])


@pytest.mark.parametrize("dynamic,cap", [(False, 8), (True, None)])
def test_standard_constructor_skips_cap_for_static_or_unset(
    monkeypatch, check_attention_constructor, dynamic, cap
):
    monkeypatch.setattr(
        attention, "get_pg_size", lambda *args: pytest.fail("unexpected cap metadata lookup")
    )
    check_attention_constructor(dynamic_context_parallel=dynamic, max_cp_size=cap)


def test_cross_attention_checks_mha_heads(check_attention_constructor):
    check_attention_constructor(attention.CrossAttention, num_query_groups=32, max_cp_size=16)
    with pytest.raises(ValueError, match="TP-local Q/K/V"):
        check_attention_constructor(attention.CrossAttention, num_query_groups=32, max_cp_size=32)


@pytest.mark.parametrize(
    "layer_type",
    [multi_latent_attention.MLASelfAttention, multi_latent_attention.FusedMLASelfAttention],
)
def test_mla_uses_its_expanded_heads_not_gqa_counts(check_attention_constructor, layer_type):
    check_attention_constructor(layer_type, num_query_groups=2, max_cp_size=4)
    with pytest.raises(ValueError, match="TP-local Q/K/V"):
        check_attention_constructor(layer_type, num_query_groups=2, max_cp_size=8)


def _gdn_layer(cap=2, mode="headwise", dynamic=True, local_heads=(2, 4)):
    layer = SimpleNamespace(
        config=SimpleNamespace(
            max_cp_size=cap, dynamic_context_parallel=dynamic, linear_cp_mode=mode
        ),
        cp_size=8,
        tp_size=2,
        num_key_heads=local_heads[0] * 2,
        num_value_heads=local_heads[1] * 2,
        pg_collection=SimpleNamespace(cp=_Group(8)),
    )
    layer._validate_max_cp_size = MethodType(common._GDNBase._validate_max_cp_size, layer)
    return layer


def test_gdn_headwise_allows_static_cp_above_explicit_cap():
    layer = _gdn_layer()
    layer._validate_max_cp_size()
    assert layer.cp_size == 8


@pytest.mark.parametrize("local_heads", [(1, 4), (2, 1)])
def test_gdn_headwise_validates_both_head_counts(local_heads):
    with pytest.raises(ValueError, match="local_key_heads=.*static_cp_size=8.*max_cp_size=2"):
        _gdn_layer(local_heads=local_heads)._validate_max_cp_size()


def test_gdn_chunkwise_has_no_head_divisibility_requirement():
    _gdn_layer(mode="chunkwise", local_heads=(1, 1))._validate_max_cp_size()


@pytest.mark.parametrize("mode", ["headwise", "chunkwise"])
@pytest.mark.parametrize("actual_tp", [3, 8])
def test_gdn_uses_actual_tp_without_head_replication(mode, actual_tp):
    layer = _gdn_layer(mode=mode)
    layer.config.tensor_model_parallel_size = 2
    layer.tp_size = actual_tp
    with pytest.raises(ValueError, match="actual TP size.*do not replicate heads"):
        layer._validate_max_cp_size()


@pytest.mark.parametrize("dynamic, cap", [(True, None), (False, 2)])
def test_gdn_default_or_static_path_is_unchanged(dynamic, cap):
    _gdn_layer(dynamic=dynamic, cap=cap, local_heads=(1, 1))._validate_max_cp_size()


@pytest.mark.parametrize(
    "module,forward",
    [(gdn, gdn.GatedDeltaNet._forward_training), (kda, kda.KimiDeltaAttention.forward)],
)
@pytest.mark.parametrize("mode", ["headwise", "chunkwise"])
def test_gdn_and_kda_do_not_recheck_cap_per_microbatch(monkeypatch, module, forward, mode):
    class ReachedComputation(Exception):
        pass

    class StopBeforeComputation:
        @property
        def shape(self):
            raise ReachedComputation

    def stop_at_conversion(**kwargs):
        raise ReachedComputation

    layer = _gdn_layer(mode=mode)
    layer.config.sequence_parallel = False
    layer.tp_group = _Group(2)
    layer._validate_max_cp_size = lambda *args: pytest.fail("unexpected per-microbatch cap check")
    monkeypatch.setattr(
        module, "convert_module_input_tensors_cp_partition_mode", stop_at_conversion
    )
    with pytest.raises(ReachedComputation):
        forward(layer, StopBeforeComputation(), None)


@pytest.mark.parametrize(
    "cap, dynamic, expected",
    [
        (2, True, None),
        (None, True, AssertionError),
        (2, False, AssertionError),
        (4, True, ValueError),
    ],
)
def test_gdn_constructor_uses_cap_only_for_explicit_dcp(monkeypatch, cap, dynamic, expected):
    class ReachedParameterSetup(Exception):
        pass

    observed = []

    def stop_before_parameters(self):
        observed.append((self.cp_size, self.qk_dim_local_tp))
        raise ReachedParameterSetup

    monkeypatch.setattr(common, "HAVE_FLA", True)
    monkeypatch.setattr(common._GDNBase, "_setup_variant_attrs", stop_before_parameters)
    config = SimpleNamespace(
        max_cp_size=cap,
        dynamic_context_parallel=dynamic,
        linear_cp_mode="headwise",
        sequence_parallel=False,
        hidden_size=16,
        activation_func=torch.nn.functional.silu,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
    )
    with pytest.raises(expected or ReachedParameterSetup):
        common._GDNBase(
            config,
            submodules=SimpleNamespace(),
            pg_collection=SimpleNamespace(cp=_Group(8), tp=_Group(2)),
        )
    assert observed == ([] if expected else [(8, 8)])
