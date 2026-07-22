# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration coverage for ordinary DSA and architecture-neutral enhancements."""

from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    _validate_dsa_index_share_pipeline_split,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    absorbed_mla as absorbed_mla_module,
)
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_kernels
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAttention,
    DSAttentionSubmodules,
    is_dsa_skip_topk_layer,
    source_dsa_compute_layer,
)
from megatron.training.arguments import (
    _add_experimental_attention_variant_args,
)


def _index_share_config():
    return SimpleNamespace(
        dsa_indexer_topk=8, dsa_indexer_topk_freq=4, dsa_indexer_skip_topk_offset=1, kv_channels=16
    )


def test_index_share_schedule_and_pipeline_split_validation():
    assert not is_dsa_skip_topk_layer(1, 1, 4)
    assert is_dsa_skip_topk_layer(2, 1, 4)
    assert is_dsa_skip_topk_layer(4, 1, 4)
    assert source_dsa_compute_layer(4, 1, 4) == 1
    assert source_dsa_compute_layer(6, 1, 4) == 5

    with pytest.raises(ValueError, match="layer_number"):
        is_dsa_skip_topk_layer(0, 0, 1)
    with pytest.raises(ValueError, match="skip_topk_offset"):
        is_dsa_skip_topk_layer(1, -1, 1)
    with pytest.raises(ValueError, match="topk_freq"):
        is_dsa_skip_topk_layer(1, 0, 0)

    config = SimpleNamespace(
        experimental_attention_variant="dsa",
        dsa_indexer_topk_freq=4,
        dsa_indexer_skip_topk_offset=1,
    )
    _validate_dsa_index_share_pipeline_split(config, [0, 1, 2, 3])
    with pytest.raises(RuntimeError, match="pipeline split is invalid"):
        _validate_dsa_index_share_pipeline_split(config, [1, 2, 3, 4])


def test_mtp_layer_number_offsets_index_share_schedule():
    """An MTP DSA layer must share top-k against its global expanded layer number."""
    config = SimpleNamespace(
        num_layers=8,
        dsa_indexer_topk=4,
        dsa_indexer_topk_freq=4,
        dsa_indexer_skip_topk_offset=1,
        kv_channels=16,
    )
    pg_collection = SimpleNamespace(tp=object(), cp=object())

    attention = DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=object()),
        layer_number=2,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="all_gather",
        pg_collection=pg_collection,
        is_mtp_layer=True,
    )

    assert attention.layer_number == 10
    assert attention.skip_topk
    assert attention.source_layer == 9
    assert attention.indexer is None


def test_index_share_skip_layer_uses_request_scoped_holders(monkeypatch):
    def fail_build_module(*_args, **_kwargs):
        raise AssertionError("skip layers must not build indexer modules")

    monkeypatch.setattr(
        "megatron.core.transformer.experimental_attention_variant.dsa.build_module",
        fail_build_module,
    )
    config = _index_share_config()
    attention = DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=object()),
        layer_number=2,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        softmax_scale=1.0,
        pg_collection=SimpleNamespace(),
    )

    assert attention.skip_topk
    assert attention.indexer is None
    assert attention.source_layer == 1

    packed_seq_params = PackedSeqParams(qkv_format="thd")
    attention_mask = torch.empty(1)
    topk_holder = attention._get_index_share_topk_holder(packed_seq_params, attention_mask)
    length_holder = attention._get_index_share_topk_length_holder(packed_seq_params, attention_mask)

    assert topk_holder is getattr(packed_seq_params, DSAttention._HOLDER_ATTR)
    assert length_holder is getattr(packed_seq_params, DSAttention._LENGTH_HOLDER_ATTR)
    assert not hasattr(attention_mask, DSAttention._HOLDER_ATTR)
    assert not hasattr(config, DSAttention._HOLDER_ATTR)


def test_indexer_loss_autograd_surface_is_owned_by_dsa():
    assert issubclass(dsa_module.FusedDSAIndexerLoss, torch.autograd.Function)
    assert not hasattr(dsa_module, "OrdinaryFusedDSAIndexerLoss")
    assert not hasattr(dsa_module, "DSv4FusedDSAIndexerLoss")


def test_ordinary_dsa_kernels_selection_cache_and_import_errors(monkeypatch):
    config = SimpleNamespace(attention_backend="auto", dsa_kernel_backend="none")
    assert dsa_kernels._get_backend_module_name(config) is None
    assert not dsa_kernels.use_fused_dsa_kernels(config)

    config.dsa_kernel_backend = "tilelang"
    assert (
        dsa_kernels._get_backend_module_name(config)
        == "megatron.core.transformer.experimental_attention_variant.dsa_tilelang_kernels"
    )
    assert dsa_kernels.use_fused_dsa_kernels(config)

    config.dsa_kernel_backend = "cudnn"
    assert (
        dsa_kernels._get_backend_module_name(config)
        == "megatron.core.transformer.experimental_attention_variant.dsa_cudnn_kernels"
    )
    config.attention_backend = "unfused"
    assert not dsa_kernels.use_fused_dsa_kernels(config)

    config.attention_backend = "auto"
    config.dsa_kernel_backend = "invalid"
    with pytest.raises(ValueError, match="dsa_kernel_backend"):
        dsa_kernels._get_backend_module_name(config)

    config.dsa_kernel_backend = "tilelang"
    fake_backend = SimpleNamespace()
    imported = []

    def fake_import_module(module_name):
        imported.append(module_name)
        return fake_backend

    monkeypatch.setattr(dsa_kernels, "import_module", fake_import_module)
    monkeypatch.setattr(dsa_kernels, "_BACKEND", None)
    monkeypatch.setattr(dsa_kernels, "_BACKEND_SELECTION", None)
    assert dsa_kernels._load_backend(config) is fake_backend
    assert dsa_kernels._load_backend(config) is fake_backend
    assert imported == [
        "megatron.core.transformer.experimental_attention_variant.dsa_tilelang_kernels"
    ]

    config.dsa_kernel_backend = "cudnn"
    monkeypatch.setattr(
        dsa_kernels,
        "import_module",
        lambda _module_name: (_ for _ in ()).throw(OSError("missing backend")),
    )
    with pytest.raises(RuntimeError, match="Failed to import DSA kernel backend"):
        dsa_kernels._load_backend(config)


def test_deprecated_dsa_kernel_fusion_cli_has_no_implicit_true_default():
    parser = ArgumentParser()
    _add_experimental_attention_variant_args(parser)

    args = parser.parse_args([])
    assert args.apply_dsa_kernel_fusion is None

    args = parser.parse_args(["--no-dsa-kernel-fusion"])
    assert args.apply_dsa_kernel_fusion is False


def test_ordinary_dsa_kernels_dependency_validation(monkeypatch):
    from megatron.core import utils as core_utils

    core_utils._validate_dsa_kernel_backend_dependencies("none")
    with pytest.raises(ValueError, match="dsa_kernel_backend"):
        core_utils._validate_dsa_kernel_backend_dependencies("invalid")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        core_utils._validate_dsa_kernel_backend_dependencies("tilelang")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        core_utils, "_missing_tilelang_dsa_kernel_dependencies", lambda: ["TileLang SparseMLA"]
    )
    with pytest.raises(ValueError, match="TileLang SparseMLA"):
        core_utils._validate_dsa_kernel_backend_dependencies("tilelang")

    monkeypatch.setattr(core_utils, "_missing_tilelang_dsa_kernel_dependencies", lambda: [])
    core_utils._validate_dsa_kernel_backend_dependencies("tilelang")

    monkeypatch.setattr(core_utils, "_missing_cudnn_dsa_kernel_dependencies", lambda: ["flash_mla"])
    with pytest.raises(ValueError, match="flash_mla"):
        core_utils._validate_dsa_kernel_backend_dependencies("cudnn")


def test_checkpointed_absorbed_attention_keeps_metadata_out_of_tensor_args(monkeypatch):
    packed_seq_params = PackedSeqParams(qkv_format="thd")
    checkpoint_args = None

    def fake_checkpoint(run_function, distribute_saved_activations, *args):
        nonlocal checkpoint_args
        del distribute_saved_activations
        checkpoint_args = args
        assert all(torch.is_tensor(arg) for arg in args)
        return run_function(*args)

    class CoreAttention(torch.nn.Module):
        def forward(self, query, key, *, value, attention_mask, **kwargs):
            del query, key, value, attention_mask
            assert kwargs["packed_seq_params"] is packed_seq_params
            assert kwargs["position_ids"] is None
            return kwargs["x"]

    dummy_attention = SimpleNamespace(
        attn_mask_type=AttnMaskType.causal, core_attention=CoreAttention()
    )
    monkeypatch.setattr(absorbed_mla_module.tensor_parallel, "checkpoint", fake_checkpoint)

    hidden_states = torch.randn(4, 1, 8)
    output = AbsorbedMLASelfAttention._checkpointed_attention_forward(
        dummy_attention,
        q_absorbed=torch.randn(4, 1, 2, 8),
        k_compressed=torch.randn(4, 1, 1, 8),
        hidden_states=hidden_states,
        q_compressed=torch.randn(4, 1, 8),
        attention_mask=torch.empty(1),
        up_v_weight=torch.randn(2, 4, 4),
        position_ids=None,
        packed_seq_params=packed_seq_params,
    )

    assert checkpoint_args is not None
    assert all(arg is not packed_seq_params for arg in checkpoint_args)
    assert all(arg is not None for arg in checkpoint_args)
    assert output is hidden_states


def test_absorbed_mla_forward_uses_and_restores_dynamic_cp_group():
    original_cp_group = object()
    dynamic_cp_group = object()
    pg_collection = SimpleNamespace(cp=original_cp_group)
    observed_groups = []

    hidden_states = torch.randn(4, 1, 6)
    q_absorbed = torch.randn(4, 1, 2, 2)
    kv_compressed = torch.randn(4, 1, 1, 2)
    q_compressed = torch.randn(4, 1, 2)
    v_up_weight = torch.randn(2, 3, 2)

    def get_query_key_value_tensors(
        hidden_states_arg, key_value_states, packed_seq_params, inference_context=None
    ):
        del hidden_states_arg, key_value_states, packed_seq_params, inference_context
        observed_groups.append(pg_collection.cp)
        return q_absorbed, kv_compressed, q_compressed

    class CoreAttention:
        consumes_absorbed_v_up_projection = True

        def __call__(self, query, key, **kwargs):
            del query, key
            observed_groups.append(pg_collection.cp)
            return kwargs["x"]

    def linear_proj(core_attn_out):
        observed_groups.append(pg_collection.cp)
        return core_attn_out, None

    dummy_attention = SimpleNamespace(
        training=False,
        cache_mla_latents=False,
        pg_collection=pg_collection,
        get_query_key_value_tensors=get_query_key_value_tensors,
        _get_v_up_weight=lambda: v_up_weight,
        checkpoint_core_attention=False,
        core_attention=CoreAttention(),
        attn_mask_type=AttnMaskType.causal,
        num_attention_heads_per_partition=2,
        config=SimpleNamespace(kv_lora_rank=2, v_head_dim=3, tensor_model_parallel_size=1),
        recompute_up_proj=False,
        linear_proj=linear_proj,
    )
    packed_seq_params = PackedSeqParams(
        qkv_format="thd", local_cp_size=2, cp_group=dynamic_cp_group
    )

    output, bias = AbsorbedMLASelfAttention.forward(
        dummy_attention, hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
    )

    assert bias is None
    assert output is hidden_states
    assert observed_groups == [dynamic_cp_group, dynamic_cp_group, dynamic_cp_group]
    assert pg_collection.cp is original_cp_group


@pytest.mark.parametrize("target_layout", ["combined", "split"])
def test_absorbed_mla_loads_combined_and_split_kv_up_checkpoints(monkeypatch, target_layout):
    dummy_attention = object.__new__(AbsorbedMLASelfAttention)
    dummy_attention.num_attention_heads_per_partition = 2
    dummy_attention.config = SimpleNamespace(qk_head_dim=2, v_head_dim=3, kv_lora_rank=4)
    dummy_attention._uses_combined_kv_up_projection = target_layout == "combined"

    prefix = "self_attention."
    k_weight = torch.arange(2 * 2 * 4, dtype=torch.float32).view(2 * 2, 4)
    v_weight = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2 * 3, 4)
    combined_weight = (
        torch.cat((k_weight.view(2, 2, 4), v_weight.view(2, 3, 4)), dim=1)
        .contiguous()
        .view(2 * (2 + 3), 4)
    )
    extra_state = torch.empty(0)
    if target_layout == "combined":
        state_dict = {
            f"{prefix}linear_k_up_proj.weight": k_weight.clone(),
            f"{prefix}linear_v_up_proj.weight": v_weight.clone(),
            f"{prefix}linear_k_up_proj._extra_state": extra_state,
            f"{prefix}linear_v_up_proj._extra_state": extra_state.clone(),
        }
    else:
        state_dict = {
            f"{prefix}linear_kv_up_proj.weight": combined_weight.clone(),
            f"{prefix}linear_kv_up_proj._extra_state": extra_state,
        }

    captured_state_dict = {}

    def fake_super_load(self, state_dict, *args, **kwargs):
        del self, args, kwargs
        captured_state_dict.update(state_dict)

    monkeypatch.setattr(absorbed_mla_module.Attention, "_load_from_state_dict", fake_super_load)
    AbsorbedMLASelfAttention._load_from_state_dict(
        dummy_attention, state_dict, prefix, {}, True, [], [], []
    )

    if target_layout == "combined":
        torch.testing.assert_close(
            captured_state_dict[f"{prefix}linear_kv_up_proj.weight"], combined_weight
        )
        assert f"{prefix}linear_k_up_proj.weight" not in captured_state_dict
        assert f"{prefix}linear_v_up_proj.weight" not in captured_state_dict
        assert f"{prefix}linear_kv_up_proj._extra_state" in captured_state_dict
    else:
        torch.testing.assert_close(
            captured_state_dict[f"{prefix}linear_k_up_proj.weight"], k_weight
        )
        torch.testing.assert_close(
            captured_state_dict[f"{prefix}linear_v_up_proj.weight"], v_weight
        )
        assert f"{prefix}linear_kv_up_proj.weight" not in captured_state_dict
        assert f"{prefix}linear_k_up_proj._extra_state" in captured_state_dict
        assert f"{prefix}linear_v_up_proj._extra_state" in captured_state_dict


def test_indexer_loss_tracker_grows_for_mtp_layer_numbers():
    """MTP-expanded hybrid layer numbers must not overrun the base-model tracker."""
    helper = dsa_module.DSAIndexerLossLoggingHelper
    reduce_group = object()
    avg_group = object()
    helper.tracker.clear()
    helper.tracker["values"] = torch.tensor([1.0, 2.0])

    try:
        helper.save_loss_to_tracker(
            loss=torch.tensor(3.0),
            layer_number=5,
            num_layers=2,
            reduce_group=reduce_group,
            avg_group=avg_group,
        )

        torch.testing.assert_close(
            helper.tracker["values"], torch.tensor([1.0, 2.0, 0.0, 0.0, 3.0])
        )
        assert helper.tracker["reduce_group"] is reduce_group
        assert helper.tracker["avg_group"] is avg_group
    finally:
        helper.tracker.clear()


def test_dsv4_metric_logging_preserves_graph_groups_and_uses_indexer_layer_count(monkeypatch):
    """CUDA Graph reuse keeps groups, and only ratio-4 DSv4 layers enter the average."""
    helper = dsa_module.DSAIndexerLossLoggingHelper
    reduce_group = object()
    avg_group = object()
    helper.tracker.clear()
    helper.tracker.update(
        {
            "values": torch.tensor([2.0, 0.0, 6.0, 0.0]),
            "reduce_group": reduce_group,
            "avg_group": avg_group,
        }
    )
    recorded = []

    class Writer:
        @staticmethod
        def add_scalar(name, value, iteration):
            recorded.append((name, value.clone(), iteration))

    monkeypatch.setattr(helper, "reduce_loss_in_tracker", lambda num_layers=None: None)

    try:
        helper.track_indexer_metrics(
            loss_scale=0.5, iteration=7, writer=Writer(), num_indexer_layers=2, preserve_groups=True
        )

        assert len(recorded) == 1
        name, value, iteration = recorded[0]
        assert name == "indexer loss"
        torch.testing.assert_close(value, torch.tensor(2.0))
        assert iteration == 7
        torch.testing.assert_close(helper.tracker["values"], torch.zeros(4))
        assert helper.tracker["reduce_group"] is reduce_group
        assert helper.tracker["avg_group"] is avg_group
    finally:
        helper.tracker.clear()
