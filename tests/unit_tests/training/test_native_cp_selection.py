# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from argparse import ArgumentParser
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from megatron.training.arguments import (
    add_megatron_arguments,
    core_transformer_config_from_args,
    parse_args,
    validate_args,
)
from megatron.training.initialize import (
    _native_cp_model_unavailable_reason,
    _native_cp_transport_checker,
)
from megatron.training.models.gpt import GPTModelConfig


def _arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_native_cp_selection"])
    args = parse_args()
    args.num_layers = 2
    args.hidden_size = 128
    args.num_attention_heads = 8
    args.max_position_embeddings = 256
    args.seq_length = 256
    args.micro_batch_size = 1
    args.train_iters = 1
    args.position_embedding_type = "rope"
    args.dynamic_context_parallel = True
    args.calculate_per_token_loss = True
    args.max_seqlen_per_dp_cp_rank = 256
    args.transformer_impl = "transformer_engine"
    args.distributed_backend = "nccl"
    args.cp_comm_type = ["p2p"]
    args.bf16 = True
    return args


@pytest.fixture
def native_cp_args(monkeypatch):
    args = _arguments(monkeypatch)
    validate_args(args)
    return args


def _model(args):
    return GPTModelConfig(transformer=core_transformer_config_from_args(args), vocab_size=1024)


def _dsv4_model(args, hybrid, config_class=MLATransformerConfig):
    """Isolate model recognition from optional DSv4 kernel/config validation."""
    config = core_transformer_config_from_args(args, config_class=config_class)
    # As in the existing unsupported-attention tests, mutate after constructing
    # the config: this tests selection, not the availability of cuDNN kernels.
    config.experimental_attention_variant = "dsv4_hybrid"
    config.multi_latent_attention = True
    config.dsa_kernel_backend = "cudnn"
    config.cp_partition_mode = "contiguous"
    config.attention_cp_layout = "contiguous"
    config.linear_cp_layout = "contiguous"
    config.dsa_cp_balance_indexer = False
    config.is_hybrid_model = hybrid
    if not hybrid:
        return GPTModelConfig(transformer=config, vocab_size=1024)
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
    from megatron.training.models.hybrid import HybridModelConfig

    return HybridModelConfig(
        transformer=config,
        vocab_size=1024,
        hybrid_stack_spec=hybrid_dsv4_stack_spec(config),
        hybrid_layer_pattern="WECEHE|DE/WE",
    )


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("mtp_depth", [None, 1])
def test_native_cp_known_dsv4_builders_are_candidates(native_cp_args, hybrid, mtp_depth):
    model = _dsv4_model(native_cp_args, hybrid)
    model.transformer.mtp_num_layers = mtp_depth
    assert model.transformer.params_dtype is torch.bfloat16
    assert _native_cp_model_unavailable_reason(model, native_cp_args) is None


def test_native_cp_dsv4_resolved_cli_spec_is_candidate(native_cp_args):
    from megatron.training.argument_utils import hybrid_config_from_args

    model = _dsv4_model(native_cp_args, hybrid=True)
    native_cp_args.spec = [
        "megatron.core.models.hybrid.hybrid_layer_specs",
        "hybrid_dsv4_stack_spec",
    ]
    native_cp_args.hybrid_layer_pattern = model.hybrid_layer_pattern
    native_cp_args.vocab_size = 1024
    resolved = hybrid_config_from_args(
        native_cp_args, config=model.transformer, vocab_size_from_tokenizer=False
    )
    assert _native_cp_model_unavailable_reason(resolved, native_cp_args) is None
    # Re-created partial builders must compare by structure, not object identity.
    assert _native_cp_model_unavailable_reason(deepcopy(resolved), native_cp_args) is None
    resolved.hybrid_stack_spec.submodules.csa_layer.submodules.self_attention.submodules.core_attention = (
        torch.nn.Identity
    )
    assert _native_cp_model_unavailable_reason(resolved, native_cp_args)


@pytest.mark.parametrize("dsv4", [False, True])
@pytest.mark.parametrize("missing", [None, "all_gather", "reduce_scatter", "exchange"])
def test_native_cp_checks_dsv4_transport_interfaces(monkeypatch, native_cp_args, dsv4, missing):
    methods = {name: lambda: None for name in ("all_gather", "reduce_scatter", "exchange")}
    if missing:
        methods.pop(missing)
    module = SimpleNamespace(
        NativeCPTransport=type("Transport", (), methods),
        get_native_cp_transport_unavailable_reason=lambda parent: None,
    )
    monkeypatch.setitem(
        sys.modules, "transformer_engine.pytorch.attention.native_cp_transport", module
    )
    model = _dsv4_model(native_cp_args, hybrid=True) if dsv4 else _model(native_cp_args)
    reason = _native_cp_transport_checker(model, native_cp_args)(object())
    assert bool(reason) == (dsv4 and missing is not None)


@pytest.mark.parametrize(
    "hidden,window,expected", [(256, 128, 131072), (4096, 128, 2097152), (256, 256, 262144)]
)
def test_native_dsv4_staging_budget_is_independent_of_runtime_cp(
    monkeypatch, hidden, window, expected
):
    from megatron.core.transformer.experimental_attention_variant.csa_utils.cp_utils import (
        initialize_native_dsv4_cp_transport,
    )

    calls = []
    parent = object()
    module = SimpleNamespace(
        initialize_native_cp_transport=lambda group, size: calls.append((group, size))
    )
    monkeypatch.setitem(
        sys.modules, "transformer_engine.pytorch.attention.native_cp_transport", module
    )
    config = SimpleNamespace(
        csa_window_size=window,
        csa_compress_ratios=[0, 4, 128],
        hidden_size=hidden,
        params_dtype=torch.bfloat16,
        num_moe_experts=8,
        thd_max_packed_sequences=64,
    )
    for cp_size in (1, 3, 5, 32):
        config.context_parallel_size = cp_size
        initialize_native_dsv4_cp_transport(config, parent)
    assert calls == [(parent, expected)] * 4
    with pytest.raises(ValueError, match="dp_cp"):
        initialize_native_dsv4_cp_transport(config, None)


@pytest.mark.parametrize("hybrid", [False, True])
def test_native_cp_dsv4_config_subclass_is_not_inferred(native_cp_args, hybrid):
    class CustomMLATransformerConfig(MLATransformerConfig):
        pass

    model = _dsv4_model(native_cp_args, hybrid, config_class=CustomMLATransformerConfig)
    assert _native_cp_model_unavailable_reason(model, native_cp_args)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("experimental_attention_variant", "dsa"),
        ("multi_latent_attention", False),
        ("dsa_kernel_backend", "none"),
        ("dsa_kernel_backend", "tilelang"),
        ("cp_partition_mode", "zigzag"),
        ("dsa_cp_balance_indexer", True),
        ("tensor_model_parallel_size", 2),
        ("transformer_impl", "local"),
        ("use_kitchen", True),
        ("use_kitchen_attention", True),
        ("fallback_to_eager_attn", True),
        ("params_dtype", torch.float16),
        ("params_dtype", torch.float32),
        ("dsa_indexer_precision", "mxfp8"),
        ("fp8", "hybrid"),
        ("fp4", "e2m1"),
        ("fp8_dot_product_attention", True),
        ("quant_recipe", object()),
        ("cuda_graph_impl", "transformer_engine"),
        ("cp_comm_type", "a2a+p2p"),
    ],
)
def test_native_cp_dsv4_unsupported_paths_are_not_inferred(native_cp_args, hybrid, field, value):
    model = _dsv4_model(native_cp_args, hybrid)
    setattr(model.transformer, field, value)
    assert _native_cp_model_unavailable_reason(model, native_cp_args)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("builder", "custom.DSv4Builder"),
        ("restore_modelopt_state", True),
        ("pre_wrap_hooks", [lambda model: model]),
        ("post_wrap_hooks", [lambda model: model]),
    ],
)
def test_native_cp_dsv4_custom_builders_and_hooks_are_not_inferred(
    native_cp_args, hybrid, field, value
):
    model = _dsv4_model(native_cp_args, hybrid)
    setattr(model, field, value)
    assert _native_cp_model_unavailable_reason(model, native_cp_args)


@pytest.mark.parametrize("pattern", [None, "", "WEGE", "WECE/ME"])
def test_native_cp_dsv4_hybrid_rejects_non_dsv4_patterns(native_cp_args, pattern):
    model = _dsv4_model(native_cp_args, hybrid=True)
    model.hybrid_layer_pattern = pattern
    assert _native_cp_model_unavailable_reason(model, native_cp_args)


@pytest.mark.parametrize("spec", [None, lambda config: None, ModuleSpec(module=torch.nn.Identity)])
def test_native_cp_dsv4_hybrid_rejects_unknown_stack_specs(native_cp_args, spec):
    model = _dsv4_model(native_cp_args, hybrid=True)
    model.hybrid_stack_spec = spec
    assert _native_cp_model_unavailable_reason(model, native_cp_args)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mtp_depth", [None, 2])
def test_native_cp_default_te_gpt_is_candidate(native_cp_args, dtype, mtp_depth):
    model = _model(native_cp_args)
    model.transformer.params_dtype = dtype
    model.transformer.bf16 = dtype == torch.bfloat16
    model.transformer.fp16 = dtype == torch.float16
    model.transformer.mtp_num_layers = mtp_depth
    assert model.transformer.use_native_cp_transport is False
    assert _native_cp_model_unavailable_reason(model, native_cp_args) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("builder", "custom.GPTModelBuilder"),
        ("transformer_layer_spec", ModuleSpec(module=torch.nn.Identity)),
        ("transformer_layer_spec", lambda config: None),
        ("restore_modelopt_state", True),
        ("pre_wrap_hooks", [lambda model: model]),
        ("post_wrap_hooks", [lambda model: model]),
    ],
)
def test_native_cp_custom_model_is_not_inferred(native_cp_args, field, value):
    model = _model(native_cp_args)
    setattr(model, field, value)
    assert _native_cp_model_unavailable_reason(model, native_cp_args)


def test_native_cp_unknown_model_and_subclass_are_not_inferred(native_cp_args):
    class CustomGPTModelConfig(GPTModelConfig):
        pass

    model = CustomGPTModelConfig(transformer=_model(native_cp_args).transformer, vocab_size=1024)
    assert _native_cp_model_unavailable_reason(None, native_cp_args)
    assert _native_cp_model_unavailable_reason(model, native_cp_args)


def test_native_cp_custom_transformer_config_is_not_inferred(native_cp_args):
    class CustomTransformerConfig(TransformerConfig):
        pass

    config = core_transformer_config_from_args(native_cp_args, config_class=CustomTransformerConfig)
    model = GPTModelConfig(transformer=config, vocab_size=1024)
    assert _native_cp_model_unavailable_reason(model, native_cp_args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dynamic_context_parallel", False),
        ("transformer_impl", "local"),
        ("is_hybrid_model", True),
        ("experimental_attention_variant", "gated_delta_net"),
        ("multi_latent_attention", True),
        ("use_kitchen_attention", True),
        ("fallback_to_eager_attn", True),
        ("attention_backend", AttnBackend.local),
        ("attention_backend", AttnBackend.unfused),
        ("params_dtype", torch.float32),
        ("fp8", "hybrid"),
        ("fp4", "e2m1"),
        ("fp8_dot_product_attention", True),
        ("qk_clip", True),
        ("log_max_attention_logit", True),
        ("softmax_type", "learnable"),
        ("cuda_graph_impl", "local"),
        ("cp_comm_type", "all_gather"),
        ("cp_comm_type", ["p2p", "all_gather"]),
        ("max_seqlen_per_dp_cp_rank", None),
        ("gtp_weight_remat_size", 2),
        ("expert_gtp_weight_remat_size", 2),
    ],
)
def test_native_cp_unsupported_attention_uses_legacy(native_cp_args, field, value):
    model = _model(native_cp_args)
    # Mutate after construction to isolate the selection gate from config validation.
    setattr(model.transformer, field, value)
    assert _native_cp_model_unavailable_reason(model, native_cp_args)


def test_native_cp_requires_nccl(native_cp_args):
    model = _model(native_cp_args)
    native_cp_args.distributed_backend = "gloo"
    assert _native_cp_model_unavailable_reason(model, native_cp_args)


def test_native_cp_has_no_user_cli_flag():
    parser = add_megatron_arguments(ArgumentParser(allow_abbrev=False))
    assert "--use-native-cp-transport" not in parser._option_string_actions
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--use-native-cp-transport"])
    assert error.value.code == 2


def test_native_cp_checkpoint_state_is_reset_then_internal_state_copies(monkeypatch):
    # A checkpoint's resolved choice must not bypass detection on a new runtime.
    args = _arguments(monkeypatch)
    args.use_native_cp_transport = True
    validate_args(args)
    assert args.use_native_cp_transport is False
    assert core_transformer_config_from_args(args).use_native_cp_transport is False
    # The common args-to-config path must retain the runtime's final internal result.
    args.use_native_cp_transport = True
    assert core_transformer_config_from_args(args).use_native_cp_transport is True


@pytest.mark.parametrize(
    "reasons,expected",
    [([None, None], True), ([None, "remote parent unsupported"], False), (["old TE", None], False)],
)
def test_native_cp_selection_requires_every_rank(monkeypatch, reasons, expected):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: len(reasons))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def gather(output, local_reason):
        assert local_reason == reasons[0]
        output[:] = reasons

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    assert parallel_state._agree_on_native_cp_transport(reasons[0]) is expected
