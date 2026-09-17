# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import sys
from argparse import ArgumentParser

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import (
    add_megatron_arguments,
    core_transformer_config_from_args,
    parse_args,
    validate_args,
)
from megatron.training.initialize import _native_cp_model_unavailable_reason
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
        ("fp8_dot_product_attention", True),
        ("qk_clip", True),
        ("log_max_attention_logit", True),
        ("softmax_type", "learnable"),
        ("cuda_graph_impl", "local"),
        ("cp_comm_type", "all_gather"),
        ("cp_comm_type", ["p2p", "all_gather"]),
        ("max_seqlen_per_dp_cp_rank", None),
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
