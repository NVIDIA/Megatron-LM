"""Local-SPMD type checks for Megatron's tensor-parallel modules.

Each test builds one production module on rank 0 of a fake TP2 job, types its
parameters and inputs on the TP axis, and runs it under the strict checker. To
cover a new module or config variant, add a test (or a ``parametrize`` value).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
from spmd_types.checker import typecheck as spmd_typecheck
from torch.testing._internal.distributed.fake_pg import FakeStore

from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_gated_delta_net_module_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.T5.t5_spec import decoder_model_with_local_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.spmd.annotations import annotate_model
from megatron.core.tensor_parallel import mappings
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock
from megatron.core.transformer.spec_utils import ModuleSpec, build_module, get_submodules
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer


@contextmanager
def _fake_tp(ranks: list[int] | None) -> Iterator[torch.distributed.ProcessGroup]:
    """Impersonate rank 0 of a TP2 job and yield its TP group.

    With ``ranks``, the module gets its own TP group over those ranks instead of
    the one in ``parallel_state``, as when a caller passes an explicit
    ``pg_collection``. Code that still reaches for the global group then uses a
    group outside the typed mesh.
    """
    world_size = 2 if ranks is None else 4
    torch.distributed.init_process_group(
        backend="fake", store=FakeStore(), rank=0, world_size=world_size
    )
    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=2, create_gloo_process_groups=False
        )
        model_parallel_cuda_manual_seed(123)
        if ranks is None:
            yield parallel_state.get_tensor_model_parallel_group()
        else:
            yield torch.distributed.new_group(ranks=ranks, backend="fake")
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.fixture(autouse=True)
def _default_attention_backend(monkeypatch):
    """Undo the unit-test conftest's forced unfused attention; let TE pick as in training."""
    for variable in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def tp_group() -> Iterator[torch.distributed.ProcessGroup]:
    with _fake_tp(ranks=None) as group:
        yield group


@pytest.fixture
def explicit_tp_group() -> Iterator[torch.distributed.ProcessGroup]:
    """A TP group that differs from ``parallel_state``'s, for explicit ``pg_collection`` tests."""
    with _fake_tp(ranks=[0, 2]) as group:
        yield group


@contextmanager
def typecheck(tp_group: torch.distributed.ProcessGroup) -> Iterator[None]:
    """Strictly type-check the enclosed code on a mesh with only ``tp_group``.

    Create input tensors before entering, so the checker does not type integer
    and boolean factory outputs as replicated.
    """
    from megatron.core.spmd import spmd_rules  # noqa: F401

    with (
        spmd_typecheck(strict_mode="strict", local=True),
        spmd.set_current_mesh({"TP": spmd.MeshAxis.of(tp_group)}),
        torch.compiler.set_stance("force_eager"),
    ):
        yield


def typed(tensor: torch.Tensor, tp_type: spmd.SpmdType) -> torch.Tensor:
    spmd.assert_type(tensor, {"TP": tp_type})
    return tensor


def activations(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.bfloat16, device="cuda", requires_grad=True)


def tp_type(tensor: torch.Tensor, tp_group: torch.distributed.ProcessGroup) -> spmd.SpmdType:
    return spmd.get_axis_local_type(tensor, tp_group)


def tiny_config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        ffn_hidden_size=64,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        masked_softmax_fusion=False,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def tp_only(tp_group: torch.distributed.ProcessGroup) -> ProcessGroupCollection:
    groups = ProcessGroupCollection.use_mpu_process_groups()
    groups.tp = tp_group
    return groups


def tiny_gpt(tp_group: torch.distributed.ProcessGroup) -> GPTModel:
    config = tiny_config()
    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_decoder_block_spec(config, use_transformer_engine=True),
        vocab_size=128,
        max_sequence_length=8,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        scatter_embedding_sequence_parallel=True,
        pg_collection=tp_only(tp_group),
    ).cuda()


def test_sequence_parallel_round_trip(tp_group):
    x = activations(8, 2, 4)
    with typecheck(tp_group):
        local = mappings.scatter_to_sequence_parallel_region(typed(x, spmd.I), group=tp_group)
        out = mappings.gather_from_sequence_parallel_region(
            local, group=tp_group, tensor_parallel_output_grad=False
        )
        assert tp_type(out, tp_group) is spmd.I
        out.sum().backward()


@pytest.mark.parametrize("sequence_parallel", [True, False])
def test_te_transformer_block(tp_group, sequence_parallel):
    block = TransformerBlock(
        tiny_config(sequence_parallel=sequence_parallel),
        get_gpt_layer_with_transformer_engine_spec(),
    ).cuda()
    x_type = spmd.S(0) if sequence_parallel else spmd.I
    x = activations(4, 2, 32)
    with typecheck(tp_group):
        annotate_model(block)
        out = block(hidden_states=typed(x, x_type), attention_mask=None)
        if sequence_parallel:
            out = mappings.gather_from_sequence_parallel_region(
                out, group=tp_group, tensor_parallel_output_grad=False
            )
        assert tp_type(out, tp_group) is spmd.I
        out.sum().backward()


def test_vocab_parallel_cross_entropy(tp_group):
    logits = activations(4, 2, 8)
    target = torch.zeros(4, 2, dtype=torch.long, device="cuda")
    with typecheck(tp_group):
        loss = vocab_parallel_cross_entropy(typed(logits, spmd.S(-1)), typed(target, spmd.I))
        assert tp_type(loss, tp_group) is spmd.I
        loss.sum().backward()


def test_te_gpt_forward_backward(tp_group):
    model = tiny_gpt(tp_group)
    input_ids = torch.zeros(2, 8, dtype=torch.long, device="cuda")
    position_ids = torch.zeros(2, 8, dtype=torch.long, device="cuda")
    attention_mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool, device="cuda")
    labels = torch.zeros(2, 8, dtype=torch.long, device="cuda")
    with typecheck(tp_group):
        annotate_model(model)
        loss = model(
            input_ids=typed(input_ids, spmd.R),
            position_ids=typed(position_ids, spmd.I),
            attention_mask=typed(attention_mask, spmd.I),
            labels=typed(labels, spmd.I),
        )
        assert tp_type(loss, tp_group) is spmd.I
        loss.sum().backward()


# Megatron-LM#7452.8: Learnable attention sinks are excluded from the clipping norm
@pytest.mark.xfail(strict=True, reason="7452.8: split softmax_offset lacks tensor_model_parallel")
def test_attention_sink_forward_backward(tp_group):
    attention = DotProductAttention(
        tiny_config(softmax_type="learnable"),
        1,
        AttnMaskType.causal,
        "self",
        pg_collection=tp_only(tp_group),
    ).cuda()
    # [sequence, batch, local heads, head dim]; heads are split across TP.
    query, key, value = (activations(4, 2, 2, 8) for _ in range(3))
    # Scores are computed into a shared uninitialized buffer (baddbmm with beta=0);
    # it is typed below like the per-head scores it will hold.
    scratch = parallel_state.get_global_memory_buffer()
    scratch.get_tensor((2 * 2, 4, 4), torch.bfloat16, "mpu")
    with typecheck(tp_group):
        annotate_model(attention)
        spmd.assert_type(scratch.buffer[("mpu", torch.bfloat16)], {"TP": spmd.V})
        out = attention(
            typed(query, spmd.S(2)), typed(key, spmd.S(2)), typed(value, spmd.S(2)), None
        )
        mappings.gather_from_tensor_model_parallel_region(out, group=tp_group).sum().backward()


def l2norm(x, dim=-1, eps=1e-6):
    return F.normalize(x, dim=dim, eps=eps)


# Megatron-LM#7452.11: GDN shared output-normalization weight misses the TP gradient sum
@pytest.mark.xfail(
    strict=True, reason="7452.11: out_norm weight has a partial gradient but no sequence_parallel"
)
def test_gated_delta_net_forward_backward(tp_group):
    config = tiny_config(
        sequence_parallel=False,
        normalization="RMSNorm",
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        activation_func=F.silu,
        # Use Megatron's torch reference kernels instead of flash-linear-attention.
        deterministic_mode=True,
    )
    # Deterministic mode still insists on flash-linear-attention and uses its l2norm.
    with (
        patch("megatron.core.ssm.gated_delta_net.common.HAVE_FLA", True),
        patch("megatron.core.ssm.gated_delta_net.common.l2norm", l2norm),
        patch("megatron.core.ssm.gated_delta_net.gdn.l2norm", l2norm),
    ):
        gdn = build_module(
            get_gated_delta_net_module_spec(config),
            config=config,
            layer_number=1,
            pg_collection=tp_only(tp_group),
        ).cuda()
        x = activations(8, 2, 32)
        with typecheck(tp_group):
            annotate_model(gdn)
            out, _ = gdn(typed(x, spmd.I), attention_mask=None)
            out.sum().backward()


# Megatron-LM#7452.4: MTP full recomputation uses the global TP group
@pytest.mark.xfail(
    strict=True, reason="7452.4: MTP recompute passes parallel_state's TP group, not its own"
)
def test_mtp_recompute_uses_explicit_tp_group(explicit_tp_group):
    config = tiny_config(
        mtp_num_layers=1,
        sequence_parallel=False,
        perform_initialization=False,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        distribute_saved_activations=True,
        fp8="hybrid",
    )
    spec = get_gpt_mtp_block_spec(config, get_gpt_layer_local_spec(), use_transformer_engine=False)
    block = MultiTokenPredictionBlock(config, spec, pg_collection=tp_only(explicit_tp_group))
    layer = block.layers[0].cuda()
    # Recompute a stand-in for the layer; only the checkpoint's TP group matters here.
    layer._proj_and_transformer_layer = lambda hidden_states, **kwargs: 2 * hidden_states
    hidden, decoder_input = activations(8, 2, 32), activations(8, 2, 32)
    with typecheck(explicit_tp_group):
        annotate_model(layer)
        out = layer._checkpointed_forward(typed(hidden, spmd.I), typed(decoder_input, spmd.I))
        out.sum().backward()


# Megatron-LM#7452.5: GPT scatters the MoE padding mask over the wrong TP group
@pytest.mark.xfail(
    strict=True, reason="7452.5: padding mask is scattered on parallel_state's TP group"
)
def test_gpt_padding_mask_uses_explicit_tp_group(explicit_tp_group):
    model = tiny_gpt(explicit_tp_group)
    input_ids = torch.zeros(2, 8, dtype=torch.long, device="cuda")
    position_ids = torch.zeros(2, 8, dtype=torch.long, device="cuda")
    padding_mask = torch.zeros(2, 8, dtype=torch.bool, device="cuda")
    with typecheck(explicit_tp_group):
        annotate_model(model)
        model._preprocess(
            input_ids=typed(input_ids, spmd.R),
            position_ids=typed(position_ids, spmd.I),
            padding_mask=typed(padding_mask, spmd.I),
        )


# Megatron-LM#7452.6: Cross-attention Q/KV projections omit the explicit TP group
@pytest.mark.xfail(
    strict=True, reason="7452.6: cross-attention key/value linear uses parallel_state's TP group"
)
def test_cross_attention_uses_explicit_tp_group(explicit_tp_group):
    attention = build_module(
        decoder_model_with_local_spec().submodules.cross_attention,
        config=tiny_config(perform_initialization=False),
        layer_number=1,
        pg_collection=tp_only(explicit_tp_group),
    ).cuda()
    hidden_states = activations(4, 2, 32)
    key_value_states = activations(4, 2, 32)
    with typecheck(explicit_tp_group):
        annotate_model(attention)
        attention.get_query_key_value_tensors(
            typed(hidden_states, spmd.S(0)), typed(key_value_states, spmd.S(0))
        )


# Megatron-LM#7452.7: LLaVA scatters combined embeddings over the wrong TP group
@pytest.mark.xfail(
    strict=True, reason="7452.7: token-parallel scatter uses parallel_state's TP group"
)
def test_llava_sequence_scatter_uses_explicit_tp_group(explicit_tp_group):
    language_config = tiny_config()
    language_config.language_model_type = "dummy"
    vision_config = tiny_config(hidden_size=16, num_attention_heads=2, sequence_parallel=False)
    vision_config.vision_model_type = "clip"
    projection_config = tiny_config(num_attention_heads=2, ffn_hidden_size=32)
    layer_submodules = get_gpt_layer_with_transformer_engine_submodules()
    model = LLaVAModel(
        language_transformer_config=language_config,
        language_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=layer_submodules
        ),
        language_vocab_size=128,
        language_max_sequence_length=8,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=ModuleSpec(
            module=TransformerLayer, submodules=deepcopy(layer_submodules)
        ),
        drop_vision_class_token=False,
        vision_projection_config=projection_config,
        vision_projection_layer_spec=deepcopy(get_submodules(layer_submodules.mlp)),
        img_h=28,
        img_w=28,
        patch_dim=14,
        pg_collection=tp_only(explicit_tp_group),
    ).cuda()
    embeddings = activations(8, 2, 32)
    with typecheck(explicit_tp_group):
        annotate_model(model)
        local = model._process_embedding_token_parallel(
            typed(embeddings, spmd.I), None, None, None
        )[0]
        mappings.gather_from_sequence_parallel_region(
            local, group=explicit_tp_group, tensor_parallel_output_grad=False
        ).sum().backward()
