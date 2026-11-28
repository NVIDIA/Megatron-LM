# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepSeek-V4.1 variant: the gated fusion without the short convolution.

``_v41_reference`` restates the forward of the DeepSeek-V4.1 Engram module of
NVIDIA/Megatron-LM#7224 by Deyu Fu and Hongxiao Bai, megatron/core/models/deepseek_v41/engram.py
at commit 1bb21c69c. It is kept apart from the production code so that it can catch
transcription mistakes in it.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.engram.config import EngramConfig
from megatron.core.models.engram.engram import Engram
from megatron.core.models.engram.hashing import build_ngram_hashes
from megatron.core.models.engram.variants import (
    DEEPSEEK_V41_VARIANT_NAME,
    DEEPSEEK_VARIANT_NAME,
    ENGRAM_VARIANTS,
    GATED_CONV_FUSION,
    GATED_FUSION,
)
from megatron.core.packed_seq_params import PackedSeqParams

from ._test_utils import make_module_config, make_pg_collection, write_tokenizer_map

_HIDDEN_SIZE = 8


def _engram_config(tmp_path, variant=DEEPSEEK_V41_VARIANT_NAME, **overrides):
    artifact = tmp_path / "map.json"
    if not artifact.exists():
        write_tokenizer_map(artifact, vocab_size=32, layer_ids=(1,))
    arguments = dict(
        global_vocab_sizes=(17, 19),
        layer_ids=(1,),
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=4,
        hash_seed=0,
        boundary_token_id=0,
        tokenizer_map_path=str(artifact),
        variant=variant,
    )
    arguments.update(overrides)
    return EngramConfig(**arguments)


def _module(tmp_path, dtype=torch.float32, num_streams=1, variant=DEEPSEEK_V41_VARIANT_NAME):
    torch.manual_seed(123)
    module = Engram(
        config=make_module_config(num_streams=num_streams, dtype=dtype),
        engram_config=_engram_config(tmp_path, variant=variant),
        layer_number=1,
        pg_collection=make_pg_collection(),
    )
    if module.engram_config.variant_spec.fusion == GATED_FUSION:
        with torch.no_grad():
            # Away from their initial ones, so the gate weights matter.
            module.q_weight.normal_(mean=1.0, std=0.2)
            module.k_weight.normal_(mean=1.0, std=0.2)
    return module


def _v41_reference(module, hidden_states, input_ids, token_mask=None, cu_seqlens=None):
    """Return the hidden states and the residual of the DeepSeek-V4.1 forward.

    ``token_mask`` is ``[B, S]`` and True for valid tokens, as in the DeepSeek-V4.1 module.
    """
    hashes = build_ngram_hashes(
        input_ids,
        module.tokenizer_remap,
        module.hash_multipliers,
        module.table_sizes,
        module.engram_config.max_ngram_order,
        module.engram_config.num_hash_heads,
        module.engram_config.hash_boundary_token_id,
        cu_seqlens=cu_seqlens,
    )
    embeddings = torch.cat(
        [
            F.embedding(hashes[..., table_id], table.weight)
            for table_id, table in enumerate(module.embedding.tables)
        ],
        dim=-1,
    )
    n, hidden_size = module.num_streams, module.hidden_size
    kv = F.linear(embeddings, module.wkv.weight).transpose(0, 1)
    key, value = kv.split([n * hidden_size, hidden_size], dim=-1)
    key = key.float().unflatten(-1, (n, hidden_size))
    h = hidden_states.float().unflatten(-1, (n, hidden_size))
    eps = module.config.layernorm_epsilon
    rstd = (h.square().mean(-1) + eps).rsqrt() * (key.square().mean(-1) + eps).rsqrt()
    gamma = (module.q_weight.float() * module.k_weight.float()).view(n, hidden_size)
    dot = (h * gamma * key).sum(-1)
    dot = dot * rstd * hidden_size**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.transpose(0, 1).unsqueeze(-1), 0)
    residual = gate.unsqueeze(-1) * value.float().unsqueeze(-2)
    dtype = hidden_states.dtype
    return (h + residual).flatten(-2).to(dtype), residual.flatten(-2).to(dtype)


def _inputs(dtype, num_streams, sequence_length=6, batch_size=2):
    generator = torch.Generator().manual_seed(7)
    input_ids = torch.randint(0, 32, (batch_size, sequence_length), generator=generator)
    hidden = torch.randn(
        sequence_length, batch_size, _HIDDEN_SIZE * num_streams, generator=generator
    ).to(dtype)
    return hidden, input_ids


def test_variant_descriptors_name_their_fusion():
    assert ENGRAM_VARIANTS[DEEPSEEK_V41_VARIANT_NAME].fusion == GATED_FUSION
    assert not ENGRAM_VARIANTS[DEEPSEEK_V41_VARIANT_NAME].has_short_convolution
    for name in (DEEPSEEK_VARIANT_NAME, "qwen"):
        assert ENGRAM_VARIANTS[name].fusion == GATED_CONV_FUSION
        assert ENGRAM_VARIANTS[name].has_short_convolution


@pytest.mark.parametrize("num_streams", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fusion_matches_the_deepseek_v41_forward_bit_for_bit(tmp_path, dtype, num_streams):
    module = _module(tmp_path, dtype=dtype, num_streams=num_streams)
    hidden, input_ids = _inputs(dtype, num_streams)
    # A zero hidden state gives a score of exactly zero, whose gate is sigmoid(1e-3), not 0.5.
    hidden[2] = 0

    expected_hidden, expected_residual = _v41_reference(module, hidden, input_ids)
    added = module(hidden, input_ids, add_to_hidden_states=True)
    residual = module(hidden, input_ids)

    assert added.dtype == residual.dtype == dtype
    assert added.shape == residual.shape == hidden.shape
    assert torch.equal(added, expected_hidden)
    assert torch.equal(residual, expected_residual)
    # The gate is open somewhere, so the comparison is not one of zeros.
    assert torch.count_nonzero(residual) > 0


def test_half_precision_hidden_states_are_added_in_fp32(tmp_path):
    """Adding in FP32 and rounding once differs from rounding the residual first."""
    module = _module(tmp_path, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(3)
    input_ids = torch.randint(0, 32, (4, 64), generator=generator)
    hidden = torch.randn(64, 4, _HIDDEN_SIZE, generator=generator).to(torch.bfloat16)

    added = module(hidden, input_ids, add_to_hidden_states=True)
    rounded_first = hidden + module(hidden, input_ids)
    assert torch.equal(added, _v41_reference(module, hidden, input_ids)[0])
    assert not torch.equal(added, rounded_first)
    torch.testing.assert_close(added.float(), rounded_first.float(), rtol=0.0, atol=2e-2)


@pytest.mark.parametrize("num_streams", [1, 4])
def test_gradients_match_the_deepseek_v41_forward(tmp_path, num_streams):
    module = _module(tmp_path, num_streams=num_streams)
    hidden, input_ids = _inputs(torch.float32, num_streams)
    hidden[2] = 0
    hidden.requires_grad_(True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)

    actual = module(hidden, input_ids, add_to_hidden_states=True)
    expected, _ = _v41_reference(module, reference_hidden, input_ids)
    projection = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * projection).sum(), [hidden, *module.parameters()])
    expected_grads = torch.autograd.grad(
        (expected * projection).sum(), [reference_hidden, *module.parameters()]
    )
    names = ["hidden_states", *(name for name, _ in module.named_parameters())]
    for name, actual_grad, expected_grad in zip(names, actual_grads, expected_grads):
        assert torch.count_nonzero(expected_grad) > 0, name
        assert torch.isfinite(actual_grad).all(), name
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-7, msg=name)


def test_double_precision_stays_in_double_precision(tmp_path):
    """Only half precision is widened, so FP64 modules keep FP64 gate arithmetic."""
    module = _module(tmp_path, dtype=torch.float64)
    hidden, input_ids = _inputs(torch.float64, 1)
    hidden.requires_grad_(True)

    def residual(hidden_states):
        return module(hidden_states, input_ids)

    assert residual(hidden).dtype == torch.float64
    assert torch.autograd.gradcheck(residual, (hidden,), atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gate_stays_closed_after_the_last_document_boundary(tmp_path, dtype):
    module = _module(tmp_path, dtype=dtype)
    # Two documents of 3 and 4 tokens in a row of 10: positions 7 to 9 belong to no document.
    hidden, input_ids = _inputs(dtype, 1, sequence_length=10, batch_size=1)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu_seqlens)
    token_mask = (torch.arange(10) < 7).view(1, 10)

    expected_hidden, expected_residual = _v41_reference(
        module, hidden, input_ids, token_mask=token_mask, cu_seqlens=cu_seqlens
    )
    added = module(hidden, input_ids, packed_seq_params, add_to_hidden_states=True)
    residual = module(hidden, input_ids, packed_seq_params)

    assert torch.equal(added, expected_hidden)
    assert torch.equal(residual, expected_residual)
    assert torch.equal(added[7:], hidden[7:])
    assert torch.count_nonzero(residual[7:]) == 0
    assert torch.count_nonzero(residual[:7]) > 0
    # An unpacked row has no boundary, and a last document that ends at the end of the row
    # (the --sft layout, whose padding belongs to that document) leaves no position after it:
    # every position keeps its gate.
    assert torch.count_nonzero(module(hidden, input_ids)[7:]) > 0
    whole_row = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=torch.tensor([0, 3, 10], dtype=torch.int32)
    )
    assert torch.count_nonzero(module(hidden, input_ids, whole_row)[7:]) > 0


def test_parameters_are_the_tables_one_projection_and_two_gate_weights(tmp_path):
    module = _module(tmp_path, num_streams=4)
    tables = {f"embedding.tables.{index}.weight" for index in range(4)}
    assert {name for name, _ in module.named_parameters()} == tables | {
        "wkv.weight",
        "q_weight",
        "k_weight",
    }
    # One key per residual stream and one shared value, from the memory of both n-gram orders.
    assert module.wkv.weight.shape == (5 * _HIDDEN_SIZE, 16)
    assert module.q_weight.shape == module.k_weight.shape == (4 * _HIDDEN_SIZE,)
    assert module.engram_config.conv_history_length == 0


@pytest.mark.parametrize("perform_initialization", [True, False])
def test_initialization_draws_the_projection_and_opens_the_gate_weights(
    tmp_path, perform_initialization
):
    config = make_module_config(num_streams=4, dtype=torch.float32)
    config.perform_initialization = perform_initialization
    config.init_method = lambda tensor: torch.nn.init.constant_(tensor, 0.25)
    fresh = Engram(
        config=config,
        engram_config=_engram_config(tmp_path),
        layer_number=1,
        pg_collection=make_pg_collection(),
    )
    assert torch.equal(fresh.q_weight, torch.ones_like(fresh.q_weight))
    assert torch.equal(fresh.k_weight, torch.ones_like(fresh.k_weight))
    # The model's init_method draws wkv, and only when the model initializes its parameters.
    assert torch.all(fresh.wkv.weight == 0.25).item() == perform_initialization


def test_sequence_parallel_marks_the_dense_parameters(tmp_path):
    module = Engram(
        config=make_module_config(sequence_parallel=True, dtype=torch.float32),
        engram_config=_engram_config(tmp_path),
        layer_number=1,
        pg_collection=make_pg_collection(),
    )
    for name in ("wkv.weight", "q_weight", "k_weight"):
        assert module.get_parameter(name).sequence_parallel, name


def test_hash_constants_and_tables_equal_the_deepseek_variant(tmp_path):
    v41 = _module(tmp_path)
    deepseek = _module(tmp_path, variant=DEEPSEEK_VARIANT_NAME)
    assert v41.engram_config.table_sizes(1) == deepseek.engram_config.table_sizes(1)
    assert v41.engram_config.multipliers(1) == deepseek.engram_config.multipliers(1)
    assert torch.equal(v41.tokenizer_remap, deepseek.tokenizer_remap)
    assert v41.engram_config.hash_boundary_token_id == deepseek.engram_config.hash_boundary_token_id
    # Same seed, same tables: the fusion is the only difference between the two variants.
    for v41_table, deepseek_table in zip(v41.embedding.tables, deepseek.embedding.tables):
        assert torch.equal(v41_table.weight, deepseek_table.weight)


def _startup_transformer_config(**overrides):
    config = dict(
        num_layers=2,
        hidden_size=_HIDDEN_SIZE,
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def test_kernel_size_is_checked_only_with_the_convolution(tmp_path):
    assert _engram_config(tmp_path, kernel_size=0).conv_history_length == 0
    with pytest.raises(ValueError, match="engram_kernel_size must be positive"):
        _engram_config(tmp_path, variant=DEEPSEEK_VARIANT_NAME, kernel_size=0)


def test_parallel_layouts_need_no_room_for_a_convolution_history(tmp_path):
    """Without the convolution, short context-parallel chunks and sequence-parallel slices pass."""
    v41 = _engram_config(tmp_path)
    deepseek = _engram_config(tmp_path, variant=DEEPSEEK_VARIANT_NAME)
    assert deepseek.conv_history_length == 9

    context_parallel = _startup_transformer_config(context_parallel_size=2)
    with pytest.raises(ValueError, match="each zigzag chunk to own at least 9 positions"):
        deepseek._validate_parallelism(context_parallel, sequence_length=16)
    v41._validate_parallelism(context_parallel, sequence_length=16)

    sequence_parallel = _startup_transformer_config(
        tensor_model_parallel_size=4, expert_tensor_parallel_size=4, sequence_parallel=True
    )
    with pytest.raises(ValueError, match="each tensor-parallel rank to own at least 9 positions"):
        deepseek._validate_parallelism(sequence_parallel, sequence_length=16)
    v41._validate_parallelism(sequence_parallel, sequence_length=16)
