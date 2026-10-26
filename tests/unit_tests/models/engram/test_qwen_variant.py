# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Qwen PLE variant: hash constants, EOS-reset windows, module math, and THD packing.

The reference computations in this file are written independently from the published
equations (splitmix64 steps, nth-prime allocation, per-position window reset) so they can
catch transcription mistakes in the production code. ``QWEN4_REFERENCE_PATH`` optionally
points at the official HF ``modeling_qwen4_exp.py`` for a bit-exact cross-check.
"""

import importlib.util
import math
import os

import pytest
import torch

from megatron.core.models.engram.config import (
    EngramConfig,
    allocate_qwen_table_sizes,
    build_qwen_layer_multipliers,
    is_prime,
)
from megatron.core.models.engram.engram import Engram
from megatron.core.models.engram.hashing import build_ngram_hashes, shift_right_reset_at_eos

from ._test_utils import make_module_config, make_pg_collection

_EOS = 7
_UNIGRAM_VOCAB = 32


def _qwen_config(layer_ids=(1,), base=17):
    return EngramConfig(
        global_vocab_sizes=(base, base),
        layer_ids=layer_ids,
        max_ngram_order=3,
        num_hash_heads=2,
        memory_dim=8,
        kernel_size=4,
        hash_seed=1234,
        boundary_token_id=_EOS,
        variant="qwen",
        unigram_vocab_size=_UNIGRAM_VOCAB,
    )


def test_qwen_config_guards():
    with pytest.raises(ValueError, match="engram_eos_token_id.*must be below"):
        EngramConfig(
            global_vocab_sizes=(17, 17),
            layer_ids=(1,),
            max_ngram_order=3,
            num_hash_heads=2,
            memory_dim=8,
            kernel_size=4,
            hash_seed=0,
            boundary_token_id=_UNIGRAM_VOCAB,  # out-of-vocabulary EOS
            variant="qwen",
            unigram_vocab_size=_UNIGRAM_VOCAB,
        )
    with pytest.raises(ValueError, match="must be equal"):
        EngramConfig(
            global_vocab_sizes=(17, 19),
            layer_ids=(1,),
            max_ngram_order=3,
            num_hash_heads=2,
            memory_dim=8,
            kernel_size=4,
            hash_seed=0,
            boundary_token_id=_EOS,
            variant="qwen",
            unigram_vocab_size=_UNIGRAM_VOCAB,
        )
    with pytest.raises(ValueError, match="engram_unigram_vocab_size"):
        EngramConfig(
            global_vocab_sizes=(17, 17),
            layer_ids=(1,),
            max_ngram_order=3,
            num_hash_heads=2,
            memory_dim=8,
            kernel_size=4,
            hash_seed=0,
            boundary_token_id=_EOS,
            variant="qwen",
        )


def _reference_nth_prime_after(start: int, count: int) -> int:
    """Naive restatement of the official per-head prime allocation."""
    prime = start
    for _ in range(count):
        prime += 1
        while not is_prime(prime):
            prime += 1
    return prime


def _reference_splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def test_qwen_multipliers_and_primes_match_reference_equations():
    seed, order, unigram = 1234, 3, _UNIGRAM_VOCAB
    for ple_layer_index in (0, 1, 3):
        half_bound = max(1, (((1 << 63) - 1) // unigram) // 2)
        expected = tuple(
            2
            * (
                _reference_splitmix64(
                    (seed + 10007 * ple_layer_index + 0x9E3779B97F4A7C15 * (index + 1))
                    & ((1 << 64) - 1)
                )
                % half_bound
            )
            + 1
            for index in range(order)
        )
        assert build_qwen_layer_multipliers(unigram, order, ple_layer_index, seed) == expected

    # nth-prime allocation: sequential primes after base-1, order-major heads, layers chained.
    def naive_next_primes(start, count):
        primes, value = [], start
        while len(primes) < count:
            value += 1
            if is_prime(value):
                primes.append(value)
        return primes

    layer_ids = (1, 5)
    tables = allocate_qwen_table_sizes(17, layer_ids, max_ngram_order=3, num_hash_heads=2)
    flat_expected = naive_next_primes(16, 8)
    assert tables[1] == tuple(flat_expected[:4])
    assert tables[5] == tuple(flat_expected[4:8])
    assert _reference_nth_prime_after(16, 1) == 17 and _reference_nth_prime_after(16, 3) == 23


def test_shift_right_reset_at_eos_matches_naive_semantics():
    torch.manual_seed(0)
    tokens = torch.randint(0, _UNIGRAM_VOCAB, (3, 24), dtype=torch.int64)
    tokens[0, 5] = _EOS
    tokens[0, 6] = _EOS  # adjacent EOS: zero-length segment
    tokens[1, 0] = _EOS  # EOS at row start
    tokens[2, 23] = _EOS  # EOS at row end

    def naive(row, shift):
        out = []
        segment_start = 0
        for position in range(row.numel()):
            if position - segment_start >= shift and position - shift >= 0:
                out.append(int(row[position - shift]))
            else:
                out.append(_EOS)
            if int(row[position]) == _EOS:
                segment_start = position + 1
        return torch.tensor(out, dtype=torch.int64)

    for shift in range(3):
        shifted = shift_right_reset_at_eos(tokens, shift, _EOS)
        for row_index in range(tokens.shape[0]):
            torch.testing.assert_close(shifted[row_index], naive(tokens[row_index], shift))


def test_qwen_hashes_match_naive_reference():
    config = _qwen_config()
    torch.manual_seed(1)
    tokens = torch.randint(0, _UNIGRAM_VOCAB, (2, 16), dtype=torch.int64)
    tokens[0, 4] = _EOS
    multipliers = torch.tensor(config.multipliers(1), dtype=torch.int64)
    table_sizes = torch.tensor(config.table_sizes(1), dtype=torch.int64)

    actual = build_ngram_hashes(
        input_ids=tokens,
        tokenizer_remap=None,
        multipliers=multipliers,
        table_sizes=table_sizes,
        max_ngram_order=3,
        num_hash_heads=2,
        boundary_token_id=_EOS,
        reset_at_boundary=True,
    )

    shifted = [shift_right_reset_at_eos(tokens, shift, _EOS) for shift in range(3)]
    expected_columns = []
    table_index = 0
    for order in (2, 3):
        mixed = shifted[0] * multipliers[0]
        for position in range(1, order):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        for _ in range(2):
            expected_columns.append(torch.remainder(mixed, table_sizes[table_index]))
            table_index += 1
    torch.testing.assert_close(actual, torch.stack(expected_columns, dim=-1))


def _build_qwen_module(num_streams=4):
    torch.manual_seed(99)
    module = Engram(
        config=make_module_config(num_streams=num_streams),
        engram_config=_qwen_config(),
        layer_number=1,
        pg_collection=make_pg_collection(),
    )
    with torch.no_grad():
        module.short_conv.weight.normal_(mean=0.0, std=0.03)
        module.key_norm.weight.normal_(mean=0.0, std=0.02)  # zero-centered gamma perturbation
        module.query_norm.weight.normal_(mean=0.0, std=0.02)
        module.conv_norm.weight.normal_(mean=0.0, std=0.02)
    return module


def test_qwen_module_layout_and_forward_reference(num_streams=4):
    module = _build_qwen_module(num_streams)
    # Official layout: fused bias-free projections and zero-centered group norms.
    assert module.key_projection.bias is None and module.value_projection.bias is None
    assert module.key_norm.zero_centered and module.tokenizer_remap is None
    assert module.key_projection.weight.shape == (num_streams * 8, 16)

    tokens = torch.tensor([[3, 9, _EOS, 4, 2, 6, 1, 5, 8, 2, 11, 4]], dtype=torch.int64)
    sequence = tokens.shape[1]
    hidden = torch.randn(sequence, 1, 8 * num_streams, dtype=torch.float64, requires_grad=True)
    output = module(hidden, tokens)

    # Independent reference from the published equations.
    hashes = build_ngram_hashes(
        tokens,
        None,
        module.hash_multipliers,
        module.table_sizes,
        3,
        2,
        _EOS,
        reset_at_boundary=True,
    )
    embeddings = torch.cat(
        [
            torch.nn.functional.embedding(hashes[..., t], table.weight)
            for t, table in enumerate(module.embedding.tables)
        ],
        dim=-1,
    ).transpose(0, 1)

    def group_norm(norm, tensor):
        grouped = tensor.view(*tensor.shape[:-1], num_streams, 8)
        normed = grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + norm.eps)
        return (normed * (1.0 + norm.weight.view(num_streams, 8))).flatten(-2)

    key = group_norm(module.key_norm, embeddings @ module.key_projection.weight.T)
    query = group_norm(module.query_norm, hidden)
    score = (key.view(sequence, 1, num_streams, 8) * query.view(sequence, 1, num_streams, 8)).sum(
        -1
    ) / math.sqrt(8)
    score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
    value = score.sigmoid().unsqueeze(-1) * (
        embeddings @ module.value_projection.weight.T
    ).unsqueeze(2)
    normed = group_norm(module.conv_norm, value.flatten(-2))
    padded = torch.nn.functional.pad(normed.permute(1, 2, 0), (module.conv_history_length, 0))
    convolved = torch.nn.functional.conv1d(
        padded, module.short_conv.weight, dilation=3, groups=num_streams * 8
    )
    expected = value.flatten(-2) + torch.nn.functional.silu(convolved).permute(2, 0, 1)
    torch.testing.assert_close(output, expected.view_as(output), rtol=1e-10, atol=1e-10)

    output.sum().backward()
    assert hidden.grad is not None
    assert module.key_projection.weight.grad is not None


def test_thd_packed_row_matches_per_document_rows():
    """One packed row with EOS boundaries vs. the same documents in separate rows.

    Hash IDs must match everywhere (windows reset at EOS). Module outputs match exactly at
    positions at least ``conv_history_length`` past each document start; earlier positions
    legitimately differ because the official convolution does not segment at boundaries.
    """
    module = _build_qwen_module(num_streams=1)
    history = module.conv_history_length

    doc_a = torch.randint(0, _UNIGRAM_VOCAB - 1, (1, 20), dtype=torch.int64)
    doc_b = torch.randint(0, _UNIGRAM_VOCAB - 1, (1, 19), dtype=torch.int64)
    doc_a[doc_a == _EOS] = 1
    doc_b[doc_b == _EOS] = 1
    eos = torch.tensor([[_EOS]], dtype=torch.int64)
    packed = torch.cat([doc_a, eos, doc_b], dim=1)  # [1, 40]

    hashes_packed = build_ngram_hashes(
        packed,
        None,
        module.hash_multipliers,
        module.table_sizes,
        3,
        2,
        _EOS,
        reset_at_boundary=True,
    )
    hashes_b = build_ngram_hashes(
        doc_b, None, module.hash_multipliers, module.table_sizes, 3, 2, _EOS, reset_at_boundary=True
    )
    # Document B inside the packed row hashes identically to document B standing alone.
    torch.testing.assert_close(hashes_packed[:, doc_a.shape[1] + 1 :], hashes_b)

    hidden_packed = torch.randn(packed.shape[1], 1, 8, dtype=torch.float64)
    output_packed = module(hidden_packed, packed)
    output_b = module(hidden_packed[doc_a.shape[1] + 1 :], doc_b)
    # Beyond the convolution receptive field the packed and standalone outputs agree.
    torch.testing.assert_close(output_packed[doc_a.shape[1] + 1 + history :], output_b[history:])
    # Within the receptive field they intentionally differ (official conv is unsegmented).
    assert not torch.allclose(output_packed[doc_a.shape[1] + 1], output_b[0])


@pytest.mark.skipif(
    not os.environ.get("QWEN4_REFERENCE_PATH"),
    reason="set QWEN4_REFERENCE_PATH to the official modeling_qwen4_exp.py for the bit-exact check",
)
def test_constants_match_official_reference_file():
    """Bit-exact cross-check against the official implementation's constant generators.

    The official file uses package-relative imports, so extract just the pure hash-constant
    functions (torch/math only) via AST and execute them standalone.
    """
    import ast

    source = open(os.environ["QWEN4_REFERENCE_PATH"], encoding="utf-8").read()
    tree = ast.parse(source)
    wanted = {
        "_MASK64",
        "_SPLITMIX_GAMMA",
        "_SPLITMIX_M1",
        "_SPLITMIX_M2",
        "_PRIME_1",
        "_splitmix64",
        "_build_layer_multipliers",
        "_is_prime",
        "_find_nth_prime_after",
    }
    picked = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets)
        )
    ]
    assert len(picked) == len(wanted), f"reference functions missing: found {len(picked)}"
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=picked, type_ignores=[]), "<qwen4_reference>", "exec"), namespace)

    for ple_layer_index in (0, 1):
        ours = build_qwen_layer_multipliers(_UNIGRAM_VOCAB, 3, ple_layer_index, 1234)
        theirs = namespace["_build_layer_multipliers"](_UNIGRAM_VOCAB, 3, ple_layer_index, 1234)
        assert list(ours) == theirs.tolist()
    for count in (1, 2, 5):
        assert _reference_nth_prime_after(19_999_999, count) == namespace["_find_nth_prime_after"](
            19_999_999, count
        )

    # Extract the official _shift_right_ignore_eos method and compare bit-exactly.
    from types import SimpleNamespace

    shift_method = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_shift_right_ignore_eos":
                    shift_method = item
    assert shift_method is not None
    exec(
        compile(ast.Module(body=[shift_method], type_ignores=[]), "<qwen4_shift>", "exec"),
        namespace,
    )
    official_shift = namespace["_shift_right_ignore_eos"]
    stub = SimpleNamespace(eos_token_id=_EOS)
    torch.manual_seed(3)
    tokens = torch.randint(0, _UNIGRAM_VOCAB, (4, 33), dtype=torch.int64)
    tokens[0, 10] = _EOS
    tokens[1, 0] = _EOS
    tokens[2, 31] = _EOS
    tokens[3, 5] = tokens[3, 6] = _EOS
    for shift in range(3):
        torch.testing.assert_close(
            shift_right_reset_at_eos(tokens, shift, _EOS), official_shift(stub, tokens, shift)
        )
