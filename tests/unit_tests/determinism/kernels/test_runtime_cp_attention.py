# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay packed attention while switching its runtime CP topology.

Comparisons are bit-exact for repeated execution at the *same* CP size, including after
intervening microbatches with other CP sizes. Different reduction topologies are not
required to produce identical bits. The real RoPE, TE attention, and projection kernels
run here; no compute kernel is mocked.
"""

from contextlib import contextmanager

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    bytes_equal,
    seeded,
)


def _zigzag_shard(hidden_states, lengths, cp_size, cp_rank):
    """Slice each packed sequence into its rank's paired front/back CP chunks."""
    parts = []
    for sequence in hidden_states.split(lengths):
        chunks = sequence.chunk(2 * cp_size)
        parts.extend((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]))
    return torch.cat(parts).contiguous()


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_zigzag_shard_preserves_each_packed_sequence(cp_size):
    """CPU check of the test input layout, including unequal sequence lengths."""
    lengths = (16, 24)
    tokens = torch.arange(sum(lengths)).view(-1, 1, 1)
    shards = [_zigzag_shard(tokens, lengths, cp_size, rank) for rank in range(cp_size)]
    assert torch.equal(torch.cat(shards).flatten().sort().values, tokens.flatten())
    for rank, shard in enumerate(shards):
        offset = 0
        for length, local in zip(lengths, shard.split([n // cp_size for n in lengths])):
            chunk = length // (2 * cp_size)
            front = tokens[offset + rank * chunk : offset + (rank + 1) * chunk]
            back = tokens[offset + length - (rank + 1) * chunk : offset + length - rank * chunk]
            assert torch.equal(local, torch.cat((front, back)))
            offset += length


@contextmanager
def _runtime_cp_groups():
    """Create every subgroup in a consistent global order, then release local handles."""
    groups = {}
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    try:
        for size in (1, 2, 4):
            for first in range(0, world_size, size):
                ranks = list(range(first, first + size))
                group = torch.distributed.new_group(ranks)
                if rank in ranks:
                    groups[size] = group
        yield groups
    finally:
        torch.cuda.synchronize()
        for group in reversed(list(groups.values())):
            torch.distributed.destroy_process_group(group)


def _assert_same_replay(reference, actual):
    """Compare outputs and all gradients, including their raw bit patterns."""
    for expected, observed in zip(reference, actual):
        assert expected.keys() == observed.keys()
        for name in expected:
            assert bytes_equal(expected[name], observed[name]), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Transformer Engine")
@pytest.mark.parametrize("build_cp_size", [1, 2])
@pytest.mark.parametrize("fused_rope", [False, True])
def test_packed_attention_runtime_cp_replays(build_cp_size, fused_rope, monkeypatch):
    """Replay real CP1/2/4 forward/backward across topology changes and a failed forward."""
    from megatron.core.extensions.transformer_engine import HAVE_TE

    if not HAVE_TE:
        pytest.skip("requires Transformer Engine")

    from transformer_engine.pytorch.attention.dot_product_attention import (
        dot_product_attention as te_dpa,
    )

    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    if Utils.world_size < 4 or Utils.world_size % 4:
        pytest.skip("runtime CP4 replay requires a world size divisible by four")

    Utils.initialize_model_parallel(context_parallel_size=build_cp_size)
    try:
        model_parallel_cuda_manual_seed(123)
        seeded()
        monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
        monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
        monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        # TE caches backend selection; a preceding test may have chosen another backend.
        monkeypatch.setattr(
            te_dpa,
            "_attention_backends",
            {
                "attention_params": None,
                "use_flash_attention": None,
                "flash_attention_backend": None,
                "use_fused_attention": None,
                "fused_attention_backend": None,
                "use_unfused_attention": None,
                "backend_selection_requires_update": False,
            },
        )
        monkeypatch.setattr(TEDotProductAttention, "cp_stream", None)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=512,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=128,
            context_parallel_size=build_cp_size,
            cp_comm_type="p2p",
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            add_bias_linear=False,
            apply_rope_fusion=fused_rope,
            deterministic_mode=True,
        )
        module = SelfAttention(
            config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).cuda()
        build_group = module.pg_collection.cp
        core = module.core_attention
        assert isinstance(core, TEDotProductAttention)
        original_te_group = core.cp_group
        original_te_ranks = core.cp_global_ranks

        def assert_restored():
            assert module.pg_collection.cp is build_group
            assert core.cp_group is original_te_group
            assert core.cp_global_ranks == original_te_ranks

        lengths = (1536, 2560)  # Both divisible by 2 * CP4; no padding ambiguity.
        hidden = torch.randn(
            sum(lengths), 1, config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        cu_seqlens = torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32)
        rotary = RotaryEmbedding(kv_channels=config.kv_channels, rotary_percent=1.0)(
            max(lengths), packed_seq=True
        )
        references = {}
        with _runtime_cp_groups() as groups:
            # None exercises the legacy build-time path before and after runtime overrides.
            for runtime_size in (None, 1, 2, 4, 2, 1, None):
                group = build_group if runtime_size is None else groups[runtime_size]
                packed = PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    cu_seqlens_q_padded=cu_seqlens,
                    cu_seqlens_kv_padded=cu_seqlens,
                    max_seqlen_q=max(lengths),
                    max_seqlen_kv=max(lengths),
                    local_cp_size=runtime_size,
                    cp_group=None if runtime_size is None else group,
                )
                local = _zigzag_shard(hidden, lengths, group.size(), group.rank())
                inputs = dict(
                    hidden_states=local.detach().requires_grad_(True),
                    attention_mask=None,
                    rotary_pos_emb=rotary,
                    packed_seq_params=packed,
                )
                result = assert_module_replays_bit_exact(
                    module,
                    inputs,
                    replays=3,
                    contention=True,
                    what=f"packed attention build CP{build_cp_size}, runtime CP{runtime_size}",
                )
                assert_restored()
                assert "in.hidden_states" in result[1]
                assert any("linear_qkv" in name for name in result[1])
                assert any("linear_proj" in name for name in result[1])
                for tensors in result:
                    assert all(torch.isfinite(t).all() for t in tensors.values())
                if runtime_size in references:
                    _assert_same_replay(references[runtime_size], result)
                else:
                    references[runtime_size] = result

                if runtime_size == 4:
                    assert isinstance(TEDotProductAttention.cp_stream, torch.cuda.Stream)

                    def fail_after_binding(*args, **kwargs):
                        assert core.cp_group is group
                        raise RuntimeError("synthetic failure after runtime CP binding")

                    with monkeypatch.context() as patch:
                        patch.setattr(core, "_forward", fail_after_binding)
                        with pytest.raises(RuntimeError, match="synthetic failure"):
                            module(**inputs)
                    assert_restored()
                    recovered = assert_module_replays_bit_exact(
                        module, inputs, replays=2, contention=True, what="after CP forward failure"
                    )
                    _assert_same_replay(result, recovered)
                    assert_restored()
    finally:
        Utils.destroy_model_parallel()
