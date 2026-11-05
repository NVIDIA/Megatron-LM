# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""apply_rotary_pos_emb uses the context-parallel group its caller passes.

Only packed (THD) sequences read the group. When the caller omits it for them, RoPE warns and
falls back to the global context-parallel group, as before.
"""

import warnings

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.models.common.embeddings import rope_utils
from megatron.core.process_groups_config import ProcessGroupFallbackWarning
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, forbid_global_process_groups

VERSIONS = "deprecated since Megatron Core 0.21 and will be removed in 0.23"

# Two packed sequences of 16 tokens each. With context parallelism of size 2, every rank holds
# 8 tokens of each sequence.
CU_SEQLENS = (0, 16, 32)
LOCAL_TOKENS = 16
HEADS, DIM = 2, 8


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _config():
    return TransformerConfig(
        num_layers=1, hidden_size=HEADS * DIM, num_attention_heads=HEADS, apply_rope_fusion=False
    )


def _inputs():
    generator = torch.Generator(device="cuda").manual_seed(1234)
    t = torch.randn(LOCAL_TOKENS, HEADS, DIM, device="cuda", generator=generator)
    freqs = torch.randn(CU_SEQLENS[-1], 1, 1, DIM, device="cuda", generator=generator)
    cu_seqlens = torch.tensor(CU_SEQLENS, dtype=torch.int32, device="cuda")
    return t, freqs, cu_seqlens


def _pairs_cp_group():
    """A context-parallel group of adjacent rank pairs, which the global grid does not have."""
    world_size = torch.distributed.get_world_size()
    group, _ = torch.distributed.new_subgroups_by_enumeration(
        [[rank, rank + 1] for rank in range(0, world_size, 2)]
    )
    return group


@pytest.mark.skipif(Utils.world_size < 2 or Utils.world_size % 2, reason="needs an even world")
class TestApplyRotaryPosEmbContextParallelGroup:
    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_packed_sequences_use_the_given_group(self):
        # The global grid has no context parallelism; the given group has two ranks.
        Utils.initialize_model_parallel()
        cp_group = _pairs_cp_group()
        t, freqs, cu_seqlens = _inputs()

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            output = rope_utils.apply_rotary_pos_emb(
                t, freqs, _config(), cu_seqlens=cu_seqlens, cp_group=cp_group
            )

        expected = rope_utils._apply_rotary_pos_emb_thd(t, cu_seqlens, freqs, cp_group=cp_group)
        global_cp = rope_utils._apply_rotary_pos_emb_thd(
            t, cu_seqlens, freqs, cp_group=parallel_state.get_context_parallel_group()
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert not torch.equal(output, global_cp)
        torch.distributed.destroy_process_group(cp_group)

    def test_unpacked_sequences_read_no_group(self):
        Utils.initialize_model_parallel(context_parallel_size=2)
        t = torch.randn(LOCAL_TOKENS, 1, HEADS, DIM, device="cuda")
        freqs = torch.randn(LOCAL_TOKENS, 1, 1, DIM, device="cuda")

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            output = rope_utils.apply_rotary_pos_emb(t, freqs, _config())

        torch.testing.assert_close(
            output, rope_utils._apply_rotary_pos_emb_bshd(t, freqs), rtol=0, atol=0
        )

    def test_packed_sequences_without_group_warn_and_use_the_global_group(self):
        Utils.initialize_model_parallel(context_parallel_size=2)
        t, freqs, cu_seqlens = _inputs()
        expected = rope_utils.apply_rotary_pos_emb(
            t,
            freqs,
            _config(),
            cu_seqlens=cu_seqlens,
            cp_group=parallel_state.get_context_parallel_group(),
        )

        with pytest.warns(ProcessGroupFallbackWarning) as record:
            output = rope_utils.apply_rotary_pos_emb(t, freqs, _config(), cu_seqlens=cu_seqlens)

        messages = [
            str(w.message)
            for w in record
            if issubclass(w.category, ProcessGroupFallbackWarning)
            and "apply_rotary_pos_emb was called without `cp_group`" in str(w.message)
        ]
        assert len(messages) == 1 and VERSIONS in messages[0]
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_packed_sequences_without_any_group_raise(self):
        Utils.initialize_distributed()
        if parallel_state.get_context_parallel_group(check_initialized=False) is not None:
            pytest.skip("an earlier test left the global grid initialized")
        t, freqs, cu_seqlens = _inputs()

        with pytest.warns(ProcessGroupFallbackWarning), pytest.raises(ValueError, match="cp_group"):
            rope_utils.apply_rotary_pos_emb(t, freqs, _config(), cu_seqlens=cu_seqlens)
