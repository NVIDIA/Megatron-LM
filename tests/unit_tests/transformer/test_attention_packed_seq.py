# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils


def make_test_packed_seq_params(sequence_length):
    cu_seqlens = torch.IntTensor([0, 6, 19, 22, sequence_length]).cuda()
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = seqlens.max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )
    return packed_seq_params


def make_test_packed_padded_seq_params(sequence_length):
    cu_seqlens = torch.IntTensor([0, 18, 44, 52, 96, 118]).cuda()
    cu_seqlens_padded = torch.IntTensor([0, 20, 48, 56, 100, sequence_length]).cuda()
    seqlens = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    max_seqlen = seqlens.max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )
    return packed_seq_params


class TestParallelAttentionWithPackedSequence:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        # use BF16 and a large enough hidden size to enable FlashAttention for thd format.
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
        )
        self.parallel_attention = SelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):

        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 1

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        packed_seq_params = make_test_packed_seq_params(sequence_length)
        output, bias = self.parallel_attention(
            hidden_states, attention_mask, packed_seq_params=packed_seq_params
        )

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.skipif(not is_te_min_version("1.4.0"), reason="Fused RoPE requires TE >= 1.4.0")
    def test_fused_rope_gpu_forward(self):
        self.parallel_attention.config.apply_rope_fusion = True
        config = self.parallel_attention.config
        sequence_length = 32
        micro_batch_size = 1

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None
        rotary_pos_emb = torch.ones(
            sequence_length, 1, 1, self.parallel_attention.config.kv_channels
        ).cuda()

        packed_seq_params = make_test_packed_seq_params(sequence_length)
        output, bias = self.parallel_attention(
            hidden_states, attention_mask, packed_seq_params=packed_seq_params
        )

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size
        self.parallel_attention.config.apply_rope_fusion = False

    def test_checkpointed_gpu_forward(self):
        transformer_config = self.transformer_config
        transformer_config.recompute_granularity = 'selective'
        checkpointed_parallel_attention = SelfAttention(
            transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        config = checkpointed_parallel_attention.config

        sequence_length = 32
        micro_batch_size = 1

        checkpointed_parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, checkpointed_parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        packed_seq_params = make_test_packed_seq_params(sequence_length)
        output, bias = checkpointed_parallel_attention(
            hidden_states, attention_mask, packed_seq_params=packed_seq_params
        )

        assert config.recompute_granularity == 'selective'
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size


# Note: this test requires TE >= 1.8 as well as cuDNN FusedAttention to run
class TestParallelAttentionWithPackedPaddedSequence(TestParallelAttentionWithPackedSequence):

    def test_gpu_forward(self):

        config = self.parallel_attention.config
        sequence_length = 128
        micro_batch_size = 1

        self.parallel_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        attention_mask = None

        packed_seq_params = make_test_packed_padded_seq_params(sequence_length)
        output, bias = self.parallel_attention(
            hidden_states, attention_mask, packed_seq_params=packed_seq_params
        )

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size


class TestAttentionDynamicContextParallel:
    """Regression tests for runtime (hybrid/dynamic) CP.

    A model built with static context_parallel_size == 1 can be handed a
    per-microbatch CP group at runtime via PackedSeqParams.cp_group. Three
    contracts must hold: RoPE position math must use that runtime group (not
    the static one), and TEDotProductAttention must lazily create its
    auxiliary CP stream (the constructor only allocates it for static CP > 1),
    and local_cp_size == 1 must not carry a cp_group because that convention
    means CP is disabled for the sub-sample.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
        )
        self.parallel_attention = SelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _runtime_cp_group(self):
        # A 1-rank group standing in for the group the hybrid-CP scheduler
        # binds per microbatch; identity is what the assertions check.
        return torch.distributed.new_group(ranks=[torch.distributed.get_rank()])

    def test_rope_uses_runtime_cp_group(self, monkeypatch):
        import megatron.core.transformer.attention as attention_module
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

        captured = []
        real_apply = attention_module.apply_rotary_pos_emb

        def spying_apply(t, freqs, **kwargs):
            captured.append(kwargs.get("cp_group"))
            return real_apply(t, freqs, **kwargs)

        monkeypatch.setattr(attention_module, "apply_rotary_pos_emb", spying_apply)

        # Bypass core attention: this test only checks RoPE's group selection.
        # (Patch the class forward: nn.Module.__setattr__ refuses to replace a
        # registered submodule with a plain callable.)
        hidden_size = self.transformer_config.hidden_size

        def fake_core_attention_forward(module_self, query, *args, **kwargs):
            return torch.zeros(
                (query.shape[0], hidden_size), dtype=torch.bfloat16, device=query.device
            )

        monkeypatch.setattr(
            type(self.parallel_attention.core_attention), "forward", fake_core_attention_forward
        )

        sequence_length = 32
        self.parallel_attention.cuda()
        hidden_states = torch.ones(
            (sequence_length, 1, self.transformer_config.hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        rotary_pos_emb = RotaryEmbedding(kv_channels=16, rotary_percent=1.0)(sequence_length)

        build_time_group = self.parallel_attention.pg_collection.cp

        # Microbatch with a runtime CP group: RoPE must use it during the
        # forward, then the collection must restore its build-time group.
        runtime_group = self._runtime_cp_group()
        packed_seq_params = make_test_packed_seq_params(sequence_length)
        packed_seq_params.cp_group = runtime_group
        packed_seq_params.local_cp_size = 2
        self.parallel_attention(
            hidden_states, None, rotary_pos_emb=rotary_pos_emb, packed_seq_params=packed_seq_params
        )
        assert captured and all(group is runtime_group for group in captured)
        assert self.parallel_attention.pg_collection.cp is build_time_group

        # Next microbatch without a runtime group (e.g. local_cp_size == 1):
        # the build-time group must be restored, not the previous microbatch's.
        captured.clear()
        packed_seq_params = make_test_packed_seq_params(sequence_length)
        self.parallel_attention(
            hidden_states, None, rotary_pos_emb=rotary_pos_emb, packed_seq_params=packed_seq_params
        )
        assert captured and all(group is build_time_group for group in captured)
        assert self.parallel_attention.pg_collection.cp is build_time_group

    def test_te_cp_stream_lazily_created_for_runtime_cp_group(self, monkeypatch):
        import transformer_engine.pytorch as te_pytorch

        from megatron.core.extensions.transformer_engine import TEDotProductAttention

        core_attention = self.parallel_attention.core_attention
        assert isinstance(core_attention, TEDotProductAttention)

        # Model built with context_parallel_size == 1: constructor allocated no stream.
        monkeypatch.setattr(TEDotProductAttention, "cp_stream", None)

        captured = {}

        def fake_set_context_parallel_group(self, cp_group, ranks, stream, comm_type=None):
            captured["stream"] = stream

        def fake_forward(self, query, *args, **kwargs):
            return torch.zeros(
                (query.shape[0], query.shape[-2] * query.shape[-1]),
                dtype=query.dtype,
                device=query.device,
            )

        monkeypatch.setattr(
            te_pytorch.DotProductAttention,
            "set_context_parallel_group",
            fake_set_context_parallel_group,
        )
        monkeypatch.setattr(te_pytorch.DotProductAttention, "forward", fake_forward)

        core_attention.cuda()
        query = torch.zeros(32, 4, 16, dtype=torch.bfloat16, device="cuda")
        key = torch.zeros_like(query)
        value = torch.zeros_like(query)
        packed_seq_params = make_test_packed_seq_params(32)
        packed_seq_params.cp_group = self._runtime_cp_group()
        packed_seq_params.local_cp_size = 2

        core_attention(
            query,
            key,
            value,
            None,
            AttnMaskType.padding_causal,
            packed_seq_params=packed_seq_params,
        )

        assert isinstance(captured.get("stream"), torch.cuda.Stream)
        assert isinstance(TEDotProductAttention.cp_stream, torch.cuda.Stream)

    def test_cp_group_with_local_cp_size_one_is_rejected(self):
        core_attention = self.parallel_attention.core_attention.cuda()
        query = torch.zeros(32, 4, 16, dtype=torch.bfloat16, device="cuda")
        packed_seq_params = make_test_packed_seq_params(32)
        packed_seq_params.cp_group = torch.distributed.new_group(
            ranks=[torch.distributed.get_rank()]
        )
        packed_seq_params.local_cp_size = 1

        with pytest.raises(AssertionError, match="local_cp_size == 1"):
            core_attention(
                query,
                torch.zeros_like(query),
                torch.zeros_like(query),
                None,
                AttnMaskType.padding_causal,
                packed_seq_params=packed_seq_params,
            )
