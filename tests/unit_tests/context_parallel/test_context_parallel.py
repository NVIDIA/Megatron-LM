# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
from torch import Tensor

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils
from megatron.core.context_parallel import DefaultContextParallelHandler, get_cp_handler_cls
from megatron.core import parallel_state

from megatron.core.models.common.embeddings import RotaryEmbedding
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.process_groups_config import ProcessGroupCollection


import torch
import pytest
from typing import Optional

# Assuming necessary imports are available in the running environment:
# from megatron.core import parallel_state, Utils
# from megatron.core.transformer.transformer_config import TransformerConfig
# from megatron.core.transformer.custom_layers.transformer_engine import get_gpt_layer_with_transformer_engine_spec
# from megatron.core.transformer.attention import SelfAttention, AttnMaskType
# from megatron.core.models.gpt.gpt_embedding import RotaryEmbedding
# from megatron.core.distributed import ProcessGroupCollection
# from megatron.core.context_parallel.utils import get_cp_handler_cls
# from megatron.core.utils import apply_rotary_pos_emb


class TestContextParallelHandler:
    def setup_method(self, method):
        """
        Initialize the distributed environment and model config before each test.
        Sets up Context Parallel (CP) size to 8 and uses BF16.
        """
        Utils.initialize_model_parallel(1, 1, context_parallel_size=8)
        # Mocking manual seed if necessary
        # model_parallel_cuda_manual_seed(123)
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)

        # Use BF16 and a large enough hidden size to enable FlashAttention for thd format.
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=2048,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
            context_parallel_size=8,
        )

        pgs = ProcessGroupCollection()
        pgs.tp = parallel_state.get_tensor_model_parallel_group()
        pgs.cp = parallel_state.get_context_parallel_group()

        self.parallel_attention = SelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_spec().submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=pgs,
        )

        self.rope = RotaryEmbedding(
            kv_channels=self.transformer_config.kv_channels, rotary_percent=1.0
        )

    def teardown_method(self, method):
        """Destroy the distributed environment after each test."""
        Utils.destroy_model_parallel()

    @pytest.mark.skip
    @pytest.mark.parametrize("backend", ["transformer_engine"])
    def test_dispatch_combine_thd(self, backend: str):
        """
        Test dispatch and combine logic for THD format (Packed/VarLen sequences).
        Verifies both forward pass (Rotary Embedding) and backward pass (Gradients).
        """
        cp_handler_cls = get_cp_handler_cls(backend=backend)

        config = self.parallel_attention.config
        self.parallel_attention.cuda()

        # Define cumulative sequence lengths for packed sequence
        cu_seqlens = torch.IntTensor([0, 18, 44, 52, 96, 128]).cuda()
        sequence_length = 128
        micro_batch_size = 1

        # Create hidden states: [sequence length, batch size, hidden size]
        hidden_states = torch.arange(0, sequence_length)[:, None, None].expand(
            (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        hidden_states.requires_grad_(True)
        grad_output = torch.randn_like(hidden_states)

        # Initialize the Context Parallel Handler
        handler = cp_handler_cls(
            qkv_format="thd",
            cp_group=parallel_state.get_context_parallel_group(),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
        )

        rotary_pos_emb = self.rope(handler.max_seqlen_q, cp_handler=handler)

        # 1. Test dispatch and combine raw reconstruction
        local_states = handler.dispatch(seq_dim=0, tensor=hidden_states)
        total_states = handler.combine(seq_dim=0, tensor=local_states)
        assert torch.equal(total_states, hidden_states)

        # 2. Test Forward with RoPE (Distributed)
        local_states_rotated = handler.apply_rotary_pos_emb(
            local_states, rotary_pos_emb, config=config
        )

        total_states_rotated = handler.combine(seq_dim=0, tensor=local_states_rotated)

        # 3. Test Backward (Distributed)
        total_states_rotated.backward(grad_output)
        grad_input = hidden_states.grad
        hidden_states.grad = None

        # 4. Reference Implementation (Single GPU / Global View)
        hidden_states_rotated_ref = apply_rotary_pos_emb(
            hidden_states,
            rotary_pos_emb,
            config=config,
            cu_seqlens=cu_seqlens,
            # HACK: Use TP group because its world size is 1, simulating reference behavior
            cp_group=parallel_state.get_tensor_model_parallel_group(),
        )

        hidden_states_rotated_ref.backward(grad_output)
        grad_input_ref = hidden_states.grad
        hidden_states.grad = None

        # 5. Compare Results
        torch.testing.assert_close(
            total_states_rotated, hidden_states_rotated_ref, atol=1e-10, rtol=1e-4
        )

        torch.testing.assert_close(grad_input, grad_input_ref, atol=1e-10, rtol=1e-4)

    @pytest.mark.parametrize("backend", ["transformer_engine"])
    def test_dispatch_combine_sbhd(self, backend: str):
        """
        Test dispatch and combine logic for SBHD format (Standard layout: Seq, Batch, Head, Dim).
        Verifies both forward pass (Rotary Embedding) and backward pass (Gradients).
        """
        cp_handler_cls = get_cp_handler_cls(backend=backend)
        config = self.parallel_attention.config
        self.parallel_attention.cuda()

        sequence_length = 128
        micro_batch_size = 1

        # Create hidden states: [sequence length, batch size, num_heads, head_dim]
        # Note: expand does not allocate memory, but logic holds for testing
        hidden_states = torch.arange(0, sequence_length)[:, None, None, None].expand(
            (sequence_length, micro_batch_size, config.num_attention_heads, config.kv_channels)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        # Enable gradient calculation for backward pass testing
        hidden_states.requires_grad_(True)
        grad_output = torch.randn_like(hidden_states)

        # Initialize handler for SBHD format
        handler = cp_handler_cls(
            qkv_format="sbhd",
            cp_group=parallel_state.get_context_parallel_group(),
            max_seqlen_q=sequence_length,
            max_seqlen_kv=sequence_length,
        )

        # Generate RoPE embeddings for Distributed (CP) and Reference (Global)
        rotary_pos_emb = self.rope(handler.max_seqlen_q, cp_handler=handler)
        # HACK: Use TP group because its world size is 1, simulating reference behavior
        rotary_pos_emb_ref = self.rope(
            handler.max_seqlen_q,
            cp_handler=cp_handler_cls(
                qkv_format="sbhd",
                cp_group=parallel_state.get_tensor_model_parallel_group(),
                max_seqlen_q=sequence_length,
                max_seqlen_kv=sequence_length,
            ),
        )

        # 1. Test dispatch and combine raw reconstruction (Roundtrip check)
        local_states = handler.dispatch(seq_dim=0, tensor=hidden_states)
        total_states = handler.combine(seq_dim=0, tensor=local_states)
        assert torch.equal(total_states, hidden_states)

        # 2. Distributed Path: Apply RoPE on local chunks and gather
        local_states_rotated = handler.apply_rotary_pos_emb(
            local_states, rotary_pos_emb, config=config
        )
        total_states_rotated = handler.combine(seq_dim=0, tensor=local_states_rotated)

        # Run Backward on Distributed result
        total_states_rotated.backward(grad_output)
        grad_input_dist = hidden_states.grad
        hidden_states.grad = None

        # 3. Reference Path: Apply RoPE on global tensor directly
        hidden_states_rotated_ref = apply_rotary_pos_emb(
            hidden_states,
            rotary_pos_emb_ref,
            config=config,
            # Use TP group (size=1) to act as a global reference without splitting
            cp_group=parallel_state.get_tensor_model_parallel_group(),
        )

        # Run Backward on Reference result
        hidden_states_rotated_ref.backward(grad_output)
        grad_input_ref = hidden_states.grad
        hidden_states.grad = None

        # 4. Compare Forward Results
        torch.testing.assert_close(
            total_states_rotated, hidden_states_rotated_ref, atol=1e-10, rtol=1e-4
        )

        # 5. Compare Backward Gradients
        torch.testing.assert_close(grad_input_dist, grad_input_ref, atol=1e-10, rtol=1e-4)

    @pytest.mark.skip
    def test_parallel_attention_sbhd(self):
        config = self.parallel_attention.config
        self.parallel_attention.cuda()

        sequence_length = 8192 * 8
        micro_batch_size = 1

        # [batch size, sequence length, hidden size]
        hidden_states = torch.arange(0, sequence_length)[:, None, None].expand(
            (sequence_length, micro_batch_size, config.num_attention_heads * config.kv_channels)
        )
        hidden_states = hidden_states.cuda().to(torch.bfloat16)

        handler = TEContextParallelHandler(
            qkv_format="sbhd",
            attn_backend="te_fused_attention",
            cp_comm_type="p2p",
            cp_group=parallel_state.get_context_parallel_group(),
            max_seqlen_q=sequence_length,
            max_seqlen_kv=sequence_length,
        )

        local_hidden_states = handler.dispatch(seq_dim=0, tensor=hidden_states)
        rotary_pos_emb = self.rope(handler.max_seqlen_q, cp_handler=handler)

        torch.cuda.profiler.start()
        for i in range(100):
            with torch.cuda.nvtx.range(f"iter_{i}"):
                out = self.parallel_attention(
                    local_hidden_states,
                    attention_mask=None,
                    rotary_pos_emb=rotary_pos_emb,
                    cp_handler=handler,
                )
        torch.cuda.profiler.stop()
