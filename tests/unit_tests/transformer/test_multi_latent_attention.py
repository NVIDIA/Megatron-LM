# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import warnings
from inspect import signature
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.context_parallel_layout import prebuild_thd_cp_partition_routes
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank as get_tensor_on_this_cp_rank,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import (
    FusedMLASelfAttention,
    MLASelfAttention,
    MLASelfAttentionSubmodules,
    MultiLatentAttention,
)
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import is_te_min_version, is_torch_min_version, unwrap_model
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
)
from tests.unit_tests.test_utilities import Utils


def make_test_packed_seq_params(sequence_length=None, cu_seqlens=None):
    if cu_seqlens is None:
        assert sequence_length is not None
        cu_seqlens = [0, 6, 19, 22, sequence_length]
    cu_seqlens = torch.IntTensor(cu_seqlens).cuda()
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


def make_test_packed_seq_params_with_padding(
    sequence_length=None, cu_seqlens=None, cu_seqlens_padded=None
):
    """Create PackedSeqParams with both regular and padded cu_seqlens for testing padded sequences."""
    if cu_seqlens is None:
        assert sequence_length is not None
        cu_seqlens = [
            0,
            6,
            19,
            22,
            sequence_length - 8,
        ]  # Actual sequence lengths (with some padding removed)
    if cu_seqlens_padded is None:
        assert sequence_length is not None
        cu_seqlens_padded = [0, 8, 22, 28, sequence_length]  # Padded sequence lengths

    cu_seqlens = torch.IntTensor(cu_seqlens).cuda()
    cu_seqlens_padded = torch.IntTensor(cu_seqlens_padded).cuda()

    # Use padded lengths for max_seqlen calculation
    seqlens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    max_seqlen, _ = seqlens_padded.max(dim=0, keepdim=True)
    max_seqlen = max_seqlen.tolist()[0]

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


def get_mla_self_attn_submodules(linear_qkv_down_proj=None, qk_layernorm=False):
    submodules = get_gpt_layer_with_transformer_engine_submodules(
        multi_latent_attention=True, qk_layernorm=qk_layernorm
    ).self_attention.submodules
    assert isinstance(submodules, MLASelfAttentionSubmodules)
    if linear_qkv_down_proj is not None:
        submodules.linear_q_down_proj = linear_qkv_down_proj
        submodules.linear_kv_down_proj = linear_qkv_down_proj
    return submodules


def get_fused_mla_submodules(qk_layernorm=False):
    """Get submodules for FusedMLASelfAttention via the mla_down_proj_fusion spec path."""
    submodules = get_gpt_layer_with_transformer_engine_submodules(
        multi_latent_attention=True, mla_down_proj_fusion=True, qk_layernorm=qk_layernorm
    ).self_attention.submodules
    assert isinstance(submodules, MLASelfAttentionSubmodules)
    assert submodules.linear_qkv_down_proj is not None
    return submodules


backend = TESpecProvider()
linear_qkv_down_proj_options = [backend.linear(), backend.column_parallel_linear()]


@pytest.mark.parametrize("rope_type", ('yarn', 'rope'))
class TestParallelMLAAttention:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, rope_type):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rope_type=rope_type,
            rotary_base=10000,
            original_max_position_embeddings=32,
        )
        self.parallel_attention = MLASelfAttention(
            self.transformer_config,
            get_mla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_input_params_forward(self):
        """
        Test to ensure that MultiLatentAttention has all parameters
        required by the Attention class's forward method.
        """
        # Extract parameters from the forward methods of both Attention and MultiLatentAttention
        attn_params = set(signature(Attention.forward).parameters.keys())
        mla_params = set(signature(MultiLatentAttention.forward).parameters.keys())

        # Identify parameters that are in Attention but missing in MultiLatentAttention
        missing_params = attn_params - mla_params
        assert not missing_params, f"Missing parameters in MultiLatentAttention: {missing_params}"

    def test_constructor(self):
        assert isinstance(self.parallel_attention, MLASelfAttention)
        assert self.parallel_attention.layer_number == 1

        num_weights = sum([p.numel() for p in self.parallel_attention.parameters()])
        assert num_weights == 65036

    def test_attention_latent_norm_epsilon_on_fused_projections(self):
        config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            qk_layernorm=True,
            layernorm_epsilon=1.0e-5,
            attention_latent_norm_epsilon=1.0e-6,
        )
        attention = MLASelfAttention(
            config,
            get_mla_self_attn_submodules(qk_layernorm=True),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

        assert attention.linear_q_up_proj.eps == pytest.approx(1.0e-6)
        assert attention.linear_kv_up_proj.eps == pytest.approx(1.0e-6)

    def test_cpu_forward(self):
        # we can't currently do this because the global memory buffer is on GPU
        pass

    def test_gpu_forward(self):
        if is_te_min_version("1.10.0"):
            config = self.parallel_attention.config
            sequence_length = 32
            micro_batch_size = 2

            self.parallel_attention.cuda()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
            )
            hidden_states = hidden_states.cuda()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            output, bias = self.parallel_attention(hidden_states, attention_mask)

            assert config.recompute_granularity is None
            assert output.shape[0] == sequence_length
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == config.hidden_size
            assert bias.shape[0] == config.hidden_size

    @pytest.mark.experimental
    def test_gpu_forward_with_yarn_rope_fusion(self):
        if self.transformer_config.rope_type == "rope":
            pytest.skip("Rope is not supported for this test")
        if is_te_min_version("1.10.0"):
            transformer_config = self.transformer_config
            transformer_config.apply_rope_fusion = True
            checkpointed_parallel_attention = MLASelfAttention(
                transformer_config,
                get_mla_self_attn_submodules(),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
            )
            config = checkpointed_parallel_attention.config

            sequence_length = 32
            micro_batch_size = 2

            checkpointed_parallel_attention.cuda()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (
                    sequence_length,
                    micro_batch_size,
                    checkpointed_parallel_attention.config.hidden_size,
                )
            )
            hidden_states = hidden_states.cuda()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            output, bias = checkpointed_parallel_attention(hidden_states, attention_mask)

            assert config.apply_rope_fusion == True

    def test_gpu_forward_thd(self):
        if is_te_min_version("1.10.0"):
            # use flash attention for hopper, future may support fused attention for ampere
            _environ = os.environ.copy()
            os.environ['NVTE_FUSED_ATTN'] = "1"
            os.environ['NVTE_FLASH_ATTN'] = "0"

            config = self.parallel_attention.config
            sequence_length = 32
            micro_batch_size = 1

            self.parallel_attention.cuda().bfloat16()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
            )
            hidden_states = hidden_states.cuda().bfloat16()

            attention_mask = None
            packed_seq_params = make_test_packed_seq_params(sequence_length=sequence_length)
            output, bias = self.parallel_attention(
                hidden_states, attention_mask, packed_seq_params=packed_seq_params
            )

            assert config.recompute_granularity is None
            assert output.shape[0] == sequence_length
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == config.hidden_size
            assert bias.shape[0] == config.hidden_size
            os.environ.clear()
            os.environ.update(_environ)

    def test_gpu_forward_thd_padded(self):
        """Test MLA forward pass with cu_seqlens_q_padded and cu_seqlens_kv_padded."""
        if is_te_min_version("1.10.0"):
            config = self.parallel_attention.config
            sequence_length = 32
            micro_batch_size = 1

            self.parallel_attention.cuda().bfloat16()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
            )
            hidden_states = hidden_states.cuda().bfloat16()

            attention_mask = None

            # Create packed seq params with both regular and padded cu_seqlens
            packed_seq_params = make_test_packed_seq_params_with_padding(
                sequence_length=sequence_length
            )

            # Verify that the PackedSeqParams has both regular and padded cu_seqlens
            assert packed_seq_params.cu_seqlens_q is not None
            assert packed_seq_params.cu_seqlens_kv is not None
            assert packed_seq_params.cu_seqlens_q_padded is not None
            assert packed_seq_params.cu_seqlens_kv_padded is not None

            # Test the forward pass with padded cu_seqlens
            output, bias = self.parallel_attention(
                hidden_states, attention_mask, packed_seq_params=packed_seq_params
            )

            assert config.recompute_granularity is None
            assert output.shape[0] == sequence_length
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == config.hidden_size
            assert bias.shape[0] == config.hidden_size

            # Test that the get_query_key_value_tensors function properly handles padded cu_seqlens
            query, key, value, q_compressed, kv_compressed = (
                self.parallel_attention.get_query_key_value_tensors(
                    hidden_states, None, None, packed_seq_params, None
                )
            )

            assert query is not None
            assert key is not None
            assert value is not None
            assert q_compressed is not None
            assert kv_compressed is not None
            assert query.is_contiguous()
            assert key.is_contiguous()
            assert value.is_contiguous()

    def test_gpu_forward_thd_qv_head_dim_mismatch(self):
        """Test THD MLA path when q and v head dimensions differ."""
        if is_te_min_version("1.10.0"):
            transformer_config = MLATransformerConfig(
                num_layers=2,
                hidden_size=12,
                num_attention_heads=4,
                use_cpu_initialization=True,
                q_lora_rank=32,
                kv_lora_rank=32,
                qk_head_dim=128,
                v_head_dim=64,
                qk_pos_emb_head_dim=64,
                rope_type=self.transformer_config.rope_type,
                rotary_base=self.transformer_config.rotary_base,
                original_max_position_embeddings=self.transformer_config.original_max_position_embeddings,
            )
            mismatch_attention = MLASelfAttention(
                transformer_config,
                get_mla_self_attn_submodules(),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
            )

            sequence_length = 32
            micro_batch_size = 1

            mismatch_attention.cuda().bfloat16()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (sequence_length, micro_batch_size, transformer_config.hidden_size)
            )
            hidden_states = hidden_states.cuda().bfloat16()

            attention_mask = None
            with mock.patch.dict(
                os.environ, {"NVTE_FUSED_ATTN": "1", "NVTE_FLASH_ATTN": "0"}, clear=False
            ):
                packed_seq_params = make_test_packed_seq_params(sequence_length=sequence_length)

                query, key, value, q_compressed, kv_compressed = (
                    mismatch_attention.get_query_key_value_tensors(
                        hidden_states, None, None, packed_seq_params, None
                    )
                )

                assert query is not None
                assert key is not None
                assert value is not None
                assert q_compressed is not None
                assert kv_compressed is not None
                assert query.shape[-1] != value.shape[-1]

                output, bias = mismatch_attention(
                    hidden_states, attention_mask, packed_seq_params=packed_seq_params
                )

                assert output.shape[0] == sequence_length
                assert output.shape[1] == micro_batch_size
                assert output.shape[2] == transformer_config.hidden_size
                assert bias.shape[0] == transformer_config.hidden_size

    def test_checkpointed_gpu_forward(self):
        if is_te_min_version("1.10.0"):
            transformer_config = self.transformer_config
            transformer_config.recompute_granularity = 'selective'
            checkpointed_parallel_attention = MLASelfAttention(
                transformer_config,
                get_mla_self_attn_submodules(),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
            )
            config = checkpointed_parallel_attention.config

            sequence_length = 32
            micro_batch_size = 2

            checkpointed_parallel_attention.cuda()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (
                    sequence_length,
                    micro_batch_size,
                    checkpointed_parallel_attention.config.hidden_size,
                )
            )
            hidden_states = hidden_states.cuda()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            output, bias = checkpointed_parallel_attention(hidden_states, attention_mask)

            assert config.recompute_granularity == 'selective'
            assert "core_attn" in config.recompute_modules
            assert output.shape[0] == sequence_length
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == config.hidden_size
            assert bias.shape[0] == config.hidden_size

    def test_up_proj_recomputed_gpu_forward(self):
        if is_te_min_version("1.10.0"):
            transformer_config = self.transformer_config
            transformer_config.recompute_granularity = 'selective'
            transformer_config.recompute_modules = ["mla_up_proj"]
            checkpointed_parallel_attention = MLASelfAttention(
                transformer_config,
                get_mla_self_attn_submodules(),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
            )
            config = checkpointed_parallel_attention.config

            sequence_length = 32
            micro_batch_size = 2

            checkpointed_parallel_attention.cuda()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (
                    sequence_length,
                    micro_batch_size,
                    checkpointed_parallel_attention.config.hidden_size,
                )
            )
            hidden_states = hidden_states.cuda()

            q, k, v, q_compressed, kv_compressed = (
                checkpointed_parallel_attention.get_query_key_value_tensors(hidden_states)
            )
            assert q.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            output, bias = checkpointed_parallel_attention(hidden_states, attention_mask)

            assert checkpointed_parallel_attention.recompute_up_proj == True
            assert output.shape[0] == sequence_length
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == config.hidden_size
            assert bias.shape[0] == config.hidden_size


@pytest.mark.parametrize("linear_qkv_down_proj", linear_qkv_down_proj_options)
class TestSequenceParallelMLAAttention:
    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, linear_qkv_down_proj):
        self.tensor_parallel_size = 2
        Utils.initialize_model_parallel(self.tensor_parallel_size, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rotary_base=10000,
            original_max_position_embeddings=64,
            tensor_model_parallel_size=self.tensor_parallel_size,
            sequence_parallel=True,
        )
        self.parallel_attention = MLASelfAttention(
            self.transformer_config,
            get_mla_self_attn_submodules(linear_qkv_down_proj=linear_qkv_down_proj),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        if is_te_min_version("1.10.0"):
            config = self.parallel_attention.config
            sequence_length = 64
            sub_sequence_length = sequence_length // self.tensor_parallel_size
            micro_batch_size = 2

            self.parallel_attention.cuda()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (sub_sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
            )
            hidden_states = hidden_states.cuda()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            output, bias = self.parallel_attention(hidden_states, attention_mask)

            assert config.recompute_granularity is None
            assert output.shape[0] == sub_sequence_length
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == config.hidden_size
            assert bias.shape[0] == config.hidden_size


@pytest.mark.parametrize("linear_qkv_down_proj", linear_qkv_down_proj_options)
class TestTensorParallelMLAAttention:
    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, linear_qkv_down_proj):
        self.tensor_parallel_size = 2
        Utils.initialize_model_parallel(self.tensor_parallel_size, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rotary_base=10000,
            original_max_position_embeddings=64,
            tensor_model_parallel_size=self.tensor_parallel_size,
            sequence_parallel=False,
        )
        self.parallel_attention = MLASelfAttention(
            self.transformer_config,
            get_mla_self_attn_submodules(linear_qkv_down_proj=linear_qkv_down_proj),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        if is_te_min_version("1.10.0"):
            config = self.parallel_attention.config
            sequence_length = 64
            micro_batch_size = 2

            self.parallel_attention.cuda()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
            )
            hidden_states = hidden_states.cuda()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            output, bias = self.parallel_attention(hidden_states, attention_mask)

            assert config.recompute_granularity is None
            assert output.shape[0] == sequence_length
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == config.hidden_size
            assert bias.shape[0] == config.hidden_size


@pytest.mark.experimental
@pytest.mark.skipif(
    not is_te_min_version("2.5.0", check_equality=True),
    reason="Requires TransformerEngine >= 2.5.0",
)
@pytest.mark.parametrize(
    ("rope_type", "apply_rope_fusion"),
    (('rope', False), ('rope', True), ('yarn', False), ('yarn', True)),
)
class TestContextParallelMLAAttention:

    @pytest.fixture(scope='function', autouse=True)
    def setup_method(self, rope_type, apply_rope_fusion):
        self.context_parallel_size = 4
        Utils.initialize_model_parallel(1, 1, context_parallel_size=self.context_parallel_size)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rotary_base=10000,
            original_max_position_embeddings=64,
            context_parallel_size=self.context_parallel_size,
            bf16=True,
            rope_type=rope_type,
            apply_rope_fusion=apply_rope_fusion,
        )
        self.parallel_attention = MLASelfAttention(
            self.transformer_config,
            get_mla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).bfloat16()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward(self):
        if is_te_min_version("2.5.0", check_equality=True):
            config = self.parallel_attention.config
            sequence_length = 64
            micro_batch_size = 2

            self.parallel_attention.cuda()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (
                    sequence_length // self.context_parallel_size,
                    micro_batch_size,
                    self.parallel_attention.config.hidden_size,
                )
            ).bfloat16()
            hidden_states = hidden_states.cuda()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            output, bias = self.parallel_attention(hidden_states, attention_mask)

            assert config.recompute_granularity is None
            assert output.shape[0] == sequence_length // self.context_parallel_size
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == config.hidden_size
            assert bias.shape[0] == config.hidden_size

    def test_gpu_forward_thd(self):
        if is_te_min_version("2.5.0", check_equality=True):
            config = self.parallel_attention.config
            sequence_length = 128
            micro_batch_size = 1
            cu_seqlens = [0, 16, 48, 64, 128]
            self.parallel_attention.cuda()

            # [sequence length, batch size, hidden size]
            hidden_states = torch.ones(
                (
                    sequence_length // self.context_parallel_size,
                    micro_batch_size,
                    self.parallel_attention.config.hidden_size,
                )
            ).bfloat16()
            hidden_states = hidden_states.cuda()

            attention_mask = None
            packed_seq_params = make_test_packed_seq_params(cu_seqlens=cu_seqlens)

            output, bias = self.parallel_attention(
                hidden_states, attention_mask, packed_seq_params=packed_seq_params
            )

            assert config.recompute_granularity is None
            assert output.shape[0] == sequence_length // self.context_parallel_size
            assert output.shape[1] == micro_batch_size
            assert output.shape[2] == config.hidden_size
            assert bias.shape[0] == config.hidden_size


def _thd_cp_layout_token_indices(cu_seqlens_padded, cp_size, cp_rank, cp_partition_mode):
    """Global packed-token indices held by one CP rank for a THD layout."""
    total_tokens = cu_seqlens_padded[-1]
    if cp_partition_mode == "contiguous":
        part_len = total_tokens // cp_size
        return torch.arange(cp_rank * part_len, (cp_rank + 1) * part_len, dtype=torch.long)
    assert cp_partition_mode == "zigzag"
    token_indices = []
    for seq_start, seq_end in zip(cu_seqlens_padded[:-1], cu_seqlens_padded[1:]):
        chunk_len = (seq_end - seq_start) // (2 * cp_size)
        for chunk in (cp_rank, 2 * cp_size - cp_rank - 1):
            chunk_start = seq_start + chunk * chunk_len
            token_indices.extend(range(chunk_start, chunk_start + chunk_len))
    return torch.tensor(token_indices, dtype=torch.long)


def _thd_sp_shard_token_indices(
    cu_seqlens_padded, cp_size, cp_rank, tp_size, tp_rank, cp_partition_mode
):
    """Global packed-token indices of one (cp_rank, tp_rank) sequence-parallel shard."""
    cp_local = _thd_cp_layout_token_indices(cu_seqlens_padded, cp_size, cp_rank, cp_partition_mode)
    assert cp_local.numel() % tp_size == 0
    return cp_local.chunk(tp_size)[tp_rank].contiguous()


def _gather_full_thd(local, token_indices, total_tokens, group):
    """Reassemble the full packed sequence from equal-sized shards of a process group."""
    local = local.detach().contiguous()
    token_indices = token_indices.to(local.device)
    gathered = [torch.empty_like(local) for _ in range(group.size())]
    gathered_indices = [torch.empty_like(token_indices) for _ in range(group.size())]
    dist.all_gather(gathered, local, group=group)
    dist.all_gather(gathered_indices, token_indices, group=group)
    full = torch.empty(
        (total_tokens,) + tuple(local.shape[1:]), dtype=local.dtype, device=local.device
    )
    for shard, indices in zip(gathered, gathered_indices):
        full[indices] = shard
    return full


def _assert_bitwise_equal(actual, expected, what):
    """Assert bit-exact equality and describe the mismatch when it fails."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(
            f"{what}: shape/dtype mismatch, got {tuple(actual.shape)} {actual.dtype}, "
            f"expected {tuple(expected.shape)} {expected.dtype}"
        )
    if torch.equal(actual, expected):
        return
    mismatch = actual != expected
    abs_diff = (actual.float() - expected.float()).abs()
    raise AssertionError(
        f"{what}: {int(mismatch.sum())} of {mismatch.numel()} elements differ, "
        f"max abs diff = {abs_diff.max().item():.6g}"
    )


@pytest.mark.experimental
@pytest.mark.skipif(
    not is_te_min_version("2.5.0", check_equality=True),
    reason="Requires TransformerEngine >= 2.5.0",
)
@pytest.mark.parametrize(
    ("rope_type", "apply_rope_fusion"), (('rope', False), ('rope', True), ('yarn', True))
)
@pytest.mark.parametrize(("tp_size", "cp_size"), ((1, 2), (2, 2), (2, 4), (4, 2)))
@pytest.mark.parametrize("padded", (False, True), ids=("packed", "padded"))
class TestContextParallelMLAAttentionLayoutConversion:
    """MLA accepts a contiguous THD CP layout by converting to zigzag internally.

    A module fed contiguous shards (``config.cp_partition_mode="contiguous"``) must
    produce the same outputs and gradients as an identical module fed zigzag shards,
    both for pure CP and for TP x CP with sequence parallelism (fused TP x CP route).
    """

    hidden_size = 12
    total_tokens = 128
    cu_seqlens_padded = [0, 16, 48, 64, 128]

    def _make_config(self, tp_size, cp_size, rope_type, apply_rope_fusion):
        return MLATransformerConfig(
            num_layers=2,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rotary_base=10000,
            original_max_position_embeddings=64,
            tensor_model_parallel_size=tp_size,
            sequence_parallel=tp_size > 1,
            context_parallel_size=cp_size,
            bf16=True,
            rope_type=rope_type,
            apply_rope_fusion=apply_rope_fusion,
            # The modules run in training mode; dropout would consume different RNG
            # state in the reference and candidate forwards and break bit-exactness.
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )

    def _build_attention(self, config):
        model_parallel_cuda_manual_seed(123)
        attention = MLASelfAttention(
            config,
            get_mla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        return attention.bfloat16().cuda()

    def _make_packed_seq_params(self, padded, cp_partition_mode):
        if padded:
            packed_seq_params = make_test_packed_seq_params_with_padding(
                cu_seqlens=[0, 12, 40, 60, 120], cu_seqlens_padded=self.cu_seqlens_padded
            )
        else:
            packed_seq_params = make_test_packed_seq_params(cu_seqlens=self.cu_seqlens_padded)
        packed_seq_params.cp_partition_mode = cp_partition_mode
        return packed_seq_params

    def test_contiguous_input_matches_zigzag_reference(
        self, rope_type, apply_rope_fusion, tp_size, cp_size, padded
    ):
        required_world_size = tp_size * cp_size
        if (
            not torch.cuda.is_available()
            or Utils.world_size < required_world_size
            or Utils.world_size % required_world_size != 0
        ):
            pytest.skip(f"Needs a multiple of {required_world_size} CUDA ranks.")

        Utils.initialize_model_parallel(tp_size, 1, context_parallel_size=cp_size)
        try:
            cp_group = parallel_state.get_context_parallel_group()
            tp_group = parallel_state.get_tensor_model_parallel_group()
            tp_cp_group = parallel_state.get_tensor_and_context_parallel_group()
            cp_rank, tp_rank = cp_group.rank(), tp_group.rank()
            device = torch.device("cuda", torch.cuda.current_device())

            reference = self._build_attention(
                self._make_config(tp_size, cp_size, rope_type, apply_rope_fusion)
            )
            candidate_config = self._make_config(tp_size, cp_size, rope_type, apply_rope_fusion)
            # The module reads config.cp_partition_mode as the layout of its input. The
            # config-level contiguous-CP checks describe complete model builds (packing
            # scheduler, hybrid attention variants), so set the attribute directly here.
            candidate_config.cp_partition_mode = "contiguous"
            candidate = self._build_attention(candidate_config)
            candidate.load_state_dict(reference.state_dict())

            torch.manual_seed(7)
            full_hidden = torch.randn(
                self.total_tokens, 1, self.hidden_size, dtype=torch.bfloat16, device=device
            )
            full_grad = torch.randn_like(full_hidden)
            zigzag_indices = _thd_sp_shard_token_indices(
                self.cu_seqlens_padded, cp_size, cp_rank, tp_size, tp_rank, "zigzag"
            ).to(device)
            contiguous_indices = _thd_sp_shard_token_indices(
                self.cu_seqlens_padded, cp_size, cp_rank, tp_size, tp_rank, "contiguous"
            ).to(device)

            reference_input = full_hidden.index_select(0, zigzag_indices).requires_grad_(True)
            # Padded THD (padding between sequences) leaves TE with cuDNN fused attention
            # only: FlashAttention 2/4 are excluded for THD with padding between sequences
            # and the unfused backend for pad_between_seqs, and cuDNN rejects head_dim 192
            # bf16 THD with padding on some TE/cuDNN builds. That is a capability limit of
            # the reference configuration, independent of the layout conversion, so it is
            # a skip rather than a failure; the candidate forward and all comparisons stay
            # strict.
            try:
                reference_output, reference_bias = reference(
                    reference_input,
                    None,
                    packed_seq_params=self._make_packed_seq_params(padded, "zigzag"),
                )
            except ValueError as error:
                if "No dot product attention backend" not in str(error):
                    raise
                pytest.skip(
                    "TransformerEngine has no attention backend for padded THD MLA attention "
                    "(head_dim 192, bf16) on this TE/cuDNN build; "
                    "TestParallelMLAAttention::test_gpu_forward_thd_padded fails the same way"
                )
            reference_output.mul(full_grad.index_select(0, zigzag_indices)).sum().backward()

            candidate_packed_seq_params = self._make_packed_seq_params(padded, "contiguous")
            prebuild_thd_cp_partition_routes(
                candidate_packed_seq_params, cp_group, tp_group=tp_group, tp_cp_group=tp_cp_group
            )
            candidate_input = full_hidden.index_select(0, contiguous_indices).requires_grad_(True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                candidate_output, candidate_bias = candidate(
                    candidate_input, None, packed_seq_params=candidate_packed_seq_params
                )
            assert not [
                w
                for w in caught
                if issubclass(w.category, (RuntimeWarning, FutureWarning))
                and "layout" in str(w.message)
            ], [str(w.message) for w in caught]
            # The output is handed back in the caller's (contiguous) layout.
            assert candidate_packed_seq_params.cp_partition_mode == "contiguous"
            assert candidate_output.shape == candidate_input.shape
            candidate_output.mul(full_grad.index_select(0, contiguous_indices)).sum().backward()

            full_reference_output = _gather_full_thd(
                reference_output, zigzag_indices, self.total_tokens, tp_cp_group
            )
            full_candidate_output = _gather_full_thd(
                candidate_output, contiguous_indices, self.total_tokens, tp_cp_group
            )
            # The conversion is a pure permutation, so the forward must be bit-exact.
            _assert_bitwise_equal(full_candidate_output, full_reference_output, "output")
            if reference_bias is not None:
                _assert_bitwise_equal(candidate_bias, reference_bias, "output bias")

            # Backward: the input gradient comes back in the contiguous layout, and every
            # rank accumulates parameter gradients from the same zigzag tokens. Both runs
            # see identical tensors, but attention backward kernels are not guaranteed to
            # be run-to-run deterministic (atomic accumulation), so gradients are compared
            # with a tolerance instead of bit for bit.
            full_reference_grad = _gather_full_thd(
                reference_input.grad, zigzag_indices, self.total_tokens, tp_cp_group
            )
            full_candidate_grad = _gather_full_thd(
                candidate_input.grad, contiguous_indices, self.total_tokens, tp_cp_group
            )
            torch.testing.assert_close(
                full_candidate_grad.float(), full_reference_grad.float(), rtol=2e-2, atol=2e-2
            )
            reference_grads = dict(reference.named_parameters())
            for name, param in candidate.named_parameters():
                if param.grad is None:
                    assert reference_grads[name].grad is None, name
                    continue
                torch.testing.assert_close(
                    param.grad.float(),
                    reference_grads[name].grad.float(),
                    rtol=2e-2,
                    atol=2e-2,
                    msg=lambda message, name=name: f"{name}: {message}",
                )
        finally:
            Utils.destroy_model_parallel()


@pytest.mark.parametrize("gate_granularity", ("elementwise", "headwise"))
class TestMLAOutputGate:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, gate_granularity):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rope_type="yarn",
            rotary_base=10000,
            original_max_position_embeddings=32,
            attention_output_gate=True,
            gated_attention_proj_granularity=gate_granularity,
        )
        self.parallel_attention = MLASelfAttention(
            self.transformer_config,
            get_mla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor_builds_configured_gate_projection(self, gate_granularity):
        gate = self.parallel_attention.linear_gate
        assert gate is not None
        expected_projection_size = (
            self.parallel_attention.query_projection_size
            if gate_granularity == "elementwise"
            else self.transformer_config.num_attention_heads
        )
        assert gate.weight.shape == (expected_projection_size, self.transformer_config.hidden_size)
        assert gate.bias is None or gate.bias.numel() == 0

    def test_gate_matches_independent_fp32_sigmoid_reference(self, gate_granularity):
        output_size = self.parallel_attention.query_projection_size
        core_attn_out = torch.linspace(-2.0, 2.0, 2 * output_size, dtype=torch.bfloat16).view(
            2, 1, output_size
        )
        gate_size = (
            output_size
            if gate_granularity == "elementwise"
            else self.transformer_config.num_attention_heads
        )
        gate = torch.linspace(-4.0, 4.0, 2 * gate_size, dtype=torch.bfloat16).view(2, 1, gate_size)

        output = self.parallel_attention._apply_mla_output_gate(core_attn_out, gate)
        reference_scale = torch.sigmoid(gate.float()).to(core_attn_out.dtype)
        if gate_granularity == "headwise":
            reference_scale = torch.repeat_interleave(
                reference_scale, self.transformer_config.v_head_dim, dim=-1
            )
        expected = core_attn_out * reference_scale

        assert output.dtype == core_attn_out.dtype
        torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_gate_matches_fp32_sigmoid_reference_vjp(self, gate_granularity):
        generator = torch.Generator().manual_seed(20260814)
        output_size = self.parallel_attention.query_projection_size
        gate_size = (
            output_size
            if gate_granularity == "elementwise"
            else self.transformer_config.num_attention_heads
        )
        gate_input = torch.randn((8, 1, gate_size), generator=generator, dtype=torch.bfloat16)
        core_output = torch.randn((8, 1, output_size), generator=generator, dtype=torch.bfloat16)
        output_gradient = torch.randn(core_output.shape, generator=generator, dtype=torch.bfloat16)

        actual_gate = gate_input.detach().clone().requires_grad_(True)
        actual_core = core_output.detach().clone().requires_grad_(True)
        actual_output = self.parallel_attention._apply_mla_output_gate(actual_core, actual_gate)
        actual_gradients = torch.autograd.grad(
            actual_output, (actual_gate, actual_core), output_gradient
        )

        sigmoid_fp32 = torch.sigmoid(gate_input.float())
        native_scale = sigmoid_fp32.to(core_output.dtype)
        if gate_granularity == "headwise":
            native_scale = torch.repeat_interleave(
                native_scale, self.transformer_config.v_head_dim, dim=-1
            )
            expected_core_gradient = output_gradient * native_scale
            head_shape = (*core_output.shape[:-1], gate_size, self.transformer_config.v_head_dim)
            scale_gradient = (
                output_gradient.float().view(head_shape) * core_output.float().view(head_shape)
            ).sum(dim=-1)
            # The fused broadcast reduction returns a native-dtype gradient to the
            # FP32-sigmoid cast before applying the sigmoid derivative.
            scale_gradient = scale_gradient.to(gate_input.dtype).float()
        else:
            expected_core_gradient = output_gradient * native_scale
            scale_gradient = (output_gradient * core_output).float()

        expected_output = core_output * native_scale
        expected_gate_gradient = (scale_gradient * sigmoid_fp32 * (1.0 - sigmoid_fp32)).to(
            gate_input.dtype
        )

        torch.testing.assert_close(actual_output, expected_output, atol=0, rtol=0)
        torch.testing.assert_close(actual_gradients[0], expected_gate_gradient)
        torch.testing.assert_close(actual_gradients[1], expected_core_gradient)

    def test_missing_gate_spec_raises(self):
        submodules = get_mla_self_attn_submodules()
        submodules.linear_gate = None
        with pytest.raises(ValueError, match="linear_gate module spec"):
            MLASelfAttention(
                self.transformer_config,
                submodules,
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
            )

    def test_fused_down_projection_is_rejected(self, gate_granularity):
        with pytest.raises(ValueError, match="does not support fused down projections"):
            MLATransformerConfig(
                num_layers=2,
                hidden_size=12,
                num_attention_heads=4,
                q_lora_rank=32,
                kv_lora_rank=32,
                qk_head_dim=128,
                v_head_dim=128,
                qk_pos_emb_head_dim=64,
                rope_type=self.transformer_config.rope_type,
                rotary_base=10000,
                original_max_position_embeddings=32,
                attention_output_gate=True,
                gated_attention_proj_granularity=gate_granularity,
                mla_down_proj_fusion=True,
            )

    def test_gpu_forward_thd(self):
        if not is_te_min_version("1.10.0"):
            pytest.skip("MLA requires TransformerEngine >= 1.10.0")

        attention = self.parallel_attention.cuda().bfloat16()
        call_order = []
        core_attention_hook = attention.core_attention.register_forward_pre_hook(
            lambda *_: call_order.append("core_attention")
        )
        gate_projection_hook = attention.linear_gate.register_forward_pre_hook(
            lambda *_: call_order.append("linear_gate")
        )
        sequence_length = 32
        hidden_states = torch.ones(
            (sequence_length, 1, self.transformer_config.hidden_size),
            device='cuda',
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        packed_seq_params = make_test_packed_seq_params(sequence_length=sequence_length)

        try:
            with mock.patch.dict(
                os.environ, {"NVTE_FUSED_ATTN": "1", "NVTE_FLASH_ATTN": "0"}, clear=False
            ):
                output, bias = attention(
                    hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
                )
                output.sum().backward()
        finally:
            core_attention_hook.remove()
            gate_projection_hook.remove()

        assert call_order == ["core_attention", "linear_gate"]
        assert output.shape == hidden_states.shape
        assert bias.shape == (self.transformer_config.hidden_size,)
        assert hidden_states.grad is not None
        assert attention.linear_gate.weight.grad is not None


@pytest.mark.parametrize("rope_type", ('yarn', 'rope'))
class TestParallelMLAAttentionPrecision:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, rope_type):
        self._environ_backup = os.environ.copy()
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = "0"
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rope_type=rope_type,
            rotary_base=10000,
            original_max_position_embeddings=32,
            deterministic_mode=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        self.parallel_attention = MLASelfAttention(
            self.transformer_config,
            get_mla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        os.environ.clear()
        os.environ.update(self._environ_backup)
        Utils.destroy_model_parallel()

    def test_gpu_forward_thd_precision(self):
        if is_te_min_version("1.10.0"):
            # use flash attention for hopper, future may support fused attention for ampere
            _environ = os.environ.copy()
            os.environ['NVTE_FUSED_ATTN'] = "1"
            os.environ['NVTE_FLASH_ATTN'] = "0"

            config = self.parallel_attention.config

            self.parallel_attention.cuda().bfloat16()

            # Input shape
            sequence_length = 32
            micro_batch_size = 4
            cu_seqlens = [0, 32, 64, 96, 128]
            # sbhd input shape: [sequence length, batch size, hidden size]
            hidden_states_sbhd = torch.rand(
                (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
            )
            attention_mask_sbhd = torch.ones(
                (1, 1, sequence_length, sequence_length), dtype=bool
            ).cuda()
            # thd input shape: [sequence length * batch size, 1, hidden size]
            hidden_states_sbhd = hidden_states_sbhd.cuda().bfloat16()
            hidden_states_thd = hidden_states_sbhd.transpose(0, 1).contiguous()
            hidden_states_thd = hidden_states_thd.view(
                -1, 1, self.parallel_attention.config.hidden_size
            )
            attention_mask_thd = None
            packed_seq_params = make_test_packed_seq_params(cu_seqlens=cu_seqlens)

            # fine-grained check
            query_sbhd, key_sbhd, value_sbhd, q_compressed_sbhd, kv_compressed_sbhd = (
                self.parallel_attention.get_query_key_value_tensors(
                    hidden_states_sbhd, None, None, None, None
                )
            )
            query_thd, key_thd, value_thd, q_compressed_thd, kv_compressed_thd = (
                self.parallel_attention.get_query_key_value_tensors(
                    hidden_states_thd, None, None, packed_seq_params, None
                )
            )
            _query_sbhd = query_sbhd.transpose(0, 1).contiguous().view(*query_thd.shape)
            _key_sbhd = key_sbhd.transpose(0, 1).contiguous().view(*key_thd.shape)
            _value_sbhd = value_sbhd.transpose(0, 1).contiguous().view(*value_thd.shape)
            _q_compressed_sbhd = (
                q_compressed_sbhd.transpose(0, 1).contiguous().view(*q_compressed_thd.shape)
            )
            _kv_compressed_sbhd = (
                kv_compressed_sbhd.transpose(0, 1).contiguous().view(*kv_compressed_thd.shape)
            )
            assert torch.equal(_query_sbhd, query_thd)
            assert torch.equal(_key_sbhd, key_thd)
            assert torch.equal(_value_sbhd, value_thd)
            assert torch.equal(_q_compressed_sbhd, q_compressed_thd)
            assert torch.equal(_kv_compressed_sbhd, kv_compressed_thd)

            core_attn_out_sbhd = self.parallel_attention._run_core_attention(
                query_sbhd, key_sbhd, value_sbhd, attention_mask_sbhd, packed_seq_params=None
            )
            query_thd = query_thd.squeeze(1)
            key_thd = key_thd.squeeze(1)
            value_thd = value_thd.squeeze(1)
            core_attn_out_thd = self.parallel_attention._run_core_attention(
                query_thd,
                key_thd,
                value_thd,
                attention_mask_thd,
                packed_seq_params=packed_seq_params,
            )
            core_attn_out_thd = core_attn_out_thd.reshape(core_attn_out_thd.size(0), 1, -1)
            _core_attn_out_sbhd = (
                core_attn_out_sbhd.transpose(0, 1).contiguous().view(*core_attn_out_thd.shape)
            )
            assert torch.equal(_core_attn_out_sbhd, core_attn_out_thd)

            output_sbhd, bias_sbhd = apply_module(self.parallel_attention.linear_proj)(
                core_attn_out_sbhd
            )
            output_thd, bias_thd = apply_module(self.parallel_attention.linear_proj)(
                core_attn_out_thd
            )
            _output_sbhd = output_sbhd.transpose(0, 1).contiguous().view(*output_thd.shape)
            assert torch.equal(_output_sbhd, output_thd)

            output_thd_fine_grained = output_thd
            bias_thd_fine_grained = bias_thd

            # E2E check
            # sbhd
            output_sbhd, bias_sbhd = self.parallel_attention(
                hidden_states_sbhd, attention_mask_sbhd
            )
            # thd
            output_thd, bias_thd = self.parallel_attention(
                hidden_states_thd, attention_mask_thd, packed_seq_params=packed_seq_params
            )
            _output_sbhd = output_sbhd.transpose(0, 1).contiguous().view(*output_thd.shape)
            assert torch.equal(_output_sbhd, output_thd)
            assert bias_thd.shape == bias_sbhd.shape
            assert torch.equal(bias_sbhd, bias_thd)

            assert torch.equal(output_thd, output_thd_fine_grained)
            assert torch.equal(bias_thd, bias_thd_fine_grained)

            os.environ.clear()
            os.environ.update(_environ)


@pytest.mark.experimental
@pytest.mark.skipif(
    not is_te_min_version("2.5.0", check_equality=True),
    reason="Requires TransformerEngine >= 2.5.0",
)
@pytest.mark.parametrize(
    ("rope_type", "apply_rope_fusion"),
    (('rope', False), ('rope', True), ('yarn', False), ('yarn', True)),
)
class TestContextParallelMLAAttentionPrecision:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, rope_type, apply_rope_fusion):
        self._environ_backup = os.environ.copy()
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = "0"
        self.context_parallel_size = 4
        Utils.initialize_model_parallel(1, 1, context_parallel_size=self.context_parallel_size)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rotary_base=10000,
            original_max_position_embeddings=64,
            context_parallel_size=self.context_parallel_size,
            bf16=True,
            rope_type=rope_type,
            apply_rope_fusion=apply_rope_fusion,
            deterministic_mode=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        self.parallel_attention = MLASelfAttention(
            self.transformer_config,
            get_mla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).bfloat16()

    def teardown_method(self, method):
        os.environ.clear()
        os.environ.update(self._environ_backup)
        Utils.destroy_model_parallel()

    def test_gpu_forward_thd_precision(self):
        if is_te_min_version("2.5.0", check_equality=True):
            # use flash attention for hopper, future may support fused attention for ampere
            _environ = os.environ.copy()
            os.environ['NVTE_FUSED_ATTN'] = "1"
            os.environ['NVTE_FLASH_ATTN'] = "0"
            atol, rtol = 3e-4, 3e-4

            self.parallel_attention.cuda().bfloat16()

            # Input shape
            sequence_length = 32
            micro_batch_size = 4
            cu_seqlens = [0, 32, 64, 96, 128]
            # sbhd input shape: [sequence length, batch size, hidden size]
            hidden_states_sbhd = torch.rand(
                (
                    sequence_length // self.context_parallel_size,
                    micro_batch_size,
                    self.parallel_attention.config.hidden_size,
                )
            )
            attention_mask_sbhd = None
            # thd input shape: [sequence length * batch size, 1, hidden size]
            hidden_states_sbhd = hidden_states_sbhd.cuda().bfloat16()
            hidden_states_thd = hidden_states_sbhd.transpose(0, 1).contiguous()
            hidden_states_thd = hidden_states_thd.view(
                -1, 1, self.parallel_attention.config.hidden_size
            )
            attention_mask_thd = None
            packed_seq_params = make_test_packed_seq_params(cu_seqlens=cu_seqlens)

            # fine-grained check
            query_sbhd, key_sbhd, value_sbhd, q_compressed_sbhd, kv_compressed_sbhd = (
                self.parallel_attention.get_query_key_value_tensors(
                    hidden_states_sbhd, None, None, None, None
                )
            )
            query_thd, key_thd, value_thd, q_compressed_thd, kv_compressed_thd = (
                self.parallel_attention.get_query_key_value_tensors(
                    hidden_states_thd, None, None, packed_seq_params, None
                )
            )
            _query_sbhd = query_sbhd.transpose(0, 1).contiguous().view(*query_thd.shape)
            _key_sbhd = key_sbhd.transpose(0, 1).contiguous().view(*key_thd.shape)
            _value_sbhd = value_sbhd.transpose(0, 1).contiguous().view(*value_thd.shape)
            _q_compressed_sbhd = (
                q_compressed_sbhd.transpose(0, 1).contiguous().view(*q_compressed_thd.shape)
            )
            _kv_compressed_sbhd = (
                kv_compressed_sbhd.transpose(0, 1).contiguous().view(*kv_compressed_thd.shape)
            )
            torch.testing.assert_close(_query_sbhd, query_thd, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(_key_sbhd, key_thd, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(_value_sbhd, value_thd, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(_q_compressed_sbhd, q_compressed_thd, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(_kv_compressed_sbhd, kv_compressed_thd, atol=1e-6, rtol=1e-6)

            core_attn_out_sbhd = self.parallel_attention._run_core_attention(
                query_sbhd, key_sbhd, value_sbhd, attention_mask_sbhd, packed_seq_params=None
            )
            query_thd = query_thd.squeeze(1)
            key_thd = key_thd.squeeze(1)
            value_thd = value_thd.squeeze(1)
            core_attn_out_thd = self.parallel_attention._run_core_attention(
                query_thd,
                key_thd,
                value_thd,
                attention_mask_thd,
                packed_seq_params=packed_seq_params,
            )
            core_attn_out_thd = core_attn_out_thd.reshape(core_attn_out_thd.size(0), 1, -1)
            _core_attn_out_sbhd = (
                core_attn_out_sbhd.transpose(0, 1).contiguous().view(*core_attn_out_thd.shape)
            )
            torch.testing.assert_close(_core_attn_out_sbhd, core_attn_out_thd, atol=atol, rtol=rtol)

            output_sbhd, bias_sbhd = apply_module(self.parallel_attention.linear_proj)(
                core_attn_out_sbhd
            )
            output_thd, bias_thd = apply_module(self.parallel_attention.linear_proj)(
                core_attn_out_thd
            )
            _output_sbhd = output_sbhd.transpose(0, 1).contiguous().view(*output_thd.shape)
            torch.testing.assert_close(_output_sbhd, output_thd, atol=atol, rtol=rtol)

            output_thd_fine_grained = output_thd
            bias_thd_fine_grained = bias_thd

            # E2E check
            # sbhd
            output_sbhd, bias_sbhd = self.parallel_attention(
                hidden_states_sbhd, attention_mask_sbhd
            )
            # thd
            output_thd, bias_thd = self.parallel_attention(
                hidden_states_thd, attention_mask_thd, packed_seq_params=packed_seq_params
            )
            _output_sbhd = output_sbhd.transpose(0, 1).contiguous().view(*output_thd.shape)
            torch.testing.assert_close(_output_sbhd, output_thd, atol=atol, rtol=rtol)
            assert bias_thd.shape == bias_sbhd.shape
            torch.testing.assert_close(bias_sbhd, bias_thd, atol=atol, rtol=rtol)

            assert torch.equal(output_thd, output_thd_fine_grained)
            assert torch.equal(bias_thd, bias_thd_fine_grained)

            os.environ.clear()
            os.environ.update(_environ)


@pytest.mark.experimental
@pytest.mark.skipif(not is_torch_min_version("2.5.0"), reason="Requires PyTorch >= 2.5.0")
class TestParallelMLAAttentionPrecisionWithRopeFusion:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        self._environ_backup = os.environ.copy()
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = "0"
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rope_type="yarn",
            rotary_base=10000,
            original_max_position_embeddings=32,
            deterministic_mode=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            apply_rope_fusion=True,
        )
        self.parallel_attention = MLASelfAttention(
            self.transformer_config,
            get_mla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        os.environ.clear()
        os.environ.update(self._environ_backup)
        Utils.destroy_model_parallel()

    def test_gpu_forward_thd_precision(self):
        if is_te_min_version("1.10.0"):
            # use flash attention for hopper, future may support fused attention for ampere
            _environ = os.environ.copy()
            os.environ['NVTE_FUSED_ATTN'] = "1"
            os.environ['NVTE_FLASH_ATTN'] = "0"

            config = self.parallel_attention.config

            self.parallel_attention.cuda().bfloat16()

            # Input shape
            sequence_length = 32
            micro_batch_size = 4
            cu_seqlens = [0, 32, 64, 96, 128]
            # sbhd input shape: [sequence length, batch size, hidden size]
            hidden_states_sbhd = torch.rand(
                (sequence_length, micro_batch_size, self.parallel_attention.config.hidden_size)
            )
            attention_mask_sbhd = torch.ones(
                (1, 1, sequence_length, sequence_length), dtype=bool
            ).cuda()
            # thd input shape: [sequence length * batch size, 1, hidden size]
            hidden_states_sbhd = hidden_states_sbhd.cuda().bfloat16()
            hidden_states_thd = hidden_states_sbhd.transpose(0, 1).contiguous()
            hidden_states_thd = hidden_states_thd.view(
                -1, 1, self.parallel_attention.config.hidden_size
            )
            attention_mask_thd = None
            packed_seq_params = make_test_packed_seq_params(cu_seqlens=cu_seqlens)

            # fine-grained check
            query_sbhd, key_sbhd, value_sbhd, q_compressed_sbhd, kv_compressed_sbhd = (
                self.parallel_attention.get_query_key_value_tensors(
                    hidden_states_sbhd, None, None, None, None
                )
            )
            query_thd, key_thd, value_thd, q_compressed_thd, kv_compressed_thd = (
                self.parallel_attention.get_query_key_value_tensors(
                    hidden_states_thd, None, None, packed_seq_params, None
                )
            )
            _query_sbhd = query_sbhd.transpose(0, 1).contiguous().view(*query_thd.shape)
            _key_sbhd = key_sbhd.transpose(0, 1).contiguous().view(*key_thd.shape)
            _value_sbhd = value_sbhd.transpose(0, 1).contiguous().view(*value_thd.shape)
            _q_compressed_sbhd = (
                q_compressed_sbhd.transpose(0, 1).contiguous().view(*q_compressed_thd.shape)
            )
            _kv_compressed_sbhd = (
                kv_compressed_sbhd.transpose(0, 1).contiguous().view(*kv_compressed_thd.shape)
            )
            assert torch.equal(_query_sbhd, query_thd)
            assert torch.equal(_key_sbhd, key_thd)
            assert torch.equal(_value_sbhd, value_thd)
            assert torch.equal(_q_compressed_sbhd, q_compressed_thd)
            assert torch.equal(_kv_compressed_sbhd, kv_compressed_thd)

            core_attn_out_sbhd = self.parallel_attention._run_core_attention(
                query_sbhd, key_sbhd, value_sbhd, attention_mask_sbhd, packed_seq_params=None
            )
            query_thd = query_thd.squeeze(1)
            key_thd = key_thd.squeeze(1)
            value_thd = value_thd.squeeze(1)
            core_attn_out_thd = self.parallel_attention._run_core_attention(
                query_thd,
                key_thd,
                value_thd,
                attention_mask_thd,
                packed_seq_params=packed_seq_params,
            )
            core_attn_out_thd = core_attn_out_thd.reshape(core_attn_out_thd.size(0), 1, -1)
            _core_attn_out_sbhd = (
                core_attn_out_sbhd.transpose(0, 1).contiguous().view(*core_attn_out_thd.shape)
            )
            assert torch.equal(_core_attn_out_sbhd, core_attn_out_thd)

            output_sbhd, bias_sbhd = apply_module(self.parallel_attention.linear_proj)(
                core_attn_out_sbhd
            )
            output_thd, bias_thd = apply_module(self.parallel_attention.linear_proj)(
                core_attn_out_thd
            )
            _output_sbhd = output_sbhd.transpose(0, 1).contiguous().view(*output_thd.shape)
            assert torch.equal(_output_sbhd, output_thd)

            output_thd_fine_grained = output_thd
            bias_thd_fine_grained = bias_thd

            # E2E check
            # sbhd
            output_sbhd, bias_sbhd = self.parallel_attention(
                hidden_states_sbhd, attention_mask_sbhd
            )
            # thd
            output_thd, bias_thd = self.parallel_attention(
                hidden_states_thd, attention_mask_thd, packed_seq_params=packed_seq_params
            )
            _output_sbhd = output_sbhd.transpose(0, 1).contiguous().view(*output_thd.shape)
            assert torch.equal(_output_sbhd, output_thd)
            assert bias_thd.shape == bias_sbhd.shape
            assert torch.equal(bias_sbhd, bias_thd)

            assert torch.equal(output_thd, output_thd_fine_grained)
            assert torch.equal(bias_thd, bias_thd_fine_grained)

            os.environ.clear()
            os.environ.update(_environ)


@pytest.mark.skipif(not is_te_min_version("2.9.0"), reason="QK clipping requires TE >= 2.9.0")
@pytest.mark.parametrize("rope_type", ('yarn', 'rope'))
class TestMLAClipQK:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, rope_type):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rope_type=rope_type,
            rotary_base=10000,
            original_max_position_embeddings=32,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_clip_qk_disabled_raises_error(self):
        """Test that clip_qk raises ValueError when qk_clip is not enabled."""
        if is_te_min_version("1.10.0"):
            # Create config without qk_clip
            config = MLATransformerConfig(
                num_layers=2,
                hidden_size=12,
                num_attention_heads=4,
                use_cpu_initialization=True,
                q_lora_rank=32,
                kv_lora_rank=32,
                qk_head_dim=128,
                v_head_dim=128,
                qk_pos_emb_head_dim=64,
                rotary_base=10000,
                original_max_position_embeddings=32,
                qk_clip=False,
            )
            attention = MLASelfAttention(
                config,
                get_mla_self_attn_submodules(),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
            )

            with pytest.raises(ValueError, match="qk_clip option needs to be enabled"):
                attention.clip_qk()

    def test_clip_qk_none_logits_raises_error(self):
        """Test that clip_qk raises ValueError when current_max_attn_logits is None."""
        if is_te_min_version("1.10.0"):
            attention = MLASelfAttention(
                self.transformer_config,
                get_mla_self_attn_submodules(),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
            )

            with pytest.raises(ValueError, match="current_max_attn_logits is None"):
                attention.clip_qk()

    def test_clip_qk_below_threshold_no_update(self):
        """Test that weights are not updated when max logits are below threshold."""
        if not is_te_min_version("1.10.0"):
            pytest.skip("MLA requires TransformerEngine >= 1.10.0")

        attention = MLASelfAttention(
            self.transformer_config,
            get_mla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        attention.cuda()

        # Save original weights
        if self.transformer_config.q_lora_rank is None:
            original_q_weight = attention.linear_q_proj.weight.data.clone()
        else:
            original_q_weight = attention.linear_q_up_proj.weight.data.clone()
        original_kv_weight = attention.linear_kv_up_proj.weight.data.clone()

        # Set current_max_attn_logits below threshold
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [50.0, 60.0, 70.0, 80.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should not be updated
        if self.transformer_config.q_lora_rank is None:
            assert torch.equal(attention.linear_q_proj.weight.data, original_q_weight)
        else:
            assert torch.equal(attention.linear_q_up_proj.weight.data, original_q_weight)
        assert torch.equal(attention.linear_kv_up_proj.weight.data, original_kv_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None

    def test_clip_qk_above_threshold_updates_weights(self):
        """Test that weights are updated when max logits exceed threshold."""
        if not is_te_min_version("1.10.0"):
            pytest.skip("MLA requires TransformerEngine >= 1.10.0")

        attention = MLASelfAttention(
            self.transformer_config,
            get_mla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        attention.cuda()

        # Save original weights
        if self.transformer_config.q_lora_rank is None:
            original_q_weight = attention.linear_q_proj.weight.data.clone()
        else:
            original_q_weight = attention.linear_q_up_proj.weight.data.clone()
        original_kv_weight = attention.linear_kv_up_proj.weight.data.clone()

        # Set current_max_attn_logits above threshold
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [150.0, 160.0, 170.0, 180.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should be updated
        if self.transformer_config.q_lora_rank is None:
            assert not torch.equal(attention.linear_q_proj.weight.data, original_q_weight)
        else:
            assert not torch.equal(attention.linear_q_up_proj.weight.data, original_q_weight)
        assert not torch.equal(attention.linear_kv_up_proj.weight.data, original_kv_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None

    def test_clip_qk_mixed_logits(self):
        """Test clip_qk with mixed logits (some above, some below threshold)."""
        if not is_te_min_version("1.10.0"):
            pytest.skip("MLA requires TransformerEngine >= 1.10.0")

        attention = MLASelfAttention(
            self.transformer_config,
            get_mla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        attention.cuda()

        # Save original weights
        if self.transformer_config.q_lora_rank is None:
            original_q_weight = attention.linear_q_proj.weight.data.clone()
        else:
            original_q_weight = attention.linear_q_up_proj.weight.data.clone()
        original_kv_weight = attention.linear_kv_up_proj.weight.data.clone()

        # Set mixed current_max_attn_logits (some above, some below threshold)
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [80.0, 150.0, 90.0, 200.0], device='cuda'
        )

        # Call clip_qk
        attention.clip_qk()

        # Weights should be updated since at least one head exceeds threshold
        if self.transformer_config.q_lora_rank is None:
            assert not torch.equal(attention.linear_q_proj.weight.data, original_q_weight)
        else:
            assert not torch.equal(attention.linear_q_up_proj.weight.data, original_q_weight)
        assert not torch.equal(attention.linear_kv_up_proj.weight.data, original_kv_weight)
        # current_max_attn_logits should be reset
        assert attention.core_attention.current_max_attn_logits is None

    def test_clip_qk_with_absorption_raises_error(self):
        """Test that clip_qk raises ValueError when in absorption mode."""
        if not is_te_min_version("1.10.0"):
            pytest.skip("MLA requires TransformerEngine >= 1.10.0")

        # Create config with cache_mla_latents enabled
        config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rotary_base=10000,
            original_max_position_embeddings=32,
            qk_clip=True,
            qk_clip_threshold=100.0,
            qk_clip_alpha=0.5,
        )
        attention = MLASelfAttention(
            config,
            get_mla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        attention.cuda()

        # Simulate absorption mode by setting cache_mla_latents and deleting linear_kv_up_proj
        attention.cache_mla_latents = True
        if hasattr(attention, 'linear_kv_up_proj'):
            delattr(attention, 'linear_kv_up_proj')

        # Set current_max_attn_logits
        attention.core_attention.current_max_attn_logits = torch.tensor(
            [150.0, 160.0, 170.0, 180.0], device='cuda'
        )

        with pytest.raises(
            ValueError,
            match="qk_clip is not supported when cache_mla_latents is enabled and absorption is active",
        ):
            attention.clip_qk()


@pytest.mark.experimental
@pytest.mark.parametrize(
    ("rope_type", "apply_rope_fusion"),
    [("rope", False), ("rope", True), ("yarn", False), ("yarn", True)],
)
@pytest.mark.parametrize(
    ("tp", "sp", "cp", "output_gate", "gate_granularity"),
    [
        (4, False, 1, False, "elementwise"),  # TP w/o SP
        (4, True, 1, False, "elementwise"),  # TP w/ SP
        (1, False, 4, False, "elementwise"),  # CP
        (2, False, 2, False, "elementwise"),  # CP + TP w/o SP
        (2, True, 2, False, "elementwise"),  # CP + TP w/ SP
        (4, True, 1, True, "elementwise"),  # Elementwise gate with TP + SP
        (4, True, 1, True, "headwise"),  # Headwise gate with TP + SP
        (1, False, 2, True, "elementwise"),  # Elementwise gate with CP
        (1, False, 2, True, "headwise"),  # Headwise gate with CP
    ],
)
@pytest.mark.skipif(not is_te_min_version("1.10.0"), reason="Requires TransformerEngine >= 1.10.0")
def test_parallel_multi_latent_attention_correctness(
    tmp_path_dist_ckpt, rope_type, apply_rope_fusion, tp, sp, cp, output_gate, gate_granularity
):
    if output_gate and (rope_type != "yarn" or apply_rope_fusion):
        pytest.skip("Gated MLA parallel coverage uses one representative YARN configuration.")
    if cp > 1 and not is_te_min_version("2.5.0", check_equality=True):
        pytest.skip("MLA CP requires TransformerEngine >= 2.5.0")
    if rope_type == "yarn" and apply_rope_fusion and not is_torch_min_version("2.5.0"):
        pytest.skip("MLA yarn rope fusion requires PyTorch >= 2.5.0")
    if (
        cp > 1
        and rope_type == "yarn"
        and apply_rope_fusion
        and not is_te_min_version("2.6.0", check_equality=True)
    ):
        pytest.skip("MLA CP + yarn rope fusion requires PyTorch >= 2.6.0")

    # Non-deterministic mode has bug to be fixed with MLA
    _environ = os.environ.copy()
    os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = "1"
    os.environ['NVTE_FUSED_ATTN'] = "1"
    os.environ['NVTE_FLASH_ATTN'] = "0"

    # Constants
    seed = 123
    sequence_length = 256
    micro_batch_size = 4
    hidden_size = 128

    # Model initialization function
    def initialize_gpt_model(
        pre_process=True, post_process=True, vp_stage=None, pg_collection=None, config=None
    ):
        layer_spec = get_gpt_layer_with_transformer_engine_spec(multi_latent_attention=True)
        gpt_model = GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=128,
            max_sequence_length=sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        return gpt_model

    # Initialize baseline parallel state
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )

    # Initialize input hidden states
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    input_hidden_states = (
        torch.rand((sequence_length, micro_batch_size, hidden_size))
        .cuda()
        .bfloat16()
        .requires_grad_(True)
    )

    # Initialize transformer config
    transformer_config = MLATransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=128,
        v_head_dim=128,
        qk_pos_emb_head_dim=64,
        rotary_base=10000,
        original_max_position_embeddings=64,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        bf16=True,
        rope_type=rope_type,
        apply_rope_fusion=apply_rope_fusion,
        attention_output_gate=output_gate,
        gated_attention_proj_granularity=gate_granularity,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    with TempNamedDir(tmp_path_dist_ckpt / 'test_parallel_mla', sync=True) as ckpt_dir:
        # Set argument
        mock_args = parse_args(ignore_unknown_args=True)
        set_args(mock_args)

        # Initialize baseline model
        init_basic_mock_args(mock_args, 1, 1, bf16=True)
        mock_args.context_parallel_size = 1
        mock_args.sequence_parallel = 1
        gpt_model = unwrap_model(get_model(initialize_gpt_model, config=transformer_config))

        # Initialize args and save checkpoint
        init_checkpointing_mock_args(mock_args, ckpt_dir, False)
        mock_args.no_save_optim = True
        mock_args.no_save_rng = True
        mock_args.no_load_optim = True
        mock_args.no_load_rng = True
        save_checkpoint(10, gpt_model, None, None, 0)

        # Calculate baseline output
        attention = gpt_model[0].decoder.layers[0].self_attention
        output_hidden_states_baseline, bias_hidden_states_baseline = attention(
            input_hidden_states, attention_mask=None
        )
        output_hidden_states_baseline.sum().backward()

        # Save baseline output
        input_grad_baseline = input_hidden_states.grad.detach()
        output_hidden_states_baseline = output_hidden_states_baseline.detach()
        bias_hidden_states_baseline = bias_hidden_states_baseline.detach()

        # Initialize parallel model
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=1, context_parallel_size=cp
        )
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        transformer_config.context_parallel_size = cp
        transformer_config.tensor_model_parallel_size = tp
        transformer_config.sequence_parallel = sp
        init_basic_mock_args(mock_args, tp, 1, bf16=True)
        mock_args.context_parallel_size = cp
        mock_args.sequence_parallel = sp
        gpt_model = unwrap_model(get_model(initialize_gpt_model, config=transformer_config))
        with mock.patch('megatron.training.checkpointing.check_checkpoint_args'):
            with mock.patch('megatron.training.checkpointing.update_num_microbatches'):
                load_checkpoint(gpt_model, None, None)

        # Function to get tensor on this tp and cp rank
        cp_group = parallel_state.get_context_parallel_group()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        def get_tensor_on_this_rank(tensor):
            if cp > 1:
                tensor = get_tensor_on_this_cp_rank(tensor, 0, cp_group)
            if tp > 1 and sp:
                sp_seg = sequence_length // tp // cp
                tensor = tensor[tp_rank * sp_seg : (tp_rank + 1) * sp_seg]
            return tensor

        # Calculate parallel model output
        input_hidden_states = get_tensor_on_this_rank(input_hidden_states)
        input_hidden_states = input_hidden_states.detach().requires_grad_(True)
        parallel_attention = gpt_model[0].decoder.layers[0].self_attention
        output_hidden_states_parallel, bias_hidden_states_parallel = parallel_attention(
            input_hidden_states, attention_mask=None
        )
        output_hidden_states_parallel.sum().backward()
        input_grad_parallel = input_hidden_states.grad.detach()

        # Check if the output is the same
        if cp:
            atol, rtol = 5e-3, 5e-3
        else:
            atol, rtol = 5e-4, 5e-4
        output_hidden_states_baseline = get_tensor_on_this_rank(output_hidden_states_baseline)
        input_grad_baseline = get_tensor_on_this_rank(input_grad_baseline)

        assert torch.all(
            ~torch.isnan(output_hidden_states_baseline)
        ), "output_hidden_states_baseline contains nan"
        assert torch.all(
            ~torch.isinf(output_hidden_states_baseline)
        ), "output_hidden_states_baseline contains inf"
        assert torch.all(
            ~torch.isnan(bias_hidden_states_baseline)
        ), "bias_hidden_states_baseline contains nan"
        assert torch.all(
            ~torch.isinf(bias_hidden_states_baseline)
        ), "bias_hidden_states_baseline contains inf"
        assert torch.all(~torch.isnan(input_grad_baseline)), "input_grad_baseline contains nan"
        assert torch.all(~torch.isinf(input_grad_baseline)), "input_grad_baseline contains inf"
        assert torch.all(
            ~torch.isnan(output_hidden_states_parallel)
        ), "output_hidden_states_parallel contains nan"
        assert torch.all(
            ~torch.isinf(output_hidden_states_parallel)
        ), "output_hidden_states_parallel contains inf"
        assert torch.all(
            ~torch.isnan(bias_hidden_states_parallel)
        ), "bias_hidden_states_parallel contains nan"
        assert torch.all(
            ~torch.isinf(bias_hidden_states_parallel)
        ), "bias_hidden_states_parallel contains inf"
        assert torch.all(~torch.isnan(input_grad_parallel)), "input_grad_parallel contains nan"
        assert torch.all(~torch.isinf(input_grad_parallel)), "input_grad_parallel contains inf"

        torch.testing.assert_close(
            output_hidden_states_baseline,
            output_hidden_states_parallel,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"Mismatch in output_hidden_states: {msg}",
        )
        torch.testing.assert_close(
            bias_hidden_states_baseline,
            bias_hidden_states_parallel,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"Mismatch in bias_hidden_states: {msg}",
        )
        torch.testing.assert_close(
            input_grad_baseline,
            input_grad_parallel,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"Mismatch in input_grad: {msg}",
        )

        Utils.destroy_model_parallel()

    os.environ.clear()
    os.environ.update(_environ)


@pytest.mark.parametrize("rope_type", ('yarn', 'rope'))
class TestFusedMLASelfAttention:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, rope_type):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rope_type=rope_type,
            rotary_base=10000,
            original_max_position_embeddings=32,
        )
        self.fused_attention = FusedMLASelfAttention(
            self.transformer_config,
            get_fused_mla_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.fused_attention, FusedMLASelfAttention)
        assert isinstance(self.fused_attention, MLASelfAttention)
        assert self.fused_attention.layer_number == 1
        assert hasattr(self.fused_attention, 'linear_qkv_down_proj')

    def test_attention_latent_norm_epsilon_on_fused_projections(self):
        config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            qk_layernorm=True,
            layernorm_epsilon=1.0e-5,
            attention_latent_norm_epsilon=1.0e-6,
        )
        attention = FusedMLASelfAttention(
            config,
            get_fused_mla_submodules(qk_layernorm=True),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

        assert attention.linear_q_up_proj.eps == pytest.approx(1.0e-6)
        assert attention.linear_kv_up_proj.eps == pytest.approx(1.0e-6)

    def test_fused_weight_shape(self):
        config = self.transformer_config
        expected_out = config.q_lora_rank + config.kv_lora_rank + config.qk_pos_emb_head_dim
        weight = self.fused_attention.linear_qkv_down_proj.weight
        assert weight.shape[0] == expected_out
        assert weight.shape[1] == config.hidden_size

    def test_qkv_down_projection_split(self):
        if not is_te_min_version("1.10.0"):
            pytest.skip("Requires TE >= 1.10.0")
        config = self.transformer_config
        self.fused_attention.cuda()

        seq_len, batch = 16, 2
        hidden = torch.randn(seq_len, batch, config.hidden_size).cuda()
        q_compressed, kv_combined = self.fused_attention._qkv_down_projection(hidden)

        assert q_compressed.shape == (seq_len, batch, config.q_lora_rank)
        assert kv_combined.shape == (
            seq_len,
            batch,
            config.kv_lora_rank + config.qk_pos_emb_head_dim,
        )

    def test_gpu_forward(self):
        if not is_te_min_version("1.10.0"):
            pytest.skip("Requires TE >= 1.10.0")

        config = self.fused_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.fused_attention.cuda()

        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size)).cuda()
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        output, bias = self.fused_attention(hidden_states, attention_mask)

        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def test_gpu_forward_bf16(self):
        if not is_te_min_version("1.10.0"):
            pytest.skip("Requires TE >= 1.10.0")

        config = self.fused_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.fused_attention.cuda().bfloat16()

        hidden_states = (
            torch.ones((sequence_length, micro_batch_size, config.hidden_size)).cuda().bfloat16()
        )
        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        output, bias = self.fused_attention(hidden_states, attention_mask)

        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert output.dtype == torch.bfloat16


class TestFusedMLAGradientFlow:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rope_type="rope",
            rotary_base=10000,
            original_max_position_embeddings=32,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_backward_pass(self):
        if not is_te_min_version("1.10.0"):
            pytest.skip("Requires TE >= 1.10.0")

        config = self.transformer_config
        fused = FusedMLASelfAttention(
            config, get_fused_mla_submodules(), layer_number=1, attn_mask_type=AttnMaskType.causal
        )
        fused.cuda()

        seq_len, batch = 32, 2
        hidden_states = torch.randn(
            seq_len, batch, config.hidden_size, device='cuda', requires_grad=True
        )
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device='cuda')

        output, bias = fused(hidden_states, attention_mask)
        loss = output.sum()
        loss.backward()

        assert fused.linear_qkv_down_proj.weight.grad is not None
        assert (
            fused.linear_qkv_down_proj.weight.grad.shape == fused.linear_qkv_down_proj.weight.shape
        )
        assert hidden_states.grad is not None


def test_fused_mla_training_hooks_use_fused_down_projection(monkeypatch):
    """Training hooks should use fused q/kv down projection attributes."""

    class LinearWithDelayedWgrad:
        def __init__(self, name):
            self.name = name

        def backward_dw(self):
            calls.append(self.name)

    calls = []
    fused = FusedMLASelfAttention.__new__(FusedMLASelfAttention)
    fused.linear_kv_up_proj = LinearWithDelayedWgrad("kv_up")
    fused.linear_qkv_down_proj = LinearWithDelayedWgrad("qkv_down")
    fused.linear_q_up_proj = LinearWithDelayedWgrad("q_up")
    fused.linear_proj = LinearWithDelayedWgrad("out")

    fused.backward_dw()

    assert calls == ["kv_up", "qkv_down", "q_up", "out"]

    saved_inputs = []
    mla_module = __import__(FusedMLASelfAttention.__module__, fromlist=["set_save_original_input"])
    monkeypatch.setattr(mla_module, "set_save_original_input", saved_inputs.append)

    fused.set_for_recompute_input_layernorm()

    assert saved_inputs == [fused.linear_qkv_down_proj]


class TestFusedMLALoadFromStateDict:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=32,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rope_type="rope",
            rotary_base=10000,
            original_max_position_embeddings=32,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_load_unfused_state_dict(self):
        if not is_te_min_version("1.10.0"):
            pytest.skip("Requires TE >= 1.10.0")

        unfused = MLASelfAttention(
            self.transformer_config,
            get_mla_self_attn_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )
        fused = FusedMLASelfAttention(
            self.transformer_config,
            get_fused_mla_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

        unfused_sd = unfused.state_dict()

        q_down_keys = [k for k in unfused_sd if 'linear_q_down_proj' in k]
        kv_down_keys = [k for k in unfused_sd if 'linear_kv_down_proj' in k]
        assert len(q_down_keys) > 0, "Expected q_down_proj keys in unfused state dict"
        assert len(kv_down_keys) > 0, "Expected kv_down_proj keys in unfused state dict"

        fused.load_state_dict(unfused_sd, strict=False)

        config = self.transformer_config
        expected_out = config.q_lora_rank + config.kv_lora_rank + config.qk_pos_emb_head_dim
        assert fused.linear_qkv_down_proj.weight.shape[0] == expected_out

        q_w = unfused_sd['linear_q_down_proj.weight']
        kv_w = unfused_sd['linear_kv_down_proj.weight']
        expected_fused = torch.cat([q_w, kv_w], dim=0)
        torch.testing.assert_close(fused.linear_qkv_down_proj.weight.data, expected_fused)

    def test_sharded_state_dict_splits_back(self):
        if not is_te_min_version("1.10.0"):
            pytest.skip("Requires TE >= 1.10.0")

        fused = FusedMLASelfAttention(
            self.transformer_config,
            get_fused_mla_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

        sharded_sd = fused.sharded_state_dict(prefix="")
        assert any(
            'linear_q_down_proj.weight' in k for k in sharded_sd
        ), f"Expected linear_q_down_proj.weight in sharded state dict, got keys: {list(sharded_sd.keys())}"
        assert any(
            'linear_kv_down_proj.weight' in k for k in sharded_sd
        ), f"Expected linear_kv_down_proj.weight in sharded state dict, got keys: {list(sharded_sd.keys())}"
        assert not any(
            'linear_qkv_down_proj.weight' in k for k in sharded_sd
        ), f"Unexpected linear_qkv_down_proj.weight in sharded state dict"


class TestFusedMLARequiresQLora:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_raises_without_q_lora_rank(self):
        config = MLATransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            q_lora_rank=None,
            kv_lora_rank=32,
            qk_head_dim=128,
            v_head_dim=128,
            qk_pos_emb_head_dim=64,
            rope_type="rope",
            rotary_base=10000,
            original_max_position_embeddings=32,
        )
        with pytest.raises(AssertionError, match="q_lora_rank"):
            FusedMLASelfAttention(
                config,
                get_fused_mla_submodules(),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
            )
