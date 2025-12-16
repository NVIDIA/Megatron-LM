# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
from functools import partial
from importlib.metadata import version
from inspect import signature
from unittest import mock

import pytest
import torch
import transformer_engine as te

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank as get_tensor_on_this_cp_rank,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import MLASelfAttention, MultiLatentAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import is_te_min_version, is_torch_min_version
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from megatron.training.utils import unwrap_model
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


def get_mla_self_attn_submodules(linear_qkv_down_proj=None):
    submodules = get_gpt_layer_with_transformer_engine_spec(
        multi_latent_attention=True
    ).submodules.self_attention.submodules
    if linear_qkv_down_proj is not None:
        submodules.linear_q_down_proj = linear_qkv_down_proj
        submodules.linear_kv_down_proj = linear_qkv_down_proj
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
            query, key, value = self.parallel_attention.get_query_key_value_tensors(
                hidden_states, None, None, packed_seq_params, None
            )

            assert query is not None
            assert key is not None
            assert value is not None
            assert query.is_contiguous()
            assert key.is_contiguous()
            assert value.is_contiguous()

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

            q, k, v = checkpointed_parallel_attention.get_query_key_value_tensors(hidden_states)
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
    (
        ('rope', False),
        ('yarn', False),
        ('yarn', True),  # apply_rope_fusion for MLA only works with YARN RoPE.
    ),
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
            query_sbhd, key_sbhd, value_sbhd = self.parallel_attention.get_query_key_value_tensors(
                hidden_states_sbhd, None, None, None, None
            )
            query_thd, key_thd, value_thd = self.parallel_attention.get_query_key_value_tensors(
                hidden_states_thd, None, None, packed_seq_params, None
            )
            _query_sbhd = query_sbhd.transpose(0, 1).contiguous().view(*query_thd.shape)
            _key_sbhd = key_sbhd.transpose(0, 1).contiguous().view(*key_thd.shape)
            _value_sbhd = value_sbhd.transpose(0, 1).contiguous().view(*value_thd.shape)
            assert torch.equal(_query_sbhd, query_thd)
            assert torch.equal(_key_sbhd, key_thd)
            assert torch.equal(_value_sbhd, value_thd)

            core_attn_out_sbhd = self.parallel_attention.core_attention(
                query_sbhd,
                key_sbhd,
                value_sbhd,
                attention_mask_sbhd,
                packed_seq_params=None,
                attn_mask_type=self.parallel_attention.attn_mask_type,
            )
            query_thd = query_thd.squeeze(1)
            key_thd = key_thd.squeeze(1)
            value_thd = value_thd.squeeze(1)
            core_attn_out_thd = self.parallel_attention.core_attention(
                query_thd,
                key_thd,
                value_thd,
                attention_mask_thd,
                packed_seq_params=packed_seq_params,
                attn_mask_type=self.parallel_attention.attn_mask_type,
            )
            core_attn_out_thd = core_attn_out_thd.reshape(core_attn_out_thd.size(0), 1, -1)
            _core_attn_out_sbhd = (
                core_attn_out_sbhd.transpose(0, 1).contiguous().view(*core_attn_out_thd.shape)
            )
            assert torch.equal(_core_attn_out_sbhd, core_attn_out_thd)

            output_sbhd, bias_sbhd = self.parallel_attention.linear_proj(core_attn_out_sbhd)
            output_thd, bias_thd = self.parallel_attention.linear_proj(core_attn_out_thd)
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
    (
        ('rope', False),
        ('yarn', False),
        ('yarn', True),  # apply_rope_fusion for MLA only works with YARN RoPE.
    ),
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
            query_sbhd, key_sbhd, value_sbhd = self.parallel_attention.get_query_key_value_tensors(
                hidden_states_sbhd, None, None, None, None
            )
            query_thd, key_thd, value_thd = self.parallel_attention.get_query_key_value_tensors(
                hidden_states_thd, None, None, packed_seq_params, None
            )
            _query_sbhd = query_sbhd.transpose(0, 1).contiguous().view(*query_thd.shape)
            _key_sbhd = key_sbhd.transpose(0, 1).contiguous().view(*key_thd.shape)
            _value_sbhd = value_sbhd.transpose(0, 1).contiguous().view(*value_thd.shape)
            torch.testing.assert_close(_query_sbhd, query_thd, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(_key_sbhd, key_thd, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(_value_sbhd, value_thd, atol=1e-6, rtol=1e-6)

            core_attn_out_sbhd = self.parallel_attention.core_attention(
                query_sbhd,
                key_sbhd,
                value_sbhd,
                attention_mask_sbhd,
                packed_seq_params=None,
                attn_mask_type=self.parallel_attention.attn_mask_type,
            )
            query_thd = query_thd.squeeze(1)
            key_thd = key_thd.squeeze(1)
            value_thd = value_thd.squeeze(1)
            core_attn_out_thd = self.parallel_attention.core_attention(
                query_thd,
                key_thd,
                value_thd,
                attention_mask_thd,
                packed_seq_params=packed_seq_params,
                attn_mask_type=self.parallel_attention.attn_mask_type,
            )
            core_attn_out_thd = core_attn_out_thd.reshape(core_attn_out_thd.size(0), 1, -1)
            _core_attn_out_sbhd = (
                core_attn_out_sbhd.transpose(0, 1).contiguous().view(*core_attn_out_thd.shape)
            )
            torch.testing.assert_close(_core_attn_out_sbhd, core_attn_out_thd, atol=atol, rtol=rtol)

            output_sbhd, bias_sbhd = self.parallel_attention.linear_proj(core_attn_out_sbhd)
            output_thd, bias_thd = self.parallel_attention.linear_proj(core_attn_out_thd)
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
            query_sbhd, key_sbhd, value_sbhd = self.parallel_attention.get_query_key_value_tensors(
                hidden_states_sbhd, None, None, None, None
            )
            query_thd, key_thd, value_thd = self.parallel_attention.get_query_key_value_tensors(
                hidden_states_thd, None, None, packed_seq_params, None
            )
            _query_sbhd = query_sbhd.transpose(0, 1).contiguous().view(*query_thd.shape)
            _key_sbhd = key_sbhd.transpose(0, 1).contiguous().view(*key_thd.shape)
            _value_sbhd = value_sbhd.transpose(0, 1).contiguous().view(*value_thd.shape)
            assert torch.equal(_query_sbhd, query_thd)
            assert torch.equal(_key_sbhd, key_thd)
            assert torch.equal(_value_sbhd, value_thd)

            core_attn_out_sbhd = self.parallel_attention.core_attention(
                query_sbhd,
                key_sbhd,
                value_sbhd,
                attention_mask_sbhd,
                packed_seq_params=None,
                attn_mask_type=self.parallel_attention.attn_mask_type,
            )
            query_thd = query_thd.squeeze(1)
            key_thd = key_thd.squeeze(1)
            value_thd = value_thd.squeeze(1)
            core_attn_out_thd = self.parallel_attention.core_attention(
                query_thd,
                key_thd,
                value_thd,
                attention_mask_thd,
                packed_seq_params=packed_seq_params,
                attn_mask_type=self.parallel_attention.attn_mask_type,
            )
            core_attn_out_thd = core_attn_out_thd.reshape(core_attn_out_thd.size(0), 1, -1)
            _core_attn_out_sbhd = (
                core_attn_out_sbhd.transpose(0, 1).contiguous().view(*core_attn_out_thd.shape)
            )
            assert torch.equal(_core_attn_out_sbhd, core_attn_out_thd)

            output_sbhd, bias_sbhd = self.parallel_attention.linear_proj(core_attn_out_sbhd)
            output_thd, bias_thd = self.parallel_attention.linear_proj(core_attn_out_thd)
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
    [
        ("rope", False),
        ("yarn", False),
        ("yarn", True),  # apply_rope_fusion for MLA only works with YARN RoPE.
    ],
)
@pytest.mark.parametrize(
    ("tp", "sp", "cp"),
    [
        (4, False, 1),  # TP w/o SP
        (4, True, 1),  # TP w/ SP
        (1, False, 4),  # CP
        (2, False, 2),  # CP + TP w/o SP
        (2, True, 2),  # CP + TP w/ SP
    ],
)
@pytest.mark.skipif(not is_te_min_version("1.10.0"), reason="Requires TransformerEngine >= 1.10.0")
def test_parallel_multi_latent_attention_correctness(
    tmp_path_dist_ckpt, rope_type, apply_rope_fusion, tp, sp, cp
):
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
