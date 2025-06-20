# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
from importlib.metadata import version
from inspect import signature

import pytest
import torch
import transformer_engine as te

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import MLASelfAttention, MultiLatentAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import is_te_min_version, is_torch_min_version
from tests.unit_tests.test_utilities import Utils


def make_test_packed_seq_params(sequence_length=None, cu_seqlens=None):
    if cu_seqlens is None:
        assert sequence_length is not None
        cu_seqlens = [0, 6, 19, 22, sequence_length]
    cu_seqlens = torch.IntTensor(cu_seqlens).cuda()
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen, _ = seqlens.max(dim=0, keepdim=True)
    max_seqlen = max_seqlen.tolist()[0]
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )
    return packed_seq_params


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
            get_gpt_layer_with_transformer_engine_spec(
                multi_latent_attention=True
            ).submodules.self_attention.submodules,
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
                get_gpt_layer_with_transformer_engine_spec(
                    multi_latent_attention=True
                ).submodules.self_attention.submodules,
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

    def test_checkpointed_gpu_forward(self):
        if is_te_min_version("1.10.0"):
            transformer_config = self.transformer_config
            transformer_config.recompute_granularity = 'selective'
            checkpointed_parallel_attention = MLASelfAttention(
                transformer_config,
                get_gpt_layer_with_transformer_engine_spec(
                    multi_latent_attention=True
                ).submodules.self_attention.submodules,
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
                get_gpt_layer_with_transformer_engine_spec(
                    multi_latent_attention=True
                ).submodules.self_attention.submodules,
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


class TestSequenceParallelMLAAttention:

    def setup_method(self, method):
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
            get_gpt_layer_with_transformer_engine_spec(
                multi_latent_attention=True
            ).submodules.self_attention.submodules,
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


class TestTensorParallelMLAAttention:
    def setup_method(self, method):
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
            get_gpt_layer_with_transformer_engine_spec(
                multi_latent_attention=True
            ).submodules.self_attention.submodules,
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


@pytest.mark.parametrize("rope_type", ('yarn', 'rope'))
class TestParallelMLAAttentionPrecision:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, rope_type):
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
            get_gpt_layer_with_transformer_engine_spec(
                multi_latent_attention=True
            ).submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward_thd_precision(self):
        if is_te_min_version("1.10.0"):
            # use flash attention for hopper, future may support fused attention for ampere
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


@pytest.mark.experimental
@pytest.mark.skipif(not is_torch_min_version("2.5.0"), reason="Requires PyTorch >= 2.5.0")
class TestParallelMLAAttentionPrecisionWithRopeFusion:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
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
            max_position_embeddings=32,
            deterministic_mode=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            apply_rope_fusion=True,
        )
        self.parallel_attention = MLASelfAttention(
            self.transformer_config,
            get_gpt_layer_with_transformer_engine_spec(
                multi_latent_attention=True
            ).submodules.self_attention.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_gpu_forward_thd_precision(self):
        if is_te_min_version("1.10.0"):
            # use flash attention for hopper, future may support fused attention for ampere
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
