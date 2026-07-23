# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os

import pytest
import torch
from packaging.version import Version as PkgVersion

from megatron.core.models.bert.bert_layer_specs import (
    bert_layer_local_spec,
    get_bert_layer_with_transformer_engine_spec,
    get_bert_layer_with_transformer_engine_submodules,
)
from megatron.core.models.bert.bert_model import BertModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


class TestBertModel:

    def setup_method(self, method):
        tp = 1
        pp = 1
        Utils.initialize_model_parallel(tp, pp)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            perform_initialization=True,
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            pipeline_dtype=torch.bfloat16,
            attention_backend=AttnBackend.unfused,
        )
        self.bert_model = BertModel(
            config=transformer_config,
            num_tokentypes=0,
            transformer_layer_spec=get_bert_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor(self):
        assert isinstance(self.bert_model, BertModel)

        assert self.bert_model.max_sequence_length == 4

        num_weights = sum([p.numel() for p in self.bert_model.parameters()])
        assert num_weights == 6702

    @pytest.mark.internal
    def test_set_input_tensor(self):
        config: TransformerConfig = self.bert_model.config
        sequence_length = self.bert_model.max_sequence_length
        micro_batch_size = 2

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.bert_model.set_input_tensor(input_tensor)

        assert self.bert_model.encoder.input_tensor.shape[0] == sequence_length
        assert self.bert_model.encoder.input_tensor.shape[1] == micro_batch_size
        assert self.bert_model.encoder.input_tensor.shape[2] == config.hidden_size

    @pytest.mark.internal
    def test_post_process_forward(self):
        config: TransformerConfig = self.bert_model.config
        sequence_length = self.bert_model.max_sequence_length
        micro_batch_size = 2

        self.bert_model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones((micro_batch_size, sequence_length), dtype=bool).cuda()

        logits = self.bert_model.forward(input_ids=input_ids, attention_mask=attention_mask)

        assert logits[0].shape[0] == micro_batch_size
        assert logits[0].shape[1] == sequence_length
        assert logits[0].shape[2] == self.bert_model.vocab_size

    @pytest.mark.internal
    def test_qk_layernorm_submodules_are_none(self):
        # The TE BERT spec leaves q_layernorm/k_layernorm unset (None) instead of hardcoding
        # IdentityOp, so that TransformerConfig.qk_layernorm can select the default TENorm
        # through the shared SelfAttention fallback (`submodules.q_layernorm or TENorm`).
        spec = get_bert_layer_with_transformer_engine_spec()
        assert spec.submodules.self_attention.submodules.q_layernorm is None
        assert spec.submodules.self_attention.submodules.k_layernorm is None

    @pytest.mark.internal
    def test_qk_layernorm_from_config_fallback(self):
        # With config.qk_layernorm=True and the spec's q_layernorm/k_layernorm left unset,
        # SelfAttention should fall back to instantiating a real TE LayerNorm for Q and K.
        te_pytorch = pytest.importorskip("transformer_engine.pytorch")

        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            perform_initialization=True,
            qk_layernorm=True,
            pipeline_dtype=torch.bfloat16,
            attention_backend=AttnBackend.unfused,
        )
        bert_model = BertModel(
            config=transformer_config,
            num_tokentypes=0,
            transformer_layer_spec=get_bert_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )
        attention = bert_model.encoder.layers[0].self_attention
        assert isinstance(attention.q_layernorm, te_pytorch.LayerNorm)
        assert isinstance(attention.k_layernorm, te_pytorch.LayerNorm)

    @pytest.mark.internal
    def test_packed_forward_uses_cu_seqlens_positions_and_no_attention_mask(self, mocker):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            perform_initialization=True,
            attention_backend=AttnBackend.unfused,
        )
        bert_model = BertModel(
            config=config,
            num_tokentypes=0,
            transformer_layer_spec=get_bert_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
            position_embedding_type='rope',
            post_process=False,
        )
        sequence_length = 6
        input_ids = torch.arange(sequence_length, dtype=torch.int64).unsqueeze(0)
        cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=2,
            max_seqlen_kv=4,
        )
        encoder_input = torch.ones(sequence_length, 1, config.hidden_size)
        hidden_states = torch.zeros_like(encoder_input)
        rotary_pos_emb = torch.ones(4, 1, 1, config.hidden_size // config.num_attention_heads)

        extended_attention_mask = mocker.patch.object(bert_model, 'bert_extended_attention_mask')
        embedding_forward = mocker.patch.object(
            bert_model.embedding, 'forward', return_value=encoder_input
        )
        rotary_forward = mocker.patch.object(
            bert_model.rotary_pos_emb, 'forward', return_value=rotary_pos_emb
        )
        encoder_forward = mocker.patch.object(
            bert_model.encoder, 'forward', return_value=hidden_states
        )

        output = bert_model.forward(
            input_ids=input_ids, attention_mask=None, packed_seq_params=packed_seq_params
        )

        extended_attention_mask.assert_not_called()
        assert torch.equal(
            embedding_forward.call_args.kwargs['position_ids'],
            torch.tensor([[0, 1, 0, 1, 2, 3]], dtype=torch.int64),
        )
        rotary_forward.assert_called_once_with(
            4, packed_seq=True, cp_group=packed_seq_params.cp_group
        )
        assert encoder_forward.call_args.kwargs['attention_mask'] is None
        assert encoder_forward.call_args.kwargs['packed_seq_params'] is packed_seq_params
        assert encoder_forward.call_args.kwargs['rotary_pos_emb'] is rotary_pos_emb
        assert output is hidden_states

    @pytest.mark.internal
    def test_forward_validates_dense_attention_mask_and_packed_format(self):
        sequence_length = self.bert_model.max_sequence_length
        input_ids = torch.arange(sequence_length, dtype=torch.int64).unsqueeze(0)
        attention_mask = torch.ones((1, sequence_length), dtype=bool)
        cu_seqlens = torch.tensor([0, sequence_length], dtype=torch.int32)

        with pytest.raises(ValueError, match='attention_mask must be provided'):
            self.bert_model.forward(input_ids=input_ids, attention_mask=None)

        with pytest.raises(ValueError, match='qkv_format'):
            self.bert_model.forward(
                input_ids=input_ids,
                attention_mask=None,
                packed_seq_params=PackedSeqParams(
                    qkv_format='sbhd',
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    max_seqlen_q=sequence_length,
                    max_seqlen_kv=sequence_length,
                ),
            )

        with pytest.raises(ValueError, match='attention_mask must be None'):
            self.bert_model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                packed_seq_params=PackedSeqParams(
                    qkv_format='thd',
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_kv=cu_seqlens,
                    max_seqlen_q=sequence_length,
                    max_seqlen_kv=sequence_length,
                ),
            )

    @pytest.mark.internal
    @pytest.mark.parametrize(
        ('token_ids', 'cu_seqlens', 'error_match'),
        [
            (
                torch.arange(4, dtype=torch.int64).repeat((2, 1)),
                torch.tensor([0, 4]),
                'dummy batch',
            ),
            (
                torch.arange(4, dtype=torch.int64).unsqueeze(0),
                None,
                'cu_seqlens_q must be provided',
            ),
            (torch.arange(4, dtype=torch.int64).unsqueeze(0), torch.tensor([1, 4]), 'start at 0'),
            (torch.arange(4, dtype=torch.int64).unsqueeze(0), torch.tensor([0, 3]), 'end at'),
            (
                torch.arange(4, dtype=torch.int64).unsqueeze(0),
                torch.tensor([0, 3, 2, 4]),
                'monotonically',
            ),
        ],
    )
    def test_packed_position_ids_validate_cu_seqlens(self, token_ids, cu_seqlens, error_match):
        with pytest.raises(ValueError, match=error_match):
            self.bert_model.bert_position_ids(
                token_ids, PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu_seqlens)
            )


class TestBertModelAttentionDimensions:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            pipeline_dtype=torch.bfloat16,
            attention_backend=AttnBackend.auto,
        )
        # This should convert arbitray mask to padding mask
        self.bert_model = BertModel(
            config=self.transformer_config,
            num_tokentypes=0,
            transformer_layer_spec=get_bert_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=4,
        )

    @pytest.mark.internal
    def test_local_spec(self, mocker):
        self.bert_model.config.attention_backend = AttnBackend.local
        self.bert_model.transformer_layer_spec = bert_layer_local_spec
        attn_mask_dimensions = self.bert_model._sanity_check_attention_and_get_attn_mask_dimension()
        assert (
            attn_mask_dimensions == "b1ss"
        ), f"Expected b1ss for attn_mask_dimensions but got {attn_mask_dimensions}"

    @pytest.mark.internal
    def test_local_spec_exception(self, mocker):
        self.bert_model.config.attention_backend = AttnBackend.flash
        self.bert_model.transformer_layer_spec = bert_layer_local_spec
        with pytest.raises(Exception) as exc_info:
            self.bert_model._sanity_check_attention_and_get_attn_mask_dimension()
        assert (
            str(exc_info.value)
            == 'Expected AttnBackend to be local or auto while using mcore self attention, but found AttnBackend.flash. Set --attn-backend to local or dont use MCore SelfAttention submodule in layer specs'
        )

    @pytest.mark.internal
    def test_transformer_engine_version_1_10(self, mocker):
        submodules = get_bert_layer_with_transformer_engine_submodules()
        submodules.self_attention.params['attn_mask_type'] = AttnMaskType.arbitrary

        mocker.patch("megatron.core.utils.get_te_version", return_value=PkgVersion("1.10"))
        self.bert_model.transformer_layer_spec = ModuleSpec(
            module=TransformerLayer, submodules=submodules
        )
        attn_mask_dimensions = self.bert_model._sanity_check_attention_and_get_attn_mask_dimension()
        attn_mask_type = submodules.self_attention.params['attn_mask_type']
        assert (
            attn_mask_type == AttnMaskType.padding
        ), f"Exepcted attn mask type to be padding, but got {attn_mask_type}"
        assert (
            attn_mask_dimensions == "b11s"
        ), f"Expected b11s for attn_mask_dimensions but got {attn_mask_dimensions}"

    @pytest.mark.internal
    def test_transformer_engine_version_1_7_to_1_10_flash_attn(self, mocker):
        self.bert_model.config.attention_backend = AttnBackend.flash
        mocker.patch("megatron.core.utils.get_te_version", return_value=PkgVersion("1.8"))
        self.bert_model.transformer_layer_spec = get_bert_layer_with_transformer_engine_spec()
        attn_mask_dimensions = self.bert_model._sanity_check_attention_and_get_attn_mask_dimension()
        assert (
            attn_mask_dimensions == "b11s"
        ), f"Expected b11s for attn_mask_dimensions but got {attn_mask_dimensions}"

    @pytest.mark.internal
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    def test_transformer_engine_version_1_7_to_1_10_rng_error(self, mocker):
        submodules = get_bert_layer_with_transformer_engine_submodules()
        submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding
        mocker.patch("megatron.core.utils.get_te_version", return_value=PkgVersion("1.8"))
        with pytest.raises(Exception) as exc_info:
            self.bert_model = BertModel(
                config=self.transformer_config,
                num_tokentypes=0,
                transformer_layer_spec=ModuleSpec(module=TransformerLayer, submodules=submodules),
                vocab_size=100,
                max_sequence_length=4,
            )
        assert str(exc_info.value) == (
            "Linear.__init__() got an unexpected keyword argument 'rng_tracker_name' when "
            "instantiating TERowParallelLinear when instantiating SelfAttention when "
            "instantiating TransformerLayer"
        )

    @pytest.mark.internal
    def test_transformer_engine_version_1_7_to_1_10_unfused_attention(self, mocker):
        self.bert_model.config.attention_backend = AttnBackend.unfused
        submodules = get_bert_layer_with_transformer_engine_submodules()
        submodules.self_attention.params['attn_mask_type'] = AttnMaskType.padding
        mocker.patch("megatron.core.utils.get_te_version", return_value=PkgVersion("1.8"))
        self.bert_model.transformer_layer_spec = ModuleSpec(
            module=TransformerLayer, submodules=submodules
        )
        attn_mask_dimensions = self.bert_model._sanity_check_attention_and_get_attn_mask_dimension()
        attn_mask_type = submodules.self_attention.params['attn_mask_type']
        assert (
            attn_mask_type == AttnMaskType.arbitrary
        ), f"Exepcted attn mask type to be arbitrary, but got {attn_mask_type}"
        assert (
            attn_mask_dimensions == "b1ss"
        ), f"Expected b1ss for attn_mask_dimensions but got {attn_mask_dimensions}"

    @pytest.mark.internal
    def test_transformer_engine_version_less_than_1_7(self, mocker):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        self.bert_model.config.attention_backend = AttnBackend.flash
        with pytest.raises(Exception) as exc_info:
            mocker.patch("megatron.core.utils.get_te_version", return_value=PkgVersion("1.5"))
            self.bert_model = BertModel(
                config=self.transformer_config,
                num_tokentypes=0,
                transformer_layer_spec=get_bert_layer_with_transformer_engine_spec(),
                vocab_size=100,
                max_sequence_length=4,
            )

        assert str(exc_info.value) == (
            "Flash and fused attention is not supported with transformer engine version "
            "< 1.7. Set --attention-backend to unfused or leave it to be default (auto) or upgrade transformer engine >= 1.7"
        )
