# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
import sys

import pytest
import torch
from packaging.version import Version as PkgVersion

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.bert.bert_layer_specs import (
    bert_layer_local_spec,
    get_bert_layer_with_transformer_engine_spec,
    get_bert_layer_with_transformer_engine_submodules,
)
from megatron.core.models.bert.bert_lm_head import BertLMHead
from megatron.core.models.bert.bert_model import BertModel
from megatron.core.process_groups_config import ProcessGroupCollection
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
    def test_apply_lm_head_default_creates_bert_lm_head(self):
        assert isinstance(self.bert_model.lm_head, BertLMHead)

    @pytest.mark.internal
    def test_output_layer_bias_false_disables_bias(self):
        bert_model = BertModel(
            config=self.bert_model.config,
            num_tokentypes=0,
            transformer_layer_spec=get_bert_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=self.bert_model.max_sequence_length,
            apply_lm_head=False,
            output_layer_bias=False,
        )

        assert bert_model.output_layer.bias is None

    @pytest.mark.internal
    def test_apply_lm_head_false_bypasses_head(self):
        config: TransformerConfig = self.bert_model.config
        sequence_length = self.bert_model.max_sequence_length
        micro_batch_size = 2

        bert_model = BertModel(
            config=config,
            num_tokentypes=0,
            transformer_layer_spec=get_bert_layer_with_transformer_engine_spec(),
            vocab_size=100,
            max_sequence_length=sequence_length,
            apply_lm_head=False,
        )
        assert bert_model.lm_head is None
        bert_model.cuda()

        encoder_output = {}
        bert_model.encoder.register_forward_hook(
            lambda module, args, output: encoder_output.setdefault('hidden_states', output)
        )

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones((micro_batch_size, sequence_length), dtype=bool).cuda()

        logits = bert_model.forward(input_ids=input_ids, attention_mask=attention_mask)

        assert logits[0].shape[0] == micro_batch_size
        assert logits[0].shape[1] == sequence_length
        assert logits[0].shape[2] == bert_model.vocab_size

        # With apply_lm_head=False, the output_layer must be applied directly to the
        # encoder's hidden states, without BertLMHead's dense+GeLU+LayerNorm transform.
        expected_logits, _ = bert_model.output_layer(encoder_output['hidden_states'])
        torch.testing.assert_close(logits[0], expected_logits.transpose(0, 1).contiguous())

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


_GLOBAL_GRID_PREFIXES = (
    "get_tensor_model_parallel",
    "get_pipeline_model_parallel",
    "get_context_parallel",
    "is_pipeline_",
)


def _forbid_global_grid(patch):
    """Make the global TP, PP and CP accessors, the stage predicates and the shim raise.

    Modules that imported an accessor by name hold their own reference, so those are patched too.
    """
    for name in dir(parallel_state):
        if not name.startswith(_GLOBAL_GRID_PREFIXES):
            continue
        original = getattr(parallel_state, name)

        def forbid(*args, _name=name, **kwargs):
            raise AssertionError(f"read of the global grid: parallel_state.{_name}")

        for module in list(sys.modules.values()):
            if getattr(module, "__name__", "").startswith("megatron.") and (
                getattr(module, "__dict__", {}).get(name) is original
            ):
                patch.setattr(module, name, forbid)

    def forbid_shim(cls, *args, **kwargs):
        raise AssertionError("read of the global grid: use_mpu_process_groups")

    patch.setattr(ProcessGroupCollection, "use_mpu_process_groups", classmethod(forbid_shim))


class TestBertModelWithOwnProcessGroups:
    """A BertModel given its own collection runs entirely on the groups in that collection."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        Utils.world_size < 2 or Utils.world_size % 2 != 0, reason="needs an even number of ranks"
    )
    @pytest.mark.parametrize("sequence_parallel", [False, True])
    @pytest.mark.parametrize("position_embedding_type", ["learned_absolute", "rope"])
    def test_forward_matches_global_grid_without_reading_it(
        self, monkeypatch, sequence_parallel, position_embedding_type
    ):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=2,
            sequence_parallel=sequence_parallel,
            attention_backend=AttnBackend.unfused,
        )

        def build_bert(pg_collection=None):
            return BertModel(
                config=config,
                num_tokentypes=0,
                transformer_layer_spec=get_bert_layer_with_transformer_engine_spec(),
                vocab_size=100,
                max_sequence_length=8,
                position_embedding_type=position_embedding_type,
                pg_collection=pg_collection,
            )

        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        reference = build_bert().cuda().eval()

        # Same layout as the global grid (TP=2, CP=1, PP=1), but new communicators: a module that
        # read the global groups instead of these would hit the forbidden accessors below.
        grid = HyperCommGrid([2, 1, 1, Utils.world_size // 2], ["tp", "cp", "pp", "dp"])
        try:
            pg_collection = ProcessGroupCollection(
                tp=grid.create_pg("tp"),
                cp=grid.create_pg("cp"),
                pp=grid.create_pg("pp"),
                # Single pipeline stage: no embedding groups and no GTP axis.
                embd=None,
                pos_embd=None,
                gtp_remat=None,
                expt_gtp_remat=None,
            )
            assert pg_collection.tp is not parallel_state.get_tensor_model_parallel_group()
            assert torch.distributed.get_process_group_ranks(
                pg_collection.tp
            ) == torch.distributed.get_process_group_ranks(
                parallel_state.get_tensor_model_parallel_group()
            )

            input_ids = torch.randint(
                0,
                100,
                (2, 8),
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(0),
            )
            attention_mask = torch.ones((2, 8), dtype=bool, device="cuda")
            with torch.no_grad():
                expected_logits, expected_binary_logits = reference(input_ids, attention_mask)

            with monkeypatch.context() as patch:
                _forbid_global_grid(patch)
                model = build_bert(pg_collection).cuda().eval()
                model.load_state_dict(reference.state_dict())
                with torch.no_grad():
                    logits, binary_logits = model(input_ids, attention_mask)

            assert model.embedding.tp_group is pg_collection.tp
            assert model.encoder.pg_collection is pg_collection
            assert model.output_layer.tp_group is pg_collection.tp
            assert model.pooler.tp_group is pg_collection.tp
            if position_embedding_type == "rope":
                assert model.rotary_pos_emb.cp_group is pg_collection.cp
            torch.testing.assert_close(logits, expected_logits)
            torch.testing.assert_close(binary_logits, expected_binary_logits)
        finally:
            grid.destroy()
