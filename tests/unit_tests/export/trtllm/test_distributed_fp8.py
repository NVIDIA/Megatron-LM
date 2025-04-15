from functools import partial

import pytest
import torch
from pytest_mock import mocker
from torch.optim import Adam
from torch.utils.data import DataLoader

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import compile_helpers
from megatron.core.export.data_type import DataType
from megatron.core.export.export_config import ExportConfig
from megatron.core.export.model_type import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.tokenizer.tokenizer import _NullTokenizer
from tests.unit_tests.test_utilities import Utils

VOCAB_SIZE = 256
SEQUENCE_LENGTH = 64
NUM_LAYERS = 2
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def _model_provider():
    """Build the model."""

    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=512,
        num_attention_heads=16,
        use_cpu_initialization=True,
        num_query_groups=2,
        fp8='hybrid',
        fp8_margin=0,
        fp8_interval=1,
        fp8_amax_history_len=1024,
        fp8_amax_compute_algo="max",
        tensor_model_parallel_size=2,
    )

    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=VOCAB_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
    )

    return gpt_model


def _get_train_data_iterator():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=50),
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(datasets[0], batch_size=8, shuffle=True)

    train_iterator = iter(train_dataloader)

    return train_iterator


def _forward_step_func(data_iterator, model):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}

    data = next(data_iterator)
    tokens = torch.ones_like(data['tokens']).to(DEVICE)
    attention_mask = data['attention_mask'].to(DEVICE)
    position_ids = data['position_ids'].to(DEVICE)
    labels = data['labels'].to(DEVICE)
    loss_mask = data['loss_mask'].to(DEVICE)
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


class TestTRTLLMSingleDeviceConverterFP8:
    QUANTIZED_LAYERS = [
        'transformer.layers.*.attention.dense.weight',
        'transformer.layers.*.attention.qkv.weight',
        'transformer.layers.*.mlp.fc.weight',
        'transformer.layers.*.mlp.proj.weight',
    ]
    NON_QUANTIZED_LAYERS = [
        'transformer.layers.*.attention.dense.bias',
        'transformer.layers.*.input_layernorm.weight',
        'transformer.layers.*.input_layernorm.bias',
        'transformer.layers.*.attention.qkv.bias',
        'transformer.layers.*.post_layernorm.weight',
        'transformer.layers.*.post_layernorm.bias',
        'transformer.layers.*.mlp.fc.bias',
        'transformer.layers.*.mlp.proj.bias',
        'transformer.vocab_embedding.weight',
        'transformer.position_embedding.weight',
        'lm_head.weight',
        'transformer.ln_f.weight',
        'transformer.ln_f.bias',
    ]
    SCALING_FACTORS = [
        'transformer.layers.*.attention.dense.activation_scaling_factor',
        'transformer.layers.*.attention.dense.weights_scaling_factor',
        'transformer.layers.*.attention.qkv.activation_scaling_factor',
        'transformer.layers.*.attention.qkv.weights_scaling_factor',
        'transformer.layers.*.mlp.fc.activation_scaling_factor',
        'transformer.layers.*.mlp.fc.weights_scaling_factor',
        'transformer.layers.*.mlp.proj.activation_scaling_factor',
        'transformer.layers.*.mlp.proj.weights_scaling_factor',
    ]
    KV_SCALING_FACTORS = ['transformer.layers.*.attention.kv_cache_scaling_factor']

    def _assert_has_scales(self, state_dict, quantized):
        for layer in range(NUM_LAYERS):
            for key in self.SCALING_FACTORS:
                k = key.replace('*', str(layer))

                if quantized:
                    assert k in state_dict, f'Expected {k} in the converted model'
                    assert (
                        state_dict[k].dtype == torch.float32
                    ), 'Scaling factor dtype is expected to be torch.float32'
                else:
                    assert k not in state_dict, f'Did not expect {k} in the converted model'

    def _assert_has_kv_scales(self, state_dict, kv_quantized):
        for layer in range(NUM_LAYERS):
            for key in self.KV_SCALING_FACTORS:
                k = key.replace('*', str(layer))

                if kv_quantized:
                    assert k in state_dict, f'Expected {k} in the converted model'
                    assert (
                        state_dict[k].dtype == torch.float32
                    ), 'Scaling factor dtype is expected to be torch.float32'
                else:
                    assert k not in state_dict, f'Did not expect {k} in the converted model'

    def _assert_quantizable_layers(self, state_dict, quantized):
        expected_dtype = torch.float8_e4m3fn if quantized else DTYPE

        for layer in range(NUM_LAYERS):
            for key in self.QUANTIZED_LAYERS:
                k = key.replace('*', str(layer))

                assert k in state_dict, f'Expected {k} in the converted model'
                assert (
                    state_dict[k].dtype == expected_dtype
                ), f'Expected {k} to have the dtype == {str(expected_dtype)}'

    def _assert_non_quantizable_layers(self, state_dict):
        expected_dtype = torch.bfloat16

        for layer in range(NUM_LAYERS):
            for key in self.NON_QUANTIZED_LAYERS:
                k = key.replace('*', str(layer))

                assert k in state_dict, f'Expected {k} in the converted model'
                assert (
                    state_dict[k].dtype == expected_dtype
                ), f'Expected {k} to have the dtype == {str(expected_dtype)}'

    def setup_method(self, method):
        Utils.initialize_model_parallel(2, 1)
        gpt_model = _model_provider()
        gpt_model.to(DEVICE)
        optim = Adam(gpt_model.parameters())
        train_iterator = _get_train_data_iterator()
        forward_backward_func = get_forward_backward_func()

        # Mock training to initialize constants
        for _ in range(2):
            optim.zero_grad()
            forward_backward_func(
                forward_step_func=_forward_step_func,
                data_iterator=train_iterator,
                model=gpt_model,
                num_microbatches=1,
                seq_length=SEQUENCE_LENGTH,
                micro_batch_size=8,
                decoder_seq_length=SEQUENCE_LENGTH,
                forward_only=False,
            )
            optim.step()

        self.gpt_model = gpt_model

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_get_model_weights_converter(self, mocker):
        pytest.importorskip('tensorrt_llm')
        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.distributed_trtllm_model_weights_converter.str_dtype_to_torch",
            return_value=DTYPE,
        )

        from megatron.core.export.trtllm.trtllm_helper import TRTLLMHelper

        gpt_model = self.gpt_model
        seq_len_interpolation_factor = None
        if hasattr(gpt_model, "rotary_pos_emb"):
            seq_len_interpolation_factor = gpt_model.rotary_pos_emb.seq_len_interpolation_factor
        trtllm_helper = TRTLLMHelper(
            transformer_config=gpt_model.config,
            model_type=ModelType.gpt,
            position_embedding_type=gpt_model.position_embedding_type,
            max_position_embeddings=gpt_model.max_position_embeddings,
            rotary_percentage=gpt_model.rotary_percent,
            rotary_base=gpt_model.rotary_base,
            moe_tp_mode=2,
            multi_query_mode=False,
            activation="gelu",
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            share_embeddings_and_output_weights=gpt_model.share_embeddings_and_output_weights,
        )

        for fp8_quantized in [True, False]:
            for fp8_kvcache in [True, False]:
                weight_list, config_list = (
                    trtllm_helper.get_trtllm_pretrained_config_and_model_weights(
                        model_state_dict=gpt_model.state_dict(),
                        dtype=DataType.bfloat16,
                        on_device_distributed_conversion=True,
                        vocab_size=VOCAB_SIZE,
                        gpus_per_node=2,
                        fp8_quantized=fp8_quantized,
                        fp8_kvcache=fp8_kvcache,
                    )
                )

                expected_quant = 'FP8' if fp8_quantized else None
                expected_kv_quant = 'FP8' if fp8_kvcache else None
                assert (
                    config_list[0].quantization.quant_algo == expected_quant
                ), 'Wrong quantization settings'
                assert (
                    config_list[0].quantization.kv_cache_quant_algo == expected_kv_quant
                ), 'Wrong KV-cache quantization settings'
                self._assert_has_scales(weight_list[0], fp8_quantized)
                self._assert_has_kv_scales(weight_list[0], fp8_kvcache)
                self._assert_quantizable_layers(weight_list[0], fp8_quantized)
                self._assert_non_quantizable_layers(weight_list[0])
