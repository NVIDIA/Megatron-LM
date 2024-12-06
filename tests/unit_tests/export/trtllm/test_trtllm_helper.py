import pytest

from megatron.core.export.export_config import ExportConfig
from megatron.core.export.model_type import ModelType


# TODO : Remove importorskip and handle with mocker
class TestTRTLLMHelper:

    def test_exceptions(self, mocker):
        pytest.importorskip('tensorrt_llm')

        from megatron.core.export.trtllm.trtllm_helper import TRTLLMHelper

        trtllm_helper = TRTLLMHelper(
            transformer_config=None,
            model_type=ModelType.gpt,
            share_embeddings_and_output_weights=True,
        )

        with pytest.raises(AssertionError):
            trtllm_helper.get_trtllm_pretrained_config_and_model_weights(
                model_state_dict=None,
                dtype=None,
                on_device_distributed_conversion=True,
                vocab_size=None,
                gpus_per_node=2,
            )

        with pytest.raises(AssertionError):
            trtllm_helper.get_trtllm_pretrained_config_and_model_weights(
                model_state_dict=None,
                dtype=None,
                on_device_distributed_conversion=True,
                vocab_size=100,
                gpus_per_node=2,
            )

        with pytest.raises(AssertionError):
            trtllm_helper.get_trtllm_pretrained_config_and_model_weights(
                model_state_dict=None,
                dtype=None,
                export_config=ExportConfig(),
                on_device_distributed_conversion=True,
                vocab_size=100,
                gpus_per_node=2,
            )

        with pytest.raises(AssertionError):
            trtllm_helper.get_trtllm_pretrained_config_and_model_weights(
                model_state_dict=None,
                dtype=None,
                vocab_size=100,
                on_device_distributed_conversion=True,
                gpus_per_node=None,
            )

        with pytest.raises(AssertionError):
            trtllm_helper.get_trtllm_pretrained_config_and_model_weights(
                model_state_dict=None,
                dtype=None,
                export_config=ExportConfig(use_embedding_sharing=False),
                on_device_distributed_conversion=False,
            )

        with pytest.raises(AssertionError):
            trtllm_helper.get_trtllm_pretrained_config_and_model_weights(
                model_state_dict=None,
                dtype=None,
                export_config=ExportConfig(use_embedding_sharing=True),
                vocab_size=100,
            )
