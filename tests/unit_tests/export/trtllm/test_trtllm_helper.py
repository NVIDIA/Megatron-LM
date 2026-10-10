# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.export.data_type import DataType
from megatron.core.export.export_config import ExportConfig
from megatron.core.export.model_type import ModelType
from megatron.core.process_groups_config import ProcessGroupCollection


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


class _ConverterCreated(Exception):
    """Stops the conversion once the distributed converter is constructed."""


def test_on_device_conversion_passes_process_groups_to_converter(mocker):
    """`pg_collection` reaches the distributed converter unchanged."""
    from megatron.core.export.trtllm import trtllm_helper

    converter_kwargs = {}

    def create_converter(**kwargs):
        converter_kwargs.update(kwargs)
        raise _ConverterCreated

    mocker.patch.object(
        trtllm_helper, "DistributedTRTLLMModelWeightsConverter", side_effect=create_converter
    )
    # Built without __init__, which needs TensorRT-LLM; the code under test does not.
    helper = trtllm_helper.TRTLLMHelper.__new__(trtllm_helper.TRTLLMHelper)
    helper.model_type = ModelType.gpt
    helper.transformer_config = None
    helper.multi_query_mode = False
    helper.activation = "gelu"
    pg_collection = ProcessGroupCollection()

    with pytest.raises(_ConverterCreated):
        helper.get_trtllm_pretrained_config_and_model_weights(
            model_state_dict={},
            dtype=DataType.bfloat16,
            on_device_distributed_conversion=True,
            vocab_size=100,
            gpus_per_node=2,
            pg_collection=pg_collection,
        )
    assert converter_kwargs["pg_collection"] is pg_collection
