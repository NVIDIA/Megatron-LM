# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import sys
import warnings

import pytest
import torch
from pytest_mock import mocker

from megatron.core import parallel_state, process_groups_config
from megatron.core.export.data_type import DataType
from megatron.core.export.trtllm.model_to_trllm_mapping.default_conversion_dict import (
    DEFAULT_CONVERSION_DICT,
)

# pylint: disable=line-too-long
from megatron.core.export.trtllm.trtllm_weights_converter.distributed_trtllm_model_weights_converter import (
    DistributedTRTLLMModelWeightsConverter,
)
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_SEQUENCE_LENGTH = 64
_VOCAB_SIZE = 256

_STR_DTYPE_TO_TORCH = (
    "megatron.core.export.trtllm.trtllm_weights_converter."
    "distributed_trtllm_model_weights_converter.str_dtype_to_torch"
)

# The accessors that tools/check_process_group_usage.py counts.
_ACCESSOR_SUFFIXES = ("_group", "_groups", "_gloo", "_rank", "_ranks", "_world_size", "_src_rank")
_NOT_ACCESSORS = {
    "get_nccl_options",
    "get_all_ranks",
    "get_global_memory_buffer",
    "get_virtual_pipeline_model_parallel_rank",
    "get_virtual_pipeline_model_parallel_world_size",
}


class GlobalProcessGroupRead(Exception):
    """A read of the global parallel grid where the test forbids one."""


@contextlib.contextmanager
def forbid_global_process_groups():
    """Make every read of the global parallel grid raise inside the block.

    Patches the parallel_state accessors, the copies that Megatron modules imported by name, and
    `ProcessGroupCollection.use_mpu_process_groups`.
    """
    # Keyed by id() because module attributes need not be hashable.
    names = {
        id(value): name
        for name, value in vars(parallel_state).items()
        if getattr(value, "__module__", None) == parallel_state.__name__
        and name.startswith("get_")
        and name.endswith(_ACCESSOR_SUFFIXES)
        and name not in _NOT_ACCESSORS
    }

    def forbidden(name):
        def read(*args, **kwargs):
            raise GlobalProcessGroupRead(f"read of the global parallel grid: {name}")

        return read

    with pytest.MonkeyPatch.context() as patch:
        for module_name, module in list(sys.modules.items()):
            if module is None or module_name.split(".")[0] != "megatron":
                continue
            for attribute, value in list(vars(module).items()):
                if id(value) in names:
                    patch.setattr(module, attribute, forbidden(names[id(value)]))
        patch.setattr(
            ProcessGroupCollection,
            "use_mpu_process_groups",
            classmethod(forbidden("ProcessGroupCollection.use_mpu_process_groups")),
        )
        yield


@pytest.fixture
def fresh_warning_registry(monkeypatch):
    """The test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _state_dict_without_extra_state(model):
    # The values of _extra_state entries are None.
    return {key: val for key, val in model.state_dict().items() if val is not None}


class TestTRTLLMDistributedGPUConverter:
    """
    Test Distributed converter
    """

    def setup_method(self, method):
        """
        Setup method
        """
        Utils.initialize_model_parallel(2, 1)
        model_parallel_cuda_manual_seed(123)

        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=2,
            use_cpu_initialization=True,
            pipeline_dtype=torch.float32,
            add_qkv_bias=False,
            add_bias_linear=False,
        )
        self.gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=_VOCAB_SIZE,
            max_sequence_length=_SEQUENCE_LENGTH,
        )

    def teardown_method(self, method):
        """
        teardown method
        """
        Utils.destroy_model_parallel()

    def test_get_model_weights_converter(self, mocker):
        """
        test model weights onverter
        """
        device = torch.device("cuda")
        self.gpt_model.to(device)

        transformer_config = self.gpt_model.config

        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.distributed_trtllm_model_weights_converter.str_dtype_to_torch",
            return_value=torch.float32,
        )

        dtype = DataType.bfloat16
        distributed_converter = DistributedTRTLLMModelWeightsConverter(
            transformer_config, dtype, activation="gelu"
        )

        model_state_dict = {}
        for key, val in self.gpt_model.state_dict().items():
            # val is non for _extra_state layers . We filter it out
            if val is not None:
                model_state_dict[key] = val

        distributed_converter.convert(
            model_state_dict=model_state_dict,
            trtllm_conversion_dict=DEFAULT_CONVERSION_DICT,
            tokenizer_vocab_size=_VOCAB_SIZE,
        )

        expected_result = {
            'transformer.vocab_embedding.weight': torch.Size([128, 64]),
            'transformer.position_embedding.weight': torch.Size([32, 64]),
            'lm_head.weight': torch.Size([128, 64]),
            'transformer.ln_f.weight': torch.Size([64]),
            'transformer.ln_f.bias': torch.Size([64]),
            'transformer.layers.0.input_layernorm.weight': torch.Size([64]),
            'transformer.layers.0.input_layernorm.bias': torch.Size([64]),
            'transformer.layers.0.attention.dense.weight': torch.Size([64, 32]),
            'transformer.layers.0.attention.qkv.weight': torch.Size([96, 64]),
            'transformer.layers.0.post_layernorm.weight': torch.Size([64]),
            'transformer.layers.0.post_layernorm.bias': torch.Size([64]),
            'transformer.layers.0.mlp.fc.weight': torch.Size([128, 64]),
            'transformer.layers.0.mlp.proj.weight': torch.Size([64, 128]),
            'transformer.layers.1.input_layernorm.weight': torch.Size([64]),
            'transformer.layers.1.input_layernorm.bias': torch.Size([64]),
            'transformer.layers.1.attention.dense.weight': torch.Size([64, 32]),
            'transformer.layers.1.attention.qkv.weight': torch.Size([96, 64]),
            'transformer.layers.1.post_layernorm.weight': torch.Size([64]),
            'transformer.layers.1.post_layernorm.bias': torch.Size([64]),
            'transformer.layers.1.mlp.fc.weight': torch.Size([128, 64]),
            'transformer.layers.1.mlp.proj.weight': torch.Size([64, 128]),
        }

        for key, value in distributed_converter.trtllm_model_weights.items():
            assert (
                expected_result[key] == value.shape
            ), f"Shape mismatch for {key}. Expected {expected_result[key]} but got {value.shape}"

    def test_requires_model_parallel_groups(self, mocker):
        """
        Without model parallel groups the converter raises instead of exporting TP1/PP1.
        """
        mocker.patch(
            "megatron.core.export.trtllm.trtllm_weights_converter.distributed_trtllm_model_weights_converter.str_dtype_to_torch",
            return_value=torch.float32,
        )
        transformer_config = self.gpt_model.config
        Utils.destroy_model_parallel()

        with pytest.raises(RuntimeError, match="initialize_model_parallel"):
            DistributedTRTLLMModelWeightsConverter(transformer_config, DataType.bfloat16)

    @pytest.mark.usefixtures("fresh_warning_registry")
    def test_without_process_groups_warns_once(self, mocker):
        """Without pg_collection, the converter warns once and uses the global groups."""
        mocker.patch(_STR_DTYPE_TO_TORCH, return_value=torch.float32)

        with pytest.warns(ProcessGroupFallbackWarning) as record:
            for _ in range(2):
                converter = DistributedTRTLLMModelWeightsConverter(
                    self.gpt_model.config, DataType.bfloat16
                )

        fallbacks = [w for w in record if issubclass(w.category, ProcessGroupFallbackWarning)]
        assert len(fallbacks) == 1
        message = str(fallbacks[0].message)
        owner = "DistributedTRTLLMModelWeightsConverter"
        assert f"{owner} was called without `pg_collection`" in message
        assert "deprecated since Megatron Core 0.21 and will be removed in 0.23" in message
        assert fallbacks[0].filename == __file__
        assert converter.tp_group is parallel_state.get_tensor_model_parallel_group()
        assert (converter.inference_tp_size, converter.inference_pp_size) == (2, 1)

    def test_uses_given_process_groups(self, mocker):
        """With pg_collection, the converter uses its groups and reads no global ones.

        The TP=2 groups pair the same ranks as the global grid but are distinct communicators, so
        the converted weights must match the global-group conversion while the converter keeps
        the given groups.
        """
        if Utils.world_size % 2 != 0:
            pytest.skip("needs an even world size")
        mocker.patch(_STR_DTYPE_TO_TORCH, return_value=torch.float32)
        self.gpt_model.to(torch.device("cuda"))
        # convert() renames the keys of the state dict it is given, so each conversion gets its own.
        global_state_dict = _state_dict_without_extra_state(self.gpt_model)
        model_state_dict = _state_dict_without_extra_state(self.gpt_model)
        grid = HyperCommGrid([2, Utils.world_size // 2, 1], ["tp", "dp", "pp"])
        pg_collection = ProcessGroupCollection(tp=grid.create_pg("tp"), pp=grid.create_pg("pp"))
        assert pg_collection.tp is not parallel_state.get_tensor_model_parallel_group()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ProcessGroupFallbackWarning)
            global_converter = DistributedTRTLLMModelWeightsConverter(
                self.gpt_model.config, DataType.bfloat16
            )
        global_converter.convert(
            model_state_dict=global_state_dict,
            trtllm_conversion_dict=DEFAULT_CONVERSION_DICT,
            tokenizer_vocab_size=_VOCAB_SIZE,
        )

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            converter = DistributedTRTLLMModelWeightsConverter(
                self.gpt_model.config, DataType.bfloat16, pg_collection=pg_collection
            )
            converter.convert(
                model_state_dict=model_state_dict,
                trtllm_conversion_dict=DEFAULT_CONVERSION_DICT,
                tokenizer_vocab_size=_VOCAB_SIZE,
            )

        assert converter.tp_group is pg_collection.tp
        assert (converter.inference_tp_size, converter.tp_rank) == (2, pg_collection.tp.rank())
        assert (converter.inference_pp_size, converter.pp_rank) == (1, 0)
        assert converter.trtllm_model_weights.keys() == global_converter.trtllm_model_weights.keys()
        for key, value in converter.trtllm_model_weights.items():
            assert torch.equal(value, global_converter.trtllm_model_weights[key]), key
        grid.destroy()

    @pytest.mark.parametrize("unset", ["tp", "pp"])
    def test_rejects_collection_without_groups(self, mocker, unset):
        """A collection that leaves tp or pp unset or None raises instead of exporting TP1/PP1."""
        mocker.patch(_STR_DTYPE_TO_TORCH, return_value=torch.float32)
        groups = {
            "tp": parallel_state.get_tensor_model_parallel_group(),
            "pp": parallel_state.get_pipeline_model_parallel_group(),
        }

        with forbid_global_process_groups():
            with pytest.raises(RuntimeError, match="pass a pg_collection that sets tp and pp"):
                DistributedTRTLLMModelWeightsConverter(
                    self.gpt_model.config,
                    DataType.bfloat16,
                    pg_collection=ProcessGroupCollection(**{**groups, unset: None}),
                )
            del groups[unset]
            with pytest.raises(ValueError, match=f"requires pg_collection to set {unset}"):
                DistributedTRTLLMModelWeightsConverter(
                    self.gpt_model.config,
                    DataType.bfloat16,
                    pg_collection=ProcessGroupCollection(**groups),
                )
