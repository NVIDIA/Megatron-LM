# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The data section around Megatron Core's GPTDatasetConfig: the lifecycle (validate, finalize, bind
the tokenizer, build), the serialization and the dataset-class resolution."""

import copy
from argparse import Namespace
from dataclasses import fields
from types import SimpleNamespace

import pytest
import yaml

import megatron.training.config as config_package
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.gpt_fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.config import DataConfig, DataloaderConfig
from megatron.training.config.container import ConfigContainerBase
from megatron.training.config.dataset_config import (
    CONTAINER_WRITTEN_FIELDS,
    classify_dataset_class,
    dataset_class_for,
    has_data_source,
    is_packed_sequence,
    resolve_dataset_class,
)
from megatron.training.config.instantiate_utils import InstantiationMode, instantiate
from megatron.training.config.utils import sanitize_dataclass_config
from megatron.training.config.yaml_utils import safe_yaml_representers
from megatron.training.datasets.sft_dataset import MockSFTDataset, SFTDataset
from megatron.training.datasets.varlen_dataset import MockVarlenDataset, VarlenDataset

BOOLS = dict(reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False)
BLEND = (["prefix-a", "prefix-b"], [0.3, 0.7])
FIM_TOKENS = {"prefix": "<p>", "middle": "<m>", "suffix": "<s>", "pad": "<x>", "eod": "<e>"}


def _dataset(**overrides):
    kwargs = dict(sequence_length=16, random_seed=1234, **BOOLS)
    kwargs.update(overrides)
    return GPTDatasetConfig(**kwargs)


def _fim(**overrides):
    kwargs = dict(
        sequence_length=16,
        random_seed=1234,
        blend=BLEND,
        split="9,1,0",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        fim_extra_tokens=dict(FIM_TOKENS),
        **BOOLS,
    )
    kwargs.update(overrides)
    return GPTFIMDatasetConfig(**kwargs)


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer(
        Namespace(
            vocab_size=2048,
            tokenizer_type="NullTokenizer",
            rank=0,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=1,
            pad_vocab_size=True,
        )
    )


# The section holds Megatron Core's config itself.


def test_the_section_holds_megatron_cores_config(tokenizer):
    """No wrapper class: cfg.data.dataset is Megatron Core's GPTDatasetConfig, constructed before
    the tokenizer exists (Core's mock and split_matrix follow from the sources; its validate()
    needs the tokenizer and runs when the section binds it)."""
    assert not hasattr(config_package, "GPTDatasetConfig")
    assert not hasattr(config_package, "GPTFIMDatasetConfig")
    data = DataConfig(dataset=_dataset())
    assert type(data.dataset) is GPTDatasetConfig
    assert data.dataset.tokenizer is None and data.dataset.token_dtype_code is None
    assert data.dataset.mock is True and data.dataset.split is None  # the even split is derived
    assert data.dataset.split_matrix is not None
    explicit = DataConfig(dataset=_dataset(split="949,50,1"))  # --mock-data --split 949,50,1
    assert explicit.dataset.split == "949,50,1"  # kept as given (and serialized as given)
    assert explicit.dataset.split_matrix == data.dataset.split_matrix  # the even split, as before
    explicit.validate()
    explicit.finalize()
    assert explicit.dataset_type is MockGPTDataset
    data.validate()  # the section's checks and Core's tokenizer-free ones
    data.finalize()
    assert data.dataset_type is MockGPTDataset
    with pytest.raises(RuntimeError, match="no tokenizer bound"):
        data.dataset_build()
    data.bind_tokenizer(tokenizer)  # runs Core's validate()
    assert data.dataset.tokenizer is tokenizer and data.dataset.token_dtype_code == 8
    dataset_type, config, packed = data.dataset_build()
    assert dataset_type is MockGPTDataset and config is data.dataset and packed is False
    data.bind_tokenizer(tokenizer)  # idempotent for the same tokenizer
    with pytest.raises(RuntimeError, match="another tokenizer"):
        data.bind_tokenizer(SimpleNamespace(vocab_size=2048))
    assert data.dataset.tokenizer is tokenizer
    with pytest.raises(ValueError, match="needs a tokenizer"):
        DataConfig(dataset=_dataset()).bind_tokenizer(None)
    DataConfig().bind_tokenizer(tokenizer)  # loader-only: nothing to bind
    eager = DataConfig(dataset=_dataset(tokenizer=tokenizer))  # a Python caller's tokenizer
    eager.bind_tokenizer(tokenizer)
    assert eager.dataset.token_dtype_code == 8


def test_bind_tokenizer_runs_megatron_cores_checks(tokenizer):
    """Core's validate() is where a GPT config that needs the tokenizer fails: the asserts its
    __post_init__ had, once the tokenizer is bound (the blend / split rules fail at construction,
    as they did)."""
    with pytest.raises(AssertionError):
        DataConfig(dataset=_dataset(reset_position_ids=None)).bind_tokenizer(tokenizer)
    sbhd = DataConfig(dataset=_dataset(varlen_sbhd_validation=True, hybrid_context_parallel=True))
    with pytest.raises(AssertionError, match="varlen-sbhd-validation"):
        sbhd.bind_tokenizer(tokenizer)
    counts = DataConfig(
        dataset=_dataset(blend=BLEND, split="9,1,0", sequences_per_dataset={"prefix-a": [10, 2]})
    )
    with pytest.raises(AssertionError, match="vocab size"):
        counts.bind_tokenizer(SimpleNamespace(vocab_size=None))
    with pytest.raises(AssertionError, match="split must be provided"):  # Core's, at validate()
        DataConfig(dataset=_dataset(blend=BLEND)).validate()
    with pytest.raises(AssertionError, match="equal in number"):
        DataConfig(dataset=_dataset(blend=(["a", "b"], [1.0]), split="1,1,1")).validate()
    with pytest.raises(TypeError, match="random_seed"):  # Core's requirement, unchanged
        GPTDatasetConfig(sequence_length=16, **BOOLS)
    for overrides in (
        {},
        dict(blend=BLEND, split="9,1,0"),
        dict(blend_per_split=[(["a"], None), None, (["b", "c"], [1.0, 2.0])]),
        dict(blend_per_split=[None, None, None]),
    ):
        assert _dataset(**overrides).mock is (not has_data_source(_dataset(**overrides)))


# The FIM config.


def test_fim_config_is_megatron_cores(tokenizer):
    cfg = _fim()
    assert isinstance(cfg, GPTDatasetConfig)
    assert [f.name for f in fields(cfg) if f.name.startswith("fim_")] == [
        "fim_rate", "fim_spm_rate", "fim_extra_tokens", "fim_split_sample", "fim_fragment_rate", "fim_no_prefix",
    ]  # fmt: skip
    assert not any(f.name.startswith("fim_") for f in fields(GPTDatasetConfig))
    data = DataConfig(dataset=cfg)
    data.validate()
    data.finalize()
    assert data.dataset_type is GPTFIMDataset
    data.bind_tokenizer(tokenizer)
    assert data.dataset_build()[0] is GPTFIMDataset
    DataConfig(dataset=_fim(fim_rate=0.0, fim_spm_rate=0.0)).bind_tokenizer(
        tokenizer
    )  # 0.0 is a rate
    for bad, message in (
        (dict(fim_rate=None), "fim_rate"),
        (dict(fim_spm_rate=None), "fim_spm_rate"),
        (dict(fim_extra_tokens={"prefix": "<p>"}), "FIM extra tokens"),
        (dict(fim_extra_tokens=None), "FIM extra tokens"),
    ):
        with pytest.raises(AssertionError, match=message):
            DataConfig(dataset=_fim(**bad)).bind_tokenizer(tokenizer)


# Serialization.


def test_container_written_fields_are_the_layout_and_the_validation_flags():
    """The six fields other sections own, left at Megatron Core's defaults until the container
    writes them; the seed is the section's own setting, not one of them."""
    assert CONTAINER_WRITTEN_FIELDS == (
        "data_parallel_size", "context_parallel_size", "sequence_parallel_size",
        "hybrid_context_parallel", "multiple_validation_sets", "full_validation",
    )  # fmt: skip
    defaults = {f.name: f.default for f in fields(GPTDatasetConfig)}
    assert {name: defaults[name] for name in CONTAINER_WRITTEN_FIELDS} == {
        "data_parallel_size": 1, "context_parallel_size": None, "sequence_parallel_size": 0,
        "hybrid_context_parallel": False, "multiple_validation_sets": None, "full_validation": None,
    }  # fmt: skip
    assert "random_seed" not in CONTAINER_WRITTEN_FIELDS


def test_to_cfg_dict_omits_tokenizer_owned_and_derived_fields(tokenizer):
    data = DataConfig(
        dataset=_dataset(blend=BLEND, split="9,1,0", data_parallel_size=4, full_validation=True),
        dataset_type=GPTDataset,
    )
    before = data.to_cfg_dict()
    assert before["_target_"] == "megatron.training.config.dataset_config.DataConfig"
    section = before["dataset"]
    assert section["_target_"] == "megatron.core.datasets.gpt_dataset.GPTDatasetConfig"
    assert "tokenizer" not in section and not (set(CONTAINER_WRITTEN_FIELDS) & set(section))
    assert section["random_seed"] == 1234  # the seed is saved with the section
    assert not {"split_matrix", "mock", "token_dtype_code"} & set(section)
    assert (
        section["blend"] == [["prefix-a", "prefix-b"], [0.3, 0.7]] and section["split"] == "9,1,0"
    )
    assert before["dataset_type"] is GPTDataset and before["dataloader"]["num_workers"] == 2
    data.bind_tokenizer(tokenizer)
    assert data.to_cfg_dict() == before  # binding derives nothing into a constructor field
    restored = instantiate(
        sanitize_dataclass_config(copy.deepcopy(before)), mode=InstantiationMode.STRICT
    )
    assert type(restored) is DataConfig and type(restored.dataset) is GPTDatasetConfig
    assert restored.dataset.tokenizer is None and restored.dataset.data_parallel_size == 1
    assert restored.dataset.full_validation is None and restored.dataset.random_seed == 1234
    assert restored.to_cfg_dict() == before
    fim = DataConfig(dataset=_fim()).to_cfg_dict()["dataset"]
    assert fim["_target_"] == "megatron.core.datasets.gpt_fim_dataset.GPTFIMDatasetConfig"
    assert fim["fim_extra_tokens"] == FIM_TOKENS
    assert type(instantiate(fim, mode=InstantiationMode.STRICT)) is GPTFIMDatasetConfig
    # A mock section carries no split: Megatron Core derives the even split (split_matrix) from
    # the absence of a source and writes nothing back; an explicit split is kept as given and, as
    # before, not used by the mock dataset.
    assert DataConfig(dataset=_dataset()).to_cfg_dict()["dataset"]["split"] is None
    explicit = DataConfig(dataset=_dataset(split="949,50,1")).to_cfg_dict()["dataset"]
    assert explicit["split"] == "949,50,1"
    assert DataConfig().to_cfg_dict()["dataset"] is None


def test_data_section_round_trips_through_yaml_with_the_class():
    data = DataConfig(
        dataset=_dataset(blend=BLEND, split="9,1,0"),
        dataloader=DataloaderConfig(num_workers=0),
        dataset_type=GPTDataset,
    )
    as_dict = ConfigContainerBase._convert_value_to_dict(data)
    assert as_dict == data.to_cfg_dict()
    with safe_yaml_representers():
        text = yaml.safe_dump(as_dict, default_flow_style=False)
    assert "_call_: false" in text and "gpt_dataset.GPTDataset" in text
    loaded = sanitize_dataclass_config(yaml.safe_load(text))
    restored = instantiate(loaded, mode=InstantiationMode.STRICT)
    assert type(restored) is DataConfig and restored.dataset_type is GPTDataset
    assert restored.dataloader == data.dataloader
    assert type(restored.dataset) is GPTDatasetConfig and restored.dataset.tokenizer is None
    assert restored.to_cfg_dict() == data.to_cfg_dict()
    restored.validate()
    restored.finalize()
    assert restored.dataset_type is GPTDataset


# Dataset-class resolution.


def test_implicit_class_resolution():
    assert resolve_dataset_class(_dataset(blend=BLEND, split="1,1,1"), None) is GPTDataset
    assert resolve_dataset_class(_dataset(), None) is MockGPTDataset
    assert (
        resolve_dataset_class(_dataset(blend_per_split=[None, None, None]), None) is MockGPTDataset
    )
    assert resolve_dataset_class(_fim(), None) is GPTFIMDataset
    with pytest.raises(ValueError, match="no mock dataset for the fim family"):
        resolve_dataset_class(_fim(blend=None, split=None), None)
    assert resolve_dataset_class(None, None) is None
    with pytest.raises(ValueError, match="dataset_type needs a dataset"):
        resolve_dataset_class(None, GPTDataset)


@pytest.mark.parametrize(
    ("dataset", "dataset_type", "message"),
    [
        (_dataset(blend=BLEND, split="1,1,1"), MockGPTDataset, "takes no data source"),
        (_dataset(), GPTDataset, "select MockGPTDataset"),
        (_dataset(), VarlenDataset, "select MockVarlenDataset"),
        (_fim(blend=None, split=None), GPTFIMDataset, "no mock dataset for the fim family"),
        (_dataset(blend=BLEND, split="1,1,1"), GPTFIMDataset, "go together"),
        (_fim(), GPTDataset, "go together"),
        (_fim(), SFTDataset, "go together"),
        (_dataset(varlen_sbhd_validation=True), MockVarlenDataset, "requires real data"),
        (_dataset(), object, "must be one of the dataset classes"),
        (_dataset(), "GPTDataset", "must be one of the dataset classes"),
    ],
)  # fmt: skip
def test_explicit_class_must_match_the_config(dataset, dataset_type, message):
    with pytest.raises(ValueError, match=message):
        resolve_dataset_class(dataset, dataset_type)


@pytest.mark.parametrize(
    ("dataset", "dataset_type"),
    [
        (_dataset(blend=BLEND, split="1,1,1"), GPTDataset),
        (_dataset(), MockGPTDataset),
        (_dataset(blend=BLEND, split="1,1,1"), SFTDataset),
        (_dataset(), MockSFTDataset),
        (_dataset(), MockVarlenDataset),
        (_dataset(blend=BLEND, split="1,1,1", varlen_sbhd_validation=True), VarlenDataset),
        (_fim(), GPTFIMDataset),
    ],
)
def test_explicit_classes_accepted(dataset, dataset_type):
    assert resolve_dataset_class(dataset, dataset_type) is dataset_type


def test_class_table_and_packedness():
    assert dataset_class_for("gpt", mock=True) is MockGPTDataset
    assert dataset_class_for("fim", mock=True) is None
    assert classify_dataset_class(MockVarlenDataset) == ("varlen", True)
    with pytest.raises(ValueError, match="dataset family"):
        dataset_class_for("bert", mock=False)
    plain = _dataset()
    assert is_packed_sequence(SFTDataset, plain) and is_packed_sequence(MockSFTDataset, plain)
    assert is_packed_sequence(VarlenDataset, plain) and is_packed_sequence(MockVarlenDataset, plain)
    assert not is_packed_sequence(VarlenDataset, _dataset(varlen_sbhd_validation=True))
    assert not is_packed_sequence(GPTDataset, plain) and not is_packed_sequence(
        GPTFIMDataset, plain
    )


# The data section lifecycle.


def test_loader_only_section():
    data = DataConfig(dataloader=DataloaderConfig(num_workers=0))
    data.validate()
    data.finalize()
    assert data.dataset is None and data.dataset_type is None
    assert (
        data.dataloader.dataloader_type == "single" and data.dataloader.persistent_workers is False
    )
    with pytest.raises(ValueError, match="no dataset"):
        data.dataset_build()


def test_finalize_commits_the_class_and_dataset_build_needs_the_tokenizer(tokenizer):
    data = DataConfig(dataset=_dataset(), dataloader=DataloaderConfig(dataloader_type="cyclic"))
    data.validate()
    assert data.dataset_type is None  # validation does not write
    data.finalize()
    assert data.dataset_type is MockGPTDataset and data.dataloader.dataloader_type == "cyclic"
    with pytest.raises(RuntimeError, match="no tokenizer bound"):
        data.dataset_build()
    data.bind_tokenizer(tokenizer)
    dataset_type, config, packed = data.dataset_build()
    assert dataset_type is MockGPTDataset and config is data.dataset and packed is False
    data.finalize()  # idempotent
    assert data.dataset_type is MockGPTDataset
    sft = DataConfig(dataset=_dataset(), dataset_type=MockSFTDataset)
    sft.validate()
    sft.finalize()
    sft.bind_tokenizer(tokenizer)
    assert sft.dataset_build()[2] is True


def test_dataloader_config_validation():
    with pytest.raises(ValueError, match="dataloader_type"):
        DataloaderConfig(dataloader_type="batch").validate()
    with pytest.raises(ValueError, match="num_workers"):
        DataloaderConfig(num_workers=-1).validate()
    loader = DataloaderConfig(num_workers=2)
    loader.validate()
    loader.finalize()
    assert loader.dataloader_type == "single" and loader.persistent_workers is True


def test_legacy_derived_keys_are_dropped_on_load():
    """Files written while mock, split_matrix and token_dtype_code were fields carry them as keys;
    the loader drops them (they are read-only properties now) and the section loads."""
    section = DataConfig(dataset=_dataset(blend=BLEND, split="9,1,0")).to_cfg_dict()
    section["dataset"].update(
        mock=False, split_matrix=[[0.0, 0.9], [0.9, 1.0], None], token_dtype_code=8
    )
    restored = instantiate(sanitize_dataclass_config(section), mode=InstantiationMode.STRICT)
    assert type(restored.dataset) is GPTDatasetConfig and restored.dataset.mock is False
    assert restored.dataset.split_matrix == _dataset(blend=BLEND, split="9,1,0").split_matrix
