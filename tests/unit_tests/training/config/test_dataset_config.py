# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""One dataset config object: Megatron Core's GPTDatasetConfig with deferred initialization, the
dataset-class resolution and the data section lifecycle."""

import copy
import dataclasses
from argparse import Namespace
from dataclasses import fields
from types import SimpleNamespace

import pytest
import yaml

from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig as MCoreGPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.config import (
    DataConfig,
    DataloaderConfig,
    GPTDatasetConfig,
    GPTFIMDatasetConfig,
)
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
from megatron.training.datasets.fim_dataset import GPTFIMDataset
from megatron.training.datasets.fim_dataset import GPTFIMDatasetConfig as RuntimeGPTFIMDatasetConfig
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


def _core_values(config):
    return {f.name: getattr(config, f.name) for f in fields(MCoreGPTDatasetConfig)}


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


# The class: Megatron Core's fields, deferred initialization.


def test_same_fields_as_megatron_core_and_no_eager_post_init():
    cfg = _dataset()
    assert isinstance(cfg, MCoreGPTDatasetConfig)
    assert [f.name for f in fields(cfg) if f.init] == [
        f.name for f in fields(MCoreGPTDatasetConfig) if f.init
    ]
    # No field of its own: names, order and defaults are Core's (random_seed stays required).
    assert [(f.name, f.default) for f in fields(GPTDatasetConfig)] == [
        (f.name, f.default) for f in fields(MCoreGPTDatasetConfig)
    ]
    with pytest.raises(TypeError, match="random_seed"):
        GPTDatasetConfig(sequence_length=16, **BOOLS)
    assert cfg.tokenizer is None and cfg.split_matrix is None
    assert cfg.mock is False and cfg.token_dtype_code is None
    assert not cfg.is_prepared
    unset = _dataset()
    assert unset.random_seed == 1234 and unset.context_parallel_size is None
    # The fields the container writes keep Megatron Core's fields and defaults.
    assert (unset.data_parallel_size, unset.sequence_parallel_size) == (1, 0)
    assert unset.hybrid_context_parallel is False
    assert unset.multiple_validation_sets is None and unset.full_validation is None


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        dict(blend=BLEND, split="9,1,0"),
        dict(blend_per_split=[(["a"], None), None, (["b", "c"], [1.0, 2.0])]),
        dict(blend_per_split=[None, None, None]),
    ],
)
def test_prepare_runs_megatron_core_post_init_once(tokenizer, overrides):
    cfg = _dataset(**overrides)
    cfg.validate()
    cfg.prepare(tokenizer)
    expected = MCoreGPTDatasetConfig(
        sequence_length=16, random_seed=1234, tokenizer=tokenizer, **BOOLS, **overrides
    )
    assert _core_values(cfg) == _core_values(expected)
    assert cfg.is_prepared and cfg.tokenizer is tokenizer
    assert cfg.mock is (not has_data_source(expected))
    assert cfg.token_dtype_code == 8  # split_matrix follows Core: None under blend_per_split
    derived = (cfg.split_matrix, cfg.token_dtype_code)
    cfg.prepare(tokenizer)  # same tokenizer: no-op
    assert (cfg.split_matrix, cfg.token_dtype_code) == derived
    with pytest.raises(RuntimeError, match="another tokenizer"):
        cfg.prepare(SimpleNamespace(vocab_size=2048))


def test_prepare_needs_a_tokenizer(tokenizer):
    with pytest.raises(ValueError, match="needs a tokenizer"):
        _dataset().prepare(None)
    unset = _dataset(random_seed=7)
    assert not unset.is_prepared
    unset.prepare(tokenizer)
    assert unset.random_seed == 7
    # Layout fields hold Megatron Core's defaults unless set; set values are kept.
    assert (unset.data_parallel_size, unset.sequence_parallel_size) == (1, 0)
    assert unset.hybrid_context_parallel is False and unset.context_parallel_size is None
    explicit = _dataset(
        data_parallel_size=4, sequence_parallel_size=2, hybrid_context_parallel=True
    )
    explicit.prepare(tokenizer)
    assert (explicit.data_parallel_size, explicit.sequence_parallel_size) == (4, 2)
    assert explicit.hybrid_context_parallel is True


def test_prepare_applies_megatron_core_checks(tokenizer):
    # Both rules live in Megatron Core's __post_init__; validate() announces the first one early.
    sbhd = _dataset(
        blend=BLEND, split="9,1,0", varlen_sbhd_validation=True, hybrid_context_parallel=True
    )
    with pytest.raises(ValueError, match="hybrid_context_parallel"):
        sbhd.validate()
    with pytest.raises(AssertionError):
        sbhd.prepare(tokenizer)
    assert not sbhd.is_prepared and sbhd.tokenizer is None  # a failed prepare() binds nothing
    counts = _dataset(blend=BLEND, split="9,1,0", sequences_per_dataset={"prefix-a": [10, 2]})
    counts.validate()
    with pytest.raises(AssertionError):
        counts.prepare(SimpleNamespace(vocab_size=None))
    assert not counts.is_prepared and counts.tokenizer is None
    counts.prepare(tokenizer)
    assert counts.is_prepared and counts.token_dtype_code == 8


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(sequence_length=0), "sequence_length"),
        (dict(sequence_length=None), "sequence_length"),
        (dict(reset_position_ids=None), "reset_position_ids"),
        (dict(eod_mask_loss=None), "eod_mask_loss"),
        (dict(num_dataset_builder_threads=0), "num_dataset_builder_threads"),
        (dict(fast_cache_load=True), "fast_cache_load requires path_to_cache"),
        (dict(fast_cache_load=True, path_to_cache="/c", blend=BLEND, split="1,1,1"), "cannot be used with blend"),
        (dict(defer_npy_index_mmap=True), "defer_npy_index_mmap requires"),
        (dict(blend=BLEND), "split must be provided"),
        (dict(blend=(["a", "b"], [1.0]), split="1,1,1"), "equal in number"),
        (dict(blend=BLEND, split="1,1,1", blend_per_split=[(["a"], None), None, None]), "incompatible"),
        (dict(blend_per_split=[(["a"], None), None, None], split="1,1,1"), "split and blend_per_split"),
        (dict(blend_per_split=[(["a"], None), None]), "3 blends"),
        (dict(blend_per_split=[(["a", "b"], [1.0]), None, None]), "equal in number"),
        (dict(varlen_sbhd_validation=True, hybrid_context_parallel=True), "hybrid_context_parallel"),
    ],
)  # fmt: skip
def test_validate_rejects_what_megatron_core_rejects(overrides, message):
    cfg = _dataset(**overrides)
    before = copy.deepcopy(dataclasses.asdict(cfg))
    with pytest.raises(ValueError, match=message):
        cfg.validate()
    assert dataclasses.asdict(cfg) == before


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        dict(blend=BLEND, split="9,1,0"),
        dict(blend=(["a"], None), split="1,0,0"),
        dict(blend_per_split=[(["a"], None), (["b"], None), None]),
        dict(blend_per_split=[None, None, None]),
        dict(fast_cache_load=True, defer_npy_index_mmap=True, path_to_cache="/c", blend_per_split=[(["a"], None), None, None]),
        dict(varlen_sbhd_validation=True, blend=BLEND, split="9,1,0"),
    ],
)  # fmt: skip
def test_validate_accepts_valid_sections(overrides):
    _dataset(**overrides).validate()


def test_a_bound_tokenizer_means_prepared(tokenizer):
    """Core's invariant: an object holding a tokenizer has run Core's post-init. A tokenizer given
    to the constructor is bound through prepare(); dataclasses.replace() of a prepared object runs
    Core's post-init on the copy, as for Core's class; deepcopy() copies the prepared state."""
    eager = _dataset(blend=BLEND, split="9,1,0", tokenizer=tokenizer)
    assert eager.is_prepared and eager.tokenizer is tokenizer and eager.token_dtype_code == 8
    expected = MCoreGPTDatasetConfig(
        sequence_length=16,
        random_seed=1234,
        tokenizer=tokenizer,
        blend=BLEND,
        split="9,1,0",
        **BOOLS,
    )
    assert _core_values(eager) == _core_values(expected)
    with pytest.raises(TypeError, match="random_seed"):  # Core's requirement, unchanged
        GPTDatasetConfig(sequence_length=16, tokenizer=tokenizer, **BOOLS)
    cfg = _dataset()
    cfg.prepare(tokenizer)
    copied = dataclasses.replace(cfg, blend=BLEND, split="9,1,0")
    assert copied.is_prepared and copied.tokenizer is tokenizer
    assert copied.mock is False and copied.split_matrix == expected.split_matrix
    assert not dataclasses.replace(_dataset()).is_prepared
    assert copy.deepcopy(cfg).is_prepared


# The FIM subclass.


def test_fim_subclass_carries_the_runtime_fim_fields(tokenizer):
    cfg = _fim()
    assert isinstance(cfg, RuntimeGPTFIMDatasetConfig) and isinstance(cfg, GPTDatasetConfig)
    assert [f.name for f in fields(cfg) if f.name.startswith("fim_")] == [
        "fim_rate", "fim_spm_rate", "fim_extra_tokens", "fim_split_sample", "fim_fragment_rate", "fim_no_prefix",
    ]  # fmt: skip
    assert not any(f.name.startswith("fim_") for f in fields(GPTDatasetConfig))
    cfg.validate()
    with pytest.raises(ValueError, match="fim_rate"):
        _fim(fim_rate=0).validate()
    with pytest.raises(ValueError, match="fim_spm_rate"):
        _fim(fim_spm_rate=None).validate()
    with pytest.raises(ValueError, match="FIM extra tokens"):
        _fim(fim_extra_tokens={"prefix": "<p>"}).validate()
    with pytest.raises(ValueError, match="FIM extra tokens"):
        _fim(fim_extra_tokens=None).validate()
    cfg.prepare(tokenizer)
    expected = RuntimeGPTFIMDatasetConfig(
        sequence_length=16,
        random_seed=1234,
        tokenizer=tokenizer,
        blend=BLEND,
        split="9,1,0",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        fim_extra_tokens=dict(FIM_TOKENS),
        **BOOLS,
    )
    names = [f.name for f in fields(RuntimeGPTFIMDatasetConfig)]
    assert {n: getattr(cfg, n) for n in names} == {n: getattr(expected, n) for n in names}


# Serialization.


def test_container_written_fields_are_the_layout_and_the_validation_flags():
    """The six fields other sections own, left at Megatron Core's defaults until the container
    writes them; the seed is the section's own setting, not one of them."""
    assert CONTAINER_WRITTEN_FIELDS == (
        "data_parallel_size", "context_parallel_size", "sequence_parallel_size",
        "hybrid_context_parallel", "multiple_validation_sets", "full_validation",
    )  # fmt: skip
    defaults = {f.name: f.default for f in fields(MCoreGPTDatasetConfig)}
    assert {name: defaults[name] for name in CONTAINER_WRITTEN_FIELDS} == {
        "data_parallel_size": 1, "context_parallel_size": None, "sequence_parallel_size": 0,
        "hybrid_context_parallel": False, "multiple_validation_sets": None, "full_validation": None,
    }  # fmt: skip
    assert "random_seed" not in CONTAINER_WRITTEN_FIELDS


def test_to_cfg_dict_omits_tokenizer_owned_and_derived_fields(tokenizer):
    cfg = _dataset(blend=BLEND, split="9,1,0", data_parallel_size=4, full_validation=True)
    before = cfg.to_cfg_dict()
    assert before["_target_"] == "megatron.training.config.dataset_config.GPTDatasetConfig"
    assert "tokenizer" not in before and not (set(CONTAINER_WRITTEN_FIELDS) & set(before))
    assert before["random_seed"] == 1234  # the seed is saved with the section
    assert not {"split_matrix", "mock", "token_dtype_code"} & set(before)
    assert before["blend"] == [["prefix-a", "prefix-b"], [0.3, 0.7]] and before["split"] == "9,1,0"
    cfg.prepare(tokenizer)
    after = cfg.to_cfg_dict()
    assert after == before  # the real-data case derives nothing into a constructor field
    restored = instantiate(
        sanitize_dataclass_config(copy.deepcopy(after)), mode=InstantiationMode.STRICT
    )
    assert type(restored) is GPTDatasetConfig and not restored.is_prepared
    assert restored.tokenizer is None and restored.data_parallel_size == 1
    assert restored.full_validation is None
    assert restored.random_seed == 1234
    assert restored.to_cfg_dict() == after
    fim = _fim().to_cfg_dict()
    assert (
        fim["_target_"].endswith(".GPTFIMDatasetConfig") and fim["fim_extra_tokens"] == FIM_TOKENS
    )
    assert type(instantiate(fim, mode=InstantiationMode.STRICT)) is GPTFIMDatasetConfig
    # A mock run records the split Megatron Core chose: the saved file is a snapshot of the
    # effective inputs.
    mock = _dataset()
    mock.prepare(tokenizer)
    assert mock.to_cfg_dict()["split"] == "1,1,1"


def test_data_section_round_trips_through_yaml_with_the_class():
    data = DataConfig(
        dataset=_dataset(blend=BLEND, split="9,1,0"),
        dataloader=DataloaderConfig(num_workers=0),
        dataset_type=GPTDataset,
    )
    as_dict = ConfigContainerBase._convert_value_to_dict(data)
    with safe_yaml_representers():
        text = yaml.safe_dump(as_dict, default_flow_style=False)
    assert "_call_: false" in text and "gpt_dataset.GPTDataset" in text
    loaded = sanitize_dataclass_config(yaml.safe_load(text))
    restored = instantiate(loaded, mode=InstantiationMode.STRICT)
    assert type(restored) is DataConfig and restored.dataset_type is GPTDataset
    assert restored.dataloader == data.dataloader
    assert type(restored.dataset) is GPTDatasetConfig and restored.dataset.tokenizer is None
    assert restored.dataset.to_cfg_dict() == data.dataset.to_cfg_dict()
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


def test_finalize_commits_the_class_and_dataset_build_needs_a_prepared_config(tokenizer):
    data = DataConfig(dataset=_dataset(), dataloader=DataloaderConfig(dataloader_type="cyclic"))
    data.validate()
    assert data.dataset_type is None  # validation does not write
    data.finalize()
    assert data.dataset_type is MockGPTDataset and data.dataloader.dataloader_type == "cyclic"
    with pytest.raises(RuntimeError, match="not prepared"):
        data.dataset_build()
    data.dataset.prepare(tokenizer)
    dataset_type, config, packed = data.dataset_build()
    assert dataset_type is MockGPTDataset and config is data.dataset and packed is False
    data.finalize()  # idempotent
    assert data.dataset_type is MockGPTDataset
    sft = DataConfig(dataset=_dataset(), dataset_type=MockSFTDataset)
    sft.validate()
    sft.finalize()
    sft.dataset.prepare(tokenizer)
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
