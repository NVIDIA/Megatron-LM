# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native CPU dataset tests. Run from the repository root with:

python -m pytest --confcutdir=tests/unit_tests/data tests/unit_tests/data/test_downsampling.py

Only synthetic indexed sequences are used. No process group or GPU is initialized.
"""

import hashlib
import json
import tempfile
from argparse import ArgumentParser
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest
import torch
from filelock import FileLock

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    MockGPTDataset,
    MockGPTLowLevelDataset,
)
from megatron.core.datasets.gpt_fim_dataset import GPTFIMDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset
from megatron.core.datasets.utils import Split, compile_helpers, get_blend_from_list
from megatron.core.tokenizers import MegatronTokenizer
from megatron.training.argument_utils import (
    core_gpt_dataset_config_from_args,
    gpt_data_config_from_args,
)
from megatron.training.arguments import add_megatron_arguments
from megatron.training.utils import get_blend_and_blend_per_split

_MOCK_VOCAB_SIZE = 8192


class BuilderMockGPTDataset(MockGPTDataset):
    def __init__(
        self,
        indexed_dataset,
        dataset_path,
        indexed_indices,
        num_samples,
        index_split,
        config,
        train_data_down_sampling_prob=None,
    ) -> None:
        GPTDataset.__init__(
            self,
            indexed_dataset,
            dataset_path,
            indexed_indices,
            num_samples,
            index_split,
            config,
            train_data_down_sampling_prob,
        )


@pytest.fixture(scope="module")
def tokenizer():
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        # Several CI workers can import this module before initializing a process group.
        # Serialize the shared extension build without creating a process group for CPU tests.
        key = hashlib.sha256(str(Path(__file__).resolve()).encode()).hexdigest()
        with FileLock(str(Path(tempfile.gettempdir()) / f"megatron-dataset-{key}.lock")):
            compile_helpers()
    return MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=_MOCK_VOCAB_SIZE
    )


@pytest.fixture(autouse=True)
def local_dataset_build(monkeypatch, tokenizer):
    """Exercise native local dataset construction with independent caches on every CI rank."""
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


@pytest.fixture
def cache_path(tmp_path):
    return str(tmp_path)


def make_config(tokenizer, **kwargs):
    config_kwargs = {
        "random_seed": 1234,
        "sequence_length": 8,
        "split": "1,0,0",
        "reset_position_ids": False,
        "reset_attention_mask": False,
        "eod_mask_loss": False,
        "tokenizer": tokenizer,
    }
    config_kwargs.update(kwargs)
    return GPTDatasetConfig(**config_kwargs)


def make_dataset(tokenizer, monkeypatch, probability=None, seed=1234, path_to_cache=None):
    monkeypatch.setattr(MockGPTLowLevelDataset, "size", 100)
    monkeypatch.setattr(MockGPTLowLevelDataset, "max_sequence_length", 16)
    low_level_dataset = MockGPTLowLevelDataset(tokenizer)
    indices = numpy.arange(len(low_level_dataset), dtype=numpy.int32)
    config = make_config(tokenizer, path_to_cache=path_to_cache, train_data_down_sampling_seed=seed)
    dataset = MockGPTDataset(
        low_level_dataset, "/data/a", indices, 10, Split.train, config, probability
    )
    return dataset, indices


def make_data_args(**kwargs):
    values = {
        "data_path": None,
        "data_args_path": None,
        "train_data_path": None,
        "valid_data_path": None,
        "test_data_path": None,
        "per_split_data_args_path": None,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_weighted_blend_remains_supported():
    prefixes, weights = get_blend_from_list(["0.25", "/data/a", "0.75", "/data/b"])

    assert prefixes == ["/data/a", "/data/b"]
    assert weights == [0.25, 0.75]


def test_per_source_triple_blend():
    blend = get_blend_from_list(
        ["0.25", "/data/a", "0.5", "0.75", "/data/b", "1.0"], allow_per_source_downsampling=True
    )

    assert blend == (["/data/a", "/data/b"], [0.25, 0.75], [0.5, 1.0])


def test_numeric_json_weights_and_probabilities():
    blend = get_blend_from_list(
        [1, "/data/a", 0.5, 2.0, "/data/b", 1], allow_per_source_downsampling=True
    )
    assert blend == (["/data/a", "/data/b"], [1.0, 2.0], [0.5, 1.0])
    assert get_blend_from_list([1, "/data/a", 2.0, "/data/b"]) == (
        ["/data/a", "/data/b"],
        [1.0, 2.0],
    )
    assert get_blend_from_list(["123"]) == (["123"], None)
    with pytest.raises(ValueError):
        get_blend_from_list([1, 2, 0.5], allow_per_source_downsampling=True)


def test_per_source_triple_blend_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="not enabled"):
        get_blend_from_list(["0.25", "/data/a", "0.5", "0.75", "/data/b", "1.0"])


@pytest.mark.parametrize(
    "blend",
    [
        ["0.25", "/data/a", "0.5", "0.75", "/data/b"],
        ["0.25", "/data/a", "bad-probability"],
        ["0.25", "/data/a", "1.1"],
        ["0.25", "/data/a", "0.0"],
        ["0.25", "/data/a", "nan"],
        ["0.25", "/data/a", "inf"],
        ["0.25", "/data/a", "not-a-weight", "/data/b"],
        ["-0.25", "/data/a", "1.25", "/data/b"],
        ["0.0", "/data/a", "0.0", "/data/b"],
        ["0.0", "/data/a", "0.5", "0.0", "/data/b", "0.5"],
    ],
)
def test_malformed_or_mixed_weighted_blends_raise(blend):
    with pytest.raises(ValueError):
        get_blend_from_list(blend)


def test_unweighted_prefix_list_remains_supported():
    assert get_blend_from_list(["/data/a", "/data/b"]) == (["/data/a", "/data/b"], None)


def test_train_blend_parsing_forwards_per_source_probabilities():
    args = make_data_args(
        train_data_path=["0.25", "/data/a", "0.5", "0.75", "/data/b", "1.0"],
        valid_data_path=["/data/valid"],
    )

    blend, blend_per_split = get_blend_and_blend_per_split(args, allow_per_source_downsampling=True)

    assert blend is None
    assert blend_per_split == [
        (["/data/a", "/data/b"], [0.25, 0.75], [0.5, 1.0]),
        (["/data/valid"], None),
        None,
    ]


def test_train_downsampling_probability_override_converts_pairs_to_triples():
    args = make_data_args(
        train_data_path=["0.25", "/data/a", "0.75", "/data/b"],
        train_data_down_sampling_prob_override=0.4,
    )

    _, blend_per_split = get_blend_and_blend_per_split(args, allow_per_source_downsampling=True)

    assert blend_per_split == [(["/data/a", "/data/b"], [0.25, 0.75], [0.4, 0.4]), None, None]


def test_train_downsampling_probability_override_replaces_triple_probabilities():
    args = make_data_args(
        train_data_path=["0.25", "/data/a", "0.5", "0.75", "/data/b", "1.0"],
        train_data_down_sampling_prob_override=0.4,
    )

    _, blend_per_split = get_blend_and_blend_per_split(args, allow_per_source_downsampling=True)

    assert blend_per_split == [(["/data/a", "/data/b"], [0.25, 0.75], [0.4, 0.4]), None, None]


@pytest.mark.parametrize("paths", [{}, {"valid_data_path": ["/data/valid"]}])
@pytest.mark.parametrize("probability", [0.4, float("nan")])
def test_override_requires_valid_probability_and_train_blend(paths, probability):
    args = make_data_args(**paths, train_data_down_sampling_prob_override=probability)
    with pytest.raises(ValueError):
        get_blend_and_blend_per_split(args, allow_per_source_downsampling=True)


@pytest.mark.parametrize("probability", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_train_downsampling_probability_override_rejects_invalid_probability(probability):
    args = make_data_args(
        train_data_path=["0.25", "/data/a", "0.75", "/data/b"],
        train_data_down_sampling_prob_override=probability,
    )

    with pytest.raises(ValueError, match="must be finite and in"):
        get_blend_and_blend_per_split(args, allow_per_source_downsampling=True)


def test_json_train_blend_parsing_forwards_per_source_probabilities(tmp_path):
    data_args_path = tmp_path / "blend.json"
    data_args_path.write_text(
        json.dumps(
            {
                "train": ["0.25", "/data/a", "0.5", "0.75", "/data/b", "1.0"],
                "valid": ["/data/valid"],
                "test": None,
            }
        )
    )

    blend, blend_per_split = get_blend_and_blend_per_split(
        make_data_args(per_split_data_args_path=str(data_args_path)),
        allow_per_source_downsampling=True,
    )

    assert blend is None
    assert blend_per_split == [
        (["/data/a", "/data/b"], [0.25, 0.75], [0.5, 1.0]),
        (["/data/valid"], None),
        None,
    ]


@pytest.mark.parametrize("probability", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_gpt_dataset_rejects_invalid_probability(tokenizer, monkeypatch, probability):
    with pytest.raises(ValueError, match="must be finite and in"):
        make_dataset(tokenizer, monkeypatch, probability)


def test_gpt_dataset_requires_seed_for_downsampling(tokenizer, monkeypatch):
    with pytest.raises(ValueError, match="requires an integer seed"):
        make_dataset(tokenizer, monkeypatch, 0.5, seed=None)


def test_probability_one_preserves_indices_and_legacy_hash(tokenizer, monkeypatch):
    legacy_dataset, legacy_indices = make_dataset(tokenizer, monkeypatch)
    full_dataset, full_indices = make_dataset(tokenizer, monkeypatch, 1.0)

    assert legacy_dataset.indices is legacy_indices
    assert full_dataset.indices is full_indices
    assert legacy_dataset.unique_description_hash == full_dataset.unique_description_hash
    assert "train_data_down_sampling_prob" not in full_dataset.unique_identifiers
    assert "train_data_down_sampling_seed" not in full_dataset.unique_identifiers


def test_downsampling_is_deterministic_and_part_of_cache_identity(tokenizer, monkeypatch):
    dataset, indices = make_dataset(tokenizer, monkeypatch, 0.4)
    repeated_dataset, _ = make_dataset(tokenizer, monkeypatch, 0.4)
    other_probability_dataset, _ = make_dataset(tokenizer, monkeypatch, 0.5)
    other_seed_dataset, _ = make_dataset(tokenizer, monkeypatch, 0.4, seed=4321)

    expected_indices = numpy.random.RandomState(1234).permutation(indices)[
        : ceil(0.4 * len(indices))
    ]
    assert numpy.array_equal(dataset.indices, expected_indices)
    assert dataset.indices.flags.owndata
    assert dataset.indices.base is None
    assert numpy.array_equal(dataset.indices, repeated_dataset.indices)
    assert not numpy.array_equal(dataset.indices, other_seed_dataset.indices)
    assert dataset.unique_identifiers["train_data_down_sampling_prob"] == 0.4
    assert dataset.unique_identifiers["train_data_down_sampling_seed"] == 1234
    assert dataset.unique_description_hash == repeated_dataset.unique_description_hash
    assert dataset.unique_description_hash != other_probability_dataset.unique_description_hash
    assert dataset.unique_description_hash != other_seed_dataset.unique_description_hash


def test_same_seed_produces_nested_document_pools(tokenizer, monkeypatch):
    smaller, _ = make_dataset(tokenizer, monkeypatch, 0.24)
    middle, _ = make_dataset(tokenizer, monkeypatch, 0.45)
    larger, _ = make_dataset(tokenizer, monkeypatch, 0.8)

    assert numpy.array_equal(smaller.indices, middle.indices[: len(smaller.indices)])
    assert numpy.array_equal(middle.indices, larger.indices[: len(middle.indices)])
    assert len(numpy.unique(larger.indices)) == len(larger.indices)


@pytest.mark.parametrize("split", ["valid", "test"])
def test_per_split_parser_rejects_downsampling_outside_train(split):
    args = make_data_args(**{f"{split}_data_path": ["1.0", "/data/eval", "0.5"]})

    with pytest.raises(ValueError, match="not enabled"):
        get_blend_and_blend_per_split(args, allow_per_source_downsampling=True)


def test_single_source_builder_forwards_probability(tokenizer, monkeypatch, cache_path):
    monkeypatch.setattr(MockGPTLowLevelDataset, "size", 100)
    monkeypatch.setattr(MockGPTLowLevelDataset, "max_sequence_length", 16)
    config = make_config(
        tokenizer,
        split=None,
        path_to_cache=cache_path,
        train_data_down_sampling_seed=1234,
        blend_per_split=[(["/data/a"], [1.0], [0.4]), None, None],
    )

    train_dataset, _, _ = BlendedMegatronDatasetBuilder(
        BuilderMockGPTDataset, [10, 0, 0], lambda: True, config
    ).build()

    assert train_dataset.train_data_down_sampling_prob == 0.4
    assert len(numpy.unique(train_dataset.document_index)) == 40


def test_cache_hit_preserves_selected_documents(tokenizer, monkeypatch, cache_path):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        make_dataset(tokenizer, monkeypatch, 0.4, path_to_cache=cache_path)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    dataset, _ = make_dataset(tokenizer, monkeypatch, 0.4, path_to_cache=cache_path)
    repeated_dataset, _ = make_dataset(tokenizer, monkeypatch, 0.4, path_to_cache=cache_path)

    assert numpy.array_equal(dataset.indices, repeated_dataset.indices)
    assert len(repeated_dataset.indices) == 40
    assert numpy.array_equal(dataset.document_index, repeated_dataset.document_index)
    assert numpy.array_equal(dataset.sample_index, repeated_dataset.sample_index)
    assert numpy.array_equal(dataset.shuffle_index, repeated_dataset.shuffle_index)
    assert dataset._get_num_tokens_per_epoch() == repeated_dataset._get_num_tokens_per_epoch()


def test_fast_cache_load_uses_downsampled_indices(tokenizer, monkeypatch, cache_path):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        make_dataset(tokenizer, monkeypatch, 0.4, path_to_cache=cache_path)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    dataset, _ = make_dataset(tokenizer, monkeypatch, 0.4, path_to_cache=cache_path)
    config = make_config(
        tokenizer,
        path_to_cache=cache_path,
        train_data_down_sampling_seed=1234,
        fast_cache_load=True,
    )
    cached_dataset = MockGPTDataset(dataset.dataset, "/data/a", None, 10, Split.train, config, 0.4)

    assert cached_dataset.indices is None
    assert dataset.unique_description_hash == cached_dataset.unique_description_hash
    assert numpy.array_equal(dataset.document_index, cached_dataset.document_index)
    assert numpy.array_equal(dataset.sample_index, cached_dataset.sample_index)
    assert numpy.array_equal(dataset.shuffle_index, cached_dataset.shuffle_index)


def test_multi_source_builder_forwards_probabilities(tokenizer, monkeypatch, cache_path):
    monkeypatch.setattr(MockGPTLowLevelDataset, "size", 100)
    monkeypatch.setattr(MockGPTLowLevelDataset, "max_sequence_length", 16)
    config = make_config(
        tokenizer,
        split=None,
        path_to_cache=cache_path,
        train_data_down_sampling_seed=1234,
        blend_per_split=[(["/data/a", "/data/b"], [0.5, 0.5], [0.4, 0.6]), None, None],
    )

    train_dataset, _, _ = BlendedMegatronDatasetBuilder(
        BuilderMockGPTDataset, [10, 0, 0], lambda: True, config
    ).build()

    assert [dataset.train_data_down_sampling_prob for dataset in train_dataset.datasets] == [
        0.4,
        0.6,
    ]
    assert [len(numpy.unique(dataset.document_index)) for dataset in train_dataset.datasets] == [
        40,
        60,
    ]


def test_global_triple_blend_is_rejected_by_builder(tokenizer):
    config = make_config(tokenizer, blend=(["/data/a"], [1.0], [0.5]))
    with pytest.raises(ValueError, match="per-split train blend"):
        BlendedMegatronDatasetBuilder(MockGPTDataset, [10, 0, 0], lambda: True, config)


def test_non_train_triple_blend_is_rejected_by_builder(tokenizer):
    config = make_config(
        tokenizer, split=None, blend_per_split=[None, (["/data/a"], [1.0], [0.5]), None]
    )
    with pytest.raises(ValueError, match="only supported for train"):
        BlendedMegatronDatasetBuilder(MockGPTDataset, [0, 10, 0], lambda: True, config)


def test_config_rejects_probability_count_mismatch(tokenizer):
    with pytest.raises(ValueError, match="equal lengths"):
        make_config(
            tokenizer,
            split=None,
            blend_per_split=[(["/data/a", "/data/b"], [0.5, 0.5], [0.5]), None, None],
        ).validate()


def test_config_rejects_invalid_probability(tokenizer):
    with pytest.raises(ValueError, match=r"must be finite and in \(0, 1\]"):
        make_config(
            tokenizer, split=None, blend_per_split=[(["/data/a"], [1.0], [0.0]), None, None]
        ).validate()


def test_config_rejects_triple_without_probabilities(tokenizer):
    with pytest.raises(ValueError, match="equal lengths"):
        make_config(
            tokenizer, split=None, blend_per_split=[(["/data/a"], [1.0], None), None, None]
        ).validate()


def test_config_rejects_blend_with_more_than_three_elements(tokenizer):
    with pytest.raises(ValueError, match="must contain prefixes, weights"):
        make_config(
            tokenizer,
            split=None,
            blend_per_split=[(["/data/a"], [1.0], [0.5], "extra"), None, None],
        ).validate()


@pytest.mark.parametrize("probability", [0.5, 1.0])
def test_non_gpt_dataset_rejects_triple_blend(tokenizer, probability):
    config = make_config(
        tokenizer, split=None, blend_per_split=[(["/data/a"], [1.0], [probability]), None, None]
    )

    with pytest.raises(ValueError, match="requires a GPT dataset"):
        BlendedMegatronDatasetBuilder(MegatronDataset, [10, 0, 0], lambda: True, config)


def test_fim_dataset_forwards_downsampling_probability(monkeypatch):
    forwarded = {}

    def gpt_dataset_init(
        self,
        indexed_dataset,
        dataset_path,
        indexed_indices,
        num_samples,
        index_split,
        config,
        train_data_down_sampling_prob=None,
    ):
        self.config = config
        forwarded["probability"] = train_data_down_sampling_prob

    class Tokenizer:
        @staticmethod
        def tokens_to_ids(tokens):
            return list(range(len(tokens)))

    config = SimpleNamespace(
        random_seed=1234,
        fim_rate=0.5,
        fim_spm_rate=0.5,
        fim_fragment_rate=None,
        fim_split_sample=None,
        fim_no_prefix=None,
        fim_extra_tokens={
            "prefix": "prefix",
            "middle": "middle",
            "suffix": "suffix",
            "pad": "pad",
            "eod": "eod",
        },
        tokenizer=SimpleNamespace(_tokenizer=Tokenizer()),
    )
    monkeypatch.setattr(GPTDataset, "__init__", gpt_dataset_init)

    GPTFIMDataset(None, "/data/a", numpy.arange(1), 1, Split.train, config, 0.5)

    assert forwarded["probability"] == 0.5


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True])
def test_invalid_seed_is_rejected_before_building(tokenizer, seed):
    config = make_config(tokenizer, train_data_down_sampling_seed=seed)
    with pytest.raises(ValueError, match="integer in"):
        config.validate_without_tokenizer()


def _native_args(*extra):
    parser = ArgumentParser()
    add_megatron_arguments(parser)
    return parser.parse_args(
        ["--seq-length", "8", "--train-data-path", "1", "/data/a", "0.4", *extra]
    )


def test_central_factory_forwards_seed_and_probabilities():
    args = _native_args("--train-data-down-sampling-seed", "42")
    config = gpt_data_config_from_args(args).dataset
    assert config.train_data_down_sampling_seed == 42
    assert config.blend_per_split[0] == (["/data/a"], [1.0], [0.4])


@pytest.mark.parametrize("flag", ["--sft", "--use-varlen-dataset"])
def test_non_gpt_families_reject_downsampling_at_factory(flag):
    with pytest.raises(ValueError, match="not enabled"):
        gpt_data_config_from_args(_native_args(flag))


@pytest.mark.parametrize("flag", ["--logits-save-dir", "--logits-load-dir"])
def test_cached_logits_reject_downsampling_even_when_hash_checks_disabled(flag):
    args = _native_args(flag, "/unused/logits", "--logits-load-ignore-hash")
    with pytest.raises(ValueError, match="Cached-logit distillation"):
        gpt_data_config_from_args(args)


def test_legacy_namespace_without_new_options_remains_supported():
    args = _native_args("--train-data-path", "1", "/data/a")
    del args.train_data_down_sampling_seed
    del args.train_data_down_sampling_prob_override
    config = gpt_data_config_from_args(args).dataset
    assert config.train_data_down_sampling_seed == 1234
    assert len(config.blend_per_split[0]) == 2


def test_cache_and_training_factory_share_downsampling_config(tokenizer, monkeypatch):
    from megatron.training import argument_utils
    from tools import prepare_cache

    args = _native_args("--train-data-down-sampling-seed", "9876")
    args.data_parallel_size = 1
    monkeypatch.setattr(argument_utils, "build_tokenizer", lambda _args: tokenizer)
    train = core_gpt_dataset_config_from_args(args, tokenizer=tokenizer)
    cache = prepare_cache.core_gpt_dataset_config_from_args(args)
    assert train.train_data_down_sampling_seed == cache.train_data_down_sampling_seed == 9876
    assert train.blend_per_split == cache.blend_per_split
