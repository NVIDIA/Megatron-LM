# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import numpy
import pytest
import torch

from megatron.core.datasets.bert_dataset import (
    BERTMaskedWordPieceDataset,
    BERTMaskedWordPieceDatasetConfig,
)
from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.utils import Split, compile_helpers
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("kind,save_count", [("gpt", 3), ("blend", 2), ("bert", 1)])
def test_index_cache_recovers_after_interrupted_save(tmp_path, monkeypatch, kind, save_count):
    """Rebuild an interrupted cache and read identical samples from a fresh dataset."""
    Utils.initialize_distributed()
    if torch.distributed.get_rank() != 0:
        return
    compile_helpers()
    prefix = str(tmp_path / "tokens")
    builder = IndexedDatasetBuilder(prefix + ".bin", dtype=numpy.int32)
    for offset in range(8):
        builder.add_document([(offset + i) % 96 + 1 for i in range(64)], [16] * 4)
    builder.finalize(prefix + ".idx")
    indexed = IndexedDataset(prefix)
    tokenizer = SimpleNamespace(
        unique_identifiers={"vocab_size": 132},
        vocab_size=132,
        inv_vocab={i: str(i) for i in range(132)},
        eod=127,
        eos=127,
        cls=128,
        sep=129,
        mask=130,
        pad=0,
    )
    cache = tmp_path / "cache"
    common = dict(random_seed=1234, sequence_length=16, tokenizer=tokenizer)
    gpt_config = GPTDatasetConfig(
        **common,
        path_to_cache=str(tmp_path / "gpt"),
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )

    def build_gpt():
        return GPTDataset(
            indexed,
            prefix,
            numpy.arange(len(indexed), dtype=numpy.int32),
            32,
            Split.train,
            gpt_config,
        )

    if kind == "gpt":
        gpt_config.path_to_cache = str(cache)
        build = build_gpt
    elif kind == "blend":
        datasets = [build_gpt(), build_gpt()]
        config = BlendedMegatronDatasetConfig(**common, path_to_cache=str(cache))

        def build():
            return BlendedDataset(datasets, [1, 1], 32, config)

    else:
        config = BERTMaskedWordPieceDatasetConfig(
            **common,
            path_to_cache=str(cache),
            masking_probability=0.15,
            short_sequence_probability=0.1,
            masking_max_ngram=1,
            masking_do_full_word=False,
            masking_do_permutation=False,
            masking_use_longer_ngrams=False,
            masking_use_geometric_distribution=False,
            classification_head=False,
        )

        def build():
            return BERTMaskedWordPieceDataset(
                indexed, prefix, numpy.arange(8, dtype=numpy.int32), 32, Split.train, config
            )

    save = numpy.save
    calls = 0

    def interrupted_save(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        save(path, *args, **kwargs)
        if calls == save_count:
            with open(path, "r+b") as writer:
                writer.truncate(8)
            raise OSError("interrupted index write")

    with monkeypatch.context() as interrupted:
        interrupted.setattr(numpy, "save", interrupted_save)
        with pytest.raises(OSError, match="interrupted index write"):
            build()

    rebuilt = build()
    expected = [rebuilt[i] for i in range(4)]
    loaded = build()
    for index, sample in enumerate(expected):
        actual = loaded[index]
        if actual.keys() != sample.keys():
            pytest.fail("cache reload changed sample fields")
        for name, value in sample.items():
            if not numpy.array_equal(actual[name], value):
                pytest.fail(f"cache reload changed sample {index} field {name}")
    if any(path.is_dir() for path in cache.iterdir()):
        pytest.fail("failed index writes left temporary directories")
    descriptions = list(cache.glob("*description.txt"))
    if len(descriptions) != 1 or descriptions[0].read_text() != rebuilt.unique_description:
        pytest.fail("cache description does not match the rebuilt dataset")
    for path in cache.glob("*.npy"):
        numpy.load(path, mmap_mode="r", allow_pickle=False)
