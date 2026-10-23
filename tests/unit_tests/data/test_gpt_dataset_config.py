# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron Core's GPT dataset config completes itself in validate(), not in __post_init__: it can be
constructed before its tokenizer exists, and BlendedMegatronDatasetBuilder validates it."""

from types import SimpleNamespace

import pytest

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.gpt_fim_dataset import GPTFIMDatasetConfig
from megatron.core.tokenizers import MegatronTokenizer

BOOLS = dict(reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False)
BLEND = (["prefix-a", "prefix-b"], [0.3, 0.7])
FIM_TOKENS = {"prefix": "<p>", "middle": "<m>", "suffix": "<s>", "pad": "<x>", "eod": "<e>"}


@pytest.fixture(scope="module")
def tokenizer():
    return MegatronTokenizer.from_pretrained(
        metadata_path={"library": "null-text"}, vocab_size=2048
    )


def _config(**overrides):
    kwargs = dict(random_seed=1234, sequence_length=16, **BOOLS)
    kwargs.update(overrides)
    return GPTDatasetConfig(**kwargs)


def _base(**overrides):
    return BlendedMegatronDatasetConfig(random_seed=1234, sequence_length=16, **overrides)


def test_construction_needs_no_tokenizer_and_derives_the_sources():
    """The blend / split rules need no tokenizer and run at construction as before; mock and
    split_matrix follow from the sources (read-only properties, nothing is assigned); what needs
    the tokenizer waits for validate()."""
    mock = _config()
    assert mock.tokenizer is None and mock.token_dtype_code is None
    assert mock.mock is True and mock.split is None  # the even split is derived, not written
    assert mock.split_matrix == _base().split_matrix == _base(split="1,1,1").split_matrix
    assert not {"mock", "split_matrix", "token_dtype_code"} & set(
        GPTDatasetConfig.__dataclass_fields__
    )
    for name, value in (("mock", False), ("split_matrix", None), ("token_dtype_code", 4)):
        with pytest.raises(AttributeError):
            setattr(mock, name, value)
    real = _config(blend=BLEND, split="9,1,0")
    assert (
        real.mock is False and real.split_matrix == _base(blend=BLEND, split="9,1,0").split_matrix
    )
    per_split = _config(blend_per_split=[(["a"], None), None, (["b"], None)])
    assert per_split.mock is False and per_split.split_matrix is None
    with pytest.raises(AssertionError, match="split must be provided"):
        _config(blend=BLEND)
    with pytest.raises(AssertionError, match="incompatible"):
        _config(blend=BLEND, split="9,1,0", blend_per_split=[(["a"], None), None, None])
    with pytest.raises(AssertionError, match="data-cache-path"):
        _config(fast_cache_load=True)
    with pytest.raises(TypeError, match="random_seed"):
        GPTDatasetConfig(sequence_length=16, **BOOLS)


def test_validate_needs_the_tokenizer_and_asserts_only(tokenizer):
    """What __post_init__ asserted before: run once the config is complete. Assertions only and
    idempotent; token_dtype_code follows the tokenizer and mock / split_matrix follow the sources,
    so a config completed after construction needs no refresh."""
    cfg = _config()
    with pytest.raises(AssertionError):
        cfg.validate()  # no tokenizer
    cfg.tokenizer = tokenizer
    assert cfg.token_dtype_code == 8  # from the tokenizer, before any validate()
    cfg.validate()
    cfg.validate()
    assert cfg.token_dtype_code == 8 and cfg.mock is True
    cfg.blend, cfg.split = BLEND, "9,1,0"  # completed after construction
    assert cfg.mock is False and cfg.split_matrix == _base(blend=BLEND, split="9,1,0").split_matrix
    cfg.validate()
    for missing in ("reset_position_ids", "reset_attention_mask", "eod_mask_loss"):
        with pytest.raises(AssertionError):
            _config(tokenizer=tokenizer, **{missing: None}).validate()
    sbhd = _config(tokenizer=tokenizer, varlen_sbhd_validation=True, hybrid_context_parallel=True)
    with pytest.raises(AssertionError, match="varlen-sbhd-validation"):
        sbhd.validate()
    counts = _config(blend=BLEND, split="9,1,0", sequences_per_dataset={"prefix-a": [10, 2]})
    counts.tokenizer = SimpleNamespace(vocab_size=None)
    with pytest.raises(AssertionError, match="vocab size"):
        counts.validate()
    wide = _config(tokenizer=SimpleNamespace(vocab_size=70000))
    wide.validate()
    assert wide.token_dtype_code == 4


def test_the_builder_validates_the_config(tokenizer):
    """A config handed to BlendedMegatronDatasetBuilder is validated there: callers that pass a
    complete config need no explicit validate(), and one bound to its tokenizer after construction
    is completed there."""
    cfg = _config()
    cfg.tokenizer = tokenizer
    builder = BlendedMegatronDatasetBuilder(MockGPTDataset, [8, 8, 8], lambda: True, cfg)
    assert builder.config is cfg and cfg.token_dtype_code == 8
    with pytest.raises(AssertionError):
        BlendedMegatronDatasetBuilder(MockGPTDataset, [8, 8, 8], lambda: True, _config())


def test_fim_config_validates_its_own_fields(tokenizer):
    def fim(**overrides):
        kwargs = dict(
            random_seed=1234,
            sequence_length=16,
            tokenizer=tokenizer,
            fim_rate=0.5,
            fim_spm_rate=0.5,
            fim_extra_tokens=dict(FIM_TOKENS),
            **BOOLS,
        )
        kwargs.update(overrides)
        return GPTFIMDatasetConfig(**kwargs)

    cfg = fim()
    assert isinstance(cfg, GPTDatasetConfig)
    cfg.validate()
    assert cfg.token_dtype_code == 8
    fim(fim_rate=0.0, fim_spm_rate=0.0).validate()  # 0.0 is a rate, not a missing value
    for bad, message in (
        (dict(fim_rate=None), "fim_rate"),
        (dict(fim_spm_rate=None), "fim_spm_rate"),
        (dict(fim_extra_tokens={"prefix": "<p>"}), "FIM extra tokens"),
        (dict(fim_extra_tokens=None), "FIM extra tokens"),
    ):
        with pytest.raises(AssertionError, match=message):
            fim(**bad).validate()
