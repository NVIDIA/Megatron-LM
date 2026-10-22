# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Every GPT entry point builds the dataset config and dataset class pretrain_gpt.py built before.

The reference is the frozen factory and provider logic of commit 438ef0025
(``legacy_dataset_factories.py``), run on the same validated arguments as the new path: ``cfg.data``
built by the one factory, the mirrors written by the container, the config prepared with the run's
tokenizer and handed to the builder as is. pretrain_gpt.py's legacy factory is the oracle for all
four entry points; where the other three legacy factories differed from it (fields they never
forwarded, requests they silently downgraded), the difference is listed here as a documented change.
"""

import ast
import inspect
import json
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig as MCoreGPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset
from megatron.core.datasets.utils import Split, compile_helpers
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training import argument_utils, global_vars, training
from megatron.training.argument_utils import (
    gpt_data_config_from_args,
    pretrain_cfg_container_from_args,
)
from megatron.training.arguments import parse_args, validate_args
from megatron.training.config import GPTDatasetConfig, GPTFIMDatasetConfig
from megatron.training.config import container as container_module
from megatron.training.config.dataset_config import LAYOUT_MIRRORS
from megatron.training.datasets import providers, sft_dataset, varlen_dataset
from megatron.training.datasets.fim_dataset import GPTFIMDataset
from megatron.training.datasets.fim_dataset import GPTFIMDatasetConfig as RuntimeGPTFIMDatasetConfig
from megatron.training.datasets.sft_dataset import MockSFTDataset, SFTDataset
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.training import legacy_dataset_factories as legacy

MODEL_ARGV = (
    "--num-layers 2 --hidden-size 128 --num-attention-heads 8 --micro-batch-size 1 "
    "--global-batch-size 8 --seq-length 32 --max-position-embeddings 32 "
    "--tokenizer-type NullTokenizer --vocab-size 2048"
).split()

# Applied to the validated arguments and mirrored into the container's model and validation
# sections, as gpt_config_from_args would carry them for a real run.
PARALLEL_OVERRIDES = dict(
    context_parallel_size=2,
    tensor_model_parallel_size=2,
    sequence_parallel=True,
    data_parallel_size=4,
    hybrid_context_parallel=True,
    full_validation=True,
    multiple_validation_sets=True,
)
VALIDATION_OVERRIDES = dict(full_validation=True, multiple_validation_sets=True)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("dataset_providers")
    data_args = root / "data_args.txt"
    data_args.write_text("0.3 prefix-a 0.7 prefix-b\n")
    per_split = root / "per_split.json"
    per_split.write_text(
        json.dumps({"train": "prefix-a prefix-b", "valid": ["prefix-c"], "test": "prefix-d"})
    )
    sequences = root / "sequences.json"
    sequences.write_text(json.dumps({"prefix-a": 10, "prefix-b": 20}))
    return {
        "mock": (["--mock-data"], {}),
        "mock_parallel": (["--mock-data"], PARALLEL_OVERRIDES),
        "blend": (["--data-path", "0.3", "prefix-a", "0.7", "prefix-b", "--split", "9,1,0"], {}),
        "blend_default_split": (["--data-path", "prefix-a"], {}),
        "per_split": (
            ["--train-data-path", "prefix-a", "--valid-data-path", "prefix-b", "--test-data-path", "prefix-c"],
            {},
        ),
        "data_args_file": (["--data-args-path", str(data_args), "--split", "9,1,0"], {}),
        "per_split_file": (["--per-split-data-args-path", str(per_split)], {}),
        "sft_mock": (["--sft", "--mock-data"], {}),
        "sft": (["--sft", "--data-path", "prefix-a"], {}),
        "varlen_mock": (
            ["--use-varlen-dataset", "--mock-data", "--calculate-per-token-loss", "--max-seqlen-per-dp-cp-rank", "32"],
            {},
        ),
        "varlen_sbhd": (["--use-varlen-dataset", "--data-path", "prefix-a", "--varlen-sbhd-validation"], {}),
        "fim": (
            ["--fim-data", "--data-path", "prefix-a", "--fim-rate", "0.9", "--fim-spm-rate", "0.4",
             "--fim-split-sample", "<split>", "--fim-fragment-rate", "0.2", "--fim-no-prefix", "nofim",
             "--fim-prefix-token", "<p>", "--fim-eod-token", "<e>"],
            {},
        ),
        "cache": (
            ["--train-data-path", "prefix-a", "--valid-data-path", "prefix-b", "--test-data-path", "prefix-c",
             "--data-cache-path", "/cache", "--dataloader-fast-cache-load", "--dataloader-defer-npy-index-mmap"],
            {},
        ),
        "knobs": (
            ["--data-path", "prefix-a", "--phase-transition-iterations", "20,10",
             "--dataloader-inter-document-masking", "--reset-position-ids", "--reset-attention-mask",
             "--eod-mask-loss", "--no-create-attention-mask-in-dataloader", "--num-dataset-builder-threads", "3",
             "--mid-level-dataset-surplus", "0.1", "--object-storage-cache-path", "/object-cache",
             "--allow-ambiguous-pad-tokens", "--no-mmap-bin-files", "--per-dataset-sequences-path", str(sequences),
             "--sft-mock-dataset-config-json", "{}", "--varlen-mock-dataset-config-json", "{}",
             "--varlen-sbhd-validation", "--num-workers", "0", "--dataloader-type", "cyclic"],
            {},
        ),
        # Every field the legacy hybrid, hybrid-flex and cache factories left at Core's defaults
        # is set here (the validation flags through the overrides, as a run's validation section).
        "forwarded": (
            ["--train-data-path", "prefix-a", "--valid-data-path", "prefix-b", "--test-data-path", "prefix-c",
             "--data-cache-path", "/cache", "--dataloader-fast-cache-load", "--dataloader-defer-npy-index-mmap",
             "--dataloader-inter-document-masking", "--reset-position-ids", "--reset-attention-mask",
             "--eod-mask-loss", "--allow-ambiguous-pad-tokens", "--per-dataset-sequences-path", str(sequences),
             "--sft-mock-dataset-config-json", "{}", "--varlen-mock-dataset-config-json", "{}",
             "--varlen-sbhd-validation"],
            VALIDATION_OVERRIDES,
        ),
        "sft_fim": (["--sft", "--fim-data", "--data-path", "prefix-a"], {}),
        "varlen_fim": (
            ["--use-varlen-dataset", "--fim-data", "--data-path", "prefix-a", "--calculate-per-token-loss",
             "--max-seqlen-per-dp-cp-rank", "32"],
            {},
        ),
        "implicit_mock": ([], {}),
        # Accepted by validate_args; rejected by the factory (see the dedicated test).
        "mock_data_args_file": (["--mock-data", "--data-args-path", str(data_args), "--split", "9,1,0"], {}),
        "mock_per_split_file": (["--mock-data", "--per-split-data-args-path", str(per_split)], {}),
    }  # fmt: skip


CASE_NAMES = [
    "mock", "mock_parallel", "blend", "blend_default_split", "per_split", "data_args_file",
    "per_split_file", "sft_mock", "sft", "varlen_mock", "varlen_sbhd", "fim", "cache", "knobs",
    "forwarded", "sft_fim", "varlen_fim", "implicit_mock",
]  # fmt: skip
MIXED_FIM_CASES = {"sft_fim", "varlen_fim"}
ENTRY_POINTS = ("gpt", "hybrid", "hybrid_flex", "prepare_cache")
# What a script cannot build, it rejects itself: pretrain_hybrid_flex.py has no varlen batch path
# (it used to build plain GPT data for the request); tools/prepare_cache.py prepares the plain
# GPTDataset cache only, needs a data source, and rejects the other modes as before.
REJECTED = {
    ("hybrid_flex", "varlen_mock"): "does not support --use-varlen-dataset",
    ("hybrid_flex", "varlen_sbhd"): "does not support --use-varlen-dataset",
    ("hybrid_flex", "varlen_fim"): "does not support --use-varlen-dataset",
    ("prepare_cache", "mock"): "--mock-data is not supported",
    ("prepare_cache", "mock_parallel"): "--mock-data is not supported",
    ("prepare_cache", "sft_mock"): "--mock-data is not supported",
    ("prepare_cache", "sft"): "--sft is not supported",
    ("prepare_cache", "varlen_mock"): "--mock-data is not supported",
    ("prepare_cache", "fim"): "--fim-data is not supported",
    ("prepare_cache", "sft_fim"): "--sft is not supported",
    ("prepare_cache", "varlen_fim"): "--fim-data is not supported",
    ("prepare_cache", "implicit_mock"): "GPTDataset needs a data source",
}
# Documented changes from the legacy hybrid / hybrid-flex providers, which had no FIM branch and
# selected the real SFTDataset for --sft --mock-data (its mock build failed): (before, after).
LEGACY_CLASS_CHANGES = {
    ("hybrid", "fim"): (GPTDataset, GPTFIMDataset),
    ("hybrid", "sft_mock"): (SFTDataset, MockSFTDataset),
    ("hybrid_flex", "fim"): (GPTDataset, GPTFIMDataset),
    ("hybrid_flex", "sft_mock"): (SFTDataset, MockSFTDataset),
}
# The Core fields each legacy factory left at Megatron Core's defaults, forwarded now like every
# other data argument (the "forwarded" case sets them all). pretrain_hybrid_flex.py's layout fields
# are not among them: it builds its container without a model config, so they keep Core's defaults
# on both sides.
LEGACY_OMISSIONS = {
    "hybrid": {"sft_mock_dataset_config_json"},
    "hybrid_flex": {
        "multiple_validation_sets", "full_validation", "allow_ambiguous_pad_tokens",
        "fast_cache_load", "defer_npy_index_mmap", "sequences_per_dataset",
        "sft_mock_dataset_config_json", "varlen_mock_dataset_config_json", "varlen_sbhd_validation",
    },
    "prepare_cache": {
        "inter_document_masking", "sft_mock_dataset_config_json",
        "varlen_mock_dataset_config_json", "varlen_sbhd_validation",
    },
}  # fmt: skip
LEGACY_OWN = {
    "hybrid": legacy.legacy_hybrid_dataset_config,
    "hybrid_flex": legacy.legacy_hybrid_flex_dataset_config,
    "prepare_cache": legacy.legacy_prepare_cache_dataset_config,
}
LEGACY_BUILD = {
    "gpt": (legacy.legacy_gpt_dataset_config, lambda a: legacy.legacy_gpt_dataset_class(a)[0]),
    "hybrid": (legacy.legacy_hybrid_dataset_config, lambda a: legacy.legacy_hybrid_dataset_class(a)[0]),
    "hybrid_flex": (legacy.legacy_hybrid_flex_dataset_config, legacy.legacy_hybrid_flex_dataset_class),
}  # fmt: skip


def _entry_point(name):
    if name == "gpt":
        import pretrain_gpt as module
    elif name == "hybrid":
        import pretrain_hybrid as module
    elif name == "hybrid_flex":
        from megatron.elastification import pretrain_hybrid_flex as module
    else:
        from tools import prepare_cache as module
    return module


def _validated(extra, overrides):
    with patch("sys.argv", ["test_dataset_providers.py", *MODEL_ARGV, *extra]):
        args = validate_args(parse_args())
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _layout_from_args(args):
    """The model section a real run derives from the same arguments (gpt_config_from_args)."""
    transformer = SimpleNamespace(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        sequence_parallel=args.sequence_parallel,
        hybrid_context_parallel=args.hybrid_context_parallel,
        gtp_weight_remat_size=args.gtp_weight_remat_size,
        gtp_remat_num_sequence_shards=args.gtp_remat_num_sequence_shards,
        cuda_graph_impl="none",
    )
    return SimpleNamespace(transformer=transformer, seq_length=args.max_position_embeddings)


def _world_size_for(args):
    shards = args.gtp_remat_num_sequence_shards
    return (
        args.tensor_model_parallel_size
        * args.pipeline_model_parallel_size
        * (args.context_parallel_size // shards)
        * args.gtp_weight_remat_size
        * args.data_parallel_size
    )


def _data_config(entry_point, args):
    """The data section as the entry point builds it."""
    if entry_point == "hybrid_flex":
        return _entry_point(entry_point).flex_data_config_from_args(args)
    return gpt_data_config_from_args(args)


def _prepare(monkeypatch, extra, overrides, entry_point="gpt"):
    """Validated args, the container as the entry point builds it (not yet finalized) and the
    run's tokenizer, registered as globals. pretrain_hybrid_flex.py builds its container without a
    model config, so its layout mirrors have no owner there; the other scripts carry the layout the
    same arguments give gpt_config_from_args."""
    args = _validated(extra, overrides)
    cfg = pretrain_cfg_container_from_args(args, data_cfg=_data_config(entry_point, args))
    cfg.model = None if entry_point == "hybrid_flex" else _layout_from_args(args)
    cfg.validation.full_validation = args.full_validation
    cfg.validation.multiple_validation_sets = args.multiple_validation_sets
    monkeypatch.setattr(container_module, "_current_world_size", lambda: _world_size_for(args))
    tokenizer = build_tokenizer(args)
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", cfg)
    monkeypatch.setattr(global_vars, "_GLOBAL_TOKENIZER", tokenizer)
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    monkeypatch.setattr(training, "get_args", lambda: args)
    return args, cfg, tokenizer


def _ready(cfg):
    """What pretrain() and initialize_megatron() do before the provider runs."""
    cfg.validate()
    cfg.finalize()
    global_vars._prepare_dataset_config(cfg)
    return cfg.data


def _tokenizer_signature(tokenizer):
    signature = {"type": type(tokenizer).__name__, "vocab_size": tokenizer.vocab_size}
    for name in ("eod", "pad", "bos", "eos", "mask", "cls", "sep"):
        try:
            signature[name] = getattr(tokenizer, name)
        except Exception:  # noqa: BLE001 - a tokenizer may not define the special token
            signature[name] = "unavailable"
    return signature


def _signature(config, names):
    signature = {name: getattr(config, name) for name in names}
    signature["tokenizer"] = _tokenizer_signature(signature["tokenizer"])
    return signature


@pytest.fixture
def warnings_log(monkeypatch):
    log = []
    monkeypatch.setattr(argument_utils, "_warn", log.append)
    return log


@pytest.mark.parametrize("case", CASE_NAMES)
@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_entry_points_build_the_dataset_config_pretrain_gpt_built(
    monkeypatch, warnings_log, corpus, entry_point, case
):
    extra, overrides = corpus[case]
    module = _entry_point(entry_point)
    if (entry_point, case) in REJECTED:
        with pytest.raises(ValueError, match=REJECTED[(entry_point, case)]):
            args, _, _ = _prepare(monkeypatch, extra, overrides, entry_point)
            module.core_gpt_dataset_config_from_args(args)
        return
    args, cfg, tokenizer = _prepare(monkeypatch, extra, overrides, entry_point)
    expected = legacy.legacy_gpt_dataset_config(args)
    expected_class, expected_packed = legacy.legacy_gpt_dataset_class(args)
    if entry_point == "prepare_cache":
        actual = module.core_gpt_dataset_config_from_args(args)
        assert actual.is_prepared
        dataset_class = packed = None
    else:
        data = _ready(cfg)
        dataset_class, actual, packed = data.dataset_build()
        assert actual is data.dataset and actual.is_prepared and actual.tokenizer is tokenizer
    if case in MIXED_FIM_CASES:
        # Documented exception: the dataset class and every shared field are unchanged; the unused
        # FIM runtime wrapper pretrain_gpt built around an SFT / varlen dataset is gone.
        assert isinstance(expected, RuntimeGPTFIMDatasetConfig)
        assert not isinstance(actual, RuntimeGPTFIMDatasetConfig)
        names = [f.name for f in fields(MCoreGPTDatasetConfig)]
    else:
        assert isinstance(actual, type(expected))
        names = [f.name for f in fields(expected)]
    compared = names
    if entry_point == "hybrid_flex":
        # No model config: the layout mirrors keep Megatron Core's defaults, as the legacy factory
        # left them; the standalone wrapper fills them from args like the other wrappers.
        flex_layout = legacy.legacy_hybrid_flex_dataset_config(args)
        for name in LAYOUT_MIRRORS:
            assert getattr(actual, name) == getattr(flex_layout, name), name
        compared = [name for name in names if name not in LAYOUT_MIRRORS]
    assert _signature(actual, compared) == _signature(expected, compared)
    if entry_point == "prepare_cache":
        return
    if case == "implicit_mock":
        # Documented difference: the legacy providers keyed on --mock-data and paired a mock
        # runtime config with the real dataset class; the section selects the mock class.
        assert expected_class is GPTDataset and dataset_class is MockGPTDataset
    else:
        assert dataset_class is expected_class
    if entry_point != "hybrid_flex":  # its rank predicate has no packed-sequence parameter
        assert packed is expected_packed
    wrapper = module.core_gpt_dataset_config_from_args(args)
    assert isinstance(wrapper, type(actual)) and wrapper.is_prepared
    assert _signature(wrapper, names) == _signature(expected, names)


@pytest.mark.parametrize(("entry_point", "case"), sorted(LEGACY_CLASS_CHANGES))
def test_hybrid_entry_points_build_the_requested_family(
    monkeypatch, warnings_log, corpus, entry_point, case
):
    """pretrain_hybrid.py and pretrain_hybrid_flex.py had no FIM branch and selected the real
    SFTDataset for --sft --mock-data, whose mock build failed. Both now build what pretrain_gpt.py
    builds for the same arguments, without a substitution warning."""
    extra, overrides = corpus[case]
    args, cfg, _ = _prepare(monkeypatch, extra, overrides, entry_point)
    before, after = LEGACY_CLASS_CHANGES[(entry_point, case)]
    assert LEGACY_BUILD[entry_point][1](args) is before
    data = _ready(cfg)
    assert data.dataset_type is after is legacy.legacy_gpt_dataset_class(args)[0]
    assert isinstance(data.dataset, GPTFIMDatasetConfig) is (after is GPTFIMDataset)
    assert warnings_log == []


@pytest.mark.parametrize("entry_point", sorted(LEGACY_OMISSIONS))
def test_fields_the_legacy_factories_left_out_are_forwarded(monkeypatch, corpus, entry_point):
    """The one factory forwards every data argument: against its own legacy factory, each entry
    point differs in exactly the fields that factory left at Megatron Core's defaults, and in
    nothing else against pretrain_gpt's."""
    extra, overrides = corpus["forwarded"]
    args, cfg, _ = _prepare(monkeypatch, extra, overrides, entry_point)
    own = LEGACY_OWN[entry_point](args)
    if entry_point == "prepare_cache":
        actual = _entry_point(entry_point).core_gpt_dataset_config_from_args(args)
    else:
        _, actual, _ = _ready(cfg).dataset_build()
    names = [f.name for f in fields(MCoreGPTDatasetConfig) if f.name != "tokenizer"]
    changed = {name for name in names if getattr(actual, name) != getattr(own, name)}
    assert changed == LEGACY_OMISSIONS[entry_point]
    expected = legacy.legacy_gpt_dataset_config(args)
    same = [n for n in names if entry_point != "hybrid_flex" or n not in LAYOUT_MIRRORS]
    assert {n: getattr(actual, n) for n in same} == {n: getattr(expected, n) for n in same}


def test_hybrid_flex_rejects_the_varlen_family(monkeypatch, warnings_log, corpus):
    """Its batch path keys the packed-sequence metadata on --sft and its rank predicate has no
    packed-sequence parameter: the request is an error instead of a silent plain-GPT run, while
    --sft stays supported and is built as pretrain_gpt builds it."""
    module = _entry_point("hybrid_flex")
    extra, overrides = corpus["varlen_sbhd"]
    with pytest.raises(ValueError, match="does not support --use-varlen-dataset"):
        module.flex_data_config_from_args(_validated(extra, overrides))
    assert warnings_log == []
    _, cfg, _ = _prepare(monkeypatch, *corpus["sft"], "hybrid_flex")
    data = _ready(cfg)
    assert data.dataset_type is SFTDataset and type(data.dataset) is GPTDatasetConfig


@pytest.mark.parametrize("entry_point", ["gpt", "hybrid", "hybrid_flex"])
def test_provider_path_builds_no_second_tokenizer(monkeypatch, corpus, entry_point):
    args, cfg, tokenizer = _prepare(monkeypatch, *corpus["mock"], entry_point)
    module = _entry_point(entry_point)
    if hasattr(module, "build_tokenizer"):
        monkeypatch.setattr(module, "build_tokenizer", Mock(side_effect=AssertionError("rebuilt")))
    data = _ready(cfg)
    _, config, _ = data.dataset_build()
    assert config.tokenizer is global_vars.get_tokenizer() is tokenizer


def test_sft_and_varlen_datasets_read_no_fim_field():
    """The mixed-FIM exception rests on this: the packed datasets never read a fim_* attribute."""
    for module in (sft_dataset, varlen_dataset):
        tree = ast.parse(inspect.getsource(module))
        attribute_reads = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr.lower().startswith("fim")
        }
        assert not attribute_reads, (module.__name__, attribute_reads)


@pytest.mark.parametrize("case", ["mock_data_args_file", "mock_per_split_file"])
@pytest.mark.parametrize("entry_point", ["gpt", "hybrid"])
def test_mock_data_next_to_a_file_backed_source_is_rejected(monkeypatch, corpus, entry_point, case):
    """validate_args accepts --mock-data next to a file-backed source. The legacy providers then
    selected MockGPTDataset (keyed on --mock-data) around a runtime config built from the file
    (mock=False), which MockGPTDataset refuses, so the combination never trained. The factory
    rejects it before the container is built."""
    extra, overrides = corpus[case]
    legacy_factory, legacy_class = LEGACY_BUILD[entry_point]
    args = _validated(extra, overrides)
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", pretrain_cfg_container_from_args(args))
    legacy_config = legacy_factory(args)
    assert legacy_config.mock is False
    assert legacy_config.blend is not None or legacy_config.blend_per_split is not None
    assert legacy_class(args) is MockGPTDataset
    with pytest.raises(ValueError, match="single data source"):
        gpt_data_config_from_args(args)


def test_fim_request_without_a_source_is_rejected(monkeypatch, warnings_log):
    args = _validated(["--fim-data"], {})
    data = gpt_data_config_from_args(args)
    assert isinstance(data.dataset, GPTDatasetConfig) and data.dataset.blend is None
    with pytest.raises(ValueError, match="no mock dataset for the fim family"):
        data.validate()


class TestProvidersBuildMockDatasets:
    """The providers build mock datasets through the real builder, as the legacy code did, and the
    builder receives cfg.data.dataset itself."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(1, 1)
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        ("entry_point", "case", "oracle"),
        [
            ("gpt", "mock", "gpt"),
            ("gpt", "sft_mock", "gpt"),
            ("gpt", "varlen_mock", "gpt"),
            ("hybrid", "mock", "hybrid"),
            ("hybrid", "varlen_mock", "hybrid"),
            # pretrain_hybrid's legacy SFTDataset around a mock config could not build: the
            # reference is what pretrain_gpt built for the same arguments.
            ("hybrid", "sft_mock", "gpt"),
            ("hybrid_flex", "mock", "hybrid_flex"),
        ],
    )
    def test_provider_matches_the_legacy_build(
        self, monkeypatch, corpus, entry_point, case, oracle
    ):
        legacy_factory, legacy_class = LEGACY_BUILD[oracle]
        args, cfg, _ = _prepare(monkeypatch, *corpus[case], entry_point)
        module = _entry_point(entry_point)
        expected = BlendedMegatronDatasetBuilder(
            legacy_class(args), (8, 8, 8), lambda: True, legacy_factory(args)
        ).build()
        data = _ready(cfg)
        actual = module.train_valid_test_datasets_provider((8, 8, 8))  # the run config's data
        explicit = providers.build_gpt_datasets(
            (8, 8, 8), data_config=data, built_on_rank=module.is_dataset_built_on_rank
        )
        for built in (actual, explicit):
            _assert_same_datasets(expected, built)
            assert all(split.config is data.dataset for split in built)

    @pytest.mark.parametrize("case", ["sft_mock", "varlen_mock"])
    def test_packed_mock_datasets_ignore_the_fim_wrapper(self, monkeypatch, corpus, case):
        """Mixed-FIM exception: an SFT / varlen dataset builds the same samples whether its runtime
        config is the FIM wrapper pretrain_gpt used to build for --sft --fim-data or the plain config
        the provider builds now (mock datasets; validate_args rejects --fim-data with --mock-data,
        so the configs are constructed directly here)."""
        args, cfg, _ = _prepare(monkeypatch, *corpus[case])
        data = _ready(cfg)
        plain = data.dataset
        shared = {f.name: getattr(plain, f.name) for f in fields(MCoreGPTDatasetConfig) if f.init}
        wrapped = RuntimeGPTFIMDatasetConfig(
            **shared,
            fim_rate=0.9,
            fim_spm_rate=0.4,
            fim_extra_tokens={"prefix": "<p>", "middle": "<m>", "suffix": "<s>", "pad": "<x>", "eod": "<e>"},
        )  # fmt: skip
        dataset_class = legacy.legacy_gpt_dataset_class(args)[0]
        with_wrapper = BlendedMegatronDatasetBuilder(
            dataset_class, (8, 8, 8), lambda: True, wrapped
        ).build()
        without = BlendedMegatronDatasetBuilder(
            dataset_class, (8, 8, 8), lambda: True, plain
        ).build()
        _assert_same_datasets(with_wrapper, without)

    def test_legacy_mock_data_next_to_a_file_backed_source_did_not_build(self, monkeypatch, corpus):
        """The legacy outcome behind test_mock_data_next_to_a_file_backed_source_is_rejected: the
        class the legacy provider selected refuses the runtime config the legacy factory built.
        Constructed directly (the builder constructs on rank 0 before a barrier, so an exception
        there would leave the other rank waiting)."""
        extra, overrides = corpus["mock_data_args_file"]
        args = _validated(extra, overrides)
        monkeypatch.setattr(
            global_vars, "_GLOBAL_RUN_CONFIG", pretrain_cfg_container_from_args(args)
        )
        legacy_config = legacy.legacy_gpt_dataset_config(args)
        legacy_class = legacy.legacy_gpt_dataset_class(args)[0]
        assert legacy_class is MockGPTDataset and legacy_config.mock is False
        with pytest.raises(AssertionError):
            legacy_class(None, None, None, 8, Split.train, legacy_config)


def _assert_same_datasets(expected, built):
    for expected_split, split in zip(expected, built):
        assert type(split) is type(expected_split)
        assert len(split) == len(expected_split)
        sample, expected_sample = split[0], expected_split[0]
        assert sample.keys() == expected_sample.keys()
        for key in expected_sample:
            assert (sample[key] == expected_sample[key]).all(), key


# The shared provider body and the shared legacy wrapper.


def test_scripts_delegate_to_the_shared_provider_body(monkeypatch):
    """The provider body is megatron.training.datasets.providers.build_gpt_datasets; each script's
    train_valid_test_datasets_provider keeps the legacy shape and passes its rank predicate (the
    shared GPT / hybrid rule, or hybrid-flex's own, which accepts the packed-sequence flag)."""
    for name in ("gpt", "hybrid", "hybrid_flex"):
        module = _entry_point(name)
        calls = []
        monkeypatch.setattr(
            module, "build_gpt_datasets", lambda *a, **k: calls.append((a, k)) or (1, 2, 3)
        )
        parameters = inspect.signature(module.train_valid_test_datasets_provider).parameters
        assert list(parameters) == ["train_val_test_num_samples", "vp_stage"]  # main's shape
        built = module.train_valid_test_datasets_provider((8, 8, 8), 1)
        assert built == (1, 2, 3)
        ((positional, keywords),) = calls
        assert positional == ((8, 8, 8), 1)
        assert keywords == {"built_on_rank": module.is_dataset_built_on_rank}
        shared = module.is_dataset_built_on_rank is providers.is_dataset_built_on_rank
        assert shared is (name != "hybrid_flex")
        parameters = inspect.signature(module.is_dataset_built_on_rank).parameters
        assert {"vp_stage", "is_packed_sequence"} <= set(parameters)


def test_legacy_wrapper_is_the_shared_helper(monkeypatch, corpus):
    """pretrain_gpt and pretrain_hybrid expose argument_utils.core_gpt_dataset_config_from_args
    itself; hybrid-flex and the cache tool wrap it (the run's tokenizer; the class pinned to
    GPTDataset). The helper takes the seed from the arguments, validates, and prepares with the
    tokenizer it is given."""
    shared = argument_utils.core_gpt_dataset_config_from_args
    assert _entry_point("gpt").core_gpt_dataset_config_from_args is shared
    assert _entry_point("hybrid").core_gpt_dataset_config_from_args is shared
    args, cfg, _ = _prepare(monkeypatch, *corpus["blend"])
    cfg.rng.seed = 77  # not the dataset's seed: that is --seed
    own = SimpleNamespace(vocab_size=2048)
    config = shared(args, tokenizer=own)
    assert config.is_prepared and config.tokenizer is own and config.random_seed == args.seed
    with pytest.raises(ValueError, match="GPTDataset needs a data source"):
        shared(_validated(["--mock-data"], {}), dataset_type=GPTDataset)
