# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dataset providers decide where datasets are built from the run's process groups.

Training passes its collection to a provider that accepts one, and the providers of the GPT,
hybrid and VLM training scripts take the ranks that build their datasets from it. Callers that
pass no groups keep the global grid, with a warning.

Tests that pass groups make every global parallel_state accessor fail, or give the code groups
whose layout differs from the global grid, so a read of the global grid cannot pass unnoticed.
"""

import functools
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import pretrain_gpt
import pretrain_hybrid
import pretrain_vlm
from megatron.core import parallel_state, process_groups_config, tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.training import training
from megatron.training.config import TrainingConfig, ValidationConfig
from megatron.training.datasets import data_samplers
from megatron.training.state import TrainState
from megatron.training.utils import is_first_or_last_pipeline_stage
from tests.unit_tests.test_utilities import Utils

# Tests of the explicit path fail on any fallback warning, as under strict mode.
no_fallback = pytest.mark.filterwarnings(
    "error::megatron.core.process_groups_config.ProcessGroupFallbackWarning"
)


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


class _FakeGroup:
    """Process-group stand-in with a rank, a size and the global ranks it spans."""

    def __init__(self, rank, ranks):
        self._rank = rank
        self.ranks = list(ranks)

    def rank(self):
        return self._rank

    def size(self):
        return len(self.ranks)


def _forbid_global_process_groups(monkeypatch):
    """Fail the test on any read of the global groups, ranks, sizes or pipeline stages."""

    def fail(*args, **kwargs):
        pytest.fail("the data path must not read the global parallel state")

    for name, value in list(vars(parallel_state).items()):
        if name.startswith(("get_", "is_pipeline_")) and callable(value):
            monkeypatch.setattr(parallel_state, name, fail)
    monkeypatch.setattr(
        ProcessGroupCollection,
        "use_mpu_process_groups",
        Mock(side_effect=AssertionError("supplied groups must not be rebuilt from globals")),
    )


def _use_fake_groups(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: group.ranks)


def _groups(tp_rank, pp_rank, pp_size):
    """Groups whose global ranks differ from any global grid of the test's world."""
    return ProcessGroupCollection(
        tp=_FakeGroup(tp_rank, [5, 7]), pp=_FakeGroup(pp_rank, range(11, 11 + pp_size))
    )


def _fallback_warning(owner, argument="pg_collection"):
    return pytest.warns(
        ProcessGroupFallbackWarning, match=f"{owner} was called without `{argument}`"
    )


# build_train_valid_test_datasets


def _recording_provider(calls, with_vp_stage):
    """A provider with a pg_collection parameter that records the arguments it receives."""

    def provider(train_val_test_num_samples, vp_stage=None, pg_collection=None):
        calls.append((train_val_test_num_samples, vp_stage, pg_collection))
        return "train", "valid", "test"

    if not with_vp_stage:
        return provider
    # Training binds the model chunk of a virtual pipeline stage this way.
    vp_provider = functools.partial(provider, vp_stage=1)
    vp_provider.is_distributed = True
    return vp_provider


@no_fallback
@pytest.mark.parametrize("with_vp_stage", [False, True], ids=["plain", "vp_stage_partial"])
def test_dataset_provider_with_a_pg_collection_parameter_receives_the_run_groups(
    monkeypatch, with_vp_stage
):
    _forbid_global_process_groups(monkeypatch)
    pgs = _groups(0, 0, 1)
    calls = []
    provider = _recording_provider(calls, with_vp_stage)

    datasets = training.build_train_valid_test_datasets(provider, [4, 2, 1], pg_collection=pgs)

    assert datasets == ("train", "valid", "test")
    assert calls == [([4, 2, 1], 1 if with_vp_stage else None, pgs)]


@no_fallback
@pytest.mark.parametrize("signature", ["sizes_only", "var_keyword"])
def test_dataset_provider_without_a_pg_collection_parameter_is_called_with_the_sizes(
    monkeypatch, signature
):
    _forbid_global_process_groups(monkeypatch)
    calls = []
    if signature == "sizes_only":

        def provider(train_val_test_num_samples):
            calls.append(((train_val_test_num_samples,), {}))
            return None, None, None

    else:
        # A provider that forwards **kwargs to another one does not declare pg_collection.
        def provider(*args, **kwargs):
            calls.append((args, kwargs))
            return None, None, None

    training.build_train_valid_test_datasets(provider, [4, 2, 1], pg_collection=_groups(0, 0, 1))

    assert calls == [(([4, 2, 1],), {})]


# is_dataset_built_on_rank of the GPT and hybrid training scripts


@pytest.fixture(params=[pretrain_gpt, pretrain_hybrid], ids=["gpt", "hybrid"])
def dataset_script(request, monkeypatch):
    """A training script whose dataset rank check reads stubbed arguments and config."""
    module = request.param
    args = SimpleNamespace(
        sequence_packing_scheduler=None,
        sft=False,
        use_varlen_dataset=False,
        mock_data=True,
        fim_data=False,
    )
    config = SimpleNamespace(
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        virtual_pipeline_model_parallel_size=None,
    )
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(module, "core_transformer_config_from_args", lambda args: config)
    return SimpleNamespace(module=module, args=args, config=config)


@no_fallback
@pytest.mark.parametrize(
    "tp_rank,pp_rank,pp_size,vp_stage,vp_size,scheduler,packed,expected",
    [
        # Only the first rank of the tensor-parallel group builds datasets.
        (1, 0, 2, None, None, None, False, False),
        (1, 0, 2, None, None, None, True, False),
        (0, 0, 4, None, None, None, False, True),
        (0, 3, 4, None, None, None, False, True),
        # Middle pipeline stages read data only for packed sequences.
        (0, 1, 4, None, None, None, False, False),
        (0, 1, 4, None, None, None, True, True),
        # The packing scheduler sends middle stages their metadata itself.
        (0, 1, 4, None, None, "dp_balanced", True, False),
        (0, 3, 4, None, None, "dp_balanced", True, True),
        # With a virtual pipeline, only the first and last model chunks read data.
        (0, 0, 4, 0, 2, None, False, True),
        (0, 0, 4, 1, 2, None, False, False),
        (0, 3, 4, 1, 2, None, False, True),
        (0, 3, 4, 0, 2, "dp_balanced", True, False),
    ],
)
def test_dataset_rank_check_reads_the_supplied_groups(
    dataset_script,
    monkeypatch,
    tp_rank,
    pp_rank,
    pp_size,
    vp_stage,
    vp_size,
    scheduler,
    packed,
    expected,
):
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    dataset_script.args.sequence_packing_scheduler = scheduler
    dataset_script.config.virtual_pipeline_model_parallel_size = vp_size

    built = dataset_script.module.is_dataset_built_on_rank(
        vp_stage=vp_stage,
        is_packed_sequence=packed,
        pg_collection=_groups(tp_rank, pp_rank, pp_size),
    )

    assert built is expected


@no_fallback
@pytest.mark.parametrize("vp_size", [None, 2])
def test_dataset_rank_check_places_mtp_on_the_supplied_pipeline_group(
    dataset_script, monkeypatch, vp_size
):
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    dataset_script.config.virtual_pipeline_model_parallel_size = vp_size
    mtp_on_this_rank = Mock(return_value=True)
    monkeypatch.setattr(dataset_script.module, "mtp_on_this_rank_func", mtp_on_this_rank)
    pgs = _groups(0, 1, 4)

    # A middle stage reads data when it holds MTP layers.
    assert dataset_script.module.is_dataset_built_on_rank(vp_stage=0, pg_collection=pgs) is True
    kwargs = mtp_on_this_rank.call_args.kwargs
    assert kwargs["pp_group"] is pgs.pp
    assert kwargs["vp_stage"] == 0
    assert kwargs["vp_size"] == (vp_size or 1)


@pytest.mark.parametrize("missing", ["tp", "pp"])
def test_dataset_rank_check_rejects_a_collection_without_its_groups(dataset_script, missing):
    groups = {name: _FakeGroup(0, [5]) for name in ("tp", "pp")}
    del groups[missing]

    with pytest.raises(ValueError, match=f"pg_collection.{missing}"):
        dataset_script.module.is_dataset_built_on_rank(
            pg_collection=ProcessGroupCollection(**groups)
        )


class _RecordingBuilder:
    """BlendedMegatronDatasetBuilder stand-in that evaluates the rank check it receives."""

    def __init__(self, dataset_type, sizes, is_built_on_rank, config):
        self.is_built_on_rank = is_built_on_rank

    def build(self):
        built = self.is_built_on_rank()
        return built, built, built


@no_fallback
@pytest.mark.parametrize("tp_rank,expected", [(0, True), (1, False)])
def test_dataset_provider_builds_on_the_supplied_groups(
    dataset_script, monkeypatch, tp_rank, expected
):
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    module = dataset_script.module
    monkeypatch.setattr(module, "print_rank_0", lambda message: None)
    monkeypatch.setattr(module, "core_gpt_dataset_config_from_args", lambda args: object())
    monkeypatch.setattr(module, "BlendedMegatronDatasetBuilder", _RecordingBuilder)

    datasets = module.train_valid_test_datasets_provider(
        [4, 2, 1], vp_stage=None, pg_collection=_groups(tp_rank, 0, 2)
    )

    assert datasets == (expected, expected, expected)


# The VLM training script


def _stub_vlm_dataset_inputs(monkeypatch):
    monkeypatch.setattr(pretrain_vlm, "print_rank_0", lambda message: None)
    monkeypatch.setattr(pretrain_vlm, "get_run_config", lambda: SimpleNamespace(rng=Mock()))
    monkeypatch.setattr(pretrain_vlm, "get_args", Mock)
    monkeypatch.setattr(pretrain_vlm, "get_tokenizer", Mock)
    monkeypatch.setattr(pretrain_vlm, "MultimodalDatasetConfig", Mock)
    monkeypatch.setattr(pretrain_vlm, "BlendedMegatronDatasetBuilder", _RecordingBuilder)


@no_fallback
@pytest.mark.parametrize("tp_rank,expected", [(0, True), (1, False)])
def test_vlm_dataset_provider_builds_on_the_supplied_tensor_parallel_group(
    monkeypatch, tp_rank, expected
):
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)
    _stub_vlm_dataset_inputs(monkeypatch)

    datasets = pretrain_vlm.train_valid_test_datasets_provider(
        [4, 2, 1], pg_collection=ProcessGroupCollection(tp=_FakeGroup(tp_rank, [3, 7]))
    )

    assert datasets == (expected, expected, expected)


def test_vlm_dataset_rank_check_rejects_a_collection_without_a_tensor_parallel_group():
    with pytest.raises(ValueError, match="pg_collection.tp"):
        pretrain_vlm.is_dataset_built_on_rank(ProcessGroupCollection(pp=_FakeGroup(0, [5])))


# Callers that pass no groups keep the global grid and are warned


@pytest.mark.parametrize("tp_rank,pp_rank,expected", [(0, 1, True), (1, 1, False), (0, 0, True)])
def test_dataset_rank_check_without_groups_warns_and_uses_the_global_groups(
    dataset_script, monkeypatch, tp_rank, pp_rank, expected
):
    _use_fake_groups(monkeypatch)
    global_groups = _groups(tp_rank, pp_rank, 2)
    use_mpu = Mock(return_value=global_groups)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", use_mpu)
    owner = f"{dataset_script.module.__name__}.is_dataset_built_on_rank"

    with _fallback_warning(owner):
        built = dataset_script.module.is_dataset_built_on_rank()

    assert built is expected
    assert use_mpu.call_args.kwargs == {"required_pgs": ["tp", "pp"]}


def test_vlm_dataset_rank_check_without_groups_warns_and_uses_the_global_group(monkeypatch):
    _use_fake_groups(monkeypatch)
    use_mpu = Mock(return_value=ProcessGroupCollection(tp=_FakeGroup(1, [3, 7])))
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", use_mpu)

    with _fallback_warning("pretrain_vlm.is_dataset_built_on_rank"):
        assert pretrain_vlm.is_dataset_built_on_rank() is False
    assert use_mpu.call_args.kwargs == {"required_pgs": ["tp"]}


# The deprecated fallbacks of the data path warn


def test_stage_check_without_a_pipeline_group_warns_and_reads_the_global_grid(monkeypatch):
    calls = []

    def first_stage(ignore_virtual=True, vp_stage=None):
        calls.append(("first", ignore_virtual, vp_stage))
        return False

    def last_stage(ignore_virtual=True, vp_stage=None):
        calls.append(("last", ignore_virtual, vp_stage))
        return True

    monkeypatch.setattr(parallel_state, "is_pipeline_first_stage", first_stage)
    monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", last_stage)

    with _fallback_warning("is_first_or_last_pipeline_stage", "pp_group"):
        assert is_first_or_last_pipeline_stage(1) is True
    assert calls == [("first", False, 1), ("last", False, 1)]


@no_fallback
def test_stage_check_with_a_pipeline_group_does_not_warn(monkeypatch):
    _forbid_global_process_groups(monkeypatch)
    _use_fake_groups(monkeypatch)

    assert is_first_or_last_pipeline_stage(None, pp_group=_FakeGroup(1, [0, 1, 2])) is False


def _loader_args(dataloader_type):
    return SimpleNamespace(
        dataloader_type=dataloader_type,
        num_workers=0,
        data_sharding=True,
        dynamic_context_parallel=False,
        sequence_packing_scheduler=None,
        use_varlen_dataset=False,
        varlen_sbhd_validation=False,
    )


def test_pretraining_sampler_without_groups_warns_and_shards_over_the_global_group(
    monkeypatch, run_config
):
    run_config.train.micro_batch_size = 1
    monkeypatch.setattr(data_samplers, "get_args", lambda: _loader_args("single"))
    monkeypatch.setattr(parallel_state, "get_data_parallel_rank", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_data_parallel_world_size", lambda: 2)

    with _fallback_warning("build_pretraining_data_loader"):
        loader = data_samplers.build_pretraining_data_loader(range(8), consumed_samples=2)

    assert list(loader.batch_sampler) == [[3], [5], [7]]


@no_fallback
def test_external_data_loader_without_groups_reads_no_groups(monkeypatch, run_config):
    _forbid_global_process_groups(monkeypatch)
    monkeypatch.setattr(data_samplers, "get_args", lambda: _loader_args("external"))
    dataset = range(8)

    assert data_samplers.build_pretraining_data_loader(dataset, consumed_samples=0) is dataset


def test_full_validation_group_without_groups_warns_and_uses_the_global_group(monkeypatch):
    global_group = object()
    get_group = Mock(return_value=global_group)
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", get_group)

    with _fallback_warning("build_train_valid_test_data_iterators"):
        group = training._get_data_parallel_group(
            None, with_context_parallel=True, owner="build_train_valid_test_data_iterators"
        )
    assert group is global_group
    get_group.assert_called_once_with(with_context_parallel=True)


@pytest.fixture(params=[pretrain_gpt, pretrain_hybrid], ids=["gpt", "hybrid"])
def batch_script(request, monkeypatch):
    """A get_batch entry point with stubbed TP and CP helpers that record their groups."""
    module = request.param
    _use_fake_groups(monkeypatch)
    args = SimpleNamespace(
        sequence_packing_scheduler=None,
        dynamic_context_parallel=False,
        context_parallel_size=1,
        sft=False,
        dataloader_inter_document_masking=False,
        create_attention_mask_in_dataloader=False,
        micro_batch_size=1,
        seq_length=8,
        pipeline_model_parallel_size=1,
    )
    config = SimpleNamespace(
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        virtual_pipeline_model_parallel_size=None,
        sequence_parallel=False,
        tensor_model_parallel_size=2,
        linear_cp_layout="zigzag",
        attention_cp_layout="zigzag",
    )
    monkeypatch.setattr(module, "get_args", lambda: args)
    monkeypatch.setattr(module, "core_transformer_config_from_args", lambda args: config)
    tp_broadcast = Mock(side_effect=lambda batch, **kwargs: dict.fromkeys(module.BATCH_KEYS))
    monkeypatch.setattr(module, "get_batch_on_this_tp_rank", tp_broadcast)
    if module is pretrain_gpt:
        monkeypatch.setattr(module, "get_batch_on_this_cp_rank", lambda batch, **kwargs: batch)
    else:
        monkeypatch.setattr(module, "get_batches_on_this_cp_rank", Mock())
    return SimpleNamespace(module=module, tp_broadcast=tp_broadcast)


def test_get_batch_without_groups_warns_and_uses_the_global_groups(batch_script, monkeypatch):
    module = batch_script.module
    global_groups = ProcessGroupCollection(
        tp=_FakeGroup(1, [5, 7]),
        pp=_FakeGroup(0, [11]),
        cp=_FakeGroup(0, [5]),
        tp_cp=_FakeGroup(1, [5, 7]),
    )
    use_mpu = Mock(return_value=global_groups)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", use_mpu)

    with _fallback_warning(f"{module.__name__}.get_batch"):
        module.get_batch(None)

    kwargs = batch_script.tp_broadcast.call_args.kwargs
    assert kwargs["broadcast_group"] is global_groups.tp
    assert kwargs["broadcast_src_rank"] == 5


def test_vlm_get_batch_without_groups_warns_and_uses_the_global_group(monkeypatch):
    tp_group = _FakeGroup(1, [3, 7])
    monkeypatch.setattr(
        ProcessGroupCollection,
        "use_mpu_process_groups",
        Mock(return_value=ProcessGroupCollection(tp=tp_group)),
    )
    monkeypatch.setattr(
        pretrain_vlm,
        "get_args",
        lambda: SimpleNamespace(context_parallel_size=1, sequence_parallel=False),
    )
    groups = []

    def broadcast_data(keys, data, datatype, tp_group=None):
        groups.append(tp_group)
        return {key: torch.zeros(1, 4, dtype=datatype) for key in keys}

    monkeypatch.setattr(tensor_parallel, "broadcast_data", broadcast_data)

    with _fallback_warning("pretrain_vlm.get_batch"):
        pretrain_vlm.get_batch(None)

    assert groups == [tp_group, tp_group]


def test_vlm_model_provider_without_groups_warns_before_using_the_global_groups(monkeypatch):
    class GlobalGroupsRead(Exception):
        pass

    monkeypatch.setattr(pretrain_vlm, "get_args", Mock)
    monkeypatch.setattr(
        ProcessGroupCollection, "use_mpu_process_groups", Mock(side_effect=GlobalGroupsRead)
    )

    with _fallback_warning("pretrain_vlm.model_provider"), pytest.raises(GlobalGroupsRead):
        pretrain_vlm.model_provider()


# 4-GPU tests


def _strided_tp_group(world_size):
    """Pair rank r with rank r + world_size // 2, unlike the global grid's adjacent TP pairs."""
    rank = torch.distributed.get_rank()
    half = world_size // 2
    tp_group = None
    for first in range(half):
        ranks = [first, first + half]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            tp_group = group
    return tp_group


def _build_loaders(monkeypatch, run_config, pg_collection, provider_calls, provider_accepts_groups):
    """Run build_train_valid_test_data_loaders with a provider that records its calls."""
    run_config.train = TrainingConfig(train_samples=1, train_iters=1, global_batch_size=1)
    run_config.validation = ValidationConfig(
        eval_interval=1, eval_iters=1, eval_global_batch_size=1
    )
    args = SimpleNamespace(dataloader_type="single", phase_transition_iterations=None)

    if provider_accepts_groups:

        def provider(train_valid_test_num_samples, pg_collection=None):
            provider_calls.append(pg_collection)
            return range(1), range(1), range(1)

    else:

        def provider(train_valid_test_num_samples):
            provider_calls.append(None)
            return range(1), range(1), range(1)

    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_train_state", lambda: TrainState())
    monkeypatch.setattr(
        training,
        "build_pretraining_data_loader",
        lambda dataset, consumed_samples, pg_collection=None: dataset,
    )
    return training.build_train_valid_test_data_loaders(provider, pg_collection=pg_collection)


@no_fallback
def test_data_loaders_build_datasets_on_the_run_tensor_parallel_group(monkeypatch, run_config):
    world_size = Utils.world_size
    if world_size < 4 or world_size % 2:
        pytest.skip("needs an even world size of at least 4")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        tp_group = _strided_tp_group(world_size)
        pgs = ProcessGroupCollection(tp=tp_group)
        provider_calls = []

        with monkeypatch.context() as patch:
            _forbid_global_process_groups(patch)
            _build_loaders(patch, run_config, pgs, provider_calls, provider_accepts_groups=True)

        # The first rank of each strided pair builds; on the global grid, ranks 0, 2, ... would.
        assert provider_calls == ([pgs] if tp_group.rank() == 0 else [])
    finally:
        Utils.destroy_model_parallel()


def test_data_loaders_without_groups_warn_and_build_on_the_global_grid(monkeypatch, run_config):
    world_size = Utils.world_size
    if world_size < 4 or world_size % 2:
        pytest.skip("needs an even world size of at least 4")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        provider_calls = []

        with _fallback_warning("build_train_valid_test_data_loaders"):
            _build_loaders(
                monkeypatch, run_config, None, provider_calls, provider_accepts_groups=False
            )

        global_tp_rank = parallel_state.get_tensor_model_parallel_rank()
        assert provider_calls == ([None] if global_tp_rank == 0 else [])
    finally:
        Utils.destroy_model_parallel()


def _expected_dataset_rank(vp_stage, scheduler, packed):
    """Whether this rank builds the dataset on the global grid, read from parallel_state."""
    if parallel_state.get_tensor_model_parallel_rank() != 0:
        return False
    ignore_virtual = vp_stage is None
    is_first_or_last_stage = parallel_state.is_pipeline_first_stage(
        ignore_virtual=ignore_virtual, vp_stage=vp_stage
    ) or parallel_state.is_pipeline_last_stage(ignore_virtual=ignore_virtual, vp_stage=vp_stage)
    if scheduler is not None:
        return is_first_or_last_stage
    return packed or is_first_or_last_stage


@pytest.mark.parametrize("vp_size", [None, 2])
@pytest.mark.parametrize("scheduler", [None, "dp_balanced"])
@pytest.mark.parametrize("packed", [False, True])
def test_dataset_rank_check_matches_the_global_grid(dataset_script, vp_size, scheduler, packed):
    if Utils.world_size % 4:
        pytest.skip("needs a world size divisible by 4 for a 2 x 2 tensor x pipeline grid")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=vp_size,
    )
    try:
        dataset_script.args.sequence_packing_scheduler = scheduler
        dataset_script.config.virtual_pipeline_model_parallel_size = vp_size
        pgs = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "pp"])
        for vp_stage in [None] if vp_size is None else range(vp_size):
            expected = _expected_dataset_rank(vp_stage, scheduler, packed)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ProcessGroupFallbackWarning)
                built = dataset_script.module.is_dataset_built_on_rank(
                    vp_stage=vp_stage, is_packed_sequence=packed, pg_collection=pgs
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ProcessGroupFallbackWarning)
                built_without_groups = dataset_script.module.is_dataset_built_on_rank(
                    vp_stage=vp_stage, is_packed_sequence=packed
                )
            assert built == expected, f"{vp_stage=}"
            assert built_without_groups == expected, f"{vp_stage=}"
    finally:
        Utils.destroy_model_parallel()
