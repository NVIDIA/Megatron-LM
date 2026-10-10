# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""TransformerBlock, TransformerLayer, Float16Module and the MTP helpers check the caller's
process groups, and their deprecated global fallbacks warn and resolve the same groups as before."""

import dataclasses
import sys
import warnings
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import multi_token_prediction as mtp_module
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.multi_token_prediction import (
    MTPLossLoggingHelper,
    MultiTokenPredictionLayer,
    _compute_mtp_acceptance_counts,
    get_mtp_layer_offset,
    get_mtp_num_layers_to_build,
    process_mtp_loss,
)
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import _LAYER_PROCESS_GROUPS, TransformerLayer
from tests.unit_tests.test_utilities import Utils

_VERSIONS = "deprecated since Megatron Core 0.21 and will be removed in 0.23"

requires_two_ranks = pytest.mark.skipif(
    Utils.world_size < 2, reason="two pipeline stages need at least two ranks"
)
requires_four_ranks = pytest.mark.skipif(
    Utils.world_size % 4 != 0, reason="needs a multiple of four ranks"
)


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


@pytest.fixture(autouse=True)
def destroy_model_parallel():
    yield
    Utils.destroy_model_parallel()


def _fallback_warning(record, owner, argument="pg_collection"):
    """Return the only fallback warning in ``record``, after checking that it names ``owner``."""
    fallbacks = [w for w in record if issubclass(w.category, ProcessGroupFallbackWarning)]
    assert [str(w.message).split(" was called")[0] for w in fallbacks] == [owner]
    message = str(fallbacks[0].message)
    assert message.startswith(f"{owner} was called without `{argument}`")
    assert _VERSIONS in message
    # The warning points at the code that omitted the argument.
    assert fallbacks[0].filename == __file__
    return fallbacks[0]


def _forbid_global_process_groups(monkeypatch):
    """Make the global group, rank and size accessors and the global collection raise.

    By-name imports of an accessor in other Megatron Core modules are patched as well.
    """

    def forbid(*args, **kwargs):
        raise AssertionError("read the global process groups")

    suffixes = ("_group", "_groups", "_gloo", "_rank", "_ranks", "_world_size")
    accessors = {
        name: getattr(parallel_state, name)
        for name in dir(parallel_state)
        if name.startswith("get_") and name.endswith(suffixes) and name != "get_all_ranks"
    }
    originals = {id(accessor) for accessor in accessors.values()}
    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("megatron.core"):
            for name, value in list(vars(module).items()):
                if id(value) in originals:
                    monkeypatch.setattr(module, name, forbid)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", forbid)


def _layer_spec():
    return get_gpt_layer_with_transformer_engine_spec() if HAVE_TE else get_gpt_layer_local_spec()


def _config(**kwargs):
    kwargs = {"num_layers": 4, "hidden_size": 64, "num_attention_heads": 4, **kwargs}
    if kwargs.get("pipeline_model_parallel_size", 1) > 1:
        kwargs.setdefault("pipeline_dtype", torch.float32)
    if "mtp_num_layers" in kwargs:
        kwargs.setdefault("mtp_loss_scaling_factor", 1.0)
    return TransformerConfig(use_cpu_initialization=True, **kwargs)


def _layer_numbers(block):
    return [layer.layer_number for layer in block.layers]


def _this_stage_layer_numbers():
    """Layer numbers of this rank's stage when 4 layers are split over 2 stages."""
    stage = parallel_state.get_pipeline_model_parallel_rank()
    return [2 * stage + 1, 2 * stage + 2]


class TestExplicitCollection:
    """An explicit collection must say which pipeline stage this rank is on."""

    @requires_two_ranks
    @pytest.mark.parametrize("owner", ["TransformerBlock", "TransformerLayer"])
    def test_missing_pp_raises_under_pipeline_parallelism(self, owner):
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        config = _config(pipeline_model_parallel_size=2)
        # Without pp, every stage would read pipeline rank 0 and build the first stage's layers.
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        with pytest.raises(ValueError, match=f"{owner} requires pg_collection to set pp"):
            if owner == "TransformerBlock":
                TransformerBlock(config, _layer_spec(), pg_collection=pg_collection)
            else:
                TransformerLayer(config, _layer_spec().submodules, pg_collection=pg_collection)

    @requires_two_ranks
    def test_pp_none_raises_under_pipeline_parallelism(self):
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        config = _config(pipeline_model_parallel_size=2)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        pg_collection.pp = None
        with pytest.raises(ValueError, match="pipeline_model_parallel_size is 2, but .*pp is None"):
            TransformerBlock(config, _layer_spec(), pg_collection=pg_collection)

    @pytest.mark.parametrize("pp", ["unset", "none"])
    def test_pp_is_optional_without_pipeline_parallelism(self, pp):
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        if pp == "none":
            pg_collection.pp = None
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            block = TransformerBlock(_config(), _layer_spec(), pg_collection=pg_collection)
        assert _layer_numbers(block) == [1, 2, 3, 4]

    def test_missing_tp_raises(self):
        Utils.initialize_model_parallel()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['cp', 'pp'])
        with pytest.raises(ValueError, match="TransformerBlock requires pg_collection to set tp"):
            TransformerBlock(_config(), _layer_spec(), pg_collection=pg_collection)

    @requires_four_ranks
    def test_explicit_collection_reads_no_global_groups(self, monkeypatch):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)
        config = _config(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
        # New communicators over the ranks of the global groups: a stray global read would hand
        # a module a group object that is not in this collection.
        grid = HyperCommGrid([2, 1, Utils.world_size // 4, 2], ["tp", "cp", "dp", "pp"])
        pg_collection = ProcessGroupCollection(
            tp=grid.create_pg("tp"),
            cp=grid.create_pg("cp"),
            pp=grid.create_pg("pp"),
            dp_cp=grid.create_pg(["dp", "cp"]),
            hcp=None,
            gtp_remat=None,
            expt_gtp_remat=None,
        )
        _forbid_global_process_groups(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            block = TransformerBlock(config, _layer_spec(), pg_collection=pg_collection)
        assert block.pg_collection is pg_collection
        assert block.tp_group is pg_collection.tp
        assert _layer_numbers(block) == [
            2 * pg_collection.pp.rank() + 1,
            2 * pg_collection.pp.rank() + 2,
        ]
        for layer in block.layers:
            assert layer.pg_collection is pg_collection
            assert layer.self_attention.pg_collection is pg_collection
            assert layer.mlp.tp_group is pg_collection.tp


class TestDefaultCollection:
    """Without a collection, the fallback warns once and builds the same modules as before."""

    @requires_four_ranks
    def test_block_warns_once_and_uses_the_global_groups(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
        # No dropout, so that the two blocks below compute the same outputs.
        config = _config(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        model_parallel_cuda_manual_seed(123)
        with pytest.warns(ProcessGroupFallbackWarning) as record:
            block = TransformerBlock(config, _layer_spec())
        # Only the block warns: it hands its collection to the layers and their modules.
        _fallback_warning(record, "TransformerBlock")

        # The fallback builds the fields the layers read, each the global group.
        full = vars(ProcessGroupCollection.use_mpu_process_groups())
        groups = vars(block.pg_collection)
        assert set(groups) == set(_LAYER_PROCESS_GROUPS)
        for name, group in groups.items():
            assert group is full[name], name

        # The block builds the same layers as with the full global collection.
        model_parallel_cuda_manual_seed(123)
        reference = TransformerBlock(
            config, _layer_spec(), pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        )
        assert _layer_numbers(block) == _layer_numbers(reference) == _this_stage_layer_numbers()
        assert [(n, p.shape) for n, p in block.named_parameters()] == [
            (n, p.shape) for n, p in reference.named_parameters()
        ]
        for layer in block.layers:
            assert layer.pg_collection is block.pg_collection
            assert layer.self_attention.pg_collection is block.pg_collection

        block.cuda()
        reference.cuda()
        reference.load_state_dict(block.state_dict())
        torch.manual_seed(0)
        hidden_states = torch.randn(16, 2, config.hidden_size, device="cuda")
        mask = torch.ones((1, 1, 16, 16), dtype=torch.bool, device="cuda")
        torch.testing.assert_close(
            block(hidden_states, attention_mask=mask),
            reference(hidden_states, attention_mask=mask),
            rtol=0,
            atol=0,
        )

    @requires_two_ranks
    def test_layer_warns_once_with_its_own_name(self):
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        model_parallel_cuda_manual_seed(123)
        with pytest.warns(ProcessGroupFallbackWarning) as record:
            layer = TransformerLayer(
                _config(pipeline_model_parallel_size=2), _layer_spec().submodules, layer_number=1
            )
        _fallback_warning(record, "TransformerLayer")
        assert set(vars(layer.pg_collection)) == set(_LAYER_PROCESS_GROUPS)
        assert layer.layer_number == _this_stage_layer_numbers()[0]
        assert layer.self_attention.pg_collection is layer.pg_collection

    @pytest.mark.skipif(not HAVE_TE, reason="the MoE layer spec needs Transformer Engine")
    @pytest.mark.parametrize("moe", [False, True])
    def test_modules_read_only_the_fields_the_fallback_builds(self, monkeypatch, moe):
        Utils.initialize_model_parallel(expert_model_parallel_size=2 if moe else 1)
        model_parallel_cuda_manual_seed(123)
        moe_kwargs = (
            dict(
                num_moe_experts=4,
                moe_router_topk=2,
                moe_ffn_hidden_size=128,
                moe_token_dispatcher_type="alltoall",
                expert_model_parallel_size=2,
                add_bias_linear=False,
            )
            if moe
            else {}
        )
        config = _config(num_layers=2, **moe_kwargs)
        field_names = {f.name for f in dataclasses.fields(ProcessGroupCollection)}

        # __getattr__ only runs for a field that is not in vars(): make reading one fail.
        def read_unset_field(self, name):
            if name in field_names:
                raise AssertionError(f"read the unset field {name}")
            raise AttributeError(name)

        monkeypatch.setattr(ProcessGroupCollection, "__getattr__", read_unset_field)
        with pytest.warns(ProcessGroupFallbackWarning):
            block = TransformerBlock(config, get_gpt_decoder_block_spec(config, True)).cuda()
        hidden_states = torch.randn(16, 2, config.hidden_size, device="cuda")
        block(hidden_states, attention_mask=None).sum().backward()
        block.sharded_state_dict()
        if moe:
            mlp = block.layers[0].mlp
            assert mlp.ep_group is parallel_state.get_expert_model_parallel_group()
            assert mlp.ep_group.size() == 2


class TestFloat16ModuleCollection:
    """Float16Module converts on the pipeline stages of the wrapped module's own groups."""

    class _Recorder(torch.nn.Module):
        def forward(self, x):
            self.input_dtype = x.dtype
            return x

    class _StageGroup:
        """A pipeline group of which this rank is ``rank`` out of ``size``."""

        def __init__(self, rank, size):
            self._rank, self._size = rank, size

        def rank(self):
            return self._rank

        def size(self):
            return self._size

    @pytest.mark.parametrize(
        "rank, size, input_dtype, converted_input, output_dtype",
        [
            (0, 1, torch.float32, torch.bfloat16, torch.float32),  # only stage
            (0, 2, torch.float32, torch.bfloat16, torch.bfloat16),  # first stage
            (1, 3, torch.bfloat16, torch.bfloat16, torch.bfloat16),  # middle stage
            (1, 2, torch.bfloat16, torch.bfloat16, torch.float32),  # last stage
        ],
    )
    def test_explicit_collection_selects_the_stage(
        self, monkeypatch, rank, size, input_dtype, converted_input, output_dtype
    ):
        Utils.initialize_model_parallel()
        config = _config(bf16=True, pipeline_model_parallel_size=size)
        recorder = self._Recorder()
        pg_collection = ProcessGroupCollection(pp=self._StageGroup(rank, size))
        _forbid_global_process_groups(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            wrapper = Float16Module(config, recorder, pg_collection=pg_collection)
            output = wrapper(torch.ones(2, dtype=input_dtype, device="cuda"))
        assert recorder.input_dtype == converted_input
        assert output.dtype == output_dtype

    def test_argument_takes_precedence_over_the_module_collection(self):
        Utils.initialize_model_parallel()
        recorder = self._Recorder()
        recorder.pg_collection = ProcessGroupCollection(pp=self._StageGroup(0, 2))
        pg_collection = ProcessGroupCollection(pp=self._StageGroup(1, 2))
        config = _config(bf16=True, pipeline_model_parallel_size=2)
        wrapper = Float16Module(config, recorder, pg_collection=pg_collection)
        assert wrapper.pg_collection is pg_collection
        # The last stage of two converts outputs but not inputs.
        assert wrapper(torch.ones(2, dtype=torch.bfloat16, device="cuda")).dtype == torch.float32

    def test_module_collection_without_pp_raises_under_pipeline_parallelism(self):
        Utils.initialize_model_parallel()
        recorder = self._Recorder()
        recorder.pg_collection = ProcessGroupCollection(tp=None)
        config = _config(bf16=True, pipeline_model_parallel_size=2)
        with pytest.raises(ValueError, match="Float16Module requires pg_collection to set pp"):
            Float16Module(config, recorder)

    @requires_two_ranks
    def test_without_a_collection_warns_and_uses_the_global_pipeline_group(self):
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        config = _config(bf16=True, pipeline_model_parallel_size=2)
        wrapper = Float16Module(config, self._Recorder())
        with pytest.warns(ProcessGroupFallbackWarning) as record:
            output = wrapper(torch.ones(2, dtype=torch.bfloat16, device="cuda"))
        _fallback_warning(record, "Float16Module")
        last_stage = parallel_state.get_pipeline_model_parallel_rank() == 1
        assert output.dtype == (torch.float32 if last_stage else torch.bfloat16)


class TestMultiTokenPrediction:
    """MTP offset, layer count and loss fallbacks warn; the acceptance guard reads no globals."""

    @requires_two_ranks
    def test_layer_count_without_pp_rank_warns_and_uses_the_global_rank(self):
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        config = _config(num_layers=2, mtp_num_layers=1, pipeline_model_parallel_size=2)
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            expected = get_mtp_num_layers_to_build(config, pp_rank=pp_rank)
        with pytest.warns(ProcessGroupFallbackWarning) as record:
            assert get_mtp_num_layers_to_build(config) == expected
        _fallback_warning(record, "get_mtp_num_layers_to_build", "pp_rank")
        # Without a layout, all MTP layers sit on the last stage.
        assert expected == (1 if pp_rank == 1 else 0)

    @requires_two_ranks
    def test_layer_offset_without_pp_rank_warns_and_uses_the_global_rank(self):
        Utils.initialize_model_parallel(pipeline_model_parallel_size=2)
        config = _config(
            num_layers=2,
            mtp_num_layers=1,
            pipeline_model_parallel_size=2,
            pipeline_model_parallel_layout=[['embedding', 'decoder', 'decoder', 'mtp'], ['loss']],
        )
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        with pytest.warns(ProcessGroupFallbackWarning) as record:
            assert get_mtp_layer_offset(config) == get_mtp_layer_offset(config, pp_rank=pp_rank)
        _fallback_warning(record, "get_mtp_layer_offset", "pp_rank")

    @pytest.mark.parametrize("metric_avg_group", ["omitted", "given"])
    def test_loss_metrics_without_a_group_warn_and_use_the_global_group(
        self, monkeypatch, metric_avg_group
    ):
        Utils.initialize_model_parallel()
        captured = {}
        monkeypatch.setattr(
            MTPLossLoggingHelper,
            "save_metrics_to_tracker",
            lambda *args, **kwargs: captured.update(avg_group=kwargs["avg_group"]),
        )
        group = object() if metric_avg_group == "given" else None
        config = _config(num_layers=2, hidden_size=1, num_attention_heads=1, mtp_num_layers=1)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            process_mtp_loss(
                hidden_states=torch.ones(8, 1, 1),
                labels=torch.zeros(1, 4, dtype=torch.long),
                loss_mask=torch.ones(1, 4),
                output_layer=lambda hidden, **kwargs: (hidden, None),
                output_weight=None,
                runtime_gather_output=True,
                is_training=True,
                compute_language_model_loss=lambda labels, logits: torch.ones_like(
                    labels, dtype=logits.dtype
                ),
                config=config,
                metric_avg_group=group,
            )
        if metric_avg_group == "given":
            assert captured["avg_group"] is group
            assert not [w for w in record if issubclass(w.category, ProcessGroupFallbackWarning)]
        else:
            assert captured["avg_group"] is parallel_state.get_data_parallel_group(
                with_context_parallel=True
            )
            _fallback_warning(record, "process_mtp_loss", "metric_avg_group")

    def test_acceptance_counts_need_the_tp_group_of_vocab_sharded_logits(self, monkeypatch):
        Utils.initialize_model_parallel()

        def forbid(*args, **kwargs):
            raise AssertionError("read the global parallel state")

        monkeypatch.setattr(mtp_module.parallel_state, "is_initialized", forbid)
        _forbid_global_process_groups(monkeypatch)
        logits = torch.randn(4, 1, 8)
        labels = torch.zeros(1, 4, dtype=torch.long)
        loss_mask = torch.ones(1, 4)
        output_layer = SimpleNamespace(gather_output=False)
        # Sharded logits without their group would take the argmax of one vocab shard only.
        with pytest.raises(ValueError, match="tp_group must be provided"):
            _compute_mtp_acceptance_counts(logits, labels, loss_mask, output_layer, None)
        # Gathered logits hold the whole vocabulary and need no group.
        correct, total = _compute_mtp_acceptance_counts(
            logits, labels, loss_mask, output_layer, True
        )
        assert total.item() == 4
        assert correct.item() == (logits.argmax(dim=-1) == 0).sum().item()

    def test_mtp_layer_requires_cp(self):
        Utils.initialize_model_parallel()
        config = _config(num_layers=2, mtp_num_layers=1)
        mtp_block_spec = get_gpt_mtp_block_spec(
            config, _layer_spec(), use_transformer_engine=HAVE_TE, pp_rank=0
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp'])
        with pytest.raises(
            ValueError, match="MultiTokenPredictionLayer requires pg_collection to set cp"
        ):
            MultiTokenPredictionLayer(
                config, mtp_block_spec.layer_specs[0].submodules, pg_collection=pg_collection
            )
