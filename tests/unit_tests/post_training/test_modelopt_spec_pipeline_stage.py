# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The ModelOpt GPT spec and its builders keep the layers of the pipeline stage they are given."""

from argparse import Namespace

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.core.process_groups_config import ProcessGroupCollection, ProcessGroupFallbackWarning
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

PP_SIZE = 2
VOCAB_SIZE = 128
SEQUENCE_LENGTH = 32

# Three layers split 1 + 2 over two pipeline stages; layers 1 and 3 are MoE.
UNEVEN_SPLIT = {
    "num_layers": 3,
    "num_layers_in_first_pipeline_stage": 1,
    "moe_layer_freq": [1, 0, 1],
}
# Four layers over two pipeline stages with two virtual stages each, one layer per virtual
# stage; layers 1 and 4 are MoE. Virtual stage v of pipeline rank p holds layer 2 * v + p + 1.
VIRTUAL_PIPELINE = {
    "num_layers": 4,
    "virtual_pipeline_model_parallel_size": 2,
    "moe_layer_freq": [1, 0, 0, 1],
}

# Layer numbers each pipeline rank builds for three dense layers split 1 + 2.
DENSE_STAGE_LAYER_NUMBERS = [[1], [2, 3]]


def _forbid_global_pipeline_rank(monkeypatch):
    def forbidden():
        raise RuntimeError("the global pipeline rank was read")

    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", forbidden)


def _moe_config(**kwargs):
    return TransformerConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=2,
        pipeline_model_parallel_size=PP_SIZE,
        pipeline_dtype=torch.bfloat16,
        **kwargs,
    )


def _is_moe(layer_spec):
    return isinstance(get_submodules(layer_spec.submodules.mlp), MoESubmodules)


@pytest.mark.parametrize(
    "config_kwargs, vp_stage, pp_rank, expected_is_moe",
    [
        pytest.param(UNEVEN_SPLIT, None, 0, [True], id="uneven-pp0"),
        pytest.param(UNEVEN_SPLIT, None, 1, [False, True], id="uneven-pp1"),
        pytest.param(VIRTUAL_PIPELINE, 0, 0, [True], id="virtual-vp0-pp0"),
        pytest.param(VIRTUAL_PIPELINE, 0, 1, [False], id="virtual-vp0-pp1"),
        pytest.param(VIRTUAL_PIPELINE, 1, 0, [False], id="virtual-vp1-pp0"),
        pytest.param(VIRTUAL_PIPELINE, 1, 1, [True], id="virtual-vp1-pp1"),
    ],
)
def test_modelopt_spec_keeps_the_layers_of_the_given_stage(
    monkeypatch, config_kwargs, vp_stage, pp_rank, expected_is_moe
):
    _forbid_global_pipeline_rank(monkeypatch)
    config = _moe_config(**config_kwargs)

    block_spec = get_gpt_modelopt_spec(config, vp_stage=vp_stage, pp_rank=pp_rank)

    assert [_is_moe(layer_spec) for layer_spec in block_spec.layer_specs] == expected_is_moe


def test_modelopt_spec_without_pp_rank_uses_the_global_pipeline_rank(monkeypatch):
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda: 1)
    config = _moe_config(**UNEVEN_SPLIT)

    with pytest.warns(ProcessGroupFallbackWarning, match="called without `pp_rank`"):
        block_spec = get_gpt_modelopt_spec(config)

    assert [_is_moe(layer_spec) for layer_spec in block_spec.layer_specs] == [False, True]


def _dense_config():
    return TransformerConfig(
        num_layers=3,
        hidden_size=64,
        num_attention_heads=4,
        pipeline_model_parallel_size=PP_SIZE,
        num_layers_in_first_pipeline_stage=1,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        perform_initialization=False,
    )


def _model_args(**overrides):
    """The arguments the ModelOpt builders read to construct a GPT model with the ModelOpt spec."""
    args = Namespace(
        spec=None,
        mtp_num_layers=None,
        padded_vocab_size=VOCAB_SIZE,
        max_position_embeddings=SEQUENCE_LENGTH,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
        use_rope_scaling=False,
        rope_scaling_factor=8.0,
        export_force_local_attention=False,
        export_te_mcore_model=False,
        export_real_quant_cfg="None",
    )
    vars(args).update(overrides)
    return args


def _build_with_gpt_model_builder(config, pg_collection, monkeypatch):
    from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig

    model_config = GPTModelConfig(
        transformer=config,
        vocab_size=VOCAB_SIZE,
        restore_modelopt_state=True,
        use_arbitrary_attention_mask=False,
    )
    return GPTModelBuilder(model_config).build_model(pg_collection)


def _modelopt_model_builder(monkeypatch, args):
    pytest.importorskip("modelopt.torch.distill")
    from megatron.post_training import model_builder

    monkeypatch.setattr(model_builder, "get_args", lambda: args)
    return model_builder


def _build_with_modelopt_builder(config, pg_collection, monkeypatch):
    args = _model_args(
        export_qk_l2_norm=False,
        export_moe_apply_probs_on_input=False,
        export_model_type="GPTModel",
        export_offline_model=False,
        export_default_te_spec=False,
        export_kd_teacher_load=None,
        freeze_base_for_mtp=False,
        load=None,
    )
    model_builder = _modelopt_model_builder(monkeypatch, args)
    # The builder makes its transformer config from the arguments.
    monkeypatch.setattr(model_builder, "core_transformer_config_from_args", lambda args: config)
    return model_builder.modelopt_gpt_hybrid_builder(
        args,
        is_pp_first_stage(pg_collection.pp),
        is_pp_last_stage(pg_collection.pp),
        pg_collection=pg_collection,
    )


def _build_modelopt_teacher(config, pg_collection, monkeypatch):
    model_builder = _modelopt_model_builder(monkeypatch, _model_args())
    # The teacher is built with the student's model arguments, including its collection.
    model_kwargs = {
        "vocab_size": VOCAB_SIZE,
        "max_sequence_length": SEQUENCE_LENGTH,
        "pre_process": is_pp_first_stage(pg_collection.pp),
        "post_process": is_pp_last_stage(pg_collection.pp),
        "share_embeddings_and_output_weights": False,
        "pg_collection": pg_collection,
    }
    return model_builder._build_teacher_model(config, Namespace(), model_kwargs)


BUILDERS = [
    pytest.param(_build_with_gpt_model_builder, id="GPTModelBuilder"),
    pytest.param(_build_with_modelopt_builder, id="modelopt_gpt_hybrid_builder"),
    pytest.param(_build_modelopt_teacher, id="modelopt_kd_teacher"),
]


class TestModelOptBuildersOnTheirOwnPipelineGrid:
    """A builder given a PP=2 collection builds its stage while the global grid has PP=1."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("forbid_global_pipeline_rank", [True, False])
    @pytest.mark.parametrize("build", BUILDERS)
    def test_builder_builds_the_stage_of_its_collection(
        self, monkeypatch, build, forbid_global_pipeline_rank
    ):
        if Utils.world_size % PP_SIZE != 0:
            pytest.skip(f"needs a world size divisible by {PP_SIZE}")
        # The global grid has one pipeline stage, so reading it selects stage 0 on every rank.
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(123)
        grid = HyperCommGrid([1, 1, PP_SIZE, Utils.world_size // PP_SIZE], ["tp", "cp", "pp", "dp"])
        pg_collection = ProcessGroupCollection(
            tp=grid.create_pg("tp"), cp=grid.create_pg("cp"), pp=grid.create_pg("pp"), embd=None
        )
        pp_rank = pg_collection.pp.rank()

        if forbid_global_pipeline_rank:
            _forbid_global_pipeline_rank(monkeypatch)
        model = build(_dense_config(), pg_collection, monkeypatch)

        layer_numbers = [layer.layer_number for layer in model.decoder.layers]
        assert layer_numbers == DENSE_STAGE_LAYER_NUMBERS[pp_rank], f"{pp_rank=}"
