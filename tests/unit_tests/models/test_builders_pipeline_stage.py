# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model builders build the pipeline stage of the process-group collection they are given."""

import json
from argparse import Namespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
from megatron.core.transformer.identity_op import IdentityOp
from tests.unit_tests.test_utilities import Utils

PP_SIZE = 2
VOCAB_SIZE = 128
SEQUENCE_LENGTH = 32

# Three layers split 1 + 2 over two pipeline stages; the last layer has no MLP. For each
# pipeline rank: (layer number, whether the layer has an MLP) of every layer it builds.
STAGE_LAYERS = [[(1, True)], [(2, True), (3, False)]]


def _heterogeneous_config():
    attention = {"no_op": False, "replace_with_linear": False, "num_query_groups": 4}
    mlp = {"no_op": False, "replace_with_linear": False, "ffn_hidden_size": 128}
    no_mlp = {"no_op": True, "replace_with_linear": False, "ffn_hidden_size": None}
    block_configs = [
        {"attention": attention, "mlp": mlp},
        {"attention": attention, "mlp": mlp},
        {"attention": attention, "mlp": no_mlp},
    ]
    return HeterogeneousTransformerConfig(
        num_layers=3,
        hidden_size=64,
        num_attention_heads=4,
        normalization="RMSNorm",
        transformer_impl="local",
        pipeline_model_parallel_size=PP_SIZE,
        num_layers_in_first_pipeline_stage=1,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        perform_initialization=False,
        heterogeneous_layers_config_encoded_json=json.dumps({"block_configs": block_configs}),
    )


def _model_args(**overrides):
    """The arguments the builders read to construct a GPT model with a heterogeneous spec."""
    args = Namespace(
        spec=None,
        transformer_impl="local",
        experimental_attention_variant=None,
        num_experts=None,
        # Any path selects the heterogeneous spec; the config carries the block configs.
        heterogeneous_layers_config_path="block_configs.json",
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
    )
    vars(args).update(overrides)
    return args


def _build_with_gpt_builder(config, pg_collection, monkeypatch):
    from gpt_builders import gpt_builder

    return gpt_builder(
        _model_args(),
        is_pp_first_stage(pg_collection.pp),
        is_pp_last_stage(pg_collection.pp),
        config=config,
        pg_collection=pg_collection,
    )


def _build_with_gpt_model_builder(config, pg_collection, monkeypatch):
    from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig

    model_config = GPTModelConfig(transformer=config, vocab_size=VOCAB_SIZE)
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
        export_te_mcore_model=False,
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
    model_builder = _modelopt_model_builder(monkeypatch, _model_args(export_te_mcore_model=False))
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
    pytest.param(_build_with_gpt_builder, id="gpt_builder"),
    pytest.param(_build_with_gpt_model_builder, id="GPTModelBuilder"),
    pytest.param(_build_with_modelopt_builder, id="modelopt_gpt_hybrid_builder"),
    pytest.param(_build_modelopt_teacher, id="modelopt_kd_teacher"),
]


class TestBuildersOnTheirOwnPipelineGrid:
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

            def forbidden():
                raise RuntimeError("the global pipeline rank was read")

            monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", forbidden)
        model = build(_heterogeneous_config(), pg_collection, monkeypatch)

        layers = [
            (layer.layer_number, not isinstance(layer.mlp, IdentityOp))
            for layer in model.decoder.layers
        ]
        assert layers == STAGE_LAYERS[pp_rank], f"{pp_rank=}"


@pytest.mark.parametrize(
    "args_overrides, spec_builder",
    [
        pytest.param(
            {"experimental_attention_variant": "gated_delta_net"},
            "get_transformer_block_with_experimental_attention_variant_spec",
            id="experimental_attention_variant",
        ),
        pytest.param({"num_experts": 4}, "get_gpt_decoder_block_spec", id="moe"),
        pytest.param({}, "get_gpt_heterogeneous_layer_spec", id="heterogeneous"),
    ],
)
@pytest.mark.parametrize("has_collection", [True, False])
def test_gpt_builder_passes_its_pipeline_position(args_overrides, spec_builder, has_collection):
    """gpt_builder gives each block spec its stage; without a collection, the global one."""
    from gpt_builders import gpt_builder

    if has_collection:
        pg_collection = Mock()
        pg_collection.pp.rank.return_value = 1
    else:
        pg_collection = None
    args = _model_args(
        normalization="RMSNorm", qk_l2_norm=False, mtp_num_layers=1, **args_overrides
    )

    # The mocked block spec has no layers, so the MTP block takes the plain layer spec.
    with (
        patch("gpt_builders.GPTModel"),
        patch(f"gpt_builders.{spec_builder}") as block_spec,
        patch("gpt_builders._get_transformer_layer_spec"),
        patch("gpt_builders.get_gpt_mtp_block_spec") as mtp_block_spec,
    ):
        gpt_builder(args, True, True, vp_stage=0, config=MagicMock(), pg_collection=pg_collection)

    expected_pp_rank = 1 if has_collection else None
    assert block_spec.call_args.kwargs["vp_stage"] == 0
    assert block_spec.call_args.kwargs["pp_rank"] == expected_pp_rank
    assert mtp_block_spec.call_args.kwargs["pp_rank"] == expected_pp_rank
