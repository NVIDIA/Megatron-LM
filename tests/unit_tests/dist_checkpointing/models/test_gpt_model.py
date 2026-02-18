# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import functools
import pathlib
from collections.abc import Callable
from typing import Optional

import pytest
import torch

from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec as gpt_local_spec
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec as gpt_te_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.dist_checkpointing.models.common import (
    common_test_parallel_reconfiguration_e2e,
    common_test_simple_sharded_state_dict_save_load,
    common_test_state_dict_comparison,
    common_test_vocab_size_padding_change,
)
from tests.unit_tests.test_utilities import Utils

# List of model spec functions
_spec_fn_list: list[Callable[[], ModuleSpec]] = [gpt_te_spec, gpt_local_spec]
_gpt_te_spec_op_fuser: Optional[Callable] = None
if is_te_min_version("1.13.0"):
    _gpt_te_spec_op_fuser = functools.partial(gpt_te_spec, use_te_op_fuser=True)
    _spec_fn_list.append(_gpt_te_spec_op_fuser)


def initialize_gpt_model(seed, layer_spec_fn=gpt_te_spec, vocab_size=128, **config_kwargs):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs = dict(
        num_layers=8,
        hidden_size=16,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
    )
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    pre_process = ps.is_pipeline_first_stage()
    post_process = ps.is_pipeline_last_stage()
    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=layer_spec_fn(),
        vocab_size=vocab_size,
        max_sequence_length=4,
        pre_process=pre_process,
        post_process=post_process,
    )

    with torch.no_grad():
        for p in model.parameters():
            p.random_()
    return model


class TestGPTModel:
    @pytest.mark.parametrize('src_layer_spec_fn', _spec_fn_list)
    @pytest.mark.parametrize('dst_layer_spec_fn', _spec_fn_list)
    def test_sharded_state_dict_save_load(
        self,
        tmp_path_dist_ckpt: pathlib.Path,
        src_layer_spec_fn: Callable[[], ModuleSpec],
        dst_layer_spec_fn: Callable[[], ModuleSpec],
    ):
        common_test_simple_sharded_state_dict_save_load(
            initialize_gpt_model, tmp_path_dist_ckpt, src_layer_spec_fn, dst_layer_spec_fn
        )


class TestGPTModelReconfiguration:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        (
            'use_fpsl',
            'load_order',
            'store_order',
            'src_tp_pp',
            'dest_tp_pp',
            'singleton_local_shards',
            'src_layer_spec_fn',
            'dst_layer_spec_fn',
        ),
        [
            (False, 'tp-dp-pp', 'tp-dp-pp', (2, 4), (4, 2), True, gpt_te_spec, gpt_te_spec),
            (False, 'tp-pp-dp', 'tp-pp-dp', (1, 8), (8, 1), False, gpt_te_spec, gpt_te_spec),
            (True, 'tp-dp-pp', 'tp-pp-dp', (2, 1), (1, 8), True, gpt_te_spec, gpt_te_spec),
            (False, 'tp-dp-pp', 'tp-dp-pp', (1, 1), (2, 2), True, gpt_te_spec, gpt_te_spec),
            (True, 'tp-pp-dp', 'tp-pp-dp', (2, 1), (1, 8), False, gpt_local_spec, gpt_local_spec),
            (False, 'tp-dp-pp', 'tp-pp-dp', (1, 1), (2, 4), False, gpt_te_spec, gpt_local_spec),
            (True, 'tp-dp-pp', 'tp-dp-pp', (2, 4), (4, 2), True, gpt_local_spec, gpt_te_spec),
            (False, 'tp-pp-dp', 'tp-pp-dp', (2, 1), (1, 8), False, gpt_te_spec, gpt_local_spec),
            (False, 'tp-dp-pp', 'tp-pp-dp', (2, 4), (2, 4), True, gpt_local_spec, gpt_local_spec),
            (
                False,
                'tp-dp-pp',
                'tp-dp-pp',
                (2, 4),
                (4, 2),
                False,
                gpt_te_spec,
                _gpt_te_spec_op_fuser,
            ),
            (
                False,
                'tp-dp-pp',
                'tp-dp-pp',
                (2, 4),
                (4, 2),
                False,
                _gpt_te_spec_op_fuser,
                gpt_te_spec,
            ),
        ],
    )
    def test_parallel_reconfiguration_e2e(
        self,
        tmp_path_dist_ckpt: pathlib.Path,
        src_tp_pp: tuple[int, int],
        dest_tp_pp: tuple[int, int],
        src_layer_spec_fn: Optional[Callable[[], ModuleSpec]],
        dst_layer_spec_fn: Optional[Callable[[], ModuleSpec]],
        use_fpsl: bool,
        load_order: str,
        store_order: str,
        singleton_local_shards: bool,
    ):
        """Test model saving and loading with different TP/PP"""
        if src_layer_spec_fn is None or dst_layer_spec_fn is None:
            pytest.skip("Spec function is not supported")
        Utils.initialize_model_parallel(src_tp_pp[0], src_tp_pp[1])
        common_test_parallel_reconfiguration_e2e(
            initialize_gpt_model,
            tmp_path_dist_ckpt,
            src_tp_pp,
            dest_tp_pp,
            src_layer_spec_fn,
            dst_layer_spec_fn,
            use_fpsl,
            load_order,
            store_order,
            metadata={'singleton_local_shards': singleton_local_shards},
        )

    def test_state_dict_comparison(self, tmp_path_dist_ckpt):
        common_test_state_dict_comparison(initialize_gpt_model, tmp_path_dist_ckpt)

    @pytest.mark.parametrize(
        "vocab_size_base,src_tp_pp,dest_tp_pp",
        [
            (128, (2, 4), (4, 2)),
            (17, (1, 8), (8, 1)),
            (127, (1, 8), (8, 1)),
            (31123, (1, 1), (1, 8)),
            (17, (1, 1), (1, 8)),
        ],
    )
    def test_vocab_size_padding_change(
        self,
        tmp_path_dist_ckpt: pathlib.Path,
        vocab_size_base: int,
        src_tp_pp: tuple[int, int],
        dest_tp_pp: tuple[int, int],
    ) -> None:
        """Test model loading with different vocab size (caused by TP padding)."""
        Utils.initialize_model_parallel(src_tp_pp[0], src_tp_pp[1])
        common_test_vocab_size_padding_change(
            initialize_gpt_model, tmp_path_dist_ckpt, vocab_size_base, src_tp_pp, dest_tp_pp
        )

    @pytest.mark.parametrize(
        ('src_tp_pp', 'dest_tp_pp', 'src_layer_spec_fn', 'dst_layer_spec_fn'),
        [
            ((2, 4), (4, 2), gpt_te_spec, gpt_te_spec),
            ((2, 4), (4, 2), gpt_te_spec, gpt_local_spec),
            ((2, 4), (4, 2), gpt_local_spec, gpt_te_spec),
            ((2, 4), (4, 2), gpt_te_spec, _gpt_te_spec_op_fuser),
            ((2, 4), (4, 2), _gpt_te_spec_op_fuser, gpt_te_spec),
        ],
    )
    def test_mlp_with_glu(
        self,
        tmp_path_dist_ckpt: pathlib.Path,
        src_tp_pp: tuple[int, int],
        dest_tp_pp: tuple[int, int],
        src_layer_spec_fn: Optional[Callable[[], ModuleSpec]],
        dst_layer_spec_fn: Optional[Callable[[], ModuleSpec]],
    ) -> None:
        """Test model loading when MLP activation is gated linear unit."""
        if src_layer_spec_fn is None or dst_layer_spec_fn is None:
            pytest.skip("Spec function is not supported")
        Utils.initialize_model_parallel(src_tp_pp[0], src_tp_pp[1])
        common_test_parallel_reconfiguration_e2e(
            functools.partial(initialize_gpt_model, gated_linear_unit=True),
            tmp_path_dist_ckpt,
            src_tp_pp,
            dest_tp_pp,
            src_layer_spec_fn,
            dst_layer_spec_fn,
            False,  # use_fpsl
        )
