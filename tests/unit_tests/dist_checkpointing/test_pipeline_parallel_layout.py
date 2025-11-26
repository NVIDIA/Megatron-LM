# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core import mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec as gpt_te_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.dist_checkpointing.models.common import (
    common_test_parallel_reconfiguration_e2e,
)
from tests.unit_tests.test_utilities import Utils


def initialize_gpt_model(
    seed,
    layer_spec_fn=gpt_te_spec,
    vocab_size=128,
    virtual_pipeline_model_parallel_size=None,
    is_moe=False,
    **config_kwargs
):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs = dict(
        num_layers=8,
        hidden_size=16,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
    )
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)
    if is_moe:
        transformer_config.moe_layer_freq = [0, 1, 1, 1, 1, 0, 1, 0]
        transformer_config.moe_ffn_hidden_size = 128
        transformer_config.num_moe_experts = 4
    model = []
    for i in range(virtual_pipeline_model_parallel_size or 1):
        if is_moe:
            layer_spec = layer_spec_fn(transformer_config, use_transformer_engine=True, vp_stage=i)
        else:
            layer_spec = layer_spec_fn()
        pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
        post_process = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
        this_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=4,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=i,
        )
        this_model.model_type = ModelType.encoder_or_decoder
        model.append(this_model)

    with torch.no_grad():
        for m in model:
            for p in m.parameters():
                p.random_()
    if virtual_pipeline_model_parallel_size is None:
        model = model[0]
    return model


# Dense Model Only
@pytest.mark.internal
def test_save_and_load_checkpoint_pp(tmp_path_dist_ckpt):
    src_layer_spec_fn = gpt_te_spec
    dst_layer_spec_fn = gpt_te_spec
    use_fpsl = False
    load_order = 'tp-dp-pp'
    store_order = 'tp-dp-pp'
    src_tp_pp = (1, 4)
    src_model_init_kwargs = {
        "pipeline_model_parallel_layout": [
            ["embedding"] + ["decoder"] * 2,
            ["decoder"] * 3,
            [],
            ["decoder"] * 3 + ["loss"],
        ]
    }
    dest_tp_pp = (2, 1)

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
        src_model_init_kwargs=src_model_init_kwargs,
    )


@pytest.fixture
def create_args():
    """Setup dummy args."""
    args = SimpleNamespace()
    args.finetune = False
    args.non_persistent_global_ckpt_dir = None
    args.non_persistent_ckpt_type = None
    args.non_persistent_save_interval = None
    args.exit_on_missing_checkpoint = True
    args.async_save = False
    args.data_parallel_random_init = False
    args.log_progress = False
    args.ckpt_fully_parallel_save = False
    args.ckpt_fully_parallel_load = False
    args.auto_detect_ckpt_format = False
    args.retro_add_retriever = False
    args.ckpt_convert_update_legacy_dist_opt_format = False
    args.ckpt_step = None
    args.use_dist_ckpt = True
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    args.vocab_file = None
    args.add_position_embedding = False
    args.ckpt_assume_constant_structure = True
    args.dist_ckpt_strictness = "assume_ok_unexpected"
    args.fp16 = False
    args.bf16 = True
    args.no_save_optim = True
    args.no_save_rng = True
    args.no_load_optim = True
    args.no_load_rng = True
    args.use_distributed_optimizer = True

    yield args


# Dense and MoE Models
@pytest.mark.parametrize(
    ('src_tp_pp_vpp', 'dst_tp_pp_vpp', 'src_pp_layout', 'dst_pp_layout', 'is_moe'),
    [
        (
            (1, 4, 2),
            (1, 2, 1),
            [
                ["embedding"],
                ["decoder"],
                ["decoder"] * 2,
                ["decoder"],
                [],
                ["decoder"],
                ["decoder"],
                ["decoder"] * 2 + ["loss"],
            ],
            [["embedding"] + ["decoder"] * 4, ["decoder"] * 4 + ["loss"]],
            False,
        ),
        (
            (1, 4, 2),
            (2, 1, None),
            [
                ["embedding"],
                ["decoder"],
                ["decoder"] * 2,
                ["decoder"],
                [],
                ["decoder"],
                ["decoder"],
                ["decoder"] * 2 + ["loss"],
            ],
            None,
            False,
        ),
        (
            (4, 1, None),
            (1, 2, 1),
            None,
            [["embedding"] + ["decoder"] * 4, ["decoder"] * 4 + ["loss"]],
            False,
        ),
        (
            (1, 2, 1),
            (1, 4, 2),
            [["embedding"] + ["decoder"] * 4, ["decoder"] * 4 + ["loss"]],
            [
                ["embedding"],
                ["decoder"],
                ["decoder"] * 2,
                ["decoder"],
                [],
                ["decoder"],
                ["decoder"],
                ["decoder"] * 2 + ["loss"],
            ],
            False,
        ),
        (
            (1, 4, 2),
            (1, 2, 1),
            [
                ["embedding"],
                ["decoder"],
                ["decoder"] * 2,
                ["decoder"],
                [],
                ["decoder"],
                ["decoder"],
                ["decoder"] * 2 + ["loss"],
            ],
            [["embedding"] + ["decoder"] * 4, ["decoder"] * 4 + ["loss"]],
            True,
        ),
        (
            (1, 4, 2),
            (2, 1, None),
            [
                ["embedding"],
                ["decoder"],
                ["decoder"] * 2,
                ["decoder"],
                [],
                ["decoder"],
                ["decoder"],
                ["decoder"] * 2 + ["loss"],
            ],
            None,
            True,
        ),
        (
            (4, 1, None),
            (1, 2, 1),
            None,
            [["embedding"] + ["decoder"] * 4, ["decoder"] * 4 + ["loss"]],
            True,
        ),
        (
            (1, 2, 1),
            (1, 4, 2),
            [["embedding"] + ["decoder"] * 4, ["decoder"] * 4 + ["loss"]],
            [
                ["embedding"],
                ["decoder"],
                ["decoder"] * 2,
                ["decoder"],
                [],
                ["decoder"],
                ["decoder"],
                ["decoder"] * 2 + ["loss"],
            ],
            True,
        ),
    ],
)
def test_save_and_load_checkpoint_vpp(
    create_args,
    tmp_path_dist_ckpt,
    src_tp_pp_vpp,
    src_pp_layout,
    dst_tp_pp_vpp,
    dst_pp_layout,
    is_moe,
):
    args = create_args
    # Model config
    args.num_layers = 8
    args.hidden_size = 8
    args.num_attention_heads = 8
    # Ckpt format
    args.ckpt_format = "torch_dist"
    set_args(args)

    def set_tp_pp_vpp(tp, pp, vpp=None, pp_layout=None, destroy_first=True):
        if destroy_first:
            Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tp, pp, vpp)
        args.tensor_model_parallel_size = tp
        args.pipeline_model_parallel_size = pp
        args.virtual_pipeline_model_parallel_size = vpp
        args.pipeline_model_parallel_layout = pp_layout

    def set_ckpt_path(ckpt_path):
        args.save = ckpt_path
        args.load = ckpt_path

    set_tp_pp_vpp(*src_tp_pp_vpp, pp_layout=src_pp_layout, destroy_first=False)
    init_num_microbatches_calculator(0, None, 1, 1, 1)

    iteration = 123
    layer_spec_fn = get_gpt_decoder_block_spec if is_moe else gpt_te_spec
    model = initialize_gpt_model(
        1,
        layer_spec_fn=layer_spec_fn,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_layout=args.pipeline_model_parallel_layout,
        is_moe=is_moe,
    )
    model = model if isinstance(model, list) else [model]
    optimizer = None
    opt_param_scheduler = None
    num_floating_point_operations_so_far = 456

    with TempNamedDir(
        tmp_path_dist_ckpt / 'test_gpt_model_reconfiguration_model_A'
    ) as ckpt_dir_A, TempNamedDir(
        tmp_path_dist_ckpt / 'test_gpt_model_reconfiguration_model_B'
    ) as ckpt_dir_B:
        set_ckpt_path(ckpt_dir_A)
        save_checkpoint(
            iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far
        )

        expected_ckpt_path = args.save / "iter_0000123" / ".metadata"
        assert os.path.exists(expected_ckpt_path)

        set_tp_pp_vpp(*dst_tp_pp_vpp, pp_layout=dst_pp_layout)
        new_model = initialize_gpt_model(
            2,
            layer_spec_fn=layer_spec_fn,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_layout=args.pipeline_model_parallel_layout,
            is_moe=is_moe,
        )
        new_model = new_model if isinstance(new_model, list) else [new_model]

        load_checkpoint(new_model, optimizer, opt_param_scheduler, strict=False)
        set_ckpt_path(ckpt_dir_B)
        save_checkpoint(
            iteration,
            new_model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
        )

        set_tp_pp_vpp(1, 1)
        set_ckpt_path(ckpt_dir_A)
        model_A = initialize_gpt_model(
            123,
            layer_spec_fn=layer_spec_fn,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_layout=args.pipeline_model_parallel_layout,
            is_moe=is_moe,
        )
        load_checkpoint([model_A], optimizer, opt_param_scheduler, strict=False)

        set_ckpt_path(ckpt_dir_B)
        model_B = initialize_gpt_model(
            321,
            layer_spec_fn=layer_spec_fn,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_layout=args.pipeline_model_parallel_layout,
            is_moe=is_moe,
        )
        load_checkpoint([model_B], optimizer, opt_param_scheduler, strict=False)

        for k in model_A.state_dict():
            if "_extra_state" in k:  # Ignore extra states
                continue
            tensor_a = model_A.state_dict()[k]
            tensor_b = model_B.state_dict()[k]
            assert tensor_a is not None, k
            assert tensor_b is not None, k
            assert torch.equal(tensor_a, tensor_b), k

    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()
