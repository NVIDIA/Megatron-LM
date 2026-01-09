# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed

from megatron.core import mpu, parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec as gpt_te_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args, set_global_variables, unset_global_variables
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.dist_checkpointing.models.common import (
    common_test_parallel_reconfiguration_e2e,
)
from tests.unit_tests.test_utilities import Utils


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
    args.use_megatron_fsdp = False
    args.dist_ckpt_save_pre_mcore_014 = False
    args.dist_ckpt_optim_fully_reshardable = False
    args.distrib_optim_fully_reshardable_mem_efficient = False
    args.data_path = None
    args.mock_data = True
    args.data_args_path = None
    args.train_data_path, args.valid_data_path, args.test_data_path = None, None, None
    args.per_split_data_args_path = None
    args.rank = int(os.getenv('RANK', '0'))
    # Model config
    args.num_layers = 8
    args.hidden_size = 128
    args.num_attention_heads = 8
    # Ckpt format
    args.ckpt_format = "torch_dist"
    args.tokenizer_type = 'NullTokenizer'
    args.vocab_size = 16384
    args.make_vocab_size_divisible_by = 128
    args.padded_vocab_size = 16512
    args.reset_position_ids = False
    args.reset_attention_mask = False
    args.eod_mask_loss = False
    args.multi_latent_attention = False
    args.heterogeneous_layers_config_path = None
    args.no_persist_layer_norm = True
    args.apply_layernorm_1p = False
    args.norm_epsilon = 1e-6
    args.params_dtype = torch.bfloat16
    args.overlap_p2p_comm = True
    args.rotary_interleaved = False
    args.decoder_first_pipeline_num_layers = None
    args.decoder_last_pipeline_num_layers = None
    args.fp8_param_gather = False
    args.swiglu = True
    args.bias_swiglu_fusion = False
    args.squared_relu = False
    args.init_method_xavier_uniform = False
    args.quick_geglu = False
    args.group_query_attention = True
    args.config_logger_dir = None
    args.rope_type = None
    args.is_hybrid_model = False
    args.num_query_groups = 4
    args.cp_comm_type = ['p2p']
    args.seed = 123
    args.rampup_batch_size = None
    args.global_batch_size = 256
    args.micro_batch_size = 1
    args.decrease_batch_size_if_needed = False
    args.enable_one_logger = True
    args.one_logger_async = False
    args.adlr_autoresume = False
    args.adlr_autoresume_interval = 1000
    args.timing_log_level = 0
    args.timing_log_option = "minmax"
    args.enable_experimental = False
    args.exit_signal_handler = False
    args.disable_jit_fuser = False
    args.one_logger_project = "megatron-lm"
    args.one_logger_run_name = None
    args.tensorboard_dir = None
    args.tensorboard_queue_size = 1000
    args.wandb_project = ""
    args.wandb_exp_name = ""
    args.wandb_save_dir = ""
    args.wandb_entity = ""
    args.iteration = 0
    args.train_samples = 100000
    args.full_validation = False
    args.train_iters = 100
    args.dataloader_type = "single"
    args.eval_iters = 32
    args.eval_interval = 500
    args.save_interval = 500
    args.exit_interval = None
    args.exit_duration_in_mins = None
    args.legacy_tokenizer = True
    args.split = "99,1,0"
    args.multiple_validation_sets = False
    args.num_dataset_builder_threads = 1
    args.num_workers = 2
    args.skip_train = False
    args.data_cache_path = None
    args.mmap_bin_files = False
    args.object_storage_cache_path = None
    args.mid_level_dataset_surplus = 0.005
    args.create_attention_mask_in_dataloader = False
    args.sft_mock_dataset_config_json = None
    args.hybrid_context_parallel_scheduler = "balanced"
    args.check_for_nan_in_loss_and_grad = False
    args.check_for_spiky_loss = False
    args.sequence_parallel = False
    args.untie_embeddings_and_output_weights = True
    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0
    args.moe_ffn_hidden_size = 768
    args.use_legacy_models = False
    args.allow_ambiguous_pad_tokens = False
    args.add_bias_linear = False
    args.sft = True
    args.overlap_moe_expert_parallel_comm = False
    args.sft_mock_dataset_config_json = '{"mode":"distribution","type":"lognormal","min_seq_len":1024,"max_seq_len":8192,"mean_seq_len":4096,"lognormal_sigma":1.1}'
    args.world_size = 8
    args.seq_length = 8192
    args.max_position_embeddings = 8192
    args.max_seqlen_per_cp_rank = None
    args.variable_seq_lengths = False
    args.moe_token_dispatcher_type = "allgather"

    yield args


def initialize_gpt_model(
    args,
    layer_spec_fn=gpt_te_spec,
    virtual_pipeline_model_parallel_size=None,
    is_moe=False,
    with_mtp=False,
    **config_kwargs,
):
    torch.manual_seed(args.seed)
    model_parallel_cuda_manual_seed(args.seed)

    # NOTE: This unit test uses TP/PP/CP (and optionally hybrid-CP). We must pass the
    # model-parallel sizes into TransformerConfig; otherwise it defaults to cp=1 which
    # breaks RoPE sharding (cp_group.size()>1 but config.context_parallel_size==1).
    default_config_kwargs = dict(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        use_cpu_initialization=True,
        pipeline_dtype=args.params_dtype,
        bf16=args.bf16,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        sequence_parallel=args.sequence_parallel,
        hybrid_context_parallel=args.hybrid_context_parallel,
        hybrid_context_parallel_scheduler=getattr(
            args, "hybrid_context_parallel_scheduler", "balanced"
        ),
        sft_sequence_packing=getattr(args, "sft_sequence_packing", False),
        max_seqlen_per_cp_rank=getattr(args, "max_seqlen_per_cp_rank", None),
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        hidden_dropout=args.hidden_dropout,
        attention_dropout=args.attention_dropout,
        mtp_num_layers=1 if with_mtp else None,
        mtp_loss_scaling_factor=1.0 if with_mtp else None,
        variable_seq_lengths=args.variable_seq_lengths,
        moe_token_dispatcher_type=args.moe_token_dispatcher_type,
    )
    default_config_kwargs.update(**config_kwargs)

    transformer_config = TransformerConfig(**default_config_kwargs)
    if is_moe:
        # transformer_config.moe_layer_freq = [0, 1, 1, 1, 1, 0, 1, 0]
        transformer_config.moe_ffn_hidden_size = args.moe_ffn_hidden_size
        transformer_config.num_moe_experts = args.num_experts
        transformer_config.add_bias_linear = args.add_bias_linear

    model = []
    for i in range(virtual_pipeline_model_parallel_size or 1):
        if is_moe:
            layer_spec = layer_spec_fn(transformer_config, use_transformer_engine=True, vp_stage=i)
        else:
            layer_spec = layer_spec_fn()

        if with_mtp and mtp_on_this_rank(transformer_config, ignore_virtual=False, vp_stage=i):
            if is_moe:
                transformer_layer_spec_for_mtp = gpt_te_spec(transformer_config)
            else:
                transformer_layer_spec_for_mtp = layer_spec
            mtp_block_spec = get_gpt_mtp_block_spec(
                transformer_config,
                transformer_layer_spec_for_mtp,
                use_transformer_engine=True,
                vp_stage=i,
            )
        else:
            mtp_block_spec = None

        # print("========================")
        # print("[DEBUG] mtp_block_spec is ", mtp_block_spec)
        # exit()
        pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
        post_process = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
        this_model = (
            GPTModel(
                config=transformer_config,
                transformer_layer_spec=layer_spec,
                vocab_size=args.padded_vocab_size,
                pre_process=pre_process,
                post_process=post_process,
                position_embedding_type="rope",
                vp_stage=i,
                mtp_block_spec=mtp_block_spec,
                share_embeddings_and_output_weights=False,
                max_sequence_length=args.seq_length,
            )
            .bfloat16()
            .cuda()
        )
        this_model.model_type = ModelType.encoder_or_decoder
        model.append(this_model)

    if virtual_pipeline_model_parallel_size is None:
        model = model[0]
    return model


def get_data_iterator(args):
    """
    Get the data iterator for the test.

    Args:
        args: args namespace
    """
    from megatron.core.datasets.blended_megatron_dataset_builder import (
        BlendedMegatronDatasetBuilder,
    )
    from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
    from megatron.training import get_tokenizer
    from megatron.training.datasets.sft_dataset import MockSFTDataset, MockSFTLowLevelDataset
    from megatron.training.training import build_train_valid_test_data_iterators
    from megatron.training.utils import (
        get_batch_on_this_cp_rank,
        get_batch_on_this_tp_rank,
        get_blend_and_blend_per_split,
        is_first_or_last_pipeline_stage,
    )
    from pretrain_gpt import is_dataset_built_on_rank, train_valid_test_datasets_provider

    blend, blend_per_split = get_blend_and_blend_per_split(args)
    # rebuild_tokenizer(args)
    tokenizer = get_tokenizer()
    dataset_config = GPTDatasetConfig(
        random_seed=123,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split='969, 30, 1',
        tokenizer=tokenizer,
        create_attention_mask=False,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        context_parallel_size=args.context_parallel_size,
        data_parallel_size=args.data_parallel_size,
        sequence_parallel_size=args.tensor_model_parallel_size,
        hybrid_context_parallel=args.hybrid_context_parallel,
        sft_mock_dataset_config_json=args.sft_mock_dataset_config_json,
        sft_sequence_packing=args.sft_sequence_packing,
    )
    train_ds, test_ds, valid_ds = BlendedMegatronDatasetBuilder(
        MockSFTDataset,
        [100000, 2560, 2560],
        partial(is_dataset_built_on_rank, vp_stage=None),
        dataset_config,
    ).build()

    train_data_iterator, valid_data_iterator, test_data_iterator = (
        build_train_valid_test_data_iterators(train_valid_test_datasets_provider)
    )

    return train_data_iterator


# Dense and MoE Models
@pytest.mark.parametrize(
    ('tp_pp_cp_vpp', 'is_moe'),
    [
        ((1, 2, 1, None), True),
        # ((1, 4, 1, None), True),
        # ((2, 2, 4, None), True),
        # ((2, 4, 4, None), True),
        # ((2, 1, 4, None), True),
        # ((1, 1, 2, None), True),
        # ((1, 2, 1, None), False),
        # ((1, 4, 1, None), False),
        # ((2, 2, 2, None), False),
        # ((2, 4, 1, None), False),
        # ((2, 1, 4, None), False),
        # ((1, 1, 2, None), False),
    ],
)
def test_packing_and_hybrid_cp(create_args, tp_pp_cp_vpp, is_moe):
    def _assert_loss_close(loss, loss_ref, *, atol=1e-6, msg="loss mismatch"):
        # Megatron's forward_backward_func(forward_only=True) typically returns a list of dicts
        # (per-microbatch), where each dict maps loss-name -> tensor.
        def _normalize_if_sum_and_count(t: torch.Tensor) -> torch.Tensor:
            # Some Megatron losses are returned as a 2-vector: [loss_sum, num_tokens].
            # In that case, compare per-token loss to make results comparable across
            # different effective sequence lengths (e.g., packing vs non-packing).
            if torch.is_tensor(t) and t.dim() == 1 and t.numel() == 2:
                denom = t[1].clamp_min(1.0)
                return t[0] / denom
            return t

        if isinstance(loss, dict):
            assert isinstance(loss_ref, dict), f"{msg}: type {type(loss)} vs {type(loss_ref)}"
            assert loss.keys() == loss_ref.keys(), f"{msg}: keys {loss.keys()} vs {loss_ref.keys()}"
            for k in loss.keys():
                v = loss[k]
                v_ref = loss_ref[k]
                if torch.is_tensor(v) and torch.is_tensor(v_ref):
                    v_n = _normalize_if_sum_and_count(v)
                    v_ref_n = _normalize_if_sum_and_count(v_ref)
                    assert torch.allclose(v_n, v_ref_n, atol=atol), f"{msg} at key={k}"
                else:
                    assert v == v_ref, f"{msg} at key={k}: {v} vs {v_ref}"
        else:
            assert torch.is_tensor(loss) and torch.is_tensor(
                loss_ref
            ), f"{msg}: expected tensors, got {type(loss)} and {type(loss_ref)}"
            loss_n = _normalize_if_sum_and_count(loss)
            loss_ref_n = _normalize_if_sum_and_count(loss_ref)
            assert torch.allclose(loss_n, loss_ref_n, atol=atol), msg

    args = create_args
    losses_reduced_baseline, is_last_stage = dummy_forward_func(
        args,
        is_sft_sequence_packing=False,
        is_hybrid_context_parallel=False,
        tp_pp_cp_vpp=tp_pp_cp_vpp,
        is_moe=is_moe,
    )
    losses_reduce_packing, _ = dummy_forward_func(
        args,
        is_sft_sequence_packing=True,
        is_hybrid_context_parallel=False,
        tp_pp_cp_vpp=tp_pp_cp_vpp,
        is_moe=is_moe,
    )
    losses_reduced_hybrid, _ = dummy_forward_func(
        args,
        is_sft_sequence_packing=True,
        is_hybrid_context_parallel=True,
        tp_pp_cp_vpp=tp_pp_cp_vpp,
        is_moe=is_moe,
    )
    # NOTE: dummy_forward_func() destroys model-parallel groups before returning.
    # So we must not query parallel_state after it returns.
    if is_last_stage:
        for loss, loss_baseline in zip(losses_reduce_packing, losses_reduced_baseline):
            _assert_loss_close(
                loss,
                loss_baseline,
                atol=1e-6,
                msg="losses_reduce_packing and losses_reduced_baseline are not equal",
            )
        for loss, loss_baseline in zip(losses_reduced_hybrid, losses_reduced_baseline):
            _assert_loss_close(
                loss,
                loss_baseline,
                atol=1e-6,
                msg="losses_reduced_hybrid and losses_reduced_baseline are not equal",
            )
    print("test_packing_and_hybrid_cp passed with tp_pp_cp_vpp: ", tp_pp_cp_vpp, "is_moe: ", is_moe)


def dummy_forward_func(
    args, is_sft_sequence_packing, is_hybrid_context_parallel, tp_pp_cp_vpp, is_moe
):
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from pretrain_gpt import forward_step, get_batch

    args.sft_sequence_packing = is_sft_sequence_packing
    args.hybrid_context_parallel = is_hybrid_context_parallel

    if is_moe:
        args.num_experts = 4

    def set_tp_pp_vpp(tp, pp, cp, vpp=None, destroy_first=True):
        if destroy_first:
            Utils.destroy_model_parallel()
        args.tensor_model_parallel_size = tp
        args.pipeline_model_parallel_size = pp
        args.virtual_pipeline_model_parallel_size = vpp
        args.data_parallel_size = 8 // (tp * pp)
        # Hybrid-CP requires context_parallel_size == 1; CP is achieved via DPxCP hybrid groups.
        args.context_parallel_size = 1 if args.hybrid_context_parallel else cp
        if tp > 1:
            args.sequence_parallel = True
        Utils.initialize_model_parallel(
            tp,
            pp,
            vpp,
            context_parallel_size=args.context_parallel_size,
            hybrid_context_parallel=args.hybrid_context_parallel,
            min_hybrid_context_parallel_size=getattr(args, "min_hybrid_context_parallel_size", 1),
        )

    set_tp_pp_vpp(*tp_pp_cp_vpp)
    if is_sft_sequence_packing:
        args.variable_seq_lengths = True
        # TODO(tailaim): add support for other dispatcher types
        print(
            f"Setting moe_token_dispatcher_type to alltoall for sft sequence packing with pipeline parallelism"
        )
        args.moe_token_dispatcher_type = "alltoall"
        if is_hybrid_context_parallel:
            args.max_seqlen_per_cp_rank = args.seq_length // args.data_parallel_size
        else:
            args.max_seqlen_per_cp_rank = args.seq_length // args.context_parallel_size

    set_global_variables(args)
    # set_args(args)

    # init_num_microbatches_calculator(0, None, 256, 1, args.data_parallel_size)

    layer_spec_fn = get_gpt_decoder_block_spec if is_moe else gpt_te_spec
    model = initialize_gpt_model(
        args,
        layer_spec_fn=layer_spec_fn,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        is_moe=True,
        with_mtp=False,
    )
    model = model if isinstance(model, list) else [model]

    data_iterator = get_data_iterator(args)

    # #debugmtl
    # print(f"iterator: {next(data_iterator)}")

    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=[data_iterator] * len(model),
        model=model,
        num_microbatches=args.global_batch_size
        // args.data_parallel_size
        * args.context_parallel_size
        // args.micro_batch_size,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        forward_only=True,
    )

    # Capture pipeline stage info BEFORE destroying model-parallel state.
    is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=True)

    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()

    unset_global_variables()

    return losses_reduced, is_last_stage
