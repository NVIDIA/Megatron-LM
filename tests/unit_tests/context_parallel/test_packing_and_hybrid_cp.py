# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy
import torch.distributed

from megatron.core import mpu, parallel_state
from megatron.core.datasets.data_schedule import PackingScheduler, wrap_dataloader, get_batch_on_this_rank_for_sequence_packing
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
from megatron.core.rerun_state_machine import RerunDataIterator
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
    args.sequence_packing_scheduler = None
    args.check_for_nan_in_loss_and_grad = False
    args.check_for_spiky_loss = False
    args.sequence_parallel = False
    args.untie_embeddings_and_output_weights = True
    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0
    args.moe_ffn_hidden_size = None
    args.use_legacy_models = False
    args.allow_ambiguous_pad_tokens = False
    args.add_bias_linear = False
    args.sft = True
    args.overlap_moe_expert_parallel_comm = False
    args.sft_mock_dataset_config_json = '{"mode":"distribution","type":"lognormal","min_seq_len":1024,"max_seq_len":8192,"mean_seq_len":4096,"lognormal_sigma":1.1}'
    args.world_size = 8
    args.seq_length = 8192
    args.max_position_embeddings = 8192
    args.max_seqlen_per_dp_cp_rank = None
    args.variable_seq_lengths = False
    args.moe_token_dispatcher_type = "allgather"
    args.moe_latent_size = None
    args.te_precision_config_file = None

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
        sequence_packing_scheduler=args.sequence_packing_scheduler,
        sequence_packing=getattr(args, "sequence_packing", False),
        max_seqlen_per_dp_cp_rank=getattr(args, "max_seqlen_per_dp_cp_rank", None),
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
        get_blend_and_blend_per_split
    )
    from pretrain_gpt import is_dataset_built_on_rank

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
        sequence_packing=args.sequence_packing,
    )
    train_ds, _, _ = BlendedMegatronDatasetBuilder(
        MockSFTDataset,
        [100000, 2560, 2560],
        partial(is_dataset_built_on_rank, vp_stage=None),
        dataset_config,
    ).build()
    
    is_tp_first = parallel_state.get_tensor_model_parallel_rank() == 0
    is_pp_first = parallel_state.get_pipeline_model_parallel_rank() == 0
    is_pp_last = (parallel_state.get_pipeline_model_parallel_rank() == 
        parallel_state.get_pipeline_model_parallel_world_size() - 1)

    if is_tp_first and (is_pp_first or is_pp_last):
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_size = args.data_parallel_size // args.context_parallel_size
        num_microbatches = args.global_batch_size // dp_size // args.micro_batch_size
        start_index = dp_rank * num_microbatches
        end_index = start_index + num_microbatches
        samples = [train_ds[i] for i in range(start_index, end_index)]
        if args.sequence_packing:
            data_iterator = RerunDataIterator(iter([samples]))
        else:
            for sample in samples:
                sample['tokens'] = sample['tokens'].unsqueeze(0)
                sample['labels'] = sample['labels'].unsqueeze(0)
                sample['loss_mask'] = sample['loss_mask'].unsqueeze(0)
                sample['position_ids'] = sample['position_ids'].unsqueeze(0)
            data_iterator = RerunDataIterator(iter(samples))
    else:
        data_iterator = None
        
    if (args.virtual_pipeline_model_parallel_size is not None and 
        args.virtual_pipeline_model_parallel_size > 1):
        vpp_size = args.virtual_pipeline_model_parallel_size
        if is_pp_first:
            data_iterator = [data_iterator] + [None for _ in range(vpp_size - 1)]
        elif is_pp_last:
            data_iterator = [None for _ in range(vpp_size - 1)] + [data_iterator]
        else:
            data_iterator = [None for _ in range(vpp_size)]
            
    return data_iterator

# Dense and MoE Models
@pytest.mark.parametrize(
    ('tp_pp_cp_vpp', 'is_moe'),
    [
        ((1, 2, 1, None), True),
        ((1, 4, 1, None), True),
        ((2, 2, 2, None), True),
        ((1, 1, 2, None), True),
        ((1, 2, 1, None), False),
        ((1, 4, 1, None), False),
        ((2, 2, 2, None), False),
        ((1, 1, 2, None), False),
    ],
)
def test_packing_and_hybrid_cp(create_args, tp_pp_cp_vpp, is_moe):
    
    def _compute_avg_loss(losses_list):
        """计算所有 micro-batches 的平均 loss"""
        total_loss_sum = 0.0
        total_tokens = 0.0
        for loss_dict in losses_list:
            if isinstance(loss_dict, dict) and 'lm loss' in loss_dict:
                t = loss_dict['lm loss']
                if torch.is_tensor(t) and t.dim() == 1 and t.numel() == 2:
                    total_loss_sum += t[0].item()
                    total_tokens += t[1].item()
        if total_tokens > 0:
            return total_loss_sum / total_tokens
        return 0.0

    args = create_args
    losses_reduced_baseline, is_last_stage = dummy_forward_func(
        args,
        is_sequence_packing=False,
        is_hybrid_context_parallel=False,
        tp_pp_cp_vpp=tp_pp_cp_vpp,
        is_moe=is_moe,
    )
    losses_reduce_packing, _ = dummy_forward_func(
        args,
        is_sequence_packing=True,
        is_hybrid_context_parallel=False,
        tp_pp_cp_vpp=tp_pp_cp_vpp,
        is_moe=is_moe,
    )
    losses_reduced_hybrid, _ = dummy_forward_func(
        args,
        is_sequence_packing=True,
        is_hybrid_context_parallel=True,
        tp_pp_cp_vpp=tp_pp_cp_vpp,
        is_moe=is_moe,
    )
    if is_last_stage and torch.distributed.get_rank() == 0:
        avg_baseline = _compute_avg_loss(losses_reduced_baseline)
        avg_packing = _compute_avg_loss(losses_reduce_packing)
        avg_hybrid = _compute_avg_loss(losses_reduced_hybrid)
        print(f"avg_loss_baseline: {avg_baseline:.6f}")
        print(f"avg_loss_packing: {avg_packing:.6f}")
        print(f"avg_loss_hybrid: {avg_hybrid:.6f}")
    
    # NOTE: dummy_forward_func() destroys model-parallel groups before returning.
    # So we must not query parallel_state after it returns.
    if is_last_stage:
        avg_baseline = _compute_avg_loss(losses_reduced_baseline)
        avg_packing = _compute_avg_loss(losses_reduce_packing)
        avg_hybrid = _compute_avg_loss(losses_reduced_hybrid)
        
        rtol = 1e-3  # 相对误差 0.1%
    
        # 相对误差: |a - b| / |b| < rtol
        rel_err_packing = abs(avg_packing - avg_baseline) / abs(avg_baseline) if avg_baseline != 0 else 0
        rel_err_hybrid = abs(avg_hybrid - avg_baseline) / abs(avg_baseline) if avg_baseline != 0 else 0

        assert rel_err_packing < rtol, \
            f"packing avg loss {avg_packing:.6f} vs baseline {avg_baseline:.6f}, rel_err={rel_err_packing:.6e}"
        assert rel_err_hybrid < rtol, \
            f"hybrid avg loss {avg_hybrid:.6f} vs baseline {avg_baseline:.6f}, rel_err={rel_err_hybrid:.6e}"
            
    print("test_packing_and_hybrid_cp passed with tp_pp_cp_vpp: ", tp_pp_cp_vpp, "is_moe: ", is_moe)


def dummy_forward_func(
    args, is_sequence_packing, is_hybrid_context_parallel, tp_pp_cp_vpp, is_moe
):
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from pretrain_gpt import forward_step, get_batch

    args.sequence_packing = is_sequence_packing
    args.hybrid_context_parallel = is_hybrid_context_parallel

    args.num_experts = 4 if is_moe else None
    args.moe_ffn_hidden_size = 768 if is_moe else None

    def set_tp_pp_vpp(tp, pp, cp, vpp=None, destroy_first=True):
        if destroy_first:
            Utils.destroy_model_parallel()
        args.tensor_model_parallel_size = tp
        args.pipeline_model_parallel_size = pp
        args.virtual_pipeline_model_parallel_size = vpp
        args.data_parallel_size = 8 // (tp * pp)
        # Hybrid-CP requires context_parallel_size == 1; CP is achieved via DPxCP hybrid groups.
        args.context_parallel_size = 1 if args.hybrid_context_parallel else cp
        if args.hybrid_context_parallel:
            dp_cp_size = args.data_parallel_size * args.context_parallel_size
            if dp_cp_size % 2 != 0:
                pytest.skip(
                    "Hybrid context parallel requires an even dp-cp group size"
                )
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
    if is_sequence_packing:
        args.variable_seq_lengths = True
        # TODO(tailaim): add support for other dispatcher types
        print(
            f"Setting moe_token_dispatcher_type to alltoall for sft sequence packing with pipeline parallelism"
        )
        args.moe_token_dispatcher_type = "alltoall"
        if is_hybrid_context_parallel:
            args.max_seqlen_per_dp_cp_rank = args.seq_length // args.data_parallel_size
            args.sequence_packing_scheduler = "default_hybrid_cp"
        else:
            args.max_seqlen_per_dp_cp_rank = args.seq_length // args.context_parallel_size
            args.sequence_packing_scheduler = "naive_sequence_packing"
    else:
        args.sequence_packing_scheduler = None

    set_global_variables(args)
    # set_args(args)

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
        is_moe=is_moe,
        with_mtp=False,
    )
    model = model if isinstance(model, list) else [model]

    data_iterator = get_data_iterator(args)

    forward_backward_func = get_forward_backward_func()
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
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


class MockVariableLengthSequencePackingDataIterator:
    """
    Mock data iterator for testing get_batch_on_this_rank_for_sequence_packing.

    Generates variable-length (THD format) packed sequences with deterministic
    data for verification across parallel ranks.
    """

    def __init__(
        self,
        total_seq_length: int,
        sequence_lengths: list,
        local_cp_size: int = None,
        device: str = "cuda",
        seed: int = 42,
    ):
        """
        Args:
            total_seq_length: Total length of packed sequences
            sequence_lengths: List of individual sequence lengths (variable-length).
                              If None, generates random variable lengths.
            local_cp_size: Local CP size for hybrid context parallel
            device: Device to create tensors on
            seed: Random seed for reproducibility
        """
        self.total_seq_length = total_seq_length
        self.sequence_lengths = sequence_lengths
        self.local_cp_size = local_cp_size
        self.device = device
        self.seed = seed
        assert (
            sum(self.sequence_lengths) == total_seq_length
        ), f"Sequence lengths sum {sum(self.sequence_lengths)} != total {total_seq_length}"

    def __iter__(self):
        """Interface for the data iterator."""
        return self

    def __next__(self):
        """Generate a mock batch with variable-length THD format."""
        dev = self.device
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        tokens = torch.randint(0, 16384, (self.total_seq_length,), dtype=torch.int64, device=dev)

        # Create position_ids that reset for each sequence (THD format)
        position_ids = []
        for seq_len in self.sequence_lengths:
            position_ids.extend(range(seq_len))
        position_ids = torch.tensor(position_ids, dtype=torch.int64, device=dev)

        # Labels are tokens shifted by 1 for easy verification
        labels = tokens + 1

        # Loss mask: 1.0 for all positions except padding (none here)
        loss_mask = torch.ones(self.total_seq_length, dtype=torch.float32, device=dev)

        # Create cu_seqlens for variable-length packed sequences
        cu_seqlens = [0]
        for seq_len in self.sequence_lengths:
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=dev)
        cu_seqlens_padded = cu_seqlens.clone()

        max_seqlen = torch.tensor([max(self.sequence_lengths)], dtype=torch.int32, device=dev)

        batch = {
            "tokens": tokens,
            "position_ids": position_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_padded": cu_seqlens_padded,
            "max_seqlen": max_seqlen,
        }

        if not (
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ):
            batch["tokens"] = None
            batch["position_ids"] = None
            batch["labels"] = None
            batch["loss_mask"] = None

        if self.local_cp_size is not None:
            batch["local_cp_size"] = torch.tensor(
                [self.local_cp_size], dtype=torch.int32, device=dev
            )

        return batch


def _gather_tensor_from_tp_group(tensor):
    """Gather tensors from all TP ranks for comparison."""
    assert tensor is not None, "Tensor should not be None"
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(tp_size)]
    torch.distributed.all_gather(
        gathered, tensor, group=parallel_state.get_tensor_model_parallel_group()
    )
    return gathered


def _gather_tensor_from_all_ranks(tensor):
    """Gather tensors from all PP ranks for comparison."""
    assert tensor is not None, "Tensor should not be None"
    if type(tensor) is int:
        tensor = torch.tensor(tensor, dtype=torch.int32, device=torch.cuda.current_device())
    gathered = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, tensor)
    return gathered


@pytest.mark.parametrize(
    ("tp", "pp", "cp", "hybrid_cp"),
    [
        (1, 1, 1, False),  # Basic case: no parallelism
        (2, 1, 1, False),  # Tensor parallel only
        (1, 2, 1, False),  # Pipeline parallel only
        (2, 2, 1, False),  # TP + PP
        (1, 1, 2, False),  # CP only
        (2, 1, 2, False),  # TP + CP
        (1, 2, 2, False),  # PP + CP
        (1, 4, 1, False),  # Has middle pp stage
        (1, 1, 1, True),  # Hybrid CP enabled (CP=1 with hybrid groups)
        (2, 1, 1, True),  # TP + Hybrid CP
    ],
)
def test_get_batch_on_this_rank_for_sequence_packing(tp, pp, cp, hybrid_cp):
    """
    Test get_batch_on_this_rank_for_sequence_packing function with variable-length THD format.

    This test verifies:
    1. TP ranks: All ranks within a TP group receive identical data after broadcast
    2. PP ranks: Middle PP ranks have the same packed_seq_params as first/last stages
    3. CP ranks: Data is correctly partitioned with proper shape and values
    4. Variable-length (THD) format: Different sequence lengths are handled correctly
    """
    args = SimpleNamespace()
    args.tensor_model_parallel_size = tp
    args.pipeline_model_parallel_size = pp
    args.context_parallel_size = cp
    args.hybrid_context_parallel = hybrid_cp
    args.virtual_pipeline_model_parallel_size = None
    args.data_parallel_size = 8 // (tp * pp * cp)
    args.seq_length = 8192

    # Skip invalid configurations
    if args.data_parallel_size < 1:
        raise ValueError(f"Invalid config: tp={tp}, pp={pp}, cp={cp} exceeds world size 8")

    # Initialize model parallel
    Utils.initialize_model_parallel(
        tp,
        pp,
        None,
        context_parallel_size=cp,
        hybrid_context_parallel=hybrid_cp,
        min_hybrid_context_parallel_size=1,
    )

    try:
        # Create mock data iterator with variable-length sequences
        # Only TP rank 0 needs the iterator; other TP ranks pass None
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        local_cp_size = 8 // (tp * pp) if hybrid_cp else None

        if tp_rank == 0:
            # Use deterministic seed based on DP rank so same data within TP/PP/CP group
            dp_rank = parallel_state.get_data_parallel_rank()
            sequence_lengths = [1024, 2048, 512, 1536, 3072]
            assert (
                sum(sequence_lengths) == args.seq_length
            ), f"Sequence lengths sum {sum(sequence_lengths)} != total {args.seq_length}"
            data_iterator = iter(
                MockVariableLengthSequencePackingDataIterator(
                    total_seq_length=args.seq_length,
                    sequence_lengths=sequence_lengths,  # Variable lengths, sum=8192
                    local_cp_size=local_cp_size,
                    seed=42 + dp_rank,  # Same seed within PP/CP group
                )
            )
        else:
            # Non-TP-rank-0 ranks don't need the iterator
            data_iterator = None

        # Call the function under test
        result = get_batch_on_this_rank_for_sequence_packing(
            data_iterator=data_iterator,
            mtp_on_this_rank=False,
            vp_stage=None,
            hybrid_context_parallel=hybrid_cp,
        )

        # Unpack the result
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = result

        # Get parallel state info
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        cp_rank = parallel_state.get_context_parallel_rank()
        is_first_stage = parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        is_last_stage = parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        is_first_or_last = is_first_stage or is_last_stage

        # =====================================================================
        # TEST 1: Verify data based on pipeline stage
        # =====================================================================
        if is_first_stage:
            assert tokens is not None, "First stage should have tokens"
            assert position_ids is not None, "First stage should have position_ids"
            assert tokens.dim() == 2, "Tokens should be 2D (batch, seq)"
            assert position_ids.dim() == 2, "Position IDs should be 2D (batch, seq)"
            assert tokens.size(0) == 1, "batch should be 1 in THD format"
            assert position_ids.size(0) == 1, "batch should be 1 in THD format"
        else:
            assert tokens is None, "Non-first stage should not have tokens"
            assert position_ids is None, "Non-first stage should not have position_ids"

        if is_last_stage:
            assert labels is not None, "Last stage should have labels"
            assert loss_mask is not None, "Last stage should have loss_mask"
            assert labels.dim() == 2, "Labels should be 2D (batch, seq)"
            assert loss_mask.dim() == 2, "Loss mask should be 2D (batch, seq)"
            assert labels.size(0) == 1, "batch should be 1 in THD format"
            assert loss_mask.size(0) == 1, "batch should be 1 in THD format"
        else:
            assert labels is None, "Non-last stage should not have labels"
            assert loss_mask is None, "Non-last stage should not have loss_mask"

        # =====================================================================
        # TEST 2: Verify all ranks have consistent packed_seq_params
        # =====================================================================
        assert packed_seq_params is not None
        assert packed_seq_params.qkv_format == "thd"
        if hybrid_cp:
            assert packed_seq_params.local_cp_size is not None
            assert packed_seq_params.cp_group is not None

        test_keys = [
            "cu_seqlens_q",
            "cu_seqlens_q_padded",
            "max_seqlen_q",
            "cu_seqlens_kv",
            "cu_seqlens_kv_padded",
            "max_seqlen_kv",
        ]
        if hybrid_cp:
            test_keys.append("local_cp_size")
        for key in test_keys:
            tensor = getattr(packed_seq_params, key)
            assert tensor is not None
            gathered_tensor = _gather_tensor_from_all_ranks(tensor)
            for i in range(1, len(gathered_tensor)):
                assert torch.equal(
                    gathered_tensor[0], gathered_tensor[i]
                ), f"Rank 0 and rank {i} have different {key}"

        # =====================================================================
        # TEST 3: Verify TP ranks receive identical data after broadcast
        # =====================================================================
        if tp > 1:
            test_tensors = []
            if is_first_stage:
                test_tensors.extend([tokens, position_ids])
            if is_last_stage:
                test_tensors.extend([labels, loss_mask])

            for tensor in test_tensors:
                gathered_tensors = _gather_tensor_from_tp_group(tensor)
                for i in range(1, tp):
                    assert torch.equal(
                        gathered_tensors[0], gathered_tensors[i]
                    ), f"TP rank 0 and rank {i} have different data"

        # =====================================================================
        # TEST 4: Verify CP partitioning
        # =====================================================================
        if cp > 1 or hybrid_cp:
            if hybrid_cp:
                assert packed_seq_params.local_cp_size is not None
                cp_size = packed_seq_params.local_cp_size
                assert packed_seq_params.cp_group == (
                    parallel_state.get_hybrid_data_context_parallel_groups(group_size=cp_size)
                )
            else:
                cp_size = cp

            # With CP, the sequence should be partitioned
            expected_seq_len = args.seq_length // cp_size

            if is_first_stage:
                actual_seq_len = tokens.shape[1]
                assert (
                    actual_seq_len == expected_seq_len
                ), f"CP partitioned tokens have wrong shape: {actual_seq_len} != {expected_seq_len}"

            # Verify labels only if all CP ranks are at last stage
            if is_last_stage:
                actual_seq_len = labels.shape[1]
                assert (
                    actual_seq_len == expected_seq_len
                ), f"CP partitioned labels have wrong shape: {actual_seq_len} != {expected_seq_len}"

    finally:
        Utils.destroy_model_parallel()
        unset_global_variables()


@pytest.mark.parametrize(
    ("tp", "pp", "cp", "vpp","scheduler_type"),
    [
        (1, 1, 1, None, PackingScheduler.DEFAULT_HYBRID_CP),
        (1, 1, 8, None, PackingScheduler.NAIVE_SEQUENCE_PACKING),
        (2, 1, 1, None, PackingScheduler.DEFAULT_HYBRID_CP),
        (2, 1, 4, None, PackingScheduler.NAIVE_SEQUENCE_PACKING),
        (2, 4, 1, None, PackingScheduler.NAIVE_SEQUENCE_PACKING),
        (2, 2, 1, None, PackingScheduler.DEFAULT_HYBRID_CP),
        (2, 2, 1, None, PackingScheduler.NAIVE_SEQUENCE_PACKING),
        (1, 4, 1, 4, PackingScheduler.DEFAULT_HYBRID_CP),
        (1, 4, 1, 4, PackingScheduler.NAIVE_SEQUENCE_PACKING),
    ],
)
def test_wrap_dataloader(tp, pp, cp, vpp, scheduler_type):
    '''
    Test wrap_dataloader function with different scheduler types.
    '''
    args = SimpleNamespace()
    args.tensor_model_parallel_size = tp
    args.pipeline_model_parallel_size = pp
    args.context_parallel_size = cp
    args.virtual_pipeline_model_parallel_size = None
    args.data_parallel_size = 8 // (tp * pp * cp)
    args.seq_length = 8192
    args.max_seqlen_per_dp_cp_rank = 8192
    if scheduler_type is PackingScheduler.DEFAULT_HYBRID_CP:
        args.hybrid_context_parallel = True
    elif scheduler_type is PackingScheduler.NAIVE_SEQUENCE_PACKING:
        args.hybrid_context_parallel = False

    # Skip invalid configurations
    if args.data_parallel_size < 1:
        raise ValueError(f"Invalid config: tp={tp}, pp={pp}, cp={cp} exceeds world size 8")

    def _create_single_sample(seq_len):
        # hard code the padding size to 16
        pad_size = 16
        seq_len_padded = ((seq_len + pad_size - 1) // pad_size) * pad_size
        device = torch.device("cuda", torch.cuda.current_device())
        tokens = torch.randint(0, 16384, (seq_len_padded,), dtype=torch.int64, device=device)
        labels = tokens + 1
        position_ids = torch.arange(seq_len_padded, dtype=torch.int64, device=device)
        loss_mask = torch.ones(seq_len_padded, dtype=torch.float32, device=device)
        loss_mask[0:seq_len] = 1
        loss_mask[seq_len:] = 0
        original_seq_len = torch.tensor(seq_len, dtype=torch.int32, device=tokens.device)
        padded_seq_len = torch.tensor(seq_len_padded, dtype=torch.int32, device=tokens.device)
        return {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            "original_seq_len": original_seq_len,
            "padded_seq_len": padded_seq_len,
        }

    # Initialize model parallel
    Utils.initialize_model_parallel(
        tp,
        pp,
        vpp,
        context_parallel_size=cp,
        hybrid_context_parallel=args.hybrid_context_parallel,
        min_hybrid_context_parallel_size=1,
    )
    
    global_batch_size = 64
    micro_batch_size = 1
    import random
    nums = [random.randint(2048, args.seq_length) for _ in range(global_batch_size)] # 64 sequences 
    
    config = SimpleNamespace()
    config.max_seqlen_per_dp_cp_rank = args.max_seqlen_per_dp_cp_rank
    config.microbatch_group_size_per_vp_stage = pp
    config.hybrid_context_parallel = args.hybrid_context_parallel
    config.virtual_pipeline_model_parallel_size = vpp
    
    
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    
    is_pp_first = (pp_rank == 0)
    is_pp_last = (pp_rank == pp - 1)
    is_tp_first = (tp_rank == 0)
    
    if is_tp_first and (is_pp_first or is_pp_last):
        num_per_dp = global_batch_size // dp_size // micro_batch_size
        samples = [_create_single_sample(num) for num in nums[dp_rank*num_per_dp:(dp_rank+1)*num_per_dp]]
        data_iterator = RerunDataIterator(iter([samples]))
    else:
        data_iterator = None
    
    if is_tp_first:
        if vpp is not None and vpp > 1:
            if is_pp_first:
                data_iterator = [data_iterator] + [None for _ in range(vpp - 1)]
            elif is_pp_last:
                data_iterator = [None for _ in range(vpp - 1)] + [data_iterator]
            else:
                data_iterator = [None for _ in range(vpp)]
    try:
        # Call the function under test
        (new_data_iterator, num_micro_batches, num_total_tokens_this_global_batch, sequence_square_sum_this_global_batch) = wrap_dataloader(data_iterator, config, scheduler_type)
        
        # check the result
        assert type(num_micro_batches) is int
        assert type(num_total_tokens_this_global_batch) is float or type(num_total_tokens_this_global_batch) is numpy.float32
        assert type(sequence_square_sum_this_global_batch) is float or type(sequence_square_sum_this_global_batch) is numpy.float32
        
        def _check_batch(batch_all, batch_keys):
            for batch in batch_all:
                assert set(batch.keys()) == set(batch_keys), f"batch keys: {set(batch.keys())} != {set(batch_keys)}"
                for key in batch_keys:
                    assert batch[key] is not None
                    
        # verify the result
        if is_tp_first:
            batch_keys = ["cu_seqlens","max_seqlen","cu_seqlens_padded"]
            if scheduler_type is PackingScheduler.DEFAULT_HYBRID_CP:
                batch_keys.append("local_cp_size")
            if vpp is not None and vpp > 1:
                if is_pp_first:
                    for data_iterator in new_data_iterator[1:]:
                        batch_all = [next(data_iterator) for _ in range(num_micro_batches)]
                        _check_batch(batch_all, batch_keys)
                    new_data_iterator = new_data_iterator[0]
                elif is_pp_last:
                    for data_iterator in new_data_iterator[:-1]:
                        batch_all = [next(data_iterator) for _ in range(num_micro_batches)]
                        _check_batch(batch_all, batch_keys)
                    new_data_iterator = new_data_iterator[-1]
                else:
                    for data_iterator in new_data_iterator:
                        batch_all = [next(data_iterator) for _ in range(num_micro_batches)]
                        _check_batch(batch_all, batch_keys)
                    new_data_iterator = new_data_iterator[0]
            
            batch_all = [next(new_data_iterator) for _ in range(num_micro_batches)]
            if is_pp_first or is_pp_last:
                batch_keys += ["tokens", "position_ids", "labels", "loss_mask"]
                
            _check_batch(batch_all, batch_keys)
        else:
            if vpp is not None and vpp > 1:
                assert type(new_data_iterator) is list and len(new_data_iterator) == vpp
                for data_iterator in new_data_iterator:
                    assert data_iterator is None
            else:
                assert new_data_iterator is None
        
    finally:
        if torch.distributed.get_rank() == 0:
            print(f"rank:0, exit test_wrap_dataloader successfully with tp:{tp}, pp:{pp}, cp:{cp}, vpp:{vpp}, scheduler_type:{scheduler_type}",flush=True)
        Utils.destroy_model_parallel()
        unset_global_variables()