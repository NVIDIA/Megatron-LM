# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2024, Zhipu AI CORPORATION.  All rights reserved.
"""Pretrain LLaVA."""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
from functools import partial
from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.models.multimodal import GPTVisionModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank_auto,
    average_losses_across_data_parallel_group
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)

stimer = StragglerDetector()

from megatron.core.models.vision.eva_clip_model import Eva2ClipModel
from megatron.core.transformer.transformer_config import VisionTransformerConfig
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec_for_eva_clip,
)
from megatron.training.checkpointing import load_checkpoint

class MegatronVisionModel(torch.nn.Module):
    def __init__(self, pre_process):
        super().__init__()
        args = get_args()
        self.image_seq_length = args.image_seq_length
        eva_args = torch.load(os.path.join(args.vit_load, "iter_0000001/mp_rank_00/model_optim_rng.pt"), map_location="cpu")["args"]
        eva_args.independent_parallel = True
        assert args.tensor_model_parallel_size == eva_args.tensor_model_parallel_size
        print('building EVA model ...')
        config = core_transformer_config_from_args(eva_args, VisionTransformerConfig)
        assert config.independent_parallel
        transformer_layer_spec = get_vit_layer_with_transformer_engine_spec_for_eva_clip()
        self.vit = Eva2ClipModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=1,
            pre_process=pre_process,
        )
        eva_args.load = args.vit_load
        load_checkpoint([self.vit], None, None, args=eva_args)
        self.linear_proj = torch.nn.Linear(eva_args.hidden_size, args.hidden_size)
    
    def forward(self, **kw_args):
        kw_args.pop('indices', None)
        kw_args.pop('src_indices', None)
        kw_args.pop('tgt_indices', None)
        kw_args.pop('pre_len', None)
        external_inputs = {"images": kw_args.pop('images')}
        if 'attention_mask' not in kw_args:
            kw_args['attention_mask'] = None
        vit_output = self.vit(**kw_args, external_inputs=external_inputs)
        if mpu.get_context_parallel_world_size() != 1:
            cp_size = mpu.get_context_parallel_world_size()
            cp_rank = mpu.get_context_parallel_rank()
            calibration_index = torch.arange(self.image_seq_length, device='cuda').view(2 * cp_size, self.image_seq_length // (2 * cp_size))[[cp_rank, (2 * cp_size - cp_rank - 1)]].view(-1)
            ci_list = [torch.empty_like(calibration_index) for _ in range(cp_size)]
            torch.distributed.all_gather(ci_list, calibration_index, group=mpu.get_context_parallel_group())
            calibration_index = torch.cat(ci_list)
            vo_list = [torch.empty_like(vit_output) for _ in range(cp_size)]
            torch.distributed.all_gather(vo_list, vit_output, group=mpu.get_context_parallel_group())
            vit_output_all = torch.cat(vo_list)
            vit_output = torch.empty_like(vit_output_all)
            vit_output[calibration_index] = vit_output_all
        return self.linear_proj(vit_output.transpose(0, 1))

def eva_model_provider(config):
    model = MegatronVisionModel(True)
    return model

def model_provider(pre_process=True, post_process=True):
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        The returned model
    """
    args = get_args()

    config = core_transformer_config_from_args(get_args())
    assert config.first_pipeline_num_layers
    print("building megatron core visual language model!!!!!!!!!!!!!!")
    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
    model = GPTVisionModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        external_feature_model_provider=eva_model_provider,
        allow_missing_keys=['external_feature_model']
    )

    # total_trainable = 0
    # enable = ['linear_proj'] # You can make partial parameters trainable.
    # for n, p in model.named_parameters():
    #     flag = False
    #     for e in enable:
    #         if e.lower() in n.lower():
    #             flag = True
    #             break
    #     if not flag:
    #         p.requires_grad_(False)
    #     else:
    #         print_rank_0(n)
    #         total_trainable += p.numel()
    # print_rank_0("***** Total trainable parameters: "+str(total_trainable)+" *****")
    return model


def get_batch(data_iterator):
    """Generate a batch."""
    args = get_args()

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return [None] * 6

    # get batches based on the TP rank you are on
    if mpu.is_pipeline_first_stage():
        keys = ['tokens', 'position_ids', 'indices', 'external_images', 'external_input_ids', 'external_position_ids']
        batch = get_batch_on_this_tp_rank_auto(data_iterator, keys)
    else:
        keys = ['labels', 'loss_mask', 'attention_mask', "indices"]
        batch = get_batch_on_this_tp_rank_auto(data_iterator, keys)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    external_dict = {}
    for k in list(batch.keys()):
        if 'external_' in k or 'indices' in k:
            if 'external_' in k:
                external_dict[k[len('external_'):]] = batch.pop(k)
            else:
                external_dict[k] = batch.pop(k)
    batch['external_dict'] = external_dict

    return batch.values()

def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, external_dict = [None] * 6
        if mpu.is_pipeline_first_stage():
            tokens, position_ids, external_dict = get_batch(data_iterator)
        elif mpu.is_pipeline_last_stage():
            labels, loss_mask, attention_mask, external_dict = get_batch(data_iterator)
        else:
            external_dict = {}
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                            labels=labels, external_inputs=external_dict)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0

def build_dataset(cls, *args, **kwargs):
    if torch.distributed.is_initialized():
        if not is_dataset_built_on_rank():
            return None
        return cls(*args, **kwargs)
    return cls(*args, **kwargs)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    args = get_args()
    from torch.utils.data import default_collate
    def collate_fn(batch):
        keys = list(set().union(*[set(x.keys()) for x in batch]))
        new_batch = [{} for _ in range(len(batch))]
        for k in keys:
            if 'external' in k or k == 'indices':
                for x, y in zip(new_batch, batch):
                    if k in y:
                        x[k] = y.pop(k)
        old_batch = default_collate(batch)
        for k in keys:
            if 'external' in k or k == 'indices':
                cat_dim = 0 if k != 'indices' else 1
                if k == 'indices':
                    cnt = 0
                    for sample in new_batch:
                        if k in sample:
                            sample[k][0] = cnt
                        cnt += 1
                old_batch[k] = torch.cat([x[k] for x in new_batch if k in x], dim=cat_dim)
        return old_batch
    args.collate_fn = collate_fn

    from dataset import llama2_text_processor, llama2_tokenizer, blip2_image_processor_func_megatron, BlipImageEvalProcessor
    blip2_image_processor_megatron_224 = partial(blip2_image_processor_func_megatron, args.image_seq_length, BlipImageEvalProcessor(224))
    tokenizer = llama2_tokenizer('/'.join(args.tokenizer_model.split('/')[:-1]))
    text_processor = llama2_text_processor(tokenizer, image_length=args.image_seq_length, max_target_length=args.seq_length)

    def process_fn(item):
        prompt = "Describe the image."
        caption = item["caption"]
        ret = text_processor(caption, prompt)
        num_image = 1
        if num_image > 0:
            img_indices_b = torch.zeros(num_image, text_processor.image_length, dtype=torch.int64) # This will change in collate_fn
            img_indices_s = torch.arange(text_processor.image_length).unsqueeze(0).repeat(num_image, 1)
            indices = torch.stack([img_indices_b, img_indices_s], dim=0) # 2, num_image, image_length
        return {**ret, **blip2_image_processor_megatron_224(item["image"]), "indices": indices}

    from dataset import ImageJsonDataset
    dataset = build_dataset(ImageJsonDataset, args.train_data_path[0], single_process_fn=process_fn)
    
    print("build dataset over...")
    return dataset, dataset, None

def add_vit_load_args(parser):
    group = parser.add_argument_group(title='vit load')
    group.add_argument("--vit-load", type=str,
                       help='path of vit')
    group.add_argument("--image-seq-length", type=int,
                       help='vit image length')
    return parser

if __name__ == "__main__":
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        # for mpi initialization
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={},
             extra_args_provider=add_vit_load_args)
