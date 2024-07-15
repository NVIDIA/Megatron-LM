# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain or SFT multimodal."""
from copy import deepcopy
from functools import partial
import os
import sys
import warnings

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))

from megatron.training import get_args, get_timers, get_tokenizer, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from config import get_language_model_config, get_vision_model_config, get_vision_projection_config
from megatron.core.models.multimodal.llava_model import LLaVAModel
from layer_specs import get_layer_spec, get_mlp_module_spec, get_layer_spec_te
from megatron.training import pretrain
from megatron.training.utils import average_losses_across_data_parallel_group
from dataloader_provider import train_valid_test_dataloaders_provider


def model_provider(
    pre_process=True, post_process=True, add_encoder=True, add_decoder=True,
    parallel_output=True) -> LLaVAModel:
    """Builds the model.

    Args:
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism). Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True. When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
        parallel_output (bool): Enable parallel model output.

    Returns:
        model: A multimodal model.
    """
    args = get_args()

    use_te = args.use_te

    print_rank_0('building a multimodal model ...')

    num_image_tokens = get_image_token_count()

    old_seq_length = args.seq_length
    args.decoder_seq_length = args.seq_length + num_image_tokens
    args.seq_length = num_image_tokens
    if torch.distributed.get_rank() == 0:
        warnings.warn("Changed decoder_seq_length to num_image_tokens ({num_image_tokens}) + user-specified seq_length ({old_seq_length}).")

    if args.decoder_seq_length > args.max_position_embeddings:
        args.max_position_embeddings = args.decoder_seq_length
        warnings.warn("Expanded max_position_embeddings to {args.max_position_embeddings} to accommodate the full sequence of vit output + llm output.")

    base_config = core_transformer_config_from_args(get_args())
    base_config.language_model_type = args.language_model_type

    language_config = deepcopy(base_config)
    language_config = get_language_model_config(language_config)

    if use_te:
        language_transformer_layer_spec = get_layer_spec_te(is_vit=False)
    else:
        language_transformer_layer_spec = get_layer_spec(is_vit=False)

    vision_config = deepcopy(base_config)
    vision_config = get_vision_model_config(vision_config, apply_query_key_layer_scaling=args.apply_query_key_layer_scaling)
    if args.pipeline_model_parallel_size > 1:
        assert args.encoder_pipeline_model_parallel_size == 1, "ViT can only live on 1 pipeline stage."
        vision_config.pipeline_model_parallel_size = args.encoder_pipeline_model_parallel_size

    if use_te:
        vision_transformer_layer_spec = get_layer_spec_te(is_vit=True)
    else:
        vision_transformer_layer_spec = get_layer_spec(is_vit=True)

    vision_projection_config = deepcopy(base_config)
    vision_projection_config = get_vision_projection_config(vision_projection_config, language_config.hidden_size)
    vision_projection_layer_spec = get_mlp_module_spec(use_te=use_te).submodules

    model = LLaVAModel(
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        drop_vision_class_token=args.disable_vision_class_token,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_layer_spec,
        vision_projection_type="mlp",
        allow_missing_vision_projection_checkpoint=args.allow_missing_vision_projection_checkpoint,
        parallel_output=parallel_output,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        language_rotary_base=args.rotary_base,
        img_embedding_idx=args.img_embedding_idx,
    )

    model.freeze(freeze_language_model=args.freeze_LM, freeze_vision_model=args.freeze_ViT, freeze_vision_projection=False)

    return model


def get_batch(data_iterator):
    """Generate a batch"""

    args = get_args()

    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_text = tensor_parallel.broadcast_data(["text"], data, torch.int64)["text"]
    data_img = tensor_parallel.broadcast_data(["img"], data, torch.float32)
    prompt_len = tensor_parallel.broadcast_data(["prompt_len"], data, torch.int64)["prompt_len"]

    torch.cuda.nvtx.range_pop()

    tokens_ = data_text.long()

    img_raw = data_img['img'].reshape(-1, 3, args.img_h, args.img_w)

    torch.cuda.nvtx.range_push("index tokens")
    tokenizer = get_tokenizer()
    text_length = args.decoder_seq_length - args.seq_length
    tokens = tokens_[:, :text_length].contiguous()
    labels = tokens_[:, 1:text_length+1].contiguous()

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("get_ltor_masks_and_position_ids")
    if hasattr(tokenizer, 'eod'):
        eod_token = tokenizer.eod
    elif hasattr(tokenizer, 'eos_id'):
        eod_token = tokenizer.eos_id
    attention_mask, loss_mask, position_ids = \
        get_ltor_masks_and_position_ids(tokens, eod_token,
                                        args.reset_position_ids,
                                        args.reset_attention_mask,
                                        args.eod_mask_loss,
                                        question_length=prompt_len)
    torch.cuda.nvtx.range_pop()

    loss_mask, labels, attention_mask = _preprocess_data_for_llava(loss_mask, labels, attention_mask)

    tokens = tokens[:, 1:]  # drop image index token

    return tokens, labels, loss_mask, attention_mask, position_ids, img_raw


def get_image_token_count():
    args = get_args()

    add_class_token = not args.disable_vision_class_token

    num_patches_per_dim_h = args.img_h // args.patch_dim
    num_patches_per_dim_w = args.img_w // args.patch_dim
    num_patches = num_patches_per_dim_h * num_patches_per_dim_w
    num_image_tokens = num_patches + (1 if add_class_token else 0)

    return num_image_tokens


def _preprocess_data_for_llava(loss_mask, labels, attention_mask):
    """Preprocess data sample to the format expected by a LLaVA model."""
    num_image_tokens = get_image_token_count()

    batch_size = loss_mask.shape[0]

    loss_mask2 = torch.cat(
        [torch.zeros(batch_size, num_image_tokens - 1, dtype=torch.float32, device=loss_mask.device), loss_mask], dim=1
    )
    labels2 = torch.cat([torch.zeros(batch_size, num_image_tokens - 1, dtype=torch.int64, device=labels.device), labels], dim=1)

    full_seq_length = len(labels2[0])
    attention_mask2 = torch.tril(torch.ones((1, 1, full_seq_length, full_seq_length), device=attention_mask.device))
    attention_mask2 = attention_mask2 < 0.5

    return loss_mask2, labels2, attention_mask2


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    question_length=None,
                                    weights=None):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()


    if question_length is not None:
        for b in range(micro_batch_size):
            loss_mask[b, :max(0, question_length[b].item())] = 0.0

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)
    if weights is not None:
        loss_mask = loss_mask * weights

    return attention_mask, loss_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    if loss_mask is not None:
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / max( 1,loss_mask.sum() )
    else:
        loss = torch.mean(losses)

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}



def forward_step(data_iterator, model: LLaVAModel):
    """Forward training step.

    Args:
        data_iterator (torch.utils.data.dataloader): Input data iterator
        model: Multimodal model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, images = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(images, tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)

def add_multimodal_extra_args(parser):
    """Extra arguments."""
    group = parser.add_argument_group(title='multimodal arguments')
    group.add_argument('--valid-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--dataset-config', type=str, default=None)
    group.add_argument("--prompt-path", type=str, default=None)
    group.add_argument('--freeze-LM', action='store_true', default=False)
    group.add_argument('--freeze-ViT', action='store_true', default=False)
    group.add_argument('--language-model-type', type=str, required=True)
    group.add_argument("--disable-vision-class-token", action="store_true", default=False)
    group.add_argument("--allow-missing-vision-projection-checkpoint", action="store_true", default=False)
    group.add_argument("--use-te", action="store_true", default=False)
    group.add_argument("--img-embedding-idx", type=int, default=0,
                       help='Llava specific parameter. Defines at which index'
                       'in the language_embedding tensor the image_embeddings'
                       'should be inserted')
    return parser


def llava_embedding_ranks(pp_ranks):
    """LLava's embedding ranks consist of the decoder's first and last ranks (ie, the ViT has no embeddings).
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1 or pp_ranks[epp] == last_rank:
        return [last_rank]
    else:
        return [pp_ranks[epp], last_rank]


def llava_position_embedding_ranks(pp_ranks):
    """LLava's embedding ranks consist of the singular rank of the model or the decoder's first rank.
    Args:
        pp_ranks: A list of global ranks that constitute a pipeline group.
    """
    args = get_args()

    # encoder size is also the index to the first rank of the decoder.
    epp = args.encoder_pipeline_model_parallel_size

    last_rank = pp_ranks[-1]
    if len(pp_ranks) == 1:
        return [last_rank]
    else:
        return [pp_ranks[epp]]


if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(
        train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_multimodal_extra_args,
        get_embedding_ranks=llava_embedding_ranks,
        get_position_embedding_ranks=llava_position_embedding_ranks,
    )
