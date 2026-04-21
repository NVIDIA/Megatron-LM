"""Pretrain or SFT multimodal."""
import math
import os
import sys
from functools import partial

import torch
import yaml

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

from dataloader_provider import train_valid_test_dataloaders_provider, is_first_or_last_stage
from model import model_provider
from multimodal_args import add_multimodal_extra_args

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.multimodal import context_parallel
from megatron.core.models.multimodal.llava_model import IGNORE_INDEX, LLaVAModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    is_pipeline_last_stage,
)
from megatron.training import get_args, get_timers, get_tokenizer, pretrain
from megatron.training.utils import is_last_rank, get_batch_on_this_cp_rank


def get_batch(data_iterator, image_token_index, img_seq_len):
    """Generate a batch

    Note: attn_mask_type in layer_specs.py sets the attention mask. Attention mask is None here.
    """
    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None
    num_tiles = None
    num_frames = None
    packed_seq_params = None
    imgs_sizes = None
    vision_cu_lengths = None
    vision_max_lengths = None
    vision_packed_seq_params = None
    has_pad_img = None
    samples_seen = None
    sound_clips = None
    sound_timestamps = None
    num_sound_clips = None
    sound_length = None

    args = get_args()

    # Dataloader doesn't run on the middle stages in a pipeline parallel model.
    pp_size = get_pipeline_model_parallel_world_size()
    if not is_first_or_last_stage(pp_size, args.encoder_pipeline_model_parallel_size):
        # Note these are all set to None above.
        return (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            imgs,
            num_tiles,
            num_frames,
            packed_seq_params,
            imgs_sizes,
            vision_packed_seq_params,
            has_pad_img,
            sound_clips,
            sound_length,
            sound_timestamps,
            num_sound_clips,
            samples_seen,
        )

    # Broadcast data.
    torch.cuda.nvtx.range_push("get_data")
    if data_iterator is not None and get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
    else:
        data = None

    data_text = tensor_parallel.broadcast_data(["tokens"], data, torch.int64, optimize=args.optimize_broadcast)["tokens"]
    labels = tensor_parallel.broadcast_data(["labels"], data, torch.int64, optimize=args.optimize_broadcast)["labels"]

    imgs = tensor_parallel.broadcast_data(["imgs"], data, torch.float32, optimize=args.optimize_broadcast)["imgs"]

    # Handle datasets that don't provide num_frames (for backward compatibility with image-only datasets)
    if get_tensor_model_parallel_rank() == 0 and data is not None and "num_frames" not in data:
        # For image-only datasets, each tile corresponds to 1 frame
        if "num_tiles" in data:
            data["num_frames"] = torch.ones_like(data["num_tiles"], dtype=torch.int32)
        else:
            data["num_frames"] = torch.tensor([], dtype=torch.int32)

    tiles_and_frames = tensor_parallel.broadcast_data(["num_tiles", "num_frames"], data, torch.int32, optimize=args.optimize_broadcast)
    num_tiles, num_frames = tiles_and_frames["num_tiles"], tiles_and_frames["num_frames"]

    cu_lengths = tensor_parallel.broadcast_data(["cu_lengths"], data, torch.int32, optimize=args.optimize_broadcast)["cu_lengths"]
    cu_lengths_padded = tensor_parallel.broadcast_data(["cu_lengths_padded"], data, torch.int32, optimize=args.optimize_broadcast)["cu_lengths_padded"]
    max_lengths = tensor_parallel.broadcast_data(["max_lengths"], data, torch.int32, optimize=args.optimize_broadcast)["max_lengths"]

    if get_tensor_model_parallel_rank() == 0 and 'samples_seen' not in data:
        data['samples_seen'] = torch.tensor(1, dtype=torch.int32, device=data_text.device)

    samples_seen = tensor_parallel.broadcast_data(["samples_seen"], data, torch.int32, optimize=args.optimize_broadcast)["samples_seen"]

    imgs_sizes = tensor_parallel.broadcast_data(["imgs_sizes"], data, torch.int32, optimize=args.optimize_broadcast)["imgs_sizes"]

    vision_cu_lengths = tensor_parallel.broadcast_data(["vision_cu_lengths"], data, torch.int32, optimize=args.optimize_broadcast)["vision_cu_lengths"]
    vision_max_lengths = tensor_parallel.broadcast_data(["vision_max_lengths"], data, torch.int32, optimize=args.optimize_broadcast)["vision_max_lengths"]
    has_pad_img = tensor_parallel.broadcast_data(["has_pad_img"], data, torch.bool, optimize=args.optimize_broadcast)["has_pad_img"]

    sound1 = tensor_parallel.broadcast_data(["sound_clips", "sound_timestamps"], data, torch.float32)
    sound_clips, sound_timestamps = sound1["sound_clips"], sound1["sound_timestamps"]
    sound2 = tensor_parallel.broadcast_data(["num_sound_clips", "sound_length"], data, torch.int64)
    num_sound_clips, sound_length = sound2["num_sound_clips"], sound2["sound_length"]

    # No image input (text-only sample) if the dataloader returned a size 1 image.
    if imgs.shape == torch.Size([1, 1]):
        # FSDP can hang with text-only samples. A workaround is to run a valid dummy image through the vision
        # model and then add image embeddings with a zero multiplier.
        if args.use_torch_fsdp2:
            imgs = torch.zeros((1, 3, args.img_h, args.img_w), dtype=torch.float32, device=data_text.device)
        else:
            # Similar workaround is not needed without FSDP and we can use an empty image.
            # FIXME: text-only data can cause still cause a hang in the special case where
            # the vision model is own its own pipeline rank and --freeze-ViT is enabled.
            imgs = torch.tensor([], dtype=torch.float32, device=data_text.device)
        num_tiles = torch.tensor([], dtype=torch.int, device=data_text.device)
        num_frames = torch.tensor([], dtype=torch.int, device=data_text.device)

    # TODO: Sound encoder from HF/Nemo can hang with text-only samples. Find a better way to handle this.
    is_sound_frozen = args.freeze_sound_model and args.freeze_sound_projection
    if getattr(args, "sound_model_type", None) and sound_clips is not None and sound_clips.shape == torch.Size([1, 1]) and not is_sound_frozen:
        sound_clips = torch.zeros((1, 1600), dtype=sound_clips.dtype, device=sound_clips.device)
        sound_length = torch.tensor([1600], dtype=sound_length.dtype, device=sound_length.device)
        sound_timestamps = torch.tensor([], dtype=sound_timestamps.dtype, device=sound_timestamps.device)

    # Last pipeline parallel stage doesn't need images.
    if pp_size > 1 and is_pipeline_last_stage():
        imgs = None

    # If cu_lengths and max_lengths are non-dummy, construct PackedSeqParams. Otherwise, leave it at None.
    if cu_lengths.shape != torch.Size([1, 1]):
        assert (
            cu_lengths.shape[0] == max_lengths.shape[0] == 1
        ), "micro-batch-size must be 1 for packing"
        cu_lengths = cu_lengths[0]
        cu_lengths_padded = cu_lengths_padded[0]
        max_lengths = max_lengths[0]

        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_lengths,
            cu_seqlens_kv=cu_lengths,
            cu_seqlens_q_padded=cu_lengths_padded,
            cu_seqlens_kv_padded=cu_lengths_padded,
            max_seqlen_q=max_lengths,
            max_seqlen_kv=max_lengths,
        )

    # If cu_lengths and max_lengths are non-dummy, construct PackedSeqParams. Otherwise, leave it at None.
    vision_packed_seq_params = None
    if vision_cu_lengths.shape != torch.Size([1, 1]):
        assert (
            vision_cu_lengths.shape[0] == vision_max_lengths.shape[0] == 1
        ), "micro-batch-size must be 1 for packing"
        vision_cu_lengths = vision_cu_lengths[0]
        vision_max_lengths = vision_max_lengths[0]

        vision_packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=vision_cu_lengths,
            cu_seqlens_kv=vision_cu_lengths,
            max_seqlen_q=vision_max_lengths,
            max_seqlen_kv=vision_max_lengths,
        )

    torch.cuda.nvtx.range_pop()

    tokens_ = data_text.long()

    torch.cuda.nvtx.range_push("index tokens")
    tokenizer = get_tokenizer()
    text_length = tokens_.shape[1]
    tokens = tokens_[:, :text_length].contiguous()
    labels = labels[:, 1 : text_length + 1].contiguous()

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("get_ltor_masks_and_position_ids")
    loss_mask, position_ids = get_ltor_masks_and_position_ids(tokens, labels, tokenizer.pad)
    torch.cuda.nvtx.range_pop()

    return (
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        imgs,
        num_tiles,
        num_frames,
        packed_seq_params,
        imgs_sizes,
        vision_packed_seq_params,
        has_pad_img,
        sound_clips,
        sound_length,
        sound_timestamps,
        num_sound_clips,
        samples_seen,
    )


def get_ltor_masks_and_position_ids(input_ids, target, pad_token):
    """Build masks and position id for left to right model."""
    seq_length = input_ids.shape[1]

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    # Loss mask.
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[target == pad_token] = 0.0  # mask paddings
    loss_mask[target == IGNORE_INDEX] = 0.0  # mask prompts

    return loss_mask, position_ids


def get_mask_start_and_end_idx(arr):
    """
    Returns a list of tuples holding the start and end index in arr of the non-zeros contiguuous
    sub arrays.

    For instance, if arr = [0, 1, 0, 0, 1, 1]
    get_mask_start_and_end_idx(arr) = [(1, 1), (4, 5)]
    such that arr[1:1+1] = [1] and arr[4:5+1] = [1, 1]
    """
    mask = (arr != 0)

    mask_int = mask.int()

    diff = mask_int[1:] - mask_int[:-1]
    start_indices = (diff == 1).nonzero(as_tuple=False).flatten() + 1
    end_indices = (diff == -1).nonzero(as_tuple=False).flatten()
    if len(mask)==0: return []
    if mask[0]:
        start_indices = torch.cat((torch.tensor([0], device=arr.device), start_indices))
    if mask[-1]:
        end_indices = torch.cat((end_indices, torch.tensor([len(arr) - 1], device=arr.device)))
    sequences = list(zip(start_indices.tolist(), end_indices.tolist()))
    return sequences


def scaled_loss_func(loss_mask, output_tensor, samples_seen):
    """
    Scaled loss function

    Scale the loss for each conversation turn using the formula:

    1 / sum_j[ sqrt(length(loss_turn_j)) ] * sum_i[ sum(loss_turn_i) / sqrt(length(loss_turn_i)) ]

    Where we use the loss mask to infer the start / end of the conversation turns.
    """
    args = get_args()
    assert args.context_parallel_size == 1, "this loss func is incorrect for context parallel"
    losses = output_tensor.float()

    loss_list = []
    num_valid_labels_list = []
    for idx in range(losses.shape[0]):
        loss_this_sample = losses[idx]
        turn_start_end_list = get_mask_start_and_end_idx(loss_mask[idx])
        for turn_start, turn_end in turn_start_end_list:
            # compute loss for each turn
            loss_this_turn = loss_this_sample[turn_start:turn_end+1].sum()
            assert (1 - loss_mask)[idx][turn_start:turn_end+1].sum() < 1.0
            num_valid_labels_this_turn = turn_end - turn_start + 1
            loss_this_turn = loss_this_turn / num_valid_labels_this_turn
            loss_list.append(loss_this_turn)
            # append num of valid labels for each turn
            num_valid_labels_list.append(num_valid_labels_this_turn)
    base_num = sum([math.sqrt(each) for each in num_valid_labels_list])
    for idx in range(len(loss_list)):
        # normalize loss for each turn
        loss_list[idx] = loss_list[idx] * math.sqrt(num_valid_labels_list[idx]) / base_num

    if len(loss_list) > 0:
        total_loss = torch.stack(loss_list).sum()
        total_tokens = torch.ones_like(total_loss)
    else:
        raise RuntimeError("loss_list for loss scaling per conversation unexpectedly got empty list")

    num_tokens = total_tokens.clone().detach().to(torch.int)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), num_tokens.view(1)])

    return (
        total_loss,
        num_tokens,
        {
            'lm loss': reporting_loss,
            '_samples_seen': samples_seen.detach(),
        }
    )


def loss_func(loss_mask, output_tensor, samples_seen):
    args = get_args()

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.contiguous().view(-1).float()
    loss = torch.sum(losses * loss_mask)

    if args.no_calculate_per_token_loss:
        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        num_tokens = torch.clamp(num_tokens, min=1)
    elif args.use_loss_scaling and args.context_parallel_size > 1:
        # num_tokens are all-reduced from all CP ranks and loss will be divided by the total num_tokens = args.context_parallel_size.
        # So we need to multiply loss by args.context_parallel_size to get the correct loss.
        num_tokens = torch.tensor(1, dtype=torch.int, device=losses.device)
        loss *= args.context_parallel_size
    else:
        num_tokens = loss_mask.sum().clone().detach().to(torch.int)

    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (
        loss,
        num_tokens,
        {
            'lm loss': reporting_loss,
            '_samples_seen': samples_seen.detach(),
        },
    )

def forward_step(data_iterator, model: LLaVAModel):
    """Forward training step.

    Args:
        data_iterator (torch.utils.data.dataloader): Input data iterator
        model: Multimodal model

    Returns:
        output_tensor (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape [b, s, vocab_size].
        loss_func (callable): Loss function with a loss mask specified.
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    (
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        images,
        num_image_tiles,
        num_frames,
        packed_seq_params,
        imgs_sizes,
        vision_packed_seq_params,
        has_pad_img,
        sound_clips,
        sound_length,
        sound_timestamps,
        num_sound_clips,
        samples_seen,
    ) = get_batch(data_iterator, model.module.module.image_token_index, model.module.module.img_seq_len)
    timers('batch-generator').stop()

    output_tensor, loss_mask = model(
        images,
        tokens,
        position_ids,
        attention_mask,
        labels,
        loss_mask,
        num_image_tiles=num_image_tiles,
        num_frames=num_frames,
        packed_seq_params=packed_seq_params,
        imgs_sizes=imgs_sizes,
        vision_packed_seq_params=vision_packed_seq_params,
        has_pad_img=has_pad_img,
        sound_clips=sound_clips,
        sound_length=sound_length,
        sound_timestamps=sound_timestamps,
        num_sound_clips=num_sound_clips,
    )
    args = get_args()
    if args.use_loss_scaling and args.context_parallel_size <= 1:
        loss_function = partial(scaled_loss_func, loss_mask, samples_seen=samples_seen)
    else:
        # For context parallel, we use the regular loss func because the scaling factors are already applied to loss_mask.
        # We do this because the CP sharding ignores turn boundaries in the conversation.
        loss_function = partial(loss_func, loss_mask, samples_seen=samples_seen)

    return output_tensor, loss_function


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


def run_online_eval(model):
    """Run an evaluation benchmark during training."""
    args = get_args()

    # Online evaluation config is not defined. Do nothing.
    if not args.online_evaluation_config:
        return []

    from config import EvaluationConfig
    # Import the common evaluation functions
    from run_text_generation import get_evaluation_configs, run_evaluation_loop

    # Use the common config loading function
    configs = get_evaluation_configs(config_path=args.online_evaluation_config)

    # The inference code assumes the first rank is the leader.
    # Tensorboard writer is on the last rank.
    # We must write to a storage space that all ranks see.
    output_dir = os.path.join(args.save, "online_eval")
    os.makedirs(output_dir, exist_ok=True)

    # Use the common evaluation loop
    scores = run_evaluation_loop(model[0].module, configs, output_dir_override=output_dir, print_output=False)

    return [scores]


def write_eval_to_tensorboard(data, iteration, writer, walltime=None):
    """Write evaluation data to Tensorboard."""
    if not writer:
        return

    for item in data:
        for k, v in item.items():
            writer.add_scalar(k, v, iteration, walltime=walltime)


def write_online_eval_to_tensorboard(data, iteration, writer, walltime=None):
    """Write online evaluation data to Tensorboard."""
    import shutil
    args = get_args()

    # Define source and destination directories
    source_dir = os.path.join(args.save, "online_eval")
    destination_dir = os.path.join(args.save, f"online_eval_{iteration}")
    if os.path.exists(source_dir):
        print("Moving online eval data from", source_dir, "to", destination_dir)

        # Move the directory (back up the generation)
        shutil.move(source_dir, destination_dir)

    write_eval_to_tensorboard(data, iteration, writer, walltime)


def post_init_func():
    # Debug only a specific rank (set DEBUG_RANK env var)
    # Assumes the job was launched with torch.distributed.run (which sets LOCAL_RANK)
    debug_rank = os.environ.get('DEBUG_RANK', None)
    if debug_rank is not None:
        local_rank = os.environ.get('LOCAL_RANK', None)
        if local_rank is None:
            raise ValueError("Expected LOCAL_RANK to be set from torch.distributed.run when using DEBUG_RANK")

        if int(local_rank) == int(debug_rank):
            import debugpy
            import socket
            hostname = socket.gethostname()
            debug_port = int(os.environ.get('DEBUG_PORT', 3009))
            debugpy.listen(("0.0.0.0", debug_port))
            print(f"[Rank {local_rank}] Waiting for debugger. Attach to host: {hostname}, port: {debug_port}...")
            debugpy.wait_for_client()
        else:
            print(f"[Rank {local_rank}] Waiting for rank {debug_rank}...")

        torch.distributed.barrier()
        print(f"[Rank {local_rank}] Debugger attached, continuing training...")

if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    try:
        pretrain(
            train_valid_test_dataloaders_provider,
            model_provider,
            ModelType.encoder_and_decoder,
            forward_step,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
            extra_args_provider=add_multimodal_extra_args,
            process_non_loss_data_func=write_online_eval_to_tensorboard,
            get_embedding_ranks=llava_embedding_ranks,
            get_position_embedding_ranks=llava_position_embedding_ranks,
            non_loss_data_func=run_online_eval,
            post_init_func=post_init_func,
        )
    except Exception as e:
        # If using DEBUG_RANK to debug, don't exit on failure (or torchrun will kill all ranks)
        debug_rank = os.environ.get('DEBUG_RANK', None)
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        if debug_rank is not None and local_rank != int(debug_rank):
            import time
            import traceback
            print(f"\n[Rank {local_rank}] Caught exception during debugging (pausing to keep DEBUG_RANK alive):")
            traceback.print_exc()
            while True:
                time.sleep(60)

        raise
