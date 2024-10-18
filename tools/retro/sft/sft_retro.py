# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Pretrain GPT"""

import torch
from functools import partial, reduce
import sys, os

sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), "../../../"))))
from megatron.training import get_args, get_retro_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.training import pretrain
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.training.utils import average_losses_across_data_parallel_group
from pretrain_gpt import model_provider, is_dataset_built_on_rank
from tools.retro.sft.dataset_conv import JsonQADataset, JsonQADatasetConfig, RetroJsonQADataset, RetroJsonQADatasetConfig


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    # parameters for the knowledgeable dialogue generation
    group.add_argument('--task', type=str, default=None,
                       help='Task name.')
    group.add_argument('--epochs', type=int, default=None,
                       help='Number of finetunning epochs. Zero results in '
                            'evaluation only.')
    group.add_argument('--keep-last', action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                            'the data loader')
    group.add_argument('--pretrained-checkpoint', type=str, default=None,
                       help='Pretrained checkpoint used for finetunning.')
    group.add_argument('--data-folder', type=str, default=None,
                       help='dataset folder')
    group.add_argument('--answer-loss-only', action='store_true', default=False,
                       help='take the loss from answer part, ignore the context')
    group.add_argument('--weight', type=float, default=1)
    group.add_argument('--adaptor', action='store_true', default=False)
    group.add_argument('--project-size', type=int, default=256)
    group.add_argument('--cyclic-train-iters', type=int, default=None)
    group.add_argument('--stored_params', type=dict, default=dict())
    group.add_argument('--eval_ppl', action='store_true', default=False)
    group.add_argument('--debug', action='store_true', default=False)
    group.add_argument('--add_retriever', action='store_true', default=False)
    group.add_argument('--return_doc_ids', action='store_true', default=False)
    group.add_argument('--return_neighbor_ids', action='store_true', default=False)
    group.add_argument('--add_offset_doc_ids', action='store_true', default=False)
    group.add_argument('--offset_dict_path', type=str, default='')
    group.add_argument('--neighbors_path', type=str, default='')
    group.add_argument('--valid_neighbors_path', type=str, default='')
    group.add_argument('--database_path', type=str, default='')
    group.add_argument('--valid_database_path', type=str, default='')
    group.add_argument('--encoder-layers', type=int, default=12)
    group.add_argument('--encoder-hidden-dropout', type=float, default=0.1)
    group.add_argument('--encoder-attention-dropout', type=float, default=0.1)
    group.add_argument('--k', type=int, default=2)
    group.add_argument('--r', type=int, default=128)
    group.add_argument('--m', type=int, default=64)
    group.add_argument('--dpr-mode', type=str, default="multi")
    group.add_argument('--faiss-ckpt', type=str, default='')
    group.add_argument('--original-db-file', type=str, default="")
    group.add_argument('--ft_neighbours', type=int, default=1)
    group.add_argument('--reuse-top', action='store_true', default=False)
    group.add_argument('--shuffle_topn', action='store_true', default=False)
    group.add_argument('--chunk0', action='store_true', default=False)
    group.add_argument('--disable-encoder', action='store_true', default=False)
    group.add_argument('--qa-space-pad', action='store_true', default=False)
    group.add_argument('--retro-mask-encoder', action='store_true', default=False)
    group.add_argument('--without-title', action='store_true', default=False)
    group.add_argument('--longform-answer', action='store_true', default=False)
    group.add_argument('--bert-retriever-neighbours', action='store_true', default=False)
    group.add_argument('--prefix', action='store_true', default=False)
    group.add_argument('--question-in-encoder', action='store_true', default=False)
    group.add_argument('--reset_eval', type=bool, default=True)  ## by default reset eval for each eval
    return parser


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text', 'answer_mask']
    datatype = torch.int64

    if args.retro_add_retriever:
        keys += 'neighbor_tokens', 'context_len'

    # Broadcast data.
    if data_iterator is not None:
        try:
            data = next(data_iterator)

        except Exception:
            data = data_iterator
            raise ValueError("error with data_iterator")
    else:
        data = None

    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    chunk_size = torch.min(data_b['context_len'])
    retro_args = get_retro_args()
    # two chunk retro has at least seq_len / 2 of chunk size
    retro_args.retro_gpt_chunk_length = max(args.seq_length // 2, args.seq_length - chunk_size.item())

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    answer_mask = data_b["answer_mask"].float()[:, 1:].contiguous()

    if args.retro_add_retriever:
        neighbor_tokens = data_b['neighbor_tokens'].view(-1,
                                                         retro_args.retro_gpt_retrieved_length).long()  # [bs * l * k, r]

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    if args.answer_loss_only:
        loss_mask = loss_mask * answer_mask

    if args.retro_add_retriever:
        _, _, neighbor_position_ids = get_ltor_masks_and_position_ids(
            neighbor_tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        neighbor_attention_mask = None
        return tokens, labels, loss_mask, attention_mask, position_ids, \
            neighbor_tokens, neighbor_attention_mask, neighbor_position_ids
    else:
        return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    if args.retro_add_retriever:
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids, \
            neighbor_tokens, neighbor_attention_mask, neighbor_position_ids = get_batch(
            data_iterator)
        timers('batch-generator').stop()
        output_tensor = model(tokens, position_ids, attention_mask,
                              retriever_input_ids=neighbor_tokens,
                              retriever_position_ids=neighbor_position_ids,
                              retriever_attn_mask=neighbor_attention_mask,
                              labels=labels)
    else:
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
        timers('batch-generator').stop()
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    retro_args = get_retro_args()

    tokenizer = get_tokenizer()

    def fix_and_split_blend_pair(pair):
        weight, name = pair
        return [
            [weight, os.path.join(args.data_folder, name, f"{name}_QA_train.json")],
            [weight, os.path.join(args.data_folder, name, f"{name}_QA_dev.json")],
            None,
        ]

    blend = [args.data_path[i:i+2] for i in range(0, len(args.data_path), 2)]

    if len(blend) == 1:
        blend_per_split =  [
            os.path.join(args.data_folder, blend[0], f"{blend[0]}_QA_train.json"),
            os.path.join(args.data_folder, blend[0], f"{blend[0]}_QA_dev.json"),
            None,
        ]
    else:
        blend_per_split = [
            list(
                reduce(
                    lambda x, y: x + y,
                    list(zip(*map(fix_and_split_blend_pair, blend)))[0]
                )
            ),
            None,
            None,
        ]

    blend_per_split = [get_blend_from_list(blend) for blend in blend_per_split]

    extra_kwargs = {}

    if args.retro_add_retriever:
        dataset_cls = RetroJsonQADataset
        config_cls = RetroJsonQADatasetConfig
        extra_kwargs["retro_num_neighbors"] = args.retro_num_neighbors
        extra_kwargs["retro_gpt_retrieved_length"] = retro_args.retro_gpt_retrieved_length
    else:
        dataset_cls = JsonQADataset
        config_cls = JsonQADatasetConfig

    config = config_cls(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend_per_split=blend_per_split,
        split=args.split,
        path_to_cache=args.data_cache_path,
        tokenizer=tokenizer,
        ft_neighbours=args.ft_neighbours,
        bert_retriever_neighbours=args.bert_retriever_neighbours,
        longform_answer=args.longform_answer,
        inference_only=False,
        retrieved_neighbours=False,
        fix_newsqa=True,
        **extra_kwargs
    )

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_cls,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider, model_provider,
        ModelType.retro_decoder,  # ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_tasks_args
    )
