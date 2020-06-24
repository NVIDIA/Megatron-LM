# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT2 zero-shot evaluation."""

import math

import torch

from megatron import get_args, print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.model import GPT2Model
from megatron.training import get_model
from megatron.utils import get_ltor_masks_and_position_ids
from tasks.finetune_utils import build_data_loader

from .datasets import build_dataset


def get_model_provider(eval_metric):
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""

    def model_provider():
        """Build the model."""

        if eval_metric == 'loss':
            parallel_output = True
        elif eval_metric == 'accuracy':
            parallel_output = False
        else:
            raise NotImplementedError('output type for {} evaluation metric '
                                      'is not supported.'.format(eval_metric))

        print_rank_0('building GPT2 model ...')
        model = GPT2Model(num_tokentypes=0, parallel_output=parallel_output)

        return model

    return model_provider


def process_batch(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()
    tokenizer = get_tokenizer()

    loss_mask = batch['pad_mask'].long().cuda().contiguous().byte()
    tokens_ = batch['text'].long().cuda().contiguous()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, attention_mask, position_ids, loss_mask


def forward_step(batch, model, eval_metric):
    """Forward step."""

    # Get the batch.
    tokens, labels, attention_mask, position_ids, loss_mask = process_batch(
        batch)

    # Forward model.
    output = model(tokens, position_ids, attention_mask)

    # For loss, return the unreduced loss.
    if eval_metric == 'loss':
        losses = mpu.vocab_parallel_cross_entropy(
            output.contiguous().float(), labels.contiguous())
        loss = torch.sum(
            losses.view(-1) * loss_mask.contiguous().view(-1).float())
        return loss

    # For accuracy, return the number of correctly predicted samples.
    if eval_metric == 'accuracy':
        outputs = torch.argmax(output, -1)
        correct = (outputs == labels).float()
        correct[(1 - loss_mask).bool()] = 1
        correct = correct.prod(-1)
        return correct.sum()

    raise NotImplementedError('forward method for evaluation metric {} '
                              'is not implemented.'.format(eval_metric))


def evaluate(data_loader, model, eval_metric):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_output = 0.0
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(data_loader):
            if iteration % args.log_interval == 0:
                print_rank_0('> working on iteration: {}'.format(iteration))
            # Forward evaluation.
            output = forward_step(batch, model, eval_metric)

            # Reduce across processes.
            torch.distributed.all_reduce(output,
                                         group=mpu.get_data_parallel_group())

            total_output += output

    return total_output


def evaluate_and_print_results(task, data_loader, model, eval_metric):
    """Evaluate and print results on screen."""

    # Evaluate and get results.
    output = evaluate(data_loader, model, eval_metric)

    string = ' validation results on {} | '.format(task)
    if eval_metric == 'loss':
        num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
        num_original_tokens = data_loader.dataset.num_original_tokens
        val_loss = output / (num_tokenized_tokens - 1)
        ppl = math.exp(min(20, val_loss))
        token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens - 1)
        adjusted_ppl = math.exp(min(20, val_loss * token_ratio))
        string += 'avg loss: {:.4E} | '.format(val_loss)
        string += 'ppl: {:.4E} | '.format(ppl)
        string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
        string += 'token ratio: {} |'.format(token_ratio)

    elif eval_metric == 'accuracy':
        num_examples = len(data_loader.dataset)
        acc = output / num_examples
        string += 'number correct: {:.4E} | '.format(output)
        string += 'total examples: {:.4E} | '.format(num_examples)
        string += 'avg accuracy: {:.4E}'.format(acc)

    else:
        raise NotImplementedError('evaluation method for {} metric is not '
                                  'implemented yet.'.format(eval_metric))

    length = len(string) + 1
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)


def main():
    """Main program."""
    args = get_args()

    if args.task == 'LAMBADA':
        eval_metric = 'accuracy'
    elif args.task == 'WIKITEXT103':
        eval_metric = 'loss'
    else:
        raise NotImplementedError('{} task is not implemented.'.format(
            args.task))

    # Set up model and load checkpoint.
    model = get_model(get_model_provider(eval_metric))
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    # Data stuff.
    dataset = build_dataset(args.task)
    dataloader = build_data_loader(dataset, args.batch_size,
                                   args.num_workers, drop_last=False)

    # Run evaluation.
    evaluate_and_print_results(args.task, dataloader, model, eval_metric)

    print_rank_0('done :-)')
