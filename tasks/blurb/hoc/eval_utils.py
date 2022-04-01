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

"""Evaluation utilities."""

import os
import time
from functools import partial

import torch
import numpy as np

from megatron import get_args
from megatron import print_rank_last, is_last_rank
from megatron import mpu
from megatron.schedules import get_forward_backward_func
from tasks.blurb.hoc.finetune_utils import build_data_loader
from tasks.blurb.hoc.finetune_utils import process_batch
from megatron.utils import average_losses_across_data_parallel_group


def accuracy_func_provider(single_dataset_provider):
    """Provide function that calculates accuracies."""
    args = get_args()

    # Build dataloaders.
    datapaths = args.valid_data
    dataloaders = []
    for datapath in datapaths:
        dataset = single_dataset_provider(datapath)
        dataloader = build_data_loader(
            dataset, args.orig_micro_batch_size, num_workers=args.num_workers,
            drop_last=(mpu.get_data_parallel_world_size() > 1))
        dataloaders.append((dataset.dataset_name, dataloader))

    def metrics_func(model, epoch, output_predictions=False):
        print_rank_last('calculating metrics ...')
        num_classes=10
        correct = np.zeros(num_classes, dtype=int)
        total = 0
        if output_predictions:
            assert mpu.get_data_parallel_world_size() == 1
            named_predictions = []
            names = 'predictions'
        
        for name, dataloader in dataloaders:
            output = calculate_correct_answers(name, model, dataloader,
                                               epoch, output_predictions)
            if not output_predictions:
                correct_ans, total_count = output
            else:
                correct_ans, total_count, predictions = output
                named_predictions.append((name, predictions))
                names += '_' + name
            if mpu.is_pipeline_last_stage():
            #if is_last_rank():
                for i in range(num_classes):
                    correct[i] += correct_ans[i]
                total += total_count
        if is_last_rank():
            for i in range(num_classes):
                percent = float(correct[i]) * 100.0 / float(total)
                print(' >> |epoch: {}| overall: correct / total = {} / {} = '
                    '{:.4f} %'.format(epoch+1, correct[i], total, percent))

        if output_predictions and is_last_rank():
            assert args.load is not None
            filename = os.path.join(args.load, names + '.pt')
            torch.save(named_predictions, filename)

    return metrics_func


def calculate_correct_answers(name, model, dataloader,
                              epoch, output_predictions):
    """Calculate correct over total answers and return prediction if the
    `output_predictions` is true."""
    args = get_args()
    forward_backward_func = get_forward_backward_func()
    start_time = time.time()
    for m in model:
        m.eval()
    saved_micro_batch_size = args.micro_batch_size
    saved_global_batch_size = args.global_batch_size

    ds = dataloader.dataset
    if hasattr(ds, 'sample_multiplier'):
        # If our dataset as a sample_multiplier attribute that means
        # each "sample" from the dataset actually has multiple samples
        # that will collapse into the batch dimension (for example in
        # the RACE dataset that has several options), we need to
        # account for that when setting the micro batch size.
        sample_multiplier = ds.sample_multiplier
    else:
        sample_multiplier = 1
    micro_batch_size_times_data_parallel = args.orig_micro_batch_size * args.data_parallel_size
    num_micro_batches = args.orig_global_batch_size // micro_batch_size_times_data_parallel

    #def loss_func(output_predictions, labels, output_tensor, bs):
    def loss_func(output_predictions, labels, output_tensor):

        loss_fcn = torch.nn.CrossEntropyLoss()
        num_classes = 10
        loss = None
        loss_dict = {}
        for i in range(num_classes):
            if loss is None:
                loss = loss_fcn(output_tensor[:,i,:],labels[:,i])
            else:
                loss += loss_fcn(output_tensor[:,i,:],labels[:,i])


            predicted = torch.argmax(output_tensor[:,i,:], dim=-1)
            corrects = (predicted == labels[:,i])

            loss_dict['correct{%d}' % i] = corrects.sum().item()

        loss_dict['total'] = labels.size(dim=0)
        #loss_dict['total'] = bs

        #averaged_loss = average_losses_across_data_parallel_group([loss])
        #return loss, {'lm loss': averaged_loss[0]}, loss_dict
        return 0, loss_dict

    # defined inside to capture output_predictions
    def correct_answers_forward_step(batch, model):
        try:
            batch_ = next(batch)
        except BaseException:
            batch_ = batch
        tokens, types, labels, attention_mask, abstract_ids = process_batch(batch_)

        # Forward model.
        args = get_args()
        output_tensor = model(tokens, attention_mask, tokentype_ids=types)

        #bs = len(batch['label'])
        #return output_tensor, partial(loss_func, output_predictions, labels, bs)
        return output_tensor, partial(loss_func, output_predictions, labels)

    num_classes = 10
    with torch.no_grad():
        # For all the batches in the dataset.
        total = 0
        correct = np.zeros(num_classes, dtype=int)
        if output_predictions:
            # This option is only possible when data parallel size is 1.
            assert mpu.get_data_parallel_world_size() == 1
            softmaxes = []
            labels = []
            ids = []
        for _, batch in enumerate(dataloader):
            # For evaluation only mode we use drop_last = False to get all the
            # samples, which means we might not have a full batch, so we
            # adjust batch_size here to actual batch size of data

            # ... applying sample_multiplier if necessary
            actual_batch_size = len(batch['label'])
            args.micro_batch_size = actual_batch_size * sample_multiplier
            args.global_batch_size = actual_batch_size * sample_multiplier * num_micro_batches

            loss_dicts = forward_backward_func(correct_answers_forward_step, batch, model,
                                               optimizer=None, timers=None, forward_only=True)

            for loss_dict in loss_dicts:
                if output_predictions:
                    softmaxes.extend(loss_dict['softmaxes'])
                    labels.extend(loss_dict['labels'])
                    ids.extend(loss_dict['ids'])

                total += loss_dict['total']
                correct[0] += loss_dict['correct{0}']
                correct[1] += loss_dict['correct{1}']
                correct[2] += loss_dict['correct{2}']
                correct[3] += loss_dict['correct{3}']
                correct[4] += loss_dict['correct{4}']
                correct[5] += loss_dict['correct{5}']
                correct[6] += loss_dict['correct{6}']
                correct[7] += loss_dict['correct{7}']
                correct[8] += loss_dict['correct{8}']
                correct[9] += loss_dict['correct{9}']
                #for i in range(num_classes):
                #    correct[i] += loss_dict['correct{%d}' % i]

    for m in model:
        m.train()
    args.micro_batch_size = saved_micro_batch_size
    args.global_batch_size = saved_global_batch_size

    # Reduce.
    if mpu.is_pipeline_last_stage():
        correct_ans = np.zeros(num_classes,dtype=int)
        for i in range(num_classes):
            unreduced = torch.cuda.LongTensor([correct[i], total])
            torch.distributed.all_reduce(unreduced,
                                        group=mpu.get_data_parallel_group())
            # Print on screen.
            correct_ans[i] = unreduced[0].item()
            total_count = unreduced[1].item()
            percent = float(correct_ans[i]) * 100.0 / float(total_count)
            elapsed_time = time.time() - start_time
            print_rank_last(' > |epoch: {}| metrics for {}: correct / total '
                            '= {} / {} = {:.4f} %, elapsed time (sec): {:.3f}'.format(
                                epoch+1, name, correct_ans[i], total_count,
                                percent, elapsed_time))

        if output_predictions:
            return correct_ans, total_count, (softmaxes, labels, ids)
        return correct_ans, total_count
    if output_predictions:
        return 0, 0, ()
    return 0, 0
