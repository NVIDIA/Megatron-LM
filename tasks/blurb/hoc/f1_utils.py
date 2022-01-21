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
#from megatron.schedules_output import get_forward_backward_func
from megatron.schedules import get_forward_backward_func
from tasks.blurb.hoc.finetune_utils import build_data_loader
from tasks.blurb.hoc.finetune_utils import process_batch
from megatron.utils import average_losses_across_data_parallel_group

from sklearn.metrics import f1_score


def accuracy_f1_func_provider(single_dataset_provider):
    """Provide function that calculates accuracies."""
    args = get_args()

    # Build dataloaders.
    datapaths = args.valid_data
    dataloaders = []
    for datapath in datapaths:
        dataset = single_dataset_provider(datapath)
        #Set batch_size to 1, when calculating F1 scores
        args.f1_micro_batch_size = 1
        args.f1_global_batch_size = args.f1_micro_batch_size*args.data_parallel_size
        dataloader = build_data_loader(
            dataset, args.f1_micro_batch_size, num_workers=args.num_workers,
            drop_last=(mpu.get_data_parallel_world_size() > 1))
        dataloaders.append((dataset.dataset_name, dataloader))
        #dataloader = build_data_loader(
        #    dataset, args.orig_micro_batch_size, num_workers=args.num_workers,
        #    drop_last=(mpu.get_data_parallel_world_size() > 1))
        #dataloaders.append((dataset.dataset_name, dataloader))

    def metrics_func(model, epoch, output_predictions=False):
        print_rank_last('calculating metrics ...')
        num_classes=10
        f1 = np.zeros(num_classes)
        total = 0
        correct = 0
        if output_predictions:
            assert mpu.get_data_parallel_world_size() == 1
            named_predictions = []
            names = 'predictions'
        
        for name, dataloader in dataloaders:
            output = calculate_correct_answers(name, model, dataloader,
                                               epoch, output_predictions)
            if not output_predictions:
                #correct_ans, total_count = output
                #f1_scores, correct_ans, total_count = output
                f1_scores, total_count = output
            else:
                correct_ans, total_count, predictions = output
                named_predictions.append((name, predictions))
                names += '_' + name
            if mpu.is_pipeline_last_stage():
            #if is_last_rank():
                for i in range(num_classes):
                    f1[i] += f1_scores[i]
                total += total_count
                #correct += correct_ans
        if is_last_rank():
            for i in range(num_classes):
                #percent = float(correct[i]) * 100.0 / float(total)
                print(' >> |epoch: {}| overall: correct / total = {} / {} | '
                    'F1 Scores: {:.4f} '.format(epoch, correct, total, f1[i]))
        #if is_last_rank():
        #    for i in range(num_classes):
        #        percent = float(correct[i]) * 100.0 / float(total)
        #        print(' >> |epoch: {}| overall: correct / total = {} / {} = '
        #            '{:.4f} %'.format(epoch, correct[i], total, percent))

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
    #micro_batch_size_times_data_parallel = args.orig_micro_batch_size * args.data_parallel_size
    micro_batch_size_times_data_parallel = args.f1_micro_batch_size * args.data_parallel_size
    #num_micro_batches = args.orig_global_batch_size // micro_batch_size_times_data_parallel
    num_micro_batches = args.f1_global_batch_size // micro_batch_size_times_data_parallel

    #def loss_func(output_predictions, labels, output_tensor, bs):
    def loss_func(output_predictions, labels, output_tensor):

        loss_fcn = torch.nn.CrossEntropyLoss()
        num_classes = 10
        loss_dict = {}
        loss = None
        for i in range(num_classes):
            if loss is None:
                loss = loss_fcn(output_tensor[:,i,:],labels[:,i])
            else:
                loss += loss_fcn(output_tensor[:,i,:],labels[:,i])

            predicted = torch.argmax(output_tensor[:,i,:], dim=-1).cpu()
            loss_dict['predicted{%d}' % i] = predicted

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
    abstract_scores = {}
    abstract_truth = {}
    correct = np.zeros(num_classes)
    total = 0
    with torch.no_grad():
        # For all the batches in the dataset.
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


            tokens,types,labels,attention_mask,abstract_ids = process_batch(batch)
            
            #output_tensor = forward_backward_func(correct_answers_forward_step, batch, model, 
            #                                   optimizer=None, timers=None, forward_only=True)
            loss_dicts = forward_backward_func(correct_answers_forward_step, batch, model, 
                                               optimizer=None, timers=None, forward_only=True)

            abstract_id = abstract_ids.cpu()[0]
            batch_labels = labels.cpu()[0]

            if abstract_id not in abstract_scores:
                abstract_scores[abstract_id] = np.zeros(num_classes)
            if abstract_id not in abstract_truth:
                abstract_truth[abstract_id] = batch_labels
            else:
                abstract_truth[abstract_id] += batch_labels

            for loss_dict in loss_dicts:
                total += loss_dict['total']
                abstract_scores[abstract_id][0] += loss_dict['predicted{0}']
                abstract_scores[abstract_id][1] += loss_dict['predicted{1}']
                abstract_scores[abstract_id][2] += loss_dict['predicted{2}']
                abstract_scores[abstract_id][3] += loss_dict['predicted{3}']
                abstract_scores[abstract_id][4] += loss_dict['predicted{4}']
                abstract_scores[abstract_id][5] += loss_dict['predicted{5}']
                abstract_scores[abstract_id][6] += loss_dict['predicted{6}']
                abstract_scores[abstract_id][7] += loss_dict['predicted{7}']
                abstract_scores[abstract_id][8] += loss_dict['predicted{8}']
                abstract_scores[abstract_id][9] += loss_dict['predicted{9}']

    pred_labels = np.zeros((len(abstract_scores), num_classes), dtype=np.int32)
    actual_labels = np.zeros((len(abstract_scores), num_classes), dtype=np.int32)         
    for i,abstract_id in enumerate(abstract_scores.keys()):
        pred_labels[i,:] = np.clip(abstract_scores[abstract_id], 0, 1)
        actual_labels[i,:] = np.clip(abstract_truth[abstract_id], 0, 1)
        correct += 1.0*((abstract_scores[abstract_id] > 0) == (abstract_truth[abstract_id] > 0))

    f1 = np.zeros(num_classes)
    for j in range(num_classes):
        f1[j] = f1_score(actual_labels[:,j], pred_labels[:,j])

    for m in model:
        m.train()
    args.micro_batch_size = saved_micro_batch_size
    args.global_batch_size = saved_global_batch_size

    # Reduce.
    if mpu.is_pipeline_last_stage():
        f1_scores = np.zeros(num_classes)
        for i in range(num_classes):
            #unreduced = torch.cuda.LongTensor([correct, total])
            unreduced = torch.cuda.LongTensor([total])
            torch.distributed.all_reduce(unreduced,
                                        group=mpu.get_data_parallel_group())
            total_count = unreduced[0].item()

            unreducedFloat = torch.cuda.FloatTensor([f1[i]])
            torch.distributed.all_reduce(unreducedFloat,
                                        group=mpu.get_data_parallel_group())
            f1_scores[i] = unreducedFloat[0].item()
            elapsed_time = time.time() - start_time

        if output_predictions:
            return correct_ans, total_count, (softmaxes, labels, ids)
        #return f1_scores, correct, total_count
        return f1_scores, total_count
    if output_predictions:
        return 0, 0, ()
    return 0, 0
