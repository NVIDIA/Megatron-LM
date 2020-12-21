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

import torch

from megatron import get_args
from megatron import print_rank_last, is_last_rank
from megatron import mpu
from megatron.training import communicate
from tasks.finetune_utils import build_data_loader
from tasks.finetune_utils import process_batch


def accuracy_func_provider(single_dataset_provider):
    """Provide function that calculates accuracies."""
    args = get_args()

    # Build dataloaders.
    datapaths = args.valid_data
    dataloaders = []
    for datapath in datapaths:
        dataset = single_dataset_provider(datapath)
        dataloader = build_data_loader(
            dataset, args.micro_batch_size, num_workers=args.num_workers,
            drop_last=(mpu.get_data_parallel_world_size() > 1))
        dataloaders.append((dataset.dataset_name, dataloader))

    def metrics_func(model, epoch, output_predictions=False):
        print_rank_last('calculating metrics ...')
        correct = 0
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
            correct += correct_ans
            total += total_count
        if is_last_rank():
            percent = float(correct) * 100.0 / float(total)
            print(' >> |epoch: {}| overall: correct / total = {} / {} = '
                  '{:.4f} %'.format(epoch, correct, total, percent))

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
    start_time = time.time()
    model.eval()
    saved_batch_size = args.micro_batch_size
    with torch.no_grad():
        # For all the batches in the dataset.
        total = 0
        correct = 0
        if output_predictions:
            # This option is only possible when data parallel size is 1.
            assert mpu.get_data_parallel_world_size() == 1
            softmaxes = []
            labels = []
            ids = []
        for _, batch in enumerate(dataloader):
            # Run the model forward.
            tokens, types, labels_, attention_mask = process_batch(batch)

            # For evaluation only mode we use drop_last = False to get all the
            # samples, which means we might not have a full batch, so we
            # adjust batch_size here to actual batch size of data
            actual_batch_size = len(labels_)
            # ... applying sample_multiplier if necessary
            ds = dataloader.dataset
            if hasattr(ds, 'sample_multiplier'):
                actual_batch_size *= ds.sample_multiplier
            args.micro_batch_size = actual_batch_size

            if not mpu.is_pipeline_first_stage():
                input_tensor, _ = communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_forward=True,
                    recv_backward=False)
            else:
                input_tensor = None

            # Forward model.
            if mpu.is_pipeline_first_stage():
                assert input_tensor is None
                output_tensor = model(tokens, attention_mask, tokentype_ids=types)
            else:
                assert input_tensor is not None
                output_tensor = model(input_tensor, attention_mask)

            if mpu.is_pipeline_last_stage():
                logits = output_tensor

                # Add output predictions.
                if output_predictions:
                    softmaxes.extend(torch.nn.Softmax(dim=-1)(
                        logits.float()).data.cpu().numpy().tolist())
                    labels.extend(labels_.data.cpu().numpy().tolist())
                    ids.extend(batch['uid'].cpu().numpy().tolist())
                # Compute the correct answers.
                predicted = torch.argmax(logits, dim=-1)
                corrects = (predicted == labels_)
                # Add to the counters.
                total += labels_.size(0)
                correct += corrects.sum().item()
            else:
                communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_forward=False,
                    recv_backward=False)

    model.train()
    args.micro_batch_size = saved_batch_size

    # Reduce.
    if mpu.is_pipeline_last_stage():
        unreduced = torch.cuda.LongTensor([correct, total])
        torch.distributed.all_reduce(unreduced,
                                     group=mpu.get_data_parallel_group())

        # Print on screen.

        correct_ans = unreduced[0].item()
        total_count = unreduced[1].item()
        percent = float(correct_ans) * 100.0 / float(total_count)
        elapsed_time = time.time() - start_time
        print_rank_last(' > |epoch: {}| metrics for {}: correct / total '
                        '= {} / {} = {:.4f} %, elapsed time (sec): {:.3f}'.format(
                            epoch, name, correct_ans, total_count,
                            percent, elapsed_time))

        if output_predictions:
            return correct_ans, total_count, (softmaxes, labels, ids)
        return correct_ans, total_count
    if output_predictions:
        return 0, 0, ()
    return 0, 0
