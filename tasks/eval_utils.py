# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Evaluation utilities."""

import os
import time
from functools import partial

import torch

from megatron import get_args
from megatron import print_rank_last, is_last_rank
from megatron.core import mpu
from megatron.schedules import get_forward_backward_func
from tasks.finetune_utils import build_data_loader
from tasks.finetune_utils import process_batch
from deepspeed.accelerator import get_accelerator


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
            percent = 0
            if total > 0:
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

    def loss_func(output_predictions, labels, output_tensor):
        args = get_args()
        logits = output_tensor

        loss_dict = {}
        # Add output predictions.
        if output_predictions:
            assert False
            loss_dict['softmaxes'] = torch.nn.Softmax(dim=-1)(
                logits.float()).data.cpu().numpy().tolist()
            loss_dict['labels'] = labels.data.cpu().numpy().tolist()
            loss_dict['ids'] = batch['uid'].cpu().numpy().tolist()
        # Compute the correct answers.
        if args.finetune and args.task == 'CoLA':
            predicted = torch.argmax(logits, dim=-1)
            loss_dict['labels'] = labels.data.cpu().numpy().tolist()
            loss_dict['predicted'] = predicted.data.cpu().numpy().tolist()
        elif args.finetune and args.task == 'STS-B':
            predicted = torch.squeeze(logits)
            loss_dict['labels'] = labels.data.cpu().numpy().tolist()
            loss_dict['predicted'] = predicted.data.cpu().numpy().tolist()
        else:
            predicted = torch.argmax(logits, dim=-1)
            corrects = (predicted == labels)
            # Add to the counters.
            loss_dict['total'] = labels.size(0)
            loss_dict['correct'] = corrects.sum().item()

        return 0, loss_dict

    # defined inside to capture output_predictions
    def correct_answers_forward_step(batch, model):
        try:
            batch_ = next(batch)
        except BaseException:
            batch_ = batch
        tokens, types, labels, attention_mask = process_batch(batch_)

        # Forward model.
        args = get_args()
        output_tensor = model(tokens, attention_mask, tokentype_ids=types)

        return output_tensor, partial(loss_func, output_predictions, labels)

    with torch.no_grad():
        # For all the batches in the dataset.
        total = 0
        correct = 0
        labels = []
        predicted = []
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
            actual_batch_size = len(batch['label'])
            # ... applying sample_multiplier if necessary
            args.micro_batch_size = actual_batch_size * sample_multiplier
            args.global_batch_size = actual_batch_size * sample_multiplier * num_micro_batches

            loss_dicts = forward_backward_func(correct_answers_forward_step, batch, model,
                                               optimizer=None, timers=None, forward_only=True)

            for loss_dict in loss_dicts:
                if output_predictions:
                    softmaxes.extend(loss_dict['softmaxes'])
                    labels.extend(loss_dict['labels'])
                    ids.extend(loss_dict['ids'])
                if args.finetune and args.task in ['CoLA', 'STS-B']:
                    labels.extend(loss_dict['labels'])
                    predicted.extend(loss_dict['predicted'])
                else:
                    total += loss_dict['total']
                    correct += loss_dict['correct']


    for m in model:
        m.train()
    args.micro_batch_size = saved_micro_batch_size
    args.global_batch_size = saved_global_batch_size

    # Reduce.
    if mpu.is_pipeline_last_stage():
        if args.finetune and args.task in ['CoLA', 'STS-B']:
            if args.task == 'CoLA':
                labels = get_accelerator().LongTensor(labels)
                predicted = get_accelerator().LongTensor(predicted)
                labels_gather = [torch.zeros(len(labels), dtype=torch.long,
                    device=labels.device) for _ in range(mpu.get_data_parallel_world_size())]
                predicted_gather = [torch.zeros(len(predicted), dtype=torch.long,
                    device=predicted.device) for _ in range(mpu.get_data_parallel_world_size())]
            else:
                labels = get_accelerator().FloatTensor(labels)
                predicted = get_accelerator().FloatTensor(predicted)
                labels_gather = [torch.zeros(len(labels), dtype=torch.float,
                    device=labels.device) for _ in range(mpu.get_data_parallel_world_size())]
                predicted_gather = [torch.zeros(len(predicted), dtype=torch.float,
                    device=predicted.device) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(labels_gather, labels,
                group=mpu.get_data_parallel_group())
            torch.distributed.all_gather(predicted_gather, predicted,
                group=mpu.get_data_parallel_group())

            labels_gather = sum([x.data.cpu().numpy().tolist() for x in labels_gather], [])
            predicted_gather = sum([x.data.cpu().numpy().tolist() for x in predicted_gather], [])

            # Print on screen.
            if args.task == 'CoLA':
                from sklearn.metrics import matthews_corrcoef
                mcc = matthews_corrcoef(labels_gather, predicted_gather)
                elapsed_time = time.time() - start_time
                print_rank_last(' > |epoch: {}| metrics for {}: mcc '
                                '= {} , elapsed time (sec): {:.3f}'.format(
                                    epoch, name, mcc, elapsed_time))
            else:
                from scipy.stats import pearsonr, spearmanr
                pearson_corr = pearsonr(predicted_gather, labels_gather)[0]
                spearman_corr = spearmanr(predicted_gather, labels_gather)[0]
                corr = (pearson_corr + spearman_corr) / 2
                elapsed_time = time.time() - start_time
                print_rank_last(' > |epoch: {}| metrics for {}: pearson '
                                '= {} spearmanr = {} corr = {} elapsed time (sec): {:.3f}'.format(
                                    epoch, name, pearson_corr, spearman_corr,
                                    corr, elapsed_time))

            if output_predictions:
                return 0, 0, ()
            return 0, 0
        else:
            unreduced = get_accelerator().LongTensor([correct, total])
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
