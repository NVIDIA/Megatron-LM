# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Vision-classification finetuning/evaluation."""

import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
from megatron import get_args, get_timers
from megatron import print_rank_0, print_rank_last
from megatron.core import mpu
from tasks.vision.finetune_utils import finetune
from tasks.vision.finetune_utils import build_data_loader
from megatron.utils import average_losses_across_data_parallel_group
from megatron.schedules import get_forward_backward_func
from tasks.vision.segmentation.data import build_train_valid_datasets
from tasks.vision.segmentation.seg_models import SegformerSegmentationModel
from megatron.model.vision.utils import resize


def calculate_iou(hist_data):
    acc = np.diag(hist_data).sum() / hist_data.sum()
    acc_cls = np.diag(hist_data) / hist_data.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    divisor = hist_data.sum(axis=1) + hist_data.sum(axis=0) - \
        np.diag(hist_data)
    iu = np.diag(hist_data) / divisor
    return iu, acc, acc_cls


def fast_hist(pred, gtruth, num_classes):
    # mask indicates pixels we care about
    mask = (gtruth >= 0) & (gtruth < num_classes)

    # stretch ground truth labels by num_classes
    #   class 0  -> 0
    #   class 1  -> 19
    #   class 18 -> 342
    #
    # TP at 0 + 0, 1 + 1, 2 + 2 ...
    #
    # TP exist where value == num_classes*class_id + class_id
    # FP = row[class].sum() - TP
    # FN = col[class].sum() - TP
    hist = np.bincount(num_classes * gtruth[mask].astype(int) + pred[mask],
                       minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)
    return hist


def segmentation():

    def train_valid_datasets_provider():
        """Build train and validation dataset."""
        args = get_args()

        train_ds, valid_ds = build_train_valid_datasets(
            data_path=args.data_path,
            image_size=(args.img_h, args.img_w)

        )
        return train_ds, valid_ds

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        args = get_args()

        model = SegformerSegmentationModel(num_classes=args.num_classes,
                                           pre_process=pre_process,
                                           post_process=post_process)
        print_rank_0("model = {}".format(model))
        return model

    def process_batch(batch):
        """Process batch and produce inputs for the model."""
        images = batch[0].cuda().contiguous()
        masks = batch[1].cuda().contiguous()
        return images, masks

    def calculate_weight(masks, num_classes):
        bins = torch.histc(masks, bins=num_classes, min=0.0, max=num_classes)
        hist_norm = bins.float()/bins.sum()
        hist = ((bins != 0).float() * (1. - hist_norm)) + 1.0
        return hist

    def cross_entropy_loss_func(images, masks, output_tensor,
                                non_loss_data=False):
        args = get_args()
        ignore_index = args.ignore_index
        color_table = args.color_table
        logits = output_tensor.contiguous().float()
        logits = resize(logits, size=masks.shape[1:],
                        mode='bilinear', align_corners=False)
      
        # Cross-entropy loss.
        # weight = calculate_weight(masks, num_classes)
        loss = F.cross_entropy(logits, masks, ignore_index=ignore_index)

        if not non_loss_data:
            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])
            return loss, {'lm loss': averaged_loss[0]}
        else:
            seg_mask = logits.argmax(dim=1)
            output_mask = F.embedding(seg_mask, color_table).permute(0, 3, 1, 2)
            gt_mask = F.embedding(masks, color_table).permute(0, 3, 1, 2)
            return torch.cat((images, output_mask, gt_mask), dim=2), loss

    def _cross_entropy_forward_step(batch, model):
        """Simple forward step with cross-entropy loss."""
        timers = get_timers()

        # Get the batch.
        timers("batch generator", log_level=2).start()
        import types
        if isinstance(batch, types.GeneratorType):
            batch_ = next(batch)
        else:
            batch_ = batch
        images, masks = process_batch(batch_)
        timers("batch generator").stop()

        # Forward model.
        output_tensor = model(images)

        return output_tensor, partial(cross_entropy_loss_func, images, masks)

    def calculate_correct_answers(model, dataloader, epoch):
        """Calculate correct over total answers"""

        forward_backward_func = get_forward_backward_func()
        for m in model:
            m.eval()

        def loss_func(labels, output_tensor):
            args = get_args()
            logits = output_tensor
            logits = resize(logits, size=labels.shape[1:],
                            mode='bilinear', align_corners=False)

            loss_dict = {}
            # Compute the correct answers.
            probs = logits.contiguous().float().softmax(dim=1)
            max_probs, preds = torch.max(probs, 1)

            preds = preds.cpu().numpy()
            performs = fast_hist(preds.flatten(),
                                 labels.cpu().numpy().flatten(),
                                 args.ignore_index)
            loss_dict['performs'] = performs
            return 0, loss_dict

        # defined inside to capture output_predictions
        def correct_answers_forward_step(batch, model):
            try:
                batch_ = next(batch)
            except BaseException:
                batch_ = batch
            images, labels = process_batch(batch_)

            # Forward model.
            output_tensor = model(images)

            return output_tensor, partial(loss_func, labels)

        with torch.no_grad():
            # For all the batches in the dataset.
            performs = None
            for _, batch in enumerate(dataloader):
                loss_dicts = forward_backward_func(correct_answers_forward_step,
                                                   batch, model,
                                                   optimizer=None,
                                                   timers=None,
                                                   forward_only=True)
                for loss_dict in loss_dicts:
                    if performs is None:
                        performs = loss_dict['performs']
                    else:
                        performs += loss_dict['performs']

        for m in model:
            m.train()
        # Reduce.
        if mpu.is_pipeline_last_stage():
            performs_tensor = torch.cuda.FloatTensor(performs)
            torch.distributed.all_reduce(performs_tensor,
                                         group=mpu.get_data_parallel_group())
            hist = performs_tensor.cpu().numpy()
            iu, acc, acc_cls = calculate_iou(hist)
            miou = np.nanmean(iu)

            return iu, miou

    def accuracy_func_provider():
        """Provide function that calculates accuracies."""
        args = get_args()

        train_ds, valid_ds = build_train_valid_datasets(
            data_path=args.data_path,
            image_size=(args.img_h, args.img_w)
        )
        dataloader = build_data_loader(
            valid_ds,
            args.micro_batch_size,
            num_workers=args.num_workers,
            drop_last=(mpu.get_data_parallel_world_size() > 1),
            shuffle=False
        )

        def metrics_func(model, epoch):
            print_rank_0("calculating metrics ...")
            iou, miou = calculate_correct_answers(model, dataloader, epoch)
            print_rank_last(
                " >> |epoch: {}| overall: iou = {},"
                "miou = {:.4f} %".format(epoch, iou, miou*100.0)
            )
        return metrics_func

    def dump_output_data(data, iteration, writer):
        for (output_tb, loss) in data:
            # output_tb[output_tb < 0] = 0
            # output_tb[output_tb > 1] = 1
            writer.add_images("image-outputseg-realseg", output_tb,
                              global_step=None, walltime=None,
                              dataformats='NCHW')

    """Finetune/evaluate."""
    finetune(
        train_valid_datasets_provider,
        model_provider,
        forward_step=_cross_entropy_forward_step,
        process_non_loss_data_func=dump_output_data,
        end_of_epoch_callback_provider=accuracy_func_provider,
    )


def main():
    segmentation()

