# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Vision-classification finetuning/evaluation."""

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
from tasks.vision.segmentation.metrics import CFMatrix
from tasks.vision.segmentation.data import build_train_valid_datasets
from tasks.vision.segmentation.seg_models import SetrSegmentationModel
from tasks.vision.segmentation.utils import slidingcrops, slidingjoins

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

        return SetrSegmentationModel(num_classes=args.num_classes,
                                     pre_process=pre_process,
                                     post_process=post_process)

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

    def cross_entropy_loss_func(images, masks, output_tensor, non_loss_data=False):
        args = get_args()
        ignore_index = args.ignore_index
        color_table = args.color_table
        weight = calculate_weight(masks, args.num_classes)
        logits = output_tensor.contiguous().float()
        loss = F.cross_entropy(logits, masks, weight=weight, ignore_index=ignore_index)

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
        args = get_args()
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
        if not model.training:
            images, masks, _, _ = slidingcrops(images, masks)
        #print_rank_0("images size = {}".format(images.size()))
       
        if not model.training:
            output_tensor = torch.cat([model(image) for image in torch.split(images, args.micro_batch_size)])
        else:
            output_tensor = model(images)

        return output_tensor, partial(cross_entropy_loss_func, images, masks)

    def calculate_correct_answers(model, dataloader, epoch):
        """Calculate correct over total answers"""

        forward_backward_func = get_forward_backward_func()
        for m in model:
            m.eval()

        def loss_func(labels, slices_info, img_size, output_tensor):
            args = get_args()
            logits = output_tensor

            loss_dict = {}
            # Compute the correct answers.
            probs = logits.contiguous().float().softmax(dim=1)
            max_probs, preds = torch.max(probs, 1)
            preds = preds.int()
            preds, labels = slidingjoins(preds, max_probs, labels, slices_info, img_size)
            _, performs = CFMatrix()(preds, labels, args.ignore_index)

            loss_dict['performs'] = performs
            return 0, loss_dict

        # defined inside to capture output_predictions
        def correct_answers_forward_step(batch, model):
            args = get_args()
            try:
                batch_ = next(batch)
            except BaseException:
                batch_ = batch
            images, labels = process_batch(batch_)

            assert not model.training
            images, labels, slices_info, img_size = slidingcrops(images, labels)
            # Forward model.
            output_tensor = torch.cat([model(image) for image in torch.split(images, args.micro_batch_size)])

            return output_tensor, partial(loss_func, labels, slices_info, img_size)

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
            torch.distributed.all_reduce(performs,
                                         group=mpu.get_data_parallel_group())
            # Print on screen.
            # performs[int(ch), :] = [nb_tp, nb_fp, nb_tn, nb_fn]
            true_positive = performs[:, 0]
            false_positive = performs[:, 1]
            false_negative = performs[:, 3]

            iou = true_positive / (true_positive + false_positive + false_negative)
            miou = iou[~torch.isnan(iou)].mean()

            return iou.tolist(), miou.item()

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

