# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain VIT"""

import torch
import torch.nn.functional as F
from functools import partial
from megatron import get_args, get_timers, print_rank_0, print_rank_last
from megatron.core.enums import ModelType
from megatron.data.vit_dataset import build_train_valid_datasets
from megatron.model.vision.inpainting import VitInpaintingModel
from megatron.model.vision.inpainting import MitInpaintingModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from tasks.vision.metrics import SSIM, PSNR
from megatron.arguments import core_transformer_config_from_args

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()
    config = core_transformer_config_from_args(args)
    if args.vision_backbone_type == 'vit':
        model = VitInpaintingModel(config,
                                   pre_process=pre_process,
                                   post_process=post_process)
    elif args.vision_backbone_type == 'mit':
        model = MitInpaintingModel(pre_process=pre_process,
                                   post_process=post_process)
    else:
        raise Exception('{} vision backbone is not supported.'.format(
                              args.vision_backbone_type))
    return model


def get_batch(data_iterator):
    """Build the batch."""
    data = next(data_iterator)

    # only data parallelism; no need for broadcast
    images = data[0][0].cuda()
    masks = data[0][1].cuda()
    return images, masks


def loss_func(images, masks, masked_images, outputs, collect_data=False):
    outputs = outputs.contiguous().float()
    masks_flip = 1-masks
    flip_masked_outputs = outputs.masked_fill(masks_flip.bool(), 0)
    flip_masked_images = images.masked_fill(masks_flip.bool(), 0)

    ssim_fun = SSIM()
    psnr_fun = PSNR()

    if not collect_data:
        mask_count = torch.count_nonzero(masks)
        loss = F.mse_loss(
            flip_masked_outputs,
            flip_masked_images.float(),
            reduction="sum"
        )
        loss = loss/mask_count
        ssim = ssim_fun(flip_masked_outputs, flip_masked_images.float())
        psnr = psnr_fun(flip_masked_outputs, flip_masked_images.float())

        averaged_loss = average_losses_across_data_parallel_group(
            [loss, psnr, ssim]
        )

        return loss, {"loss": averaged_loss[0],
                      "psnr": averaged_loss[1],
                      'ssim': averaged_loss[2]}
    else:
        synth_images = masked_images.float() + flip_masked_outputs
        ssim = ssim_fun(synth_images, images.float())
        psnr = psnr_fun(synth_images, images.float())
        return torch.cat((images, masked_images, synth_images), dim=2), ssim, psnr


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    (
        images,
        masks,
    ) = get_batch(data_iterator)
    timers("batch-generator").stop()

    masked_images = images.masked_fill(masks.bool(), 0)
    outputs = model(masked_images)

    # Forward mode
    return outputs, partial(loss_func, images, masks, masked_images)


def process_non_loss_data(data, iteration, writer):
    psnr_sum = 0
    ssim_sum = 0
    for (output_tb, ssim, psnr) in data:
        output_tb[output_tb < 0] = 0
        output_tb[output_tb > 1] = 1
        writer.add_images("gt-input-output-vald", output_tb,
                          global_step=iteration, walltime=None,
                          dataformats='NCHW')
        psnr_sum = psnr_sum + psnr.item()
        ssim_sum = ssim_sum + ssim.item()
    psnr = psnr_sum/len(data)
    ssim = ssim_sum/len(data)
    writer.add_scalar('PSNR generate value-validation', psnr, iteration)
    writer.add_scalar('SSIM generate value-validation', ssim, iteration)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0(
        "> building train, validation, and test datasets " "for VIT ..."
    )
    train_ds, valid_ds = build_train_valid_datasets(
        data_path=args.data_path,
        image_size=(args.img_h, args.img_w)
    )
    print_rank_0("> finished creating VIT datasets ...")

    return train_ds, valid_ds, None


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        process_non_loss_data,
        args_defaults={'dataloader_type': 'cyclic', 'vision_pretraining': True}
    )
