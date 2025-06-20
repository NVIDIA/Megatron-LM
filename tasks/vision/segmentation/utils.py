import math
import torch
import numpy as np
from megatron.training import get_args

def slidingcrops(img, mask):
    # img: [b c h w]
    # mask: [b h w]
    args = get_args()
    assert args.img_h == args.img_w
    crop_size = args.img_h
    stride = args.seg_stride
    ignore_index = args.ignore_index
    n, c, h, w = img.shape
    assert h >= crop_size
    assert w >= crop_size
    long_size = max(h, w)

    img_slices, mask_slices, slices_info = [], [], []
    if long_size > crop_size:
        assert stride <= crop_size
        h_step_num = int(math.ceil((h - crop_size) / float(stride))) + 1
        w_step_num = int(math.ceil((w - crop_size) / float(stride))) + 1
        for yy in range(h_step_num):
            for xx in range(w_step_num):
                sy, sx = yy * stride, xx * stride
                ey, ex = sy + crop_size, sx + crop_size
                img_sub = img[:, :, sy: ey, sx: ex]
                mask_sub = mask[:, sy: ey, sx: ex]

                # padding
                sub_h, sub_w = img_sub.shape[2:]
                pad_h = max(crop_size - sub_h, 0)
                pad_w = max(crop_size - sub_w, 0)
                img_sub = torch.nn.functional.pad(img_sub, pad=(0, pad_w, 0, pad_h), value=ignore_index)
                mask_sub = torch.nn.functional.pad(mask_sub, pad=(0, pad_w, 0, pad_h))

                img_slices.append(img_sub)
                mask_slices.append(mask_sub)
                slices_info.append([sy, ey, sx, ex, sub_h, sub_w])

        return torch.cat(img_slices), torch.cat(mask_slices), slices_info, (h, w)
    else:
        return img, mask, [[0, h, 0, w, h, w]], (h, w)


def slidingjoins(preds, probs, labels, slices_info, img_size):
    args = get_args()
    num_slices = len(slices_info)

    if num_slices == 1:
        return preds, labels

    h, w = img_size
    split_size = args.micro_batch_size

    preds_split = torch.split(preds, split_size)
    probs_split = torch.split(probs, split_size)
    labels_split = torch.split(labels, split_size)

    assert(len(preds_split) == num_slices)

    total_max_probs = torch.zeros((split_size, h, w), dtype=torch.float, device='cuda')
    total_preds = torch.zeros((split_size, h, w), dtype=torch.int, device='cuda')
    total_labels = torch.zeros((split_size, h, w), dtype=torch.int, device='cuda')

    for i in range(num_slices):
        sy, ey, sx, ex, sub_h, sub_w = slices_info[i]
        assert sy + sub_h <= h
        assert sx + sub_w <= w
        curr_max_probs = total_max_probs[:, sy:sy + sub_h, sx:sx + sub_w]
        curr_preds = total_preds[:, sy:sy + sub_h, sx:sx + sub_w]

        local_max_probs = probs_split[i][:, :sub_h, : sub_w]
        local_preds = preds_split[i][:, :sub_h, :sub_w]

        result_max_probs = torch.maximum(curr_max_probs, local_max_probs)
        result_preds = torch.where(curr_max_probs >= local_max_probs, curr_preds, local_preds)

        total_max_probs[:, sy:sy + sub_h, sx:sx + sub_w] = result_max_probs
        total_preds[:, sy:sy + sub_h, sx:sx + sub_w] = result_preds
        total_labels[:, sy:sy + sub_h, sx:sx + sub_w] = labels_split[i][0, :sub_h, :sub_w]

    return total_preds, total_labels

