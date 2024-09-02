# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved. Except portions as noted which are Copyright (c) 2023 OpenGVLab and licensed under the MIT license found in LICENSE.
import numpy as np
import torch

from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.transforms import Compose, RandAugment, RandomResizedCrop, Resize, ToPILImage


# Imagenet's mean and std.
pixel_mean = [123.675, 116.28, 103.53]
pixel_std = [58.395, 57.12, 57.375]

# Reshape for broadcasting.
pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)


def convert_to_rgb(image):
    return image.convert("RGB")

def _transform_train_aug(img_h, img_w):
    return Compose([
        ToPILImage(),
        RandomResizedCrop((img_h, img_w), scale=(0.5, 1.0)),
        convert_to_rgb,
        RandAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    ])

def _transform_test(img_h, img_w):
    return Compose([
        ToPILImage(),
        Resize((img_h, img_w)),
        convert_to_rgb,
    ])


def standardize_image(img):
    """Standardize image pixel values."""
    return (torch.Tensor(np.array(img)).permute(2, 0, 1) - pixel_mean) / pixel_std


def get_visual_transform(img, img_h, img_w, use_tiling=False, max_num_tiles=1, use_thumbnail=False, augment=False):
    if use_tiling:
        assert img_h == img_w, "dynamic tiling expects equal tile height and width"
        imgs = dynamic_preprocess(img, min_num=1, max_num=max_num_tiles, image_size=img_h, use_thumbnail=use_thumbnail)
        imgs = [standardize_image(img.convert("RGB")) for img in imgs]
    else:
        img = np.array(img)
        original_h, original_w = img.shape[0], img.shape[1]
        ratio = float(max(img_h, img_w)) / max(original_h, original_w)
        scaled_h, scaled_w = int(original_h * ratio + 0.5), int(original_w * ratio + 0.5)

        if augment:
            visual_transform = _transform_train_aug(scaled_h, scaled_w)
        else:
            visual_transform = _transform_test(scaled_h, scaled_w)

        img = visual_transform(img)

        # Standardize pixel values.
        img = standardize_image(img)

        # Pad to target image size.
        delta_h, delta_w = img_h - scaled_h, img_w - scaled_w
        img = torch.nn.functional.pad(img, (0, delta_w, 0, delta_h))
        imgs = [img]

    return imgs


# From https://github.com/OpenGVLab/InternVL/blob/c62fa4f7c850165d7386bdc48ac6bc5a6fab0864/internvl_chat/internvl/train/dataset.py#L685
# Copyright (c) 2023 OpenGVLab.
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


# From https://github.com/OpenGVLab/InternVL/blob/c62fa4f7c850165d7386bdc48ac6bc5a6fab0864/internvl_chat/internvl/train/dataset.py#L702
# Copyright (c) 2023 OpenGVLab.
def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
