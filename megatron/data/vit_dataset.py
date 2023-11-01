# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import os
import random
import numpy as np
import torch
import torchvision.transforms as T
from torchvision import datasets
from megatron import get_args
from megatron.data.image_folder import ImageFolder
from megatron.data.autoaugment import ImageNetPolicy
from megatron.data.data_samplers import RandomSeedDataset
from PIL import Image, ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class ClassificationTransform():
    def __init__(self, image_size, train=True):
        args = get_args()
        assert args.fp16 or args.bf16
        self.data_type = torch.half if args.fp16 else torch.bfloat16
        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                ImageNetPolicy(),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                T.ConvertImageDtype(self.data_type)
            ])
        else:
            self.transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                T.ConvertImageDtype(self.data_type)
            ])

    def __call__(self, input):
        output = self.transform(input)
        return output


class InpaintingTransform():
    def __init__(self, image_size, train=True):

        args = get_args()
        self.mask_factor = args.mask_factor
        self.mask_type = args.mask_type
        self.image_size = image_size
        self.patch_size = args.patch_dim
        self.mask_size = int(self.mask_factor*(image_size[0]/self.patch_size)*(image_size[1]/self.patch_size))
        self.train = train
        assert args.fp16 or args.bf16
        self.data_type = torch.half if args.fp16 else torch.bfloat16
     
        if self.train:
            self.transform = T.Compose([
                T.RandomResizedCrop(self.image_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                ImageNetPolicy(),
                T.ToTensor(),
                T.ConvertImageDtype(self.data_type)
            ])
        else:
            self.transform = T.Compose([
                T.Resize(self.image_size, interpolation=2),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.ConvertImageDtype(self.data_type)
            ])

    def gen_mask(self, image_size, mask_size, mask_type, patch_size):
        # output: mask as a list with indices for missing patches
        action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        assert image_size[0] == image_size[1]
        img_size_patch = image_size[0] // patch_size

        # drop masked patches
        mask = torch.zeros((image_size[0], image_size[1]), dtype=torch.float)

        if mask_type == 'random':
            x = torch.randint(0, img_size_patch, ())
            y = torch.randint(0, img_size_patch, ())
            for i in range(mask_size):
                r = torch.randint(0, len(action_list), ())
                x = torch.clamp(x + action_list[r][0], min=0, max=img_size_patch - 1)
                y = torch.clamp(y + action_list[r][1], min=0, max=img_size_patch - 1)
                x_offset = x * patch_size
                y_offset = y * patch_size
                mask[x_offset:x_offset+patch_size, y_offset:y_offset+patch_size] = 1
        else:
            assert mask_type == 'row'
            count = 0
            for x in reversed(range(img_size_patch)):
                for y in reversed(range(img_size_patch)):
                    if (count < mask_size):
                        count += 1
                        x_offset = x * patch_size
                        y_offset = y * patch_size
                        mask[x_offset:x_offset+patch_size, y_offset:y_offset+patch_size] = 1
        return mask

    def __call__(self, input):
        trans_input = self.transform(input)
        mask = self.gen_mask(self.image_size, self.mask_size, 
			     self.mask_type, self.patch_size)
        mask = mask.unsqueeze(dim=0)
        return trans_input, mask


class DinoTransform(object):
    def __init__(self, image_size, train=True):
        args = get_args()
        self.data_type = torch.half if args.fp16 else torch.bfloat16

        flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4,
			       saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])

        if args.fp16 or args.bf16:
            normalize = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                T.ConvertImageDtype(self.data_type)
            ])
        else:
            normalize = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        # first global crop
        scale_const = 0.4
        self.global_transform1 = T.Compose([
            T.RandomResizedCrop(image_size,
                                scale=(scale_const, 1),
                                interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize
        ])
        # second global crop
        self.global_transform2 = T.Compose([
            T.RandomResizedCrop(image_size,
                                scale=(scale_const, 1),
                                interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize
        ])
        # transformation for the local small crops
        self.local_crops_number = args.dino_local_crops_number
        self.local_transform = T.Compose([
            T.RandomResizedCrop(args.dino_local_img_size,
                                scale=(0.05, scale_const),
                                interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


def build_train_valid_datasets(data_path, image_size=224):
    args = get_args()

    if args.vision_pretraining_type == 'classify':
        train_transform = ClassificationTransform(image_size)
        val_transform = ClassificationTransform(image_size, train=False)
    elif args.vision_pretraining_type == 'inpaint':
        train_transform = InpaintingTransform(image_size, train=False)
        val_transform = InpaintingTransform(image_size, train=False)
    elif args.vision_pretraining_type == 'dino':
        train_transform = DinoTransform(image_size, train=True)
        val_transform = ClassificationTransform(image_size, train=False)
    else:
        raise Exception('{} vit pretraining type is not supported.'.format(
                args.vit_pretraining_type))

    # training dataset
    train_data_path = data_path[0] if len(data_path) <= 2 else data_path[2]
    train_data = ImageFolder(
        root=train_data_path,
        transform=train_transform,
        classes_fraction=args.classes_fraction,
        data_per_class_fraction=args.data_per_class_fraction
    )
    train_data = RandomSeedDataset(train_data)

    # validation dataset
    val_data_path = data_path[1]
    val_data = ImageFolder(
        root=val_data_path,
        transform=val_transform
    )
    val_data = RandomSeedDataset(val_data)

    return train_data, val_data
