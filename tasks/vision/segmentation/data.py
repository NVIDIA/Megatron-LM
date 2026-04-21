import random
import os
import math
import mmcv
import torch
import numpy as np
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import Dataset
from megatron.legacy.data.autoaugment import ImageNetPolicy
from tasks.vision.segmentation.cityscapes import Cityscapes
import tasks.vision.segmentation.transforms as ET
from megatron.legacy.data.autoaugment import ImageNetPolicy
from megatron.training import get_args
from PIL import Image, ImageOps


class VitSegmentationJointTransform():
    def __init__(self, train=True, resolution=None):
        self.train = train
        if self.train:
            self.transform0 = ET.RandomSizeAndCrop(resolution)
            self.transform1 = ET.RandomHorizontallyFlip()

    def __call__(self, img, mask):
        if self.train:
            img, mask = self.transform0(img, mask)
            img, mask = self.transform1(img, mask)
        return img, mask


class VitSegmentationImageTransform():
    def __init__(self, train=True, resolution=None):
        args = get_args()
        self.train = train
        assert args.fp16 or args.bf16
        self.data_type = torch.half if args.fp16 else torch.bfloat16
        self.mean_std = args.mean_std
        if self.train:
            assert resolution is not None
            self.transform = T.Compose([
                ET.PhotoMetricDistortion(),
                T.ToTensor(),
                T.Normalize(*self.mean_std),
                T.ConvertImageDtype(self.data_type)
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(*self.mean_std),
                T.ConvertImageDtype(self.data_type)
            ])

    def __call__(self, input):
        output = self.transform(input)
        return output


class VitSegmentationTargetTransform():
    def __init__(self, train=True, resolution=None):
        self.train = train

    def __call__(self, input):
        output = torch.from_numpy(np.array(input, dtype=np.int32)).long()
        return output


class RandomSeedSegmentationDataset(Dataset):
    def __init__(self,
                 dataset,
                 joint_transform,
                 image_transform,
                 target_transform):

        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = self.base_seed
        self.dataset = dataset
        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + 100 * epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        img, mask = self.dataset[idx]

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        img, mask = self.joint_transform(img, mask)
        img = self.image_transform(img)
        mask = self.target_transform(mask)

        return img, mask


def build_cityscapes_train_valid_datasets(data_path, image_size):
    args = get_args()
    args.num_classes = Cityscapes.num_classes
    args.ignore_index = Cityscapes.ignore_index
    args.color_table = Cityscapes.color_table
    args.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_joint_transform = \
        VitSegmentationJointTransform(train=True, resolution=image_size)
    val_joint_transform = \
        VitSegmentationJointTransform(train=False, resolution=image_size)
    train_image_transform = \
        VitSegmentationImageTransform(train=True, resolution=image_size)
    val_image_transform = \
        VitSegmentationImageTransform(train=False, resolution=image_size)
    train_target_transform = \
        VitSegmentationTargetTransform(train=True, resolution=image_size)
    val_target_transform = \
        VitSegmentationTargetTransform(train=False, resolution=image_size)

    # training dataset
    train_data = Cityscapes(
        root=data_path[0],
        split='train',
        mode='fine',
        resolution=image_size
    )
    train_data = RandomSeedSegmentationDataset(
        train_data,
        joint_transform=train_joint_transform,
        image_transform=train_image_transform,
        target_transform=train_target_transform)

    # validation dataset
    val_data = Cityscapes(
        root=data_path[0],
        split='val',
        mode='fine',
        resolution=image_size
    )

    val_data = RandomSeedSegmentationDataset(
        val_data,
        joint_transform=val_joint_transform,
        image_transform=val_image_transform,
        target_transform=val_target_transform)

    return train_data, val_data


def build_train_valid_datasets(data_path, image_size):
    return build_cityscapes_train_valid_datasets(data_path, image_size)
