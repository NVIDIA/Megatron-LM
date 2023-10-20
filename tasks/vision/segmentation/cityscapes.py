# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016, 
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# code taken from 
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/cityscapes.py
# modified it to change max label index from 255 to 19 (num_classes)

import torch
import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
from torchvision.datasets.utils import extract_archive, verify_str_arg, iterable_to_str
from torchvision.datasets import VisionDataset
from PIL import Image
from megatron import print_rank_0


class Cityscapes(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    Examples:
        Get semantic segmentation target
        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')
            img, smnt = dataset[0]
        Get multiple targets
        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])
            img, (inst, col, poly) = dataset[0]
        Validate on the "coarse" set
        .. code-block:: python
            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')
            img, smnt = dataset[0]
    """
    num_classes = 19
    ignore_index = 19
    color_table = torch.tensor(
        [[128, 64, 128],
         [244, 35, 232],
         [70, 70, 70],
         [102, 102, 156],
         [190, 153, 153],
         [153, 153, 153],
         [250, 170, 30],
         [220, 220, 0],
         [107, 142, 35],
         [152, 251, 152],
         [70, 130, 180],
         [220, 20, 60],
         [255, 0, 0],
         [0, 0, 142],
         [0, 0, 70],
         [0, 60, 100],
         [0, 80, 100],
         [0, 0, 230],
         [119, 11, 32],
         [0, 0, 0]], dtype=torch.float, device='cuda')


    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 
        'category', 'category_id', 'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 19, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 19, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 19, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 19, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 19, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 19, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 19, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 19, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 19, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 19, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    # label2trainid
    label2trainid   = { label.id  : label.train_id for label in classes}

    def __init__(
            self,
            root: str,
            split: str = "train",
            mode: str = "fine",
            resolution: int = 1024,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(Cityscapes, self).__init__(root, transforms, transform, target_transform)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit_trainvaltest/leftImg8bit', split)
        self.targets_dir = os.path.join(self.root, 'gtFine_trainvaltest/gtFine', split)
        self.split = split
        self.resolution = resolution
        self.images = []
        self.targets = []

        for city in sorted(os.listdir(self.images_dir)):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_name = '{}_{}_labelIds.png'.format(file_name.split('_leftImg8bit')[0], self.mode)
                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(os.path.join(target_dir, target_name))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        
        target = Image.open(self.targets[index]) 
        target = np.array(target)

        target_copy = target.copy()
        for k, v in Cityscapes.label2trainid.items():
            binary_target = (target == k)
            target_copy[binary_target] = v
        target = target_copy

        target = Image.fromarray(target.astype(np.uint8))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        # len(self.images)
        return len(self.images)

