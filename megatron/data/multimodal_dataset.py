# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import torch

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage, RandomResizedCrop, Resize

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(img_h, img_w):
    return Compose([
        ToPILImage(),
        RandomResizedCrop((img_h, img_w), scale=(0.5, 1.0), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class MultiModalDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, indexed_dataset,
                 num_samples, seq_length, seed, img_h, img_w):

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.doc_idx = indexed_dataset.get_doc_idx()
        self.visual_transform = _transform(img_h, img_w)

    def __len__(self):
        return self.indexed_dataset.sizes.shape[0]

    def __getitem__(self, idx):
        text_sample, mode = self.indexed_dataset.get(self.doc_idx[idx])
        assert mode == 0
        img_sample, mode = self.indexed_dataset.get(self.doc_idx[idx]+1)
        assert mode == 1
        img_pad = img_sample[0].item()
        xs = img_sample[1:].tobytes(order='C')
        xs = xs[:len(xs)-img_pad]

        img_sample = np.array(Image.open(io.BytesIO(xs)))
        img_sample = self.visual_transform(img_sample).reshape(-1)

        return {'text': np.array(text_sample, dtype=np.int64),
                'img': np.array(img_sample, dtype=np.float32)}
