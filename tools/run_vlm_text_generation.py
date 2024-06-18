# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Generate text using a vision language model."""
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from functools import partial

# Add megatron to the path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToPILImage

from megatron.inference.text_generation.api import generate_and_post_process
from megatron.inference.text_generation.forward_step import ForwardStep
from megatron.training import get_args, get_model, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from pretrain_vlm import model_provider


def add_text_generation_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='Vision language model text generation')

    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0, help='Top k sampling.')
    group.add_argument(
        "--out-seq-length", type=int, default=1024, help='Size of the output generated text.'
    )
    group.add_argument("--output-path", type=str, required=True, help='Output file path')
    group.add_argument('--input-path', type=str, required=True, help="Input directory")
    group.add_argument(
        '--num-partitions', type=int, default=0, help="Number of partitions for inputs."
    )
    group.add_argument('--partition-id', type=int, default=0, help="Partition index")
    group.add_argument("--drop-vision-class-token", action="store_true", default=False)
    group.add_argument("--gt-path", type=str, help="Optional ground truth file")

    return parser


def preprocess_image(target_h, target_w, img):
    """Example image preprocessing. Resizes input image to target size.

    Args:
        target_h (int): Target height in pixels.
        target_w (int): Target width in pixels
        img (np.array [h, w, c]): Input image in a numpy array.

    Returns:
        output_img (torch.Tensor [c, h, w]): Input image resized to target size.
    """
    # Imagenet's mean and std for normalization.
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)

    # Resize image considering ratio between input and target image sizes.
    img_h, img_w = img.shape[0], img.shape[1]
    ratio = float(max(target_h, target_w)) / max(img_h, img_w)

    scaled_h, scaled_w = int(img_h * ratio + 0.5), int(img_w * ratio + 0.5)

    image_transform = Compose(
        [ToPILImage(), Resize((scaled_h, scaled_w)), lambda x: x.convert("RGB")]
    )
    img = image_transform(img)

    # Normalize pixel values.
    img = (torch.Tensor(np.array(img)).permute(2, 0, 1) - pixel_mean) / pixel_std

    # Pad to target size.
    delta_h, delta_w = target_h - scaled_h, target_w - scaled_w
    output_img = torch.nn.functional.pad(img, (0, delta_w, 0, delta_h))

    return output_img


def generate_samples(model):
    """Text generation using a trained vision language model. This is an example for the COCO dataset."""
    args = get_args()

    image_files = sorted(glob.glob(args.input_path + "/*"))
    # Optionally, process only a subset of the input files.
    if args.num_partitions > 0:
        per_part = len(image_files) // args.num_partitions
        image_files = image_files[per_part * args.partition_id : per_part * (args.partition_id + 1)]

    num_samples = len(image_files)
    images = []

    # Run image preprocessing.
    for image_file in image_files:
        img = np.array(Image.open(image_file))
        img = preprocess_image(args.img_h, args.img_w, img)

        images.append(img.reshape(-1, 3, args.img_h, args.img_w))

    # Load optional ground truth.
    gt_image_id_to_captions = defaultdict(list)
    if args.gt_path:
        gts = json.load(open(args.gt_path))
        for gt in gts["annotations"]:
            gt_image_id_to_captions[gt["image_id"]].append(gt['caption'])

    idx = 0
    while True:
        image = images[idx].cuda()
        image_id = int(image_files[idx].split("_")[-1].split(".")[0])

        forward_step = partial(VLMForwardStep, image)

        if torch.distributed.get_rank() == 0:
            prompt = "Give a short and clear explanation of the subsequent image.\n"

            resp_sentences, _, _, _ = generate_and_post_process(
                model,
                forward_step=forward_step,
                prompts=[prompt],
                tokens_to_generate=args.out_seq_length,
                return_output_log_probs=False,
                top_k_sampling=args.top_k,
                top_p_sampling=args.top_p,
                add_BOS=False,
                temperature=args.temperature,
                random_seed=123,
            )

            for prompt, generation in zip([prompt], resp_sentences):
                output = {
                    "question_id": image_id,
                    "prompt": prompt,
                    "caption": generation[len(prompt) :],
                }

                output["ground_truth"] = gt_image_id_to_captions[image_id]

                print_rank_0(output)

                yield output
                idx += 1
                if idx >= num_samples:
                    break
        else:
            generate_and_post_process(model, forward_step=forward_step)

            idx += 1
            if idx >= num_samples:
                break


def generate_and_write_samples(model):
    args = get_args()

    for output in generate_samples(model):
        if torch.distributed.get_rank() == 0:
            with open(args.output_path, 'a') as f:
                f.write(json.dumps(output) + "\n")


class VLMForwardStep(ForwardStep):
    def __init__(self, images, model, max_batch_size, max_sequence_length):
        super().__init__(model, max_batch_size, max_sequence_length)
        self._images = images

    def _forward(self, tokens, position_ids, attention_mask):
        return self.model(
            self._images,
            tokens,
            position_ids,
            attention_mask,
            inference_params=self.inference_params,
        )

    def __call__(self, tokens, position_ids, attention_mask):
        logits = super().__call__(tokens, position_ids, attention_mask)

        # On the first inference iteration, we compute image tokens.
        # Update the sequence length offset by the number of image tokens.
        num_tokens = tokens.size(1)
        if num_tokens > 1:
            self.inference_params.sequence_len_offset += self.inference_params.key_value_memory_dict[
                "image_tokens_count"
            ]

        return logits


def main():
    """Vision language model text generation."""

    logging.getLogger(__name__).warning("Models using pipeline parallelism are not supported yet.")

    initialize_megatron(extra_args_provider=add_text_generation_args)

    # Set up model and load checkpoint.
    model = get_model(model_provider, wrap_with_ddp=False)

    args = get_args()
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    model = model[0]
    model.eval()

    generate_and_write_samples(model)


if __name__ == "__main__":
    main()
