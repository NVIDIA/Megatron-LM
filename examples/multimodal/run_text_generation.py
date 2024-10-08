# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Generate text using a vision language model."""
import glob
import itertools
import json
import logging
import os
import re
import sys
from collections import defaultdict
from functools import partial

# Add megatron to the path.
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

import datasets
import numpy as np
import torch
import yaml
from config import EvaluationConfig
from dataset_helpers import tokenizer_image_token
from image_processing import get_visual_transform
from MMMU.mmmu.utils.data_utils import (
    CAT_SHORT2LONG,
    construct_prompt,
    load_yaml,
    process_single_sample,
)
from MMMU.mmmu.utils.eval_utils import parse_multi_choice_response
from model import model_provider
from multimodal_args import add_multimodal_extra_args
from PIL import Image
from torchvision.io import read_video

from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN_INDEX
from megatron.core.models.vision.clip_vit_model import get_num_image_embeddings
from megatron.inference.text_generation.api import generate_and_post_process
from megatron.inference.text_generation.forward_step import ForwardStep
from megatron.training import get_args, get_model, get_tokenizer, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron


def add_text_generation_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='Vision language model text generation arguments')

    group.add_argument("--temperature", type=float, default=1.0, help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0, help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0, help='Top k sampling.')
    group.add_argument(
        "--out-seq-length", type=int, default=1024, help='Length of the output generated text.'
    )
    group.add_argument("--output-path", type=str, help='Output file path')
    group.add_argument('--input-image-path', type=str, help="Input image directory")
    group.add_argument(
        '--num-partitions', type=int, default=0, help="Number of partitions for inputs."
    )
    group.add_argument('--partition-id', type=int, default=0, help="Partition index")
    group.add_argument("--gt-path", type=str, help="Optional ground truth file")
    group.add_argument(
        "--task",
        type=str,
        choices=["captioning", "TextVQA", "VQAv2", "ChartQA", "MMMU", "VideoMME"],
        help="Generation task to run",
    )
    group.add_argument(
        "--num-samples-per-partition", type=int, default=0, help="Number of samples per partition"
    )
    group.add_argument(
        "--prompt-format",
        type=str,
        default="mistral",
        choices=["llama3", "mistral"],
        help="Prompting format to use",
    )
    group.add_argument("--config-path", type=str, help="Config file to use.")

    # Add common multimodal arguments needed for e.g. building the model.
    parser = add_multimodal_extra_args(parser)

    return parser


def _get_partition_bounds(
    total_num_samples, num_samples_per_partition, num_partitions, partition_id
):
    if num_samples_per_partition == 0:
        samples_per_partition = [
            int(x) for x in np.linspace(0, total_num_samples, num_partitions + 1)
        ]
        return samples_per_partition[partition_id], samples_per_partition[partition_id + 1]
    return num_samples_per_partition * partition_id, num_samples_per_partition * (partition_id + 1)


def get_evaluation_dataset(
    task,
    input_image_path,
    gt_path,
    img_h,
    img_w,
    use_tiling,
    max_num_tiles,
    use_thumbnail,
    num_samples_per_partition,
    num_partitions,
    partition_id,
    num_frames,
):
    """Build evaluation dataset."""
    images = []
    tile_counts = []
    questions, answers = [], []
    samples, sample_ids = [], []

    if task == "TextVQA":
        samples = json.load(open(gt_path, encoding='utf-8'))['data']

        # Optionally, process only a subset of the input files.
        if num_partitions > 0:
            lb, ub = _get_partition_bounds(
                len(samples), num_samples_per_partition, num_partitions, partition_id
            )
            samples = samples[lb:ub]

        for i in range(len(samples)):
            sample = samples[i]

            img_file = "{}/{}.jpg".format(input_image_path, sample["image_id"])
            if not os.path.exists(img_file):
                img_file = img_file.replace('.jpg', '.png')

            img = Image.open(img_file)
            imgs = get_visual_transform(
                img, img_h, img_w, use_tiling, max_num_tiles, use_thumbnail, augment=False
            )

            images.append(imgs)
            tile_counts.append(torch.tensor([len(imgs)], dtype=torch.int))

            questions.append(sample["question"])
            answers.append(sample["answers"])
            sample_ids.append(sample["question_id"])
    elif task == "VQAv2":
        samples = json.load(open(gt_path, encoding='utf-8'))

        # Optionally, process only a subset of the input files.
        if num_partitions > 0:
            lb, ub = _get_partition_bounds(
                len(samples), num_samples_per_partition, num_partitions, partition_id
            )
            samples = samples[lb:ub]

        for i in range(len(samples)):
            sample = samples[i]

            img_file = "{}/{}".format(input_image_path, sample["image"])

            img = Image.open(img_file)
            imgs = get_visual_transform(
                img, img_h, img_w, use_tiling, max_num_tiles, use_thumbnail, augment=False
            )

            images.append(imgs)
            tile_counts.append(torch.tensor([len(imgs)], dtype=torch.int))

            questions.append(sample["question"])
            answers.append(sample["answer"])
            sample_ids.append(sample["question_id"])
    elif task == "ChartQA":
        samples = json.load(open(gt_path, encoding='utf-8'))

        # Optionally, process only a subset of the input files.
        if num_partitions > 0:
            lb, ub = _get_partition_bounds(
                len(samples), num_samples_per_partition, num_partitions, partition_id
            )
            samples = samples[lb:ub]

        for i in range(len(samples)):
            sample = samples[i]

            img_file = "{}/{}".format(input_image_path, sample["imgname"])

            img = Image.open(img_file)
            imgs = get_visual_transform(
                img, img_h, img_w, use_tiling, max_num_tiles, use_thumbnail, augment=False
            )

            images.append(imgs)
            tile_counts.append(torch.tensor([len(imgs)], dtype=torch.int))

            questions.append(sample["query"])
            answers.append(sample["label"])
            sample_ids.append(i)
    elif task == "captioning":
        image_files = sorted(glob.glob(input_image_path + "/*"))
        # Optionally, process only a subset of the input files.
        if num_partitions > 0:
            lb, ub = _get_partition_bounds(
                len(image_files), num_samples_per_partition, num_partitions, partition_id
            )
            image_files = image_files[lb:ub]

        gts = json.load(open(gt_path))
        answers = defaultdict(list)
        for gt in gts["annotations"]:
            answers[gt["image_id"]].append(gt['caption'])

        # Run image preprocessing.
        for i in range(len(image_files)):
            image_file = image_files[i]
            img = Image.open(image_file)
            imgs = get_visual_transform(
                img, img_h, img_w, use_tiling, max_num_tiles, use_thumbnail, augment=False
            )

            images.append(imgs)
            tile_counts.append(torch.tensor([len(imgs)], dtype=torch.int))

            image_id = int(image_file.split("_")[-1].split(".")[0])
            sample_ids.append(image_id)
    elif task == 'MMMU':
        # The following downloads the MMMU dataset from HuggingFace and uses the API from the MMMU github repo to run MMMU evaluation.
        all_mmmu_datasets = []

        hf_datasets_cache = os.environ["HF_DATASETS_CACHE"]
        assert hf_datasets_cache != "", "Please set the environment variable HF_DATASETS_CACHE."

        for subject in CAT_SHORT2LONG.values():
            subject_dataset = datasets.load_dataset(
                "MMMU/MMMU", subject, split=datasets.Split.VALIDATION, cache_dir=hf_datasets_cache
            )
            all_mmmu_datasets.append(subject_dataset)

        dataset = datasets.concatenate_datasets(all_mmmu_datasets)

        dataset = [s for s in dataset if s['id'].startswith("val")]

        # Optionally, process only a subset of the input files.
        start_idx = 0
        end_idx = len(dataset)
        if num_partitions > 0:
            start_idx, end_idx = _get_partition_bounds(
                len(dataset), num_samples_per_partition, num_partitions, partition_id
            )

        end_idx = min(len(dataset), end_idx)

        # Using the LLaVA config from the MMMU repo.
        config = load_yaml("examples/multimodal/MMMU/mmmu/configs/llava1.5.yaml")
        for k, v in config.items():
            if isinstance(v, list):
                assert len(v) == 1, "only one value supported."
                config[k] = v[0]

        for idx in range(start_idx, end_idx):
            sample = dataset[idx]

            single_image = True
            # Use the single image approach from the MMMU repo.
            if single_image:
                sample = process_single_sample(sample)
                sample = construct_prompt(sample, config)

                img = sample["image"]
                sample_imgs = get_visual_transform(
                    img, img_h, img_w, use_tiling, max_num_tiles, use_thumbnail, augment=False
                )
                sample_num_tiles = [len(sample_imgs)]
            else:
                sample = construct_prompt(sample, config)

                sample_imgs = []
                sample_num_tiles = []

                img_indices = re.findall(r"<image (\d+)", sample["final_input_prompt"])
                # If there are multiple input images, we need to avoid the number of image embeddings getting too large.
                adjusted_max_num_tiles = max(1, max_num_tiles // len(img_indices))

                for img_idx in img_indices:
                    img_key = f"image_{img_idx}"
                    img_str = f"<image {img_idx}>"

                    img = sample[img_key]
                    assert img is not None, f"{img_str} is in prompt but not in sample images"

                    # Note: Only replace the current image tag.
                    sample["final_input_prompt"] = sample["final_input_prompt"].replace(
                        img_str, "<image>", 1
                    )

                    imgs = get_visual_transform(
                        img,
                        img_h,
                        img_w,
                        use_tiling,
                        adjusted_max_num_tiles,
                        use_thumbnail,
                        augment=False,
                    )  # List of tiles.

                    sample_imgs.extend(imgs)
                    sample_num_tiles.append(len(imgs))

                # Sanity check.
                for i in range(1, 8):
                    assert (
                        f"<image {i}>" not in sample["final_input_prompt"]
                    ), "prompt contains unhandled image tags"

            images.append(sample_imgs)
            tile_counts.append(torch.tensor(sample_num_tiles, dtype=torch.int))

            sample_ids.append(sample['id'])

            prompt = sample['final_input_prompt']
            if single_image:
                for i in range(8):
                    prompt = prompt.replace(f"<image {i}>", "")
                prompt = f"<image>\n{prompt}"
            questions.append(prompt)

            answers.append(sample['answer'])
            samples.append(sample)
    elif task == "VideoMME":
        ground_truth_original = json.load(open(gt_path))
        ground_truth = []
        for gt in ground_truth_original:
            video_path = gt["url"]
            video_path = video_path.replace("https://www.youtube.com/watch?v=", "")
            video_path = video_path.replace("https://m.youtube.com/watch?v=", "")
            video_path = os.path.join(input_image_path, video_path + ".mp4")
            if not os.path.exists(video_path):
                continue
            gt["video_path"] = video_path
            ground_truth.append(gt)

        ground_truth = sorted(ground_truth, key=lambda gt: gt["video_path"])
        print_rank_0(f"Found {len(ground_truth)} videos to process.")

        if num_partitions > 0:
            start_idx, end_idx = _get_partition_bounds(
                len(ground_truth), num_samples_per_partition, num_partitions, partition_id
            )
            ground_truth = ground_truth[start_idx:end_idx]

        # Run image preprocessing.
        for idx, gt in enumerate(ground_truth):
            print_rank_0(f"Processing input video: {idx} / {len(ground_truth)}")
            video, _, _ = read_video(gt["video_path"], start_pts=0, end_pts=None, pts_unit='sec')
            video = video.numpy()
            selected_frames = torch.linspace(0, video.shape[0] - 1, num_frames).long()
            video_frames = video[selected_frames]
            if num_frames == 1:
                video_frames = video_frames[None]

            imgs = list(
                itertools.chain.from_iterable(
                    get_visual_transform(
                        img, img_h, img_w, use_tiling, max_num_tiles, use_thumbnail, augment=False
                    )
                    for img in video_frames
                )
            )

            for question in gt["questions"]:
                # Very hacky, but we essentially re-create gt holding only the
                # question of interest. This is the make this generation script
                # compatible with the Video MME evaluation script.
                question_dict = {
                    "video_id": gt["video_id"],
                    "duration_category": gt["duration_category"],
                    "video_category": gt["video_category"],
                    "video_subcategory": gt["video_subcategory"],
                    "url": gt["url"],
                    "questions": [question],
                }
                images.append(imgs)
                tile_counts.append(torch.tensor([len(imgs)], dtype=torch.int))
                questions.append(question_dict)
                sample_ids.append(question["question_id"])
    else:
        raise NotImplementedError("unsupported task")

    return images, tile_counts, samples, sample_ids, questions, answers


def generate_samples(model, config: EvaluationConfig):
    """Text generation using a trained vision language model."""
    args = get_args()
    images, tile_counts, samples, sample_ids, questions, answers = get_evaluation_dataset(
        config.task,
        config.input_image_path,
        config.gt_path,
        args.img_h,
        args.img_w,
        args.use_tiling,
        args.max_num_tiles,
        args.use_thumbnail,
        config.num_samples_per_partition,
        config.num_partitions,
        config.partition_id,
        args.num_frames,
    )

    num_img_embeddings_per_tile = get_num_image_embeddings(
        args.img_h, args.img_w, args.patch_dim, args.disable_vision_class_token, 1
    )
    num_samples = len(sample_ids)
    idx = 0
    while idx < num_samples:
        imgs = torch.stack(images[idx]).cuda()
        num_tiles = tile_counts[idx].cuda()
        sample_id = sample_ids[idx]

        prompt = get_prompt(config.task, questions, idx, config.prompt_format)

        forward_step = partial(VLMForwardStep, num_img_embeddings_per_tile, imgs, num_tiles)

        if torch.distributed.get_rank() == 0:
            resp_sentences, _, _, _ = generate_and_post_process(
                model,
                forward_step=forward_step,
                prompts=[prompt],
                tokens_to_generate=config.out_seq_length,
                top_k_sampling=config.top_k,
                top_p_sampling=config.top_p,
                add_BOS=False,
                temperature=config.temperature,
                random_seed=args.seed,
                detokenize_segments=False,
            )

            for prompt, generation in zip([prompt], resp_sentences):
                output = {"sample_id": sample_id, "prompt": prompt}

                output_name = ""
                if config.task == "captioning":
                    output_name = "caption"
                elif config.task in ("TextVQA", "VQAv2", "ChartQA"):
                    output_name = "answer"
                elif config.task in ("MMMU"):
                    output_name = "text"
                elif config.task == "VideoMME":
                    output_name = "response"
                    output = questions[idx]

                generated = get_generated(generation, args.prompt_format)
                if config.task == "VideoMME":
                    output["questions"][0][output_name] = generated
                else:
                    output[output_name] = generated

                if config.task == "captioning":
                    output["ground_truth"] = answers[sample_id]
                elif config.task in ("TextVQA", "VQAv2"):
                    output["gt_answer"] = [ans for ans in answers[idx]]
                elif config.task == "ChartQA":
                    output["gt_answer"] = [answers[idx]]
                elif config.task == "MMMU":
                    sample = samples[idx]

                    prediction = generated
                    if sample["question_type"] == "multiple-choice":
                        prediction = parse_multi_choice_response(
                            generated, sample["all_choices"], sample["index2ans"]
                        )

                    output["prediction"] = prediction

                print_rank_0(output)

                yield output
                idx += 1
        else:
            generate_and_post_process(model, forward_step=forward_step, detokenize_segments=False)

            idx += 1


def get_evaluation_config():
    """Get evaluation config from a config file or command-line arguments."""
    args = get_args()
    if args.config_path:
        with open(args.config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        config = EvaluationConfig(**config_dict)
    else:
        config = EvaluationConfig(
            task=args.task,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            out_seq_length=args.out_seq_length,
            output_path=args.output_path,
            input_image_path=args.input_image_path,
            gt_path=args.gt_path,
            num_partitions=args.num_partitions,
            partition_id=args.partition_id,
            num_samples_per_partition=args.num_samples_per_partition,
            prompt_format=args.prompt_format,
        )

    # Default output path if not defined...
    if not config.output_path:
        os.makedirs("generated", exist_ok=True)
        config.output_path = "generated/" + args.language_model_type

    return config


def generate_and_write_samples(model, config):
    """Generate text and write to an output file."""
    for output in generate_samples(model, config):
        if torch.distributed.get_rank() == 0:
            with open(config.output_path, 'a') as f:
                f.write(json.dumps(output) + "\n")


class VLMForwardStep(ForwardStep):
    """Inference forward step for a multimodal model."""

    def __init__(
        self,
        num_img_embeddings_per_tile,
        images,
        num_tiles,
        model,
        max_batch_size,
        max_sequence_length,
    ):
        """Create multimodal forward step."""
        total_num_tiles = torch.sum(num_tiles).item()
        num_img_embeddings = num_img_embeddings_per_tile * total_num_tiles

        super().__init__(model, max_batch_size, max_sequence_length + num_img_embeddings)
        self._images = images
        self._num_tiles = num_tiles

    def _forward(self, tokens, position_ids, attention_mask):
        return self.model(
            self._images,
            tokens,
            position_ids,
            attention_mask=None,
            inference_params=self.inference_params,
            num_image_tiles=self._num_tiles,
            runtime_gather_output=True,
        )

    def __call__(self, tokens, position_ids, attention_mask):
        logits = super().__call__(tokens, position_ids, attention_mask)

        # On the first inference iteration, we compute image tokens.
        # Update the sequence length offset by the number of image tokens.
        num_image_tokens = (tokens == -200).sum().item()
        num_tokens = tokens.size(1)
        if num_tokens > 1 and num_image_tokens > 0:
            self.inference_params.sequence_len_offset += (
                self.inference_params.key_value_memory_dict["image_tokens_count"] - num_image_tokens
            )

        return logits


def get_prompt(task, questions, idx, prompt_format):
    """Get a prompt for the evaluation task."""
    if task == "captioning":
        if prompt_format == "llama3":
            prompt = "<|start_header_id|>system<|end_header_id|>\n\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\nProvide a one-sentence caption for provided image.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif prompt_format == "mistral":
            prompt = (
                "[INST] <image>Give a short and clear explanation of the subsequent image. [/INST]"
            )
    elif task == "TextVQA":
        question = questions[idx]

        if prompt_format == "llama3":
            prompt = "<|start_header_id|>system<|end_header_id|>\n\nAnswer the questions.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\n{}\nAnswer the question using a single word or phrase.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".format(
                question
            )
        elif prompt_format == "mistral":
            prompt = "[INST] <image>\n{}\nAnswer the question using a single word or phrase. [/INST]".format(
                question
            )
    elif task == "VQAv2":
        question = questions[idx]

        if prompt_format == "llama3":
            prompt = "<|start_header_id|>system<|end_header_id|>\n\nAnswer the questions.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\n{}\nAnswer the question using a single word or phrase.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".format(
                question
            )
        elif prompt_format == "mistral":
            prompt = "[INST] <image>\n{}\nAnswer the question using a single word or phrase. [/INST]".format(
                question
            )
    elif task == "ChartQA":
        question = questions[idx]

        if prompt_format == "llama3":
            prompt = "<|start_header_id|>system<|end_header_id|>\n\nAnswer the questions.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n<image>\n{}\nAnswer the question using a single word or phrase.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".format(
                questions
            )
        elif prompt_format == "mistral":
            prompt = "[INST] <image>\n{}\nAnswer the question using a single word or phrase. [/INST]".format(
                question
            )
    elif task == "MMMU":
        question = questions[idx]

        if prompt_format == "llama3":
            prompt = "<|start_header_id|>system<|end_header_id|>\n\nAnswer the questions.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            prompt = prompt.format(question)
        elif prompt_format == "mistral":
            prompt = "[INST] {} [/INST]".format(question)
    elif task == "VideoMME":
        question = (
            "Select the best answer to the following multiple-choice "
            "question based on the video. Respond with only the letter "
            "(A, B, C, or D) of the correct option.\n"
        )
        question += questions[idx]["questions"][0]["question"] + "\n"
        question += questions[idx]["questions"][0]["choices"][0] + "\n"
        question += questions[idx]["questions"][0]["choices"][1] + "\n"
        question += questions[idx]["questions"][0]["choices"][2] + "\n"
        question += questions[idx]["questions"][0]["choices"][3] + "\n"

        if prompt_format == "llama3":
            prompt = "<|start_header_id|>system<|end_header_id|>\n\nAnswer the questions.<|eot_id|>{}<|start_header_id|>user<|end_header_id|>\n\n<image>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            prompt = prompt.format("", question)
        elif prompt_format == "mistral":
            prompt = "<image>\n{}".format(question)

    return prompt


def get_generated(prompt_and_generation, prompt_format):
    """Strip prompt and other unnecessary text from generation."""
    if prompt_format == "llama3":
        generated = prompt_and_generation.split(
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )[-1]
        generated = generated.split("<|eot_id|>")[0]
    elif prompt_format == "mistral":
        generated = prompt_and_generation.split("[/INST]")[-1]
        generated = generated.split("</s>")[0]

    generated = generated.strip()
    generated = generated.split("\n\n")[0]
    generated = generated.split("\n")[0]

    return generated


def patch_tokenizer(args):
    """Patch tokenizer with image token support."""

    def _decorate_tokenize(f):
        # When tokenizing, replace <image> with the image token index (-200)
        def wrapper(prompt):
            tokens = tokenizer_image_token(args, prompt, f)

            return tokens

        return wrapper

    def _decorate_detokenize(f):
        # When detokenizing, skip image token index.
        def wrapper(tokens):
            tokens = np.array(tokens)
            tokens = tokens[tokens != IMAGE_TOKEN_INDEX]
            tokens = tokens.tolist()

            return f(tokens)

        return wrapper

    tokenizer = get_tokenizer()
    tokenizer.tokenize = _decorate_tokenize(tokenizer.tokenize)
    tokenizer.detokenize = _decorate_detokenize(tokenizer.detokenize)


def main():
    """Vision language model text generation."""
    logging.getLogger(__name__).warning("Models using pipeline parallelism are not supported yet.")

    initialize_megatron(extra_args_provider=add_text_generation_args)

    args = get_args()

    patch_tokenizer(args)  # Make the tokenizer support image tokens.

    def wrapped_model_provider(pre_process, post_process):
        return model_provider(pre_process, post_process, parallel_output=False)

    # Set up model and load checkpoint.
    model = get_model(wrapped_model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    model = model[0]

    model.eval()

    config = get_evaluation_config()

    generate_and_write_samples(model, config)


if __name__ == "__main__":
    main()
