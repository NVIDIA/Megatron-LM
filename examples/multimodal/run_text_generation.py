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

from megatron.core import parallel_state
from megatron.core.models.multimodal.llava_model import IMAGE_TOKEN
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
    group.add_argument("--config-path", type=str, help="Evaluation config file to use.")

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


class VQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_image_path,
        gt_path,
        num_samples_per_partition,
        num_partitions,
        partition_id,
        keys,
        img_h,
        img_w,
        use_tiling,
        max_num_tiles,
        use_thumbnail,
    ):
        samples = json.load(open(gt_path, encoding='utf-8'))
        if "data" in samples:
            samples = samples["data"]

        # Optionally, process only a subset of the input files.
        if num_partitions > 0:
            lb, ub = _get_partition_bounds(
                len(samples), num_samples_per_partition, num_partitions, partition_id
            )
            samples = samples[lb:ub]

        self._keys = keys
        self._samples = samples
        self._input_image_path = input_image_path
        self._img_h = img_h
        self._img_w = img_w
        self._use_tiling = use_tiling
        self._max_num_tiles = max_num_tiles
        self._use_thumbnail = use_thumbnail

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        sample = self._samples[idx]

        img_file = "{}/{}".format(self._input_image_path, sample[self._keys["image_id"]])
        if not os.path.exists(img_file):
            img_file += ".jpg"

            if not os.path.exists(img_file):
                img_file = img_file.replace('.jpg', '.png')

        img = Image.open(img_file)
        imgs = get_visual_transform(
            img,
            self._img_h,
            self._img_w,
            self._use_tiling,
            self._max_num_tiles,
            self._use_thumbnail,
            augment=False,
        )
        tile_count = torch.tensor([len(imgs)], dtype=torch.int)

        sample_id = idx
        if "sample_id" in self._keys:
            sample_id = sample[self._keys["sample_id"]]

        metadata = ""  # Not used.

        return (
            torch.stack(imgs),
            tile_count,
            sample_id,
            sample[self._keys["question"]],
            sample[self._keys["answer"]],
            metadata,
        )


class CaptioningDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_image_path,
        gt_path,
        num_samples_per_partition,
        num_partitions,
        partition_id,
        img_h,
        img_w,
        use_tiling,
        max_num_tiles,
        use_thumbnail,
    ):
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

        self._image_files = image_files
        self._answers = answers
        self._img_h = img_h
        self._img_w = img_w
        self._use_tiling = use_tiling
        self._max_num_tiles = max_num_tiles
        self._use_thumbnail = use_thumbnail

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, idx):
        img_file = self._image_files[idx]
        image_id = int(img_file.split("_")[-1].split(".")[0])

        img = Image.open(img_file)
        imgs = get_visual_transform(
            img,
            self._img_h,
            self._img_w,
            self._use_tiling,
            self._max_num_tiles,
            self._use_thumbnail,
            augment=False,
        )

        tile_count = torch.tensor([len(imgs)], dtype=torch.int)

        question = ""  # Fixed for all samples.
        metadata = ""  # Not used.

        return torch.stack(imgs), tile_count, image_id, question, self._answers[image_id], metadata


class MMMUDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_image_path,
        num_samples_per_partition,
        num_partitions,
        partition_id,
        img_h,
        img_w,
        use_tiling,
        max_num_tiles,
        use_thumbnail,
        single_image,
    ):
        # The following downloads the MMMU dataset from HuggingFace and uses the API from the MMMU github repo to run MMMU evaluation.
        all_mmmu_datasets = []

        hf_datasets_cache = os.environ["HF_DATASETS_CACHE"]
        assert hf_datasets_cache != "", "Please set the environment variable HF_DATASETS_CACHE."

        for subject in CAT_SHORT2LONG.values():
            # Use a local copy of the dataset if exists (can be faster) or the HF one.
            if os.path.exists(input_image_path):
                subject_dataset = datasets.load_dataset(
                    os.path.join(input_image_path, subject),
                    split=datasets.Split.VALIDATION,
                    cache_dir=hf_datasets_cache,
                    verification_mode="no_checks",
                )
            else:
                subject_dataset = datasets.load_dataset(
                    "MMMU/MMMU",
                    subject,
                    split=datasets.Split.VALIDATION,
                    cache_dir=hf_datasets_cache,
                )

            all_mmmu_datasets.append(subject_dataset)

        dataset = datasets.concatenate_datasets(all_mmmu_datasets)

        dataset = [s for s in dataset if s['id'].startswith("val")]

        # Optionally, process only a subset of the input files.
        if num_partitions > 0:
            lb, ub = _get_partition_bounds(
                len(dataset), num_samples_per_partition, num_partitions, partition_id
            )
            dataset = dataset[lb:ub]

        # Using the LLaVA config from the MMMU repo.
        config = load_yaml("examples/multimodal/MMMU/mmmu/configs/llava1.5.yaml")
        for k, v in config.items():
            if isinstance(v, list):
                assert len(v) == 1, "only one value supported."
                config[k] = v[0]

        self._config = config

        self._dataset = dataset

        self._img_h = img_h
        self._img_w = img_w
        self._use_tiling = use_tiling
        self._max_num_tiles = max_num_tiles
        self._use_thumbnail = use_thumbnail
        self._single_image = single_image

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample = self._dataset[idx]

        # Use the single image approach from the MMMU repo.
        if self._single_image:
            sample = process_single_sample(sample)
            sample = construct_prompt(sample, self._config)

            img = sample["image"]
            sample_imgs = get_visual_transform(
                img,
                self._img_h,
                self._img_w,
                self._use_tiling,
                self._max_num_tiles,
                self._use_thumbnail,
                augment=False,
            )
            sample_num_tiles = [len(sample_imgs)]
        else:
            sample = construct_prompt(sample, self._config)

            sample_imgs = []
            sample_num_tiles = []

            img_indices = re.findall(r"<image (\d+)", sample["final_input_prompt"])
            # If there are multiple input images, we need to avoid the number of image embeddings getting too large.
            adjusted_max_num_tiles = max(1, self._max_num_tiles // len(img_indices))

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
                    self._img_h,
                    self._img_w,
                    self._use_tiling,
                    adjusted_max_num_tiles,
                    self._use_thumbnail,
                    augment=False,
                )  # List of tiles.

                sample_imgs.extend(imgs)
                sample_num_tiles.append(len(imgs))

            # Sanity check.
            for i in range(1, 8):
                assert (
                    f"<image {i}>" not in sample["final_input_prompt"]
                ), "prompt contains unhandled image tags"

        # MMMU specific metadata.
        metadata = {"question_type": sample["question_type"]}
        if sample["question_type"] == "multiple-choice":
            metadata["index2ans"] = sample["index2ans"]
            metadata["all_choices"] = sample["all_choices"]

        prompt = sample['final_input_prompt']
        if self._single_image:
            for i in range(8):
                prompt = prompt.replace(f"<image {i}>", "")
            prompt = f"<image>\n{prompt}"

        tile_count = torch.tensor(sample_num_tiles, dtype=torch.int)

        return (
            torch.stack(sample_imgs),
            tile_count,
            sample["id"],
            prompt,
            sample["answer"],
            metadata,
        )


class VideoMMMEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_image_path,
        gt_path,
        num_samples_per_partition,
        num_partitions,
        partition_id,
        img_h,
        img_w,
        use_tiling,
        max_num_tiles,
        use_thumbnail,
        num_frames,
    ):
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

        self._ground_truth = ground_truth
        self._img_h = img_h
        self._img_w = img_w
        self._use_tiling = use_tiling
        self._max_num_tiles = max_num_tiles
        self._use_thumbnail = use_thumbnail
        self._num_frames = num_frames

    def __len__(self):
        return len(self._ground_truth)

    def __getitem__(self, idx):
        gt = self._ground_truth[idx]

        video, _, _ = read_video(gt["video_path"], start_pts=0, end_pts=None, pts_unit='sec')
        video = video.numpy()
        selected_frames = torch.linspace(0, video.shape[0] - 1, self._num_frames).long()
        video_frames = video[selected_frames]
        if self._num_frames == 1:
            video_frames = video_frames[None]

        imgs = list(
            itertools.chain.from_iterable(
                get_visual_transform(
                    img,
                    self._img_h,
                    self._img_w,
                    self._use_tiling,
                    self._max_num_tiles,
                    self._use_thumbnail,
                    augment=False,
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

        num_tiles = torch.tensor([len(imgs)], dtype=torch.int)

        answer = ""
        metadata = ""

        return (
            torch.stack(imgs),
            num_tiles,
            question["question_id"],
            question_dict,
            answer,
            metadata,
        )


def get_evaluation_dataloader(
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
    num_workers,
):
    """Build evaluation dataset."""
    if task == "TextVQA":
        keys = {
            "image_id": "image_id",
            "sample_id": "question_id",
            "question": "question",
            "answer": "answers",
        }

        dataset = VQADataset(
            input_image_path,
            gt_path,
            num_samples_per_partition,
            num_partitions,
            partition_id,
            keys,
            img_h,
            img_w,
            use_tiling,
            max_num_tiles,
            use_thumbnail,
        )
    elif task == "VQAv2":
        keys = {
            "image_id": "image",
            "sample_id": "question_id",
            "question": "question",
            "answer": "answer",
        }

        dataset = VQADataset(
            input_image_path,
            gt_path,
            num_samples_per_partition,
            num_partitions,
            partition_id,
            keys,
            img_h,
            img_w,
            use_tiling,
            max_num_tiles,
            use_thumbnail,
        )
    elif task == "ChartQA":
        keys = {"image_id": "imgname", "question": "query", "answer": "label"}

        dataset = VQADataset(
            input_image_path,
            gt_path,
            num_samples_per_partition,
            num_partitions,
            partition_id,
            keys,
            img_h,
            img_w,
            use_tiling,
            max_num_tiles,
            use_thumbnail,
        )
    elif task == "captioning":
        dataset = CaptioningDataset(
            input_image_path,
            gt_path,
            num_samples_per_partition,
            num_partitions,
            partition_id,
            img_h,
            img_w,
            use_tiling,
            max_num_tiles,
            use_thumbnail,
        )
    elif task == 'MMMU':
        # Note: single_image=True uses only one image like in the MMMU repo example.
        # single_image=False uses all images in the sample.
        dataset = MMMUDataset(
            input_image_path,
            num_samples_per_partition,
            num_partitions,
            partition_id,
            img_h,
            img_w,
            use_tiling,
            max_num_tiles,
            use_thumbnail,
            single_image=True,
        )
    elif task == "VideoMME":
        dataset = VideoMMMEDataset(
            input_image_path,
            gt_path,
            num_samples_per_partition,
            num_partitions,
            partition_id,
            img_h,
            img_w,
            use_tiling,
            max_num_tiles,
            use_thumbnail,
            num_frames,
        )
    else:
        raise NotImplementedError(f"unsupported task {task}")

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()

    sampler = torch.utils.data.DistributedSampler(
        dataset, shuffle=False, num_replicas=dp_world_size, rank=dp_rank
    )
    # TODO: Batched inference is not supported yet.
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=num_workers, sampler=sampler, pin_memory=True
    )

    return dataloader


def generate_samples(model, config: EvaluationConfig, print_output):
    """Text generation using a trained vision language model."""
    args = get_args()

    dataloader = get_evaluation_dataloader(
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
        args.num_workers,
    )

    num_img_embeddings_per_tile = get_num_image_embeddings(
        args.img_h, args.img_w, args.patch_dim, args.vision_model_type, args.disable_vision_class_token, 1
    )

    for idx, (imgs, num_tiles, sample_id, question, answers, metadata) in enumerate(dataloader):
        imgs = imgs.to("cuda")
        num_tiles = num_tiles.to("cuda")

        conv = get_conversation(config.task, question)

        forward_step = partial(VLMForwardStep, num_img_embeddings_per_tile, imgs, num_tiles)

        if is_first_rank():
            resp_sentences, _, _, _ = generate_and_post_process(
                model,
                forward_step=forward_step,
                prompts=[conv],
                tokens_to_generate=config.out_seq_length,
                top_k_sampling=config.top_k,
                top_p_sampling=config.top_p,
                add_BOS=False,
                temperature=config.temperature,
                random_seed=args.seed,
                detokenize_segments=False,
                data_parallel=True,
            )

            for generation in resp_sentences:
                if isinstance(sample_id, torch.Tensor):
                    sample_id = sample_id.item()

                output = {"sample_id": sample_id}

                output_name = ""
                if config.task == "captioning":
                    output_name = "caption"
                elif config.task in ("TextVQA", "VQAv2", "ChartQA"):
                    output_name = "answer"
                elif config.task in ("MMMU"):
                    output_name = "text"
                elif config.task == "VideoMME":
                    output_name = "response"
                    output = question

                prompt, generated = get_prompt_and_generated(
                    generation, args.tokenizer_prompt_format
                )
                if config.task == "VideoMME":
                    output["questions"][0][output_name] = generated
                else:
                    output[output_name] = generated
                    output["prompt"] = prompt

                if config.task == "captioning":
                    output["ground_truth"] = answers
                elif config.task in ("TextVQA", "VQAv2"):
                    output["gt_answer"] = [ans for ans in answers]
                elif config.task == "ChartQA":
                    output["gt_answer"] = [answers]
                elif config.task == "MMMU":
                    prediction = generated
                    if metadata["question_type"] == "multiple-choice":
                        prediction = parse_multi_choice_response(
                            generated, metadata["all_choices"], metadata["index2ans"]
                        )

                    output["prediction"] = prediction

                if print_output:
                    print(output)

                yield output
                idx += 1
        else:
            generate_and_post_process(
                model, forward_step=forward_step, detokenize_segments=False, data_parallel=True
            )

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
        )

    # Default output path if not defined...
    if not config.output_path:
        os.makedirs("generated", exist_ok=True)
        config.output_path = "generated/" + args.language_model_type

    return config


def is_first_rank():
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        and parallel_state.get_tensor_model_parallel_rank() == 0
    )


def get_output_path(config, dp_rank):
    return (
        f"{config.output_path}-{config.task}-dprank={dp_rank}-partition={config.partition_id}.jsonl"
    )


def generate_and_write_samples(model, config, print_output=True):
    """Generate text and write to an output file."""
    dp_rank = parallel_state.get_data_parallel_rank()

    if is_first_rank():
        output_path = get_output_path(config, dp_rank)
        output_file = open(output_path, "w")
        print(f"output path: {output_file.name}")

    with torch.no_grad():
        for output in generate_samples(model, config, print_output):
            if is_first_rank():
                output_file.write(json.dumps(output) + "\n")
                output_file.flush()

    if is_first_rank():
        output_file.close()


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
        num_image_tokens = (tokens == self.model.module.image_token_index).sum().item()
        num_tokens = tokens.size(1)
        if num_tokens > 1 and num_image_tokens > 0:
            self.inference_params.sequence_len_offset += (
                self.inference_params.key_value_memory_dict["image_tokens_count"] - num_image_tokens
            )

        return logits


def get_conversation(task, question):
    conversation = []

    # In all cases, the tokenizer adds possible header tokens for the assistant.
    if task == "captioning":
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {
                "role": "user",
                "content": "<image>Provide a one-sentence caption for provided image.",
            },
        ]
    elif task in ("TextVQA", "VQAv2", "ChartQA"):
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {
                "role": "user",
                "content": f"<image>\n{question}\nAnswer the question using a single word or phrase.",
            },
        ]
    elif task == "MMMU":
        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": question},
        ]
    elif task == "VideoMME":
        q = (
            "Select the best answer to the following multiple-choice "
            "question based on the video. Respond with only the letter "
            "(A, B, C, or D) of the correct option.\n"
        )
        q += question["questions"][0]["question"] + "\n"
        q += question["questions"][0]["choices"][0] + "\n"
        q += question["questions"][0]["choices"][1] + "\n"
        q += question["questions"][0]["choices"][2] + "\n"
        q += question["questions"][0]["choices"][3] + "\n"

        conversation = [
            {"role": "system", "content": "Answer the questions."},
            {"role": "user", "content": f"<image>\n{question}"},
        ]

    return conversation


def get_prompt_and_generated(prompt_and_generation, prompt_format):
    """Strip prompt and other unnecessary text from generation."""
    if prompt_format == "llama3":
        splitted = prompt_and_generation.split("<|start_header_id|>assistant<|end_header_id|>\n\n")
        prompt = splitted[0]
        generated = splitted[1]
        generated = generated.split("<|eot_id|>")[0]
    elif prompt_format == "mistral":
        splitted = prompt_and_generation.split("[/INST]")
        prompt = splitted[0]
        generated = splitted[1]
        generated = generated.split("</s>")[0]
    elif prompt_format == "chatml":
        splitted = prompt_and_generation.split("<|im_start|> assistant\n")
        prompt = splitted[0]
        generated = splitted[1]
        generated = generated.split("<|im_end|>")[0]

    # Remove possible garbage.
    generated = generated.strip()
    generated = generated.split("\n\n")[0]
    generated = generated.split("\n")[0]

    return prompt, generated


def main():
    """Vision language model text generation."""
    initialize_megatron(extra_args_provider=add_text_generation_args)

    if torch.distributed.get_rank() == 0:
        logging.getLogger(__name__).warning(
            "Models using pipeline parallelism are not supported yet."
        )

    args = get_args()

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
