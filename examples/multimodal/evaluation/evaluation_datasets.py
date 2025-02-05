# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Evaluation datasets."""
import glob
import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from image_processing import get_visual_transform
from PIL import Image

from megatron.training import print_rank_0


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
    """VQA evaluation dataset."""

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
        vision_model_type,
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
        self._vision_model_type = vision_model_type

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
            vision_model_type=self._vision_model_type,
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
    """Captioning evaluation dataset."""

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
        vision_model_type,
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
        self._vision_model_type = vision_model_type

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
            vision_model_type=self._vision_model_type,
        )

        tile_count = torch.tensor([len(imgs)], dtype=torch.int)

        question = ""  # Fixed for all samples.
        metadata = ""  # Not used.

        return torch.stack(imgs), tile_count, image_id, question, self._answers[image_id], metadata


class MMMUDataset(torch.utils.data.Dataset):
    """MMMU evaluation dataset."""

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
        prompt_style,
        vision_model_type,
    ):
        import datasets
        from MMMU.mmmu.utils.data_utils import CAT_SHORT2LONG, load_yaml

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
        self._prompt_style = prompt_style
        self._vision_model_type = vision_model_type

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        from MMMU.mmmu.utils.data_utils import construct_prompt, process_single_sample

        sample = self._dataset[idx]

        # Use the single image approach from the MMMU repo.
        if self._prompt_style == "single_image":
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
                vision_model_type=self._vision_model_type,
            )
            sample_num_tiles = [len(sample_imgs)]

            prompt = sample["final_input_prompt"]
            for i in range(8):
                prompt = prompt.replace(f"<image {i}>", "")
            sample["final_input_prompt"] = f"<image>\n{prompt}"
        elif self._prompt_style == "vlmevalkit":
            sample = construct_prompt(sample, self._config)

            if sample["question_type"] == "multiple-choice":
                question = sample["question"]

                options = ""
                for k, v in sample["index2ans"].items():
                    options += f"{k}. {v}\n"

                final_prompt = f"{question}\n"
                if "hint" in sample:
                    final_prompt += f"Hint: {sample['hint']}\n"

                if "task_instructions" in sample:
                    final_prompt += f"Task instructions: {sample['task_instructions']}\n"

                final_prompt += options
                final_prompt += "Answer with the option's letter from the given choices directly."

                sample["final_input_prompt"] = final_prompt.rstrip()
            else:
                question = sample["question"]
                final_prompt = f"{question}\n"
                final_prompt += "Answer the question directly."
                sample["final_input_prompt"] = final_prompt.rstrip()

            sample_imgs = []
            sample_num_tiles = []

            img_indices = sorted(list(set(re.findall(r"<image (\d+)", sample["final_input_prompt"]))))
            # If there are multiple input images, we need to avoid the number of image embeddings getting too large.
            adjusted_max_num_tiles = max(1, self._max_num_tiles // len(img_indices))
            adjusted_max_num_tiles = min(adjusted_max_num_tiles, self._max_num_tiles)

            for img_idx in img_indices:
                img_key = f"image_{img_idx}"
                img_str = f"<image {img_idx}>"

                img = sample[img_key]
                assert img is not None, f"{img_str} is in prompt but not in sample images"

                imgs = get_visual_transform(
                    img,
                    self._img_h,
                    self._img_w,
                    self._use_tiling,
                    adjusted_max_num_tiles,
                    self._use_thumbnail,
                    augment=False,
                    vision_model_type=self._vision_model_type,
                )  # List of tiles.

                sample_imgs.extend(imgs)
                sample_num_tiles.append(len(imgs))

            sample["final_input_prompt"] = " ".join([f'<image {i + 1}><image>' for i in range(len(img_indices))]) + "\n" + sample["final_input_prompt"]
        elif self._prompt_style == "multi_image":
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
                    vision_model_type=self._vision_model_type,
                )  # List of tiles.

                sample_imgs.extend(imgs)
                sample_num_tiles.append(len(imgs))

            # Sanity check.
            for i in range(1, 8):
                assert (
                    f"<image {i}>" not in sample["final_input_prompt"]
                ), "prompt contains unhandled image tags"
        else:
            raise ValueError(f"unknown prompt style {self._prompt_style}")

        # MMMU specific metadata.
        metadata = {"question_type": sample["question_type"]}
        if sample["question_type"] == "multiple-choice":
            metadata["index2ans"] = sample["index2ans"]
            metadata["all_choices"] = sample["all_choices"]

        prompt = sample['final_input_prompt']

        tile_count = torch.tensor(sample_num_tiles, dtype=torch.int)

        return (
            torch.stack(sample_imgs),
            tile_count,
            sample["id"],
            prompt,
            sample["answer"],
            metadata,
        )


class VideoMMEDataset(torch.utils.data.Dataset):
    "Video MME evaluation dataset."

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
        vision_model_type,
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
        self._use_tiling = False
        self._max_num_tiles = max_num_tiles
        self._use_thumbnail = use_thumbnail
        self._num_frames = num_frames
        self._vision_model_type = vision_model_type

    def __len__(self):
        return len(self._ground_truth)

    def __getitem__(self, idx):
        from torchvision.io import read_video

        gt = self._ground_truth[idx]

        video, _, _ = read_video(gt["video_path"], start_pts=0, end_pts=None, pts_unit='sec')
        video = video.numpy()
        selected_frames = torch.linspace(0, video.shape[0] - 1, self._num_frames).long()
        video_frames = video[selected_frames]
        if self._num_frames == 1:
            video_frames = video_frames[None]

        imgs = []
        for img in video_frames:
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            img = to_pil(img)
            imgs += get_visual_transform(
                img, self._img_h, self._img_w, self._use_tiling, self._max_num_tiles,
                self._use_thumbnail, augment=False, vision_model_type=self._vision_model_type
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


class OCRBenchDataset(torch.utils.data.Dataset):
    """OCRBench evaluation dataset."""

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
        vision_model_type,
    ):
        gt = json.load(open(gt_path, encoding='utf-8'))

        if num_partitions > 0:
            start_idx, end_idx = _get_partition_bounds(
                len(gt), num_samples_per_partition, num_partitions, partition_id
            )
            gt = gt[start_idx:end_idx]

        self._input_image_path = input_image_path
        self._gt = gt
        self._img_h = img_h
        self._img_w = img_w
        self._use_tiling = use_tiling
        self._max_num_tiles = max_num_tiles
        self._use_thumbnail = use_thumbnail
        self._vision_model_type = vision_model_type

    def __len__(self):
        return len(self._gt)

    def __getitem__(self, idx):
        img_path = os.path.join(self._input_image_path, self._gt[idx]['image_path'])

        img = Image.open(img_path)
        imgs = get_visual_transform(
            img,
            self._img_h,
            self._img_w,
            self._use_tiling,
            self._max_num_tiles,
            self._use_thumbnail,
            augment=False,
            vision_model_type=self._vision_model_type,
        )

        tile_count = torch.tensor([len(imgs)], dtype=torch.int)

        metadata = {
            "dataset_name": self._gt[idx]["dataset_name"],
            "data_type": self._gt[idx]["type"],
        }

        return (
            torch.stack(imgs),
            tile_count,
            idx,
            self._gt[idx]["question"],
            self._gt[idx]["answers"],
            metadata,
        )


class MathVistaDataset(torch.utils.data.Dataset):
    """MathVista evaluation dataset."""

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
        vision_model_type,
    ):
        import datasets

        hf_datasets_cache = os.environ["HF_DATASETS_CACHE"]
        assert hf_datasets_cache != "", "Please set the environment variable HF_DATASETS_CACHE."

        if os.path.exists(input_image_path):
            dataset = datasets.load_dataset(
                input_image_path, cache_dir=hf_datasets_cache, verification_mode="no_checks"
            )
        else:
            dataset = datasets.load_dataset(
                "AI4Math/MathVista", split="testmini", cache_dir=hf_datasets_cache
            )

        if num_partitions > 0:
            start_idx, end_idx = _get_partition_bounds(
                len(dataset), num_samples_per_partition, num_partitions, partition_id
            )
            dataset = dataset[start_idx:end_idx]

        self._dataset = dataset
        self._img_h = img_h
        self._img_w = img_w
        self._use_tiling = use_tiling
        self._max_num_tiles = max_num_tiles
        self._use_thumbnail = use_thumbnail
        self._vision_model_type = vision_model_type

    def __len__(self):
        return len(self._dataset["pid"])

    def __getitem__(self, idx):
        # Already a PIL object.
        img = self._dataset['decoded_image'][idx]

        imgs = get_visual_transform(
            img,
            self._img_h,
            self._img_w,
            self._use_tiling,
            self._max_num_tiles,
            self._use_thumbnail,
            augment=False,
            vision_model_type=self._vision_model_type,
        )

        tile_count = torch.tensor([len(imgs)], dtype=torch.int)

        question_id = self._dataset["pid"][idx]
        question = self._dataset["question"][idx]
        question_type = self._dataset["question_type"][idx]  # free_form or multi_choice
        query = self._dataset["query"][idx]
        choices = self._dataset["choices"][idx]
        answer = self._dataset["answer"][idx]

        if question_type == 'multi_choice':
            start_chr = 'A'
            choices_str = ''
            index2ans = {}
            all_choices = []
            for choice in choices:
                all_choices.append(start_chr)
                index2ans[start_chr] = choice
                choices_str += f"{start_chr}. {choice}\n"
                start_chr = chr(ord(start_chr) + 1)

            question = question + '\n' + choices_str
            question = question + "Answer with the option's letter from the given choices directly."
            answer = chr(ord('A') + choices.index(answer))
        else:
            question = query.replace("Hint: ", "")
            index2ans = {}
            all_choices = []

        metadata = {
            "question_type": question_type,
            "index2ans": index2ans,
            "all_choices": all_choices,
        }

        return torch.stack(imgs), tile_count, question_id, question, answer, metadata


class AI2DDataset(torch.utils.data.Dataset):
    """AI2D evaluation dataset."""

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
        no_mask,
        vision_model_type,
    ):
        with open(gt_path, 'r') as f:
            jsonl = list(f)

        gt = [json.loads(json_str) for json_str in jsonl]

        if num_partitions > 0:
            start_idx, end_idx = _get_partition_bounds(
                len(gt), num_samples_per_partition, num_partitions, partition_id
            )
            gt = gt[start_idx:end_idx]

        self._gt = gt
        self._input_image_path = input_image_path
        self._img_h = img_h
        self._img_w = img_w
        self._use_tiling = use_tiling
        self._max_num_tiles = max_num_tiles
        self._use_thumbnail = use_thumbnail
        self._no_mask = no_mask
        self._vision_model_type = vision_model_type

    def __len__(self):
        return len(self._gt)

    def __getitem__(self, idx):
        img_path = os.path.join(self._input_image_path, self._gt[idx]['image'])
        if self._no_mask:
            img_path.replace("AI2D_TEST", "AI2D_TEST_NO_MASK_IMAGES")

        img = Image.open(img_path)
        imgs = get_visual_transform(
            img,
            self._img_h,
            self._img_w,
            self._use_tiling,
            self._max_num_tiles,
            self._use_thumbnail,
            augment=False,
            vision_model_type=self._vision_model_type,
        )

        tile_count = torch.tensor([len(imgs)], dtype=torch.int)

        metadata = ""  # Not used.

        return (
            torch.stack(imgs),
            tile_count,
            self._gt[idx]["question_id"],
            self._gt[idx]["question"],
            self._gt[idx]["answer"],
            metadata,
        )


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
    vision_model_type,
):
    """Get an evaluation dataset."""
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
            vision_model_type,
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
            vision_model_type,
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
            vision_model_type,
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
            vision_model_type,
        )
    elif task == 'MMMU':
        # Note:
        # - prompt_style="single_image" uses only one image like in the MMMU repo example.
        # - prompt_style="multi_image" uses multiple input images.
        # - prompt_style="vlmevalkit" is similar to https://github.com/open-compass/VLMEvalKit/blob/5d3cebcf18ef4bfbadc3bd3ef80bdc7aad2c6557/vlmeval/vlm/internvl_chat.py#L499
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
            prompt_style="single_image",
            vision_model_type=vision_model_type,
        )
    elif task == "VideoMME":
        dataset = VideoMMEDataset(
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
            vision_model_type,
        )
    elif task == "OCRBench":
        dataset = OCRBenchDataset(
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
            vision_model_type,
        )
    elif task == "MathVista":
        dataset = MathVistaDataset(
            input_image_path,
            num_samples_per_partition,
            num_partitions,
            partition_id,
            img_h,
            img_w,
            use_tiling,
            max_num_tiles,
            use_thumbnail,
            vision_model_type,
        )
    elif task == "AI2D":
        dataset = AI2DDataset(
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
            no_mask=False,
            vision_model_type=vision_model_type,
        )
    else:
        raise NotImplementedError(f"unsupported task {task}")

    return dataset
