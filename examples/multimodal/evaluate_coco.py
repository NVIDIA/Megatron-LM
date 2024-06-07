# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import argparse
import glob
import json

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO


def convert_to_coco_format(input_path):
    """Convert input files to COCO compatible format."""
    output_file_path = input_path + "-captioning-merged.json"

    pattern = input_path + "-captioning-[0-9].*jsonl"
    input_file_paths = glob.glob(pattern)

    captions = []

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)

                question_id = res['sample_id']
                caption = res['caption'].rstrip('.').lower()

                captions.append({"image_id": question_id, "caption": caption})

    with open(output_file_path, "w") as output_file:
        json.dump(captions, output_file)

    return output_file_path


def coco_captioning_eval(input_path, groundtruth_file):
    """Run COCO captioning evaluation."""
    coco = COCO(groundtruth_file)
    input_file = convert_to_coco_format(input_path)
    coco_result = coco.loadRes(input_file)

    coco_eval = COCOEvalCap(coco, coco_result)

    # Evaluate on the input subset of images.
    coco_eval.params['image_id'] = coco_result.getImgIds()

    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(metric, score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to input file(s)")
    parser.add_argument(
        "--groundtruth-path", type=str, required=True, help="Path to groundtruth file"
    )
    args = parser.parse_args()

    coco_captioning_eval(args.input_path, args.groundtruth_path)
