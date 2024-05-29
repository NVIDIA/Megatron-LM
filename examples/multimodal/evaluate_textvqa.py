import argparse
import glob
import json
import re

# This can help resolve an import error of an mmf dependency that is not needed.
try:
    from mmf.utils.m4c_evaluators import TextVQAAccuracyEvaluator
except ModuleNotFoundError:
    from mmf.utils.m4c_evaluators import TextVQAAccuracyEvaluator


def merge_input_files(input_path):
    """Merge input files to a format compatible with the evaluator."""
    output_file_path = input_path + "-TextVQA-merged.json"

    pattern = input_path + "-TextVQA-[0-9].*jsonl"
    input_file_paths = glob.glob(pattern)

    results = []

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)
                results.append(res)

    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file)

    return output_file_path


# Note: This is based on https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/eval/eval_textvqa.py#L17
# and slightly modified.
def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif "Reference OCR token: " in prompt and len(prompt.split("\n")) == 3:
        if prompt.startswith("Reference OCR token:"):
            question = prompt.split("\n")[1]
        else:
            question = prompt.split("\n")[0]
    elif len(prompt.split("\n")) == 2:
        question = prompt.split("\n")[0]
    else:
        raise RuntimeError("unexpected prompt format")

    return question.lower()


# Note: This is based on https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/eval/eval_textvqa.py#L35
# and slightly modified.
def evaluate(result_file_path, groundtruth_path):
    with open(groundtruth_path) as groundtruth_file:
        groundtruth = json.load(groundtruth_file)["data"]

    groundtruth = {(gt["image_id"], gt["question"].lower()): gt["answers"] for gt in groundtruth}

    with open(result_file_path, "r") as result_file:
        results = json.load(result_file)

    predictions = []
    for result in results:
        gt_answers = groundtruth[(result["sample_id"], prompt_processor(result["prompt"]))]
        predictions.append({"pred_answer": result["text"], "gt_answers": gt_answers})

    evaluator = TextVQAAccuracyEvaluator()
    print(
        'Samples: {}\nAccuracy: {:.2f}%\n'.format(
            len(predictions), 100.0 * evaluator.eval_pred_list(predictions)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help="Path to input file(s)")
    parser.add_argument('--groundtruth-path', type=str, help="Path to groundtruth file")
    args = parser.parse_args()

    result_file_path = merge_input_files(args.input_path)

    evaluate(result_file_path, args.groundtruth_path)
