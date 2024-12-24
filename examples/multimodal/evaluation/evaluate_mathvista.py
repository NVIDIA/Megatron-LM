import argparse
import json
import re

from evaluate_mmmu import get_input_output_paths
from MMMU.mmmu.utils.eval_utils import parse_multi_choice_response
from open_flamingo.eval.vqa_metric import VQAEval


def merge_input_files(input_path):
    """Merge input files to a format compatible with the evaluator."""
    input_file_paths, output_file_path = get_input_output_paths(input_path, task="MathVista")

    results = dict()

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)
                sample_id = res["sample_id"]

                # Remove possible duplicates.
                if sample_id in results:
                    continue

                results[sample_id] = res

    results = list(results.values())

    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file)

    return output_file_path


def extra_processing(text):
    """Extra processing."""
    # Max decimal point capped to 2 decimal point
    regex = re.compile(r'^\d+\.\d+$')
    decimal = regex.findall(text)

    if len(decimal) > 0:
        non_decimal = len(decimal[0].split(".")[0])

        # if decimal values are all 0, trim them
        decimal_digits = [int(d) for d in decimal[0].split(".")[1]]
        if sum(decimal_digits) == 0:
            text = decimal[0][:non_decimal]
        else:
            text = decimal[0][: non_decimal + 3]

    # remove % and trailing .
    text = text.replace("%", "")
    if text[-1] == ".":
        text = text[:-1]

    return text


def extract_answer(text):
    """Extract answer."""
    alphabet = re.findall(r'[a-zA-Z]+', text)
    if len(alphabet) > 0 and "e+" not in text:
        template = re.findall(r'answer is -*\d+\.*\d*', text)
        if len(template) > 0:
            text = template[0]

            numbers = re.findall(r'-*\d+\.*\d*', text)
            text = numbers[0] if len(numbers) > 0 else text

    return text


def compute_mathvista_accuracy(result_file):
    """Compute MathVista accuracy."""
    merged_results = json.load(open(result_file))

    vqa = VQAEval(vqa=None, vqaRes=None)
    acc = 0
    for res in merged_results:
        pred_ans = res["answer"]
        if res["question_type"] == "multi_choice":
            pred_ans = parse_multi_choice_response(pred_ans, res["all_choices"], res["index2ans"])
        else:
            pred_ans = vqa.processPunctuation(pred_ans)
            pred_ans = vqa.processDigitArticle(pred_ans)
            # Extra processing and extraction.
            pred_ans = extra_processing(pred_ans)
            pred_ans = extract_answer(pred_ans)

        gt_ans = res["gt_answer"]
        if isinstance(gt_ans, list):
            assert len(gt_ans) == 1, f"Expected 1 groundtruth, got {gt_ans}"
            gt_ans = gt_ans[0]

        if res["question_type"] != "multi_choice":
            gt_ans = vqa.processPunctuation(gt_ans)
            gt_ans = vqa.processDigitArticle(gt_ans)

            gt_ans = extra_processing(gt_ans)

        if pred_ans == gt_ans:
            acc += 1
    acc = acc / len(merged_results) * 100
    return acc


def mathvista_eval(input_path):
    """Run MathVista evaluation."""
    result_file_path = merge_input_files(input_path)
    acc = compute_mathvista_accuracy(result_file_path)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help="Path to input file(s)")
    args = parser.parse_args()

    acc = mathvista_eval(args.input_path)

    print(f"===== MathVista accuracy: {acc} =====")
