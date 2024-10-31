import argparse
import json

from evaluate_mmmu import get_input_output_paths


def merge_input_files(input_path):
    """Merge input files to a format compatible with the evaluator."""
    input_file_paths, output_file_path = get_input_output_paths(input_path, task="OCRBench")

    results = []

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)
                results.append(res)

    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file)

    return output_file_path


def compute_ocrbench_score(result_file):
    """Compute OCRBench score."""
    merged_results = json.load(open(result_file))

    # OCRBench score calculation is adopted from https://github.com/Yuliang-Liu/MultimodalOCR/blob/1b7713f44c91f30f64efb6d3e494c416861ef15f/example.py#L1
    # MIT License. Copyright (c) 2023 Yuliang Liu
    score = {
        "Regular Text Recognition": 0,
        "Irregular Text Recognition": 0,
        "Artistic Text Recognition": 0,
        "Handwriting Recognition": 0,
        "Digit String Recognition": 0,
        "Non-Semantic Text Recognition": 0,
        "Scene Text-centric VQA": 0,
        "Doc-oriented VQA": 0,
        "Doc-oriented VQA": 0,
        "Key Information Extraction": 0,
        "Handwritten Mathematical Expression Recognition": 0,
    }

    for res in merged_results:
        predict = res["answer"]
        answers = res["gt_answer"]

        dataset_name = res["dataset_name"]
        ocr_type = res["data_type"]

        if dataset_name == "HME100k":
            if isinstance(answers, list):
                for j in range(len(answers)):
                    answer = answers[j].strip().replace("\n", " ").replace(" ", "")
                    predict = predict.strip().replace("\n", " ").replace(" ", "")
                    if answer in predict:
                        score[ocr_type] += 1
            else:
                answers = answers.strip().replace("\n", " ").replace(" ", "")
                predict = predict.strip().replace("\n", " ").replace(" ", "")
                if answers in predict:
                    score[ocr_type] += 1
        else:
            if isinstance(answers, list):
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace("\n", " ")
                    predict = predict.lower().strip().replace("\n", " ")
                    if answer in predict:
                        score[ocr_type] += 1
            else:
                answers = answers.lower().strip().replace("\n", " ")
                predict = predict.lower().strip().replace("\n", " ")
                if answers in predict:
                    score[ocr_type] += 1

    recognition_score = (
        score['Regular Text Recognition']
        + score['Irregular Text Recognition']
        + score['Artistic Text Recognition']
        + score['Handwriting Recognition']
        + score['Digit String Recognition']
        + score['Non-Semantic Text Recognition']
    )
    final_score = (
        recognition_score
        + score['Scene Text-centric VQA']
        + score['Doc-oriented VQA']
        + score['Key Information Extraction']
        + score['Handwritten Mathematical Expression Recognition']
    )
    result_log = f"""###########################OCRBench##############################
Text Recognition(Total 300): {recognition_score}
------------------Details of Recognition Score-------------------
Regular Text Recognition(Total 50): {score['Regular Text Recognition']}
Irregular Text Recognition(Total 50): {score['Irregular Text Recognition']}
Artistic Text Recognition(Total 50): {score['Artistic Text Recognition']}
Handwriting Recognition(Total 50): {score['Handwriting Recognition']}
Digit String Recognition(Total 50): {score['Digit String Recognition']}
Non-Semantic Text Recognition(Total 50): {score['Non-Semantic Text Recognition']}
----------------------------------------------------------------
Scene Text-centric VQA(Total 200): {score['Scene Text-centric VQA']}
----------------------------------------------------------------
Doc-oriented VQA(Total 200): {score['Doc-oriented VQA']}
----------------------------------------------------------------
Key Information Extraction(Total 200): {score['Key Information Extraction']}
----------------------------------------------------------------
Handwritten Mathematical Expression Recognition(Total 100): {score['Handwritten Mathematical Expression Recognition']}
----------------------Final Score-------------------------------
Final Score(Total 1000): {final_score}"""

    return result_log, final_score


def ocrbench_eval(input_path):
    """Run OCRBench evaluation."""
    result_file_path = merge_input_files(input_path)
    result_log, score = compute_ocrbench_score(result_file_path)
    return result_log, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help="Path to input file(s)")
    args = parser.parse_args()

    result_log, _ = ocrbench_eval(args.input_path)

    print(result_log)
