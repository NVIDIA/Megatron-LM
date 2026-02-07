import argparse
import json

from .evaluate_mmmu import get_input_output_paths


def merge_input_files(input_path):
    """Merge input files to a format compatible with the evaluator."""
    input_file_paths, output_file_path = get_input_output_paths(input_path, task="MVBench")

    results = []
    collected = set()


    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)
                res["question_id"] = "{}-{}".format(res['task_type'], res['sample_id'])
                if res['sample_id'] in collected:
                    continue
                collected.add(res['sample_id'])

                results.append(res)

    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file, indent=4, sort_keys=True)

    return output_file_path


# The following code is adapted from
# https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/mvbench.ipynb
# which is licensed under the MIT license. More details on the license can be
# found at https://github.com/OpenGVLab/Ask-Anything/tree/main?tab=MIT-1-ov-file#readme
def check_ans(pred, gt):
    flag = False
        
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
    
    return flag

def create_result_dict(result_list):

    correct = 0
    total = 0
    res_list = []
    acc_dict = {}

    for idx, result_obj in enumerate(result_list):
        task_type = result_obj['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0]  # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        pred = result_obj['answer']
        gt = result_obj['gt_answer'][0]
        
        res_list.append({
            'pred': pred,
            'gt': gt
        })
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1

    print(f"Total Acc: {correct / total * 100 :.2f}%")
    print('-' * 30, task_type, '-' * 30)

    return acc_dict
        

def combine_all_res(acc_dict):
    final_res = dict()
    correct = 0
    total = 0
    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
        correct += v[0]
        total += v[1]    
    final_res['total-acc'] = correct / total * 100

    print(final_res)

    return final_res


def mvbench_eval(input_path):
    result_file_path = merge_input_files(input_path)
    
    merged_results = json.load(open(result_file_path))
    acc_dict = create_result_dict(merged_results)
    
    return combine_all_res(acc_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help="Path to input file(s)")
    args = parser.parse_args()

    avg_acc_dict = mvbench_eval(args.input_path)

    print(f"MVBench {avg_acc_dict}")


    
