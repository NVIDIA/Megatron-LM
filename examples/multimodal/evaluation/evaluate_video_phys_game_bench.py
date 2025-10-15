import argparse
import json

from .evaluate_mmmu import get_input_output_paths


def merge_input_files(input_path):
    """Merge input files to a format compatible with the evaluator."""
    input_file_paths, output_file_path = get_input_output_paths(input_path, task="PhysGameBench")

    results = []
    collected = set()

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)
                res["question_id"] = res["sample_id"]
                if res['sample_id'] in collected:
                    continue
                collected.add(res['sample_id'])

                results.append(res)

    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file, indent=4, sort_keys=True)

    return output_file_path


# The following function is adapted from
# https://github.com/PhysGame/PhysGame/blob/main/physvlm/test/PhysGame_bench/utils.py#L101
# which is licensed under the Apache 2.0 license. More details on the license can be
# found at https://github.com/PhysGame/PhysGame/tree/main?tab=Apache-2.0-1-ov-file#readme
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

def compute_all_acc(result_list):
    correct, total = 0, 0
    subclass_cnt = {}
    for res in result_list:
        total += 1
        pred = res['answer']
        gt = res['gt_answer'][0]
        subclass = res['subclass']
        if gt.lower().replace(".", "") == pred.lower().replace(".", ""):
            correct += 1
            if subclass not in subclass_cnt.keys():
                subclass_cnt.update({subclass: [1, 1]})
            else:
                subclass_cnt[subclass][0] += 1
                subclass_cnt[subclass][1] += 1
        else:
            if subclass not in subclass_cnt.keys():
                subclass_cnt.update({subclass: [0, 1]})
            else:
                subclass_cnt[subclass][1] += 1
    
    result_acc_dict = {
        "Physgame-Total-Acc": correct / total * 100
    }
    print (f'Physgame-Total-Acc: {correct / total * 100 :.2f}%', )
    for sub_i in subclass_cnt.keys():
        print(f'Physgame-{sub_i}-Acc: {subclass_cnt[sub_i][0] / subclass_cnt[sub_i][1] * 100 :.2f}%')
        result_acc_dict[f'Physgame-{sub_i}-Acc'] = subclass_cnt[sub_i][0] / subclass_cnt[sub_i][1] * 100
    
    return result_acc_dict
        
def phys_game_bench_eval(input_path):
    result_file_path = merge_input_files(input_path)
    
    merged_results = json.load(open(result_file_path))
    
    return compute_all_acc(merged_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help="Path to input file(s)")
    args = parser.parse_args()

    avg_acc = phys_game_bench_eval(args.input_path)

    print(f"PhysGameBench accuracy: {avg_acc:.2f}")
