import os
import statistics

def gather_numbers(fname, match_keywords, index_keywords, index_offsets):
    results = {}
    for k in index_keywords:
        results[k] = []
    file1 = open(fname, 'r')
    while True:
        line = file1.readline()
        if not line:
            break
        splits = line.split(' ')
        for i in range(len(match_keywords)):
            if match_keywords[i] in line:
                ref_idx = splits.index(index_keywords[i])
                results[index_keywords[i]].append(float(splits[ref_idx+index_offsets[i]]))
    file1.close()
    return results

def gather_GLUE_results(result_path, key, lr):
    result = []
    mnli_matched_result = []
    mnli_mismatched_result = []
    for file in os.listdir(result_path):
        if file.startswith(key) and lr in file:
            fname = f'{result_path}/{file}/output.log'
            if os.path.exists(fname):
                if key == "STS-B":
                    results = gather_numbers(fname, ['metrics for'], ['spearmanr'], [2])
                    overall_candidate = results['spearmanr']
                    overall_candidate = [x * 100.0 for x in overall_candidate]
                elif key == "CoLA":
                    results = gather_numbers(fname, ['metrics for'], ['mcc'], [2])
                    overall_candidate = results['mcc']
                    overall_candidate = [x * 100.0 for x in overall_candidate]
                elif key == "MNLI":
                    results = gather_numbers(fname,
                        ['overall:', 'metrics for dev-matched:', 'metrics for dev-mismatched:'],
                        ['overall:', 'dev-matched:', 'dev-mismatched:'],
                        [9, 9, 9])
                    overall_candidate = results['overall:']
                    matched_candidate = results['dev-matched:']
                    mismatched_candidate = results['dev-mismatched:']
                else:
                    results = gather_numbers(fname, ['overall:'], ['overall:'], [9])
                    overall_candidate = results['overall:']
                if len(overall_candidate) > 0:
                    if len(overall_candidate) != 3:
                        print(f"{result_path} task {key} lr {lr} only has {len(overall_candidate)} epoch")
                    best_index = overall_candidate.index(max(overall_candidate))
                    result.append(overall_candidate[best_index])
                    if key == "MNLI":
                        mnli_matched_result.append(matched_candidate[best_index])
                        mnli_mismatched_result.append(mismatched_candidate[best_index])
    if len(result) > 0:
        if len(result) != 5:
            print(f"{result_path} task {key} lr {lr} only has {len(result)} seed")
        if key == "MNLI":
            best_index = result.index(statistics.median_high(result))
            return round(mnli_matched_result[best_index],2), round(statistics.stdev(mnli_matched_result),2), round(mnli_mismatched_result[best_index],2), round(statistics.stdev(mnli_mismatched_result),2)
        else:
            return round(statistics.median_high(result),2), round(statistics.stdev(result),2)
    else:
        if key == "MNLI":
            return None, None, None, None
        else:
            return None, None

def gather_finetune_results(result_path, extra_col=[], lr="2e-5"):
    output = ""
    for field in extra_col:
        output += f"{field} &"
    task_output = ""
    median_list, std_list = [], []
    m_median, m_std, mm_median, mm_std = gather_GLUE_results(result_path, "MNLI", lr)
    if m_median is not None:
        median_list += [m_median, mm_median]
        std_list += [m_std, mm_std]
    task_output += f"{m_median}±{m_std} & {mm_median}±{mm_std} &"
    tasks = ["QQP", "QNLI", "SST-2", "CoLA", "STS-B", "MRPC", "RTE"]
    for task in tasks:
        t_median, t_std = gather_GLUE_results(result_path, task, lr)
        if t_median is not None:
            median_list += [t_median]
            std_list += [t_std]
        if task == "RTE":
            task_output += f"{t_median}±{t_std} "
        else:
            task_output += f"{t_median}±{t_std} &"
    overall_median = round(sum(median_list) / len(median_list), 2)
    overall_std = round(sum(std_list) / len(std_list), 2)
    output += f"{overall_median}±{overall_std} &"
    output += task_output
    output += " \\\\"
    print(output)

if __name__ == '__main__':
    print("\\begin{table}")
    print("\centering")
    print("\\tiny")
    text = "\\begin{tabular}{@{}l|"
    for _ in range(11):
        text += "c"
    text += "@{}}"
    print(text)
    print("\\toprule")
    print("Case & Train tokens & Average & MNLI-m & MNLI-mm & QQP & QNLI & SST-2 & CoLA & STS-B & MRPC & RTE \\\\")
    print("\midrule")
    
    result_path='/blob/users/conglli/project/bert_with_pile/checkpoint/bert-pile-0.336B-iters-2M-lr-1e-4-min-1e-5-wmup-10000-dcy-2M-sty-linear-gbs-1024-mbs-16-gpu-64-zero-0-mp-1-pp-1-nopp-finetune/'
    gather_finetune_results(result_path)
    
    print("\\bottomrule")
    print("\end{tabular}")
    print("\end{table}")
    print("")
    print("")