import json
import os
import math
from math import log10, floor
import copy

def mean(arr):
    return sum(arr) / len(arr)


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


def median(arr):
    return arr[len(arr) // 2]

metric_dict = {
    "hellaswag":"acc_norm",
    "lambada":"acc",
    "triviaqa":"acc",
    "webqs":"acc",
    "winogrande":"acc",
    "piqa":"acc_norm",
    "arc_challenge":"acc_norm",
    "arc_easy":"acc_norm",
    "openbookqa":"acc_norm",
    "race":"acc",
    "boolq":"acc",
    "cb":"acc",
    "copa":"acc",
    "rte":"acc",
    "wic":"acc",
    "wsc":"acc",
    "multirc":"acc",
    "record":"f1",
    "anli_r1":"acc",
    "anli_r2":"acc",
    "anli_r3":"acc",
    "wikitext":"word_perplexity",
    "logiqa":"acc_norm",
    "mathqa":"acc_norm",
    "mc_taco":"f1",
    "mrpc":"acc",
    "prost":"acc_norm",
    "pubmedqa":"acc",
    "qnli":"acc",
    "qqp":"acc",
    "sciq":"acc_norm",
    "sst":"acc",
    "wnli":"acc"
}

official_dict = {
    "hellaswag":["HellaSwag","acc"],
    "lambada":["LAMBADA","acc"],
    "triviaqa":["TriviaQA","acc"],
    "webqs":["WebQs","acc"],
    "winogrande":["Winogrande","acc"],
    "piqa":["PIQA","acc"],
    "arc_challenge":["ARC Challenge","acc"],
    "arc_easy":["ARC Easy","acc"],
    "openbookqa":["OpenBookQA","acc"],
    "race":["RACE-h","acc"],
    "boolq":["BoolQ","acc"],
    "cb":["CB","acc"],
    "copa":["Copa","acc"],
    "rte":["RTE","acc"],
    "wic":["WiC","acc"],
    "wsc":["WSC","acc"],
    "multirc":["MultiRC","acc"],
    "record":["ReCoRD","f1"],
    "anli_r1":["ANLI R1","acc"],
    "anli_r2":["ANLI R2","acc"],
    "anli_r3":["ANLI R3","acc"],
    "wikitext":["WikiText-2","ppl"],
    "logiqa":["LogiQA","acc"],
    "mathqa":["MathQA","acc"],
    "mc_taco":["MC-TACO","f1"],
    "mrpc":["MRPC","acc"],
    "prost":["PROST","acc"],
    "pubmedqa":["PubMedQA","acc"],
    "qnli":["QNLI","acc"],
    "qqp":["QQP","acc"],
    "sciq":["SciQ","acc"],
    "sst":["SST-2","acc"],
    "wnli":["WNLI","acc"]
}

# When comparing with gpt3 paper, the most trustful tasks are the hellaswag to
# anli_r3, who have >= 1000 samples (less variation), and have <= 43% data
# contamination in the paper.
gpt3paper_zeroshoteval = {
    "hellaswag":[33.7,43.6,51.0,54.7,62.8,67.4,70.9,78.9],
    "lambada":[42.7,54.3,60.4,63.6,67.1,70.3,72.5,76.2],
    "triviaqa":[4.15,7.61,14.0,19.7,31.3,38.7,41.8,64.3],
    "webqs":[1.77,3.20,4.33,4.63,7.92,7.73,8.22,14.4],
    "winogrande":[52.0,52.1,57.4,58.7,62.3,64.5,67.9,70.2],
    "piqa":[64.6,70.2,72.9,75.1,75.6,78.0,78.5,81.0],
    "arc_challenge":[26.6,29.5,31.8,35.5,38.0,41.4,43.7,51.4],
    "arc_easy":[43.6,46.5,53.0,53.8,58.2,60.2,63.8,68.8],
    "anli_r1":[33.4,34.2,33.4,33.4,34.2,32.3,33.2,34.6],
    "anli_r2":[33.2,31.9,33.3,33.3,33.8,33.5,33.5,35.4],
    "anli_r3":[33.6,34.0,33.8,33.4,35.3,34.8,34.4,34.5],
    "openbookqa":[35.6,43.2,45.2,46.8,53.0,50.4,55.6,57.6],
    "race":[35.2,37.9,40.1,40.9,42.4,44.1,44.6,45.5],
    "boolq":[49.7,60.3,58.9,62.4,67.1,65.4,66.2,60.5],
    "cb":[0.00,32.1,8.93,19.6,19.6,28.6,19.6,46.4],
    "copa":[66.0,68.0,73.0,77.0,76.0,80.0,84.0,91.0],
    "rte":[47.7,49.8,48.4,56.0,46.6,55.2,62.8,63.5],
    "wic":[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
    "wsc":[59.6,56.7,65.4,61.5,66.3,60.6,64.4,65.4],
    "multirc":[4.72,9.65,12.3,13.6,14.3,18.4,24.2,27.6],
    "record":[71.9,79.2,82.8,85.2,87.3,89.5,90.4,91.0]
}

gpt3paper_fewshoteval = {
    "hellaswag":[33.5,43.1,51.3,54.9,62.9,67.3,71.3,79.3],
    "lambada":[22.0,40.4,63.2,57.0,78.1,79.1,81.3,86.4],
    "triviaqa":[6.96,16.3,26.5,32.1,42.3,51.6,57.5,71.2],
    "webqs":[5.46,12.6,15.9,19.6,24.8,27.7,33.5,41.5],
    "winogrande":[51.3,52.6,57.5,59.1,62.6,67.4,70.0,77.7],
    "piqa":[64.3,69.4,72.0,74.3,75.4,77.8,79.9,82.3],
    "arc_challenge":[25.5,28.4,32.3,36.7,39.5,43.7,44.8,51.5],
    "arc_easy":[42.7,51.0,58.1,59.1,62.1,65.8,69.1,70.1],
    "anli_r1":[32.1,32.5,30.9,32.5,33.5,33.1,33.3,36.8],
    "anli_r2":[35.7,33.8,32.1,31.4,32.6,33.3,32.6,34.0],
    "anli_r3":[35.0,34.4,35.1,36.0,32.7,33.9,34.5,40.2],
    "openbookqa":[37.0,43.6,48.0,50.6,55.6,55.2,60.8,65.4],
    "race":[34.3,37.0,40.4,41.4,42.3,44.7,45.1,46.8],
    "boolq":[43.1,60.6,62.0,64.1,70.3,70.0,70.2,77.5],
    "cb":[42.9,58.9,53.6,69.6,67.9,60.7,66.1,82.1],
    "copa":[67.0,64.0,72.0,77.0,83.0,83.0,86.0,92.0],
    "rte":[52.3,48.4,46.9,50.9,56.3,49.5,60.6,72.9],
    "wic":[49.8,55.0,53.0,53.0,51.6,53.1,51.1,55.3],
    "wsc":[58.7,60.6,54.8,49.0,62.5,67.3,75.0,75.0],
    "multirc":[6.09,11.8,16.8,20.8,24.7,23.8,25.0,32.5],
    "record":[70.7,77.9,82.1,84.0,87.5,88.8,89.8,90.1]
}

gpt3paper_zeroshoteval_index = {
    "125M":0, # Small
    "350M":1, # Medium
    "760M":2, # Large
    "1.3B":3, # XL
    "2.7B":4,
    "6.7B":5,
    "13B":6,
    "175B":7
}

def round_sig(x, sig=3):
    if x == 0:
        return 0
    return round(x, sig-int(floor(log10(abs(x))))-1)

def generate_result_table(tab_header, configs, task_order, caption, avg_range,
    avg_tag, avg_only=False, fontsize="\\footnotesize", find_best=False,
    candidate_range=None, candidate_task=None, split_name_by_space=False,
    print_stderr=False, few_shot=False):
    # Gather results
    result_list = []
    for i in range(len(configs)):
        result_dict = {}
        eval_path = configs[i][-1]
        if "paper" in configs[i][0]:
            assert eval_path is None
        if eval_path is None:
            assert "paper" in configs[i][0]
            assert configs[i][1] in gpt3paper_zeroshoteval_index, "the second element has to be the model size"
            paper_result_idx = gpt3paper_zeroshoteval_index[configs[i][1]]
            if few_shot:
                for task in gpt3paper_fewshoteval:
                    result_dict[task] = [gpt3paper_fewshoteval[task][paper_result_idx]]
            else:
                for task in gpt3paper_zeroshoteval:
                    result_dict[task] = [gpt3paper_zeroshoteval[task][paper_result_idx]]
        else:
            for file in os.listdir(eval_path):
                if file.endswith(".json"):
                    result = json.load(open(eval_path+"/"+file, "r"))
                    for task in result['results']:
                        if task != "wikitext":
                            result_dict[task] = [100.0*result['results'][task][metric_dict[task]]]
                        else:
                            result_dict[task] = [result['results'][task][metric_dict[task]]]
        result_list.append(result_dict)
    avg_list = []
    for i in range(len(configs)):
        average_results = []
        for j in range(len(avg_range)):
            results = []
            for k in range(avg_range[j]+1):
                if task_order[k] in result_list[i]:
                    results.append(result_list[i][task_order[k]][0])
            if len(results) > 0:
                average_results.append(float(sum(results))/len(results))
            else:
                average_results.append(0)
        avg_list.append(average_results)

    if find_best:
        best_avg_value = [0 for _ in range(len(avg_range))]
        best_avg_idx = [0 for _ in range(len(avg_range))]
        best_task_value = [0 for _ in range(len(candidate_task))]
        best_task_idx = [0 for _ in range(len(candidate_task))]
        for i in range(candidate_range, len(configs)):
            for j in range(len(avg_range)):
                if avg_list[i][j] > best_avg_value[j]:
                    best_avg_value[j] = avg_list[i][j]
                    best_avg_idx[j] = i
            for j in range(len(candidate_task)):
                if result_list[i][candidate_task[j]] > best_task_value[j]:
                    best_task_value[j] = result_list[i][candidate_task[j]]
                    best_task_idx[j] = i
        # reorder configs, result_list, avg_list to only keep the best cases
        new_configs = configs[:candidate_range]
        new_result_list = result_list[:candidate_range]
        new_avg_list = avg_list[:candidate_range]
        for i in range(len(avg_range)):
            selected_config = copy.deepcopy(configs[best_avg_idx[i]])
            selected_config[0] = "({})Best Avg{}".format(len(new_configs),
                avg_tag[i])
            new_configs.append(selected_config)
            new_result_list.append(result_list[best_avg_idx[i]])
            new_avg_list.append(avg_list[best_avg_idx[i]])

        for i in range(len(candidate_task)):
            selected_config = copy.deepcopy(configs[best_task_idx[i]])
            selected_config[0] = "({})Best {}".format(len(new_configs),
                official_dict[candidate_task[i]][0])
            new_configs.append(selected_config)
            new_result_list.append(result_list[best_task_idx[i]])
            new_avg_list.append(avg_list[best_task_idx[i]])
        configs = new_configs
        result_list = new_result_list
        avg_list = new_avg_list

    # split the case names by space
    if split_name_by_space:
        max_num_row = 1
        splitted_names = []
        for i in range(len(configs)):
            new_name = configs[i][0].split()
            max_num_row = max(max_num_row, len(new_name))
            splitted_names.append(new_name)
        tab_header = ["" for _ in range(max_num_row-1)] + tab_header
        for i in range(len(configs)):
            padding = ["" for _ in range(max_num_row-len(splitted_names[i]))]
            configs[i] = padding + splitted_names[i] + configs[i][1:]
    
    # generate the table
    print("\\begin{table}")
    print("\centering")
    print(fontsize)
    print("\caption{"+caption+"}")
    text = "\\begin{tabular}{@{}l|"
    for _ in range(len(configs)):
        text += "c"
    text += "@{}}"
    print(text)
    print("\\toprule")
    for i in range(len(tab_header)):
        text = "{} &".format(tab_header[i])
        for j in range(len(configs)):
            if j != len(configs) - 1:
                text += (configs[j][i] + "& ")
            else:
                text += (configs[j][i] + "\\\\")
        print(text)
    print("\midrule")
    for i in range(len(avg_range)):
        text = ("Avg. " + avg_tag[i])
        arr = []
        for j in range(len(configs)):
            arr.append(avg_list[j][i])
            text += " & {}".format(round_sig(avg_list[j][i]))
        text += "\\\\"
        if print_stderr:
            arr_mean = mean(arr)
            arr_std = sample_stddev(arr)
            text += " % mean {:.3f}, std {:.3f}, mean+1std {:.3f}, mean+2std {:.3f}, mean+3std {:.3f}".format(
                arr_mean, arr_std, arr_mean+arr_std, arr_mean+arr_std*2, arr_mean+arr_std*3)
        print(text)
    if not avg_only:
        print("\midrule")
        for i in range(len(task_order)):
            task = task_order[i]
            text = "({}) {}".format(i, official_dict[task][0])
            arr = []
            for j in range(len(configs)):
                result_dict = result_list[j]
                if task in result_dict:
                    text += " & {}".format(round_sig(result_dict[task][0]))
                    arr.append(result_dict[task][0])
                else:
                    text += " & N/A"
            text += "\\\\"
            if print_stderr:
                arr_mean = mean(arr)
                arr_std = sample_stddev(arr)
                if task != "wikitext":
                    text += " % mean {:.3f}, std {:.3f}, mean+1std {:.3f}, mean+2std {:.3f}, mean+3std {:.3f}".format(
                        arr_mean, arr_std, arr_mean+arr_std, arr_mean+arr_std*2, arr_mean+arr_std*3)
                else:
                    text += " % mean {:.3f}, std {:.3f}, mean-1std {:.3f}, mean-2std {:.3f}, mean-3std {:.3f}".format(
                        arr_mean, arr_std, arr_mean-arr_std, arr_mean-arr_std*2, arr_mean-arr_std*3)
            print(text)
    print("\\bottomrule")
    print("\end{tabular}")
    print("\end{table}")
    print("")
    print("")

if __name__ == '__main__':
    task_order = ["hellaswag","lambada","triviaqa","webqs","winogrande","piqa",
        "arc_challenge","arc_easy","anli_r1","anli_r2","anli_r3","openbookqa",
        "race","boolq","copa","rte","wsc","multirc","record","wikitext"]
    avg_range = [18]
    avg_tag = ["0-18"]
    tab_header = ["Case","Model size","Train tokens","Batch size","Bsz warmup","LR","min LR","LR warmup","LR decay","decay style"]

    configs = [
        ["(0)paper","125M","300B","256","4B","6e-4","6e-5","375M","260B","cosine", None], # gpt3 paper orig results, thus result path is None
        ["(1)repro","125M","300B","256","4B","6e-4","6e-5","375M","260B","cosine",
         '/blob/users/conglli/project/data_efficiency_gpt/eval_results/gpt-pile-0.125B-tok300B-lr6.0e-4-min6.0e-5-wup375M-dcy260B-sty-cosine-gbs256-mbs4-gpu64-zero0-mp1-pp1-nopp-seed1234-bwup4B/global_step591581/'],
        ["(2)fixedBsz","125M","300B","256","N/A","6e-4","6e-5","3000M","260B","cosine",
         '/blob/users/conglli/project/data_efficiency_gpt/eval_results/gpt-pile-0.125B-tok300B-lr6.0e-4-min6.0e-5-wup3000M-dcy260B-sty-cosine-gbs256-mbs4-gpu64-zero0-mp1-pp1-nopp-seed1234/global_step572205/'],
        ["(3)fixedBsz 300B+minLR","125M","300B","256","N/A","6e-4","1e-6","3000M","300B","cosine",
         '/blob/users/conglli/project/data_efficiency_gpt/eval_results/gpt-pile-0.125B-tok300B-lr6.0e-4-min1.0e-6-wup3000M-dcy300B-sty-cosine-gbs256-mbs4-gpu64-zero0-mp1-pp1-nopp-seed1234/global_step572205/']
    ]
    caption = 'Conglong: GPT-3 125M results zero-shot'
    generate_result_table(tab_header, configs, task_order, caption, avg_range,
        avg_tag, split_name_by_space=True, fontsize="\\tiny")

    configs = [
        ["(0)paper","125M","300B","256","4B","6e-4","6e-5","375M","260B","cosine", None], # gpt3 paper orig results, thus result path is None
        ["(1)repro","125M","300B","256","4B","6e-4","6e-5","375M","260B","cosine",
         '/blob/users/conglli/project/data_efficiency_gpt/eval_results_fewshot/gpt-pile-0.125B-tok300B-lr6.0e-4-min6.0e-5-wup375M-dcy260B-sty-cosine-gbs256-mbs4-gpu64-zero0-mp1-pp1-nopp-seed1234-bwup4B/global_step591581/'],
        ["(2)fixedBsz","125M","300B","256","N/A","6e-4","6e-5","3000M","260B","cosine",
         '/blob/users/conglli/project/data_efficiency_gpt/eval_results_fewshot/gpt-pile-0.125B-tok300B-lr6.0e-4-min6.0e-5-wup3000M-dcy260B-sty-cosine-gbs256-mbs4-gpu64-zero0-mp1-pp1-nopp-seed1234/global_step572205/'],
        ["(3)fixedBsz 300B+minLR","125M","300B","256","N/A","6e-4","1e-6","3000M","300B","cosine",
         '/blob/users/conglli/project/data_efficiency_gpt/eval_results_fewshot/gpt-pile-0.125B-tok300B-lr6.0e-4-min1.0e-6-wup3000M-dcy300B-sty-cosine-gbs256-mbs4-gpu64-zero0-mp1-pp1-nopp-seed1234/global_step572205/'],
    ]
    caption = 'Conglong: GPT-3 125M results few-shot'
    generate_result_table(tab_header, configs, task_order, caption, avg_range,
        avg_tag, split_name_by_space=True, fontsize="\\tiny", few_shot=True)

