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

def gather_MNLI_results(result_path):
    overall = []
    matched = []
    mismatched = []
    for file in os.listdir(result_path):
        if file.startswith('MNLI'):
            fname = f'{result_path}/{file}/output.log'
            if os.path.exists(fname):
                results = gather_numbers(fname,
                    ['overall:', 'metrics for dev-matched:', 'metrics for dev-mismatched:'],
                    ['overall:', 'dev-matched:', 'dev-mismatched:'],
                    [9, 9, 9])
                overall_candidate = results['overall:']
                matched_candidate = results['dev-matched:']
                mismatched_candidate = results['dev-mismatched:']
                if len(overall_candidate) > 0:
                    assert len(overall_candidate) == len(matched_candidate) and len(overall_candidate) == len(mismatched_candidate)
                    best_index = overall_candidate.index(max(overall_candidate))
                    overall.append(overall_candidate[best_index])
                    matched.append(matched_candidate[best_index])
                    mismatched.append(mismatched_candidate[best_index])
    if len(overall) > 0:
        if len(overall) % 2 == 1:
            median_idx = overall.index(statistics.median(overall))
        else:
            median_idx = overall.index(statistics.median_high(overall))
        print(f'MNLI how Megatron paper reported: overall results median {statistics.median(overall)}, corresponding matched/mismatched: {matched[median_idx]}/{mismatched[median_idx]}')
        print(f'MNLI other results:')
        print(f'MNLI overall results {overall}, median {statistics.median(overall)} (corresponding matched/mismatched {matched[median_idx]}/{mismatched[median_idx]}), mean {statistics.mean(overall)}, std {statistics.stdev(overall)}')
        print(f'MNLI matched results {matched}, median {statistics.median(matched)}, mean {statistics.mean(matched)}, std {statistics.stdev(matched)}')
        print(f'MNLI mismatched results {mismatched}, median {statistics.median(mismatched)}, mean {statistics.mean(mismatched)}, std {statistics.stdev(mismatched)}')
    else:
        print("Didn't find any MNLI result")

def gather_QQP_results(result_path):
    overall = []
    for file in os.listdir(result_path):
        if file.startswith('QQP'):
            fname = f'{result_path}/{file}/output.log'
            if os.path.exists(fname):
                results = gather_numbers(fname, ['overall:'], ['overall:'], [9])
                overall_candidate = results['overall:']
                if len(overall_candidate) > 0:
                    best_index = overall_candidate.index(max(overall_candidate))
                    overall.append(overall_candidate[best_index])
    if len(overall) > 0:
        print(f'QQP how Megatron paper reported: overall results median {statistics.median(overall)}')
        print(f'QQP other results:')
        print(f'QQP overall results {overall}, median {statistics.median(overall)}, mean {statistics.mean(overall)}, std {statistics.stdev(overall)}')
    else:
        print("Didn't find any QQP result")

def gather_RACE_results(result_path, task):
    dev = []
    test = []
    for file in os.listdir(result_path):
        if file.startswith(f'RACE-{task}'):
            fname = f'{result_path}/{file}/output.log'
            if os.path.exists(fname):
                results = gather_numbers(fname,
                    [f'metrics for dev-{task}:', f'metrics for test-{task}:'],
                    [f'dev-{task}:', f'test-{task}:'],
                    [9, 9])
                dev_candidate = results[f'dev-{task}:']
                test_candidate = results[f'test-{task}:']
                if len(dev_candidate) > 0:
                    assert len(dev_candidate) == len(test_candidate)
                    dev.append(max(dev_candidate))
                    test.append(max(test_candidate))
    if len(dev) > 0:
        if len(dev) % 2 == 1:
            median_idx = dev.index(statistics.median(dev))
        else:
            median_idx = dev.index(statistics.median_high(dev))
        print(f'RACE-{task} how Megatron paper reported: test result from the median of dev results {test[median_idx]}')
        print(f'RACE-{task} other results:')
        print(f'RACE-{task} dev results {dev}, median {statistics.median(dev)}, mean {statistics.mean(dev)}, std {statistics.stdev(dev)}')
        print(f'RACE-{task} test results {test}, median {statistics.median(test)}, mean {statistics.mean(test)}, std {statistics.stdev(test)}')
    else:
        print(f"Didn't find any RACE-{task} result")

def gather_finetune_results(result_path):
    print(f'Gather finetune results for {result_path}')
    gather_MNLI_results(result_path)
    gather_QQP_results(result_path)
    gather_RACE_results(result_path, 'middle')
    gather_RACE_results(result_path, 'high')

if __name__ == '__main__':
    result_path='/blob/users/conglli/project/bert_with_pile/checkpoint/bert-pile-0.336B-iters-2M-lr-1e-4-min-1e-5-wmup-10000-dcy-2M-sty-linear-gbs-1024-mbs-16-gpu-64-zero-0-mp-1-pp-1-nopp-finetune/'
    gather_finetune_results(result_path)