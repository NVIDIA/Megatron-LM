import os
import glob
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-prefix-paths', type=str, required=True,
                       help='Glob path to folder where all the bin, idx files are.')
    parser.add_argument('--domain-ratio-from-json', type=str, required=True,
                       help='Domain multiplier from a json file.')
    parser.add_argument('--lang-select-prob-json', type=str, required=True,
                       help='Path to a json file that indicates the lang selection prob.')
    parser.add_argument('--exclude-iterator-json', type=str, required=True,
                       help='Path to a json file that list the restricted iterator name.')
    parser.add_argument('--total-token', type=int, required=True,
                       help='Total token to be sampled.')
    parser.add_argument('--verbose', action='store_true',
                       help='Print additional information')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    source_prefix_paths = glob.glob(args.source_prefix_paths)
    domain_dict = json.load(open(args.domain_ratio_from_json))
    lang_prob_dict = json.load(open(args.lang_select_prob_json))
    exclude_iterator_list = json.load(open(args.exclude_iterator_json))['exclude_iterator_name']


    source_prefix_paths = sorted([ source_prefix_path.replace(".bin", "") for source_prefix_path in source_prefix_paths if source_prefix_path.endswith(".bin")])
    data_dist_by_lang, tot_token_by_lang, tot_sampled_token_by_lang = {}, {}, {}
    
    print("\n\nFile Found ...")
    for idx, source_prefix_path in enumerate(source_prefix_paths):
        source_prefix_path = os.path.basename(source_prefix_path)
        if source_prefix_path in exclude_iterator_list:
            continue
        print(f"\t{source_prefix_path}")
        dc = int(source_prefix_path.split("dc=")[1].split("_")[0])
        sc = int(source_prefix_path.split("sc=")[1].split("_")[0])
        tc = int(source_prefix_path.split("tc=")[1].split("_")[0])
        lang = source_prefix_path.split("_")[0]
        domain = source_prefix_path.split("_")[1]
        if lang not in data_dist_by_lang: data_dist_by_lang[lang] = []
        if lang not in tot_token_by_lang: tot_token_by_lang[lang] = 0
        if lang not in tot_sampled_token_by_lang: tot_sampled_token_by_lang[lang] = 0
        data_dist_by_lang[lang].append(
            ( tc * domain_dict[domain], source_prefix_path )
        )
        tot_token_by_lang[lang] += tc
        tot_sampled_token_by_lang[lang] += (tc * domain_dict[domain])
    
    iterator_selection_prob = []
    for lang, iterator_list in  data_dist_by_lang.items():
        tot_prob_covered = 0.0
        for (iterator_tok_cnt, iterators_name) in iterator_list:
            domain = iterators_name.split("_")[1]
            prob = iterator_tok_cnt/tot_sampled_token_by_lang[lang] * lang_prob_dict[lang]
            iterator_selection_prob.append(
                [prob, iterators_name, int(prob*args.total_token), iterator_tok_cnt//domain_dict[domain]]
            )
            tot_prob_covered += prob
        assert abs(lang_prob_dict[lang] - tot_prob_covered) < 1e-6
    
    print(f"\n\n> Total token by language.")
    for lang, token in tot_token_by_lang.items():
        print(f"\t\t{lang}:{token:_}")
    print(f"> Total token that will be sampled by language.")
    for lang, token in tot_sampled_token_by_lang.items():
        print(f"\t\t{lang}:{token:_}")
    print("\n")
    for prob, iterator_name, total_token_to_be_sampled, total_token_exists in iterator_selection_prob:
        if args.verbose:
            print(f"{prob} {os.path.basename(iterator_name)} {total_token_to_be_sampled:_} {total_token_exists:_} {total_token_to_be_sampled/total_token_exists}")
        else:
            print(f"{prob} {os.path.basename(iterator_name)}")

    

"""
python examples/pretrain-llama/data-processing/data_ratio_from_file.py \
     --source-prefix-paths "../DUMPED/allam_data_2-1_splits-llama2-indexed_data/llama2_bin_idx/*.bin" \
     --domain-ratio-from-json examples/pretrain-llama/data-processing/sampling_assets/data_ratio.json \ 
     --lang-select-prob-json examples/pretrain-llama/data-processing/sampling_assets/lang_prob.json \
     --total-token 1000000000000 \ 
     --exclude-iterator-json examples/pretrain-llama/data-processing/sampling_assets/exclude_iterator.json 
"""