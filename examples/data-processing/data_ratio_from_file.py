import os
import glob
import json
import copy
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-prefix-paths', default=None, type=str,
                       help='Glob path to folder where all the bin, idx files are.')
    parser.add_argument('--prefix-paths-from-json', default=None, type=str,
                       help='File names listed in a json.')
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
    parser.add_argument('--export-script', type=str,
                       help='Export output to this file for running it in megatron format atgument.')
    parser.add_argument('--prefix-for-file-path', type=str,
                       help='Add additional prefix to the file path.')
    args = parser.parse_args()
    if args.source_prefix_paths is not None:
        assert args.prefix_paths_from_json is None
    if args.prefix_paths_from_json is not None:
        assert args.source_prefix_paths is None
    if args.prefix_for_file_path is not None:
        if args.prefix_for_file_path.endswith("/"):
            args.prefix_for_file_path = args.prefix_for_file_path[:-1]
    return args

def normalize(in_json):
    total = sum(in_json.values())
    for k, v in in_json.items():
        in_json[k] = v/total
    return in_json

if __name__ == "__main__":
    args = get_args()
    if args.source_prefix_paths is not None:
        source_prefix_paths = glob.glob(args.source_prefix_paths)
    else:
        source_prefix_paths = json.load(open(args.prefix_paths_from_json))
    domain_dict = json.load(open(args.domain_ratio_from_json))
    lang_prob_dict = normalize(json.load(open(args.lang_select_prob_json)))
    exclude_iterator_list = json.load(open(args.exclude_iterator_json))['exclude_iterator_name']
    
    exclude_iterator_list = sorted([ exclude_iterator.replace(".bin", "") for exclude_iterator in exclude_iterator_list if exclude_iterator.endswith(".bin")])
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
        domain_multiplier = copy.deepcopy(domain_dict)

        if lang in domain_dict: domain_multiplier = domain_multiplier[lang]
        if domain in domain_multiplier: domain_multiplier = domain_multiplier[domain]
        if isinstance(domain_multiplier, dict): continue
        if lang not in data_dist_by_lang: data_dist_by_lang[lang] = []
        if lang not in tot_token_by_lang: tot_token_by_lang[lang] = 0
        if lang not in tot_sampled_token_by_lang: tot_sampled_token_by_lang[lang] = 0
        data_dist_by_lang[lang].append(
            ( tc * domain_multiplier, source_prefix_path )
        )
        tot_token_by_lang[lang] += tc
        tot_sampled_token_by_lang[lang] += (tc * domain_multiplier)
    
    iterator_selection_prob = []
    for lang, iterator_list in  data_dist_by_lang.items():
        tot_prob_covered = 0.0
        for (iterator_tok_cnt, iterators_name) in iterator_list:
            if lang == "en":
                print(f"{iterators_name=} {iterator_tok_cnt=} {tot_sampled_token_by_lang[lang]=}")
            domain = iterators_name.split("_")[1]
            prob = iterator_tok_cnt/tot_sampled_token_by_lang[lang] * lang_prob_dict[lang]
            domain_multiplier = copy.deepcopy(domain_dict)
            if lang in domain_dict: domain_multiplier = domain_multiplier[lang]
            if domain in domain_multiplier:
                domain_multiplier = domain_multiplier[domain]
            assert not isinstance(domain_multiplier, dict)
            iterator_selection_prob.append(
                [prob, iterators_name, int(prob*args.total_token), iterator_tok_cnt//domain_multiplier]
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

    print("> Iterator selection probability.\n")
    lang_token = {k:0 for k, _ in lang_prob_dict.items()}
    if args.export_script is not None:
        if not args.export_script.endswith(".sh"): args.export_script = args.export_script + ".sh"
        out_file_ptr = open(f"{args.export_script}", "w")
        out_file_ptr.write("DATA_PATH=( --data-path ")
    for prob, iterator_name, total_token_to_be_sampled, total_token_exists in iterator_selection_prob:
        lang = iterator_name.split("_")[0]
        lang_token[lang] += total_token_to_be_sampled
        if args.verbose:
            print(f"\t{prob} {os.path.basename(iterator_name)} {total_token_to_be_sampled:_} {total_token_exists:_} {total_token_to_be_sampled/total_token_exists}")
        __output_format = os.path.basename(iterator_name).replace('=', '\\=')
        if not args.verbose:
            print(f"\t{prob} {args.prefix_for_file_path}/{__output_format}")
        if args.export_script is not None:
            out_file_ptr.write(f"\n{prob} {args.prefix_for_file_path}/{__output_format}")
    if args.export_script is not None:
        # out_file_ptr.write("\n)\nexport DATA_PATH=${_DATA_PATH[@]}")
        out_file_ptr.write("\n)")
        out_file_ptr.close()
    print(f"\n\nOut of {args.total_token} token, language wise token distribution.\n{json.dumps(lang_token, indent=4)}")