
python examples/pretrain-llama/data-processing/data_ratio_from_file.py \
 --source-prefix-paths "../DUMPED/allam_data_2-1_splits-llama2-indexed_data/llama2_bin_idx/*.bin" \
 --domain-ratio-from-json examples/pretrain-llama/data-processing/sampling_assets/data_ratio.json \ 
 --lang-select-prob-json examples/pretrain-llama/data-processing/sampling_assets/lang_prob.json \
 --total-token 1000000000000 \ 
 --exclude-iterator-json examples/pretrain-llama/data-processing/sampling_assets/exclude_iterator.json 