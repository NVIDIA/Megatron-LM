
python examples/pretrain-llama/data-processing/data_ratio_from_file.py \
 --prefix-paths-from-json "examples/pretrain-llama/training/llama_ve_en_ar_v1/allam_data_2-1_splits-llama2-VE-indexed_data.json" \
 --domain-ratio-from-json "examples/pretrain-llama/training/llama_ve_en_ar_v1/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-llama/training/llama_ve_en_ar_v1/lang_prob.json" \
 --total-token 500_000_000_000 \
 --exclude-iterator-json "examples/pretrain-llama/training/llama_ve_en_ar_v1/exclude_iterator.json"