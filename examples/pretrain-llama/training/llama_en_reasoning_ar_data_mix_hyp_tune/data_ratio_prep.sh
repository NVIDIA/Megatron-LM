
python examples/pretrain-llama/data-processing/data_ratio_from_file.py \
 --prefix-paths-from-json "examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/en_reasoning_and_arabic_files.json" \
 --domain-ratio-from-json "examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/lang_prob.json" \
 --total-token 500_000_000_000 \
 --exclude-iterator-json "examples/pretrain-llama/training/llama_en_reasoning_ar_data_mix_hyp_tune/exclude_iterator.json"