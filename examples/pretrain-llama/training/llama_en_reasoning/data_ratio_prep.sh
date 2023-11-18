
python examples/pretrain-llama/data-processing/data_ratio_from_file.py \
 --source-prefix-paths "../DUMPED/reasoning-llama2-indexed_data/*.bin" \
 --domain-ratio-from-json "examples/pretrain-llama/training/llama_en_reasoning/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-llama/training/llama_en_reasoning/lang_prob.json" \
 --total-token 500_000_000_000 \
 --exclude-iterator-json "examples/pretrain-llama/training/llama_en_reasoning/exclude_iterator.json"