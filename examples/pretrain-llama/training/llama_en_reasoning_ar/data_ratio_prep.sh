
python examples/pretrain-llama/data-processing/data_ratio_from_file.py \
 --prefix-paths-from-json "examples/pretrain-llama/training/llama_en_reasoning_ar/en_reasoning_and_arabic_files.json" \
 --domain-ratio-from-json "examples/pretrain-llama/training/llama_en_reasoning_ar/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-llama/training/llama_en_reasoning_ar/lang_prob.json" \
 --total-token 500_000_000_000 \
 --exclude-iterator-json "examples/pretrain-llama/training/llama_en_reasoning_ar/exclude_iterator.json"