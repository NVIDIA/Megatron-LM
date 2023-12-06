
python examples/data-processing/data_ratio_from_file.py \
 --prefix-paths-from-json "examples/pretrain-llama/training/llama_en_reasoning_ar_with_trans_from_scratch/en_reasoning_and_arabic_files_with_trans.json" \
 --domain-ratio-from-json "examples/pretrain-llama/training/llama_en_reasoning_ar_with_trans_from_scratch/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-llama/training/llama_en_reasoning_ar_with_trans_from_scratch/lang_prob.json" \
 --total-token 2400_000_000_000 \
 --exclude-iterator-json "examples/pretrain-llama/training/llama_en_reasoning_ar_with_trans_from_scratch/exclude_iterator.json" \
 --prefix-for-file-path "\$BIN_IDX_PATH" \
 --export-script "examples/pretrain-llama/training/llama_en_reasoning_ar_with_trans_from_scratch/iterator_selection_prob.sh"