python examples/data-processing/data_ratio_from_file.py \
 --prefix-paths-from-json "examples/pretrain-llama/training/llama_en_ar_reasoning_warmup_optim_cont/data.json" \
 --domain-ratio-from-json "examples/pretrain-llama/training/llama_en_ar_reasoning_warmup_optim_cont/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-llama/training/llama_en_ar_reasoning_warmup_optim_cont/lang_prob.json" \
 --total-token 10_000_000_000 \
 --exclude-iterator-json "examples/pretrain-llama/training/llama_en_ar_reasoning_warmup_optim_cont/exclude_iterator.json" \
 --export-script "examples/pretrain-llama/training/llama_en_ar_reasoning_warmup_optim_cont/iter_prob.sh" \
 --prefix-for-file-path "\$BIN_IDX_PATH/"