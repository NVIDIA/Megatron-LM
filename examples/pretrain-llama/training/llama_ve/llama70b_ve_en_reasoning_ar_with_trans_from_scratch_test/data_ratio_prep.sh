if [ ! -f "examples/pretrain-llama/training/llama_ve/llama70b_ve_en_reasoning_ar_with_trans_from_scratch_test/data.json" ]; then
    python examples/data-processing/remote_list.py \
    --az-configs "examples/configs/azure_login_configs.json" \
    --input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/data_repo/tokenize_by_llama2-ve/meglm_llama2-ve_bin_idx/" \
    --export-data-signature "examples/pretrain-llama/training/llama_ve/llama70b_ve_en_reasoning_ar_with_trans_from_scratch_test/data.json"
fi


python examples/data-processing/data_ratio_from_file.py \
 --prefix-paths-from-json "examples/pretrain-llama/training/llama_ve/llama70b_ve_en_reasoning_ar_with_trans_from_scratch_test/data.json" \
 --domain-ratio-from-json "examples/pretrain-llama/training/llama_ve/llama70b_ve_en_reasoning_ar_with_trans_from_scratch_test/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-llama/training/llama_ve/llama70b_ve_en_reasoning_ar_with_trans_from_scratch_test/lang_prob.json" \
 --total-token 2000_000_000_000 \
 --exclude-iterator-json "examples/pretrain-llama/training/llama_ve/llama70b_ve_en_reasoning_ar_with_trans_from_scratch_test/exclude_iterator.json" \
 --prefix-for-file-path "\$BIN_IDX_PATH" \
 --export-script "examples/pretrain-llama/training/llama_ve/llama70b_ve_en_reasoning_ar_with_trans_from_scratch_test/iter_prob.sh" \
 --verbose