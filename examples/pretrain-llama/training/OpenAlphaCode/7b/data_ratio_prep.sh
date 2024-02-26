if [ ! -f "examples/pretrain-llama/training/OpenAlphaCode/7b/data.json" ]; then
    python examples/data-processing/remote_list.py \
    --az-configs "examples/configs/azure_west_europe.json" \
    --input-folder-path "https://provisioningte0256624006.blob.core.windows.net/llm-data/data_repo/tokenize_by_llama2/meglm_llama2_bin_idx/" \
    --export-data-signature "examples/pretrain-llama/training/OpenAlphaCode/7b/data.json"
fi


python examples/data-processing/data_ratio_from_file.py \
 --prefix-paths-from-json "examples/pretrain-llama/training/OpenAlphaCode/7b/data.json" \
 --domain-ratio-from-json "examples/pretrain-llama/training/OpenAlphaCode/7b/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-llama/training/OpenAlphaCode/7b/lang_prob.json" \
 --total-token 800_000_000 \
 --exclude-iterator-json "examples/pretrain-llama/training/OpenAlphaCode/7b/exclude_iterator.json" \
 --prefix-for-file-path "\$BIN_IDX_PATH" \
 --export-script "examples/pretrain-llama/training/OpenAlphaCode/7b/iter_prob.sh" \
 --verbose