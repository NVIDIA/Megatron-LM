if [ ! -f "examples/pretrain-allam/training/allam-7bv3-0-1/data.json" ]; then
    python examples/data-processing/remote_list.py \
    --az-configs "examples/configs/azure_login_configs.json" \
    --input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/data_repo/tokenize_by_v5_improved/meglm_tok_v5_improved_bin_idx/" \
    --export-data-signature "examples/pretrain-allam/training/allam-7bv3-0-1/data.json"
fi

python examples/data-processing/data_ratio_from_file.py \
 --prefix-paths-from-json "examples/pretrain-allam/training/allam-7bv3-0-1/data.json" \
 --domain-ratio-from-json  "examples/pretrain-allam/training/allam-7bv3-0-1/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-allam/training/allam-7bv3-0-1/lang_prob.json" \
 --exclude-iterator-json "examples/pretrain-allam/training/allam-7bv3-0-1/exclude_iterator.json" \
 --total-token 500_000_000_000 \
 --export-script examples/pretrain-allam/training/allam-7bv3-0-1/iterator_prob.sh \
 --prefix-for-file-path "\$BIN_IDX_PATH/" 
