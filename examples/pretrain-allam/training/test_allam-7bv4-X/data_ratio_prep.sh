if [ ! -f "examples/pretrain-allam/training/test_allam-7bv4-X/data.json" ]; then
    python examples/data-processing/remote_list.py \
    --az-configs "examples/configs/azure_login_configs.json" \
    --input-folder-path "https://allamllmuksstandard.blob.core.windows.net/llm-data/data_repo/tokenize_by_v5_improved/meglm_tok_v5_improved_bin_idx/" \
    --export-data-signature "examples/pretrain-allam/training/test_allam-7bv4-X/data.json"
fi

python examples/data-processing/data_ratio_from_file.py \
 --prefix-paths-from-json "examples/pretrain-allam/training/test_allam-7bv4-X/data.json" \
 --domain-ratio-from-json  "examples/pretrain-allam/training/test_allam-7bv4-X/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-allam/training/test_allam-7bv4-X/lang_prob.json" \
 --exclude-iterator-json "examples/pretrain-allam/training/test_allam-7bv4-X/exclude_iterator.json" \
 --total-token 4_000_000_000_000 \
 --export-script examples/pretrain-allam/training/test_allam-7bv4-X/iterator_prob.sh \
 --prefix-for-file-path "\$BIN_IDX_PATH/" 
