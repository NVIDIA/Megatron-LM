
python examples/data-processing/data_ratio_from_file.py \
 --prefix-paths-from-json "examples/pretrain-allam/training/allam-1b_SlimPajama_mlm/data_signature.json" \
 --domain-ratio-from-json  "examples/pretrain-allam/training/allam-1b_SlimPajama_mlm/data_ratio.json" \
 --lang-select-prob-json "examples/pretrain-allam/training/allam-1b_SlimPajama_mlm/lang_prob.json" \
 --exclude-iterator-json "examples/pretrain-allam/training/allam-1b_SlimPajama_mlm/exclude_iterator.json" \
 --total-token 300000000000 \
 --export-script "examples/pretrain-allam/training/allam-1b_SlimPajama_mlm/iter_prob.sh" \
 --prefix-for-file-path "\$BIN_IDX_PATH/"