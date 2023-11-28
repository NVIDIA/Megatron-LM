
NUM_TOKEN=43486543872

# C4
# python examples/data-processing/merge_shard.py \
# --input-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/data/c4/" \
# --output-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/merged_shards/" \
# --prefix-name "en_dolma_c4_" \
# --az-configs "examples/configs/azure_login_configs.json" \
# --az-sample-yaml-job-file "examples/data-processing/az_templates/template_merge_shard.yaml" \
# --shard-size $NUM_TOKEN 

python examples/data-processing/merge_shard.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/data/common-crawl/" \
--output-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/merged_shards/" \
--prefix-name "en_dolma_common-crawl_" \
--az-configs "examples/configs/azure_login_configs.json" \
--az-sample-yaml-job-file "examples/data-processing/az_templates/template_merge_shard.yaml" \
--shard-size $NUM_TOKEN 
