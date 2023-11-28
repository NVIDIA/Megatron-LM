
NUM_TOKEN=43486543872
python examples/pretrain-llama/data-processing/merge_shard/merge_shard.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/data/c4/" \
--output-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/merged_shards/" \
--prefix-name "en_dolma_c4_" \
--az-configs "examples/pretrain-llama/data-processing/tokenize/azure_login_configs.json" \
--shard-size $NUM_TOKEN 

python examples/pretrain-llama/data-processing/merge_shard/merge_shard.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/data/c4/" \
--output-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/dolma/merged_shards/" \
--prefix-name "en_dolma_c4_" \
--az-configs "examples/pretrain-llama/data-processing/tokenize/azure_login_configs.json" \
--shard-size $NUM_TOKEN 