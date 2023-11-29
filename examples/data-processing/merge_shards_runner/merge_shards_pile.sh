
SIZE=53486543872
python examples/data-processing/merge_shards.py \
--input-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/the_pile/" \
--output-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/the_pile/merged_shards" \
--prefix-name "en_pile_" \
--az-configs "examples/configs/azure_login_configs.json" \
--az-sample-yaml-job-file "examples/data-processing/az_templates/template_merge_shard.yaml" \
--compute-target 'azure' \
--shard-size $SIZE