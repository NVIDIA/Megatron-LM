
NUM_SIZE=4348654387

bash examples/data-processing/merge_shards_runner/slim_pajama_process.sh

mkdir -p "../RAW_DATA_FOLDER/SlimPajama-627B/data_by_domain"
python examples/data-processing/sample_json_data_from_meta.py \
--input-folder-path "../RAW_DATA_FOLDER/SlimPajama-627B/export/" \
--output-folder-path "../RAW_DATA_FOLDER/SlimPajama-627B/data_by_domain" \
--meta-keys 'meta' 'redpajama_set_name'

# # train
# python examples/data-processing/merge_shard.py \
# --input-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/SlimPajama-627B/" \
# --output-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/SlimPajama-627B/merged_shards" \
# --prefix-name "en_slim_pajama_" \
# --az-configs "examples/configs/azure_login_configs.json" \
# --az-sample-yaml-job-file "examples/data-processing/az_templates/template_merge_shard.yaml" \
# --compute-target 'azure' \
# --shard-size $NUM_TOKEN 

# --input-folder-path "../RAW_DATA_FOLDER/SlimPajama-627B/train/" \
# --output-folder-path "../RAW_DATA_FOLDER/SlimPajama-627B/merged_shards" \
# --prefix-name "en_slim_pajama_" \
# --compute-target 'local' \
# --shard-size $NUM_TOKEN 

# # validation
# python examples/data-processing/merge_shard.py \
# --input-folder-path "../RAW_DATA_FOLDER/SlimPajama-627B/validation/" \
# --output-folder-path "../RAW_DATA_FOLDER/SlimPajama-627B/merged_shards" \
# --prefix-name "en_slim_pajama_" \
# --compute-target 'local' \
# --shard-size $NUM_TOKEN 

# # test
# python examples/data-processing/merge_shard.py \
# --input-folder-path $PATH_TO_SLIM_PAJAMA/test.jsonl \
# --output-folder-path "$PATH_TO_SLIM_PAJAMA/merged_shards" \
# --prefix-name "en_slim_pajama_" \
# --compute-target 'local' \
# --shard-size $NUM_SIZE 