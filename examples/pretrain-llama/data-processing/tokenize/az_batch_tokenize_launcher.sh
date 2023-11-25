python examples/pretrain-llama/data-processing/tokenize/az_batch_tokenize_launcher.py \
--az-blob-input-folder-path "https://allamllmuksstandard.blob.core.windows.net/vocab-expanded-training-data/processed/ar/translated" \
--az-blob-bin-idx-folder-path "azureml://subscriptions/c7209a17-0d9f-41df-8e45-e0172343698d/resourcegroups/llm-test/workspaces/Provisioning-Test/datastores/llama_pretraining/paths/test_bin_idx" \
--az-tokenizer-model "azureml://subscriptions/c7209a17-0d9f-41df-8e45-e0172343698d/resourcegroups/llm-test/workspaces/Provisioning-Test/datastores/llama_pretraining/paths/allam_data_2-1_splits-llama2-VE-indexed_data/tokenizer/tokenizer.model" \
--az-sas-token "sp=racwdli&st=2023-11-25T01:04:32Z&se=2023-11-25T09:04:32Z&spr=https&sv=2022-11-02&sr=c&sig=htlh%2B5OEmWy0ikTxgrAJny0F7g3PdgXxYFIQwsEHYh0%3D" \
--az-num-proc 16
