# Preprocess data

## Single-node interactive node

bash preprocess_data_wikipedia.sh  db-build
bash preprocess_data_wikipedia.sh  index-train
bash preprocess_data_wikipedia.sh  query-pretraining-neighbors

# Pretraining

## Single-node interactive node

bash tools/retro/examples/tests/pretrain_model_wiki.sh

## Multi-node run with sbatch

sbatch tools/retro/examples/tests/pretrain-nextllm-800m-retro.sh
sbatch tools/retro/examples/tests/pretrain-nextllm-800m-gpt.sh
sbatch tools/retro/examples/tests/pretrain-nextllm-43b-retro.sh

## Check the training curves and see whether they are aligned