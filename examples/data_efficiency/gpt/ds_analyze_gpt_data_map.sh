#!/bin/bash

num_workers=1 # Num nodes to run the map job
num_threads=40 # Num threads on each node. Set this based on #CPU cores

# If different data epochs have slightly different data samples (e.g., due
# to randomness), then you need to specify large enough num_epochs that cover
# whole pretraining. If different data epochs are the same, set num_epochs to
# 1 to only index 1 epoch, and during pretraining DeepSpeed data efficiency
# library will automatically handle reshuffling when reaching another epoch.
num_epochs=1

# Which node is this node (start with 0 and end with num_workers-1). This
# script only launch the map job on 1 worker node, since we don't expect
# running on many nodes and workers don't need any communication. But you
# can modify this script to add a MPI/torch distributed launcher.
worker_id=$1
save_path="/blob/users/conglli/data/analysis_pile_gpt_${num_epochs}epoch/"

metric='total_vocab_freq'
# metric='vocab_rarity' # this requires the result of total_vocab_freq

seq_len=2048
batch_size=10000

jobname="gpt-pile-analyzing-${metric}-${num_epochs}epoch-map-worker${worker_id}"
# Public the Pile dataset, can be downloaded at
# https://mystic.the-eye.eu/public/AI/pile_neox/
## Change data_home to your own training data path.
# data_home="/vc_data_blob/users/conglli/the_pile_public_merged_nopreprocessing"
data_home="/blob/data/the_pile_public_merged_nopreprocessing"
data_path="${data_home}/pile_text_document"

vocab_path="gpt2-vocab.json"
if [ ! -f "$vocab_path" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
fi
merge_path="gpt2-merges.txt"
if [ ! -f "$merge_path" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
fi

# Make sure the "--split" is the same as what you will use for pre-training.
options=" \
    --analyzing-task map \
    --analyzing-data-type GPT \
    --analyzing-metric ${metric} \
    --analyzing-num-workers ${num_workers} \
    --analyzing-worker-id ${worker_id} \
    --analyzing-num-threads ${num_threads} \
    --vocab-file ${vocab_path} \
    --merge-file ${merge_path} \
    --data-path ${data_path} \
    --data-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --micro-batch-size ${batch_size} \
    --global-batch-size ${batch_size} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --num-layers 1 \
    --hidden-size 1 \
    --num-attention-heads 1 \
    --split 949,50,1 \
    --distributed-backend gloo \
    --train-data-exact-num-epochs ${num_epochs} \
    --return-data-index \
    --save-interval 1 \
    --save ${save_path}"

python ../analyze_data.py ${options} &> ${jobname}.log