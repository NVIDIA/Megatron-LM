#!/bin/bash
dir=`pwd`
###############################################################################
### Main configs
### The main configs are from Megatron-LM paper
### https://arxiv.org/abs/1909.08053. Choose based on your desired model size
### or build your own configs.
seq_len=512
global_batch_size=1024
lr=1e-4
min_lr=1e-5

## init_std is the standard deviation for weight initialization. Usually larger
## model needs lower std. Here we roughly follow a heuristic equation of
## sqrt(1/3/hidden_size) from https://arxiv.org/pdf/2201.11990.pdf

## In addition, we find that the 3.9B model (even after tuning init_std) has
## NaN loss issue from the beginning thus unable to train. This is probably
## because in this example we use the public Pile data, which is a more diverse
## (and potentially more noisy) data than what used in Megatron paper. One
## potential solution is only use the sub datasets in Pile that are also
## used by Megatron paper.

## BERT 110M (same config as original BERT-Base model)
## This config is not included in Megatron-LM paper
# model_size=0.11
# num_layers=12
# hidden_size=768
# num_attn_heads=12
# init_std=0.02

## BERT 336M (same config as original BERT-Large model)
model_size=0.336
num_layers=24
hidden_size=1024
num_attn_heads=16
init_std=0.02

## BERT 1.3B
# model_size=1.3
# num_layers=24
# hidden_size=2048
# num_attn_heads=32
# init_std=0.013

## BERT 3.9B
# model_size=3.9
# num_layers=48
# hidden_size=2560
# num_attn_heads=40
# init_std=0.011
###############################################################################
### Training duration configs
## The main termination condition, original Megatron paper trains for 2M iters.
train_iters_in_million=2
train_iters=$((${train_iters_in_million} * 1000000))
###############################################################################
### lr configs
## lr warmup and decay duration. Original Megatron paper uses 10000 warmup
## iters. Decay iters is the same as train iters.
lr_warmup_iters=10000
lr_decay_iters_in_million=${train_iters_in_million}
lr_decay_iters=$((${lr_decay_iters_in_million} * 1000000))
lr_decay_style="linear"
###############################################################################
### Parallelism configs
## Model parallelism, 1 is no MP
mp_size=1

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Currently pipeline parallelism is not supported for BERT model: DeepSpeed's
## pipeline parallelism is only integrated with the GPT case, and currently
## DeepSpeed is not integrated with Megatron's own pipeline parallelism.
pp_size=1
no_pp="true"

## ZeRO stage
zero_stage=0

## Total number of GPUs. ds_ssh is from DeepSpeed library.
num_gpus=$(($(ds_ssh nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)-2))
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_node=$(( ${num_gpus} / ${num_gpus_pernode} ))
## Data parallel size.
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} ))

## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
## Below batch_size calculation assumes the case without gradient accumulation.
## Manually set it to a lower value if you hit out of memory during training.
batch_size=$(( ${global_batch_size} / ${dp_size} ))
###############################################################################
### Misc configs
log_interval=100
eval_iters=10
eval_interval=1000
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
num_save=100
save_interval=$((${train_iters} / ${num_save}))

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"

## Public the Pile dataset, see prepare_pile_data.py in the same directory
## about how to download and preprocess the data.
jobname="bert-pile"
## For internal use. Change data_home to your own training data path.
data_home="/vc_data_blob/users/conglli/the_pile_bert"
if [[ "$host" == *"webxt"* ]]; then
    data_home="/blob/data/the_pile_bert"
fi
data_path="${data_home}/pile_bert_train_text_sentence"

vocab_path="bert-large-uncased-vocab.txt"
if [ ! -f "$vocab_path" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt
fi

## Number of workers for dataloader. We found that for BERT pre-training,
## num_workers will greatly affect data loading time and overall training
## time. In our experiment with 64 GPUs, the performance reaches peak at
## num_workers = 4 but it may differ depending on hardware. Also note that
## larger num_workers add more CPU computation/memory overhead.
num_workers=4

jobname="${jobname}-${model_size}B-iters-${train_iters_in_million}M"
jobname="${jobname}-lr-${lr}-min-${min_lr}-wmup-${lr_warmup_iters}-dcy-${lr_decay_iters_in_million}M-sty-${lr_decay_style}"
jobname="${jobname}-gbs-${global_batch_size}-mbs-${batch_size}-gpu-${num_gpus}-zero-${zero_stage}-mp-${mp_size}-pp-${pp_size}"
if [ "${no_pp}" = "true" ]; then
    jobname="${jobname}-nopp"
fi

username=$(whoami)
output_home="/vc_data_blob/users/${username}/project/bert_with_pile"
if [[ "$host" == *"webxt"* ]]; then
    output_home="/blob/users/${username}/project/bert_with_pile"
fi
log_path="${output_home}/log/"
checkpoint_path="${output_home}/checkpoint/${jobname}"
## Microsoft internal constraint: because tensorboard is logged by last rank,
## it's better to put the path in NFS instead of Blob.
tensorboard_dir="/vc_data/users/${username}/project/bert_with_pile/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}
###############################################################################
data_options=" \
    --vocab-file ${vocab_path} \
    --data-path ${data_path} \
    --data-impl mmap"

megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --init-method-std ${init_std} \
    --tensor-model-parallel-size ${mp_size} \
    --lr-decay-iters ${lr_decay_iters} \
    --lr-warmup-iters ${lr_warmup_iters} \
    --micro-batch-size ${batch_size} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --train-iters ${train_iters} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --split 949,50,1 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --save-interval ${save_interval} \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --num-workers ${num_workers} \
    --fp16 \
    --load ${checkpoint_path} \
    --save ${checkpoint_path} \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --tensorboard-dir ${tensorboard_path}"

if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

template_json="ds_config_bert_TEMPLATE.json"
config_json="ds_config_bert_bsz${global_batch_size}_mbsz${batch_size}_log${log_interval}_zero${zero_stage}.json"
if [[ $zero_stage -gt 0 ]]; then
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/false/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
      > ${config_json}
else
sed "s/CONFIG_BATCH_SIZE/${global_batch_size}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
      > ${config_json}
fi

deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} \
    --pipeline-model-parallel-size ${pp_size}"

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

if [ "${activation_checkpoint}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
    --deepspeed-activation-checkpointing"
fi

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0
for (( node = 0; node <= num_node-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$iteration_file\""); then
        local_iteration=$(ssh -q worker-"$node" cat $iteration_file)
        iteration=$(( ${local_iteration} > ${iteration} ? ${local_iteration} :  ${iteration} ))
    fi
done
if [[ $iteration -gt 0 ]]; then
    iteration_2="global_step${iteration}"
    ds_ssh "echo $iteration > $iteration_file"
    ds_ssh "echo $iteration_2 > $iteration_file_2"
fi

deepspeed ${dir}/../../pretrain_bert.py ${megatron_options} ${data_options} ${deepspeed_options} &>> ${log_path}/${jobname}_${host}_${current_time}.log