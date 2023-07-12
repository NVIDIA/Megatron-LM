#!/bin/bash
dir=`pwd`
###############################################################################
### Main configs
### The main configs are from Megatron-LM paper
### https://arxiv.org/abs/1909.08053. Choose based on your desired model size
### or build your own configs.
seq_len=512
global_batch_size=1024
# lr=1e-4
lr=$1
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
## We changed to token-based termination since data efficiency techniques could
## change token per step.
calc() { awk "BEGIN{ printf \"%.0f\n\", $* }"; }
# train_iters_in_million=2
train_iters_in_million=$2
train_tokens=$(calc $train_iters_in_million*1000000*$seq_len*$global_batch_size)
train_tokens_in_billion=$(calc $train_tokens/1000000000)

## A large enough number of iters, just to make sure we index enough data. The
## only effective termination condition is the train_tokens above.
train_iters=4000000

## Another wall-clock time termination condition in minutes. Set it large
## enough to avoid undesired early termination.
exit_duration=30000000
###############################################################################
### lr configs
## lr warmup and decay duration. Original Megatron paper uses 10000 warmup
## iters. We changed lr decay to token based since data efficiency techniques
## could change token per step.
lr_warmup_iters=10000
lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens=${train_tokens}
lr_decay_style="linear"
###############################################################################
### Parallelism configs
## Model parallelism, 1 is no MP
mp_size=1

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Currently pipeline parallelism is not supported for BERT model: DeepSpeed's
## pipeline parallelism is only integrated with the GPT case, and currently
## DeepSpeed is not integrated with Megatron's own pipeline parallelism.
## Note that currently both curriculum learning and random-LTD are NOT
## compatible with pipeline parallelism.
pp_size=1
no_pp="true"

## ZeRO-based data parallelism, stage=0 will disable ZeRO
zero_stage=0

## Total number of GPUs. ds_ssh is from DeepSpeed library.
num_gpus=$(($(ds_ssh nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)-2))
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_node=$(( ${num_gpus} / ${num_gpus_pernode} ))

## Data parallel size.
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} ))

## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
## Reduce it manually if GPU OOM
batch_size=$(( ${global_batch_size} / ${dp_size} ))
###############################################################################
### Random layerwise token dropping (random-LTD) configs
## random-LTD's main switch. "false" means disabled. "true" means enabled.
ltd_enabled=${3:-'false'}
## How much dropping ratio to start with. The value denotes the seqlen after
## dropping.
ltd_start=${4:-512}
## How many steps for random-LTD to gradually reduce dropping ratio to zero.
ltd_step_in_million=${5:-1}

# ltd_enabled="true"
# ltd_start=200
# ltd_step_in_million=1.8
ltd_step=$(calc $ltd_step_in_million*1000000)

## For BERT pretraining, we observe that random-LTD when combined with zero
## dropout can achieve better finetune accuracy on certain tasks. However, this
## is not guaranteed for all models/tasks. It is still recommend to try both
## with and without dropout for random-LTD.
dropout=${6:-0.1}
###############################################################################
### Curriculum learning (CL) configs
## CL's main switch. "false" means disabled. "true" means enabled.
cl_enabled=${7:-'false'}
## Number of CL metrics to use.
cl_num_metric=${8:-1}

## Name of difficulty metric
cl_1st_metric=${9:-'dummy'}
## Path to the data indexes for this difficulty metric. Samples on ith row of
## index_to_sample have the difficulty value equals to ith row of
## index_to_metric.
cl_1st_index_to_sample_path=${10:-'dummy'}
cl_1st_index_to_metric_path=${11:-'dummy'}
## During training, whether increase difficulty by value- or percentile-based.
cl_1st_difficulty_type=${12:-'value'}
## "single_cluster" means no clustering required and probably CL is achieved by
## data postprocessing. "schedule_based" means will cluster data based on the
## difficulty schedule (pacing function) below.
cl_1st_clustering_type=${13:-'single_cluster'}
## Start difficulty
cl_1st_min=${14:-512}
## End difficulty
cl_1st_max=${15:-512}
## Total step to reach end difficulty
cl_1st_total_step_in_million=${16:-1}
## When changing difficulty, always make sure it's a multiple of the
## difficulty_step below.
cl_1st_difficulty_step=${17:-1}
## Root degree of the schedule (pacing function).
cl_1st_root=${18:-1}

cl_2nd_metric=${19:-'dummy'}
cl_2nd_index_to_sample_path=${20:-'dummy'}
cl_2nd_index_to_metric_path=${21:-'dummy'}
cl_2nd_difficulty_type=${22:-'value'}
cl_2nd_clustering_type=${23:-'single_cluster'}
cl_2nd_min=${24:-2048}
cl_2nd_max=${25:-2048}
cl_2nd_total_step_in_million=${26:-1}
cl_2nd_difficulty_step=${27:-1}
cl_2nd_root=${28:-1}

# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# ## The *_index_to_sample_percentile_merged is a concatenated index for perf
# ## improvement, but it only works when you set difficulty_type="percentile" in
# ## ds_config. If you use difficulty_type="value", you need to change this to
# ## *_index_to_sample
# # cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_sample"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_bert_5epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="value"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=600
# cl_1st_max=9069
# cl_1st_total_step_in_million=0.96
# cl_1st_difficulty_step=1
# cl_1st_root=2

# cl_2nd_metric="seqlen_truncate"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=128
# cl_2nd_max=512
# cl_2nd_total_step_in_million=0.96
# cl_2nd_difficulty_step=8
# cl_2nd_root=1

cl_1st_total_step=$(calc $cl_1st_total_step_in_million*1000000)
cl_2nd_total_step=$(calc $cl_2nd_total_step_in_million*1000000)
###############################################################################
### Misc configs
log_interval=100
eval_iters=10
eval_interval=1000
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
num_save=100
estimated_train_iter=$((${train_tokens} / ${seq_len} / ${global_batch_size}))
save_interval=$((${estimated_train_iter} / ${num_save}))

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
host="${HOSTNAME}"
seed=1234
## Number of workers for dataloader. We found that for BERT pre-training,
## num_workers will greatly affect data loading time and overall training
## time. In our experiment with 64 GPUs, the performance reaches peak at
## num_workers = 4 but it may differ depending on hardware. Also note that
## larger num_workers add more CPU computation/memory overhead.
num_workers=4

## Public the Pile dataset, see ../pile_data_download_preprocess.py about how
## to download and preprocess the data. Change data_home to where you store the
## pile_bert_train_text_sentence.bin and pile_bert_train_text_sentence.idx.
data_home="/vc_data_blob/users/conglli/the_pile_bert"
if [[ "$host" == *"webxt"* ]]; then
    data_home="/blob/data/the_pile_bert"
fi
data_path="${data_home}/pile_bert_train_text_sentence"
## train_idx_path forces Megatron to use a specific data index file generated
## when we analyze data. This is needed because our index for curriculum
## learning difficulty metric is based on this data index.
train_idx_path="${data_home}/pile_bert_train_text_sentence_train_indexmap_exact5ep_509msl_0.10ssp_1234s.npy"

vocab_path="bert-large-uncased-vocab.txt"
if [ ! -f "$vocab_path" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt
fi

prescale_grad="true"
jobname="bert_${model_size}B_tok${train_tokens_in_billion}B"
jobname="${jobname}_lr${lr}_min${min_lr}_w${lr_warmup_iters}_d${lr_decay_tokens_in_billion}B_${lr_decay_style}"
jobname="${jobname}_gbs${global_batch_size}_mbs${batch_size}_g${num_gpus}"
if [[ $zero_stage -gt 0 ]]; then
    jobname="${jobname}_z${zero_stage}"
    prescale_grad="false"
fi
if [[ $mp_size -gt 1 ]]; then
    jobname="${jobname}_mp${mp_size}"
fi
if [ "${no_pp}" = "false" ]; then
    jobname="${jobname}_pp${pp_size}"
fi
jobname="${jobname}_seed${seed}"
if [ "${ltd_enabled}" = "true" ]; then
    jobname="${jobname}_ltd_${ltd_start}_${ltd_step_in_million}M_drop${dropout}"
fi
if [ "${cl_enabled}" = "true" ]; then
    jobname="${jobname}_cl_${cl_1st_metric}_${cl_1st_min}_${cl_1st_max}_${cl_1st_total_step_in_million}M_${cl_1st_root}"
    if [[ $cl_num_metric -gt 1 ]]; then
        jobname="${jobname}_${cl_2nd_metric}_${cl_2nd_min}_${cl_2nd_max}_${cl_2nd_total_step_in_million}M_${cl_2nd_root}"
    fi
fi

username=$(whoami)
output_home="/blob/users/${username}/project/data_efficient_bert"
log_path="${output_home}/log/"
checkpoint_path="${output_home}/checkpoint/${jobname}"
## Microsoft internal constraint: because tensorboard is logged by last rank,
## it's better to put the path in NFS instead of Blob.
tensorboard_dir="/vc_data/users/${username}/project/data_efficient_bert/tensorboard/"
tensorboard_path="${tensorboard_dir}${jobname}_${host}_${current_time}"
mkdir -p ${log_path}
mkdir -p ${checkpoint_path}
mkdir -p ${tensorboard_path}
if [ "${cl_enabled}" = "true" ]; then
    data_cluster_path="${output_home}/data_cluster/${jobname}"
    mkdir -p ${data_cluster_path}
fi
###############################################################################
data_options=" \
    --vocab-file ${vocab_path} \
    --data-path ${data_path} \
    --data-impl mmap"

## If CL is used, make sure to set "--split" the same as what you used during
## offline data analysis&indexing.
megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --tensor-model-parallel-size ${mp_size} \
    --init-method-std ${init_std} \
    --lr-decay-tokens ${lr_decay_tokens} \
    --lr-warmup-iters ${lr_warmup_iters} \
    --micro-batch-size ${batch_size} \
    --exit-duration-in-mins ${exit_duration} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --train-tokens ${train_tokens} \
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
    --seed ${seed} \
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

if [ "${ltd_enabled}" = "true" ]; then
megatron_options="${megatron_options} \
    --attention-dropout ${dropout} \
    --hidden-dropout ${dropout} \
    --random-ltd"
fi

if [ "${cl_enabled}" = "true" ]; then
megatron_options="${megatron_options} \
    --train-idx-path ${train_idx_path} \
    --data-efficiency-curriculum-learning"
fi

config_json="ds_config_gbs${global_batch_size}_mbs${batch_size}_log${log_interval}_zero${zero_stage}_seed${seed}"
if [ "${ltd_enabled}" = "true" ]; then
    config_json="${config_json}_ltd_${ltd_start}_${ltd_step}"
fi
if [ "${cl_enabled}" = "true" ]; then
    config_json="${config_json}_cl_${cl_1st_metric}_${cl_1st_min}_${cl_1st_max}_${cl_1st_total_step}_${cl_1st_root}"
    if [[ $cl_num_metric -gt 1 ]]; then
        config_json="${config_json}_${cl_2nd_metric}_${cl_2nd_min}_${cl_2nd_max}_${cl_2nd_total_step}_${cl_2nd_root}"
    fi
fi
config_json="${config_json}.json"
if [[ $cl_num_metric -gt 1 ]]; then
template_json="ds_config_bert_2clmetrics_TEMPLATE.json"
sed "s/GBSIZE/${global_batch_size}/" ${template_json} \
    | sed "s/MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/${prescale_grad}/" \
    | sed "s/DATA_EFFICIENCY_SEED/${seed}/" \
    | sed "s/LTD_ENABLED/${ltd_enabled}/" \
    | sed "s/LTD_MIN/${ltd_start}/" \
    | sed "s/LTD_MAX/${seq_len}/" \
    | sed "s/LTD_STEP/${ltd_step}/" \
    | sed "s/CL_ENABLED/${cl_enabled}/" \
    | sed "s/DATA_SAMPLING_NUM_WORKERS/${num_workers}/" \
    | sed "s#CL_CLUSTER_PATH#${data_cluster_path}#" \
    | sed "s#CL_1st_METRIC_NAME#${cl_1st_metric}#" \
    | sed "s#CL_1st_SAMPLE_PATH#${cl_1st_index_to_sample_path}#" \
    | sed "s#CL_1st_METRIC_PATH#${cl_1st_index_to_metric_path}#" \
    | sed "s#CL_1st_DIFF_TYPE#${cl_1st_difficulty_type}#" \
    | sed "s#CL_1st_CLUSTER_TYPE#${cl_1st_clustering_type}#" \
    | sed "s/CL_1st_MIN/${cl_1st_min}/" \
    | sed "s/CL_1st_MAX/${cl_1st_max}/" \
    | sed "s/CL_1st_TOTAL_STEP/${cl_1st_total_step}/" \
    | sed "s/CL_1st_DIFF_STEP/${cl_1st_difficulty_step}/" \
    | sed "s/CL_1st_ROOT/${cl_1st_root}/" \
    | sed "s#CL_2nd_METRIC_NAME#${cl_2nd_metric}#" \
    | sed "s#CL_2nd_SAMPLE_PATH#${cl_2nd_index_to_sample_path}#" \
    | sed "s#CL_2nd_METRIC_PATH#${cl_2nd_index_to_metric_path}#" \
    | sed "s#CL_2nd_DIFF_TYPE#${cl_2nd_difficulty_type}#" \
    | sed "s#CL_2nd_CLUSTER_TYPE#${cl_2nd_clustering_type}#" \
    | sed "s/CL_2nd_MIN/${cl_2nd_min}/" \
    | sed "s/CL_2nd_MAX/${cl_2nd_max}/" \
    | sed "s/CL_2nd_TOTAL_STEP/${cl_2nd_total_step}/" \
    | sed "s/CL_2nd_DIFF_STEP/${cl_2nd_difficulty_step}/" \
    | sed "s/CL_2nd_ROOT/${cl_2nd_root}/" \
      > ${config_json}
else
template_json="ds_config_bert_1clmetric_TEMPLATE.json"
sed "s/GBSIZE/${global_batch_size}/" ${template_json} \
    | sed "s/MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/${prescale_grad}/" \
    | sed "s/DATA_EFFICIENCY_SEED/${seed}/" \
    | sed "s/LTD_ENABLED/${ltd_enabled}/" \
    | sed "s/LTD_MIN/${ltd_start}/" \
    | sed "s/LTD_MAX/${seq_len}/" \
    | sed "s/LTD_STEP/${ltd_step}/" \
    | sed "s/CL_ENABLED/${cl_enabled}/" \
    | sed "s/DATA_SAMPLING_NUM_WORKERS/${num_workers}/" \
    | sed "s#CL_CLUSTER_PATH#${data_cluster_path}#" \
    | sed "s#CL_1st_METRIC_NAME#${cl_1st_metric}#" \
    | sed "s#CL_1st_SAMPLE_PATH#${cl_1st_index_to_sample_path}#" \
    | sed "s#CL_1st_METRIC_PATH#${cl_1st_index_to_metric_path}#" \
    | sed "s#CL_1st_DIFF_TYPE#${cl_1st_difficulty_type}#" \
    | sed "s#CL_1st_CLUSTER_TYPE#${cl_1st_clustering_type}#" \
    | sed "s/CL_1st_MIN/${cl_1st_min}/" \
    | sed "s/CL_1st_MAX/${cl_1st_max}/" \
    | sed "s/CL_1st_TOTAL_STEP/${cl_1st_total_step}/" \
    | sed "s/CL_1st_DIFF_STEP/${cl_1st_difficulty_step}/" \
    | sed "s/CL_1st_ROOT/${cl_1st_root}/" \
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

deepspeed ${dir}/../../../../pretrain_bert.py ${megatron_options} ${data_options} ${deepspeed_options} &>> ${log_path}/${jobname}_${host}_${current_time}.log
