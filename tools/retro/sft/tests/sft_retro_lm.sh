#!/bin/bash
# bash examples/qa/finetune_normal_lm.sh landrover_tasb_retrieved 843m 1 3e-6 1

blend_name=$1
model_size=$2
global_bsz=$3
lr=$4
ft_neighbours=1
model_card=pp1
ckpt=$5
TASK=none

train_iters=1000


DATA_HOME="/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/"
data_folder="$DATA_HOME"

SFT_HOME="/lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM"

TOKENIZER_MODEL="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"


if [[ $model_size == "843m" ]]; then
    mod_par=1
    layers=24
    hid_dim=1024
    heads=16
    pip_par=1
fi

if [[ $model_size == "43b" ]]; then
    mod_par=8
    layers=48
    hid_dim=8192
    heads=64
    pip_par=4
    if [[ $model_card == *pp1* ]]; then
        pip_par=1
    fi
fi

GPT_ARGS="--apply-layernorm-1p \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --no-position-embedding \
        --use-rotary-position-embeddings \
        --rotary-percent 0.5 \
        --swiglu \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --pipeline-model-parallel-size $pip_par \
        --tensor-model-parallel-size $mod_par \
        --num-layers $layers \
        --hidden-size $hid_dim \
        --num-attention-heads $heads \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --lr-decay-style cosine \
        --tokenizer-type GPTSentencePieceTokenizer \
        --tokenizer-model ${TOKENIZER_MODEL} \
        --clip-grad 1.0 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.98 \
        --log-params-norm \
        --log-num-zeros-in-grad \
        --bf16 \
"

if [[ $model_card == *pp1* ]]; then
    GPT_ARGS+=" --use-distributed-optimizer"
fi

FT_ARGS="--eod-mask-loss \
    --answer-loss-only \
    --ft_neighbours ${ft_neighbours} \
    --task $TASK"

num_nodes=1
num_gpus=8

if [[ $model_size == "843m" ]]; then
    num_nodes=1
    lr=5e-6
    min_lr=5e-6
fi


if [[ $model_size == "43b" ]]; then
    num_nodes=64
    lr=5e-6
    min_lr=5e-6
fi

PRETRAINED_CHECKPOINT=${ckpt}

SAVENAME="retro-${blend_name}_${model_card}_same_format_ctx${ft_neighbours}_${model_size}_${global_bsz}_${lr}"
CHECKPOINT_PATH="${SFT_HOME}/checkpoints/applications/${SAVENAME}"
TENSORBOARD_DIR="${SFT_HOME}/tensorboard/${SAVENAME}"
mkdir -p ${TENSORBOARD_DIR}

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 200 \
             --tensorboard-dir ${TENSORBOARD_DIR} \
             --log-validation-ppl-to-tensorboard \
             --eval-iters 100"

. ./tools/retro/sft/tests/${blend_name}.sh

RETRO_WORKDIR=/lustre/fsw/adlr/adlr-nlp/boxinw/next-llm
K=2

options=" \
    $GPT_ARGS \
    --retro-workdir ${RETRO_WORKDIR} \
    --retro-add-retriever \
    --retro-num-neighbors ${K} \
    --retro-attention-gate 0 \
    --data-path ${DATA_BLEND} \
    --data-folder ${data_folder} \
    --recompute-activations \
    --lr $lr \
    --micro-batch-size 1 \
    --global-batch-size ${global_bsz} \
    --min-lr ${min_lr} \
    --retro-cyclic-train-iters ${train_iters} \
    --train-iters ${train_iters} \
    --dataloader-type cyclic \
    --save $CHECKPOINT_PATH \
    $OUTPUT_ARGS \
    $FT_ARGS"

if [[ -d "$CHECKPOINT_PATH" ]]; then
  options="$options \
      --load $CHECKPOINT_PATH "
else
  echo $PRETRAINED_CHECKPOINT
  options="$options \
      --load $PRETRAINED_CHECKPOINT \
      --finetune \
      --no-load-rng \
      --no-load-optim "
fi

DIR=`pwd`
# -m torch.distributed.launch --nproc_per_node 8
run_cmd="python -u ${DIR}/tools/retro/sft/sft_retro.py ${options}"
# srun -l \
#      --container-image "gitlab-master.nvidia.com/adlr/megatron-lm/boxinw/faissgpu" \
#      --container-mounts "/home/pengx/projects/retro/:/home/pengx/projects/retro/" \
#      --output=$DIR/logs/%x_%j_$DATETIME.log sh -c "${run_cmd}"
# $run_cmd

export SUBMIT_LOGS="${SFT_HOME}/megatron-lm/logs"
mkdir -p $SUBMIT_LOGS
export NCCL_DEBUG=INFO

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

DOCKER="gitlab-master.nvidia.com/adlr/megatron-lm/boxinw/retro.23.04"
MOUNTS="/lustre/fsw/"
PARTITION="luna"
LAUNCH="${ADLR_UTILS}/mp_launch"

echo ${run_cmd}
submit_job --gpu ${num_gpus} --nodes ${num_nodes} --email_mode never  --mounts $MOUNTS --partition $PARTITION  --image $DOCKER -c "$LAUNCH ${run_cmd}" -n "${SAVENAME}" --duration 3  # --dependent_clones 1
