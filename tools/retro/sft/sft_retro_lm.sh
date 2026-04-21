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


DATA_HOME="<path/to/instruction/tuning/data/directory>"
data_folder="$DATA_HOME"

SFT_HOME="<path/to/megatron/repo>"

TOKENIZER_MODEL="<path/to/gpt/tokenizer/model>"

RETRO_WORKDIR="<path/to/retro/workdir>"

K=2

PRETRAINED_CHECKPOINT=${ckpt}

SAVENAME="retro-${blend_name}_${model_card}_same_format_ctx${ft_neighbours}_${model_size}_${global_bsz}_${lr}"
CHECKPOINT_PATH="${SFT_HOME}/checkpoints/applications/${SAVENAME}"
TENSORBOARD_DIR="${SFT_HOME}/tensorboard/${SAVENAME}"
mkdir -p ${TENSORBOARD_DIR}

. ./tools/retro/sft/"${blend_name}".sh


if [[ $model_size == "843m" ]]; then
    # model param
    mod_par=1
    layers=24
    hid_dim=1024
    heads=16
    pip_par=1

    # node param
    num_nodes=1
    lr=5e-6
    min_lr=5e-6
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
        --use-distributed-optimizer \
"

FT_ARGS="--eod-mask-loss \
    --answer-loss-only \
    --ft_neighbours ${ft_neighbours} \
    --task $TASK"


OUTPUT_ARGS="--log-interval 10 \
             --save-interval 500 \
             --eval-interval 200 \
             --tensorboard-dir ${TENSORBOARD_DIR} \
             --log-validation-ppl-to-tensorboard \
             --eval-iters 100"

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

######## Command. ########

run_cmd="python -u ${SFT_HOME}/tools/retro/sft/sft_retro.py ${options}"

export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPROCS=8
CMD="\
    pwd && cd ${SFT_HOME} && pwd && \
    export PYTHONPATH=$PYTHONPATH:${SFT_HOME} && \
    python -m torch.distributed.run \
    --nproc_per_node ${NPROCS} \
    --nnodes 1 \
    --node_rank 0 \
    --master_port 6000 \
    ${run_cmd} \
"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "CMD = '$CMD'."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~"
eval $CMD

