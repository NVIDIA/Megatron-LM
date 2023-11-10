#!/bin/bash

TASK=$1
model_size=$2
sampling=$3
split=$4
gen_start=$5
num_gen=$6
ckpt_step=${7}
ft_neighbours=${8}
model_card=${9}
ckpt=${10}
K=${11}
retrieve=${12}

QA_HOME="/lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM"

TOKENIZER_MODEL="/lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"

RETRO_WORKDIR=/lustre/fsw/adlr/adlr-nlp/boxinw/next-llm


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

num_nodes=1
num_gpus=8

sample_input_file="/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/$TASK/${split}.json"
DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/boxinw/instruction_tuning_data/$TASK"
FEWSHOT_INPUT_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa"

if [[ $TASK == "nq" ]]; then
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ/${split}.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/single-turn-qa/NQ/fewshot_samples.json"
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/pengx/retro/data/NQ"
fi

if [[ $TASK == "doc2dial" ]]; then
    DATA_FOLDER="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK"
    sample_input_file="/lustre/fsw/adlr/adlr-nlp/zihanl/datasets/foundational-qa/multi-turn-qa/$TASK/${TASK}_ftdragon_chatgptgen7k_chunk150_QA_test.json"
    fewshot_input_file="${FEWSHOT_INPUT_FOLDER}/multi-turn-qa/doc2dial/fewshot_samples.json"
fi

top_k=1
micro_bsz=1
SAMPLE_ARGS="--top_k $top_k"

if [[ $sampling == "beam" ]]; then
    micro_bsz=1
    SAMPLE_ARGS="--beam-search"
fi

CHECKPOINT_PATH=${ckpt}
sample_output_file="${CHECKPOINT_PATH}/retro-generate-${TASK}_${ft_neighbours}_${K}_${model_size}_${split}_${sampling}_${gen_start}_${num_gen}_${ckpt_step}.txt"

DIR=`pwd`

echo $sample_input_file
echo $sample_output_file


GEN_ARGS="$SAMPLE_ARGS \
          --gen-start-idx $gen_start \
          --num-gen $num_gen \
          --ckpt-step ${ckpt_step} \
          --sample-input-file $sample_input_file \
          --sample-output-file $sample_output_file \
          --retro-workdir ${RETRO_WORKDIR} \
          --retro-add-retriever \
          --retro-num-neighbors ${K} \
          --reuse-top \
          --retro-attention-gate 0 \
          "

if [[ $retrieve == 1 ]]; then
    GEN_ARGS="$GEN_ARGS \
          --use-retrieved-neighbours \
          "
fi

FT_ARGS="--eod-mask-loss \
    --answer-loss-only \
    --ft_neighbours ${ft_neighbours} \
    --task $TASK"

DISTRIBUTED_ARGS="--nproc_per_node ${mod_par} \
                  --nnodes ${pip_par} \
                  --node_rank 0 \
                  --master_port 8889"

COMMAND="python -m torch.distributed.run $DISTRIBUTED_ARGS ${DIR}/tools/retro/text_generation/retro_text_generation.py"

COMMAND="$COMMAND \
       $GPT_ARGS \
       $GEN_ARGS \
       --load $CHECKPOINT_PATH \
       --micro-batch-size $micro_bsz \
       $FT_ARGS"

export SUBMIT_LOGS="${QA_HOME}/megatron-lm/logs"
mkdir -p $SUBMIT_LOGS
export NCCL_DEBUG=INFO

export NCCL_IB_TIMEOUT=19
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

MOUNTS="/lustre/fsw/adlr/adlr-nlp/"
PARTITION="luna"
DOCKER="gitlab-master.nvidia.com/adlr/megatron-lm/boxinw/retro.23.04"

submit_job --gpu ${mod_par} --nodes ${pip_par} --email_mode never  --mounts $MOUNTS --partition $PARTITION --image $DOCKER  -c "$COMMAND" -n "generate_${model_size}_${TASK}" --duration 4
# $COMMAND
# -m torch.distributed.launch $DISTRIBUTED_ARGS 
