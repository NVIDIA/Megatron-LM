# This is an example zero-shot eval script. Please first read the readme_evalharness.md under the same directory.

CHECKPOINT_PATH=/blob/users/conglli/project/gpt3_with_pile/checkpoint/gpt3-with-pile-0.125B-lr-2.4e-3-minlr-6.0e-5-bs-2048-gpus-128-zero-0-mp-1-pp-1-no_pp-cl-startseqlen-72-step-20728-token-45B/global_step81566/
CONFIG_PATH=ds_config_gpt3-with-pile-0.125B-lr-2.4e-3-minlr-6.0e-5-bs-2048-gpus-128-zero-0-mp-1-pp-1-no_pp-cl-startseqlen-72-step-20728-token-45B.json
RESULT_PATH=gpt3-with-pile-0.125B-lr-2.4e-3-minlr-6.0e-5-bs-2048-gpus-128-zero-0-mp-1-pp-1-no_pp-cl-startseqlen-72-step-20728-token-45B_global_step81566.log

PP_SIZE=1
TP_SIZE=1
NO_PP="true"
EP_PARALLEL_SIZE=1
# Currently eval harness does not support data parallel
# However, for MoE models it's possible to enable a "fake data parallel"
# in order to load experts on multiple gpus. At the same time, it's not
# real data parallel because we load the same data on all gpus.
# On the other hand, it's better to use less number of gpus than training,
# to reduce communication overhead.
NUM_NODE=1
NUM_GPU_PER_NODE=1

TASKS="lambada"
# WikiText-2, not used in GPT-3 paper but used in GPT-2 paper
# TASKS="wikitext"
# Tasks that appeared in GPT-3 paper (sorted based on the order in paper), plus WikiText-2.
# TASKS="hellaswag,lambada,triviaqa,webqs,winogrande,piqa,arc_challenge,arc_easy,openbookqa,race,boolq,cb,copa,rte,wic,wsc,multirc,record,anli_r1,anli_r2,anli_r3,wikitext"
# All tasks that confirmed to work, there are more tasks on https://github.com/EleutherAI/lm-evaluation-harness that we didn't test.
# TASKS="hellaswag,lambada,triviaqa,webqs,winogrande,piqa,arc_challenge,arc_easy,openbookqa,race,boolq,cb,copa,rte,wic,wsc,multirc,record,anli_r1,anli_r2,anli_r3,wikitext,logiqa,mathqa,mc_taco,mrpc,prost,pubmedqa,qnli,qqp,sciq,sst,wnli"

VOCAB_FILE=/data/Megatron-LM/data/gpt2-vocab.json
MERGE_FILE=/data/Megatron-LM/data/gpt2-merges.txt

# export HF_DATASETS_OFFLINE=1

# Dummy arguments to make megatron happy. No need to configure them.
# The reason we don't need to configure them and many other arguments is
# because the eval framework will read the arguments from checkpoint file.
MEGATRON_REQUIRED_ARGS="\
    --num-layers -1\
    --hidden-size -1\
    --num-attention-heads -1\
    --seq-length -1 \
    --max-position-embeddings -1
"

CMD="../../tasks/eval_harness/evaluate.py \
    --load $CHECKPOINT_PATH\
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE\
    --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
    --vocab-file $VOCAB_FILE\
    --merge-file $MERGE_FILE\
    --micro-batch-size 12\
    --no-load-optim \
    --no-load-rng \
    --inference \
    --disable-moe-token-dropping \
    --tokenizer-type GPT2BPETokenizer \
    --adaptive_seq_len\
    --eval_fp32\
    --task_list $TASKS\
    --results_path $RESULT_PATH \
    --deepspeed \
    --deepspeed_config $CONFIG_PATH \
    $MEGATRON_REQUIRED_ARGS\
    "

if [[ "${NO_PP}" = "true" ]]; then
CMD="${CMD} \
    --no-pipeline-parallel"
fi

LAUNCHER="deepspeed --num_nodes $NUM_NODE --num_gpus $NUM_GPU_PER_NODE"
$LAUNCHER $CMD