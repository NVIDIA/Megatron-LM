 WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

# TRAIN_DATA="$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/outputs/data/MNLI/train.tsv"
# VALID_DATA="$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/outputs/data/MNLI/dev_matched.tsv \
#             $AZUREML_CR_EXECUTION_WORKING_DIR_PATH/outputs/data/MNLI/dev_mismatched.tsv"
TRAIN_DATA="$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/outputs/data/FILTER/alpaca_data_train.json"
VALID_DATA="$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/outputs/data/FILTER/alpaca_data_valid.json"
# PRETRAINED_CHECKPOINT=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/outputs/models/bert_cased_345m
# PRETRAINED_CHECKPOINT=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/outputs/models/gpt_345m
PRETRAINED_CHECKPOINT=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/outputs/models/allam_pile_1b
# VOCAB_FILE=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/outputs/models/gpt_345m/gpt2-vocab.json
# MERGE_FILE=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/outputs/models/gpt_345m/gpt2-merges.txt
MODEL_FILE=$AZUREML_CR_DATA_CAPABILITY_PATH/INPUT_tokenizer_model_path/ar_en.model
# VOCAB_FILE=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/outputs/models/bert_cased_345m/bert-large-cased-vocab.txt
CHECKPOINT_PATH=$AZUREML_CR_EXECUTION_WORKING_DIR_PATH/outputs/checkpoints/allam_1b_filter
mkdir -p $CHECKPOINT_PATH
GPT_MODEL_ARGS=(
    --seq-length 2048 
    --max-position-embeddings 2048 
    --num-layers 24
    --hidden-size 2048
    --ffn-hidden-size 3072
    --num-attention-heads 16
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --make-vocab-size-divisible-by 128
    --norm-epsilon 1.0e-05
    --disable-bias-linear
    --swiglu
    --tokenizer-type Llama2Tokenizer
    --untie-embeddings-and-output-weights
    --use-rotary-position-embeddings
    --normalization RMSNorm
    --no-position-embedding
    --no-masked-softmax-fusion
    --no-query-key-layer-scaling
)
# --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
torchrun $DISTRIBUTED_ARGS allam-megatron/tasks/main.py \
               --task FILTER \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-model $MODEL_FILE \
               --epochs 5 \
               --tensor-model-parallel-size 1 \
               ${GPT_MODEL_ARGS[@]} \
               --micro-batch-size 4 \
               --lr 5.0e-5 \
               --lr-decay-style linear \
               --lr-warmup-fraction 0.065 \
               --save-interval 500000 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --save $CHECKPOINT_PATH \
               --log-interval 10 \
               --eval-interval 1000 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --bf16  2>&1| tee  $CHECKPOINT_PATH/log.txt