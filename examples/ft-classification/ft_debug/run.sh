# TOKENIZER_MODEL=$1
# BIN_IDX_PATH=$2
# DATA_CACHE=$3
# CHECKPOINT_DIR=$4
# TENSORBOARD_LOGS_PATH=$5

# # DISTRIBUTED_ARGS=(
# #     --nproc_per_node $GPUS_PER_NODE 
# #     --nnodes $NUM_NODES 
# #     --master_addr $MASTER_ADDR 
# #     --master_port $MASTER_PORT
# # )
# TRANSFORMER_ENG_ARGS=(
#     # --fp8-format
#     # --fp8-margin
#     # --fp8-interval
#     # --fp8-amax-history-len
#     # --fp8-amax-compute-algo
#     # --no-fp8-wgrad
#     # --transformer-impl transformer_engine
#     # sequence parallel on transformer engine
# )
#     # --spec megatron.core.models.gpt.allam_layer_specs allam_layer_with_transformer_engine_spec
# GPT_MODEL_ARGS=(
#     --use-mcore-models
#     --seq-length 2048 
#     --max-position-embeddings 2048 
#     --num-layers 24
#     --hidden-size 2048
#     --ffn-hidden-size 3072
#     --num-attention-heads 16
#     --hidden-dropout 0.0
#     --attention-dropout 0.0
#     --make-vocab-size-divisible-by 128
#     --norm-epsilon 1.0e-05
#     --disable-bias-linear
#     --swiglu
#     --tokenizer-type Llama2Tokenizer
#     --untie-embeddings-and-output-weights
#     --use-rotary-position-embeddings
#     --normalization RMSNorm
#     --no-position-embedding
#     --no-masked-softmax-fusion
#     --no-query-key-layer-scaling
# )

# LOGISTICS_ARGS=(
#     --save $CHECKPOINT_DIR 
#     --tokenizer-model $TOKENIZER_MODEL
#     --split 99990,8,2 
#     --log-interval 100
#     --save-interval 5000 
#     --eval-interval 1000 
#     --eval-iters 50
#     --tensorboard-dir $TENSORBOARD_LOGS_PATH 
#     --tensorboard-log-interval 100
#     --data-cache-path $DATA_CACHE
#     --log-validation-ppl-to-tensorboard 
#     --seed 1234
# )

# TRAINING_ARGS=(
#     --micro-batch-size 8
#     --global-batch-size 512
#     --train-iters 286000
#     --lr 0.0002 
#     --lr-decay-style cosine 
#     --weight-decay 0.1 
#     --adam-beta1 0.9 
#     --adam-beta2 0.95 
#     --init-method-std 0.01
#     --clip-grad 1.0 
#     --min-lr 2.0e-05
#     --lr-warmup-iters 400
#     --use-flash-attn
#     --bf16
# )


# MODEL_PARALLEL_ARGS=(
# 	--tensor-model-parallel-size 1
#     --pipeline-model-parallel-size 1
#     --no-async-tensor-model-parallel-allreduce
# )

# DATA_PATH=(
#     --data-path 1.0 $BIN_IDX_PATH/00.jsonl_text_document_dc\=7021342_sc\=7021342_tc\=13216350335
# 1.0 $BIN_IDX_PATH/01.jsonl_text_document_dc\=7021312_sc\=7021312_tc\=13192588541
# 1.0 $BIN_IDX_PATH/02.jsonl_text_document_dc\=7023178_sc\=7023178_tc\=13180654647
# 1.0 $BIN_IDX_PATH/03.jsonl_text_document_dc\=7020288_sc\=7020288_tc\=13180292856
# 1.0 $BIN_IDX_PATH/04.jsonl_text_document_dc\=7019454_sc\=7019454_tc\=13154424129
# 1.0 $BIN_IDX_PATH/05.jsonl_text_document_dc\=7017130_sc\=7017130_tc\=13204113790
# 1.0 $BIN_IDX_PATH/06.jsonl_text_document_dc\=7021715_sc\=7021715_tc\=13222972231
# 1.0 $BIN_IDX_PATH/07.jsonl_text_document_dc\=7020629_sc\=7020629_tc\=13271324679
# 1.0 $BIN_IDX_PATH/08.jsonl_text_document_dc\=7019675_sc\=7019675_tc\=13188918882
# 1.0 $BIN_IDX_PATH/09.jsonl_text_document_dc\=7025523_sc\=7025523_tc\=13185447117
# 1.0 $BIN_IDX_PATH/10.jsonl_text_document_dc\=7023035_sc\=7023035_tc\=13202537584
# 1.0 $BIN_IDX_PATH/11.jsonl_text_document_dc\=7014863_sc\=7014863_tc\=13195477232
# 1.0 $BIN_IDX_PATH/12.jsonl_text_document_dc\=7018266_sc\=7018266_tc\=13231916405
# 1.0 $BIN_IDX_PATH/13.jsonl_text_document_dc\=7018250_sc\=7018250_tc\=13176952976
# 1.0 $BIN_IDX_PATH/14.jsonl_text_document_dc\=7019039_sc\=7019039_tc\=13195505586
# 1.0 $BIN_IDX_PATH/15.jsonl_text_document_dc\=7017302_sc\=7017302_tc\=13249197248
# 1.0 $BIN_IDX_PATH/16.jsonl_text_document_dc\=7019911_sc\=7019911_tc\=13250578136
# 1.0 $BIN_IDX_PATH/17.jsonl_text_document_dc\=7024783_sc\=7024783_tc\=13267409500
# 1.0 $BIN_IDX_PATH/18.jsonl_text_document_dc\=7021699_sc\=7021699_tc\=13256048253
# 1.0 $BIN_IDX_PATH/19.jsonl_text_document_dc\=7022783_sc\=7022783_tc\=13258500817
# 1.0 $BIN_IDX_PATH/20.jsonl_text_document_dc\=7020245_sc\=7020245_tc\=13174906477
# 1.0 $BIN_IDX_PATH/21.jsonl_text_document_dc\=7020057_sc\=7020057_tc\=13258376058
# 1.0 $BIN_IDX_PATH/22.jsonl_text_document_dc\=7017578_sc\=7017578_tc\=13272712232
# 1.0 $BIN_IDX_PATH/23.jsonl_text_document_dc\=7022460_sc\=7022460_tc\=13258551739
# 1.0 $BIN_IDX_PATH/24.jsonl_text_document_dc\=7019701_sc\=7019701_tc\=13179413045
# 1.0 $BIN_IDX_PATH/25.jsonl_text_document_dc\=7014308_sc\=7014308_tc\=13158260474
# 1.0 $BIN_IDX_PATH/26.jsonl_text_document_dc\=7020461_sc\=7020461_tc\=13167180838
# 1.0 $BIN_IDX_PATH/27.jsonl_text_document_dc\=7017749_sc\=7017749_tc\=13189021404
# 1.0 $BIN_IDX_PATH/28.jsonl_text_document_dc\=7017905_sc\=7017905_tc\=13193366378
# 1.0 $BIN_IDX_PATH/29.jsonl_text_document_dc\=7024343_sc\=7024343_tc\=13186702029
# )

# # torchrun ${\DISTRIBUTED_ARGS[@]}\ pretrain_gpt.py\ \
# python pretrain_gpt.py \
#     ${GPT_MODEL_ARGS[@]} \
#     ${LOGISTICS_ARGS[@]} \
#     ${TRAINING_ARGS[@]} \
#     ${MODEL_PARALLEL_ARGS[@]} \
#     ${DATA_PATH[@]}
  
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
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --bf16  2>&1| tee  $CHECKPOINT_PATH/log.txt