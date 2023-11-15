# DISTRIBUTED_ARGS=(
#     --nproc_per_node 8
#     --nnodes 1
#     --master_addr "localhost"
#     --master_port "12345"
# )
# torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \

python pretrain_gpt.py \
--tensor-model-parallel-size 2 \
--pipeline-model-parallel-size 2 \
--seq-length 2048 \
--max-position-embeddings 2048 \
--tokenizer-type "Llama2Tokenizer" \
--tokenizer-model "../../cap/data-capability/wd/INPUT_tokenizer/tokenizer.model" \
--load "../../cap/data-capability/wd/INPUT_load_initial_model/" \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--no-initialization \
--no-load-optim \
--no-load-rng \
--bf16 \
--untie-embeddings-and-output-weights \
--use-rotary-position-embeddings \
--normalization "RMSNorm" \
--no-position-embedding \
--no-masked-softmax-fusion \
--no-query-key-layer-scaling \
--micro-batch-size 2 \
--global-batch-size 32 \
--train-iters 3000 \
--lr 0.0003 \
--lr-decay-style "cosine" \
--min-lr 0.00003  \
--lr-warmup-iters 20 \
--weight-decay 0.1 \
--clip-grad 1.0 \
--save "../../cap/data-capability/wd/model_dir" \
--data-path 1.0 "../../cap/data-capability/wd/INPUT_data/ar_books_split_00_text_document" \
--data-cache-path "../../cap/data-capability/wd/cache/" \
--split "9998,1,1" \
--log-interval 10 \
--log-validation-ppl-to-tensorboard \
--save-interval 200 \
--eval-interval 200 \
--eval-iters 100 \
--tensorboard-dir "../../cap/data-capability/wd/model_dir/tensorboard" \
--tensorboard-log-interval 100 \
--no-async-tensor-model-parallel-allreduce \
--use-flash-attn