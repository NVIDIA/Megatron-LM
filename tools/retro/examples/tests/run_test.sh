# Preprocess data

## Single-node interactive node

bash preprocess_data_wikipedia.sh  db-build
bash preprocess_data_wikipedia.sh  index-train
bash preprocess_data_wikipedia.sh  query-pretraining-neighbors

# Pretraining

## Single-node interactive node

bash tools/retro/examples/tests/pretrain_model_wiki.sh

## Multi-node run with sbatch

sbatch tools/retro/examples/tests/pretrain-nextllm-800m-retro.sh
sbatch tools/retro/examples/tests/pretrain-nextllm-800m-gpt.sh
sbatch tools/retro/examples/tests/pretrain-nextllm-43b-retro.sh

## Check the training curves and see whether they are aligned

python -m torch.distributed.run --nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000 pretrain_retro.py --sequence-parallel --recompute-activations --use-flash-attn --apply-layernorm-1p --untie-embeddings-and-output-weights --disable-bias-linear --no-position-embedding --use-rotary-position-embeddings --rotary-percent 0.5 --swiglu --attention-dropout 0.0 --hidden-dropout 0.0 --exit-duration-in-mins 220 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --save-interval 2000 --save /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting-github-mr --load /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/checkpoints/gpt3-843m-multi-1.1t-gtc-llr --no-load-optim --finetune --tensorboard-dir /lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting-github-mr/tensorboard --log-validation-ppl-to-tensorboard --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --seq-length 4096 --max-position-embeddings 4096 --micro-batch-size 2 --global-batch-size 128 --train-samples 25000000 --lr-decay-samples 23750000 --lr-warmup-samples 16667 --lr 2.5e-5 --min-lr 2.5e-6 --lr-decay-style cosine --log-interval 100 --eval-iters 32 --eval-interval 1260 --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /lustre/fsw/adlr/adlr-nlp/adlr-nlp-sharing/nvllm-1.1t/utils/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model --data-path 0.01920 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/Books3_shuf_text_document 0.01602 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/OpenWebText2_shuf_text_document 0.00751 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/StackExchange_shuf_text_document 0.00324 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/PubMedAbs_shuf_text_document 0.00653 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/Wikipedia_shuf_text_document 0.00193 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/Gutenberg_shuf_text_document 0.00117 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/BookCorpus2_shuf_text_document 0.00023 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/NIHExporter_shuf_text_document 0.01143 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/ArXiv_shuf_text_document 0.00366 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/Stories_shuf_text_document 0.03992 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/BigScience/BigScience_shuf_text_document 0.04768 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/Reddit-Plus/Reddit_all_dialogue_shuf_text_document 0.07199 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/CC-NEWS/CC-NEWS_shuf_text_document 0.02180 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/Pile-CC_shuf_text_document 0.07633 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/CC-MAIN-2020-50/CC-MAIN-2020-50_shuf_text_document 0.07644 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/CC-MAIN-2022-40/CC-MAIN-2022-40_00_shuf_text_document 0.07644 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/CC-MAIN-2022-40/CC-MAIN-2022-40_01_shuf_text_document 0.09414 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/CC-MAIN-2019-35/CC-MAIN-2019-35_shuf_text_document 0.03890 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/MTNLG/CC-2021-04_shuf_text_document 0.08544 /lustre/fsw/adlr/adlr-nlp/boxinw/retro/data/english/mc4-en_1T-url/mc4-en_shuf_text_document --split 98,2,0 --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --init-method-std 0.007 --log-params-norm --log-num-zeros-in-grad --bf16 --retro-fix-sub-epoch --retro-workdir /lustre/fsw/adlr/adlr-nlp/boxinw/next-llm --retro-add-retriever
