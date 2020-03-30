#!/bin/bash
LENGTH=512
CHKPT="chkpts/debug"
COMMAND="/home/scratch.gcf/adlr-utils/release/cluster-interface/latest/mp_launch python pretrain_bert_ict.py \
       --num-layers 6 \
       --hidden-size 768\
       --num-attention-heads 12 \
       --batch-size 1 \
       --checkpoint-activations \
       --seq-length $LENGTH \
       --max-position-embeddings $LENGTH \
       --train-iters 1000 \
       --no-save-optim --no-save-rng \
       --save $CHKPT \
       --resume-dataloader \
       --train-data /home/universal-lm-data.cosmos549/datasets/wikipedia/wikidump_lines.json \
       --presplit-sentences \
       --loose-json \
       --text-key text \
       --data-loader lazy \
       --tokenizer-type BertWordPieceTokenizer \
       --cache-dir cache \
       --split 58,1,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --num-workers 0 \
       --no-load-optim --finetune \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --save-interval 1000 \
       --fp16 --adlr-autoresume --adlr-autoresume-interval 5000"
submit_job --image 'http://gitlab-master.nvidia.com/adlr/megatron-lm/megatron:rouge_score' --mounts /home/universal-lm-data.cosmos549,/home/raulp -c "${COMMAND}" --name ict_test --partition interactive --gpu 8 --nodes 2 --autoresume_timer 300 -i
