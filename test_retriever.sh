COMMAND="/home/scratch.gcf/adlr-utils/release/cluster-interface/latest/mp_launch python hashed_index.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --batch-size 8 \
    --checkpoint-activations \
    --seq-length 288 \
    --max-position-embeddings 288 \
    --train-iters 100000 \
    --load /home/dcg-adlr-nkant-output.cosmos1203/chkpts/realm_debug \
    --ict-load /home/dcg-adlr-nkant-output.cosmos1203/chkpts/ict_best \
    --save /home/dcg-adlr-nkant-output.cosmos1203/chkpts/realm_debug \
    --data-path /home/universal-lm-data.cosmos549/datasets/wiki-indexed/wikipedia_lines \
    --titles-data-path /home/universal-lm-data.cosmos549/datasets/wiki-indexed/wikipedia_lines-titles \
    --hash-data-path /home/dcg-adlr-nkant-data.cosmos1202/hash_data/ict_best.pkl \
    --vocab-file /home/universal-lm-data.cosmos549/scratch/mshoeybi/data/albert/vocab.txt \
    --split 58,1,1 \
    --distributed-backend nccl \
    --lr 0.0001 \
    --num-workers 2 \
    --lr-decay-style linear \
    --warmup .01 \
    --save-interval 3000 \
    --fp16 \
    --adlr-autoresume \
    --adlr-autoresume-interval 100"

submit_job --image 'http://gitlab-master.nvidia.com/adlr/megatron-lm/megatron:20.03' --mounts /home/universal-lm-data.cosmos549,/home/dcg-adlr-nkant-source.cosmos1204,/home/dcg-adlr-nkant-data.cosmos1202,/home/dcg-adlr-nkant-output.cosmos1203,/home/nkant --name test_retriever --partition interactive --gpu 1 --nodes 1 --autoresume_timer 300 -c "${COMMAND}"
