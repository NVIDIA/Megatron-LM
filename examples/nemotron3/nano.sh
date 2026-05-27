# Megatron-LM legacy-CLI launch line for Nemotron-3 Nano (30B-A3B MoE).
#
# Faithful 1:1 with the canonical Megatron-Bridge recipe at
# Megatron-Bridge/src/megatron/bridge/recipes/nemotronh/nemotron_3_nano.py
# ::nemotron_3_nano_pretrain_config.
#
# This file is the legacy ``--hybrid-layer-pattern`` baseline; the recommended
# path is the Python recipe at examples/nemotron3/nano.py
# (entry point: ``--model-recipe examples.nemotron3.nano``). Both produce
# byte-identical models when seeded the same way.
#
# Usage (assumes a running torchrun environment):
#   torchrun ... pretrain_hybrid.py $(cat examples/nemotron3/nano.sh)

  --use-mcore-models \
  --transformer-impl transformer_engine \
  --group-query-attention \
  --squared-relu \
  --bf16 \
  --first-last-layers-bf16 \
  --use-fused-weighted-squared-relu \
  --split 9999,8,2 \
  --cuda-graph-scope full \
  --cuda-graph-impl none \
  --cuda-graph-warmup-steps 3 \
  --seq-length 8192 \
  --max-position-embeddings 8192 \
  --seed 1234 \
  --micro-batch-size 2 \
  --global-batch-size 3072 \
  --train-iters 39735 \
  --manual-gc-interval 0 \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --sequence-parallel \
  --context-parallel-size 1 \
  --expert-model-parallel-size 8 \
  --expert-tensor-parallel-size 1 \
  --tp-comm-overlap \
  --cross-entropy-loss-fusion \
  --cross-entropy-fusion-impl native \
  --no-overlap-p2p-communication \
  --num-layers 52 \
  --hybrid-layer-pattern "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME" \
  --hidden-size 2688 \
  --num-attention-heads 32 \
  --attention-backend fused \
  --num-query-groups 2 \
  --ffn-hidden-size 1856 \
  --kv-channels 128 \
  --hidden-dropout 0.0 \
  --attention-dropout 0.0 \
  --norm-epsilon 1e-05 \
  --disable-bias-linear \
  --normalization RMSNorm \
  --init-method-std 0.0173 \
  --no-rope-fusion \
  --mamba-num-heads 64 \
  --mamba-state-dim 128 \
  --mamba-head-dim 64 \
  --mamba-num-groups 8 \
  --num-experts 128 \
  --moe-shared-expert-intermediate-size 3712 \
  --moe-layer-freq 1 \
  --moe-ffn-hidden-size 1856 \
  --moe-router-load-balancing-type seq_aux_loss \
  --moe-router-topk 6 \
  --moe-router-num-groups 1 \
  --moe-router-group-topk 1 \
  --moe-router-topk-scaling-factor 2.5 \
  --moe-router-score-function sigmoid \
  --moe-router-dtype fp32 \
  --moe-router-enable-expert-bias \
  --moe-router-bias-update-rate 0.001 \
  --moe-grouped-gemm \
  --moe-aux-loss-coeff 0.0001 \
  --moe-token-dispatcher-type flex \
  --moe-flex-dispatcher-backend deepep \
  --moe-hybridep-num-sms 16 \
  --moe-permute-fusion \
  --untie-embeddings-and-output-weights \
  --position-embedding-type none \
  --rotary-percent 1.0 \
  --rotary-base 10000 \
  --make-vocab-size-divisible-by 128 \
  --lr 0.0016 \
  --min-lr 1.6e-05 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-08 \
  --clip-grad 1.0 \
  --grad-reduce-in-fp32 \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --use-distributed-optimizer \
  --eval-iters 32 \
  --eval-interval 500 \
  --lr-decay-style cosine \
  --lr-warmup-iters 333 \
  --lr-warmup-samples 0 \
  --lr-warmup-init 0.0 \
  --override-opt-param-scheduler \
  --start-weight-decay 0.1 \
  --end-weight-decay 0.1 \
  --weight-decay-incr-style constant \
  --dataloader-type single \
  --num-workers 8 \
  --num-dataset-builder-threads 1 \
  --no-mmap-bin-files \
  --log-interval 10 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --save-interval 200 \
  --ckpt-format torch_dist \
  --use-persistent-ckpt-worker \
  --dist-ckpt-strictness log_all \
  --distributed-timeout-minutes 10
