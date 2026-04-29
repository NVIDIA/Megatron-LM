# ═══════════════════════════════════════════════════════════
# Megatron-LM pretrain_gpt.py args (translated from Bridge)
# ═══════════════════════════════════════════════════════════
#
# NOTE: squared_relu: model.activation_func=squared_relu → --squared-relu
# NOTE: max-position-embeddings not set; defaulting to seq-length for MLM assert
#
  --use-mcore-models \
  --transformer-impl transformer_engine \
  --group-query-attention \
  --squared-relu \
  --bf16 \
  --split 9999,8,2 \
  --cuda-graph-scope full \
  --cuda-graph-impl none \
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
  --cross-entropy-loss-fusion \
  --cross-entropy-fusion-impl native \
  --no-overlap-p2p-communication \
  --num-layers 52 \
  --mtp-num-layers 0 \
  --mtp-loss-scaling-factor 0.1 \
  --hidden-size 2688 \
  --num-attention-heads 32 \
  --attention-backend AttnBackend.fused \
  --num-query-groups 2 \
  --ffn-hidden-size 1856 \
  --kv-channels 128 \
  --hidden-dropout 0.0 \
  --attention-dropout 0.0 \
  --norm-epsilon 1e-05 \
  --disable-bias-linear \
  --num-experts 128 \
  --normalization RMSNorm \
  --init-method-std 0.0173 \
  --no-rope-fusion \
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
  --moe-permute-fusion \
  --cuda-graph-warmup-steps 3 \
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
  --lr-wsd-decay-style minus_sqrt \
  --lr-warmup-iters 333 \
  --lr-warmup-samples 0 \
  --lr-warmup-init 0.0 \
  --override-opt-param-scheduler \
  --start-weight-decay 0.033 \
  --end-weight-decay 0.033 \
  --weight-decay-incr-style constant \
  --dataloader-type single \
  --num-workers 8 \
  --num-dataset-builder-threads 1 \
  --no-mmap-bin-files \
  --log-interval 10 \
  --tensorboard-dir /workspace/Megatron-Bridge/nemo_experiments/default/tb_logs \
  --log-timers-to-tensorboard \
  --logging-level 20 \
  --vocab-extra-ids 0 \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --tiktoken-num-special-tokens 1000 \
  --save /workspace/Megatron-Bridge/nemo_experiments/default/checkpoints \
  --save-interval 200 \
  --load /workspace/Megatron-Bridge/nemo_experiments/default/checkpoints \
  --ckpt-format torch_dist \
  --use-persistent-ckpt-worker \
  --dist-ckpt-strictness log_all \
  --distributed-timeout-minutes 10 \

# ── Not supported: mapped keys with no value (set explicitly to use) ──
#   train.train_samples: mapped to --train-samples but value is None (not set)
#   train.exit_interval: mapped to --exit-interval but value is None (not set)
#   train.exit_duration_in_mins: mapped to --exit-duration-in-mins but value is None (not set)
#   model.virtual_pipeline_model_parallel_size: mapped to --num-virtual-stages-per-pipeline-rank but value is None (not set)
#   model.microbatch_group_size_per_vp_stage: mapped to --microbatch-group-size-per-virtual-pipeline-stage but value is None (not set)
#   model.num_layers_in_first_pipeline_stage: mapped to --decoder-first-pipeline-num-layers but value is None (not set)
#   model.num_layers_in_last_pipeline_stage: mapped to --decoder-last-pipeline-num-layers but value is None (not set)
#   model.pipeline_model_parallel_layout: mapped to --pipeline-model-parallel-layout but value is None (not set)
#   model.window_size: mapped to --window-size but value is None (not set)
#   model.recompute_granularity: mapped to --recompute-granularity but value is None (not set)
#   model.recompute_method: mapped to --recompute-method but value is None (not set)
#   model.recompute_num_layers: mapped to --recompute-num-layers but value is None (not set)
#   model.seq_len_interpolation_factor: mapped to --rotary-seq-len-interpolation-factor but value is None (not set)
#   model.vocab_size: mapped to --padded-vocab-size but value is None (not set)
#   optimizer.decoupled_lr: mapped to --decoupled-lr but value is None (not set)
#   optimizer.decoupled_min_lr: mapped to --decoupled-min-lr but value is None (not set)
#   scheduler.lr_decay_iters: mapped to --lr-decay-iters but value is None (not set)
#   scheduler.lr_decay_samples: mapped to --lr-decay-samples but value is None (not set)
#   scheduler.lr_wsd_decay_iters: mapped to --lr-wsd-decay-iters but value is None (not set)
#   scheduler.lr_wsd_decay_samples: mapped to --lr-wsd-decay-samples but value is None (not set)
#   scheduler.lr_warmup_fraction: mapped to --lr-warmup-fraction but value is None (not set)
#   dataset.path_to_cache: mapped to --data-cache-path but value is None (not set)
#   logger.wandb_project: mapped to --wandb-project but value is None (not set)
#   logger.wandb_exp_name: mapped to --wandb-exp-name but value is None (not set)
#   tokenizer.vocab_size: mapped to --vocab-size but value is None (not set)
#   tokenizer.vocab_file: mapped to --vocab-file but value is None (not set)
#   tokenizer.merge_file: mapped to --merge-file but value is None (not set)
#   checkpoint.pretrained_checkpoint: mapped to --pretrained-checkpoint but value is None (not set)

# ── Unknown Bridge keys (not mapped) ─────────────────────
#   tensor_inspect=None

# ── Skipped: 468 Bridge-only keys (no MLM equivalent)
