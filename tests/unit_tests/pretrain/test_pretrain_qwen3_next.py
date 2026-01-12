# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import shlex

import pytest


def get_bash_cmd(model_args, env_args=None, num_gpus_per_node=8):
    args = [
        "python3",
        "-m",
        "torch.distributed.launch",
        "--nproc-per-node=" + str(num_gpus_per_node),
    ]
    megatron_path = os.getenv("MEGATRON_PATH", "/opt/megatron-lm")
    script_path = os.path.join(megatron_path, "pretrain_gpt.py")
    args.append(script_path)
    for key, value in model_args.items():
        if value is None:
            pass
        if isinstance(value, bool):
            if value:
                args.append(key)
        else:
            args.extend([key, str(value)])
    if env_args is not None:
        for key, value in env_args.items():
            args = [f"{key}={value}"] + args
    return args


ENV_ARGS = {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "NCCL_ALGO": "Ring",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVTE_FUSED_ATTN": "1",
}

MODELS = {
    "qwen3_next": {
        "--untie-embeddings-and-output-weights": False,
        "--max-position-embeddings": 1024,
        "--num-layers": 6,
        "--hidden-size": 512,
        "--num-attention-heads": 8,
        "--group-query-attention": True,
        "--num-query-groups": 2,
        "--swiglu": True,
        "--position-embedding-type": "rope",
        "--rotary-percent": 0.25,
        "--apply-layernorm-1p": True,
        "--attention-output-gate": True,
        "--no-weight-decay-cond-type": "apply_wd_to_qk_layernorm",
        "--linear-attention-type": "gated_delta_net",
        "--linear-attention-freq": 3,
        "--linear-conv-kernel-dim": 4,
        "--linear-key-head-dim": 64,
        "--linear-value-head-dim": 64,
        "--linear-num-key-heads": 4,
        "--linear-num-value-heads": 8,
        "--tokenizer-type": "HuggingFaceTokenizer",
        "--tokenizer-model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "--no-rope-fusion": True,
        # MoE args
        "--num-experts": 32,
        "--moe-ffn-hidden-size": 64,
        "--moe-shared-expert-intermediate-size": 64,
        "--moe-shared-expert-gate": True,
        "--moe-router-load-balancing-type": "aux_loss",
        "--moe-router-topk": 8,
        "--disable-bias-linear": True,
        # "--moe-router-pre-softmax": False, # Different from Qwen2-MoE
        # "--moe-grouped-gemm": True,
        # "--moe-aux-loss-coeff": 1e-3,
        # "--moe-token-dispatcher-type": "flex",
        # "--moe-enable-deepep": True,
        # "--moe-permute-fusion": True,
        "--moe-router-dtype": "fp32",
        # "--moe-router-fusion": True,
    }
}

TRAINING_ARGS = {
    "--micro-batch-size": 4,
    "--global-batch-size": 32,
    "--seq-length": 1024,
    "--lr-decay-iters": 320000,
    "--mock-data": True,
    "--lr": 0.00015,
    "--lr-decay-style": "cosine",
    "--min-lr": 1.0e-5,
    "--weight-decay": 1e-2,
    "--clip-grad": 1.0,
    "--lr-warmup-fraction": 0.01,
    "--transformer-impl": "transformer_engine",
    "--deterministic-mode": True,
    "--no-gradient-accumulation-fusion": True,
    "--attention-softmax-in-fp32": True,
    "--use-mcore-models": True,
    "--ckpt-format": "torch_dist",
    "--bf16": True,
    "--attention-backend": "fused",
    "--eval-interval": 1000,
    "--eval-iters": 1,
    "--log-interval": 1,
    "--save-interval": 5,
    "--train-iters": 15,
}


@pytest.mark.parametrize("model_name", MODELS.keys())
@pytest.mark.parametrize(("tp", "sp", "pp", "ep", "cp"), [(1, False, 1, 1, 1), (2, True, 2, 2, 1)])
def test_qwen3_next_pretrain_and_resume(tmp_path_dist_ckpt, model_name, tp, sp, pp, ep, cp):
    save_path = tmp_path_dist_ckpt

    parallel_args = {
        "--tensor-model-parallel-size": tp,
        "--pipeline-model-parallel-size": pp,
        "--sequence-parallel": sp,
        "--expert-model-parallel-size": ep,
        "--context-parallel-size": cp,
    }

    # Load default arguments

    # Stage 1: Pretrain from scratch with parallel
    cmd = get_bash_cmd(
        model_args={
            **MODELS[model_name],
            **TRAINING_ARGS,
            **parallel_args,
            "--save": save_path,
            "--load": None,  # train from scratch
            "--exit-interval": 5,
        },
        env_args=ENV_ARGS,
        num_gpus_per_node=8,
    )
    cmd_str = shlex.join(cmd)
    ret = os.system(cmd_str)  # TODO(yuzhongw): use subprocess.run instead of os.system
    assert ret == 0, f"Failed to pretrain {model_name} at the first stage with command: {cmd_str}"

    # Stage 2: Resume training w/o parallel
    cmd = get_bash_cmd(
        model_args={
            **MODELS[model_name],
            **TRAINING_ARGS,
            "--save": save_path,
            "--load": save_path,  # load from checkpoint
            "--exit-interval": 10,
        },
        env_args=ENV_ARGS,
        num_gpus_per_node=8,
    )
    cmd_str = shlex.join(cmd)
    ret = os.system(cmd_str)
    assert (
        ret == 0
    ), f"Failed to resume training {model_name} at the second stage with command: {cmd_str}"

    # Stage 3: Resume training with parallel
    cmd = get_bash_cmd(
        model_args={
            **MODELS[model_name],
            **TRAINING_ARGS,
            **parallel_args,
            "--save": save_path,
            "--load": save_path,  # load from checkpoint
        },
        env_args=ENV_ARGS,
        num_gpus_per_node=8,
    )
    cmd_str = shlex.join(cmd)
    ret = os.system(cmd_str)
    assert (
        ret == 0
    ), f"Failed to resume training {model_name} at the third stage with command: {cmd_str}"
