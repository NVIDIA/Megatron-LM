#!/usr/bin/env bash
# Decode-attention bake-off before any integration work: does flashinfer's trtllm-gen
# kernel beat FA2 at our shapes, and does it accept Megatron's 256-token pages?
set -uo pipefail
VENV=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/envs/megatron_lm/dd356431262b5db4/.venv
export PYTHONPATH="$PWD:$PWD/dev/moe_fused:${PYTHONPATH:-}"
$VENV/bin/python dev/moe_fused/harness_attn.py --seqlens 512,1024,2048 --pages 256,128,64
