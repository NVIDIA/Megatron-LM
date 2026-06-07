# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Numerical parity for ``Qwen35VLPatchMerger`` vs HuggingFace reference.

The HuggingFace reference (``Qwen3VLVisionPatchMerger`` in
``transformers/models/qwen3_vl/modeling_qwen3_vl.py``, branch
``use_postshuffle_norm=False``) is reproduced inline so this test does not
require ``transformers`` to be installed.  The HF module is verbatim::

    self.norm       = nn.LayerNorm(hidden_size, eps=1e-6)
    self.linear_fc1 = nn.Linear(merge_dim, merge_dim)
    self.act_fn     = nn.GELU()             # default approximate='none'
    self.linear_fc2 = nn.Linear(merge_dim, out_hidden_size)

    x = self.norm(x)
    x = x.view(-1, merge_dim)
    x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))

With matching dims and weights copied across, the Megatron
implementation must agree:

  * fp32 forward:  max-abs diff <= 1e-4  (TE LayerNorm vs nn.LayerNorm
    have different fused reduction order; ~1e-5 absolute residual is
    structural and not a real divergence)
  * bf16 forward:  max-abs diff <= 5e-2  (bf16 ceiling for two-layer MLP)

Run with::

    torchrun --nproc_per_node=1 \\
        examples/multimodal_dev/tests/test_vision_patch_merger_parity.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../.."),
)
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from megatron.core import parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLPatchMerger


# Match Qwen3.5-VL 9B / 397B-A17B vision tower dims.
HIDDEN_SIZE = 1152
OUT_HIDDEN_SIZE = 3584
SPATIAL_MERGE_SIZE = 2
NUM_PATCHES = 64  # must be divisible by spatial_merge_size ** 2

ATOL_FP32 = 1e-4
RTOL_FP32 = 1e-3
ATOL_BF16 = 5e-2
RTOL_BF16 = 5e-2


class HFPatchMergerReference(nn.Module):
    """Inline HF ``Qwen3VLVisionPatchMerger`` (use_postshuffle_norm=False)."""

    def __init__(self, hidden_size: int, out_hidden_size: int, spatial_merge_size: int):
        super().__init__()
        self.merge_dim = hidden_size * (spatial_merge_size ** 2)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merge_dim, self.merge_dim)
        self.act_fn = nn.GELU()  # approximate='none' by default
        self.linear_fc2 = nn.Linear(self.merge_dim, out_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(hidden_states)
        x = x.view(-1, self.merge_dim)
        x = self.linear_fc1(x)
        x = self.act_fn(x)
        x = self.linear_fc2(x)
        return x


def _init_distributed() -> int:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def _init_megatron_parallel() -> None:
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(42)


def _build_config(dtype: torch.dtype) -> TransformerConfig:
    is_bf16 = dtype is torch.bfloat16
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=HIDDEN_SIZE,
        num_attention_heads=8,
        kv_channels=HIDDEN_SIZE // 8,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        bf16=is_bf16,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        add_bias_linear=True,
        gated_linear_unit=False,
        normalization="LayerNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )


def _copy_hf_to_megatron(hf: HFPatchMergerReference, mcore: Qwen35VLPatchMerger) -> None:
    """TP=1: 1:1 parameter copy between HF nn.Module and the MCore module."""
    with torch.no_grad():
        mcore.patch_norm.weight.copy_(hf.norm.weight.to(mcore.patch_norm.weight.dtype))
        mcore.patch_norm.bias.copy_(hf.norm.bias.to(mcore.patch_norm.bias.dtype))
        mcore.linear_fc1.weight.copy_(hf.linear_fc1.weight.to(mcore.linear_fc1.weight.dtype))
        mcore.linear_fc1.bias.copy_(hf.linear_fc1.bias.to(mcore.linear_fc1.bias.dtype))
        mcore.linear_fc2.weight.copy_(hf.linear_fc2.weight.to(mcore.linear_fc2.weight.dtype))
        mcore.linear_fc2.bias.copy_(hf.linear_fc2.bias.to(mcore.linear_fc2.bias.dtype))


def _run_one(dtype: torch.dtype, atol: float, rtol: float, device: torch.device, seed: int = 42) -> None:
    torch.manual_seed(seed)

    hf_ref = HFPatchMergerReference(
        hidden_size=HIDDEN_SIZE,
        out_hidden_size=OUT_HIDDEN_SIZE,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
    ).to(device=device, dtype=dtype).eval()

    config = _build_config(dtype)
    mcore = Qwen35VLPatchMerger(
        config=config,
        hidden_size=HIDDEN_SIZE,
        out_hidden_size=OUT_HIDDEN_SIZE,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
    ).to(device=device, dtype=dtype).eval()

    _copy_hf_to_megatron(hf_ref, mcore)

    x = torch.randn(NUM_PATCHES, HIDDEN_SIZE, device=device, dtype=dtype)

    with torch.no_grad():
        y_hf = hf_ref(x)
        y_mcore = mcore(x)

    assert y_hf.shape == y_mcore.shape, (y_hf.shape, y_mcore.shape)
    diff = (y_hf - y_mcore).abs()
    print(
        f"[{dtype}] shape={tuple(y_hf.shape)} "
        f"max_abs_diff={diff.max().item():.3e} "
        f"mean_abs_diff={diff.mean().item():.3e} "
        f"hf_norm={y_hf.float().norm().item():.4f} "
        f"mcore_norm={y_mcore.float().norm().item():.4f}"
    )
    torch.testing.assert_close(y_mcore, y_hf, atol=atol, rtol=rtol)


def main() -> None:
    local_rank = _init_distributed()
    _init_megatron_parallel()
    device = torch.device(f"cuda:{local_rank}")

    _run_one(torch.float32, ATOL_FP32, RTOL_FP32, device)
    _run_one(torch.bfloat16, ATOL_BF16, RTOL_BF16, device)

    if int(os.environ.get("RANK", 0)) == 0:
        print(
            "\nPASS: Qwen35VLPatchMerger logits match HF Qwen3VLVisionPatchMerger "
            "in both fp32 and bf16."
        )


if __name__ == "__main__":
    main()
