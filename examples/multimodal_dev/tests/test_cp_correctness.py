# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Distributed correctness test for Context Parallelism (CP) support.

Verifies that CP>1 produces the same (or numerically close) loss as CP=1
for the Qwen3.5-VL multimodal model by running forward passes with
deterministic data and comparing the per-rank reduced losses.

Launch with torchrun (N must be >= 2*max_cp_size for zigzag splitting):

    # Test CP=2 on 2 GPUs:
    torchrun --nproc_per_node=2 examples/multimodal_dev/tests/test_cp_correctness.py --cp-size 2

    # Test CP=4 on 4 GPUs:
    torchrun --nproc_per_node=4 examples/multimodal_dev/tests/test_cp_correctness.py --cp-size 4

The test:
  1. Builds a tiny proxy model (2 layers, no MoE, no vision encoder).
  2. Generates a deterministic batch (same seed on all ranks).
  3. Runs forward with CP=1 (each rank processes the full sequence independently).
  4. Re-initialises model-parallel groups with the target CP size.
  5. Runs forward with CP=target (sequence is split across ranks).
  6. Compares the all-reduced loss values.

Exit code 0 = PASS, 1 = FAIL.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

# Ensure the repo root is on the path so that megatron and examples are importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _parse_args():
    parser = argparse.ArgumentParser(description="CP correctness test")
    parser.add_argument(
        "--cp-size", type=int, default=2,
        help="Target context-parallel size to compare against CP=1 baseline",
    )
    parser.add_argument(
        "--seq-len", type=int, default=128,
        help="Sequence length (must be divisible by 2*max(cp_size, tp_size*cp_size))",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-4,
        help="Absolute tolerance for loss comparison",
    )
    parser.add_argument(
        "--rtol", type=float, default=5e-2,
        help="Relative tolerance for loss comparison (default 5%%)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    # Megatron adds extra args; ignore them.
    args, _ = parser.parse_known_args()
    return args


def _init_distributed():
    """Initialise torch.distributed if not already done."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def _init_megatron_parallel(tp_size=1, pp_size=1, cp_size=1, seed=42):
    """(Re-)initialise Megatron model-parallel groups and RNG tracker."""
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
    )
    model_parallel_cuda_manual_seed(seed)


def _make_deterministic_batch(seed, batch_size, seq_len, vocab_size, device):
    """Create a deterministic batch identical on all ranks."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), generator=rng,
    ).to(device)
    labels = torch.randint(
        0, vocab_size, (batch_size, seq_len), generator=rng,
    ).to(device)
    loss_mask = torch.ones(batch_size, seq_len, device=device)
    # Standard position_ids [B, S]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
    }


def _build_tiny_model(cp_size, device):
    """Build a minimal GPTModel for testing (no vision, no MoE)."""
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_config import TransformerConfig

    hidden_size = 256
    num_heads = 4
    config = TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        ffn_hidden_size=hidden_size * 4,
        num_attention_heads=num_heads,
        kv_channels=hidden_size // num_heads,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        bf16=True,
        context_parallel_size=cp_size,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        sequence_parallel=False,
    )

    spec = get_gpt_layer_with_transformer_engine_spec()

    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=1024,
        max_sequence_length=4096,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
    )
    model = model.to(device=device, dtype=torch.bfloat16)
    return model, config


def _forward_with_cp(model, batch, cp_size):
    """Run forward pass, handling CP splitting of the batch.

    When cp_size > 1, splits the batch tensors using the same zigzag
    logic as multimodal_dev/models/base.py.
    """
    from examples.multimodal_dev.models.base import _cp_split_tensor
    from megatron.core import parallel_state as ps

    input_ids = batch["input_ids"].clone()
    labels = batch["labels"].clone()
    loss_mask = batch["loss_mask"].clone()
    position_ids = batch["position_ids"].clone()

    if cp_size > 1:
        cp_rank = ps.get_context_parallel_rank()
        input_ids = _cp_split_tensor(input_ids, seq_dim=1, cp_size=cp_size, cp_rank=cp_rank)
        labels = _cp_split_tensor(labels, seq_dim=1, cp_size=cp_size, cp_rank=cp_rank)
        loss_mask = _cp_split_tensor(loss_mask, seq_dim=1, cp_size=cp_size, cp_rank=cp_rank)
        # position_ids are NOT split — the RoPE layer handles CP slicing internally.

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            attention_mask=None,
        )

    # output is the per-token loss [B, S/CP]
    masked_loss = (output.float() * loss_mask.float()).sum()
    num_tokens = loss_mask.sum()

    # All-reduce across CP ranks to get global loss
    if cp_size > 1:
        cp_group = ps.get_context_parallel_group()
        dist.all_reduce(masked_loss, group=cp_group)
        dist.all_reduce(num_tokens, group=cp_group)

    avg_loss = masked_loss / num_tokens.clamp(min=1)
    return avg_loss.item()


def main():
    args = _parse_args()
    local_rank = _init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    target_cp = args.cp_size
    if world_size < target_cp:
        if rank == 0:
            print(
                f"SKIP: world_size={world_size} < cp_size={target_cp}. "
                f"Need at least {target_cp} GPUs.",
                flush=True,
            )
        dist.destroy_process_group()
        sys.exit(0)
    if world_size % target_cp != 0:
        if rank == 0:
            print(
                f"SKIP: world_size={world_size} is not divisible by cp_size={target_cp}.",
                flush=True,
            )
        dist.destroy_process_group()
        sys.exit(0)

    vocab_size = 1024

    # Ensure seq_len is divisible by 2 * target_cp
    seq_len = args.seq_len
    align = 2 * target_cp
    if seq_len % align != 0:
        seq_len = ((seq_len + align - 1) // align) * align
        if rank == 0:
            print(f"Adjusted seq_len to {seq_len} for alignment with CP={target_cp}", flush=True)

    # --- Step 1: CP=1 baseline ---
    if rank == 0:
        print(f"=== CP=1 baseline (world_size={world_size}) ===", flush=True)

    _init_megatron_parallel(cp_size=1)

    # Set deterministic seed for model init
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_cp1, _ = _build_tiny_model(cp_size=1, device=device)

    batch = _make_deterministic_batch(
        seed=args.seed + 1, batch_size=1, seq_len=seq_len,
        vocab_size=vocab_size, device=device,
    )

    loss_cp1 = _forward_with_cp(model_cp1, batch, cp_size=1)

    if rank == 0:
        print(f"  CP=1 loss: {loss_cp1:.6f}", flush=True)

    # Save model state for reuse
    state_dict = model_cp1.state_dict()
    del model_cp1
    torch.cuda.empty_cache()

    # --- Step 2: CP=target ---
    if rank == 0:
        print(f"=== CP={target_cp} (world_size={world_size}) ===", flush=True)

    _init_megatron_parallel(cp_size=target_cp)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_cpN, _ = _build_tiny_model(cp_size=target_cp, device=device)

    # Load the same weights to ensure identical model
    model_cpN.load_state_dict(state_dict, strict=True)
    del state_dict

    loss_cpN = _forward_with_cp(model_cpN, batch, cp_size=target_cp)

    if rank == 0:
        print(f"  CP={target_cp} loss: {loss_cpN:.6f}", flush=True)

    del model_cpN
    torch.cuda.empty_cache()

    # --- Step 3: Compare ---
    if rank == 0:
        diff = abs(loss_cpN - loss_cp1)
        rel_diff = diff / max(abs(loss_cp1), 1e-10)

        print(f"\n=== Comparison ===", flush=True)
        print(f"  CP=1 loss:         {loss_cp1:.6f}", flush=True)
        print(f"  CP={target_cp} loss:         {loss_cpN:.6f}", flush=True)
        print(f"  Absolute diff:     {diff:.6e}", flush=True)
        print(f"  Relative diff:     {rel_diff:.6e}", flush=True)
        print(f"  Tolerance (atol):  {args.atol:.6e}", flush=True)
        print(f"  Tolerance (rtol):  {args.rtol:.6e}", flush=True)

        passed = diff <= args.atol + args.rtol * abs(loss_cp1)
        if passed:
            print(f"\nPASS: CP={target_cp} matches CP=1 baseline", flush=True)
        else:
            print(f"\nFAIL: CP={target_cp} loss differs from CP=1 beyond tolerance", flush=True)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0 and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
